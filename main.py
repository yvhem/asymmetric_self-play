import os
import random
import time
from collections import deque
from copy import deepcopy

import click
import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mujoco_py import MjViewer

from robogym.envs.rearrange.blocks_train import make_env
from robogym.utils.rotation import normalize_angles, euler2quat

from agent import ABCBuffer, PPOAgent, RolloutBuffer
from config import ActionConfig, DEVICE, GoalValidationConfig, HardwareConfig, RewardConfig, SelfPlayConfig
from utils import get_obs_space_dims

@ray.remote
def rollout_worker(worker_id, seed, num_objects, alice_weights, bob_weights, alice_obs_rms=None, bob_obs_rms=None):
    # each worker has its own environment and agents
    num_episode_objects = num_objects if num_objects > 0 else np.random.randint(1, 3)
    env_params = {
        "simulation_params": {
            "num_objects": num_episode_objects,
            "max_num_objects": 2
        }
    }
    env = make_env(parameters=env_params, starting_seed=seed + worker_id)
    obs_space_dims = get_obs_space_dims(env)
    action_dim = ActionConfig.ACTION_DIMS * ActionConfig.ACTION_BINS

    alice = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=True)
    bob = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=False)
    
    # use CPU
    alice.policy_old.to('cpu'); bob.policy_old.to('cpu')
    
    alice.policy_old.load_state_dict(alice_weights)
    bob.policy_old.load_state_dict(bob_weights)
    #alice.obs_rms = alice_obs_rms
    #bob.obs_rms = bob_obs_rms

    alice_memory = RolloutBuffer()
    bob_rl_memory = RolloutBuffer()
    bob_abc_memory = ABCBuffer()
    
    stats = {"alice_total_reward": 0.0, "bob_successes": 0, "goals_attempted": 0, "alice_valid_goals": 0}

    # run one episode
    env.reset()
    turn_start_sim_state = env.unwrapped.sim.get_state()
    
    alice_lstm_state = alice.policy_old.get_initial_lstm_state(1, 'cpu')
    bob_lstm_state = bob.policy_old.get_initial_lstm_state(1, 'cpu')
    bob_has_failed = False

    for _ in range(SelfPlayConfig.NUM_GOALS_PER_EPISODE):
        stats['goals_attempted'] += 1
        
        # Alice's turn
        env.unwrapped.sim.set_state(turn_start_sim_state)
        alice_current_obs = env.observe()
        initial_turn_pos = alice_current_obs['obj_pos'].copy()
        
        alice_turn_traj = RolloutBuffer()
        done = False
        for _ in range(SelfPlayConfig.ALICE_GOAL_SETTING_STEPS):
            alice_obs_no_goal = {k: v for k, v in alice_current_obs.items() if not k.startswith('goal_')}
            action, value, logprob, new_lstm = alice.select_action(alice_obs_no_goal, alice_lstm_state)
            
            alice_turn_traj.states.append(alice_current_obs)
            alice_turn_traj.actions.append(action)
            alice_turn_traj.logprobs.append(logprob)
            alice_turn_traj.values.append(value)
            
            alice_current_obs, _, done, _ = env.step(action)
            alice_lstm_state = new_lstm
            if done: break
        if done: break
        
        goal_pos = alice_current_obs['obj_pos'].copy()
        goal_rot = alice_current_obs['obj_rot'].copy()
        
        # goal validation
        pos_diff = np.linalg.norm(goal_pos - initial_turn_pos, axis=1)
        obj_moved = np.any(pos_diff > GoalValidationConfig.SUCCESS_THRESHOLD_POS)
        num_actual_objects = env.unwrapped.mujoco_simulation.num_objects
        is_off_table = np.any(env.unwrapped.mujoco_simulation.check_objects_off_table(goal_pos[:num_actual_objects]))

        if not obj_moved or is_off_table:
            final_reward = 0.0
            alice_turn_traj.rewards = [0.0] * (len(alice_turn_traj.actions) - 1) + [final_reward]
            alice_turn_traj.is_terminals = [False] * (len(alice_turn_traj.actions) - 1) + [True]

            alice_memory.actions.extend(alice_turn_traj.actions)
            alice_memory.states.extend(alice_turn_traj.states)
            alice_memory.logprobs.extend(alice_turn_traj.logprobs)
            alice_memory.rewards.extend(alice_turn_traj.rewards)
            alice_memory.values.extend(alice_turn_traj.values)
            alice_memory.is_terminals.extend(alice_turn_traj.is_terminals)
            break

        stats['alice_valid_goals'] += 1
        
        is_in_placement_area_mask = env.unwrapped.mujoco_simulation.check_objects_in_placement_area(goal_pos[:num_actual_objects])
        is_out_of_zone = not np.all(is_in_placement_area_mask)

        bob_succeeded = False
        if not bob_has_failed:
            # bob's turn
            env.unwrapped.sim.set_state(turn_start_sim_state)
            env.unwrapped._goal['obj_pos'] = goal_pos
            env.unwrapped._goal['obj_rot'] = goal_rot
            qpos_goal = env.unwrapped.sim.data.qpos.copy()
            for i in range(num_actual_objects):
                joint_name = f"object{i}:joint"
                qpos_addr_start = env.unwrapped.sim.model.get_joint_qpos_addr(joint_name)[0]
                qpos_goal[qpos_addr_start : qpos_addr_start + 3] = goal_pos[i]
                qpos_goal[qpos_addr_start + 3 : qpos_addr_start + 7] = euler2quat(goal_rot[i])
            env.unwrapped._goal['qpos_goal'] = qpos_goal
            bob_current_obs = env.observe()
            
            bob_turn_traj = RolloutBuffer()
            bob_achieved_obj_mask = np.zeros(num_actual_objects, dtype=bool)
            
            for step_count in range(SelfPlayConfig.BOB_MAX_GOAL_SOLVING_STEPS):
                state_to_store = bob_current_obs
                bob_obs_with_goal = bob_current_obs.copy()
                num_bob_objs = bob_obs_with_goal['obj_pos'].shape[0]
                bob_obs_with_goal['goal_obj_pos'] = goal_pos[:num_bob_objs]
                bob_obs_with_goal['goal_obj_rot'] = goal_rot[:num_bob_objs]
                bob_obs_with_goal['rel_goal_obj_pos'] = goal_pos[:num_bob_objs] - bob_current_obs['obj_pos'][:num_bob_objs]
                bob_obs_with_goal['rel_goal_obj_rot'] = normalize_angles(goal_rot[:num_bob_objs] - bob_current_obs['obj_rot'][:num_bob_objs])
                
                action, value, logprob, new_lstm = bob.select_action(bob_obs_with_goal, bob_lstm_state)
                
                bob_turn_traj.states.append(state_to_store)
                bob_turn_traj.actions.append(action)
                bob_turn_traj.logprobs.append(logprob)
                bob_turn_traj.values.append(value)
                
                bob_current_obs, _, done, _ = env.step(action)
                bob_lstm_state = new_lstm

                # check for success
                dist_pos = np.linalg.norm(bob_current_obs['obj_pos'][:num_actual_objects] - goal_pos[:num_actual_objects], axis=-1)
                dist_rot = np.linalg.norm(normalize_angles(bob_current_obs['obj_rot'][:num_actual_objects] - goal_rot[:num_actual_objects]), axis=-1)
            
                per_obj_success = (dist_pos < GoalValidationConfig.SUCCESS_THRESHOLD_POS) & (dist_rot < GoalValidationConfig.SUCCESS_THRESHOLD_ROT)
                reward_for_step = 0
                newly_achieved_mask = per_obj_success & ~bob_achieved_obj_mask
                reward_for_step += np.sum(newly_achieved_mask) * RewardConfig.OBJECT_PLACED_REWARD
                moved_away_mask = ~per_obj_success & bob_achieved_obj_mask
                reward_for_step += np.sum(moved_away_mask) * RewardConfig.OBJECT_MOVED_AWAY_PENALTY

                bob_achieved_obj_mask = per_obj_success
                is_turn_success = np.all(per_obj_success)
                
                if is_turn_success:
                    reward_for_step += RewardConfig.GOAL_SUCCESS_REWARD
                    bob_succeeded = True

                bob_turn_traj.rewards.append(reward_for_step)
                bob_turn_traj.is_terminals.append(is_turn_success or done)
                
                if is_turn_success or done:
                    break
            
            # collect Bob's trajectory for PPO update, regardless of success
            bob_rl_memory.actions.extend(bob_turn_traj.actions)
            bob_rl_memory.states.extend(bob_turn_traj.states)
            bob_rl_memory.logprobs.extend(bob_turn_traj.logprobs)
            bob_rl_memory.rewards.extend(bob_turn_traj.rewards)
            bob_rl_memory.values.extend(bob_turn_traj.values)
            bob_rl_memory.is_terminals.extend(bob_turn_traj.is_terminals)

        # assign rewards and prepare ABC data for update
        alice_final_reward = RewardConfig.VALID_GOAL_REWARD
        if bob_succeeded:
            stats['bob_successes'] += 1
        else:
            bob_has_failed = True
            alice_final_reward += RewardConfig.BOB_FAILURE_REWARD
            
            bob_abc_memory.actions.extend(alice_turn_traj.actions)
            bob_abc_memory.states.extend(alice_turn_traj.states)
            bob_abc_memory.goals.extend([{'goal_obj_pos': goal_pos, 'goal_obj_rot': goal_rot}] * len(alice_turn_traj.actions))

            demo_logprobs = []
            bob_abc_lstm_state = bob.policy_old.get_initial_lstm_state(1, 'cpu')
            for i in range(len(alice_turn_traj.actions)):
                # construct the observation Bob would see
                abc_obs = alice_turn_traj.states[i].copy()
                num_abc_objs = abc_obs['obj_pos'].shape[0]
                abc_obs['goal_obj_pos'] = goal_pos[:num_abc_objs]
                abc_obs['goal_obj_rot'] = goal_rot[:num_abc_objs]
                abc_obs['rel_goal_obj_pos'] = goal_pos[:num_abc_objs] - abc_obs['obj_pos'][:num_abc_objs]
                abc_obs['rel_goal_obj_rot'] = normalize_angles(goal_rot[:num_abc_objs] - abc_obs['obj_rot'][:num_abc_objs])
                
                
                # get logprob from Bob's perspective
                _, _, logprob, bob_abc_lstm_state = bob.select_action(abc_obs, bob_abc_lstm_state)
                demo_logprobs.append(logprob)
            bob_abc_memory.logprobs.extend(demo_logprobs)

        if is_out_of_zone:
            alice_final_reward += RewardConfig.OUT_OF_ZONE_PENALTY

        print(f"Bob {'succeeded' if bob_succeeded else 'failed'}. Alice reward: {alice_final_reward:.2f}")
        
        alice_turn_traj.rewards = [0.0] * (len(alice_turn_traj.actions) - 1) + [alice_final_reward]
        alice_turn_traj.is_terminals = [False] * (len(alice_turn_traj.actions) - 1) + [True]
        
        alice_memory.actions.extend(alice_turn_traj.actions)
        alice_memory.states.extend(alice_turn_traj.states)
        alice_memory.logprobs.extend(alice_turn_traj.logprobs)
        alice_memory.rewards.extend(alice_turn_traj.rewards)
        alice_memory.values.extend(alice_turn_traj.values)
        alice_memory.is_terminals.extend(alice_turn_traj.is_terminals)

        turn_start_sim_state = env.unwrapped.sim.get_state()

    env.close()
    return alice_memory, bob_rl_memory, bob_abc_memory, stats

def run_debug_visualization(load_step, num_objects):
    num_debug_workers = 2
    envs = []
    viewers = []
    for i in range(num_debug_workers):
        env_params = {"simulation_params": {"max_num_objects": 2, "num_objects": num_objects if num_objects > 0 else 1}}
        # fixed seeds for repeatability
        env = make_env(parameters=env_params, starting_seed=1337 + i)
        envs.append(env)
        viewers.append(MjViewer(env.unwrapped.sim))

    obs_space_dims = get_obs_space_dims(envs[0])
    action_dim = ActionConfig.ACTION_DIMS * ActionConfig.ACTION_BINS
    
    alice = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=True)
    bob = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=False)

    if load_step:
        print(f"Loading models from step {load_step}")
        alice.load_models(f"models/alice_{load_step}")
        bob.load_models(f"models/bob_{load_step}")

    for training_step in range(1, 21):
        print(f"Step {training_step}")

        main_alice_memory = RolloutBuffer()
        main_bob_rl_memory = RolloutBuffer()
        main_bob_abc_memory = ABCBuffer()

        for worker_id in range(num_debug_workers):
            env = envs[worker_id]
            viewer = viewers[worker_id]
            
            print(f"Worker {worker_id}")
            
            alice.policy_old.load_state_dict(alice.policy.state_dict())
            bob.policy_old.load_state_dict(bob.policy.state_dict())
            
            worker_seed = training_step * num_debug_workers + worker_id

            alice_mem, bob_rl_mem, bob_abc_mem = run_visualized_episode(
                worker_id, env, viewer, alice, bob, worker_seed
            )

            main_alice_memory.actions.extend(alice_mem.actions); main_alice_memory.states.extend(alice_mem.states); main_alice_memory.logprobs.extend(alice_mem.logprobs); main_alice_memory.rewards.extend(alice_mem.rewards); main_alice_memory.values.extend(alice_mem.values); main_alice_memory.is_terminals.extend(alice_mem.is_terminals)
            main_bob_rl_memory.actions.extend(bob_rl_mem.actions); main_bob_rl_memory.states.extend(bob_rl_mem.states); main_bob_rl_memory.logprobs.extend(bob_rl_mem.logprobs); main_bob_rl_memory.rewards.extend(bob_rl_mem.rewards); main_bob_rl_memory.values.extend(bob_rl_mem.values); main_bob_rl_memory.is_terminals.extend(bob_rl_mem.is_terminals)
            main_bob_abc_memory.actions.extend(bob_abc_mem.actions); main_bob_abc_memory.states.extend(bob_abc_mem.states); main_bob_abc_memory.goals.extend(bob_abc_mem.goals); main_bob_abc_memory.logprobs.extend(bob_abc_mem.logprobs)
        
        if len(main_alice_memory) > 0:
            alice.update(main_alice_memory)
        if len(main_bob_rl_memory) > 0 or len(main_bob_abc_memory) > 0:
            bob.update(main_bob_rl_memory, main_bob_abc_memory)

def run_visualized_episode(worker_id, env, viewer, alice, bob, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    alice_memory = RolloutBuffer()
    bob_rl_memory = RolloutBuffer()
    bob_abc_memory = ABCBuffer()
    
    env.reset()
    turn_start_sim_state = env.unwrapped.sim.get_state()
    
    alice_lstm_state = alice.policy_old.get_initial_lstm_state(1, 'cpu')
    bob_lstm_state = bob.policy_old.get_initial_lstm_state(1, 'cpu')
    bob_has_failed = False

    for goal_num in range(SelfPlayConfig.NUM_GOALS_PER_EPISODE):
        env.unwrapped.sim.set_state(turn_start_sim_state)
        viewer.update_sim(env.unwrapped.sim)
        alice_current_obs = env.observe()
        initial_turn_pos = alice_current_obs['obj_pos'].copy()
        
        alice_turn_traj = RolloutBuffer()
        done = False
        for _ in range(SelfPlayConfig.ALICE_GOAL_SETTING_STEPS):
            alice_obs_no_goal = {k: v for k, v in alice_current_obs.items() if not k.startswith('goal_')}
            action, value, logprob, new_lstm = alice.select_action(alice_obs_no_goal, alice_lstm_state)
            alice_turn_traj.states.append(alice_current_obs)
            alice_turn_traj.actions.append(action); alice_turn_traj.logprobs.append(logprob); alice_turn_traj.values.append(value)
            alice_current_obs, _, done, _ = env.step(action)
            alice_lstm_state = new_lstm
            viewer.render()
            if done: break
        if done: break
        
        goal_pos = alice_current_obs['obj_pos'].copy(); goal_rot = alice_current_obs['obj_rot'].copy()
        
        pos_diff = np.linalg.norm(goal_pos - initial_turn_pos, axis=1)
        obj_moved = np.any(pos_diff > GoalValidationConfig.SUCCESS_THRESHOLD_POS)
        num_actual_objects = env.unwrapped.mujoco_simulation.num_objects
        is_off_table = np.any(env.unwrapped.mujoco_simulation.check_objects_off_table(goal_pos[:num_actual_objects]))

        if not obj_moved or is_off_table:
            final_reward = 0.0
            alice_turn_traj.rewards = [0.0] * (len(alice_turn_traj.actions) - 1) + [final_reward]
            alice_turn_traj.is_terminals = [False] * (len(alice_turn_traj.actions) - 1) + [True]
            alice_memory.actions.extend(alice_turn_traj.actions); alice_memory.states.extend(alice_turn_traj.states); alice_memory.logprobs.extend(alice_turn_traj.logprobs); alice_memory.rewards.extend(alice_turn_traj.rewards); alice_memory.values.extend(alice_turn_traj.values); alice_memory.is_terminals.extend(alice_turn_traj.is_terminals)
            break
        
        is_in_placement_area_mask = env.unwrapped.mujoco_simulation.check_objects_in_placement_area(goal_pos[:num_actual_objects])
        is_out_of_zone = not np.all(is_in_placement_area_mask)
        
        bob_succeeded = False
        if not bob_has_failed:
            env.unwrapped.sim.set_state(turn_start_sim_state)
            viewer.update_sim(env.unwrapped.sim)
            bob_current_obs = env.observe()
            bob_turn_traj = RolloutBuffer()
            for step_count in range(SelfPlayConfig.BOB_MAX_GOAL_SOLVING_STEPS):
                state_to_store = bob_current_obs
                bob_obs_with_goal = bob_current_obs.copy()
                num_bob_objs = bob_obs_with_goal['obj_pos'].shape[0]
                bob_obs_with_goal['goal_obj_pos'] = goal_pos[:num_bob_objs]
                bob_obs_with_goal['goal_obj_rot'] = goal_rot[:num_bob_objs]
                bob_obs_with_goal['rel_goal_obj_pos'] = goal_pos[:num_bob_objs] - bob_current_obs['obj_pos'][:num_bob_objs]
                bob_obs_with_goal['rel_goal_obj_rot'] = normalize_angles(goal_rot[:num_bob_objs] - bob_current_obs['obj_rot'][:num_bob_objs])
                
                action, value, logprob, new_lstm = bob.select_action(bob_obs_with_goal, bob_lstm_state)
                bob_turn_traj.states.append(state_to_store); bob_turn_traj.actions.append(action); bob_turn_traj.logprobs.append(logprob); bob_turn_traj.values.append(value)
                bob_current_obs, _, done, _ = env.step(action)
                bob_lstm_state = new_lstm
                viewer.render()
                
                dist_pos = np.linalg.norm(bob_current_obs['obj_pos'][:num_actual_objects] - goal_pos[:num_actual_objects], axis=-1)
                dist_rot = np.linalg.norm(normalize_angles(bob_current_obs['obj_rot'][:num_actual_objects] - goal_rot[:num_actual_objects]), axis=-1)
                is_turn_success = np.all((dist_pos < GoalValidationConfig.SUCCESS_THRESHOLD_POS) & (dist_rot < GoalValidationConfig.SUCCESS_THRESHOLD_ROT))
                if is_turn_success:
                    bob_succeeded = True
                    break
                if done: break
            bob_rl_memory.actions.extend(bob_turn_traj.actions); bob_rl_memory.states.extend(bob_turn_traj.states); bob_rl_memory.logprobs.extend(bob_turn_traj.logprobs); bob_rl_memory.rewards.extend(bob_turn_traj.rewards); bob_rl_memory.values.extend(bob_turn_traj.values); bob_rl_memory.is_terminals.extend(bob_turn_traj.is_terminals)

        alice_final_reward = RewardConfig.VALID_GOAL_REWARD
        if bob_succeeded:
            pass
        else:
            bob_has_failed = True
            alice_final_reward += RewardConfig.BOB_FAILURE_REWARD
            bob_abc_memory.actions.extend(alice_turn_traj.actions); bob_abc_memory.states.extend(alice_turn_traj.states); bob_abc_memory.goals.extend([{'goal_obj_pos': goal_pos, 'goal_obj_rot': goal_rot}] * len(alice_turn_traj.actions)); bob_abc_memory.logprobs.extend(alice_turn_traj.logprobs)
        
        if is_out_of_zone:
            alice_final_reward += RewardConfig.OUT_OF_ZONE_PENALTY
        
        print(f"Worker {worker_id}: Bob {'succeeded' if bob_succeeded else 'failed'}. Alice reward: {alice_final_reward:.2f}")

        alice_turn_traj.rewards = [0.0] * (len(alice_turn_traj.actions) - 1) + [alice_final_reward]
        alice_turn_traj.is_terminals = [False] * (len(alice_turn_traj.actions) - 1) + [True]
        alice_memory.actions.extend(alice_turn_traj.actions); alice_memory.states.extend(alice_turn_traj.states); alice_memory.logprobs.extend(alice_turn_traj.logprobs); alice_memory.rewards.extend(alice_turn_traj.rewards); alice_memory.values.extend(alice_turn_traj.values); alice_memory.is_terminals.extend(alice_turn_traj.is_terminals)

        turn_start_sim_state = env.unwrapped.sim.get_state()

    return alice_memory, bob_rl_memory, bob_abc_memory


@click.command()
@click.option('--debug', is_flag=True, help="Run a 2-worker debug session.")
@click.option('--load-step', type=int, default=None, help="Training step to load models from.")
@click.option('--num-objects', type=int, default=0, help="Number of objects (1 or 2). 0 for random.")
def main(debug, load_step, num_objects):
    if debug:
        run_debug_visualization(load_step, num_objects)
    else:
        ray.init(num_cpus=HardwareConfig.NUM_WORKERS, ignore_reinit_error=True)
        
        env_params = {"simulation_params": {"max_num_objects": 2}}
        dummy_env = make_env(parameters=env_params, starting_seed=0)
        obs_space_dims = get_obs_space_dims(dummy_env)
        action_dim = ActionConfig.ACTION_DIMS * ActionConfig.ACTION_BINS
        dummy_env.close()

        log_dir_suffix = f"_objects={num_objects}" if num_objects > 0 else "_objects=random"
        writer = SummaryWriter(comment=log_dir_suffix)

        alice = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=True)
        bob = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=False)

        start_step = 1
        if load_step:
            print(f"Loading models from step {load_step}")
            alice.load_models(f"models/alice_{load_step}")
            bob.load_models(f"models/bob_{load_step}")
            start_step = load_step + 1

        cumulative_alice_valid_goals = 0.0
        cumulative_bob_successes = 0.0

        alice_policy_pool = deque([deepcopy(alice.policy.state_dict()) for _ in range(10)], maxlen=10)
        bob_policy_pool = deque([deepcopy(bob.policy.state_dict()) for _ in range(10)], maxlen=10)

        for training_step in range(start_step, 200001):
            if training_step > 1:
                alice_policy_pool.append(deepcopy(alice.policy.state_dict()))
                bob_policy_pool.append(deepcopy(bob.policy.state_dict()))
            
            # select policies for this batch
            alice_weights_for_worker = alice.policy.state_dict()
            bob_weights_for_worker = bob.policy.state_dict()
            
            # with p=0.2 one of the agents plays against a past version of the other
            if len(bob_policy_pool) > 0 and random.random() < SelfPlayConfig.PAST_POLICY_PROB:
                bob_weights_for_worker = random.choice(bob_policy_pool)
            elif len(alice_policy_pool) > 0 and random.random() < SelfPlayConfig.PAST_POLICY_PROB:
                alice_weights_for_worker = random.choice(alice_policy_pool)

            all_worker_stats = [] 

            alice_weights_id = ray.put({k: v.cpu() for k, v in alice_weights_for_worker.items()})
            bob_weights_id = ray.put({k: v.cpu() for k, v in bob_weights_for_worker.items()})

            #alice_rms_id = ray.put(deepcopy(alice.obs_rms))
            #bob_rms_id = ray.put(deepcopy(bob.obs_rms))

            alice_main_memory = RolloutBuffer()
            bob_main_rl_memory = RolloutBuffer()
            bob_main_abc_memory = ABCBuffer()
            
            rollout_promises = [
                rollout_worker.remote(i, training_step * HardwareConfig.NUM_WORKERS + i, num_objects, 
                                      alice_weights_id, bob_weights_id#, alice_rms_id, bob_rms_id) 
                )
                for i in range(HardwareConfig.NUM_WORKERS)
            ]

            with tqdm(total=HardwareConfig.BLOCK_BATCH_SIZE, desc=f"Step {training_step}") as pbar:
                while pbar.n < HardwareConfig.BLOCK_BATCH_SIZE:
                    ready, rollout_promises = ray.wait(rollout_promises, num_returns=1)
                    if not ready: continue
                    
                    result_id = ready[0]
                    alice_mem, bob_rl_mem, bob_abc_mem, stats = ray.get(result_id)
                    all_worker_stats.append(stats)
                    
                    size = len(alice_mem) + len(bob_rl_mem) + len(bob_abc_mem)
                    pbar.update(size)
                    
                    alice_main_memory.actions.extend(alice_mem.actions); alice_main_memory.states.extend(alice_mem.states); alice_main_memory.logprobs.extend(alice_mem.logprobs); alice_main_memory.rewards.extend(alice_mem.rewards); alice_main_memory.values.extend(alice_mem.values); alice_main_memory.is_terminals.extend(alice_mem.is_terminals)
                    bob_main_rl_memory.actions.extend(bob_rl_mem.actions); bob_main_rl_memory.states.extend(bob_rl_mem.states); bob_main_rl_memory.logprobs.extend(bob_rl_mem.logprobs); bob_main_rl_memory.rewards.extend(bob_rl_mem.rewards); bob_main_rl_memory.values.extend(bob_rl_mem.values); bob_main_rl_memory.is_terminals.extend(bob_rl_mem.is_terminals)
                    bob_main_abc_memory.actions.extend(bob_abc_mem.actions); bob_main_abc_memory.states.extend(bob_abc_mem.states); bob_main_abc_memory.goals.extend(bob_abc_mem.goals); bob_main_abc_memory.logprobs.extend(bob_abc_mem.logprobs)

                    # launch a new task to replace the finished one
                    new_worker_seed = pbar.n + training_step * HardwareConfig.NUM_WORKERS
                    rollout_promises.append(
                        rollout_worker.remote(-1, new_worker_seed, num_objects,
                                                alice_weights_id, bob_weights_id#, alice_rms_id, bob_rms_id)
                        )
                    )
                
            # cancel any remaining tasks
            for promise in rollout_promises:
                ray.cancel(promise)

            current_step_valid_goals = sum(s['alice_valid_goals'] for s in all_worker_stats)
            current_step_bob_successes = sum(s['bob_successes'] for s in all_worker_stats)

            cumulative_alice_valid_goals += current_step_valid_goals
            cumulative_bob_successes += current_step_bob_successes

            if cumulative_alice_valid_goals > 0:
                bob_alice_ratio = cumulative_bob_successes / cumulative_alice_valid_goals
            else:
                bob_alice_ratio = 0.0

            writer.add_scalar('Custom/Successes_vs_ValidGoals', bob_alice_ratio, training_step)
            writer.add_scalar('Custom/CumulativeAliceValidGoals', cumulative_alice_valid_goals, training_step)
            writer.add_scalar('Custom/CumulativeBobSuccesses', cumulative_bob_successes, training_step)

            alice_loss_info = alice.update(alice_main_memory)
            bob_loss_info = bob.update(bob_main_rl_memory, bob_main_abc_memory)
            
            for k, v in alice_loss_info.items(): writer.add_scalar(f'Alice/{k}', v, training_step)
            for k, v in bob_loss_info.items(): writer.add_scalar(f'Bob/{k}', v, training_step)

            if training_step % 10 == 0:
                alice.save_models(f"models/alice_{training_step}")
                bob.save_models(f"models/bob_{training_step}")

        writer.close()
        ray.shutdown()

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    main()
