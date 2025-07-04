import click
import numpy as np
import torch
import time
import mujoco_py
import importlib
import pdb

from agent import PPOAgent
from config import ActionConfig, GoalValidationConfig
from utils import get_obs_space_dims
from robogym.utils.rotation import normalize_angles, euler2quat

# A dictionary to map task names to their respective environment creation functions.
ENV_MAKERS = {
    "push": "robogym.envs.rearrange.blocks:make_env",
    "flip": "robogym.envs.rearrange.blocks:make_env",
    "pick-and-place": "robogym.envs.rearrange.blocks_pickandplace:make_env",
    "stack": "robogym.envs.rearrange.blocks_stack:make_env",
}

def set_goal_for_task(env, task_name):
    """ Manually set a goal for the specified holdout task. """
    obs = env.unwrapped.observe()
    num_objs = env.unwrapped.mujoco_simulation.num_objects

    goal_pos = obs['obj_pos'].copy()
    goal_rot = obs['obj_rot'].copy()

    if task_name == "push":
        goal_pos[0, 0] += 0.1
    elif task_name == "flip":
        current_euler = obs['obj_rot'][0]
        current_euler[2] += np.pi / 2
        goal_rot[0] = current_euler
    elif task_name == "pick-and-place":
        goal_pos[0, 0] += 0.1
        goal_pos[0, 2] += 0.1
    elif task_name == "stack":
        if num_objs > 1:
            goal_pos[1] = goal_pos[0].copy()
            obj_size = env.unwrapped.mujoco_simulation.simulation_params.object_size
            goal_pos[1, 2] += obj_size * 2

    env.unwrapped._goal['obj_pos'] = goal_pos
    env.unwrapped._goal['obj_rot'] = goal_rot
    
    sim = env.unwrapped.mujoco_simulation
    sim.set_target_pos(goal_pos[:num_objs])
    sim.set_target_rot(goal_rot[:num_objs])

    qpos_goal = sim.mj_sim.data.qpos.copy()
    for i in range(num_objs):
        joint_name = f"object{i}:joint"
        qpos_addr_start = sim.mj_sim.model.get_joint_qpos_addr(joint_name)[0]
        qpos_goal[qpos_addr_start : qpos_addr_start + 3] = goal_pos[i]
        qpos_goal[qpos_addr_start + 3 : qpos_addr_start + 7] = euler2quat(goal_rot[i])
    env.unwrapped._goal['qpos_goal'] = qpos_goal
    sim.forward()

    return goal_pos, goal_rot


@click.command()
@click.option('--task', type=click.Choice(['push', 'flip', 'pick-and-place', 'stack']), required=True, help="The holdout task to evaluate.")
@click.option('--load-step', type=int, required=True, help="Training step to load Bob's model from.")
@click.option('--episodes', type=int, default=10, help="Number of episodes to run for evaluation.")
@click.option('--max-steps', type=int, default=200, help="Maximum number of steps per episode.")
def evaluate(task, load_step, episodes, max_steps):
    """Evaluate a trained Bob agent on a specified holdout task."""
    print(f"--- Evaluating Bob on task: {task} ---")
    
    env_maker_path = ENV_MAKERS[task]
    module_path, func_name = env_maker_path.rsplit(":", 1)
    env_module = importlib.import_module(module_path)
    make_env = getattr(env_module, func_name)

    env_params = {"simulation_params": {"max_num_objects": 2}}
    if task == 'stack':
        env_params["simulation_params"]["num_objects"] = 2
    else:
        env_params["simulation_params"]["num_objects"] = 1

    env_constants = {
        "max_timesteps_per_goal_per_obj": max_steps
    }

    env = make_env(parameters=env_params, constants=env_constants)
    
    obs_space_dims = get_obs_space_dims(env)
    action_dim = ActionConfig.ACTION_DIMS * ActionConfig.ACTION_BINS

    bob = PPOAgent(obs_space_dims=obs_space_dims, action_dim=action_dim, is_alice=False)
    
    eval_device = torch.device('cpu')
    bob.policy_old.to(eval_device)

    print(f"Loading Bob's model from step {load_step}")
    bob.load_models(f"models/bob_{load_step}")

    viewer = mujoco_py.MjViewer(env.unwrapped.sim)

    success_count = 0
    total_steps = []

    for episode in range(episodes):
        print(f"\n--- Evaluation Episode {episode+1}/{episodes} ---")
        obs = env.reset()
        viewer.update_sim(env.unwrapped.sim)

        goal_pos, goal_rot = set_goal_for_task(env, task)
        
        bob_lstm_state = bob.policy_old.get_initial_lstm_state(1, eval_device)
        done = False
        succeeded = False
        
        num_actual_objects = env.unwrapped.mujoco_simulation.num_objects

        for step_count in range(max_steps):
            bob_obs_with_goal = obs.copy()
            bob_obs_with_goal['goal_obj_pos'] = goal_pos[:num_actual_objects]
            bob_obs_with_goal['goal_obj_rot'] = goal_rot[:num_actual_objects]
            bob_obs_with_goal['rel_goal_obj_pos'] = goal_pos[:num_actual_objects] - obs['obj_pos'][:num_actual_objects]
            bob_obs_with_goal['rel_goal_obj_rot'] = normalize_angles(goal_rot[:num_actual_objects] - obs['obj_rot'][:num_actual_objects])
            
            action, _, _, bob_lstm_state = bob.select_action(bob_obs_with_goal, bob_lstm_state, deterministic=True)
            obs, _, done, _ = env.step(action)
            viewer.render()

            dist_pos = np.linalg.norm(obs['obj_pos'][:num_actual_objects] - goal_pos[:num_actual_objects], axis=-1)
            dist_rot = np.linalg.norm(normalize_angles(obs['obj_rot'][:num_actual_objects] - goal_rot[:num_actual_objects]), axis=-1)
            is_success = np.all((dist_pos < GoalValidationConfig.SUCCESS_THRESHOLD_POS) & 
                                (dist_rot < GoalValidationConfig.SUCCESS_THRESHOLD_ROT))
            
            #if np.all(dist_pos < 0.04):
                #pdb.set_trace()
            
            if is_success:
                print(f"Success! Achieved goal in {step_count+1} steps.")
                success_count += 1
                total_steps.append(step_count + 1)
                succeeded = True
                break
            
            if done:
                print(f"Episode terminated by environment at step {step_count+1}.")
                break
        
        if not succeeded and not done:
            print(f"Failure. Agent did not achieve goal within {max_steps} steps.")
        elif not succeeded and done:
             print("Failure. Agent did not achieve the goal before episode terminated.")

        
        time.sleep(1)

    env.close()

    success_rate = (success_count / episodes) * 100
    avg_steps = np.mean(total_steps) if total_steps else float('inf')
    
    print("\n--- Evaluation Summary ---")
    print(f"Task: {task}")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{episodes})")
    print(f"Average Steps to Success: {avg_steps:.2f}")


if __name__ == '__main__':
    evaluate()
