import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pickle

from models import ActorCritic
from config import PPOConfig, ActionConfig, DEVICE
from utils import compute_gae, process_obs_for_policy#, RunningMeanStd
from robogym.utils.rotation import normalize_angles

# store trajectories for PPO updates
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def __len__(self):
        return len(self.actions)

# store demos for ABC
class ABCBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.goals = []
        self.logprobs = []

    def __len__(self):
        return len(self.actions)

class PPOAgent:
    def __init__(self, obs_space_dims, action_dim, is_alice):
        self.is_alice = is_alice
        self.action_dims = ActionConfig.ACTION_DIMS
        self.action_bins = ActionConfig.ACTION_BINS
        self.max_num_objects = obs_space_dims['max_num_objects']
        
        self.policy = ActorCritic(obs_space_dims, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=PPOConfig.LEARNING_RATE)

        self.policy_old = ActorCritic(obs_space_dims, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()

        #self.obs_rms = {
        #    'robot_state': RunningMeanStd(shape=(obs_space_dims['robot_joint_pos'] + obs_space_dims['gripper_pos'],)),
        #    'object_state': RunningMeanStd(shape=(obs_space_dims['object_state'],)),
        #    'goal_state': RunningMeanStd(shape=(obs_space_dims['goal_state'],))
        #}

    #def _normalize_obs(self, obs_dict):
    #    norm_obs = {}
    #    for key, obs in obs_dict.items():
    #        if key in self.obs_rms:
    #            var_sqrt = np.sqrt(self.obs_rms[key].var + 1e-8)
    #            norm_obs[key] = np.clip((obs - self.obs_rms[key].mean) / var_sqrt, -10, 10)
    #        else:
    #            norm_obs[key] = obs
    #    return norm_obs

    def select_action(self, obs, lstm_state, deterministic=False):
        processed_obs = process_obs_for_policy(obs, self.max_num_objects)
        #normalized_processed_obs = self._normalize_obs(processed_obs)
        
        device = next(self.policy_old.parameters()).device
        #obs_tensor = {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in normalized_processed_obs.items()}
        obs_tensor = {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in processed_obs.items()}
        h_0, c_0 = lstm_state
        lstm_state_on_device = (h_0.to(device), c_0.to(device))

        with torch.no_grad():
            logits, value, new_lstm_state = self.policy_old(obs_tensor, lstm_state_on_device)

        logits = logits.view(-1, self.action_dims, self.action_bins)
        dist = Categorical(logits=logits) 

        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().numpy().squeeze(axis=0), value.item(), action_logprob.item(), new_lstm_state

    def update(self, memory, abc_memory=None):
        loss_info = {}

        # prepare RL data
        if len(memory) > 0:
            processed_states_raw = [process_obs_for_policy(s, self.max_num_objects) for s in memory.states]
            
            # update running mean and std
            #for key in self.obs_rms.keys():
            #    if key in processed_states_raw[0]:
            #        batch_data = np.stack([s[key] for s in processed_states_raw])
            #        if key in ['object_state', 'goal_state']:
            #            mask = np.stack([s['object_mask'] for s in processed_states_raw]).astype(bool)
            #            if np.any(mask): self.obs_rms[key].update(batch_data[mask])
            #        else:
            #            self.obs_rms[key].update(batch_data)

            # process RL data for training
            rewards_t = torch.tensor(memory.rewards, dtype=torch.float32).to(DEVICE)
            is_terminals_t = torch.tensor(memory.is_terminals, dtype=torch.float32).to(DEVICE)
            values_t = torch.tensor(memory.values, dtype=torch.float32).to(DEVICE)
            next_value = 0.0 if memory.is_terminals[-1] else values_t[-1].item()
            advantages = compute_gae(rewards_t, values_t, is_terminals_t, next_value)
            returns = advantages + values_t
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # normalized_states_raw = [self._normalize_obs(s) for s in processed_states_raw]
            # old_states = {k: torch.from_numpy(np.stack([s[k] for s in normalized_states_raw])).float().to(DEVICE) for k in normalized_states_raw[0]}
            old_states = {k: torch.from_numpy(np.stack([s[k] for s in processed_states_raw])).float().to(DEVICE) for k in processed_states_raw[0]}
            old_actions = torch.from_numpy(np.stack(memory.actions)).long().to(DEVICE)
            old_logprobs = torch.from_numpy(np.stack(memory.logprobs)).float().to(DEVICE)
        
        # prepare ABC data
        abc_data = self._prepare_abc_data(abc_memory) if not self.is_alice and abc_memory and len(abc_memory) > 0 else None

        # PPO update loop
        for _ in range(PPOConfig.SAMPLE_REUSE):
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=DEVICE)

            # RL loss
            if len(memory) > 0:
                h_0, c_0 = self.policy.get_initial_lstm_state(old_actions.shape[0], DEVICE)
                logits, values, _ = self.policy(old_states, (h_0, c_0))
                
                values = values.squeeze(-1)
                logits = logits.view(old_actions.shape[0], self.action_dims, self.action_bins)
                dist = Categorical(logits=logits) 
                
                log_ratio = dist.log_prob(old_actions).sum(dim=-1) - old_logprobs.detach()
                ratios = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - PPOConfig.PPO_CLIPPING_EPSILON, 1 + PPOConfig.PPO_CLIPPING_EPSILON) * advantages
                rl_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * F.mse_loss(values, returns)
                entropy_loss = -PPOConfig.ENTROPY_COEFFICIENT * dist.entropy().mean()
                
                total_loss += rl_loss + PPOConfig.VALUE_LOSS_WEIGHT * value_loss + entropy_loss
                
                if 'rl_loss' not in loss_info:
                    loss_info.update({'rl_loss': rl_loss.item(), 'value_loss': value_loss.item(), 'entropy': -entropy_loss.item() / PPOConfig.ENTROPY_COEFFICIENT})

            # ABC loss
            if abc_data:
                abc_loss = self._compute_abc_loss(abc_data)
                if not torch.isnan(abc_loss):
                    total_loss += PPOConfig.ABC_LOSS_WEIGHT * abc_loss
                    if 'abc_loss' not in loss_info: loss_info['abc_loss'] = abc_loss.item()

            if total_loss.requires_grad:
                total_loss.backward()
                #nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_info

    def _prepare_abc_data(self, abc_memory):
        demo_states_raw = [process_obs_for_policy(s, self.max_num_objects) for s in abc_memory.states]
        
        # add goal information to states
        for i, processed_state in enumerate(demo_states_raw):
            goal_obs = abc_memory.goals[i]
            state_obs = abc_memory.states[i]
            num_objs = goal_obs['goal_obj_pos'].shape[0]

            rel_goal_pos = goal_obs['goal_obj_pos'] - state_obs['obj_pos'][:num_objs]
            rel_goal_rot = normalize_angles(goal_obs['goal_obj_rot'] - state_obs['obj_rot'][:num_objs])
            
            goal_state_raw = np.concatenate([goal_obs['goal_obj_pos'], goal_obs['goal_obj_rot'], rel_goal_pos, rel_goal_rot], axis=-1)
            
            padding_rows = self.max_num_objects - goal_state_raw.shape[0]
            if padding_rows > 0:
                padding = np.zeros((padding_rows, goal_state_raw.shape[1]), dtype=np.float32)
                padded_goal = np.concatenate([goal_state_raw, padding], axis=0)
            else:
                padded_goal = goal_state_raw[:self.max_num_objects]

            processed_state['goal_state'] = padded_goal
        
        demo_states_with_goals = {k: torch.from_numpy(np.stack([s[k] for s in demo_states_raw])).float().to(DEVICE) for k in demo_states_raw[0]}
        demo_actions = torch.from_numpy(np.stack(abc_memory.actions)).long().to(DEVICE)
        old_demo_logprobs = torch.from_numpy(np.stack(abc_memory.logprobs)).float().to(DEVICE)

        return {"demo_states": demo_states_with_goals, "demo_actions": demo_actions, "old_logprobs": old_demo_logprobs}

    def _compute_abc_loss(self, abc_data):
        h_0, c_0 = self.policy.get_initial_lstm_state(abc_data['demo_actions'].shape[0], DEVICE)
        logits, _, _ = self.policy(abc_data['demo_states'], (h_0, c_0))
        logits = logits.view(-1, self.action_dims, self.action_bins)
        
        dist = Categorical(logits=logits)
        log_ratio = dist.log_prob(abc_data['demo_actions']).sum(dim=-1) - abc_data['old_logprobs'].detach()
        ratios = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
        
        advantage = 1.0 
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - PPOConfig.ABC_CLIPPING_EPSILON, 1 + PPOConfig.ABC_CLIPPING_EPSILON) * advantage
        
        return -torch.min(surr1, surr2).mean()

    def save_models(self, path):
        torch.save(self.policy.state_dict(), f"{path}_policy.pth")
        #with open(f"{path}_obs_rms.pkl", 'wb') as f:
        #    pickle.dump(self.obs_rms, f)

    def load_models(self, path):
        try:
            self.policy.load_state_dict(torch.load(f"{path}_policy.pth", map_location=DEVICE))
            self.policy_old.load_state_dict(self.policy.state_dict())
            print(f"Loaded policy from {path}_policy.pth")
        except FileNotFoundError:
            print(f"Policy model not found at {path}_policy.pth => training from scratch")
        except Exception as e:
            print(f"Error loading policy model from {path}_policy.pth: {e} => training from scratch")

        #try:
        #    with open(f"{path}_obs_rms.pkl", 'rb') as f:
        #        self.obs_rms = pickle.load(f)
        #    print(f"Loaded observation normalizer from {path}_obs_rms.pkl")
        #except FileNotFoundError:
        #    print(f"Observation normalizer not found at {path}_obs_rms.pkl => using default")
        #except Exception as e:
        #    print(f"Error loading observation normalizer from {path}_obs_rms.pkl: {e} => using default")