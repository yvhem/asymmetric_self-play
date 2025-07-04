import torch
import numpy as np

from config import PPOConfig

#class RunningMeanStd:
#    def __init__(self, epsilon=1e-4, shape=()):
#        self.mean = np.zeros(shape, 'float64')
#        self.var = np.ones(shape, 'float64')
#        self.count = epsilon
#
#    def update(self, x):
#        batch_mean = np.mean(x, axis=0)
#        batch_var = np.var(x, axis=0)
#        batch_count = x.shape[0]
#        self.update_from_moments(batch_mean, batch_var, batch_count)
#
#    def update_from_moments(self, batch_mean, batch_var, batch_count):
#        delta = batch_mean - self.mean
#        tot_count = self.count + batch_count
#        new_mean = self.mean + delta * batch_count / tot_count
#        m_a = self.var * self.count
#        m_b = batch_var * batch_count
#        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
#        new_var = M2 / tot_count
#        self.mean = new_mean
#        self.var = new_var
#        self.count = tot_count

def compute_gae(rewards, values, is_terminals, next_value):
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0
    values_detached = values.detach()
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - is_terminals[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - is_terminals[t]
            next_val = values_detached[t + 1]
        
        delta = rewards[t] + PPOConfig.DISCOUNT_FACTOR * next_val * next_non_terminal - values_detached[t]
        advantages[t] = last_gae_lam = delta + PPOConfig.DISCOUNT_FACTOR * PPOConfig.GAE_LAMBDA * next_non_terminal * last_gae_lam
    return advantages

def get_obs_space_dims(env):
    obs = env.reset()
    max_num_objects = env.unwrapped.parameters.simulation_params.max_num_objects
    if max_num_objects is None:
        max_num_objects = obs['obj_pos'].shape[0]

    object_feature_dim = (
        obs['obj_pos'].shape[1] + obs['obj_rot'].shape[1] +
        obs['obj_vel_pos'].shape[1] + obs['obj_vel_rot'].shape[1] +
        obs['obj_gripper_contact'].shape[1] + obs['obj_rel_pos'].shape[1]
    )
    goal_feature_dim = (
        obs['goal_obj_pos'].shape[1] + obs['goal_obj_rot'].shape[1] +
        obs['rel_goal_obj_pos'].shape[1] + obs['rel_goal_obj_rot'].shape[1]
    )

    dims = {
        'robot_joint_pos': obs['robot_joint_pos'].shape[0],
        'gripper_pos': obs['gripper_pos'].shape[0],
        'object_state': object_feature_dim,
        'goal_state': goal_feature_dim,
        'max_num_objects': max_num_objects
    }
    return dims

def process_obs_for_policy(obs, max_num_objects):
    num_objects = obs['obj_pos'].shape[0]

    def pad(arr, feature_dim):
        padding_rows = max_num_objects - arr.shape[0]
        if padding_rows > 0:
            padding = np.zeros((padding_rows, feature_dim), dtype=np.float32)
            return np.concatenate([arr, padding], axis=0)
        return arr[:max_num_objects]

    object_state_raw = np.concatenate([
        obs['obj_pos'], obs['obj_rot'], obs['obj_vel_pos'],
        obs['obj_vel_rot'], obs['obj_gripper_contact'],
        obs['obj_rel_pos']
    ], axis=-1)
    object_state_padded = pad(object_state_raw, object_state_raw.shape[1])

    object_mask = np.zeros(max_num_objects, dtype=np.float32)
    object_mask[:num_objects] = 1.0

    processed = {
        'robot_state': np.concatenate([obs['robot_joint_pos'], obs['gripper_pos']]),
        'object_state': object_state_padded,
        'object_mask': object_mask
    }

    if 'goal_obj_pos' in obs:
        goal_state_raw = np.concatenate([
            obs['goal_obj_pos'], obs['goal_obj_rot'],
            obs['rel_goal_obj_pos'], obs['rel_goal_obj_rot']
        ], axis=-1)
        goal_state_padded = pad(goal_state_raw, goal_state_raw.shape[1])
        processed['goal_state'] = goal_state_padded

    return processed
