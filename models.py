import torch
import torch.nn as nn
import torch.nn.functional as F

class PermutationInvariantEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x, mask):
        embedded = self.mlp(x)
        mask_expanded = mask.unsqueeze(-1).expand_as(embedded)
        masked_embedded = embedded * mask_expanded
        
        # max pooling for permutation invariance
        pooled, _ = torch.max(masked_embedded, dim=1)
        return pooled

class ActorCritic(nn.Module):
    """The policy and value network for both Alice and Bob."""
    def __init__(self, obs_space_dims, actor_output_dim):
        super().__init__()
        self.obs_space_dims = obs_space_dims

        # robot state embedding
        robot_state_dim = obs_space_dims['robot_joint_pos'] + obs_space_dims['gripper_pos']
        self.robot_state_embedding = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # object state embedding
        object_feature_dim = obs_space_dims['object_state']
        goal_feature_dim = obs_space_dims['goal_state']
        self.obj_embedding = PermutationInvariantEmbedding(object_feature_dim + goal_feature_dim, 512)
        
        # MLP
        combined_feature_dim = 256 + 512  # robot_emb + obj_emb
        self.mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # out dim for LSTM
        )
        
        # LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        
        # output heads
        self.actor_head = nn.Linear(256, actor_output_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs, lstm_state):
        # robot state embedding
        robot_emb = self.robot_state_embedding(obs['robot_state'])

        # object state embedding
        if 'goal_state' in obs: # bob
            obj_goal_concat = torch.cat([obs['object_state'], obs['goal_state']], dim=-1)
        else:  # alice
            b, n, _ = obs['object_state'].shape
            goal_feature_dim = self.obs_space_dims['goal_state']
            goal_placeholder = torch.zeros(b, n, goal_feature_dim, device=obs['object_state'].device)
            obj_goal_concat = torch.cat([obs['object_state'], goal_placeholder], dim=-1)

        obj_emb = self.obj_embedding(obj_goal_concat, obs['object_mask'])
        
        # concat features
        combined_features = torch.cat([robot_emb, obj_emb], dim=-1)
            
        # MLP and LSTM processing
        x = self.mlp(F.relu(combined_features))
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, new_lstm_state = self.lstm(x, lstm_state)
        
        if lstm_out.shape[1] == 1:
            lstm_out = lstm_out.squeeze(1)
            
        # output heads
        action_logits = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)
        
        return action_logits, value, new_lstm_state

    def get_initial_lstm_state(self, batch_size, device):
        return (torch.zeros(1, batch_size, 256, device=device),
                torch.zeros(1, batch_size, 256, device=device))