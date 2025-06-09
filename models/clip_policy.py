import torch
import torch.nn as nn


class CLIPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()

        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.feature_net(x)
        action_mean = self.policy_head(features)
        action_mean = torch.tanh(action_mean)  # Bound actions to [-1, 1]
        value = self.value_head(features)
        return action_mean, value
