import torch.nn as nn


class CLIPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        # Actions are represented as means in a Gaussian distribution.
        self.action_head = nn.Linear(256, action_dim)
        # Value head for estimating the value of the state.
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        action_mean = self.action_head(x)
        value = self.value_head(x)
        return action_mean, value
