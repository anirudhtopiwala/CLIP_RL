import torch
import torch.nn as nn


class CLIPPolicy(nn.Module):
    def __init__(self, embedding_dim, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, clip_feat):
        return self.mlp(clip_feat)