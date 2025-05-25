import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGL.so.1"

import torch

from env.robosuite_env import create_env
from models.clip_policy import CLIPPolicy
from utils.clip_utils import get_image_embedding, get_text_embedding

env = create_env()
obs = env.reset()
print("Acion Dim:", env.action_dim)
policy = CLIPPolicy(embedding_dim=512, action_dim=env.action_dim).eval()
policy.load_state_dict(torch.load("models/clip_policy_1k.pt", map_location="cpu"))
policy.eval()

text_goal = "pick up the red cube"
goal_feat = get_text_embedding(text_goal)

done = False
while not done:
    frame = obs["frontview_image"]
    img_feat = get_image_embedding(frame)

    state_feat = img_feat + goal_feat
    state_feat = state_feat.unsqueeze(0)

    action = policy(state_feat).squeeze().detach().numpy()
    print("Action:", action)
    obs, reward, done, _ = env.step(action)
    print("Reward:", reward)
