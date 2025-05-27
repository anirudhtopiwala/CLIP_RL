import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGL.so.1"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

import torch

from env.robosuite_env import create_env
from models.clip_policy import CLIPPolicy
from utils.clip_utils import get_image_embedding, get_text_embedding

env = create_env()
obs = env.reset()
print("Acion Dim:", env.action_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for compatibility

proprio_dim = env.observation_spec()["robot0_proprio-state"].shape[0]
embedding_dim = 512 + 512 + proprio_dim  # image + text + proprio

policy = CLIPPolicy(input_dim=embedding_dim, action_dim=env.action_dim).to(device)
policy.load_state_dict(
    torch.load(
        "models/clip_policy_episodes_100_20250526_201626.pt", map_location=device
    )
)
policy.eval()
print("Policy loaded successfully.")

text_goal = "pick up the cube"
goal_feat = get_text_embedding(text_goal)

done = False
while not done:
    img_feat = get_image_embedding(obs["frontview_image"]).to(device)
    proprio = torch.tensor(
        obs["robot0_proprio-state"], dtype=torch.float32, device=device
    )
    state_feat = torch.cat([img_feat, goal_feat, proprio], dim=-1).unsqueeze(0)

    with torch.no_grad():
        action_mean, _ = policy(state_feat)
    action = action_mean.squeeze(0).cpu().numpy()

    obs, reward, done, _ = env.step(action)
    print("Reward:", reward)
