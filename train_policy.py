import argparse
import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGL.so.1"

import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env.robosuite_env import create_env
from models.clip_policy import CLIPPolicy
from utils.clip_utils import get_image_embedding, get_text_embedding


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_policy(
    policy, optimizer, env, device, writer, num_episodes=1000, grad_clip=1.0
):
    policy.train()
    print("Training CLIPPolicy...")
    for episode in range(num_episodes):
        obs = env.reset()
        text_goal = "pick up the red cube"  # You can randomize this for goal-conditioned training
        goal_feat = get_text_embedding(text_goal)
        goal_feat = goal_feat.to(device)

        log_probs = []
        rewards = []
        done = False
        while not done:
            frame = obs["frontview_image"]
            img_feat = get_image_embedding(frame).to(device)
            state_feat = img_feat + goal_feat
            state_feat = state_feat.unsqueeze(0)

            action_mean = policy(state_feat)
            action_dist = torch.distributions.Normal(action_mean, 1.0)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()

            obs, reward, done, _ = env.step(action.squeeze().cpu().detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)

        returns = compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        log_probs = torch.stack(log_probs)
        if log_probs.shape != returns.shape:
            min_len = min(log_probs.shape[0], returns.shape[0])
            log_probs = log_probs[:min_len]
            returns = returns[:min_len]
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        total_reward = sum(rewards)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

        # TensorBoard logging
        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Loss/policy_loss", loss.item(), episode)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train CLIPPolicy with Robosuite")
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint to warm start",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="clip_policy.pt",
        help="Path to save trained model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = create_env(is_renderer=False)
    print("Action Dim:", env.action_dim)
    policy = CLIPPolicy(embedding_dim=512, action_dim=env.action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        policy.load_state_dict(torch.load(args.model_path, map_location=device))

    writer = SummaryWriter()
    train_policy(policy, optimizer, env, device, writer, num_episodes=args.num_episodes)
    writer.close()

    # Save the trained model
    torch.save(policy.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
