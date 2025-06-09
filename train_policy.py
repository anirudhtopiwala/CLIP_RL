import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGL.so.1"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"

import datetime
import random
import time

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


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation (GAE) for computing advantages and returns."""
    advantages = []
    gae = 0
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = (
            rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        )
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


def collect_trajectory(
    env, policy, device, max_steps=512, exploration_std=0.1, action_repeat=4
):
    obs = env.reset()
    text_goal = "pick up the red cube"
    goal_feat = get_text_embedding(text_goal).to(device)

    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    for _ in range(max_steps):
        img_feat = get_image_embedding(obs["frontview_image"]).to(device)
        proprio = torch.tensor(
            obs["robot0_proprio-state"], dtype=torch.float32, device=device
        )
        state_feat = torch.cat([img_feat, goal_feat, proprio], dim=-1).unsqueeze(0)

        with torch.no_grad():
            action_mean, value = policy(state_feat)
            action_dist = torch.distributions.Normal(action_mean, exploration_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

        # Apply action repeat
        total_reward = 0
        for _ in range(action_repeat):
            next_obs, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                break

        states.append(state_feat.squeeze(0).cpu())
        actions.append(action.squeeze(0).cpu())
        log_probs.append(log_prob.cpu())
        rewards.append(total_reward)
        dones.append(float(done))
        values.append(value.squeeze(0).cpu().item())

        if done:
            break
    return states, actions, log_probs, rewards, dones, values


def ppo_update(
    policy,
    optimizer,
    device,
    batch,
    ppo_epochs=6,
    mini_batch_size=64,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    grad_clip=1.0,
    exploration_std=0.1,
):
    states, actions, old_log_probs, returns, advantages = batch
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    old_log_probs = torch.stack(old_log_probs).to(device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0
    for _ in range(ppo_epochs):
        idx = torch.randperm(states.size(0))
        for start in range(0, states.size(0), mini_batch_size):
            end = start + mini_batch_size
            mb_idx = idx[start:end]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            action_mean, value = policy(mb_states)
            action_dist = torch.distributions.Normal(action_mean, exploration_std)
            log_probs = action_dist.log_prob(mb_actions).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1).mean()

            ratio = (log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                * mb_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - value.squeeze(-1)).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / (ppo_epochs * max(1, states.size(0) // mini_batch_size))


def train_policy(
    policy, optimizer, env, device, writer, num_episodes, model_name, grad_clip=1.0
):
    policy.train()
    print("Training CLIPPolicy with PPO...")

    # Curriculum learning parameters
    initial_exploration_std = 0.5  # Start with high exploration
    final_exploration_std = 0.05  # End with low exploration
    exploration_decay = (initial_exploration_std - final_exploration_std) / num_episodes
    current_exploration_std = initial_exploration_std

    for episode in range(num_episodes):
        episode_start_time = time.time()

        # Decay exploration as training progresses
        current_exploration_std = max(
            final_exploration_std, initial_exploration_std - episode * exploration_decay
        )
        print(
            f"Episode {episode+1}/{num_episodes}, Exploration Std: {current_exploration_std:.4f}"
        )
        states, actions, log_probs, rewards, dones, values = collect_trajectory(
            env,
            policy,
            device,
            max_steps=512,
            exploration_std=current_exploration_std,
            action_repeat=4,
        )
        advantages, returns = compute_gae(rewards, values, dones)
        loss = ppo_update(
            policy,
            optimizer,
            device,
            batch=(states, actions, log_probs, returns, advantages),
            grad_clip=grad_clip,
            exploration_std=current_exploration_std,
        )
        total_reward = sum(rewards)
        episode_duration = time.time() - episode_start_time
        print(
            f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Loss: {loss:.4f}, Duration: {episode_duration:.2f}s"
        )
        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Loss/ppo_loss", loss, episode)
        writer.add_scalar("Episode/Duration", episode_duration, episode)
        writer.add_scalar("Training/exploration_std", current_exploration_std, episode)

        if (episode + 1) % 100 == 0:
            checkpoint_path = f"models/{model_name}_checkpoint_ep{episode+1}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    return policy


def train_policy_non_ppo(
    policy, optimizer, env, device, writer, num_episodes=1000, grad_clip=1.0
):
    policy.train()
    print("Training CLIPPolicy...")
    total_start_time = time.time()
    for episode in range(num_episodes):
        episode_start_time = time.time()
        obs = env.reset()
        text_goal = "pick up the red cube"  # You can randomize this for goal-conditioned training
        goal_feat = get_text_embedding(text_goal)
        goal_feat = goal_feat.to(device)

        log_probs = []
        rewards = []
        done = False
        steps_in_episode = 0
        while not done:
            frame = obs["frontview_image"]
            img_feat = get_image_embedding(frame).to(device)
            state_feat = img_feat + goal_feat
            state_feat = state_feat.unsqueeze(0)

            action_mean = policy(state_feat)
            action_dist = torch.distributions.Normal(action_mean, 0.1)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()

            obs, reward, done, _ = env.step(action.squeeze().cpu().detach().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)
            steps_in_episode += 1

        episode_duration = time.time() - episode_start_time
        print(
            f"Episode {episode+1} finished after {steps_in_episode} steps in {episode_duration:.2f} seconds."
        )
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
        writer.add_scalar("Episode/Duration", episode_duration, episode)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train CLIPPolicy with Robosuite")
    parser.add_argument(
        "--num_episodes", type=int, default=250, help="Number of training episodes"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint to warm start",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="clip_policy",
        help="Name of the policy model. (for saving purposes)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device", device)

    env = create_env(is_renderer=False)
    proprio_dim = env.observation_spec()["robot0_proprio-state"].shape[0]
    embedding_dim = 512 + 512 + proprio_dim  # image + text + proprio
    policy = CLIPPolicy(input_dim=embedding_dim, action_dim=env.action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        policy.load_state_dict(torch.load(args.model_path, map_location=device))

    writer = SummaryWriter()
    train_policy(
        policy,
        optimizer,
        env,
        device,
        writer,
        num_episodes=args.num_episodes,
        model_name=args.model_name,
    )
    writer.close()

    # Save the trained model with time stamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/{args.model_name}_episodes_{args.num_episodes}_{timestamp}.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
