"""
AntBulletEnv-v0 with Soft Actor-Critic (SAC)
=============================================
Training script to solve the AntBulletEnv-v0 environment using SAC.
Target: Average total reward of over 2500 over 100 consecutive episodes.

Based on the work from:
  https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms
  Original SAC code based on Pranjal Tandon's code (https://github.com/pranz24)

References:
  SAC: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor
  https://arxiv.org/abs/1801.01290
"""

import gym
import pybullet_envs
import numpy as np
import torch
import time
import os
from collections import deque

from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory

def train_sac(env_name='AntBulletEnv-v0',
              seed=0,
              hidden_size=256,
              lr=0.0001,
              gamma=0.99,
              tau=0.005,
              alpha=0.2,
              batch_size=256,
              replay_size=1000000,
              start_steps=10000,
              max_episodes=3000,
              max_steps=1000,
              target_score=2500,
              updates_per_step=1,
              print_every=1,
              save_dir='dir_ant_lr0001-sc2500'):
    """
    Train a SAC agent on the AntBulletEnv-v0 environment.

    Args:
        env_name: Name of the gym environment.
        seed: Random seed.
        hidden_size: Hidden layer size for networks.
        lr: Learning rate.
        gamma: Discount factor.
        tau: Soft update coefficient.
        alpha: Entropy regularization coefficient.
        batch_size: Batch size for training.
        replay_size: Maximum replay buffer size.
        start_steps: Number of random exploration steps before training.
        max_episodes: Maximum number of episodes.
        max_steps: Maximum steps per episode.
        target_score: Target average score to solve the environment.
        updates_per_step: Number of gradient updates per environment step.
        print_every: Print frequency (episodes).
        save_dir: Directory to save model checkpoints.
    """

    # Create environment
    env = gym.make(env_name)
    env.seed(seed)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment info
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    print(f"Environment: {env_name}")
    print(f"State space dimension: {state_dim}")
    print(f"Action space dimension: {action_space}")
    print(f"Max steps in episode: {max_steps}")

    # Create agent
    agent = soft_actor_critic_agent(
        num_inputs=state_dim,
        action_space=action_space,
        device=device,
        hidden_size=hidden_size,
        seed=seed,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha
    )

    # Replay Memory
    memory = ReplayMemory(seed, replay_size)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    total_steps = 0
    scores_window = deque(maxlen=100)
    scores_all = []
    avg_scores_all = []
    start_time = time.time()
    solved = False

    print("\n--- Training Started ---\n")
    print(f"Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  Tau: {tau}")
    print(f"  Alpha: {alpha}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Replay size: {replay_size}")
    print(f"  Start steps: {start_steps}")
    print(f"  Target score: {target_score}")
    print()

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(max_steps):
            # Select action
            if total_steps < start_steps:
                action = env.action_space.sample()  # Random exploration
            else:
                action = agent.select_action(state)

            # Step environment
            next_state, reward, done, info = env.step(action)
            episode_steps += 1
            total_steps += 1
            episode_reward += reward

            # Store in replay buffer (mask: 1 if not done, 0 if done)
            mask = 1.0 if not done else 0.0
            memory.push(state, action, reward, next_state, mask)

            state = next_state

            # Update agent
            if len(memory) > batch_size:
                for _ in range(updates_per_step):
                    agent.update_parameters(memory, batch_size)

            if done:
                break

        scores_window.append(episode_reward)
        scores_all.append(episode_reward)
        avg_score = np.mean(scores_window)
        avg_scores_all.append(avg_score)

        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        if episode % print_every == 0:
            print(f"Ep.: {episode}, Total Steps: {total_steps}, "
                  f"Ep.Steps: {episode_steps}, Score: {episode_reward:.2f}, "
                  f"Avg.Score: {avg_score:.2f}, Time: {elapsed_str}")

        # Check if solved
        if avg_score >= target_score and len(scores_window) >= 100:
            print(f"\nSolved environment with Avg Score: {avg_score}")
            print(f"Solved in episode {episode} after {elapsed_str}")

            # Save model
            agent.save_model(
                env_name,
                suffix=f"solved_ep{episode}",
                actor_path=os.path.join(save_dir, 'sac_actor_solved.pth'),
                critic_path=os.path.join(save_dir, 'sac_critic_solved.pth')
            )
            solved = True
            break

        # Periodic save
        if episode % 200 == 0:
            agent.save_model(
                env_name,
                suffix=f"ep{episode}",
                actor_path=os.path.join(save_dir, f'sac_actor_ep{episode}.pth'),
                critic_path=os.path.join(save_dir, f'sac_critic_ep{episode}.pth')
            )

    env.close()

    if not solved:
        print(f"\nTraining completed without solving. Best Avg Score: {max(avg_scores_all):.2f}")

    # Save training data
    np.save(os.path.join(save_dir, 'scores.npy'), np.array(scores_all))
    np.save(os.path.join(save_dir, 'avg_scores.npy'), np.array(avg_scores_all))

    return scores_all, avg_scores_all


def test_agent(env_name='AntBulletEnv-v0',
               actor_path='dir_ant_lr0001-sc2500/sac_actor_solved.pth',
               critic_path='dir_ant_lr0001-sc2500/sac_critic_solved.pth',
               seed=0,
               hidden_size=256,
               num_episodes=10,
               render=True):
    """
    Test a trained SAC agent.
    """
    env = gym.make(env_name)
    env.seed(seed)

    if render:
        env.render(mode="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    agent = soft_actor_critic_agent(
        num_inputs=state_dim,
        action_space=action_space,
        device=device,
        hidden_size=hidden_size,
        seed=seed,
        lr=0.0001,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    )

    agent.load_model(actor_path, critic_path)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, eval=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        print(f"Episode {episode}: Score = {episode_reward:.2f}")

    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SAC for AntBulletEnv-v0')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='train or test mode')
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0',
                        help='Environment name')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Entropy regularization coefficient')
    parser.add_argument('--target_score', type=float, default=2500,
                        help='Target average score')
    parser.add_argument('--max_episodes', type=int, default=3000,
                        help='Maximum training episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    parser.add_argument('--actor_path', type=str,
                        default='dir_ant_lr0001-sc2500/sac_actor_solved.pth',
                        help='Path to actor model for testing')
    parser.add_argument('--critic_path', type=str,
                        default='dir_ant_lr0001-sc2500/sac_critic_solved.pth',
                        help='Path to critic model for testing')

    args = parser.parse_args()

    if args.mode == 'train':
        train_sac(
            env_name=args.env,
            seed=args.seed,
            lr=args.lr,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            target_score=args.target_score,
            max_episodes=args.max_episodes,
        )
    else:
        test_agent(
            env_name=args.env,
            actor_path=args.actor_path,
            critic_path=args.critic_path,
            seed=args.seed,
            hidden_size=args.hidden_size,
            render=args.render,
        )
