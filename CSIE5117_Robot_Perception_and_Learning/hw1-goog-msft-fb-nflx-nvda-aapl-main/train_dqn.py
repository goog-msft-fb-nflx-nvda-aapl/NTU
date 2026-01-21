import os
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.env import CartPoleEnv
from src.dqn import DQNAgent, DQNController


def train_dqn(
    num_episodes: int = 500,
    max_steps: int = 1000,
    eval_freq: int = 50,
    save_path: str = "models/dqn_cartpole.pt",
    config_path: str = "configs/dqn.yaml"
):
    """Train DQN agent on CartPole"""
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Create environment
    env = CartPoleEnv(
        max_episode_steps=config.env.max_episode_steps,
        dt=config.env.dt,
        force_mag=config.env.force_mag,
        render_mode=None
    )
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=config.agent.hidden_dims,
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        epsilon_start=config.agent.epsilon_start,
        epsilon_end=config.agent.epsilon_end,
        epsilon_decay=config.agent.epsilon_decay,
        buffer_size=config.agent.buffer_size,
        batch_size=config.agent.batch_size,
        target_update_freq=config.agent.target_update_freq,
        device=config.agent.device
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_episodes = []
    
    print("="*60)
    print("Training DQN on CartPole")
    print("="*60)
    
    for episode in range(num_episodes):
        obs = env.reset(seed=episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.get_action(obs, training=True)
            next_obs, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train
            loss = agent.train_step()
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # End episode
        agent.end_episode()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1:4d}: "
                  f"Reward={episode_reward:6.1f}, "
                  f"Length={episode_length:4d}, "
                  f"Avg10={avg_reward:6.1f}, "
                  f"Epsilon={agent.epsilon:.3f}, "
                  f"Buffer={len(agent.replay_buffer)}")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_dqn(agent, env, num_eval=5)
            eval_rewards.append(eval_reward)
            eval_episodes.append(episode + 1)
            print(f"{'='*60}")
            print(f"Evaluation at episode {episode+1}: Avg Reward = {eval_reward:.1f}")
            print(f"{'='*60}")
    
    env.close()
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Save training statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes,
        'losses': agent.losses
    }
    np.save(save_path.replace('.pt', '_stats.npy'), stats)
    
    # Plot results
    plot_training_results(stats, save_path.replace('.pt', '_training.png'))
    
    return agent, stats


def evaluate_dqn(agent: DQNAgent, env: CartPoleEnv, num_eval: int = 10, seed: int = 42):
    """Evaluate DQN agent"""
    eval_rewards = []
    
    for i in range(num_eval):
        obs = env.reset(seed=seed + i)
        episode_reward = 0
        
        for _ in range(1000):
            action = agent.get_action(obs, training=False)
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_results(stats, save_path):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'], alpha=0.3, label='Raw')
    window = 50
    if len(stats['episode_rewards']) >= window:
        smoothed = np.convolve(stats['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(stats['episode_rewards'])), 
                       smoothed, label=f'MA({window})', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(stats['episode_lengths'], alpha=0.3, label='Raw')
    if len(stats['episode_lengths']) >= window:
        smoothed = np.convolve(stats['episode_lengths'], 
                               np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(stats['episode_lengths'])), 
                       smoothed, label=f'MA({window})', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Evaluation rewards
    if stats['eval_rewards']:
        axes[1, 0].plot(stats['eval_episodes'], stats['eval_rewards'], 
                       marker='o', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Avg Evaluation Reward')
        axes[1, 0].set_title('Evaluation Performance')
        axes[1, 0].grid(True)
    
    # Training loss
    if stats['losses']:
        axes[1, 1].plot(stats['losses'], alpha=0.3, label='Raw')
        window_loss = min(100, len(stats['losses']) // 10)
        if len(stats['losses']) >= window_loss:
            smoothed = np.convolve(stats['losses'], 
                                   np.ones(window_loss)/window_loss, mode='valid')
            axes[1, 1].plot(range(window_loss-1, len(stats['losses'])), 
                           smoothed, label=f'MA({window_loss})', linewidth=2)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    agent, stats = train_dqn(
        num_episodes=500,
        max_steps=1000,
        eval_freq=50,
        save_path="models/dqn_cartpole.pt",
        config_path="configs/dqn.yaml"
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Final 10-episode average reward: {np.mean(stats['episode_rewards'][-10:]):.1f}")
    print(f"Best evaluation reward: {max(stats['eval_rewards']):.1f}")
    print("="*60)