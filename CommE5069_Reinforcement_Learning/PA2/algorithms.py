# # bias_variance_analysis.py
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from algorithms import MonteCarloPrediction, TDPrediction
# # from gridworld import GridWorld

# # # Parameters (same as main.py)
# # STEP_REWARD = -0.1
# # GOAL_REWARD = 1.0
# # TRAP_REWARD = -1.0
# # INIT_POS = [0]
# # DISCOUNT_FACTOR = 0.9
# # POLICY = None
# # MAX_EPISODE = 300
# # LEARNING_RATE = 0.01

# # def run_bias_variance_analysis(num_seeds=50):
# #     """Run MC and TD(0) prediction for multiple seeds and analyze bias/variance"""
    
# #     # Load ground truth values
# #     gt_values = np.load('sample_solutions/prediction_GT.npy')
    
# #     mc_results = []
# #     td_results = []
    
# #     print(f"Running {num_seeds} seeds for bias/variance analysis...")
    
# #     for seed in range(num_seeds):
# #         if (seed + 1) % 10 == 0:
# #             print(f"  Completed {seed + 1}/{num_seeds} seeds")
        
# #         # Run MC Prediction
# #         grid_world = GridWorld(
# #             "maze.txt",
# #             step_reward=STEP_REWARD,
# #             goal_reward=GOAL_REWARD,
# #             trap_reward=TRAP_REWARD,
# #             init_pos=INIT_POS,
# #         )
# #         mc_pred = MonteCarloPrediction(
# #             grid_world,
# #             discount_factor=DISCOUNT_FACTOR,
# #             policy=POLICY,
# #             max_episode=MAX_EPISODE,
# #             seed=seed
# #         )
# #         mc_pred.run()
# #         mc_results.append(mc_pred.get_all_state_values())
        
# #         # Run TD(0) Prediction
# #         grid_world = GridWorld(
# #             "maze.txt",
# #             step_reward=STEP_REWARD,
# #             goal_reward=GOAL_REWARD,
# #             trap_reward=TRAP_REWARD,
# #             init_pos=INIT_POS,
# #         )
# #         td_pred = TDPrediction(
# #             grid_world,
# #             learning_rate=LEARNING_RATE,
# #             discount_factor=DISCOUNT_FACTOR,
# #             policy=POLICY,
# #             max_episode=MAX_EPISODE,
# #             seed=seed
# #         )
# #         td_pred.run()
# #         td_results.append(td_pred.get_all_state_values())
    
# #     # Convert to numpy arrays: shape (num_seeds, num_states)
# #     mc_results = np.array(mc_results)
# #     td_results = np.array(td_results)
    
# #     print("\nCalculating bias and variance...")
    
# #     # Calculate average predictions across all seeds
# #     mc_avg = np.mean(mc_results, axis=0)  # shape: (num_states,)
# #     td_avg = np.mean(td_results, axis=0)
    
# #     # Calculate bias for each state
# #     mc_bias = mc_avg - gt_values
# #     td_bias = td_avg - gt_values
    
# #     # Calculate variance for each state
# #     mc_variance = np.mean((mc_results - mc_avg) ** 2, axis=0)
# #     td_variance = np.mean((td_results - td_avg) ** 2, axis=0)
    
# #     # Print statistics
# #     print("\n" + "="*60)
# #     print("BIAS AND VARIANCE ANALYSIS")
# #     print("="*60)
# #     print(f"\nMonte Carlo Prediction:")
# #     print(f"  Mean Absolute Bias: {np.mean(np.abs(mc_bias)):.6f}")
# #     print(f"  Mean Variance: {np.mean(mc_variance):.6f}")
# #     print(f"  Max Absolute Bias: {np.max(np.abs(mc_bias)):.6f}")
# #     print(f"  Max Variance: {np.max(mc_variance):.6f}")
    
# #     print(f"\nTD(0) Prediction:")
# #     print(f"  Mean Absolute Bias: {np.mean(np.abs(td_bias)):.6f}")
# #     print(f"  Mean Variance: {np.mean(td_variance):.6f}")
# #     print(f"  Max Absolute Bias: {np.max(np.abs(td_bias)):.6f}")
# #     print(f"  Max Variance: {np.max(td_variance):.6f}")
    
# #     print("\n" + "="*60)
    
# #     # Create visualizations
# #     num_states = len(gt_values)
# #     state_indices = np.arange(num_states)
    
# #     # Figure 1: Bias Comparison
# #     plt.figure(figsize=(14, 5))
    
# #     plt.subplot(1, 2, 1)
# #     plt.bar(state_indices, mc_bias, alpha=0.7, label='MC Bias')
# #     plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
# #     plt.xlabel('State Index')
# #     plt.ylabel('Bias')
# #     plt.title('Monte Carlo Prediction - Bias per State')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(1, 2, 2)
# #     plt.bar(state_indices, td_bias, alpha=0.7, color='orange', label='TD(0) Bias')
# #     plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
# #     plt.xlabel('State Index')
# #     plt.ylabel('Bias')
# #     plt.title('TD(0) Prediction - Bias per State')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('bias_comparison.png', dpi=300, bbox_inches='tight')
# #     print("\nSaved: bias_comparison.png")
    
# #     # Figure 2: Variance Comparison
# #     plt.figure(figsize=(14, 5))
    
# #     plt.subplot(1, 2, 1)
# #     plt.bar(state_indices, mc_variance, alpha=0.7, label='MC Variance')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Variance')
# #     plt.title('Monte Carlo Prediction - Variance per State')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(1, 2, 2)
# #     plt.bar(state_indices, td_variance, alpha=0.7, color='orange', label='TD(0) Variance')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Variance')
# #     plt.title('TD(0) Prediction - Variance per State')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('variance_comparison.png', dpi=300, bbox_inches='tight')
# #     print("Saved: variance_comparison.png")
    
# #     # Figure 3: Side-by-side comparison
# #     plt.figure(figsize=(14, 10))
    
# #     plt.subplot(2, 2, 1)
# #     plt.bar(state_indices, np.abs(mc_bias), alpha=0.7, label='MC')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Absolute Bias')
# #     plt.title('MC - Absolute Bias')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(2, 2, 2)
# #     plt.bar(state_indices, np.abs(td_bias), alpha=0.7, color='orange', label='TD(0)')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Absolute Bias')
# #     plt.title('TD(0) - Absolute Bias')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(2, 2, 3)
# #     plt.bar(state_indices, mc_variance, alpha=0.7, label='MC')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Variance')
# #     plt.title('MC - Variance')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(2, 2, 4)
# #     plt.bar(state_indices, td_variance, alpha=0.7, color='orange', label='TD(0)')
# #     plt.xlabel('State Index')
# #     plt.ylabel('Variance')
# #     plt.title('TD(0) - Variance')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('bias_variance_summary.png', dpi=300, bbox_inches='tight')
# #     print("Saved: bias_variance_summary.png")
    
# #     # Figure 4: Direct comparison
# #     plt.figure(figsize=(14, 5))
    
# #     x = np.arange(num_states)
# #     width = 0.35
    
# #     plt.subplot(1, 2, 1)
# #     plt.bar(x - width/2, np.abs(mc_bias), width, label='MC', alpha=0.7)
# #     plt.bar(x + width/2, np.abs(td_bias), width, label='TD(0)', alpha=0.7)
# #     plt.xlabel('State Index')
# #     plt.ylabel('Absolute Bias')
# #     plt.title('Absolute Bias Comparison')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.subplot(1, 2, 2)
# #     plt.bar(x - width/2, mc_variance, width, label='MC', alpha=0.7)
# #     plt.bar(x + width/2, td_variance, width, label='TD(0)', alpha=0.7)
# #     plt.xlabel('State Index')
# #     plt.ylabel('Variance')
# #     plt.title('Variance Comparison')
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('bias_variance_direct_comparison.png', dpi=300, bbox_inches='tight')
# #     print("Saved: bias_variance_direct_comparison.png")
    
# #     # Save numerical results
# #     results = {
# #         'mc_bias': mc_bias,
# #         'td_bias': td_bias,
# #         'mc_variance': mc_variance,
# #         'td_variance': td_variance,
# #         'mc_avg': mc_avg,
# #         'td_avg': td_avg,
# #         'gt_values': gt_values
# #     }
# #     np.savez('bias_variance_results.npz', **results)
# #     print("Saved: bias_variance_results.npz")
    
# #     print("\nAnalysis complete!")
# #     return results

# # learning_loss_curves.py
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from collections import deque
# # from gridworld import GridWorld

# # # Modified versions of the control algorithms that track rewards and losses
# # # Copy the base classes from algorithms.py but add tracking

# # class ModelFreeControl:
# #     """Base class for model free control algorithms"""

# #     def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
# #         self.grid_world = grid_world
# #         self.discount_factor = discount_factor
# #         self.action_space = grid_world.get_action_space()
# #         self.state_space = grid_world.get_state_space()
# #         self.q_values = np.zeros((self.state_space, self.action_space))
# #         self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
# #         self.policy_index = np.zeros(self.state_space, dtype=int)
        
# #         # Tracking for learning and loss curves
# #         self.episode_rewards = []
# #         self.episode_losses = []
# #         self.current_episode_rewards = []
# #         self.current_episode_losses = []

# #     def get_policy_index(self) -> np.ndarray:
# #         for s_i in range(self.state_space):
# #             self.policy_index[s_i] = self.q_values[s_i].argmax()
# #         return self.policy_index

# #     def get_max_state_values(self) -> np.ndarray:
# #         max_values = np.zeros(self.state_space)
# #         for i in range(self.state_space):
# #             max_values[i] = self.q_values[i].max()
# #         return max_values


# # class MonteCarloPolicyIteration_Tracked(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.rng = np.random.default_rng()

# #     def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
# #         G = 0
# #         for t in range(len(state_trace) - 1, -1, -1):
# #             G = self.discount_factor * G + reward_trace[t]
# #             state = state_trace[t]
# #             action = action_trace[t]
            
# #             # Track estimation loss
# #             estimation_loss = abs(G - self.q_values[state, action])
# #             self.current_episode_losses.append(estimation_loss)
            
# #             # Update Q-value
# #             self.q_values[state, action] += self.lr * (G - self.q_values[state, action])

# #     def policy_improvement(self) -> None:
# #         for s in range(self.state_space):
# #             best_action = np.argmax(self.q_values[s])
# #             for a in range(self.action_space):
# #                 if a == best_action:
# #                     self.policy[s, a] = 1 - self.epsilon + self.epsilon / self.action_space
# #                 else:
# #                     self.policy[s, a] = self.epsilon / self.action_space

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         state_trace = []
# #         action_trace = []
# #         reward_trace = []
        
# #         while iter_episode < max_episode:
# #             current_state = self.grid_world.get_current_state()
# #             action_probs = self.policy[current_state]
# #             action = self.rng.choice(self.action_space, p=action_probs)
            
# #             state_trace.append(current_state)
# #             action_trace.append(action)
            
# #             next_state, reward, done = self.grid_world.step(action)
# #             reward_trace.append(reward)
# #             self.current_episode_rewards.append(reward)
            
# #             if done:
# #                 self.current_episode_losses = []
# #                 self.policy_evaluation(state_trace, action_trace, reward_trace)
# #                 self.policy_improvement()
                
# #                 # Record episode statistics
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 # Reset for next episode
# #                 state_trace = []
# #                 action_trace = []
# #                 reward_trace = []
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1


# # class SARSA_Tracked(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.rng = np.random.default_rng()

# #     def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> float:
# #         if is_done:
# #             td_target = r
# #         else:
# #             td_target = r + self.discount_factor * self.q_values[s2, a2]
        
# #         estimation_loss = abs(td_target - self.q_values[s, a])
# #         self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
# #         best_action = np.argmax(self.q_values[s])
# #         for action in range(self.action_space):
# #             if action == best_action:
# #                 self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
# #             else:
# #                 self.policy[s, action] = self.epsilon / self.action_space
        
# #         return estimation_loss

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         action_probs = self.policy[current_state]
# #         current_action = self.rng.choice(self.action_space, p=action_probs)
        
# #         while iter_episode < max_episode:
# #             s = self.grid_world.get_current_state()
# #             a = current_action
            
# #             next_state, reward, done = self.grid_world.step(a)
# #             self.current_episode_rewards.append(reward)
            
# #             action_probs = self.policy[next_state]
# #             next_action = self.rng.choice(self.action_space, p=action_probs)
            
# #             loss = self.policy_eval_improve(s, a, reward, next_state, next_action, done)
# #             self.current_episode_losses.append(loss)
            
# #             current_action = next_action
            
# #             if done:
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1
                
# #                 current_state = self.grid_world.get_current_state()
# #                 action_probs = self.policy[current_state]
# #                 current_action = self.rng.choice(self.action_space, p=action_probs)


# # class Q_Learning_Tracked(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, 
# #                  buffer_size: int, update_frequency: int, sample_batch_size: int):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.buffer = deque(maxlen=buffer_size)
# #         self.update_frequency = update_frequency
# #         self.sample_batch_size = sample_batch_size
# #         self.rng = np.random.default_rng()

# #     def add_buffer(self, s, a, r, s2, d) -> None:
# #         self.buffer.append((s, a, r, s2, d))

# #     def sample_batch(self) -> np.ndarray:
# #         batch_size = min(self.sample_batch_size, len(self.buffer))
# #         indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
# #         return indices

# #     def policy_eval_improve(self, s, a, r, s2, is_done) -> float:
# #         if is_done:
# #             td_target = r
# #         else:
# #             td_target = r + self.discount_factor * np.max(self.q_values[s2])
        
# #         estimation_loss = abs(td_target - self.q_values[s, a])
# #         self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
# #         best_action = np.argmax(self.q_values[s])
# #         for action in range(self.action_space):
# #             if action == best_action:
# #                 self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
# #             else:
# #                 self.policy[s, action] = self.epsilon / self.action_space
        
# #         return estimation_loss

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         transition_count = 0
        
# #         while iter_episode < max_episode:
# #             s = self.grid_world.get_current_state()
# #             action_probs = self.policy[s]
# #             a = self.rng.choice(self.action_space, p=action_probs)
            
# #             s2, r, done = self.grid_world.step(a)
# #             self.current_episode_rewards.append(r)
            
# #             self.add_buffer(s, a, r, s2, done)
# #             transition_count += 1
            
# #             if transition_count % self.update_frequency == 0:
# #                 batch_indices = self.sample_batch()
# #                 for idx in batch_indices:
# #                     s_b, a_b, r_b, s2_b, d_b = self.buffer[idx]
# #                     loss = self.policy_eval_improve(s_b, a_b, r_b, s2_b, d_b)
# #                     self.current_episode_losses.append(loss)
            
# #             if done:
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1


# # def compute_learning_curve(episode_rewards, window=10):
# #     """Compute average non-discounted episodic reward for last 10 episodes"""
# #     learning_curve = []
# #     for i in range(len(episode_rewards)):
# #         start_idx = max(0, i - window + 1)
# #         recent_episodes = episode_rewards[start_idx:i+1]
        
# #         # Calculate average reward per episode
# #         avg_reward = np.mean([np.mean(ep_rewards) for ep_rewards in recent_episodes])
# #         learning_curve.append(avg_reward)
    
# #     return learning_curve


# # def compute_loss_curve(episode_losses, window=10):
# #     """Compute average absolute estimation loss for last 10 episodes"""
# #     loss_curve = []
# #     for i in range(len(episode_losses)):
# #         start_idx = max(0, i - window + 1)
# #         recent_episodes = episode_losses[start_idx:i+1]
        
# #         # Calculate average loss per episode
# #         avg_loss = np.mean([np.mean(ep_losses) for ep_losses in recent_episodes if len(ep_losses) > 0])
# #         loss_curve.append(avg_loss)
    
# #     return loss_curve


# # def run_experiments(epsilon_values=[0.1, 0.2, 0.3, 0.4], max_episode=10000):
# #     """Run MC, SARSA, and Q-Learning with different epsilon values"""
    
# #     STEP_REWARD = -0.1
# #     GOAL_REWARD = 1.0
# #     TRAP_REWARD = -1.0
# #     DISCOUNT_FACTOR = 0.9
# #     LEARNING_RATE = 0.01
# #     BUFFER_SIZE = 10000
# #     UPDATE_FREQUENCY = 200
# #     SAMPLE_BATCH_SIZE = 500
    
# #     results = {}
    
# #     for epsilon in epsilon_values:
# #         print(f"\n{'='*60}")
# #         print(f"Running experiments with epsilon = {epsilon}")
# #         print(f"{'='*60}")
        
# #         results[epsilon] = {}
        
# #         # Monte Carlo Policy Iteration
# #         print(f"  Running MC Policy Iteration...")
# #         grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD, 
# #                               goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD)
# #         mc = MonteCarloPolicyIteration_Tracked(grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon)
# #         mc.run(max_episode)
# #         results[epsilon]['MC'] = {
# #             'learning_curve': compute_learning_curve(mc.episode_rewards),
# #             'loss_curve': compute_loss_curve(mc.episode_losses)
# #         }
# #         print(f"    Completed {len(mc.episode_rewards)} episodes")
        
# #         # SARSA
# #         print(f"  Running SARSA...")
# #         grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD,
# #                               goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD)
# #         sarsa = SARSA_Tracked(grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon)
# #         sarsa.run(max_episode)
# #         results[epsilon]['SARSA'] = {
# #             'learning_curve': compute_learning_curve(sarsa.episode_rewards),
# #             'loss_curve': compute_loss_curve(sarsa.episode_losses)
# #         }
# #         print(f"    Completed {len(sarsa.episode_rewards)} episodes")
        
# #         # Q-Learning
# #         print(f"  Running Q-Learning...")
# #         grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD,
# #                               goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD)
# #         qlearn = Q_Learning_Tracked(grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon,
# #                                     BUFFER_SIZE, UPDATE_FREQUENCY, SAMPLE_BATCH_SIZE)
# #         qlearn.run(max_episode)
# #         results[epsilon]['Q-Learning'] = {
# #             'learning_curve': compute_learning_curve(qlearn.episode_rewards),
# #             'loss_curve': compute_loss_curve(qlearn.episode_losses)
# #         }
# #         print(f"    Completed {len(qlearn.episode_rewards)} episodes")
    
# #     return results


# # def plot_learning_curves(results, max_episode=10000):
# #     """Plot learning curves for all algorithms and epsilon values"""
    
# #     epsilon_values = sorted(results.keys())
# #     algorithms = ['MC', 'SARSA', 'Q-Learning']
    
# #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
# #     for idx, algorithm in enumerate(algorithms):
# #         ax = axes[idx]
        
# #         for epsilon in epsilon_values:
# #             learning_curve = results[epsilon][algorithm]['learning_curve']
# #             episodes = range(len(learning_curve))
# #             ax.plot(episodes, learning_curve, label=f'ε={epsilon}', linewidth=2, alpha=0.8)
        
# #         ax.set_xlabel('Episode', fontsize=12)
# #         ax.set_ylabel('Average Reward (Last 10 Episodes)', fontsize=12)
# #         ax.set_title(f'{algorithm} - Learning Curve', fontsize=14, fontweight='bold')
# #         ax.legend(fontsize=10)
# #         ax.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
# #     print("\nSaved: learning_curves.png")


# # def plot_loss_curves(results, max_episode=10000):
# #     """Plot loss curves for all algorithms and epsilon values"""
    
# #     epsilon_values = sorted(results.keys())
# #     algorithms = ['MC', 'SARSA', 'Q-Learning']
    
# #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
# #     for idx, algorithm in enumerate(algorithms):
# #         ax = axes[idx]
        
# #         for epsilon in epsilon_values:
# #             loss_curve = results[epsilon][algorithm]['loss_curve']
# #             episodes = range(len(loss_curve))
# #             ax.plot(episodes, loss_curve, label=f'ε={epsilon}', linewidth=2, alpha=0.8)
        
# #         ax.set_xlabel('Episode', fontsize=12)
# #         ax.set_ylabel('Average Absolute Loss (Last 10 Episodes)', fontsize=12)
# #         ax.set_title(f'{algorithm} - Loss Curve', fontsize=14, fontweight='bold')
# #         ax.legend(fontsize=10)
# #         ax.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
# #     plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
# #     print("Saved: loss_curves.png")


# # if __name__ == "__main__":
# #     print("Starting learning and loss curve experiments...")
# #     print("This will take several minutes...\n")
    
# #     epsilon_values = [0.1, 0.2, 0.3, 0.4]
# #     max_episode = 10000
    
# #     results = run_experiments(epsilon_values, max_episode)
    
# #     print("\n" + "="*60)
# #     print("Generating plots...")
# #     print("="*60)
    
# #     plot_learning_curves(results, max_episode)
# #     plot_loss_curves(results, max_episode)
    
# #     # Save results
# #     np.save('learning_loss_results.npy', results)
# #     print("Saved: learning_loss_results.npy")
    
# #     print("\nAll experiments completed!")



# # wandb_integration.py
# """
# Weights & Biases Integration for RL Assignment 2
# This script logs all experiments to W&B for visualization and tracking.
# """

# # import numpy as np
# # import wandb
# # from collections import deque
# # from gridworld import GridWorld

# # # Import the tracked versions from learning_loss_curves.py
# # # You can copy the class definitions here or import them

# # class ModelFreeControl:
# #     """Base class for model free control algorithms"""

# #     def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
# #         self.grid_world = grid_world
# #         self.discount_factor = discount_factor
# #         self.action_space = grid_world.get_action_space()
# #         self.state_space = grid_world.get_state_space()
# #         self.q_values = np.zeros((self.state_space, self.action_space))
# #         self.policy = np.ones((self.state_space, self.action_space)) / self.action_space
# #         self.policy_index = np.zeros(self.state_space, dtype=int)
        
# #         self.episode_rewards = []
# #         self.episode_losses = []
# #         self.current_episode_rewards = []
# #         self.current_episode_losses = []

# #     def get_policy_index(self) -> np.ndarray:
# #         for s_i in range(self.state_space):
# #             self.policy_index[s_i] = self.q_values[s_i].argmax()
# #         return self.policy_index

# #     def get_max_state_values(self) -> np.ndarray:
# #         max_values = np.zeros(self.state_space)
# #         for i in range(self.state_space):
# #             max_values[i] = self.q_values[i].max()
# #         return max_values


# # class MonteCarloPolicyIteration_WandB(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.rng = np.random.default_rng()

# #     def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
# #         G = 0
# #         for t in range(len(state_trace) - 1, -1, -1):
# #             G = self.discount_factor * G + reward_trace[t]
# #             state = state_trace[t]
# #             action = action_trace[t]
# #             estimation_loss = abs(G - self.q_values[state, action])
# #             self.current_episode_losses.append(estimation_loss)
# #             self.q_values[state, action] += self.lr * (G - self.q_values[state, action])

# #     def policy_improvement(self) -> None:
# #         for s in range(self.state_space):
# #             best_action = np.argmax(self.q_values[s])
# #             for a in range(self.action_space):
# #                 if a == best_action:
# #                     self.policy[s, a] = 1 - self.epsilon + self.epsilon / self.action_space
# #                 else:
# #                     self.policy[s, a] = self.epsilon / self.action_space

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         state_trace = []
# #         action_trace = []
# #         reward_trace = []
        
# #         while iter_episode < max_episode:
# #             current_state = self.grid_world.get_current_state()
# #             action_probs = self.policy[current_state]
# #             action = self.rng.choice(self.action_space, p=action_probs)
            
# #             state_trace.append(current_state)
# #             action_trace.append(action)
            
# #             next_state, reward, done = self.grid_world.step(action)
# #             reward_trace.append(reward)
# #             self.current_episode_rewards.append(reward)
            
# #             if done:
# #                 self.current_episode_losses = []
# #                 self.policy_evaluation(state_trace, action_trace, reward_trace)
# #                 self.policy_improvement()
                
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 # Log to W&B
# #                 avg_reward = np.mean(self.current_episode_rewards)
# #                 avg_loss = np.mean(self.current_episode_losses) if len(self.current_episode_losses) > 0 else 0
                
# #                 # Compute moving average over last 10 episodes
# #                 if len(self.episode_rewards) >= 10:
# #                     last_10_rewards = [np.mean(ep) for ep in self.episode_rewards[-10:]]
# #                     last_10_losses = [np.mean(ep) for ep in self.episode_losses[-10:] if len(ep) > 0]
# #                     ma_reward = np.mean(last_10_rewards)
# #                     ma_loss = np.mean(last_10_losses) if len(last_10_losses) > 0 else 0
# #                 else:
# #                     ma_reward = avg_reward
# #                     ma_loss = avg_loss
                
# #                 wandb.log({
# #                     "episode": iter_episode,
# #                     "avg_reward": avg_reward,
# #                     "avg_loss": avg_loss,
# #                     "ma_reward_10": ma_reward,
# #                     "ma_loss_10": ma_loss,
# #                     "episode_length": len(self.current_episode_rewards)
# #                 })
                
# #                 state_trace = []
# #                 action_trace = []
# #                 reward_trace = []
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1


# # class SARSA_WandB(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.rng = np.random.default_rng()

# #     def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> float:
# #         if is_done:
# #             td_target = r
# #         else:
# #             td_target = r + self.discount_factor * self.q_values[s2, a2]
        
# #         estimation_loss = abs(td_target - self.q_values[s, a])
# #         self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
# #         best_action = np.argmax(self.q_values[s])
# #         for action in range(self.action_space):
# #             if action == best_action:
# #                 self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
# #             else:
# #                 self.policy[s, action] = self.epsilon / self.action_space
        
# #         return estimation_loss

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         action_probs = self.policy[current_state]
# #         current_action = self.rng.choice(self.action_space, p=action_probs)
        
# #         while iter_episode < max_episode:
# #             s = self.grid_world.get_current_state()
# #             a = current_action
            
# #             next_state, reward, done = self.grid_world.step(a)
# #             self.current_episode_rewards.append(reward)
            
# #             action_probs = self.policy[next_state]
# #             next_action = self.rng.choice(self.action_space, p=action_probs)
            
# #             loss = self.policy_eval_improve(s, a, reward, next_state, next_action, done)
# #             self.current_episode_losses.append(loss)
            
# #             current_action = next_action
            
# #             if done:
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 # Log to W&B
# #                 avg_reward = np.mean(self.current_episode_rewards)
# #                 avg_loss = np.mean(self.current_episode_losses)
                
# #                 if len(self.episode_rewards) >= 10:
# #                     last_10_rewards = [np.mean(ep) for ep in self.episode_rewards[-10:]]
# #                     last_10_losses = [np.mean(ep) for ep in self.episode_losses[-10:]]
# #                     ma_reward = np.mean(last_10_rewards)
# #                     ma_loss = np.mean(last_10_losses)
# #                 else:
# #                     ma_reward = avg_reward
# #                     ma_loss = avg_loss
                
# #                 wandb.log({
# #                     "episode": iter_episode,
# #                     "avg_reward": avg_reward,
# #                     "avg_loss": avg_loss,
# #                     "ma_reward_10": ma_reward,
# #                     "ma_loss_10": ma_loss,
# #                     "episode_length": len(self.current_episode_rewards)
# #                 })
                
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1
                
# #                 current_state = self.grid_world.get_current_state()
# #                 action_probs = self.policy[current_state]
# #                 current_action = self.rng.choice(self.action_space, p=action_probs)


# # class Q_Learning_WandB(ModelFreeControl):
# #     def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, 
# #                  buffer_size: int, update_frequency: int, sample_batch_size: int):
# #         super().__init__(grid_world, discount_factor)
# #         self.lr = learning_rate
# #         self.epsilon = epsilon
# #         self.buffer = deque(maxlen=buffer_size)
# #         self.update_frequency = update_frequency
# #         self.sample_batch_size = sample_batch_size
# #         self.rng = np.random.default_rng()

# #     def add_buffer(self, s, a, r, s2, d) -> None:
# #         self.buffer.append((s, a, r, s2, d))

# #     def sample_batch(self) -> np.ndarray:
# #         batch_size = min(self.sample_batch_size, len(self.buffer))
# #         indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
# #         return indices

# #     def policy_eval_improve(self, s, a, r, s2, is_done) -> float:
# #         if is_done:
# #             td_target = r
# #         else:
# #             td_target = r + self.discount_factor * np.max(self.q_values[s2])
        
# #         estimation_loss = abs(td_target - self.q_values[s, a])
# #         self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
# #         best_action = np.argmax(self.q_values[s])
# #         for action in range(self.action_space):
# #             if action == best_action:
# #                 self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
# #             else:
# #                 self.policy[s, action] = self.epsilon / self.action_space
        
# #         return estimation_loss

# #     def run(self, max_episode=1000) -> None:
# #         iter_episode = 0
# #         current_state = self.grid_world.reset()
# #         transition_count = 0
        
# #         while iter_episode < max_episode:
# #             s = self.grid_world.get_current_state()
# #             action_probs = self.policy[s]
# #             a = self.rng.choice(self.action_space, p=action_probs)
            
# #             s2, r, done = self.grid_world.step(a)
# #             self.current_episode_rewards.append(r)
            
# #             self.add_buffer(s, a, r, s2, done)
# #             transition_count += 1
            
# #             if transition_count % self.update_frequency == 0:
# #                 batch_indices = self.sample_batch()
# #                 for idx in batch_indices:
# #                     s_b, a_b, r_b, s2_b, d_b = self.buffer[idx]
# #                     loss = self.policy_eval_improve(s_b, a_b, r_b, s2_b, d_b)
# #                     self.current_episode_losses.append(loss)
            
# #             if done:
# #                 self.episode_rewards.append(self.current_episode_rewards.copy())
# #                 self.episode_losses.append(self.current_episode_losses.copy())
                
# #                 # Log to W&B
# #                 avg_reward = np.mean(self.current_episode_rewards)
# #                 avg_loss = np.mean(self.current_episode_losses) if len(self.current_episode_losses) > 0 else 0
                
# #                 if len(self.episode_rewards) >= 10:
# #                     last_10_rewards = [np.mean(ep) for ep in self.episode_rewards[-10:]]
# #                     last_10_losses = [np.mean(ep) for ep in self.episode_losses[-10:] if len(ep) > 0]
# #                     ma_reward = np.mean(last_10_rewards)
# #                     ma_loss = np.mean(last_10_losses) if len(last_10_losses) > 0 else 0
# #                 else:
# #                     ma_reward = avg_reward
# #                     ma_loss = avg_loss
                
# #                 wandb.log({
# #                     "episode": iter_episode,
# #                     "avg_reward": avg_reward,
# #                     "avg_loss": avg_loss,
# #                     "ma_reward_10": ma_reward,
# #                     "ma_loss_10": ma_loss,
# #                     "episode_length": len(self.current_episode_rewards),
# #                     "buffer_size": len(self.buffer)
# #                 })
                
# #                 self.current_episode_rewards = []
# #                 self.current_episode_losses = []
# #                 iter_episode += 1


# # def run_wandb_experiments():
# #     """Run all experiments with W&B logging"""
    
# #     STEP_REWARD = -0.1
# #     GOAL_REWARD = 1.0
# #     TRAP_REWARD = -1.0
# #     DISCOUNT_FACTOR = 0.9
# #     LEARNING_RATE = 0.01
# #     BUFFER_SIZE = 10000
# #     UPDATE_FREQUENCY = 200
# #     SAMPLE_BATCH_SIZE = 500
    
# #     epsilon_values = [0.1, 0.2, 0.3, 0.4]
# #     algorithms = ['MC', 'SARSA', 'Q-Learning']
# #     max_episode = 10000
    
# #     for algorithm in algorithms:
# #         for epsilon in epsilon_values:
# #             # Initialize W&B run
# #             run_name = f"{algorithm}_epsilon_{epsilon}"
# #             wandb.init(
# #                 project="rl-hw2-gridworld",
# #                 name=run_name,
# #                 config={
# #                     "algorithm": algorithm,
# #                     "epsilon": epsilon,
# #                     "learning_rate": LEARNING_RATE,
# #                     "discount_factor": DISCOUNT_FACTOR,
# #                     "max_episode": max_episode,
# #                     "step_reward": STEP_REWARD,
# #                     "goal_reward": GOAL_REWARD,
# #                     "trap_reward": TRAP_REWARD
# #                 }
# #             )
            
# #             print(f"\nRunning {algorithm} with epsilon={epsilon}")
            
# #             grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD,
# #                                   goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD)
            
# #             if algorithm == 'MC':
# #                 agent = MonteCarloPolicyIteration_WandB(
# #                     grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon
# #                 )
# #             elif algorithm == 'SARSA':
# #                 agent = SARSA_WandB(
# #                     grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon
# #                 )
# #             elif algorithm == 'Q-Learning':
# #                 agent = Q_Learning_WandB(
# #                     grid_world, DISCOUNT_FACTOR, LEARNING_RATE, epsilon,
# #                     BUFFER_SIZE, UPDATE_FREQUENCY, SAMPLE_BATCH_SIZE
# #                 )
            
# #             agent.run(max_episode)
            
# #             # Log final metrics
# #             wandb.log({
# #                 "final_max_q_value": np.max(agent.q_values),
# #                 "final_mean_q_value": np.mean(agent.q_values),
# #                 "total_episodes": len(agent.episode_rewards)
# #             })
            
# #             wandb.finish()
# #             print(f"  Completed: {len(agent.episode_rewards)} episodes")


# # def run_bias_variance_wandb():
# #     """Run bias/variance analysis with W&B logging"""
    
# #     wandb.init(
# #         project="rl-hw2-gridworld",
# #         name="bias_variance_analysis",
# #         config={
# #             "num_seeds": 50,
# #             "max_episode": 300,
# #             "algorithms": ["MC", "TD0"]
# #         }
# #     )
    
# #     STEP_REWARD = -0.1
# #     GOAL_REWARD = 1.0
# #     TRAP_REWARD = -1.0
# #     INIT_POS = [0]
# #     DISCOUNT_FACTOR = 0.9
# #     POLICY = None
# #     MAX_EPISODE = 300
# #     LEARNING_RATE = 0.01
    
# #     gt_values = np.load('sample_solutions/prediction_GT.npy')
    
# #     from algorithms import MonteCarloPrediction, TDPrediction
    
# #     mc_results = []
# #     td_results = []
    
# #     print("Running bias/variance analysis with W&B logging...")
    
# #     for seed in range(50):
# #         # MC Prediction
# #         grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD,
# #                               goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD, init_pos=INIT_POS)
# #         mc_pred = MonteCarloPrediction(grid_world, discount_factor=DISCOUNT_FACTOR,
# #                                        policy=POLICY, max_episode=MAX_EPISODE, seed=seed)
# #         mc_pred.run()
# #         mc_values = mc_pred.get_all_state_values()
# #         mc_results.append(mc_values)
        
# #         # TD Prediction
# #         grid_world = GridWorld("maze.txt", step_reward=STEP_REWARD,
# #                               goal_reward=GOAL_REWARD, trap_reward=TRAP_REWARD, init_pos=INIT_POS)
# #         td_pred = TDPrediction(grid_world, learning_rate=LEARNING_RATE,
# #                               discount_factor=DISCOUNT_FACTOR, policy=POLICY,
# #                               max_episode=MAX_EPISODE, seed=seed)
# #         td_pred.run()
# #         td_values = td_pred.get_all_state_values()
# #         td_results.append(td_values)
        
# #         # Log per-seed results
# #         mc_mse = np.mean((mc_values - gt_values) ** 2)
# #         td_mse = np.mean((td_values - gt_values) ** 2)
        
# #         wandb.log({
# #             "seed": seed,
# #             "mc_mse": mc_mse,
# #             "td_mse": td_mse
# #         })
    
# #     # Calculate final statistics
# #     mc_results = np.array(mc_results)
# #     td_results = np.array(td_results)
    
# #     mc_avg = np.mean(mc_results, axis=0)
# #     td_avg = np.mean(td_results, axis=0)
    
# #     mc_bias = mc_avg - gt_values
# #     td_bias = td_avg - gt_values
    
# #     mc_variance = np.mean((mc_results - mc_avg) ** 2, axis=0)
# #     td_variance = np.mean((td_results - td_avg) ** 2, axis=0)
    
# #     # Log summary statistics
# #     wandb.log({
# #         "MC_mean_abs_bias": np.mean(np.abs(mc_bias)),
# #         "MC_mean_variance": np.mean(mc_variance),
# #         "MC_max_abs_bias": np.max(np.abs(mc_bias)),
# #         "MC_max_variance": np.max(mc_variance),
# #         "TD_mean_abs_bias": np.mean(np.abs(td_bias)),
# #         "TD_mean_variance": np.mean(td_variance),
# #         "TD_max_abs_bias": np.max(np.abs(td_bias)),
# #         "TD_max_variance": np.max(td_variance)
# #     })
    
# #     # Create and log tables
# #     bias_table = wandb.Table(
# #         columns=["State", "MC_Bias", "TD_Bias", "MC_Variance", "TD_Variance"],
# #         data=[[i, mc_bias[i], td_bias[i], mc_variance[i], td_variance[i]] 
# #               for i in range(len(gt_values))]
# #     )
# #     wandb.log({"bias_variance_table": bias_table})
    
# #     wandb.finish()
# #     print("Bias/variance analysis with W&B complete!")


# # if __name__ == "__main__":
# #     print("="*60)
# #     print("Reinforcement Learning HW2 - W&B Integration")
# #     print("="*60)
    
# #     # First, make sure you're logged in to W&B
# #     # Run: wandb login
    
# #     choice = input("\nWhat would you like to run?\n"
# #                    "1. Bias/Variance Analysis\n"
# #                    "2. Learning/Loss Curves (All algorithms, all epsilons)\n"
# #                    "3. Both\n"
# #                    "Choice (1/2/3): ")
    
# #     if choice == "1":
# #         run_bias_variance_wandb()
# #     elif choice == "2":
# #         run_wandb_experiments()
# #     elif choice == "3":
# #         run_bias_variance_wandb()
# #         run_wandb_experiments()
# #     else:
# #         print("Invalid choice!")
    
# #     print("\n" + "="*60)
# #     print("Visit https://wandb.ai to view your results!")
# #     print("="*60)

import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)
        if policy is not None:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """
        current_state = self.grid_world.get_current_state()
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  
        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter += 1
        return next_state, reward, done
        
class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)
        self.returns = [[] for _ in range(self.state_space)]

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # Initialize Returns(s) for all states - stores all returns for each state
        returns = {s: [] for s in range(self.state_space)}
        
        # Reset environment at the start
        current_state = self.grid_world.reset()
        
        while self.episode_counter < self.max_episode:
            # Generate an episode following the policy
            episode_states = []
            episode_rewards = []
            
            # Collect one complete episode
            done = False
            while not done:
                current_state = self.grid_world.get_current_state()
                episode_states.append(current_state)
                
                next_state, reward, done = self.collect_data()
                episode_rewards.append(reward)
            
            # Now we have a complete episode
            # Calculate returns and update values using First-Visit MC
            G = 0
            T = len(episode_states)
            
            # Process episode backwards: t = T-1, T-2, ..., 0
            for t in range(T - 1, -1, -1):
                # G <- gamma * G + R_{t+1}
                G = self.discount_factor * G + episode_rewards[t]
                
                # Check if S_t is a first visit (not in S_0, S_1, ..., S_{t-1})
                if episode_states[t] not in episode_states[:t]:
                    # Append G to Returns(S_t)
                    returns[episode_states[t]].append(G)
                    # V(S_t) <- average(Returns(S_t))
                    self.values[episode_states[t]] = np.mean(returns[episode_states[t]])

class TDPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)
        self.lr = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        current_state = self.grid_world.reset()
        
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.get_current_state()
            next_state, reward, done = self.collect_data()
            
            # TD(0) update: V(S) <- V(S) + α[R + γV(S') - V(S)]
            if done:
                # Terminal state has value 0
                td_target = reward
            else:
                td_target = reward + self.discount_factor * self.values[next_state]
            
            td_error = td_target - self.values[current_state]
            self.values[current_state] += self.lr * td_error

class NstepTDPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world, policy, discount_factor, max_episode, seed)
        self.lr = learning_rate
        self.n = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        current_state = self.grid_world.reset()
        
        while self.episode_counter < self.max_episode:
            # Store states and rewards for this episode
            states = [self.grid_world.get_current_state()]
            rewards = [0]  # R_0 doesn't exist, placeholder
            
            T = float('inf')
            t = 0
            
            while True:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    states.append(next_state)
                    rewards.append(reward)
                    
                    if done:
                        T = t + 1
                
                tau = t - self.n + 1
                
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.discount_factor ** (i - tau - 1)) * rewards[i]
                    
                    if tau + self.n < T:
                        G += (self.discount_factor ** self.n) * self.values[states[tau + self.n]]
                    
                    self.values[states[tau]] += self.lr * (G - self.values[states[tau]])
                
                if tau == T - 1:
                    break
                
                t += 1

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values

class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # Every-Visit Monte-Carlo: update Q(s,a) for every occurrence
        # Note: len(action_trace) = len(reward_trace) = len(state_trace) - 1
        T = len(action_trace)
        
        # Calculate returns backward for efficiency: O(T) instead of O(T^2)
        # G_t = R_{t+1} + γ*G_{t+1}
        G = 0
        for t in range(T - 1, -1, -1):
            s_t = state_trace[t]
            a_t = action_trace[t]
            
            # Update return: G_t = R_{t+1} + γ*G_{t+1}
            G = reward_trace[t] + self.discount_factor * G
            
            # Update Q(S_t, A_t) using constant-alpha MC update
            # Q(S_t, A_t) ← Q(S_t, A_t) + α[G_t - Q(S_t, A_t)]
            self.q_values[s_t, a_t] += self.lr * (G - self.q_values[s_t, a_t])

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # Update policy using ε-greedy improvement
        for s in range(self.state_space):
            # Find the best action for this state
            best_action = np.argmax(self.q_values[s])
            
            # ε-greedy policy:
            # π(a|s) = 1 - ε + ε/m, if a* = argmax Q(s,a)
            # π(a|s) = ε/m, otherwise
            for a in range(self.action_space):
                if a == best_action:
                    self.policy[s, a] = 1 - self.epsilon + self.epsilon / self.action_space
                else:
                    self.policy[s, a] = self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        
        while iter_episode < max_episode:
            # Sample action from current policy
            action_probs = self.policy[current_state]
            action = np.random.choice(self.action_space, p=action_probs)
            
            # Take action and observe next state and reward
            next_state, reward, done = self.grid_world.step(action)
            
            # Store transition
            action_trace.append(action)
            reward_trace.append(reward)
            state_trace.append(next_state)
            
            if done:
                # Episode finished - perform policy evaluation and improvement
                self.policy_evaluation(state_trace, action_trace, reward_trace)
                self.policy_improvement()
                
                # Reset for next episode
                iter_episode += 1
                current_state = self.grid_world.reset()
                state_trace = [current_state]
                action_trace = []
                reward_trace = []
            else:
                # Continue episode
                current_state = next_state

class SARSA(ModelFreeControl):
    def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.rng = np.random.default_rng()

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        if is_done:
            td_target = r
        else:
            td_target = r + self.discount_factor * self.q_values[s2, a2]
        
        # SARSA update
        self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
        # Epsilon-greedy improvement
        best_action = np.argmax(self.q_values[s])
        for action in range(self.action_space):
            if action == best_action:
                self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
            else:
                self.policy[s, action] = self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()
        
        # Choose initial action
        action_probs = self.policy[current_state]
        current_action = self.rng.choice(self.action_space, p=action_probs)
        
        while iter_episode < max_episode:
            s = self.grid_world.get_current_state()
            a = current_action
            
            # Take action
            next_state, reward, done = self.grid_world.step(a)
            
            # Choose next action
            action_probs = self.policy[next_state]
            next_action = self.rng.choice(self.action_space, p=action_probs)
            
            # Update Q-value and policy
            self.policy_eval_improve(s, a, reward, next_state, next_action, done)
            
            current_action = next_action
            
            if done:
                iter_episode += 1
                # Choose action for new episode
                current_state = self.grid_world.get_current_state()
                action_probs = self.policy[current_state]
                current_action = self.rng.choice(self.action_space, p=action_probs)

class Q_Learning(ModelFreeControl):
    def __init__(self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.buffer = deque(maxlen=buffer_size)
        self.update_frequency = update_frequency
        self.sample_batch_size = sample_batch_size
        self.rng = np.random.default_rng()

    def add_buffer(self, s, a, r, s2, d) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        batch_size = min(self.sample_batch_size, len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        return indices

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        if is_done:
            td_target = r
        else:
            td_target = r + self.discount_factor * np.max(self.q_values[s2])
        
        # Q-Learning update
        self.q_values[s, a] += self.lr * (td_target - self.q_values[s, a])
        
        # Epsilon-greedy improvement
        best_action = np.argmax(self.q_values[s])
        for action in range(self.action_space):
            if action == best_action:
                self.policy[s, action] = 1 - self.epsilon + self.epsilon / self.action_space
            else:
                self.policy[s, action] = self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0
        
        while iter_episode < max_episode:
            s = self.grid_world.get_current_state()
            
            # Choose action using epsilon-greedy
            action_probs = self.policy[s]
            a = self.rng.choice(self.action_space, p=action_probs)
            
            # Take action
            s2, r, done = self.grid_world.step(a)
            
            # Store transition
            self.add_buffer(s, a, r, s2, done)
            transition_count += 1
            
            # Update Q-values using sampled batch
            if transition_count % self.update_frequency == 0:
                batch_indices = self.sample_batch()
                for idx in batch_indices:
                    s_b, a_b, r_b, s2_b, d_b = self.buffer[idx]
                    self.policy_eval_improve(s_b, a_b, r_b, s2_b, d_b)
            
            if done:
                iter_episode += 1