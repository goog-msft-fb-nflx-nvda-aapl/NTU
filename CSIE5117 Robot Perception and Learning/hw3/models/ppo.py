"""
PPO implementation used for training on continuous-control environments.
Warm-up version for homework: students only implement the PPO objective.
"""

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self, policy_class, env, **hyperparameters):
        """
        PPO agent with separate actor / critic networks.
        """
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Env info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Actor / Critic
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
        self.critic = policy_class(self.obs_dim, 1).to(self.device)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Action distribution covariance
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # Logger
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,
            "i_so_far": 0,
            "batch_lens": [],
            "batch_rews": [],
            "actor_losses": [],
        }

    # ---------------------------------------------------------------------- #
    #                        Main training loop                              #
    # ---------------------------------------------------------------------- #

    def learn(self, total_timesteps: int):
        """
        Main PPO training loop.
        """
        print(
            f"Learning... {self.max_timesteps_per_episode} timesteps/episode, "
            f"{self.timesteps_per_batch} timesteps/batch, total {total_timesteps} timesteps"
        )

        t_so_far = 0  # total timesteps
        i_so_far = 0  # total iterations

        while t_so_far < total_timesteps:
            # ------------------------------------------------------------------
            # 1. Rollout: collect on-policy data
            # ------------------------------------------------------------------
            (
                batch_obs,
                batch_acts,
                batch_log_probs,
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
            ) = self.rollout()

            batch_obs = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)
            batch_log_probs = batch_log_probs.to(self.device)

            # ------------------------------------------------------------------
            # 2. GAE Advantage + target values
            # ------------------------------------------------------------------
            advantages = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(
                self.device
            )

            with torch.no_grad():
                values = self.critic(batch_obs).squeeze()
            batch_rtgs = advantages + values  # target values (V-target)

            # Update counters
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Normalize advantages (stabilizes training)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            # ------------------------------------------------------------------
            # 3. PPO update (mini-batch SGD for several epochs)
            # ------------------------------------------------------------------
            num_steps = batch_obs.size(0)
            indices = np.arange(num_steps)
            minibatch_size = num_steps // self.num_minibatches
            epoch_actor_losses = []

            for _ in range(self.n_updates_per_iteration):
                # Learning rate annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = max(self.lr * (1.0 - frac), 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                self.logger["lr"] = new_lr

                np.random.shuffle(indices)

                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = indices[start:end]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs = batch_obs[mb_idx]
                    mb_acts = batch_acts[mb_idx]
                    mb_log_probs_old = batch_log_probs[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_rtgs = batch_rtgs[mb_idx]

                    # Forward pass: evaluate V(s), log π(a|s), entropy
                    V_pred, log_probs_new, entropy = self.evaluate(mb_obs, mb_acts)

                    # ==========================================================
                    # TODO: Implement PPO clipped surrogate objective
                    #
                    # Given:
                    #   - log_probs_new: log πθ(a|s)
                    #   - mb_log_probs_old: log πθ_old(a|s)
                    #   - mb_adv: Advantage estimates A_t
                    #   - self.clip: ε (clip range)
                    # 
                    # 1. Compute the probability ratio r_t(θ):
                    # 2. Compute the clipped surrogate objective:
                    # 3. Add entropy bonus (to encourage exploration):
                    # 4. Compute value function loss (critic_loss) using MSE:
                    # 5. Compute approximate KL divergence for early stopping:
                    # ==========================================================

                    # ====== Your code starts here ======
                    # 1. Probability ratio r_t(θ) = π_θ(a|s) / π_old(a|s)
                    log_ratio = log_probs_new - mb_log_probs_old
                    ratio = torch.exp(log_ratio)

                    # 5. Approximate KL divergence for early stopping
                    # Using the unbiased estimator: KL ≈ (r - 1) - log(r)
                    approx_kl = ((ratio - 1) - log_ratio).mean().detach()

                    # 2. Clipped surrogate objective
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # 3. Entropy bonus to encourage exploration
                    actor_loss -= self.ent_coef * entropy.mean()

                    # 4. Value function loss (critic MSE)
                    critic_loss = nn.MSELoss()(V_pred, mb_rtgs.detach())
                    # ====== Your code ends here ======

                    # Backward: actor
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Backward: critic
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    epoch_actor_losses.append(actor_loss.detach())

                # Early stop if KL too large
                if approx_kl > self.target_kl:
                    break

            # ------------------------------------------------------------------
            # 4. Logging
            # ------------------------------------------------------------------
            avg_actor_loss = sum(epoch_actor_losses) / len(epoch_actor_losses)
            self.logger["actor_losses"].append(avg_actor_loss)

            if self.logger["i_so_far"] % 10 == 0:
                wandb.log({"advantage_hist": wandb.Histogram(advantages.cpu().numpy())})

            self._log_summary()

            # Save model periodically
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./checkpoints/ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./checkpoints/ppo_critic.pth")
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")

    # ---------------------------------------------------------------------- #
    #                          GAE & Rollout                                #
    # ---------------------------------------------------------------------- #

    def calculate_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) over a batch of episodes.
        """
        all_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0.0

            # backward through episode
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    delta = ep_rews[t] - ep_vals[t]

                adv = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = adv
                advantages.insert(0, adv)

            all_advantages.extend(advantages)

        return torch.tensor(all_advantages, dtype=torch.float32)

    def rollout(self):
        """
        Collect one batch of on-policy data.
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if (
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                ep_dones.append(done)

                t += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                value = self.critic(obs_tensor).detach()

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_rews.append(rew)
                ep_vals.append(value.item())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)

        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_lens,
            batch_vals,
            batch_dones,
        )

    # ---------------------------------------------------------------------- #
    #                         Action & Evaluation                            #
    # ---------------------------------------------------------------------- #

    def compute_rtgs(self, batch_rews):
        """
        (Legacy) Compute Reward-To-Go for each timestep in the batch.
        Not used when GAE is enabled, but kept for reference.
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return torch.tensor(batch_rtgs, dtype=torch.float32)

    def get_action(self, obs):
        """
        Sample an action from the current policy π(a|s).
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mean = self.actor(obs_tensor)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach().cpu()

    def evaluate(self, batch_obs, batch_acts):
        """
        Given a batch of (s, a), compute:
        - V(s)
        - log π(a|s)
        - entropy[π(·|s)]
        """
        values = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return values, log_probs, dist.entropy()

    # ---------------------------------------------------------------------- #
    #                        Hyperparameters & Logging                       #
    # ---------------------------------------------------------------------- #

    def _init_hyperparameters(self, hyperparameters):
        """
        Set default hyperparameters and override with user-provided values.
        """
        # Core PPO hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        # Extras
        self.lam = 0.98
        self.num_minibatches = 6
        self.ent_coef = 0.0
        self.target_kl = 0.02
        self.max_grad_norm = 0.5
        self.deterministic = False

        # Misc
        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None
        self.run_name = "unnamed_run"

        # Override defaults
        for param, val in hyperparameters.items():
            setattr(self, param, val)

        if self.seed is not None:
            assert isinstance(self.seed, int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Print and log statistics for the latest iteration.
        """
        delta_t_prev = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_len = np.mean(self.logger["batch_lens"])
        avg_ep_ret = np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
        avg_actor_loss = np.mean(
            [losses.float().mean().cpu().item() for losses in self.logger["actor_losses"]]
        )

        avg_ep_len = str(round(avg_ep_len, 2))
        avg_ep_ret = str(round(avg_ep_ret, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        print()
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_len}", flush=True)
        print(f"Average Episodic Return: {avg_ep_ret}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("------------------------------------------------------", flush=True)
        print()

        wandb.log(
            {
                "iteration": i_so_far,
                "timesteps_so_far": t_so_far,
                "avg_episode_length": float(avg_ep_len),
                "avg_episode_return": float(avg_ep_ret),
                "avg_actor_loss": float(avg_actor_loss),
                "iteration_duration_sec": float(delta_t),
            }
        )

        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
        
# """
# PPO implementation used for training on continuous-control environments.
# Warm-up version for homework: students only implement the PPO objective.
# """

# import time

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import wandb
# from torch.optim import Adam
# from torch.distributions import MultivariateNormal


# class PPO:
#     def __init__(self, policy_class, env, **hyperparameters):
#         """
#         PPO agent with separate actor / critic networks.
#         """
#         assert isinstance(env.observation_space, gym.spaces.Box)
#         assert isinstance(env.action_space, gym.spaces.Box)

#         self._init_hyperparameters(hyperparameters)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Env info
#         self.env = env
#         self.obs_dim = env.observation_space.shape[0]
#         self.act_dim = env.action_space.shape[0]

#         # Actor / Critic
#         self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
#         self.critic = policy_class(self.obs_dim, 1).to(self.device)

#         # Optimizers
#         self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

#         # Action distribution covariance
#         self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
#         self.cov_mat = torch.diag(self.cov_var).to(self.device)

#         # Logger
#         self.logger = {
#             "delta_t": time.time_ns(),
#             "t_so_far": 0,
#             "i_so_far": 0,
#             "batch_lens": [],
#             "batch_rews": [],
#             "actor_losses": [],
#         }

#     # ---------------------------------------------------------------------- #
#     #                        Main training loop                              #
#     # ---------------------------------------------------------------------- #

#     def learn(self, total_timesteps: int):
#         """
#         Main PPO training loop.
#         """
#         print(
#             f"Learning... {self.max_timesteps_per_episode} timesteps/episode, "
#             f"{self.timesteps_per_batch} timesteps/batch, total {total_timesteps} timesteps"
#         )

#         t_so_far = 0  # total timesteps
#         i_so_far = 0  # total iterations

#         while t_so_far < total_timesteps:
#             # ------------------------------------------------------------------
#             # 1. Rollout: collect on-policy data
#             # ------------------------------------------------------------------
#             (
#                 batch_obs,
#                 batch_acts,
#                 batch_log_probs,
#                 batch_rews,
#                 batch_lens,
#                 batch_vals,
#                 batch_dones,
#             ) = self.rollout()

#             batch_obs = batch_obs.to(self.device)
#             batch_acts = batch_acts.to(self.device)
#             batch_log_probs = batch_log_probs.to(self.device)

#             # ------------------------------------------------------------------
#             # 2. GAE Advantage + target values
#             # ------------------------------------------------------------------
#             advantages = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(
#                 self.device
#             )

#             with torch.no_grad():
#                 values = self.critic(batch_obs).squeeze()
#             batch_rtgs = advantages + values  # target values (V-target)

#             # Update counters
#             t_so_far += np.sum(batch_lens)
#             i_so_far += 1
#             self.logger["t_so_far"] = t_so_far
#             self.logger["i_so_far"] = i_so_far

#             # Normalize advantages (stabilizes training)
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

#             # ------------------------------------------------------------------
#             # 3. PPO update (mini-batch SGD for several epochs)
#             # ------------------------------------------------------------------
#             num_steps = batch_obs.size(0)
#             indices = np.arange(num_steps)
#             minibatch_size = num_steps // self.num_minibatches
#             epoch_actor_losses = []

#             for _ in range(self.n_updates_per_iteration):
#                 # Learning rate annealing
#                 frac = (t_so_far - 1.0) / total_timesteps
#                 new_lr = max(self.lr * (1.0 - frac), 0.0)
#                 self.actor_optim.param_groups[0]["lr"] = new_lr
#                 self.critic_optim.param_groups[0]["lr"] = new_lr
#                 self.logger["lr"] = new_lr

#                 np.random.shuffle(indices)

#                 for start in range(0, num_steps, minibatch_size):
#                     end = start + minibatch_size
#                     mb_idx = indices[start:end]
#                     if len(mb_idx) == 0:
#                         continue

#                     mb_obs = batch_obs[mb_idx]
#                     mb_acts = batch_acts[mb_idx]
#                     mb_log_probs_old = batch_log_probs[mb_idx]
#                     mb_adv = advantages[mb_idx]
#                     mb_rtgs = batch_rtgs[mb_idx]

#                     # Forward pass: evaluate V(s), log π(a|s), entropy
#                     V_pred, log_probs_new, entropy = self.evaluate(mb_obs, mb_acts)

#                     # ==========================================================
#                     # TODO: Implement PPO clipped surrogate objective
#                     #
#                     # Given:
#                     #   - log_probs_new: log πθ(a|s)
#                     #   - mb_log_probs_old: log πθ_old(a|s)
#                     #   - mb_adv: Advantage estimates A_t
#                     #   - self.clip: ε (clip range)
#                     # 
#                     # 1. Compute the probability ratio r_t(θ):
#                     # 2. Compute the clipped surrogate objective:
#                     # 3. Add entropy bonus (to encourage exploration):
#                     # 4. Compute value function loss (critic_loss) using MSE:
#                     # 5. Compute approximate KL divergence for early stopping:
#                     # ==========================================================

#                     # ====== Your code starts here ======
#                     # log_ratio = ...
#                     # ratio = ...
#                     # approx_kl = ...

#                     # surr1 = ...
#                     # surr2 = ...
#                     # actor_loss = ...
#                     # actor_loss -= ...

#                     # critic_loss = ...
#                     raise NotImplementedError("PPO clipped objective is not implemented yet.")
#                     # ====== Your code ends here ======

#                     # Backward: actor
#                     self.actor_optim.zero_grad()
#                     actor_loss.backward(retain_graph=True)
#                     nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
#                     self.actor_optim.step()

#                     # Backward: critic
#                     self.critic_optim.zero_grad()
#                     critic_loss.backward()
#                     nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#                     self.critic_optim.step()

#                     epoch_actor_losses.append(actor_loss.detach())

#                 # Early stop if KL too large
#                 if approx_kl > self.target_kl:
#                     break

#             # ------------------------------------------------------------------
#             # 4. Logging
#             # ------------------------------------------------------------------
#             avg_actor_loss = sum(epoch_actor_losses) / len(epoch_actor_losses)
#             self.logger["actor_losses"].append(avg_actor_loss)

#             if self.logger["i_so_far"] % 10 == 0:
#                 wandb.log({"advantage_hist": wandb.Histogram(advantages.cpu().numpy())})

#             self._log_summary()

#             # Save model periodically
#             if i_so_far % self.save_freq == 0:
#                 torch.save(self.actor.state_dict(), "./checkpoints/ppo_actor.pth")
#                 torch.save(self.critic.state_dict(), "./checkpoints/ppo_critic.pth")
#                 wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
#                 wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")

#     # ---------------------------------------------------------------------- #
#     #                          GAE & Rollout                                #
#     # ---------------------------------------------------------------------- #

#     def calculate_gae(self, rewards, values, dones):
#         """
#         Compute Generalized Advantage Estimation (GAE) over a batch of episodes.
#         """
#         all_advantages = []

#         for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
#             advantages = []
#             last_advantage = 0.0

#             # backward through episode
#             for t in reversed(range(len(ep_rews))):
#                 if t + 1 < len(ep_rews):
#                     delta = (
#                         ep_rews[t]
#                         + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
#                         - ep_vals[t]
#                     )
#                 else:
#                     delta = ep_rews[t] - ep_vals[t]

#                 adv = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
#                 last_advantage = adv
#                 advantages.insert(0, adv)

#             all_advantages.extend(advantages)

#         return torch.tensor(all_advantages, dtype=torch.float32)

#     def rollout(self):
#         """
#         Collect one batch of on-policy data.
#         """
#         batch_obs = []
#         batch_acts = []
#         batch_log_probs = []
#         batch_rews = []
#         batch_lens = []
#         batch_vals = []
#         batch_dones = []

#         t = 0

#         while t < self.timesteps_per_batch:
#             ep_rews = []
#             ep_vals = []
#             ep_dones = []

#             obs, _ = self.env.reset()
#             done = False

#             for ep_t in range(self.max_timesteps_per_episode):
#                 if (
#                     self.render
#                     and (self.logger["i_so_far"] % self.render_every_i == 0)
#                     and len(batch_lens) == 0
#                 ):
#                     self.env.render()

#                 ep_dones.append(done)

#                 t += 1
#                 batch_obs.append(obs)

#                 action, log_prob = self.get_action(obs)
#                 obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
#                 value = self.critic(obs_tensor).detach()

#                 obs, rew, terminated, truncated, _ = self.env.step(action)
#                 done = terminated or truncated

#                 ep_rews.append(rew)
#                 ep_vals.append(value.item())
#                 batch_acts.append(action)
#                 batch_log_probs.append(log_prob)

#                 if done:
#                     break

#             batch_lens.append(ep_t + 1)
#             batch_rews.append(ep_rews)
#             batch_vals.append(ep_vals)
#             batch_dones.append(ep_dones)

#         batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
#         batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
#         batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)

#         self.logger["batch_rews"] = batch_rews
#         self.logger["batch_lens"] = batch_lens

#         return (
#             batch_obs,
#             batch_acts,
#             batch_log_probs,
#             batch_rews,
#             batch_lens,
#             batch_vals,
#             batch_dones,
#         )

#     # ---------------------------------------------------------------------- #
#     #                         Action & Evaluation                            #
#     # ---------------------------------------------------------------------- #

#     def compute_rtgs(self, batch_rews):
#         """
#         (Legacy) Compute Reward-To-Go for each timestep in the batch.
#         Not used when GAE is enabled, but kept for reference.
#         """
#         batch_rtgs = []

#         for ep_rews in reversed(batch_rews):
#             discounted_reward = 0
#             for rew in reversed(ep_rews):
#                 discounted_reward = rew + discounted_reward * self.gamma
#                 batch_rtgs.insert(0, discounted_reward)

#         return torch.tensor(batch_rtgs, dtype=torch.float32)

#     def get_action(self, obs):
#         """
#         Sample an action from the current policy π(a|s).
#         """
#         obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
#         mean = self.actor(obs_tensor)
#         dist = MultivariateNormal(mean, self.cov_mat)

#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         return action.detach().cpu().numpy(), log_prob.detach().cpu()

#     def evaluate(self, batch_obs, batch_acts):
#         """
#         Given a batch of (s, a), compute:
#         - V(s)
#         - log π(a|s)
#         - entropy[π(·|s)]
#         """
#         values = self.critic(batch_obs).squeeze()

#         mean = self.actor(batch_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)
#         log_probs = dist.log_prob(batch_acts)

#         return values, log_probs, dist.entropy()

#     # ---------------------------------------------------------------------- #
#     #                        Hyperparameters & Logging                       #
#     # ---------------------------------------------------------------------- #

#     def _init_hyperparameters(self, hyperparameters):
#         """
#         Set default hyperparameters and override with user-provided values.
#         """
#         # Core PPO hyperparameters
#         self.timesteps_per_batch = 4800
#         self.max_timesteps_per_episode = 1600
#         self.n_updates_per_iteration = 5
#         self.lr = 0.005
#         self.gamma = 0.95
#         self.clip = 0.2

#         # Extras
#         self.lam = 0.98
#         self.num_minibatches = 6
#         self.ent_coef = 0.0
#         self.target_kl = 0.02
#         self.max_grad_norm = 0.5
#         self.deterministic = False

#         # Misc
#         self.render = True
#         self.render_every_i = 10
#         self.save_freq = 10
#         self.seed = None
#         self.run_name = "unnamed_run"

#         # Override defaults
#         for param, val in hyperparameters.items():
#             setattr(self, param, val)

#         if self.seed is not None:
#             assert isinstance(self.seed, int)
#             torch.manual_seed(self.seed)
#             print(f"Successfully set seed to {self.seed}")

#     def _log_summary(self):
#         """
#         Print and log statistics for the latest iteration.
#         """
#         delta_t_prev = self.logger["delta_t"]
#         self.logger["delta_t"] = time.time_ns()
#         delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
#         delta_t = str(round(delta_t, 2))

#         t_so_far = self.logger["t_so_far"]
#         i_so_far = self.logger["i_so_far"]
#         avg_ep_len = np.mean(self.logger["batch_lens"])
#         avg_ep_ret = np.mean([np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]])
#         avg_actor_loss = np.mean(
#             [losses.float().mean().cpu().item() for losses in self.logger["actor_losses"]]
#         )

#         avg_ep_len = str(round(avg_ep_len, 2))
#         avg_ep_ret = str(round(avg_ep_ret, 2))
#         avg_actor_loss = str(round(avg_actor_loss, 5))

#         print()
#         print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
#         print(f"Average Episodic Length: {avg_ep_len}", flush=True)
#         print(f"Average Episodic Return: {avg_ep_ret}", flush=True)
#         print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
#         print(f"Timesteps So Far: {t_so_far}", flush=True)
#         print(f"Iteration took: {delta_t} secs", flush=True)
#         print("------------------------------------------------------", flush=True)
#         print()

#         wandb.log(
#             {
#                 "iteration": i_so_far,
#                 "timesteps_so_far": t_so_far,
#                 "avg_episode_length": float(avg_ep_len),
#                 "avg_episode_return": float(avg_ep_ret),
#                 "avg_actor_loss": float(avg_actor_loss),
#                 "iteration_duration_sec": float(delta_t),
#             }
#         )

#         self.logger["batch_lens"] = []
#         self.logger["batch_rews"] = []
#         self.logger["actor_losses"] = []
