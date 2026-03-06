"""
TRPO implementation.

This class follows the standard TRPO pipeline:
1. Collect on-policy rollouts.
2. Estimate advantages with GAE.
3. Update the actor with a trust-region update (KL constraint + conjugate gradient + line search).
4. Update the critic via value regression (MSE).

Homework Part 2 (TRPO):
- You need to complete the TODO parts:
  (1) _surrogate_loss_and_kl(...)
  (2) Trust-region step scaling and line search acceptance in learn().
"""

import time
import gymnasium as gym
import wandb
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import MultivariateNormal


class TRPO:
    """
    TRPO agent used as the main RL algorithm.
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
        Initialize TRPO agent and hyperparameters.

        Args:
            policy_class: nn.Module class used for both actor and critic.
            env: gymnasium environment with Box observation & action spaces.
            hyperparameters: additional keyword arguments overriding default hparams.
        """
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Actor / Critic
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
        self.critic = policy_class(self.obs_dim, 1).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Fixed diagonal covariance for continuous actions
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

    # ====================================================================== #
    #                               TRAIN LOOP                               #
    # ====================================================================== #

    def learn(self, total_timesteps: int):
        """
        Main TRPO training loop.

        - Collect rollouts.
        - Compute GAE advantages.
        - TRPO trust-region update for actor.
        - Value-function regression for critic.
        """
        print(
            f"Learning with TRPO... Running {self.max_timesteps_per_episode} timesteps "
            f"per episode, {self.timesteps_per_batch} timesteps per batch for a total of "
            f"{total_timesteps} timesteps"
        )

        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            # -------- Rollout -------- #
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

            # -------- GAE Advantage -------- #
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)

            with torch.no_grad():
                V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V

            # Update counters
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # -------- TRPO Actor Update -------- #
            with torch.no_grad():
                old_mean = self.actor(batch_obs)
                old_dist = MultivariateNormal(old_mean, self.cov_mat)

            # Old loss as baseline for line search
            old_loss_pi, _ = self._surrogate_loss_and_kl(
                batch_obs, batch_acts, batch_log_probs, A_k, old_dist
            )
            old_loss_pi = old_loss_pi.detach()

            # Gradient of current surrogate loss
            loss_pi, _ = self._surrogate_loss_and_kl(
                batch_obs, batch_acts, batch_log_probs, A_k, old_dist
            )
            g = self._flat_grad(loss_pi, self.actor, retain_graph=True).detach()

            # Conjugate gradient to get approximate natural gradient step
            def Avp_func(v):
                return self._fisher_vector_product(batch_obs, old_dist, v)

            step_dir = self._conjugate_gradient(Avp_func, g, self.cg_iters)

            # --------------------------------------------------------------
            # TODO 1: Trust-region step scaling
            #
            # We want to scale the direction `step_dir` so that:
            #   0.5 * step^T F step <= max_kl
            #
            # Hints:
            #   F_step = Avp_func(step_dir)          # F @ step_dir
            #   step_dir_F_step = (step_dir * F_step).sum()   # s^T F s
            #   step_size = sqrt(2 * max_kl / (step_dir_F_step + 1e-8))
            #   full_step = -step_size * step_dir    # negative because we minimize loss
            # --------------------------------------------------------------

            # ===== Your code (trust-region scaling) starts here =====
            F_step = Avp_func(step_dir)                            # F @ step_dir
            step_dir_F_step = (step_dir * F_step).sum()           # s^T F s
            step_size = torch.sqrt(
                2.0 * self.max_kl / (step_dir_F_step + 1e-8)
            )
            full_step = -step_size * step_dir                      # negative: we minimize
            # ===== Your code (trust-region scaling) ends here =====

            # Store old parameters for line search
            old_params = self._get_flat_params(self.actor)

            # -------- Line Search -------- #
            success = False
            for j in range(self.backtrack_iters):
                coeff = self.backtrack_coeff ** j
                new_params = old_params + coeff * full_step
                self._set_flat_params(self.actor, new_params)

                loss_pi_new, kl_new = self._surrogate_loss_and_kl(
                    batch_obs, batch_acts, batch_log_probs, A_k, old_dist
                )
                loss_pi_new_val = loss_pi_new.detach().item()
                kl_new_val = kl_new.detach().item()

                # ----------------------------------------------------------
                # TODO 2: Line search acceptance rule
                #
                # We only accept this update if:
                #   (1) The new surrogate loss is not worse than the old one.
                #   (2) The new KL divergence is within the trust region.
                #
                # That is, in math:
                #   L_new <= L_old    and    KL_new <= max_kl
                #
                # Please implement the condition below.
                # ----------------------------------------------------------

                # ===== Your code (line search condition) starts here =====
                if loss_pi_new_val <= old_loss_pi.item() and kl_new_val <= self.max_kl:
                    success = True
                    break
                # ===== Your code (line search condition) ends here =====

            if not success:
                # Restore old params if line search fails
                self._set_flat_params(self.actor, old_params)
                print("[TRPO] Line search failed. Keeping old parameters.")

            # -------- Critic Update (Value Regression) -------- #
            for _ in range(self.n_updates_per_iteration):
                self.critic_optim.zero_grad()
                V_pred = self.critic(batch_obs).squeeze()
                value_loss = nn.MSELoss()(V_pred, batch_rtgs.detach())
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

            # -------- Logging -------- #
            final_loss_pi, _ = self._surrogate_loss_and_kl(
                batch_obs, batch_acts, batch_log_probs, A_k, old_dist
            )
            self.logger["actor_losses"].append(final_loss_pi.detach())

            if i_so_far % 10 == 0:
                wandb.log({"advantage_hist": wandb.Histogram(A_k.cpu().numpy())})

            self._log_summary()

            # Save model periodically
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./checkpoints/trpo_actor.pth")
                torch.save(self.critic.state_dict(), "./checkpoints/trpo_critic.pth")
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")

    # ====================================================================== #
    #                         ROLLOUT & ADVANTAGE                            #
    # ====================================================================== #
    
    def calculate_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) over a batch of episodes.
        """
        batch_advantages = []

        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0.0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = (
                        ep_rews[t]
                        + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
                        - ep_vals[t]
                    )
                else:
                    # Last step in episode
                    delta = ep_rews[t] - ep_vals[t]

                advantage = (
                    delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                )
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def rollout(self):
        """
        Collect on-policy trajectories until `timesteps_per_batch` is reached.
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
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                val = self.critic(obs_tensor).detach()

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_rews.append(rew)
                ep_vals.append(val.item())
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

    def compute_rtgs(self, batch_rews):
        """
        (Optional) Reward-to-go computation if you want to use it instead of GAE.
        """
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0.0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return torch.tensor(batch_rtgs, dtype=torch.float32)

    # ====================================================================== #
    #                           POLICY & VALUE                               #
    # ====================================================================== #

    @torch.no_grad()
    def get_action(self, obs):
        """
        Sample an action from the current policy.

        Args:
            obs: numpy observation.

        Returns:
            action (np.ndarray), log_prob (torch.Tensor on CPU)
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mean = self.actor(obs_tensor)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu()

    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluate value function and log-probs for given batch.
        """
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()

        return V, log_probs, entropy
    
    # ====================================================================== #
    #                        HYPERPARAMETERS & UTILS                         #
    # ====================================================================== #
    
    def _init_hyperparameters(self, hyperparameters):
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5  # critic updates per iteration
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2  # kept for compatibility; unused in TRPO

        # Extras
        self.lam = 0.98  # GAE lambda
        self.num_minibatches = 6  # unused in TRPO
        self.ent_coef = 0.0
        self.target_kl = 0.02  # PPO-style; TRPO uses max_kl instead

        # TRPO specific
        self.max_kl = 0.01
        self.cg_iters = 10
        self.cg_damping = 0.1
        self.backtrack_coeff = 0.8
        self.backtrack_iters = 10

        # Misc
        self.max_grad_norm = 0.5
        self.deterministic = False

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

    # ---------- Flat parameter helpers ---------- #

    def _get_flat_params(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    def _set_flat_params(self, model, flat_params):
        prev_ind = 0
        for p in model.parameters():
            flat_size = p.numel()
            p.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view_as(p))
            prev_ind += flat_size

    def _flat_grad(self, loss, model, retain_graph=False, create_graph=False):
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            retain_graph=retain_graph,
            create_graph=create_graph,
        )
        return torch.cat([g.contiguous().view(-1) for g in grads])

    # ---------- Conjugate Gradient & Fisher-Vector Product ---------- #

    def _conjugate_gradient(self, Avp_func, b, nsteps, residual_tol=1e-10):
        """
        Solve Ax = b using conjugate gradient where Avp_func(v) = A @ v.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(nsteps):
            Avp = Avp_func(p)
            denom = torch.dot(p, Avp) + 1e-8
            alpha = rdotr / denom
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr

        return x


    def _fisher_vector_product(self, batch_obs, old_dist, v):
        """
        Fisher-vector product: Fv ≈ ∇²_θ KL(π_old || π_θ) v + damping * v
        """
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, self.actor.parameters())
        flat_fvp = torch.cat([g.contiguous().view(-1) for g in grads2]).detach()

        return flat_fvp + self.cg_damping * v

    def _surrogate_loss_and_kl(
        self, batch_obs, batch_acts, batch_log_probs_old, advantages, old_dist
    ):
        """
        Compute TRPO surrogate loss and KL(π_old || π).

        Homework TODO (Priority 1):
        - Implement the TRPO surrogate objective:
              L(θ) = E[ r_t(θ) * A_t ]
          where r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
        - Compute KL(π_old || π_θ) for the trust-region constraint.
        """

        # --------------------------------------------------------------
        # TODO 3 Surrogate loss and KL
        #
        # Given:
        #   - batch_obs: states s_t
        #   - batch_acts: actions a_t
        #   - batch_log_probs_old: log π_old(a_t|s_t)
        #   - advantages: A_t
        #   - old_dist: π_old(·|s) as a MultivariateNormal
        #
        # Steps:
        #   1. Build current policy distribution:
        #   2. Compute log_probs under current policy:
        #   3. Compute ratios r_t(θ):
        #   4. Surrogate loss (note the negative sign, since we MINIMIZE):
        #   5. Add entropy bonus:
        #   6. KL divergence (for trust region):
        #
        # Return:
        #   loss_pi, kl
        # --------------------------------------------------------------

        # ===== Your code (surrogate loss + KL) starts here =====
        # 1. Build current policy distribution
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        # 2. Compute log probs under current policy
        log_probs = dist.log_prob(batch_acts)

        # 3. Compute importance ratios r_t(θ)
        ratios = torch.exp(log_probs - batch_log_probs_old)

        # 4. Surrogate loss — negative because we MINIMIZE (optimizer does gradient descent)
        surr_loss = -(ratios * advantages).mean()

        # 5. Entropy bonus
        entropy = dist.entropy().mean()
        loss_pi = surr_loss - self.ent_coef * entropy

        # 6. KL divergence KL(π_old || π_θ) for trust region constraint
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

        return loss_pi, kl
        # ===== Your code (surrogate loss + KL) ends here =====

    # ====================================================================== #
    #                                LOGGING                                 #
    # ====================================================================== #
    
    def _log_summary(self):
        """
        Print and log summary stats for the latest iteration.
        """
        delta_t_prev = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
        delta_t = round(delta_t, 2)

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = float(np.mean(self.logger["batch_lens"]))
        avg_ep_rews = float(np.mean([np.sum(ep) for ep in self.logger["batch_rews"]]))
        avg_actor_loss = float(
            np.mean([loss.float().mean().cpu().item() for loss in self.logger["actor_losses"]])
        )

        print()
        print(f"-------------------- Iteration #{i_so_far} --------------------")
        print(f"Average Episodic Length: {avg_ep_lens}")
        print(f"Average Episodic Return: {avg_ep_rews}")
        print(f"Average Loss: {avg_actor_loss}")
        print(f"Timesteps So Far: {t_so_far}")
        print(f"Iteration took: {delta_t} secs")
        print("------------------------------------------------------")
        print()

        wandb.log(
            {
                "iteration": i_so_far,
                "timesteps_so_far": t_so_far,
                "avg_episode_length": avg_ep_lens,
                "avg_episode_return": avg_ep_rews,
                "avg_actor_loss": avg_actor_loss,
                "iteration_duration_sec": delta_t,
            }
        )

        # Reset batch-specific logs
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
        
# """
# TRPO implementation.

# This class follows the standard TRPO pipeline:
# 1. Collect on-policy rollouts.
# 2. Estimate advantages with GAE.
# 3. Update the actor with a trust-region update (KL constraint + conjugate gradient + line search).
# 4. Update the critic via value regression (MSE).

# Homework Part 2 (TRPO):
# - You need to complete the TODO parts:
#   (1) _surrogate_loss_and_kl(...)
#   (2) Trust-region step scaling and line search acceptance in learn().
# """

# import time
# import gymnasium as gym
# import wandb
# import numpy as np
# import torch
# import torch.nn as nn

# from torch.optim import Adam
# from torch.distributions import MultivariateNormal


# class TRPO:
#     """
#     TRPO agent used as the main RL algorithm.
#     """

#     def __init__(self, policy_class, env, **hyperparameters):
#         """
#         Initialize TRPO agent and hyperparameters.

#         Args:
#             policy_class: nn.Module class used for both actor and critic.
#             env: gymnasium environment with Box observation & action spaces.
#             hyperparameters: additional keyword arguments overriding default hparams.
#         """
#         assert isinstance(env.observation_space, gym.spaces.Box)
#         assert isinstance(env.action_space, gym.spaces.Box)

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Initialize hyperparameters
#         self._init_hyperparameters(hyperparameters)

#         # Environment info
#         self.env = env
#         self.obs_dim = env.observation_space.shape[0]
#         self.act_dim = env.action_space.shape[0]

#         # Actor / Critic
#         self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
#         self.critic = policy_class(self.obs_dim, 1).to(self.device)

#         self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

#         # Fixed diagonal covariance for continuous actions
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

#     # ====================================================================== #
#     #                               TRAIN LOOP                               #
#     # ====================================================================== #

#     def learn(self, total_timesteps: int):
#         """
#         Main TRPO training loop.

#         - Collect rollouts.
#         - Compute GAE advantages.
#         - TRPO trust-region update for actor.
#         - Value-function regression for critic.
#         """
#         print(
#             f"Learning with TRPO... Running {self.max_timesteps_per_episode} timesteps "
#             f"per episode, {self.timesteps_per_batch} timesteps per batch for a total of "
#             f"{total_timesteps} timesteps"
#         )

#         t_so_far = 0
#         i_so_far = 0

#         while t_so_far < total_timesteps:
#             # -------- Rollout -------- #
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

#             # -------- GAE Advantage -------- #
#             A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)

#             with torch.no_grad():
#                 V = self.critic(batch_obs).squeeze()
#             batch_rtgs = A_k + V

#             # Update counters
#             t_so_far += np.sum(batch_lens)
#             i_so_far += 1
#             self.logger["t_so_far"] = t_so_far
#             self.logger["i_so_far"] = i_so_far

#             # Normalize advantages
#             A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

#             # -------- TRPO Actor Update -------- #
#             with torch.no_grad():
#                 old_mean = self.actor(batch_obs)
#                 old_dist = MultivariateNormal(old_mean, self.cov_mat)

#             # Old loss as baseline for line search
#             old_loss_pi, _ = self._surrogate_loss_and_kl(
#                 batch_obs, batch_acts, batch_log_probs, A_k, old_dist
#             )
#             old_loss_pi = old_loss_pi.detach()

#             # Gradient of current surrogate loss
#             loss_pi, _ = self._surrogate_loss_and_kl(
#                 batch_obs, batch_acts, batch_log_probs, A_k, old_dist
#             )
#             g = self._flat_grad(loss_pi, self.actor, retain_graph=True).detach()

#             # Conjugate gradient to get approximate natural gradient step
#             def Avp_func(v):
#                 return self._fisher_vector_product(batch_obs, old_dist, v)

#             step_dir = self._conjugate_gradient(Avp_func, g, self.cg_iters)

#             # --------------------------------------------------------------
#             # TODO 1: Trust-region step scaling
#             #
#             # We want to scale the direction `step_dir` so that:
#             #   0.5 * step^T F step <= max_kl
#             #
#             # Hints:
#             #   F_step = Avp_func(step_dir)          # F @ step_dir
#             #   step_dir_F_step = (step_dir * F_step).sum()   # s^T F s
#             #   step_size = sqrt(2 * max_kl / (step_dir_F_step + 1e-8))
#             #   full_step = -step_size * step_dir    # negative because we minimize loss
#             # --------------------------------------------------------------

#             # ===== Your code (trust-region scaling) starts here =====
#             # F_step = ...
#             # step_dir_F_step = ...
#             # step_size = ...
#             # full_step = ...

#             raise NotImplementedError("TRPO trust-region step scaling is not implemented yet.")
#             # ===== Your code (trust-region scaling) ends here =====

#             # Store old parameters for line search
#             old_params = self._get_flat_params(self.actor)

#             # -------- Line Search -------- #
#             success = False
#             for j in range(self.backtrack_iters):
#                 coeff = self.backtrack_coeff ** j
#                 new_params = old_params + coeff * full_step
#                 self._set_flat_params(self.actor, new_params)

#                 loss_pi_new, kl_new = self._surrogate_loss_and_kl(
#                     batch_obs, batch_acts, batch_log_probs, A_k, old_dist
#                 )
#                 loss_pi_new_val = loss_pi_new.detach().item()
#                 kl_new_val = kl_new.detach().item()

#                 # ----------------------------------------------------------
#                 # TODO 2: Line search acceptance rule
#                 #
#                 # We only accept this update if:
#                 #   (1) The new surrogate loss is not worse than the old one.
#                 #   (2) The new KL divergence is within the trust region.
#                 #
#                 # That is, in math:
#                 #   L_new <= L_old    and    KL_new <= max_kl
#                 #
#                 # Please implement the condition below.
#                 # ----------------------------------------------------------

#                 # ===== Your code (line search condition) starts here =====
#                 # if ( ... ):
#                 #     success = True
#                 #     break
#                 raise NotImplementedError("TRPO line search acceptance rule is not implemented yet.")
#                 # ===== Your code (line search condition) ends here =====

#             if not success:
#                 # Restore old params if line search fails
#                 self._set_flat_params(self.actor, old_params)
#                 print("[TRPO] Line search failed. Keeping old parameters.")

#             # -------- Critic Update (Value Regression) -------- #
#             for _ in range(self.n_updates_per_iteration):
#                 self.critic_optim.zero_grad()
#                 V_pred = self.critic(batch_obs).squeeze()
#                 value_loss = nn.MSELoss()(V_pred, batch_rtgs.detach())
#                 value_loss.backward()
#                 nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#                 self.critic_optim.step()

#             # -------- Logging -------- #
#             final_loss_pi, _ = self._surrogate_loss_and_kl(
#                 batch_obs, batch_acts, batch_log_probs, A_k, old_dist
#             )
#             self.logger["actor_losses"].append(final_loss_pi.detach())

#             if i_so_far % 10 == 0:
#                 wandb.log({"advantage_hist": wandb.Histogram(A_k.cpu().numpy())})

#             self._log_summary()

#             # Save model periodically
#             if i_so_far % self.save_freq == 0:
#                 torch.save(self.actor.state_dict(), "./checkpoints/trpo_actor.pth")
#                 torch.save(self.critic.state_dict(), "./checkpoints/trpo_critic.pth")
#                 wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
#                 wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")

#     # ====================================================================== #
#     #                         ROLLOUT & ADVANTAGE                            #
#     # ====================================================================== #
    
#     def calculate_gae(self, rewards, values, dones):
#         """
#         Compute Generalized Advantage Estimation (GAE) over a batch of episodes.
#         """
#         batch_advantages = []

#         for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
#             advantages = []
#             last_advantage = 0.0

#             for t in reversed(range(len(ep_rews))):
#                 if t + 1 < len(ep_rews):
#                     delta = (
#                         ep_rews[t]
#                         + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1])
#                         - ep_vals[t]
#                     )
#                 else:
#                     # Last step in episode
#                     delta = ep_rews[t] - ep_vals[t]

#                 advantage = (
#                     delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
#                 )
#                 last_advantage = advantage
#                 advantages.insert(0, advantage)

#             batch_advantages.extend(advantages)

#         return torch.tensor(batch_advantages, dtype=torch.float)

#     def rollout(self):
#         """
#         Collect on-policy trajectories until `timesteps_per_batch` is reached.
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
#                 obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
#                 val = self.critic(obs_tensor).detach()

#                 obs, rew, terminated, truncated, _ = self.env.step(action)
#                 done = terminated or truncated

#                 ep_rews.append(rew)
#                 ep_vals.append(val.item())
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

#     def compute_rtgs(self, batch_rews):
#         """
#         (Optional) Reward-to-go computation if you want to use it instead of GAE.
#         """
#         batch_rtgs = []

#         for ep_rews in reversed(batch_rews):
#             discounted_reward = 0.0
#             for rew in reversed(ep_rews):
#                 discounted_reward = rew + discounted_reward * self.gamma
#                 batch_rtgs.insert(0, discounted_reward)

#         return torch.tensor(batch_rtgs, dtype=torch.float32)

#     # ====================================================================== #
#     #                           POLICY & VALUE                               #
#     # ====================================================================== #

#     @torch.no_grad()
#     def get_action(self, obs):
#         """
#         Sample an action from the current policy.

#         Args:
#             obs: numpy observation.

#         Returns:
#             action (np.ndarray), log_prob (torch.Tensor on CPU)
#         """
#         obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
#         mean = self.actor(obs_tensor)
#         dist = MultivariateNormal(mean, self.cov_mat)

#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         return action.cpu().numpy(), log_prob.cpu()

#     def evaluate(self, batch_obs, batch_acts):
#         """
#         Evaluate value function and log-probs for given batch.
#         """
#         V = self.critic(batch_obs).squeeze()
#         mean = self.actor(batch_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)
#         log_probs = dist.log_prob(batch_acts)
#         entropy = dist.entropy()

#         return V, log_probs, entropy
    
#     # ====================================================================== #
#     #                        HYPERPARAMETERS & UTILS                         #
#     # ====================================================================== #
    
#     def _init_hyperparameters(self, hyperparameters):
#         # Algorithm hyperparameters
#         self.timesteps_per_batch = 4800
#         self.max_timesteps_per_episode = 1600
#         self.n_updates_per_iteration = 5  # critic updates per iteration
#         self.lr = 0.005
#         self.gamma = 0.95
#         self.clip = 0.2  # kept for compatibility; unused in TRPO

#         # Extras
#         self.lam = 0.98  # GAE lambda
#         self.num_minibatches = 6  # unused in TRPO
#         self.ent_coef = 0.0
#         self.target_kl = 0.02  # PPO-style; TRPO uses max_kl instead

#         # TRPO specific
#         self.max_kl = 0.01
#         self.cg_iters = 10
#         self.cg_damping = 0.1
#         self.backtrack_coeff = 0.8
#         self.backtrack_iters = 10

#         # Misc
#         self.max_grad_norm = 0.5
#         self.deterministic = False

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

#     # ---------- Flat parameter helpers ---------- #

#     def _get_flat_params(self, model):
#         return torch.cat([p.data.view(-1) for p in model.parameters()])

#     def _set_flat_params(self, model, flat_params):
#         prev_ind = 0
#         for p in model.parameters():
#             flat_size = p.numel()
#             p.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view_as(p))
#             prev_ind += flat_size

#     def _flat_grad(self, loss, model, retain_graph=False, create_graph=False):
#         grads = torch.autograd.grad(
#             loss,
#             model.parameters(),
#             retain_graph=retain_graph,
#             create_graph=create_graph,
#         )
#         return torch.cat([g.contiguous().view(-1) for g in grads])

#     # ---------- Conjugate Gradient & Fisher-Vector Product ---------- #

#     def _conjugate_gradient(self, Avp_func, b, nsteps, residual_tol=1e-10):
#         """
#         Solve Ax = b using conjugate gradient where Avp_func(v) = A @ v.
#         """
#         x = torch.zeros_like(b)
#         r = b.clone()
#         p = b.clone()
#         rdotr = torch.dot(r, r)

#         for _ in range(nsteps):
#             Avp = Avp_func(p)
#             denom = torch.dot(p, Avp) + 1e-8
#             alpha = rdotr / denom
#             x += alpha * p
#             r -= alpha * Avp
#             new_rdotr = torch.dot(r, r)
#             if new_rdotr < residual_tol:
#                 break
#             beta = new_rdotr / (rdotr + 1e-8)
#             p = r + beta * p
#             rdotr = new_rdotr

#         return x


#     def _fisher_vector_product(self, batch_obs, old_dist, v):
#         """
#         Fisher-vector product: Fv ≈ ∇²_θ KL(π_old || π_θ) v + damping * v
#         """
#         mean = self.actor(batch_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)

#         kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

#         grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
#         flat_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads])

#         kl_v = (flat_grad_kl * v).sum()
#         grads2 = torch.autograd.grad(kl_v, self.actor.parameters())
#         flat_fvp = torch.cat([g.contiguous().view(-1) for g in grads2]).detach()

#         return flat_fvp + self.cg_damping * v

#     def _surrogate_loss_and_kl(
#         self, batch_obs, batch_acts, batch_log_probs_old, advantages, old_dist
#     ):
#         """
#         Compute TRPO surrogate loss and KL(π_old || π).

#         Homework TODO (Priority 1):
#         - Implement the TRPO surrogate objective:
#               L(θ) = E[ r_t(θ) * A_t ]
#           where r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
#         - Compute KL(π_old || π_θ) for the trust-region constraint.
#         """

#         # --------------------------------------------------------------
#         # TODO 3 Surrogate loss and KL
#         #
#         # Given:
#         #   - batch_obs: states s_t
#         #   - batch_acts: actions a_t
#         #   - batch_log_probs_old: log π_old(a_t|s_t)
#         #   - advantages: A_t
#         #   - old_dist: π_old(·|s) as a MultivariateNormal
#         #
#         # Steps:
#         #   1. Build current policy distribution:
#         #   2. Compute log_probs under current policy:
#         #   3. Compute ratios r_t(θ):
#         #   4. Surrogate loss (note the negative sign, since we MINIMIZE):
#         #   5. Add entropy bonus:
#         #   6. KL divergence (for trust region):
#         #
#         # Return:
#         #   loss_pi, kl
#         # --------------------------------------------------------------

#         # ===== Your code (surrogate loss + KL) starts here =====
#         # mean = ...
#         # dist = ...
#         # log_probs = ...
#         # ratios = ...
#         # surr_loss = ...
#         # entropy = ...
#         # loss_pi = ...
#         # kl = ...
#         raise NotImplementedError("TRPO surrogate loss and KL are not implemented yet.")
#         # ===== Your code (surrogate loss + KL) ends here =====

#     # ====================================================================== #
#     #                                LOGGING                                 #
#     # ====================================================================== #
    
#     def _log_summary(self):
#         """
#         Print and log summary stats for the latest iteration.
#         """
#         delta_t_prev = self.logger["delta_t"]
#         self.logger["delta_t"] = time.time_ns()
#         delta_t = (self.logger["delta_t"] - delta_t_prev) / 1e9
#         delta_t = round(delta_t, 2)

#         t_so_far = self.logger["t_so_far"]
#         i_so_far = self.logger["i_so_far"]
#         avg_ep_lens = float(np.mean(self.logger["batch_lens"]))
#         avg_ep_rews = float(np.mean([np.sum(ep) for ep in self.logger["batch_rews"]]))
#         avg_actor_loss = float(
#             np.mean([loss.float().mean().cpu().item() for loss in self.logger["actor_losses"]])
#         )

#         print()
#         print(f"-------------------- Iteration #{i_so_far} --------------------")
#         print(f"Average Episodic Length: {avg_ep_lens}")
#         print(f"Average Episodic Return: {avg_ep_rews}")
#         print(f"Average Loss: {avg_actor_loss}")
#         print(f"Timesteps So Far: {t_so_far}")
#         print(f"Iteration took: {delta_t} secs")
#         print("------------------------------------------------------")
#         print()

#         wandb.log(
#             {
#                 "iteration": i_so_far,
#                 "timesteps_so_far": t_so_far,
#                 "avg_episode_length": avg_ep_lens,
#                 "avg_episode_return": avg_ep_rews,
#                 "avg_actor_loss": avg_actor_loss,
#                 "iteration_duration_sec": delta_t,
#             }
#         )

#         # Reset batch-specific logs
#         self.logger["batch_lens"] = []
#         self.logger["batch_rews"] = []
#         self.logger["actor_losses"] = []
