from dataclasses import dataclass
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import wandb

from .ppo import PPO  # shared utilities / hyperparameters / logging


@dataclass
class FpoActionInfo:
    """
    Container for additional information returned by the FPO actor.
    """
    x_t_path: torch.Tensor         # (*, flow_steps, action_dim)
    loss_eps: torch.Tensor         # (*, sample_dim, action_dim)
    loss_t: torch.Tensor           # (*, sample_dim, 1)
    initial_cfm_loss: torch.Tensor # (*,)


class FPO(PPO):
    """
    Flow Matching Policy Optimization (FPO).

    - Actor: diffusion / flow-matching policy.
    - Critic: standard value network.
    - Uses PPO-style clipped objective, where the policy ratio is derived
      from CFM loss differences instead of log-prob ratios.
    """

    def __init__(self, actor_class, critic_class, env, **hyperparameters):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize hyperparameters via PPO base
        self._init_hyperparameters(hyperparameters)

        # Actor input dimension: observation + action + time (or extra feature)
        self.obs_dim_actor = (
            env.observation_space.shape[0]
            + env.action_space.shape[0]
            + 1
        )
        self.obs_dim_critic = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ----- Actor (diffusion policy) -----
        actor_kwargs = getattr(self, "actor_kwargs", {})
        self.actor = actor_class(
            self.obs_dim_actor,
            self.act_dim,
            **actor_kwargs,
        ).to(self.device)

        # FPO-specific hyperparameters
        self.num_train_samples = hyperparameters["num_fpo_samples"]
        self.positive_advantage = hyperparameters.get("positive_advantage", False)

        print(f"[FPO] training with {self.num_train_samples} samples per state")
        print(f"[FPO] positive_advantage = {self.positive_advantage}")

        # ----- Critic -----
        self.critic = critic_class(self.obs_dim_critic, 1).to(self.device)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Kept for compatibility with PPO/TRPO interfaces, not used by FPO actor
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
    #                          ACTION SAMPLING                               #
    # ====================================================================== #

    def get_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, torch.Tensor, FpoActionInfo]:
        """
        Sample an action from the FPO policy.

        Returns:
            action: np.ndarray, environment action.
            log_prob: dummy tensor for API compatibility (not used).
            action_info: FpoActionInfo with CFM-related tensors.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, x_t_path, eps, t, initial_cfm_loss = \
                self.actor.sample_action_with_info(
                    obs_tensor,
                    self.num_train_samples,
                )

        action_info = FpoActionInfo(
            x_t_path=x_t_path,
            loss_eps=eps,
            loss_t=t,
            initial_cfm_loss=initial_cfm_loss,
        )

        # Dummy log_prob to stay compatible with PPO/TRPO signatures
        log_prob = torch.tensor(0.0)

        return action.squeeze().cpu().numpy(), log_prob, action_info

    # ====================================================================== #
    #                               ROLLOUT                                  #
    # ====================================================================== #

    def rollout(self):
        """
        Collect one on-policy batch.

        Returns:
            batch_obs          : [T, obs_dim]
            batch_acts         : [T, act_dim]
            batch_log_probs    : [T] (dummy, unused)
            batch_rews         : list of per-episode reward lists
            batch_lens         : list of episode lengths
            batch_vals         : list of per-episode value lists
            batch_dones        : list of per-episode done flags
            batch_action_info  : list of FpoActionInfo, length T
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews: List[List[float]] = []
        batch_lens: List[int] = []
        batch_vals: List[List[float]] = []
        batch_dones: List[List[bool]] = []
        batch_action_info: List[FpoActionInfo] = []

        t = 0

        while t < self.timesteps_per_batch:
            ep_rews: List[float] = []
            ep_vals: List[float] = []
            ep_dones: List[bool] = []

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

                action, log_prob, action_info = self.get_action(obs)

                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                )
                val = self.critic(obs_tensor).detach()

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_rews.append(rew)
                ep_vals.append(val.item())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_action_info.append(action_info)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            self.logger["batch_rews"] = batch_rews
            self.logger["batch_lens"] = batch_lens

        return (
            torch.tensor(np.array(batch_obs), dtype=torch.float32),
            torch.tensor(np.array(batch_acts), dtype=torch.float32),
            torch.tensor(np.array(batch_log_probs), dtype=torch.float32),
            batch_rews,
            batch_lens,
            batch_vals,
            batch_dones,
            batch_action_info,
        )

    # ====================================================================== #
    #                              TRAINING LOOP                             #
    # ====================================================================== #

    def learn(self, total_timesteps: int):
        """
        Main FPO training loop.

        - Collect rollouts.
        - Compute GAE and returns.
        - Use CFM loss differences to construct rho_s.
        - Plug rho_s into a PPO-style clipped surrogate objective.
        """
        t_so_far, i_so_far = 0, 0

        while t_so_far < total_timesteps:
            (
                batch_obs,
                batch_acts,
                batch_log_probs,   # unused (dummy)
                batch_rews,
                batch_lens,
                batch_vals,
                batch_dones,
                batch_action_info,
            ) = self.rollout()

            batch_obs = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # ----- GAE & returns ----- #
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones).to(self.device)

            with torch.no_grad():
                V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V

            if self.positive_advantage:
                A_k = F.softplus(A_k)
            else:
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # ----- Minibatch training ----- #
            num_steps = batch_obs.size(0)
            indices = np.arange(num_steps)
            minibatch_size = num_steps // self.num_minibatches

            actor_loss_history: List[torch.Tensor] = []

            for _ in range(self.n_updates_per_iteration):
                # Linear learning-rate annealing
                frac = (t_so_far - 1.0) / float(total_timesteps)
                new_lr = max(self.lr * (1.0 - frac), 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                self.logger["lr"] = new_lr

                np.random.shuffle(indices)

                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    idx = indices[start:end]
                    if len(idx) == 0:
                        continue

                    mini_obs = batch_obs[idx]      # [B, obs_dim]
                    mini_acts = batch_acts[idx]    # [B, act_dim]
                    mini_adv = A_k[idx]            # [B]
                    mini_rtgs = batch_rtgs[idx]    # [B]
                    mini_infos = [batch_action_info[i] for i in idx]

                    # ----- CFM-related tensors (already collected in rollout) -----
                    loss_eps = torch.stack(
                        [info.loss_eps for info in mini_infos]
                    ).to(self.device)                  # [B, N, act_dim]
                    loss_t = torch.stack(
                        [info.loss_t for info in mini_infos]
                    ).to(self.device)                   # [B, N, 1]
                    initial_cfm_loss = torch.stack(
                        [info.initial_cfm_loss for info in mini_infos]
                    ).to(self.device)                   # [B]

                    # Critic prediction
                    V_pred = self.critic(mini_obs).squeeze(-1)

                    # Placeholder entropy (can be replaced if actor exposes it)
                    entropy = torch.tensor(0.0, device=self.device)

                    # ========================================================== #
                    # TODO-F1: reshape & compute CFM loss / CFM difference
                    # ========================================================== #
                    #
                    # We have:
                    #   - mini_obs:          [B, obs_dim_critic]
                    #   - mini_acts:         [B, act_dim]
                    #   - loss_eps:          [B, N, act_dim]
                    #   - loss_t:            [B, N, 1]
                    #   - initial_cfm_loss:  [B]
                    #
                    # The FPO actor's `compute_cfm_loss` expects *flattened*
                    # inputs of shape [B*N, ...]. For each state, we have N
                    # samples (eps, t), and we want to evaluate the CFM loss
                    # of the *current* actor at the same (obs, act, eps, t).
                    #
                    # Steps:
                    #   1. Let B, N, D = loss_eps.shape.
                    #   2. Repeat observations and actions across N:
                    #        flat_obs:  [B*N, obs_dim_actor]
                    #        flat_acts: [B*N, act_dim]
                    #   3. Flatten eps, t, and initial_cfm_loss:
                    #        flat_eps:       [B*N, act_dim]
                    #        flat_t:         [B*N, 1]
                    #        flat_init_loss: [B*N]
                    #   4. Compute the new CFM loss:
                    #        cfm_loss = self.actor.compute_cfm_loss(
                    #            flat_obs, flat_acts, flat_eps, flat_t
                    #        )                 # [B*N]
                    #   5. Compute the per-sample CFM loss difference:
                    #        diff = flat_init_loss - cfm_loss
                    #      and reshape back to [B, N]:
                    #        cfm_difference = diff.view(B, N)
                    #
                    # This `cfm_difference` will later be turned into a
                    # state-wise ratio ρ_s.
                    # ========================================================== #

                    # ===== Your code (F1) starts here =====
                    B, N, D = loss_eps.shape

                    # Repeat obs and acts across N samples per state
                    # mini_obs is [B, obs_dim_critic]; actor needs obs_dim_actor
                    # but compute_cfm_loss takes state_norm of obs_dim_critic shape
                    flat_obs = mini_obs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)  # [B*N, obs_dim_critic]
                    flat_acts = mini_acts.unsqueeze(1).expand(B, N, -1).reshape(B * N, D)  # [B*N, act_dim]

                    flat_eps = loss_eps.reshape(B * N, D)           # [B*N, act_dim]
                    flat_t = loss_t.reshape(B * N, 1)               # [B*N, 1]
                    flat_init_loss = initial_cfm_loss.reshape(B * N)  # [B*N]

                    # New CFM loss under current actor parameters
                    cfm_loss = self.actor.compute_cfm_loss(
                        flat_obs, flat_acts, flat_eps, flat_t
                    )  # [B*N]

                    # CFM loss difference: old_loss - new_loss → higher means policy improved
                    diff = flat_init_loss - cfm_loss                # [B*N]
                    cfm_difference = diff.view(B, N)                # [B, N]
                    # ===== Your code (F1) ends here =====

                    # ========================================================== #
                    # TODO-F2: build state-wise policy ratio rho_s from CFM diff
                    # ========================================================== #
                    #
                    # From the paper, we use the CFM loss difference as a
                    # surrogate "log-ratio" between new and old policies.
                    #
                    # For numerical stability:
                    #   1. Clamp cfm_difference to [-3, 3].
                    #   2. Average over the N samples (dim=1) to get one
                    #      scalar per state:   delta_s = mean(cfm_difference, dim=1).
                    #   3. Clamp delta_s again to [-3, 3].
                    #   4. Exponentiate to get the policy ratio ρ_s:
                    #
                    #        rho_s = exp(delta_s)
                    #
                    # where rho_s has shape [B], just like PPO's r_t(θ).
                    # ========================================================== #

                    # ===== Your code (F2) starts here =====
                    # Clamp per-sample differences for numerical stability
                    cfm_difference = torch.clamp(cfm_difference, -3.0, 3.0)

                    # Average over N samples to get one scalar per state
                    delta_s = cfm_difference.mean(dim=1)            # [B]
                    delta_s = torch.clamp(delta_s, -3.0, 3.0)

                    # Exponentiate to get policy ratio ρ_s
                    rho_s = torch.exp(delta_s)                      # [B]
                    # ===== Your code (F2) ends here =====

                    # ========================================================== #
                    # TODO-F3: PPO-style clipped surrogate objective with rho_s
                    # ========================================================== #
                    #
                    # Now treat rho_s as a policy ratio, analogous to PPO:
                    #
                    #   surr1 = rho_s * mini_adv
                    #   surr2 = clamp(rho_s, 1 - clip, 1 + clip) * mini_adv
                    #
                    # Then define:
                    #
                    #   policy_loss = -mean( min(surr1, surr2) )
                    #   actor_loss  = policy_loss - ent_coef * entropy.mean()
                    #
                    # Critic loss remains standard MSE:
                    #
                    #   critic_loss = MSE(V_pred, mini_rtgs)
                    #
                    # ========================================================== #

                    # ===== Your code (F3) starts here =====
                    surr1 = rho_s * mini_adv
                    surr2 = torch.clamp(rho_s, 1 - self.clip, 1 + self.clip) * mini_adv

                    policy_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = policy_loss - self.ent_coef * entropy.mean()

                    critic_loss = nn.MSELoss()(V_pred, mini_rtgs.detach())
                    # ===== Your code (F3) ends here =====

                    # ----- Actor step ----- #
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # ----- Critic step ----- #
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    actor_loss_history.append(actor_loss.detach())

                    # ----- Logging (per minibatch) ----- #
                    # NOTE: you may want to move these logs *after* filling in
                    # rho_s / cfm_difference in your TODOs above.
                    metrics = {
                        "clipped_ratio_mean": (
                            (torch.abs(rho_s - 1.0) > self.clip)
                            .float()
                            .mean()
                            .item()
                        ),
                        "cfm_difference": cfm_difference.mean().item(),
                        "policy_ratio_mean": rho_s.mean().item(),
                        "policy_ratio_min": rho_s.min().item(),
                        "policy_ratio_max": rho_s.max().item(),
                        "policy_loss": policy_loss.item(),
                        "adv": mini_adv.mean().item(),
                        "surrogate_loss1_mean": surr1.mean().item(),
                        "surrogate_loss2_mean": surr2.mean().item(),
                        "action_min": mini_acts.min().item(),
                        "action_max": mini_acts.max().item(),
                    }
                    wandb.log(metrics)

            avg_loss = sum(actor_loss_history) / max(len(actor_loss_history), 1)
            self.logger["actor_losses"].append(avg_loss)

            if self.logger["i_so_far"] % 10 == 0:
                wandb.log({"advantage_hist": wandb.Histogram(A_k.cpu().numpy())})

            self._log_summary()

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./checkpoints/fpo_actor.pth")
                torch.save(self.critic.state_dict(), "./checkpoints/fpo_critic.pth")
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic_iter{i_so_far}.pth")