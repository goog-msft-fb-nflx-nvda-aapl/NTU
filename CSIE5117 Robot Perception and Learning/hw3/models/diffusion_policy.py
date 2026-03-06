import torch
import torch.nn.functional as F
from .network import FeedForwardNN


class DiffusionPolicy(FeedForwardNN):
    """
    A simple diffusion-style policy built on FeedForwardNN.

    - Forward pass: f(state, x_t, t) -> velocity.
    - Inference: perform Euler integration from noise to action.
    - Training: use conditional flow matching loss (CFM).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device = None,
        num_steps: int = 10,
        fixed_noise_inference: bool = False,
    ):
        super().__init__(in_dim, out_dim)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_steps = num_steps
        self.fixed_noise_inference = fixed_noise_inference

        # Pre-sampled noise used when fixed_noise_inference=True
        self.init_noise = torch.randn(1, out_dim, device=self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def sample_action(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Deterministic Euler integration sampling.

        Args:
            state_norm: Tensor of shape (D_s,) or (1, D_s)

        Returns:
            Tensor (D_a,) — final action
        """
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0)
        state_norm = state_norm.to(self.device)

        dt = 1.0 / self.num_steps

        x_t = (
            self.init_noise.clone()
            if self.fixed_noise_inference
            else torch.randn(1, self.out_dim, device=self.device)
        )

        for step in range(self.num_steps):
            t_val = step * dt
            t_tensor = torch.full((1, 1), t_val, device=self.device)
            inp = torch.cat([state_norm, x_t, t_tensor], dim=1)

            with torch.no_grad():
                velocity = self(inp)

            x_t = x_t + dt * velocity

        return x_t[0]

    # ------------------------------------------------------------------
    # Inference + additional training info
    # ------------------------------------------------------------------
    def sample_action_with_info(
        self,
        state_norm: torch.Tensor,
        num_train_samples: int = 100,
        include_inference_eps: bool = False,  # unused but preserved for compatibility
    ):
        """
        Same as sample_action(), but also returns:
        - full x_t trajectory
        - CFM loss samples

        Returns:
            pred_action:       (D_a,)
            x_t_path:          [1, T+1, D_a]
            eps_sample:        [N, D_a]
            t_sample:          [N, 1]
            cfm_loss_initial:  [N]
        """
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0)
        state_norm = state_norm.to(self.device)

        dt = 1.0 / self.num_steps

        eps = (
            self.init_noise.clone()
            if self.fixed_noise_inference
            else torch.randn(1, self.out_dim, device=self.device)
        )

        x_t = eps.clone()
        x_t_path = [x_t.detach().clone()]

        # Euler integration trajectory
        for step in range(self.num_steps):
            t_val = step * dt
            t_tensor = torch.full((1, 1), t_val, device=self.device)
            inp = torch.cat([state_norm, x_t, t_tensor], dim=1)
            velocity = self(inp)
            x_t = x_t + dt * velocity
            x_t_path.append(x_t.detach().clone())

        x_t_path = torch.stack(x_t_path, dim=1)  # [1, T+1, D_a]

        # Samples used for CFM loss
        eps_sample = torch.randn(num_train_samples, self.out_dim, device=self.device)
        t_sample = torch.rand(num_train_samples, 1, device=self.device)
        x1 = x_t.repeat(num_train_samples, 1).detach()
        state_tile = state_norm.expand(num_train_samples, -1)

        cfm_loss_initial = self.compute_cfm_loss(
            state_tile, x1, eps_sample, t_sample
        ).detach()

        return x_t[0], x_t_path, eps_sample, t_sample, cfm_loss_initial

    # ------------------------------------------------------------------
    # Conditional Flow Matching loss
    # ------------------------------------------------------------------
    def compute_cfm_loss(
        self,
        state_norm: torch.Tensor,
        x1: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Conditional flow matching loss:
        Predict velocity = x1 - eps and compare against model velocity.

        Args:
            state_norm: [B, D_s]
            x1:         [B, D_a]
            eps:        [B, D_a]
            t:          [B, 1]

        Returns:
            loss: [B]
        """
        B, D_a = eps.shape

        assert x1.shape == (B, D_a)
        assert state_norm.shape[0] == B
        assert t.shape == (B, 1)

        # Linear interpolation between noise and final point
        x_t = (1 - t) * eps + t * x1

        inp = torch.cat([state_norm, x_t, t], dim=1)
        velocity_pred = self(inp)

        return F.mse_loss(
            velocity_pred,
            x1 - eps,
            reduction="none",
        ).mean(dim=1)
