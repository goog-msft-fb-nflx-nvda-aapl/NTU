import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=96, out_dim=2, device='cuda'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.device = device
    def forward(self, x, t):
        # for simplicity, we ignore the time encoding
        return self.net(torch.cat([x, t], dim=-1))

    def sample_target(self, x0, n_steps):
      x = x0.clone().to(self.device)
      dt = 1.0 / n_steps

      for i in range(n_steps):
          t = torch.full((x.shape[0], 1), i / n_steps, device=x.device)
          u = self(x, t)
          x = self.step(x, u, dt)

      return x

    def step(self, x, u, dt, reverse=False):
      if reverse:
        return x - u * dt
      return x + u * dt


class Meanflow(nn.Module):
    def __init__(self, in_dim=4, hidden=96, out_dim=2, device='cuda'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.device = device
    def forward(self, xt, r, t):
        # for simplicity, we ignore the time encoding
        return self.net(torch.cat([xt, r, t], dim=-1))

    def sample_target(self, x0):
        x = x0.clone().to(self.device)
        x = x + self(x0, 0, 1)
        return x