import torch


class OneMoon:
    def __init__(self, noise, rotate):
        self.noise = noise
        self.rotate = rotate

    @staticmethod
    def R(theta: float) -> torch.Tensor:
        c = torch.cos(torch.tensor(theta))
        s = torch.sin(torch.tensor(theta))
        return torch.tensor(
            [
                [c, -s],
                [s,  c],
            ],
            dtype=torch.float32,
        ) * 3

    @staticmethod
    def sample(noise, R, n):
        t = torch.rand(n) * torch.pi
        x2 = torch.stack([1 - torch.cos(t), 1 - torch.sin(t) - 0.5],1)
        X = x2 + noise * torch.randn(n, 2)
        X = X @ R.T
        return X + torch.tensor([8.0, -2.0])

    def sample_target(self, n_sample):
        n = 300
        samples = self.sample(self.noise, self.R(self.rotate), n_sample)
        idx = torch.randperm(n_sample)[:n]
        samples = samples[idx]
        idx = torch.randint(0, n, (n_sample,), device=samples.device)
        samples = samples[idx]
        return samples


class Gaussian:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def sample(mean, sigma, n):
      return mean + sigma * torch.randn(n, 2)

    def sample_source(self, n_sample):
        return self.sample(self.mean, self.sigma, n_sample)
