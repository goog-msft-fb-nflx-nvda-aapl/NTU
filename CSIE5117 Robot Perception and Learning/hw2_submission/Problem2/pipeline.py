from torch.func import jvp
import torch
import torch.nn.functional as F
import torch.optim as optim


class FMTrainer:
    def __init__(self, model, device):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
        self.device = device
        self.state = {"loss": []}

    def fit(self, dataloader,  n_epoch):
        self.model.train()
        for _ in range(n_epoch):
            for x0, x1 in dataloader:
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                t = torch.rand(x0.shape[0], 1, device=self.device)
                # TODO: interpolate x0 and x1 to xt. Noted that when t=0, xt=x0; when t=1, xt=x1
                # xt = ...
                xt = (1 - t) * x0 + t * x1

                # TODO: compute the target velocity u
                # u = ...
                u = x1 - x0

                # TODO: predict the velocity at (xt, t) using self.model
                # pred = ...
                pred = self.model(xt, t)

                # TODO: compute the MSE loss between pred and u
                # loss = ...
                loss = F.mse_loss(pred, u)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.state["loss"].append(loss.item())

    def fit_meanflow(self, dataloader, n_epoch):
        self.model.train()
        for _ in range(n_epoch):
            for x0, x1 in dataloader:
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                t = torch.rand(x0.shape[0], 1, device=self.device)
                r = torch.rand(x0.shape[0], 1, device=self.device)
                mask = torch.gt(r, t)
                r[mask],t[mask] = t[mask],r[mask]
                assert torch.all(torch.le(r, t))

                # TODO: interpolate x0 and x1 to xt. Noted that when t=0, xt=x1; when t=1, xt=x0, which is different to flow matching to align with the meanflow paper
                # xt = ...
                xt = (1 - t) * x1 + t * x0

                # TODO: compute the target instantaneous velocity v
                # v = ...
                v = x0 - x1

                # TODO: predict the average velocity at (xt, r, t) and dudt using self.model and jvp
                # u_pred, dudt = ...
                fn = lambda xt_, r_, t_: self.model(xt_, r_, t_)
                u_pred, dudt = jvp(
                    fn,
                    (xt, r, t),
                    (v, torch.zeros_like(r), torch.ones_like(t))
                )

                # TODO: compute the target average velocity u_tgt
                # u_tgt = ...
                u_tgt = v - (t - r) * dudt

                # TODO: compute the MSE loss between u_pred and u_tgt. Noted that u_tgt should be detached from the computation graph as stopgrad
                # error = ...
                # loss = (error ** 2).mean()
                error = u_pred - u_tgt.detach()
                loss = (error ** 2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.state["loss"].append(loss.item())


class FMPipeline:
    def __init__(self, model, device, n_step):
        self.model = model
        self.device = device
        self.n_step = n_step

    def step(self, x, u, dt):
        return x + u * dt

    @torch.no_grad()
    def traj(self, x0):
        self.model.eval()
        x = x0.clone().to(self.device)
        dt = 1.0 / self.n_step
        traj = [x.cpu().numpy()]

        for i in range(self.n_step):
            t = torch.full((x.shape[0], 1), i / self.n_step, device=x.device)
            u = self.model(x, t)
            x = self.step(x, u, dt)
            traj.append(x.cpu().numpy())

        return {"sample": x.cpu(), "traj": traj}

    @torch.no_grad()
    def meanflow_traj(self, x0):
        self.model.eval()
        x = x0.clone().to(self.device)
        dt = 1.0 / self.n_step
        traj = [x.cpu().numpy()]

        for i in range(self.n_step, 0, -1):
            t = torch.full((x.shape[0], 1), i / self.n_step, device=x.device)
            r = torch.full((x.shape[0], 1), (i-1) / self.n_step, device=x.device)
            u = self.model(x, r, t)
            x = x - u*dt
            traj.append(x.cpu().numpy())

        return {"sample": x.cpu(), "traj": traj}