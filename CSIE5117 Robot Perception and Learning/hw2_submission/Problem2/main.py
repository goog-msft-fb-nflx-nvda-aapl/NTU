import torch
from data import OneMoon, Gaussian
from model import MLP, Meanflow
from pipeline import FMTrainer, FMPipeline
from visualize import plot_traj, plot_loss
from optimal_transport import OTPlanSampler
import argparse
import numpy as np
import os

if not os.path.exists('outputs'):
    os.makedirs('outputs')
gaussian = Gaussian(mean=0.0, sigma=1.0)
X0 = gaussian.sample_source(3000)
onemoon = OneMoon(noise=0.06, rotate=torch.pi/2)
X1 = onemoon.sample_target(3000)


def get_output_traj_and_plot(args, model, n_step, X0, X1):
    flow = FMPipeline(model=model, device='cuda', n_step=n_step)
    if args.method == "meanflow":
        output = flow.meanflow_traj(X0)
    else:
        output = flow.traj(X0)
    plot_traj(X0.numpy(), output["sample"], X1.numpy(), n_vis=300, traj=output["traj"], title=f"{args.method}_traj_{n_step}")
    return output


def main(args):
    if args.method == "flow" or args.method == "reflow":
        print(f"Start {args.method} experiment")

    # Stage 1: Flow Matching
        model = MLP().to("cuda")
        trainer = FMTrainer(model=model, device="cuda")

        dataset = torch.utils.data.TensorDataset(X0, X1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        trainer.fit(dataloader, n_epoch=100)
        plot_loss(trainer, title="flow_matching_loss")

        output = get_output_traj_and_plot(args, model, 1, X0, X1)
        output = get_output_traj_and_plot(args, model, 1000, X0, X1)


    elif args.method == "optimal_coupling":
        # Stage 2: Optimal Coupling
        print("Start optimal coupling experiment")
        model_ot = MLP().to("cuda")
        trainer = FMTrainer(model=model_ot, device="cuda")

        otplansampler = OTPlanSampler(method="exact")
        # TODO: Change X0, X1 to X0_ot, X1_ot using optimal_transport.py's sample_plan()
        # X0_ot, X1_ot = ...
        X0_ot, X1_ot = otplansampler.sample_plan(X0, X1)

        dataset = torch.utils.data.TensorDataset(X0_ot, X1_ot)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        trainer.fit(dataloader, n_epoch=100)
        plot_loss(trainer, title="ot_flow_matching_loss")

        output_ot = get_output_traj_and_plot(args, model_ot, 1, X0, X1)
        output_ot = get_output_traj_and_plot(args, model_ot, 1000, X0, X1)

    elif args.method == "meanflow":
        # Stage 4: Meanflow
        print("Start meanflow experiment")
        model_meanflow = Meanflow().to("cuda")
        trainer = FMTrainer(model=model_meanflow, device='cuda')
        dataset = torch.utils.data.TensorDataset(X0, X1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        trainer.fit_meanflow(dataloader, n_epoch=100)
        plot_loss(trainer, title="meanflow_loss")

        output_meanflow = get_output_traj_and_plot(args, model_meanflow, 1, X0, X1)

    if args.method == "reflow":
        # Stage 3: Reflow
        # TODO: Change X1 to X1_reflow using the output of pretrained flow model
        # X1_reflow = ...
        flow_sampler = FMPipeline(model=model, device='cuda', n_step=1000)
        X1_reflow = flow_sampler.traj(X0)["sample"].to(X0.device)

        dataset = torch.utils.data.TensorDataset(X0, X1_reflow)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        trainer = FMTrainer(model=model, device='cuda')
        trainer.fit(dataloader, n_epoch=100)
        plot_loss(trainer, title="reflow_loss")

        output_reflow = get_output_traj_and_plot(args, model, 1, X0, X1)
        output_reflow = get_output_traj_and_plot(args, model, 1000, X0, X1)


def get_args_parser():
    parser = argparse.ArgumentParser('Flow Matching Experiments', add_help=False)
    parser.add_argument('--method', default='flow', type=str, help='method to run: flow, optimal_coupling, reflow, meanflow')
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    main(args)
