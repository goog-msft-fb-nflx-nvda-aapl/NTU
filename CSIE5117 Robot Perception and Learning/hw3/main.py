"""
Entry point for training and evaluating PPO / TRPO / FPO on GridWorld.
"""

import sys
from datetime import datetime

import torch
import wandb

from utils.arguments import get_args
from models.ppo import PPO
from models.fpo import FPO
from models.trpo import TRPO
from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.eval_policy import eval_policy
from utils.gridworld import GridWorldEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def train(env, hyperparams, actor_path, critic_path, algo):
    """
    Train a policy on the given environment.
    """
    print(f"Training using {algo.upper()}", flush=True)

    if algo == "ppo":
        model = PPO(policy_class=FeedForwardNN, env=env, **hyperparams)
    elif algo == "trpo":
        model = TRPO(policy_class=FeedForwardNN, env=env, **hyperparams)
    elif algo == "fpo":
        model = FPO(actor_class=DiffusionPolicy, critic_class=FeedForwardNN, env=env, **hyperparams)
    else:
        print(f"Unsupported method: {algo}")
        sys.exit(1)

    model.actor.to(device)
    model.critic.to(device)

    # Optionally resume from checkpoint
    if actor_path and critic_path:
        print(f"Loading {actor_path} and {critic_path}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_path, map_location=device))
        model.critic.load_state_dict(torch.load(critic_path, map_location=device))
        print("Checkpoint loaded.", flush=True)
    elif actor_path or critic_path:
        print("Error: specify both actor and critic checkpoints, or none.")
        sys.exit(1)
    else:
        print("Training from scratch.", flush=True)

    model.learn(total_timesteps=200_000)


def test(env, actor_path, algo):
    """
    Evaluate a trained policy.
    """
    print(f"Testing {actor_path}", flush=True)

    if not actor_path:
        print("No model file specified. Exiting.", flush=True)
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if algo in ("ppo", "trpo"):
        policy = FeedForwardNN(obs_dim, act_dim).to(device)
    elif algo == "fpo":
        policy = DiffusionPolicy(obs_dim + act_dim + 1, act_dim, device=device).to(device)
    else:
        print(f"Unsupported method: {algo}")
        sys.exit(1)

    policy.load_state_dict(torch.load(actor_path, map_location=device))

    if algo == "fpo":
        eval_policy(policy=policy.sample_action, env=env, render=True)
    else:
        eval_policy(policy=policy, env=env, render=True)


def main(args):
    """
    Parse arguments, set up env and hyperparameters, then train or test.
    """
    hyperparams = {
        "timesteps_per_batch": 2048,
        "max_timesteps_per_episode": 200,
        "gamma": 0.99,
        "n_updates_per_iteration": 10,
        "lr": 3e-4,
        "clip": 0.2,
        "render": True,
        "render_every_i": 1000,
        # FPO-specific:
        "grid_mode": "two_walls",
        "num_fpo_samples": 50,
        "positive_advantage": False,
    }

    print("Hyperparameters:", hyperparams)

    env = GridWorldEnv(mode=hyperparams["grid_mode"])

    if args.mode == "train":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr = hyperparams["lr"]
        batch_size = hyperparams["timesteps_per_batch"]

        run_name = f"{args.method}_lr{lr}_bs{batch_size}_{timestamp}"

        if args.method == "fpo":
            n_samples = hyperparams["num_fpo_samples"]
            run_name += f"_N{n_samples}"

        print(f"Run name: {run_name}")
        hyperparams["run_name"] = run_name

        wandb.init(
            project="fpo-diffusion-grid",
            name=run_name,
            config=hyperparams,
            tags=[args.method, "gridworld", args.mode],
        )

        train(
            env=env,
            hyperparams=hyperparams,
            actor_path=args.actor_model,
            critic_path=args.critic_model,
            algo=args.method,
        )
    else:
        test(env=env, actor_path=args.actor_model, algo=args.method)


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args)
