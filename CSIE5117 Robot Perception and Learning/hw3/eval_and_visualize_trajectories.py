#!/usr/bin/env python3
"""
Evaluate a policy from fixed start points and visualize the resulting trajectories.

This script supports three modes:
1. Full pipeline (default): evaluate policy → save trajectories (.pkl) → save figure (.png).
2. Evaluation only (--no-visualize): only save trajectories (.pkl).
3. Visualization only (--visualize-only --input <file.pkl>): load pkl and save figure (.png).
"""

import sys
import argparse
import time
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.gridworld import GridWorldEnv
from utils.eval_policy import _log_summary


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_trajectories(args):
    """Run the policy from a set of fixed start points and save all trajectories."""
    print(f"Evaluating model: {args.actor_model}", flush=True)

    if not args.actor_model:
        print("Error: Missing model file.")
        sys.exit(1)

    # Environment & device
    env = GridWorldEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load policy
    if args.method == "ppo":
        policy = FeedForwardNN(obs_dim, act_dim).to(device)
    elif args.method == "fpo":
        policy = DiffusionPolicy(
            in_dim=obs_dim + act_dim + 1,
            out_dim=act_dim,
            device=device
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    policy.load_state_dict(torch.load(args.actor_model, map_location=device))
    policy.eval()

    # Fixed start points used for visualization
    start_points = [
        (17, 12),
        (11, 13),
        (6, 9),
        (20, 17),
    ]

    # Determine output file name
    model_base = os.path.splitext(os.path.basename(args.actor_model))[0]
    out_path = args.output or f"outputs/{model_base}_fixed_traj.pkl"

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    episode_count = 0

    for sid, start_xy in enumerate(start_points):
        for _ in range(args.episodes):
            # Reset & force starting location
            _, _ = env.reset()
            env.pos = np.array(start_xy, dtype=int)
            obs = env._get_obs()

            traj = [tuple(env.pos)]
            ep_return = 0.0
            done = False
            step = 0

            while not done:
                step += 1

                # Optional rendering
                if args.render:
                    env.render()
                    time.sleep(args.sleep)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

                # Deterministic action
                if args.method == "ppo":
                    action = policy(obs_tensor).detach().cpu().numpy()
                else:
                    action = policy.sample_action(obs_tensor).cpu().numpy()

                # Optional Gaussian noise
                if args.noise > 0.0:
                    action += np.random.normal(scale=args.noise, size=action.shape)
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                # Step environment
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += rew

                traj.append(tuple(env.pos))

            # Logging
            _log_summary(ep_len=step, ep_ret=ep_return, ep_num=episode_count)
            episode_count += 1

            results.append({
                "start_idx": sid,
                "start": start_xy,
                "traj": traj,
                "ep_len": step,
                "ep_ret": ep_return
            })

    # Save all trajectory data
    with open(out_path, "wb") as f:
        pickle.dump({
            "start_points": start_points,
            "episodes": args.episodes,
            "noise": args.noise,
            "trajectories": results
        }, f)

    print(f"Saved {len(results)} trajectories → {out_path}", flush=True)
    return out_path


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_trajectories(trajectory_file, args):
    """Plot time-colored trajectory curves over the GridWorld map and save as image."""
    with open(trajectory_file, "rb") as f:
        data = pickle.load(f)

    start_points = data["start_points"]
    episodes = data["episodes"]
    trajs = data["trajectories"]

    env = GridWorldEnv("two_walls")
    size = env.grid_size

    fig, ax = plt.subplots(figsize=(6, 6))

    # Base terrain (light gray)
    img = np.ones((size, size, 3), dtype=np.uint8) * 240

    # Mark goal cells
    goal_cells = np.array(list(env.config.goal_cells), dtype=float)
    for gx, gy in goal_cells:
        img[int(gy), int(gx)] = [42, 157, 143]

    ax.imshow(img, extent=(0, size, 0, size))

    # Mark death cells
    death_cells = np.array(list(env.config.death_cells), dtype=float)
    if death_cells.size > 0:
        ax.scatter(
            death_cells[:, 0] + 0.5,
            death_cells[:, 1] + 0.5,
            marker="s",
            s=200,
            color=np.array([229, 57, 70]) / 255.0,
            label="Death"
        )

    # Mark goal cells again (overlay)
    if goal_cells.size > 0:
        ax.scatter(
            goal_cells[:, 0] + 0.5,
            goal_cells[:, 1] + 0.5,
            marker="s",
            s=200,
            color=np.array([42, 157, 143]) / 255.0,
            label="Goal"
        )

    # Grid lines
    majors = np.arange(0, size + 1, 5)
    minors = np.arange(0, size + 1, 1)
    ax.set_xticks(majors)
    ax.set_yticks(majors)
    ax.set_xticks(minors, minor=True)
    ax.set_yticks(minors, minor=True)
    ax.grid(which="major", color="#bbb", linestyle="--", linewidth=1)
    ax.grid(which="minor", color="#ddd", linestyle="-", linewidth=0.5)

    # Draw trajectories
    cmap = plt.get_cmap(args.cmap)
    for ep in trajs:
        pts = np.array(ep["traj"], dtype=float)
        xs, ys = pts[:, 0] + 0.5, pts[:, 1] + 0.5

        segs = np.stack([
            np.column_stack([xs[:-1], ys[:-1]]),
            np.column_stack([xs[1:], ys[1:]])
        ], axis=1)

        norm = Normalize(0, len(xs) - 1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=args.linewidth)
        lc.set_array(np.arange(len(xs)))

        ax.add_collection(lc)

    # Start & end markers
    for ep in trajs:
        pts = np.array(ep["traj"], dtype=float)

        ax.scatter(
            pts[0, 0] + 0.5, pts[0, 1] + 0.5,
            marker="o", s=150,
            facecolors="white", edgecolors="black",
            label="Start" if ep == trajs[0] else ""
        )
        ax.scatter(
            pts[-1, 0] + 0.5, pts[-1, 1] + 0.5,
            marker="X", s=150,
            facecolors="black", edgecolors="black",
            label="End" if ep == trajs[0] else ""
        )

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    # Deduplicated legend
    handles, labels = ax.get_legend_handles_labels()
    unique = {lbl: h for h, lbl in zip(handles, labels) if lbl}
    ax.legend(unique.values(), unique.keys(), loc="upper right", bbox_to_anchor=(1.3, 1.0))

    model_name = os.path.splitext(os.path.basename(trajectory_file))[0]
    ax.set_title(
        f"{model_name}\n{episodes} runs from {len(start_points)} start points (color shows time)",
        color="#333"
    )

    plt.tight_layout()

    # Decide output figure path
    if args.figure_output:
        fig_path = args.figure_output
    else:
        base = os.path.splitext(os.path.basename(trajectory_file))[0]
        fig_path = f"outputs/{base}.png"

    fig_dir = os.path.dirname(fig_path)
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved figure → {fig_path}", flush=True)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy from fixed start points and visualize trajectories."
    )

    # Mode selection
    parser.add_argument("--visualize-only", action="store_true",
                        help="Only visualize an existing trajectory file (requires --input).")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization after evaluation.")

    # Evaluation
    parser.add_argument("--actor_model", help="Path to actor .pth file.")
    parser.add_argument("--method", choices=["ppo", "fpo"],
                        help="Policy type.")
    parser.add_argument("--episodes", "-n", type=int, default=10,
                        help="Episodes per start point (default: 10).")
    parser.add_argument("--render", action="store_true",
                        help="Render environment each step.")
    parser.add_argument("--sleep", type=float, default=0.001,
                        help="Delay for rendering mode.")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Stddev of Gaussian noise added to actions.")
    parser.add_argument("--output",
                        help="Path to save trajectory pickle.")

    # Visualization
    parser.add_argument("--input",
                        help="Trajectory pickle file (for --visualize-only).")
    parser.add_argument("--figure-output",
                        help="Save figure instead of using default path.")
    parser.add_argument("--cmap", default="plasma",
                        help="Colormap for time progression.")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Transparency for background (currently unused).")
    parser.add_argument("--linewidth", type=float, default=3.0,
                        help="Line width for trajectories.")

    args = parser.parse_args()

    if args.visualize_only:
        if not args.input:
            print("Error: --visualize-only requires --input")
            sys.exit(1)
        visualize_trajectories(args.input, args)
        return

    # Evaluation mode
    if not args.actor_model or not args.method:
        print("Error: --actor_model and --method are required.")
        sys.exit(1)

    traj_file = evaluate_trajectories(args)

    if not args.no_visualize:
        visualize_trajectories(traj_file, args)


if __name__ == "__main__":
    main()
