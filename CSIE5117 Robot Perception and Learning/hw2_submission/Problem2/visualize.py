from matplotlib import pyplot as plt
import numpy as np


def plot_traj(x0, pred_x1, X1, n_vis, traj, title):
    plt.figure(figsize=(6,6))

    subset = np.linspace(0, len(traj)-1, 1000, dtype=int)
    for idx in range(x0.shape[0]):
        xs = [traj[t][idx,0] for t in subset]
        ys = [traj[t][idx,1] for t in subset]
        plt.plot(xs, ys, linewidth=0.3, alpha=0.1, color="gray")

    plt.scatter(x0[:,0], x0[:,1], s=6, alpha=0.8, label="start: x0")
    plt.scatter(pred_x1[:,0], pred_x1[:,1], s=6, alpha=0.8, label="end: pred_x1")
    plt.scatter(X1[:n_vis,0], X1[:n_vis,1], s=6, alpha=0.8, label="Target")

    plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"outputs/{title}.png", dpi=300)


def plot_loss(trainer, title):
    plt.figure(figsize=(6,4))
    plt.plot(trainer.state["loss"])
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title(f"{title}")
    plt.tight_layout()
    fig_path = f"outputs/{title}.png"
    plt.savefig(fig_path, dpi=160)