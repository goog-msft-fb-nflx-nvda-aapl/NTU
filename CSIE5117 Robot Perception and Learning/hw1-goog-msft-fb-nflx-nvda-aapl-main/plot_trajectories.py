import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from src.env import CartPoleEnv
from src.params import ControlParams
from src.lqr import BaseLQR, ContinuousLQR, DiscreteLQR
from src.ilqr import ILQRController
from src.mpc import ModelPredictiveControllerWrapper


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="cont_lqr",
        choices=["cont_lqr", "disc_lqr", "ilqr", "all"],
        help="Controller configuration to run (or 'all' for all three)"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="results/plots",
        help="Output directory for plots"
    )
    return parser.parse_args()


class TrajectoryRecorder:
    """Records state trajectories during simulation."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.times = []
        self.positions = []
        self.angles = []
        self.velocities = []
        self.angular_velocities = []
        self.step_count = 0
    
    def record(self, obs, dt):
        """Record observation at current timestep."""
        self.times.append(self.step_count * dt)
        self.positions.append(obs[0])
        self.velocities.append(obs[1])
        self.angles.append(obs[2])
        self.angular_velocities.append(obs[3])
        self.step_count += 1
    
    def get_data(self):
        """Return recorded data as numpy arrays."""
        return {
            'time': np.array(self.times),
            'position': np.array(self.positions),
            'velocity': np.array(self.velocities),
            'angle': np.array(self.angles),
            'angular_velocity': np.array(self.angular_velocities)
        }


class Runner:
    def __init__(
        self,
        controller: BaseLQR,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode=None,
    ):
        self.controller = controller
        self.dt = dt
        self.env = CartPoleEnv(
            max_episode_steps=max_episode_steps,
            dt=dt,
            force_mag=force_mag,
            render_mode=render_mode,
        )
        self.recorder = TrajectoryRecorder()

    def run(self, max_steps=10000, seed=0, deadband=0.0, save_path=None):
        """
        Run the controller and record trajectory data.
        
        Parameters
        ----------
        max_steps : int
            Maximum number of steps to run
        seed : int
            Random seed for environment reset
        deadband : float
            Control deadband threshold
        save_path : str, optional
            Ignored in this version (kept for compatibility with config files)
        """
        self.recorder.reset()
        obs = self.env.reset(seed=seed)
        total_r = 0.0
        
        # Record initial state
        self.recorder.record(obs, self.dt)
        
        for _ in range(max_steps):
            action = self.controller.get_action(obs, deadband=deadband)
            out = self.env.step(action)

            if len(out) == 5:
                obs, r, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, r, done = out
            
            total_r += r
            self.recorder.record(obs, self.dt)
            
            if done:
                break

        print(f"[Runner]: return={total_r}, steps={self.recorder.step_count}")
        self.env.close()
        
        return self.recorder.get_data()


def plot_trajectories(data_dict, output_dir):
    """
    Create plots for time-angle and time-position for all controllers.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with keys as controller names and values as trajectory data
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for each controller
    colors = {
        'cont_lqr': '#2E86AB',  # Blue
        'disc_lqr': '#A23B72',  # Purple
        'ilqr': '#F18F01'       # Orange
    }
    
    labels = {
        'cont_lqr': 'Continuous LQR',
        'disc_lqr': 'Discrete LQR',
        'ilqr': 'iLQR'
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time vs Angle
    for controller_name, data in data_dict.items():
        ax1.plot(
            data['time'],
            np.degrees(data['angle']),  # Convert to degrees for better readability
            label=labels[controller_name],
            color=colors[controller_name],
            linewidth=2,
            alpha=0.8
        )
    
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Pole Angle vs Time', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Plot 2: Time vs Position
    for controller_name, data in data_dict.items():
        ax2.plot(
            data['time'],
            data['position'],
            label=labels[controller_name],
            color=colors[controller_name],
            linewidth=2,
            alpha=0.8
        )
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cart Position (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Cart Position vs Time', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_dir, 'combined_trajectories.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {combined_path}")
    
    # Also save individual plots for each controller
    for controller_name, data in data_dict.items():
        fig_ind, (ax1_ind, ax2_ind) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Angle plot
        ax1_ind.plot(
            data['time'],
            np.degrees(data['angle']),
            color=colors[controller_name],
            linewidth=2,
            alpha=0.8
        )
        ax1_ind.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax1_ind.set_ylabel('Angle (degrees)', fontsize=12, fontweight='bold')
        ax1_ind.set_title(f'{labels[controller_name]}: Pole Angle vs Time', 
                         fontsize=14, fontweight='bold', pad=20)
        ax1_ind.grid(True, alpha=0.3, linestyle='--')
        ax1_ind.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Position plot
        ax2_ind.plot(
            data['time'],
            data['position'],
            color=colors[controller_name],
            linewidth=2,
            alpha=0.8
        )
        ax2_ind.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2_ind.set_ylabel('Cart Position (m)', fontsize=12, fontweight='bold')
        ax2_ind.set_title(f'{labels[controller_name]}: Cart Position vs Time', 
                         fontsize=14, fontweight='bold', pad=20)
        ax2_ind.grid(True, alpha=0.3, linestyle='--')
        ax2_ind.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        
        individual_path = os.path.join(output_dir, f'{controller_name}_trajectory.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot to: {individual_path}")
        plt.close(fig_ind)
    
    plt.close(fig)


def run_controller(config_name):
    """Run a single controller and return trajectory data."""
    print(f"\n{'='*60}")
    print(f"Running {config_name.upper()}")
    print(f"{'='*60}")
    
    config = OmegaConf.load(f"configs/{config_name}.yaml")
    params = ControlParams(**config.params)

    if config_name == "cont_lqr":
        cont_lqr = ContinuousLQR(
            A=params.A,
            B=params.B,
            Q=params.Q,
            R=params.R,
        )
        Kc = cont_lqr.solve()
        print(f"Continuous LQR Gain K:\n{Kc}")
        # Override render_mode to None for data collection
        runner_params = dict(config.runner_args)
        runner_params['render_mode'] = None
        runner = Runner(cont_lqr, **runner_params)
        data = runner.run(**config.run_params)
        
    elif config_name == "disc_lqr":
        disc_lqr = DiscreteLQR(
            A=params.A,
            B=params.B,
            Q=params.Q,
            R=params.R,
            dt=config.runner_args.dt,
        )
        Kd = disc_lqr.solve()
        print(f"Discrete LQR Gain K:\n{Kd}")
        # Override render_mode to None for data collection
        runner_params = dict(config.runner_args)
        runner_params['render_mode'] = None
        runner = Runner(disc_lqr, **runner_params)
        data = runner.run(**config.run_params)
        
    elif config_name == "ilqr":
        ilqr = ILQRController(
            g=config.params.g,
            m_c=config.params.m_c,
            m_p=config.params.m_p,
            l=config.params.l,
            Q=params.Q,
            R=params.R,
            Qf=params.Qf,
            dt=config.runner_args.dt,
            **config.controller,
        )
        ilqr_mpc = ModelPredictiveControllerWrapper(
            controller=ilqr,
            dt=config.runner_args.dt,
            T_hor=config.runner_args.T_hor,
            force_mag=config.runner_args.force_mag
        )
        runner_params = {
            'max_episode_steps': config.runner_args.max_episode_steps,
            'dt': config.runner_args.dt,
            'force_mag': config.runner_args.force_mag,
            'render_mode': None,
        }
        runner = Runner(ilqr_mpc, **runner_params)
        data = runner.run(**config.run_params)
    
    return data


def main():
    args = parse_arguments()
    
    data_dict = {}
    
    if args.config == "all":
        configs_to_run = ["cont_lqr", "disc_lqr", "ilqr"]
    else:
        configs_to_run = [args.config]
    
    # Run each controller and collect data
    for config_name in configs_to_run:
        data = run_controller(config_name)
        data_dict[config_name] = data
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    plot_trajectories(data_dict, args.output_dir)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    for controller_name, data in data_dict.items():
        print(f"\n{controller_name.upper()}:")
        print(f"  Total steps: {len(data['time'])}")
        print(f"  Final time: {data['time'][-1]:.2f} s")
        print(f"  Max angle deviation: {np.max(np.abs(np.degrees(data['angle']))):.3f} degrees")
        print(f"  Max position deviation: {np.max(np.abs(data['position'])):.3f} m")
        print(f"  Final angle: {np.degrees(data['angle'][-1]):.3f} degrees")
        print(f"  Final position: {data['position'][-1]:.3f} m")


if __name__ == "__main__":
    main()