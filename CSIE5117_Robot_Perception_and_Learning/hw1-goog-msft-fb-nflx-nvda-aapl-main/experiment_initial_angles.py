"""
Experiment script to test different initial angles for CartPole controllers.
This script systematically tests CONT-LQR, DISC-LQR, and iLQR with various initial pole angles.
"""

import os
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from typing import Tuple

from src.env import CartPoleEnv
from src.params import ControlParams
from src.lqr import ContinuousLQR, DiscreteLQR
from src.ilqr import ILQRController
from src.mpc import ModelPredictiveControllerWrapper


class CustomCartPoleEnv(CartPoleEnv):
    """Extended CartPoleEnv that allows setting initial angle."""
    
    def reset_with_angle(self, angle: float, seed: int = None) -> np.ndarray:
        """
        Reset environment with a specific initial angle.
        
        Parameters
        ----------
        angle : float
            Initial pole angle in radians.
        seed : int, optional
            Random seed for reproducibility.
            
        Returns
        -------
        obs : np.ndarray
            Initial observation [x, x_dot, theta, theta_dot].
        """
        # First do a normal reset to initialize the environment
        obs, _ = self.env.reset(seed=seed)
        
        # Then override the state with our desired initial angle
        # State is [x, x_dot, theta, theta_dot]
        self.env.unwrapped.state = np.array([0.0, 0.0, angle, 0.0], dtype=np.float32)
        
        # Return the new state as observation
        obs = np.array(self.env.unwrapped.state, dtype=float)
        
        return obs


def run_experiment(
    controller,
    max_steps: int,
    seed: int,
    initial_angle: float,
    dt: float = 0.02,
    force_mag: float = 10.0,
    deadband: float = 0.0,
) -> Tuple[float, int, bool]:
    """
    Run a single experiment with given controller and initial angle.
    
    Parameters
    ----------
    controller : BaseLQR or ModelPredictiveControllerWrapper
        The controller to test.
    max_steps : int
        Maximum number of steps to run.
    seed : int
        Random seed.
    initial_angle : float
        Initial pole angle in radians.
    dt : float
        Time step.
    force_mag : float
        Force magnitude.
    deadband : float
        Deadband for control.
        
    Returns
    -------
    total_reward : float
        Total reward accumulated.
    steps_survived : int
        Number of steps before failure (or max_steps if successful).
    success : bool
        Whether the episode reached max_steps.
    """
    env = CustomCartPoleEnv(
        max_episode_steps=10000,
        dt=dt,
        force_mag=force_mag,
        render_mode=None,  # No rendering for batch experiments
    )
    
    # Reset environment for iLQR MPC wrapper if needed
    if hasattr(controller, 'reset'):
        controller.reset()
    
    # Reset with the specified initial angle
    obs = env.reset_with_angle(initial_angle, seed=seed)
    
    total_reward = 0.0
    steps = 0
    
    for step in range(max_steps):
        action = controller.get_action(obs, deadband=deadband)
        obs, reward, done = env.step(action)
        
        total_reward += reward
        steps = step + 1
        
        if done:
            break
    
    env.close()
    success = (steps >= max_steps)
    
    return total_reward, steps, success


def main():
    # Define initial angles to test (in degrees)
    angles_degrees = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
    angles_radians = [np.deg2rad(angle) for angle in angles_degrees]
    
    # Controllers to test
    controllers_config = ['cont_lqr', 'disc_lqr', 'ilqr']
    
    # Store results
    results = []
    
    print("=" * 80)
    print("CartPole Initial Angle Experiment")
    print("=" * 80)
    print(f"Testing angles: {angles_degrees} degrees")
    print(f"Controllers: {controllers_config}")
    print("=" * 80)
    print()
    
    for config_name in controllers_config:
        print(f"\n{'='*80}")
        print(f"Testing: {config_name.upper()}")
        print(f"{'='*80}")
        
        # Load configuration
        config = OmegaConf.load(f"configs/{config_name}.yaml")
        params = ControlParams(**config.params)
        
        for angle_deg, angle_rad in zip(angles_degrees, angles_radians):
            print(f"  Initial angle: {angle_deg:+6.1f}° ({angle_rad:+7.4f} rad) ... ", end="", flush=True)
            
            # Create controller based on type
            if config_name == "cont_lqr":
                controller = ContinuousLQR(
                    A=params.A,
                    B=params.B,
                    Q=params.Q,
                    R=params.R,
                )
                controller.solve()
                
            elif config_name == "disc_lqr":
                controller = DiscreteLQR(
                    A=params.A,
                    B=params.B,
                    Q=params.Q,
                    R=params.R,
                    dt=config.runner_args.dt,
                )
                controller.solve()
                
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
                controller = ModelPredictiveControllerWrapper(
                    controller=ilqr,
                    dt=config.runner_args.dt,
                    T_hor=config.runner_args.T_hor,
                    force_mag=config.runner_args.force_mag
                )
            
            # Run experiment
            try:
                total_reward, steps, success = run_experiment(
                    controller=controller,
                    max_steps=config.run_params.max_steps,
                    seed=config.run_params.seed,
                    initial_angle=angle_rad,
                    dt=config.runner_args.dt,
                    force_mag=config.runner_args.force_mag,
                    deadband=config.run_params.deadband,
                )
                
                results.append({
                    'Controller': config_name.upper().replace('_', '-'),
                    'Initial Angle (deg)': angle_deg,
                    'Initial Angle (rad)': f"{angle_rad:.4f}",
                    'Steps Survived': steps,
                    'Total Reward': f"{total_reward:.1f}",
                    'Success': 'Yes' if success else 'No',
                })
                
                status = "✓ SUCCESS" if success else f"✗ FAILED (step {steps})"
                print(f"{status}")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results.append({
                    'Controller': config_name.upper().replace('_', '-'),
                    'Initial Angle (deg)': angle_deg,
                    'Initial Angle (rad)': f"{angle_rad:.4f}",
                    'Steps Survived': 0,
                    'Total Reward': '0.0',
                    'Success': 'Error',
                })
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Print full results table
    print(df.to_string(index=False))
    print()
    
    # Save to CSV
    output_file = "results/initial_angle_experiment.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create pivot table for better visualization
    print("\n" + "=" * 80)
    print("STEPS SURVIVED BY CONTROLLER AND ANGLE")
    print("=" * 80)
    
    # Convert Steps Survived to numeric for pivot
    df_numeric = df.copy()
    df_numeric['Steps Survived'] = pd.to_numeric(df_numeric['Steps Survived'], errors='coerce')
    
    pivot = df_numeric.pivot_table(
        values='Steps Survived',
        index='Initial Angle (deg)',
        columns='Controller',
        aggfunc='first'
    )
    print(pivot.to_string())
    
    # Save pivot table
    pivot_file = "results/initial_angle_pivot.csv"
    pivot.to_csv(pivot_file)
    print(f"\nPivot table saved to: {pivot_file}")
    
    # Print success statistics
    print("\n" + "=" * 80)
    print("SUCCESS STATISTICS")
    print("=" * 80)
    for controller in controllers_config:
        controller_name = controller.upper().replace('_', '-')
        controller_df = df[df['Controller'] == controller_name]
        success_count = (controller_df['Success'] == 'Yes').sum()
        total_count = len(controller_df)
        success_rate = success_count / total_count * 100
        print(f"{controller_name:12s}: {success_count:2d}/{total_count:2d} successful ({success_rate:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("Experiment completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()