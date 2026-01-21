import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
import yaml

from src.env import CartPoleEnv
from src.params import ControlParams
from src.lqr import ContinuousLQR, DiscreteLQR
from src.ilqr import ILQRController
from src.mpc import ModelPredictiveControllerWrapper


def parse_arguments():
    parser = ArgumentParser(description="Parameter sensitivity study for cart-pole controllers")
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        default="cont_lqr",
        choices=["cont_lqr", "disc_lqr", "ilqr", "all"],
        help="Controller type to study (or 'all' for all three)"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="results/parameter_study",
        help="Output directory for results"
    )
    return parser.parse_args()


class ParameterStudyRunner:
    """Runs parameter sensitivity studies for cart-pole controllers."""
    
    def __init__(self, output_dir="results/parameter_study"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_experiment(
        self, 
        controller_type: str,
        Q: List[float],
        R: float,
        max_steps: int,
        seed: int = 0,
        experiment_name: str = ""
    ) -> Dict:
        """Run a single experiment with given parameters."""
        
        # Load base config
        config = OmegaConf.load(f"configs/{controller_type}.yaml")
        
        # Override parameters
        config.params.Q = Q
        config.params.R = [R]
        if controller_type == "ilqr":
            config.params.Qf = Q  # Use same Q for terminal cost
        config.run_params.max_steps = max_steps
        
        params = ControlParams(**config.params)
        
        # Create controller
        if controller_type == "cont_lqr":
            controller = ContinuousLQR(
                A=params.A, B=params.B, Q=params.Q, R=params.R
            )
            controller.solve()
            K = controller.K
        elif controller_type == "disc_lqr":
            controller = DiscreteLQR(
                A=params.A, B=params.B, Q=params.Q, R=params.R,
                dt=config.runner_args.dt
            )
            controller.solve()
            K = controller.K
        elif controller_type == "ilqr":
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
            K = None
        
        # Run simulation
        runner_params = dict(config.runner_args)
        runner_params['render_mode'] = None
        
        # Remove parameters that CartPoleEnv doesn't accept
        env_params = {
            'max_episode_steps': runner_params.get('max_episode_steps', 10000),
            'dt': runner_params.get('dt', 0.02),
            'force_mag': runner_params.get('force_mag', 10.0),
            'render_mode': runner_params.get('render_mode', None)
        }
        
        env = CartPoleEnv(**env_params)
        obs = env.reset(seed=seed)
        
        trajectory = {
            'time': [0],
            'position': [obs[0]],
            'velocity': [obs[1]],
            'angle': [obs[2]],
            'angular_velocity': [obs[3]],
            'control': []
        }
        
        total_reward = 0
        dt = config.runner_args.dt
        
        for step in range(max_steps):
            action = controller.get_action(obs, deadband=0.0)
            out = env.step(action)
            
            if len(out) == 5:
                obs, r, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, r, done = out
            
            total_reward += r
            trajectory['time'].append((step + 1) * dt)
            trajectory['position'].append(obs[0])
            trajectory['velocity'].append(obs[1])
            trajectory['angle'].append(obs[2])
            trajectory['angular_velocity'].append(obs[3])
            trajectory['control'].append(action)
            
            if done:
                break
        
        env.close()
        
        # Convert to numpy arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        
        # Compute metrics
        steps_completed = len(trajectory['time']) - 1
        final_angle_error = abs(np.degrees(trajectory['angle'][-1]))
        final_position_error = abs(trajectory['position'][-1])
        max_angle_dev = np.max(np.abs(np.degrees(trajectory['angle'])))
        max_position_dev = np.max(np.abs(trajectory['position']))
        
        # Control effort metrics
        if len(trajectory['control']) > 0:
            control_switches = np.sum(np.diff(trajectory['control']) != 0)
            control_switch_rate = control_switches / steps_completed if steps_completed > 0 else 0
        else:
            control_switches = 0
            control_switch_rate = 0
        
        results = {
            'controller_type': controller_type,
            'Q': Q,
            'R': R,
            'max_steps': max_steps,
            'K': K.tolist() if K is not None else None,
            'steps_completed': steps_completed,
            'total_reward': total_reward,
            'success': steps_completed >= max_steps,
            'final_angle_error': final_angle_error,
            'final_position_error': final_position_error,
            'max_angle_dev': max_angle_dev,
            'max_position_dev': max_position_dev,
            'control_switches': control_switches,
            'control_switch_rate': control_switch_rate,
            'trajectory': trajectory,
            'experiment_name': experiment_name
        }
        
        return results


def plot_comparison(results_list: List[Dict], output_path: str, title: str):
    """Plot comparison of multiple experiments."""
    
    n_experiments = len(results_list)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_experiments))
    
    for idx, result in enumerate(results_list):
        traj = result['trajectory']
        label = result['experiment_name']
        color = colors[idx]
        
        # Plot angle
        axes[0].plot(traj['time'], np.degrees(traj['angle']), 
                    label=label, color=color, linewidth=2, alpha=0.8)
        
        # Plot position
        axes[1].plot(traj['time'], traj['position'], 
                    label=label, color=color, linewidth=2, alpha=0.8)
        
        # Plot control (as continuous line showing action value)
        if len(traj['control']) > 0:
            control_time = traj['time'][:-1]  # Control has one less point
            axes[2].plot(control_time, traj['control'], 
                        label=label, color=color, linewidth=1.5, alpha=0.8)
    
    # Configure angle plot
    axes[0].set_ylabel('Pole Angle (degrees)', fontsize=12, fontweight='bold')
    axes[0].set_title(title, fontsize=14, fontweight='bold', pad=20)
    axes[0].legend(fontsize=9, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Configure position plot
    axes[1].set_ylabel('Cart Position (m)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Configure control plot
    axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Control Action', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=9, loc='best', framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_ylim([-0.2, 1.2])
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Left (0)', 'Right (1)'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_markdown_report(all_results: Dict[str, List[Dict]], output_dir: str, controller_type: str):
    """Generate markdown report summarizing all experiments."""
    
    report_path = os.path.join(output_dir, "Q1_parameter_study.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Q1. Parameter Sensitivity Study - {controller_type.upper()}\n\n")
        f.write("## Overview\n\n")
        f.write(f"This study investigates the effects of adjusting Q matrix (state cost), ")
        f.write(f"R value (control cost), and max_steps on the performance of the {controller_type.upper()} controller.\n\n")
        
        # Q matrix effects
        if 'q_variation' in all_results:
            f.write("## 1. Effect of Q Matrix (State Cost Weighting)\n\n")
            f.write("The Q matrix penalizes deviations in state variables: Q = diag([q_pos, q_vel, q_angle, q_ang_vel])\n\n")
            f.write("### Experiments Conducted\n\n")
            
            f.write("![Q Matrix Comparison](parameter_study/q_variation_comparison.png)\n\n")
            
            f.write("| Experiment | Q Matrix | Success | Steps | Max Angle (°) | Max Position (m) | Switch Rate |\n")
            f.write("|------------|----------|---------|-------|---------------|------------------|-------------|\n")
            
            for result in all_results['q_variation']:
                Q_str = str([f"{q:.1f}" for q in result['Q']]).replace("'", "")
                success = "✅" if result['success'] else "❌"
                f.write(f"| {result['experiment_name']} | {Q_str} | {success} | "
                       f"{result['steps_completed']} | {result['max_angle_dev']:.3f} | "
                       f"{result['max_position_dev']:.3f} | {result['control_switch_rate']:.3f} |\n")
            
            f.write("\n### Key Observations:\n\n")
            f.write("- **High Angle Weight (q_angle ↑)**: Prioritizes keeping the pole upright, ")
            f.write("may allow more cart position drift\n")
            f.write("- **High Position Weight (q_pos ↑)**: Keeps cart centered, ")
            f.write("but may allow larger angle deviations\n")
            f.write("- **Balanced Weights**: Achieves good overall performance with trade-offs\n\n")
        
        # R value effects
        if 'r_variation' in all_results:
            f.write("## 2. Effect of R Value (Control Cost)\n\n")
            f.write("The R parameter penalizes control effort (aggressive actions).\n\n")
            f.write("### Experiments Conducted\n\n")
            
            f.write("![R Value Comparison](parameter_study/r_variation_comparison.png)\n\n")
            
            f.write("| Experiment | R Value | Success | Steps | Max Angle (°) | Max Position (m) | Switch Rate |\n")
            f.write("|------------|---------|---------|-------|---------------|------------------|-------------|\n")
            
            for result in all_results['r_variation']:
                success = "✅" if result['success'] else "❌"
                f.write(f"| {result['experiment_name']} | {result['R']:.2f} | {success} | "
                       f"{result['steps_completed']} | {result['max_angle_dev']:.3f} | "
                       f"{result['max_position_dev']:.3f} | {result['control_switch_rate']:.3f} |\n")
            
            f.write("\n### Key Observations:\n\n")
            f.write("- **Low R (R → 0)**: Aggressive control, frequent switching, fast response\n")
            f.write("- **High R (R ↑)**: Conservative control, smoother actions, potentially slower response\n")
            f.write("- **Optimal R**: Balances control effort with performance requirements\n\n")
        
        # max_steps effects
        if 'steps_variation' in all_results:
            f.write("## 3. Effect of max_steps (Episode Length)\n\n")
            f.write("The max_steps parameter determines how long the controller must maintain stability.\n\n")
            f.write("### Experiments Conducted\n\n")
            
            f.write("![Steps Comparison](parameter_study/steps_variation_comparison.png)\n\n")
            
            f.write("| Experiment | Max Steps | Success | Steps Completed | Final Angle (°) | Final Position (m) |\n")
            f.write("|------------|-----------|---------|-----------------|-----------------|-------------------|\n")
            
            for result in all_results['steps_variation']:
                success = "✅" if result['success'] else "❌"
                f.write(f"| {result['experiment_name']} | {result['max_steps']} | {success} | "
                       f"{result['steps_completed']} | {result['final_angle_error']:.3f} | "
                       f"{result['final_position_error']:.3f} |\n")
            
            f.write("\n### Key Observations:\n\n")
            f.write("- **Longer Episodes**: Reveal long-term stability and drift issues\n")
            f.write("- **Shorter Episodes**: Easier to complete but may hide control deficiencies\n")
            f.write("- **Drift Accumulation**: Position drift typically increases with episode length\n\n")
        
        f.write("## Summary and Recommendations\n\n")
        f.write("### Parameter Tuning Guidelines\n\n")
        f.write("1. **Start with balanced Q matrix**: Equal emphasis on all states\n")
        f.write("2. **Increase angle weight**: If pole stability is most critical\n")
        f.write("3. **Increase position weight**: If cart centering is important\n")
        f.write("4. **Adjust R**: Lower for aggressive control, higher for smooth control\n")
        f.write("5. **Test with longer episodes**: To verify long-term stability\n\n")
        
        f.write("### Trade-offs\n\n")
        f.write("- **Performance vs. Control Effort**: Lower R → better performance but more energy\n")
        f.write("- **Angle vs. Position**: Higher q_angle → better pole balance but more cart drift\n")
        f.write("- **Responsiveness vs. Smoothness**: Aggressive gains → fast response but chattering\n\n")
    
    print(f"Markdown report saved to: {report_path}")


def run_study_for_controller(controller_type: str, output_dir: str):
    """Run complete parameter study for a single controller."""
    
    controller_output_dir = os.path.join(output_dir, controller_type)
    os.makedirs(controller_output_dir, exist_ok=True)
    runner = ParameterStudyRunner(controller_output_dir)
    
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"PARAMETER SENSITIVITY STUDY FOR {controller_type.upper()}")
    print(f"{'='*70}\n")
    
    # Study 1: Q matrix variation
    print("Study 1: Q Matrix Variation")
    print("-" * 70)
    
    q_experiments = [
        ([0.2, 0.2, 2.0, 0.5], "Baseline (balanced)"),
        ([2.0, 0.2, 2.0, 0.5], "High position weight"),
        ([0.2, 0.2, 10.0, 0.5], "High angle weight"),
        ([1.0, 1.0, 1.0, 1.0], "All equal weights"),
        ([0.1, 0.1, 5.0, 1.0], "Angle-focused"),
    ]
    
    q_results = []
    for Q, name in q_experiments:
        print(f"\nRunning: {name}")
        print(f"  Q = {Q}, R = 1.5")
        result = runner.run_experiment(
            controller_type=controller_type,
            Q=Q, R=1.5, max_steps=1000,
            experiment_name=name
        )
        q_results.append(result)
        print(f"  → Steps: {result['steps_completed']}/1000, "
              f"Max angle: {result['max_angle_dev']:.3f}°, "
              f"Max pos: {result['max_position_dev']:.3f}m")
    
    all_results['q_variation'] = q_results
    plot_comparison(q_results, f"{controller_output_dir}/q_variation_comparison.png",
                   f"{controller_type.upper()}: Q Matrix Variation")
    
    # Study 2: R value variation
    print(f"\n{'='*70}")
    print("Study 2: R Value Variation")
    print("-" * 70)
    
    r_experiments = [
        (0.5, "Low R (aggressive)"),
        (1.5, "Baseline R"),
        (5.0, "High R (conservative)"),
        (10.0, "Very high R"),
    ]
    
    r_results = []
    for R, name in r_experiments:
        print(f"\nRunning: {name}")
        print(f"  Q = [0.2, 0.2, 2.0, 0.5], R = {R}")
        result = runner.run_experiment(
            controller_type=controller_type,
            Q=[0.2, 0.2, 2.0, 0.5], R=R, max_steps=1000,
            experiment_name=name
        )
        r_results.append(result)
        print(f"  → Steps: {result['steps_completed']}/1000, "
              f"Switches: {result['control_switches']}, "
              f"Switch rate: {result['control_switch_rate']:.3f}")
    
    all_results['r_variation'] = r_results
    plot_comparison(r_results, f"{controller_output_dir}/r_variation_comparison.png",
                   f"{controller_type.upper()}: R Value Variation")
    
    # Study 3: max_steps variation
    print(f"\n{'='*70}")
    print("Study 3: max_steps Variation")
    print("-" * 70)
    
    steps_experiments = [
        (200, "Short episode (200)"),
        (500, "Medium episode (500)"),
        (1000, "Standard episode (1000)"),
        (2000, "Long episode (2000)"),
    ]
    
    steps_results = []
    for max_steps, name in steps_experiments:
        print(f"\nRunning: {name}")
        print(f"  Q = [0.2, 0.2, 2.0, 0.5], R = 1.5, max_steps = {max_steps}")
        result = runner.run_experiment(
            controller_type=controller_type,
            Q=[0.2, 0.2, 2.0, 0.5], R=1.5, max_steps=max_steps,
            experiment_name=name
        )
        steps_results.append(result)
        success_str = "SUCCESS" if result['success'] else "FAILED"
        print(f"  → {success_str}: {result['steps_completed']}/{max_steps} steps, "
              f"Final angle: {result['final_angle_error']:.3f}°, "
              f"Final pos: {result['final_position_error']:.3f}m")
    
    all_results['steps_variation'] = steps_results
    plot_comparison(steps_results, f"{controller_output_dir}/steps_variation_comparison.png",
                   f"{controller_type.upper()}: Episode Length Variation")
    
    # Generate comprehensive report
    print(f"\n{'='*70}")
    print("Generating markdown report...")
    print(f"{'='*70}\n")
    generate_markdown_report(all_results, controller_output_dir, controller_type)
    
    print("\n" + "="*70)
    print(f"STUDY COMPLETE FOR {controller_type.upper()}!")
    print("="*70)
    print(f"Results saved to: {controller_output_dir}/")
    print(f"  - Q variation plot: q_variation_comparison.png")
    print(f"  - R variation plot: r_variation_comparison.png")
    print(f"  - Steps variation plot: steps_variation_comparison.png")
    print(f"  - Report: Q1_parameter_study.md")
    
    return all_results


def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.controller == "all":
        controllers_to_run = ["cont_lqr", "disc_lqr", "ilqr"]
    else:
        controllers_to_run = [args.controller]
    
    all_controller_results = {}
    
    for controller_type in controllers_to_run:
        results = run_study_for_controller(controller_type, args.output_dir)
        all_controller_results[controller_type] = results
    
    print("\n" + "="*70)
    print("ALL STUDIES COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    for controller in controllers_to_run:
        print(f"  - {controller}/ subdirectory")


if __name__ == "__main__":
    main()