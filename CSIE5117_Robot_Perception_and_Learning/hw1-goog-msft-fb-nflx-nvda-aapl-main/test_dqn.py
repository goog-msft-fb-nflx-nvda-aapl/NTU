"""
Test DQN and compare with LQR/iLQR
"""
import os
import numpy as np
import time
from omegaconf import OmegaConf

from src.env import CartPoleEnv
from src.params import ControlParams
from src.lqr import ContinuousLQR, DiscreteLQR
from src.ilqr import ILQRController
from src.mpc import ModelPredictiveControllerWrapper
from src.dqn import DQNAgent, DQNController


class Runner:
    def __init__(
        self,
        controller,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode=None,
    ):
        self.controller = controller
        self.env = CartPoleEnv(
            max_episode_steps=max_episode_steps,
            dt=dt,
            force_mag=force_mag,
            render_mode=render_mode,
        )

    def run(self, max_steps=10000, seed=0, deadband=0.0, save_path="videos/cartpole.gif"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.env.start_recording(fps=30, step_capture="post")
        obs = self.env.reset(seed=seed)
        total_r = 0.0
        steps = 0
        
        start_time = time.time()
        for _ in range(max_steps):
            action = self.controller.get_action(obs, deadband=deadband)
            out = self.env.step(action)

            if len(out) == 5:
                obs, r, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, r, done = out
            total_r += r
            steps += 1
            if done:
                break
        
        elapsed_time = time.time() - start_time

        self.env.stop_recording()
        self.env.save_gif(save_path)
        self.env.close()
        return total_r, steps, elapsed_time


def test_controller(controller_type, seeds=[0, 42, 123, 456, 789]):
    """Test a controller across multiple seeds"""
    print(f"\n{'='*60}")
    print(f"Testing {controller_type}")
    print(f"{'='*60}")
    
    results = []
    
    for seed in seeds:
        if controller_type == "Continuous LQR":
            config = OmegaConf.load("configs/cont_lqr.yaml")
            params = ControlParams(**config.params)
            controller = ContinuousLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
            )
            controller.solve()
            
        elif controller_type == "Discrete LQR":
            config = OmegaConf.load("configs/disc_lqr.yaml")
            params = ControlParams(**config.params)
            controller = DiscreteLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
                dt=config.runner_args.dt,
            )
            controller.solve()
            
        elif controller_type == "iLQR":
            config = OmegaConf.load("configs/ilqr.yaml")
            params = ControlParams(**config.params)
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
            
        elif controller_type == "DQN":
            config = OmegaConf.load("configs/dqn.yaml")
            agent = DQNAgent(
                state_dim=4,
                action_dim=2,
                hidden_dims=config.agent.hidden_dims,
                learning_rate=config.agent.learning_rate,
                gamma=config.agent.gamma,
                device=config.agent.device
            )
            agent.load(config.training.save_path)
            controller = DQNController(agent)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        # Get appropriate config
        if controller_type == "DQN":
            run_config = config
        else:
            run_config = config
        
        # Run
        runner = Runner(
            controller,
            max_episode_steps=run_config.runner_args.max_episode_steps if hasattr(run_config, 'runner_args') else run_config.env.max_episode_steps,
            dt=run_config.runner_args.dt if hasattr(run_config, 'runner_args') else run_config.env.dt,
            force_mag=run_config.runner_args.force_mag if hasattr(run_config, 'runner_args') else run_config.env.force_mag,
            render_mode=None,
        )
        
        save_path = f"videos/comparison/{controller_type.lower().replace(' ', '_')}_seed{seed}.gif"
        reward, steps, elapsed = runner.run(
            max_steps=1000,
            seed=seed,
            deadband=0.0,
            save_path=save_path
        )
        
        results.append({
            'seed': seed,
            'reward': reward,
            'steps': steps,
            'time': elapsed
        })
        
        print(f"Seed {seed:3d}: Reward={reward:6.1f}, Steps={steps:4d}, Time={elapsed:.3f}s")
    
    return results


def compare_all_methods():
    """Compare all methods"""
    print("\n" + "#"*60)
    print("# COMPREHENSIVE COMPARISON: LQR vs iLQR vs DQN")
    print("#"*60)
    
    methods = [
        "Continuous LQR",
        "Discrete LQR",
        "iLQR",
        "DQN"
    ]
    
    all_results = {}
    
    for method in methods:
        try:
            results = test_controller(method)
            all_results[method] = results
        except Exception as e:
            print(f"Error testing {method}: {e}")
            all_results[method] = None
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Method':<20} {'Success':<10} {'Avg Reward':<15} {'Avg Steps':<15} {'Avg Time (s)':<15}")
    print("-"*60)
    
    for method, results in all_results.items():
        if results is None:
            print(f"{method:<20} {'N/A':<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue
        
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        times = [r['time'] for r in results]
        
        success_rate = f"{sum(1 for r in rewards if r >= 1000)}/5"
        avg_reward = f"{np.mean(rewards):.1f} ± {np.std(rewards):.1f}"
        avg_steps = f"{np.mean(steps):.1f} ± {np.std(steps):.1f}"
        avg_time = f"{np.mean(times):.3f} ± {np.std(times):.3f}"
        
        print(f"{method:<20} {success_rate:<10} {avg_reward:<15} {avg_steps:<15} {avg_time:<15}")
    
    # Detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    for method, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n{method}:")
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        times = [r['time'] for r in results]
        
        print(f"  Reward:  mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}, "
              f"min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")
        print(f"  Steps:   mean={np.mean(steps):.2f}, std={np.std(steps):.2f}, "
              f"min={np.min(steps):.0f}, max={np.max(steps):.0f}")
        print(f"  Time:    mean={np.mean(times):.4f}s, std={np.std(times):.4f}s, "
              f"min={np.min(times):.4f}s, max={np.max(times):.4f}s")
    
    return all_results


if __name__ == "__main__":
    # Check if DQN model exists
    if not os.path.exists("models/dqn_cartpole.pt"):
        print("DQN model not found. Please train first using:")
        print("  uv run python train_dqn.py")
        print("\nRunning comparison with LQR/iLQR only...")
        
        methods = ["Continuous LQR", "Discrete LQR", "iLQR"]
        all_results = {}
        for method in methods:
            results = test_controller(method)
            all_results[method] = results
    else:
        all_results = compare_all_methods()
    
    print("\n" + "#"*60)
    print("# Comparison completed!")
    print("#"*60)