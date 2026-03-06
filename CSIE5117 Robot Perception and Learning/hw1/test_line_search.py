"""
Test with more challenging scenarios to demonstrate line search benefits
"""
import os
from omegaconf import OmegaConf
from src.env import CartPoleEnv
from src.params import ControlParams
from src.ilqr_ls import ILQRController
from src.mpc import ModelPredictiveControllerWrapper
import numpy as np


class Runner:
    def __init__(
        self,
        controller,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode="human",
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

        self.env.stop_recording()
        self.env.save_gif(save_path)
        self.env.close()
        return total_r, steps


def run_experiment_with_params(use_line_search, name, seed, max_iter, tol, T_hor, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Line Search = {use_line_search}, max_iter = {max_iter}, T_hor = {T_hor}")
        print(f"{'='*60}")
    
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
        max_iter=max_iter,
        tol=tol,
        use_line_search=use_line_search,
    )
    
    ilqr_mpc = ModelPredictiveControllerWrapper(
        controller=ilqr,
        dt=config.runner_args.dt,
        T_hor=T_hor,
        force_mag=config.runner_args.force_mag
    )
    
    runner_params = {
        'max_episode_steps': config.runner_args.max_episode_steps,
        'dt': config.runner_args.dt,
        'force_mag': config.runner_args.force_mag,
        'render_mode': None,
    }
    runner = Runner(ilqr_mpc, **runner_params)
    
    save_path = f"videos/cartpole_{name}_seed{seed}.gif"
    total_return, steps = runner.run(
        max_steps=config.run_params.max_steps,
        seed=seed,
        deadband=config.run_params.deadband,
        save_path=save_path
    )
    
    stats = {
        'return': total_return,
        'steps': steps,
        'line_search_stats': ilqr.line_search_stats if hasattr(ilqr, 'line_search_stats') else [],
        'max_iter': max_iter,
        'T_hor': T_hor
    }
    
    if verbose and hasattr(ilqr, 'line_search_stats') and len(ilqr.line_search_stats) > 0:
        ls_stats = ilqr.line_search_stats
        print(f"\nConvergence info:")
        print(f"  Iterations: {len(ls_stats)}")
        print(f"  Initial cost: {ls_stats[0]['J_old']:.2f}")
        print(f"  Final cost: {ls_stats[-1]['J_new']:.2f}")
        print(f"  Total improvement: {ls_stats[0]['J_old'] - ls_stats[-1]['J_new']:.2f}")
        
        if use_line_search:
            # Show which alphas were used
            alpha_used = [stat['alpha'] for stat in ls_stats]
            print(f"  Alphas used: {alpha_used[:10]}" + ("..." if len(alpha_used) > 10 else ""))
    
    return stats


def test_scenario(scenario_name, max_iter, tol, T_hor, seeds):
    """Test a specific scenario"""
    print(f"\n{'#'*60}")
    print(f"# SCENARIO: {scenario_name}")
    print(f"# max_iter={max_iter}, tol={tol}, T_hor={T_hor}")
    print(f"{'#'*60}")
    
    results = {'no_ls': [], 'with_ls': []}
    
    for use_ls, key, name in [(False, 'no_ls', 'no_ls'), (True, 'with_ls', 'with_ls')]:
        print(f"\n{'='*60}")
        print(f"{'WITH' if use_ls else 'WITHOUT'} LINE SEARCH")
        print(f"{'='*60}")
        
        for seed in seeds:
            stats = run_experiment_with_params(
                use_line_search=use_ls,
                name=name,
                seed=seed,
                max_iter=max_iter,
                tol=tol,
                T_hor=T_hor,
                verbose=False
            )
            results[key].append(stats)
            
            # Show iteration count and cost for first planning step
            if stats['line_search_stats']:
                ls = stats['line_search_stats']
                n_iter = len(ls)
                final_cost = ls[-1]['J_new']
                print(f"Seed {seed:3d}: Return={stats['return']:6.1f}, Steps={stats['steps']:4d}, "
                      f"Iters={n_iter:2d}, FinalCost={final_cost:7.2f}")
            else:
                print(f"Seed {seed:3d}: Return={stats['return']:6.1f}, Steps={stats['steps']:4d}")
    
    # Summary
    print(f"\n{'-'*60}")
    print(f"SUMMARY - {scenario_name}")
    print(f"{'-'*60}")
    
    for key, label in [('no_ls', 'Without LS'), ('with_ls', 'With LS')]:
        returns = [s['return'] for s in results[key]]
        steps = [s['steps'] for s in results[key]]
        
        if results[key][0]['line_search_stats']:
            iters = [len(s['line_search_stats']) for s in results[key]]
            final_costs = [s['line_search_stats'][-1]['J_new'] for s in results[key] if s['line_search_stats']]
            
            print(f"\n{label}:")
            print(f"  Success: {sum(1 for r in returns if r >= 1000)}/{len(returns)}")
            print(f"  Avg Return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
            print(f"  Avg Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
            print(f"  Avg Iterations: {np.mean(iters):.1f} ± {np.std(iters):.1f}")
            print(f"  Avg Final Cost: {np.mean(final_costs):.2f} ± {np.std(final_costs):.2f}")
        else:
            print(f"\n{label}:")
            print(f"  Success: {sum(1 for r in returns if r >= 1000)}/{len(returns)}")
            print(f"  Avg Return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
            print(f"  Avg Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    
    return results


if __name__ == "__main__":
    seeds = [0, 42, 123, 456, 789]
    
    # Test different scenarios
    scenarios = [
        ("Baseline", 50, 1e-6, 2.0),
        ("Limited Iterations", 10, 1e-6, 2.0),
        ("Tighter Convergence", 50, 1e-8, 2.0),
        ("Shorter Horizon", 50, 1e-6, 1.0),
        ("Very Limited", 5, 1e-6, 1.5),
    ]
    
    all_results = {}
    for scenario_name, max_iter, tol, T_hor in scenarios:
        all_results[scenario_name] = test_scenario(scenario_name, max_iter, tol, T_hor, seeds)
    
    # Overall comparison
    print(f"\n\n{'#'*60}")
    print(f"# OVERALL COMPARISON")
    print(f"{'#'*60}")
    
    for scenario_name in all_results.keys():
        results = all_results[scenario_name]
        
        no_ls_success = sum(1 for s in results['no_ls'] if s['return'] >= 1000)
        with_ls_success = sum(1 for s in results['with_ls'] if s['return'] >= 1000)
        
        no_ls_avg_iters = np.mean([len(s['line_search_stats']) for s in results['no_ls'] if s['line_search_stats']])
        with_ls_avg_iters = np.mean([len(s['line_search_stats']) for s in results['with_ls'] if s['line_search_stats']])
        
        print(f"\n{scenario_name}:")
        print(f"  Success: {no_ls_success}/5 vs {with_ls_success}/5 (no LS vs with LS)")
        print(f"  Avg Iterations: {no_ls_avg_iters:.1f} vs {with_ls_avg_iters:.1f}")