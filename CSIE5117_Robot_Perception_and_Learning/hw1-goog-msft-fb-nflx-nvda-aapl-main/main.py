import os
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
    )
    return parser.parse_args()


class Runner:
    def __init__(
        self,
        controller: BaseLQR,
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
        for _ in range(max_steps):
            action = self.controller.get_action(obs, deadband=deadband)
            out = self.env.step(action)

            if len(out) == 5:
                obs, r, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, r, done = out
            total_r += r
            if done:
                break

        print(f"[Runner]: return={total_r}")
        self.env.stop_recording()
        self.env.save_gif(save_path)
        self.env.close()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(f"configs/{args.config}.yaml")
    params = ControlParams(**config.params)

    match args.config:
        case "cont_lqr":
            cont_lqr = ContinuousLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
            )
            Kc = cont_lqr.solve()
            runner = Runner(cont_lqr, **config.runner_args)
            runner.run(**config.run_params)

        case "disc_lqr":
            disc_lqr = DiscreteLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
                dt=config.runner_args.dt,
            )
            Kd = disc_lqr.solve()
            runner = Runner(disc_lqr, **config.runner_args)
            runner.run(**config.run_params)

        case "ilqr":
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
            # MODIFIED: Extract only the parameters that Runner accepts from config.runner_args
            # to avoid passing T_hor which causes TypeError
            runner_params = {
                'max_episode_steps': config.runner_args.max_episode_steps,
                'dt': config.runner_args.dt,
                'force_mag': config.runner_args.force_mag,
                'render_mode': config.runner_args.render_mode,
            }
            runner = Runner(ilqr_mpc, **runner_params)
            runner.run(**config.run_params)

        case _:
            raise ValueError(f"Unknown method: {args.config}")