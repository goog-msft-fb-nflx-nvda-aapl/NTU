import numpy as np


class ModelPredictiveControllerWrapper:
    def __init__(self, controller, dt: float, T_hor: float, force_mag: float):
        self.controller = controller
        self.dt = dt
        self.force_mag = force_mag
        self.N_hor = int(T_hor / dt)
        assert self.N_hor >= 5, "Horizon too short."

        self.U_ws = np.zeros((self.N_hor, 1), dtype=float)

    def reset(self):
        self.U_ws[:] = 0.0
        if hasattr(self.controller, "reset_episode"):
            self.controller.reset_episode()

    def get_action(self, obs: np.ndarray, deadband: float = 0.0) -> int:
        # 1. Re-plan using iLQR
        X_opt, U_opt, K_seq, k_seq, _ = self.controller.plan(obs, self.U_ws)

        # 2. Compute first control with feedback
        action = self.controller.get_action(obs, deadband, X_opt[0], U_opt[0], k_seq[0], K_seq[0])

        # 3. Warm-start: shift U forward for next step
        self.U_ws[:-1, 0] = U_opt[1:, 0]
        self.U_ws[-1, 0] = 0.0

        return action
