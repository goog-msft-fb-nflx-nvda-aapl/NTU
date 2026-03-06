import numpy as np
from typing import Tuple, Optional
from numpy.linalg import inv
from scipy.linalg import eig, expm, ordqz
from src.riccati_solvers import solve_care_custom, solve_dare_custom


class BaseLQR:
    def u(self, x: np.ndarray) -> float:
        """
        Compute the continuous-valued control signal u = -Kx.
        This must be called after the controller has solved for the gain matrix K.
        """
        if not hasattr(self, "K"):
            raise RuntimeError("LQR gain K is not computed. Call solve() first.")
        return float(-(self.K @ x))

    def get_action(self, x: np.ndarray, deadband: float = 0.0) -> int:
        """
        Map the continuous control value u to the discrete CartPole action space.

        Parameters
        ----------
        x : np.ndarray
            State vector [x, x_dot, theta, theta_dot].
        deadband : float
            Optional deadband around zero to avoid control chattering.

        Returns
        -------
        int
            Discrete action: 0 = push left, 1 = push right.
        """
        u = self.u(x)
        if abs(u) <= deadband:
            return 1  # Default to pushing right within deadband
        return 1 if u > 0 else 0


class ContinuousLQR(BaseLQR):
    """Continuous-time LQR controller for linearized CartPole dynamics."""

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = None

    def solve(self) -> np.ndarray:
        # ==================== CUSTOM CARE SOLVER ====================
        # Solve the continuous-time algebraic Riccati equation (CARE)
        # CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
        # Then compute optimal gain: K = R^{-1} B^T P
        P = solve_care_custom(self.A, self.B, self.Q, self.R)
        self.K = inv(self.R) @ self.B.T @ P
        # ==================== END OF CUSTOM SOLVER ====================
        return self.K


def c2d(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous-time state-space model to discrete-time using Zero-Order Hold (ZOH).

    Parameters
    ----------
    A : np.ndarray
        Continuous-time system matrix.
    B : np.ndarray
        Continuous-time input matrix.
    dt : float
        Sampling time step.

    Returns
    -------
    Ad : np.ndarray
        Discrete-time system matrix.
    Bd : np.ndarray
        Discrete-time input matrix.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


class DiscreteLQR(BaseLQR):
    """Discrete-time LQR controller for linearized CartPole dynamics."""

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
        self.dt = dt
        self.Ad, self.Bd = c2d(A, B, dt)
        self.Q = Q
        self.R = R
        self.K = None

    def solve(self) -> np.ndarray:
        # ==================== CUSTOM DARE SOLVER ====================
        # Solve the discrete-time algebraic Riccati equation (DARE)
        # DARE: A^T P A - P - A^T P B(R + B^T P B)^{-1} B^T P A + Q = 0
        # Then compute optimal gain: K = (R + B^T P B)^{-1} B^T P A
        P = solve_dare_custom(self.Ad, self.Bd, self.Q, self.R)
        self.K = inv(self.R + self.Bd.T @ P @ self.Bd) @ (self.Bd.T @ P @ self.Ad)
        # ==================== END OF CUSTOM SOLVER ====================
        return self.K