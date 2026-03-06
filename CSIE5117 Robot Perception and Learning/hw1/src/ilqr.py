import math
import numpy as np
from typing import Tuple, List, Optional

from .utils import to_scalar, to_vec, row1


def dynamic_func(
    x: np.ndarray,
    u: float,
    g: float,
    m_c: float,
    m_p: float,
    l: float,
) -> np.ndarray:
    """
    Nonlinear dynamics for cart-pole system.
    State: x = [x, x_dot, theta, theta_dot]^T
    
    Equations:
    theta_dd = (g*sin(theta) - cos(theta)*(u + m_p*l*theta_dot^2*sin(theta))/M) / (l*(4/3 - m_p*cos^2(theta)/M))
    x_dd = (u + m_p*l*(theta_dot^2*sin(theta) - theta_dd*cos(theta))) / M
    """
    M = m_c + m_p
    
    # Extract state variables
    x_pos = x[0]
    x_dot = x[1]
    theta = x[2]
    th_dot = x[3]
    
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    
    # Compute theta_dd (angular acceleration)
    numerator = g * sin_th - cos_th * (u + m_p * l * th_dot**2 * sin_th) / M
    denominator = l * (4.0/3.0 - m_p * cos_th**2 / M)
    th_dd = numerator / denominator
    
    # Compute x_dd (cart acceleration)
    x_dd = (u + m_p * l * (th_dot**2 * sin_th - th_dd * cos_th)) / M
    
    return np.array(
        [
            x_dot,
            x_dd,
            th_dot,
            th_dd,
        ],
    )


def euler_step(x: np.ndarray, u: float, dt: float, g: float, m_c: float, m_p: float, l: float) -> np.ndarray:
    """
    Euler integration: x_{k+1} = x_k + dt * f(x_k, u_k)
    """
    x_dot = dynamic_func(x, u, g, m_c, m_p, l)
    x_next = x + dt * x_dot
    return x_next


def fd_jacobian(f_d, x: np.ndarray, u: float, dt: float, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    n = x.size
    A = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        A[:, i] = (f_d(x + dx, u, dt) - f_d(x - dx, u, dt)) / (2 * eps)
    du = eps
    B = ((f_d(x, u + du, dt) - f_d(x, u - du, dt)) / (2 * du)).reshape(-1, 1)
    return A, B


class ILQRController:
    def __init__(
        self,
        g: float,
        m_c: float,
        m_p: float,
        l: float,
        Q: np.ndarray,
        R: float,
        Qf: np.ndarray,
        dt: float,
        max_iter: int = 100,
        tol: float = 1e-6,
        force_mag: float = 10.0,
    ):
        self.g = g
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.Q = Q
        self.R = float(R)
        self.Qf = Qf
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.force_mag = force_mag

        self.X_ref = None
        self.U_ff = None
        self.K_seq = None
        self.k_seq = None

    def f_discrete(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        return euler_step(x, u, dt, self.g, self.m_c, self.m_p, self.l)

    def stage_cost(self, x: np.ndarray, u: float) -> float:
        """
        Stage cost: l(x, u) = x^T * Q * x + u^T * R * u
        """
        J = float(x.T @ self.Q @ x + self.R * u**2)
        return J

    def terminal_cost(self, x: np.ndarray) -> float:
        return float(x.T @ self.Qf @ x)

    def plan(self, x0: np.ndarray, U_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[float], float]:
        x0 = to_vec(x0)
        U = np.asarray(U_init, dtype=float).reshape(-1, 1)

        # 1) initial rollout
        X, J = self.initial_rollout(x0, U)

        for _ in range(self.max_iter):
            # 2) linearize dynamics along current trajectory

            A_seq, B_seq = self.linearize_traj(X, U)

            # 3) backward pass
            bp_ok, K_seq, k_seq, _, _ = self.backward_pass(X, U, A_seq, B_seq)

            # 4) forward pass
            X_new, U_new, J_new, accepted = self.forward_pass(x0, X, U, K_seq, k_seq, J)

            if accepted:
                X, U, J = X_new, U_new, J_new

            if abs(J - J_new) < self.tol:
                break

        # 5) final backward pass to store gains for closed-loop use
        K_seq, k_seq = self.final_backward_pass(X, U)
        self.X_ref, self.U_ff, self.K_seq, self.k_seq = X, U, K_seq, k_seq
        return X, U, K_seq, k_seq, float(J)

    # Step 1: rollout
    def initial_rollout(self, x0: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, float]:
        N = U.shape[0]
        n = x0.size
        X = np.zeros((N + 1, n), dtype=float); X[0] = x0
        J = 0.0
        for k in range(N):
            X[k + 1] = self.f_discrete(X[k], float(U[k, 0]), self.dt)
            J += self.stage_cost(X[k], float(U[k, 0]))
        J += self.terminal_cost(X[-1])
        return X, float(J)

    # Step 2: linearize
    def linearize_traj(self, X: np.ndarray, U: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        N = U.shape[0]
        A_seq, B_seq = [], []
        for k in range(N):
            A, B = fd_jacobian(self.f_discrete, X[k], float(U[k, 0]), self.dt)
            A_seq.append(A)
            B_seq.append(B)
        return A_seq, B_seq

    # Step 3: backward pass
    def backward_pass(
        self,
        X: np.ndarray,
        U: np.ndarray,
        A_seq: List[np.ndarray],
        B_seq: List[np.ndarray],
    ) -> Tuple[bool, List[np.ndarray], List[float], np.ndarray, np.ndarray]:

        N = U.shape[0]
        n = X.shape[1]

        Vx  = to_vec(self.Qf @ X[-1])
        Vxx = self.Qf.copy()
        K_seq = [None] * N
        k_seq = [None] * N

        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]
            xk = X[k]
            uk = float(U[k, 0])

            # Cost function derivatives
            # l(x,u) = x^T Q x + u^T R u
            lx  = 2.0 * self.Q @ xk  # shape (n,)
            lu  = 2.0 * self.R * uk  # scalar
            lxx = 2.0 * self.Q       # shape (n, n)
            luu = 2.0 * self.R       # scalar
            lux = np.zeros((1, n))   # shape (1, n) - zero because l is separable in x and u

            # Q-function derivatives
            Qx  = lx + A.T @ Vx                    # shape (n,)
            Qu  = lu + (B.T @ Vx).item()           # scalar
            Qxx = lxx + A.T @ Vxx @ A              # shape (n, n)
            Qux = lux + B.T @ Vxx @ A              # shape (1, n)
            Quu = luu + (B.T @ Vxx @ B).item()     # scalar

            # Optimal feedback gains
            kff = -Qu / Quu                        # scalar (feedforward term)
            Kfb = -Qux / Quu                       # shape (1, n) (feedback gain)

            k_seq[k] = float(kff)
            K_seq[k] = row1(Kfb)

            Vx = to_vec(Qx - Qux.T.flatten() * Quu**-1 * Qu)
            Vxx = Qxx - Qux.T @ Qux / Quu
        return True, K_seq, k_seq, Vx, Vxx

    # Step 4: forward pass
    def forward_pass(
        self,
        x0: np.ndarray,
        X: np.ndarray,
        U: np.ndarray,
        K_seq: List[np.ndarray],
        k_seq: List[float],
        J_old: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:

        N = U.shape[0]
        X_new = np.zeros_like(X)
        X_new[0] = x0
        U_new = np.zeros_like(U)
        J_new = 0.0

        for k in range(N):
            u  = float(U[k, 0] + k_seq[k] + to_scalar(K_seq[k] @ to_vec(X_new[k] - X[k])))
            u  = self._clip_u(u)
            X_new[k + 1] = self.f_discrete(X_new[k], u, self.dt)
            U_new[k, 0]  = u
            J_new += self.stage_cost(X_new[k], u)
        J_new += self.terminal_cost(X_new[-1])

        return X_new, U_new, J_new, True

    # Step 5: final backward pass
    def final_backward_pass(self, X: np.ndarray, U: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        A_seq, B_seq = self.linearize_traj(X, U)

        N = U.shape[0]
        n = X.shape[1]
        Vx  = to_vec(self.Qf @ X[-1])
        Vxx = self.Qf.copy()
        K_seq = [None] * N
        k_seq = [None] * N

        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]
            xk = X[k]
            uk = float(U[k, 0])

            # Cost function derivatives
            lx  = 2.0 * self.Q @ xk
            lu  = 2.0 * self.R * uk
            lxx = 2.0 * self.Q
            luu = 2.0 * self.R
            lux = np.zeros((1, n))

            # Q-function derivatives
            Qx  = lx + A.T @ Vx
            Qu  = lu + (B.T @ Vx).item()
            Qxx = lxx + A.T @ Vxx @ A
            Qux = lux + B.T @ Vxx @ A
            Quu = luu + (B.T @ Vxx @ B).item()

            # Optimal feedback gains
            kff = -Qu / Quu
            Kfb = -Qux / Quu

            k_seq[k] = float(kff)
            K_seq[k] = row1(Kfb)

            Vx = to_vec(Qx - Qux.T.flatten() * Quu**-1 * Qu)
            Vxx = Qxx - Qux.T @ Qux / Quu
        return K_seq, k_seq

    def _clip_u(self, u: float) -> float:
        return float(np.clip(u, -self.force_mag, self.force_mag))

    def get_action(
        self,
        obs: np.ndarray,
        deadband: float = 0.0,
        x: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        kt: Optional[float] = None,
        Kt: Optional[np.ndarray] = None,
    ) -> int:

        u = float(u + kt + to_scalar(Kt @ (to_vec(obs) - to_vec(x))))
        u = self._clip_u(u)

        if abs(u) <= deadband:
            return 1
        return 1 if u > 0 else 0