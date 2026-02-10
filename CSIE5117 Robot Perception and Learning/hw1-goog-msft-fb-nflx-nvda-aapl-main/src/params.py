from typing import Optional, List
import numpy as np


class ControlParams:
    def __init__(
        self,
        g: float = 9.8,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        Q: Optional[List[float]] = None,
        Qf: Optional[List[float]] = None,
        R: Optional[List[float]] = None,
        *args,
        **kwargs,
    ):
        M = m_c + m_p
        D = l * (4.0 / 3.0 - m_p / M)
        alpha = (m_p / M) / (4.0 / 3.0 - m_p / M)
        
        # ==================== MODIFIED: TODO COMPLETED ====================
        # Linearized dynamics: x_dot = A*x + B*u
        # State: x = [x, x_dot, theta, theta_dot]^T
        # Linearization around theta ~ 0: sin(theta) ~ theta, cos(theta) ~ 1
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, -(m_p * g) / (M * D / l), 0],
            [0, 0, 0, 1],
            [0, 0, g * (M / m_p) * alpha / l, 0]
        ])
        
        self.B = np.array([
            [0],
            [1 / (M * D / l)],
            [0],
            [-1 / (M * D)]
        ])
        # ==================== END OF MODIFICATION ====================
        
        self.Q = np.diag(Q) if Q is not None else None
        self.Qf = np.diag(Qf) if Qf is not None else None
        self.R = np.array([R]) if R is not None else None