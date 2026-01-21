"""
Custom implementations of CARE and DARE solvers.

This module provides custom solvers for:
- CARE: Continuous-time Algebraic Riccati Equation
- DARE: Discrete-time Algebraic Riccati Equation

Both use the Schur decomposition method.
"""

import numpy as np
from scipy.linalg import schur, eigvals


def solve_care_custom(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation (CARE):
        A^T P + P A - P B R^{-1} B^T P + Q = 0
    
    Uses the Hamiltonian matrix method with Schur decomposition.
    
    Parameters
    ----------
    A : np.ndarray
        State matrix (n x n)
    B : np.ndarray
        Input matrix (n x m)
    Q : np.ndarray
        State cost matrix (n x n), must be positive semi-definite
    R : np.ndarray
        Control cost matrix (m x m), must be positive definite
    
    Returns
    -------
    P : np.ndarray
        Solution to CARE (n x n), positive semi-definite
    
    Algorithm
    ---------
    1. Form the Hamiltonian matrix:
       H = [ A          -B R^{-1} B^T ]
           [-Q          -A^T          ]
    
    2. Compute the ordered Schur decomposition of H
    3. Extract the stable invariant subspace
    4. Compute P from the subspace basis
    """
    n = A.shape[0]
    
    # Compute R^{-1} B^T
    R_inv = np.linalg.inv(R)
    BR_invBT = B @ R_inv @ B.T
    
    # Form the Hamiltonian matrix (2n x 2n)
    H = np.block([
        [A, -BR_invBT],
        [-Q, -A.T]
    ])
    
    # Compute Schur decomposition: H = U T U^T
    # We need the ordered Schur form with stable eigenvalues first
    T, U, sdim = schur(H, output='real', sort='lhp')
    
    # Extract the upper-left and lower-left blocks of U
    # corresponding to the stable invariant subspace
    U11 = U[:n, :n]
    U21 = U[n:, :n]
    
    # Solve for P: P U11 = U21
    # Since U11 should be invertible for the stable subspace
    P = U21 @ np.linalg.inv(U11)
    
    # Symmetrize to account for numerical errors
    P = (P + P.T) / 2
    
    return P


def solve_dare_custom(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solve the discrete-time algebraic Riccati equation (DARE):
        A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0
    
    Uses the symplectic matrix method with Schur decomposition.
    
    Parameters
    ----------
    A : np.ndarray
        State transition matrix (n x n)
    B : np.ndarray
        Input matrix (n x m)
    Q : np.ndarray
        State cost matrix (n x n), must be positive semi-definite
    R : np.ndarray
        Control cost matrix (m x m), must be positive definite
    
    Returns
    -------
    P : np.ndarray
        Solution to DARE (n x n), positive semi-definite
    
    Algorithm
    ---------
    1. Form the symplectic matrix:
       M = [ A + B R^{-1} B^T (A^{-1})^T Q    -B R^{-1} B^T (A^{-1})^T ]
           [        -(A^{-1})^T Q                    (A^{-1})^T        ]
    
    2. Alternatively, use the equivalent formulation:
       M = [ A^{-1}                          A^{-1} B (R + B^T P B)^{-1} B^T ]
           [ Q A^{-1}        A^T + Q A^{-1} B (R + B^T P B)^{-1} B^T      ]
    
    3. We use a simplified approach via the discrete Hamiltonian:
       G = [ A^{-1}              A^{-1} B R^{-1} B^T ]
           [ Q A^{-1}            A^T + Q A^{-1} B R^{-1} B^T ]
    """
    n = A.shape[0]
    
    # Compute inverses and products
    A_inv = np.linalg.inv(A)
    R_inv = np.linalg.inv(R)
    BR_invBT = B @ R_inv @ B.T
    
    # Form the symplectic/Hamiltonian matrix for DARE
    # Using formulation from Laub's method
    G = np.block([
        [A + BR_invBT @ A_inv.T @ Q, -BR_invBT @ A_inv.T],
        [-A_inv.T @ Q, A_inv.T]
    ])
    
    # Compute ordered Schur decomposition
    # Sort eigenvalues inside the unit circle (stable for discrete-time)
    # Use a custom sort function for discrete-time stability
    def sort_disc_stable(x):
        """Return True if eigenvalue is stable (inside unit circle)."""
        return abs(x) < 1.0
    
    T, U, sdim = schur(G, output='real', sort=sort_disc_stable)
    
    # Select eigenvalues inside unit circle
    # sdim should equal n for a valid solution
    if sdim != n:
        import warnings
        warnings.warn(f"Expected {n} stable eigenvalues, got {sdim}. Solution may be inaccurate.")
    
    # Extract the stable invariant subspace
    U11 = U[:n, :n]
    U21 = U[n:, :n]
    
    # Solve for P: P = U21 @ inv(U11)
    P = U21 @ np.linalg.inv(U11)
    
    # Symmetrize
    P = (P + P.T) / 2
    
    return P


def verify_care_solution(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> float:
    """
    Verify the CARE solution by computing the residual:
        residual = A^T P + P A - P B R^{-1} B^T P + Q
    
    Returns the Frobenius norm of the residual.
    """
    R_inv = np.linalg.inv(R)
    residual = A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    return np.linalg.norm(residual, 'fro')


def verify_dare_solution(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray) -> float:
    """
    Verify the DARE solution by computing the residual:
        residual = A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    
    Returns the Frobenius norm of the residual.
    """
    S = R + B.T @ P @ B
    S_inv = np.linalg.inv(S)
    residual = A.T @ P @ A - P - A.T @ P @ B @ S_inv @ B.T @ P @ A + Q
    return np.linalg.norm(residual, 'fro')