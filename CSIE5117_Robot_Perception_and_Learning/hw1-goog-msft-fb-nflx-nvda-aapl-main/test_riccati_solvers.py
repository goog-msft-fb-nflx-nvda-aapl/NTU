"""
Test script to verify custom CARE and DARE implementations.

This script:
1. Tests the custom solvers against scipy's implementations
2. Verifies the solutions satisfy the Riccati equations
3. Tests with the actual CartPole parameters
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from src.riccati_solvers import (
    solve_care_custom, 
    solve_dare_custom,
    verify_care_solution,
    verify_dare_solution
)
from src.params import ControlParams


def test_care():
    """Test custom CARE solver against scipy implementation."""
    print("=" * 70)
    print("Testing Custom CARE Solver")
    print("=" * 70)
    
    # Get CartPole parameters
    params = ControlParams(
        Q=[0.2, 0.2, 2.0, 0.5],
        R=[1.5]
    )
    
    A = params.A
    B = params.B
    Q = params.Q
    R = params.R
    
    print("\nSystem dimensions:")
    print(f"  A: {A.shape}, B: {B.shape}, Q: {Q.shape}, R: {R.shape}")
    
    # Solve using scipy
    print("\n[1] Solving CARE using scipy.linalg.solve_continuous_are...")
    P_scipy = solve_continuous_are(A, B, Q, R)
    residual_scipy = verify_care_solution(A, B, Q, R, P_scipy)
    print(f"    Residual norm (scipy): {residual_scipy:.2e}")
    
    # Solve using custom implementation
    print("\n[2] Solving CARE using custom implementation...")
    P_custom = solve_care_custom(A, B, Q, R)
    residual_custom = verify_care_solution(A, B, Q, R, P_custom)
    print(f"    Residual norm (custom): {residual_custom:.2e}")
    
    # Compare solutions
    diff = np.linalg.norm(P_scipy - P_custom, 'fro')
    print(f"\n[3] Difference between solutions: {diff:.2e}")
    
    # Compute gains
    K_scipy = np.linalg.inv(R) @ B.T @ P_scipy
    K_custom = np.linalg.inv(R) @ B.T @ P_custom
    K_diff = np.linalg.norm(K_scipy - K_custom)
    
    print(f"\n[4] LQR Gains:")
    print(f"    K (scipy):  {K_scipy.flatten()}")
    print(f"    K (custom): {K_custom.flatten()}")
    print(f"    Gain difference: {K_diff:.2e}")
    
    # Check eigenvalues of closed-loop system
    A_cl_scipy = A - B @ K_scipy
    A_cl_custom = A - B @ K_custom
    
    eig_scipy = np.linalg.eigvals(A_cl_scipy)
    eig_custom = np.linalg.eigvals(A_cl_custom)
    
    print(f"\n[5] Closed-loop eigenvalues:")
    print(f"    scipy:  {eig_scipy}")
    print(f"    custom: {eig_custom}")
    print(f"    All stable (scipy):  {np.all(np.real(eig_scipy) < 0)}")
    print(f"    All stable (custom): {np.all(np.real(eig_custom) < 0)}")
    
    success = diff < 1e-6 and residual_custom < 1e-9
    print(f"\n{'✓ CARE TEST PASSED' if success else '✗ CARE TEST FAILED'}")
    print("=" * 70)
    
    return success


def test_dare():
    """Test custom DARE solver against scipy implementation."""
    print("\n" + "=" * 70)
    print("Testing Custom DARE Solver")
    print("=" * 70)
    
    # Get CartPole parameters
    params = ControlParams(
        Q=[0.2, 0.2, 2.0, 0.5],
        R=[1.5]
    )
    
    A = params.A
    B = params.B
    Q = params.Q
    R = params.R
    dt = 0.02
    
    # Discretize the system
    from scipy.linalg import expm
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    
    print("\nDiscrete system dimensions:")
    print(f"  Ad: {Ad.shape}, Bd: {Bd.shape}, Q: {Q.shape}, R: {R.shape}")
    print(f"  Sampling time: {dt} s")
    
    # Solve using scipy
    print("\n[1] Solving DARE using scipy.linalg.solve_discrete_are...")
    P_scipy = solve_discrete_are(Ad, Bd, Q, R)
    residual_scipy = verify_dare_solution(Ad, Bd, Q, R, P_scipy)
    print(f"    Residual norm (scipy): {residual_scipy:.2e}")
    
    # Solve using custom implementation
    print("\n[2] Solving DARE using custom implementation...")
    P_custom = solve_dare_custom(Ad, Bd, Q, R)
    residual_custom = verify_dare_solution(Ad, Bd, Q, R, P_custom)
    print(f"    Residual norm (custom): {residual_custom:.2e}")
    
    # Compare solutions
    diff = np.linalg.norm(P_scipy - P_custom, 'fro')
    print(f"\n[3] Difference between solutions: {diff:.2e}")
    
    # Compute gains
    K_scipy = np.linalg.inv(R + Bd.T @ P_scipy @ Bd) @ (Bd.T @ P_scipy @ Ad)
    K_custom = np.linalg.inv(R + Bd.T @ P_custom @ Bd) @ (Bd.T @ P_custom @ Ad)
    K_diff = np.linalg.norm(K_scipy - K_custom)
    
    print(f"\n[4] LQR Gains:")
    print(f"    K (scipy):  {K_scipy.flatten()}")
    print(f"    K (custom): {K_custom.flatten()}")
    print(f"    Gain difference: {K_diff:.2e}")
    
    # Check eigenvalues of closed-loop system
    A_cl_scipy = Ad - Bd @ K_scipy
    A_cl_custom = Ad - Bd @ K_custom
    
    eig_scipy = np.linalg.eigvals(A_cl_scipy)
    eig_custom = np.linalg.eigvals(A_cl_custom)
    
    print(f"\n[5] Closed-loop eigenvalues:")
    print(f"    scipy:  {eig_scipy}")
    print(f"    custom: {eig_custom}")
    print(f"    All stable (scipy):  {np.all(np.abs(eig_scipy) < 1)}")
    print(f"    All stable (custom): {np.all(np.abs(eig_custom) < 1)}")
    
    success = diff < 1e-6 and residual_custom < 1e-9
    print(f"\n{'✓ DARE TEST PASSED' if success else '✗ DARE TEST FAILED'}")
    print("=" * 70)
    
    return success


def test_simple_example():
    """Test with a simple 2x2 system."""
    print("\n" + "=" * 70)
    print("Testing with Simple 2x2 System")
    print("=" * 70)
    
    # Simple unstable system
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    
    print("\nSystem:")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"Q =\n{Q}")
    print(f"R =\n{R}")
    
    # CARE
    print("\n--- CARE ---")
    P_scipy_c = solve_continuous_are(A, B, Q, R)
    P_custom_c = solve_care_custom(A, B, Q, R)
    
    res_scipy_c = verify_care_solution(A, B, Q, R, P_scipy_c)
    res_custom_c = verify_care_solution(A, B, Q, R, P_custom_c)
    
    print(f"Residual (scipy): {res_scipy_c:.2e}")
    print(f"Residual (custom): {res_custom_c:.2e}")
    print(f"Difference: {np.linalg.norm(P_scipy_c - P_custom_c):.2e}")
    
    # DARE (discretize first)
    dt = 0.1
    from scipy.linalg import expm
    M = np.block([[A, B], [np.zeros((1, 2)), np.zeros((1, 1))]])
    Md = expm(M * dt)
    Ad = Md[:2, :2]
    Bd = Md[:2, 2:]
    
    print("\n--- DARE ---")
    P_scipy_d = solve_discrete_are(Ad, Bd, Q, R)
    P_custom_d = solve_dare_custom(Ad, Bd, Q, R)
    
    res_scipy_d = verify_dare_solution(Ad, Bd, Q, R, P_scipy_d)
    res_custom_d = verify_dare_solution(Ad, Bd, Q, R, P_custom_d)
    
    print(f"Residual (scipy): {res_scipy_d:.2e}")
    print(f"Residual (custom): {res_custom_d:.2e}")
    print(f"Difference: {np.linalg.norm(P_scipy_d - P_custom_d):.2e}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "RICCATI EQUATION SOLVER TESTS" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Run tests
    test_simple_example()
    care_pass = test_care()
    dare_pass = test_dare()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"CARE Test: {'✓ PASSED' if care_pass else '✗ FAILED'}")
    print(f"DARE Test: {'✓ PASSED' if dare_pass else '✗ FAILED'}")
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if care_pass and dare_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 70)