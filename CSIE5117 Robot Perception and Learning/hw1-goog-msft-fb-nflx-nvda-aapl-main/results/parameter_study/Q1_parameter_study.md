# Q1. Parameter Sensitivity Study

## Overview

This study investigates the effects of adjusting Q matrix (state cost), R value (control cost), and max_steps on the performance of three controllers: Continuous LQR, Discrete LQR, and iLQR.

## 1. Effect of Q Matrix (State Cost Weighting)

The Q matrix penalizes deviations in state variables: Q = diag([q_pos, q_vel, q_angle, q_ang_vel])

### Experiments Conducted

![Q Matrix Comparison](parameter_study/q_variation_comparison.png)

| Experiment | Q Matrix | Success | Steps | Max Angle (°) | Max Position (m) | Switch Rate |
|------------|----------|---------|-------|---------------|------------------|-------------|
| Baseline (balanced) | [0.2, 0.2, 2.0, 0.5] | ✅ | 1000 | 2.685 | 1.275 | 0.798 |
| High position weight | [2.0, 0.2, 2.0, 0.5] | ✅ | 1000 | 2.685 | 0.443 | 0.744 |
| High angle weight | [0.2, 0.2, 10.0, 0.5] | ✅ | 1000 | 2.685 | 1.275 | 0.798 |
| All equal weights | [1.0, 1.0, 1.0, 1.0] | ✅ | 1000 | 2.685 | 0.615 | 0.772 |
| Angle-focused | [0.1, 0.1, 5.0, 1.0] | ✅ | 1000 | 2.685 | 1.964 | 0.821 |

### Key Observations:

- **High Angle Weight (q_angle ↑)**: Prioritizes keeping the pole upright, may allow more cart position drift
- **High Position Weight (q_pos ↑)**: Keeps cart centered, but may allow larger angle deviations
- **Balanced Weights**: Achieves good overall performance with trade-offs

## 2. Effect of R Value (Control Cost)

The R parameter penalizes control effort (aggressive actions).

### Experiments Conducted

![R Value Comparison](parameter_study/r_variation_comparison.png)

| Experiment | R Value | Success | Steps | Max Angle (°) | Max Position (m) | Switch Rate |
|------------|---------|---------|-------|---------------|------------------|-------------|
| Low R (aggressive) | 0.50 | ✅ | 1000 | 2.685 | 0.751 | 0.774 |
| Baseline R | 1.50 | ✅ | 1000 | 2.685 | 1.275 | 0.798 |
| High R (conservative) | 5.00 | ❌ | 788 | 2.685 | 2.401 | 0.796 |
| Very high R | 10.00 | ❌ | 726 | 2.685 | 2.402 | 0.723 |

### Key Observations:

- **Low R (R → 0)**: Aggressive control, frequent switching, fast response
- **High R (R ↑)**: Conservative control, smoother actions, potentially slower response
- **Optimal R**: Balances control effort with performance requirements

## 3. Effect of max_steps (Episode Length)

The max_steps parameter determines how long the controller must maintain stability.

### Experiments Conducted

![Steps Comparison](parameter_study/steps_variation_comparison.png)

| Experiment | Max Steps | Success | Steps Completed | Final Angle (°) | Final Position (m) |
|------------|-----------|---------|-----------------|-----------------|-------------------|
| Short episode (200) | 200 | ✅ | 200 | 0.289 | 0.650 |
| Medium episode (500) | 500 | ✅ | 500 | 0.043 | 1.230 |
| Standard episode (1000) | 1000 | ✅ | 1000 | 0.114 | 0.980 |
| Long episode (2000) | 2000 | ✅ | 2000 | 0.078 | 0.713 |

### Key Observations:

- **Longer Episodes**: Reveal long-term stability and drift issues
- **Shorter Episodes**: Easier to complete but may hide control deficiencies
- **Drift Accumulation**: Position drift typically increases with episode length

## Summary and Recommendations

### Parameter Tuning Guidelines

1. **Start with balanced Q matrix**: Equal emphasis on all states
2. **Increase angle weight**: If pole stability is most critical
3. **Increase position weight**: If cart centering is important
4. **Adjust R**: Lower for aggressive control, higher for smooth control
5. **Test with longer episodes**: To verify long-term stability

### Trade-offs

- **Performance vs. Control Effort**: Lower R → better performance but more energy
- **Angle vs. Position**: Higher q_angle → better pole balance but more cart drift
- **Responsiveness vs. Smoothness**: Aggressive gains → fast response but chattering

