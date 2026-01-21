# RPL HW1: Optimal Control for CartPole Probelm

This repository contains the implementation for Homework 1 of the Robot Perception and Learning course at National Taiwan University (Fall 2025).

In this assignment, you will learn how to design and optimize controllers for the classic cart–pole balancing task, a fundamental problem in control theory and robotics.
We explore two complementary approaches:
- LQR (Continuous & Discrete): A classic linear optimal control method that stabilizes the cart–pole system around the upright equilibrium.
- iLQR (Iterative LQR): A nonlinear extension of LQR that refines the control policy through iterative optimization with a receding horizon.

![demo](assets/demp.gif)


## Setup
To set up the environment and install all required packages:
```
uv sync
```
> [!TIP]
> Make sure uv is installed (`pip install uv`).


## Run the Linear Quadratic Regulator (LQR)
```
uv run main.py -c <config: cont_lqr, disc_lqr, ilqr>
```
