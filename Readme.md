🛡️ Institutional Bayesian Credit Framework
Quantifying Uncertainty in Illiquid Term Structures

📌 Overview
This repository implements a Hierarchical Bayesian Factor Model for constructing credit spread curves. Unlike deterministic interpolation (which assumes market quotes are perfect), this framework treats market data as noisy observations and utilizes Cubic B-Splines to derive a smooth, economically consistent term structure.

This project was developed under the mentorship of a Quantitative Researcher at Bank of Montreal (BMO) to explore how probabilistic methods can stabilize credit curves in emerging or illiquid markets (specifically focusing on Malaysian AAA Corporate Spreads).

🏗️ Core Architecture
1. The Mathematical Engine (src/spline_utils.py)

To ensure the Forward Curve (the marginal cost of credit) is smooth and non-oscillatory, I implemented:

Cubic B-Splines (C 
2
  continuity): Ensuring the second derivative is continuous across all tenors.

Clamped Boundary Conditions: Utilizing padded knots to force the model to respect terminal market pillars.

Vectorized Design Matrix: High-performance construction of the basis functions using scipy.interpolate.

2. The Probabilistic Model (src/model.py)

Built using PyMC, the model moves away from "Best Fit" lines toward "Distributional Truth":

Stochastic Factor Drivers: Incorporates VIX and Nasdaq returns as latent priors to drive level and slope shifts.

MCMC Sampling: Uses the No-U-Turn Sampler (NUTS) to generate thousands of possible curve realizations, forming a "Credible Interval" (Uncertainty Cloud).

3. Validation & Stress Testing (03_validation.ipynb)

Out-of-Sample Censoring: The model is tested by "hiding" the 5-year tenor and evaluating its ability to reconstruct the missing data point based on the global shape of the curve.

Stress Engine: A Streamlit dashboard that simulates "Black Swan" events (e.g., VIX spikes) to observe the non-linear reaction of the credit curve.

🚀 Key Features
Bayesian Mean Curve: Robust against outliers and market noise.

Expected Shortfall (95%): Real-time metric for tail risk in the term structure.

Synthetic Data Generation: Includes a simulator for Geometric Brownian Motion (GBM) and Ornstein-Uhlenbeck (OU) processes to stress-test the pipeline.

🛠️ Tech Stack
Language: Python

Probabilistic Logic: PyMC, PyTensor

Numerical Analysis: NumPy, SciPy, Pandas

Visualization: ArviZ, Matplotlib, Streamlit

🧬 Academic Context
This project serves as a practical exploration of Numerical Analysis and Stochastic Processes, aligning with the rigorous mathematical standards of the French Classe Internationale (Science Track) at Université Grenoble Alpes.