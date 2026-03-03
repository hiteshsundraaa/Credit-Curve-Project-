# 🏛️ Adversarial Bayesian Credit Lab
**A Multi-Asset Term Structure Framework for Systemic Risk Analysis**

This project implements a **Bayesian P-Spline** model to estimate credit term structures, integrated with a **stochastic market engine** simulating Nasdaq-100 (GBM) and VIX (OU) dynamics.

## 🔬 Core Methodology
- **Hamiltonian Monte Carlo (HMC)**: Utilized for high-dimensional posterior inference of spline coefficients.
- **Wei et al. (2025) Guidance**: Implementation of adversarial stress-testing to identify non-linear credit regimes.
- **Stochastic Data Engine**: Cross-asset simulation of Equity-Credit contagion via correlated Geometric Brownian Motion and Ornstein-Uhlenbeck processes.

## 🛠️ Architecture
- `src/model.py`: PyMC implementation of the robust Student-T Likelihood spline model.
- `src/data_generator.py`: Stochastic simulation of Nasdaq, VIX, and Spread history.
- `app.py`: Streamlit-based institutional risk dashboard.

## 📊 Key Insights
- **Level vs. Slope Shifts**: Quantifies parallel and non-parallel curve deformations.
- **Tail Risk**: Derivation of 95% Credible Intervals and Expected Shortfall (ES).