import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from src.spline_utils import get_bspline_matrix
from src.model import build_credit_model

# 1. Malaysian Market Data (AAA Corporate Spreads)
tenors = np.array([1, 2, 3, 5, 7, 10])
spreads = np.array([85, 102, 120, 155, 180, 210]) 

# 2. Setup High-Resolution Basis
# Using more knots or degree=3 (Cubic) for maximum smoothness
knots = np.array([1, 3, 5, 7, 10])
X_market = get_bspline_matrix(tenors, knots, degree=3)

# 3. Execution: Bayesian MCMC Sampling
print("🚀 Initializing Advanced Bayesian Inference...")
model = build_credit_model(X_market, spreads)

with model:
    # We use more draws for higher precision in the Forward Curve derivatives
    trace = pm.sample(
        draws=2000, 
        tune=1000, 
        target_accept=0.95, 
        return_inferencedata=True,
        random_seed=42
    )

# 4. Generate Predictions & Forward Rates
x_plot = np.linspace(1, 10, 200)
dt = 0.01 # Small step for numerical differentiation

# Basis matrices for Spot and 'Forward' logic
X_plot = get_bspline_matrix(x_plot, knots, degree=3)
X_plot_dt = get_bspline_matrix(x_plot + dt, knots, degree=3)

# Extract posterior coefficients
post_beta = trace.posterior["beta"].stack(sample=("chain", "draw")).values

# Calculate Spot Curves
spot_curves = np.dot(X_plot, post_beta)

# Calculate Instantaneous Forward Spreads: f(t) = s(t) + t * s'(t)
spot_curves_dt = np.dot(X_plot_dt, post_beta)
deriv = (spot_curves_dt - spot_curves) / dt
fwd_curves = spot_curves + (x_plot[:, None] * deriv)

# 5. Professional Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left Plot: Spot Spread with Uncertainty
az.plot_hdi(x_plot, spot_curves.T, ax=ax1, hdi_prob=0.95, fill_kwargs={"alpha": 0.3, "label": "95% CI"})
ax1.plot(x_plot, spot_curves.mean(axis=1), color='navy', lw=2, label="Bayesian Mean (Spot)")
ax1.scatter(tenors, spreads, color='red', zorder=5, label="MY Market Quotes")
ax1.set_title("Malaysian AAA Spot Spread Curve")
ax1.set_ylabel("Spread (bps)")

# Right Plot: Implied Forward Curve
# This shows the 'marginal cost' of credit - very sensitive to model quality!
az.plot_hdi(x_plot, fwd_curves.T, ax=ax2, hdi_prob=0.95, fill_kwargs={"color": "orange", "alpha": 0.3})
ax2.plot(x_plot, fwd_curves.mean(axis=1), color='darkorange', lw=2, label="Implied Forward")
ax2.set_title("Implied Forward Spread (Marginal Cost)")
ax2.set_ylabel("Instantaneous Spread (bps)")

for ax in [ax1, ax2]:
    ax.legend()
    ax.set_xlabel("Tenor (Years)")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Quantitative Convergence Check
print("\n--- Model Diagnostics ---")
print(az.summary(trace, var_names=["tau", "sigma_obs"]))