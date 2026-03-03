import numpy as np
import matplotlib.pyplot as plt
from src.model import build_credit_model
import pymc as pm

# 1. Baseline Data (from Part 2)
tenors = np.array([1, 2, 3, 5, 7, 10])
base_spreads = np.array([85, 102, 120, 155, 180, 210])

# 2. Shocked Data: Imagine the 5Y tenor spreads blow out by 100bps
shocked_spreads = base_spreads.copy()
shocked_spreads[3] += 100  # The 5Y point is now an outlier

# 3. Run the model on the shocked data
# (Note: Use the same X_market and knots from before)
shock_model = build_credit_model(X_market, shocked_spreads)

with shock_model:
    shock_trace = pm.sample(1000, tune=1000, target_accept=0.9)