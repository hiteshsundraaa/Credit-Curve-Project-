import pymc as pm
import pytensor.tensor as pt

def build_credit_model(X_matrix, observed_spreads, tenors, vix_input=0, slope_input=0):
    """
    Advanced Factor Model:
    Spread(t) = B-Spline(t) + (Gamma_1 * VIX) + (Gamma_2 * Slope_Shock * Tenor)
    """
    with pm.Model() as model:
        # 1. THE SPLINE (Baseline Curve)
        tau = pm.Exponential("tau", 1.0)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=X_matrix.shape[1])
        beta = pm.Deterministic("beta", pt.extra_ops.cumsum(beta_raw * tau))
        
        # 2. MULTI-FACTOR DRIVERS
        # Gamma_1: Parallel Shift (Level)
        gamma_vix = pm.Normal("gamma_vix", mu=0, sigma=10.0)
        # Gamma_2: Curve Twist (Slope)
        gamma_slope = pm.Normal("gamma_slope", mu=0, sigma=5.0)
        
        # 3. THE MEAN FUNCTION
        # VIX affects the whole curve. Slope affects tenors linearly (t/max_t)
        level_effect = gamma_vix * vix_input
        slope_effect = gamma_slope * slope_input * (tenors / tenors.max())
        
        mu = pm.math.dot(X_matrix, beta) + level_effect + slope_effect
        
        # 4. LIKELIHOOD
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=5.0)
        pm.StudentT("obs", nu=3, mu=mu, sigma=sigma_obs, observed=observed_spreads)
        
    return model