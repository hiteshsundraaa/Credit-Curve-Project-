"""
Walk-Forward Backtesting Engine
================================
Implements expanding-window walk-forward validation for the Bayesian P-Spline
credit curve model. At each step t, the model is trained on data[0:t] and
evaluated on data[t:t+h] — no lookahead bias.

Metrics produced:
  - Curve RMSE per tenor bucket
  - PnL from a simple carry + roll-down strategy
  - Hit ratio (directional accuracy of spread change forecasts)
  - Calibration: % of realizations inside 95% CI
"""

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# 1.  Nelson-Siegel-Svensson benchmark
# ---------------------------------------------------------------------------

def nss_curve(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    """Nelson-Siegel-Svensson parametric credit curve."""
    t1 = tau / lambda1
    t2 = tau / lambda2
    term1 = beta1 * (1 - np.exp(-t1)) / t1
    term2 = beta2 * ((1 - np.exp(-t1)) / t1 - np.exp(-t1))
    term3 = beta3 * ((1 - np.exp(-t2)) / t2 - np.exp(-t2))
    return beta0 + term1 + term2 + term3


def fit_nss(tenors, spreads):
    """Fit NSS curve; returns fitted spread array or None on failure."""
    try:
        p0 = [spreads.mean(), -20, 10, 10, 2.0, 5.0]
        bounds = ([-np.inf]*4 + [0.1, 0.1], [np.inf]*4 + [30, 30])
        popt, _ = curve_fit(nss_curve, tenors, spreads, p0=p0,
                            bounds=bounds, maxfev=10000)
        return nss_curve(tenors, *popt), popt
    except RuntimeError:
        return None, None


# ---------------------------------------------------------------------------
# 2.  Lightweight spline fitting (no MCMC — fast for backtesting)
# ---------------------------------------------------------------------------

def get_bspline_matrix(x, knots, degree=3):
    knots_padded = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    n_basis = len(knots) + degree - 1
    X = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        X[:, i] = BSpline(knots_padded, coeffs, degree)(x)
    return X


def fit_spline_ols(tenors, spreads, knots):
    """Fast OLS spline fit — used for walk-forward speed."""
    X = get_bspline_matrix(tenors, knots, degree=3)
    beta, _, _, _ = np.linalg.lstsq(X, spreads, rcond=None)
    return beta


def predict_spline(x_new, beta, knots):
    X = get_bspline_matrix(x_new, knots, degree=3)
    return X @ beta


# ---------------------------------------------------------------------------
# 3.  Synthetic multi-date spread history generator
# ---------------------------------------------------------------------------

def generate_spread_history(n_days=504, seed=42):
    """
    Simulates daily cross-sectional credit spread quotes for 6 tenors.
    Uses a 3-factor (level, slope, curvature) driving process with
    VIX-regime-dependent volatility.

    Returns
    -------
    pd.DataFrame  columns: Date, Tenor_1Y … Tenor_10Y, VIX, Regime
    """
    rng = np.random.default_rng(seed)
    tenors = np.array([1, 2, 3, 5, 7, 10])

    # Factor dynamics
    level   = np.zeros(n_days);  level[0]  = 130
    slope   = np.zeros(n_days);  slope[0]  = -40
    curve   = np.zeros(n_days);  curve[0]  = 20
    vix     = np.zeros(n_days);  vix[0]    = 18

    # VIX OU process
    kappa, theta_v, sigma_v = 0.15, 20, 3.5

    # Regime: 0=normal, 1=stressed  (Markov)
    P_normal_to_stress = 0.02
    P_stress_to_normal = 0.10
    regime = np.zeros(n_days, dtype=int)

    for t in range(1, n_days):
        if regime[t-1] == 0:
            regime[t] = int(rng.random() < P_normal_to_stress)
        else:
            regime[t] = int(rng.random() > P_stress_to_normal)

        vol_mult = 3.0 if regime[t] == 1 else 1.0

        vix[t] = (vix[t-1]
                  + kappa * (theta_v - vix[t-1])
                  + sigma_v * vol_mult * rng.standard_normal())
        vix[t] = np.clip(vix[t], 10, 80)

        vix_shock = (vix[t] - 20) / 15

        level[t]  = level[t-1]  + 0.3 * vix_shock + 0.8 * rng.standard_normal() * vol_mult
        slope[t]  = slope[t-1]  - 0.1 * vix_shock + 0.5 * rng.standard_normal() * vol_mult
        curve[t]  = curve[t-1]  + 0.0             + 0.3 * rng.standard_normal() * vol_mult

        # Mean reversion
        level[t]  += -0.05 * (level[t-1] - 130)
        slope[t]  += -0.08 * (slope[t-1] + 40)
        curve[t]  += -0.10 * (curve[t-1] - 20)

    # Nelson-Siegel factor loadings → cross-sectional spreads
    t1 = tenors / 2.0
    L1 = (1 - np.exp(-t1)) / t1
    L2 = L1 - np.exp(-t1)
    L3 = (1 - np.exp(-tenors / 5.0)) / (tenors / 5.0) - np.exp(-tenors / 5.0)

    spreads = (level[:, None]
               + slope[:, None] * L1[None, :]
               + curve[:, None] * L2[None, :]
               + 5 * L3[None, :]
               + rng.normal(0, 2, (n_days, len(tenors))))
    spreads = np.clip(spreads, 20, 600)

    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    df = pd.DataFrame(spreads, columns=[f"Tenor_{int(t)}Y" for t in tenors])
    df.insert(0, "Date", dates)
    df["VIX"]    = vix
    df["Regime"] = regime
    return df, tenors


# ---------------------------------------------------------------------------
# 4.  Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward_backtest(df, tenors, knots, min_train=60, step=5, horizon=5):
    """
    Expanding-window walk-forward backtest.

    Parameters
    ----------
    df        : DataFrame from generate_spread_history
    tenors    : array of tenor values (years)
    knots     : spline knot vector
    min_train : minimum observations before first test window
    step      : re-estimation frequency (trading days)
    horizon   : forecast horizon (trading days)

    Returns
    -------
    results : dict with per-step metrics
    """
    spread_cols = [f"Tenor_{int(t)}Y" for t in tenors]
    n = len(df)

    records = []
    nss_records = []

    for t in range(min_train, n - horizon, step):
        train_df = df.iloc[:t]
        test_df  = df.iloc[t: t + horizon]

        train_spreads = train_df[spread_cols].values  # (t, 6)
        test_spreads  = test_df[spread_cols].values   # (horizon, 6)

        # ---- Fit on last available cross-section (most recent day) --------
        y_train = train_spreads[-1]
        beta_spline = fit_spline_ols(tenors, y_train, knots)

        # ---- NSS benchmark ------------------------------------------------
        nss_fit, nss_params = fit_nss(tenors, y_train)

        # ---- Forecast: naive carry (no change) vs. model interpolation ----
        # P-Spline prediction at observed tenors
        spline_pred = predict_spline(tenors, beta_spline, knots)

        # ---- Evaluate on each day in the horizon --------------------------
        for h in range(horizon):
            y_actual = test_spreads[h]

            spline_rmse = np.sqrt(np.mean((spline_pred - y_actual) ** 2))
            naive_rmse  = np.sqrt(np.mean((y_train - y_actual) ** 2))

            if nss_fit is not None:
                nss_rmse = np.sqrt(np.mean((nss_fit - y_actual) ** 2))
            else:
                nss_rmse = np.nan

            # Directional accuracy (did we get the sign of spread change right?)
            direction_actual = np.sign(y_actual - y_train)
            direction_pred   = np.sign(spline_pred - y_train)
            hit_ratio = np.mean(direction_actual == direction_pred)

            # Simple carry-roll PnL:
            # Long 10Y spread → profit if spread widens, loss if tightens
            # (approximation: DV01 = 1 per bp for simplicity)
            pnl_long_10y = y_actual[-1] - y_train[-1]   # realised move in 10Y
            pnl_long_5y  = y_actual[3]  - y_train[3]    # 5Y

            records.append({
                "Date"        : test_df.index[h] if hasattr(test_df.index[h], 'date') else test_df.iloc[h]["Date"],
                "Train_End"   : train_df.iloc[-1]["Date"],
                "Horizon_Day" : h + 1,
                "Spline_RMSE" : spline_rmse,
                "NSS_RMSE"    : nss_rmse,
                "Naive_RMSE"  : naive_rmse,
                "Hit_Ratio"   : hit_ratio,
                "PnL_10Y"     : pnl_long_10y,
                "PnL_5Y"      : pnl_long_5y,
                "VIX"         : train_df.iloc[-1]["VIX"],
                "Regime"      : train_df.iloc[-1]["Regime"],
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5.  Regime detection  (2-state Gaussian HMM via Baum-Welch)
# ---------------------------------------------------------------------------

class GaussianHMM2State:
    """
    Minimal 2-state Gaussian HMM fitted with Baum-Welch (EM).
    State 0 = low-vol (normal), State 1 = high-vol (stressed).
    """

    def __init__(self, n_iter=100):
        self.n_iter = n_iter

    def fit(self, X):
        """X: 1-D array of observations (e.g., VIX or spread changes)."""
        X = np.asarray(X, dtype=float)
        T = len(X)

        # Init
        self.A  = np.array([[0.95, 0.05], [0.10, 0.90]])   # transition
        self.pi = np.array([0.8, 0.2])                       # initial
        idx = X.argsort()
        self.mu    = np.array([X[idx[:T//2]].mean(), X[idx[T//2:]].mean()])
        self.sigma = np.array([X[idx[:T//2]].std() + 1e-3,
                               X[idx[T//2:]].std() + 1e-3])

        for _ in range(self.n_iter):
            # E-step
            alpha, beta_, gamma, xi = self._e_step(X)
            # M-step
            self._m_step(X, gamma, xi)

        self.state_sequence_ = np.argmax(
            self._e_step(X)[2], axis=1)
        return self

    def _emission(self, x):
        """Returns (T, 2) emission probabilities."""
        from scipy.stats import norm
        return np.column_stack([
            norm.pdf(x, self.mu[k], self.sigma[k]) + 1e-300
            for k in range(2)
        ])

    def _e_step(self, X):
        T = len(X)
        B = self._emission(X)

        # Forward
        alpha = np.zeros((T, 2))
        alpha[0] = self.pi * B[0]
        alpha[0] /= alpha[0].sum()
        scale = np.zeros(T)
        scale[0] = 1.0
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            s = alpha[t].sum()
            scale[t] = s if s > 0 else 1e-300
            alpha[t] /= scale[t]

        # Backward
        beta_ = np.zeros((T, 2))
        beta_[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta_[t] = (self.A * B[t+1] * beta_[t+1]).sum(axis=1)
            beta_[t] /= scale[t+1]

        gamma = alpha * beta_
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T-1, 2, 2))
        for t in range(T-1):
            xi[t] = (alpha[t, :, None] * self.A
                     * B[t+1] * beta_[t+1])
            xi[t] /= xi[t].sum()

        return alpha, beta_, gamma, xi

    def _m_step(self, X, gamma, xi):
        self.pi = gamma[0]
        self.A  = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True)
        for k in range(2):
            w = gamma[:, k]
            self.mu[k]    = (w * X).sum() / w.sum()
            self.sigma[k] = np.sqrt((w * (X - self.mu[k])**2).sum() / w.sum()) + 1e-3

    def predict(self, X):
        return self._e_step(np.asarray(X))[2]   # posterior state probs


# ---------------------------------------------------------------------------
# 6.  Spread factor decomposition
# ---------------------------------------------------------------------------

def decompose_spread_factors(df, tenors):
    """
    Decomposes daily spread changes into:
      - Level factor  (parallel shift = mean change across tenors)
      - Slope factor  (10Y minus 1Y change)
      - Curvature     (2Y + 10Y - 2*5Y change, butterfly)
      - Idiosyncratic (residual after factor projection)

    Returns DataFrame with columns per factor + per-tenor idiosyncratic.
    """
    spread_cols = [f"Tenor_{int(t)}Y" for t in tenors]
    dS = df[spread_cols].diff().dropna()

    level  = dS.mean(axis=1)
    slope  = dS["Tenor_10Y"] - dS["Tenor_1Y"]

    # Curvature: butterfly around 5Y
    if "Tenor_5Y" in dS.columns:
        curve = dS["Tenor_2Y"] + dS["Tenor_10Y"] - 2 * dS["Tenor_5Y"]
    else:
        curve = pd.Series(np.zeros(len(dS)), index=dS.index)

    # Systematic component via OLS projection
    factor_matrix = np.column_stack([
        np.ones(len(dS)),
        level.values,
        slope.values,
        curve.values
    ])

    idio = {}
    for col in spread_cols:
        y = dS[col].values
        beta_f, _, _, _ = np.linalg.lstsq(factor_matrix, y, rcond=None)
        fitted = factor_matrix @ beta_f
        idio[col + "_idio"] = y - fitted

    result = pd.DataFrame({
        "Date"      : df["Date"].iloc[1:].values,
        "Level"     : level.values,
        "Slope"     : slope.values,
        "Curvature" : curve.values,
        "VIX"       : df["VIX"].iloc[1:].values,
        "Regime"    : df["Regime"].iloc[1:].values,
    })
    for k, v in idio.items():
        result[k] = v

    return result
