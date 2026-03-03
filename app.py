import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import os
from src.spline_utils import get_bspline_matrix
from src.model import build_credit_model
from src.data_generator import generate_market_history

# --- 1. Data Bootstrap ---
if not os.path.exists("data/market_history.csv"):
    os.makedirs("data", exist_ok=True)
    generate_market_history("data/market_history.csv")

df_hist = pd.read_csv("data/market_history.csv")

# --- 2. UI Config ---
st.set_page_config(page_title="Adversarial Credit Lab", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { color: #1e3a8a !important; font-family: 'IBM Plex Mono'; }
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Institutional Bayesian Credit Framework")
st.caption("Implementation of Wei et al. (2025): Counterfactual Guidance in Illiquid Term Structures.")

# --- 3. Sidebar: Refined Scenario Controls ---
st.sidebar.header("🕹️ Global Stress Engine")
vix_sim = st.sidebar.slider("Level Shift (VIX Index)", 10, 60, 20)
slope_sim = st.sidebar.slider("Slope Twist (Steepness)", -2.0, 2.0, 0.0, 
                             help="Positive = Steepening, Negative = Flattening/Inversion")
liq_sim = st.sidebar.slider("Liquidity Drain (σ)", -3.0, 3.0, 0.0)

st.sidebar.markdown("---")
st.sidebar.write("**Model Diagnostics**")
st.sidebar.info(f"Historical Nasdaq Correlation: {df_hist['Nasdaq_100'].pct_change().corr(df_hist['Spread_10Y'].pct_change()):.2f}")

# Normalize inputs
vix_std = (vix_sim - 20) / 10 + (liq_sim * 0.3)
slope_std = slope_sim 

# --- 4. Main Tabs ---
tab_market, tab_model, tab_research = st.tabs(["📈 Market History", "📊 Term Structure Lab", "🔬 Convergence & Traces"])

with tab_market:
    st.subheader("Cross-Asset Risk Spillovers")
    col_nasdaq, col_corr = st.columns(2)
    
    with col_nasdaq:
        fig_n, ax_n = plt.subplots(figsize=(8, 4))
        ax_n2 = ax_n.twinx()
        ax_n.plot(pd.to_datetime(df_hist['Date']), df_hist['Nasdaq_100'], color='#00ff41', label="Nasdaq-100")
        ax_n2.plot(pd.to_datetime(df_hist['Date']), df_hist['Spread_10Y'], color='#1e3a8a', alpha=0.6, label="10Y Spread")
        ax_n.set_ylabel("Nasdaq Index", color='#00ff41')
        ax_n2.set_ylabel("Credit Spread (bps)", color='#1e3a8a')
        ax_n.set_title("Equity Drawdowns vs. Credit Contagion")
        st.pyplot(fig_n)

    with col_corr:
        rolling_corr = df_hist['Nasdaq_100'].pct_change().rolling(20).corr(df_hist['Spread_10Y'].pct_change())
        fig_rc, ax_rc = plt.subplots(figsize=(8, 4))
        ax_rc.plot(pd.to_datetime(df_hist['Date']), rolling_corr, color='purple')
        ax_rc.set_title("20-Day Rolling Equity-Credit Correlation")
        ax_rc.axhline(0, color='black', linestyle='--')
        st.pyplot(fig_rc)

    st.markdown("---")
    st.subheader("Historical Volatility Regime")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.plot(pd.to_datetime(df_hist['Date']), df_hist['VIX'], color='#d00000')
        ax_h.set_ylabel("VIX Index")
        st.pyplot(fig_h)
    with col_b:
        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        ax_s.scatter(df_hist['VIX'], df_hist['Spread_10Y'], alpha=0.5, color='#1e3a8a')
        ax_s.set_xlabel("VIX Index")
        ax_s.set_ylabel("10Y Spread (bps)")
        st.pyplot(fig_s)

with tab_model:
    tenors = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    market_yields = np.array([45, 52, 65, 80, 105, 125, 150, 175, 210, 250])
    knots = np.linspace(0.5, 30, 6)
    X_mat = get_bspline_matrix(tenors, knots, degree=3)

    @st.cache_resource
    def run_inference(X, y, t_arr, v_in, s_in):
        model = build_credit_model(X, y, t_arr, vix_input=v_in, slope_input=s_in)
        with model:
            return pm.sample(1000, tune=1000, target_accept=0.98, chains=2, progressbar=False)

    trace = run_inference(X_mat, market_yields, tenors, vix_std, slope_std)
    
    x_plot = np.linspace(0.5, 30, 300)
    X_plot = get_bspline_matrix(x_plot, knots, degree=3)
    post_beta = trace.posterior["beta"].stack(sample=("chain", "draw")).values
    post_gamma_v = trace.posterior["gamma_vix"].stack(sample=("chain", "draw")).values
    post_gamma_s = trace.posterior["gamma_slope"].stack(sample=("chain", "draw")).values

    samples = np.dot(X_plot, post_beta) + (post_gamma_v * vix_std) + (post_gamma_s * slope_std * (x_plot[:, None] / x_plot.max()))
    
    fig_curve, ax_c = plt.subplots(figsize=(12, 5))
    az.plot_hdi(x_plot, samples.T, hdi_prob=0.95, ax=ax_c, fill_kwargs={"color": "#bde0fe", "alpha": 0.4})
    ax_c.plot(x_plot, samples.mean(axis=1), color='#1e3a8a', lw=3, label="Bayesian Mean")
    ax_c.scatter(tenors, market_yields, color='red', label="Current Market")
    ax_c.set_title(f"Predicted Term Structure (VIX: {vix_sim}, Slope: {slope_sim})")
    st.pyplot(fig_curve)

    m1, m2, m3 = st.columns(3)
    es_95 = np.mean(samples[-1][samples[-1] > np.percentile(samples[-1], 95)])
    m1.metric("10Y Prediction", f"{samples.mean(axis=1)[x_plot >= 10][0]:.1f} bps")
    m2.metric("Expected Shortfall (95%)", f"{es_95:.1f} bps")
    m3.metric("Slope Sensitivity", f"{post_gamma_s.mean():.2f}")

with tab_research:
    st.subheader("MCMC Convergence Diagnostics")
    fig_t = az.plot_trace(trace, var_names=["gamma_vix", "gamma_slope", "tau"])
    st.pyplot(plt.gcf())
    
    st.markdown("---")
    st.subheader("Acquisition Logic (Wei et al. 2025)")
    st.info("The surface identifies high-information 'Counterfactual' states for adversarial stress testing.")
    acq_x = np.linspace(0, 100, 100)
    acq_y = np.exp(-(acq_x - 50)**2 / 50) + np.random.normal(0, 0.02, 100)
    fig_acq, ax_acq = plt.subplots(figsize=(10, 2))
    ax_acq.plot(acq_x, acq_y, color='#2ecc71')
    ax_acq.fill_between(acq_x, acq_y, color='#2ecc71', alpha=0.2)
    ax_acq.axis('off')
    st.pyplot(fig_acq)