
"""
finance_formulas.py

A compact toolkit implementing 9 groups of investment formulas:
1) Returns & portfolio aggregation (including covariance-based volatility)
2) Sharpe ratio
3) CAGR / TWR / IRR
4) Total return (with external cash inflows)
5) Fees (TER) and account value simulation with contributions
6) Derived time series for visualization (NAV / cumulative return)
7) Benchmark comparison (excess cumulative, tracking diff/error)
8) Linear regression prediction & residual std
9) DCA vs Lump Sum utilities

Dependencies: numpy, pandas
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, Optional
import numpy as np
import pandas as pd

# -----------------------------
# 1) RETURNS & PORTFOLIO
# -----------------------------

def simple_returns_from_prices(prices: Sequence[float]) -> np.ndarray:
    """
    Compute simple periodic returns r_t = (P_t - P_{t-1}) / P_{t-1}
    prices: sequence of prices (len >= 2)
    returns: ndarray of length len(prices)-1
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be 1D and length >= 2")
    return p[1:] / p[:-1] - 1.0


def log_returns_from_prices(prices: Sequence[float]) -> np.ndarray:
    """
    Compute log returns l_t = ln(P_t / P_{t-1}).
    """
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be 1D and length >= 2")
    return np.log(p[1:] / p[:-1])


def portfolio_returns(weights: Sequence[float], returns_matrix: Iterable[Sequence[float]]) -> np.ndarray:
    """
    Combine asset returns into portfolio returns by weighted sum:
      r_p,t = sum_i w_i * r_i,t
    weights: length N
    returns_matrix: shape (T, N) or (N, T); function will auto-detect
    returns: shape (T,)
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    R = np.asarray(returns_matrix, dtype=float)
    # allow (T, N) or (N, T)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D")
    if R.shape[1] == w.size:
        # (T, N)
        return R @ w
    elif R.shape[0] == w.size:
        # (N, T)
        return (w @ R)
    else:
        raise ValueError("returns_matrix shape not compatible with weights length")


def portfolio_variance(weights: Sequence[float], cov_matrix: Sequence[Sequence[float]]) -> float:
    """
    Portfolio variance: w^T Σ w
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    Sigma = np.asarray(cov_matrix, dtype=float)
    if Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] != w.size:
        raise ValueError("cov_matrix must be square and match weights size")
    return float(w.T @ Sigma @ w)


def volatility_from_returns(returns: Sequence[float], ddof: int = 1) -> float:
    """
    Sample standard deviation of returns.
    ddof=1 for sample std (unbiased); ddof=0 for population std.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be 1D, length >= 2")
    return float(r.std(ddof=ddof))


def annualize_volatility(periodic_vol: float, periods_per_year: int) -> float:
    """Annualize volatility by sqrt(periods_per_year)."""
    if periodic_vol < 0 or periods_per_year <= 0:
        raise ValueError("volatility must be >= 0 and periods_per_year > 0")
    return periodic_vol * math.sqrt(periods_per_year)


# -----------------------------
# 2) SHARPE RATIO
# -----------------------------

def sharpe_ratio(returns: Sequence[float], risk_free_rate_annual: float, periods_per_year: int, ddof: int = 1) -> float:
    """
    Annualized Sharpe ratio using periodic returns r_t (e.g., monthly).
    Convert annual risk-free rate into per-period: rf_p = (1+rf)^(1/m) - 1

    Sharpe_ann = (mean(r - rf_p) / std(r)) * sqrt(m)
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("returns must be 1D, length >= 2")
    m = periods_per_year
    rf_p = (1.0 + risk_free_rate_annual) ** (1.0 / m) - 1.0
    ex = r - rf_p
    mu = ex.mean()
    sd = ex.std(ddof=ddof)
    if sd == 0:
        return float("nan")
    return float(mu / sd * math.sqrt(m))


# -----------------------------
# 3) CAGR / TWR / IRR
# -----------------------------

def cagr(v0: float, vT: float, years: float) -> float:
    """Compound Annual Growth Rate with no external cash flows."""
    if v0 <= 0 or vT <= 0 or years <= 0:
        raise ValueError("v0, vT, years must be > 0")
    return (vT / v0) ** (1.0 / years) - 1.0


def time_weighted_return(period_returns: Sequence[float], periods_per_year: Optional[int] = None) -> Tuple[float, Optional[float]]:
    """
    Time-Weighted Return (TWR): multiply subperiod gross returns.
    period_returns: iterable of simple returns r_t
    Returns:
      total_return (TWR_total - 1), optional annualized if periods_per_year provided.
    """
    r = np.asarray(period_returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("period_returns must be 1D and non-empty")
    growth = float(np.prod(1.0 + r))
    twr_total = growth - 1.0
    if periods_per_year is None:
        return twr_total, None
    years = r.size / periods_per_year
    ann = growth ** (1.0 / years) - 1.0
    return twr_total, ann


def irr(cash_flows: Sequence[float], times_years: Optional[Sequence[float]] = None, guess: float = 0.1, tol: float = 1e-10, maxiter: int = 1000) -> float:
    """
    Compute IRR (or XIRR if times_years provided) by Newton's method.
    cash_flows: list of CF amounts (negative for outflow/investment, positive for inflow/withdrawal)
    times_years: same length as cash_flows, times in years (0 for initial). If None, assumes equal spacing 1 unit.
    Returns annualized IRR.
    """
    cf = np.asarray(cash_flows, dtype=float)
    if times_years is None:
        t = np.arange(cf.size, dtype=float)
    else:
        t = np.asarray(times_years, dtype=float)
    if cf.size != t.size:
        raise ValueError("cash_flows and times_years length mismatch")

    def npv(rate: float) -> float:
        return float(np.sum(cf / (1.0 + rate) ** t))

    def d_npv(rate: float) -> float:
        return float(np.sum(-t * cf / (1.0 + rate) ** (t + 1.0)))

    r = guess
    for _ in range(maxiter):
        f = npv(r)
        df = d_npv(r)
        if abs(df) < 1e-18:
            break
        r_next = r - f / df
        if abs(r_next - r) < tol:
            return r_next
        r = r_next
    return r


# -----------------------------
# 4) TOTAL RETURN (with cash inflows)
# -----------------------------

def total_return_final_value(final_value: float, inflows: Sequence[float]) -> float:
    """
    Total return given final account value and list of inflows (including initial investment).
    R_total = (V_T - sum(inflows)) / sum(inflows)
    """
    s = float(np.sum(inflows))
    if s <= 0:
        raise ValueError("sum of inflows must be > 0")
    return (final_value - s) / s


# -----------------------------
# 5) FEES (TER) & ACCOUNT SIMULATION
# -----------------------------

def per_period_fee_from_ter(ter_annual: float, periods_per_year: int) -> float:
    """
    Convert annual TER to per-period fee approximation.
    Simple linear allocation: ter_per = ter_annual / m
    """
    if ter_annual < 0 or periods_per_year <= 0:
        raise ValueError("ter_annual >=0 and periods_per_year >0")
    return ter_annual / periods_per_year


def simulate_account_value(
    gross_returns: Sequence[float],
    contributions: Sequence[float],
    ter_annual: float = 0.0,
    periods_per_year: int = 12,
    v0: float = 0.0
) -> np.ndarray:
    """
    Simulate account value over time with contributions (C_t) and gross returns r_t,
    deducting TER spread across periods.
    V_t = (V_{t-1} + C_t) * (1 + r_t - ter_annual/periods_per_year)
    Returns array of V_t for each period.
    """
    r = np.asarray(gross_returns, dtype=float).reshape(-1)
    C = np.asarray(contributions, dtype=float).reshape(-1)
    if r.size != C.size:
        raise ValueError("gross_returns and contributions must have same length")
    fee_per = per_period_fee_from_ter(ter_annual, periods_per_year)
    V = np.empty_like(r, dtype=float)
    v = float(v0)
    for t in range(r.size):
        v = (v + C[t]) * (1.0 + r[t] - fee_per)
        V[t] = v
    return V


# -----------------------------
# 6) DERIVED SERIES FOR VISUALIZATION
# -----------------------------

def cumulative_nav_from_returns(returns: Sequence[float], start_nav: float = 1.0) -> np.ndarray:
    """
    NAV_t = start * Π(1 + r_k)
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    growth = np.cumprod(1.0 + r)
    return start_nav * growth


# -----------------------------
# 7) BENCHMARK COMPARISON
# -----------------------------

def cumulative_excess_vs_benchmark(portfolio_returns_seq: Sequence[float], benchmark_returns_seq: Sequence[float]) -> np.ndarray:
    """
    Excess_t = Π[(1+r_p)/(1+r_b)] - 1
    Returned as a series over time.
    """
    rp = np.asarray(portfolio_returns_seq, dtype=float).reshape(-1)
    rb = np.asarray(benchmark_returns_seq, dtype=float).reshape(-1)
    if rp.size != rb.size:
        raise ValueError("portfolio and benchmark must have same length")
    ratio = (1.0 + rp) / (1.0 + rb)
    cum = np.cumprod(ratio) - 1.0
    return cum


def tracking_difference(returns_p: Sequence[float], returns_b: Sequence[float]) -> float:
    """Average difference E[r_p - r_b]."""
    rp = np.asarray(returns_p, dtype=float); rb = np.asarray(returns_b, dtype=float)
    if rp.size != rb.size:
        raise ValueError("length mismatch")
    return float((rp - rb).mean())


def tracking_error_annualized(returns_p: Sequence[float], returns_b: Sequence[float], periods_per_year: int, ddof: int = 1) -> float:
    """Annualized std of (r_p - r_b)."""
    rp = np.asarray(returns_p, dtype=float); rb = np.asarray(returns_b, dtype=float)
    if rp.size != rb.size:
        raise ValueError("length mismatch")
    diff_std = float((rp - rb).std(ddof=ddof))
    return diff_std * math.sqrt(periods_per_year)


# -----------------------------
# 8) LINEAR REGRESSION & RESIDUAL STD
# -----------------------------

def linear_regression_fit(y: Sequence[float], X: Optional[Sequence[Sequence[float]]] = None, add_intercept: bool = True) -> Tuple[np.ndarray, float]:
    """
    Fit OLS: y = alpha + X beta + eps
    If X is None, we fit a time trend X = [[t]] with t=0..T-1.
    Returns (params, residual_std), where params includes intercept if add_intercept=True.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
    T = y_arr.shape[0]
    if X is None:
        X_arr = np.arange(T, dtype=float).reshape(-1, 1)
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != T:
            raise ValueError("X must have same number of rows as y")

    if add_intercept:
        X_design = np.hstack([np.ones((T, 1)), X_arr])
    else:
        X_design = X_arr

    # OLS closed-form: beta = (X'X)^{-1} X'y
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X_design.T @ y_arr  # (k x 1)
    y_hat = X_design @ beta
    resid = y_arr - y_hat
    k = X_design.shape[1]
    dof = max(T - k, 1)
    resid_std = float(np.sqrt((resid**2).sum() / dof))
    return beta.reshape(-1), resid_std


def linear_regression_predict(beta: Sequence[float], X_next: Sequence[float], add_intercept: bool = True) -> float:
    """
    Predict y_next given fitted params beta and a feature row X_next.
    """
    b = np.asarray(beta, dtype=float).reshape(-1)
    x = np.asarray(X_next, dtype=float).reshape(-1)
    if add_intercept:
        x = np.hstack([1.0, x])
    if x.size != b.size:
        raise ValueError("X_next dimension mismatch with beta")
    return float(b @ x)


# -----------------------------
# 9) DCA vs LUMP SUM
# -----------------------------

def fv_lump_sum(v0: float, returns: Sequence[float], ter_annual: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Final value for lump sum given a sequence of gross returns and TER.
    """
    fee_per = per_period_fee_from_ter(ter_annual, periods_per_year)
    growth = float(np.prod(1.0 + np.asarray(returns, dtype=float) - fee_per))
    return v0 * growth


def fv_dca_theoretical(C: float, r_per: float, n: int, at_begin: bool = False) -> float:
    """
    Theoretical FV for constant contribution C each period at constant rate r_per (ordinary annuity).
    If at_begin=True, treat as annuity due (multiply by (1+r_per)).
    """
    if r_per == 0:
        fv = C * n
    else:
        fv = C * ((1.0 + r_per) ** n - 1.0) / r_per
    if at_begin:
        fv *= (1.0 + r_per)
    return fv


def fv_dca_simulated(contributions: Sequence[float], returns: Sequence[float], ter_annual: float = 0.0, periods_per_year: int = 12) -> float:
    """
    Simulated DCA final value with varying per-period returns and TER.
    Uses the same recursion as simulate_account_value, returns last V.
    """
    V = simulate_account_value(returns, contributions, ter_annual=ter_annual, periods_per_year=periods_per_year, v0=0.0)
    return float(V[-1])


# -----------------------------
# Small helper: input-friendly wrappers
# -----------------------------

def as_percent(x: float, digits: int = 4) -> str:
    """Format a decimal (e.g., 0.06123) as percentage string (e.g., '6.1230%')."""
    return f"{x*100:.{digits}f}%"


if __name__ == "__main__":
    # Tiny interactive demo (you can run: python finance_formulas.py)
    print("Demo: basic usage")
    prices = [100, 103, 101, 104]
    r = simple_returns_from_prices(prices)
    print("Prices:", prices)
    print("Returns:", np.round(r, 6))

    w = [0.6, 0.4]
    R_assets = np.column_stack([r, r*0.3 + 0.01])  # mock a 2-asset matrix (T x N)
    rp = portfolio_returns(w, R_assets)
    print("Portfolio returns:", np.round(rp, 6))

    vol_m = volatility_from_returns(rp)
    vol_a = annualize_volatility(vol_m, periods_per_year=12)
    print("Vol (monthly):", round(vol_m, 6), "Vol (annualized):", round(vol_a, 6))

    sh = sharpe_ratio(rp, risk_free_rate_annual=0.02, periods_per_year=12)
    print("Sharpe (ann):", round(sh, 6))

    twr_total, twr_ann = time_weighted_return(rp, periods_per_year=12)
    print("TWR total:", round(twr_total, 6), "TWR ann:", round(twr_ann, 6))

    c = cagr(100, 150, years=3)
    print("CAGR:", as_percent(c))

    V = simulate_account_value(gross_returns=rp, contributions=[100]*len(rp), ter_annual=0.002, periods_per_year=12, v0=0.0)
    print("Simulated account values (first 3):", np.round(V[:3], 2), "... last:", round(float(V[-1]), 2))

    # Benchmark mock
    rb = rp - 0.002  # pretend benchmark slightly different
    excess_cum = cumulative_excess_vs_benchmark(rp, rb)
    print("Excess cumulative (last):", round(float(excess_cum[-1]), 6))

    # Regression
    beta, s_eps = linear_regression_fit(y=rp)  # time trend
    pred_next = linear_regression_predict(beta, X_next=[len(rp)], add_intercept=True)
    print("OLS beta:", np.round(beta, 6), "resid std:", round(s_eps, 6), "next pred:", round(pred_next, 6))

    # DCA vs Lump Sum
    fv_ls = fv_lump_sum(10000, returns=rp, ter_annual=0.002, periods_per_year=12)
    fv_dca = fv_dca_simulated(contributions=[500]*len(rp), returns=rp, ter_annual=0.002, periods_per_year=12)
    print("FV LumpSum:", round(fv_ls, 2), "FV DCA:", round(fv_dca, 2))
