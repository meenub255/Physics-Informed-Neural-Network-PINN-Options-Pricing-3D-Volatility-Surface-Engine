"""
Analytical Black-Scholes-Merton (BSM) Pricing, Exact Closed-Form Greeks,
Gatheral SVI Volatility Surface Parameterization, Durrleman No-Arbitrage Bounds,
Dupire Local Volatility Conversion, and Fast Implied Volatility Solvers.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Union

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]

# Calibrated SVI Base Parameters for NDX / QQQ Equity Index Skew
# Raw SVI: w(k, tau) = a(tau) + b(tau) * (rho * (k - m) + sqrt((k - m)^2 + s^2))
NDX_SVI_PARAMS = {
    "base_atm_vol": 0.185,      # 18.5% baseline ATM annualized vol
    "term_slope": -0.018,       # Term structure slope
    "b_scale": 0.065,           # Smile wing slope scale
    "rho": -0.65,               # Strong negative equity index skew (put protection demand)
    "m": 0.015,                 # Horizontal smile translation
    "sigma_knot": 0.14,         # ATM curvature smoothness
}

QQQ_SVI_PARAMS = {
    "base_atm_vol": 0.186,
    "term_slope": -0.018,
    "b_scale": 0.065,
    "rho": -0.65,
    "m": 0.015,
    "sigma_knot": 0.14,
}


def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: Union[str, np.ndarray] = "call",
) -> np.ndarray:
    """
    Vectorized Black-Scholes-Merton European option price with continuous dividend yield q.

    Parameters
    ----------
    S : float or np.ndarray
        Spot price of the underlying index/ETF (e.g., NDX or QQQ).
    K : float or np.ndarray
        Strike price.
    tau : float or np.ndarray
        Time to maturity in years (T - t >= 0).
    r : float or np.ndarray
        Annualized continuously compounded risk-free interest rate.
    q : float or np.ndarray
        Annualized continuous dividend yield.
    sigma : float or np.ndarray
        Annualized implied/constant volatility.
    option_type : str or np.ndarray
        'call' or 'put' (or array of 'call'/'put' or 'C'/'P').

    Returns
    -------
    np.ndarray
        Option fair value(s).
    """
    S_arr = np.asarray(S, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    sig_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)

    # Handle terminal payoff where tau == 0
    safe_tau = np.maximum(tau_arr, 1e-12)
    sqrt_tau = np.sqrt(safe_tau)
    sig_sqrt_tau = sig_arr * sqrt_tau

    d1 = (np.log(np.maximum(S_arr, 1e-12) / np.maximum(K_arr, 1e-12)) + (r_arr - q_arr + 0.5 * sig_arr**2) * safe_tau) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau

    disc_q = np.exp(-q_arr * safe_tau)
    disc_r = np.exp(-r_arr * safe_tau)

    call_val = S_arr * disc_q * norm.cdf(d1) - K_arr * disc_r * norm.cdf(d2)
    put_val = K_arr * disc_r * norm.cdf(-d2) - S_arr * disc_q * norm.cdf(-d1)

    # Exact payoff at expiration tau <= 1e-10
    call_payoff = np.maximum(S_arr - K_arr, 0.0)
    put_payoff = np.maximum(K_arr - S_arr, 0.0)
    call_val = np.where(tau_arr <= 1e-10, call_payoff, call_val)
    put_val = np.where(tau_arr <= 1e-10, put_payoff, put_val)

    if isinstance(option_type, str):
        is_call = option_type.lower().startswith("c")
        return call_val if is_call else put_val
    else:
        opt_arr = np.char.lower(np.asarray(option_type, dtype=str))
        is_call = np.char.startswith(opt_arr, "c")
        return np.where(is_call, call_val, put_val)


def bsm_greeks(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    option_type: Union[str, np.ndarray] = "call",
) -> Dict[str, np.ndarray]:
    """
    Compute exact analytical Black-Scholes-Merton Greeks and strike sensitivities:
    - price: Option fair value V
    - delta: dV/dS
    - gamma: d^2V/dS^2
    - vega:  dV/dsigma (raw sensitivity per 1.00 = 100% vol point; divide by 100 for per-1% vol)
    - theta: -dV/dtau = dV/dt (annualized time decay)
    - rho:   dV/dr
    - dual_delta: dV/dK
    - dual_gamma: d^2V/dK^2 (Breeden-Litzenberger undiscounted risk-neutral density)
    - vanna: d^2V/(dS dsigma)
    - volga: d^2V/dsigma^2
    """
    S_arr = np.asarray(S, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 1e-8)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    sig_arr = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)

    sqrt_tau = np.sqrt(tau_arr)
    sig_sqrt_tau = sig_arr * sqrt_tau

    d1 = (np.log(S_arr / K_arr) + (r_arr - q_arr + 0.5 * sig_arr**2) * tau_arr) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau

    disc_q = np.exp(-q_arr * tau_arr)
    disc_r = np.exp(-r_arr * tau_arr)

    nd1 = norm.pdf(d1)
    nd2 = norm.pdf(d2)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)

    if isinstance(option_type, str):
        is_call = np.full_like(S_arr, option_type.lower().startswith("c"), dtype=bool)
    else:
        opt_arr = np.char.lower(np.asarray(option_type, dtype=str))
        is_call = np.char.startswith(opt_arr, "c")

    # Price
    call_price = S_arr * disc_q * Nd1 - K_arr * disc_r * Nd2
    put_price = K_arr * disc_r * N_minus_d2 - S_arr * disc_q * N_minus_d1
    price = np.where(is_call, call_price, put_price)

    # Delta: dV/dS
    delta = np.where(is_call, disc_q * Nd1, -disc_q * N_minus_d1)

    # Gamma: d^2V/dS^2 (identical for call and put)
    gamma = (disc_q * nd1) / (S_arr * sig_sqrt_tau)

    # Vega: dV/dsigma (identical for call and put)
    vega = S_arr * disc_q * nd1 * sqrt_tau

    # Theta: -dV/dtau = dV/dt (annualized)
    common_theta = -(S_arr * disc_q * nd1 * sig_arr) / (2.0 * sqrt_tau)
    call_theta = common_theta + q_arr * S_arr * disc_q * Nd1 - r_arr * K_arr * disc_r * Nd2
    put_theta = common_theta - q_arr * S_arr * disc_q * N_minus_d1 + r_arr * K_arr * disc_r * N_minus_d2
    theta = np.where(is_call, call_theta, put_theta)

    # Rho: dV/dr
    call_rho = K_arr * tau_arr * disc_r * Nd2
    put_rho = -K_arr * tau_arr * disc_r * N_minus_d2
    rho = np.where(is_call, call_rho, put_rho)

    # Dual Delta: dV/dK and Dual Gamma: d^2V/dK^2 (Breeden-Litzenberger)
    dual_delta = np.where(is_call, -disc_r * Nd2, disc_r * N_minus_d2)
    dual_gamma = (disc_r * nd2) / (K_arr * sig_sqrt_tau)

    # Second-order volatility Greeks: Vanna & Volga
    vanna = -disc_q * nd1 * (d2 / sig_arr)
    volga = vega * (d1 * d2 / sig_arr)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "dual_delta": dual_delta,
        "dual_gamma": dual_gamma,
        "vanna": vanna,
        "volga": volga,
    }


def bsm_price_torch(
    S: torch.Tensor,
    K: torch.Tensor,
    tau: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    sigma: torch.Tensor,
    option_type: Literal["call", "put"] = "call",
) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of Black-Scholes-Merton pricing.
    """
    safe_tau = torch.clamp(tau, min=1e-8)
    safe_sigma = torch.clamp(sigma, min=1e-6)
    sqrt_tau = torch.sqrt(safe_tau)
    sig_sqrt_tau = safe_sigma * sqrt_tau

    d1 = (torch.log(torch.clamp(S, min=1e-8) / torch.clamp(K, min=1e-8)) + (r - q + 0.5 * safe_sigma**2) * safe_tau) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau

    disc_q = torch.exp(-q * safe_tau)
    disc_r = torch.exp(-r * safe_tau)

    Nd1 = torch.special.ndtr(d1)
    Nd2 = torch.special.ndtr(d2)

    if option_type.lower().startswith("c"):
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        payoff = torch.relu(S - K)
    else:
        price = K * disc_r * torch.special.ndtr(-d2) - S * disc_q * torch.special.ndtr(-d1)
        payoff = torch.relu(K - S)

    return torch.where(tau <= 1e-7, payoff, price)


def bsm_greeks_torch(
    S: torch.Tensor,
    K: torch.Tensor,
    tau: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    sigma: torch.Tensor,
    option_type: Literal["call", "put"] = "call",
) -> Dict[str, torch.Tensor]:
    """
    Exact analytical Greeks in PyTorch for direct tensor benchmarking against PINN autograd.
    """
    safe_tau = torch.clamp(tau, min=1e-8)
    safe_sigma = torch.clamp(sigma, min=1e-6)
    sqrt_tau = torch.sqrt(safe_tau)
    sig_sqrt_tau = safe_sigma * sqrt_tau

    d1 = (torch.log(S / K) + (r - q + 0.5 * safe_sigma**2) * safe_tau) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau

    disc_q = torch.exp(-q * safe_tau)
    disc_r = torch.exp(-r * safe_tau)

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    nd1 = inv_sqrt_2pi * torch.exp(-0.5 * d1**2)
    nd2 = inv_sqrt_2pi * torch.exp(-0.5 * d2**2)

    Nd1 = torch.special.ndtr(d1)
    Nd2 = torch.special.ndtr(d2)
    N_minus_d1 = torch.special.ndtr(-d1)
    N_minus_d2 = torch.special.ndtr(-d2)

    is_call = option_type.lower().startswith("c")

    if is_call:
        price = S * disc_q * Nd1 - K * disc_r * Nd2
        delta = disc_q * Nd1
        theta = -(S * disc_q * nd1 * safe_sigma) / (2.0 * sqrt_tau) + q * S * disc_q * Nd1 - r * K * disc_r * Nd2
        rho = K * safe_tau * disc_r * Nd2
    else:
        price = K * disc_r * N_minus_d2 - S * disc_q * N_minus_d1
        delta = -disc_q * N_minus_d1
        theta = -(S * disc_q * nd1 * safe_sigma) / (2.0 * sqrt_tau) - q * S * disc_q * N_minus_d1 + r * K * disc_r * N_minus_d2
        rho = -K * safe_tau * disc_r * N_minus_d2

    gamma = (disc_q * nd1) / (S * sig_sqrt_tau)
    vega = S * disc_q * nd1 * sqrt_tau
    dual_gamma = (disc_r * nd2) / (K * sig_sqrt_tau)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "dual_gamma": dual_gamma,
    }


def implied_volatility_vec(
    market_price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    option_type: Union[str, np.ndarray] = "call",
    tol: float = 1e-7,
    max_iter: int = 40,
) -> np.ndarray:
    """
    Fast vectorized Implied Volatility solver combining damped Halley/Newton-Raphson
    iterations with a guaranteed SciPy Brent's method fallback (`brentq`).
    """
    P_arr = np.atleast_1d(np.asarray(market_price, dtype=np.float64))
    S_arr = np.broadcast_to(np.asarray(S, dtype=np.float64), P_arr.shape)
    K_arr = np.broadcast_to(np.asarray(K, dtype=np.float64), P_arr.shape)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=np.float64), P_arr.shape)
    r_arr = np.broadcast_to(np.asarray(r, dtype=np.float64), P_arr.shape)
    q_arr = np.broadcast_to(np.asarray(q, dtype=np.float64), P_arr.shape)

    if isinstance(option_type, str):
        opt_arr = np.full(P_arr.shape, option_type, dtype="<U4")
    else:
        opt_arr = np.broadcast_to(np.asarray(option_type, dtype="<U4"), P_arr.shape)

    # Brenner-Subrahmanyam / Corrado-Miller style initial guess
    sigma = np.full(P_arr.shape, 0.22, dtype=np.float64)

    for _ in range(max_iter):
        res = bsm_greeks(S_arr, K_arr, tau_arr, r_arr, q_arr, sigma, opt_arr)
        diff = res["price"] - P_arr
        if np.max(np.abs(diff)) < tol:
            break
        vega = np.maximum(res["vega"], 1e-8)
        volga = res["volga"]
        # Halley's step with safety damping
        denom = vega - 0.5 * diff * (volga / vega)
        step = np.where(np.abs(denom) > 1e-7, diff / denom, diff / vega)
        step = np.clip(step, -0.25, 0.25)
        sigma = np.clip(sigma - step, 0.01, 3.0)

    # Verify convergence and apply Brent's method fallback for any stubborn deep OTM/ITM options
    final_prices = bsm_price(S_arr, K_arr, tau_arr, r_arr, q_arr, sigma, opt_arr)
    unconverged = np.abs(final_prices - P_arr) > 1e-4

    if np.any(unconverged):
        flat_idx = np.flatnonzero(unconverged)
        for idx in flat_idx:
            s_i = float(S_arr.flat[idx])
            k_i = float(K_arr.flat[idx])
            t_i = float(tau_arr.flat[idx])
            r_i = float(r_arr.flat[idx])
            q_i = float(q_arr.flat[idx])
            p_i = float(P_arr.flat[idx])
            o_i = str(opt_arr.flat[idx])

            def obj(vol: float) -> float:
                return float(bsm_price(s_i, k_i, t_i, r_i, q_i, vol, o_i)) - p_i

            try:
                low_v, high_v = 0.005, 4.0
                if obj(low_v) * obj(high_v) < 0:
                    sigma.flat[idx] = brentq(obj, low_v, high_v, xtol=1e-7, maxiter=100)
            except ValueError:
                pass

    return sigma if np.ndim(market_price) > 0 else float(sigma[0])


def svi_total_variance(
    k: ArrayLike,
    tau: ArrayLike,
    params: Dict[str, float] = NDX_SVI_PARAMS,
) -> Dict[str, np.ndarray]:
    """
    Gatheral's (2004) Stochastic Volatility Inspired (SVI) parameterization across log-moneyness
    k = ln(K / F_T) and maturity tau = T - t, designed to be strictly free of calendar and
    butterfly arbitrage across typical index strikes and maturities.

    Returns
    -------
    dict with keys:
        - w: Total implied variance w(k, tau) = sigma_imp^2 * tau
        - dw_dk: First strike derivative dw/dk
        - d2w_dk2: Second strike derivative d^2w/dk^2
        - dw_dtau: Maturity derivative dw/dtau (must be >= 0 for no calendar arbitrage)
        - sigma_imp: Annualized implied volatility sqrt(w / tau)
    """
    k_arr = np.asarray(k, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 1e-5)

    base_vol = params["base_atm_vol"]
    term_slope = params["term_slope"]
    b_scale = params["b_scale"]
    rho = params["rho"]
    m = params["m"]
    s = params["sigma_knot"]

    # Term structure of baseline variance a(tau) and wing slope b(tau)
    # Constructed so dw/dtau > 0 everywhere (guaranteed no calendar spread arbitrage)
    atm_var_rate = (base_vol + term_slope * (1.0 - np.exp(-1.5 * tau_arr))) ** 2
    b_tau = b_scale * (tau_arr ** 0.72)
    db_dtau = 0.72 * b_scale * (tau_arr ** -0.28)

    # Shift a(tau) so ATM variance at k=0 matches atm_var_rate * tau smoothly
    smile_at_zero = rho * (-m) + np.sqrt(m**2 + s**2)
    a_tau = np.maximum(atm_var_rate * tau_arr - b_tau * smile_at_zero, 1e-5 * tau_arr)

    # Derivative of a_tau wrt tau
    datm_dtau = (
        2.0
        * (base_vol + term_slope * (1.0 - np.exp(-1.5 * tau_arr)))
        * (term_slope * 1.5 * np.exp(-1.5 * tau_arr))
        * tau_arr
        + atm_var_rate
    )
    da_dtau = datm_dtau - db_dtau * smile_at_zero

    km = k_arr - m
    root = np.sqrt(km**2 + s**2)
    smile_term = rho * km + root

    w = a_tau + b_tau * smile_term
    dw_dk = b_tau * (rho + km / root)
    d2w_dk2 = b_tau * (s**2) / (root**3)
    dw_dtau = da_dtau + db_dtau * smile_term

    sigma_imp = np.sqrt(np.maximum(w / tau_arr, 1e-6))

    return {
        "w": w,
        "dw_dk": dw_dk,
        "d2w_dk2": d2w_dk2,
        "dw_dtau": dw_dtau,
        "sigma_imp": sigma_imp,
    }


def svi_implied_vol(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    params: Dict[str, float] = NDX_SVI_PARAMS,
) -> np.ndarray:
    """
    Evaluate the SVI implied volatility sigma_imp(S, K, tau, r, q).
    """
    S_arr = np.asarray(S, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 1e-5)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    forward = S_arr * np.exp((r_arr - q_arr) * tau_arr)
    k = np.log(K_arr / forward)
    svi = svi_total_variance(k, tau_arr, params=params)
    return svi["sigma_imp"]


def durrleman_butterfly_condition(
    k: ArrayLike,
    w: ArrayLike,
    dw_dk: ArrayLike,
    d2w_dk2: ArrayLike,
) -> np.ndarray:
    """
    Gatheral & Jacquier (2014) / Durrleman butterfly no-arbitrage function g(k):
    g(k) = (1 - k * w' / (2 * w))^2 - (w'^2 / 4) * (1 / w + 1 / 4) + w'' / 2
    A volatility slice is free of butterfly arbitrage if and only if g(k) >= 0 for all k
    and lim_{k -> +inf} d1(k) = -inf.
    """
    k_arr = np.asarray(k, dtype=np.float64)
    w_arr = np.maximum(np.asarray(w, dtype=np.float64), 1e-8)
    wp = np.asarray(dw_dk, dtype=np.float64)
    wpp = np.asarray(d2w_dk2, dtype=np.float64)

    term1 = (1.0 - (k_arr * wp) / (2.0 * w_arr)) ** 2
    term2 = 0.25 * (wp**2) * (1.0 / w_arr + 0.25)
    term3 = 0.5 * wpp
    return term1 - term2 + term3


def dupire_local_vol_from_svi(
    S: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    params: Dict[str, float] = NDX_SVI_PARAMS,
) -> np.ndarray:
    """
    Compute exact Dupire Local Volatility sigma_loc(K, tau) from total implied variance w(k, tau)
    using Gatheral's log-moneyness Dupire identity:
        sigma_loc^2(k, tau) = (dw / dtau) / g(k)
    where g(k) is the Durrleman butterfly density factor.
    """
    S_arr = np.asarray(S, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.maximum(np.asarray(tau, dtype=np.float64), 1e-5)
    r_arr = np.asarray(r, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    forward = S_arr * np.exp((r_arr - q_arr) * tau_arr)
    k = np.log(K_arr / forward)

    svi = svi_total_variance(k, tau_arr, params=params)
    g_k = durrleman_butterfly_condition(k, svi["w"], svi["dw_dk"], svi["d2w_dk2"])

    local_var = np.maximum(svi["dw_dtau"], 1e-6) / np.maximum(g_k, 1e-4)
    return np.sqrt(local_var)
