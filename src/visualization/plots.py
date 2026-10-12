"""
Clean, Uncrowded 3D Volatility Surface, Dupire Local Volatility, Breeden-Litzenberger Density,
Autograd Greek Parity, and Nasdaq Multi-Exchange Surveillance Visualization Suite.

Note: Figure titles are intentionally omitted from the internal Plotly canvas and rendered
via clean HTML card headers in the UI to guarantee zero title/legend text overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from src.analytics.black_scholes import bsm_greeks
from src.models.pinn_net import DualPINNOptionsEngine

NASDAQ_IV_COLORSCALE = [
    [0.00, "#0A1931"],
    [0.20, "#003566"],
    [0.45, "#0066FF"],
    [0.70, "#00B4D8"],
    [0.88, "#00F5D4"],
    [1.00, "#E0FBF6"],
]

DUPIRE_LOCAL_COLORSCALE = [
    [0.00, "#0F172A"],
    [0.25, "#1E3A8A"],
    [0.50, "#3B82F6"],
    [0.75, "#38BDF8"],
    [1.00, "#FCD34D"],
]

DENSITY_COLORSCALE = [
    [0.00, "#090D16"],
    [0.30, "#1E293B"],
    [0.60, "#0284C7"],
    [0.85, "#10B981"],
    [1.00, "#A7F3D0"],
]

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _apply_clean_3d_layout(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    z_title: str,
    height: int = 560,
) -> go.Figure:
    fig.update_layout(
        title=None,
        font=dict(family=FONT_FAMILY, color="#CBD5E1", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(
                title=dict(text=x_title, font=dict(size=11, color="#94A3B8")),
                backgroundcolor="#0B111E",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
                tickfont=dict(size=10, color="#94A3B8"),
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(size=11, color="#94A3B8")),
                backgroundcolor="#0B111E",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
                tickfont=dict(size=10, color="#94A3B8"),
            ),
            zaxis=dict(
                title=dict(text=z_title, font=dict(size=11, color="#94A3B8")),
                backgroundcolor="#090E1A",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
                tickfont=dict(size=10, color="#94A3B8"),
            ),
            camera=dict(eye=dict(x=1.62, y=-1.58, z=0.92)),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.1, z=0.72),
        ),
        margin=dict(l=20, r=30, b=15, t=40),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.96,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#1E293B",
            borderwidth=1,
            font=dict(size=11.5, color="#E2E8F0"),
        ),
    )
    return fig


def evaluate_dense_surface_grid(
    engine: DualPINNOptionsEngine,
    spot: float = 20500.0,
    r: float = 0.045,
    q: float = 0.008,
    n_strikes: int = 48,
    n_maturities: int = 38,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    strikes = np.linspace(0.84 * spot, 1.16 * spot, n_strikes)
    taus = np.linspace(14.0 / 365.0, 1.0, n_maturities)
    K_grid, tau_grid = np.meshgrid(strikes, taus)

    S_flat = torch.full((K_grid.size,), float(spot), dtype=torch.float32, device=device)
    K_flat = torch.tensor(K_grid.ravel(), dtype=torch.float32, device=device)
    tau_flat = torch.tensor(tau_grid.ravel(), dtype=torch.float32, device=device)
    r_flat = torch.full((K_grid.size,), float(r), dtype=torch.float32, device=device)
    q_flat = torch.full((K_grid.size,), float(q), dtype=torch.float32, device=device)

    forward_flat = S_flat * torch.exp((r_flat - q_flat) * tau_flat)
    k_flat = torch.log(K_flat / forward_flat)

    surf = engine.vol_net.evaluate_surface_and_arbitrage(k_flat, tau_flat, create_graph=False)
    sig_flat = surf["sigma_imp"].detach()

    greeks = engine.pricing_net.compute_autograd_greeks(
        S_flat, K_flat, tau_flat, r_flat, q_flat, sig_flat, option_type="call", create_graph=False
    )

    shape = K_grid.shape
    return {
        "K_grid": K_grid,
        "tau_grid": tau_grid,
        "dte_grid": tau_grid * 365.0,
        "moneyness_grid": K_grid / spot,
        "iv_pct": (sig_flat.cpu().numpy().reshape(shape)) * 100.0,
        "local_vol_pct": (surf["dupire_local_vol"].detach().cpu().numpy().reshape(shape)) * 100.0,
        "durrleman_g": surf["durrleman_g"].detach().cpu().numpy().reshape(shape),
        "dw_dtau": surf["dw_dtau"].detach().cpu().numpy().reshape(shape),
        "total_var": surf["w"].detach().cpu().numpy().reshape(shape),
        "call_price": greeks["price"].cpu().numpy().reshape(shape),
        "delta": greeks["delta"].cpu().numpy().reshape(shape),
        "gamma": greeks["gamma"].cpu().numpy().reshape(shape),
        "vega": (greeks["vega"].cpu().numpy().reshape(shape)) / 100.0,
        "theta_daily": (greeks["theta"].cpu().numpy().reshape(shape)) / 365.0,
        "bl_density": greeks["bl_density"].cpu().numpy().reshape(shape),
    }


def create_3d_iv_surface_plotly(
    engine: DualPINNOptionsEngine,
    signals_df: pd.DataFrame,
    symbol: str = "NDX",
) -> go.Figure:
    sub = signals_df[signals_df["symbol"] == symbol].copy()
    spot = float(sub["spot_price"].iloc[0])
    r = float(sub["risk_free_rate"].iloc[0])
    q = float(sub["dividend_yield"].iloc[0])

    grid = evaluate_dense_surface_grid(engine, spot=spot, r=r, q=q)

    fig = go.Figure()

    # 1. Continuous PINN No-Arbitrage 3D Surface
    fig.add_trace(
        go.Surface(
            x=grid["K_grid"],
            y=grid["dte_grid"],
            z=grid["iv_pct"],
            colorscale=NASDAQ_IV_COLORSCALE,
            opacity=0.92,
            name="PINN Surface",
            contours=dict(
                x=dict(show=True, color="rgba(255,255,255,0.08)", width=1),
                y=dict(show=True, color="rgba(255,255,255,0.08)", width=1),
            ),
            colorbar=dict(
                title=dict(text="IV (%)", font=dict(size=11, color="#94A3B8")),
                tickfont=dict(size=10, color="#94A3B8"),
                thickness=14,
                len=0.65,
                x=0.96,
                outlinecolor="#1E293B",
                outlinewidth=1,
            ),
            hovertemplate=(
                "<b>PINN Manifold</b><br>"
                "Strike: $%{x:,.1f}<br>"
                "Tenor: %{y:.0f} DTE<br>"
                "Implied Vol: %{z:.2f}%<extra></extra>"
            ),
        )
    )

    # 2. Fair-priced Nasdaq Exchange Quotes
    fair_df = sub[sub["detected_signal"] == "FAIR_PRICED"]
    fig.add_trace(
        go.Scatter3d(
            x=fair_df["strike"],
            y=fair_df["maturity_days"],
            z=fair_df["market_iv"] * 100.0,
            mode="markers",
            name="Exchange Quotes (PHLX/ISE/NOM/BX/GEMX/MRX)",
            marker=dict(size=2.8, color="#38BDF8", opacity=0.72),
            text=fair_df["exchange_code"] + " • " + fair_df["option_type"].str.upper(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Strike: $%{x:,.1f} | Tenor: %{y} DTE<br>"
                "Market IV: %{z:.2f}%<extra></extra>"
            ),
        )
    )

    # 3. Detected Mispriced & Arbitrage-Violating Quotes
    anom_df = sub[sub["detected_signal"] != "FAIR_PRICED"]
    if len(anom_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=anom_df["strike"],
                y=anom_df["maturity_days"],
                z=anom_df["market_iv"] * 100.0,
                mode="markers",
                name="Flagged Arbitrage / Mispricing",
                marker=dict(
                    size=5.8,
                    color="#F43F5E",
                    symbol="diamond",
                    line=dict(color="#FFE4E6", width=1.2),
                ),
                text=(
                    anom_df["exchange_code"]
                    + " • "
                    + anom_df["detected_signal"]
                    + " ("
                    + anom_df["iv_residual_bps"].round(1).astype(str)
                    + " bps)"
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Strike: $%{x:,.1f} | Tenor: %{y} DTE<br>"
                    "Quoted IV: %{z:.2f}%<extra></extra>"
                ),
            )
        )

    return _apply_clean_3d_layout(
        fig=fig,
        x_title=f"Strike ($ {symbol})",
        y_title="Tenor (DTE)",
        z_title="Implied Vol (%)",
        height=540,
    )


def create_3d_local_vol_surface_plotly(
    engine: DualPINNOptionsEngine,
    signals_df: pd.DataFrame,
    symbol: str = "NDX",
) -> go.Figure:
    sub = signals_df[signals_df["symbol"] == symbol].copy()
    spot = float(sub["spot_price"].iloc[0])
    r = float(sub["risk_free_rate"].iloc[0])
    q = float(sub["dividend_yield"].iloc[0])

    grid = evaluate_dense_surface_grid(engine, spot=spot, r=r, q=q)

    fig = go.Figure(
        data=[
            go.Surface(
                x=grid["K_grid"],
                y=grid["dte_grid"],
                z=grid["local_vol_pct"],
                colorscale=DUPIRE_LOCAL_COLORSCALE,
                opacity=0.95,
                contours=dict(
                    x=dict(show=True, color="rgba(255,255,255,0.07)", width=1),
                    y=dict(show=True, color="rgba(255,255,255,0.07)", width=1),
                ),
                colorbar=dict(
                    title=dict(text="Local Vol (%)", font=dict(size=11, color="#94A3B8")),
                    tickfont=dict(size=10, color="#94A3B8"),
                    thickness=14,
                    len=0.65,
                    x=0.96,
                    outlinecolor="#1E293B",
                    outlinewidth=1,
                ),
                hovertemplate=(
                    "<b>Dupire Local Volatility</b><br>"
                    "Strike: $%{x:,.1f}<br>"
                    "Tenor: %{y:.0f} DTE<br>"
                    "Local Vol: %{z:.2f}%<extra></extra>"
                ),
            )
        ]
    )

    return _apply_clean_3d_layout(
        fig=fig,
        x_title=f"Strike ($ {symbol})",
        y_title="Tenor (DTE)",
        z_title="Local Vol (%)",
        height=540,
    )


def create_3d_density_surface_plotly(
    engine: DualPINNOptionsEngine,
    signals_df: pd.DataFrame,
    symbol: str = "NDX",
) -> go.Figure:
    sub = signals_df[signals_df["symbol"] == symbol].copy()
    spot = float(sub["spot_price"].iloc[0])
    r = float(sub["risk_free_rate"].iloc[0])
    q = float(sub["dividend_yield"].iloc[0])

    grid = evaluate_dense_surface_grid(engine, spot=spot, r=r, q=q)

    fig = go.Figure(
        data=[
            go.Surface(
                x=grid["K_grid"],
                y=grid["dte_grid"],
                z=grid["bl_density"],
                colorscale=DENSITY_COLORSCALE,
                opacity=0.95,
                contours=dict(
                    x=dict(show=True, color="rgba(255,255,255,0.07)", width=1),
                    y=dict(show=True, color="rgba(255,255,255,0.07)", width=1),
                ),
                colorbar=dict(
                    title=dict(text="Density p(K)", font=dict(size=11, color="#94A3B8")),
                    tickfont=dict(size=10, color="#94A3B8"),
                    thickness=14,
                    len=0.65,
                    x=0.96,
                    outlinecolor="#1E293B",
                    outlinewidth=1,
                ),
                hovertemplate=(
                    "<b>Risk-Neutral Density</b><br>"
                    "Strike: $%{x:,.1f}<br>"
                    "Tenor: %{y:.0f} DTE<br>"
                    "p(K): %{z:.6f}<extra></extra>"
                ),
            )
        ]
    )

    return _apply_clean_3d_layout(
        fig=fig,
        x_title=f"Strike ($ {symbol})",
        y_title="Tenor (DTE)",
        z_title="Density p(K)",
        height=540,
    )


def create_volatility_smile_slices_plotly(
    engine: DualPINNOptionsEngine,
    signals_df: pd.DataFrame,
    symbol: str = "NDX",
) -> go.Figure:
    """
    Clean 2D Implied Volatility Smile Cross-Sections with concise horizontal legend
    and zero title/legend overlap.
    """
    sub = signals_df[(signals_df["symbol"] == symbol) & (signals_df["option_type"] == "call")].copy()
    tenors = [30, 60, 120, 180]
    palette = {30: "#38BDF8", 60: "#00F5D4", 120: "#818CF8", 180: "#FBBF24"}

    fig = go.Figure()

    for dte in tenors:
        slice_df = sub[sub["maturity_days"] == dte].sort_values("strike")
        if len(slice_df) == 0:
            continue
        col = palette[dte]
        fig.add_trace(
            go.Scatter(
                x=slice_df["strike"],
                y=slice_df["pinn_iv"] * 100.0,
                mode="lines",
                name=f"{dte} DTE",
                line=dict(color=col, width=2.5),
            )
        )
        fair_s = slice_df[slice_df["detected_signal"] == "FAIR_PRICED"]
        fig.add_trace(
            go.Scatter(
                x=fair_s["strike"],
                y=fair_s["market_iv"] * 100.0,
                mode="markers",
                name=f"Quotes ({dte}D)",
                marker=dict(color=col, size=5, opacity=0.65),
                showlegend=False,
            )
        )

    anom_all = sub[(sub["maturity_days"].isin(tenors)) & (sub["detected_signal"] != "FAIR_PRICED")]
    if len(anom_all) > 0:
        fig.add_trace(
            go.Scatter(
                x=anom_all["strike"],
                y=anom_all["market_iv"] * 100.0,
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(
                    color="#F43F5E",
                    size=9,
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=1.2),
                ),
                text=anom_all["exchange_code"] + " • " + anom_all["detected_signal"] + " (" + anom_all["maturity_days"].astype(str) + "D)",
                hovertemplate="<b>%{text}</b><br>Strike: $%{x:,.1f}<br>Quoted IV: %{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=None,
        font=dict(family=FONT_FAMILY, color="#CBD5E1", size=11.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B111E",
        height=440,
        margin=dict(l=55, r=25, b=50, t=45),
        xaxis=dict(title=f"Strike Price ($ {symbol})", gridcolor="#1E293B", zerolinecolor="#334155"),
        yaxis=dict(title="Implied Volatility (%)", gridcolor="#1E293B", zerolinecolor="#334155"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
    )
    return fig


def create_greeks_comparison_plotly(
    engine: DualPINNOptionsEngine,
    spot: float = 500.0,
    tau: float = 60.0 / 365.0,
    r: float = 0.045,
    q: float = 0.0065,
    sigma: float = 0.20,
) -> go.Figure:
    strikes = np.linspace(0.82 * spot, 1.18 * spot, 90)
    S_t = torch.full((90,), float(spot), dtype=torch.float32)
    K_t = torch.tensor(strikes, dtype=torch.float32)
    tau_t = torch.full((90,), float(tau), dtype=torch.float32)
    r_t = torch.full((90,), float(r), dtype=torch.float32)
    q_t = torch.full((90,), float(q), dtype=torch.float32)
    sig_t = torch.full((90,), float(sigma), dtype=torch.float32)

    pinn_g = engine.pricing_net.compute_autograd_greeks(
        S_t, K_t, tau_t, r_t, q_t, sig_t, option_type="call", create_graph=False
    )
    ana_g = bsm_greeks(spot, strikes, tau, r, q, sigma, option_type="call")

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
        subplot_titles=(
            "DELTA (Δ = ∂V/∂S)",
            "GAMMA (Γ = ∂²V/∂S²)",
            "VEGA (V = ∂V/∂σ PER 1% VOL)",
            "THETA (Θ = -∂V/∂τ DAILY DECAY)",
        ),
    )

    pairs = [
        (1, 1, pinn_g["delta"].numpy(), ana_g["delta"], "Delta (Δ)"),
        (1, 2, pinn_g["gamma"].numpy(), ana_g["gamma"], "Gamma (Γ)"),
        (2, 1, pinn_g["vega"].numpy() / 100.0, ana_g["vega"] / 100.0, "Vega ($ / 1% Vol)"),
        (2, 2, pinn_g["theta"].numpy() / 365.0, ana_g["theta"] / 365.0, "Theta ($ / Day)"),
    ]

    for row_i, col_i, pinn_arr, ana_arr, label in pairs:
        show_leg = row_i == 1 and col_i == 1
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=ana_arr,
                mode="lines",
                name="Analytical BSM Closed-Form",
                line=dict(color="#00D4FF", width=2.5),
                showlegend=show_leg,
            ),
            row=row_i,
            col=col_i,
        )
        fig.add_trace(
            go.Scatter(
                x=strikes[::2],
                y=pinn_arr[::2],
                mode="markers",
                name="PyTorch Autograd PINN",
                marker=dict(color="#10B981", size=5.5, symbol="circle-open", line=dict(width=1.5)),
                showlegend=show_leg,
            ),
            row=row_i,
            col=col_i,
        )
        fig.update_xaxes(
            title_text="Strike Price ($)",
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            row=row_i,
            col=col_i,
        )
        fig.update_yaxes(
            title_text=label,
            gridcolor="#1E293B",
            zerolinecolor="#334155",
            row=row_i,
            col=col_i,
        )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11.5, color="#94A3B8", family=FONT_FAMILY)

    fig.update_layout(
        title=None,
        font=dict(family=FONT_FAMILY, color="#CBD5E1", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B111E",
        height=540,
        margin=dict(l=50, r=20, b=40, t=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
    )
    return fig


def create_no_arbitrage_diagnostics_plotly(
    engine: DualPINNOptionsEngine,
    spot: float = 20500.0,
    r: float = 0.045,
    q: float = 0.008,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=(
            "BUTTERFLY CONVEXITY: DURRLEMAN FUNCTION g(k, τ) ≥ 0",
            "CALENDAR MONOTONICITY: TOTAL VARIANCE w(k, τ) = σ²τ",
        ),
    )

    k_vals = np.linspace(-0.18, 0.18, 80)
    dte_slices = [14, 30, 60, 120, 270, 365]
    colors = ["#38BDF8", "#00F5D4", "#3B82F6", "#818CF8", "#F59E0B", "#10B981"]

    for dte, col in zip(dte_slices, colors):
        tau_val = dte / 365.0
        k_t = torch.tensor(k_vals, dtype=torch.float32)
        tau_t = torch.full_like(k_t, tau_val)
        out = engine.vol_net.evaluate_surface_and_arbitrage(k_t, tau_t, create_graph=False)

        fig.add_trace(
            go.Scatter(
                x=k_vals,
                y=out["durrleman_g"].detach().numpy(),
                mode="lines",
                name=f"{dte}D",
                line=dict(color=col, width=2.4),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=k_vals,
                y=out["w"].detach().numpy(),
                mode="lines",
                name=f"{dte}D",
                line=dict(color=col, width=2.4),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#F43F5E",
        annotation_text="Arbitrage Boundary g(k) = 0",
        annotation_font=dict(color="#F43F5E", size=10),
        row=1,
        col=1,
    )

    for annotation in fig["layout"]["annotations"]:
        if "text" in annotation and "Boundary" not in annotation["text"]:
            annotation["font"] = dict(size=11.5, color="#94A3B8", family=FONT_FAMILY)

    fig.update_xaxes(
        title_text="Log-Moneyness k = ln(K / F_T)",
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Durrleman Density Factor g(k, τ)",
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Log-Moneyness k = ln(K / F_T)",
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Total Implied Variance w(k, τ)",
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        row=1,
        col=2,
    )

    fig.update_layout(
        title=None,
        font=dict(family=FONT_FAMILY, color="#CBD5E1", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B111E",
        height=430,
        margin=dict(l=50, r=20, b=45, t=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
    )
    return fig


def create_exchange_anomalies_bar_plotly(ex_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=(
            "ANOMALIES BY SIGNAL TYPE & NASDAQ VENUE",
            "ACTIONABLE DOLLAR EDGE CAPTURED BY VENUE ($ USD)",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=ex_df["exchange"],
            y=ex_df["buy_signals"],
            name="Buy Undervalued",
            marker_color="#10B981",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ex_df["exchange"],
            y=ex_df["sell_signals"],
            name="Sell Overvalued",
            marker_color="#F43F5E",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ex_df["exchange"],
            y=ex_df["arb_violations"],
            name="Butterfly / Calendar Arb",
            marker_color="#F59E0B",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=ex_df["exchange"],
            y=ex_df["total_actionable_edge_usd"],
            name="Actionable Edge ($)",
            marker_color="#0066FF",
            text=[f"${v:,.0f}" for v in ex_df["total_actionable_edge_usd"]],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11.5, color="#94A3B8", family=FONT_FAMILY)

    fig.update_layout(
        title=None,
        barmode="stack",
        font=dict(family=FONT_FAMILY, color="#CBD5E1", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B111E",
        height=330,
        margin=dict(l=45, r=20, b=35, t=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15, 23, 42, 0.8)",
        ),
    )
    fig.update_xaxes(gridcolor="#1E293B", row=1, col=1)
    fig.update_yaxes(title_text="Flagged Contracts", gridcolor="#1E293B", row=1, col=1)
    fig.update_xaxes(gridcolor="#1E293B", row=1, col=2)
    fig.update_yaxes(title_text="Actionable Edge ($ USD)", gridcolor="#1E293B", row=1, col=2)
    return fig


def save_static_diagnostic_report(
    engine: DualPINNOptionsEngine,
    signals_df: pd.DataFrame,
    output_path: str = "outputs/pinn_nasdaq_diagnostic_report.png",
) -> Path:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 11), dpi=160)
    fig.patch.set_facecolor("#070B14")

    ndx_df = signals_df[signals_df["symbol"] == "NDX"].copy()
    spot = float(ndx_df["spot_price"].iloc[0])
    r = float(ndx_df["risk_free_rate"].iloc[0])
    q = float(ndx_df["dividend_yield"].iloc[0])
    grid = evaluate_dense_surface_grid(engine, spot=spot, r=r, q=q)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_facecolor("#070B14")
    surf1 = ax1.plot_surface(
        grid["K_grid"],
        grid["dte_grid"],
        grid["iv_pct"],
        cmap="viridis",
        alpha=0.82,
        edgecolor="none",
    )
    anom = ndx_df[ndx_df["detected_signal"] != "FAIR_PRICED"]
    ax1.scatter(
        anom["strike"],
        anom["maturity_days"],
        anom["market_iv"] * 100.0,
        color="#F43F5E",
        s=42,
        marker="D",
        label="Flagged Exchange Anomaly",
    )
    ax1.set_title("NDX 3D Implied Volatility Surface & Mispriced Quotes", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Strike K ($)")
    ax1.set_ylabel("Maturity (DTE)")
    ax1.set_zlabel("Implied Vol (%)")
    ax1.view_init(elev=26, azim=-52)
    ax1.legend(loc="upper left", fontsize=8)
    fig.colorbar(surf1, ax=ax1, shrink=0.55, pad=0.08, label="IV (%)")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.set_facecolor("#070B14")
    surf2 = ax2.plot_surface(
        grid["K_grid"],
        grid["dte_grid"],
        grid["local_vol_pct"],
        cmap="plasma",
        alpha=0.88,
        edgecolor="none",
    )
    ax2.set_title("Dupire Local Volatility Surface σ_loc(K, τ)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Strike K ($)")
    ax2.set_ylabel("Maturity (DTE)")
    ax2.set_zlabel("Local Vol (%)")
    ax2.view_init(elev=26, azim=-52)
    fig.colorbar(surf2, ax=ax2, shrink=0.55, pad=0.08, label="Local Vol (%)")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#0B111E")
    strikes_q = np.linspace(420.0, 580.0, 80)
    S_t = torch.full((80,), 500.0, dtype=torch.float32)
    K_t = torch.tensor(strikes_q, dtype=torch.float32)
    tau_t = torch.full((80,), 60.0 / 365.0, dtype=torch.float32)
    r_t = torch.full((80,), 0.045, dtype=torch.float32)
    q_t = torch.full((80,), 0.0065, dtype=torch.float32)
    sig_t = torch.full((80,), 0.20, dtype=torch.float32)

    pinn_g = engine.pricing_net.compute_autograd_greeks(S_t, K_t, tau_t, r_t, q_t, sig_t, option_type="call")
    ana_g = bsm_greeks(500.0, strikes_q, 60.0 / 365.0, 0.045, 0.0065, 0.20, option_type="call")

    ax3.plot(strikes_q, ana_g["delta"], color="#00D4FF", lw=2.5, label="Analytical Δ (Closed-Form)")
    ax3.scatter(strikes_q[::3], pinn_g["delta"].numpy()[::3], color="#F43F5E", s=28, zorder=5, label="PINN Autograd Δ")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(strikes_q, ana_g["gamma"], color="#10B981", lw=2.0, ls="--", label="Analytical Γ (Closed-Form)")
    ax3_twin.scatter(strikes_q[::3], pinn_g["gamma"].numpy()[::3], color="#F59E0B", s=24, marker="^", zorder=5, label="PINN Autograd Γ")

    ax3.set_title("Sub-Millisecond PyTorch Autograd Greeks vs. Closed-Form BSM (QQQ)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Strike Price K ($)")
    ax3.set_ylabel("Delta (Δ = ∂V/∂S)", color="#00D4FF")
    ax3_twin.set_ylabel("Gamma (Γ = ∂²V/∂S²)", color="#10B981")
    ax3.grid(alpha=0.2)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#0B111E")
    k_vals = np.linspace(-0.16, 0.16, 70)
    palette = ["#38BDF8", "#00F5D4", "#818CF8", "#F43F5E", "#F59E0B"]
    for dte, col in zip([14, 30, 90, 180, 365], palette):
        k_t = torch.tensor(k_vals, dtype=torch.float32)
        tau_t = torch.full_like(k_t, dte / 365.0)
        out = engine.vol_net.evaluate_surface_and_arbitrage(k_t, tau_t, create_graph=False)
        ax4.plot(k_vals, out["w"].detach().numpy(), color=col, lw=2.2, label=f"Total Variance w(k) @ {dte} DTE")

    ax4.set_title("Calendar No-Arbitrage: Monotonic Total Variance Slices w(k, τ) = σ²τ", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Log-Moneyness k = ln(K / F_T)")
    ax4.set_ylabel("Total Implied Variance w(k, τ)")
    ax4.grid(alpha=0.2)
    ax4.legend(loc="upper right", fontsize=8.5)

    plt.suptitle(
        "Nasdaq-100 (NDX / QQQ) Physics-Informed Neural Network (PINN) Options & Volatility Engine",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_file), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file
