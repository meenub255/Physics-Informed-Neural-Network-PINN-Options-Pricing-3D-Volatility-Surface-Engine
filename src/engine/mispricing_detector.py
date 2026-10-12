"""
Sub-Millisecond Mispricing & Arbitrage Detection Engine for Nasdaq-100 (NDX / QQQ) Option Chains.
Computes PINN fair values, 3D Implied & Dupire Local Volatility Surfaces, exact Autograd Greeks,
and flags actionable Butterfly, Calendar, and Bid/Ask Mispricing anomalies across Nasdaq's 6 exchanges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.analytics.black_scholes import bsm_greeks
from src.models.pinn_net import DualPINNOptionsEngine


@dataclass
class BenchmarkMetrics:
    single_batch_pricing_latency_ms: float
    single_batch_greeks_latency_ms: float
    per_contract_latency_us: float
    iv_surface_rmse_bps: float
    price_mae_pct_of_spot: float
    delta_mae: float
    gamma_mae: float
    vega_mae_pct: float
    theta_mae_daily: float
    rho_mae: float
    mean_abs_pde_residual: float
    min_pinn_durrleman_g: float
    min_pinn_calendar_slope: float
    anomaly_precision: float
    anomaly_recall: float
    anomaly_f1: float
    total_detected_anomalies: int
    total_actionable_edge_usd: float


class MispricingArbitrageDetector:
    """
    Scans multi-exchange NDX & QQQ option chains using the trained DualPINNOptionsEngine
    to detect mispriced contracts and structural arbitrage violations.
    """

    def __init__(self, engine: DualPINNOptionsEngine, device: str = "cpu") -> None:
        self.engine = engine.to(device)
        self.engine.eval()
        self.device = torch.device(device)

    def benchmark_sub_millisecond_latency(self, batch_size: int = 128, n_runs: int = 50) -> Dict[str, float]:
        """
        Measure exact wall-clock inference latency for:
          1. Forward pricing + IV surface evaluation on a batch of contracts
          2. Full torch.autograd Greek + PDE residual evaluation
        """
        S = torch.full((batch_size,), 500.0, dtype=torch.float32, device=self.device)
        K = torch.linspace(430.0, 570.0, batch_size, dtype=torch.float32, device=self.device)
        tau = torch.full((batch_size,), 0.25, dtype=torch.float32, device=self.device)
        r = torch.full((batch_size,), 0.045, dtype=torch.float32, device=self.device)
        q = torch.full((batch_size,), 0.0065, dtype=torch.float32, device=self.device)

        # Warm-up pass
        with torch.no_grad():
            sig = self.engine.predict_iv(S, K, tau, r, q)
            _ = self.engine.pricing_net(S, K, tau, r, q, sig, option_type="call")

        # 1. Pricing + IV Surface latency
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                sig = self.engine.predict_iv(S, K, tau, r, q)
                _ = self.engine.pricing_net(S, K, tau, r, q, sig, option_type="call")
        pricing_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

        # 2. Full Autograd Greeks (Delta, Gamma, Vega, Theta, Rho, Dual-Gamma) latency
        _ = self.engine.pricing_net.compute_autograd_greeks(S, K, tau, r, q, sig, option_type="call")
        t1 = time.perf_counter()
        for _ in range(n_runs):
            _ = self.engine.pricing_net.compute_autograd_greeks(S, K, tau, r, q, sig, option_type="call")
        greeks_ms = ((time.perf_counter() - t1) / n_runs) * 1000.0

        return {
            "batch_size": float(batch_size),
            "pricing_latency_ms": pricing_ms,
            "greeks_latency_ms": greeks_ms,
            "per_contract_us": (pricing_ms * 1000.0) / batch_size,
        }

    def analyze_option_chain(
        self,
        chain_df: pd.DataFrame,
        iv_threshold_bps: float = 135.0,
        z_score_threshold: float = 1.65,
    ) -> Tuple[pd.DataFrame, BenchmarkMetrics]:
        """
        Evaluate an entire SQL option chain DataFrame, computing PINN implied vol, Dupire local vol,
        PINN fair value, exact Autograd Greeks, and classifying mispricing & arbitrage signals.
        """
        df = chain_df.copy().reset_index(drop=True)

        S_t = torch.tensor(df["spot_price"].values, dtype=torch.float32, device=self.device)
        K_t = torch.tensor(df["strike"].values, dtype=torch.float32, device=self.device)
        tau_t = torch.tensor(df["tau"].values, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(df["risk_free_rate"].values, dtype=torch.float32, device=self.device)
        q_t = torch.tensor(df["dividend_yield"].values, dtype=torch.float32, device=self.device)
        k_t = torch.tensor(df["log_moneyness"].values, dtype=torch.float32, device=self.device)
        is_call_t = torch.tensor(
            (df["option_type"].str.lower() == "call").astype(np.float32).values,
            dtype=torch.float32,
            device=self.device,
        )

        # Evaluate 3D Volatility Surface & Arbitrage Derivatives via torch.autograd
        surf_out = self.engine.vol_net.evaluate_surface_and_arbitrage(k_t, tau_t, create_graph=False)
        pinn_iv = surf_out["sigma_imp"].detach()

        # Evaluate PINN Option Fair Value & Autograd Greeks at the PINN arbitrage-free IV
        greeks_out = self.engine.pricing_net.compute_autograd_greeks(
            S=S_t,
            K=K_t,
            tau=tau_t,
            r=r_t,
            q=q_t,
            sigma=pinn_iv,
            option_type=is_call_t,
            create_graph=False,
        )

        df["pinn_iv"] = pinn_iv.cpu().numpy()
        df["dupire_local_vol"] = surf_out["dupire_local_vol"].detach().cpu().numpy()
        df["durrleman_g"] = surf_out["durrleman_g"].detach().cpu().numpy()
        df["calendar_var_slope"] = surf_out["dw_dtau"].detach().cpu().numpy()

        df["pinn_price"] = greeks_out["price"].cpu().numpy()
        df["delta"] = greeks_out["delta"].cpu().numpy()
        df["gamma"] = greeks_out["gamma"].cpu().numpy()
        df["vega"] = greeks_out["vega"].cpu().numpy()
        df["theta"] = greeks_out["theta"].cpu().numpy()
        df["rho"] = greeks_out["rho"].cpu().numpy()
        df["bl_density"] = greeks_out["bl_density"].cpu().numpy()
        df["pde_residual"] = greeks_out["pde_residual"].cpu().numpy()

        df["market_mid"] = df["mid_price"]
        df["market_bid"] = df["bid"]
        df["market_ask"] = df["ask"]

        # Residuals: Market - PINN Equilibrium
        df["price_residual"] = df["market_mid"] - df["pinn_price"]
        df["iv_residual_bps"] = (df["market_iv"] - df["pinn_iv"]) * 10000.0
        half_spread = np.maximum(0.5 * df["spread"].values, 1e-4)
        df["spread_z_score"] = df["price_residual"] / half_spread

        # Detect Empirical Market Butterfly & Calendar Arbitrage Violations in raw exchange quotes
        df["mkt_total_var"] = (df["market_iv"] ** 2) * df["tau"]
        emp_butterfly_flag = np.zeros(len(df), dtype=bool)
        emp_calendar_flag = np.zeros(len(df), dtype=bool)

        # 1. Empirical strike second-difference check (Butterfly convexity kink)
        for (_, _, _), group in df.groupby(["symbol", "maturity_days", "option_type"]):
            if len(group) < 3:
                continue
            g_sorted = group.sort_values("strike")
            iv_vals = g_sorted["market_iv"].values
            idxs = g_sorted.index.values
            for i in range(1, len(iv_vals) - 1):
                local_iv_jump = iv_vals[i] - 0.5 * (iv_vals[i - 1] + iv_vals[i + 1])
                if local_iv_jump > 0.021:
                    emp_butterfly_flag[idxs[i]] = True

        # 2. Empirical maturity variance monotonicity check (Calendar spread violation)
        for (_, _, _), group in df.groupby(["symbol", "strike", "option_type"]):
            if len(group) < 2:
                continue
            g_sorted = group.sort_values("tau")
            w_vals = g_sorted["mkt_total_var"].values
            iv_res_vals = g_sorted["iv_residual_bps"].values
            idxs = g_sorted.index.values
            for i in range(1, len(w_vals)):
                if (w_vals[i] < w_vals[i - 1] + 0.0005) and (iv_res_vals[i] < -150.0):
                    emp_calendar_flag[idxs[i]] = True

        # Signal Classification
        signals = []
        edges = []
        for i, row in df.iterrows():
            iv_bps = float(row["iv_residual_bps"])
            z_sc = float(row["spread_z_score"])
            mkt_bid = float(row["market_bid"])
            mkt_ask = float(row["market_ask"])
            pinn_val = float(row["pinn_price"])

            if emp_butterfly_flag[i]:
                sig = "BUTTERFLY_ARB"
                edge = max(0.0, abs(float(row["price_residual"])) - 0.5 * float(row["spread"]))
            elif emp_calendar_flag[i]:
                sig = "CALENDAR_ARB"
                edge = max(0.0, abs(float(row["price_residual"])) - 0.5 * float(row["spread"]))
            elif iv_bps <= -iv_threshold_bps and z_sc <= -z_score_threshold:
                sig = "BUY_UNDERVALUED"
                edge = max(0.0, pinn_val - mkt_ask)
            elif iv_bps >= iv_threshold_bps and z_sc >= z_score_threshold:
                sig = "SELL_OVERVALUED"
                edge = max(0.0, mkt_bid - pinn_val)
            else:
                sig = "FAIR_PRICED"
                edge = 0.0

            signals.append(sig)
            edges.append(edge)

        df["detected_signal"] = signals
        df["actionable_edge_dollars"] = edges

        # Compute Analytical BSM Greeks at the PINN IV to verify Autograd Greek parity
        ana_greeks = bsm_greeks(
            S=df["spot_price"].values,
            K=df["strike"].values,
            tau=df["tau"].values,
            r=df["risk_free_rate"].values,
            q=df["dividend_yield"].values,
            sigma=df["pinn_iv"].values,
            option_type=df["option_type"].values,
        )

        # Clean (non-anomalous) quotes for surface reconstruction accuracy
        clean_mask = df["is_injected_anomaly"] == 0
        iv_rmse_bps = float(
            np.sqrt(np.mean(((df.loc[clean_mask, "pinn_iv"] - df.loc[clean_mask, "theoretical_iv"]) * 10000.0) ** 2))
        )
        price_mae_pct = float(
            np.mean(np.abs(df["pinn_price"] - ana_greeks["price"]) / df["spot_price"]) * 100.0
        )

        delta_mae = float(np.mean(np.abs(df["delta"] - ana_greeks["delta"])))
        gamma_mae = float(np.mean(np.abs(df["gamma"] - ana_greeks["gamma"])))
        vega_mae_pct = float(np.mean(np.abs(df["vega"] - ana_greeks["vega"]) / 100.0))
        theta_mae_daily = float(np.mean(np.abs(df["theta"] - ana_greeks["theta"]) / 365.0))
        rho_mae = float(np.mean(np.abs(df["rho"] - ana_greeks["rho"]) / 100.0))

        # Anomaly Detection Precision / Recall / F1
        y_true = (df["is_injected_anomaly"] == 1).values
        y_pred = (df["detected_signal"] != "FAIR_PRICED").values
        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum((~y_true) & y_pred))
        fn = int(np.sum(y_true & (~y_pred)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)

        lat_info = self.benchmark_sub_millisecond_latency(batch_size=128, n_runs=40)

        metrics = BenchmarkMetrics(
            single_batch_pricing_latency_ms=lat_info["pricing_latency_ms"],
            single_batch_greeks_latency_ms=lat_info["greeks_latency_ms"],
            per_contract_latency_us=lat_info["per_contract_us"],
            iv_surface_rmse_bps=iv_rmse_bps,
            price_mae_pct_of_spot=price_mae_pct,
            delta_mae=delta_mae,
            gamma_mae=gamma_mae,
            vega_mae_pct=vega_mae_pct,
            theta_mae_daily=theta_mae_daily,
            rho_mae=rho_mae,
            mean_abs_pde_residual=float(np.mean(np.abs(df["pde_residual"]))),
            min_pinn_durrleman_g=float(np.min(df["durrleman_g"])),
            min_pinn_calendar_slope=float(np.min(df["calendar_var_slope"])),
            anomaly_precision=precision,
            anomaly_recall=recall,
            anomaly_f1=f1,
            total_detected_anomalies=int(np.sum(y_pred)),
            total_actionable_edge_usd=float(np.sum(df["actionable_edge_dollars"])),
        )

        return df, metrics
