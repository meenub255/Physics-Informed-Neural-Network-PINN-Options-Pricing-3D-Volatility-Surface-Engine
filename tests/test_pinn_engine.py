"""
Comprehensive Unit & Integration Tests for the Physics-Informed Neural Network (PINN)
Options Pricing & 3D Volatility Surface Engine.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analytics.black_scholes import (
    NDX_SVI_PARAMS,
    bsm_greeks,
    bsm_price,
    durrleman_butterfly_condition,
    implied_volatility_vec,
    svi_total_variance,
)
from src.engine.mispricing_detector import MispricingArbitrageDetector
from src.models.losses import PINNOptionsLoss
from src.models.pinn_net import DualPINNOptionsEngine
from src.sql.database import NASDAQ_EXCHANGES, OptionChainDatabase


def test_bsm_put_call_parity_and_iv_inversion():
    S = np.array([20500.0, 500.0, 500.0])
    K = np.array([20000.0, 480.0, 530.0])
    tau = np.array([0.25, 0.50, 0.10])
    r = np.array([0.045, 0.045, 0.045])
    q = np.array([0.008, 0.0065, 0.0065])
    sigma = np.array([0.19, 0.22, 0.17])

    call_p = bsm_price(S, K, tau, r, q, sigma, "call")
    put_p = bsm_price(S, K, tau, r, q, sigma, "put")

    # Verify Put-Call Parity: C - P == S * exp(-q*tau) - K * exp(-r*tau)
    lhs = call_p - put_p
    rhs = S * np.exp(-q * tau) - K * np.exp(-r * tau)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-8)

    # Verify Implied Volatility inversion recovers exact sigma
    recovered_iv = implied_volatility_vec(call_p, S, K, tau, r, q, "call")
    np.testing.assert_allclose(recovered_iv, sigma, rtol=1e-5, atol=1e-5)


def test_svi_surface_is_arbitrage_free():
    k_grid = np.linspace(-0.25, 0.25, 100)
    for dte in [7, 30, 90, 180, 365]:
        tau = dte / 365.0
        svi = svi_total_variance(k_grid, tau, params=NDX_SVI_PARAMS)
        g_k = durrleman_butterfly_condition(k_grid, svi["w"], svi["dw_dk"], svi["d2w_dk2"])
        # Butterfly no-arbitrage: g(k) > 0 everywhere
        assert np.all(g_k > 0.0), f"Butterfly violation found at {dte} DTE"
        # Calendar no-arbitrage: dw/dtau > 0 everywhere
        assert np.all(svi["dw_dtau"] > 0.0), f"Calendar violation found at {dte} DTE"


def test_pinn_autograd_greeks_and_pde_residual():
    engine = DualPINNOptionsEngine()
    engine.eval()

    S = torch.tensor([20500.0, 500.0, 500.0], dtype=torch.float32)
    K = torch.tensor([20500.0, 500.0, 480.0], dtype=torch.float32)
    tau = torch.tensor([0.25, 0.25, 0.50], dtype=torch.float32)
    r = torch.tensor([0.045, 0.045, 0.045], dtype=torch.float32)
    q = torch.tensor([0.008, 0.0065, 0.0065], dtype=torch.float32)
    sigma = torch.tensor([0.185, 0.185, 0.21], dtype=torch.float32)

    pinn_out = engine.pricing_net.compute_autograd_greeks(
        S, K, tau, r, q, sigma, option_type="call", create_graph=False
    )
    ana_out = bsm_greeks(
        S.numpy(), K.numpy(), tau.numpy(), r.numpy(), q.numpy(), sigma.numpy(), option_type="call"
    )

    assert pinn_out["delta"].shape == (3,)
    assert pinn_out["gamma"].shape == (3,)
    assert torch.all(pinn_out["gamma"] > 0.0)
    assert torch.all(pinn_out["bl_density"] > 0.0)
    np.testing.assert_allclose(pinn_out["delta"].numpy(), ana_out["delta"], atol=0.02)
    assert float(torch.max(torch.abs(pinn_out["pde_residual"]))) < 1e-2


def test_vol_surface_pinn_and_loss_backward():
    engine = DualPINNOptionsEngine()
    loss_fn = PINNOptionsLoss()

    k = torch.linspace(-0.15, 0.15, 32, dtype=torch.float32)
    tau = torch.linspace(0.05, 1.0, 32, dtype=torch.float32)
    target_iv = torch.full_like(k, 0.19)

    vol_losses = loss_fn.compute_surface_loss(engine, k, tau, target_iv, k, tau)
    assert vol_losses["total_vol_loss"].requires_grad
    vol_losses["total_vol_loss"].backward()

    grad_norms = [p.grad.norm().item() for p in engine.vol_net.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and sum(grad_norms) > 0.0


def test_sql_warehouse_and_sub_millisecond_detector(tmp_path):
    db_file = tmp_path / "test_nasdaq.db"
    db = OptionChainDatabase(str(db_file))
    stats = db.seed_nasdaq_data(seed=42, force_refresh=True)

    assert stats["exchanges"] == len(NASDAQ_EXCHANGES) == 6
    assert stats["total_contracts"] > 500
    assert stats["injected_anomalies"] > 20

    df = db.get_option_chain_df(symbol="QQQ")
    assert len(df) > 200
    assert set(df["exchange_code"].unique()) == {"PHLX", "ISE", "GEMX", "MRX", "NOM", "BX"}

    engine = DualPINNOptionsEngine()
    detector = MispricingArbitrageDetector(engine=engine, device="cpu")
    signals_df, metrics = detector.analyze_option_chain(df)

    assert metrics.single_batch_pricing_latency_ms < 25.0
    assert metrics.min_pinn_durrleman_g > 0.0
    assert metrics.min_pinn_calendar_slope > 0.0

    saved = db.save_pinn_signals(signals_df)
    assert saved == len(signals_df)
    ex_summary = db.get_exchange_summary_sql(symbol="QQQ")
    assert len(ex_summary) == 6
