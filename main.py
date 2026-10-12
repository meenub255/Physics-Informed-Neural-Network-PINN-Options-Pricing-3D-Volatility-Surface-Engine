"""
End-to-End CLI Pipeline for the Nasdaq-100 (NDX / QQQ) Physics-Informed Neural Network (PINN)
Options Pricing & 3D Volatility Surface Engine.

Executes:
  1. SQL Option Chain Data Warehouse initialization across Nasdaq's 6 US Options Exchanges
     (PHLX, ISE, GEMX, MRX, NOM, BX) with calibrated SVI surfaces and injected anomalies.
  2. Dual-Network PyTorch PINN Training enforcing the Black-Scholes-Merton PDE and
     Butterfly (Durrleman g(k) >= 0, d^2C/dK^2 >= 0) + Calendar (dw/dtau >= 0) no-arbitrage bounds.
  3. Sub-millisecond inference & torch.autograd Greek parity benchmark (Delta, Gamma, Vega, Theta, Rho).
  4. Multi-Exchange Mispricing & Arbitrage Detection persisted back to SQLite.
  5. Export of Interactive 3D Plotly HTML dashboards and high-resolution Matplotlib diagnostic reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.mispricing_detector import MispricingArbitrageDetector
from src.engine.trainer import PINNTrainer
from src.sql.database import OptionChainDatabase
from src.visualization.plots import (
    create_3d_iv_surface_plotly,
    create_3d_local_vol_surface_plotly,
    create_greeks_comparison_plotly,
    create_no_arbitrage_diagnostics_plotly,
    save_static_diagnostic_report,
)


def run_pipeline(epochs: int = 160, force_retrain: bool = True) -> None:
    print("=" * 88)
    print(" NASDAQ-100 (NDX / QQQ) PINN OPTIONS PRICING & 3D VOLATILITY SURFACE ENGINE")
    print(" Exchanges Modeled: Nasdaq PHLX | ISE | GEMX | MRX | NOM | BX Options")
    print("=" * 88)

    # 1. Seed SQL Option Chain Warehouse
    print("\n[1/5] Initializing SQLite Multi-Exchange Option Chain Warehouse...")
    db = OptionChainDatabase("data/nasdaq_options_warehouse.db")
    seed_stats = db.seed_nasdaq_data(seed=42, force_refresh=True)
    chain_df = db.get_option_chain_df()
    print(
        f"  -> Seeded {seed_stats['total_contracts']} NDX & QQQ option contracts across "
        f"{seed_stats['exchanges']} Nasdaq exchanges ({seed_stats['injected_anomalies']} injected anomalies)."
    )

    # 2. Train or Load Dual-PINN Engine
    trainer = PINNTrainer(device="cpu", seed=42)
    ckpt_path = Path("models/pinn_engine_checkpoint.pt")
    if ckpt_path.exists() and not force_retrain:
        print(f"\n[2/5] Loading pre-trained Dual-PINN checkpoint from {ckpt_path}...")
        history = trainer.load_checkpoint(str(ckpt_path))
    else:
        print(
            f"\n[2/5] Training Dual-PINN Engine ({epochs} epochs) with BSM PDE + "
            f"Butterfly & Calendar No-Arbitrage Constraints..."
        )
        history = trainer.train(chain_df=chain_df, epochs=epochs, verbose=True)
        trainer.save_checkpoint(str(ckpt_path), history=history)
        print(f"  -> Training completed in {history.training_time_sec:.2f}s. Checkpoint saved to {ckpt_path}.")

    # 3. Run Sub-Millisecond Inference, Autograd Greeks, and Mispricing Scanner
    print("\n[3/5] Running Sub-Millisecond PINN Valuation, Autograd Greeks & Arbitrage Scanner...")
    detector = MispricingArbitrageDetector(engine=trainer.engine, device="cpu")
    signals_df, metrics = detector.analyze_option_chain(chain_df)
    saved_count = db.save_pinn_signals(signals_df)
    print(f"  -> Persisted {saved_count} evaluated PINN option signal records to SQL table `pinn_signals`.")

    # 4. Print Quantitative Benchmark & SQL Exchange Summary
    print("\n[4/5] Quantitative Benchmark & Nasdaq Exchange Summary:")
    print("-" * 88)
    print(f"  * Batch Pricing + IV Surface Latency (128 opts) : {metrics.single_batch_pricing_latency_ms:.3f} ms "
          f"({metrics.per_contract_latency_us:.2f} us/option)")
    print(f"  * Full Autograd Greeks Latency (Delta..Gamma)   : {metrics.single_batch_greeks_latency_ms:.3f} ms")
    print(f"  * 3D Implied Volatility Surface RMSE (Clean)    : {metrics.iv_surface_rmse_bps:.2f} bps")
    print(f"  * Normalized BSM PDE Residual MAE               : {metrics.mean_abs_pde_residual:.2e}")
    print(f"  * Min Durrleman Butterfly Condition g(k, tau)   : {metrics.min_pinn_durrleman_g:+.4f} (>= 0 Arbitrage-Free)")
    print(f"  * Min Calendar Variance Slope dw/dtau           : {metrics.min_pinn_calendar_slope:+.4f} (>= 0 Arbitrage-Free)")
    print("-" * 88)
    print("  * PyTorch Autograd Greeks vs. Closed-Form Analytical Black-Scholes Parity:")
    print(f"      Delta MAE : {metrics.delta_mae:.6f}   |   Gamma MAE : {metrics.gamma_mae:.7f}")
    print(f"      Vega MAE  : {metrics.vega_mae_pct:.6f}   |   Theta MAE : {metrics.theta_mae_daily:.6f} ($/day)")
    print(f"      Rho MAE   : {metrics.rho_mae:.6f}")
    print("-" * 88)
    print("  * Mispricing & Arbitrage Anomaly Detection Performance:")
    print(f"      Flagged Anomalies : {metrics.total_detected_anomalies} contracts")
    print(f"      Precision         : {metrics.anomaly_precision * 100.0:.1f}%")
    print(f"      Recall            : {metrics.anomaly_recall * 100.0:.1f}%")
    print(f"      F1-Score          : {metrics.anomaly_f1 * 100.0:.1f}%")
    print(f"      Actionable Edge   : ${metrics.total_actionable_edge_usd:,.2f}")
    print("-" * 88)

    print("\n  [SQL Aggregation Across Nasdaq's 6 US Options Exchanges]")
    ex_summary = db.get_exchange_summary_sql()
    print(ex_summary.to_string(index=False))

    print("\n  [Top 12 High-Conviction Mispriced NDX & QQQ Contracts Flagged by PINN]")
    top_signals = db.get_top_actionable_signals_sql(limit=12)
    display_cols = [
        "symbol",
        "exchange_code",
        "type",
        "strike",
        "dte",
        "mkt_mid",
        "pinn_fair",
        "mkt_iv_pct",
        "pinn_iv_pct",
        "iv_diff_bps",
        "delta",
        "gamma",
        "detected_signal",
        "edge_usd",
    ]
    print(top_signals[display_cols].to_string(index=False))

    # 5. Export Visualizations
    print("\n[5/5] Generating Interactive 3D Plotly HTML Dashboards & Static Diagnostic Report...")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_static_diagnostic_report(
        trainer.engine, signals_df, output_path=str(out_dir / "pinn_nasdaq_diagnostic_report.png")
    )
    fig_iv_ndx = create_3d_iv_surface_plotly(trainer.engine, signals_df, symbol="NDX")
    fig_iv_ndx.write_html(str(out_dir / "ndx_3d_iv_surface.html"), include_plotlyjs="cdn")

    fig_loc_ndx = create_3d_local_vol_surface_plotly(trainer.engine, signals_df, symbol="NDX")
    fig_loc_ndx.write_html(str(out_dir / "ndx_3d_local_vol_and_density.html"), include_plotlyjs="cdn")

    fig_greeks = create_greeks_comparison_plotly(trainer.engine)
    fig_greeks.write_html(str(out_dir / "autograd_greeks_parity.html"), include_plotlyjs="cdn")

    fig_no_arb = create_no_arbitrage_diagnostics_plotly(trainer.engine)
    fig_no_arb.write_html(str(out_dir / "no_arbitrage_diagnostics.html"), include_plotlyjs="cdn")

    print(f"  -> Saved static 4-panel report : {png_path}")
    print(f"  -> Saved interactive 3D HTMLs  : {out_dir.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nasdaq-100 PINN Options Pricing & Volatility Surface Engine")
    parser.add_argument("--epochs", type=int, default=160, help="Number of PINN training epochs")
    parser.add_argument("--use-cached", action="store_true", help="Load existing model checkpoint if available")
    args = parser.parse_args()
    run_pipeline(epochs=args.epochs, force_retrain=not args.use_cached)
