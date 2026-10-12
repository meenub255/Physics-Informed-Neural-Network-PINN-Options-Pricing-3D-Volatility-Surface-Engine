"""
Institutional 3D Plotly & Publication-Quality Matplotlib Visualizations for the PINN Options Engine.
"""

from .plots import (
    evaluate_dense_surface_grid,
    create_3d_iv_surface_plotly,
    create_3d_local_vol_surface_plotly,
    create_3d_density_surface_plotly,
    create_volatility_smile_slices_plotly,
    create_greeks_comparison_plotly,
    create_no_arbitrage_diagnostics_plotly,
    create_exchange_anomalies_bar_plotly,
    save_static_diagnostic_report,
)

__all__ = [
    "evaluate_dense_surface_grid",
    "create_3d_iv_surface_plotly",
    "create_3d_local_vol_surface_plotly",
    "create_3d_density_surface_plotly",
    "create_volatility_smile_slices_plotly",
    "create_greeks_comparison_plotly",
    "create_no_arbitrage_diagnostics_plotly",
    "create_exchange_anomalies_bar_plotly",
    "save_static_diagnostic_report",
]
