"""
Physics-Informed & No-Arbitrage Loss Functions for the Dual-PINN Options Engine.

Embeds:
  1. Black-Scholes-Merton PDE Residual Loss (L_PDE) via torch.autograd
  2. Terminal & Asymptotic Spatial Boundary Conditions (L_BC)
  3. Butterfly Spread No-Arbitrage Penalty:
     - Breeden-Litzenberger strike convexity: d^2C/dK^2 >= 0
     - Gatheral-Jacquier Durrleman condition: g(k, tau) >= 0
  4. Calendar Spread No-Arbitrage Penalty:
     - Total implied variance monotonicity: dw(k, tau)/dtau >= 0
  5. Robust Huber Market Calibration Loss (filters out outlier/stale quotes on exchanges)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pinn_net import DualPINNOptionsEngine


class PINNOptionsLoss(nn.Module):
    """
    Composite Physics-Informed Loss Function for training the DualPINNOptionsEngine.
    """

    def __init__(
        self,
        lambda_pde: float = 20.0,
        lambda_bc: float = 20.0,
        lambda_butterfly: float = 25.0,
        lambda_calendar: float = 25.0,
        lambda_surface_data: float = 250.0,
        lambda_price_data: float = 40.0,
        huber_delta_iv: float = 0.0025,
    ) -> None:
        super().__init__()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_butterfly = lambda_butterfly
        self.lambda_calendar = lambda_calendar
        self.lambda_surface_data = lambda_surface_data
        self.lambda_price_data = lambda_price_data
        self.huber_delta_iv = huber_delta_iv

    def compute_surface_loss(
        self,
        engine: DualPINNOptionsEngine,
        k_data: torch.Tensor,
        tau_data: torch.Tensor,
        market_iv_data: torch.Tensor,
        k_colloc: torch.Tensor,
        tau_colloc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the ArbitrageFreeVolSurfacePINN:
          - Robust Huber fit to market implied volatilities (resists injected anomalies)
          - Durrleman butterfly no-arbitrage penalty: ReLU(-g(k, tau) + 0.02)^2
          - Calendar spread no-arbitrage penalty: ReLU(-dw/dtau + 1e-4)^2
          - Smoothness regularization on d^2w/dk^2
        """
        pred_iv_data = engine.vol_net(k_data, tau_data)
        # Robust Smooth-L1 / Huber loss prevents outlier quotes from distorting the arbitrage-free surface
        loss_iv_data = F.smooth_l1_loss(pred_iv_data, market_iv_data, beta=self.huber_delta_iv)

        # Collocation points across (k, tau) domain for enforcing global no-arbitrage bounds
        surf_colloc = engine.vol_net.evaluate_surface_and_arbitrage(
            k_colloc, tau_colloc, create_graph=True
        )

        # 1. Butterfly No-Arbitrage (Durrleman condition g(k, tau) >= 0.02)
        butterfly_violation = F.relu(0.02 - surf_colloc["durrleman_g"])
        loss_butterfly = torch.mean(butterfly_violation**2)

        # 2. Calendar Spread No-Arbitrage (dw/dtau >= 0.001)
        calendar_violation = F.relu(0.001 - surf_colloc["dw_dtau"])
        loss_calendar = torch.mean(calendar_violation**2)

        total_vol_loss = (
            self.lambda_surface_data * loss_iv_data
            + self.lambda_butterfly * loss_butterfly
            + self.lambda_calendar * loss_calendar
        )

        return {
            "total_vol_loss": total_vol_loss,
            "loss_iv_data": loss_iv_data.detach(),
            "loss_butterfly_vol": loss_butterfly.detach(),
            "loss_calendar_vol": loss_calendar.detach(),
            "min_durrleman_g": torch.min(surf_colloc["durrleman_g"]).detach(),
            "min_dw_dtau": torch.min(surf_colloc["dw_dtau"]).detach(),
        }

    def compute_pricing_pde_loss(
        self,
        engine: DualPINNOptionsEngine,
        S_colloc: torch.Tensor,
        K_colloc: torch.Tensor,
        tau_colloc: torch.Tensor,
        r_colloc: torch.Tensor,
        q_colloc: torch.Tensor,
        sigma_colloc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Physics-Informed Black-Scholes-Merton PDE loss, terminal payoff boundary loss,
        and strike-convexity butterfly no-arbitrage loss (d^2C/dK^2 >= 0).
        """
        greeks_out = engine.pricing_net.compute_autograd_greeks(
            S=S_colloc,
            K=K_colloc,
            tau=tau_colloc,
            r=r_colloc,
            q=q_colloc,
            sigma=sigma_colloc,
            option_type="call",
            create_graph=True,
        )

        # 1. BSM PDE Residual: -dV/dtau + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + (r - q)*S*dV/dS - r*V = 0
        pde_res = greeks_out["pde_residual"]
        loss_pde = torch.mean(pde_res**2)

        # 2. Breeden-Litzenberger Butterfly Convexity: d^2C/dK^2 >= 0 and Gamma = d^2C/dS^2 >= 0
        butterfly_price_violation = F.relu(-greeks_out["dual_gamma"]) + F.relu(-greeks_out["gamma"])
        loss_butterfly_price = torch.mean(butterfly_price_violation**2)

        # 3. Terminal Payoff Boundary Condition at tau -> 0
        tau_zero = torch.full_like(tau_colloc, 1e-7)
        pred_terminal = engine.pricing_net(
            S_colloc, K_colloc, tau_zero, r_colloc, q_colloc, sigma_colloc, option_type="call"
        )
        exact_terminal = F.relu(S_colloc - K_colloc)
        loss_bc = torch.mean(((pred_terminal - exact_terminal) / torch.clamp(K_colloc, min=1.0)) ** 2)

        total_pde_loss = (
            self.lambda_pde * loss_pde
            + self.lambda_bc * loss_bc
            + self.lambda_butterfly * loss_butterfly_price
        )

        return {
            "total_pde_loss": total_pde_loss,
            "loss_pde": loss_pde.detach(),
            "loss_bc": loss_bc.detach(),
            "loss_butterfly_price": loss_butterfly_price.detach(),
            "max_abs_pde_residual": torch.max(torch.abs(pde_res)).detach(),
        }
