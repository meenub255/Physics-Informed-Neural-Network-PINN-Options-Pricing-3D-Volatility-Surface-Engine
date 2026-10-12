"""
Physics-Informed Neural Network (PINN) Trainer using Quasi-Monte Carlo (Latin Hypercube)
Collocation Sampling and Coupled No-Arbitrage Optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc

from src.models.losses import PINNOptionsLoss
from src.models.pinn_net import DualPINNOptionsEngine


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)
    pde_loss: List[float] = field(default_factory=list)
    iv_data_loss: List[float] = field(default_factory=list)
    butterfly_loss: List[float] = field(default_factory=list)
    calendar_loss: List[float] = field(default_factory=list)
    min_durrleman_g: List[float] = field(default_factory=list)
    min_dw_dtau: List[float] = field(default_factory=list)
    training_time_sec: float = 0.0


class PINNTrainer:
    """
    Trains the DualPINNOptionsEngine (ArbitrageFreeVolSurfacePINN + BSMPricingPINN)
    using Latin Hypercube PDE collocation points and SQL option chain quotes.
    """

    def __init__(
        self,
        engine: Optional[DualPINNOptionsEngine] = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device)
        self.engine = (engine or DualPINNOptionsEngine()).to(self.device)
        self.loss_fn = PINNOptionsLoss().to(self.device)
        self.sampler_surf = qmc.LatinHypercube(d=2, seed=seed)
        self.sampler_pde = qmc.LatinHypercube(d=5, seed=seed + 100)

    def sample_surface_collocation(self, n_points: int = 512) -> Dict[str, torch.Tensor]:
        """
        Sample Quasi-Monte Carlo collocation points (k, tau) across the log-moneyness
        and maturity domain for enforcing Durrleman butterfly and calendar no-arbitrage.
        """
        u = self.sampler_surf.random(n=n_points)
        # k in [-0.22, +0.22], tau in [0.02, 1.10]
        k = -0.22 + 0.44 * u[:, 0]
        tau = 0.02 + 1.08 * u[:, 1]
        return {
            "k": torch.tensor(k, dtype=torch.float32, device=self.device),
            "tau": torch.tensor(tau, dtype=torch.float32, device=self.device),
        }

    def sample_pde_collocation(self, n_points: int = 512) -> Dict[str, torch.Tensor]:
        """
        Sample Quasi-Monte Carlo collocation points across (S, K, tau, r, q, sigma)
        for enforcing the Black-Scholes-Merton PDE and strike convexity.
        """
        u = self.sampler_pde.random(n=n_points)
        # Normalized K = 500.0 (QQQ scale) or 20500.0 (NDX scale)
        K = np.where(u[:, 0] > 0.5, 20500.0, 500.0)
        moneyness = 0.78 + 0.44 * u[:, 0]
        S = K * moneyness
        tau = 0.02 + 1.05 * u[:, 1]
        r = 0.02 + 0.04 * u[:, 2]
        q = 0.004 + 0.012 * u[:, 3]
        sigma = 0.10 + 0.35 * u[:, 4]

        return {
            "S": torch.tensor(S, dtype=torch.float32, device=self.device),
            "K": torch.tensor(K, dtype=torch.float32, device=self.device),
            "tau": torch.tensor(tau, dtype=torch.float32, device=self.device),
            "r": torch.tensor(r, dtype=torch.float32, device=self.device),
            "q": torch.tensor(q, dtype=torch.float32, device=self.device),
            "sigma": torch.tensor(sigma, dtype=torch.float32, device=self.device),
        }

    def train(
        self,
        chain_df: pd.DataFrame,
        epochs: int = 160,
        lr_vol: float = 4e-3,
        lr_pde: float = 2e-3,
        n_colloc: int = 384,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train both the ArbitrageFreeVolSurfacePINN and BSMPricingPINN on the SQL option chain
        and interior PDE collocation domain.
        """
        t_start = time.perf_counter()
        self.engine.train()

        # Prepare market data tensors from SQL option chain
        k_data = torch.tensor(chain_df["log_moneyness"].values, dtype=torch.float32, device=self.device)
        tau_data = torch.tensor(chain_df["tau"].values, dtype=torch.float32, device=self.device)
        mkt_iv_data = torch.tensor(chain_df["market_iv"].values, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            [
                {"params": self.engine.vol_net.parameters(), "lr": lr_vol, "weight_decay": 1e-5},
                {"params": self.engine.pricing_net.parameters(), "lr": lr_pde, "weight_decay": 1e-5},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

        history = TrainingHistory()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            surf_colloc = self.sample_surface_collocation(n_points=n_colloc)
            pde_colloc = self.sample_pde_collocation(n_points=n_colloc)

            vol_metrics = self.loss_fn.compute_surface_loss(
                engine=self.engine,
                k_data=k_data,
                tau_data=tau_data,
                market_iv_data=mkt_iv_data,
                k_colloc=surf_colloc["k"],
                tau_colloc=surf_colloc["tau"],
            )

            pde_metrics = self.loss_fn.compute_pricing_pde_loss(
                engine=self.engine,
                S_colloc=pde_colloc["S"],
                K_colloc=pde_colloc["K"],
                tau_colloc=pde_colloc["tau"],
                r_colloc=pde_colloc["r"],
                q_colloc=pde_colloc["q"],
                sigma_colloc=pde_colloc["sigma"],
            )

            total_loss = vol_metrics["total_vol_loss"] + pde_metrics["total_pde_loss"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                history.epochs.append(epoch)
                history.total_loss.append(float(total_loss.item()))
                history.pde_loss.append(float(pde_metrics["loss_pde"].item()))
                history.iv_data_loss.append(float(vol_metrics["loss_iv_data"].item()))
                history.butterfly_loss.append(
                    float((vol_metrics["loss_butterfly_vol"] + pde_metrics["loss_butterfly_price"]).item())
                )
                history.calendar_loss.append(float(vol_metrics["loss_calendar_vol"].item()))
                history.min_durrleman_g.append(float(vol_metrics["min_durrleman_g"].item()))
                history.min_dw_dtau.append(float(vol_metrics["min_dw_dtau"].item()))

                if verbose and (epoch == 1 or epoch % 40 == 0 or epoch == epochs):
                    print(
                        f"  [Epoch {epoch:03d}/{epochs:03d}] "
                        f"Total: {total_loss.item():.6f} | "
                        f"PDE MSE: {pde_metrics['loss_pde'].item():.2e} | "
                        f"IV Huber: {vol_metrics['loss_iv_data'].item():.2e} | "
                        f"Min g(k): {vol_metrics['min_durrleman_g'].item():+.4f} | "
                        f"Min dw/dtau: {vol_metrics['min_dw_dtau'].item():+.4f}"
                    )

        history.training_time_sec = time.perf_counter() - t_start
        self.engine.eval()
        return history

    def save_checkpoint(
        self,
        checkpoint_path: str = "models/pinn_engine_checkpoint.pt",
        history: Optional[TrainingHistory] = None,
    ) -> Path:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.engine.state_dict(),
            "history": history.__dict__ if history else {},
        }
        torch.save(payload, str(path))
        return path

    def load_checkpoint(self, checkpoint_path: str = "models/pinn_engine_checkpoint.pt") -> Optional[TrainingHistory]:
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        payload = torch.load(str(path), map_location=self.device, weights_only=False)
        self.engine.load_state_dict(payload["model_state_dict"])
        self.engine.eval()
        hist_dict = payload.get("history", {})
        if hist_dict:
            return TrainingHistory(**hist_dict)
        return TrainingHistory()
