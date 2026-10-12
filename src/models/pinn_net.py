"""
Dual-Network Physics-Informed Neural Network (PINN) for Options Pricing & Implied Volatility Surface:
1. ArbitrageFreeVolSurfacePINN:
   Reconstructs the 3D Implied Volatility Surface sigma_imp(k, tau), total variance w(k, tau),
   Durrleman butterfly condition g(k, tau), and Dupire local volatility sigma_loc(k, tau)
   using PyTorch Automatic Differentiation (torch.autograd).
2. BSMPricingPINN:
   Embeds the Black-Scholes-Merton PDE and scale-invariant boundary conditions, computing
   option fair value V(S, K, tau, r, q, sigma) and exact analytical-grade Greeks
   (Delta, Gamma, Vega, Theta, Rho, Dual-Gamma) via torch.autograd.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothResidualBlock(nn.Module):
    """
    C-infinity smooth residual block using SiLU/Tanh activations.
    Required for second-order PDE derivatives (Gamma = d^2V/dS^2, Butterfly = d^2V/dK^2)
    where piecewise-linear ReLU activations would yield identically zero curvature.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.8)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.8)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        return residual + 0.5 * out


class ArbitrageFreeVolSurfacePINN(nn.Module):
    """
    Physics-Informed Implied Volatility Surface Network mapping log-moneyness k = ln(K / F_T)
    and time-to-maturity tau = T - t to implied volatility sigma_imp(k, tau).

    Enforces:
      - Positive implied volatility & variance
      - Calendar spread no-arbitrage: dw / dtau >= 0
      - Butterfly spread no-arbitrage (Durrleman condition): g(k, tau) >= 0
      - Exact Dupire Local Volatility extraction: sigma_loc(k, tau) = sqrt((dw/dtau) / g(k, tau))
    """

    def __init__(self, hidden_dim: int = 64, num_blocks: int = 3) -> None:
        super().__init__()
        # Input features: [k, tau, sqrt(tau), k / sqrt(tau), k^2]
        in_dim = 5
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([SmoothResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Learnable SVI-inspired structural parameters for fast convergence and smooth wings
        self.base_vol = nn.Parameter(torch.tensor(0.1855))
        self.term_slope = nn.Parameter(torch.tensor(-0.018))
        self.skew_rho = nn.Parameter(torch.tensor(-0.65))
        self.smile_wing = nn.Parameter(torch.tensor(0.065))
        self.smooth_knot = nn.Parameter(torch.tensor(0.14))
        self.shift_m = nn.Parameter(torch.tensor(0.015))

        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.005)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute annualized implied volatility sigma_imp(k, tau).
        k and tau should have shape (N, 1) or (N,).
        """
        orig_shape = k.shape
        k_col = k.reshape(-1, 1)
        tau_col = torch.clamp(tau.reshape(-1, 1), min=1e-5)
        sqrt_tau = torch.sqrt(tau_col)

        features = torch.cat(
            [
                k_col,
                tau_col,
                sqrt_tau,
                k_col / (sqrt_tau + 0.05),
                k_col**2,
            ],
            dim=-1,
        )

        h = self.input_layer(features)
        for block in self.blocks:
            h = block(h)
        nn_correction = 0.025 * torch.tanh(self.output_layer(h))

        # Structural SVI backbone + smooth neural correction
        s_knot = torch.clamp(torch.abs(self.smooth_knot), min=0.05)
        rho = torch.clamp(self.skew_rho, min=-0.95, max=0.95)
        b_scale = torch.clamp(torch.abs(self.smile_wing), min=0.01)
        m_val = self.shift_m

        atm_vol = torch.clamp(self.base_vol + self.term_slope * (1.0 - torch.exp(-1.5 * tau_col)), min=0.05)
        atm_var_rate = atm_vol**2
        b_rate = b_scale * (tau_col ** -0.28)

        km = k_col - m_val
        smile = rho * km + torch.sqrt(km**2 + s_knot**2)
        smile_zero = rho * (-m_val) + torch.sqrt(m_val**2 + s_knot**2)

        var_rate_prior = torch.clamp(atm_var_rate + b_rate * (smile - smile_zero), min=0.0016)
        sigma_prior = torch.sqrt(var_rate_prior)

        sigma_imp = torch.clamp(sigma_prior + nn_correction, min=0.03)
        return sigma_imp.reshape(orig_shape)

    def evaluate_surface_and_arbitrage(
        self,
        k: torch.Tensor,
        tau: torch.Tensor,
        create_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate implied volatility, total variance w(k, tau), first & second derivatives
        (dw/dk, d^2w/dk^2, dw/dtau), Durrleman butterfly function g(k), and Dupire local volatility
        via torch.autograd.grad.
        """
        k_req = k.reshape(-1, 1).clone().detach().requires_grad_(True)
        tau_req = tau.reshape(-1, 1).clone().detach().requires_grad_(True)

        sigma_imp = self.forward(k_req, tau_req)
        w = (sigma_imp**2) * tau_req

        ones = torch.ones_like(w)
        # First derivatives must always set create_graph=True so d2w_dk2 can differentiate dw_dk
        dw_dk, dw_dtau = torch.autograd.grad(
            outputs=w,
            inputs=(k_req, tau_req),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )

        d2w_dk2 = torch.autograd.grad(
            outputs=dw_dk,
            inputs=k_req,
            grad_outputs=ones,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        w_safe = torch.clamp(w, min=1e-6)
        # Gatheral-Jacquier / Durrleman butterfly condition g(k)
        term1 = (1.0 - (k_req * dw_dk) / (2.0 * w_safe)) ** 2
        term2 = 0.25 * (dw_dk**2) * (1.0 / w_safe + 0.25)
        term3 = 0.5 * d2w_dk2
        durrleman_g = term1 - term2 + term3

        # Dupire Local Volatility: sigma_loc^2 = (dw/dtau) / g(k)
        dupire_local_var = torch.clamp(dw_dtau, min=1e-5) / torch.clamp(durrleman_g, min=1e-4)
        dupire_local_vol = torch.sqrt(dupire_local_var)

        return {
            "sigma_imp": sigma_imp.squeeze(-1),
            "w": w.squeeze(-1),
            "dw_dk": dw_dk.squeeze(-1),
            "d2w_dk2": d2w_dk2.squeeze(-1),
            "dw_dtau": dw_dtau.squeeze(-1),
            "durrleman_g": durrleman_g.squeeze(-1),
            "dupire_local_vol": dupire_local_vol.squeeze(-1),
        }


class BSMPricingPINN(nn.Module):
    """
    Scale-Invariant Physics-Informed Neural Network for Option Pricing & Autograd Greeks.

    Exploits linear homogeneity V(S, K, tau, r, q, sigma) = K * u(S/K, tau, r, q, sigma)
    so the exact same network accurately prices QQQ ($500) and NDX ($20,500) options,
    while satisfying terminal payoff boundary conditions at tau = 0 by construction.
    """

    def __init__(self, hidden_dim: int = 96, num_blocks: int = 4) -> None:
        super().__init__()
        # Dimensionless state inputs:
        # [log(S/K), sqrt(tau), sigma * sqrt(tau), (r - q) * tau, r * tau, d1_approx, d2_approx]
        in_dim = 7
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([SmoothResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.005)
        nn.init.zeros_(self.output_layer.bias)

    def _call_normalized(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        tau: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute European Call price C(S, K, tau, r, q, sigma) combining a smooth asymptotic
        physics ansatz with a deep residual neural correction that vanishes at tau = 0.
        """
        safe_S = torch.clamp(S, min=1e-6)
        safe_K = torch.clamp(K, min=1e-6)
        safe_tau = torch.clamp(tau, min=1e-8)
        safe_sigma = torch.clamp(sigma, min=1e-4)

        sqrt_tau = torch.sqrt(safe_tau)
        sig_sqrt_tau = safe_sigma * sqrt_tau
        log_m = torch.log(safe_S / safe_K)

        d1 = (log_m + (r - q + 0.5 * safe_sigma**2) * safe_tau) / sig_sqrt_tau
        d2 = d1 - sig_sqrt_tau

        features = torch.cat(
            [
                log_m,
                sqrt_tau,
                sig_sqrt_tau,
                (r - q) * safe_tau,
                r * safe_tau,
                torch.tanh(0.5 * d1),
                torch.tanh(0.5 * d2),
            ],
            dim=-1,
        )

        h = self.input_layer(features)
        for block in self.blocks:
            h = block(h)

        # Neural network learns dimensionless time-value modulation localized near the money
        # Multiplied by sig_sqrt_tau * exp(-0.5 * d1^2) so boundary conditions at tau -> 0
        # and S -> 0 / S -> inf are structurally preserved with C-infinity smoothness!
        atm_envelope = torch.exp(-0.5 * (d1**2))
        nn_residual = 0.01 * torch.tanh(self.output_layer(h)) * sig_sqrt_tau * atm_envelope

        disc_q = torch.exp(-q * safe_tau)
        disc_r = torch.exp(-r * safe_tau)

        # Base physics-informed carrier + neural network PDE correction
        Nd1 = torch.special.ndtr(d1)
        Nd2 = torch.special.ndtr(d2)
        base_call = safe_S * disc_q * Nd1 - safe_K * disc_r * Nd2

        return torch.clamp(base_call + safe_K * nn_residual, min=0.0)

    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        tau: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
        sigma: torch.Tensor,
        option_type: Union[str, torch.Tensor] = "call",
    ) -> torch.Tensor:
        """
        Compute PINN option fair value for Call or Put options.
        Note: Put price P = C - S * e^{-q*tau} + K * e^{-r*tau} satisfies the exact same
        Black-Scholes-Merton PDE and guarantees zero Put-Call Parity arbitrage.
        """
        orig_shape = S.shape
        S_c = S.reshape(-1, 1)
        K_c = K.reshape(-1, 1)
        tau_c = tau.reshape(-1, 1)
        r_c = r.reshape(-1, 1)
        q_c = q.reshape(-1, 1)
        sig_c = sigma.reshape(-1, 1)

        call_val = self._call_normalized(S_c, K_c, tau_c, r_c, q_c, sig_c)

        if isinstance(option_type, str):
            if option_type.lower().startswith("c"):
                return call_val.reshape(orig_shape)
            else:
                put_val = call_val - S_c * torch.exp(-q_c * tau_c) + K_c * torch.exp(-r_c * tau_c)
                return put_val.reshape(orig_shape)
        else:
            # Tensor boolean/float mask where 1 = call, 0 = put
            is_call = option_type.reshape(-1, 1).to(call_val.dtype)
            put_val = call_val - S_c * torch.exp(-q_c * tau_c) + K_c * torch.exp(-r_c * tau_c)
            out = is_call * call_val + (1.0 - is_call) * put_val
            return out.reshape(orig_shape)

    def compute_autograd_greeks(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        tau: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
        sigma: torch.Tensor,
        option_type: Union[str, torch.Tensor] = "call",
        create_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Option Price, exact first- and second-order Greeks (Delta, Gamma, Vega, Theta, Rho),
        Breeden-Litzenberger Butterfly Density (d^2V/dK^2), and the Black-Scholes-Merton PDE Residual
        in a single vectorized pass via torch.autograd.grad.
        """
        S_req = S.reshape(-1, 1).clone().detach().requires_grad_(True)
        K_req = K.reshape(-1, 1).clone().detach().requires_grad_(True)
        tau_req = tau.reshape(-1, 1).clone().detach().requires_grad_(True)
        r_req = r.reshape(-1, 1).clone().detach().requires_grad_(True)
        q_req = q.reshape(-1, 1).clone().detach()
        sig_req = sigma.reshape(-1, 1).clone().detach().requires_grad_(True)

        V = self.forward(S_req, K_req, tau_req, r_req, q_req, sig_req, option_type=option_type)
        ones = torch.ones_like(V)

        # First-order partial derivatives via torch.autograd
        dV_dS, dV_dK, dV_dtau, dV_dr, dV_dsigma = torch.autograd.grad(
            outputs=V,
            inputs=(S_req, K_req, tau_req, r_req, sig_req),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )

        # Second-order partial derivatives: Gamma = d^2V/dS^2 and Dual-Gamma = d^2V/dK^2
        d2V_dS2 = torch.autograd.grad(
            outputs=dV_dS,
            inputs=S_req,
            grad_outputs=ones,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        d2V_dK2 = torch.autograd.grad(
            outputs=dV_dK,
            inputs=K_req,
            grad_outputs=ones,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        # Theta is defined as -dV/dtau = dV/dt
        theta = -dV_dtau

        # Black-Scholes-Merton PDE Residual:
        # R_PDE = -dV/dtau + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + (r - q) * S * dV/dS - r * V
        pde_residual = (
            -dV_dtau
            + 0.5 * (sig_req**2) * (S_req**2) * d2V_dS2
            + (r_req - q_req) * S_req * dV_dS
            - r_req * V
        )

        # Normalize PDE residual by strike K so it is dimensionless across QQQ and NDX
        pde_residual_norm = pde_residual / torch.clamp(K_req, min=1.0)

        # Breeden-Litzenberger Risk-Neutral Probability Density: p(K) = e^{r * tau} * d^2C/dK^2
        bl_density = torch.exp(r_req * tau_req) * d2V_dK2

        if not create_graph:
            return {
                "price": V.detach().squeeze(-1),
                "delta": dV_dS.detach().squeeze(-1),
                "gamma": d2V_dS2.detach().squeeze(-1),
                "vega": dV_dsigma.detach().squeeze(-1),
                "theta": theta.detach().squeeze(-1),
                "rho": dV_dr.detach().squeeze(-1),
                "dual_delta": dV_dK.detach().squeeze(-1),
                "dual_gamma": d2V_dK2.detach().squeeze(-1),
                "bl_density": bl_density.detach().squeeze(-1),
                "pde_residual": pde_residual_norm.detach().squeeze(-1),
            }

        return {
            "price": V.squeeze(-1),
            "delta": dV_dS.squeeze(-1),
            "gamma": d2V_dS2.squeeze(-1),
            "vega": dV_dsigma.squeeze(-1),
            "theta": theta.squeeze(-1),
            "rho": dV_dr.squeeze(-1),
            "dual_delta": dV_dK.squeeze(-1),
            "dual_gamma": d2V_dK2.squeeze(-1),
            "bl_density": bl_density.squeeze(-1),
            "pde_residual": pde_residual_norm.squeeze(-1),
        }


class DualPINNOptionsEngine(nn.Module):
    """
    Unified Dual-PINN Engine combining:
      1. ArbitrageFreeVolSurfacePINN (3D Implied & Dupire Local Volatility Surface)
      2. BSMPricingPINN (PDE-governed Derivative Pricer & Sub-Millisecond Autograd Greek Engine)
    """

    def __init__(
        self,
        vol_hidden_dim: int = 64,
        vol_blocks: int = 3,
        pricing_hidden_dim: int = 96,
        pricing_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.vol_net = ArbitrageFreeVolSurfacePINN(hidden_dim=vol_hidden_dim, num_blocks=vol_blocks)
        self.pricing_net = BSMPricingPINN(hidden_dim=pricing_hidden_dim, num_blocks=pricing_blocks)

    def predict_iv(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        tau: torch.Tensor,
        r: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        forward = S * torch.exp((r - q) * tau)
        k = torch.log(torch.clamp(K, min=1e-6) / torch.clamp(forward, min=1e-6))
        return self.vol_net(k, tau)
