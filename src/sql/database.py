"""
SQLite Data Warehouse & Multi-Exchange Option Chain Pipeline for Nasdaq-100 (NDX & QQQ).
Models option quotes across Nasdaq's 6 US Options Exchanges:
  1. PHLX (Nasdaq PHLX - Pro-Rata / Hybrid Institutional)
  2. ISE  (Nasdaq ISE  - Pro-Rata / Index & Complex Book)
  3. GEMX (Nasdaq GEMX - Priority Customer / Pro-Rata)
  4. MRX  (Nasdaq MRX  - Customer Priority / Pro-Rata)
  5. NOM  (Nasdaq Options Market - Price-Time Maker-Taker)
  6. BX   (Nasdaq BX Options - Price-Time Taker-Maker)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.analytics.black_scholes import (
    NDX_SVI_PARAMS,
    QQQ_SVI_PARAMS,
    bsm_greeks,
    bsm_price,
    dupire_local_vol_from_svi,
    svi_implied_vol,
)

NASDAQ_EXCHANGES: List[Dict[str, object]] = [
    {
        "exchange_code": "PHLX",
        "exchange_name": "Nasdaq PHLX",
        "allocation_model": "Pro-Rata / Floor-Electronic Hybrid",
        "fee_structure": "Customer Priority / Payment for Order Flow",
        "avg_latency_us": 18.5,
        "spread_multiplier": 1.00,
    },
    {
        "exchange_code": "ISE",
        "exchange_name": "Nasdaq ISE",
        "allocation_model": "Pro-Rata Institutional Complex",
        "fee_structure": "Maker-Taker / Volume Tiers",
        "avg_latency_us": 15.2,
        "spread_multiplier": 0.92,
    },
    {
        "exchange_code": "GEMX",
        "exchange_name": "Nasdaq GEMX",
        "allocation_model": "Pro-Rata Priority Customer",
        "fee_structure": "Maker-Taker Tiered Rebates",
        "avg_latency_us": 16.0,
        "spread_multiplier": 1.05,
    },
    {
        "exchange_code": "MRX",
        "exchange_name": "Nasdaq MRX",
        "allocation_model": "Customer Priority Pro-Rata",
        "fee_structure": "Low-Cost Execution Fee",
        "avg_latency_us": 17.1,
        "spread_multiplier": 1.08,
    },
    {
        "exchange_code": "NOM",
        "exchange_name": "Nasdaq Options Market",
        "allocation_model": "Price-Time Priority",
        "fee_structure": "Maker-Taker Penny Pilot",
        "avg_latency_us": 12.4,
        "spread_multiplier": 0.90,
    },
    {
        "exchange_code": "BX",
        "exchange_name": "Nasdaq BX Options",
        "allocation_model": "Price-Time Priority",
        "fee_structure": "Taker-Maker Rebate",
        "avg_latency_us": 13.8,
        "spread_multiplier": 0.96,
    },
]


class OptionChainDatabase:
    """
    SQL Option Chain Engine managing NDX & QQQ option chains, exchange metadata,
    and PINN arbitrage/mispricing detection signals.
    """

    def __init__(self, db_path: str = "data/nasdaq_options_warehouse.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS nasdaq_exchanges (
                    exchange_code TEXT PRIMARY KEY,
                    exchange_name TEXT NOT NULL,
                    allocation_model TEXT NOT NULL,
                    fee_structure TEXT NOT NULL,
                    avg_latency_us REAL NOT NULL,
                    spread_multiplier REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS option_chains (
                    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange_code TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    spot_price REAL NOT NULL,
                    strike REAL NOT NULL,
                    moneyness REAL NOT NULL,
                    log_moneyness REAL NOT NULL,
                    maturity_days INTEGER NOT NULL,
                    tau REAL NOT NULL,
                    risk_free_rate REAL NOT NULL,
                    dividend_yield REAL NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    mid_price REAL NOT NULL,
                    spread REAL NOT NULL,
                    market_iv REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    open_interest INTEGER NOT NULL,
                    theoretical_iv REAL NOT NULL,
                    theoretical_local_vol REAL NOT NULL,
                    theoretical_price REAL NOT NULL,
                    is_injected_anomaly INTEGER NOT NULL DEFAULT 0,
                    anomaly_type TEXT NOT NULL DEFAULT 'NONE',
                    FOREIGN KEY (exchange_code) REFERENCES nasdaq_exchanges(exchange_code)
                );

                CREATE INDEX IF NOT EXISTS idx_chain_symbol_tau_strike
                ON option_chains(symbol, tau, strike, option_type);

                CREATE INDEX IF NOT EXISTS idx_chain_exchange
                ON option_chains(exchange_code, symbol);

                CREATE TABLE IF NOT EXISTS pinn_signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    option_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange_code TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike REAL NOT NULL,
                    tau REAL NOT NULL,
                    market_mid REAL NOT NULL,
                    market_bid REAL NOT NULL,
                    market_ask REAL NOT NULL,
                    market_iv REAL NOT NULL,
                    pinn_price REAL NOT NULL,
                    pinn_iv REAL NOT NULL,
                    dupire_local_vol REAL NOT NULL,
                    price_residual REAL NOT NULL,
                    iv_residual_bps REAL NOT NULL,
                    spread_z_score REAL NOT NULL,
                    delta REAL NOT NULL,
                    gamma REAL NOT NULL,
                    vega REAL NOT NULL,
                    theta REAL NOT NULL,
                    rho REAL NOT NULL,
                    bl_density REAL NOT NULL,
                    durrleman_g REAL NOT NULL,
                    calendar_var_slope REAL NOT NULL,
                    pde_residual REAL NOT NULL,
                    detected_signal TEXT NOT NULL,
                    actionable_edge_dollars REAL NOT NULL,
                    FOREIGN KEY (option_id) REFERENCES option_chains(option_id)
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol_type
                ON pinn_signals(symbol, detected_signal);
                """
            )
            conn.commit()

    def seed_nasdaq_data(self, seed: int = 42, force_refresh: bool = True) -> Dict[str, int]:
        """
        Populate the SQL warehouse with calibrated NDX ($20,500) and QQQ ($500) option chains
        across Nasdaq's 6 options exchanges, including realistic microstructure bid-ask spreads
        and deterministic injected mispricing & arbitrage violations.
        """
        rng = np.random.default_rng(seed)

        with self._connect() as conn:
            cursor = conn.cursor()
            if force_refresh:
                cursor.execute("DELETE FROM pinn_signals;")
                cursor.execute("DELETE FROM option_chains;")
                cursor.execute("DELETE FROM nasdaq_exchanges;")

            # Seed Nasdaq Exchange Metadata
            cursor.executemany(
                """
                INSERT OR REPLACE INTO nasdaq_exchanges (
                    exchange_code, exchange_name, allocation_model,
                    fee_structure, avg_latency_us, spread_multiplier
                ) VALUES (
                    :exchange_code, :exchange_name, :allocation_model,
                    :fee_structure, :avg_latency_us, :spread_multiplier
                );
                """,
                NASDAQ_EXCHANGES,
            )

            # Generate NDX and QQQ Option Chains
            configs = [
                {
                    "symbol": "NDX",
                    "spot": 20500.0,
                    "r": 0.045,
                    "q": 0.008,
                    "svi_params": NDX_SVI_PARAMS,
                    "strike_step": 100.0,
                },
                {
                    "symbol": "QQQ",
                    "spot": 500.0,
                    "r": 0.045,
                    "q": 0.0065,
                    "svi_params": QQQ_SVI_PARAMS,
                    "strike_step": 2.5,
                },
            ]

            maturity_days_list = [14, 30, 60, 90, 120, 180, 270, 365]
            exchange_codes = [ex["exchange_code"] for ex in NASDAQ_EXCHANGES]
            exchange_spread_mult = {
                str(ex["exchange_code"]): float(ex["spread_multiplier"]) for ex in NASDAQ_EXCHANGES
            }

            all_rows = []
            anomaly_count = 0

            for cfg in configs:
                sym = str(cfg["symbol"])
                S0 = float(cfg["spot"])
                r = float(cfg["r"])
                q = float(cfg["q"])
                svi_p = cfg["svi_params"]
                step = float(cfg["strike_step"])

                # 25 strikes spanning [0.85 * S0, 1.15 * S0]
                raw_strikes = np.linspace(0.85 * S0, 1.15 * S0, 25)
                strikes = np.round(raw_strikes / step) * step

                for mat_days in maturity_days_list:
                    tau = mat_days / 365.0
                    forward = S0 * np.exp((r - q) * tau)

                    for idx_k, K in enumerate(strikes):
                        moneyness = K / S0
                        log_k = float(np.log(K / forward))
                        # Liquid OTM representation + near-ATM overlap:
                        # Generate both Call and Put for comprehensive parity & surface coverage
                        for opt_type in ("call", "put"):
                            theo_iv = float(svi_implied_vol(S0, K, tau, r, q, params=svi_p))
                            theo_loc_vol = float(dupire_local_vol_from_svi(S0, K, tau, r, q, params=svi_p))
                            theo_price = float(bsm_price(S0, K, tau, r, q, theo_iv, opt_type))
                            greeks = bsm_greeks(S0, K, tau, r, q, theo_iv, opt_type)
                            vega_val = float(greeks["vega"])

                            # Skip deep OTM contracts with near-zero value
                            if theo_price < 0.05 * (S0 / 500.0):
                                continue

                            # Assign exchange venue deterministically + rotation
                            ex_code = str(exchange_codes[(idx_k + mat_days + (0 if opt_type == "call" else 3)) % len(exchange_codes)])
                            sp_mult = exchange_spread_mult[ex_code]

                            # Realistic microstructure noise (1 to 4 bps of IV)
                            iv_noise = float(rng.normal(0.0, 0.00035))
                            is_anomaly = 0
                            anomaly_type = "NONE"

                            # Inject controlled, realistic market anomalies (~5.5% of quotes)
                            # 1) Undervalued Ask (Stale liquidity provider quote on PHLX/NOM)
                            if (idx_k == 10 and mat_days in (30, 90, 180) and opt_type == "call") or (
                                idx_k == 8 and mat_days in (60, 120) and opt_type == "put"
                            ):
                                iv_noise = -0.0185  # -185 bps IV discount (significant underpricing)
                                is_anomaly = 1
                                anomaly_type = "UNDERPRICED_ASK"
                                anomaly_count += 1
                            # 2) Overvalued Bid (Aggressive institutional limit order on ISE/GEMX)
                            elif (idx_k == 14 and mat_days in (30, 60, 270) and opt_type == "call") or (
                                idx_k == 6 and mat_days in (30, 180) and opt_type == "put"
                            ):
                                iv_noise = +0.0195  # +195 bps IV premium (significant overpricing)
                                is_anomaly = 1
                                anomaly_type = "OVERPRICED_BID"
                                anomaly_count += 1
                            # 3) Butterfly Arbitrage Violation (local strike concavity kink at idx_k == 12)
                            elif idx_k == 12 and mat_days in (60, 180):
                                iv_noise = +0.0240  # Sharp local IV spike violating strike convexity
                                is_anomaly = 1
                                anomaly_type = "BUTTERFLY_ARB_VIOLATION"
                                anomaly_count += 1
                            # 4) Calendar Arbitrage Violation (inverted total variance at mat_days == 120, idx_k == 16)
                            elif idx_k == 16 and mat_days == 120:
                                iv_noise = -0.0225  # Total variance drops below 90-day slice
                                is_anomaly = 1
                                anomaly_type = "CALENDAR_ARB_VIOLATION"
                                anomaly_count += 1

                            mkt_iv = max(0.04, theo_iv + iv_noise)
                            mid_price = float(bsm_price(S0, K, tau, r, q, mkt_iv, opt_type))

                            # Realistic bid-ask spread proportional to vega (~18 bps of vol) + minimum tick
                            half_spread = max(0.02 * (S0 / 500.0), 0.0009 * vega_val * sp_mult)
                            bid = max(0.01, mid_price - half_spread)
                            ask = mid_price + half_spread
                            spread = ask - bid

                            # Liquidity distribution peaking near ATM
                            atm_weight = float(np.exp(-18.0 * (log_k**2)))
                            volume = int(max(15, rng.poisson(1800 * atm_weight + 60)))
                            open_interest = int(max(100, volume * rng.uniform(4.5, 11.0)))

                            all_rows.append(
                                (
                                    sym,
                                    ex_code,
                                    opt_type,
                                    S0,
                                    float(K),
                                    float(moneyness),
                                    log_k,
                                    int(mat_days),
                                    float(tau),
                                    r,
                                    q,
                                    round(bid, 4),
                                    round(ask, 4),
                                    round(mid_price, 4),
                                    round(spread, 4),
                                    round(mkt_iv, 6),
                                    volume,
                                    open_interest,
                                    round(theo_iv, 6),
                                    round(theo_loc_vol, 6),
                                    round(theo_price, 4),
                                    is_anomaly,
                                    anomaly_type,
                                )
                            )

            cursor.executemany(
                """
                INSERT INTO option_chains (
                    symbol, exchange_code, option_type, spot_price, strike,
                    moneyness, log_moneyness, maturity_days, tau,
                    risk_free_rate, dividend_yield, bid, ask, mid_price, spread,
                    market_iv, volume, open_interest, theoretical_iv,
                    theoretical_local_vol, theoretical_price,
                    is_injected_anomaly, anomaly_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                all_rows,
            )
            conn.commit()

        return {
            "total_contracts": len(all_rows),
            "injected_anomalies": anomaly_count,
            "exchanges": len(NASDAQ_EXCHANGES),
        }

    def get_option_chain_df(
        self,
        symbol: Optional[str] = None,
        option_type: Optional[str] = None,
        exchange_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query option chains from SQLite with optional filtering by symbol, option_type, and exchange.
        """
        query = """
            SELECT
                c.*,
                e.exchange_name,
                e.allocation_model,
                e.avg_latency_us
            FROM option_chains c
            JOIN nasdaq_exchanges e ON c.exchange_code = e.exchange_code
            WHERE 1=1
        """
        params: List[object] = []
        if symbol:
            query += " AND c.symbol = ?"
            params.append(symbol.upper())
        if option_type:
            query += " AND c.option_type = ?"
            params.append(option_type.lower())
        if exchange_code and exchange_code.upper() != "ALL":
            query += " AND c.exchange_code = ?"
            params.append(exchange_code.upper())

        query += " ORDER BY c.symbol, c.tau, c.strike, c.option_type;"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def save_pinn_signals(self, signals_df: pd.DataFrame) -> int:
        """
        Persist PINN valuation, autograd Greeks, and detected mispricing/arbitrage signals to SQL.
        """
        required_cols = [
            "option_id",
            "symbol",
            "exchange_code",
            "option_type",
            "strike",
            "tau",
            "market_mid",
            "market_bid",
            "market_ask",
            "market_iv",
            "pinn_price",
            "pinn_iv",
            "dupire_local_vol",
            "price_residual",
            "iv_residual_bps",
            "spread_z_score",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "bl_density",
            "durrleman_g",
            "calendar_var_slope",
            "pde_residual",
            "detected_signal",
            "actionable_edge_dollars",
        ]
        records = signals_df[required_cols].to_dict(orient="records")
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pinn_signals;")
            cursor.executemany(
                """
                INSERT INTO pinn_signals (
                    option_id, symbol, exchange_code, option_type, strike, tau,
                    market_mid, market_bid, market_ask, market_iv,
                    pinn_price, pinn_iv, dupire_local_vol,
                    price_residual, iv_residual_bps, spread_z_score,
                    delta, gamma, vega, theta, rho,
                    bl_density, durrleman_g, calendar_var_slope, pde_residual,
                    detected_signal, actionable_edge_dollars
                ) VALUES (
                    :option_id, :symbol, :exchange_code, :option_type, :strike, :tau,
                    :market_mid, :market_bid, :market_ask, :market_iv,
                    :pinn_price, :pinn_iv, :dupire_local_vol,
                    :price_residual, :iv_residual_bps, :spread_z_score,
                    :delta, :gamma, :vega, :theta, :rho,
                    :bl_density, :durrleman_g, :calendar_var_slope, :pde_residual,
                    :detected_signal, :actionable_edge_dollars
                );
                """,
                records,
            )
            conn.commit()
        return len(records)

    def get_exchange_summary_sql(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Execute an analytical SQL aggregation across Nasdaq's 6 options exchanges showing
        contract volume, average bid-ask spread, detected mispricings, and actionable dollar edge.
        """
        where_clause = "WHERE s.symbol = ?" if symbol else ""
        params = [symbol.upper()] if symbol else []
        query = f"""
            SELECT
                e.exchange_code AS exchange,
                e.exchange_name,
                e.allocation_model,
                e.avg_latency_us,
                COUNT(s.signal_id) AS total_quotes,
                SUM(c.volume) AS total_volume,
                ROUND(AVG(c.spread), 4) AS avg_bid_ask_spread,
                SUM(CASE WHEN s.detected_signal != 'FAIR_PRICED' THEN 1 ELSE 0 END) AS flagged_anomalies,
                SUM(CASE WHEN s.detected_signal = 'BUY_UNDERVALUED' THEN 1 ELSE 0 END) AS buy_signals,
                SUM(CASE WHEN s.detected_signal = 'SELL_OVERVALUED' THEN 1 ELSE 0 END) AS sell_signals,
                SUM(CASE WHEN s.detected_signal LIKE '%ARB%' THEN 1 ELSE 0 END) AS arb_violations,
                ROUND(SUM(s.actionable_edge_dollars), 2) AS total_actionable_edge_usd
            FROM pinn_signals s
            JOIN option_chains c ON s.option_id = c.option_id
            JOIN nasdaq_exchanges e ON s.exchange_code = e.exchange_code
            {where_clause}
            GROUP BY e.exchange_code, e.exchange_name, e.allocation_model, e.avg_latency_us
            ORDER BY flagged_anomalies DESC, total_volume DESC;
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_top_actionable_signals_sql(self, symbol: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """
        Fetch the highest-conviction mispriced and arbitrage-violating contracts via SQL.
        """
        where_sym = "AND s.symbol = ?" if symbol else ""
        params: List[object] = [symbol.upper()] if symbol else []
        params.append(limit)
        query = f"""
            SELECT
                s.option_id,
                s.symbol,
                s.exchange_code,
                e.exchange_name,
                UPPER(s.option_type) AS type,
                s.strike,
                c.maturity_days AS dte,
                ROUND(s.market_bid, 2) AS bid,
                ROUND(s.market_ask, 2) AS ask,
                ROUND(s.market_mid, 2) AS mkt_mid,
                ROUND(s.pinn_price, 2) AS pinn_fair,
                ROUND(s.market_iv * 100.0, 2) AS mkt_iv_pct,
                ROUND(s.pinn_iv * 100.0, 2) AS pinn_iv_pct,
                ROUND(s.iv_residual_bps, 1) AS iv_diff_bps,
                ROUND(s.spread_z_score, 2) AS z_score,
                ROUND(s.delta, 4) AS delta,
                ROUND(s.gamma, 6) AS gamma,
                ROUND(s.vega / 100.0, 4) AS vega_1pct,
                ROUND(s.theta / 365.0, 4) AS theta_daily,
                s.detected_signal,
                c.anomaly_type AS ground_truth_anomaly,
                ROUND(s.actionable_edge_dollars, 2) AS edge_usd
            FROM pinn_signals s
            JOIN option_chains c ON s.option_id = c.option_id
            JOIN nasdaq_exchanges e ON s.exchange_code = e.exchange_code
            WHERE s.detected_signal != 'FAIR_PRICED'
            {where_sym}
            ORDER BY ABS(s.iv_residual_bps) DESC
            LIMIT ?;
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def run_custom_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a read-only analytical SQL query against the SQLite warehouse.
        """
        cleaned = sql_query.strip().upper()
        if not cleaned.startswith("SELECT") and not cleaned.startswith("WITH"):
            raise ValueError("Only SELECT or WITH analytical queries are permitted.")
        with self._connect() as conn:
            return pd.read_sql_query(sql_query, conn)
