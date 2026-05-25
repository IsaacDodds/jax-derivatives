"""
Discrete-time delta hedging — Monte Carlo PnL of a short-call position.

A market-maker sells a European call and then rebalances a stock hedge at
fixed intervals to stay delta-neutral. Under continuous frictionless hedging
in a Black-Scholes world the strategy is exactly replicating and PnL is zero;
in discrete time with transaction costs there is residual PnL whose
distribution depends on rebalance frequency and cost level.

This module simulates GBM paths, runs the strategy, and characterises the
PnL distribution as a function of rebalance frequency and transaction cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from black_scholes import bs_price, analytic_delta

jax.config.update("jax_enable_x64", True)


@dataclass
class HedgeConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.25     # 3 months
    r: float = 0.03
    sigma: float = 0.20
    n_steps: int = 63   # daily rebalance, ~63 trading days in 3M
    n_paths: int = 5000
    txn_bps_per_share: float = 1.0   # 1 bp of share notional per side
    option_type: str = "call"
    seed: int = 0


def simulate_gbm_paths(cfg: HedgeConfig):
    """Generate (n_paths, n_steps+1) GBM price paths under the risk-neutral measure."""
    dt = cfg.T / cfg.n_steps
    rng = np.random.default_rng(cfg.seed)
    z = rng.standard_normal(size=(cfg.n_paths, cfg.n_steps))
    log_returns = (cfg.r - 0.5 * cfg.sigma ** 2) * dt + cfg.sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((cfg.n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return cfg.S0 * np.exp(log_paths)


def run_hedge_simulation(cfg: HedgeConfig):
    """Run discrete-time delta hedge against a short-call book; return PnL stats.

    Convention: trader is SHORT one call (option premium received up front),
    holds delta_t shares as the hedge, finances the position at r, pays
    transaction costs on |delta_t - delta_{t-1}| shares per rebalance.

    Final PnL = premium received * accreted - max(S_T - K, 0)
              + sum of hedge-portfolio cash flows
              - cumulative transaction costs
    """
    paths = simulate_gbm_paths(cfg)
    n_paths, n_bars = paths.shape
    dt = cfg.T / cfg.n_steps

    # Initial option premium (cash credit to seller).
    premium = float(bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma, option_type=cfg.option_type))

    delta_prev = np.zeros(n_paths)
    cash = np.full(n_paths, premium)
    txn_cost_total = np.zeros(n_paths)
    cost_per_share = cfg.txn_bps_per_share / 1e4

    for t in range(cfg.n_steps):
        S_t = paths[:, t]
        tau = cfg.T - t * dt
        # Hedge delta of the SHORT call = -delta_call (we need +delta_call shares
        # of stock to neutralise a short call's negative delta-exposure).
        delta_t = np.asarray(analytic_delta(S_t, cfg.K, tau, cfg.r, cfg.sigma, option_type=cfg.option_type))
        share_change = delta_t - delta_prev
        # Cost in cash for the rebalance (buy shares -> cash out; sell shares -> cash in).
        cash -= share_change * S_t
        # Transaction cost (always a loss).
        txn = np.abs(share_change) * S_t * cost_per_share
        cash -= txn
        txn_cost_total += txn
        # Accrete cash at risk-free rate.
        cash *= np.exp(cfg.r * dt)
        delta_prev = delta_t

    S_T = paths[:, -1]
    # Liquidate stock at expiry, settle option payoff (short call obligation).
    final_cash = cash + delta_prev * S_T
    if cfg.option_type == "call":
        option_payoff = np.maximum(S_T - cfg.K, 0.0)
    else:
        option_payoff = np.maximum(cfg.K - S_T, 0.0)
    pnl = final_cash - option_payoff

    return {
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "pnl_p5": float(np.percentile(pnl, 5)),
        "pnl_p95": float(np.percentile(pnl, 95)),
        "txn_cost_mean": float(txn_cost_total.mean()),
        "premium_received": premium,
        "pnl_path": pnl,
        "config": cfg,
    }


def hedge_frequency_sweep(cfg: HedgeConfig, frequencies):
    """Run the simulation across multiple rebalance frequencies."""
    rows = []
    for n in frequencies:
        c = HedgeConfig(**{**cfg.__dict__, "n_steps": int(n)})
        out = run_hedge_simulation(c)
        rows.append({
            "n_steps": n,
            "pnl_mean": out["pnl_mean"],
            "pnl_std": out["pnl_std"],
            "pnl_p5": out["pnl_p5"],
            "pnl_p95": out["pnl_p95"],
            "txn_cost_mean": out["txn_cost_mean"],
        })
    return rows


if __name__ == "__main__":
    cfg = HedgeConfig()
    out = run_hedge_simulation(cfg)
    print(f"Short-call delta hedge — S0={cfg.S0}, K={cfg.K}, T={cfg.T}y, "
          f"sigma={cfg.sigma}, n_steps={cfg.n_steps}, n_paths={cfg.n_paths}, "
          f"txn={cfg.txn_bps_per_share}bps/share")
    print(f"  Option premium received : {out['premium_received']:.4f}")
    print(f"  PnL mean                : {out['pnl_mean']:+.4f}")
    print(f"  PnL std                 : {out['pnl_std']:.4f}")
    print(f"  PnL 5th percentile      : {out['pnl_p5']:+.4f}")
    print(f"  PnL 95th percentile     : {out['pnl_p95']:+.4f}")
    print(f"  Mean cumulative txn cost: {out['txn_cost_mean']:.4f}")

    print("\nFrequency sweep:")
    print(f"  {'n_steps':>8} {'pnl_mean':>10} {'pnl_std':>10} {'txn_cost':>10}")
    for row in hedge_frequency_sweep(cfg, [1, 5, 21, 63, 252]):
        print(f"  {row['n_steps']:>8} {row['pnl_mean']:>+10.4f} {row['pnl_std']:>10.4f} {row['txn_cost_mean']:>10.4f}")
