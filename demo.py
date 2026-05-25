"""
End-to-end demo: BS Greeks (analytic vs autodiff), SVI calibration, delta-hedge
frequency-cost trade-off. Generates three plots + a metrics summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from black_scholes import (
    analytic_delta, analytic_gamma, analytic_vega, analytic_theta, analytic_rho,
    autodiff_greeks, bs_price,
)
from svi import SVIParams, calibrate_svi, svi_implied_vol, svi_total_variance
from delta_hedge import HedgeConfig, hedge_frequency_sweep, run_hedge_simulation

OUT = Path(__file__).resolve().parent


def demo_greeks():
    """Verify autodiff Greeks match analytic across a strike grid."""
    S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    strikes = np.linspace(70.0, 130.0, 13)
    rows = []
    for K in strikes:
        d_a = float(analytic_delta(S, K, T, r, sigma))
        g_a = float(analytic_gamma(S, K, T, r, sigma))
        v_a = float(analytic_vega(S, K, T, r, sigma))
        d_ad, g_ad, v_ad, _, _ = autodiff_greeks(S, K, T, r, sigma)
        rows.append({
            "K": float(K),
            "delta_analytic": d_a, "delta_autodiff": float(d_ad),
            "gamma_analytic": g_a, "gamma_autodiff": float(g_ad),
            "vega_analytic": v_a, "vega_autodiff": float(v_ad),
        })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    K_arr = np.array([r["K"] for r in rows])
    for ax, name in zip(axes, ["delta", "gamma", "vega"]):
        ax.plot(K_arr, [r[f"{name}_analytic"] for r in rows], "o", label="analytic", ms=8)
        ax.plot(K_arr, [r[f"{name}_autodiff"] for r in rows], "x", label="JAX autodiff", ms=10, mew=2)
        ax.set_xlabel("Strike K")
        ax.set_ylabel(name)
        ax.set_title(f"Call {name} — S={S}, T={T}y, sigma={sigma}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "greeks_analytic_vs_autodiff.png", dpi=140)
    plt.close(fig)
    max_err = max(
        max(abs(r["delta_analytic"] - r["delta_autodiff"]),
            abs(r["gamma_analytic"] - r["gamma_autodiff"]),
            abs(r["vega_analytic"] - r["vega_autodiff"])) for r in rows
    )
    return {"max_abs_error_across_grid": max_err}


def demo_svi():
    """Fit SVI to a synthetic noisy smile."""
    T = 0.5
    k_grid = jnp.linspace(-0.6, 0.6, 41)
    true = SVIParams(a=0.02, b=0.12, rho=-0.5, m=-0.05, sigma=0.18)
    w_true = svi_total_variance(k_grid, true)
    rng = np.random.default_rng(42)
    w_obs = np.asarray(w_true) + rng.normal(scale=0.001, size=w_true.shape)

    fitted = calibrate_svi(k_grid, w_obs)
    w_fit = np.asarray(svi_total_variance(k_grid, fitted))
    vol_obs = np.sqrt(np.maximum(w_obs, 1e-8) / T)
    vol_fit = np.sqrt(w_fit / T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(np.asarray(k_grid), vol_obs, label="market (noisy)", alpha=0.7)
    axes[0].plot(np.asarray(k_grid), vol_fit, "r-", label="SVI fit", lw=2)
    axes[0].set_xlabel("log-moneyness k")
    axes[0].set_ylabel("implied vol")
    axes[0].set_title(f"SVI calibration — T = {T}y")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(np.asarray(k_grid), np.asarray(w_obs), "o", label="market w", alpha=0.6)
    axes[1].plot(np.asarray(k_grid), w_fit, "r-", label="SVI w", lw=2)
    axes[1].set_xlabel("log-moneyness k")
    axes[1].set_ylabel("total variance w(k)")
    axes[1].set_title("Total variance")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "svi_fit.png", dpi=140)
    plt.close(fig)
    mse = float(np.mean((w_fit - np.asarray(w_true)) ** 2))
    return {
        "true_params": {k: float(v) for k, v in vars(true).items()},
        "fitted_params": {k: float(v) for k, v in vars(fitted).items()},
        "mse_vs_noiseless_truth": mse,
    }


def demo_hedge():
    """Frequency-cost trade-off of discrete-time delta hedging."""
    cfg = HedgeConfig(n_paths=5000, txn_bps_per_share=1.0)
    frequencies = [1, 5, 13, 21, 63, 126, 252]
    rows = hedge_frequency_sweep(cfg, frequencies)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    n = np.array([r["n_steps"] for r in rows])
    std = np.array([r["pnl_std"] for r in rows])
    cost = np.array([r["txn_cost_mean"] for r in rows])
    ax2 = ax.twinx()
    ax.plot(n, std, "o-", color="crimson", lw=2, label="PnL std (lower = better hedge)")
    ax2.plot(n, cost, "s--", color="navy", lw=2, label="mean cum. txn cost")
    ax.set_xscale("log")
    ax.set_xlabel("Rebalances over the option's life (log scale)")
    ax.set_ylabel("PnL standard deviation", color="crimson")
    ax2.set_ylabel("Cumulative transaction cost (mean across paths)", color="navy")
    ax.set_title(
        f"Discrete-time delta hedge frequency-cost trade-off\n"
        f"short call, S0={cfg.S0}, K={cfg.K}, T={cfg.T}y, sigma={cfg.sigma}, txn={cfg.txn_bps_per_share}bps/share"
    )
    ax.grid(alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.85))
    fig.tight_layout()
    fig.savefig(OUT / "delta_hedge_frequency_cost.png", dpi=140)
    plt.close(fig)
    return {"frequency_sweep": rows}


def main():
    print("[1/3] Greeks: analytic vs JAX autodiff ...")
    g = demo_greeks()
    print(f"      max |analytic - autodiff| across strike grid: {g['max_abs_error_across_grid']:.2e}")

    print("[2/3] SVI calibration on synthetic noisy smile ...")
    s = demo_svi()
    print(f"      MSE vs noiseless ground truth: {s['mse_vs_noiseless_truth']:.3e}")

    print("[3/3] Delta-hedge frequency-cost sweep ...")
    h = demo_hedge()
    for row in h["frequency_sweep"]:
        print(f"      n={row['n_steps']:>4}  pnl_std={row['pnl_std']:.4f}  txn={row['txn_cost_mean']:.4f}")

    summary = {"greeks_check": g, "svi_calibration": s, "delta_hedge": h}
    (OUT / "demo_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
