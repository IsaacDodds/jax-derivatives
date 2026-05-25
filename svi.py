"""
SVI raw-parametrisation implied-variance surface calibration.

Gatheral's raw SVI parametrises total implied variance for one expiry as a
function of log-moneyness k = log(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Five parameters (a, b, rho, m, sigma). Constraints to keep the slice
arbitrage-free and the variance non-negative:
    b >= 0
    -1 < rho < 1
    sigma > 0
    a + b * sigma * sqrt(1 - rho^2) >= 0  (non-negative-variance condition)

The fit minimises weighted squared error between SVI total variance and the
market total variance w_mkt(k) = (sigma_mkt(k))^2 * T. Weights default to 1.

Usage:
    params = calibrate_svi(k, w_mkt)
    w_hat = svi_total_variance(k, params)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr) -> "SVIParams":
        a, b, rho, m, sigma = (float(x) for x in arr)
        return cls(a=a, b=b, rho=rho, m=m, sigma=sigma)


def svi_total_variance(k, params):
    """Total implied variance w(k) = sigma_imp(k)^2 * T under SVI."""
    if isinstance(params, SVIParams):
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    else:
        a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + jnp.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k, T, params):
    """Implied volatility from SVI total-variance parametrisation."""
    w = svi_total_variance(k, params)
    return jnp.sqrt(jnp.maximum(w, 1e-12) / T)


def _loss(theta, k, w_mkt, weights):
    """Mean weighted squared error in total variance."""
    w_model = svi_total_variance(k, theta)
    err = (w_model - w_mkt) ** 2
    return float(jnp.mean(weights * err))


def calibrate_svi(
    k,
    w_mkt,
    weights=None,
    init=None,
    bounds=None,
) -> SVIParams:
    """Calibrate SVI raw parameters to a single expiry slice.

    Parameters
    ----------
    k : array of log-moneyness values
    w_mkt : array of market total variance values w = sigma_imp^2 * T
    weights : optional per-strike weights (defaults to ones)
    init : optional initial parameter guess; defaults to a sensible heuristic
    bounds : optional bounds; defaults to standard arbitrage-prevention box
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    w_mkt = jnp.asarray(w_mkt, dtype=jnp.float64)
    if weights is None:
        weights = jnp.ones_like(w_mkt)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float64)

    if init is None:
        init = np.array([float(jnp.min(w_mkt)) * 0.5, 0.1, -0.3, 0.0, 0.1])

    if bounds is None:
        bounds = [
            (1e-8, 5.0),     # a >= 0
            (1e-8, 5.0),     # b >= 0
            (-0.999, 0.999), # |rho| < 1
            (-2.0, 2.0),     # m
            (1e-4, 5.0),     # sigma > 0
        ]

    res = minimize(
        _loss,
        init,
        args=(k, w_mkt, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    return SVIParams.from_array(res.x)


if __name__ == "__main__":
    # Generate a synthetic vol smile, then check we recover it.
    T = 0.5
    k_grid = jnp.linspace(-0.5, 0.5, 21)
    true = SVIParams(a=0.02, b=0.10, rho=-0.4, m=0.0, sigma=0.15)
    w_true = svi_total_variance(k_grid, true)

    # Add a touch of noise so it isn't trivially perfect.
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.0005, size=w_true.shape)
    w_obs = np.asarray(w_true) + noise

    fitted = calibrate_svi(k_grid, w_obs)
    print("True   :", true)
    print("Fitted :", fitted)

    w_fit = np.asarray(svi_total_variance(k_grid, fitted))
    mse = float(np.mean((w_fit - np.asarray(w_true)) ** 2))
    print(f"MSE vs noiseless ground truth: {mse:.6g}")
