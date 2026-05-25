"""
Black-Scholes pricer with analytic AND autodiff Greeks.

The point of the file: the same one-line price function is differentiated by
JAX to produce delta, gamma, theta, vega and rho — and the result matches the
closed-form analytic Greeks to within floating-point noise. This is a clean
demonstration of why autodiff matters in derivatives quant work: you can build
a complicated payoff once and get every Greek free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

jax.config.update("jax_enable_x64", True)


def _d1(S, K, T, r, sigma, q=0.0):
    return (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))


def _d2(S, K, T, r, sigma, q=0.0):
    return _d1(S, K, T, r, sigma, q) - sigma * jnp.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Black-Scholes-Merton price of a European option.

    S: spot, K: strike, T: time to expiry (years), r: risk-free rate,
    sigma: vol, q: continuous dividend yield, option_type: 'call' or 'put'.
    """
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return K * jnp.exp(-r * T) * norm.cdf(-d2) - S * jnp.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


# ----- Analytic Greeks (closed form) -----

def analytic_delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1 = _d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return jnp.exp(-q * T) * norm.cdf(d1)
    return jnp.exp(-q * T) * (norm.cdf(d1) - 1.0)


def analytic_gamma(S, K, T, r, sigma, q=0.0):
    d1 = _d1(S, K, T, r, sigma, q)
    return jnp.exp(-q * T) * norm.pdf(d1) / (S * sigma * jnp.sqrt(T))


def analytic_vega(S, K, T, r, sigma, q=0.0):
    d1 = _d1(S, K, T, r, sigma, q)
    return S * jnp.exp(-q * T) * norm.pdf(d1) * jnp.sqrt(T)  # per 1.0 vol unit


def analytic_theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)
    first = -S * jnp.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * jnp.sqrt(T))
    if option_type == "call":
        return first - r * K * jnp.exp(-r * T) * norm.cdf(d2) + q * S * jnp.exp(-q * T) * norm.cdf(d1)
    return first + r * K * jnp.exp(-r * T) * norm.cdf(-d2) - q * S * jnp.exp(-q * T) * norm.cdf(-d1)


def analytic_rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    d2 = _d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * jnp.exp(-r * T) * norm.cdf(d2)
    return -K * T * jnp.exp(-r * T) * norm.cdf(-d2)


# ----- Autodiff Greeks (built once, free for any payoff) -----

def autodiff_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Returns (delta, gamma, vega, theta, rho) using JAX autodiff on bs_price.

    Sign of theta is per-year (decrease in option value as T decreases by 1
    year), matching the analytic convention.
    """
    f = lambda s, k, t, rr, sg: bs_price(s, k, t, rr, sg, q, option_type)
    delta = jax.grad(f, argnums=0)(S, K, T, r, sigma)
    gamma = jax.grad(jax.grad(f, argnums=0), argnums=0)(S, K, T, r, sigma)
    vega = jax.grad(f, argnums=4)(S, K, T, r, sigma)
    # theta = -dV/dT (calendar-time interpretation)
    theta = -jax.grad(f, argnums=2)(S, K, T, r, sigma)
    rho = jax.grad(f, argnums=3)(S, K, T, r, sigma)
    return delta, gamma, vega, theta, rho


if __name__ == "__main__":
    # Quick self-check.
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    price_call = bs_price(S, K, T, r, sigma, option_type="call")
    delta_a = analytic_delta(S, K, T, r, sigma, option_type="call")
    gamma_a = analytic_gamma(S, K, T, r, sigma)
    vega_a = analytic_vega(S, K, T, r, sigma)
    theta_a = analytic_theta(S, K, T, r, sigma, option_type="call")
    rho_a = analytic_rho(S, K, T, r, sigma, option_type="call")

    delta_ad, gamma_ad, vega_ad, theta_ad, rho_ad = autodiff_greeks(S, K, T, r, sigma, option_type="call")

    print(f"Call price       : {price_call:.6f}")
    print(f"Delta  analytic={delta_a:.6f}  autodiff={delta_ad:.6f}")
    print(f"Gamma  analytic={gamma_a:.6f}  autodiff={gamma_ad:.6f}")
    print(f"Vega   analytic={vega_a:.6f}  autodiff={vega_ad:.6f}")
    print(f"Theta  analytic={theta_a:.6f}  autodiff={theta_ad:.6f}")
    print(f"Rho    analytic={rho_a:.6f}  autodiff={rho_ad:.6f}")
