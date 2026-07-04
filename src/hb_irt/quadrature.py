"""Gauss-Hermite quadrature helpers for EAP estimation (spec Appendix A.1)."""

from __future__ import annotations

import numpy as np

DEFAULT_N_POINTS = 21


def hermite_nodes_weights(n_points: int = DEFAULT_N_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Physicists' Gauss-Hermite nodes/weights for integrating against e^{-x^2}."""
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    return np.polynomial.hermite.hermgauss(n_points)


def quadrature_grid(
    mu: float, sigma: float, n_points: int = DEFAULT_N_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights approximating integration against N(mu, sigma^2).

    theta_q = mu + sqrt(2) * sigma * x_q
    weight_q = w_q / sqrt(pi)

    weight_q sums to 1 and approximates the Gaussian prior density mass at theta_q,
    so `sum(f(theta_q) * weight_q) ~= E[f(theta)]` for theta ~ N(mu, sigma^2).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    x, w = hermite_nodes_weights(n_points)
    theta = mu + np.sqrt(2.0) * sigma * x
    weight = w / np.sqrt(np.pi)
    return theta, weight


def posterior_moments(
    log_likelihood: np.ndarray,
    theta: np.ndarray,
    prior_weight: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Posterior mean/variance/weights via Gauss-Hermite quadrature (spec eq 7-8, A.1).

    `log_likelihood[q]` is the log-likelihood of the observed responses at node
    `theta[q]`; `prior_weight[q]` is the quadrature weight from `quadrature_grid`
    representing the prior density mass at that node. Returns the EAP mean,
    posterior variance (clamped at 0 for numerical safety), and normalized
    posterior weights over the quadrature grid.
    """
    if not (len(log_likelihood) == len(theta) == len(prior_weight)):
        raise ValueError("log_likelihood, theta, and prior_weight must have equal length")
    ll = log_likelihood - np.max(log_likelihood)
    unnormalized = np.exp(ll) * prior_weight
    total = unnormalized.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("degenerate posterior: likelihood times prior integrates to zero")
    post_weight = unnormalized / total
    mean = float(np.sum(theta * post_weight))
    variance = float(np.sum(theta**2 * post_weight) - mean**2)
    return mean, max(variance, 0.0), post_weight
