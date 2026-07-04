"""EAP and MAP ability estimation (spec §4.1-4.3, eq 7-9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from ..models.base import ItemModel
from ..quadrature import DEFAULT_N_POINTS, posterior_moments, quadrature_grid
from ..types import Posterior

ModelResponse = tuple[ItemModel, float]


def _total_loglik(models_responses: Sequence[ModelResponse], theta: float) -> float:
    return sum(model.loglik(value, theta) for model, value in models_responses)


def eap_estimate(
    models_responses: Sequence[ModelResponse],
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    n_points: int = DEFAULT_N_POINTS,
) -> Posterior:
    """Expected A Posteriori estimate via Gauss-Hermite quadrature (spec eq 7, §4.2).

    `models_responses` pairs each response with the `ItemModel` that scored it
    (3PL, GRM, or CRM); mixing model types in one call is supported since each
    contributes only a scalar log-likelihood. With no responses, this reduces to
    the prior itself.
    """
    theta, prior_weight = quadrature_grid(prior_mu, prior_sigma, n_points)
    log_likelihood = np.array([_total_loglik(models_responses, t) for t in theta])
    mean, variance, _ = posterior_moments(log_likelihood, theta, prior_weight)
    return Posterior(mu=mean, variance=variance)


def map_estimate(
    models_responses: Sequence[ModelResponse],
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> float:
    """Maximum A Posteriori ability estimate (spec §4.2): argmax of
    log-likelihood + log-prior, found by bounded scalar optimization.
    """
    if prior_sigma <= 0:
        raise ValueError(f"prior_sigma must be positive, got {prior_sigma}")

    def neg_log_posterior(theta: float) -> float:
        log_prior = norm.logpdf(theta, loc=prior_mu, scale=prior_sigma)
        return -(_total_loglik(models_responses, theta) + log_prior)

    result = minimize_scalar(neg_log_posterior, bounds=bounds, method="bounded")
    return float(result.x)
