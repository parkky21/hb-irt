"""Information-maximizing MSAT module selection (spec §2.3-2.4, eq 2-3)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models.threepl import ThreePLModel
from ..quadrature import DEFAULT_N_POINTS, quadrature_grid
from ..types import Posterior, TestModule
from .module_bank import ModuleBank

DEFAULT_ALPHA = 2.0  # exploration bonus weight (spec §2.3)
DEFAULT_BETA = 5.0  # exposure decay rate (spec §2.3)


def information_at_estimate(module: TestModule, theta: float) -> float:
    """I_Q(mu_curr) = sum_i I_i(mu_curr), the module's Fisher information
    evaluated at a single ability value (spec Algorithm §2.3, step 1a)."""
    return sum(ThreePLModel(item).info(theta) for item in module.items)


def expected_information_gain(
    module: TestModule,
    posterior: Posterior,
    n_points: int = DEFAULT_N_POINTS,
) -> float:
    """EIG(Q) = integral of I_Q(theta) over the current posterior (spec eq 3),
    approximated via Gauss-Hermite quadrature (spec Appendix A.1)."""
    theta_q, weight_q = quadrature_grid(posterior.mu, posterior.sem, n_points=n_points)
    return float(sum(w * information_at_estimate(module, t) for t, w in zip(theta_q, weight_q)))


def expected_variance_reduction(module: TestModule, posterior: Posterior) -> float:
    """Delta_sigma^2 = sigma_curr^2 - E[sigma_new^2 | Q] (spec Algorithm §2.3, step 1b).

    E[sigma_new^2 | Q] is approximated by the standard normal-approximation
    update to posterior precision: precision_new = precision_curr + I_Q(mu_curr).
    This is the same asymptotic approximation used operationally in CAT engines
    to avoid simulating every possible response pattern for Q.
    """
    info_q = information_at_estimate(module, posterior.mu)
    if posterior.variance <= 0:
        return 0.0
    precision_curr = 1.0 / posterior.variance
    variance_new = 1.0 / (precision_curr + info_q)
    return posterior.variance - variance_new


def selection_score(
    module: TestModule,
    posterior: Posterior,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """S(Q) = I_Q(mu_curr) + alpha * exp(-N_Q / beta) (spec Algorithm §2.3, step 1c)."""
    info_q = information_at_estimate(module, posterior.mu)
    exploration_bonus = alpha * math.exp(-module.n_exposures / beta)
    return info_q + exploration_bonus


def select_next_module(
    bank: ModuleBank,
    posterior: Posterior,
    administered_ids: Iterable[str],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> TestModule:
    """Select the next module to administer (spec Algorithm §2.3, steps 1-3)."""
    candidates = bank.available(administered_ids)
    if not candidates:
        raise ValueError("no modules remain available in the bank for this candidate")
    return max(candidates, key=lambda module: selection_score(module, posterior, alpha, beta))
