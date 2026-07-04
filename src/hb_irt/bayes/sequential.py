"""Sequential Bayesian updating across test modules (spec §4.4, eq 10).

Each module's posterior becomes the prior for the next module, so uncertainty
accumulates evidence monotonically: posterior variance cannot increase from one
module to the next (spec §4.4, "Posterior variance monotonically decreases").
"""

from __future__ import annotations

from collections.abc import Sequence

from ..quadrature import DEFAULT_N_POINTS
from ..types import Posterior
from .estimation import ModelResponse, eap_estimate

_MIN_SIGMA = 1e-6


def sequential_update(
    prior: Posterior,
    module_responses: Sequence[ModelResponse],
    n_points: int = DEFAULT_N_POINTS,
) -> Posterior:
    """Update `prior` with one module's responses (spec Algorithm §4.4, steps 1-5)."""
    prior_sigma = prior.sem if prior.sem > _MIN_SIGMA else _MIN_SIGMA
    return eap_estimate(
        module_responses,
        prior_mu=prior.mu,
        prior_sigma=prior_sigma,
        n_points=n_points,
    )


def sequential_update_all(
    initial_prior: Posterior,
    modules: Sequence[Sequence[ModelResponse]],
    n_points: int = DEFAULT_N_POINTS,
) -> list[Posterior]:
    """Apply `sequential_update` across modules in order, returning the full
    posterior history (one `Posterior` per module)."""
    history: list[Posterior] = []
    posterior = initial_prior
    for responses in modules:
        posterior = sequential_update(posterior, responses, n_points=n_points)
        history.append(posterior)
    return history
