"""MSAT stopping rules (spec §2.5, Table 3)."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Posterior


@dataclass(frozen=True)
class StoppingConfig:
    """Default parameter values per spec Table 3."""

    sigma_min: float = 0.3
    max_modules: int = 8
    min_items: int = 20
    delta_saturation: float = 0.01


@dataclass(frozen=True)
class StoppingDecision:
    should_stop: bool
    reasons: tuple[str, ...]


def evaluate_stopping(
    posterior: Posterior,
    previous_posterior: Posterior | None,
    n_modules: int,
    n_items: int,
    config: StoppingConfig = StoppingConfig(),
) -> StoppingDecision:
    """Evaluate MSAT stopping rules after a module (spec Table 3, §2.5).

    "Minimum Items" is treated as a floor that gates the other three rules
    rather than an independent trigger: the spec's intent is to ensure enough
    measurement has occurred before terminating, not to force a stop the
    moment N_min items have been seen regardless of precision.
    """
    if n_items < config.min_items:
        return StoppingDecision(should_stop=False, reasons=())

    reasons: list[str] = []
    if posterior.sem < config.sigma_min:
        reasons.append("precision_threshold")
    if n_modules >= config.max_modules:
        reasons.append("maximum_modules")
    if previous_posterior is not None:
        delta = previous_posterior.variance - posterior.variance
        if delta < config.delta_saturation:
            reasons.append("information_saturation")

    return StoppingDecision(should_stop=bool(reasons), reasons=tuple(reasons))
