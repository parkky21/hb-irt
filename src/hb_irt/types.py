"""Shared data types for items, responses, posteriors, and scores (spec Table 5, 7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    """3PL item parameters (spec §3.1, Table 5).

    a: discrimination (typical range 0.5-2.5)
    b: difficulty, ability level at which P(correct) = (1+c)/2
    c: guessing / lower asymptote (typical range 0.20-0.25)
    """

    item_id: str
    a: float
    b: float
    c: float = 0.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"discrimination a must be > 0, got {self.a}")
        if not (0.0 <= self.c < 1.0):
            raise ValueError(f"guessing c must be in [0, 1), got {self.c}")


@dataclass(frozen=True)
class Response:
    """A single observed response to an item.

    `value` is model-dependent: 0/1 for a 3PL (MCQ) item, an integer category
    index in [0, n_categories) for a GRM item, or a raw continuous score for a
    CRM item.
    """

    item_id: str
    value: float


@dataclass(frozen=True)
class Posterior:
    """Ability posterior N(mu, variance) on the logit theta scale (spec §4.1-4.3)."""

    mu: float
    variance: float

    def __post_init__(self) -> None:
        if self.variance < 0:
            raise ValueError(f"variance must be >= 0, got {self.variance}")

    @property
    def sem(self) -> float:
        """Standard error of measurement: sqrt(variance) (spec eq 9)."""
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Symmetric normal-approximation credible interval (spec eq 9).

        `level` must be in (0, 1); 0.95 uses the z=1.96 multiplier from the spec.
        """
        if not (0.0 < level < 1.0):
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = _z_for_level(level)
        margin = z * self.sem
        return self.mu - margin, self.mu + margin


def _z_for_level(level: float) -> float:
    """Two-sided normal critical value for a given credible-interval level."""
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2.0))


@dataclass(frozen=True)
class TestModule:
    """A pre-constructed test module of 5-20 items (spec §2.2, eq 1)."""

    __test__ = False  # not a pytest test class despite the name

    module_id: str
    items: tuple[Item, ...]
    module_type: str = "medium"
    n_exposures: int = 0

    def __post_init__(self) -> None:
        if not (5 <= len(self.items) <= 20):
            raise ValueError(
                f"module must contain 5-20 items, got {len(self.items)}"
            )

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class SubskillScore:
    """Sub-skill score data model with full uncertainty information (spec Table 7)."""

    subskill_id: str
    theta_eap: float
    theta_variance: float
    theta_sem: float
    score_0_100: float
    margin_error_95: float
    ci_lower_95: float
    ci_upper_95: float
    items_administered: int
    modules_completed: int
    level_thetas: dict[str, float] = field(default_factory=dict)
