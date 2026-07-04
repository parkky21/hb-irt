"""Continuous Response Model for QA responses on a continuous 0-100 scale.

Not part of the spec (which only defines the binary 3PL MCQ item); added to cover
"IRT for QA where answer is a score 0-100." Implements Samejima's (1973)
homogeneous continuous response model: the raw score xi in (0, max_score) is
logit-transformed to z(xi), and z is modeled as Normal(theta - b, 1/a^2) --
i.e. a linear-Gaussian measurement model on the transformed scale, exactly
analogous in spirit to the 3PL's location-scale role for `b` and `a`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from .base import ItemModel

_EPS = 1e-9


@dataclass(frozen=True)
class CRMItem:
    """CRM item parameters.

    a: discrimination (precision of the transformed response given theta)
    b: difficulty / location
    max_score: upper bound of the raw response scale (e.g. 100.0)
    """

    item_id: str
    a: float
    b: float
    max_score: float = 100.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"discrimination a must be > 0, got {self.a}")
        if self.max_score <= 0:
            raise ValueError(f"max_score must be > 0, got {self.max_score}")


class CRMModel(ItemModel):
    """Samejima continuous response model on a bounded raw score scale."""

    def __init__(self, item: CRMItem) -> None:
        self.item = item
        self.item_id = item.item_id

    def _clip(self, value: float) -> float:
        lo, hi = _EPS * self.item.max_score, self.item.max_score * (1.0 - _EPS)
        if not (0.0 <= value <= self.item.max_score):
            raise ValueError(
                f"CRM response must be in [0, {self.item.max_score}], got {value}"
            )
        return min(max(value, lo), hi)

    def transform(self, value: float) -> float:
        """z(xi) = ln(xi / (max_score - xi)), the logit transform to the real line."""
        xi = self._clip(value)
        return math.log(xi / (self.item.max_score - xi))

    def _transform_derivative(self, value: float) -> float:
        xi = self._clip(value)
        return self.item.max_score / (xi * (self.item.max_score - xi))

    def loglik(self, value: float, theta: float) -> float:
        """log f(xi | theta) = log[a * phi(a*(z-(theta-b)))] + log|dz/dxi|."""
        a, b = self.item.a, self.item.b
        z = self.transform(value)
        dz = self._transform_derivative(value)
        density = a * norm.pdf(a * (z - (theta - b))) * dz
        density = max(density, 1e-300)
        return math.log(density)

    def info(self, theta: float) -> float:
        """Fisher information is constant (a^2): z|theta is linear-Gaussian with
        variance 1/a^2, independent of theta."""
        return self.item.a**2
