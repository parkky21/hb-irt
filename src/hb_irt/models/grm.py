"""Samejima Graded Response Model for ordinal QA responses (0-10 scale, 11 categories).

Not part of the spec (which only defines the binary 3PL MCQ item); added to cover
"IRT for QA where answer is a score 0-10." Uses the standard Samejima (1969)
cumulative-boundary parameterization so it composes with the rest of the package
(same theta scale, same loglik/info interface as the 3PL model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import ItemModel

_EPS = 1e-12


@dataclass(frozen=True)
class GRMItem:
    """GRM item: one discrimination `a` and K-1 strictly increasing boundaries.

    `boundaries[k]` is the difficulty of the k-th category boundary (b_1 < b_2 <
    ... < b_{K-1}); the number of ordered categories is `len(boundaries) + 1`
    (e.g. 10 boundaries -> 11 categories for a 0-10 score).
    """

    item_id: str
    a: float
    boundaries: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"discrimination a must be > 0, got {self.a}")
        if len(self.boundaries) < 1:
            raise ValueError("GRM item needs at least 1 boundary (2 categories)")
        if any(x >= y for x, y in zip(self.boundaries, self.boundaries[1:])):
            raise ValueError("boundaries must be strictly increasing")

    @property
    def n_categories(self) -> int:
        return len(self.boundaries) + 1


class GRMModel(ItemModel):
    """Graded Response Model.

    Cumulative boundary response function (k=1..K-1):
        P*_k(theta) = P(X >= k | theta) = 1 / (1 + exp(-a*(theta - b_k)))
    with P*_0 = 1 and P*_K = 0. Category probability:
        P_k(theta) = P*_k(theta) - P*_{k+1}(theta), k = 0..K-1
    """

    def __init__(self, item: GRMItem) -> None:
        self.item = item
        self.item_id = item.item_id

    def _boundary_prob(self, theta: float, k: int) -> float:
        if k <= 0:
            return 1.0
        if k >= self.item.n_categories:
            return 0.0
        b = self.item.boundaries[k - 1]
        return 1.0 / (1.0 + math.exp(-self.item.a * (theta - b)))

    def category_probabilities(self, theta: float) -> np.ndarray:
        """P_k(theta) for k=0..K-1, summing to 1."""
        k_max = self.item.n_categories
        pstar = np.array([self._boundary_prob(theta, k) for k in range(k_max + 1)])
        return pstar[:-1] - pstar[1:]

    def loglik(self, value: float, theta: float) -> float:
        k = int(value)
        if k != value or not (0 <= k < self.item.n_categories):
            raise ValueError(
                f"GRM response must be an integer category in "
                f"[0, {self.item.n_categories - 1}], got {value}"
            )
        probs = self.category_probabilities(theta)
        p = max(probs[k], _EPS)
        return math.log(p)

    def info(self, theta: float) -> float:
        """Item information: sum_k [P_k'(theta)]^2 / P_k(theta) (Samejima 1969)."""
        a = self.item.a
        k_max = self.item.n_categories
        pstar = np.array([self._boundary_prob(theta, k) for k in range(k_max + 1)])
        dpstar = a * pstar * (1.0 - pstar)
        probs = pstar[:-1] - pstar[1:]
        dprobs = dpstar[:-1] - dpstar[1:]
        probs_safe = np.clip(probs, _EPS, None)
        return float(np.sum(dprobs**2 / probs_safe))
