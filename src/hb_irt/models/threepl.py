"""Three-Parameter Logistic model for MCQ items (spec §3.1, eq 4; §A.2)."""

from __future__ import annotations

import math

from ..types import Item
from .base import ItemModel

_EPS = 1e-12


class ThreePLModel(ItemModel):
    """3PL model: P(correct | theta) = c + (1-c) / (1 + exp(-a*(theta-b)))."""

    def __init__(self, item: Item) -> None:
        self.item = item
        self.item_id = item.item_id

    def probability(self, theta: float) -> float:
        """P(X=1 | theta) per spec eq (4)."""
        a, b, c = self.item.a, self.item.b, self.item.c
        return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))

    def loglik(self, value: float, theta: float) -> float:
        if value not in (0, 1):
            raise ValueError(f"3PL response value must be 0 or 1, got {value}")
        p = min(max(self.probability(theta), _EPS), 1.0 - _EPS)
        return math.log(p) if value == 1 else math.log(1.0 - p)

    def info(self, theta: float) -> float:
        """Fisher item information per spec eq (2) / (A.2):

        I(theta) = a^2 * (P(theta)-c)^2 * (1-P(theta)) / ((1-c)^2 * P(theta))
        """
        a, c = self.item.a, self.item.c
        p = min(max(self.probability(theta), _EPS), 1.0 - _EPS)
        q = 1.0 - p
        if c >= 1.0 - _EPS:
            return 0.0
        return a**2 * (p - c) ** 2 * q / ((1.0 - c) ** 2 * p)
