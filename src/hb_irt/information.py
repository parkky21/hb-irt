"""Fisher test information and standard error of measurement (spec eq 2, A.2)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models.base import ItemModel


def test_information(models: Sequence[ItemModel], theta: float) -> float:
    """I(theta) = sum_i I_i(theta); information is additive across items (spec A.2)."""
    return sum(model.info(theta) for model in models)


def standard_error(models: Sequence[ItemModel], theta: float) -> float:
    """SE(theta) = 1 / sqrt(I(theta)) (spec A.2). Returns +inf if I(theta) == 0."""
    info = test_information(models, theta)
    if info <= 0:
        return math.inf
    return 1.0 / math.sqrt(info)
