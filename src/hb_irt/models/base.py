"""Common interface for IRT item models (3PL, GRM, CRM)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ItemModel(ABC):
    """An item model gives log-likelihood and Fisher information at a given ability.

    Concrete models interpret `value` differently: 0/1 for a 3PL (MCQ) item, an
    integer category index for a GRM item, or a raw continuous score for a CRM
    item. `loglik` and `info` are the two operations required by the Bayesian
    estimation and MSAT selection code in this package.
    """

    item_id: str

    @abstractmethod
    def loglik(self, value: float, theta: float) -> float:
        """Log-likelihood of an observed response `value` at ability `theta`."""

    @abstractmethod
    def info(self, theta: float) -> float:
        """Fisher information the item provides at ability `theta`."""
