"""Dispatch from an item's data type to its `ItemModel` (3PL / GRM / CRM).

Lets code that only holds item data (e.g. `TestModule.items`, MSAT selection)
build the right model without hardcoding which item type it's working with.
"""

from __future__ import annotations

from ..types import Item
from .base import ItemModel
from .crm import CRMItem, CRMModel
from .grm import GRMItem, GRMModel
from .threepl import ThreePLModel


def build_model(item: Item | GRMItem | CRMItem) -> ItemModel:
    """Construct the `ItemModel` matching `item`'s concrete type."""
    if isinstance(item, Item):
        return ThreePLModel(item)
    if isinstance(item, GRMItem):
        return GRMModel(item)
    if isinstance(item, CRMItem):
        return CRMModel(item)
    raise TypeError(f"no ItemModel registered for item type {type(item).__name__}")
