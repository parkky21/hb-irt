"""Module bank: repository of pre-calibrated test modules (spec §2.1-2.2, Table 1-2)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..types import TestModule

MODULE_TYPES = ("easy", "medium", "hard", "challenge")

# Target ability range per module type (spec Table 2).
MODULE_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "easy": (-3.0, -0.5),
    "medium": (-1.0, 1.0),
    "hard": (0.5, 3.0),
    "challenge": (2.0, 4.0),
}


@dataclass(frozen=True)
class ModuleBank:
    """Repository of test modules queryable by type and administration history."""

    modules: tuple[TestModule, ...]

    def __post_init__(self) -> None:
        unknown = {m.module_type for m in self.modules} - set(MODULE_TYPES)
        if unknown:
            raise ValueError(f"unknown module types: {sorted(unknown)}")
        ids = [m.module_id for m in self.modules]
        if len(ids) != len(set(ids)):
            raise ValueError("module_id values must be unique within a bank")

    def by_type(self, module_type: str) -> tuple[TestModule, ...]:
        if module_type not in MODULE_TYPES:
            raise ValueError(f"unknown module type: {module_type!r}")
        return tuple(m for m in self.modules if m.module_type == module_type)

    def available(self, administered_ids: Iterable[str]) -> tuple[TestModule, ...]:
        """Modules not yet administered to this candidate (B \\ H, spec §2.3)."""
        administered = set(administered_ids)
        return tuple(m for m in self.modules if m.module_id not in administered)

    def get(self, module_id: str) -> TestModule:
        for m in self.modules:
            if m.module_id == module_id:
                return m
        raise KeyError(f"no module with id {module_id!r}")
