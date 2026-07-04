"""Bayesian updating and IRT (MCQ 3PL, graded/continuous QA) for candidate skill assessment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hb-irt")
except PackageNotFoundError:  # pragma: no cover - only when package isn't installed
    __version__ = "0.0.0"
