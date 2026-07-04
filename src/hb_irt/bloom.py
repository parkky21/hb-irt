"""Bloom level to IRT difficulty anchor mapping with shrinkage (spec §3.3, eq 6, Table 6)."""

from __future__ import annotations

BLOOM_DIFFICULTY_ANCHORS: dict[str, float] = {
    "L1": -1.5,  # Recall
    "L2": -0.5,  # Comprehension
    "L3": 0.5,  # Application
    "L4": 1.2,  # Analysis
    "L5": 2.0,  # Evaluation
    "L6": 2.8,  # Synthesis
}

BLOOM_LEVEL_NAMES: dict[str, str] = {
    "L1": "Recall",
    "L2": "Comprehension",
    "L3": "Application",
    "L4": "Analysis",
    "L5": "Evaluation",
    "L6": "Synthesis",
}


def difficulty_anchor(level: str) -> float:
    """b_l for a Bloom level (spec Table 6)."""
    try:
        return BLOOM_DIFFICULTY_ANCHORS[level]
    except KeyError as exc:
        raise ValueError(f"unknown Bloom level: {level!r}") from exc


def shrink_difficulty(
    raw_difficulty: float,
    raw_variance: float,
    level: str,
    sigma_b: float,
) -> float:
    """Empirical-Bayes shrinkage of a raw calibrated difficulty toward its Bloom
    level anchor (spec eq 6: b_i = b_l + epsilon_i, epsilon_i ~ N(0, sigma_b^2)).

    Treats b_l as the prior mean and sigma_b^2 as the prior variance for item
    difficulty; the raw calibration estimate (raw_difficulty, raw_variance) is
    the likelihood. The shrinkage estimate is their precision-weighted mean,
    the same precision-weighting pattern used elsewhere in the spec (eq 14, 17).
    """
    if raw_variance <= 0:
        raise ValueError(f"raw_variance must be > 0, got {raw_variance}")
    if sigma_b <= 0:
        raise ValueError(f"sigma_b must be > 0, got {sigma_b}")
    anchor = difficulty_anchor(level)
    prior_precision = 1.0 / sigma_b**2
    raw_precision = 1.0 / raw_variance
    return (raw_precision * raw_difficulty + prior_precision * anchor) / (
        raw_precision + prior_precision
    )
