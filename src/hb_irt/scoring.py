"""0-100 rescaling and precision-weighted level aggregation (spec §5.1-5.3, eq 11-15)."""

from __future__ import annotations

from collections.abc import Mapping

from .types import Posterior, SubskillScore

Z_95 = 1.96
TAU_SQUARED_DEFAULT = 0.5  # tau^2 regularization for level aggregation (spec eq 14)


def rescale_0_100(posterior: Posterior) -> tuple[float, float, float, float]:
    """Rescale a logit-scale posterior to 0-100 (spec eq 11-13).

    Returns (score, margin_error_95, ci_lower_95, ci_upper_95).
    """
    score = 50.0 + 10.0 * posterior.mu
    margin = 10.0 * Z_95 * posterior.sem  # = 19.6 * SE (eq 12)
    return score, margin, score - margin, score + margin


def aggregate_levels(
    level_thetas: Mapping[str, float],
    level_variances: Mapping[str, float],
    tau_squared: float = TAU_SQUARED_DEFAULT,
) -> Posterior:
    """Precision-weighted aggregation of per-Bloom-level EAP estimates into a
    single sub-skill posterior (spec eq 14-15):

    theta_k = sum_l(w_l * theta_kl) / sum_l(w_l), w_l = 1 / (var_kl + tau^2)
    var_k = 1 / sum_l(w_l)
    """
    if level_thetas.keys() != level_variances.keys():
        raise ValueError("level_thetas and level_variances must have the same keys")
    if not level_thetas:
        raise ValueError("at least one Bloom level estimate is required")
    if tau_squared <= 0:
        raise ValueError(f"tau_squared must be > 0, got {tau_squared}")

    weights = {level: 1.0 / (level_variances[level] + tau_squared) for level in level_thetas}
    total_weight = sum(weights.values())
    theta_k = sum(weights[level] * level_thetas[level] for level in level_thetas) / total_weight
    var_k = 1.0 / total_weight
    return Posterior(mu=theta_k, variance=var_k)


def build_subskill_score(
    subskill_id: str,
    posterior: Posterior,
    items_administered: int,
    modules_completed: int,
    level_thetas: Mapping[str, float] | None = None,
) -> SubskillScore:
    """Assemble the full sub-skill score data model (spec Table 7)."""
    score, margin, lower, upper = rescale_0_100(posterior)
    return SubskillScore(
        subskill_id=subskill_id,
        theta_eap=posterior.mu,
        theta_variance=posterior.variance,
        theta_sem=posterior.sem,
        score_0_100=score,
        margin_error_95=margin,
        ci_lower_95=lower,
        ci_upper_95=upper,
        items_administered=items_administered,
        modules_completed=modules_completed,
        level_thetas=dict(level_thetas) if level_thetas else {},
    )
