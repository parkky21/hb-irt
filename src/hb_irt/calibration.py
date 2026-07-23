"""MMLE/EM calibration of 3PL item parameters (spec §3.2, eq 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .models.crm import CRMItem
from .models.grm import GRMItem
from .quadrature import DEFAULT_N_POINTS, quadrature_grid
from .types import Item

_A_BOUNDS = (0.05, 4.0)
_B_BOUNDS = (-4.0, 4.0)
_C_BOUNDS = (0.0, 0.5)
_LOG_GAP_BOUNDS = (float(np.log(1e-3)), float(np.log(8.0)))
_P_EPS = 1e-10


@dataclass(frozen=True)
class CalibrationResult:
    """Result of MMLE/EM calibration: fitted items and EM convergence info."""

    items: tuple[Item, ...]
    n_iterations: int
    converged: bool


def _p3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def _fit_item(
    theta_q: np.ndarray,
    n_q: np.ndarray,
    r_q: np.ndarray,
    a0: float,
    b0: float,
    c0: float,
    fixed_c: float | None,
) -> tuple[float, float, float]:
    """M-step: maximize the expected complete-data log-likelihood for one item."""

    def neg_expected_ll(params: np.ndarray) -> float:
        if fixed_c is not None:
            a_, b_ = params
            c_ = fixed_c
        else:
            a_, b_, c_ = params
        p = np.clip(_p3pl(theta_q, a_, b_, c_), _P_EPS, 1.0 - _P_EPS)
        return -float(np.sum(r_q * np.log(p) + (n_q - r_q) * np.log(1.0 - p)))

    if fixed_c is not None:
        x0 = [a0, b0]
        bounds = [_A_BOUNDS, _B_BOUNDS]
    else:
        x0 = [a0, b0, c0]
        bounds = [_A_BOUNDS, _B_BOUNDS, _C_BOUNDS]

    result = minimize(neg_expected_ll, x0=x0, bounds=bounds, method="L-BFGS-B")
    if fixed_c is not None:
        a_, b_ = result.x
        return float(a_), float(b_), fixed_c
    a_, b_, c_ = result.x
    return float(a_), float(b_), float(c_)


def calibrate_3pl(
    responses: np.ndarray,
    item_ids: list[str] | None = None,
    fixed_c: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_points: int = DEFAULT_N_POINTS,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> CalibrationResult:
    """Marginal Maximum Likelihood via EM (Bock & Aitkin) for a battery of 3PL
    items (spec eq 5). `responses` is an (n_examinees, n_items) binary matrix.

    `fixed_c` fixes the guessing parameter across all items, recommended when
    fewer than 500 calibration responses are available (spec §3.2).
    """
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 2:
        raise ValueError("responses must be a 2D array (examinees x items)")
    n_examinees, n_items = responses.shape
    if n_examinees == 0 or n_items == 0:
        raise ValueError("responses must be non-empty in both dimensions")
    if not np.isin(responses, (0.0, 1.0)).all():
        raise ValueError("responses must be binary (0/1)")
    if item_ids is None:
        item_ids = [f"item_{j}" for j in range(n_items)]
    if len(item_ids) != n_items:
        raise ValueError("item_ids length must match number of items (columns)")

    theta_q, prior_weight = quadrature_grid(prior_mu, prior_sigma, n_points)

    a = np.full(n_items, 1.0)
    b = np.zeros(n_items)
    c = np.full(n_items, fixed_c if fixed_c is not None else 0.2)

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        # E-step: examinee-level posterior over the quadrature grid.
        p = np.stack([_p3pl(theta_q, a[j], b[j], c[j]) for j in range(n_items)], axis=1)
        p = np.clip(p, _P_EPS, 1.0 - _P_EPS)  # (n_quad, n_items)

        loglik = responses @ np.log(p).T + (1.0 - responses) @ np.log(1.0 - p).T
        loglik -= loglik.max(axis=1, keepdims=True)
        unnormalized = np.exp(loglik) * prior_weight[None, :]
        posterior = unnormalized / unnormalized.sum(axis=1, keepdims=True)

        n_q = posterior.sum(axis=0)
        r_iq = posterior.T @ responses  # (n_quad, n_items)

        new_a, new_b, new_c = a.copy(), b.copy(), c.copy()
        for j in range(n_items):
            new_a[j], new_b[j], new_c[j] = _fit_item(
                theta_q, n_q, r_iq[:, j], a[j], b[j], c[j], fixed_c
            )

        delta = max(
            np.max(np.abs(new_a - a)),
            np.max(np.abs(new_b - b)),
            np.max(np.abs(new_c - c)),
        )
        a, b, c = new_a, new_b, new_c
        if delta < tol:
            converged = True
            break

    items = tuple(
        Item(item_id=item_ids[j], a=float(a[j]), b=float(b[j]), c=float(c[j]))
        for j in range(n_items)
    )
    return CalibrationResult(items=items, n_iterations=iteration, converged=converged)


@dataclass(frozen=True)
class CRMCalibrationResult:
    """Result of CRM MMLE/EM calibration: fitted items and EM convergence info."""

    items: tuple[CRMItem, ...]
    n_iterations: int
    converged: bool


def calibrate_crm(
    scores: np.ndarray,
    item_ids: list[str] | None = None,
    max_score: float = 100.0,
    boundary_clip: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_points: int = DEFAULT_N_POINTS,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> CRMCalibrationResult:
    """Marginal Maximum Likelihood via EM for a battery of CRM items. Not part
    of the spec (which only defines the binary 3PL MCQ item); implements
    Samejima's (1973) continuous response model, matching `models/crm.py`.

    `scores` is an (n_examinees, n_items) matrix of raw 0-max_score responses.
    Because the CRM is linear-Gaussian on the logit-transformed scale (z_ij ~
    N(theta_i - b_j, 1/a_j^2)), the M-step is a closed-form weighted mean/
    variance of residuals rather than the numeric optimization `calibrate_3pl`
    needs for the 3PL's nonlinear response function.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array (examinees x items)")
    n_examinees, n_items = scores.shape
    if n_examinees == 0 or n_items == 0:
        raise ValueError("scores must be non-empty in both dimensions")
    if not np.logical_and(scores >= 0.0, scores <= max_score).all():
        raise ValueError(f"scores must be in [0, {max_score}]")
    if not (0.0 < boundary_clip < max_score / 2.0):
        raise ValueError(
            f"boundary_clip must be in (0, max_score/2), got {boundary_clip} "
            f"(max_score={max_score})"
        )
    if item_ids is None:
        item_ids = [f"item_{j}" for j in range(n_items)]
    if len(item_ids) != n_items:
        raise ValueError("item_ids length must match number of items (columns)")

    lo, hi = boundary_clip, max_score - boundary_clip
    clipped = np.clip(scores, lo, hi)
    z = np.log(clipped / (max_score - clipped))  # (n_examinees, n_items)

    theta_q, prior_weight = quadrature_grid(prior_mu, prior_sigma, n_points)

    a = np.full(n_items, 1.0)
    b = np.zeros(n_items)

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        # E-step: posterior over the quadrature grid for each examinee, from
        # all items jointly. The Jacobian/normalization terms of the CRM
        # density don't depend on theta_q, so they cancel during the
        # per-examinee normalization below and can be dropped here.
        resid = (
            z[:, None, :] - theta_q[None, :, None] + b[None, None, :]
        )  # (n_examinees, n_quad, n_items)
        ll = -0.5 * np.sum((a[None, None, :] * resid) ** 2, axis=2)  # (n_examinees, n_quad)
        ll -= ll.max(axis=1, keepdims=True)
        unnormalized = np.exp(ll) * prior_weight[None, :]
        w = unnormalized / unnormalized.sum(axis=1, keepdims=True)  # (n_examinees, n_quad)

        # M-step: closed-form weighted least squares per item.
        e_theta = w @ theta_q  # (n_examinees,) posterior mean theta per examinee
        new_b = e_theta.mean() - z.mean(axis=0)  # (n_items,)

        resid_new = z[:, None, :] - theta_q[None, :, None] + new_b[None, None, :]
        mean_sq = np.sum(w[:, :, None] * resid_new**2, axis=(0, 1)) / n_examinees  # (n_items,)
        new_a = np.clip(1.0 / np.sqrt(np.maximum(mean_sq, 1e-12)), *_A_BOUNDS)

        delta = max(np.max(np.abs(new_a - a)), np.max(np.abs(new_b - b)))
        a, b = new_a, new_b
        if delta < tol:
            converged = True
            break

    items = tuple(
        CRMItem(
            item_id=item_ids[j],
            a=float(a[j]),
            b=float(b[j]),
            max_score=max_score,
            boundary_clip=boundary_clip,
        )
        for j in range(n_items)
    )
    return CRMCalibrationResult(items=items, n_iterations=iteration, converged=converged)


@dataclass(frozen=True)
class GRMCalibrationResult:
    """Result of GRM MMLE/EM calibration: fitted items and EM convergence info."""

    items: tuple[GRMItem, ...]
    n_iterations: int
    converged: bool


def _grm_category_probs(
    theta_q: np.ndarray, a: float, boundaries: tuple[float, ...]
) -> np.ndarray:
    """P_k(theta_q) for k=0..K-1 (same cumulative-boundary math as `GRMModel`,
    vectorized over the quadrature grid instead of a single theta)."""
    n_quad = theta_q.shape[0]
    k_max = len(boundaries) + 1
    pstar = np.empty((n_quad, k_max + 1))
    pstar[:, 0] = 1.0
    pstar[:, k_max] = 0.0
    for k in range(1, k_max):
        b = boundaries[k - 1]
        pstar[:, k] = 1.0 / (1.0 + np.exp(-a * (theta_q - b)))
    return pstar[:, :-1] - pstar[:, 1:]


def _fit_grm_item(
    theta_q: np.ndarray,
    n_qk: np.ndarray,
    a0: float,
    boundaries0: tuple[float, ...],
) -> tuple[float, tuple[float, ...]]:
    """M-step: maximize the expected complete-data log-likelihood for one GRM
    item. Boundaries are parameterized as `b_1` plus strictly-positive gaps
    (optimized in log-space) so the reconstructed boundaries are always
    increasing regardless of the optimizer's search direction.
    """
    n_gaps = len(boundaries0) - 1
    gaps0 = np.diff(np.asarray(boundaries0))
    log_gaps0 = np.log(np.maximum(gaps0, 1e-3))
    x0 = np.concatenate([[a0, boundaries0[0]], log_gaps0])
    bounds = [_A_BOUNDS, _B_BOUNDS] + [_LOG_GAP_BOUNDS] * n_gaps

    def neg_expected_ll(params: np.ndarray) -> float:
        a_ = params[0]
        b1 = params[1]
        gaps = np.exp(params[2:])
        boundaries = tuple(b1 + np.concatenate([[0.0], np.cumsum(gaps)]))
        probs = np.clip(_grm_category_probs(theta_q, a_, boundaries), _P_EPS, None)
        return -float(np.sum(n_qk * np.log(probs)))

    result = minimize(neg_expected_ll, x0=x0, bounds=bounds, method="L-BFGS-B")
    a_ = float(result.x[0])
    b1 = result.x[1]
    gaps = np.exp(result.x[2:])
    boundaries = tuple((b1 + np.concatenate([[0.0], np.cumsum(gaps)])).tolist())
    return a_, boundaries


def calibrate_grm(
    responses: np.ndarray,
    n_categories: int,
    item_ids: list[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
    n_points: int = DEFAULT_N_POINTS,
    prior_mu: float = 0.0,
    prior_sigma: float = 1.0,
) -> GRMCalibrationResult:
    """Marginal Maximum Likelihood via EM for a battery of GRM items. Not part
    of the spec (which only defines the binary 3PL MCQ item); implements the
    Bock & Aitkin EM algorithm applied to Samejima's (1969) graded response
    model, matching `models/grm.py`.

    `responses` is an (n_examinees, n_items) matrix of integer category
    indices in `[0, n_categories)`.
    """
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 2:
        raise ValueError("responses must be a 2D array (examinees x items)")
    n_examinees, n_items = responses.shape
    if n_examinees == 0 or n_items == 0:
        raise ValueError("responses must be non-empty in both dimensions")
    if n_categories < 2:
        raise ValueError(f"n_categories must be >= 2, got {n_categories}")
    int_responses = responses.astype(int)
    if not np.array_equal(int_responses, responses):
        raise ValueError("responses must contain integer category indices")
    if not np.logical_and(int_responses >= 0, int_responses < n_categories).all():
        raise ValueError(f"responses must be integer categories in [0, {n_categories})")
    if item_ids is None:
        item_ids = [f"item_{j}" for j in range(n_items)]
    if len(item_ids) != n_items:
        raise ValueError("item_ids length must match number of items (columns)")
    for j in range(n_items):
        if len(np.unique(int_responses[:, j])) < 2:
            raise ValueError(
                f"item {item_ids[j]!r} has fewer than 2 distinct observed "
                "categories; its boundaries are unidentifiable"
            )

    theta_q, prior_weight = quadrature_grid(prior_mu, prior_sigma, n_points)
    n_quad = len(theta_q)

    a = np.full(n_items, 1.0)
    init_boundaries = (
        tuple(np.linspace(-2.0, 2.0, n_categories - 1)) if n_categories > 2 else (0.0,)
    )
    boundaries = [init_boundaries for _ in range(n_items)]

    one_hot = np.zeros((n_items, n_examinees, n_categories))
    for j in range(n_items):
        one_hot[j, np.arange(n_examinees), int_responses[:, j]] = 1.0

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        # E-step: joint posterior over the quadrature grid for each examinee,
        # from all items' current parameters (conditional independence given theta).
        ll = np.zeros((n_examinees, n_quad))
        for j in range(n_items):
            probs = np.clip(_grm_category_probs(theta_q, a[j], boundaries[j]), _P_EPS, None)
            ll += one_hot[j] @ np.log(probs).T
        ll -= ll.max(axis=1, keepdims=True)
        unnormalized = np.exp(ll) * prior_weight[None, :]
        w = unnormalized / unnormalized.sum(axis=1, keepdims=True)  # (n_examinees, n_quad)

        new_a = a.copy()
        new_boundaries = list(boundaries)
        for j in range(n_items):
            n_qk = one_hot[j].T @ w  # (n_categories, n_quad)
            new_a[j], new_boundaries[j] = _fit_grm_item(
                theta_q, n_qk.T, a[j], boundaries[j]
            )

        delta_a = float(np.max(np.abs(new_a - a)))
        delta_b = max(
            max(abs(x - y) for x, y in zip(new_boundaries[j], boundaries[j]))
            for j in range(n_items)
        )
        a, boundaries = new_a, new_boundaries
        if max(delta_a, delta_b) < tol:
            converged = True
            break

    items = tuple(
        GRMItem(item_id=item_ids[j], a=float(a[j]), boundaries=boundaries[j])
        for j in range(n_items)
    )
    return GRMCalibrationResult(items=items, n_iterations=iteration, converged=converged)
