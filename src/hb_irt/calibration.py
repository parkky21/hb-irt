"""MMLE/EM calibration of 3PL item parameters (spec §3.2, eq 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .quadrature import DEFAULT_N_POINTS, quadrature_grid
from .types import Item

_A_BOUNDS = (0.05, 4.0)
_B_BOUNDS = (-4.0, 4.0)
_C_BOUNDS = (0.0, 0.5)
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
