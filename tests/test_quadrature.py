import math

import numpy as np
import pytest

from hb_irt.quadrature import hermite_nodes_weights, posterior_moments, quadrature_grid


def test_hermite_nodes_weights_symmetric_and_sums_to_sqrt_pi():
    x, w = hermite_nodes_weights(21)
    assert len(x) == 21
    assert np.isclose(np.sort(x), -np.sort(x)[::-1]).all()
    assert math.isclose(w.sum(), math.sqrt(math.pi), rel_tol=1e-9)


def test_hermite_nodes_weights_rejects_invalid_n_points():
    with pytest.raises(ValueError):
        hermite_nodes_weights(0)


def test_quadrature_grid_weights_sum_to_one():
    theta, weight = quadrature_grid(mu=0.0, sigma=1.0, n_points=21)
    assert math.isclose(weight.sum(), 1.0, rel_tol=1e-9)
    assert math.isclose(np.sum(theta * weight), 0.0, abs_tol=1e-9)


def test_quadrature_grid_matches_prior_mean_and_variance():
    mu, sigma = 0.5, 1.3
    theta, weight = quadrature_grid(mu=mu, sigma=sigma, n_points=40)
    approx_mean = np.sum(theta * weight)
    approx_var = np.sum(theta**2 * weight) - approx_mean**2
    assert math.isclose(approx_mean, mu, abs_tol=1e-6)
    assert math.isclose(approx_var, sigma**2, rel_tol=1e-6)


def test_quadrature_grid_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        quadrature_grid(mu=0.0, sigma=0.0, n_points=21)
    with pytest.raises(ValueError):
        quadrature_grid(mu=0.0, sigma=-1.0, n_points=21)


def test_quadrature_grid_rejects_invalid_n_points():
    with pytest.raises(ValueError):
        quadrature_grid(mu=0.0, sigma=1.0, n_points=0)


def test_posterior_moments_uniform_likelihood_recovers_prior():
    theta, prior_weight = quadrature_grid(mu=0.2, sigma=0.9, n_points=31)
    log_likelihood = np.zeros_like(theta)
    mean, var, post_weight = posterior_moments(log_likelihood, theta, prior_weight)
    assert math.isclose(mean, 0.2, abs_tol=1e-6)
    assert math.isclose(var, 0.9**2, rel_tol=1e-6)
    assert math.isclose(post_weight.sum(), 1.0, rel_tol=1e-9)


def test_posterior_moments_peaked_likelihood_shifts_mean():
    theta, prior_weight = quadrature_grid(mu=0.0, sigma=1.5, n_points=41)
    # Likelihood strongly favors theta near 1.0
    log_likelihood = -0.5 * ((theta - 1.0) / 0.2) ** 2
    mean, var, _ = posterior_moments(log_likelihood, theta, prior_weight)
    assert mean > 0.5
    assert var < 1.5**2


def test_posterior_moments_rejects_mismatched_lengths():
    theta, prior_weight = quadrature_grid(mu=0.0, sigma=1.0, n_points=21)
    with pytest.raises(ValueError):
        posterior_moments(np.zeros(5), theta, prior_weight)


def test_posterior_moments_rejects_degenerate_posterior():
    theta, prior_weight = quadrature_grid(mu=0.0, sigma=1.0, n_points=21)
    log_likelihood = np.full_like(theta, -1e300)
    prior_weight = np.zeros_like(prior_weight)
    with pytest.raises(ValueError):
        posterior_moments(log_likelihood, theta, prior_weight)
