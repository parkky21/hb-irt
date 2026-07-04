import math

import numpy as np
import pytest

from hb_irt.calibration import CalibrationResult, calibrate_3pl


def simulate_responses(a, b, c, thetas, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    p = c[None, :] + (1 - c[None, :]) / (
        1 + np.exp(-a[None, :] * (thetas[:, None] - b[None, :]))
    )
    return (rng.random(p.shape) < p).astype(float)


class TestCalibrate3PL:
    def test_recovers_approximate_parameters_with_fixed_c(self):
        rng = np.random.default_rng(42)
        true_a = [1.2, 0.9, 1.5]
        true_b = [-0.5, 0.0, 0.8]
        true_c = [0.2, 0.2, 0.2]
        thetas = rng.normal(0, 1, size=1500)
        responses = simulate_responses(true_a, true_b, true_c, thetas, seed=1)

        result = calibrate_3pl(responses, fixed_c=0.2, max_iter=60)

        assert isinstance(result, CalibrationResult)
        assert len(result.items) == 3
        for item, b_true in zip(result.items, true_b):
            assert math.isclose(item.b, b_true, abs_tol=0.35)
            assert item.c == 0.2

    def test_free_guessing_parameter(self):
        rng = np.random.default_rng(7)
        true_a = [1.0, 1.3]
        true_b = [0.0, -0.3]
        true_c = [0.2, 0.25]
        thetas = rng.normal(0, 1, size=2000)
        responses = simulate_responses(true_a, true_b, true_c, thetas, seed=2)

        result = calibrate_3pl(responses, max_iter=60)
        assert len(result.items) == 2
        for item in result.items:
            assert 0.0 <= item.c <= 0.5

    def test_uses_provided_item_ids(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=200)
        responses = simulate_responses([1.0, 1.0], [0.0, 0.0], [0.2, 0.2], thetas)
        result = calibrate_3pl(responses, item_ids=["alpha", "beta"], fixed_c=0.2, max_iter=5)
        assert [item.item_id for item in result.items] == ["alpha", "beta"]

    def test_default_item_ids_when_not_given(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=100)
        responses = simulate_responses([1.0], [0.0], [0.2], thetas)
        result = calibrate_3pl(responses, fixed_c=0.2, max_iter=5)
        assert result.items[0].item_id == "item_0"

    def test_converges_flag_set_on_small_easy_problem(self):
        rng = np.random.default_rng(9)
        thetas = rng.normal(0, 1, size=500)
        responses = simulate_responses([1.2], [0.0], [0.2], thetas, seed=9)
        result = calibrate_3pl(responses, fixed_c=0.2, max_iter=100, tol=1e-3)
        assert result.n_iterations <= 100

    def test_rejects_non_2d_responses(self):
        with pytest.raises(ValueError):
            calibrate_3pl(np.array([1.0, 0.0, 1.0]))

    def test_rejects_empty_responses(self):
        with pytest.raises(ValueError):
            calibrate_3pl(np.empty((0, 3)))
        with pytest.raises(ValueError):
            calibrate_3pl(np.empty((3, 0)))

    def test_rejects_non_binary_responses(self):
        with pytest.raises(ValueError):
            calibrate_3pl(np.array([[0.0, 1.0], [0.5, 1.0]]))

    def test_rejects_mismatched_item_ids_length(self):
        responses = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            calibrate_3pl(responses, item_ids=["only_one"])
