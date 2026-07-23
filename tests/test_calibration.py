import math

import numpy as np
import pytest

from hb_irt.calibration import (
    CalibrationResult,
    CRMCalibrationResult,
    GRMCalibrationResult,
    calibrate_3pl,
    calibrate_crm,
    calibrate_grm,
)
from hb_irt.models.grm import GRMItem, GRMModel


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


def simulate_crm_scores(a, b, thetas, max_score=100.0, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    z = (thetas[:, None] - b[None, :]) + rng.normal(size=(len(thetas), len(a))) / a[None, :]
    xi = max_score / (1.0 + np.exp(-z))
    return np.clip(xi, 0.0, max_score)


class TestCalibrateCRM:
    def test_recovers_approximate_parameters(self):
        rng = np.random.default_rng(42)
        true_a = [1.2, 0.9, 1.5, 1.0]
        true_b = [-0.5, 0.0, 0.8, 0.3]
        thetas = rng.normal(0, 1, size=800)
        scores = simulate_crm_scores(true_a, true_b, thetas, seed=1)

        result = calibrate_crm(scores, max_iter=200, tol=1e-5)

        assert isinstance(result, CRMCalibrationResult)
        assert len(result.items) == 4
        for item, a_true, b_true in zip(result.items, true_a, true_b):
            assert math.isclose(item.a, a_true, abs_tol=0.2)
            assert math.isclose(item.b, b_true, abs_tol=0.2)

    def test_converges_flag_set(self):
        rng = np.random.default_rng(9)
        thetas = rng.normal(0, 1, size=500)
        scores = simulate_crm_scores([1.2], [0.0], thetas, seed=9)
        result = calibrate_crm(scores, max_iter=200, tol=1e-4)
        assert result.converged
        assert result.n_iterations <= 200

    def test_handles_exact_boundary_scores_in_input(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=300)
        scores = simulate_crm_scores([1.0, 1.3], [0.0, -0.2], thetas, seed=4)
        scores[0, 0] = 100.0
        scores[1, 0] = 0.0
        result = calibrate_crm(scores, max_iter=100)
        for item in result.items:
            assert math.isfinite(item.a)
            assert math.isfinite(item.b)

    def test_uses_provided_item_ids(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=200)
        scores = simulate_crm_scores([1.0, 1.0], [0.0, 0.0], thetas)
        result = calibrate_crm(scores, item_ids=["alpha", "beta"], max_iter=10)
        assert [item.item_id for item in result.items] == ["alpha", "beta"]

    def test_default_item_ids_when_not_given(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=100)
        scores = simulate_crm_scores([1.0], [0.0], thetas)
        result = calibrate_crm(scores, max_iter=10)
        assert result.items[0].item_id == "item_0"

    def test_fitted_items_carry_max_score_and_boundary_clip(self):
        rng = np.random.default_rng(5)
        thetas = rng.normal(0, 1, size=100)
        scores = simulate_crm_scores([1.0], [0.0], thetas, max_score=10.0)
        result = calibrate_crm(scores, max_score=10.0, boundary_clip=0.2, max_iter=10)
        assert result.items[0].max_score == 10.0
        assert result.items[0].boundary_clip == 0.2

    def test_rejects_non_2d_scores(self):
        with pytest.raises(ValueError):
            calibrate_crm(np.array([50.0, 60.0]))

    def test_rejects_empty_scores(self):
        with pytest.raises(ValueError):
            calibrate_crm(np.empty((0, 3)))
        with pytest.raises(ValueError):
            calibrate_crm(np.empty((3, 0)))

    def test_rejects_out_of_range_scores(self):
        with pytest.raises(ValueError):
            calibrate_crm(np.array([[50.0, 105.0], [30.0, -1.0]]))

    def test_rejects_invalid_boundary_clip(self):
        scores = np.array([[50.0, 60.0], [40.0, 55.0]])
        with pytest.raises(ValueError):
            calibrate_crm(scores, boundary_clip=0.0)
        with pytest.raises(ValueError):
            calibrate_crm(scores, boundary_clip=50.0)

    def test_rejects_mismatched_item_ids_length(self):
        scores = np.array([[50.0, 60.0], [40.0, 55.0]])
        with pytest.raises(ValueError):
            calibrate_crm(scores, item_ids=["only_one"])


def simulate_grm_responses(a, boundaries, thetas, rng):
    model = GRMModel(GRMItem(item_id="x", a=a, boundaries=boundaries))
    out = np.empty(len(thetas), dtype=int)
    for i, theta in enumerate(thetas):
        probs = model.category_probabilities(theta)
        out[i] = rng.choice(len(probs), p=probs / probs.sum())
    return out


class TestCalibrateGRM:
    N_CATEGORIES = 11  # 0-10 ordinal scale

    def _simulate_two_items(self, seed=11, n=1500):
        rng = np.random.default_rng(seed)
        true_a = [1.2, 0.9]
        true_boundaries = [
            tuple(np.linspace(-2.0, 2.0, self.N_CATEGORIES - 1)),
            tuple(np.linspace(-1.5, 1.8, self.N_CATEGORIES - 1)),
        ]
        thetas = rng.normal(0, 1, size=n)
        responses = np.stack(
            [
                simulate_grm_responses(true_a[0], true_boundaries[0], thetas, rng),
                simulate_grm_responses(true_a[1], true_boundaries[1], thetas, rng),
            ],
            axis=1,
        )
        return responses, true_a, true_boundaries

    def test_recovers_approximate_parameters(self):
        responses, true_a, true_boundaries = self._simulate_two_items()
        result = calibrate_grm(responses, n_categories=self.N_CATEGORIES, max_iter=100)

        assert isinstance(result, GRMCalibrationResult)
        assert len(result.items) == 2
        for item, a_true, b_true in zip(result.items, true_a, true_boundaries):
            assert math.isclose(item.a, a_true, abs_tol=0.3)
            for fitted, true_val in zip(item.boundaries, b_true):
                assert math.isclose(fitted, true_val, abs_tol=0.5)

    def test_fitted_boundaries_are_strictly_increasing(self):
        responses, _, _ = self._simulate_two_items()
        result = calibrate_grm(responses, n_categories=self.N_CATEGORIES, max_iter=100)
        for item in result.items:
            assert all(x < y for x, y in zip(item.boundaries, item.boundaries[1:]))

    def test_converges_flag_set(self):
        responses, _, _ = self._simulate_two_items()
        result = calibrate_grm(responses, n_categories=self.N_CATEGORIES, max_iter=100, tol=1e-4)
        assert result.converged
        assert result.n_iterations <= 100

    def test_handles_binary_like_two_category_item(self):
        rng = np.random.default_rng(3)
        thetas = rng.normal(0, 1, size=500)
        responses = simulate_grm_responses(1.3, (0.2,), thetas, rng).reshape(-1, 1)
        result = calibrate_grm(responses, n_categories=2, max_iter=100)
        assert len(result.items[0].boundaries) == 1
        assert math.isfinite(result.items[0].a)

    def test_uses_provided_item_ids(self):
        responses, _, _ = self._simulate_two_items(n=100)
        result = calibrate_grm(
            responses, n_categories=self.N_CATEGORIES, item_ids=["alpha", "beta"], max_iter=5
        )
        assert [item.item_id for item in result.items] == ["alpha", "beta"]

    def test_default_item_ids_when_not_given(self):
        responses, _, _ = self._simulate_two_items(n=100)
        result = calibrate_grm(responses, n_categories=self.N_CATEGORIES, max_iter=5)
        assert result.items[0].item_id == "item_0"
        assert result.items[1].item_id == "item_1"

    def test_rejects_non_2d_responses(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([1.0, 2.0, 3.0]), n_categories=5)

    def test_rejects_empty_responses(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.empty((0, 3)), n_categories=5)
        with pytest.raises(ValueError):
            calibrate_grm(np.empty((3, 0)), n_categories=5)

    def test_rejects_n_categories_below_two(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([[0, 1], [1, 0]]), n_categories=1)

    def test_rejects_non_integer_responses(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([[0.5, 1.0], [1.0, 0.0]]), n_categories=3)

    def test_rejects_out_of_range_category(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([[0, 5], [1, 0]]), n_categories=3)

    def test_rejects_item_with_fewer_than_two_distinct_categories(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([[0, 0], [0, 0], [0, 0]]), n_categories=3)

    def test_rejects_mismatched_item_ids_length(self):
        with pytest.raises(ValueError):
            calibrate_grm(np.array([[0, 1], [1, 0]]), n_categories=3, item_ids=["only_one"])
