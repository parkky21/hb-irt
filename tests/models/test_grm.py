import math

import numpy as np
import pytest

from hb_irt.models.grm import GRMItem, GRMModel


def make_model(a=1.0, boundaries=(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7)):
    return GRMModel(GRMItem(item_id="qa1", a=a, boundaries=tuple(boundaries)))


class TestGRMItem:
    def test_n_categories(self):
        item = GRMItem(item_id="qa1", a=1.0, boundaries=tuple(range(10)))
        assert item.n_categories == 11

    def test_rejects_nonpositive_discrimination(self):
        with pytest.raises(ValueError):
            GRMItem(item_id="qa1", a=0.0, boundaries=(0.0,))

    def test_rejects_empty_boundaries(self):
        with pytest.raises(ValueError):
            GRMItem(item_id="qa1", a=1.0, boundaries=())

    def test_rejects_non_increasing_boundaries(self):
        with pytest.raises(ValueError):
            GRMItem(item_id="qa1", a=1.0, boundaries=(0.0, 0.0))
        with pytest.raises(ValueError):
            GRMItem(item_id="qa1", a=1.0, boundaries=(1.0, 0.0))


class TestCategoryProbabilities:
    def test_probabilities_sum_to_one(self):
        model = make_model()
        for theta in (-4, -1, 0, 1, 4):
            probs = model.category_probabilities(theta)
            assert math.isclose(probs.sum(), 1.0, abs_tol=1e-9)

    def test_probabilities_nonnegative(self):
        model = make_model()
        probs = model.category_probabilities(0.0)
        assert (probs >= 0).all()

    def test_low_theta_favors_lowest_category(self):
        model = make_model()
        probs = model.category_probabilities(-10.0)
        assert np.argmax(probs) == 0

    def test_high_theta_favors_highest_category(self):
        model = make_model()
        probs = model.category_probabilities(10.0)
        assert np.argmax(probs) == model.item.n_categories - 1


class TestLoglik:
    def test_loglik_matches_category_probability(self):
        model = make_model()
        probs = model.category_probabilities(0.5)
        for k in range(model.item.n_categories):
            assert math.isclose(model.loglik(k, 0.5), math.log(probs[k]), rel_tol=1e-9)

    def test_loglik_rejects_out_of_range_category(self):
        model = make_model()
        with pytest.raises(ValueError):
            model.loglik(-1, 0.0)
        with pytest.raises(ValueError):
            model.loglik(model.item.n_categories, 0.0)

    def test_loglik_rejects_non_integer_value(self):
        model = make_model()
        with pytest.raises(ValueError):
            model.loglik(2.5, 0.0)

    def test_loglik_boundary_categories(self):
        model = make_model()
        assert math.isfinite(model.loglik(0, 0.0))
        assert math.isfinite(model.loglik(model.item.n_categories - 1, 0.0))


class TestInfo:
    def test_info_nonnegative(self):
        model = make_model()
        for theta in (-4, -1, 0, 1, 4):
            assert model.info(theta) >= 0

    def test_info_finite_at_extreme_theta(self):
        model = make_model()
        assert math.isfinite(model.info(-10.0))
        assert math.isfinite(model.info(10.0))

    def test_higher_discrimination_increases_info_near_center(self):
        low = make_model(a=0.5)
        high = make_model(a=2.0)
        assert high.info(2.5) > low.info(2.5)
