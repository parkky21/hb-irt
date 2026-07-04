import math

import pytest

from hb_irt.models.threepl import ThreePLModel
from hb_irt.types import Item


def make_model(a=1.2, b=0.0, c=0.2):
    return ThreePLModel(Item(item_id="q1", a=a, b=b, c=c))


class TestProbability:
    def test_probability_at_difficulty_equals_midpoint(self):
        # Table 5: at theta=b, P(correct) = (1+c)/2
        model = make_model(a=1.5, b=0.3, c=0.2)
        p = model.probability(theta=0.3)
        assert math.isclose(p, (1 + 0.2) / 2, abs_tol=1e-9)

    def test_probability_approaches_guessing_at_low_theta(self):
        model = make_model(a=1.5, b=0.0, c=0.2)
        p = model.probability(theta=-10.0)
        assert math.isclose(p, 0.2, abs_tol=1e-6)

    def test_probability_approaches_one_at_high_theta(self):
        model = make_model(a=1.5, b=0.0, c=0.2)
        p = model.probability(theta=10.0)
        assert math.isclose(p, 1.0, abs_tol=1e-6)

    def test_zero_guessing_item(self):
        model = make_model(a=1.0, b=0.0, c=0.0)
        p = model.probability(theta=0.0)
        assert math.isclose(p, 0.5, abs_tol=1e-9)


class TestLoglik:
    def test_loglik_correct_response(self):
        model = make_model()
        p = model.probability(theta=1.0)
        assert math.isclose(model.loglik(1, 1.0), math.log(p))

    def test_loglik_incorrect_response(self):
        model = make_model()
        p = model.probability(theta=1.0)
        assert math.isclose(model.loglik(0, 1.0), math.log(1 - p))

    def test_loglik_rejects_invalid_value(self):
        model = make_model()
        with pytest.raises(ValueError):
            model.loglik(2, 0.0)

    def test_loglik_finite_at_extreme_theta(self):
        model = make_model()
        assert math.isfinite(model.loglik(1, -8.0))
        assert math.isfinite(model.loglik(0, 8.0))


class TestInfo:
    def test_info_nonnegative(self):
        model = make_model()
        for theta in (-4, -1, 0, 1, 4):
            assert model.info(theta) >= 0

    def test_info_zero_guessing_matches_2pl_formula(self):
        # With c=0: I(theta) = a^2 * P * Q
        model = make_model(a=1.3, b=0.0, c=0.0)
        theta = 0.5
        p = model.probability(theta)
        expected = 1.3**2 * p * (1 - p)
        assert math.isclose(model.info(theta), expected, rel_tol=1e-6)

    def test_info_peaks_near_difficulty(self):
        model = make_model(a=1.5, b=0.0, c=0.2)
        info_at_b = model.info(0.0)
        info_far = model.info(4.0)
        assert info_at_b > info_far

    def test_info_handles_extreme_guessing(self):
        model = make_model(a=1.0, b=0.0, c=1.0 - 5e-13)
        assert model.info(0.0) == 0.0

    def test_info_finite_at_extreme_theta(self):
        model = make_model()
        assert math.isfinite(model.info(-8.0))
        assert math.isfinite(model.info(8.0))
