import math

import pytest

from hb_irt.bayes.estimation import eap_estimate
from hb_irt.models.crm import CRMItem, CRMModel


def make_model(a=1.0, b=0.0, max_score=100.0, boundary_clip=0.5):
    return CRMModel(
        CRMItem(item_id="qa2", a=a, b=b, max_score=max_score, boundary_clip=boundary_clip)
    )


class TestCRMItem:
    def test_rejects_nonpositive_discrimination(self):
        with pytest.raises(ValueError):
            CRMItem(item_id="qa2", a=0.0, b=0.0)

    def test_rejects_nonpositive_max_score(self):
        with pytest.raises(ValueError):
            CRMItem(item_id="qa2", a=1.0, b=0.0, max_score=0.0)

    def test_rejects_nonpositive_boundary_clip(self):
        with pytest.raises(ValueError):
            CRMItem(item_id="qa2", a=1.0, b=0.0, boundary_clip=0.0)

    def test_rejects_boundary_clip_at_or_above_half_max_score(self):
        with pytest.raises(ValueError):
            CRMItem(item_id="qa2", a=1.0, b=0.0, max_score=100.0, boundary_clip=50.0)


class TestTransform:
    def test_transform_midpoint_is_zero(self):
        model = make_model(max_score=100.0)
        assert math.isclose(model.transform(50.0), 0.0, abs_tol=1e-9)

    def test_transform_increasing(self):
        model = make_model()
        assert model.transform(20.0) < model.transform(50.0) < model.transform(80.0)

    def test_transform_rejects_out_of_range(self):
        model = make_model(max_score=100.0)
        with pytest.raises(ValueError):
            model.transform(-1.0)
        with pytest.raises(ValueError):
            model.transform(101.0)

    def test_transform_handles_exact_boundary_values(self):
        model = make_model(max_score=100.0)
        assert math.isfinite(model.transform(0.0))
        assert math.isfinite(model.transform(100.0))

    def test_transform_clips_to_boundary_clip_amount(self):
        model = make_model(max_score=100.0, boundary_clip=0.5)
        assert math.isclose(model.transform(100.0), math.log(99.5 / 0.5))
        assert math.isclose(model.transform(0.0), math.log(0.5 / 99.5))

    def test_transform_respects_custom_boundary_clip(self):
        loose = make_model(max_score=100.0, boundary_clip=5.0)
        tight = make_model(max_score=100.0, boundary_clip=0.1)
        # a looser clip pulls the boundary score further from the extreme
        assert abs(loose.transform(100.0)) < abs(tight.transform(100.0))


class TestLoglik:
    def test_loglik_finite_across_score_range(self):
        model = make_model()
        for value in (0.0, 1.0, 25.0, 50.0, 75.0, 99.0, 100.0):
            assert math.isfinite(model.loglik(value, theta=0.0))

    def test_loglik_peaks_when_theta_matches_transformed_score(self):
        model = make_model(a=1.0, b=0.0)
        z = model.transform(70.0)
        ll_matched = model.loglik(70.0, theta=z)
        ll_far = model.loglik(70.0, theta=z - 5.0)
        assert ll_matched > ll_far

    def test_loglik_rejects_out_of_range_value(self):
        model = make_model()
        with pytest.raises(ValueError):
            model.loglik(-5.0, theta=0.0)
        with pytest.raises(ValueError):
            model.loglik(105.0, theta=0.0)

    def test_loglik_respects_difficulty_shift(self):
        easy = make_model(b=-1.0)
        hard = make_model(b=1.0)
        # A high raw score is more likely under the easier item at a fixed theta
        assert easy.loglik(90.0, theta=0.0) > hard.loglik(90.0, theta=0.0)


class TestInfo:
    def test_info_is_constant_a_squared(self):
        model = make_model(a=1.7)
        expected = 1.7**2
        for theta in (-4, -1, 0, 1, 4):
            assert math.isclose(model.info(theta), expected)

    def test_info_scales_with_discrimination(self):
        low = make_model(a=0.5)
        high = make_model(a=2.0)
        assert high.info(0.0) > low.info(0.0)


class TestBoundaryScoreRegression:
    """A perfect (or zero) raw score must not produce a degenerate posterior
    (regression: previously reported score ~128.5 +/- 0.0 for a 100/100)."""

    def test_perfect_score_yields_plausible_finite_posterior(self):
        model = make_model(a=1.2, b=0.3)
        posterior = eap_estimate([(model, 100.0)], prior_mu=0.0, prior_sigma=1.0)
        assert math.isfinite(posterior.mu)
        assert -6.0 < posterior.mu < 6.0
        assert posterior.variance > 0.0

    def test_zero_score_yields_plausible_finite_posterior(self):
        model = make_model(a=1.2, b=0.3)
        posterior = eap_estimate([(model, 0.0)], prior_mu=0.0, prior_sigma=1.0)
        assert math.isfinite(posterior.mu)
        assert -6.0 < posterior.mu < 6.0
        assert posterior.variance > 0.0
