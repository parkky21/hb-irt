import math

import pytest

from hb_irt.scoring import aggregate_levels, build_subskill_score, rescale_0_100
from hb_irt.types import Posterior


class TestRescale0100:
    def test_zero_theta_gives_score_50(self):
        score, margin, lower, upper = rescale_0_100(Posterior(mu=0.0, variance=0.0234))
        assert math.isclose(score, 50.0)

    def test_spec_example_56_plus_minus_3(self):
        # spec §5.1: theta_eap=0.62 -> score ~56, margin ~3 (Table 7 example)
        posterior = Posterior(mu=0.62, variance=0.0234)
        score, margin, lower, upper = rescale_0_100(posterior)
        assert math.isclose(score, 56.2, abs_tol=0.05)
        assert math.isclose(margin, 3.0, abs_tol=0.1)
        assert math.isclose(lower, score - margin)
        assert math.isclose(upper, score + margin)

    def test_ci_bounds_are_symmetric(self):
        posterior = Posterior(mu=-0.3, variance=0.05)
        score, margin, lower, upper = rescale_0_100(posterior)
        assert math.isclose((lower + upper) / 2, score, abs_tol=1e-9)


class TestAggregateLevels:
    def test_aggregation_is_weighted_average(self):
        thetas = {"L1": 0.0, "L2": 1.0}
        variances = {"L1": 0.1, "L2": 0.1}
        posterior = aggregate_levels(thetas, variances)
        # equal variances -> simple average
        assert math.isclose(posterior.mu, 0.5, abs_tol=1e-9)

    def test_lower_variance_level_dominates(self):
        thetas = {"L1": 0.0, "L2": 2.0}
        variances = {"L1": 0.01, "L2": 5.0}
        posterior = aggregate_levels(thetas, variances)
        assert posterior.mu < 1.0  # closer to the precise L1 estimate

    def test_variance_decreases_as_levels_added(self):
        one = aggregate_levels({"L1": 0.0}, {"L1": 0.2})
        two = aggregate_levels({"L1": 0.0, "L2": 0.0}, {"L1": 0.2, "L2": 0.2})
        assert two.variance < one.variance

    def test_mismatched_keys_raises(self):
        with pytest.raises(ValueError):
            aggregate_levels({"L1": 0.0}, {"L2": 0.1})

    def test_empty_levels_raises(self):
        with pytest.raises(ValueError):
            aggregate_levels({}, {})

    def test_rejects_nonpositive_tau_squared(self):
        with pytest.raises(ValueError):
            aggregate_levels({"L1": 0.0}, {"L1": 0.1}, tau_squared=0.0)


class TestBuildSubskillScore:
    def test_full_data_model_fields(self):
        posterior = Posterior(mu=0.62, variance=0.0234)
        s = build_subskill_score(
            subskill_id="sk_python_debugging",
            posterior=posterior,
            items_administered=42,
            modules_completed=4,
            level_thetas={"L1": 0.5, "L2": 0.7},
        )
        assert s.subskill_id == "sk_python_debugging"
        assert math.isclose(s.theta_eap, 0.62)
        assert math.isclose(s.theta_variance, 0.0234)
        assert math.isclose(s.theta_sem, math.sqrt(0.0234))
        assert math.isclose(s.score_0_100, 56.2, abs_tol=0.05)
        assert s.items_administered == 42
        assert s.modules_completed == 4
        assert s.level_thetas == {"L1": 0.5, "L2": 0.7}

    def test_defaults_empty_level_thetas(self):
        s = build_subskill_score(
            subskill_id="sk1",
            posterior=Posterior(mu=0.0, variance=0.1),
            items_administered=5,
            modules_completed=1,
        )
        assert s.level_thetas == {}
