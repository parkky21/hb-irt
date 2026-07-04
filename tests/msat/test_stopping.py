from hb_irt.msat.stopping import StoppingConfig, evaluate_stopping
from hb_irt.types import Posterior


class TestMinimumItemsGate:
    def test_does_not_stop_before_minimum_items(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=0.01),  # would pass precision threshold
            previous_posterior=None,
            n_modules=1,
            n_items=5,
            config=StoppingConfig(min_items=20),
        )
        assert decision.should_stop is False
        assert decision.reasons == ()


class TestPrecisionThreshold:
    def test_stops_when_sem_below_threshold(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=0.3**2 * 0.5),  # sem below 0.3
            previous_posterior=None,
            n_modules=2,
            n_items=25,
            config=StoppingConfig(),
        )
        assert decision.should_stop is True
        assert "precision_threshold" in decision.reasons

    def test_does_not_stop_when_sem_above_threshold(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=2.0),
            previous_posterior=None,
            n_modules=2,
            n_items=25,
            config=StoppingConfig(),
        )
        assert "precision_threshold" not in decision.reasons


class TestMaximumModules:
    def test_stops_at_max_modules(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=2.0),
            previous_posterior=None,
            n_modules=8,
            n_items=25,
            config=StoppingConfig(max_modules=8),
        )
        assert decision.should_stop is True
        assert "maximum_modules" in decision.reasons

    def test_does_not_stop_below_max_modules(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=2.0),
            previous_posterior=None,
            n_modules=3,
            n_items=25,
            config=StoppingConfig(max_modules=8),
        )
        assert "maximum_modules" not in decision.reasons


class TestInformationSaturation:
    def test_stops_when_variance_gain_marginal(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=0.5000),
            previous_posterior=Posterior(mu=0.0, variance=0.5005),
            n_modules=3,
            n_items=25,
            config=StoppingConfig(delta_saturation=0.01),
        )
        assert decision.should_stop is True
        assert "information_saturation" in decision.reasons

    def test_does_not_stop_with_large_variance_reduction(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=0.5),
            previous_posterior=Posterior(mu=0.0, variance=2.0),
            n_modules=3,
            n_items=25,
            config=StoppingConfig(delta_saturation=0.01),
        )
        assert "information_saturation" not in decision.reasons

    def test_skipped_when_no_previous_posterior(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=2.0),
            previous_posterior=None,
            n_modules=1,
            n_items=25,
            config=StoppingConfig(),
        )
        assert "information_saturation" not in decision.reasons


class TestCombinedRules:
    def test_multiple_reasons_can_trigger_together(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=0.01),
            previous_posterior=Posterior(mu=0.0, variance=0.0105),
            n_modules=8,
            n_items=25,
            config=StoppingConfig(),
        )
        assert decision.should_stop is True
        assert set(decision.reasons) == {
            "precision_threshold",
            "maximum_modules",
            "information_saturation",
        }

    def test_no_rule_triggers_continues_testing(self):
        decision = evaluate_stopping(
            posterior=Posterior(mu=0.0, variance=2.0),
            previous_posterior=Posterior(mu=0.0, variance=3.0),
            n_modules=2,
            n_items=25,
            config=StoppingConfig(),
        )
        assert decision.should_stop is False
        assert decision.reasons == ()
