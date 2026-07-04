import math

from hb_irt.bayes.sequential import sequential_update, sequential_update_all
from hb_irt.models.threepl import ThreePLModel
from hb_irt.types import Item, Posterior


def three_pl(a=1.5, b=0.0, c=0.2):
    return ThreePLModel(Item(item_id="q", a=a, b=b, c=c))


class TestSequentialUpdate:
    def test_updates_prior_with_new_module(self):
        prior = Posterior(mu=0.0, variance=1.0)
        model = three_pl()
        posterior = sequential_update(prior, [(model, 1)] * 5)
        assert posterior.mu > prior.mu

    def test_variance_does_not_increase(self):
        prior = Posterior(mu=0.0, variance=1.0)
        model = three_pl()
        posterior = sequential_update(prior, [(model, 1), (model, 0), (model, 1)])
        assert posterior.variance <= prior.variance

    def test_handles_zero_variance_prior(self):
        prior = Posterior(mu=0.5, variance=0.0)
        model = three_pl()
        posterior = sequential_update(prior, [(model, 1)])
        assert math.isfinite(posterior.mu)
        assert math.isfinite(posterior.variance)


class TestSequentialUpdateAll:
    def test_returns_one_posterior_per_module(self):
        prior = Posterior(mu=0.0, variance=1.0)
        model = three_pl()
        modules = [[(model, 1)] * 5, [(model, 1)] * 5, [(model, 0)] * 3]
        history = sequential_update_all(prior, modules)
        assert len(history) == 3
        assert all(isinstance(p, Posterior) for p in history)

    def test_variance_monotonically_non_increasing_across_modules(self):
        prior = Posterior(mu=0.0, variance=1.0)
        model = three_pl()
        modules = [[(model, 1), (model, 0)] for _ in range(5)]
        history = sequential_update_all(prior, modules)
        variances = [prior.variance] + [p.variance for p in history]
        assert all(v2 <= v1 + 1e-9 for v1, v2 in zip(variances, variances[1:]))

    def test_empty_modules_list_returns_empty_history(self):
        prior = Posterior(mu=0.1, variance=0.5)
        assert sequential_update_all(prior, []) == []
