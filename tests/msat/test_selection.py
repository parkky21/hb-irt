import math

import pytest

from hb_irt.msat.module_bank import ModuleBank
from hb_irt.msat.selection import (
    expected_information_gain,
    expected_variance_reduction,
    information_at_estimate,
    select_next_module,
    selection_score,
)
from hb_irt.types import Item, Posterior, TestModule


def make_module(module_id, b=0.0, module_type="medium", n_items=5, n_exposures=0):
    items = tuple(
        Item(item_id=f"{module_id}_q{i}", a=1.3, b=b, c=0.2) for i in range(n_items)
    )
    return TestModule(
        module_id=module_id, items=items, module_type=module_type, n_exposures=n_exposures
    )


class TestInformationAtEstimate:
    def test_matches_sum_of_item_info(self):
        module = make_module("m1", b=0.0, n_items=5)
        info = information_at_estimate(module, theta=0.0)
        assert info > 0
        double = make_module("double", b=0.0, n_items=10)
        assert math.isclose(info, 0.5 * information_at_estimate(double, 0.0))

    def test_peaks_near_module_difficulty(self):
        module = make_module("m1", b=1.0)
        assert information_at_estimate(module, 1.0) > information_at_estimate(module, 4.0)


class TestExpectedInformationGain:
    def test_positive_for_reasonable_posterior(self):
        module = make_module("m1", b=0.0)
        eig = expected_information_gain(module, Posterior(mu=0.0, variance=1.0))
        assert eig > 0

    def test_close_to_point_estimate_for_low_variance_posterior(self):
        module = make_module("m1", b=0.0)
        eig = expected_information_gain(module, Posterior(mu=0.0, variance=1e-6))
        point = information_at_estimate(module, 0.0)
        assert math.isclose(eig, point, rel_tol=0.05)


class TestExpectedVarianceReduction:
    def test_positive_reduction_when_module_informative(self):
        module = make_module("m1", b=0.0)
        reduction = expected_variance_reduction(module, Posterior(mu=0.0, variance=1.0))
        assert reduction > 0

    def test_zero_when_prior_variance_is_zero(self):
        module = make_module("m1", b=0.0)
        reduction = expected_variance_reduction(module, Posterior(mu=0.0, variance=0.0))
        assert reduction == 0.0

    def test_more_informative_module_reduces_variance_more(self):
        weak = make_module("weak", b=5.0)  # far from theta=0, low info
        strong = make_module("strong", b=0.0)  # at theta=0, high info
        posterior = Posterior(mu=0.0, variance=1.0)
        assert expected_variance_reduction(strong, posterior) > expected_variance_reduction(
            weak, posterior
        )


class TestSelectionScore:
    def test_score_includes_exploration_bonus(self):
        module = make_module("m1", b=0.0, n_exposures=0)
        posterior = Posterior(mu=0.0, variance=1.0)
        score = selection_score(module, posterior)
        info = information_at_estimate(module, 0.0)
        assert score > info

    def test_high_exposure_reduces_bonus(self):
        fresh = make_module("fresh", b=0.0, n_exposures=0)
        stale = make_module("stale", b=0.0, n_exposures=1000)
        posterior = Posterior(mu=0.0, variance=1.0)
        assert selection_score(fresh, posterior) > selection_score(stale, posterior)


class TestSelectNextModule:
    def test_selects_most_informative_module(self):
        bank = ModuleBank(
            modules=(
                make_module("far", b=5.0),
                make_module("near", b=0.0),
                make_module("mid", b=2.0),
            )
        )
        chosen = select_next_module(bank, Posterior(mu=0.0, variance=1.0), administered_ids=[])
        assert chosen.module_id == "near"

    def test_excludes_administered_modules(self):
        bank = ModuleBank(modules=(make_module("near", b=0.0), make_module("mid", b=2.0)))
        chosen = select_next_module(
            bank, Posterior(mu=0.0, variance=1.0), administered_ids=["near"]
        )
        assert chosen.module_id == "mid"

    def test_raises_when_no_modules_remain(self):
        bank = ModuleBank(modules=(make_module("m1"),))
        with pytest.raises(ValueError):
            select_next_module(bank, Posterior(mu=0.0, variance=1.0), administered_ids=["m1"])
