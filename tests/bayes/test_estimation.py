import math

import pytest

from hb_irt.bayes.estimation import eap_estimate, map_estimate
from hb_irt.models.crm import CRMItem, CRMModel
from hb_irt.models.grm import GRMItem, GRMModel
from hb_irt.models.threepl import ThreePLModel
from hb_irt.types import Item, Posterior


def three_pl(a=1.2, b=0.0, c=0.2):
    return ThreePLModel(Item(item_id="q", a=a, b=b, c=c))


class TestEAPEstimate:
    def test_no_responses_recovers_prior(self):
        post = eap_estimate([], prior_mu=0.3, prior_sigma=1.2, n_points=31)
        assert math.isclose(post.mu, 0.3, abs_tol=1e-6)
        assert math.isclose(post.variance, 1.2**2, rel_tol=1e-6)

    def test_all_correct_shifts_mean_upward(self):
        model = three_pl(a=1.5, b=0.0, c=0.2)
        responses = [(model, 1)] * 5
        post = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
        assert post.mu > 0.0

    def test_all_incorrect_shifts_mean_downward(self):
        model = three_pl(a=1.5, b=0.0, c=0.2)
        responses = [(model, 0)] * 5
        post = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
        assert post.mu < 0.0

    def test_more_items_reduces_variance(self):
        model = three_pl(a=1.5, b=0.0, c=0.2)
        few = eap_estimate([(model, 1)], prior_mu=0.0, prior_sigma=1.0)
        many = eap_estimate([(model, 1)] * 10, prior_mu=0.0, prior_sigma=1.0)
        assert many.variance < few.variance

    def test_mixed_model_types(self):
        mcq = three_pl()
        grm = GRMModel(GRMItem(item_id="qa1", a=1.0, boundaries=(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7)))
        crm = CRMModel(CRMItem(item_id="qa2", a=1.0, b=0.0))
        responses = [(mcq, 1), (grm, 8), (crm, 80.0)]
        post = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
        assert post.mu > 0.0
        assert post.variance > 0.0

    def test_returns_posterior_instance(self):
        model = three_pl()
        post = eap_estimate([(model, 1)])
        assert isinstance(post, Posterior)


class TestMapEstimate:
    def test_no_responses_recovers_prior_mode(self):
        theta = map_estimate([], prior_mu=0.4, prior_sigma=1.0)
        assert math.isclose(theta, 0.4, abs_tol=1e-3)

    def test_all_correct_shifts_estimate_upward(self):
        model = three_pl(a=1.5, b=0.0, c=0.2)
        theta = map_estimate([(model, 1)] * 5, prior_mu=0.0, prior_sigma=1.0)
        assert theta > 0.0

    def test_close_to_eap_for_moderate_evidence(self):
        model = three_pl(a=1.2, b=0.0, c=0.2)
        responses = [(model, 1), (model, 0), (model, 1)]
        eap = eap_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
        mp = map_estimate(responses, prior_mu=0.0, prior_sigma=1.0)
        assert math.isclose(eap.mu, mp, abs_tol=0.3)

    def test_rejects_nonpositive_prior_sigma(self):
        with pytest.raises(ValueError):
            map_estimate([], prior_mu=0.0, prior_sigma=0.0)
