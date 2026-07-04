import math

from hb_irt.information import standard_error
from hb_irt.information import test_information as calc_test_information
from hb_irt.models.threepl import ThreePLModel
from hb_irt.types import Item


def three_pl(a=1.0, b=0.0, c=0.2):
    return ThreePLModel(Item(item_id="q", a=a, b=b, c=c))


class TestTestInformation:
    def test_additive_across_items(self):
        m1, m2 = three_pl(a=1.0), three_pl(a=1.5)
        theta = 0.0
        total = calc_test_information([m1, m2], theta)
        assert math.isclose(total, m1.info(theta) + m2.info(theta))

    def test_empty_models_is_zero(self):
        assert calc_test_information([], 0.0) == 0.0


class TestStandardError:
    def test_matches_1_over_sqrt_info(self):
        models = [three_pl(a=1.2), three_pl(a=0.8)]
        theta = 0.5
        info = calc_test_information(models, theta)
        assert math.isclose(standard_error(models, theta), 1.0 / math.sqrt(info))

    def test_infinite_when_no_information(self):
        assert standard_error([], 0.0) == math.inf
