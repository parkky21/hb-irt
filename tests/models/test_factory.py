import pytest

from hb_irt.models.crm import CRMItem, CRMModel
from hb_irt.models.factory import build_model
from hb_irt.models.grm import GRMItem, GRMModel
from hb_irt.models.threepl import ThreePLModel
from hb_irt.types import Item


class TestBuildModel:
    def test_dispatches_3pl_item(self):
        model = build_model(Item(item_id="q1", a=1.2, b=0.0, c=0.2))
        assert isinstance(model, ThreePLModel)

    def test_dispatches_grm_item(self):
        model = build_model(GRMItem(item_id="q2", a=1.1, boundaries=(-1.0, 0.0, 1.0)))
        assert isinstance(model, GRMModel)

    def test_dispatches_crm_item(self):
        model = build_model(CRMItem(item_id="q3", a=1.0, b=0.0))
        assert isinstance(model, CRMModel)

    def test_rejects_unknown_item_type(self):
        with pytest.raises(TypeError):
            build_model(object())
