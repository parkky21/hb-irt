import pytest

from hb_irt.models.base import ItemModel


def test_item_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ItemModel()


def test_item_model_subclass_must_implement_abstract_methods():
    class Incomplete(ItemModel):
        pass

    with pytest.raises(TypeError):
        Incomplete()
