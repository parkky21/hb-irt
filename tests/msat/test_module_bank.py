import pytest

from hb_irt.msat.module_bank import ModuleBank
from hb_irt.types import Item, TestModule


def make_module(module_id, module_type="medium", n_items=5, n_exposures=0):
    items = tuple(Item(item_id=f"{module_id}_q{i}", a=1.0, b=0.0, c=0.2) for i in range(n_items))
    return TestModule(module_id=module_id, items=items, module_type=module_type, n_exposures=n_exposures)


class TestModuleBankConstruction:
    def test_valid_bank(self):
        bank = ModuleBank(modules=(make_module("m1"), make_module("m2")))
        assert len(bank.modules) == 2

    def test_rejects_unknown_module_type(self):
        with pytest.raises(ValueError):
            ModuleBank(modules=(make_module("m1", module_type="impossible"),))

    def test_rejects_duplicate_module_ids(self):
        with pytest.raises(ValueError):
            ModuleBank(modules=(make_module("m1"), make_module("m1")))


class TestByType:
    def test_filters_by_type(self):
        bank = ModuleBank(
            modules=(
                make_module("e1", module_type="easy"),
                make_module("h1", module_type="hard"),
                make_module("e2", module_type="easy"),
            )
        )
        easy = bank.by_type("easy")
        assert {m.module_id for m in easy} == {"e1", "e2"}

    def test_rejects_unknown_type_query(self):
        bank = ModuleBank(modules=(make_module("m1"),))
        with pytest.raises(ValueError):
            bank.by_type("nonsense")


class TestAvailable:
    def test_excludes_administered_modules(self):
        bank = ModuleBank(modules=(make_module("m1"), make_module("m2"), make_module("m3")))
        available = bank.available(["m1"])
        assert {m.module_id for m in available} == {"m2", "m3"}

    def test_all_available_when_none_administered(self):
        bank = ModuleBank(modules=(make_module("m1"), make_module("m2")))
        available = bank.available([])
        assert len(available) == 2


class TestGet:
    def test_returns_module_by_id(self):
        bank = ModuleBank(modules=(make_module("m1"), make_module("m2")))
        assert bank.get("m2").module_id == "m2"

    def test_raises_for_unknown_id(self):
        bank = ModuleBank(modules=(make_module("m1"),))
        with pytest.raises(KeyError):
            bank.get("missing")
