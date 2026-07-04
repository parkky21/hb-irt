import math

import pytest

from hb_irt.types import Item, Posterior, Response, SubskillScore, TestModule


class TestItem:
    def test_valid_item(self):
        item = Item(item_id="q1", a=1.2, b=0.0, c=0.2)
        assert item.item_id == "q1"
        assert item.a == 1.2

    def test_default_guessing_is_zero(self):
        item = Item(item_id="q1", a=1.0, b=0.0)
        assert item.c == 0.0

    def test_rejects_nonpositive_discrimination(self):
        with pytest.raises(ValueError):
            Item(item_id="q1", a=0.0, b=0.0, c=0.2)
        with pytest.raises(ValueError):
            Item(item_id="q1", a=-1.0, b=0.0, c=0.2)

    def test_rejects_invalid_guessing(self):
        with pytest.raises(ValueError):
            Item(item_id="q1", a=1.0, b=0.0, c=-0.1)
        with pytest.raises(ValueError):
            Item(item_id="q1", a=1.0, b=0.0, c=1.0)

    def test_is_frozen(self):
        item = Item(item_id="q1", a=1.0, b=0.0, c=0.2)
        with pytest.raises(AttributeError):
            item.a = 2.0


class TestResponse:
    def test_construction(self):
        r = Response(item_id="q1", value=1)
        assert r.item_id == "q1"
        assert r.value == 1


class TestPosterior:
    def test_sem_is_sqrt_variance(self):
        p = Posterior(mu=0.5, variance=0.25)
        assert math.isclose(p.sem, 0.5)

    def test_rejects_negative_variance(self):
        with pytest.raises(ValueError):
            Posterior(mu=0.0, variance=-1.0)

    def test_credible_interval_95(self):
        p = Posterior(mu=0.0, variance=1.0)
        lower, upper = p.credible_interval(0.95)
        assert math.isclose(lower, -1.959963, abs_tol=1e-4)
        assert math.isclose(upper, 1.959963, abs_tol=1e-4)

    def test_credible_interval_symmetric_about_mu(self):
        p = Posterior(mu=1.5, variance=0.04)
        lower, upper = p.credible_interval(0.9)
        assert math.isclose((lower + upper) / 2, 1.5, abs_tol=1e-9)

    def test_credible_interval_rejects_invalid_level(self):
        p = Posterior(mu=0.0, variance=1.0)
        with pytest.raises(ValueError):
            p.credible_interval(0.0)
        with pytest.raises(ValueError):
            p.credible_interval(1.0)


class TestTestModule:
    def _items(self, n):
        return tuple(Item(item_id=f"q{i}", a=1.0, b=0.0, c=0.2) for i in range(n))

    def test_valid_module(self):
        module = TestModule(module_id="m1", items=self._items(10))
        assert module.size == 10
        assert module.module_type == "medium"

    def test_rejects_too_few_items(self):
        with pytest.raises(ValueError):
            TestModule(module_id="m1", items=self._items(4))

    def test_rejects_too_many_items(self):
        with pytest.raises(ValueError):
            TestModule(module_id="m1", items=self._items(21))

    def test_boundary_sizes_allowed(self):
        TestModule(module_id="m1", items=self._items(5))
        TestModule(module_id="m1", items=self._items(20))


class TestSubskillScore:
    def test_construction_defaults(self):
        s = SubskillScore(
            subskill_id="sk_python_debugging",
            theta_eap=0.62,
            theta_variance=0.0234,
            theta_sem=0.153,
            score_0_100=56.2,
            margin_error_95=3.0,
            ci_lower_95=53.2,
            ci_upper_95=59.2,
            items_administered=42,
            modules_completed=4,
        )
        assert s.subskill_id == "sk_python_debugging"
        assert s.level_thetas == {}

    def test_level_thetas_populated(self):
        s = SubskillScore(
            subskill_id="sk1",
            theta_eap=0.0,
            theta_variance=0.1,
            theta_sem=math.sqrt(0.1),
            score_0_100=50.0,
            margin_error_95=6.0,
            ci_lower_95=44.0,
            ci_upper_95=56.0,
            items_administered=10,
            modules_completed=1,
            level_thetas={"L1": 0.1, "L2": 0.2},
        )
        assert s.level_thetas["L2"] == 0.2
