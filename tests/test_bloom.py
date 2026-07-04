import math

import pytest

from hb_irt.bloom import (
    BLOOM_DIFFICULTY_ANCHORS,
    BLOOM_LEVEL_NAMES,
    difficulty_anchor,
    shrink_difficulty,
)


class TestDifficultyAnchor:
    @pytest.mark.parametrize("level", list(BLOOM_DIFFICULTY_ANCHORS))
    def test_known_levels(self, level):
        assert difficulty_anchor(level) == BLOOM_DIFFICULTY_ANCHORS[level]

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            difficulty_anchor("L7")

    def test_all_levels_have_names(self):
        assert set(BLOOM_LEVEL_NAMES) == set(BLOOM_DIFFICULTY_ANCHORS)

    def test_anchors_increase_with_level(self):
        levels = ["L1", "L2", "L3", "L4", "L5", "L6"]
        values = [BLOOM_DIFFICULTY_ANCHORS[l] for l in levels]
        assert values == sorted(values)


class TestShrinkDifficulty:
    def test_shrinks_toward_anchor(self):
        # raw estimate far from anchor, high raw variance (imprecise) -> pulled toward anchor
        anchor = difficulty_anchor("L1")
        raw = anchor + 5.0
        shrunk = shrink_difficulty(raw_difficulty=raw, raw_variance=10.0, level="L1", sigma_b=0.1)
        assert abs(shrunk - anchor) < abs(raw - anchor)

    def test_precise_raw_estimate_dominates(self):
        anchor = difficulty_anchor("L3")
        raw = anchor + 2.0
        shrunk = shrink_difficulty(
            raw_difficulty=raw, raw_variance=1e-6, level="L3", sigma_b=1.0
        )
        assert math.isclose(shrunk, raw, abs_tol=1e-3)

    def test_equal_precision_gives_midpoint(self):
        anchor = difficulty_anchor("L2")
        raw = anchor + 1.0
        shrunk = shrink_difficulty(raw_difficulty=raw, raw_variance=1.0, level="L2", sigma_b=1.0)
        assert math.isclose(shrunk, anchor + 0.5, abs_tol=1e-9)

    def test_rejects_nonpositive_raw_variance(self):
        with pytest.raises(ValueError):
            shrink_difficulty(raw_difficulty=0.0, raw_variance=0.0, level="L1", sigma_b=1.0)

    def test_rejects_nonpositive_sigma_b(self):
        with pytest.raises(ValueError):
            shrink_difficulty(raw_difficulty=0.0, raw_variance=1.0, level="L1", sigma_b=0.0)

    def test_rejects_unknown_level(self):
        with pytest.raises(ValueError):
            shrink_difficulty(raw_difficulty=0.0, raw_variance=1.0, level="L9", sigma_b=1.0)
