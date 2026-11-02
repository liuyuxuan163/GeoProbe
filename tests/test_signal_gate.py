"""测试：置换检验闸门（``signal_ok``）必须两边同一采样规模。

契约
----
``signal_ok`` 比较的是：

    隐层（R 次重复的中位误差均值） < 0.6 × 置换基线（R 次重复的均值）

**不能用** ``control.median_km``（最后一次单次划分）—— 那是苹果比橘子，
而且"哪次划分排最后"是随机的，判定会跟着抖。

真实案例（2026-10-30，Qwen2.5-0.5B @ layer 6，319 词，repeats=8）：
    单次划分 4,055 km → 4,055 / 6,154 = 0.66 → 判「失败」
    8 次均值 3,318 km → 3,318 / 6,154 = 0.54 → 判「通过」
同一份数据、同一批基线，只因左边取了哪个数就翻转结论。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.experiment import ExperimentResult, RepeatStats
from geoprobe.metrics import ErrorStats


def _make_result(hidden_single_km: float, repeat_medians, permuted_km):
    result = ExperimentResult()
    result.control = ErrorStats(n=64, median_km=hidden_single_km)
    result.control_repeats = RepeatStats(medians=list(repeat_medians))
    if permuted_km is not None:
        result.baselines = {"permuted": {"median_km": permuted_km, "median_std_km": 500.0}}
    return result


class TestSignalGateUsesRepeatMean:
    def test_gate_passes_even_if_last_split_was_unlucky(self):
        """单次划分 4,055、8 次均值 3,318、置换 6,154 → 旧代码判 False，新代码判 True。"""
        result = _make_result(
            hidden_single_km=4055.0,
            repeat_medians=[3318.0] * 8,
            permuted_km=6154.0,
        )
        assert result.control.median_km > 0.6 * 6154.0, "前提：单次划分本应判失败"
        assert result.signal_ok is True, "应当用 8 次均值判定（0.54 < 0.6）"

    def test_gate_fails_when_repeat_mean_is_too_close(self):
        """真没信号时仍然必须判失败（不能因为改口径就把闸门放水）。"""
        result = _make_result(
            hidden_single_km=3000.0,
            repeat_medians=[5900.0] * 8,
            permuted_km=6154.0,
        )
        assert result.signal_ok is False

    def test_missing_permuted_baseline_is_false(self):
        """没有置换基线（layer-sweep 的非末层）→ 一律 False，不能默认放行。"""
        result = _make_result(1000.0, [1000.0] * 8, permuted_km=None)
        assert result.signal_ok is False

    def test_falls_back_to_single_split_without_repeats(self):
        """手工构造、没有重复记录时退回单次划分，而不是直接崩/恒 False。"""
        result = _make_result(1000.0, [], permuted_km=6000.0)
        assert result.signal_ok is True
        result_bad = _make_result(5500.0, [], permuted_km=6000.0)
        assert result_bad.signal_ok is False

    def test_nan_never_passes(self):
        result = _make_result(float("nan"), [float("nan")] * 8, permuted_km=6000.0)
        assert result.signal_ok is False
