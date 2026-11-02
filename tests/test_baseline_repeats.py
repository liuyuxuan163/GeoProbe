"""测试：基线的重复划分次数必须跟随主实验的 `--repeats`。

契约（改坏会在这里报错）
------------------------
1. ``_run_baselines(..., n_repeats=N)`` → 每条基线都记录 ``n_repeats == N``；
2. 显式配置 ``baselines.n_repeats = M`` → 覆盖调用方，改用 M；
3. ``run_experiment(n_repeats=N)`` → ``baselines`` 里**所有**行（含 hidden、mlp）
   的 ``n_repeats`` 都等于 N —— 即全表同一采样规模、同一组划分种子。

历史背景
--------
``_run_baselines`` 曾把次数写死成 ``config.get("baselines.n_repeats", 5)``，
而主实验用 ``--repeats 8``。后果不是"算错"，而是同一张基线表里 hidden 那行的
±std 来自 8 次划分、其余 4 行来自 5 次；且种子 42..46 与 42..49 不是同一组，
比较不是配对的。见 ``docs/实验流程图解.md`` 第 6 节第 4 条。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.config import load_config
from geoprobe.experiment import _run_baselines, run_experiment
from geoprobe.wordlists import load_wordlists


@pytest.fixture()
def config():
    config = load_config()
    config.set("model.mock", True)
    config.set("evaluation.bootstrap_samples", 100)
    return config


@pytest.fixture()
def lists():
    return load_wordlists()


def _fake_hidden(lists, dim: int = 24, seed: int = 0) -> np.ndarray:
    """合成隐层矩阵：维度很小，只为了快速跑通基线（不需要真实模型）。"""
    rng = np.random.default_rng(seed)
    n = len(lists["control"])
    return rng.normal(0.0, 1.0, size=(n, dim))


class TestBaselineRepeatsFollowCaller:
    def test_all_baselines_use_caller_repeats(self, config, lists):
        X = _fake_hidden(lists)
        Y = lists["control"].coords()
        out = _run_baselines(config, lists, X, Y, 0.2, None, n_repeats=3)
        assert out, "基线结果不应为空"
        for name, payload in out.items():
            assert payload["n_repeats"] == 3, f"{name} 没有跟随 n_repeats=3"
            assert np.isfinite(payload["median_km"]), f"{name} 的中位误差不是有限值"

    def test_explicit_config_overrides_caller(self, config, lists):
        # 关掉最慢的一条，本条只验证优先级
        config.set("baselines.char_ngram", False)
        config.set("baselines.n_repeats", 2)
        X = _fake_hidden(lists)
        Y = lists["control"].coords()
        out = _run_baselines(config, lists, X, Y, 0.2, None, n_repeats=7)
        assert out, "基线结果不应为空"
        for name, payload in out.items():
            assert payload["n_repeats"] == 2, f"{name} 没有被显式配置覆盖"


class TestRunExperimentRecordsRepeats:
    def test_every_baseline_row_shares_one_sampling_scale(self, config, lists):
        result = run_experiment(config, lists=lists, n_repeats=3)
        assert result.baselines, "应包含基线表"
        for name, payload in result.baselines.items():
            assert payload.get("n_repeats") == 3, f"{name} 的采样规模与主实验不一致"

    def test_hidden_row_equals_main_result(self, config, lists):
        """``baselines["hidden"]`` 必须就是主结果本身，否则同表对比没意义。"""
        result = run_experiment(config, lists=lists, n_repeats=2)
        assert result.baselines["hidden"]["median_km"] == pytest.approx(
            result.control_repeats.median_mean
        )
        assert result.baselines["hidden"]["median_std_km"] == pytest.approx(
            result.control_repeats.median_std
        )
