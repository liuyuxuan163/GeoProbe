"""测试：人口分档（`scripts/stratified_analysis.py` 的 P0 功能）。

契约
----
1. ``load_population`` 只收「能转成正数」的条目；缺失 / 空串 / 非数字 / ≤0 一律跳过，
   **不填 0**（填 0 会把"没数据"伪装成"人口最少"，污染最低档）。
2. ``summarise_population`` 按人口降序**等词数**切档：Q1 = 人口最多 = 最知名；
   每档大小最多差 1；档成员与输入词的顺序无关。
3. 每档的中位误差在该档**内部**算，缺失（NaN）的误差被跳过但词仍计入 ``n``。
4. ``spearman`` 负值 = 越知名误差越小；并列值按平均秩处理。

测试刻意不依赖具体词表规模（交接文档 §7.9），全部用合成数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stratified_analysis import (  # noqa: E402
    bucket_label,
    category_buckets,
    load_population,
    spearman,
    summarise_population,
)


# --------------------------------------------------------------------------- #
# 类别分桶（319 词上曾直接 KeyError: '地区'）
# --------------------------------------------------------------------------- #


class TestCategoryBucket:
    LABELS = ["城市", "国家", "其他"]

    def test_unknown_category_folds_into_other(self):
        assert bucket_label("地区", self.LABELS) == "其他"
        assert bucket_label("城市", self.LABELS) == "城市"
        assert bucket_label("国家", self.LABELS) == "国家"

    def test_every_word_lands_in_exactly_one_bucket(self):
        words = ["A", "B", "C", "D"]
        categories = {"A": "城市", "B": "国家", "C": "地区", "D": ""}
        buckets = category_buckets(words, categories, self.LABELS)
        flattened = [w for label in self.LABELS for w in buckets[label]]
        assert sorted(flattened) == sorted(words), "不能丢词，也不能重复计数"
        assert buckets["其他"] == ["C", "D"]

    def test_missing_category_defaults_to_other(self):
        buckets = category_buckets(["X"], {}, self.LABELS)
        assert buckets["其他"] == ["X"]


# --------------------------------------------------------------------------- #
# load_population
# --------------------------------------------------------------------------- #


class TestLoadPopulation:
    def test_skips_missing_and_invalid(self, workdir):
        raw = [
            {"word": "Big", "population": 10_000_000},
            {"word": "Small", "population": 20_000},
            {"word": "NoField"},
            {"word": "Empty", "population": ""},
            {"word": "Null", "population": None},
            {"word": "Text", "population": "many"},
            {"word": "Zero", "population": 0},
            {"word": "Negative", "population": -5},
        ]
        path = workdir / "words.json"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

        population = load_population(path)

        assert population == {"Big": 10_000_000.0, "Small": 20_000.0}
        assert "Zero" not in population, "0 不能被当成「人口最少」"
        assert "NoField" not in population


# --------------------------------------------------------------------------- #
# summarise_population
# --------------------------------------------------------------------------- #


def _toy(n_per_bin: int = 3, n_bins: int = 4):
    """构造 n_per_bin × n_bins 个词，人口从高到低排列，误差随知名度下降而上升。"""
    words, population, per_word = [], {}, {}
    total = n_per_bin * n_bins
    for index in range(total):
        word = f"w{index:02d}"
        words.append(word)
        # 人口：从 10,000,000 递减到约 10,000
        population[word] = float(10_000_000 / (10 ** (index * 3 / max(1, total - 1))))
        # 误差：与人口负相关（越不知名越差）
        per_word[word] = 1_000.0 + index * 500
    return words, population, per_word


class TestSummarisePopulation:
    def test_bins_are_balanced(self):
        words, population, per_word = _toy(n_per_bin=3, n_bins=4)
        summary = summarise_population(words, population, per_word, n_bins=4)
        sizes = [q["n"] for q in summary["quantiles"]]
        assert sizes == [3, 3, 3, 3]
        assert summary["n_with_population"] == 12
        assert summary["n_without_population"] == 0

    def test_remainder_goes_to_leading_bins(self):
        words, population, per_word = _toy(n_per_bin=1, n_bins=1)
        # 10 个词切 4 档 → 3,3,2,2
        words = [f"w{i:02d}" for i in range(10)]
        population = {w: float(1000 - i * 10) for i, w in enumerate(words)}
        per_word = {w: 100.0 + i for i, w in enumerate(words)}
        summary = summarise_population(words, population, per_word, n_bins=4)
        assert [q["n"] for q in summary["quantiles"]] == [3, 3, 2, 2]

    def test_q1_is_the_most_populous(self):
        words, population, per_word = _toy(n_per_bin=2, n_bins=4)
        summary = summarise_population(words, population, per_word, n_bins=4)
        q1, q4 = summary["quantiles"][0], summary["quantiles"][-1]
        assert q1["population_min"] > q4["population_max"]
        assert q1["ratio_to_q1"] == pytest.approx(1.0)
        assert q4["ratio_to_q1"] > 1.0, "误差更高的一档，相对 Q1 的倍数应 > 1"

    def test_medians_are_computed_inside_each_bin(self):
        words = [f"w{i}" for i in range(8)]
        population = {f"w{i}": float(1000 - i * 100) for i in range(8)}
        # Q1 的前两个词误差 100/300 → 中位 200；Q2 的 500/700 → 中位 600
        per_word = {"w0": 100.0, "w1": 300.0, "w2": 500.0, "w3": 700.0,
                    "w4": 900.0, "w5": 1100.0, "w6": 1300.0, "w7": 1500.0}
        summary = summarise_population(words, population, per_word, n_bins=4)
        assert summary["quantiles"][0]["median_km"] == pytest.approx(200.0)
        assert summary["quantiles"][1]["median_km"] == pytest.approx(600.0)

    def test_nan_errors_are_skipped_but_word_still_counted(self):
        words = [f"w{i}" for i in range(4)]
        population = {f"w{i}": float(1000 - i) for i in range(4)}
        per_word = {"w0": 100.0, "w1": float("nan"), "w2": 900.0, "w3": 1100.0}
        summary = summarise_population(words, population, per_word, n_bins=2)
        q1 = summary["quantiles"][0]
        assert q1["n"] == 2, "缺误差的词仍属于该档"
        assert q1["median_km"] == pytest.approx(100.0), "中位数只用有限的误差"

    def test_bins_do_not_depend_on_input_order(self):
        words, population, per_word = _toy(n_per_bin=3, n_bins=4)
        a = summarise_population(words, population, per_word, n_bins=4)
        rng = np.random.default_rng(7)
        shuffled = list(words)
        rng.shuffle(shuffled)
        b = summarise_population(shuffled, population, per_word, n_bins=4)
        assert [q["words"] for q in a["quantiles"]] == [q["words"] for q in b["quantiles"]]

    def test_monotone_flag(self):
        words, population, per_word = _toy(n_per_bin=3, n_bins=4)
        assert summarise_population(words, population, per_word, n_bins=4)["monotone"] is True
        # 把最有名的一档误差抬高 → 非单调
        per_word = dict(per_word)
        per_word["w00"] = 9_999.0
        per_word["w01"] = 9_999.0
        per_word["w02"] = 9_999.0
        assert summarise_population(words, population, per_word, n_bins=4)["monotone"] is False

    def test_words_without_population_are_excluded(self):
        words = [f"w{i}" for i in range(6)]
        population = {f"w{i}": float(1000 - i) for i in range(4)}   # 只有前 4 个有人口
        per_word = {w: 100.0 for w in words}
        summary = summarise_population(words, population, per_word, n_bins=2)
        assert summary["n_with_population"] == 4
        assert summary["n_without_population"] == 2
        assert sum(q["n"] for q in summary["quantiles"]) == 4

    def test_empty_population_does_not_crash(self):
        summary = summarise_population(["a", "b"], {}, {"a": 1.0, "b": 2.0}, n_bins=4)
        assert summary["n_with_population"] == 0
        assert summary["spearman"]["n"] == 0
        assert np.isnan(summary["spearman"]["rho_log10_population_vs_error"])


# --------------------------------------------------------------------------- #
# spearman
# --------------------------------------------------------------------------- #


class TestSpearman:
    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        assert spearman(x, y) == pytest.approx(-1.0)

    def test_perfect_positive(self):
        assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_ties_use_average_rank(self):
        # y 里有并列；秩相关仍应有限且方向正确
        rho = spearman([1, 2, 3, 4], [10, 10, 20, 30])
        assert np.isfinite(rho) and rho > 0.5

    def test_ignores_nan_pairs(self):
        rho = spearman([1, 2, 3, float("nan")], [10, 20, 30, 40])
        assert rho == pytest.approx(1.0)

    def test_too_few_points_is_nan(self):
        assert np.isnan(spearman([1, 2], [3, 4]))
