"""测试：作图脚本（``scripts/plot_results.py``）。

测两件事：

1. **数据取对了没**：slug 两套写法能对上、相对深度算对、峰值/末层取值正确；
2. **图能出**：三个 plot 函数在无显示环境下（Agg）都能写出非空 PNG。

刻意不测像素内容 —— 字体、色彩、布局随时会调，测它们只会一直误报。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_results import (  # noqa: E402
    alignment_table,
    ambiguous_points,
    best_and_last,
    canonical_slug,
    layer_rows,
    plot_alignment,
    plot_ambiguous,
    plot_baselines,
    plot_layer_curve,
    plot_population,
)


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #


def _cross_model() -> dict:
    """两个模型、5 层；3B 故意用 compare_encodings 那套 slug（带本地路径前缀）。"""
    return {
        "models": [
            {
                "slug": "qwen2.5-0.5b-instruct",
                "rows": [
                    {"layer": i, "sincos_median_km": 5000 - 400 * i} for i in range(5)
                ],
            },
            {
                "slug": "d-models-qwen2.5-3b-instruct",
                "rows": [
                    {"layer": i, "sincos_median_km": 4000 - 500 * i} for i in range(5)
                ],
            },
        ]
    }


def _results() -> dict:
    return {
        "qwen2.5-0.5b-instruct": {
            "summary": {
                "baselines": {
                    "hidden": {"median_km": 3100.0},
                    "char_ngram": {"median_km": 5200.0},
                    "mean": {"median_km": 6200.0},
                    "random": {"median_km": 6400.0},
                    "permuted": {"median_km": 7800.0},
                }
            }
        },
        "qwen2.5-3b-instruct": {
            "summary": {
                "baselines": {
                    "hidden": {"median_km": 2400.0},
                    "char_ngram": {"median_km": 5200.0},
                    "mean": {"median_km": 6200.0},
                    "random": {"median_km": 6800.0},
                    "permuted": {"median_km": 7800.0},
                }
            }
        },
    }


def _stratified() -> dict:
    def model(base: float) -> dict:
        return {
            "by_population": {
                "quantiles": [
                    {"label": "Q1", "n": 70, "population_median": 5_638_765, "median_km": base},
                    {"label": "Q2", "n": 70, "population_median": 823_728, "median_km": base + 100},
                    {"label": "Q3", "n": 70, "population_median": 55_868, "median_km": base + 600},
                    {"label": "Q4", "n": 70, "population_median": 21_086, "median_km": base + 1200},
                ]
            }
        }

    return {
        "models": {
            "qwen2.5-0.5b-instruct": model(2300),
            "qwen2.5-3b-instruct": model(2600),
        }
    }


# --------------------------------------------------------------------------- #
# 数据层
# --------------------------------------------------------------------------- #


def _results_with_alt() -> dict:
    """带 ``alt_analysis`` 的最小结果（歧义散点用）。"""
    return {
        "qwen2.5-3b-instruct": {
            "alt_reliability_used": 2500.0,
            "alt_analysis": [
                {
                    "word": "Georgia",
                    "dist_primary_km": 3000.0,
                    "dist_alt_km": 9000.0,
                    "closer_to": "primary",
                    "reliable": True,
                },
                {
                    "word": "Memphis",
                    "dist_primary_km": 9000.0,
                    "dist_alt_km": 3000.0,
                    "closer_to": "alt",
                    "reliable": False,
                },
            ],
        }
    }


class TestSlugAndRows:
    def test_two_slug_conventions_resolve_to_the_same_model(self):
        assert canonical_slug("qwen2.5-3b-instruct") == "qwen2.5-3b-instruct"
        assert canonical_slug("d-models-qwen2.5-3b-instruct") == "qwen2.5-3b-instruct"
        assert canonical_slug("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b-instruct"

    def test_layer_rows_accepts_local_path_slug(self):
        rows = layer_rows(_cross_model(), "qwen2.5-3b-instruct")
        assert len(rows) == 5
        assert rows[0]["km"] == pytest.approx(4000.0)
        assert rows[-1]["layer"] == 4
        # 5 层 → 最后一层是 100% 深度
        assert rows[-1]["depth_pct"] == pytest.approx(100.0)
        assert rows[0]["depth_pct"] == pytest.approx(0.0)

    def test_layer_rows_raises_for_unknown_model(self):
        with pytest.raises(KeyError):
            layer_rows(_cross_model(), "qwen2.5-7b-instruct")

    def test_best_and_last(self):
        info = best_and_last(layer_rows(_cross_model(), "qwen2.5-0.5b-instruct"))
        # 误随层数单调下降 → 最优在最后一层，退化 0
        assert info["best_layer"] == 4
        assert info["last_km"] == pytest.approx(info["best_km"])
        assert info["degradation"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #


class TestFigures:
    def test_layer_curve_writes_png(self, workdir):
        out = plot_layer_curve(_cross_model(), workdir / "layer.png", mean_baseline_km=6200.0)
        assert out.exists() and out.stat().st_size > 2000

    def test_baselines_writes_png(self, workdir):
        out = plot_baselines(_results(), workdir / "baselines.png")
        assert out.exists() and out.stat().st_size > 2000

    def test_population_writes_png(self, workdir):
        out = plot_population(_stratified(), workdir / "population.png")
        assert out.exists() and out.stat().st_size > 2000

    def test_layer_curve_without_stability_or_baseline(self, workdir):
        """平台区间和均值基线都是可选的，缺了也要能出图。"""
        out = plot_layer_curve(_cross_model(), workdir / "layer2.png")
        assert out.exists()

    def test_plateau_range_tolerates_missing_model(self, workdir):
        from scripts.plot_results import plateau_range

        assert plateau_range(None, "qwen2.5-3b-instruct") is None
        assert plateau_range({"models": {}}, "qwen2.5-3b-instruct") is None
        assert plateau_range({"models": {"qwen2.5-3b-instruct": {"summary": {}}}}, "qwen2.5-3b-instruct") is None
        got = plateau_range(
            {"models": {"qwen2.5-3b-instruct": {"summary": {"plateau_run": [32, 33]}}}},
            "qwen2.5-3b-instruct",
        )
        assert got == (32, 33)

    def test_alignment_writes_png(self, workdir):
        out = plot_alignment(_cross_model(), workdir / "alignment.png")
        assert out.exists() and out.stat().st_size > 2000

    def test_ambiguous_writes_png(self, workdir):
        out = plot_ambiguous(_results_with_alt(), workdir / "ambiguous.png")
        assert out.exists() and out.stat().st_size > 2000


# --------------------------------------------------------------------------- #
# 对齐口径 / 歧义散点的数据层
# --------------------------------------------------------------------------- #


class TestAlignmentTable:
    def test_six_alignments_with_consistent_layer_pairing(self):
        table = alignment_table(_cross_model())
        assert len(table) == 6
        names = [item["name"] for item in table]
        assert names[0] == "各取最优层" and names[1] == "同为最后一层"
        # 合成数据里误差随层单调下降 → 两个模型的"最优层"就是"最后一层"，
        # 所以这两种对齐必须给出同一个变化；且 3B 的斜率更陡 → 更准
        assert table[0]["change"] == pytest.approx(table[1]["change"])
        assert table[0]["change"] < 0

    def test_depth_matching_is_deterministic(self):
        """偶数层（5 层）时 50% 深度取中间那一层，且与列表顺序无关。"""
        table = alignment_table(_cross_model())
        depth_rows = {item["name"]: item for item in table}
        # 5 层 → round(0.5*4)=2 → L2
        assert "L2" in depth_rows["匹配 50% 深度"]["detail"]


class TestAmbiguousPoints:
    def test_extracts_fields(self):
        points = ambiguous_points(_results_with_alt()["qwen2.5-3b-instruct"])
        assert len(points) == 2
        assert points[0]["word"] == "Georgia"
        assert points[0]["reliable"] is True
        assert points[1]["reliable"] is False
        assert points[1]["closer_to"] == "alt"


# --------------------------------------------------------------------------- #
# 组合总览（PIL 拼图）
# --------------------------------------------------------------------------- #


class TestOverview:
    def _make_pngs(self, workdir, n=5):
        Image = pytest.importorskip("PIL.Image")
        paths = []
        for i in range(n):
            path = workdir / f"panel{i}.png"
            Image.new("RGB", (620, 340), (250 - i * 8, 250, 255)).save(path)
            paths.append(path)
        return paths

    def test_composes_grid_and_writes_file(self, workdir):
        from PIL import Image

        from scripts.plot_results import compose_overview

        paths = self._make_pngs(workdir)
        out = compose_overview(paths, workdir / "overview.png", cell=(320, 180))
        assert out.exists() and out.stat().st_size > 2000
        with Image.open(out) as im:
            # 2 列 × 3 行（5 张图 + 1 个文字面板），加边距后必然比单格大
            assert im.width > 320 * 2
            assert im.height > 180 * 3

    def test_missing_panel_raises(self, workdir):
        from scripts.plot_results import compose_overview

        with pytest.raises(FileNotFoundError):
            compose_overview([workdir / "nope.png"], workdir / "overview.png")


def test_real_reports_render_if_present(workdir):
    """如果本地有真实 JSON，就用它跑一遍（没跑过分析脚本的机器上自动跳过）。"""
    reports = ROOT / "reports"
    cross = reports / "cross_model_comparison.json"
    if not cross.exists():
        pytest.skip("没有 reports/cross_model_comparison.json")
    data = json.loads(cross.read_text(encoding="utf-8"))
    out = plot_layer_curve(data, workdir / "real_layer.png")
    assert out.exists()
