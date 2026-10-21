"""测试：端到端流程（Mock 模式）与报告生成。

**这一组测试是项目的回归防线。**
因为 Mock 模式不依赖模型，所以可以在 CI 里跑完整 pipeline，
确保"改了指标或探针逻辑之后，整个流程仍然自洽"。

关键断言有两个方向：
* `signal=True`  → 探针必须明显优于基线（流程正确）
* `signal=False` → 探针必须无法优于基线（基线有效，没有假信号）
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoprobe.config import load_config
from geoprobe.experiment import alt_preference_analysis, run_experiment, save_result
from geoprobe.extract import extract_dataset
from geoprobe.report import render_conclusion, render_markdown, write_report
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


class TestEndToEnd:
    def test_mock_run_completes(self, config, lists):
        result = run_experiment(config, lists=lists, n_repeats=2)
        assert result.mock is True
        assert result.control.n > 0
        assert result.ambiguous.n > 0
        assert len(result.ambiguous_detail) == len(lists["ambiguous"])
        assert len(result.nongeo_predictions) == len(lists["nongeo"])

    def test_probe_beats_baselines_when_signal_present(self, config, lists):
        """探针必须明显优于三个基线 —— 否则流程有问题。"""
        result = run_experiment(config, lists=lists, n_repeats=2)
        hidden = result.control_repeats.median_mean
        assert result.signal_ok, "Mock 有信号时，探针应显著优于均值基线"
        for name in ("mean", "random", "char_ngram"):
            baseline = result.baselines.get(name, {}).get("median_km")
            if baseline is None:
                continue
            assert hidden < baseline, f"隐层({hidden:.0f}) 未优于 {name}({baseline:.0f})"

    def test_no_false_signal_without_signal(self, config, lists):
        """关掉信号后，探针不应该还能"学到东西"。

        这是**基线有效性的反向验证**：如果无信号时探针仍然很准，
        说明实验设计里有数据泄露。
        """
        config.set("mock.signal", False)
        result = run_experiment(config, lists=lists, n_repeats=2)
        mean_baseline = result.baselines["mean"]["median_km"]
        hidden = result.control_repeats.median_mean
        # 无信号时，探针与均值基线应在同一量级（允许 30% 波动）
        assert hidden > mean_baseline * 0.7

    def test_repeats_have_spread(self, config, lists):
        result = run_experiment(config, lists=lists, n_repeats=3)
        assert len(result.control_repeats.medians) == 3
        assert result.control_repeats.median_mean > 0

    def test_layer_selection(self, config, lists):
        """Mock 只有 1 层，显式指定层 0 应能工作。"""
        result = run_experiment(config, lists=lists, layer=0, n_repeats=1)
        assert result.layer == 0

    def test_bad_layer_raises(self, config, lists):
        with pytest.raises(KeyError, match="不在结果里"):
            run_experiment(config, lists=lists, layer=99, n_repeats=1)


class TestAltPreference:
    def test_detects_closer_to_alternative(self, lists):
        """如果预测故意放在备选含义的坐标上，应被判定为 closer_to='alt'。"""
        entries = [entry for entry in lists["ambiguous"].entries if entry.has_alt]
        predictions = np.array([[entry.alt_lat, entry.alt_lon] for entry in entries])
        rows = alt_preference_analysis(entries, predictions)
        assert all(row["closer_to"] == "alt" for row in rows)
        assert all(row["dist_alt_km"] < row["dist_primary_km"] for row in rows)

    def test_detects_closer_to_primary(self, lists):
        entries = [entry for entry in lists["ambiguous"].entries if entry.has_alt]
        predictions = np.array([[entry.lat, entry.lon] for entry in entries])
        rows = alt_preference_analysis(entries, predictions)
        assert all(row["closer_to"] == "primary" for row in rows)

    def test_skips_entries_without_alt(self, lists):
        entries = lists["ambiguous"].entries
        predictions = np.zeros((len(entries), 2))
        rows = alt_preference_analysis(entries, predictions)
        assert len(rows) == len(lists["ambiguous"].with_alt())


class TestReport:
    def test_markdown_contains_sections(self, config, lists):
        result = run_experiment(config, lists=lists, n_repeats=1)
        markdown = render_markdown(result)
        for section in ("## 配置", "## 主结果", "## 基线对比", "## 结论", "## 已知局限"):
            assert section in markdown

    def test_conclusions_are_data_driven(self, config, lists):
        """结论必须由数据推出，而不是写死。

        换了结果，结论文字应该跟着变 —— 这里用两种信号强度验证。
        """
        strong = run_experiment(config, lists=lists, n_repeats=1)
        text_strong = " ".join(render_conclusion(strong))

        config.set("mock.signal", False)
        weak = run_experiment(config, lists=lists, n_repeats=1)
        text_weak = " ".join(render_conclusion(weak))

        assert text_strong != text_weak
        assert "探针有效" in text_strong
        # 无信号时应给出警告
        assert "未" in text_weak or "⚠" in text_weak

    def test_warning_when_no_signal(self, config, lists):
        config.set("mock.signal", False)
        result = run_experiment(config, lists=lists, n_repeats=1)
        assert result.warnings

    def test_write_report_and_json(self, config, lists, workdir):
        result = run_experiment(config, lists=lists, n_repeats=1)
        md_path = write_report(result, workdir)
        json_path = save_result(result, workdir / "result.json")
        assert md_path.exists() and json_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert "summary" in payload
        assert "ambiguous_detail" in payload
        assert "alt_analysis" in payload
        assert payload["summary"]["mock"] is True

    def test_markdown_has_tables(self, config, lists):
        result = run_experiment(config, lists=lists, n_repeats=1)
        markdown = render_markdown(result)
        assert markdown.count("|---") >= 2
        assert "Georgia" in markdown or "Washington" in markdown


class TestExtractDataset:
    def test_all_groups_extracted(self, config, lists):
        results = extract_dataset(config, lists)
        assert set(results) == {"control", "ambiguous", "nongeo"}
        for group, extract in results.items():
            assert extract.n_words == len(lists[group])
            assert extract.mock is True

    def test_matrix_shape_consistent(self, config, lists):
        results = extract_dataset(config, lists)
        dim = results["control"].matrix().shape[1]
        for extract in results.values():
            assert extract.matrix().shape[1] == dim

    def test_save_and_load_roundtrip(self, config, lists, workdir):
        results = extract_dataset(config, lists)
        path = results["control"].save(workdir / "control.npz")
        loaded = type(results["control"]).load(path)
        assert loaded.words == results["control"].words
        assert np.allclose(loaded.matrix(), results["control"].matrix())
        assert loaded.mock is True
