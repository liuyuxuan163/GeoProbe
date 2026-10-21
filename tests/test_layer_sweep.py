"""测试：逐层扫描（layer sweep）与多层激活复用。

**这一组测试是为了防止一个真实 bug 复发。**

现象：`python -m geoprobe layer-sweep` 报
``KeyError: 层 0 不在结果里（已提取 [23]）``。

根因：`run_experiment()` 在没传 `extract_results` 时会自己调用
`extract_dataset()`，而后者**默认只提取最后一层**。
早期版本在循环里对每层各调用一次 `run_experiment()`，
于是每次都重新提取（且只提最后一层），请求第 0 层时自然找不到。

修复：一次性提取全部层 → 循环里复用同一个 `extract_results`。

**为什么 Mock 模式测不出这个 bug**：Mock 只有 1 个合成层（layers=[0]），
永远不会出现"请求 0 层但只有 23 层"的情况。
所以这里手工构造一个多层的 `ExtractResult`。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.config import load_config
from geoprobe.experiment import run_experiment
from geoprobe.extract import ExtractResult
from geoprobe.wordlists import load_wordlists


def _multi_layer_results(lists, n_layers: int = 4, dim: int = 16, seed: int = 0) -> dict:
    """构造一份"多层"的 mocked 提取结果，其中只有中间某层真正带有地理信号。

    这样既能测"每一层都能被取到"，也能测"探针确实能挑出有信号的那一层"。
    """
    rng = np.random.default_rng(seed)
    helpful_layer = n_layers // 2
    results = {}

    for group, word_list in lists.items():
        words = word_list.words
        coords = word_list.coords() if word_list.has_truth else None
        matrices = {}
        for layer in range(n_layers):
            noise = rng.normal(0, 1.0, size=(len(words), dim))
            if coords is not None and layer == helpful_layer:
                # 在这一层把坐标编码进去（放大，使信号明显强于噪声）
                noise[:, 0] += coords[:, 0] / 2.0
                noise[:, 1] += coords[:, 1] / 4.0
            matrices[layer] = noise.astype(np.float32)
        results[group] = ExtractResult(
            embeddings=matrices,
            layers=list(range(n_layers)),
            words=words,
            pooling="mock",
            model_name="mock-multilayer",
            mock=True,
        )
    return results


@pytest.fixture()
def config():
    config = load_config()
    config.set("model.mock", True)
    config.set("evaluation.bootstrap_samples", 100)
    return config


@pytest.fixture()
def lists():
    return load_wordlists()


class TestExtractResultLayers:
    def test_all_layers_available(self, lists):
        results = _multi_layer_results(lists)
        extract = results["control"]
        assert extract.layers == [0, 1, 2, 3]
        for layer in extract.layers:
            assert extract.matrix(layer).shape[1] == 16

    def test_missing_layer_raises_with_actionable_hint(self, lists):
        """报错信息要能指引用户怎么修，而不是只给一个 KeyError。"""
        results = _multi_layer_results(lists)
        extract = results["control"]
        with pytest.raises(KeyError, match="layer_indices"):
            extract.matrix(99)

    def test_default_is_last_layer(self, lists):
        results = _multi_layer_results(lists)
        extract = results["control"]
        assert np.array_equal(extract.matrix(), extract.matrix(3))


class TestLayerSweepLoop:
    """模拟 `cmd_layer_sweep` 的正确循环方式：提取一次，逐层复用。"""

    def test_every_layer_runs(self, config, lists):
        """修复的核心断言：循环里每一层都能跑通，不再 KeyError。"""
        extract_results = _multi_layer_results(lists)
        ran = []
        for layer in extract_results["control"].layers:
            result = run_experiment(
                config,
                lists=lists,
                extract_results=extract_results,
                layer=layer,
                n_repeats=1,
                run_baselines=False,
            )
            assert result.layer == layer
            ran.append(layer)
        assert ran == [0, 1, 2, 3]

    def test_signal_layer_stands_out(self, config, lists):
        """有信号的那一层误差应该明显低于其他层 —— 证明逐层扫描有意义。"""
        extract_results = _multi_layer_results(lists)
        errors = {}
        for layer in extract_results["control"].layers:
            result = run_experiment(
                config,
                lists=lists,
                extract_results=extract_results,
                layer=layer,
                n_repeats=1,
                run_baselines=False,
            )
            errors[layer] = result.control_repeats.median_mean

        best = min(errors, key=errors.get)
        assert best == 2, f"应挑出有信号的第 2 层，实际 {errors}"
        assert errors[2] < errors[0] * 0.5

    def test_no_reextraction_needed(self, config, lists):
        """复用同一份 extract_results 时，不应触发重新提取。

        Mock 提取器带 `model_name` 标记；如果被重新提取，
        model_name 会变成 MockExtractor 的默认值。
        """
        extract_results = _multi_layer_results(lists)
        result = run_experiment(
            config,
            lists=lists,
            extract_results=extract_results,
            layer=1,
            n_repeats=1,
        )
        assert result.model_name == "mock-multilayer"

    def test_baselines_only_computed_once(self, config, lists):
        """只有最后一层需要跑基线（其余层省掉，节省时间）。"""
        extract_results = _multi_layer_results(lists)
        with_baselines = run_experiment(
            config, lists=lists, extract_results=extract_results, layer=3, n_repeats=1
        )
        without = run_experiment(
            config,
            lists=lists,
            extract_results=extract_results,
            layer=0,
            n_repeats=1,
            run_baselines=False,
        )
        assert with_baselines.baselines
        assert not without.baselines


class TestExtractionDatasetDefaults:
    def test_default_extracts_only_last_layer(self, config, lists):
        """确认 `extract_dataset` 的默认行为（这正是 bug 的来源，必须被文档化）。"""
        from geoprobe.extract import build_extractor, extract_dataset

        extractor = build_extractor(config)
        results = extract_dataset(config, lists, extractor=extractor)
        # Mock 只有 1 层，用它的层数作为"模型总层数"的替身
        assert results["control"].layers == [0]

    def test_explicit_layer_indices_respected(self, config, lists):
        """显式传全部层号时应该全部被提取。"""
        from geoprobe.extract import build_extractor, extract_dataset

        extractor = build_extractor(config)
        results = extract_dataset(config, lists, layer_indices=[0], extractor=extractor)
        assert results["control"].layers == [0]
