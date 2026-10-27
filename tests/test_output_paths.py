"""测试：输出命名（防止覆盖）与内存预估。

**这一组测试保护的是一个反复发作的 bug。**

同一个"文件名里少一个区分维度"的错误已经犯了三次：

1. `run --mock` 与真实结果都写 `result.json` → Mock 冲掉真实结果
2. latlon 与 sincos 都写 `layer_sweep.json` → 换编码冲掉上一次
3. 0.5B / 1.5B / 3B 都写 `result.sincos.json` → **换模型冲掉上一次**（实际发生过）

现在文件名同时带「模型标识」和「目标编码」，本组测试确保它不会再退化。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.config import load_config, model_slug, output_paths
from geoprobe.extract import (
    estimate_model_memory_gb,
    find_cached_activations,
)


# --------------------------------------------------------------------------- #
# 模型标识
# --------------------------------------------------------------------------- #


class TestModelSlug:
    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
            ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b-instruct"),
            ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3b-instruct"),
            ("meta-llama/Llama-3.2-1B-Instruct", "llama-3.2-1b-instruct"),
            ("gpt2", "gpt2"),
        ],
    )
    def test_slugify(self, model_name, expected):
        config = load_config()
        config.set("model.name", model_name)
        assert model_slug(config) == expected

    def test_mock_has_own_slug(self):
        config = load_config()
        config.set("model.mock", True)
        assert model_slug(config) == "mock"

    def test_slug_is_filename_safe(self):
        config = load_config()
        config.set("model.name", "Some Org/Weird Model (v2)!!")
        slug = model_slug(config)
        for char in '/\\:*?"<>| ()!':
            assert char not in slug

    def test_empty_name_does_not_crash(self):
        config = load_config()
        config.set("model.name", "")
        assert model_slug(config) == "unknown"

    def test_local_path_gives_same_slug_as_hub_name(self):
        """从本地目录加载时，slug 必须和 hub 名字一致。

        否则 ``--model-name D:\\models\\Qwen2.5-3B-Instruct`` 会写出
        ``d-models-qwen2.5-3b-instruct.*`` 这种和之前结果对不上的文件名，
        跨模型对比就会把这些结果误当成"另一个模型"。
        """
        expected = None
        for name in (
            "Qwen/Qwen2.5-3B-Instruct",
            r"D:\models\Qwen2.5-3B-Instruct",
            "/home/user/models/Qwen2.5-3B-Instruct",
            "Qwen2.5-3B-Instruct",
        ):
            config = load_config()
            config.set("model.name", name)
            slug = model_slug(config)
            if expected is None:
                expected = slug
            assert slug == expected, f"{name} → {slug}，期望 {expected}"


class TestOutputPaths:
    def test_different_models_never_collide(self):
        """**核心断言**：三个模型的输出文件名必须互不相同。"""
        results = {}
        for name in (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
        ):
            config = load_config()
            config.set("model.name", name)
            config.set("probe.target_encoding", "sincos")
            results[name] = output_paths(config)["result"].name

        assert len(set(results.values())) == 3, f"文件名冲突：{results}"

    def test_different_encodings_never_collide(self):
        results = set()
        for encoding in ("latlon", "sincos"):
            config = load_config()
            config.set("probe.target_encoding", encoding)
            results.add(output_paths(config)["result"].name)
        assert len(results) == 2

    def test_mock_and_real_never_collide(self):
        config = load_config()
        config.set("probe.target_encoding", "latlon")
        real = output_paths(config)["result"].name
        config.set("model.mock", True)
        mock = output_paths(config)["result"].name
        assert real != mock
        assert "mock" in mock

    def test_all_keys_present(self):
        paths = output_paths(load_config())
        for key in ("result", "layer_sweep", "report", "activations", "model_slug", "encoding"):
            assert key in paths

    def test_activations_template_has_placeholder(self):
        template = output_paths(load_config())["activations"]
        assert "{group}" in str(template)


# --------------------------------------------------------------------------- #
# 缓存查找：不允许误用别的模型的激活
# --------------------------------------------------------------------------- #


class TestFindCachedActivations:
    def test_missing_when_nothing_cached(self, workdir, monkeypatch):
        config = load_config()
        config.set("model.name", "Qwen/Qwen2.5-3B-Instruct")
        # 把 out_dir 指到一个空目录
        config.set("report.out_dir", str(workdir))
        cached, missing = find_cached_activations(config)
        assert cached == {}
        assert len(missing) == 3

    def test_rejects_legacy_cache_of_other_model(self, workdir):
        """旧命名文件如果记录的是别的模型，必须拒绝使用 —— 否则对比会无声出错。"""
        from geoprobe.extract import ExtractResult

        # 造一个"属于 0.5B"的旧命名缓存
        legacy = workdir / "activations.control.npz"
        ExtractResult(
            embeddings={0: np.zeros((4, 8), dtype=np.float32)},
            layers=[0],
            words=["a", "b", "c", "d"],
            pooling="entity_token",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            mock=False,
        ).save(legacy)

        config = load_config()
        config.set("model.name", "Qwen/Qwen2.5-1.5B-Instruct")
        config.set("report.out_dir", str(workdir))
        cached, missing = find_cached_activations(config, ["control"])
        assert "control" not in cached
        assert missing

    def test_accepts_legacy_cache_of_same_model(self, workdir):
        from geoprobe.extract import ExtractResult

        legacy = workdir / "activations.control.npz"
        ExtractResult(
            embeddings={0: np.zeros((4, 8), dtype=np.float32)},
            layers=[0],
            words=["a", "b", "c", "d"],
            pooling="entity_token",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            mock=False,
        ).save(legacy)

        config = load_config()
        config.set("model.name", "Qwen/Qwen2.5-0.5B-Instruct")
        config.set("report.out_dir", str(workdir))
        cached, missing = find_cached_activations(config, ["control"])
        assert "control" in cached
        assert not missing

    def test_prefers_new_naming(self, workdir):
        from geoprobe.extract import ExtractResult

        for name, model in (
            ("activations.control.npz", "Qwen/Qwen2.5-0.5B-Instruct"),
            ("activations.qwen2.5-1.5b-instruct.control.npz", "Qwen/Qwen2.5-1.5B-Instruct"),
        ):
            ExtractResult(
                embeddings={0: np.zeros((4, 8), dtype=np.float32)},
                layers=[0],
                words=["a", "b", "c", "d"],
                pooling="entity_token",
                model_name=model,
                mock=False,
            ).save(workdir / name)

        config = load_config()
        config.set("model.name", "Qwen/Qwen2.5-1.5B-Instruct")
        config.set("report.out_dir", str(workdir))
        cached, _missing = find_cached_activations(config, ["control"])
        assert cached["control"].model_name == "Qwen/Qwen2.5-1.5B-Instruct"


# --------------------------------------------------------------------------- #
# 内存预估
# --------------------------------------------------------------------------- #


class TestMemoryEstimate:
    def test_larger_model_needs_more(self):
        small = estimate_model_memory_gb("Qwen/Qwen2.5-0.5B-Instruct", "float32")
        large = estimate_model_memory_gb("Qwen/Qwen2.5-3B-Instruct", "float32")
        assert small < large

    def test_bfloat16_is_half_of_float32(self):
        fp32 = estimate_model_memory_gb("Qwen/Qwen2.5-1.5B-Instruct", "float32")
        bf16 = estimate_model_memory_gb("Qwen/Qwen2.5-1.5B-Instruct", "bfloat16")
        assert bf16 == pytest.approx(fp32 / 2, rel=0.01)

    def test_unknown_model_returns_none(self):
        # 名字里没有参数量 → 无法估算 → 返回 None（不阻塞）
        assert estimate_model_memory_gb("gpt2", "float32") is None

    def test_budget_check_passes_for_tiny_model(self):
        from geoprobe.extract import check_memory_budget

        # 极小模型不应抛异常
        check_memory_budget("dummy-0.01B", "float32", "cpu")

    def test_budget_check_raises_with_actionable_message(self):
        from geoprobe.extract import check_memory_budget

        with pytest.raises(MemoryError) as excinfo:
            check_memory_budget("huge/Model-9999B", "float32", "cpu")
        message = str(excinfo.value)
        assert "bfloat16" in message  # 给出降精度的出路
        assert "内存" in message or "显存" in message
