"""测试：探针、基线、隐层池化、词表校验。

重点是验证"探针真能学到线性关系"以及"没有信号时探针应该学不到" ——
这两个方向必须都成立，否则整个实验的论证是空的。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.baselines import CharNgramBaseline, RandomBaseline
from geoprobe.config import Config, DEFAULT_CONFIG, loads_yaml, load_config
from geoprobe.extract import MockExtractor, pool_activations, resolve_transformer_layers
from geoprobe.probe import MLPProbe, MeanProbe, RidgeProbe, build_probe
from geoprobe.wordlists import WordList, WordEntry, load_wordlists


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_dot_access(self):
        config = Config({"a": {"b": {"c": 3}}})
        assert config.get("a.b.c") == 3
        assert config.get("a.b.missing", "fallback") == "fallback"

    def test_defaults_loaded(self):
        config = load_config()
        assert config.get("model.name")
        assert config.get("probe.ridge_alphas")
        assert config.get("mock.dim") == 32

    def test_set_overrides(self):
        config = Config({"a": {"b": 1}})
        config.set("a.b", 9)
        assert config.get("a.b") == 9

    def test_mini_yaml_parses_nested(self):
        data = loads_yaml(
            "model:\n  name: test\n  mock: true\nprobe:\n  alphas: [1, 2]\n",
            force_mini=True,
        )
        assert data["model"]["name"] == "test"
        assert data["model"]["mock"] is True
        assert data["probe"]["alphas"] == [1, 2]

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GEO_PROBE_MODEL_MOCK", "true")
        config = load_config()
        assert config.get("model.mock") is True


# --------------------------------------------------------------------------- #
# 词表
# --------------------------------------------------------------------------- #


class TestWordlists:
    def test_loads_three_groups(self):
        lists = load_wordlists()
        assert set(lists) == {"control", "ambiguous", "nongeo"}

    def test_sizes_reasonable(self):
        lists = load_wordlists()
        assert len(lists["control"]) >= 40
        assert len(lists["ambiguous"]) >= 10
        assert len(lists["nongeo"]) >= 10

    def test_control_has_truth(self):
        assert load_wordlists()["control"].has_truth

    def test_nongeo_has_no_truth(self):
        lists = load_wordlists()
        assert not lists["nongeo"].has_truth
        with pytest.raises(ValueError, match="缺少真实坐标"):
            lists["nongeo"].coords()

    def test_ambiguous_has_alternatives(self):
        entries = load_wordlists()["ambiguous"].with_alt()
        assert len(entries) >= 10
        words = {entry.word for entry in entries}
        # 这两个词必须有备选含义，是实验的核心样本
        assert "Georgia" in words
        assert "Washington" in words

    def test_coords_shape(self):
        coords = load_wordlists()["control"].coords()
        assert coords.shape[1] == 2
        assert np.all(np.abs(coords[:, 0]) <= 90)   # 纬度
        assert np.all(np.abs(coords[:, 1]) <= 180)  # 经度

    def test_duplicate_word_rejected(self, workdir):
        import json

        (workdir / "words_control.json").write_text(
            json.dumps([{"word": "Paris", "lat": 1, "lon": 2}] * 12), encoding="utf-8"
        )
        (workdir / "words_ambiguous.json").write_text(
            json.dumps([{"word": "Paris", "lat": 1, "lon": 2}]), encoding="utf-8"
        )
        (workdir / "words_nongeo.json").write_text(json.dumps([{"word": "x"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="重复"):
            load_wordlists(workdir)

    def test_invalid_latitude_rejected(self, workdir):
        import json

        bad = [{"word": f"w{i}", "lat": 1, "lon": 2} for i in range(12)]
        bad[0]["lat"] = 999
        (workdir / "words_control.json").write_text(json.dumps(bad), encoding="utf-8")
        (workdir / "words_ambiguous.json").write_text(
            json.dumps([{"word": "amb", "lat": 1, "lon": 2}]), encoding="utf-8"
        )
        (workdir / "words_nongeo.json").write_text(json.dumps([{"word": "x"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="纬度越界"):
            load_wordlists(workdir)


# --------------------------------------------------------------------------- #
# 池化
# --------------------------------------------------------------------------- #


class TestPooling:
    def test_last_token(self):
        hidden = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert np.array_equal(pool_activations(hidden, "last_token"), hidden[-1])

    def test_mean(self):
        hidden = np.ones((5, 3), dtype=np.float32) * 2
        assert np.allclose(pool_activations(hidden, "mean"), 2.0)

    def test_entity_token_uses_span(self):
        hidden = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)
        pooled = pool_activations(hidden, "entity_token", entity_span=(1, 3))
        assert pooled[0] == pytest.approx(1.0)

    def test_entity_token_without_span_falls_back(self):
        hidden = np.arange(8, dtype=np.float32).reshape(4, 2)
        assert np.array_equal(pool_activations(hidden, "entity_token"), hidden[-1])

    def test_span_is_clamped(self):
        hidden = np.ones((3, 2), dtype=np.float32)
        # 越界的 span 不应崩溃
        assert pool_activations(hidden, "entity_token", entity_span=(10, 99)).shape == (2,)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="未知的池化"):
            pool_activations(np.ones((2, 2)), "bogus")

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="期望"):
            pool_activations(np.ones(5), "mean")


# --------------------------------------------------------------------------- #
# 层路径兼容（这是最容易卡住的坑）
# --------------------------------------------------------------------------- #


class _FakeQwen2:
    """模拟 Qwen2/2.5 的结构：model.model.layers"""

    class _Inner:
        layers = [object() for _ in range(4)]

    def __init__(self):
        self.model = self._Inner()


class _FakeGPT2:
    """模拟 GPT-2 / Qwen1 的结构：model.transformer.h"""

    class _Inner:
        h = [object() for _ in range(6)]

    def __init__(self):
        self.transformer = self._Inner()


class _FakeUnknown:
    pass


class TestLayerResolution:
    def test_qwen2_style(self):
        layers = resolve_transformer_layers(_FakeQwen2())
        assert len(layers) == 4

    def test_gpt2_style(self):
        layers = resolve_transformer_layers(_FakeGPT2())
        assert len(layers) == 6

    def test_unknown_model_raises_with_hint(self):
        with pytest.raises(AttributeError, match="无法识别模型结构"):
            resolve_transformer_layers(_FakeUnknown())


# --------------------------------------------------------------------------- #
# Mock 提取器
# --------------------------------------------------------------------------- #


class TestMockExtractor:
    def test_deterministic(self):
        extractor = MockExtractor()
        assert np.allclose(extractor.embed("Paris", (48.8, 2.3)), extractor.embed("Paris", (48.8, 2.3)))

    def test_different_words_differ(self):
        extractor = MockExtractor()
        assert not np.allclose(extractor.embed("Paris"), extractor.embed("Tokyo"))

    def test_signal_encoded(self):
        """有信号时，坐标相近的词向量也应该相近。"""
        extractor = MockExtractor(with_signal=True, noise=0.01)
        north = extractor.embed("A", (60.0, 10.0))
        also_north = extractor.embed("B", (61.0, 11.0))
        south = extractor.embed("C", (-40.0, 10.0))
        assert np.linalg.norm(north - also_north) < np.linalg.norm(north - south)

    def test_no_signal_is_noise(self):
        extractor = MockExtractor(with_signal=False, noise=0.5)
        north = extractor.embed("A", (60.0, 10.0))
        south = extractor.embed("B", (-40.0, 10.0))
        # 无信号时，距离关系不应由坐标决定（期望与随机噪声同量级）
        assert np.linalg.norm(north - south) < 10.0


# --------------------------------------------------------------------------- #
# 探针
# --------------------------------------------------------------------------- #


def _make_linear_data(n: int = 120, dim: int = 16, noise: float = 0.0, seed: int = 0):
    """造一份"坐标由隐层**线性**决定"的数据。

    注意：早期版本用 `90 * tanh(...)` 把值压到合法范围，但那让关系变成非线性的，
    Ridge 自然学不好 —— 会让"线性探针能恢复线性信号"这条测试失败。
    这里改成纯线性（用缩放而非 tanh 控制量级）。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    Y = np.stack([X[:, 0] * 20.0, X[:, 1] * 35.0], axis=1)
    if noise > 0:
        Y = Y + rng.normal(0, noise, size=Y.shape)
    return X, Y


class TestRidgeProbe:
    def test_recovers_linear_signal(self):
        X, Y = _make_linear_data()
        train = slice(0, 90)
        test = slice(90, 120)
        probe = RidgeProbe().fit(X[train], Y[train])
        pred = probe.predict(X[test])
        # 无噪声的线性关系应该被几乎完美恢复
        assert np.max(np.abs(pred - Y[test])) < 6.0

    def test_alpha_selected_from_grid(self):
        X, Y = _make_linear_data()
        probe = RidgeProbe(alphas=[0.01, 1.0, 100.0]).fit(X, Y)
        assert probe.alpha_ in (0.01, 1.0, 100.0)

    def test_noise_reduces_accuracy(self):
        X_clean, Y_clean = _make_linear_data(noise=0.0, seed=1)
        X_noisy, Y_noisy = _make_linear_data(noise=30.0, seed=1)
        err_clean = np.mean(
            np.abs(RidgeProbe().fit(X_clean[:90], Y_clean[:90]).predict(X_clean[90:]) - Y_clean[90:])
        )
        err_noisy = np.mean(
            np.abs(RidgeProbe().fit(X_noisy[:90], Y_noisy[:90]).predict(X_noisy[90:]) - Y_noisy[90:])
        )
        assert err_noisy > err_clean

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="尚未 fit"):
            RidgeProbe().predict(np.zeros((1, 2)))

    def test_describe_contains_alpha(self):
        X, Y = _make_linear_data()
        probe = RidgeProbe().fit(X, Y)
        assert "alpha" in probe.describe()


class TestMLPProbe:
    def test_fits_nonlinear_relation(self):
        """MLP 应该能拟合 Ridge 拟合不了的非线性关系。"""
        rng = np.random.default_rng(0)
        X = rng.uniform(-np.pi, np.pi, size=(200, 1))
        Y = np.stack([90.0 * np.sin(X[:, 0]), 180.0 * np.cos(X[:, 0])], axis=1)

        ridge_err = np.mean(
            np.abs(RidgeProbe().fit(X[:150], Y[:150]).predict(X[150:]) - Y[150:])
        )
        mlp_err = np.mean(
            np.abs(MLPProbe(hidden=64, max_iter=2000).fit(X[:150], Y[:150]).predict(X[150:]) - Y[150:])
        )
        assert mlp_err < ridge_err


class TestMeanProbe:
    def test_always_predicts_mean(self):
        X = np.zeros((5, 3))
        Y = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
        probe = MeanProbe().fit(X, Y)
        pred = probe.predict(np.zeros((4, 3)))
        assert np.allclose(pred, [[10.0, 20.0]] * 4)

    def test_build_probe_dispatch(self):
        config = Config(DEFAULT_CONFIG)
        assert isinstance(build_probe(config, "ridge"), RidgeProbe)
        assert isinstance(build_probe(config, "mean"), MeanProbe)
        assert isinstance(build_probe(config, "mlp"), MLPProbe)
        with pytest.raises(ValueError, match="未知探针"):
            build_probe(config, "nope")


# --------------------------------------------------------------------------- #
# 基线
# --------------------------------------------------------------------------- #


class TestBaselines:
    def test_random_is_deterministic(self):
        baseline = RandomBaseline(dim=8, seed=1)
        assert np.allclose(baseline.transform(["a", "b"]), baseline.transform(["a", "b"]))

    def test_random_shape(self):
        assert RandomBaseline(dim=8).transform(["a", "b", "c"]).shape == (3, 8)

    def test_char_ngram_captures_spelling(self):
        """字符 n-gram 应该让拼写相近的词获得相近的特征。

        （这正是它作为基线有价值的原因：如果它也能拿到不错的 R²，
        说明"地理知识"可能只是拼写规律的副产品。）
        """
        baseline = CharNgramBaseline(ngram_range=(1, 3))
        matrix = baseline.fit_transform(["Kazakhstan", "Uzbekistan", "Paris"])
        # 两个 -stan 国家的向量应该比与 Paris 更接近
        d_central_asia = np.linalg.norm(matrix[0] - matrix[1])
        d_to_paris = np.linalg.norm(matrix[0] - matrix[2])
        assert d_central_asia < d_to_paris

    def test_char_ngram_requires_fit(self):
        with pytest.raises(RuntimeError, match="先调用 fit_transform"):
            CharNgramBaseline().transform(["x"])
