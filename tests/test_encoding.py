"""测试：目标编码（经度循环编码）与「偏向判定」的可靠性门槛。

**这两组测试对应两个真实的建模缺陷**（都在第一次真实模型实验里暴露出来）：

1. 经度是循环量，线性编码对它有偏 → `TargetEncoder("sincos")`
2. "偏向哪个含义"在两个候选都很远时毫无意义 → `reliable` 门槛

测试的重点是**证明修复确实生效**，而不只是"代码能跑"。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.config import load_config
from geoprobe.experiment import alt_preference_analysis, run_experiment
from geoprobe.targets import TargetEncoder, angular_distance_deg
from geoprobe.wordlists import WordEntry, load_wordlists


# --------------------------------------------------------------------------- #
# TargetEncoder
# --------------------------------------------------------------------------- #


class TestTargetEncoderBasics:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="未知的目标编码"):
            TargetEncoder("mercator")

    def test_dims(self):
        assert TargetEncoder("latlon").dim == 2
        assert TargetEncoder("sincos").dim == 3

    def test_encode_shape(self):
        coords = np.array([[10.0, 20.0], [-30.0, 40.0]])
        assert TargetEncoder("latlon").encode(coords).shape == (2, 2)
        assert TargetEncoder("sincos").encode(coords).shape == (2, 3)

    def test_rejects_wrong_input_shape(self):
        with pytest.raises(ValueError, match="期望形状"):
            TargetEncoder("latlon").encode(np.zeros((3, 3)))

    def test_decode_rejects_wrong_dim(self):
        with pytest.raises(ValueError, match="期望形状"):
            TargetEncoder("sincos").decode(np.zeros((3, 2)))


class TestLatLonRoundtrip:
    def test_identity(self):
        coords = np.array([[48.8566, 2.3522], [-33.8688, 151.2093]])
        encoder = TargetEncoder("latlon")
        assert np.allclose(encoder.decode(encoder.encode(coords)), coords)

    def test_latitude_clipped(self):
        """回归可能外推出界，纬度必须裁到 [-90, 90]。"""
        encoder = TargetEncoder("latlon")
        decoded = encoder.decode(np.array([[120.0, 0.0], [-120.0, 0.0]]))
        assert decoded[0, 0] == 90.0
        assert decoded[1, 0] == -90.0

    def test_longitude_wraps_instead_of_clipping(self):
        """经度外推时应该"环绕"而不是裁到 ±180。

        裁的话 200° 会变成 180°（错），环绕的话变成 -160°（对，200° = -160°）。
        """
        encoder = TargetEncoder("latlon")
        decoded = encoder.decode(np.array([[0.0, 200.0]]))
        assert decoded[0, 1] == pytest.approx(-160.0)


class TestSinCosRoundtrip:
    def test_roundtrip_preserves_coordinates(self):
        coords = np.array(
            [
                [48.8566, 2.3522],
                [-33.8688, 151.2093],
                [0.0, 0.0],
                [64.1466, -21.9426],
                [35.6762, 139.6503],
            ]
        )
        encoder = TargetEncoder("sincos")
        restored = encoder.decode(encoder.encode(coords))
        assert np.allclose(restored, coords, atol=1e-9)

    def test_roundtrip_at_antimeridian(self):
        """±180° 附近必须能精确还原 —— 这是线性编码做不到的。"""
        coords = np.array([[0.0, 179.999], [0.0, -179.999], [0.0, 180.0]])
        encoder = TargetEncoder("sincos")
        restored = encoder.decode(encoder.encode(coords))
        assert np.allclose(restored[:, 0], coords[:, 0])
        # 经度可能等价地表示为 +180 或 -180，用角距离比较
        for got, want in zip(restored[:, 1], coords[:, 1]):
            assert angular_distance_deg(got, want) < 1e-6

    def test_sin_cos_on_unit_circle(self):
        """(sin λ, cos λ) 必须落在单位圆上 —— 这是编码正确的直接证据。"""
        lons = np.linspace(-180, 180, 37)
        coords = np.stack([np.zeros_like(lons), lons], axis=1)
        encoded = TargetEncoder("sincos").encode(coords)
        radius = np.sqrt(encoded[:, 1] ** 2 + encoded[:, 2] ** 2)
        assert np.allclose(radius, 1.0, atol=1e-9)


class TestAntimeridianIsTheWholePoint:
    """这一组测试是「为什么必须换成 sin/cos」的直接证据。"""

    def test_linear_encoding_calls_neighbours_far_apart(self):
        """线性编码下，179° 与 -179° 相差 358 度 —— 这是缺陷的根源。"""
        encoded = TargetEncoder("latlon").encode(np.array([[0.0, 179.0], [0.0, -179.0]]))
        gap = abs(encoded[0, 1] - encoded[1, 1])
        assert gap == pytest.approx(358.0)

    def test_sincos_encoding_keeps_neighbours_close(self):
        """sin/cos 编码下，同样两点只差 0.035 —— 相邻关系被正确表达。"""
        encoded = TargetEncoder("sincos").encode(np.array([[0.0, 179.0], [0.0, -179.0]]))
        gap = float(np.linalg.norm(encoded[0] - encoded[1]))
        assert gap < 0.05
        # 对比：线性编码的差距是 358，相差四个数量级
        assert gap * 1000 < 358.0

    def test_angular_distance_helper(self):
        assert angular_distance_deg(179.0, -179.0) == pytest.approx(2.0)
        assert angular_distance_deg(0.0, 90.0) == pytest.approx(90.0)
        assert angular_distance_deg(-170.0, 170.0) == pytest.approx(20.0)


class TestEncodingChangesWhatProbeCanLearn:
    """sincos 在什么情况下真的有用？

    **注意这里的假设**：下面的测试构造了一个"隐层是位置的**循环平滑函数**"
    的合成数据（即隐层里存的是 ``cos λ`` 和 ``sin λ``，而不是原始的度数）。

    * 在这种表示下，``sincos`` 目标是**线性可达**的 → 探针学得好
    * 而 ``latlon`` 目标需要经过 ``atan2`` 还原 → 非线性 → 线性探针学不好

    **这正是我们要验证的假设**：如果真实模型的表征是循环/旋转式的，
    那么"经度不可解码"的结论就是**编码方式的错**，不是模型的错。

    反过来说，如果隐层里存的**就是**经度的线性值（比如 Mock 合成向量那样），
    那 ``latlon`` 反而更贴合 —— 这也是为什么不能用 Mock 结果比较两种编码。
    """

    @staticmethod
    def _make_cyclic_representation(seed: int = 0, n: int = 300):
        """造一份"隐层由 (cos λ, sin λ, lat) 平滑生成"的数据。"""
        rng = np.random.default_rng(seed)
        lon = rng.uniform(-180.0, 180.0, size=n)
        lat = rng.uniform(-90.0, 90.0, size=n)
        lam = np.radians(lon)
        base = np.stack([np.cos(lam), np.sin(lam), lat / 90.0], axis=1)
        # 用随机投影扩到更高维，模拟分布式的表征
        projection = rng.normal(size=(3, 16))
        X = base @ projection + rng.normal(0, 0.01, size=(n, 16))
        return X, np.stack([lat, lon], axis=1)

    def test_sincos_target_is_learnable_on_cyclic_representation(self):
        """表示是循环的时候，sincos 目标应该能被线性探针精确恢复。"""
        from geoprobe.probe import RidgeProbe

        X, coords = self._make_cyclic_representation()
        encoder = TargetEncoder("sincos")
        probe = RidgeProbe().fit(X[:240], encoder.encode(coords)[:240])
        pred = encoder.decode(probe.predict(X[240:]))

        lon_error = np.array(
            [angular_distance_deg(a, b) for a, b in zip(pred[:, 1], coords[240:, 1])]
        )
        lat_error = np.abs(pred[:, 0] - coords[240:, 0])
        assert np.median(lon_error) < 1.0
        assert np.median(lat_error) < 1.0

    def test_latlon_target_is_much_worse_on_cyclic_representation(self):
        """同一份数据换成 latlon 目标：经度误差应该显著更大。

        **这就是"编码缺陷"的直接量化证据。**
        """
        from geoprobe.probe import RidgeProbe

        X, coords = self._make_cyclic_representation()
        train, test = slice(0, 240), slice(240, 300)
        sincos = TargetEncoder("sincos")
        latlon = TargetEncoder("latlon")

        pred_sincos = sincos.decode(RidgeProbe().fit(X[train], sincos.encode(coords)[train]).predict(X[test]))
        pred_latlon = latlon.decode(RidgeProbe().fit(X[train], latlon.encode(coords)[train]).predict(X[test]))

        def median_lon_error(pred):
            return float(
                np.median(
                    [angular_distance_deg(a, b) for a, b in zip(pred[:, 1], coords[test][:, 1])]
                )
            )

        err_sincos = median_lon_error(pred_sincos)
        err_latlon = median_lon_error(pred_latlon)
        # latlon 的误差应该大一个量级以上
        assert err_latlon > err_sincos * 10, f"latlon={err_latlon:.1f}° sincos={err_sincos:.1f}°"


# --------------------------------------------------------------------------- #
# 可靠性门槛
# --------------------------------------------------------------------------- #


def _entry(word: str, lat: float, lon: float, alt_lat: float, alt_lon: float) -> WordEntry:
    return WordEntry(
        word=word,
        lat=lat,
        lon=lon,
        label="主",
        alt_lat=alt_lat,
        alt_lon=alt_lon,
        alt_label="备选",
    )


class TestAltReliability:
    def test_far_candidates_are_unreliable(self):
        """两个候选都很远时，必须标记 reliable=False。

        这正是第一次实验里的陷阱：Santiago 到主 6687 km、到备选 6675 km，
        差值只有 12 km —— 却被算成了"偏向备选"。
        """
        entry = _entry("Santiago", -33.4489, -70.6693, 42.8782, -8.5446)
        # 预测点在大西洋中部，离两边都是几千公里
        predictions = np.array([[10.0, -40.0]])
        rows = alt_preference_analysis([entry], predictions, reliability_km=2500.0)
        assert rows[0]["min_km"] > 2500.0
        assert rows[0]["reliable"] is False

    def test_adaptive_threshold_rescues_otherwise_excluded_case(self):
        """自适应门槛应该把「模型定位正常」的词救回来。

        真实数据里的例子：Valencia 距主含义 4386 km、距备选 2990 km。
        模型自身中位误差是 3103 km，所以 2990 其实是**正常的**定位精度 ——
        固定门槛 2500 会把它误排除，自适应门槛（3103）才会保留它。
        """
        entry = _entry("Valencia", 39.4699, -0.3763, 10.1620, -68.0077)
        predictions = np.array([[20.0, -5.0]])  # 造一个离两边都有距离的点
        rows = alt_preference_analysis([entry], predictions, reliability_km=3103.0)
        assert rows[0]["min_km"] < 3103.0
        assert rows[0]["reliable"] is True

    def test_near_candidate_is_reliable(self):
        """预测点贴近主含义时，判定必须可靠。"""
        entry = _entry("Georgia", 42.3154, 43.3569, 32.1656, -82.9001)
        predictions = np.array([[42.5, 43.0]])  # 就在格鲁吉亚
        rows = alt_preference_analysis([entry], predictions, reliability_km=2500.0)
        assert rows[0]["min_km"] < 2500.0
        assert rows[0]["reliable"] is True
        assert rows[0]["closer_to"] == "primary"

    def test_near_alternative_is_reliable(self):
        """预测点贴近备选含义时，判定同样必须可靠。"""
        entry = _entry("Georgia", 42.3154, 43.3569, 32.1656, -82.9001)
        predictions = np.array([[32.0, -83.0]])  # 就在美国佐治亚州
        rows = alt_preference_analysis([entry], predictions, reliability_km=2500.0)
        assert rows[0]["reliable"] is True
        assert rows[0]["closer_to"] == "alt"

    def test_threshold_is_configurable(self):
        entry = _entry("Georgia", 42.3154, 43.3569, 32.1656, -82.9001)
        predictions = np.array([[50.0, 20.0]])  # 距格鲁吉亚约 1500 km
        strict = alt_preference_analysis([entry], predictions, reliability_km=500.0)
        loose = alt_preference_analysis([entry], predictions, reliability_km=5000.0)
        assert strict[0]["reliable"] is False
        assert loose[0]["reliable"] is True

    def test_delta_and_min_reported(self):
        entry = _entry("Georgia", 42.3154, 43.3569, 32.1656, -82.9001)
        rows = alt_preference_analysis([entry], np.array([[42.5, 43.0]]))
        row = rows[0]
        assert {"delta_km", "min_km", "reliable", "closer_to"} <= set(row)
        assert row["min_km"] == pytest.approx(min(row["dist_primary_km"], row["dist_alt_km"]))
        assert row["delta_km"] == pytest.approx(abs(row["dist_primary_km"] - row["dist_alt_km"]))

    def test_skips_entries_without_alt(self):
        entries = load_wordlists()["ambiguous"].entries
        predictions = np.zeros((len(entries), 2))
        rows = alt_preference_analysis(entries, predictions)
        assert len(rows) == len(load_wordlists()["ambiguous"].with_alt())


# --------------------------------------------------------------------------- #
# 端到端：编码配置贯通
# --------------------------------------------------------------------------- #


@pytest.fixture()
def config():
    config = load_config()
    config.set("model.mock", True)
    config.set("evaluation.bootstrap_samples", 100)
    return config


@pytest.fixture()
def lists():
    return load_wordlists()


class TestEncodingWireThrough:
    def test_sincos_runs_end_to_end(self, config, lists):
        config.set("probe.target_encoding", "sincos")
        result = run_experiment(config, lists=lists, n_repeats=2)
        assert result.target_encoding == "sincos"
        assert result.control.n > 0
        # 预测坐标必须仍落在合法地理范围内（解码正确）
        for row in result.ambiguous_detail:
            assert -90 <= row["pred_lat"] <= 90
            assert -180 <= row["pred_lon"] <= 180

    def test_encoding_recorded_in_summary(self, config, lists):
        config.set("probe.target_encoding", "sincos")
        result = run_experiment(config, lists=lists, n_repeats=1)
        assert result.summary_dict()["target_encoding"] == "sincos"

    def test_reliability_threshold_is_adaptive(self, config, lists):
        """有效门槛 = max(配置下限, 模型自身中位误差)。

        为什么不能用固定下限：如果门槛比模型自身误差还严，
        会把**模型定位正常的词**也判成不可靠 —— 那等于什么都没分析。
        """
        config.set("evaluation.alt_reliability_km", 100.0)
        result = run_experiment(config, lists=lists, n_repeats=2)
        control_median = result.control_repeats.median_mean
        assert result.alt_reliability_used == pytest.approx(max(100.0, control_median))

        # 门槛确实被抬到了模型自身误差量级，所以应有部分判定可靠
        assert result.alt_reliability_used >= control_median - 1e-6
        assert any(row["reliable"] for row in result.alt_analysis)

    def test_reliability_floor_dominates_when_higher(self, config, lists):
        """配置下限高于模型误差时，以下限为准。"""
        config.set("evaluation.alt_reliability_km", 99999.0)
        result = run_experiment(config, lists=lists, n_repeats=1)
        assert result.alt_reliability_used == pytest.approx(99999.0)
        assert all(row["reliable"] for row in result.alt_analysis)

    def test_mock_warns_against_encoding_comparison(self, config, lists):
        """Mock 的合成向量是经度线性的，不能被用来比较两种编码。"""
        result = run_experiment(config, lists=lists, n_repeats=1)
        assert any("Mock" in warning and "target_encoding" in warning for warning in result.warnings)

    def test_report_mentions_encoding(self, config, lists):
        from geoprobe.report import render_markdown

        config.set("probe.target_encoding", "sincos")
        result = run_experiment(config, lists=lists, n_repeats=1)
        markdown = render_markdown(result)
        assert "目标编码" in markdown
        assert "sincos" in markdown

    def test_report_shows_reliability_column(self, config, lists):
        from geoprobe.report import render_markdown

        result = run_experiment(config, lists=lists, n_repeats=1)
        markdown = render_markdown(result)
        assert "可靠" in markdown
        assert "最近候选" in markdown
