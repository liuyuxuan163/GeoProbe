"""测试：经度专用指标（圆形 R²）与汇总 R²。

**这两组测试对应两个会让结论完全失真的度量问题**：

1. 单次划分算 R² 极不稳定 —— 22 个留出点上，同一层的 R²(经度)
   实测能在 -1.01 ~ 0.85 之间跳。必须汇总所有重复的预测再算。
2. 普通线性 R² 对经度**无效** —— 真实 -179°、预测 +179° 地理上只差 2°，
   线性 R² 却记成 358° 的巨大残差。必须用圆形 R²。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.metrics import angular_diff_deg, circular_r2, haversine_km


class TestAngularDiff:
    def test_wraparound_is_short_way(self):
        assert angular_diff_deg(np.array([179.0]), np.array([-179.0]))[0] == pytest.approx(-2.0)
        assert angular_diff_deg(np.array([-179.0]), np.array([179.0]))[0] == pytest.approx(2.0)

    def test_zero_for_same_angle(self):
        assert angular_diff_deg(np.array([45.0]), np.array([45.0]))[0] == pytest.approx(0.0)

    def test_range_is_within_180(self):
        truth = np.linspace(-180, 180, 73)
        pred = np.linspace(180, -180, 73)
        diffs = angular_diff_deg(truth, pred)
        assert np.all(np.abs(diffs) <= 180.0)

    def test_consistent_with_haversine_at_equator(self):
        """角距离与 Haversine 在赤道上应当一致（1 度 ≈ 111.19 km）。"""
        diff = angular_diff_deg(np.array([179.0]), np.array([-179.0]))[0]
        distance = haversine_km(0.0, 179.0, 0.0, -179.0)
        assert abs(diff) * 111.19 == pytest.approx(distance, rel=0.01)


class TestCircularR2:
    def test_perfect_prediction(self):
        truth = np.array([10.0, 50.0, -170.0, 179.0])
        assert circular_r2(truth, truth) == pytest.approx(1.0)

    def test_wraparound_prediction_is_rewarded(self):
        """预测跨过日界线时，圆形 R² 应该远好于线性 R²。

        关键点：这里**不是**要求圆形 R² 接近 1 ——
        R² 是相对「永远猜均值」的基线，误差比基线还大时它就是负的。
        这里要证明的是：**同样的预测，圆形 R² 给出合理正值，线性 R² 被 358° 残差毁掉。**
        """
        truth = np.array([170.0, 175.0, 180.0, -175.0, -170.0])
        pred = np.array([175.0, 180.0, -175.0, -170.0, -165.0])  # 每点 +5°，其中一个环绕

        circ = circular_r2(truth, pred)
        assert circ > 0.4, f"圆形 R² 应为合理正值，实际 {circ:.3f}"

        # 线性 R² 的对照计算
        residual = truth - pred
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        linear = 1.0 - ss_res / ss_tot

        assert linear < 0.2, f"线性 R² 应被日界线拖累，实际 {linear:.3f}"
        assert circ > linear * 2, f"圆形 {circ:.3f} 应明显优于线性 {linear:.3f}"

    def test_linear_r2_is_catastrophic_on_equivalent_angles(self):
        """真实 -179°、预测 +179°（地理上只差 2°）—— 线性 R² 会给出灾难性负值。

        这是「不能用线性 R² 评价经度」最直接的证据。
        """
        truth = np.array([-179.0, 179.0])
        pred = np.array([179.0, -179.0])
        assert np.allclose(np.abs(angular_diff_deg(truth, pred)), 2.0)

        residual = truth - pred
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        linear = 1.0 - ss_res / ss_tot
        # 残差 358° 而数据本身只散布 ±1°，线性 R² 崩到 -3
        assert linear < -1.0

    def test_zero_equals_circular_mean_baseline(self):
        """永远预测圆形均值时，圆形 R² 应约为 0。"""
        truth = np.array([0.0, 90.0, 180.0, 270.0])  # 圆形均值无定义时取近似
        rad = np.radians(truth)
        mean_angle = np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
        pred = np.full_like(truth, mean_angle)
        assert circular_r2(truth, pred) == pytest.approx(0.0, abs=1e-9)

    def test_negative_when_worse_than_mean(self):
        truth = np.array([0.0, 10.0, 20.0, 30.0])
        pred = np.array([180.0, 190.0, 200.0, 210.0])  # 全都错了 180 度
        assert circular_r2(truth, pred) < 0

    def test_empty_is_nan(self):
        assert np.isnan(circular_r2(np.array([]), np.array([])))

    def test_monotone_in_error_magnitude(self):
        """误差越大，圆形 R² 越低。"""
        truth = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
        r2s = [circular_r2(truth, truth + shift) for shift in (0.0, 10.0, 30.0, 60.0)]
        assert r2s == sorted(r2s, reverse=True)


class TestPooledR2:
    """汇总 R² 必须比单次划分稳定得多。"""

    @staticmethod
    def _setup(weak_signal: bool = False):
        from geoprobe.config import load_config
        from geoprobe.extract import ExtractResult
        from geoprobe.wordlists import load_wordlists

        lists = load_wordlists()
        config = load_config()
        config.set("evaluation.bootstrap_samples", 100)

        # 造一份 4 层的合成激活，让第 2 层带地理信号。
        # weak_signal=True 时把信号缩小，用来观察 R² 的抽样不稳定性。
        scale_lat = 1.0 / 12.0 if weak_signal else 1.0 / 2.0
        scale_lon = 1.0 / 24.0 if weak_signal else 1.0 / 4.0
        rng = np.random.default_rng(0)
        cached = {}
        for group, word_list in lists.items():
            coords = word_list.coords() if word_list.has_truth else None
            matrices = {}
            for layer in range(4):
                mat = rng.normal(0, 1.0, size=(len(word_list), 12))
                if coords is not None and layer == 2:
                    mat[:, 0] += coords[:, 0] * scale_lat
                    mat[:, 1] += coords[:, 1] * scale_lon
                matrices[layer] = mat.astype(np.float32)
            cached[group] = ExtractResult(
                embeddings=matrices,
                layers=list(range(4)),
                words=word_list.words,
                pooling="mock",
                model_name="mock-multilayer",
                mock=True,
            )
        return config, lists, cached

    def test_pooled_n_reflects_repeats(self):
        from geoprobe.experiment import run_experiment

        config, lists, cached = self._setup()
        result = run_experiment(
            config, lists=lists, extract_results=cached, layer=2,
            n_repeats=4, run_baselines=False,
        )
        # 108 个词 × 20% 留出 = 22 个，4 次重复 → 约 88
        assert result.pooled_n == pytest.approx(88, abs=4)

    def test_single_split_r2_is_unstable(self):
        """单次划分的 R² 会在不同 seed 下波动 —— 这是必须汇总的原因。

        用**弱信号**构造（真实模型上信号也弱）：如果合成数据信号很强，
        R² 在任何划分下都接近 1，反而看不出问题。
        """
        from geoprobe.experiment import run_experiment

        config, lists, cached = self._setup(weak_signal=True)
        singles = []
        for offset in range(8):
            config.set("probe.random_state", 200 + offset)
            result = run_experiment(
                config, lists=lists, extract_results=cached, layer=2,
                n_repeats=1, run_baselines=False,
            )
            singles.append(result.control.r2_lon)

        spread = max(singles) - min(singles)
        assert spread > 0.1, f"弱信号下单次 R² 应明显波动，实测跨度 {spread:.3f}"

        # 汇总后应当落在单次值的范围内且更接近整体
        config.set("probe.random_state", 42)
        pooled = run_experiment(
            config, lists=lists, extract_results=cached, layer=2,
            n_repeats=8, run_baselines=False,
        )
        assert pooled.pooled_n > 100  # 22 × 8
        assert min(singles) - 0.3 <= pooled.pooled_r2_lon <= max(singles) + 0.3

    def test_circular_r2_present_and_bounded(self):
        from geoprobe.experiment import run_experiment

        config, lists, cached = self._setup()
        result = run_experiment(
            config, lists=lists, extract_results=cached, layer=2,
            n_repeats=4, run_baselines=False,
        )
        assert not np.isnan(result.pooled_circular_r2_lon)
        # 有信号的第 2 层，圆形 R²(经度) 应明显为正
        assert result.pooled_circular_r2_lon > 0.3
