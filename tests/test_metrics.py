"""测试：Haversine 距离、bootstrap 置信区间、误差统计。

这一组测试的重点是**数值正确性** —— 因为整个项目的结论都建立在这些数字上。
算错了不会被任何程序发现，只会得出错误的科学结论。
"""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.metrics import (
    EARTH_RADIUS_KM,
    ErrorStats,
    bootstrap_ci,
    evaluate_predictions,
    haversine_errors,
    haversine_km,
    list_errors,
    ratio_with_comment,
)


class TestHaversine:
    """用已知的真实距离验证公式实现。"""

    def test_same_point_is_zero(self):
        assert haversine_km(48.8566, 2.3522, 48.8566, 2.3522) == pytest.approx(0.0, abs=1e-9)

    def test_paris_to_london(self):
        # 真实大圆距离约 343.5 km
        distance = haversine_km(48.8566, 2.3522, 51.5074, -0.1278)
        assert distance == pytest.approx(343.5, abs=2.0)

    def test_new_york_to_los_angeles(self):
        # 真实大圆距离约 3936 km
        distance = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
        assert distance == pytest.approx(3936, abs=15)

    def test_one_degree_at_equator(self):
        # 赤道上 1 度经度 ≈ 111.19 km
        assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.1)

    def test_one_degree_longitude_shrinks_at_high_latitude(self):
        """同一个 1 度经度，在北纬 60 度只有赤道的一半左右。

        这正是"不能用欧氏距离评估经纬度"的直接证据。
        """
        at_equator = haversine_km(0.0, 0.0, 0.0, 1.0)
        at_60 = haversine_km(60.0, 0.0, 60.0, 1.0)
        assert at_60 == pytest.approx(at_equator / 2, rel=0.02)

    def test_pole_to_pole(self):
        # 半个大圆 = π * R
        assert haversine_km(90.0, 0.0, -90.0, 0.0) == pytest.approx(np.pi * EARTH_RADIUS_KM, abs=1.0)

    def test_antimeridian_short_way(self):
        """跨越 180 度经线时应该算出"短的那一边"。

        (0, 179) 与 (0, -179) 实际只差 2 度经度（约 222 km），
        而不是 358 度。Haversine 用 sin²(Δλ/2) 天然处理了这一点。
        """
        distance = haversine_km(0.0, 179.0, 0.0, -179.0)
        assert distance == pytest.approx(222.4, abs=1.0)

    def test_symmetry(self):
        forward = haversine_km(10.0, 20.0, -30.0, 40.0)
        backward = haversine_km(-30.0, 40.0, 10.0, 20.0)
        assert forward == pytest.approx(backward, rel=1e-9)


class TestHaversineErrors:
    def test_vectorized_matches_scalar(self):
        truth = np.array([[48.8566, 2.3522], [35.6762, 139.6503]])
        pred = np.array([[51.5074, -0.1278], [37.5665, 126.9780]])
        batch = haversine_errors(truth, pred)
        assert batch[0] == pytest.approx(haversine_km(*truth[0], *pred[0]), rel=1e-9)
        assert batch[1] == pytest.approx(haversine_km(*truth[1], *pred[1]), rel=1e-9)

    def test_zero_when_identical(self):
        coords = np.array([[10.0, 20.0], [-30.0, 40.0]])
        assert np.allclose(haversine_errors(coords, coords), 0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="形状不一致"):
            haversine_errors(np.zeros((3, 2)), np.zeros((4, 2)))

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="期望形状"):
            haversine_errors(np.zeros((3, 3)), np.zeros((3, 3)))

    def test_empty_is_allowed(self):
        result = haversine_errors(np.zeros((0, 2)), np.zeros((0, 2)))
        assert result.shape == (0,)


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        values = rng.normal(100, 20, size=200)
        low, high = bootstrap_ci(values, "median", samples=500, seed=1)
        assert low < np.median(values) < high

    def test_ci_narrows_with_more_data(self):
        """样本越多，置信区间越窄 —— 这是 bootstrap 应有的行为。"""
        rng = np.random.default_rng(1)
        small = rng.normal(100, 20, size=30)
        large = rng.normal(100, 20, size=3000)
        low_s, high_s = bootstrap_ci(small, "median", samples=400, seed=2)
        low_l, high_l = bootstrap_ci(large, "median", samples=400, seed=2)
        assert (high_l - low_l) < (high_s - low_s)

    def test_deterministic_with_same_seed(self):
        values = [1.0, 2.0, 3.0, 10.0, 50.0]
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_single_value(self):
        low, high = bootstrap_ci([42.0])
        assert low == high == 42.0

    def test_empty_returns_nan(self):
        low, high = bootstrap_ci([])
        assert np.isnan(low) and np.isnan(high)

    def test_mean_statistic_supported(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = bootstrap_ci(values, statistic="mean", samples=200, seed=3)
        assert low <= 3.0 <= high


class TestEvaluatePredictions:
    def test_perfect_prediction(self):
        coords = np.array([[48.8566, 2.3522], [35.6762, 139.6503], [-33.8688, 151.2093]])
        stats = evaluate_predictions(coords, coords, bootstrap_samples=100)
        assert stats.n == 3
        assert stats.median_km == pytest.approx(0.0, abs=1e-6)
        assert stats.r2_lat == pytest.approx(1.0, abs=1e-9)
        assert stats.r2_lon == pytest.approx(1.0, abs=1e-9)

    def test_r2_reported_per_dimension_not_averaged(self):
        """R² 必须分维度报告。经度的可预测性通常低于纬度，
        取平均会把两者的差异糊掉（这是原实验的一个缺陷）。"""
        truth = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
        pred = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 99.0]])  # 只有经度错
        stats = evaluate_predictions(truth, pred, bootstrap_samples=50)
        assert stats.r2_lat == pytest.approx(1.0, abs=1e-6)
        assert stats.r2_lon < 0.9
        assert stats.r2_lat != stats.r2_lon

    def test_percentiles_reported(self):
        truth = np.zeros((10, 2))
        # 造出从 0 到 9 度纬度的误差（约 0~1000 km）
        pred = np.stack([np.arange(10, dtype=float), np.zeros(10)], axis=1)
        stats = evaluate_predictions(truth, pred, bootstrap_samples=100)
        assert stats.p90_km > stats.median_km
        assert stats.max_km >= stats.p90_km

    def test_to_row_renders(self):
        stats = ErrorStats(n=5, median_km=1234.0, mean_km=2000.0, p90_km=3000.0,
                           ci_low=1000.0, ci_high=1500.0, r2_lat=0.8, r2_lon=0.6)
        row = stats.to_row()
        assert row["n"] == "5"
        assert "1,234" in row["median_km"]
        assert "0.80" in row["R²(lat/lon)"]


class TestRatioWithComment:
    def test_large_ratio(self):
        ratio, comment = ratio_with_comment(300.0, 100.0)
        assert ratio == pytest.approx(3.0)
        assert "显著" in comment

    def test_small_ratio(self):
        ratio, comment = ratio_with_comment(50.0, 100.0)
        assert ratio == pytest.approx(0.5)
        assert "更准" in comment

    def test_zero_denominator_is_safe(self):
        ratio, comment = ratio_with_comment(100.0, 0.0)
        assert np.isnan(ratio)
        assert comment == "—"


class TestListErrors:
    def test_sorted_descending(self):
        words = ["a", "b", "c"]
        truth = np.zeros((3, 2))
        pred = np.array([[1.0, 0.0], [5.0, 0.0], [2.0, 0.0]])
        rows = list_errors(words, truth, pred)
        assert [row["word"] for row in rows] == ["b", "c", "a"]
        assert rows[0]["error_km"] > rows[-1]["error_km"]

    def test_contains_coordinates(self):
        rows = list_errors(["Paris"], np.array([[48.8566, 2.3522]]), np.array([[48.8566, 2.3522]]))
        assert rows[0]["true_lat"] == pytest.approx(48.8566)
        assert rows[0]["pred_lon"] == pytest.approx(2.3522)
