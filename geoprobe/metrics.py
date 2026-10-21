"""评估指标：Haversine 距离、bootstrap 置信区间、误差统计。

**为什么这个模块是本项目的核心之一**
------------------------------------
课程原实验用 `R²` 和 `mean_squared_error` 评估探针，有两个致命问题：

1. **R² 对经纬度两维取平均**（sklearn 多输出默认 `uniform_average`），
   把"经度可预测性"和"纬度可预测性"糊成一个数，信息丢失。
2. **MSE 的单位是"度的平方"，没有物理意义。** 12.5 平方度是好是差？没人能判断。

更要紧的是：**经纬度是球面坐标，不能用欧氏距离评估。**
在赤道附近 1 度经度约 111 km，但在北纬 60 度只有约 55 km —— 同一"度"误差，
在不同纬度意味着不同的实际距离。

所以本项目主指标换成 **Haversine 大圆距离（公里）**：

    R² = 0.85 听起来不错，但平均误差可能是 3000 公里 ——
    意味着模型知道"在欧洲"，却分不清"巴黎"和"罗马"。
    **公里数能直接说明这一点，R² 不能。**

而且公里数**有物理意义、可写进简历、可被任何人验证**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "haversine_errors",
    "bootstrap_ci",
    "ErrorStats",
    "evaluate_predictions",
    "angular_diff_deg",
    "circular_r2",
]

EARTH_RADIUS_KM = 6371.0088  # IUGG 平均地球半径


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """球面上两点的大圆距离（公里）。

    公式::

        a = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
        d = 2R · arcsin(√a)

    其中 φ 是纬度、λ 是经度（单位：弧度）。

    为什么不用欧氏距离：见模块文档。同一个"度"误差在赤道和极地对应的实际距离差 2 倍以上。
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    # 数值安全：a 理论上 ∈ [0,1]，浮点误差可能轻微越界导致 arcsin 报 nan
    a = np.clip(a, 0.0, 1.0)
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def haversine_errors(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """向量化版本的 Haversine，返回每条的公里误差。

    Args:
        truth: 形状 ``(n, 2)``，每行 ``[lat, lon]``。
        pred: 形状 ``(n, 2)``，同 truth。

    Returns:
        形状 ``(n,)`` 的公里误差数组。
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"形状不一致: truth{truth.shape} vs pred{pred.shape}")
    if truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError(f"期望形状 (n, 2)，实际 {truth.shape}")

    phi1 = np.radians(truth[:, 0])
    phi2 = np.radians(pred[:, 0])
    dphi = np.radians(pred[:, 0] - truth[:, 0])
    dlambda = np.radians(pred[:, 1] - truth[:, 1])
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "median",
    samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """用 bootstrap 重采样估计统计量的置信区间。

    为什么不用"均值 ± 1.96 标准误"：
    公里误差的分布是**长尾**的（少数地点错得极远），均值不稳健。
    中位数更合适，但中位数的解析置信区间不好算 —— bootstrap 是最简单的通用解法。

    Args:
        values: 观测值（这里是逐条误差的公里数）。
        statistic: ``"median"`` | ``"mean"``。
        samples: 重采样次数。
        ci: 置信水平（0.95 表示取 2.5% 与 97.5% 分位）。

    Returns:
        ``(lower, upper)``
    """
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))

    reducer = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(samples), dtype=float)
    n = data.size
    for i in range(int(samples)):
        idx = rng.integers(0, n, size=n)
        estimates[i] = reducer(data[idx])

    alpha = (1.0 - float(ci)) / 2.0
    lower, upper = np.percentile(estimates, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return (float(lower), float(upper))


@dataclass
class ErrorStats:
    """一组预测的误差汇总。"""

    n: int = 0
    median_km: float = float("nan")
    mean_km: float = float("nan")
    p75_km: float = float("nan")
    p90_km: float = float("nan")
    max_km: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    r2_lat: float = float("nan")
    """纬度的 R²（单独报告，不取平均）"""
    r2_lon: float = float("nan")
    """经度的 R²（单独报告，不取平均）"""
    extra: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> Dict[str, str]:
        """格式化成报告表格的一行（全部保留 0 位小数的公里数）。"""
        return {
            "n": str(self.n),
            "median_km": _fmt_km(self.median_km),
            "mean_km": _fmt_km(self.mean_km),
            "p90_km": _fmt_km(self.p90_km),
            "95%CI": f"[{_fmt_km(self.ci_low)}, {_fmt_km(self.ci_high)}]",
            "R²(lat/lon)": f"{self.r2_lat:.2f} / {self.r2_lon:.2f}",
        }


def _fmt_km(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.0f}"


def _r2_single(truth: np.ndarray, pred: np.ndarray) -> float:
    """一维 R²。"""
    truth = np.asarray(truth, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def evaluate_predictions(
    truth: np.ndarray,
    pred: np.ndarray,
    bootstrap_samples: int = 1000,
    ci: float = 0.95,
    percentile_list: Optional[Sequence[float]] = None,
    seed: int = 42,
) -> ErrorStats:
    """计算一组预测的完整误差统计。

    同时给出：
    * **Haversine 公里误差**（中位数 / 均值 / 高分位 / 95% CI）—— 主指标
    * **分维度的 R²**（lat 与 lon 分开，不平均）—— 辅助参考

    为什么主指标是中位数：误差长尾，均值会被少数离谱的点拉高；
    中位数回答"典型情况下错多远"，更稳健。均值也一并报告，便于对比。
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    errors = haversine_errors(truth, pred)
    percentiles = list(percentile_list or [50, 75, 90])

    lower, upper = bootstrap_ci(errors, "median", bootstrap_samples, ci, seed)
    p_values = np.percentile(errors, percentiles) if errors.size else np.array([])
    p_map = {f"p{int(p)}": float(v) for p, v in zip(percentiles, p_values)}

    return ErrorStats(
        n=int(errors.size),
        median_km=float(np.median(errors)) if errors.size else float("nan"),
        mean_km=float(np.mean(errors)) if errors.size else float("nan"),
        p75_km=p_map.get("p75", float("nan")),
        p90_km=p_map.get("p90", float("nan")),
        max_km=float(np.max(errors)) if errors.size else float("nan"),
        ci_low=lower,
        ci_high=upper,
        r2_lat=_r2_single(truth[:, 0], pred[:, 0]) if errors.size else float("nan"),
        r2_lon=_r2_single(truth[:, 1], pred[:, 1]) if errors.size else float("nan"),
    )


def angular_diff_deg(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """两个经度之间的**最短带符号角距离**，落在 ``(-180, 180]``。

    ``angular_diff_deg(179, -179) = 2``（不是 -358）。
    这是经度作为**循环量**的正确距离定义。
    """
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return (truth - pred + 180.0) % 360.0 - 180.0


def circular_r2(truth_deg: np.ndarray, pred_deg: np.ndarray) -> float:
    """**经度专用的 R²**（圆形 R²）。

    为什么不能直接对经度用普通 R²
    -----------------------------
    经度是循环量。普通 R² 拿线性差值 ``truth - pred`` 算残差，于是：

    * 真实 -179°、预测 +179° —— 地理上只差 2°，普通 R² 却记成 358° 的巨大残差
    * 零模型用**算术均值** —— 但 ±180° 附近的数据算术均值没有地理意义

    本项目实测到过这个后果：同一层同一编码下，单次划分的普通 R²(经度)
    可以在 -1.01 ~ 0.85 之间乱跳，而 Haversine 公里误差是稳定的。
    所以**普通 R²(经度) 不是一个可信的指标**。

    圆形 R² 的做法
    --------------
    用最短角距离替代线性差值，零模型换成**圆形均值**（向量平均的方向）::

        R²_circ = 1 - Σ d(θ, θ̂)² / Σ d(θ, θ̄)²
        θ̄ = atan2(mean(sin θ), mean(cos θ))

    取值含义与普通 R² 一致：1 完美，0 等于"永远猜圆形均值"，负值比均值还差。
    """
    truth = np.asarray(truth_deg, dtype=float).ravel()
    pred = np.asarray(pred_deg, dtype=float).ravel()
    if truth.size == 0:
        return float("nan")

    residual = angular_diff_deg(truth, pred)
    ss_res = float(np.sum(residual**2))

    # 零模型：圆形均值
    rad = np.radians(truth)
    mean_angle = np.degrees(np.arctan2(float(np.mean(np.sin(rad))), float(np.mean(np.cos(rad)))))
    baseline = angular_diff_deg(truth, np.full_like(truth, mean_angle))
    ss_tot = float(np.sum(baseline**2))

    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def ratio_with_comment(numerator: float, denominator: float) -> tuple[float, str]:
    """算倍数并给一句人话解释（用于"歧义放大倍数"）。"""
    if denominator <= 0 or np.isnan(numerator) or np.isnan(denominator):
        return float("nan"), "—"
    ratio = numerator / denominator
    if ratio >= 2.0:
        comment = "歧义显著放大了误差"
    elif ratio >= 1.3:
        comment = "歧义轻微放大误差"
    elif ratio >= 0.8:
        comment = "歧义无明显影响"
    else:
        comment = "歧义词反而更准（值得进一步分析）"
    return ratio, comment


def list_errors(
    words: Sequence[str],
    truth: np.ndarray,
    pred: np.ndarray,
) -> List[Dict[str, object]]:
    """逐词误差明细，按误差降序（用于错误分析：哪些词错得最离谱）。"""
    errors = haversine_errors(truth, pred)
    rows = []
    for word, err, t, p in zip(words, errors, np.asarray(truth), np.asarray(pred)):
        rows.append(
            {
                "word": word,
                "error_km": float(err),
                "true_lat": float(t[0]),
                "true_lon": float(t[1]),
                "pred_lat": float(p[0]),
                "pred_lon": float(p[1]),
            }
        )
    rows.sort(key=lambda r: -float(r["error_km"]))
    return rows
