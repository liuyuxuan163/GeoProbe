"""目标编码：把「经纬度」变成探针要回归的目标，以及反向还原。

**为什么需要这一层（这是本项目发现的一个真实建模缺陷）**
----------------------------------------------------
第一次真实模型实验里，分维度 R² 出现强烈不对称：

    R²(纬度) = 0.427
    R²(经度) = 0.064      ← 几乎不可解码

一开始的解释是"模型不编码经度"。但检查预测值后发现，
**所有预测经度都被压缩到 -37° ~ +24° 的窄区间**，而真实跨度是 -103° ~ +43°。
这是典型的"回归到均值"。

进一步追下去，发现有一个**真实的方法缺陷**：

    经度是**循环量**。179° 和 -179° 在地理上只差 2°（相邻），
    但在线性目标空间里相差 358°。线性回归无法表达这种周期性。

于是只要地名跨过国际日期变更线，线性模型就必然给出巨大误差 ——
**这会让"经度不可解码"这个结论失效**，因为分不清是模型的限制还是编码的缺陷。

**正确做法**：把经度编码到单位圆上::

    latlon 模式（线性）:  y = [lat,  lon]                   维度 2
    sincos 模式（循环）:  y = [lat,  sin(λ), cos(λ)]         维度 3

在 sincos 空间里，179° → (0.0175, -0.9998)，-179° → (-0.0175, -0.9998)，
两者欧氏距离只有 0.035 —— **相邻关系被正确表达**。

预测时用 ``atan2(sin, cos)`` 还原经度（自动落在 [-180°, 180°]）。

用法::

    encoder = TargetEncoder("sincos")
    y = encoder.encode(coords)          # (n,2) → (n,3)
    lonlat = encoder.decode(predictions) # (n,3) → (n,2)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["TargetEncoder", "ENCODING_MODES"]

ENCODING_MODES = ("latlon", "sincos")

# 纬度的物理范围（用于把预测值 clip 回来）
LAT_MIN, LAT_MAX = -90.0, 90.0


class TargetEncoder:
    """在「地理坐标」与「探针目标」之间双向转换。

    Args:
        mode: ``"latlon"``（线性，向后兼容）或 ``"sincos"``（经度循环编码，推荐）。

    Attributes:
        dim: 编码后的目标维度（latlon=2，sincos=3）。
    """

    def __init__(self, mode: str = "latlon") -> None:
        if mode not in ENCODING_MODES:
            raise ValueError(f"未知的目标编码方式: {mode!r}（可选 {ENCODING_MODES}）")
        self.mode = mode

    @property
    def dim(self) -> int:
        return 3 if self.mode == "sincos" else 2

    # ------------------------------------------------------------------ #
    def encode(self, coords: np.ndarray) -> np.ndarray:
        """``(n, 2)`` 的 ``[lat, lon]`` → 探针目标。

        sincos 模式下经度被映射到单位圆上的 ``(sin λ, cos λ)``，
        从而让「跨日界线」的点在目标空间里也是相邻的。
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"期望形状 (n, 2)，实际 {coords.shape}")
        if self.mode == "latlon":
            return coords.copy()

        lat = coords[:, 0]
        lam = np.radians(coords[:, 1])
        return np.stack([lat, np.sin(lam), np.cos(lam)], axis=1)

    def decode(self, predictions: np.ndarray) -> np.ndarray:
        """探针输出 → ``(n, 2)`` 的 ``[lat, lon]``。

        纬度会被裁到 ``[-90, 90]``（回归可能外推）。
        经度用 ``atan2`` 还原，结果自动在 ``[-180, 180]``。
        """
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] != self.dim:
            raise ValueError(f"期望形状 (n, {self.dim})，实际 {predictions.shape}")

        lat = np.clip(predictions[:, 0], LAT_MIN, LAT_MAX)
        if self.mode == "latlon":
            lon = predictions[:, 1]
            # 线性模式也可能外推出界，这里做一次环绕（比裁到 ±180 更符合地理语义）
            lon = ((lon + 180.0) % 360.0) - 180.0
            return np.stack([lat, lon], axis=1)

        sin_part = predictions[:, 1]
        cos_part = predictions[:, 2]
        lon = np.degrees(np.arctan2(sin_part, cos_part))
        return np.stack([lat, lon], axis=1)

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "dim": self.dim,
            "note": (
                "经度线性编码（对循环量有偏，仅用于与旧结果对比）"
                if self.mode == "latlon"
                else "经度编码为 (sin, cos)，正确处理 180° 环绕"
            ),
        }


def angular_distance_deg(lon_a: float, lon_b: float) -> float:
    """两个经度之间的**最短角距离**（考虑环绕）。

    用于诊断：线性编码下 ``|179 - (-179)| = 358°``，
    但真实距离是 ``2°``。这个函数给出后者。
    """
    diff = abs(float(lon_a) - float(lon_b)) % 360.0
    return 360.0 - diff if diff > 180.0 else diff
