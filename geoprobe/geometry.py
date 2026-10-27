"""表征几何：残差流范数、层间旋转、有效维度。

**这个模块要回答的问题**

logit lens 失效之后（见 `lens.py`），"为什么深层的地理可解码性会下降"仍然没有答案。
本模块用**不需要 lm_head** 的三个几何量来回答它 —— 而且**只需要缓存的激活**，
不必重新加载模型。

三个量分别对应三种可能的机制
============================

1. **残差流范数** ``‖h_l‖``
   如果范数随深度急剧增长，那么隐层向量的**方向**信息虽然还在，
   但两个词的差异会被共同的巨大分量淹没 —— 线性探针需要从这个巨大分量里
   减去均值才能看到差异，残余的数值精度变差。
   → 预测：范数增长与探针误差正相关。

2. **层间旋转** ``cos(h_l, h_{l+1})``
   如果相邻层的方向变化很快（余弦接近 0），说明这一层把表征"转向"了别处。
   → 预测：误差上升的层段对应低余弦（大转角）。

3. **相对「最佳层」的累计漂移** ``1 - cos(h_l, h_best)``
   直接量化"表征离那个能读出地理信息的状态有多远"。
   → 预测：这条曲线应当与探针误差曲线形状一致。这是三者中最直接的检验。

4. **有效维度（参与比）**
   ``(Σλ)² / Σλ²``，λ 为协方差特征值。它回答"这些向量实际用到了多少维"。
   如果有效维度很低（比如 2048 维里只有 20 维在变化），
   那么**线性探针能利用的信息量本身就很有限** ——
   这正好是"规模效应弱"的一个可能原因（见文档讨论）。

为什么这些量能从缓存算出来
==========================
它们都只依赖隐层向量本身，不需要词表矩阵。所以三个模型的 npz 就够用，
几分钟能跑完，也不受 logit lens 近似失效的影响。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "LayerGeometry",
    "residual_norm",
    "centered_norm",
    "specificity",
    "inter_layer_cosine",
    "drift_from_layer",
    "effective_dimension",
    "norm_dispersion",
    "analyze_geometry",
    "pearson",
]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def residual_norm(matrix: np.ndarray) -> float:
    """这一层隐层向量的平均 L2 范数（**含**公共分量）。"""
    return float(np.mean(np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)))


def centered_norm(matrix: np.ndarray) -> float:
    """去掉均值后的平均范数 —— 也就是**只属于各个词**的那部分信号。

    为什么必须区分它
    ----------------
    实测 Qwen2.5-0.5B 的残差流范数从 layer 0 的 6.8 涨到 layer 5 的 315，
    而同时"有效维度"降到 1.0 —— 意思是 108 个词的隐层向量**几乎共线**：
    存在一个 ~315 量级的公共分量，词与词之间的差异只是它上面的微小扰动。

    这时用总范数解释"可解码性"会被公共分量带偏。真正相关的是
    **词间差异的大小**，即本函数算的量。
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[0] == 0:
        return float("nan")
    centered = values - values.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(centered, axis=1)))


def specificity(matrix: np.ndarray) -> float:
    """``centered_norm / residual_norm``：残差流里"能区分词"的那部分占比。

    * 接近 1：表征主要由词间差异构成，地理信号"显眼"，线性探针好读
    * 接近 0：词间差异被公共分量淹没，探针要在巨大背景上找极小的方向 ——
      在只有 86 个训练样本的情况下，Ridge 的正则会把这个方向压掉
    """
    total = residual_norm(matrix)
    if not np.isfinite(total) or total <= 1e-12:
        return float("nan")
    return centered_norm(matrix) / total


def inter_layer_cosine(layer_a: np.ndarray, layer_b: np.ndarray) -> float:
    """相邻两层隐层的平均余弦相似度（对词取平均）。

    值接近 1 表示方向几乎不变；接近 0 表示层与层之间几乎是正交的（大转角）。
    **注意**：不同层的隐层处在不同的基里，余弦只能当"相对变化"的粗指标，
    不能解读为"语义相似度"。
    """
    a = _normalize_rows(np.asarray(layer_a, dtype=np.float64))
    b = _normalize_rows(np.asarray(layer_b, dtype=np.float64))
    return float(np.mean(np.sum(a * b, axis=1)))


def drift_from_layer(reference: np.ndarray, other: np.ndarray) -> float:
    """``1 - cos(reference, other)``：相对某层的累计漂移。

    用**最佳层**作参照时，这个量直接回答"这个层离能读出地理信息的状态有多远"。
    """
    return 1.0 - inter_layer_cosine(reference, other)


def effective_dimension(matrix: np.ndarray, normalize: bool = False) -> float:
    """有效维度（参与比 participation ratio）。

    ``PR = (Σλ)² / Σλ²``，λ 是协方差矩阵的特征值。

    * 全部方差集中在一个方向 → PR = 1
    * 方差均匀分布在 k 个方向 → PR = k

    ⚠ **必须区分是否先做 L2 归一化**

    实测 Qwen2.5 的隐层范数在词间相差可达 **200 倍**（layer 3：8.3 到 1735）。
    这时协方差的最大特征值几乎完全对应"范数大小的差异"，
    于是**未归一化**的 PR 会一律趋近 1 —— 那不是"表征只有一维"，
    而是"范数差异吃掉了所有方差"。

    所以：

    * ``normalize=False`` → 反映"总方差"的结构，会被范数差异主导，参考价值低
    * ``normalize=True``  → 先归一化再算，才是"词间差异用了多少维"，
      是用来说明"表征自由度是否随规模增长"的正确量
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[0] < 2:
        return float("nan")
    if normalize:
        values = _normalize_rows(values)
    centered = values - values.mean(axis=0, keepdims=True)
    # 用 SVD 代替协方差特征值分解：数值更稳，且样本数 << 维度时更快
    singular = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular**2
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return float("nan")
    return float(total**2 / float(np.sum(eigenvalues**2)))


def norm_dispersion(matrix: np.ndarray) -> float:
    """词间范数的离散程度：``max / median``。

    实测值可达 200 倍 —— 说明"不同词的隐层向量量级差异极大"。
    这是一个真实表征性质；但**不要**据此认为它是缺陷：
    本项目实测过"先 L2 归一化再探针"，结果 0.5B **变差 2–17%**，
    说明**幅度本身携带信息**（常见词的迹更"干净"）。
    """
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)
    median = float(np.median(norms))
    if median <= 1e-12:
        return float("nan")
    return float(np.max(norms) / median)


def pearson(xs: List[float], ys: List[float]) -> float:
    """皮尔逊相关（只保留有限值）。"""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denominator)


@dataclass
class LayerGeometry:
    """单层的几何指标。"""

    layer: int
    residual_norm: float = float("nan")
    """‖h‖ 的平均值（含公共分量）"""
    centered_norm: float = float("nan")
    """去掉均值后的平均范数 —— 只属于词的那部分信号"""
    specificity: float = float("nan")
    """``centered_norm / residual_norm``，残差流里能区分词的占比"""
    inter_layer_cosine: float = float("nan")
    """与上一层的平均余弦；第 0 层为 NaN"""
    drift_from_best: float = float("nan")
    """1 - cos(h_l, h_best)；需要先知道最佳层"""
    effective_dimension: float = float("nan")
    """参与比（**未归一化**）：会被范数差异主导，通常趋近 1，参考价值低"""
    effective_dimension_normed: float = float("nan")
    """参与比（**先 L2 归一化**）：才是"词间差异用了多少维"，用来说明表征自由度"""
    norm_dispersion: float = float("nan")
    """词间范数的 ``max/median``：量级差异有多大"""
    probe_error_km: Optional[float] = None
    """该层的探针误差（若已有 encoding_comparison 结果）"""


def analyze_geometry(
    embeddings: Dict[int, np.ndarray],
    best_layer: Optional[int] = None,
    probe_errors: Optional[Dict[int, float]] = None,
) -> List[LayerGeometry]:
    """对每一层算几何指标。

    Args:
        embeddings: ``{layer: (n_words, hidden_dim)}``，直接用缓存里的激活。
        best_layer: 参照层（通常取探针误差最低的那层）。不给就用最后一层。
        probe_errors: ``{layer: 中位误差 km}``，用于后面算相关性。
    """
    layers = sorted(embeddings)
    if not layers:
        return []
    if best_layer is None or best_layer not in embeddings:
        best_layer = layers[-1]
    reference = embeddings[best_layer]

    results: List[LayerGeometry] = []
    previous: Optional[np.ndarray] = None
    for layer in layers:
        matrix = embeddings[layer]
        entry = LayerGeometry(
            layer=layer,
            residual_norm=residual_norm(matrix),
            centered_norm=centered_norm(matrix),
            specificity=specificity(matrix),
            effective_dimension=effective_dimension(matrix),
            effective_dimension_normed=effective_dimension(matrix, normalize=True),
            norm_dispersion=norm_dispersion(matrix),
            drift_from_best=drift_from_layer(reference, matrix),
            probe_error_km=(probe_errors or {}).get(layer),
        )
        if previous is not None:
            entry.inter_layer_cosine = inter_layer_cosine(previous, matrix)
        results.append(entry)
        previous = matrix
    return results
