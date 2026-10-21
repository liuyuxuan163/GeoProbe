"""实验编排：训练探针 → 三组测试 → 基线对比 → 歧义偏向分析。

实验流程
--------
::

    A 组 control（108 个无歧义地名）
        │
        ├─ 随机划分 80% 训练 / 20% 留出（重复 n_repeats 次，取均值±标准差）
        │
        ├─ 用 80% 训练探针（Ridge / MLP）
        │
        ├─ 在 20% 留出集上评估  ──→ 对照误差（探针能力上限）
        │
        ├─ 在同一探针上测 B 组（27 个歧义地名）──→ 歧义误差
        │      └─ 若该词有备选坐标 → 判断"预测更接近哪个含义"
        │
        └─ 测 C 组（24 个非地名）──→ 只输出预测坐标（无真值），
                                    看落点是否退化（负对照）

为什么要重复划分
----------------
单次 `random_state=42` 的划分结果可能只是运气好。
重复 5 次不同种子，报 **中位误差的均值 ± 标准差**，
才能说清"这个数字有多稳"。

对应课程原实验的局限 6（无显著性）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .baselines import build_baseline_features
from .config import Config
from .extract import ExtractResult, MockExtractor, build_extractor, extract_dataset
from .metrics import (
    ErrorStats,
    circular_r2,
    evaluate_predictions,
    haversine_errors,
    list_errors,
    ratio_with_comment,
)
from .probe import MLPProbe, RidgeProbe, build_probe
from .targets import TargetEncoder
from .wordlists import WordList, load_wordlists

__all__ = ["ExperimentResult", "run_experiment", "alt_preference_analysis"]


@dataclass
class RepeatStats:
    """多次重复划分的统计。"""

    medians: List[float] = field(default_factory=list)
    means: List[float] = field(default_factory=list)

    @property
    def median_mean(self) -> float:
        return float(np.mean(self.medians)) if self.medians else float("nan")

    @property
    def median_std(self) -> float:
        return float(np.std(self.medians)) if self.medians else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_repeats": len(self.medians),
            "median_mean_km": self.median_mean,
            "median_std_km": self.median_std,
        }


@dataclass
class ExperimentResult:
    """完整实验的结果容器。"""

    model_name: str = ""
    mock: bool = False
    pooling: str = ""
    layer: Optional[int] = None
    n_layers: int = 0
    target_encoding: str = "latlon"
    """目标编码方式：latlon（线性）| sincos（经度循环编码）"""

    control: ErrorStats = field(default_factory=ErrorStats)
    """A 组留出集误差 —— 探针能力上限"""
    ambiguous: ErrorStats = field(default_factory=ErrorStats)
    """B 组误差 —— 歧义地名"""
    control_repeats: RepeatStats = field(default_factory=RepeatStats)
    ambiguous_repeats: RepeatStats = field(default_factory=RepeatStats)

    ambiguity_ratio: float = float("nan")
    ambiguity_comment: str = ""

    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """各基线在 A 组留出集上的中位误差（km）"""

    ambiguous_detail: List[Dict[str, Any]] = field(default_factory=list)
    """B 组逐词误差与预测坐标"""
    alt_analysis: List[Dict[str, Any]] = field(default_factory=list)
    """歧义词"更接近哪个含义"的分析"""
    alt_reliability_used: float = float("nan")
    """实际使用的可靠性门槛（km）= max(配置下限, 模型自身中位误差)"""

    pooled_r2_lat: float = float("nan")
    """**汇总** R²（纬度）：把所有重复的留出预测拼接后计算，比单次划分稳定得多"""
    pooled_r2_lon: float = float("nan")
    """**汇总** 线性 R²（经度）—— ⚠ 对经度是**无效指标**（180° 环绕），仅作对照"""
    pooled_circular_r2_lon: float = float("nan")
    """**汇总圆形 R²（经度）** —— 经度专用，正确处理环绕，应以此为准"""
    pooled_n: int = 0
    """汇总 R² 所用的样本数（= 留出集大小 × 重复次数）"""
    nongeo_predictions: List[Dict[str, Any]] = field(default_factory=list)
    """C 组预测坐标（负对照）"""

    warnings: List[str] = field(default_factory=list)

    @property
    def signal_ok(self) -> bool:
        """探针是否真的学到了东西（决定结论是否可信）。

        **判据：置换检验（permutation test）**

        把训练集的坐标随机打乱，用同样的探针去拟合同一个 X，
        再在真实测试集上评估。如果真实坐标的误差**远小于**打乱坐标的误差，
        说明探针确实从隐层里读到了与"真实地理"对应的信息。

        这比"优于预测均值基线"严谨得多 —— 后者在纯噪声特征上也能碰巧通过
        （因为随机特征过拟合后，个别点的预测会更接近真值，中位数会被拉低）。
        """
        permuted = self.baselines.get("permuted", {}).get("median_km")
        if permuted is None or np.isnan(self.control.median_km) or permuted <= 0:
            return False
        return self.control.median_km < 0.6 * permuted

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "mock": self.mock,
            "pooling": self.pooling,
            "layer": self.layer,
            "n_layers": self.n_layers,
            "target_encoding": self.target_encoding,
            "control_median_km": self.control.median_km,
            "ambiguous_median_km": self.ambiguous.median_km,
            "ambiguity_ratio": self.ambiguity_ratio,
            "ambiguity_comment": self.ambiguity_comment,
            "baselines": self.baselines,
            "signal_ok": self.signal_ok,
            "warnings": self.warnings,
        }


def alt_preference_analysis(
    entries: Sequence[Any],
    predictions: np.ndarray,
    reliability_km: float = 2500.0,
) -> List[Dict[str, Any]]:
    """对带备选坐标的歧义词，判断"预测更接近哪个含义"。

    这是本项目最有意思的分析：它回答的不是"误差多大"，
    而是"**模型把这个词放在哪了**"。

    **可靠性门槛（``reliability_km``）解决一个真实的解读陷阱**：

    第一次实验里，"20 个歧义词有 6 个更接近备选含义"听起来像结论，
    但检查后发现其中 4 个是这样：

    ==========  ==============  ===========  ==========
    词           到主含义         到备选       差值
    ==========  ==============  ===========  ==========
    Santiago     6,687 km        6,675 km     **12 km**
    Panama       7,416 km        7,276 km     140 km
    Mexico       7,015 km        6,859 km     156 km
    Georgia      7,469 km        4,607 km     2,862 km
    ==========  ==============  ===========  ==========

    **两个候选都在几千公里外时，"谁更近"只是噪声。**
    所以只有当 ``min(d_primary, d_alt) <= reliability_km`` 时，
    这个"偏向"判定才算 ``reliable``。报告与结论只统计可靠的条目。

    Returns:
        每项含 ``word`` / ``dist_primary_km`` / ``dist_alt_km`` / ``delta_km`` /
        ``min_km`` / ``closer_to`` / ``reliable`` / ``pred_lat`` / ``pred_lon``。
    """
    rows: List[Dict[str, Any]] = []
    for entry, pred in zip(entries, np.asarray(predictions, dtype=float)):
        if not entry.has_alt:
            continue
        primary = np.array([[entry.lat, entry.lon]], dtype=float)
        alt = np.array([[entry.alt_lat, entry.alt_lon]], dtype=float)
        pred_row = pred.reshape(1, 2)
        d_primary = float(haversine_errors(primary, pred_row)[0])
        d_alt = float(haversine_errors(alt, pred_row)[0])
        nearest = min(d_primary, d_alt)
        rows.append(
            {
                "word": entry.word,
                "label": entry.label,
                "alt_label": entry.alt_label,
                "dist_primary_km": d_primary,
                "dist_alt_km": d_alt,
                "delta_km": abs(d_primary - d_alt),
                "min_km": nearest,
                "closer_to": "primary" if d_primary <= d_alt else "alt",
                # ★ 只有"最近的那个候选本来就够近"时，偏向判定才有信息量
                "reliable": bool(nearest <= float(reliability_km)),
                "pred_lat": float(pred[0]),
                "pred_lon": float(pred[1]),
            }
        )
    return rows


def _r2_dim(truth: np.ndarray, pred: np.ndarray) -> float:
    """一维 R²。"""
    truth = np.asarray(truth, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _split_train_test(
    n: int, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * float(test_size))))
    return indices[n_test:], indices[:n_test]


def run_experiment(
    config: Config,
    lists: Optional[Dict[str, WordList]] = None,
    extract_results: Optional[Dict[str, ExtractResult]] = None,
    layer: Optional[int] = None,
    probe_kinds: Sequence[str] = ("ridge",),
    n_repeats: int = 5,
    test_size: float = 0.2,
    run_baselines: bool = True,
) -> ExperimentResult:
    """跑完整实验。

    Args:
        layer: 用第几层的隐层；``None`` 表示最后一层。
        probe_kinds: 要跑哪些探针，默认只跑 Ridge（MLP 较慢，按需开启）。
        n_repeats: 重复划分次数（用于估计稳定性）。
        test_size: A 组留出比例。
    """
    lists = lists or load_wordlists(config.path_of("data"))
    control = lists["control"]
    ambiguous = lists["ambiguous"]
    nongeo = lists["nongeo"]

    if extract_results is None:
        extract_results = extract_dataset(config, lists)

    control_extract = extract_results["control"]
    ambiguous_extract = extract_results["ambiguous"]
    nongeo_extract = extract_results["nongeo"]

    actual_layer = control_extract.layers[-1] if layer is None else int(layer)
    X_control = control_extract.matrix(actual_layer)
    X_ambiguous = ambiguous_extract.matrix(actual_layer)
    Y_control = control.coords()
    Y_ambiguous = ambiguous.coords()

    # ★ 目标编码：经度是循环量，线性编码对它有偏。
    #   sincos 模式把经度放到单位圆上（lat, sin λ, cos λ），
    #   从而能区分"经度学不好"到底是模型的限制还是编码的缺陷。
    encoder = TargetEncoder(str(config.get("probe.target_encoding", "latlon")))
    Y_control_enc = encoder.encode(Y_control)
    Y_ambiguous_enc = encoder.encode(Y_ambiguous)
    reliability_floor = float(config.get("evaluation.alt_reliability_km", 2500.0))

    result = ExperimentResult(
        model_name=control_extract.model_name,
        mock=control_extract.mock,
        pooling=control_extract.pooling,
        layer=actual_layer,
        n_layers=len(control_extract.layers),
        target_encoding=encoder.mode,
    )

    primary_kind = probe_kinds[0]
    control_repeats = RepeatStats()
    ambiguous_repeats = RepeatStats()
    all_control_preds: List[np.ndarray] = []
    all_control_truth: List[np.ndarray] = []
    ambiguous_preds: List[np.ndarray] = []
    last_probe = None

    for repeat in range(max(1, int(n_repeats))):
        seed = int(config.get("probe.random_state", 42)) + repeat
        train_idx, test_idx = _split_train_test(len(X_control), test_size, seed)

        probe = build_probe(config, primary_kind)
        # 在**编码空间**里拟合
        probe.fit(X_control[train_idx], Y_control_enc[train_idx])

        # --- A 组留出集：先解码回经纬度，再用 Haversine 评估 ---
        pred_control = encoder.decode(probe.predict(X_control[test_idx]))
        stats_control = evaluate_predictions(
            Y_control[test_idx],
            pred_control,
            bootstrap_samples=int(config.get("evaluation.bootstrap_samples", 1000)),
            ci=float(config.get("evaluation.ci", 0.95)),
            percentile_list=config.get("evaluation.percentiles", [50, 75, 90]),
            seed=seed,
        )
        control_repeats.medians.append(stats_control.median_km)
        control_repeats.means.append(stats_control.mean_km)
        all_control_preds.append(pred_control)
        all_control_truth.append(Y_control[test_idx])

        # --- B 组（用同一个探针）---
        pred_ambiguous = encoder.decode(probe.predict(X_ambiguous))
        stats_ambiguous = evaluate_predictions(
            Y_ambiguous,
            pred_ambiguous,
            bootstrap_samples=int(config.get("evaluation.bootstrap_samples", 1000)),
            ci=float(config.get("evaluation.ci", 0.95)),
            percentile_list=config.get("evaluation.percentiles", [50, 75, 90]),
            seed=seed,
        )
        ambiguous_repeats.medians.append(stats_ambiguous.median_km)
        ambiguous_repeats.means.append(stats_ambiguous.mean_km)
        ambiguous_preds.append(pred_ambiguous)
        last_probe = probe

    # 用最后一次重复的完整统计作为代表值（重复统计另外给出）
    result.control = stats_control
    result.ambiguous = stats_ambiguous
    result.control_repeats = control_repeats
    result.ambiguous_repeats = ambiguous_repeats

    # ★ 汇总 R²：把所有重复的留出预测**拼起来**再算。
    #
    # 为什么不能只用最后一次重复的 R²：
    # 每次留出集只有 22 个词，而 R² 在这个量级上**极不稳定** ——
    # 同一层不同划分会给出 0.06 ~ 0.67 的巨大差异（实测过的）。
    # 拼起来后样本量变成 22×n_repeats，R² 才有可比性。
    if all_control_preds:
        pooled_pred = np.vstack(all_control_preds)
        pooled_truth = np.vstack(all_control_truth)
        result.pooled_r2_lat = _r2_dim(pooled_truth[:, 0], pooled_pred[:, 0])
        result.pooled_r2_lon = _r2_dim(pooled_truth[:, 1], pooled_pred[:, 1])
        # ★ 经度专用：圆形 R²（正确处理 180° 环绕）
        result.pooled_circular_r2_lon = circular_r2(pooled_truth[:, 1], pooled_pred[:, 1])
        result.pooled_n = int(len(pooled_pred))

    result.ambiguity_ratio, result.ambiguity_comment = ratio_with_comment(
        ambiguous_repeats.median_mean, control_repeats.median_mean
    )

    # --- 用平均预测做逐词分析（多重复平均更稳）---
    mean_ambiguous_pred = np.mean(np.stack(ambiguous_preds), axis=0)
    result.ambiguous_detail = list_errors(ambiguous.words, Y_ambiguous, mean_ambiguous_pred)

    # 「偏向判定」的**自适应**可靠性门槛：
    #   有效门槛 = max(配置的绝对下限, 模型自身在 A 组的典型误差)
    #
    # 为什么不能只用固定门槛：如果门槛（如 2500 km）比模型自身的中位误差（如 3103 km）
    # 还严，就会把**模型定位正常的词**也判成不可靠 —— 那等于什么都没分析。
    # 反过来，判据的自然含义是："最近候选至少要和模型平时一样准，才谈得上'偏向'。"
    result.alt_reliability_used = max(reliability_floor, float(control_repeats.median_mean))
    result.alt_analysis = alt_preference_analysis(
        ambiguous.entries, mean_ambiguous_pred, reliability_km=result.alt_reliability_used
    )

    # --- C 组负对照（无真值，只记录预测坐标的离散度）---
    pred_nongeo = last_probe.predict(nongeo_extract.matrix(actual_layer)) if last_probe else np.zeros((0, 2))
    result.nongeo_predictions = [
        {"word": w, "pred_lat": float(p[0]), "pred_lon": float(p[1])}
        for w, p in zip(nongeo.words, pred_nongeo)
    ]

    # --- 基线 ---
    if run_baselines:
        result.baselines = _run_baselines(config, lists, X_control, Y_control, test_size, encoder)
        result.baselines.setdefault("hidden", {})["median_km"] = control_repeats.median_mean
        result.baselines["hidden"]["median_std_km"] = control_repeats.median_std

    # --- 备选探针（MLP 对照）---
    if "mlp" in probe_kinds:
        result.baselines["mlp"] = _run_single_probe(config, "mlp", X_control, Y_control, test_size, encoder)

    # --- 警告 ---
    if extract_results is None:  # pragma: no cover - 防御
        result.warnings.append("未提供 extract_results，内部已重新提取")
    if result.mock:
        # Mock 的合成向量是"经度线性"的，所以 latlon 目标天然更贴合它。
        # 不警告的话，读者会把 Mock 下 sincos 更差误读为"sin/cos 编码没用"。
        result.warnings.append(
            "⚠ 当前是 Mock 模式（合成向量），其隐层是**经度线性**构造的，"
            "因此 {latlon} 目标天然更贴合它 —— "
            "**不要用 Mock 结果比较 target_encoding**。"
            "这个对比必须用真实模型跑。"
        )
    if not result.signal_ok:
        result.warnings.append(
            "⚠ 探针未通过置换检验（真实坐标的误差未能显著低于打乱坐标）—— "
            "说明隐层里可能没有可线性解码的地理信息（或层选得不对）。"
            "此时 B 组的结果不应被解读为「歧义效应」。"
        )
    if actual_layer < 0:
        result.warnings.append("层号异常")

    return result


def _run_baselines(
    config: Config,
    lists: Dict[str, WordList],
    X_control: np.ndarray,
    Y_control: np.ndarray,
    test_size: float,
    encoder: Optional[TargetEncoder] = None,
) -> Dict[str, Dict[str, float]]:
    """在 A 组留出集上跑各基线，返回中位误差。

    ``encoder`` 必须与主实验一致 —— 否则基线在"线性目标"上拟合、
    主探针在"sincos 目标"上拟合，两者不可比。
    """
    encoder = encoder or TargetEncoder("latlon")
    Y_enc = encoder.encode(Y_control)
    out: Dict[str, Dict[str, float]] = {}
    n_repeats = max(1, int(config.get("baselines.n_repeats", 5) or 5))

    features = build_baseline_features(config, lists, hidden_dim=X_control.shape[1])

    # 均值基线
    if bool(config.get("baselines.mean", True)):
        medians = []
        for repeat in range(n_repeats):
            seed = int(config.get("probe.random_state", 42)) + repeat
            train_idx, test_idx = _split_train_test(len(X_control), test_size, seed)
            probe = build_probe(config, "mean")
            probe.fit(X_control[train_idx], Y_enc[train_idx])
            pred = encoder.decode(probe.predict(X_control[test_idx]))
            stats = evaluate_predictions(Y_control[test_idx], pred, bootstrap_samples=200, seed=seed)
            medians.append(stats.median_km)
        out["mean"] = {"median_km": float(np.mean(medians)), "median_std_km": float(np.std(medians))}

    # 置换基线：打乱训练集的坐标，用同样的 X 拟合，再在真实测试集上评估
    if bool(config.get("baselines.permuted", True)):
        permuted_medians = []
        for repeat in range(n_repeats):
            seed = int(config.get("probe.random_state", 42)) + repeat
            train_idx, test_idx = _split_train_test(len(X_control), test_size, seed)
            rng = np.random.default_rng(seed + 1000)
            shuffled = Y_enc[train_idx].copy()
            rng.shuffle(shuffled, axis=0)
            probe = build_probe(config, "ridge")
            probe.fit(X_control[train_idx], shuffled)
            pred = encoder.decode(probe.predict(X_control[test_idx]))
            stats = evaluate_predictions(Y_control[test_idx], pred, bootstrap_samples=200, seed=seed)
            permuted_medians.append(stats.median_km)
        out["permuted"] = {
            "median_km": float(np.mean(permuted_medians)),
            "median_std_km": float(np.std(permuted_medians)),
        }

    # 随机 / 字符 n-gram 基线
    for kind in ("random", "char_ngram"):
        if kind not in features:
            continue
        X_base = features[kind]["control"]
        medians = []
        for repeat in range(n_repeats):
            seed = int(config.get("probe.random_state", 42)) + repeat
            train_idx, test_idx = _split_train_test(len(X_base), test_size, seed)
            probe = build_probe(config, "ridge")
            probe.fit(X_base[train_idx], Y_enc[train_idx])
            pred = encoder.decode(probe.predict(X_base[test_idx]))
            stats = evaluate_predictions(
                Y_control[test_idx], pred, bootstrap_samples=200, seed=seed
            )
            medians.append(stats.median_km)
        out[kind] = {"median_km": float(np.mean(medians)), "median_std_km": float(np.std(medians))}

    return out


def _run_single_probe(
    config: Config,
    kind: str,
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float,
    encoder: Optional[TargetEncoder] = None,
    n_repeats: int = 3,
) -> Dict[str, float]:
    encoder = encoder or TargetEncoder("latlon")
    Y_enc = encoder.encode(Y)
    medians = []
    for repeat in range(n_repeats):
        seed = int(config.get("probe.random_state", 42)) + repeat
        train_idx, test_idx = _split_train_test(len(X), test_size, seed)
        probe = build_probe(config, kind)
        probe.fit(X[train_idx], Y_enc[train_idx])
        pred = encoder.decode(probe.predict(X[test_idx]))
        stats = evaluate_predictions(Y[test_idx], pred, bootstrap_samples=200, seed=seed)
        medians.append(stats.median_km)
    return {"median_km": float(np.mean(medians)), "median_std_km": float(np.std(medians))}


def save_result(result: ExperimentResult, path: str | Path) -> Path:
    """把结果写成 JSON（便于对比不同配置）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result.summary_dict(),
        "control": {
            "n": result.control.n,
            "median_km": result.control.median_km,
            "mean_km": result.control.mean_km,
            "p90_km": result.control.p90_km,
            "ci": [result.control.ci_low, result.control.ci_high],
            # ⚠ 以下是**单次划分**的 R²，抽样噪声极大，不要用它下结论。
            #   结论请用下面的 "pooled" 块。
            "single_split_r2_lat": result.control.r2_lat,
            "single_split_r2_lon": result.control.r2_lon,
        },
        "ambiguous": {
            "n": result.ambiguous.n,
            "median_km": result.ambiguous.median_km,
            "mean_km": result.ambiguous.mean_km,
            "p90_km": result.ambiguous.p90_km,
            "ci": [result.ambiguous.ci_low, result.ambiguous.ci_high],
        },
        "repeats": {
            "control": result.control_repeats.to_dict(),
            "ambiguous": result.ambiguous_repeats.to_dict(),
        },
        # ★ 汇总 R² 单独成块：单次划分的 R² 抽样噪声可达 ±0.5，不能用于结论。
        #   经度一律用圆形 R²（circular_r2_lon），线性值只留作对照。
        "pooled": {
            "n": result.pooled_n,
            "r2_lat": result.pooled_r2_lat,
            "circular_r2_lon": result.pooled_circular_r2_lon,
            "linear_r2_lon_deprecated": result.pooled_r2_lon,
            "note": "r2_lat / circular_r2_lon 为汇总值（所有重复的留出预测拼接），"
            "比单次划分稳定。线性 R²(经度) 对循环量无效，仅作对照。",
        },
        "ambiguous_detail": result.ambiguous_detail,
        "alt_analysis": result.alt_analysis,
        "alt_reliability_used": result.alt_reliability_used,
        "nongeo_predictions": result.nongeo_predictions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
