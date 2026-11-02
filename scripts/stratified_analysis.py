"""分层统计：按词类（城市/国家/其他）**和人口分档**看三个模型的表现。

**为什么需要这个脚本**

之前的结论是「参数 6 倍只让误差降 19%，规模效应很弱」。但词表里混着两种
真值约定：

* **城市** —— 真值是市中心，几乎无歧义（±20 km）
* **国家** —— 真值是面积质心，而"一个国家对应哪个点"本身有 ±1,000 km 级歧义
  （巴西的质心在亚马逊，最大城市却是圣保罗，首都又是巴西利亚）

国家条目的固有歧义构成一个**噪声地板**，把真实的规模效应盖住了。
本脚本把两者拆开算，回答："规模效应到底有多强？"

**人口分档（2026-10-30 新增，交接文档 P0）**

"城市/国家"这一层拆分解决的是**真值约定**不同；而"决定精度的首要因素是
地名知名度"这个结论需要另一种拆分：**在真值约定相同（都是城市市中心）的
样本内部**，按人口（知名度的代理）再切几档。

这样做的关键好处：**排除了类别混淆**。如果只看"城市 vs 国家"，
国家更差既可以解释为"质心歧义"，也可以解释为"国家更大/更抽象"；
在城市内部按人口分层，两档的真值约定完全一致，差多少就只能归因于知名度。

口径（脚本里固定，报告里也写出来）：

* 只有 ``population`` 有值的条目参与（本项目**仅 280 个城市**，国家/地区没有）
* 按人口**降序排名**后**等词数**切档（默认 4 档），Q1 = 人口最多 = 最知名
* 为什么等词数而不是人口等距：人口跨 5 个数量级，人口等距会把 99% 的词
  塞进最低档；等词数让每档的中位数误差精度一致，档间才能直接比
* 为什么在排名上切而不是取分位点边界：人口重尾，边界附近并列值会让档失衡

**方法（逐词误差怎么来的）**

对每个模型、在每个划分上：

1. 80% 训练 / 20% 留出
2. 记录**留出词**的逐词 Haversine 误差
3. 跑多个种子，让每个词被测试多次（约 20% × 种子数 次）
4. 按词类 / 人口档聚合，取中位数

这样每个词都有多个独立测量的误差均值，比单次划分稳得多。

**类别从哪来**

优先读 ``data/words_control.json`` 里的 ``category`` 字段；
没有的话从 ``note`` 推断。人口读 ``population`` 字段。

用法::

    python scripts/stratified_analysis.py
    python scripts/stratified_analysis.py --population-bins 5
    python scripts/stratified_analysis.py --repeats 20 --out docs/分层统计.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoprobe.config import load_config  # noqa: E402
from geoprobe.experiment import _split_train_test  # noqa: E402
from geoprobe.extract import ExtractResult  # noqa: E402
from geoprobe.metrics import haversine_errors  # noqa: E402
from geoprobe.probe import build_probe  # noqa: E402
from geoprobe.targets import TargetEncoder  # noqa: E402
from geoprobe.wordlists import load_wordlists  # noqa: E402

GROUPS = ("control", "ambiguous", "nongeo")

MIN_BUCKET_WORDS = 10
"""桶内词数少于此值时，**不报倍数与规模效应**。

真实教训：319 词里「其他」只有 2 个（Scotland / Wales），却在规模效应表里
算出 −67.1% 的"改善"—— 两个词的中位数只差一次划分的运气。这类数字一旦
被引用就无法解释，所以宁可不报。（和交接文档 §7.5「不要写绝对判断」同源。）
"""


# --------------------------------------------------------------------------- #
# 类别标注
# --------------------------------------------------------------------------- #


def load_categories(path: Path) -> Dict[str, str]:
    """读取 A 组每个词的类别。

    优先用 ``category`` 字段；否则从 ``note`` 里识别中文字样。
    识别不出来就标 ``"其他"`` —— **不猜**，因为猜错会污染分层统计。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    categories: Dict[str, str] = {}
    for entry in raw:
        word = entry.get("word")
        if not word:
            continue
        explicit = entry.get("category")
        if explicit:
            categories[word] = str(explicit)
            continue
        note = str(entry.get("note", ""))
        if "城市" in note:
            categories[word] = "城市"
        elif "国家" in note:
            categories[word] = "国家"
        else:
            categories[word] = "其他"
    return categories


# --------------------------------------------------------------------------- #
# 人口分档（知名度的代理）
# --------------------------------------------------------------------------- #


def load_population(path: Path) -> Dict[str, float]:
    """读取 A 组每个词的 ``population``。

    实测：本项目**只有 280 个城市**有这个字段，37 个国家和 2 个地区没有。
    所以人口分档天然只在城市内部进行 —— 这恰好是它的优点：真值约定
    （市中心）完全一致，档间差异只能归因于知名度。

    非法值（缺失 / 空串 / 非数字 / ≤0）一律跳过，**不填默认值** ——
    填 0 会把"没有数据"伪装成"人口最少"，直接污染最低档。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for entry in raw:
        word = entry.get("word")
        value = entry.get("population")
        if not word or value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0:
            out[str(word)] = number
    return out


def _average_rank(values: np.ndarray) -> np.ndarray:
    """秩（从小到大），并列取**平均秩**。Spearman 里必需，否则并列会算错。"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denominator)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 秩相关（不依赖 scipy）。

    这里算 ``rho(log10 人口, 逐词误差)``：**负数 = 越知名越准**。
    用秩而不是线性相关，是因为人口跨 5 个数量级、误差重尾，
    线性相关系数会被几个极端值带偏。
    """
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return float("nan")
    return _pearson(_average_rank(xs[mask]), _average_rank(ys[mask]))


def summarise_population(
    words: Sequence[str],
    population: Dict[str, float],
    per_word: Dict[str, float],
    n_bins: int = 4,
) -> Dict[str, Any]:
    """把逐词误差按人口分档汇总。

    Args:
        words: A 组全部词（顺序无关）。
        population: ``load_population`` 的结果。
        per_word: ``{word: 平均留出误差 km}``，来自 ``per_word_errors``。
        n_bins: 切几档（默认 4）。

    Returns:
        ``{"n_bins", "n_with_population", "n_without_population",
        "quantiles": [每档...], "spearman": {...}, "monotone": bool}``

        每档含 ``label / n / population_min / population_median /
        population_max / median_km / mean_km / ratio_to_q1 / words``。
    """
    n_bins = max(1, int(n_bins))
    eligible = [w for w in words if w in population]
    # 人口降序：Q1 = 最知名
    ranked = sorted(eligible, key=lambda w: (-population[w], w))
    n = len(ranked)

    # 等词数切档（余数分给前面几档，保证大小最多差 1）
    base, extra = divmod(n, n_bins) if n else (0, 0)
    quantiles: List[Dict[str, Any]] = []
    cursor = 0
    q1_median = float("nan")
    for index in range(n_bins):
        size = base + (1 if index < extra else 0)
        chunk = ranked[cursor : cursor + size]
        cursor += size
        if not chunk:
            quantiles.append(
                {
                    "label": f"Q{index + 1}",
                    "n": 0,
                    "population_min": float("nan"),
                    "population_median": float("nan"),
                    "population_max": float("nan"),
                    "median_km": float("nan"),
                    "mean_km": float("nan"),
                    "ratio_to_q1": float("nan"),
                    "words": [],
                }
            )
            continue
        pops = [population[w] for w in chunk]
        errors = [per_word.get(w, float("nan")) for w in chunk]
        clean = [e for e in errors if np.isfinite(e)]
        median_km = float(np.median(clean)) if clean else float("nan")
        if index == 0:
            q1_median = median_km
        ratio = median_km / q1_median if np.isfinite(q1_median) and q1_median > 0 else float("nan")
        quantiles.append(
            {
                "label": f"Q{index + 1}",
                "n": len(chunk),
                "population_min": float(min(pops)),
                "population_median": float(np.median(pops)),
                "population_max": float(max(pops)),
                "median_km": median_km,
                "mean_km": float(np.mean(clean)) if clean else float("nan"),
                "ratio_to_q1": ratio,
                "words": chunk,
            }
        )

    pops_all = [population[w] for w in ranked]
    errs_all = [per_word.get(w, float("nan")) for w in ranked]
    finite_pairs = [(p, e) for p, e in zip(pops_all, errs_all) if np.isfinite(e)]
    rho = spearman(
        [np.log10(p) for p, _ in finite_pairs],
        [e for _, e in finite_pairs],
    )

    # 误差随「知名度下降」是否单调上升（用来判断"知名度效应"稳不稳）
    medians = [q["median_km"] for q in quantiles]
    finite_medians = [m for m in medians if np.isfinite(m)]
    monotone = bool(
        len(finite_medians) >= 2
        and all(b >= a for a, b in zip(finite_medians, finite_medians[1:]))
    )

    return {
        "n_bins": n_bins,
        "n_with_population": n,
        "n_without_population": len(words) - n,
        "quantiles": quantiles,
        "spearman": {
            "rho_log10_population_vs_error": rho,
            "n": len(finite_pairs),
            "note": "负值 = 越知名误差越小；用秩相关，抗人口重尾与误差长尾。",
        },
        "monotone": monotone,
        "scope": "仅含 population 字段的条目（本项目为全部 280 个城市）",
    }


# --------------------------------------------------------------------------- #
# 逐词误差（跨多次划分累积）
# --------------------------------------------------------------------------- #


def per_word_errors(
    config,
    X: np.ndarray,
    Y: np.ndarray,
    Y_enc: np.ndarray,
    n_repeats: int = 20,
    test_size: float = 0.2,
) -> np.ndarray:
    """返回每个词的平均留出误差（形状 ``(n_words,)``）。

    每个词在 ``n_repeats`` 次划分里大约会被留出 ``n_repeats × test_size`` 次；
    对它出现过的那些次取平均。从未被留出的词返回 NaN。
    """
    encoder = TargetEncoder(str(config.get("probe.target_encoding", "sincos")))
    totals = np.zeros(len(X), dtype=float)
    counts = np.zeros(len(X), dtype=int)

    base_seed = int(config.get("probe.random_state", 42))
    for repeat in range(n_repeats):
        train_idx, test_idx = _split_train_test(len(X), test_size, base_seed + repeat)
        probe = build_probe(config, "ridge").fit(X[train_idx], Y_enc[train_idx])
        pred = encoder.decode(probe.predict(X[test_idx]))
        errors = haversine_errors(Y[test_idx], pred)
        totals[test_idx] += errors
        counts[test_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        averages = totals / counts
    averages[counts == 0] = np.nan
    return averages


def _median(values: List[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.median(clean)) if clean else float("nan")


def bucket_label(label: str, label_order: Sequence[str]) -> str:
    """把类别名映射到统计用的桶：**不在 ``label_order`` 里的一律归入「其他」**。

    为什么需要这一步（真实踩过）
    ----------------------------
    319 词的 A 组里有 2 条 ``category`` 是「地区」（Scotland / Wales）。
    而统计桶只有 城市 / 国家 / 其他 三个 —— 直接拿 category 当下标会
    ``KeyError: '地区'``，整个脚本在 319 词上**跑不起来**。

    108 词时代之所以没暴露：那时没有 ``category`` 字段，类别靠 ``note`` 推断，
    这 2 条落在「其他」。词表扩充后字段带上了「地区」，脚本就崩了
    —— 和交接文档 §7.9「不要写死数据规模」是同一类问题的另一面：
    这一次是**写死了类别的取值集合**。

    归入「其他」而不是新开一列，是为了和历史数字可比（108 词时代
    「其他」就是这 2 条）。
    """
    return label if label in label_order else "其他"


def category_buckets(
    words: Sequence[str],
    categories: Dict[str, str],
    label_order: Sequence[str],
) -> Dict[str, List[str]]:
    """按统计桶分组（保证每个词恰好落进一个桶，总数守恒）。"""
    buckets: Dict[str, List[str]] = {label: [] for label in label_order}
    for word in words:
        buckets[bucket_label(categories.get(word, "其他"), label_order)].append(word)
    return buckets


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def discover_cached(reports: Path) -> Dict[str, Dict[str, Path]]:
    found: Dict[str, Dict[str, Path]] = {}
    for path in sorted(reports.glob("activations.*.npz")):
        stem = path.stem
        if not stem.startswith("activations."):
            continue
        for group in GROUPS:
            suffix = f".{group}"
            if stem.endswith(suffix):
                slug = stem[len("activations.") : -len(suffix)]
                found.setdefault(slug, {})[group] = path
                break
    return {slug: paths for slug, paths in found.items() if len(paths) == len(GROUPS)}


def platform_layer(slug: str, reports: Path, fallback_layers: List[int]) -> int:
    """取该模型的「平台层」：层稳定性结果里的 plateau 中位数，
    否则用编码对比里误差最低的层，再否则用中间层。"""
    stability = reports / "layer_stability.json"
    if stability.exists():
        data = json.loads(stability.read_text(encoding="utf-8"))
        model = data.get("models", {}).get(slug)
        if model:
            run = model.get("summary", {}).get("plateau_run") or []
            if run:
                return int(run[len(run) // 2])

    comparison = reports / f"encoding_comparison.{slug}.json"
    if comparison.exists():
        data = json.loads(comparison.read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        if rows:
            best = min(rows, key=lambda row: row.get("sincos_median_km") or 1e18)
            return int(best["layer"])

    return fallback_layers[len(fallback_layers) // 2]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按词类（城市/国家/其他）分层统计")
    parser.add_argument("--repeats", type=int, default=20, help="每模型的划分次数")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", nargs="+", help="只分析这些 slug")
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="逐层输出分层结果（用来检查「国家更差」这个结论是否层依赖）",
    )
    parser.add_argument("--out", help="把结果写成 Markdown 文件")
    parser.add_argument(
        "--population-bins",
        type=int,
        default=4,
        help="按人口（知名度的代理）等词数切几档，Q1 = 人口最多 = 最知名（默认 4）",
    )
    parser.add_argument("--config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.set("model.mock", False)
    config.set("probe.target_encoding", "sincos")

    lists = load_wordlists(config.path_of("data"))
    words = lists["control"].words
    Y = lists["control"].coords()
    encoder = TargetEncoder("sincos")
    Y_enc = encoder.encode(Y)
    categories = load_categories(config.path_of("data") / "words_control.json")
    population = load_population(config.path_of("data") / "words_control.json")

    reports = config.out_dir()
    cached_models = discover_cached(reports)
    if not cached_models:
        print("[FAIL] 没有缓存的激活，请先跑 layer-sweep")
        return 1
    if args.model:
        cached_models = {k: v for k, v in cached_models.items() if k in set(args.model)}
        if not cached_models:
            print("[FAIL] 指定的模型没有缓存")
            return 1

    label_order = ["城市", "国家", "其他"]
    word_buckets = category_buckets(words, categories, label_order)
    counts = {label: len(word_buckets[label]) for label in label_order}
    print(f"A 组 {len(words)} 词：" + "　".join(f"{k} {v}" for k, v in counts.items()))
    extra_labels = sorted({v for v in categories.values() if v not in label_order})
    if extra_labels:
        print(
            "  注意：这些 category 值不在统计桶里，已并入「其他」→ "
            + "、".join(extra_labels)
        )
    print(
        f"人口字段可用：{len(population)} 词（无 population 的 {len(words) - len(population)} 词"
        "不参与人口分档）"
    )
    print()

    results: Dict[str, Any] = {}
    layer_sweep_results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for slug, paths in sorted(cached_models.items()):
        extract = ExtractResult.load(paths["control"])
        layer = platform_layer(slug, reports, extract.layers)
        if layer not in extract.layers:
            print(f"[SKIP] {slug}: 没有第 {layer} 层")
            continue

        # --all-layers：逐层算一遍，看分层结论是否稳定
        if args.all_layers:
            name = extract.model_name.split("\\")[-1]
            print(f"=== {name}（逐层）===")
            print(f"  {'层':>3}{'全部':>9}{'城市':>9}{'国家':>9}{'国家/城市':>10}")
            print("  " + "-" * 42)
            sweep: Dict[int, Dict[str, float]] = {}
            for candidate in extract.layers:
                matrix = extract.matrix(candidate)
                errors = per_word_errors(
                    config, matrix, Y, Y_enc,
                    n_repeats=args.repeats, test_size=args.test_size,
                )
                buckets: Dict[str, List[float]] = {label: [] for label in label_order}
                for word, error in zip(words, errors):
                    if np.isfinite(error):
                        label = bucket_label(categories.get(word, "其他"), label_order)
                        buckets[label].append(float(error))
                entry = {
                    "overall": _median(list(errors)),
                    "城市": _median(buckets["城市"]),
                    "国家": _median(buckets["国家"]),
                    "其他": _median(buckets["其他"]),
                }
                ratio = entry["国家"] / entry["城市"] if entry["城市"] else float("nan")
                entry["country_city_ratio"] = ratio
                sweep[candidate] = entry
                print(
                    f"  {candidate:>3}{entry['overall']:>9,.0f}{entry['城市']:>9,.0f}"
                    f"{entry['国家']:>9,.0f}{ratio:>9.2f}×"
                )
            layer_sweep_results[slug] = sweep
            print()
            # all-layers 模式下仍记录代表层，便于后面汇总
            layer_rows = [row for row in sweep.values() if np.isfinite(row.get("country_city_ratio", float("nan")))]
            if layer_rows:
                ratios = [row["country_city_ratio"] for row in layer_rows]
                worse = sum(1 for ratio in ratios if ratio > 1.0)
                print(
                    f"  汇总：{len(ratios)} 层中有 {worse} 层的「国家」比「城市」更差"
                    f"（中位比值 {_median(ratios):.2f}×）"
                )
                print()

        X = extract.matrix(layer)

        errors = per_word_errors(
            config, X, Y, Y_enc, n_repeats=args.repeats, test_size=args.test_size
        )

        by_category: Dict[str, List[float]] = {label: [] for label in label_order}
        for word, error in zip(words, errors):
            if not np.isfinite(error):
                continue
            label = bucket_label(categories.get(word, "其他"), label_order)
            by_category[label].append(float(error))

        overall = _median(list(errors))
        per_category = {label: _median(values) for label, values in by_category.items()}

        name = extract.model_name.split("\\")[-1]
        print(f"=== {name}（layer {layer}, {X.shape[1]} 维）===")
        print(f"  {'类别':<8}{'词数':>5}{'中位误差':>11}")
        print("  " + "-" * 26)
        print(f"  {'全部':<8}{len(words):>5}{overall:>11,.0f}")
        for label in label_order:
            if by_category[label]:
                n_bucket = len(by_category[label])
                flag = "" if n_bucket >= MIN_BUCKET_WORDS else f"　⚠ n={n_bucket}，勿引用"
                print(
                    f"  {label:<8}{n_bucket:>5}{per_category[label]:>11,.0f}{flag}"
                )

        # ---- 人口分档（复用同一次 per_word_errors，零额外计算）----
        per_word = {
            word: (float(error) if np.isfinite(error) else float("nan"))
            for word, error in zip(words, errors)
        }
        by_population = summarise_population(
            words, population, per_word, n_bins=args.population_bins
        )
        print(
            f"\n  人口分档（{by_population['n_with_population']} 词 × "
            f"{by_population['n_bins']} 档，Q1 = 最知名）"
        )
        print(f"    {'档':<4}{'词数':>5}{'人口中位':>13}{'中位误差':>11}{'相对Q1':>9}")
        print("    " + "-" * 40)
        for q in by_population["quantiles"]:
            if q["n"] == 0:
                continue
            print(
                f"    {q['label']:<4}{q['n']:>5}{q['population_median']:>13,.0f}"
                f"{q['median_km']:>11,.0f}{q['ratio_to_q1']:>8.2f}×"
            )
        rho = by_population["spearman"]["rho_log10_population_vs_error"]
        print(
            f"    Spearman ρ(人口, 误差) = {rho:+.3f}"
            f"（负 = 越知名越准，n={by_population['spearman']['n']}）"
            + ("" if by_population["monotone"] else "　⚠ 档间误差非单调")
        )
        print()

        results[slug] = {
            "model": extract.model_name,
            "layer": layer,
            "hidden_dim": int(X.shape[1]),
            "overall_median_km": overall,
            "by_category": per_category,
            "by_population": by_population,
            "counts": {label: len(by_category[label]) for label in label_order},
            "per_word_km": {
                word: (float(error) if np.isfinite(error) else None)
                for word, error in zip(words, errors)
            },
        }

    # ---- 跨模型对比表 ----
    slugs = sorted(results)
    if not slugs:
        print("[FAIL] 没有可用结果")
        return 1

    print("=" * 78)
    print("跨模型分层对比（中位误差 km）")
    print("=" * 78)
    header = f"{'模型':<26}{'全部':>10}{'城市':>10}{'国家':>10}{'其他':>10}"
    print(header)
    print("-" * len(header))
    for slug in slugs:
        entry = results[slug]
        name = entry["model"].split("\\")[-1]
        values = [entry["overall_median_km"]] + [
            entry["by_category"].get(label, float("nan")) for label in label_order
        ]
        print(f"{name:<26}" + "".join(f"{v:>10,.0f}" for v in values))

    # ---- 跨模型人口分档（同一套分档，档成员与模型无关）----
    pop_labels = [
        q["label"]
        for q in results[slugs[0]]["by_population"]["quantiles"]
        if q["n"] > 0
    ]
    if pop_labels:
        print()
        print("跨模型人口分档对比（中位误差 km，Q1 = 最知名）")
        print("=" * 78)
        header_pop = f"{'模型':<26}" + "".join(f"{lab:>10}" for lab in pop_labels) + f"{'ρ':>9}"
        print(header_pop)
        print("-" * len(header_pop))
        for slug in slugs:
            entry = results[slug]
            name = entry["model"].split("\\")[-1]
            by_label = {q["label"]: q["median_km"] for q in entry["by_population"]["quantiles"]}
            rho = entry["by_population"]["spearman"]["rho_log10_population_vs_error"]
            print(
                f"{name:<26}"
                + "".join(f"{by_label.get(lab, float('nan')):>10,.0f}" for lab in pop_labels)
                + f"{rho:>+9.3f}"
            )

    # ---- 规模效应（最小 vs 最大模型）----
    if len(slugs) >= 2:
        ordered = sorted(slugs, key=lambda s: results[s]["hidden_dim"])
        small, large = ordered[0], ordered[-1]
        print()
        print("规模效应（小模型 → 大模型）：")
        print(f"  {results[small]['model'].split(chr(92))[-1]} → {results[large]['model'].split(chr(92))[-1]}")
        rows = [("全部", "overall_median_km")]
        for label in label_order:
            rows.append((label, label))
        skipped = []
        for name, key in rows:
            if key == "overall_median_km":
                before = results[small]["overall_median_km"]
                after = results[large]["overall_median_km"]
            else:
                if counts.get(name, 0) < MIN_BUCKET_WORDS:
                    skipped.append(f"{name}(n={counts.get(name, 0)})")
                    continue
                before = results[small]["by_category"].get(name, float("nan"))
                after = results[large]["by_category"].get(name, float("nan"))
            if np.isfinite(before) and np.isfinite(after) and before > 0:
                change = (after / before - 1) * 100
                print(f"    {name:<6}{before:>9,.0f} → {after:>8,.0f}　{change:>+7.1f}%")
        if skipped:
            print(f"    （已跳过样本过小的桶：{'、'.join(skipped)}）")

    payload = {
        "_meta": {
            "n_repeats": args.repeats,
            "test_size": args.test_size,
            "target_encoding": "sincos",
            "category_source": "data/words_control.json 的 category 或 note",
            "population_source": "data/words_control.json 的 population（本项目仅城市有）",
            "population_bins": args.population_bins,
            "population_binning": "按人口降序排名后等词数切档，Q1 = 人口最多 = 最知名",
            "note": "逐词误差为多次留出测量的平均；类别/人口档中位数均在逐词误差上计算。",
        },
        "counts": counts,
        "models": results,
    }
    out_json = reports / "stratified_analysis.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[OK] 结果：{out_json.name}")

    # ---- Markdown ----
    if args.out:
        lines = [
            "# 按词类分层统计",
            "",
            f"> 每模型 {args.repeats} 次划分（留出 {args.test_size:.0%}），"
            f"逐词误差取多次留出测量的平均，类别内取中位数。",
            "",
            f"A 组共 {len(words)} 词：" +
            "、".join(f"{label} {counts[label]}" for label in label_order),
            "",
            "| 模型 | 层 | 全部 | 城市 | 国家 | 其他 |",
            "|---|---|---|---|---|---|",
        ]
        for slug in slugs:
            entry = results[slug]
            name = entry["model"].split("\\")[-1]
            cells = [f"{entry['overall_median_km']:,.0f}"] + [
                (f"{entry['by_category'][label]:,.0f}" if np.isfinite(entry["by_category"].get(label, float("nan"))) else "—")
                for label in label_order
            ]
            lines.append(f"| `{name}` | {entry['layer']} | " + " | ".join(cells) + " |")
        small_buckets = [
            f"{label}(n={counts[label]})"
            for label in label_order
            if 0 < counts[label] < MIN_BUCKET_WORDS
        ]
        if small_buckets:
            lines += ["", f"> ⚠ 样本过小、数字不宜引用：{'、'.join(small_buckets)}。"]
        lines.append("")

        # ---- 人口分档表 ----
        if pop_labels:
            scope = results[slugs[0]]["by_population"]
            lines += [
                "## 人口分档（Q1 = 最知名）",
                "",
                f"> 只用有 `population` 字段的 {scope['n_with_population']} 个条目"
                f"（其余 {scope['n_without_population']} 个没有，不参与）；"
                f"按人口降序**等词数**切 {args.population_bins} 档。",
                "> 同一档内真值约定一致（都是城市市中心），所以档间差异只能归因于知名度。",
                "",
                "| 模型 | 层 | " + " | ".join(pop_labels) + " | Spearman ρ(人口,误差) |",
                "|---|---|" + "---|" * len(pop_labels) + "---|",
            ]
            for slug in slugs:
                entry = results[slug]
                name = entry["model"].split("\\")[-1]
                by_label = {q["label"]: q["median_km"] for q in entry["by_population"]["quantiles"]}
                cells = [
                    (f"{by_label[lab]:,.0f}" if np.isfinite(by_label.get(lab, float("nan"))) else "—")
                    for lab in pop_labels
                ]
                rho = entry["by_population"]["spearman"]["rho_log10_population_vs_error"]
                lines.append(
                    f"| `{name}` | {entry['layer']} | " + " | ".join(cells)
                    + f" | {rho:+.3f} |"
                )
            lines += ["", "### 人口档的边界与人口中位数", ""]
            lines += ["| 档 | 词数 | 人口中位 | 人口范围 |", "|---|---|---|---|"]
            for q in scope["quantiles"]:
                if q["n"] == 0:
                    continue
                lines.append(
                    f"| {q['label']} | {q['n']} | {q['population_median']:,.0f} | "
                    f"{q['population_min']:,.0f} – {q['population_max']:,.0f} |"
                )
            lines.append("")

        if len(slugs) >= 2:
            lines += ["## 规模效应", "", "| 类别 | 小模型 | 大模型 | 变化 |", "|---|---|---|---|"]
            skipped_md = []
            for name, key in [("全部", "overall_median_km")] + [(l, l) for l in label_order]:
                if key == "overall_median_km":
                    before = results[small]["overall_median_km"]
                    after = results[large]["overall_median_km"]
                else:
                    if counts.get(name, 0) < MIN_BUCKET_WORDS:
                        skipped_md.append(f"{name}(n={counts.get(name, 0)})")
                        continue
                    before = results[small]["by_category"].get(name, float("nan"))
                    after = results[large]["by_category"].get(name, float("nan"))
                if np.isfinite(before) and np.isfinite(after) and before > 0:
                    lines.append(
                        f"| {name} | {before:,.0f} km | {after:,.0f} km | "
                        f"{(after / before - 1) * 100:+.1f}% |"
                    )
            if skipped_md:
                lines += ["", f"> 已跳过样本过小的桶：{'、'.join(skipped_md)}。"]
            lines.append("")
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Markdown：{out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
