"""按词类分层统计：城市 / 国家 / 其他，分开看三个模型的表现。

**为什么需要这个脚本**

之前的结论是「参数 6 倍只让误差降 19%，规模效应很弱」。但词表里混着两种
真值约定：

* **城市** —— 真值是市中心，几乎无歧义（±20 km）
* **国家** —— 真值是面积质心，而"一个国家对应哪个点"本身有 ±1,000 km 级歧义
  （巴西的质心在亚马逊，最大城市却是圣保罗，首都又是巴西利亚）

国家条目的固有歧义构成一个**噪声地板**，把真实的规模效应盖住了。
本脚本把两者拆开算，回答："规模效应到底有多强？"

**方法**

对每个模型、在每个划分上：

1. 80% 训练 / 20% 留出
2. 记录**留出词**的逐词 Haversine 误差
3. 跑多个种子，让每个词被测试多次（约 20% × 种子数 次）
4. 按词的类别聚合，取中位数

这样每个词都有多个独立测量的误差均值，比单次划分稳得多。

**类别从哪来**

优先读 ``data/words_control.json`` 里的 ``category`` 字段；
没有的话从 ``note`` 推断（本项目现有数据用的是 note）。

用法::

    python scripts/stratified_analysis.py
    python scripts/stratified_analysis.py --repeats 20 --out docs/分层统计.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    counts = {
        label: sum(1 for word in words if categories.get(word) == label)
        for label in label_order
    }
    print(f"A 组 {len(words)} 词：" + "　".join(f"{k} {v}" for k, v in counts.items()))
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
                        buckets[categories.get(word, "其他")].append(float(error))
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
            by_category[categories.get(word, "其他")].append(float(error))

        overall = _median(list(errors))
        per_category = {label: _median(values) for label, values in by_category.items()}

        name = extract.model_name.split("\\")[-1]
        print(f"=== {name}（layer {layer}, {X.shape[1]} 维）===")
        print(f"  {'类别':<8}{'词数':>5}{'中位误差':>11}")
        print("  " + "-" * 26)
        print(f"  {'全部':<8}{len(words):>5}{overall:>11,.0f}")
        for label in label_order:
            if by_category[label]:
                print(f"  {label:<8}{len(by_category[label]):>5}{per_category[label]:>11,.0f}")
        print()

        results[slug] = {
            "model": extract.model_name,
            "layer": layer,
            "hidden_dim": int(X.shape[1]),
            "overall_median_km": overall,
            "by_category": per_category,
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
        for name, key in rows:
            if key == "overall_median_km":
                before = results[small]["overall_median_km"]
                after = results[large]["overall_median_km"]
            else:
                before = results[small]["by_category"].get(name, float("nan"))
                after = results[large]["by_category"].get(name, float("nan"))
            if np.isfinite(before) and np.isfinite(after) and before > 0:
                change = (after / before - 1) * 100
                print(f"    {name:<6}{before:>9,.0f} → {after:>8,.0f}　{change:>+7.1f}%")

    payload = {
        "_meta": {
            "n_repeats": args.repeats,
            "test_size": args.test_size,
            "target_encoding": "sincos",
            "category_source": "data/words_control.json 的 category 或 note",
            "note": "逐词误差为多次留出测量的平均；类别中位数在逐词误差上计算。",
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
        lines.append("")
        if len(slugs) >= 2:
            lines += ["## 规模效应", "", "| 类别 | 小模型 | 大模型 | 变化 |", "|---|---|---|---|"]
            for name, key in [("全部", "overall_median_km")] + [(l, l) for l in label_order]:
                if key == "overall_median_km":
                    before = results[small]["overall_median_km"]
                    after = results[large]["overall_median_km"]
                else:
                    before = results[small]["by_category"].get(name, float("nan"))
                    after = results[large]["by_category"].get(name, float("nan"))
                if np.isfinite(before) and np.isfinite(after) and before > 0:
                    lines.append(
                        f"| {name} | {before:,.0f} km | {after:,.0f} km | "
                        f"{(after / before - 1) * 100:+.1f}% |"
                    )
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
