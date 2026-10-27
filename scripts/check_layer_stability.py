"""检查「最优层」这个结论是否稳定 —— 还是只是噪声。

**这个脚本解决一个很容易犯的错误：把一个平台上的抖动当成"最优层"。**

背景
----
逐层扫描会给出每层的中位误差，自然会想去挑"最小的那一层"。但在本项目的实测里：

```
layer 4 ~ layer 26 的误差都在 2,033 ~ 2,363 km 之间，跨度 330 km
而同一层换随机种子，误差会波动 ±115 km（标准差）
```

也就是说，**中间这一大段的层间差异和抽样噪声是同一量级** ——
"最优层是第 26 层"这种说法**过度精确**，换个种子就会变成第 25 层。

正确的说法是：

> 地理信息在**第 3–4 层之后进入平台期**，中间层之间差异不显著；
> **最后一层明显差于平台**。

本脚本用「同一层多组种子各跑一次」来量化：

* **层内波动**：同一层在不同种子下的标准差
* **层间差异**：各层均值之间的标准差
* **平台区间**：误差与最优值差异在噪声范围内的连续层段

用法::

    python scripts/check_layer_stability.py
    python scripts/check_layer_stability.py --model qwen2.5-0.5b-instruct --seeds 4
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
from geoprobe.experiment import run_experiment  # noqa: E402
from geoprobe.extract import ExtractResult  # noqa: E402
from geoprobe.wordlists import load_wordlists  # noqa: E402


def discover_cached(reports: Path) -> Dict[str, Dict[str, Path]]:
    """找出三组齐全的缓存模型（按前缀/后缀解析，slug 里含点号）。"""
    groups = ("control", "ambiguous", "nongeo")
    found: Dict[str, Dict[str, Path]] = {}
    for path in sorted(reports.glob("activations.*.npz")):
        stem = path.stem
        if not stem.startswith("activations."):
            continue
        for group in groups:
            suffix = f".{group}"
            if stem.endswith(suffix):
                slug = stem[len("activations.") : -len(suffix)]
                found.setdefault(slug, {})[group] = path
                break
    return {slug: paths for slug, paths in found.items() if len(paths) == len(groups)}


def measure_layers(
    config,
    lists,
    cached: Dict[str, ExtractResult],
    layers: List[int],
    seeds: int,
    repeats: int,
    encoding: str = "sincos",
) -> Dict[int, List[float]]:
    """对每层用 ``seeds`` 组不同种子各跑一次，返回 ``{layer: [误差...]}``。"""
    config.set("probe.target_encoding", encoding)
    config.set("evaluation.bootstrap_samples", 100)
    table: Dict[int, List[float]] = {}
    for layer in layers:
        values = []
        for index in range(seeds):
            config.set("probe.random_state", 42 + index * 100)
            result = run_experiment(
                config,
                lists=lists,
                extract_results=cached,
                layer=layer,
                n_repeats=repeats,
                run_baselines=False,
            )
            values.append(result.control_repeats.median_mean)
        table[layer] = values
    return table


def summarise(table: Dict[int, List[float]], noise_factor: float = 1.0) -> Dict[str, Any]:
    """从"每层多次测量"里提炼出可靠结论。"""
    means = {layer: float(np.mean(values)) for layer, values in table.items()}
    stds = {layer: float(np.std(values)) for layer, values in table.items()}

    within = float(np.mean(list(stds.values())))          # 层内波动
    between = float(np.std(list(means.values())))         # 层间差异
    noise = max(within, 1e-9)

    best_layer = min(means, key=means.get)
    best_mean = means[best_layer]

    # 平台：误差与最优值之差在 `noise_factor × 层内噪声` 以内的层
    plateau = sorted(
        layer for layer, value in means.items() if value - best_mean <= noise_factor * noise
    )

    # 连续的平台区间（用于描述"从第几层开始进入平台"）
    def contiguous(seq: List[int]) -> List[List[int]]:
        if not seq:
            return []
        runs, current = [], [seq[0]]
        for item in seq[1:]:
            if item == current[-1] + 1:
                current.append(item)
            else:
                runs.append(current)
                current = [item]
        runs.append(current)
        return runs

    runs = contiguous(plateau)
    longest = max(runs, key=len) if runs else []

    return {
        "means": means,
        "stds": stds,
        "within_layer_noise_km": within,
        "between_layer_spread_km": between,
        "noise_dominates": within >= between * 0.6,
        "best_layer": best_layer,
        "best_mean_km": best_mean,
        "plateau_layers": plateau,
        "plateau_run": longest,
        "plateau_start": longest[0] if longest else None,
        "plateau_end": longest[-1] if longest else None,
        "last_layer": max(means),
        "last_layer_mean_km": means[max(means)],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="检查「最优层」是否只是噪声")
    parser.add_argument("--model", nargs="+", help="模型 slug（默认全部已缓存模型）")
    parser.add_argument("--seeds", type=int, default=4, help="每层用几组种子")
    parser.add_argument("--repeats", type=int, default=5, help="每组种子内重复划分几次")
    parser.add_argument("--layers", type=int, nargs="+", help="只测这些层")
    parser.add_argument("--config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.set("model.mock", False)
    lists = load_wordlists(config.path_of("data"))
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

    payload: Dict[str, Any] = {"_meta": {"seeds": args.seeds, "repeats": args.repeats}, "models": {}}

    for slug, paths in sorted(cached_models.items()):
        cached = {group: ExtractResult.load(path) for group, path in paths.items()}
        available = cached["control"].layers
        layers = [layer for layer in (args.layers or available) if layer in available]

        print(f"=== {cached['control'].model_name} ===")
        print(f"  每层 {args.seeds} 组种子 × {args.repeats} 次重复，共 {len(layers)} 层")
        table = measure_layers(config, lists, cached, layers, args.seeds, args.repeats)
        summary = summarise(table)

        print()
        print(f"  {'层':>3}  {'均值':>9} {'标准差':>8}   多次测量")
        print("  " + "-" * 62)
        for layer in layers:
            mark = " ←最低均值" if layer == summary["best_layer"] else ""
            values = ", ".join(f"{v:,.0f}" for v in table[layer])
            print(
                f"  {layer:>3}  {summary['means'][layer]:>9,.0f} "
                f"{summary['stds'][layer]:>8,.0f}   {values}{mark}"
            )

        print()
        print(f"  层内波动（同层不同种子）: {summary['within_layer_noise_km']:,.0f} km")
        print(f"  层间差异（各层均值之间）: {summary['between_layer_spread_km']:,.0f} km")
        if summary["noise_dominates"]:
            print("  ⚠ 二者同一量级 → 中间层的排序**不可信**，只能谈「平台」")
        else:
            print("  ✅ 层间差异明显更大 → 层排序有一定意义")

        if summary["plateau_run"]:
            print(
                f"\n  平台区间：layer {summary['plateau_start']} ~ {summary['plateau_end']}"
                f"（{len(summary['plateau_run'])} 层，误差与最低值差 ≤ 1×层内噪声）"
            )
        print(
            f"  最后一层 layer {summary['last_layer']}: "
            f"{summary['last_layer_mean_km']:,.0f} km "
            f"（比最低均值高 {summary['last_layer_mean_km'] - summary['best_mean_km']:+,.0f} km）"
        )
        print()

        payload["models"][slug] = {
            "model": cached["control"].model_name,
            "n_layers": len(available),
            "table": {str(k): v for k, v in table.items()},
            "summary": {k: v for k, v in summary.items() if k not in ("means", "stds")},
        }

    out_path = reports / "layer_stability.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 结果：{out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
