"""用缓存的隐层激活，离线做「编码 × 层 × 模型」对比。

**这个脚本是 P0 实验（跨规模对比）的主工具。**

它做两件事：

1. **离线对比编码与层**：从 ``reports/activations.<模型>.<group>.npz`` 读激活，
   对每一层 × 每种编码跑一次探针。**不需要再跑模型**。
2. **跨模型汇总**：把多个模型的扫描结果放在一张表里，看规律是否随规模变化。

为什么必须有第 1 条
------------------
隐层提取是唯一昂贵的一步（要跑模型）。一旦缓存成 npz，就能反复离线实验：

* 换目标编码（latlon / sincos）
* 换重复次数、留出比例
* 修好某个 bug 之后重新出结果
* **跑多个模型后横向对比** —— 不需要把 0.5B 再加载一遍

这个设计还救过一次数据事故：`layer-sweep` 早期版本用同一个文件名，
两次扫描互相覆盖；因为有缓存，被覆盖的那次可以完整重算出来。

用法::

    python scripts/compare_encodings.py                      # 自动对比全部已缓存模型
    python scripts/compare_encodings.py --model qwen2.5-1.5b-instruct
    python scripts/compare_encodings.py --repeats 8
    python scripts/compare_encodings.py --layers 0 4 8 12 16 20 23
    python scripts/compare_encodings.py --out docs/跨模型对比.md
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

GROUPS = ("control", "ambiguous", "nongeo")


def _slugify(model_name: str) -> str:
    if model_name.startswith("mock"):
        return "mock"
    slug = model_name.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.\-]+", "-", slug).strip("-")


def discover_cached(reports: Path) -> Dict[str, Dict[str, Path]]:
    """扫描 reports/ 找出所有已缓存激活的模型。

    Returns:
        ``{slug: {group: path}}`` —— 只保留三组齐全的模型。

    Note:
        文件名形如 ``activations.qwen2.5-1.5b-instruct.control.npz``。
        **不能用 split(".") 解析** —— slug 里本身有点号（`qwen2.5`）。
        用前缀 ``activations.`` 与后缀 ``.<group>`` 夹出中间的 slug。
    """
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


def load_cached(paths: Dict[str, Path]) -> Dict[str, ExtractResult]:
    return {group: ExtractResult.load(path) for group, path in paths.items()}


def scan_model(
    config,
    lists,
    cached: Dict[str, ExtractResult],
    layers: List[int],
    repeats: int,
    test_size: float,
    quiet: bool = False,
) -> Dict[str, Any]:
    """对一个模型跑「编码 × 层」全扫描。"""
    model_name = cached["control"].model_name
    results: Dict[str, Dict[int, Any]] = {}

    for encoding in ("latlon", "sincos"):
        config.set("probe.target_encoding", encoding)
        results[encoding] = {}
        if not quiet:
            print(f"  --- {encoding} ---")
        for layer in layers:
            result = run_experiment(
                config,
                lists=lists,
                extract_results=cached,
                layer=layer,
                n_repeats=repeats,
                test_size=test_size,
                run_baselines=False,
            )
            results[encoding][layer] = result
            if not quiet:
                print(
                    f"    layer {layer:>2}: control {result.control_repeats.median_mean:>8,.0f} km　"
                    f"R²(lat) {result.pooled_r2_lat:>6.3f}　"
                    f"圆R²(lon) {result.pooled_circular_r2_lon:>6.3f}"
                )

    rows = []
    for layer in layers:
        left = results["latlon"][layer]
        right = results["sincos"][layer]
        gain = (
            left.control_repeats.median_mean / right.control_repeats.median_mean
            if right.control_repeats.median_mean > 0
            else float("nan")
        )
        rows.append(
            {
                "layer": layer,
                "latlon_median_km": left.control_repeats.median_mean,
                "latlon_r2_lat": left.pooled_r2_lat,
                "latlon_circular_r2_lon": left.pooled_circular_r2_lon,
                "sincos_median_km": right.control_repeats.median_mean,
                "sincos_r2_lat": right.pooled_r2_lat,
                "sincos_circular_r2_lon": right.pooled_circular_r2_lon,
                "gain_km": gain,
            }
        )

    best_sincos = min(rows, key=lambda row: row["sincos_median_km"])
    best_latlon = min(rows, key=lambda row: row["latlon_median_km"])
    best_lon = max(rows, key=lambda row: row["sincos_circular_r2_lon"] or -9)

    return {
        "model": model_name,
        "slug": _slugify(model_name),
        "n_layers_scanned": len(layers),
        "rows": rows,
        "summary": {
            "best_layer_sincos": best_sincos["layer"],
            "best_median_km_sincos": best_sincos["sincos_median_km"],
            "best_layer_latlon": best_latlon["layer"],
            "best_median_km_latlon": best_latlon["latlon_median_km"],
            "best_circular_r2_lon": best_lon["sincos_circular_r2_lon"],
            "best_circular_r2_lon_layer": best_lon["layer"],
            "last_layer_km_sincos": rows[-1]["sincos_median_km"],
            "last_layer_km_latlon": rows[-1]["latlon_median_km"],
            "mean_gain_km": float(np.nanmean([row["gain_km"] for row in rows])),
            "mean_circular_r2_lon_sincos": float(
                np.nanmean([row["sincos_circular_r2_lon"] for row in rows])
            ),
            "mean_circular_r2_lon_latlon": float(
                np.nanmean([row["latlon_circular_r2_lon"] for row in rows])
            ),
        },
    }


def render_cross_model(all_results: List[Dict[str, Any]]) -> str:
    """生成跨模型汇总表（Markdown）。"""
    lines = [
        "# 跨模型对比",
        "",
        "| 模型 | 层数 | 最优层 | 最优层误差 | 最后一层误差 | 最后一层/最优 | 平均 sincos 增益 | 最优圆R²(经度) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in all_results:
        s = result["summary"]
        n = result["n_layers_scanned"]
        relative = f"{s['best_layer_sincos'] / max(1, n - 1) * 100:.0f}%"
        improve = (
            f"{s['last_layer_km_sincos'] / s['best_median_km_sincos']:.2f}×"
            if s["best_median_km_sincos"] > 0
            else "—"
        )
        lines.append(
            f"| `{result['model']}` | {n} | layer {s['best_layer_sincos']}（{relative} 深度） | "
            f"**{s['best_median_km_sincos']:,.0f} km** | "
            f"{s['last_layer_km_sincos']:,.0f} km | {improve} | "
            f"{s['mean_gain_km']:.2f}× | {s['best_circular_r2_lon']:.3f} |"
        )
    lines.append("")

    if len(all_results) >= 2:
        ordered = sorted(all_results, key=lambda r: r["n_layers_scanned"])
        first, last = ordered[0], ordered[-1]
        lines += ["## 规模效应", ""]
        delta_best = (
            last["summary"]["best_median_km_sincos"] - first["summary"]["best_median_km_sincos"]
        )
        lines.append(
            f"- **最优层误差**：`{first['model']}` "
            f"{first['summary']['best_median_km_sincos']:,.0f} km → "
            f"`{last['model']}` {last['summary']['best_median_km_sincos']:,.0f} km"
            f"（变化 {delta_best:+,.0f} km）"
        )
        depth_first = first["summary"]["best_layer_sincos"] / max(1, first["n_layers_scanned"] - 1) * 100
        depth_last = last["summary"]["best_layer_sincos"] / max(1, last["n_layers_scanned"] - 1) * 100
        lines.append(
            f"- **最优层相对深度**：{depth_first:.0f}% → {depth_last:.0f}%"
            + (
                "（基本不变 → 知识在固定**相对深度**形成）"
                if abs(depth_last - depth_first) < 8
                else "（明显变化 → 知识形成位置随规模移动）"
            )
        )
        lines.append(
            f"- **sincos 平均增益**：{first['summary']['mean_gain_km']:.2f}× → "
            f"{last['summary']['mean_gain_km']:.2f}×"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="离线比较「编码 × 层 × 模型」")
    parser.add_argument("--repeats", type=int, default=5, help="重复划分次数")
    parser.add_argument("--layers", type=int, nargs="+", help="只跑这些层（默认全部）")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", nargs="+", help="只跑这些 slug（默认全部已缓存模型）")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--out", help="把跨模型汇总写成 Markdown 文件")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.set("model.mock", False)
    lists = load_wordlists(config.path_of("data"))
    reports = config.out_dir()

    cached_models = discover_cached(reports)
    if not cached_models:
        print("[FAIL] 没有找到任何缓存的激活（reports/activations.<模型>.<组>.npz）")
        print("       请先跑：python -m geoprobe layer-sweep --model-name <模型>")
        return 1

    if args.model:
        wanted = set(args.model)
        missing = wanted - set(cached_models)
        if missing:
            print(f"[FAIL] 这些模型没有缓存：{sorted(missing)}")
            print(f"       已缓存：{sorted(cached_models)}")
            return 1
        cached_models = {key: value for key, value in cached_models.items() if key in wanted}

    print(f"发现 {len(cached_models)} 个已缓存模型：{sorted(cached_models)}")
    print()

    all_results: List[Dict[str, Any]] = []
    for slug, paths in sorted(cached_models.items()):
        cached = load_cached(paths)
        available = cached["control"].layers
        layers = [layer for layer in (args.layers or available) if layer in available]
        if not layers:
            print(f"[SKIP] {slug}: 没有可用层")
            continue

        print(f"=== {cached['control'].model_name} ===")
        print(
            f"  {len(available)} 层 × {cached['control'].matrix().shape[1]} 维，"
            f"本次扫描 {len(layers)} 层，重复 {args.repeats} 次"
        )

        result = scan_model(config, lists, cached, layers, args.repeats, args.test_size)
        all_results.append(result)

        out_path = reports / f"encoding_comparison.{slug}.json"
        out_path.write_text(
            json.dumps(
                {
                    "_meta": {
                        "model": result["model"],
                        "pooling": cached["control"].pooling,
                        "n_repeats": args.repeats,
                        "test_size": args.test_size,
                        "layers": layers,
                        "n_layers_total": len(available),
                        "hidden_dim": int(cached["control"].matrix().shape[1]),
                    },
                    "summary": result["summary"],
                    "rows": result["rows"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  [OK] {out_path.name}")
        print()

    markdown = render_cross_model(all_results)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"[OK] 跨模型对比：{out_path}")
    else:
        print(markdown)

    combined = {
        "_meta": {
            "n_repeats": args.repeats,
            "test_size": args.test_size,
            "models": [result["model"] for result in all_results],
        },
        "models": [
            {
                "model": result["model"],
                "slug": result["slug"],
                "summary": result["summary"],
                "rows": result["rows"],
            }
            for result in all_results
        ],
    }
    combined_path = reports / "cross_model_comparison.json"
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 机读版：{combined_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
