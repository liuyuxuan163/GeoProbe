"""用缓存的隐层激活，离线比较「目标编码 × 层」的效果。

**为什么需要这个脚本**
--------------------
隐层提取是 layer-sweep 里唯一昂贵的一步（要跑模型）。一旦提取过一次并缓存成
``reports/activations.<group>.npz``，就可以**离线**反复做下面这些事，不必再碰模型：

* 换目标编码（latlon / sincos）重跑
* 换探针（Ridge / MLP）重跑
* 换重复次数、换留出比例
* 修复了某个 bug 之后重新出结果

**这个脚本还解决了一个真实事故**：
`layer-sweep` 早期版本无论用哪种编码都写同一个 ``layer_sweep.json``，
于是跑两遍（latlon 与 sincos）时，第一遍的结果被第二遍覆盖了。
有了缓存之后，那次"丢失"的扫描可以完整重算出来。

用法::

    python scripts/compare_encodings.py                    # 默认 5 次重复
    python scripts/compare_encodings.py --repeats 10
    python scripts/compare_encodings.py --layers 0 4 8 12 16 20 23
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoprobe.config import load_config  # noqa: E402
from geoprobe.experiment import run_experiment, save_result  # noqa: E402
from geoprobe.extract import ExtractResult  # noqa: E402
from geoprobe.wordlists import load_wordlists  # noqa: E402


def load_cached(reports: Path) -> Dict[str, ExtractResult]:
    """读取三组词的缓存激活。"""
    out: Dict[str, ExtractResult] = {}
    for group in ("control", "ambiguous", "nongeo"):
        path = reports / f"activations.{group}.npz"
        if not path.exists():
            raise SystemExit(
                f"缺少缓存文件：{path}\n"
                "请先跑一次逐层扫描来生成缓存：\n"
                "    python -m geoprobe layer-sweep"
            )
        out[group] = ExtractResult.load(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="离线比较目标编码 × 层")
    parser.add_argument("--repeats", type=int, default=5, help="重复划分次数")
    parser.add_argument("--layers", type=int, nargs="+", help="只跑这些层（默认全部）")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--config", help="配置文件路径")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.set("model.mock", False)
    lists = load_wordlists(config.path_of("data"))
    reports = config.path_of(str(config.get("report.out_dir", "reports")))
    cached = load_cached(reports)

    available = cached["control"].layers
    layers = args.layers or available
    layers = [layer for layer in layers if layer in available]

    print(f"模型：{cached['control'].model_name}")
    print(f"图层：共 {len(available)} 层，本次分析 {len(layers)} 层")
    print(f"池化：{cached['control'].pooling}　重复：{args.repeats} 次")
    print()

    results: Dict[str, Dict[int, Any]] = {}
    for encoding in ("latlon", "sincos"):
        config.set("probe.target_encoding", encoding)
        results[encoding] = {}
        print(f"--- target_encoding = {encoding} ---")
        for layer in layers:
            result = run_experiment(
                config,
                lists=lists,
                extract_results=cached,
                layer=layer,
                n_repeats=args.repeats,
                test_size=args.test_size,
                run_baselines=False,
            )
            results[encoding][layer] = result
            print(
                f"  layer {layer:>2}: control {result.control_repeats.median_mean:>8,.0f} km　"
                f"R²(lat) {result.pooled_r2_lat:>6.3f}　"
                f"圆形R²(lon) {result.pooled_circular_r2_lon:>6.3f}"
            )
        print()

    # ---- 对比表 ----
    print("=" * 92)
    print("编码 × 层 对比（control 中位误差 km ｜ 圆形 R² 经度）")
    print("=" * 92)
    header = (
        f"{'层':>3}  {'latlon km':>10} {'lat R²':>7} {'lon圆R²':>8}  "
        f"{'sincos km':>10} {'lat R²':>7} {'lon圆R²':>8}  {'改善':>7}"
    )
    print(header)
    print("-" * 92)
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        left = results["latlon"][layer]
        right = results["sincos"][layer]
        gain = (
            left.control_repeats.median_mean / right.control_repeats.median_mean
            if right.control_repeats.median_mean > 0
            else float("nan")
        )
        print(
            f"{layer:>3}  {left.control_repeats.median_mean:>10,.0f} "
            f"{left.pooled_r2_lat:>7.3f} {left.pooled_circular_r2_lon:>8.3f}  "
            f"{right.control_repeats.median_mean:>10,.0f} "
            f"{right.pooled_r2_lat:>7.3f} {right.pooled_circular_r2_lon:>8.3f}  "
            f"{gain:>6.2f}×"
        )
        rows.append(
            {
                "layer": layer,
                "latlon_median_km": left.control_repeats.median_mean,
                "latlon_r2_lat": left.pooled_r2_lat,
                "latlon_circular_r2_lon": left.pooled_circular_r2_lon,
                "latlon_linear_r2_lon": left.pooled_r2_lon,
                "sincos_median_km": right.control_repeats.median_mean,
                "sincos_r2_lat": right.pooled_r2_lat,
                "sincos_circular_r2_lon": right.pooled_circular_r2_lon,
                "sincos_linear_r2_lon": right.pooled_r2_lon,
                "gain_km": gain,
            }
        )

    # ---- 结论 ----
    best_latlon = min(rows, key=lambda row: row["latlon_median_km"])
    best_sincos = min(rows, key=lambda row: row["sincos_median_km"])
    best_lon_latlon = max(rows, key=lambda row: row["latlon_circular_r2_lon"] or -9)
    best_lon_sincos = max(rows, key=lambda row: row["sincos_circular_r2_lon"] or -9)

    print()
    print("=" * 92)
    print("要点")
    print("=" * 92)
    print(
        f"  误差最优层：latlon → layer {best_latlon['layer']} "
        f"({best_latlon['latlon_median_km']:,.0f} km)　"
        f"sincos → layer {best_sincos['layer']} ({best_sincos['sincos_median_km']:,.0f} km)"
    )
    last = rows[-1]
    print(
        f"  最后一层 {last['layer']}：latlon {last['latlon_median_km']:,.0f} km　"
        f"sincos {last['sincos_median_km']:,.0f} km"
    )
    print(
        f"  经度最优层：latlon → layer {best_lon_latlon['layer']} "
        f"(圆形R²={best_lon_latlon['latlon_circular_r2_lon']:.3f})　"
        f"sincos → layer {best_lon_sincos['layer']} "
        f"(圆形R²={best_lon_sincos['sincos_circular_r2_lon']:.3f})"
    )
    mean_gain = float(np.nanmean([row["gain_km"] for row in rows]))
    mean_latlon = float(np.nanmean([row["latlon_circular_r2_lon"] for row in rows]))
    mean_sincos = float(np.nanmean([row["sincos_circular_r2_lon"] for row in rows]))
    print(f"  平均改善（公里误差）：{mean_gain:.2f}×")
    print(f"  平均圆形 R²(经度)：latlon {mean_latlon:.3f} → sincos {mean_sincos:.3f}")
    if mean_sincos > mean_latlon * 1.3:
        print("  → sincos 明显改善经度可解码性：低 R²(经度) 主要是**编码缺陷**。")
    elif mean_sincos > mean_latlon:
        print("  → sincos 略有改善，幅度有限：模型本身对经度的编码较弱。")
    else:
        print("  → sincos 未改善：经度难解码是**模型的性质**，不是编码问题。")

    out_path = reports / "encoding_comparison.json"
    out_path.write_text(
        json.dumps(
            {
                "_meta": {
                    "model": cached["control"].model_name,
                    "pooling": cached["control"].pooling,
                    "n_repeats": args.repeats,
                    "test_size": args.test_size,
                    "layers": layers,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] 对比结果：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
