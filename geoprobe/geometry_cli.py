"""几何分析的命令行：从缓存激活算范数 / 旋转 / 有效维度，并与探针误差对照。

用法::

    python -m geoprobe geometry                       # 全部已缓存模型
    python -m geoprobe geometry --model qwen2.5-3b-instruct
    python -m geoprobe geometry --out docs/表征几何.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import load_config
from .extract import ExtractResult
from .geometry import analyze_geometry, pearson

EXIT_OK, EXIT_FAIL = 0, 1


def discover_cached(reports: Path) -> Dict[str, Dict[str, Path]]:
    """找出三组齐全的缓存模型（按前后缀解析，slug 里含点号）。"""
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


def _load_probe_errors(slug: str, out_dir: Path) -> Dict[int, float]:
    path = out_dir / f"encoding_comparison.{slug}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["layer"]): float(row["sincos_median_km"])
        for row in data.get("rows", [])
        if row.get("sincos_median_km") is not None
    }


def _best_layer(probe_errors: Dict[int, float]) -> Optional[int]:
    if not probe_errors:
        return None
    return min(probe_errors, key=probe_errors.get)


def cmd_geometry(args: Any) -> int:
    config = load_config(args.config)
    reports = config.out_dir()

    cached_models = discover_cached(reports)
    if not cached_models:
        print("[FAIL] 没有缓存的激活，请先跑 layer-sweep")
        return EXIT_FAIL
    if getattr(args, "model", None):
        wanted = set(args.model)
        cached_models = {k: v for k, v in cached_models.items() if k in wanted}
        if not cached_models:
            print(f"[FAIL] 指定模型没有缓存：{sorted(wanted)}")
            return EXIT_FAIL

    payload: Dict[str, Any] = {"_meta": {"note": "只依赖缓存激活，不需要 lm_head"}, "models": {}}
    markdown: List[str] = ["# 表征几何分析", ""]

    for slug, paths in sorted(cached_models.items()):
        extract = ExtractResult.load(paths["control"])
        embeddings = {layer: extract.matrix(layer) for layer in extract.layers}
        probe_errors = _load_probe_errors(slug, reports)
        best = _best_layer(probe_errors)

        rows = analyze_geometry(embeddings, best_layer=best, probe_errors=probe_errors)
        n = len(rows)
        payload_hidden = int(embeddings[rows[0].layer].shape[1])
        print(f"=== {extract.model_name}（{n} 层 × {payload_hidden} 维）===")
        print(f"  参照层（探针误差最低）= {best if best is not None else '未提供'}")

        header = (
            f"  {'层':>3}  {'‖h‖':>7}  {'范数跨度':>8}  {'有效维':>7}  {'有效维(归一)':>11}"
            f"  {'层间cos':>8}  {'漂移':>8}"
        )
        if probe_errors:
            header += f"  {'探针误差':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in rows:
            line = (
                f"  {row.layer:>3}  {row.residual_norm:>7.1f}  {row.norm_dispersion:>8.1f}  "
                f"{row.effective_dimension:>7.1f}  {row.effective_dimension_normed:>11.1f}"
                f"  {row.inter_layer_cosine:>8.3f}  {row.drift_from_best:>8.4f}"
            )
            if probe_errors and row.probe_error_km is not None:
                line += f"  {row.probe_error_km:>9,.0f}"
            print(line)

        # ---- 相关性 ----
        print()
        if probe_errors:
            errors = [row.probe_error_km for row in rows]
            drift = [row.drift_from_best for row in rows]
            dimension = [row.effective_dimension_normed for row in rows]
            dispersion = [row.norm_dispersion for row in rows]

            r_drift = pearson(drift, errors)
            r_dim = pearson(dimension, errors)
            r_disp = pearson(dispersion, errors)

            print("  相关性（与探针误差，皮尔逊）：")
            print(f"    离参照层漂移        : {r_drift:+.3f}")
            print(f"    有效维度（归一化后）: {r_dim:+.3f}")
            print(f"    范数跨度            : {r_disp:+.3f}")
            print()
            print("  三个关键量的量级：")
            print(
                f"    范数跨度（max/median）：{min(dispersion):.1f} ~ {max(dispersion):.1f} 倍"
            )
            print(
                f"    有效维度（归一化后）：{min(dimension):.1f} ~ {max(dimension):.1f}"
                f"（隐层维度 {payload_hidden}）"
            )
            if np.isfinite(r_drift) and abs(r_drift) > 0.6:
                print(f"    → 漂移是主要解释量（{r_drift:+.2f}）")
            elif np.isfinite(r_drift):
                print("    ⚠ 漂移解释力不足 → 深层误差上升不是单纯「转动」")
        else:
            print("  （没有探针误差可对照，跳过相关性分析）")

        print()
        payload["models"][slug] = {
            "model": extract.model_name,
            "hidden_dim": int(embeddings[rows[0].layer].shape[1]),
            "reference_layer": best,
            "correlations": {
                "norm_vs_error": pearson([r.residual_norm for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "centered_norm_vs_error": pearson([r.centered_norm for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "specificity_vs_error": pearson([r.specificity for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "inter_cosine_vs_error": pearson([r.inter_layer_cosine for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "drift_vs_error": pearson([r.drift_from_best for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "effdim_vs_error": pearson([r.effective_dimension for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
                "effdim_normed_vs_error": pearson(
                    [r.effective_dimension_normed for r in rows],
                    [r.probe_error_km for r in rows],
                )
                if probe_errors
                else None,
                "dispersion_vs_error": pearson([r.norm_dispersion for r in rows], [r.probe_error_km for r in rows])
                if probe_errors
                else None,
            },
            "layers": [
                {
                    "layer": r.layer,
                    "residual_norm": r.residual_norm,
                    "centered_norm": r.centered_norm,
                    "specificity": r.specificity,
                    "norm_dispersion": r.norm_dispersion,
                    "inter_layer_cosine": r.inter_layer_cosine,
                    "drift_from_best": r.drift_from_best,
                    "effective_dimension": r.effective_dimension,
                    "effective_dimension_normed": r.effective_dimension_normed,
                    "probe_error_km": r.probe_error_km,
                }
                for r in rows
            ],
        }

    out_path = reports / "geometry.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 结果：{out_path.name}")
    return EXIT_OK
