"""L2 归一化 / 幅度信息量的对照实验（2026-10-30，池化修复后重做）。

**这个脚本回答两个不同的问题**，它们以前被混在一起：

1. 「能不能对隐层做行归一化？」—— 工程问题
   变体：raw（不处理）/ row_l2（每行 L2 归一化）/ col_std（每列 z-score，只在训练集 fit）
2. 「幅度（向量长度）里有没有方向之外的独立信息？」—— 科学问题
   判据：把 ``log‖h‖`` 作为**额外一列特征**追加到 raw 上。
   如果比 raw 更准 → 幅度有独立信息；如果不变 → 幅度是冗余的。

为什么要分开：``row_l2`` 是**非线性**变换（h → h/‖h‖），它会破坏
"y 线性于 h" 这个探针前提，所以"归一化变差"根本不能证明"幅度携带信息"。
而 ``col_std`` 是**仿射**变换（保持线性），可以用作对照。

用法::

    python scripts/check_norm_effect.py                 # 三模型全部层（约 15 分钟）
    python scripts/check_norm_effect.py --layers 20 21  # 只跑指定层（约 1 分钟）
    python scripts/check_norm_effect.py --models qwen2.5-0.5b-instruct

输出：``reports/norm_comparison.json``（含逐层四变体、比值、以及 norm↔词长/误差的相关）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoprobe.config import load_config  # noqa: E402
from geoprobe.experiment import _split_train_test  # noqa: E402
from geoprobe.extract import ExtractResult, find_cached_activations  # noqa: E402
from geoprobe.metrics import haversine_errors  # noqa: E402
from geoprobe.probe import build_probe  # noqa: E402
from geoprobe.targets import TargetEncoder  # noqa: E402
from geoprobe.wordlists import load_wordlists  # noqa: E402

VARIANTS = ("raw", "raw_norm", "raw_norm_std", "row_l2", "col_std")

CANDIDATES = [
    ("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen2.5-3b-instruct", r"D:\models\Qwen2.5-3B-Instruct"),
]


def _norm_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return norms


def build_features(X: np.ndarray, variant: str, train_idx, test_idx):
    """按变体构造训练/测试特征。**列标准化只在训练集上 fit**。"""
    if variant == "raw":
        return X[train_idx], X[test_idx]
    if variant == "raw_norm":
        Z = np.hstack([X, np.log(_norm_rows(X) + 1e-9)])
        return Z[train_idx], Z[test_idx]
    if variant == "raw_norm_std":
        # 把 log‖h‖ **标准化后**再追加：避免它因为尺度太小（~6 vs h 的 ~500）
        # 被 ridge 的 L2 惩罚直接压成 0，从而"看起来没信息"。
        logn = np.log(_norm_rows(X) + 1e-9)
        mu = logn[train_idx].mean()
        sd = logn[train_idx].std() + 1e-12
        Z = np.hstack([X, (logn - mu) / sd])
        return Z[train_idx], Z[test_idx]
    if variant == "row_l2":
        Z = X / _norm_rows(X)
        return Z[train_idx], Z[test_idx]
    if variant == "col_std":
        mu = X[train_idx].mean(axis=0, keepdims=True)
        sd = X[train_idx].std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (X[train_idx] - mu) / sd, (X[test_idx] - mu) / sd
    raise ValueError(variant)


def run_layers(
    config,
    extract: ExtractResult,
    layers: Sequence[int],
    Y: np.ndarray,
    Y_enc: np.ndarray,
    encoder: TargetEncoder,
    repeats: int,
    test_size: float,
) -> List[Dict[str, float]]:
    rows = []
    for layer in layers:
        X = np.asarray(extract.matrix(layer), dtype=np.float64)
        medians: Dict[str, List[float]] = {v: [] for v in VARIANTS}
        for repeat in range(repeats):
            seed = int(config.get("probe.random_state", 42)) + repeat
            train_idx, test_idx = _split_train_test(len(X), test_size, seed)
            for variant in VARIANTS:
                Xtr, Xte = build_features(X, variant, train_idx, test_idx)
                probe = build_probe(config, "ridge").fit(Xtr, Y_enc[train_idx])
                pred = encoder.decode(probe.predict(Xte))
                medians[variant].append(float(np.median(haversine_errors(Y[test_idx], pred))))
        row: Dict[str, float] = {"layer": int(layer)}
        for variant in VARIANTS:
            arr = np.asarray(medians[variant])
            row[variant] = float(arr.mean())
            row[f"{variant}_std"] = float(arr.std())
        row["ratio_row_l2"] = row["row_l2"] / row["raw"]
        row["ratio_col_std"] = row["col_std"] / row["raw"]
        row["ratio_raw_norm"] = row["raw_norm"] / row["raw"]
        row["ratio_raw_norm_std"] = row["raw_norm_std"] / row["raw"]
        rows.append(row)
    return rows


def summarise(rows: List[Dict[str, float]]) -> Dict[str, object]:
    r_row = np.array([r["ratio_row_l2"] for r in rows])
    r_col = np.array([r["ratio_col_std"] for r in rows])

    def block(arr: np.ndarray, key: str) -> Dict[str, object]:
        return {
            "worse_layers": int((arr > 1.0).sum()),
            "n_layers": int(arr.size),
            "ratio_median": float(np.median(arr)),
            "ratio_range": [float(arr.min()), float(arr.max())],
            "mean_km": float(np.mean([r[key] for r in rows])),
        }

    best = min(rows, key=lambda r: r["raw"])
    return {
        "raw": {
            "mean_km": float(np.mean([r["raw"] for r in rows])),
            "best_layer": int(best["layer"]),
            "best_km": float(best["raw"]),
        },
        "row_l2": block(r_row, "row_l2"),
        "col_std": block(r_col, "col_std"),
        "raw_norm": {
            "worse_layers": int((np.array([r["ratio_raw_norm"] for r in rows]) > 1.0).sum()),
            "ratio_median": float(np.median([r["ratio_raw_norm"] for r in rows])),
            "ratio_range": [
                float(min(r["ratio_raw_norm"] for r in rows)),
                float(max(r["ratio_raw_norm"] for r in rows)),
            ],
        },
        "raw_norm_std": {
            "ratio_median": float(np.median([r["ratio_raw_norm_std"] for r in rows])),
            "ratio_range": [
                float(min(r["ratio_raw_norm_std"] for r in rows)),
                float(max(r["ratio_raw_norm_std"] for r in rows)),
            ],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="L2 归一化 / 幅度信息量对照")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--layers", type=int, nargs="+", help="只跑这些层（默认全部）")
    parser.add_argument("--models", nargs="+", help="只跑这些 slug")
    parser.add_argument("--out", default="reports/norm_comparison.json")
    args = parser.parse_args(argv)

    lists = load_wordlists()
    Y = lists["control"].coords()
    encoder = TargetEncoder("sincos")
    Y_enc = encoder.encode(Y)

    payload: Dict[str, object] = {
        "_meta": {
            "repeats": args.repeats,
            "test_size": args.test_size,
            "target_encoding": "sincos",
            "variants": {
                "raw": "不处理",
                "raw_norm": "raw + 追加一列 log‖h‖（检验幅度是否有独立信息）",
                "raw_norm_std": "同上，但把 log‖h‖ 标准化后再追加（排除被 L2 惩罚压死）",
                "row_l2": "每行 L2 归一化（非线性，会破坏线性假设）",
                "col_std": "每列 z-score，只在训练集 fit（仿射，保持线性）",
            },
            "note": "池化 bug 修复后重跑；比值 >1 表示比 raw 更差",
        },
        "models": {},
    }

    for slug, model_name in CANDIDATES:
        if args.models and slug not in args.models:
            continue
        config = load_config()
        config.set("model.name", model_name)
        config.set("probe.target_encoding", "sincos")
        cached, missing = find_cached_activations(config, ["control"])
        if missing:
            print(f"[SKIP] {slug}: 缺缓存 {[p.name for p in missing]}")
            continue
        extract = cached["control"]
        layers = [l for l in (args.layers or extract.layers) if l in extract.layers]
        print(f"=== {extract.model_name}（{len(layers)} 层 × {len(VARIANTS)} 变体 × {args.repeats} 次）===")
        rows = run_layers(config, extract, layers, Y, Y_enc, encoder, args.repeats, args.test_size)
        summary = summarise(rows)
        for row in rows:
            print(
                f"  L{row['layer']:>2}  raw {row['raw']:>7,.0f} | +‖h‖ {row['raw_norm']:>7,.0f}"
                f" ({row['ratio_raw_norm']:.4f}×) | +std‖h‖ {row['raw_norm_std']:>7,.0f}"
                f" ({row['ratio_raw_norm_std']:.4f}×) | row_l2 {row['row_l2']:>7,.0f}"
                f" ({row['ratio_row_l2']:.2f}×) | col_std {row['col_std']:>7,.0f}"
                f" ({row['ratio_col_std']:.2f}×)"
            )
        s = summary
        print(
            f"  → row_l2：{s['row_l2']['worse_layers']}/{s['row_l2']['n_layers']} 层变差，"
            f"中位 {s['row_l2']['ratio_median']:.3f}×"
            f"（{s['row_l2']['ratio_range'][0]:.2f}–{s['row_l2']['ratio_range'][1]:.2f}）"
        )
        print(
            f"  → col_std：{s['col_std']['worse_layers']}/{s['col_std']['n_layers']} 层变差，"
            f"中位 {s['col_std']['ratio_median']:.3f}×"
            f"（{s['col_std']['ratio_range'][0]:.2f}–{s['col_std']['ratio_range'][1]:.2f}）"
        )
        print(
            f"  → raw+‖h‖：中位 {s['raw_norm']['ratio_median']:.4f}×，"
            f"范围 {s['raw_norm']['ratio_range'][0]:.4f}–{s['raw_norm']['ratio_range'][1]:.4f}"
            f"；标准化后再加：中位 {s['raw_norm_std']['ratio_median']:.4f}×"
            f"（{s['raw_norm_std']['ratio_range'][0]:.4f}–{s['raw_norm_std']['ratio_range'][1]:.4f}）"
        )
        print()
        payload["models"][slug] = {"model": extract.model_name, "rows": rows, "summary": summary}

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
