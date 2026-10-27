"""Logit lens 命令行：把每层的 lens 指标与探针误差并排对照。

**这个命令要回答的问题**

逐层探针发现「地理可解码性在中层最高、之后单调下降」。
一个解释是：越深的层越被"预测下一个 token"的目标挤压。

Logit lens 提供了检验它的办法 —— 同时测两个曲线：

============================  ==========================================
曲线                           含义
============================  ==========================================
**探针误差**（越深越高）        地理坐标从这一层能不能线性读出来
**lens 熵**（若越深越低）       这一层的输出分布有多"决定好了"
============================  ==========================================

如果两条曲线**反向**（误差升、熵降），"挤压"解释就得到支持。
如果熵基本不随深度变化，那解释要换成别的（比如表征被旋转，线性可读性下降）。

用法::

    python -m geoprobe logit-lens
    python -m geoprobe logit-lens --model-name Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16
    python -m geoprobe logit-lens --mode prompt --limit 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import load_config, output_paths
from .lens import LENS_MODES, LogitLens, spearman
from .wordlists import load_wordlists

EXIT_OK, EXIT_FAIL = 0, 1


def _load_probe_errors(slug: str, out_dir: Path) -> Dict[int, float]:
    """从已有的编码对比结果里读出每层的探针误差（sincos）。

    这样 logit lens 不需要重跑探针 —— 直接复用之前算好的数字。
    """
    path = out_dir / f"encoding_comparison.{slug}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["layer"]): float(row["sincos_median_km"])
        for row in data.get("rows", [])
        if row.get("sincos_median_km") is not None
    }


def cmd_logit_lens(args: argparse.Namespace) -> int:
    from .extract import ActivationExtractor, resolve_transformer_layers

    config = load_config(args.config)
    if getattr(args, "dtype", None):
        config.set("model.dtype", args.dtype)
    if getattr(args, "device", None):
        config.set("model.device", args.device)
    if getattr(args, "model_name", None):
        config.set("model.name", args.model_name)

    lists = load_wordlists(config.path_of("data"))
    # 用 A 组（无歧义地名）—— 与主实验一致
    words: List[str] = lists["control"].words
    if args.limit:
        words = words[: int(args.limit)]

    slug = output_paths(config)["model_slug"]
    out_dir = config.out_dir()

    print(f"模型：{config.get('model.name')}　精度：{config.get('model.dtype')}")
    print(f"输入模式：{args.mode}　词数：{len(words)}")
    if args.mode == "bare":
        print("  （bare：输入就是裸地名，与主实验一致，可直接和探针误差对比）")
    else:
        print("  （prompt：输入为 '<地名> is located in'，更能逼出地理内容）")
    print()

    extractor = ActivationExtractor(config)
    extractor._ensure_loaded()
    layers = resolve_transformer_layers(extractor._model)
    print(f"模型共 {len(layers)} 层，开始计算 logit lens…")

    lens = LogitLens(
        model=extractor._model,
        tokenizer=extractor._tokenizer,
        layers=layers,
        device=str(config.get("model.device", "cpu")),
    )
    metrics = lens.run(words, mode=args.mode, progress=True)

    probe_errors = _load_probe_errors(slug, out_dir)
    reference: Optional[Dict[int, float]] = probe_errors if probe_errors else None
    if reference is None:
        print("\n[提示] 没找到本模型的分层探针误差（encoding_comparison.*.json），")
        print("       只输出 lens 指标。要并排对照请先跑：")
        print(f"             python scripts/compare_encodings.py --model {slug}")
        print()

    # ---- 表格 ----
    print()
    header = f"{'层':>3}  {'lens熵':>8}  {'top1概率':>9}  {'地理词质量':>10}  {'词首率':>7}"
    if reference:
        header += f"  {'探针误差':>9}"
    print(header)
    print("-" * (len(header) + 4))

    for entry in metrics:
        line = (
            f"{entry.layer:>3}  {entry.entropy_bits:>8.3f}  {entry.top1_prob:>9.4f}  "
            f"{entry.geo_mass:>10.4%}  {entry.word_start_rate:>7.1%}"
        )
        if reference and entry.layer in reference:
            line += f"  {reference[entry.layer]:>9,.0f}"
        print(line)

    # ---- 有效性检查：必须排在趋势解读之前 ----
    valid_layers = [entry.layer for entry in metrics if entry.word_start_rate >= 0.7]
    last_entry = metrics[-1]
    lens_valid_throughout = len(valid_layers) == len(metrics)

    print()
    print("=" * 72)
    print("① 有效性检查（先确认 logit lens 在本模型上可信，再谈趋势）")
    print("=" * 72)
    print(
        f"  最后一层词首率 {last_entry.word_start_rate:.0%}"
        f"　top-1 抽样: {' | '.join(last_entry.top1_tokens[:3])}"
    )
    print(f"  词首率 ≥ 70% 的层: {valid_layers if valid_layers else '（无）'}")
    if lens_valid_throughout:
        print("  ✅ 所有层的输出都是语言上合理的接续 → 趋势可以解读")
    else:
        print("  ⚠ **中间层的 lens 输出不是语言上合理的接续**（大量词片碎片）")
        print("     → 说明这些层的残差流还没进入「接近输出」的基，")
        print("       套用最终 LayerNorm 得到的分布**没有语言意义**。")
        print("     → 结论：logit lens 在本模型的中间层**近似失效**。")
        print("       没法用它检验「深层被下一词预测挤压」这个假设。")

    # ---- 相关性分析 ----
    layers_out = [entry.layer for entry in metrics]
    entropies = [entry.entropy_bits for entry in metrics]
    top1 = [entry.top1_prob for entry in metrics]
    geo = [entry.geo_mass for entry in metrics]

    print()
    print("=" * 70)
    print("② 趋势分析（Spearman 秩相关，+1 单调升 / -1 单调降）")
    print("=" * 70)
    rho_entropy = spearman(layers_out, entropies)
    rho_top1 = spearman(layers_out, top1)
    rho_geo = spearman(layers_out, geo)
    print(f"  深度 vs lens 熵      : {rho_entropy:+.3f}")
    print(f"  深度 vs top1 概率    : {rho_top1:+.3f}")
    print(f"  深度 vs 地理词质量   : {rho_geo:+.3f}")

    rho_probe_entropy = None
    if reference:
        paired = [(e.layer, e.entropy_bits, reference[e.layer]) for e in metrics if e.layer in reference]
        if len(paired) >= 3:
            rho_probe_entropy = spearman(
                [row[1] for row in paired], [row[2] for row in paired]
            )
            print(f"  lens 熵 vs 探针误差  : {rho_probe_entropy:+.3f}")

    print()
    if not lens_valid_throughout:
        print("判读：**不做判读。**")
        print("  有效性检查没过（见 ①），上面这些相关系数是在「无效测量」上算的，")
        print("  引用它们会得到错误结论。这是本项目一以贯之的原则：")
        print("  先确认仪器有效，再解读读数。")
    else:
        print("判读：")
        if rho_entropy < -0.5:
            print("  ✅ lens 熵随深度显著下降 → 深层确实更像「已决定好的下一词预测器」")
            if rho_probe_entropy is not None and rho_probe_entropy < -0.3:
                print("  ✅ 且 lens 熵下降时探针误差上升 → **支持「挤压」解释**")
            elif rho_probe_entropy is not None:
                print("  ⚠ 但 lens 熵与探针误差未呈明显负相关 → 「挤压」解释证据不足")
        elif rho_entropy > 0.3:
            print("  ⚠ lens 熵随深度上升 → 深层更「犹豫」，不支持「深层更确信」的前提")
            print("     更可能是表征在高维被旋转，线性可读性下降")
        else:
            print("  ⚠ lens 熵基本不随深度变化 → 「挤压」解释证据不足，应另找原因")

    # ---- 抽样看 top-1 token ----
    print()
    print("抽样：每层前 3 个词的 top-1 token（看模型在这一层「想接什么词」）")
    sample_layers = [layers_out[0], layers_out[len(layers_out) // 2], layers_out[-1]]
    for entry in metrics:
        if entry.layer in sample_layers and entry.top1_tokens:
            print(f"  layer {entry.layer:>2}: {' | '.join(entry.top1_tokens[:3])}")

    # ---- 保存 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "model": str(config.get("model.name")),
            "dtype": str(config.get("model.dtype")),
            "mode": args.mode,
            "n_words": len(words),
            "n_layers": len(layers),
            "note": "logit lens 是近似：把最终 LayerNorm 套在中间层残差流上。",
        },
        "correlations": {
            "depth_vs_entropy": rho_entropy,
            "depth_vs_top1_prob": rho_top1,
            "depth_vs_geo_mass": rho_geo,
            "entropy_vs_probe_error": rho_probe_entropy,
        },
        "validity": {
            "word_start_rate_per_layer": {
                str(entry.layer): entry.word_start_rate for entry in metrics
            },
            "valid_layers": valid_layers,
            "lens_valid_throughout": lens_valid_throughout,
            "note": "词首率低的层，其 lens 输出是词片碎片，没有语言意义，趋势不可解读。",
        },
        "layers": [
            {
                "layer": entry.layer,
                "entropy_bits": entry.entropy_bits,
                "top1_prob": entry.top1_prob,
                "geo_mass": entry.geo_mass,
                "geo_mass_continents": entry.geo_mass_continents,
                "geo_mass_countries": entry.geo_mass_countries,
                "geo_mass_directions": entry.geo_mass_directions,
                "word_start_rate": entry.word_start_rate,
                "probe_error_km": reference.get(entry.layer) if reference else None,
            }
            for entry in metrics
        ],
    }
    out_path = out_dir / f"logit_lens.{slug}.{args.mode}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"[OK] 结果：{out_path.name}")
    return EXIT_OK
