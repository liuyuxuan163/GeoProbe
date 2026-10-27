"""GeoProbe 命令行入口。

用法::

    python -m geoprobe info            # 环境自检
    python -m geoprobe run             # 跑完整实验 + 生成报告
    python -m geoprobe run --mock      # Mock 模式（不加载模型，验证流程）
    python -m geoprobe layer-sweep     # 逐层扫描：地理知识在哪一层涌现
    python -m geoprobe check           # 只做数据校验

设计说明
--------
``--mock`` 是一个**一等公民**，不是临时开关：
它让整个 pipeline（探针 → 指标 → 基线 → 报告）在**没有 GPU、没有模型、
没有网络**的情况下也能完整跑通并测试。这与 DocPilot 里 MockLLM 的思路一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import load_config, output_paths
from .experiment import run_experiment, save_result
from .report import render_markdown, write_report
from .wordlists import load_wordlists

EXIT_OK, EXIT_FAIL = 0, 1


def _apply_common(config, args: argparse.Namespace):
    """把命令行参数覆盖到配置上。"""
    if getattr(args, "mock", False):
        config.set("model.mock", True)
    if getattr(args, "pooling", None):
        config.set("extract.pooling", args.pooling)
    if getattr(args, "target_encoding", None):
        config.set("probe.target_encoding", args.target_encoding)
    if getattr(args, "dtype", None):
        config.set("model.dtype", args.dtype)
    if getattr(args, "layer", None) is not None:
        config.set("extract.layer", args.layer)
    if getattr(args, "device", None):
        config.set("model.device", args.device)
    if getattr(args, "model_name", None):
        config.set("model.name", args.model_name)
    if getattr(args, "repeats", None):
        config.set("probe.n_repeats", args.repeats)
    return config


def cmd_info(args: argparse.Namespace) -> int:
    """环境自检：依赖、数据、配置。"""
    config = load_config(args.config)
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "hint": hint})

    add("python", sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "需要 3.9+")

    for module, required, hint in (
        ("numpy", True, "必装"),
        ("sklearn", True, "必装"),
        ("torch", False, "只有用真实模型时才需要；--mock 模式不需要"),
        ("transformers", False, "同上"),
        ("matplotlib", False, "画图需要"),
        ("geopandas", False, "画地图需要"),
        ("geopy", False, "刷新坐标需要"),
    ):
        try:
            __import__(module)
            add(f"dep:{module}", True, "已安装")
        except ImportError:
            add(f"dep:{module}", not required, "未安装", hint)

    try:
        lists = load_wordlists(config.path_of("data"))
        summary = "  ".join(f"{name}={len(word_list)}" for name, word_list in lists.items())
        add("wordlists", True, summary)
    except Exception as exc:  # noqa: BLE001
        add("wordlists", False, f"加载失败: {exc}")

    payload = {"ok": all(item["ok"] for item in checks), "config": config.describe(), "checks": checks}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("GeoProbe 环境自检")
        print("=" * 72)
        for item in checks:
            mark = "✅" if item["ok"] else "❌"
            print(f"{mark} {item['name']:<22} {item['detail']}")
            if item["hint"] and not item["ok"]:
                print(f"   └─ {item['hint']}")
        print("=" * 72)
        print(f"配置：{config.describe()}")
    return EXIT_OK if payload["ok"] else EXIT_FAIL


def cmd_models(args: argparse.Namespace) -> int:
    """列出候选模型的**内存预估**与可行性，并显示已跑过的缓存。

    为什么需要这个命令
    ------------------
    换更大模型是这个项目最重要的下一步（P0），但"能不能跑得动"完全取决于本机内存。
    与其等到加载时 OOM 崩溃，不如先给一张表看清楚。

    **只看预估，不下载任何东西。**
    """
    from .extract import _available_memory_gb, estimate_model_memory_gb

    candidates = [
        ("Qwen/Qwen2.5-0.5B-Instruct", "最小，CPU 秒级"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "中等，CPU 可跑"),
        ("Qwen/Qwen2.5-3B-Instruct", "较大，需要充足内存"),
        ("Qwen/Qwen2.5-7B-Instruct", "通常需要 GPU"),
        ("meta-llama/Llama-3.2-1B-Instruct", "跨模型对照"),
    ]
    available = _available_memory_gb("cpu")
    print("候选模型的内存预估（不含缓存激活）")
    print("=" * 92)
    print(f"{'模型':<38}{'fp32':>10}{'bf16':>10}{'可行性':>16}")
    print("-" * 92)

    for name, note in candidates:
        fp32 = estimate_model_memory_gb(name, "float32")
        bf16 = estimate_model_memory_gb(name, "bfloat16")
        fp32_s = f"{fp32:.1f} GB" if fp32 else "—"
        bf16_s = f"{bf16:.1f} GB" if bf16 else "—"
        if available is None or bf16 is None:
            verdict = "未知"
        elif bf16 <= available:
            verdict = "✅ bf16 可跑"
        elif fp32 <= available:
            verdict = "⚠ 只能 fp32"
        else:
            verdict = "❌ 内存不足"
        print(f"{name:<38}{fp32_s:>10}{bf16_s:>10}{verdict:>16}")

    print("-" * 92)
    if available is not None:
        print(f"当前可用内存：{available:.1f} GB（CPU）")
        print("提示：3B 的 bf16 约需 7.5 GB —— 关掉浏览器/IDE 通常能腾出几个 GB。")
        print("      如果内存一直不够，建议换一台有 GPU 的机器跑 3B。")

    # 已经跑过哪些模型（看缓存）
    print()
    print("已缓存激活（reports/activations.*.npz）")
    print("=" * 92)
    out = load_config(args.config).out_dir()
    from .extract import ExtractResult

    found = False
    for path in sorted(out.glob("activations.*.npz")):
        # ⚠ 不能用 split(".") 判断 —— 模型 slug 里本身含点号
        #（`qwen2.5-1.5b-instruct`），按点切会切错。用前后缀匹配。
        stem = path.stem
        if not (stem.startswith("activations.") and stem.endswith(".control")):
            continue
        slug = stem[len("activations.") : -len(".control")]
        try:
            result = ExtractResult.load(path)
        except Exception:  # noqa: BLE001
            continue
        found = True
        label = "mock（合成向量）" if result.mock else result.model_name
        print(
            f"  {label:<38} {len(result.layers):>2} 层 × {result.matrix().shape[1]:>4} 维  "
            f"（{len(result.words)} 词）　文件名 slug={slug}"
        )
    if not found:
        print("  （还没有缓存。跑一次：python -m geoprobe layer-sweep）")

    print()
    print("跑不同模型的命令示例：")
    for name, _note in candidates[:3]:
        print(f"  python -m geoprobe layer-sweep --model-name {name} --dtype bfloat16")
    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    """只做数据校验，不跑实验（快速确认词表没问题）。"""
    config = load_config(args.config)
    lists = load_wordlists(config.path_of("data"))
    print("词表校验通过")
    print("=" * 64)
    for name, word_list in lists.items():
        summary = word_list.summary()
        print(
            f"  {name:<12} n={summary['n']:>3}  "
            f"带真值={summary['has_truth']}  带备选={summary['n_with_alt']}  "
            f"{summary['description']}"
        )
    print()
    print("示例：")
    for name, word_list in lists.items():
        sample = ", ".join(word_list.words[:6])
        print(f"  {name:<12} {sample} …")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """跑完整实验并生成报告。"""
    config = _apply_common(load_config(args.config), args)
    lists = load_wordlists(config.path_of("data"))

    print(f"模型：{config.get('model.name')}　mock={config.get('model.mock')}")
    print(f"池化：{config.get('extract.pooling')}　重复划分：{args.repeats or 5} 次")

    # --from-cache：复用 layer-sweep 留下的隐层缓存，跳过模型前向（省 30 秒以上）
    extract_results = None
    if getattr(args, "from_cache", False):
        from .extract import ExtractResult, find_cached_activations

        # 缓存文件名带模型标识 → 换模型不会互相覆盖，也不会误用别的模型的缓存
        cached, missing = find_cached_activations(config, list(lists))
        if missing:
            print(f"[FAIL] 缺少缓存：{[p.name for p in missing]}")
            print("       请先为**这个模型**跑一次逐层扫描生成缓存：")
            print(f"             python -m geoprobe layer-sweep"
                  + (f" --model-name {config.get('model.name')}" if not config.get("model.mock") else " --mock"))
            print("       不同模型的激活不能混用 —— 文件名里带模型标识就是为了防止这个。")
            return EXIT_FAIL
        available = cached["control"].layers
        print(f"复用缓存隐层：{len(available)} 层（{available[0]}~{available[-1]}）")
        if args.layer is not None and int(args.layer) not in available:
            print(f"[FAIL] 缓存里没有第 {args.layer} 层（可用 {available}）")
            return EXIT_FAIL
        extract_results = cached
    print()

    probe_kinds = ["ridge"] + (["mlp"] if args.with_mlp else [])
    result = run_experiment(
        config,
        lists=lists,
        extract_results=extract_results,
        layer=args.layer,
        probe_kinds=probe_kinds,
        n_repeats=int(args.repeats or 5),
        test_size=float(args.test_size or 0.2),
    )

    if args.json:
        print(json.dumps(result.summary_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result, title=args.title or "GeoProbe 实验报告"))

    # ★ 文件名把「模型」与「目标编码」都编进去。
    # 同一个"少一个区分维度"的 bug 已经犯了三次：
    #   ① `run --mock` 与真实结果都写 result.json      → Mock 冲掉真实结果
    #   ② latlon 与 sincos 都写 layer_sweep.json       → 换编码冲掉上一次
    #   ③ 0.5B / 1.5B / 3B 都写 result.sincos.json     → 换模型冲掉上一次
    paths = output_paths(config)
    md_path = write_report(result, paths["report"].parent, args.out or paths["report"].name)
    json_path = save_result(result, paths["result"])
    print()
    print(f"[OK] 报告：{md_path}")
    print(f"[OK] 数据：{json_path}")
    print(f"     模型标识 {paths['model_slug']}　编码 {paths['encoding']}　"
          "（换模型/编码重跑不会覆盖这份结果）")
    if config.get("model.mock"):
        print("     注意：这是 Mock（合成向量）结果，不能代表真实模型表现。")
    return EXIT_OK


def cmd_layer_sweep(args: argparse.Namespace) -> int:
    """逐层扫描：对每一层跑一次实验，看误差随层数的变化。

    **这是 probing 论文的标准图表**：地理知识通常不是均匀分布的，
    而是在某一层附近"涌现" —— 浅层误差接近基线，深层急剧下降。

    ⚠ **实现要点（曾经踩过的坑）**：
    `extract_dataset()` 默认只提取**最后一层**（因为单层实验只需要它）。
    早期版本在循环里对每一层都调用一次 `run_experiment()`，
    而每次都重新提取（且只提最后一层），于是请求第 0 层时报
    ``KeyError: 层 0 不在结果里（已提取 [23]）``。

    正确做法是：**一次性提取全部层，然后在循环里复用** ——
    这也把 24 次重复提取降到 1 次（提速约 24 倍）。
    """
    config = _apply_common(load_config(args.config), args)
    lists = load_wordlists(config.path_of("data"))

    if config.get("model.mock"):
        print("⚠ Mock 模式下只有 1 个合成层，layer-sweep 退化为单点")
        # Mock 时仍然跑一次，便于验证代码路径
        result = run_experiment(config, lists=lists, n_repeats=1)
        print(render_markdown(result, title="GeoProbe（Mock 单层）"))
        return EXIT_OK

    from .extract import ActivationExtractor, extract_dataset

    extractor = ActivationExtractor(config)
    total = extractor.num_layers
    all_layers = list(range(total))
    n_repeats = int(args.repeats or 3)

    # --step 2 可以只扫偶数层，把耗时减半
    step = max(1, int(getattr(args, "step", 1) or 1))
    layers = all_layers[::step]
    if layers[-1] != all_layers[-1]:
        layers.append(all_layers[-1])  # 始终包含最后一层，便于与单层结果对比

    print(f"模型共 {total} 层，将扫描 {len(layers)} 层（重复划分 {n_repeats} 次）")
    print("一次性提取全部层的隐层（这是耗时主要部分）…")

    extract_results = extract_dataset(
        config, lists, layer_indices=all_layers, extractor=extractor
    )

    extracted = extract_results["control"].layers
    missing = [layer for layer in layers if layer not in extracted]
    if missing:
        print(f"⚠ 以下层没有提取到，将跳过：{missing}")
        layers = [layer for layer in layers if layer in extracted]
    if not layers:
        print("[FAIL] 没有可用层")
        return EXIT_FAIL

    # 缓存到磁盘：文件名带**模型标识**，换模型不会互相覆盖
    paths = output_paths(config)
    try:
        for group, extract in extract_results.items():
            extract.save(Path(str(paths["activations"]).replace("{group}", group)))
        print(f"隐层已缓存：{paths['activations'].name.replace('{group}', '<group>')}")
    except Exception as exc:  # noqa: BLE001 - 缓存失败不该中断实验
        print(f"⚠ 缓存失败（不影响实验）：{exc}")

    rows: List[Dict[str, Any]] = []
    for layer in layers:
        result = run_experiment(
            config,
            lists=lists,
            extract_results=extract_results,  # ★ 复用已提取的激活，不再重复提取
            layer=layer,
            n_repeats=n_repeats,
            run_baselines=(layer == layers[-1]),
        )
        rows.append(
            {
                "layer": layer,
                "control_median_km": result.control_repeats.median_mean,
                "control_median_std_km": result.control_repeats.median_std,
                "ambiguous_median_km": result.ambiguous_repeats.median_mean,
                "ratio": result.ambiguity_ratio,
                "r2_lat": result.control.r2_lat,
                "r2_lon": result.control.r2_lon,
                "signal_ok": result.signal_ok,
                "pooled_r2_lat": result.pooled_r2_lat,
                "pooled_r2_lon": result.pooled_r2_lon,
                "pooled_n": result.pooled_n,
            }
        )
        print(
            f"  layer {layer:>2}: control {result.control_repeats.median_mean:>8,.0f} km　"
            f"ambiguous {result.ambiguous_repeats.median_mean:>8,.0f} km　"
            f"汇总R²(lat/lon) {result.pooled_r2_lat:>5.2f}/{result.pooled_r2_lon:>5.2f}"
        )

    out_dir = config.out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ★ 文件名带「模型 + 编码」：换模型或换编码都不会覆盖上一次的扫描
    encoding = str(config.get("probe.target_encoding", "latlon"))
    payload = paths["layer_sweep"]
    document = {
        "_meta": {
            "model": str(config.get("model.name")),
            "pooling": str(config.get("extract.pooling")),
            "target_encoding": encoding,
            "use_instruction": bool(config.get("extract.use_instruction")),
            "n_repeats": n_repeats,
            # 用 getattr 兜底：`--test-size` 曾经只注册在 `run` 子命令上，
            # 于是 `layer-sweep` 走到这一行就 AttributeError 崩掉 ——
            # 而且是在**提取 + 缓存都成功之后**才崩，所以缓存有了、结果文件没写，
            # 现象很难解释（本项目真实踩过：3B 跑完报 AttributeError，
            # 但 activations.qwen2.5-3b-instruct.*.npz 其实是好的）。
            "test_size": float(getattr(args, "test_size", None) or 0.2),
            "n_control": len(lists["control"]),
            "n_ambiguous": len(lists["ambiguous"]),
            "layers_scanned": layers,
            "note": "R² 为「汇总 R²」（所有重复的留出预测拼接后计算），比单次划分稳定得多。",
        },
        "layers": rows,
    }
    payload.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    # 顺手给出"最佳层"，省得人工看表
    best = min(rows, key=lambda row: row["control_median_km"])
    print(f"\n[OK] 逐层结果：{payload}")
    print(f"     模型 {paths['model_slug']}　编码 {encoding}（换模型/编码重跑不会覆盖）")
    print(
        f"最佳层：layer {best['layer']}（control {best['control_median_km']:,.0f} km，"
        f"汇总R²(lat/lon) {best['pooled_r2_lat']:.2f}/{best['pooled_r2_lon']:.2f}）"
    )
    print(
        f"对比：最后一层 layer {rows[-1]['layer']} 的 control 为 "
        f"{rows[-1]['control_median_km']:,.0f} km"
    )
    print("提示：把 control_median_km 对 layer 画折线，就能看到「地理知识在哪一层涌现」。")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoprobe",
        description="LLM 地理知识探针：歧义地名的表征偏移分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", help="配置文件路径（默认 configs/default.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="环境自检")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    p_check = sub.add_parser("check", help="只校验词表数据")
    p_check.set_defaults(func=cmd_check)

    p_models = sub.add_parser("models", help="列出候选模型的内存预估与已缓存模型")
    p_models.set_defaults(func=cmd_models)

    # 表征几何：范数 / 层间旋转 / 有效维度（只用缓存，不需要 lm_head）
    from .geometry_cli import cmd_geometry

    p_geo = sub.add_parser(
        "geometry", help="表征几何：残差流范数、层间旋转、有效维度（替代失效的 logit lens）"
    )
    p_geo.add_argument("--model", nargs="+", help="只分析这些模型 slug（默认全部已缓存）")
    p_geo.set_defaults(func=cmd_geometry)

    # logit lens：检验「深层被下一词预测挤压」这个假设
    from .lens_cli import cmd_logit_lens
    from .lens import LENS_MODES

    p_lens = sub.add_parser(
        "logit-lens", help="把中间层直接投影到词表，检验「深层被下一词预测挤压」"
    )
    p_lens.add_argument("--model-name", help="覆盖配置里的模型名")
    p_lens.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    p_lens.add_argument("--device", choices=["cpu", "cuda"])
    p_lens.add_argument(
        "--mode",
        choices=list(LENS_MODES),
        default="bare",
        help="bare=裸地名（与主实验一致）| prompt=带地理提示的句子",
    )
    p_lens.add_argument("--limit", type=int, help="只用前 N 个词（快速试跑）")
    p_lens.set_defaults(func=cmd_logit_lens)

    for name, func, help_text in (
        ("run", cmd_run, "跑完整实验并生成报告"),
        ("layer-sweep", cmd_layer_sweep, "逐层扫描"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--mock", action="store_true", help="不加载模型，用合成向量验证流程")
        p.add_argument("--pooling", choices=["last_token", "mean", "entity_token"])
        p.add_argument(
            "--target-encoding",
            choices=["latlon", "sincos"],
            help="目标编码：latlon（线性，默认）| sincos（经度循环编码，推荐）",
        )
        p.add_argument("--layer", type=int, help="用第几层（默认最后一层）")
        p.add_argument("--device", choices=["cpu", "cuda"])
        p.add_argument(
            "--dtype",
            choices=["float32", "float16", "bfloat16"],
            help="权重精度。内存不够时用 bfloat16（省一半，CPU 上支持良好）",
        )
        p.add_argument("--model-name", help="覆盖配置里的模型名")
        p.add_argument("--repeats", type=int, help="重复划分次数")
        p.add_argument("--test-size", type=float, default=0.2, help="A 组留出比例")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=func)

    p_run = sub.choices["run"]
    p_run.add_argument("--with-mlp", action="store_true", help="额外跑 MLP 探针对照（较慢）")
    p_run.add_argument(
        "--from-cache",
        action="store_true",
        help="复用 layer-sweep 留下的隐层缓存，跳过模型前向",
    )
    p_run.add_argument("--title", help="报告标题")
    p_run.add_argument("--out", help="报告文件名")

    p_sweep = sub.choices["layer-sweep"]
    p_sweep.add_argument(
        "--step",
        type=int,
        default=1,
        help="层步长（2 表示只扫偶数层，耗时减半；最后一层总会包含）",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return EXIT_FAIL
