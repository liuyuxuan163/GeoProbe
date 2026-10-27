"""整理 reports/ 里"混合了多个模型"的旧文件，按新命名规范重命名。

**背景：这是同一个 bug 的第三次发作**

早先的输出文件名只带「编码方式」，不带「模型标识」，于是：

* `activations.control.npz` —— 0.5B 跑一次、1.5B 又跑一次，**后者覆盖前者**
* `result.sincos.json` —— 同理，0.5B 的结果被 1.5B 覆盖
* `layer_sweep.json` —— 同理

结果就是 reports/ 里混着不同模型的产物，**没人分得清哪个数字是哪个模型的**。

本脚本做两件事：

1. **读出每个文件内部记录的模型名**，按新规范重命名
   （``activations.control.npz`` → ``activations.qwen2.5-1.5b-instruct.control.npz``）
2. 对**无法确定**的字段（比如旧 `layer_sweep.json` 没记录编码方式）如实标注 ``unknown``，
   不猜测 —— 猜错比留空更糟

用法::

    python scripts/migrate_reports.py --dry-run    # 先看会怎么改
    python scripts/migrate_reports.py              # 实际执行
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoprobe.extract import ExtractResult  # noqa: E402


def slugify(model_name: str) -> str:
    """与 config.model_slug 保持一致的 slug 规则。"""
    if model_name.startswith("mock"):
        return "mock"
    slug = model_name.rstrip("/").split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9.\-]+", "-", slug).strip("-")
    return slug or "unknown"


def _plan_activations(reports: Path) -> List[Tuple[Path, Path, str]]:
    """旧命名 ``activations.<group>.npz`` → 带模型标识的新名字。"""
    plans: List[Tuple[Path, Path, str]] = []
    for path in sorted(reports.glob("activations.*.npz")):
        # 新命名已有三段（activations.<slug>.<group>.npz），跳过
        if len(path.stem.split(".")) >= 3:
            continue
        group = path.stem.split(".")[-1]
        try:
            result = ExtractResult.load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {path.name}: 读取失败 ({exc})")
            continue
        slug = slugify(result.model_name)
        target = reports / f"activations.{slug}.{group}.npz"
        plans.append((path, target, f"{result.model_name} ({len(result.layers)} 层)"))
    return plans


def _plan_results(reports: Path) -> List[Tuple[Path, Path, str]]:
    """``result[.mock].json`` → ``result.<slug>.<encoding>.json``。"""
    plans: List[Tuple[Path, Path, str]] = []
    for path in sorted(reports.glob("result*.json")):
        # 已经是新命名（四段以上）或特殊文件，跳过
        parts = path.stem.split(".")
        if len(parts) >= 3 or "recovered" in path.stem:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {path.name}: 读取失败 ({exc})")
            continue
        slug = slugify(str(summary.get("model", "unknown")))
        encoding = summary.get("target_encoding") or "latlon"
        target = reports / f"result.{slug}.{encoding}.json"
        plans.append((path, target, f"{summary.get('model')} layer={summary.get('layer')} enc={encoding}"))
    return plans


def _plan_layer_sweeps(
    reports: Path, assumed_model: Optional[str] = None
) -> List[Tuple[Path, Path, str]]:
    """``layer_sweep.json`` → ``layer_sweep.<slug>.<encoding>.json``。

    旧格式**没有记录模型名和编码方式**。这里：

    * 模型名优先用 ``--sweep-model`` 显式指定（最可靠）
    * 其次从编码对比文件里按"层数相同"推断
    * 都不行 → 标 ``unknown-model``，**不瞎猜**

    编码方式一律标 ``unknown`` —— 旧文件确实没记录，猜错比留空更糟。
    """
    plans: List[Tuple[Path, Path, str]] = []
    for path in sorted(reports.glob("layer_sweep*.json")):
        if len(path.stem.split(".")) >= 3:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {path.name}: 读取失败 ({exc})")
            continue

        rows = data["layers"] if isinstance(data, dict) and "layers" in data else data
        if not isinstance(rows, list):
            print(f"  [SKIP] {path.name}: 结构无法识别")
            continue

        n_layers = max(int(row["layer"]) for row in rows) + 1
        guessed_model = assumed_model
        source = "由 --sweep-model 指定"
        if not guessed_model:
            # 从编码对比文件里找层数能对上的模型
            comparison = reports / "encoding_comparison.json"
            if comparison.exists():
                meta = json.loads(comparison.read_text(encoding="utf-8"))["_meta"]
                if len(meta["layers"]) == n_layers:
                    guessed_model = meta["model"]
                    source = "按层数推断"

        slug = slugify(guessed_model) if guessed_model else "unknown-model"
        target = reports / f"layer_sweep.{slug}.unknown-encoding.json"
        note = (
            f"层数={n_layers}，模型={guessed_model + '（' + source + '）' if guessed_model else '未知'}，"
            "编码=未知（旧文件未记录）"
        )
        plans.append((path, target, note))
    return plans


def _plan_comparisons(reports: Path) -> List[Tuple[Path, Path, str]]:
    plans: List[Tuple[Path, Path, str]] = []
    for path in sorted(reports.glob("encoding_comparison*.json")):
        if len(path.stem.split(".")) >= 2:
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))["_meta"]
        except Exception:  # noqa: BLE001
            continue
        slug = slugify(str(meta.get("model", "unknown")))
        plans.append((path, reports / f"encoding_comparison.{slug}.json", str(meta.get("model"))))
    return plans


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按模型标识重命名 reports/ 里的旧文件")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不改文件")
    parser.add_argument("--copy", action="store_true", help="复制而不是重命名（保守做法）")
    parser.add_argument(
        "--sweep-model",
        help="旧 layer_sweep.json 属于哪个模型（旧文件没记录，需人工指定，如 Qwen/Qwen2.5-0.5B-Instruct）",
    )
    args = parser.parse_args(argv)

    reports = ROOT / "reports"
    if not reports.exists():
        print(f"[FAIL] 没有 {reports} 目录")
        return 1

    all_plans: List[Tuple[Path, Path, str]] = []
    for name, planner in (
        ("激活缓存", _plan_activations),
        ("单层结果", _plan_results),
        ("逐层扫描", lambda rep: _plan_layer_sweeps(rep, args.sweep_model)),
        ("编码对比", _plan_comparisons),
    ):
        print(f"\n== {name} ==")
        plans = planner(reports)
        if not plans:
            print("  （无需整理）")
        for source, target, note in plans:
            if target.exists():
                mark = "  ⚠ 目标已存在，将跳过"
            else:
                mark = ""
            print(f"  {source.name}")
            print(f"      → {target.name}{mark}")
            print(f"        {note}")
        all_plans.extend(plans)

    # 目标已存在的（内容重复）从计划里剔除，不覆盖
    skipped = [(s, t) for s, t, _n in all_plans if t.exists()]
    all_plans = [(s, t, n) for s, t, n in all_plans if not t.exists()]
    if skipped:
        print(f"\n跳过 {len(skipped)} 个（目标已存在，视为重复产物）：")
        for source, target in skipped:
            print(f"  {source.name} → {target.name}")

    if not all_plans:
        print("\n没有需要整理的文件。")
        return 0

    print(f"\n共 {len(all_plans)} 个文件待整理。")
    if args.dry_run:
        print("DRY-RUN：未改动任何文件。确认无误后去掉 --dry-run 再跑一次。")
        return 0

    conflict = [target for _src, target, _note in all_plans if target.exists()]
    if conflict:
        print(f"[FAIL] 有 {len(conflict)} 个目标文件已存在，拒绝覆盖：")
        for path in conflict:
            print(f"  {path.name}")
        print("请先手动处理这些冲突。")
        return 1

    for source, target, _note in all_plans:
        if args.copy:
            shutil.copy2(source, target)
        else:
            source.rename(target)

    print(f"\n[OK] 已{'复制' if args.copy else '重命名'} {len(all_plans)} 个文件。")
    if not args.copy:
        print("     原文件已不存在 —— 现在每个文件名都带模型标识了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
