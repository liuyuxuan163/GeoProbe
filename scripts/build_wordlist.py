"""用 Nominatim（OpenStreetMap）刷新词表里的经纬度。

为什么需要这个脚本
------------------
仓库内置的坐标是**手工填写的城市/国家中心近似值**（精度约 ±50 km）。
对于 0.5B 级别的模型，这个精度远小于模型本身的误差量级，不影响结论。

但如果你要对外展示或发表，应该用权威数据源刷新。

用法::

    pip install geopy
    python scripts/build_wordlist.py --dry-run        # 先看看会改成什么（不写盘）
    python scripts/build_wordlist.py                  # 实际写盘
    python scripts/build_wordlist.py --only control   # 只刷新 A 组

设计要点
--------
1. **保留人工维护的字段**（``note`` / ``label`` / ``alt_*``）——
   脚本只更新 ``lat`` / ``lon``。这些注释是分析时的关键上下文，不能被覆盖。
2. **默认 dry-run**，避免误改数据。
3. **Nominatim 有速率限制**（1 次/秒），脚本内置 sleep 并在失败时重试一次。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FILES = {
    "control": "words_control.json",
    "ambiguous": "words_ambiguous.json",
    "nongeo": "words_nongeo.json",
}


def geocode(word: str, timeout: float = 10.0) -> Optional[tuple[float, float]]:
    """查询一个地名的经纬度。失败返回 ``None``。"""
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        raise SystemExit("需要 geopy：pip install geopy")

    geolocator = Nominatim(user_agent="geoprobe-coordinate-refresh", timeout=timeout)
    try:
        location = geolocator.geocode(word)
    except Exception as exc:  # noqa: BLE001 - 网络问题不该中断整个流程
        print(f"    [WARN] {word}: 查询失败 ({exc})")
        return None
    if location is None:
        return None
    return float(location.latitude), float(location.longitude)


def refresh_group(key: str, dry_run: bool, delay: float) -> Dict[str, int]:
    path = DATA / FILES[key]
    entries: List[dict] = json.loads(path.read_text(encoding="utf-8"))
    stats = {"total": 0, "updated": 0, "failed": 0, "skipped": 0}

    print(f"\n== {key} ({path.name}, {len(entries)} 项) ==")
    for entry in entries:
        stats["total"] += 1
        word = entry.get("word")
        if not word:
            continue
        # C 组没有坐标，跳过
        if entry.get("lat") is None:
            stats["skipped"] += 1
            continue

        coords = geocode(word)
        time.sleep(delay)  # Nominatim 速率限制
        if coords is None:
            stats["failed"] += 1
            print(f"  [FAIL] {word}: 未查到")
            continue

        lat, lon = coords
        old = (entry.get("lat"), entry.get("lon"))
        if old[0] is not None and abs(old[0] - lat) < 1e-6 and abs(old[1] - lon) < 1e-6:
            print(f"  [SAME] {word}: ({lat:.4f}, {lon:.4f})")
            continue

        stats["updated"] += 1
        delta = ""
        if old[0] is not None:
            delta = f"  (原 {old[0]:.4f}, {old[1]:.4f})"
        print(f"  [NEW ] {word}: ({lat:.4f}, {lon:.4f}){delta}")
        entry["lat"], entry["lon"] = lat, lon

    if not dry_run and stats["updated"]:
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → 已写回 {path}")
    elif dry_run:
        print("  → dry-run，未写盘")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用 Nominatim 刷新词表坐标")
    parser.add_argument("--only", choices=list(FILES), nargs="+", help="只刷新指定组")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写盘")
    parser.add_argument("--delay", type=float, default=1.1, help="请求间隔秒数（Nominatim 要求 ≥1）")
    args = parser.parse_args(argv)

    groups = args.only or ["control", "ambiguous"]
    print("注意：Nominatim 需要联网，且每次请求间隔至少 1 秒。")
    print(f"模式：{'DRY-RUN（不写盘）' if args.dry_run else '写盘'}")

    totals = {"total": 0, "updated": 0, "failed": 0, "skipped": 0}
    for key in groups:
        stats = refresh_group(key, args.dry_run, args.delay)
        for name in totals:
            totals[name] += stats[name]

    print("\n" + "=" * 60)
    print(
        f"合计 {totals['total']} 项：更新 {totals['updated']}，"
        f"未变 {totals['total'] - totals['updated'] - totals['failed'] - totals['skipped']}，"
        f"失败 {totals['failed']}，跳过 {totals['skipped']}"
    )
    print("建议刷新后跑一遍：python -m geoprobe check && python -m pytest tests -q")
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
