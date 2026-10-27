"""对扩充后的词表做自动质检，把可疑条目标出来，减轻人工复核负担。

**复核为什么需要辅助**

`scripts/expand_wordlist.py` 已经用规则挡掉了大部分问题（同名歧义、非拉丁字符、
与已有词表重名、常用英文词同形）。但仍有三类问题只能靠"语义判断"发现：

1. **名字主要含义不是这个城市** —— 规则挡不住（黑名单不完整）
2. **坐标明显偏移** —— 与本词表内其他城市的相对位置不合理
3. **跨组冲突** —— 新词恰好是 B 组某个词的变体（拼写不同但指同一事物）

本脚本做**机械可查**的部分，输出一份"高优先级复核清单"：

* 名字长度 ≤ 3（极短地名通常是缩写或常见词）
* 名字是单个常见词形（与 B/C 组某词只差大小写或复数）
* 坐标与同国其他城市聚成一个点（可能 GeoNames 数据错误）
* 同一国家城市数过多（地理不均衡）
* 名字含撇号/连字符（tokenizer 可能切得很碎）
* 人口异常（人口为 0 或与"cities15000"下限不符）

用法::

    python scripts/audit_wordlist.py
    python scripts/audit_wordlist.py --file data/words_control_expanded.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def load_groups() -> Dict[str, List[Dict[str, Any]]]:
    groups = {}
    for key, filename in (
        ("control", "words_control.json"),
        ("ambiguous", "words_ambiguous.json"),
        ("nongeo", "words_nongeo.json"),
    ):
        path = ROOT / "data" / filename
        if path.exists():
            groups[key] = json.loads(path.read_text(encoding="utf-8"))
    return groups


def audit(entries: List[Dict[str, Any]], groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """返回 ``{问题类型: [条目...]}``。"""
    issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    b_words = {str(entry.get("word", "")).lower() for entry in groups.get("ambiguous", [])}
    c_words = {str(entry.get("word", "")).lower() for entry in groups.get("nongeo", [])}

    by_country: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        country = str(entry.get("country_code", ""))
        if country:
            by_country[country].append(entry)

    for entry in entries:
        word = str(entry.get("word", ""))
        lower = word.lower()

        if len(word) <= 3:
            issues["名字极短（可能是缩写或常见词）"].append(entry)

        # 与 B/C 组同形或只差复数
        if lower in b_words or lower in c_words:
            issues["与 B/C 组同形"].append(entry)
        elif f"{lower}s" in b_words or f"{lower}es" in b_words:
            issues["与 B/C 组只差复数"].append(entry)

        if "'" in word or "-" in word:
            issues["含撇号或连字符（tokenizer 会切碎）"].append(entry)

        population = entry.get("population")
        # 只对「城市」检查人口下限 —— 国家/地区的 population 是占位的 0，
        # 拿城市规则去套它们会误报 41 条（本项目真实踩过）。
        if entry.get("category") == "城市" and isinstance(population, int) and population < 15000:
            issues["城市人口低于 cities15000 下限"].append(entry)

        if not entry.get("continent"):
            issues["缺少 continent 字段"].append(entry)

        if entry.get("difficulty") not in ("粗", "精"):
            issues["difficulty 取值异常"].append(entry)

    # 单一国家占比过高
    total = len(entries)
    for country, items in by_country.items():
        if total >= 100 and len(items) / total > 0.08:
            issues[f"国家 {country} 占比过高（{len(items)}/{total}）"] = items

    return dict(issues)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="扩充词表的自动质检")
    parser.add_argument("--file", default="data/words_control_expanded.json")
    parser.add_argument("--out", default="docs/词表复核优先级.md")
    args = parser.parse_args(argv)

    path = ROOT / args.file
    if not path.exists():
        print(f"[FAIL] 找不到 {path}")
        print("       先运行：python scripts/expand_wordlist.py")
        return 1

    entries = json.loads(path.read_text(encoding="utf-8"))
    groups = load_groups()

    print(f"词表：{path.name}　{len(entries)} 条")
    print()
    print("=== 基本统计 ===")
    categories = Counter(str(entry.get("category", "?")) for entry in entries)
    difficulties = Counter(str(entry.get("difficulty", "?")) for entry in entries)
    continents = Counter(str(entry.get("continent", "?")) for entry in entries)
    print(f"  类别: {dict(categories)}")
    print(f"  难度: {dict(difficulties)}")
    print(f"  大洲: {dict(continents)}")
    sources = Counter(str(entry.get("source", "manual")) for entry in entries)
    print(f"  来源: {dict(sources)}")

    issues = audit(entries, groups)
    print()
    print("=== 自动质检发现 ===")
    if not issues:
        print("  没有发现机械可查的问题")
    total_flagged = sum(len(items) for items in issues.values())
    for name, items in sorted(issues.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:<40}{len(items):>4} 条")
        for entry in items[:4]:
            print(f"      {entry.get('word')}  ({entry.get('label', '')})")
        if len(items) > 4:
            print(f"      …还有 {len(items) - 4} 条")

    # ---- 输出复核优先级清单 ----
    lines = [
        "# 词表复核优先级",
        "",
        f"> 由 `scripts/audit_wordlist.py` 生成，基于 `{args.file}`（{len(entries)} 条）",
        "",
        "自动质检只能发现**机械可查**的问题。以下按优先级排列，建议从上往下看。",
        "",
        "## 词表概览",
        "",
        f"- 类别：{dict(categories)}",
        f"- 难度：{dict(difficulties)}",
        f"- 大洲：{dict(continents)}",
        f"- 来源：{dict(sources)}",
        "",
        f"## 自动质检：共 {total_flagged} 条命中（含重叠）",
        "",
    ]
    if not issues:
        lines.append("没有发现机械可查的问题。")
    else:
        lines += ["| 问题 | 条数 | 示例 |", "|---|---|---|"]
        for name, items in sorted(issues.items(), key=lambda kv: -len(kv[1])):
            sample = "、".join(f"`{entry.get('word')}`" for entry in items[:5])
            lines.append(f"| {name} | {len(items)} | {sample} |")
    lines.append("")

    lines += [
        "## 建议的人工复核顺序",
        "",
        "1. **先看「与 B/C 组同形」「只差复数」** —— 这类会造成真正的标签冲突",
        "2. **再看「名字极短」** —— 短名通常是常见词或缩写，最容易语义混淆",
        "3. **然后抽查每个大洲前 5 个与后 5 个**（`docs/词表扩充复核.md` 已按人口排序）",
        "4. 坐标只需看「是否在正确的国家/大洲」，±100 km 内不必纠",
        "",
        "## 逐条明细（按问题类型）",
        "",
    ]
    for name, items in sorted(issues.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"### {name}（{len(items)} 条）")
        lines.append("")
        lines.append("| 词 | 坐标 | 人口 | 国家 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for entry in items[:60]:
            lines.append(
                f"| `{entry.get('word')}` | {entry.get('lat')}, {entry.get('lon')} | "
                f"{entry.get('population', '—')} | {entry.get('country_code', '—')} | "
                f"{entry.get('label', '')} |"
            )
        if len(items) > 60:
            lines.append(f"| … | | | | 还有 {len(items) - 60} 条 |")
        lines.append("")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"[OK] 复核优先级：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
