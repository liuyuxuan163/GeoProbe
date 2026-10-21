"""用真实实验的结果自动更新 README 里的「实测结果」段落。

为什么需要这个脚本
------------------
README 里的数字如果靠手抄，**必然漂移**：换了模型、换了层、换了池化方式之后，
文档里的数字就成了假的。而简历/面试里被追问"这个数字怎么来的"时，
数字对不上代码是最伤可信度的情况。

这个脚本把 README 的「实测结果」区段变成**由 result.json 生成的产物**：

    python scripts/update_readme.py                 # 用 reports/result.json 更新
    python scripts/update_readme.py --dry-run       # 只看会改成什么
    python scripts/update_readme.py --result other.json

**安全设计**：默认**拒绝**用 Mock 模式的结果写进 README ——
因为 Mock 是合成向量，把它写进 README 会让人误以为是真实模型的成绩。
真要写必须显式加 `--allow-mock`。

实现要点
--------
README 里用两个 HTML 注释做标记，脚本只替换它们之间的内容::

    <!-- AUTO-RESULTS:START -->
    ...（本脚本生成，勿手改）...
    <!-- AUTO-RESULTS:END -->

这样**幂等** —— 跑多少次结果一样，也不会破坏手写的其他章节。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

START = "<!-- AUTO-RESULTS:START -->"
END = "<!-- AUTO-RESULTS:END -->"
NOTE = "<!-- 本段由 scripts/update_readme.py 从 reports/result.json 自动生成，请勿手改 -->"

# 「偏向判定」的可靠性门槛（km）。与 configs/default.yaml 的
# evaluation.alt_reliability_km 保持一致；用于兼容没有 reliable 字段的旧结果文件。
ALT_RELIABILITY_KM = 2500.0


def _fmt(value: Optional[float], unit: str = "km") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    if unit == "":
        return f"{number:.2f}"
    return f"{number:,.0f} {unit}".strip()


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> str:
    if not numerator or not denominator:
        return "—"
    return f"{numerator / denominator:.1f}×"


def diagnose(payload: Dict[str, Any]) -> List[str]:
    """**由数据推出**关键发现，而不是写死一段话。"""
    summary = payload.get("summary", {})
    control = payload.get("control", {})
    repeats = payload.get("repeats", {})
    baselines = summary.get("baselines", {})
    findings: List[str] = []

    ctrl = repeats.get("control", {}).get("median_mean_km") or summary.get("control_median_km")
    permuted = baselines.get("permuted", {}).get("median_km")
    char_base = baselines.get("char_ngram", {}).get("median_km")

    # ① 探针是否有效 + 强度
    if summary.get("signal_ok") and permuted and ctrl:
        strength = permuted / ctrl
        grade = "强" if strength >= 4 else ("中等" if strength >= 2.5 else "勉强通过")
        findings.append(
            f"**探针有效（{grade}）**：中位误差 {_fmt(ctrl)}，比置换检验"
            f"（打乱坐标，{_fmt(permuted)}）好 **{strength:.1f} 倍**。"
            + (
                "倍数偏低说明信号较弱，结论应表述为「存在但很弱」。"
                if strength < 2.5
                else ""
            )
        )
    else:
        findings.append(
            f"**⚠ 探针未通过置换检验**：中位误差 {_fmt(ctrl)}，与打乱坐标"
            f"（{_fmt(permuted)}）没有拉开差距。此时**不能**把 B 组结果解读为歧义效应。"
        )

    # ② 与拼写基线的对比（核心论证）
    if char_base and ctrl:
        if ctrl < char_base * 0.85:
            findings.append(
                f"**优于拼写基线**：字符 n-gram 基线为 {_fmt(char_base)}，"
                f"隐层为 {_fmt(ctrl)}（{char_base / ctrl:.1f} 倍更准）—— "
                "排除了「只是记住拼写规律」的解释。"
            )
        else:
            findings.append(
                f"**⚠ 未明显优于拼写基线**（char n-gram {_fmt(char_base)} vs 隐层 {_fmt(ctrl)}）—— "
                "当前结果可能主要来自地名拼写与地区的相关性，不能断言模型编码了地理知识。"
            )

    # ③ 分维度 R²：这是"分维度报告"的直接回报
    #
    # ★ 优先用汇总值 + 圆形 R²：
    #   * 单次划分的 R² 抽样噪声可达 ±0.5（同一层换种子就能从 0.01 跳到 0.55）
    #   * 线性 R²(经度) 对循环量无效（179° 与 -179° 地理相接，线性残差却是 358°）
    pooled = payload.get("pooled") or {}
    r2_lat = pooled.get("r2_lat", control.get("r2_lat"))
    r2_lon = pooled.get("circular_r2_lon", control.get("r2_lon"))
    r2_source = "汇总圆形 R²" if pooled.get("circular_r2_lon") is not None else "单次划分 R²"
    pooled_n = pooled.get("n")
    encoding = summary.get("target_encoding", "latlon")
    if r2_lat is not None and r2_lon is not None:
        sample_note = f"（n={pooled_n}）" if pooled_n else ""
        if r2_lat > 0.2 and r2_lon < r2_lat * 0.5:
            findings.append(
                f"**纬度比经度更可解码** —— R²(纬度)={r2_lat:.2f}，"
                f"圆形 R²(经度)={r2_lon:.2f}{sample_note}。"
                "模型对南北位置的表征强于东西位置。"
            )
        elif r2_lon >= r2_lat * 0.8:
            findings.append(
                f"**两个方向都可解码** —— R²(纬度)={r2_lat:.2f}，"
                f"圆形 R²(经度)={r2_lon:.2f}{sample_note}。"
                "经度并没有比纬度更差，说明模型编码了**二维**的地理结构。"
            )
        else:
            findings.append(
                f"**分维度 {r2_source}**：纬度 {r2_lat:.2f}，经度 {r2_lon:.2f}{sample_note}。"
            )
        # 编码方式是解读这条发现的前提，必须写出来
        if encoding == "latlon":
            findings.append(
                "**⚠ 本次用的是 `latlon`（经度线性）编码** —— 经度是循环量，"
                "线性目标对它有偏（179° 与 -179° 在目标空间相差 358°，实际只差 2°），"
                "**所以经度指标可能被低估**。建议用 `--target-encoding sincos` 重跑对比。"
            )

    # ④ 误差分布的形状
    median_km = control.get("median_km")
    mean_km = control.get("mean_km")
    p90 = control.get("p90_km")
    if median_km and mean_km:
        skew = mean_km / median_km
        tail = f"P90 = {_fmt(p90)}" if p90 else ""
        findings.append(
            f"**误差是长尾分布**：中位 {_fmt(median_km)}、均值 {_fmt(mean_km)}"
            f"（均值/中位 = {skew:.1f}），{tail}。"
            "说明大部分词能定到大致区域，但有少数被放到了地球另一端。"
        )

    # ⑤ 歧义效应
    ratio = summary.get("ambiguity_ratio")
    ambiguous_median = repeats.get("ambiguous", {}).get("median_mean_km")
    if ratio:
        if ratio >= 1.3:
            verdict = "歧义**显著**放大了误差"
        elif ratio >= 1.15:
            verdict = "歧义**轻微**放大误差，但幅度与噪声同量级，**证据不足**"
        else:
            verdict = "歧义**无明显影响**"
        findings.append(
            f"**歧义效应 {ratio:.2f}×**：歧义词中位误差 {_fmt(ambiguous_median)}，"
            f"对照组 {_fmt(ctrl)} —— {verdict}。"
        )

    # ⑥ 含义偏好（只统计"可靠"的判定）
    # 门槛优先用结果文件里的 alt_reliability_used（自适应：max(下限, 模型自身中位误差)）；
    # 老文件没有这个字段时，回落到常量 + 兼容没有 reliable 字段的情况。
    alt = payload.get("alt_analysis") or []
    if alt:
        threshold = float(payload.get("alt_reliability_used") or ALT_RELIABILITY_KM)

        def _min_km(row: Dict[str, Any]) -> float:
            if "min_km" in row:
                return float(row["min_km"])
            return min(float(row["dist_primary_km"]), float(row["dist_alt_km"]))

        reliable = [
            row for row in alt if row.get("reliable", _min_km(row) <= threshold)
        ]
        closer_alt = [row for row in reliable if row.get("closer_to") == "alt"]

        if not reliable:
            findings.append(
                f"**含义偏好：无法判定。** {len(alt)} 个带备选含义的歧义词中，"
                f"**没有一个**的最近候选落在 {_fmt(threshold)} 内 —— "
                "两个候选都离预测点很远，「谁更近」只是噪声。"
            )
        else:
            findings.append(
                f"**含义偏好**：{len(alt)} 个带备选含义的歧义词中，"
                f"**{len(reliable)} 个**的最近候选落在 {_fmt(threshold)} 内（判定可靠），"
                f"其中 **{len(closer_alt)} 个**更接近备选含义"
                + (
                    f"（{'、'.join(r['word'] for r in closer_alt[:5])} 等）"
                    if closer_alt
                    else ""
                )
                + "。"
                + (
                    f"其余 {len(alt) - len(reliable)} 个因两边都太远而被排除 —— "
                    "不加这个门槛会把噪声当成结论。"
                    if len(alt) > len(reliable)
                    else ""
                )
            )

    return findings


def render_section(payload: Dict[str, Any]) -> str:
    """生成 README 的实测结果段落。"""
    summary = payload.get("summary", {})
    control = payload.get("control", {})
    ambiguous = payload.get("ambiguous", {})
    repeats = payload.get("repeats", {})
    baselines = summary.get("baselines", {})

    ctrl_median = repeats.get("control", {}).get("median_mean_km") or summary.get("control_median_km")
    ctrl_std = repeats.get("control", {}).get("median_std_km")
    amb_median = repeats.get("ambiguous", {}).get("median_mean_km") or summary.get("ambiguous_median_km")
    n_repeats = repeats.get("control", {}).get("n_repeats", 1)

    lines: List[str] = [
        START,
        NOTE,
        "",
        f"**实测环境**：`{summary.get('model')}`　"
        f"池化 `{summary.get('pooling')}`　目标编码 `{summary.get('target_encoding', 'latlon')}`　"
        f"层 `{summary.get('layer')}`　重复划分 {n_repeats} 次",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　"
        f"（运行 `python scripts/update_readme.py` 可重新生成本段）",
        "",
        "### 主结果（Haversine 中位误差）",
        "",
        "| 组 | n | 中位误差 | 均值 | P90 | 95% CI | 说明 |",
        "|---|---|---|---|---|---|---|",
        (
            f"| **A · 对照（留出集）** | {control.get('n', '—')} | "
            f"**{_fmt(ctrl_median)}** | {_fmt(control.get('mean_km'))} | "
            f"{_fmt(control.get('p90_km'))} | "
            f"[{_fmt((control.get('ci') or [None])[0])}, {_fmt((control.get('ci') or [None, None])[1])}] | "
            f"探针能力上限（重复 std {_fmt(ctrl_std)}） |"
        ),
        (
            f"| **B · 歧义地名** | {ambiguous.get('n', '—')} | "
            f"**{_fmt(amb_median)}** | {_fmt(ambiguous.get('mean_km'))} | "
            f"{_fmt(ambiguous.get('p90_km'))} | "
            f"[{_fmt((ambiguous.get('ci') or [None])[0])}, {_fmt((ambiguous.get('ci') or [None, None])[1])}] | "
            f"**{summary.get('ambiguity_ratio', 0):.2f}×** 于对照组 |"
        ),
        "",
        "### 基线对比（A 组留出集）",
        "",
        "| 特征来源 | 中位误差 | 相对隐层 | 意义 |",
        "|---|---|---|---|",
    ]

    descriptions = [
        ("hidden", "模型隐层（本项目主张）", "—"),
        ("permuted", "★ 置换检验（打乱坐标）", "判别探针是否真读到信息"),
        ("char_ngram", "字符 n-gram TF-IDF", "★ 排除「只记住拼写」"),
        ("random", "随机向量", "下界"),
        ("mean", "永远预测训练集均值", "R²=0 的参照"),
        ("mlp", "隐层 + MLP 探针", "信息是否非线性编码"),
    ]
    hidden_median = baselines.get("hidden", {}).get("median_km") or ctrl_median
    for key, label, meaning in descriptions:
        if key not in baselines and key != "hidden":
            continue
        value = baselines.get(key, {}).get("median_km") or (ctrl_median if key == "hidden" else None)
        lines.append(f"| {label} | {_fmt(value)} | {_ratio(value, hidden_median)} | {meaning} |")

    lines += ["", "### 关键发现", ""]
    for finding in diagnose(payload):
        lines.append(f"- {finding}")

    lines += ["", END]
    return "\n".join(lines)


def update_readme(readme_path: Path, section: str, dry_run: bool = False) -> bool:
    """把 section 写进 README 的标记之间（不存在标记则追加）。"""
    text = readme_path.read_text(encoding="utf-8")

    if START in text and END in text:
        head = text.split(START, 1)[0]
        tail = text.split(END, 1)[1]
        new_text = f"{head}{section}{tail}"
    else:
        # 没有标记：在「## 实验设计」之前插入（找不到就追加到末尾）
        anchor = "## 实验设计"
        if anchor in text:
            head, tail = text.split(anchor, 1)
            new_text = f"{head}## 实测结果\n\n{section}\n\n---\n\n{anchor}{tail}"
        else:
            new_text = f"{text.rstrip()}\n\n## 实测结果\n\n{section}\n"

    if new_text == text:
        print("[INFO] README 内容未变化（结果与现有段落一致）")
        return False

    if dry_run:
        print("=== DRY-RUN：以下内容将替换 README 中标记之间的部分 ===")
        print(section)
        print("=== 结束 ===")
        return True

    readme_path.write_text(new_text, encoding="utf-8")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="用真实实验结果更新 README 的实测段落")
    parser.add_argument("--result", default="reports/result.json", help="结果 JSON 路径")
    parser.add_argument("--readme", default="README.md", help="README 路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    parser.add_argument("--allow-mock", action="store_true", help="允许用 Mock 结果更新（不推荐）")
    args = parser.parse_args(argv)

    result_path = Path(args.result)
    if not result_path.is_absolute():
        result_path = ROOT / result_path

    # 文件名现在带编码后缀（result.latlon.json / result.sincos.json），
    # 默认参数 result.json 可能不存在 —— 这时自动挑最新的一份真实结果。
    if not result_path.exists() and args.result == "reports/result.json":
        candidates = sorted(
            (ROOT / "reports").glob("result.*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        real = [
            path
            for path in candidates
            if ".mock" not in path.name and "recovered" not in path.name
        ]
        if real:
            result_path = real[0]
            print(f"[INFO] 自动选用最新结果文件：{result_path.name}")

    readme_path = Path(args.readme)
    if not readme_path.is_absolute():
        readme_path = ROOT / readme_path

    if not result_path.exists():
        print(f"[FAIL] 找不到结果文件：{result_path}")
        print("       请先运行：python -m geoprobe run")
        return 1
    if not readme_path.exists():
        print(f"[FAIL] 找不到 README：{readme_path}")
        return 1

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})

    if summary.get("mock") and not args.allow_mock:
        print("[FAIL] 这份结果是 Mock 模式（合成向量）产生的，拒绝写入 README。")
        print("       原因：Mock 数字会让读者误以为是真实模型的成绩。")
        print("       修复：用真实模型跑一次 ——")
        print("             python -m geoprobe run --repeats 5")
        print("       或者，如果你确实要写：加 --allow-mock")
        return 2

    if summary.get("mock"):
        print("[WARN] 正在把 Mock 结果写入 README（--allow-mock 已显式指定）")

    section = render_section(payload)
    changed = update_readme(readme_path, section, dry_run=args.dry_run)

    if changed and not args.dry_run:
        print(f"[OK] 已更新 {readme_path.relative_to(ROOT) if readme_path.is_relative_to(ROOT) else readme_path}")
        print(f"     模型 {summary.get('model')}　层 {summary.get('layer')}")
        print(f"     对照 {_fmt(summary.get('control_median_km'))}　歧义 {_fmt(summary.get('ambiguous_median_km'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
