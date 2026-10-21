"""报告生成：把实验数字变成可读的结论。

为什么报告要自动生成
--------------------
手工抄数字一定会漂移（改了代码忘了改报告）。
而「结论」部分最容易写错 —— 如果换了个模型或换了层，手写的结论就成了假的。

所以这里**所有结论都由结果数据推出**，只保留少量模板文字。

引号约定：本模块内层引用统一用中文引号「」，
避免与 Python 的双引号字符串冲突（这是个真实的踩坑点：
内层用 ASCII 双引号嵌在双引号 f-string 里会直接语法错误）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .experiment import ExperimentResult

__all__ = ["render_markdown", "write_report", "render_conclusion"]


def _fmt_km(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:,.0f}"


def render_conclusion(result: ExperimentResult) -> List[str]:
    """**由数据推出结论**，而不是写死一段话。"""
    lines: List[str] = []
    ctrl = result.control_repeats.median_mean
    amb = result.ambiguous_repeats.median_mean
    mean_base = result.baselines.get("mean", {}).get("median_km")
    char_base = result.baselines.get("char_ngram", {}).get("median_km")

    # ① 探针是否学到东西
    permuted = result.baselines.get("permuted", {}).get("median_km")
    if result.signal_ok and mean_base:
        extra = ""
        if permuted:
            extra = (
                f"置换检验也通过：打乱坐标后误差升至 {_fmt_km(permuted)} km"
                f"（约 {permuted / ctrl:.1f} 倍）。"
            )
        lines.append(
            f"**探针有效。** 无歧义地名的留出集上，Ridge 线性探针的中位误差为 "
            f"**{_fmt_km(ctrl)} km**，明显优于「永远预测训练集均值」的基线"
            f"（{_fmt_km(mean_base)} km）。{extra}"
            "这说明**地理坐标信息在所选层的表征中是线性可解码的**。"
        )
    else:
        lines.append(
            f"**⚠ 探针未通过置换检验。** 留出集中位误差 {_fmt_km(ctrl)} km，"
            f"与打乱坐标后的结果（{_fmt_km(permuted)} km）没有拉开明显差距。"
            "可能原因：层选得太浅、池化方式不合适，或模型规模太小。"
            "**此时不应把 B 组结果解读为「歧义效应」。**"
        )

    # ② 与字符基线的对比（最关键的一条论证）
    if char_base is not None and not np.isnan(char_base):
        if result.signal_ok and ctrl < char_base * 0.85:
            ratio = char_base / ctrl if ctrl > 0 else float("nan")
            lines.append(
                f"**隐层优于拼写基线。** 只用字符 n-gram 特征的探针中位误差为 {_fmt_km(char_base)} km，"
                f"而隐层探针为 {_fmt_km(ctrl)} km（约 {ratio:.1f} 倍更准）。"
                "这排除了「模型只是记住拼写规律」的解释 —— 地理信息**确实编码在表征里**。"
            )
        elif result.signal_ok:
            lines.append(
                f"**⚠ 隐层未明显优于拼写基线。** 字符 n-gram 基线 {_fmt_km(char_base)} km，"
                f"隐层 {_fmt_km(ctrl)} km。这说明当前结果**可能主要来自地名拼写与地区的相关性**，"
                "而不能断言模型编码了地理知识。这是一个需要如实报告的反例。"
            )

    # ③ 歧义效应
    if result.signal_ok and not np.isnan(result.ambiguity_ratio):
        lines.append(
            f"**歧义效应：** 歧义地名（Georgia / Turkey / Chile …）的中位误差为 "
            f"**{_fmt_km(amb)} km**，是对照组的 **{result.ambiguity_ratio:.2f} 倍**"
            f"（{result.ambiguity_comment}）。"
        )

    # ③b 分维度可解码性（用**汇总**值 + 经度专用**圆形 R²**）
    if not np.isnan(result.pooled_r2_lat) and not np.isnan(result.pooled_circular_r2_lon):
        lines.append(
            f"**两个坐标方向的可解码性：** R²(纬度) = **{result.pooled_r2_lat:.2f}**，"
            f"圆形 R²(经度) = **{result.pooled_circular_r2_lon:.2f}**"
            f"（汇总值，n={result.pooled_n}）。"
            + (
                "经度用圆形 R² 是必须的 —— 经度是循环量，普通线性 R² 会把"
                "「真实 -179°、预测 +179°」这种只差 2° 的情况记成 358° 的巨大残差。"
                "本项目实测：线性 R²(经度) 在这里不具可比性，仅作对照。"
            )
        )

    # ④ 偏向哪个含义（只统计"可靠的"判定）
    if result.alt_analysis:
        n_alt = len(result.alt_analysis)
        reliable = [row for row in result.alt_analysis if row.get("reliable")]
        n_reliable = len(reliable)
        closer_alt = [row for row in reliable if row["closer_to"] == "alt"]

        if n_reliable == 0:
            lines.append(
                f"**含义偏好：无法判定。** {n_alt} 个带备选含义的歧义词中，"
                "**没有一个**的最近候选落在可靠性门槛内 —— 也就是两个候选都离预测点很远，"
                "「谁更近」只是噪声，不能解读为含义偏好。"
            )
        elif closer_alt:
            names = "、".join(row["word"] for row in closer_alt[:5])
            lines.append(
                f"**含义偏好：** 在 {n_reliable} 个**可靠判定**的歧义词中"
                f"（最近候选 ≤ 门槛），有 **{len(closer_alt)} 个**更接近**备选含义**"
                f"（{names} 等）。这说明对这些词而言，"
                "**模型的表征被语料中更常见的那个含义主导**。"
                f"（另有 {n_alt - n_reliable} 个因候选都太远而被排除。）"
            )
        else:
            lines.append(
                f"**含义偏好：** 在 {n_reliable} 个**可靠判定**的歧义词中，"
                "**全部**的预测落点更接近主含义。"
                "这说明模型为这些歧义词保留了稳定的地理表征，"
                "没有被更常见的非地理/他国含义拉走。"
                f"（另有 {n_alt - n_reliable} 个因候选都太远而被排除。）"
            )

    # ⑤ 负对照
    if result.nongeo_predictions:
        lats = [p["pred_lat"] for p in result.nongeo_predictions]
        lons = [p["pred_lon"] for p in result.nongeo_predictions]
        lines.append(
            f"**负对照（非地名）：** 非地名词的预测落点分布为 "
            f"纬度 {np.mean(lats):.1f}±{np.std(lats):.1f}、经度 {np.mean(lons):.1f}±{np.std(lons):.1f}。"
            "它们没有真实坐标，因此只作定性检查 —— 若落点高度集中，"
            "说明探针在「没有信息时」会退化输出，符合预期。"
        )

    return lines


def render_markdown(result: ExperimentResult, title: str = "GeoProbe 实验报告") -> str:
    lines: List[str] = [f"# {title}", ""]

    # -- 配置 --
    mock_note = "（Mock 合成向量）" if result.mock else ""
    lines += [
        "## 配置",
        "",
        f"- 模型：`{result.model_name}`{mock_note}",
        f"- 池化方式：`{result.pooling}`",
        f"- 目标编码：`{result.target_encoding}`"
        + (
            "（经度线性，对循环量有偏 —— 仅用于与旧结果对比）"
            if result.target_encoding == "latlon"
            else "（经度编码为 sin/cos，正确处理 180° 环绕）"
        ),
        f"- 层：`{result.layer}`（本次只提取了 {result.n_layers} 层；"
        "用 `layer-sweep` 可扫描模型全部层）",
        "- 探针：Ridge（alpha 由训练集内部 K 折 CV 选，不使用测试集）",
        "",
    ]

    # -- 警告 --
    if result.warnings:
        lines += ["## ⚠ 警告", ""]
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # -- 主结果 --
    lines += [
        "## 主结果（Haversine 中位误差，单位 km）",
        "",
        "| 组 | n | 中位误差 | 均值 | P90 | 95% CI | 说明 |",
        "|---|---|---|---|---|---|---|",
        (
            f"| **A · 对照（留出集）** | {result.control.n} | "
            f"**{_fmt_km(result.control_repeats.median_mean)}** | "
            f"{_fmt_km(result.control.mean_km)} | {_fmt_km(result.control.p90_km)} | "
            f"[{_fmt_km(result.control.ci_low)}, {_fmt_km(result.control.ci_high)}] | "
            f"探针能力上限（重复划分标准差 {result.control_repeats.median_std:.0f} km） |"
        ),
        (
            f"| **B · 歧义地名** | {result.ambiguous.n} | "
            f"**{_fmt_km(result.ambiguous_repeats.median_mean)}** | "
            f"{_fmt_km(result.ambiguous.mean_km)} | {_fmt_km(result.ambiguous.p90_km)} | "
            f"[{_fmt_km(result.ambiguous.ci_low)}, {_fmt_km(result.ambiguous.ci_high)}] | "
            f"**{result.ambiguity_ratio:.2f}×** 于对照组 |"
        ),
        "",
    ]

    # -- 基线对比 --
    if result.baselines:
        lines += [
            "## 基线对比（A 组留出集）",
            "",
            "| 特征来源 | 中位误差 (km) | 相对隐层 | 意义 |",
            "|---|---|---|---|",
        ]
        descriptions = {
            "hidden": ("模型隐层（本项目主张）", "—"),
            "char_ngram": ("字符 n-gram TF-IDF", "★ 最关键：排除「只记住拼写」"),
            "permuted": ("★ 置换检验（打乱坐标）", "★ 最严谨：判别探针是否真读到信息"),
            "random": ("随机向量", "下界：应该很差"),
            "mean": ("永远预测训练集均值", "R²=0 的参照"),
            "mlp": ("隐层 + MLP 探针", "对照：信息是否非线性编码"),
        }
        hidden_median = result.baselines.get("hidden", {}).get("median_km")
        for kind, payload in result.baselines.items():
            median = payload.get("median_km")
            label, meaning = descriptions.get(kind, (kind, ""))
            if hidden_median and median and not np.isnan(median) and hidden_median > 0:
                rel = f"{median / hidden_median:.1f}×"
            else:
                rel = "—"
            lines.append(f"| {label} | {_fmt_km(median)} | {rel} | {meaning} |")
        lines.append("")

    # -- 含义偏好 --
    if result.alt_analysis:
        n_reliable = sum(1 for row in result.alt_analysis if row.get("reliable"))
        lines += [
            "## 歧义词的含义偏好",
            "",
            f"> 「最近候选」= 到主含义与备选含义两者中较近的那个距离。",
            f"> 只有「最近候选」≤ 可靠性门槛的条目才参与结论 —— "
            f"否则两个候选都很远，「谁更近」只是噪声。本次 {n_reliable}/{len(result.alt_analysis)} 条可靠。",
            "",
            "| 词 | 主含义 | 到主含义 | 备选含义 | 到备选 | 最近候选 | 更接近 | 可靠 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(result.alt_analysis, key=lambda r: -float(r["dist_primary_km"])):
            mark = "**备选**" if row["closer_to"] == "alt" else "主含义"
            reliable_mark = "✅" if row.get("reliable") else "—"
            lines.append(
                f"| `{row['word']}` | {row['label']} | {_fmt_km(row['dist_primary_km'])} km | "
                f"{row['alt_label']} | {_fmt_km(row['dist_alt_km'])} km | "
                f"{_fmt_km(row.get('min_km'))} km | {mark} | {reliable_mark} |"
            )
        lines.append("")

    # -- 错误分析 --
    if result.ambiguous_detail:
        lines += [
            "## 错误分析：B 组中误差最大的 10 个词",
            "",
            "| 词 | 误差 (km) | 真实坐标 | 预测坐标 |",
            "|---|---|---|---|",
        ]
        for row in result.ambiguous_detail[:10]:
            lines.append(
                f"| `{row['word']}` | {_fmt_km(row['error_km'])} | "
                f"({row['true_lat']:.1f}, {row['true_lon']:.1f}) | "
                f"({row['pred_lat']:.1f}, {row['pred_lon']:.1f}) |"
            )
        lines.append("")

    # -- 结论（自动生成）--
    lines += ["## 结论", ""]
    for item in render_conclusion(result):
        lines.append(f"- {item}")
    lines.append("")

    # -- 局限 --
    lines += [
        "## 已知局限",
        "",
        "1. **只能检测线性可解码性。** Ridge 差而信息实际存在（非线性编码）时，"
        "会被误判为「没有信息」。启用 `--with-mlp` 做对照可部分排除。",
        "2. **本报告只测了单一层。** 用 `layer-sweep` 命令可以看到地理知识在哪一层涌现。",
        "3. **样本量偏小**（A 组 108 个），置信区间较宽。",
        "4. **探针测的是「表征中可解码的信息」，不等于「模型会这样回答」。** "
        "模型可能表征里有地理信息、但输出时说不知道 —— 这正是探针与输出行为的区别，"
        "也是本实验的价值所在。",
        "5. **坐标是近似的城市/国家中心**（约 ±50 km）。远小于模型误差量级，不影响结论。",
        "",
    ]
    return "\n".join(lines)


def write_report(result: ExperimentResult, out_dir: str | Path, filename: str = "report.md") -> Path:
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
    return path
