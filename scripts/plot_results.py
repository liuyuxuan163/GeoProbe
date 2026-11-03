"""把报告 JSON 画成图（README / 讲解稿用）。

三张图，全部**只读 JSON、不加载模型**：

===================  ============================================================
图                    读什么、画什么
===================  ============================================================
``layer_curve.png``   ``cross_model_comparison.json`` → 三模型「相对深度 vs 中位误差」
                      曲线。用来看两件事：**末段的平台**和**最后一层的回升**。
                      最优层打星、最后一层打空心点、平台区间画横段、
                      再叠一条「永远预测均值」基线作参照。
``baselines.png``     ``result.<模型>.sincos.json`` → 5 类特征的中位误差（分组柱状）。
                      重点是「隐层 vs 字符 n-gram vs 置换」，并验证
                      ``mean`` / ``char_ngram`` 三模型相同（与模型无关）。
``population.png``    ``stratified_analysis.json`` → 人口 4 档 × 3 模型（分组柱状），
                      标出 Q4/Q1 比值 —— 这是「规模收益集中在冷门档」的主图。
``alignment.png``     ``cross_model_comparison.json`` → 规模效应在**六种对齐口径**下的
                      变化（各取最优层 / 同为最后一层 / 匹配 80%·50%·20% 深度 /
                      全层平均）。用来说明「同一个问题能得出 −23% 到 +16%」。
``ambiguous.png``     ``result.<模型>.sincos.json`` → 20 个歧义词的
                      「到主含义 vs 到备选含义」散点（三模型三联）。
                      对角线以上 = 更接近主含义；灰角 = 两边都太远、判定不可靠。
===================  ============================================================

用法::

    python scripts/plot_results.py                  # 默认写到 docs/figures/
    python scripts/plot_results.py --out-dir reports/figures --dpi 200
    python scripts/plot_results.py --only layer     # 只画一张

> 中文字体：脚本会按候选列表挑一个可用的 CJK 字体（YaHei / SimHei / Noto…）。
> 一个都没有时会打印警告 —— 此时图里的中文会变成方框，请装字体或改用英文标签。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前：无显示环境也能出图

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: 模型 slug → (图例短名, 颜色)。颜色沿用讲解稿的主色。
MODEL_STYLE = {
    "qwen2.5-0.5b-instruct": ("0.5B", "#2E86DE"),   # 蓝
    "qwen2.5-1.5b-instruct": ("1.5B", "#8E5BF0"),   # 紫
    "qwen2.5-3b-instruct": ("3B", "#17A673"),       # 绿
}
FALLBACK_STYLE = ("model", "#6B7280")

#: 中文字体候选（Windows / macOS / Linux 常见）
CJK_FONTS = (
    "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
)


def setup_cjk_font() -> Optional[str]:
    """挑一个可用的中文字体；找不到就返回 ``None`` 并提示。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免缺字形
            return name
    print("[WARN] 没找到中文字体（YaHei/SimHei/Noto…），图里的中文可能显示成方框")
    plt.rcParams["axes.unicode_minus"] = False
    return None


_FONT_READY = False


def ensure_font() -> None:
    """每个绘图函数入口调用一次：保证测试或别的调用方也能拿到中文字体。"""
    global _FONT_READY
    if not _FONT_READY:
        setup_cjk_font()
        _FONT_READY = True


def cjk_font_path() -> Optional[str]:
    """返回一个中文字体**文件路径**（PIL 用得到）。

    直接从 matplotlib 的字体表里取 ``fname``，不要用 ``findfont("sans-serif")``
    —— 后者在新版 matplotlib 上会因为 fontconfig 模式解析而报 ValueError。
    """
    table = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in table and Path(table[name]).exists():
            return str(table[name])
    return None


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #


def load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"缺少 {path}（先跑对应的分析脚本）")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_slug(slug: str) -> str:
    """把各种写法的 slug 归一到 ``MODEL_STYLE`` 的键。

    项目里有**两套** slug 约定：``config.model_slug()`` 走文件名的
    （``qwen2.5-3b-instruct``），``compare_encodings._slugify()`` 走模型名的
    （本地路径 ``D:\\models\\Qwen2.5-3B-Instruct`` → ``d-models-qwen2.5-3b-instruct``）。
    画图时按"参数量"匹配，两套都能对上。
    """
    text = str(slug).lower()
    for key in MODEL_STYLE:
        size = key.split("-")[1]  # 0.5b / 1.5b / 3b
        if size in text:
            return key
    return str(slug)


def layer_rows(cross_model: Dict[str, Any], slug: str) -> List[Dict[str, float]]:
    """取一个模型的逐层行：``[{layer, km, depth_pct}]``，按层号升序。

    ``slug`` 可以是任意写法，内部会归一（见 ``canonical_slug``）。
    """
    want = canonical_slug(slug)
    for model in cross_model["models"]:
        if canonical_slug(model.get("slug", "")) != want:
            continue
        rows = sorted(model["rows"], key=lambda r: r["layer"])
        n_layers = len(rows)
        out = []
        for row in rows:
            out.append(
                {
                    "layer": int(row["layer"]),
                    "km": float(row["sincos_median_km"]),
                    "depth_pct": 100.0 * row["layer"] / max(1, n_layers - 1),
                }
            )
        return out
    raise KeyError(f"cross_model_comparison.json 里没有 {slug}（归一后 {want}）")


def best_and_last(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """最优层 / 最后一层 / 退化比例（退化 = 末层/最优 − 1）。"""
    best = min(rows, key=lambda r: r["km"])
    last = rows[-1]
    return {
        "best_layer": best["layer"],
        "best_km": best["km"],
        "best_depth_pct": best["depth_pct"],
        "last_layer": last["layer"],
        "last_km": last["km"],
        "degradation": last["km"] / best["km"] - 1.0,
    }


def plateau_range(stability: Optional[Dict[str, Any]], slug: str) -> Optional[tuple[int, int]]:
    """从 ``layer_stability.json`` 取平台区间；没有就返回 ``None``。"""
    if not stability:
        return None
    want = canonical_slug(slug)
    models = stability.get("models") or {}
    model = None
    for key, value in models.items():
        if canonical_slug(key) == want:
            model = value
            break
    if not model:
        return None
    run = (model.get("summary") or {}).get("plateau_run") or []
    if len(run) < 2:
        return None
    return int(run[0]), int(run[-1])


# --------------------------------------------------------------------------- #
# 图 1：逐层曲线
# --------------------------------------------------------------------------- #


def plot_layer_curve(
    cross_model: Dict[str, Any],
    out_path: Path,
    stability: Optional[Dict[str, Any]] = None,
    mean_baseline_km: Optional[float] = None,
    dpi: int = 170,
) -> Path:
    """三模型「相对深度 vs 中位误差」曲线。

    要一眼看出来的两件事：

    * **末段的平台**：三条曲线在 85–95% 深度都压到最低，且平台很窄（横段）；
    * **最后一层的回升**：每条线最右端都有一个明显的上翘。
      退化比例不标在点上（会互相压住），而是列在左下角的摘要里。
    """
    ensure_font()
    present = {canonical_slug(m.get("slug", "")) for m in cross_model["models"]}
    slugs = [s for s in MODEL_STYLE if s in present]
    if not slugs:
        raise KeyError("cross_model_comparison.json 里一个已知模型都没匹配上")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    infos: Dict[str, Dict[str, float]] = {}
    all_km: List[float] = []

    for slug in slugs:
        label, color = MODEL_STYLE[slug]
        rows = layer_rows(cross_model, slug)
        info = best_and_last(rows)
        infos[slug] = info
        xs = [r["depth_pct"] for r in rows]
        ys = [r["km"] for r in rows]
        all_km.extend(ys)
        ax.plot(xs, ys, "-o", color=color, lw=2.0, ms=3.2, label=label, zorder=3)

        # 最优层：星标
        ax.plot(
            [info["best_depth_pct"]], [info["best_km"]], "*", color=color,
            ms=17, markeredgecolor="white", markeredgewidth=0.8, zorder=5,
        )
        # 最后一层：空心圆
        ax.plot(
            [xs[-1]], [ys[-1]], "o", color=color, ms=7.5, mfc="white",
            mew=1.8, zorder=5,
        )

        # 平台区间：在最优值高度上画一小段横线（平台 = 与最优差 ≤ 1×层内噪声的层）
        span = plateau_range(stability, slug)
        if span:
            lo, hi = span
            x0 = 100.0 * lo / max(1, len(rows) - 1)
            x1 = 100.0 * hi / max(1, len(rows) - 1)
            ax.hlines(info["best_km"], x0, x1, color=color, lw=5.0, alpha=0.32, zorder=2)

    if mean_baseline_km:
        ax.axhline(
            mean_baseline_km, color="#9CA3AF", ls="--", lw=1.3, zorder=1,
            label=f"永远预测均值 {mean_baseline_km:,.0f} km",
        )

    # 峰值所在的相对深度区间（浅灰底 + 底部标注，避开图例）
    ax.axvspan(85, 95, color="#F3F4F6", zorder=0)

    lo = min(all_km)
    hi = max(all_km + ([mean_baseline_km] if mean_baseline_km else []))
    ax.set_ylim(lo - 380, hi + 240)
    ax.text(
        90, lo - 300, "峰值都落在\n87–94% 深度",
        ha="center", va="bottom", fontsize=8.6, color="#6B7280", linespacing=1.4,
    )

    # 左下角摘要：峰值层 / 峰值 / 末层 / 退化（三行分别用模型色）
    for row, slug in enumerate(slugs):
        info = infos[slug]
        label, color = MODEL_STYLE[slug]
        ax.text(
            0.015, 0.055 + (len(slugs) - 1 - row) * 0.072,
            f"{label:<4} 峰值 L{info['best_layer']:>2} · {info['best_km']:,.0f} km"
            f"  →  末层 {info['last_km']:,.0f} km（+{info['degradation'] * 100:.0f}%）",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.2, color=color,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.75),
        )

    ax.set_xlabel("相对深度（层号 / (层数 − 1)，%）")
    ax.set_ylabel("Haversine 中位误差（km，越小越好）")
    ax.set_title("逐层扫描：末段见底、最后一层回升（319 词，sincos，8 次划分）", pad=12)
    ax.grid(alpha=0.28, ls=":")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_xlim(-2, 104)
    fig.text(
        0.5, -0.02, "注：y 轴未从 0 起（为了看清末段与平台）；星标 = 最优层，空心圆 = 最后一层",
        ha="center", va="top", fontsize=8.6, color="#6B7280",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 图 2：基线对比
# --------------------------------------------------------------------------- #


BASELINE_ORDER = (
    ("hidden", "模型隐层"),
    ("char_ngram", "字符 n-gram"),
    ("mean", "永远预测均值"),
    ("random", "随机向量"),
    ("permuted", "置换检验（打乱坐标）"),
)


def plot_baselines(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 170,
) -> Path:
    """5 类特征的中位误差分组柱状图（每类 3 根柱 = 3 个模型）。

    ``mean`` / ``char_ngram`` 不依赖模型 —— 三根柱**必须一样高**，
    这对图本身就是一个流程自检。
    """
    ensure_font()
    slugs = [s for s in MODEL_STYLE if s in results]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    n_cat = len(BASELINE_ORDER)
    bar_h = 0.8 / max(1, len(slugs))
    y_pos = list(range(n_cat))[::-1]  # 第一类画在最上面

    for i, slug in enumerate(slugs):
        label, color = MODEL_STYLE[slug]
        baselines = results[slug]["summary"]["baselines"]
        vals, offsets = [], []
        for cat, _ in BASELINE_ORDER:
            vals.append(float(baselines.get(cat, {}).get("median_km", float("nan"))))
            # 0.5B 画在最上面，与图例顺序一致
            offsets.append((len(slugs) - 1 - i) * bar_h - (len(slugs) - 1) * bar_h / 2)
        ys = [y + off for y, off in zip(y_pos, offsets)]
        bars = ax.barh(
            ys, vals, height=bar_h * 0.88, color=color,
            label=label, alpha=0.92,
            edgecolor="white", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                v + 90, bar.get_y() + bar.get_height() / 2, f"{v:,.0f}",
                va="center", fontsize=8.6, color="#374151",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([name for _, name in BASELINE_ORDER])
    ax.set_xlabel("Haversine 中位误差（km，越小越好）")
    ax.set_title("基线对比：隐层明显优于字符基线，也远优于置换检验", pad=12)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.grid(axis="x", alpha=0.28, ls=":")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_xlim(0, max(
        8000,
        max(
            float(results[s]["summary"]["baselines"].get(cat, {}).get("median_km", 0) or 0)
            for s in slugs for cat, _ in BASELINE_ORDER
        ) * 1.12,
    ))
    # 把「隐层」那一行底色挑出来，呼应正文
    ax.axhspan(y_pos[0] - 0.46, y_pos[0] + 0.46, color="#EFF6FF", zorder=0)
    fig.text(
        0.5, -0.03,
        "注：均值 / 字符 n-gram 与模型无关，三根柱应完全相同（流程自检）",
        ha="center", va="top", fontsize=8.6, color="#6B7280",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 图 3：人口分档
# --------------------------------------------------------------------------- #


def plot_population(
    stratified: Dict[str, Any],
    out_path: Path,
    dpi: int = 170,
) -> Path:
    """人口 4 档 × 3 模型的分组柱状图，标注 Q4/Q1。"""
    ensure_font()
    slugs = [s for s in MODEL_STYLE if s in stratified.get("models", {})]
    ref = stratified["models"][slugs[0]]["by_population"]
    quantiles = [q for q in ref["quantiles"] if q.get("n", 0) > 0]

    def _pop_label(value: float) -> str:
        """人口标签：≥100 万用整数，否则保留一位小数（5.6 万而不是 6 万）。"""
        wan = value / 10_000.0
        return f"{wan:,.0f} 万" if wan >= 100 else f"{wan:,.1f} 万"

    labels = [f"{q['label']}\n{_pop_label(q['population_median'])}" for q in quantiles]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    n_bins = len(quantiles)
    group_w = 0.78
    bar_w = group_w / max(1, len(slugs))
    xs = list(range(n_bins))

    for i, slug in enumerate(slugs):
        name, color = MODEL_STYLE[slug]
        results = stratified["models"][slug]["by_population"]
        by_label = {q["label"]: q for q in results["quantiles"]}
        vals = [float(by_label[q["label"]]["median_km"]) for q in quantiles]
        pos = [x + (i - (len(slugs) - 1) / 2) * bar_w for x in xs]
        bars = ax.bar(
            pos, vals, width=bar_w * 0.9, color=color, label=name,
            edgecolor="white", linewidth=0.7,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 60, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=8.4, color="#374151",
            )
        ratio = vals[-1] / vals[0] if vals[0] else float("nan")
        ax.plot(
            [pos[-1]], [vals[-1] + 420], marker="v", ms=6, color=color, clip_on=False,
        )
        ax.text(
            pos[-1], vals[-1] + 700, f"{ratio:.2f}×", ha="center", va="bottom",
            fontsize=9.0, color=color, fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("人口分档（等词数切成 4 档，Q1 = 最知名；下面一行是该档人口中位数）")
    ax.set_ylabel("Haversine 中位误差（km）")
    ax.set_title("规模收益几乎全在冷门档：Q4/Q1 从 2.08×（0.5B）缩到 1.08×（3B）", pad=12)
    ax.grid(axis="y", alpha=0.28, ls=":")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_ylim(0, max(
        float(q["median_km"]) for s in slugs
        for q in stratified["models"][s]["by_population"]["quantiles"]
        if q.get("n", 0) > 0
    ) * 1.22)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 图 4：规模效应的六种对齐口径
# --------------------------------------------------------------------------- #


def _at_depth(rows: Sequence[Dict[str, float]], frac: float) -> Dict[str, float]:
    """取相对深度最接近 ``frac`` 的那一层。

    **按索引 round(frac × (n−1)) 取，而不是按浮点距离找最近** ——
    层数为奇数时（如 3B 的 36 层、50% 深度），中间两层距离完全相同，
    按"最近的第一个"会随列表顺序变，两次运行的取层可能不一样。
    """
    index = int(round(frac * (len(rows) - 1)))
    index = max(0, min(index, len(rows) - 1))
    return rows[index]


def alignment_table(
    cross_model: Dict[str, Any],
    small_slug: str = "qwen2.5-0.5b-instruct",
    large_slug: str = "qwen2.5-3b-instruct",
) -> List[Dict[str, Any]]:
    """算 0.5B vs 3B 在六种对齐口径下的规模效应。

    Returns:
        每项 ``{name, small_km, large_km, change, detail}``，``change`` 为
        「3B 相对 0.5B 的变化比例」（负 = 3B 更好）。
    """
    small = layer_rows(cross_model, small_slug)
    large = layer_rows(cross_model, large_slug)
    bs, bl = best_and_last(small), best_and_last(large)

    def row(name: str, s_km: float, l_km: float, detail: str) -> Dict[str, Any]:
        return {
            "name": name,
            "small_km": float(s_km),
            "large_km": float(l_km),
            "change": float(l_km) / float(s_km) - 1.0,
            "detail": detail,
        }

    out = [
        row(
            "各取最优层", bs["best_km"], bl["best_km"],
            f"L{bs['best_layer']}（{bs['best_depth_pct']:.0f}%）vs "
            f"L{bl['best_layer']}（{bl['best_depth_pct']:.0f}%）",
        ),
        row(
            "同为最后一层", bs["last_km"], bl["last_km"],
            f"L{bs['last_layer']} vs L{bl['last_layer']}（位置无歧义，推荐）",
        ),
    ]
    for frac, label in ((0.8, "匹配 80% 深度"), (0.5, "匹配 50% 深度"), (0.2, "匹配 20% 深度")):
        s_row, l_row = _at_depth(small, frac), _at_depth(large, frac)
        out.append(
            row(
                label, s_row["km"], l_row["km"],
                f"L{s_row['layer']} vs L{l_row['layer']}",
            )
        )
    out.append(
        row(
            "全层平均（AUC 口径）",
            float(sum(r["km"] for r in small) / len(small)),
            float(sum(r["km"] for r in large) / len(large)),
            "所有层取均值",
        )
    )
    return out


def plot_alignment(
    cross_model: Dict[str, Any],
    out_path: Path,
    small_slug: str = "qwen2.5-0.5b-instruct",
    large_slug: str = "qwen2.5-3b-instruct",
    dpi: int = 170,
) -> Path:
    """规模效应在六种对齐口径下的变化（横向条形，负 = 3B 更好）。"""
    ensure_font()
    table = alignment_table(cross_model, small_slug, large_slug)
    small_name = MODEL_STYLE[canonical_slug(small_slug)][0]
    large_name = MODEL_STYLE[canonical_slug(large_slug)][0]

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    ys = list(range(len(table)))[::-1]
    changes = [item["change"] * 100 for item in table]
    colors = ["#17A673" if c < 0 else "#E5484D" for c in changes]
    ax.barh(ys, changes, color=colors, alpha=0.9, edgecolor="white", linewidth=0.8)

    for y, item, change in zip(ys, table, changes):
        # 条末只放百分比（短标签，不会压到 y 轴刻度）
        ax.text(
            change + (1.0 if change >= 0 else -1.0), y, f"{change:+.1f}%",
            va="center", ha="left" if change >= 0 else "right",
            fontsize=9.6, fontweight="bold", color="#111827",
        )
        # 两个绝对误差值放在条内（白字），省掉外部空间
        ax.text(
            change / 2, y, f"{item['small_km']:,.0f} → {item['large_km']:,.0f} km",
            va="center", ha="center", fontsize=8.4, color="white",
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{item['name']}\n{item['detail']}" for item in table])
    for tick, item in zip(ax.get_yticklabels(), table):
        if "最后一层" in item["name"]:
            tick.set_fontweight("bold")
    ax.axvline(0, color="#111827", lw=1.0)
    ax.set_xlabel(f"误差变化：{large_name} 相对 {small_name}（%，负 = {large_name} 更好）")
    ax.set_title(
        f"规模效应取决于对齐方式：同一个「{small_name} → {large_name}」能得出 "
        f"−{abs(min(changes)):.0f}% 到 +{max(changes):.0f}%",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.28, ls=":")
    lo = min(changes) - 10
    hi = max(changes) + 10
    ax.set_xlim(lo, hi)
    ax.text(
        0.99, 0.02,
        "对齐方式必须和结论一起说明（推荐「同为最后一层」）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8, color="#6B7280",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 图 5：歧义组偏好散点
# --------------------------------------------------------------------------- #


def ambiguous_points(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从单层结果里取歧义词的「到主含义 / 到备选含义」两点坐标。"""
    return [
        {
            "word": str(row["word"]),
            "primary_km": float(row["dist_primary_km"]),
            "alt_km": float(row["dist_alt_km"]),
            "closer_to": str(row["closer_to"]),
            "reliable": bool(row.get("reliable")),
        }
        for row in result.get("alt_analysis", [])
    ]


def plot_ambiguous(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 170,
) -> Path:
    """20 个歧义词的散点：x = 到主含义，y = 到备选含义。

    * 对角线 ``y = x`` 上 = 两边一样近；**上方 = 更接近主含义**（正常）
    * 右上角的灰区 = 两个候选都超过可靠性门槛 → 「谁更近」只是噪声
    * 实心 = 判定可靠，空心 = 不可靠；颜色 = 更接近哪一边
    """
    ensure_font()
    slugs = [s for s in MODEL_STYLE if s in results]
    if not slugs:
        raise KeyError("没有可用的 result.<模型>.sincos.json")

    n = len(slugs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), sharex=True, sharey=True)
    axes = [axes] if n == 1 else list(axes)

    for ax, slug in zip(axes, slugs):
        name, color = MODEL_STYLE[slug]
        result = results[slug]
        points = ambiguous_points(result)
        threshold = float(result.get("alt_reliability_used") or 0) or None
        limit = max(
            max(p["primary_km"], p["alt_km"]) for p in points
        ) * 1.06 if points else 1000

        # 对角线 = 等距
        ax.plot([0, limit], [0, limit], ls="--", lw=1.1, color="#9CA3AF", zorder=1)
        if threshold:
            ax.axhspan(threshold, limit, xmin=threshold / limit, color="#F3F4F6", zorder=0)
            ax.text(
                limit * 0.97, threshold * 1.02,
                f"两边都 > {threshold:,.0f} km\n→ 不可靠区",
                ha="right", va="bottom", fontsize=8.0, color="#9CA3AF",
            )

        for idx, p in enumerate(points):
            is_alt = p["closer_to"] == "alt"
            face = "#E8833A" if is_alt else "#2E86DE"
            ax.scatter(
                p["primary_km"], p["alt_km"],
                s=64 if p["reliable"] else 26,
                facecolors=face if p["reliable"] else "none",
                edgecolors=face, linewidths=1.4, alpha=0.95 if p["reliable"] else 0.75,
                zorder=3,
            )
            # 只标"信息量最大"的点：偏备选的（意外情况）+ 离对角线特别远的
            delta = abs(p["primary_km"] - p["alt_km"])
            if p["reliable"] and (is_alt or delta > 8000):
                dy = 7 if idx % 2 == 0 else -11
                ax.annotate(
                    p["word"], (p["primary_km"], p["alt_km"]),
                    xytext=(6, dy), textcoords="offset points",
                    fontsize=7.6, color="#374151",
                )

        n_rel = sum(1 for p in points if p["reliable"])
        n_alt = sum(1 for p in points if p["reliable"] and p["closer_to"] == "alt")
        ax.set_title(f"{name}　可靠 {n_rel}/{len(points)}，其中偏备选 {n_alt}", fontsize=10)
        ax.set_xlabel("到主含义的距离（km）")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.grid(alpha=0.28, ls=":")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("到备选含义的距离（km）")
    fig.suptitle(
        "歧义词偏好：对角线以上 = 更接近主含义；蓝 = 更接近主含义，橙 = 更接近备选；"
        "实心 = 判定可靠，空心 = 两边都太远（不可靠）",
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 图 0：组合总览（把上面五张拼成一张）
# --------------------------------------------------------------------------- #

OVERVIEW_TITLE = "GeoProbe · 实验总览"
OVERVIEW_SUBTITLE = (
    "319 词 · Qwen2.5 三档模型 · 88 个层-模型组合 ｜ "
    "逐层扫描 / 基线对比 / 人口分档 / 对齐口径 / 歧义偏好"
)
OVERVIEW_NOTES = (
    "关键数字",
    "· 最优层误差 3,122 / 2,683 / 2,401 km（0.5B / 1.5B / 3B）",
    "· 最后一层比峰值退化 18–31%，三模型一致",
    "· 规模效应 −21.8%（同为最后一层）；换对齐口径会变到 +16%",
    "· 规模收益几乎全在冷门档：Q4 降 41%，Q1 无显著变化",
    "· 歧义词中位误差是对照组的 1.39–1.56 倍",
)


def _fit_into(img, box_w: int, box_h: int):
    """等比例缩放进 ``box_w × box_h``（不裁剪，留白）。"""
    from PIL import Image

    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


def compose_overview(
    panel_paths: Sequence[Path],
    out_path: Path,
    cell: tuple[int, int] = (980, 520),
    margin: int = 36,
    gutter: int = 26,
) -> Path:
    """把多张图拼成 2 列网格（最后多出来的一个格子放「关键数字」面板）。

    为什么用拼图而不是重画：拼图**逐像素等于**单张图，读者在总览和细节之间
    跳转时不会看到两套不同的画法；重画则要维护两份排版。
    """
    from PIL import Image, ImageDraw, ImageFont

    pics = []
    for path in panel_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"拼图缺少 {path}（先跑一次不带 --only 的作图）")
        pics.append(Image.open(path).convert("RGB"))

    cols, cell_w, cell_h = 2, cell[0], cell[1]
    per_row = len(pics) + 1  # +1 = 文字面板
    rows = (per_row + cols - 1) // cols

    header_h = 104
    width = margin * 2 + cols * cell_w + (cols - 1) * gutter
    height = header_h + margin + rows * cell_h + (rows - 1) * gutter + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # 标题区
    font_path = cjk_font_path()
    try:
        if font_path is None:
            raise OSError("没有中文字体")
        title_font = ImageFont.truetype(font_path, 40)
        sub_font = ImageFont.truetype(font_path, 21)
        note_title = ImageFont.truetype(font_path, 27)
        note_font = ImageFont.truetype(font_path, 22)
    except OSError:
        title_font = sub_font = note_title = note_font = ImageFont.load_default()

    draw.text((margin, 22), OVERVIEW_TITLE, font=title_font, fill="#111827")
    draw.text((margin, 74), OVERVIEW_SUBTITLE, font=sub_font, fill="#6B7280")

    cells = []
    for img in pics:
        cells.append(("image", img))
    cells.append(("notes", None))

    for index, (kind, img) in enumerate(cells):
        row, col = divmod(index, cols)
        x0 = margin + col * (cell_w + gutter)
        y0 = header_h + margin + row * (cell_h + gutter)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline="#E5E7EB", width=2)

        if kind == "image":
            scaled = _fit_into(img, cell_w - 8, cell_h - 8)
            ox = x0 + (cell_w - scaled.width) // 2
            oy = y0 + (cell_h - scaled.height) // 2
            canvas.paste(scaled, (ox, oy))
        else:
            line_h = 38
            block_h = 44 + line_h * (len(OVERVIEW_NOTES) - 1)
            ty = y0 + (cell_h - block_h) // 2
            draw.text((x0 + 30, ty), OVERVIEW_NOTES[0], font=note_title, fill="#111827")
            y = ty + 48
            for line in OVERVIEW_NOTES[1:]:
                draw.text((x0 + 30, y), line, font=note_font, fill="#374151")
                y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def collect_results(reports: Path) -> Dict[str, Dict[str, Any]]:
    """收集三份单层结果 ``result.<slug>.sincos.json``。"""
    out: Dict[str, Dict[str, Any]] = {}
    for slug in MODEL_STYLE:
        path = reports / f"result.{slug}.sincos.json"
        if path.exists():
            out[slug] = load_json(path)
    return out


PANEL_KINDS = ("layer", "baselines", "population", "alignment", "ambiguous")

#: kind → 文件名（逐层那张叫 layer_curve.png，和 kind 不同名）
PANEL_FILES = {
    "layer": "layer_curve.png",
    "baselines": "baselines.png",
    "population": "population.png",
    "alignment": "alignment.png",
    "ambiguous": "ambiguous.png",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="把报告 JSON 画成 README 用的图")
    parser.add_argument("--reports", default="reports", help="JSON 所在目录")
    parser.add_argument("--out-dir", default="docs/figures", help="图片输出目录")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--only", choices=[*PANEL_KINDS, "overview"],
        help="只画某一张（overview = 先重画五张，再拼总览）",
    )
    args = parser.parse_args(argv)

    # overview 要依赖另外五张，所以它等价于"全画 + 拼图"
    wanted = set(PANEL_KINDS) if args.only in (None, "overview") else {args.only}

    font = setup_cjk_font()
    print(f"中文字体：{font or '未找到（可能显示方框）'}")

    reports = ROOT / args.reports if not Path(args.reports).is_absolute() else Path(args.reports)
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made: List[Path] = []

    if "layer" in wanted:
        cross = load_json(reports / "cross_model_comparison.json")
        stability_path = reports / "layer_stability.json"
        stability = load_json(stability_path) if stability_path.exists() else None
        results = collect_results(reports)
        mean_km = None
        if results:
            first = next(iter(results.values()))
            mean_km = float(first["summary"]["baselines"].get("mean", {}).get("median_km") or 0) or None
        made.append(plot_layer_curve(
            cross, out_dir / "layer_curve.png", stability, mean_km, dpi=args.dpi,
        ))

    if "baselines" in wanted:
        results = collect_results(reports)
        if not results:
            print("[SKIP] 没有 result.*.sincos.json，跳过基线图")
        else:
            made.append(plot_baselines(results, out_dir / "baselines.png", dpi=args.dpi))

    if "population" in wanted:
        strat_path = reports / "stratified_analysis.json"
        if not strat_path.exists():
            print("[SKIP] 没有 stratified_analysis.json，跳过人口分档图")
        else:
            made.append(plot_population(load_json(strat_path), out_dir / "population.png", dpi=args.dpi))

    if "alignment" in wanted:
        cross_path = reports / "cross_model_comparison.json"
        if not cross_path.exists():
            print("[SKIP] 没有 cross_model_comparison.json，跳过对齐口径图")
        else:
            made.append(plot_alignment(load_json(cross_path), out_dir / "alignment.png", dpi=args.dpi))

    if "ambiguous" in wanted:
        results = collect_results(reports)
        if not results:
            print("[SKIP] 没有 result.*.sincos.json，跳过歧义偏好图")
        else:
            made.append(plot_ambiguous(results, out_dir / "ambiguous.png", dpi=args.dpi))

    if args.only in (None, "overview"):
        panels = [out_dir / PANEL_FILES[kind] for kind in PANEL_KINDS]
        if all(p.exists() for p in panels):
            made.append(compose_overview(panels, out_dir / "overview.png"))
        else:
            missing = [p.name for p in panels if not p.exists()]
            print(f"[SKIP] 缺 {missing}，跳过总览拼图")

    for path in made:
        print(f"[OK] {path}  ({path.stat().st_size / 1024:.0f} KB)")
    if not made:
        print("[FAIL] 没有生成任何图")
        return 1
    print("\nREADME 里可以这样引用（把图片一起提交）：")
    for path in made:
        rel = path.relative_to(ROOT).as_posix()
        print(f"  ![{path.stem}]({rel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
