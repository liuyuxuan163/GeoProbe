"""Logit lens：把中间层的隐层直接投影到词表，看模型"每层在想什么"。

**它要检验的假设**
----------------
本项目的逐层探针发现了一个跨模型一致的形态：

```
地理可解码性在中层（约 20–40% 深度）达到峰值，之后单调下降，最后一层明显变差
  0.5B: 3,278 → 2,362（峰）→ 2,563 km
  1.5B: 3,149 → 2,175（峰）→ 2,343 km
```

一个自然的解释是：**越往后的层越被"预测下一个 token"这个训练目标挤压**，
要为词表分布服务，而不是保留"世界知识的几何结构"。

**这个解释在 logit lens 下的可检验预测**：

1. 越深的层，它的输出分布越"像一个自信的下一词预测器"
   → **top-1 概率随深度上升、熵随深度下降**
2. 而同一批层的地理可解码性却在下降
   → 两者反向 → 支持"挤压"解释

如果 logit lens 显示**熵不随深度下降**，那"挤压"解释就站不住，
应该换成别的解释（例如表征在高维空间里被"旋转"了，线性可解码性下降）。

**为什么用最后一个 token 的隐层**
-------------------------------
Logit lens 的语义是"这一层此刻会预测什么下一个 token"。
这个语义只在**最后一个位置**成立 —— 前面的位置不能看到完整的输入。
所以 lens 用 ``last_token``，而探针用 ``entity_token`` 池化。
这个差异是刻意的，不是不一致：两者回答不同的问题。

**输入模式的取舍**
----------------
* ``bare``（默认）：输入就是裸地名 ``"Paris"`` —— **与主实验的输入完全一致**，
  所以 lens 指标和探针误差可以直接放在一起比较。缺点是裸地名后面接的
  通常是语法延续，地理词的概率天然低。
* ``prompt``：输入是 ``"Paris is located in"`` —— 更能逼出地理内容，
  但和主实验的输入不同，只适合看"模型是否知道 Paris 在法国"。

要检验"挤压"假设，用 ``bare``；要看地理知识本身，用 ``prompt``。

实现注意
--------
* Logit lens 是**近似**：它把**最终层**的 LayerNorm 套在中间层的残差流上。
  这是该方法的既定做法，但要知道它不是模型真实的前向路径。
* 词表矩阵很大（151,936 × 896），所以**不要**把每层的 logits 全存下来 ——
  在钩子里算完指标就丢掉，只保留标量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "resolve_lm_head",
    "resolve_final_norm",
    "GEO_VOCAB",
    "build_geo_token_ids",
    "LensMetrics",
    "LogitLens",
    "LENS_MODES",
]

LENS_MODES = ("bare", "prompt")

#: 地理相关的英文词。用于测"每一层把多少概率质量放在地理词上"。
#: 分成几组，便于分别观察（国家名 vs 大洲 vs 方位 vs 地理术语）。
GEO_VOCAB: Dict[str, List[str]] = {
    "continents": [
        "Africa", "Asia", "Europe", "America", "Antarctica", "Oceania",
        "Eurasia", "Africa", "Americas",
    ],
    "countries": [
        "Brazil", "Argentina", "Kenya", "Norway", "Sweden", "Finland", "Poland",
        "Portugal", "Germany", "Italy", "Japan", "Canada", "Australia", "Egypt",
        "Morocco", "Vietnam", "Thailand", "Indonesia", "Philippines", "Ethiopia",
        "Bolivia", "Uruguay", "Paraguay", "Ecuador", "Iceland", "Ireland",
        "Greece", "Hungary", "Romania", "Bulgaria", "Croatia", "Ukraine", "Peru",
        "Venezuela", "Guatemala", "Mongolia", "Nepal", "France", "Spain",
        "China", "India", "Russia", "Mexico", "Turkey", "Chile",
    ],
    "directions": [
        "north", "south", "east", "west", "Northern", "Southern", "Eastern",
        "Western", "northeast", "northwest", "southeast", "southwest",
    ],
    "geo_terms": [
        "latitude", "longitude", "coordinates", "coordinate", "country",
        "capital", "city", "region", "located", "geographic", "geographical",
        "located", "border", "borders", "continent", "province", "state",
    ],
}


def resolve_lm_head(model: Any) -> Any:
    """找出词表投影矩阵（unembedding）所在模块。

    兼容：Qwen / Llama（``lm_head``）、GPT-NeoX（``embed_out``）、
    GPT-2（无独立 lm_head，与输入嵌入绑定，用 ``wte``）。
    """
    for attribute in ("lm_head", "embed_out"):
        module = getattr(model, attribute, None)
        if module is not None and hasattr(module, "weight"):
            return module
    # GPT-2 式权重共享
    inner = getattr(model, "transformer", None)
    if inner is not None and hasattr(inner, "wte"):
        return inner.wte
    raise AttributeError(
        "找不到词表投影矩阵（lm_head / embed_out / wte）。\n"
        f"模型类型: {type(model).__name__}"
    )


def resolve_final_norm(model: Any) -> Any:
    """找出最终 LayerNorm（logit lens 必须套它，否则输出分布不在正确尺度上）。"""
    candidates = [
        ("model.norm", lambda m: m.model.norm),                    # Qwen2 / Llama
        ("transformer.ln_f", lambda m: m.transformer.ln_f),        # GPT-2
        ("gpt_neox.final_layer_norm", lambda m: m.gpt_neox.final_layer_norm),
        ("model.decoder.final_layer_norm", lambda m: m.model.decoder.final_layer_norm),
    ]
    for _path, getter in candidates:
        try:
            module = getter(model)
        except AttributeError:
            continue
        if module is not None:
            return module
    raise AttributeError(
        "找不到最终 LayerNorm（logit lens 需要它）。\n"
        f"模型类型: {type(model).__name__}"
    )


def build_geo_token_ids(tokenizer: Any) -> Dict[str, List[int]]:
    """把地理词表转成 token id（每个词取**第一个** token）。

    为什么只取第一个 token：我们看的是"下一个 token"的分布，
    一个多 token 的词（如 ``"coordinates"`` 可能被切成多片）在这个分布里
    只体现为它的首片。取首片是这类分析的标准做法。
    """
    out: Dict[str, List[int]] = {}
    for group, words in GEO_VOCAB.items():
        ids: List[int] = []
        for word in words:
            encoded = tokenizer(word, add_special_tokens=False).get("input_ids", [])
            if encoded:
                ids.append(int(encoded[0]))
        out[group] = sorted(set(ids))
    return out


@dataclass
class LensMetrics:
    """单层、单批词上的 lens 指标（已对词取平均）。"""

    layer: int
    top1_prob: float = float("nan")
    """前 1 个 token 的平均概率 —— 越高说明这一层越"自信" """
    entropy_bits: float = float("nan")
    """输出分布的平均熵（bit）—— 越低说明越"决定好了" """
    geo_mass: float = float("nan")
    """放在地理词上的平均概率质量 """
    geo_mass_continents: float = float("nan")
    geo_mass_countries: float = float("nan")
    geo_mass_directions: float = float("nan")
    word_start_rate: float = float("nan")
    """top-1 是「词首 token」的比例 —— **logit lens 的有效性代理指标**

    一个裸地名后面该接什么？大概率是空格开头的新词（``" is"``）或标点。
    如果某层的 top-1 大量是词中间的碎片（如 ``"ascus"``、``"endencies"``），
    说明这一层的残差流**还没进入"接近输出"的基**，
    套用最终 LayerNorm 得到的分布没有语言意义 —— 该层的 lens 结果不可信。
    """
    top1_tokens: List[str] = field(default_factory=list)
    """前几个词各自最常出现的 top-1 token（便于人工扫一眼）"""


class LogitLens:
    """在已加载的模型上做 logit lens。

    需要传入**已加载好**的模型与分词器（由 ``ActivationExtractor`` 负责加载，
    这样精度选择、内存预检、层路径兼容都能复用）。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layers: Sequence[Any],
        device: str = "cpu",
        prompt_template: str = "{} is located in",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        self.device = device
        self.prompt_template = prompt_template
        self.lm_head = resolve_lm_head(model)
        self.final_norm = resolve_final_norm(model)
        self.geo_ids = build_geo_token_ids(tokenizer)

        # 预先检查"最终层"在 logit lens 下是否自洽：layer L-1 的 lens top-1
        # 应该和模型真实输出的 top-1 很接近（这是 logit lens 有效性的前提）。
        self._vocab_size = int(self.lm_head.weight.shape[0])

    # ------------------------------------------------------------------ #
    def _lens_logits(self, hidden: np.ndarray) -> np.ndarray:
        """一层残差流 → 词表 logits（套用最终 LayerNorm）。

        **dtype 必须对齐**：模型的 norm / lm_head 可能是 bf16/fp16，
        而钩子里抓到的是 fp32。直接相乘会报
        ``expected m1 and m2 to have the same dtype``。

        这里把 hidden 转成权重本身的 dtype 再算（而不是把权重转成 fp32）——
        后者要把 151936×896 的词表矩阵复制成 fp32，多占几百 MB 内存。
        算完把 logits 抬回 fp32 做 softmax，保证概率精度。
        """
        import torch

        weight_dtype = self.lm_head.weight.dtype
        tensor = torch.as_tensor(hidden, device=self.device).to(weight_dtype)
        with torch.no_grad():
            normed = self.final_norm(tensor.reshape(1, 1, -1))
            logits = self.lm_head(normed)
        return logits.reshape(-1).float().cpu().numpy()

    def _is_word_start(self, token_id: int) -> bool:
        """判断这个 token 是不是"词首"（空格开头 / 纯标点）。

        这是 logit lens 有效性的一个便宜代理：
        真实的"下一个词"在 BPE 里通常以 ``Ġ``（空格）开头，或者是标点。
        如果某层大量输出词中间的碎片，说明该层还没进入"接近输出"的基。
        """
        text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        if not text:
            return False
        if text.startswith(" ") or text.startswith("Ġ"):
            return True
        # 纯标点 / 纯空白也算合法接续
        stripped = text.strip()
        return bool(stripped) and all(not char.isalnum() for char in stripped)

    def _metrics_from_logits(self, logits: np.ndarray) -> Dict[str, Any]:
        """从一层 logits 算标量指标（不保留 logits 本身）。"""
        # 数值稳定 softmax
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / exp.sum()

        top_index = int(np.argmax(probs))
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        out: Dict[str, Any] = {
            "top1_id": top_index,
            "top1_prob": float(probs[top_index]),
            "entropy_bits": entropy,
            "is_word_start": self._is_word_start(top_index),
        }
        for group, ids in self.geo_ids.items():
            if ids:
                out[f"geo_mass_{group}"] = float(probs[ids].sum())
        out["geo_mass"] = float(
            sum(out.get(f"geo_mass_{group}", 0.0) for group in self.geo_ids)
        )
        return out

    def run(
        self,
        words: Sequence[str],
        mode: str = "bare",
        progress: bool = True,
    ) -> List[LensMetrics]:
        """对一批词做 logit lens，返回**每层**的平均指标。

        Args:
            mode: ``bare``（输入=裸地名）| ``prompt``（输入=带地理提示的句子）。
        """
        import torch

        if mode not in LENS_MODES:
            raise ValueError(f"未知模式 {mode!r}（可选 {LENS_MODES}）")

        n_layers = len(self.layers)
        # 累积器：{layer: [每条指标 dict]}
        collected: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_layers)}
        # 记录每个词的 top-1（取最后一层之外也留中层，便于人工看）
        top1_by_layer: Dict[int, List[str]] = {i: [] for i in range(n_layers)}

        iterator = words
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(words, desc=f"logit lens ({mode})", unit="word")
            except ImportError:
                pass

        for word in iterator:
            text = word if mode == "bare" else self.prompt_template.format(word)
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)

            captured: Dict[int, np.ndarray] = {}

            def make_hook(index: int):
                def hook(module, inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    # 只取最后一个位置的隐层（logit lens 的语义所在）
                    captured[index] = tensor.detach()[0, -1, :].float().cpu().numpy()

                return hook

            handles = []
            try:
                for index in range(n_layers):
                    handles.append(self.layers[index].register_forward_hook(make_hook(index)))
                with torch.no_grad():
                    self.model(input_ids=input_ids)
            finally:
                for handle in handles:
                    handle.remove()

            for index in range(n_layers):
                hidden = captured.get(index)
                if hidden is None:
                    continue
                logits = self._lens_logits(hidden)
                metrics = self._metrics_from_logits(logits)
                collected[index].append(metrics)
                top1_by_layer[index].append(
                    self.tokenizer.decode([metrics["top1_id"]], skip_special_tokens=True)
                )

        results: List[LensMetrics] = []
        for index in range(n_layers):
            rows = collected[index]
            if not rows:
                continue
            entry = LensMetrics(layer=index)
            entry.top1_prob = float(np.mean([row["top1_prob"] for row in rows]))
            entry.entropy_bits = float(np.mean([row["entropy_bits"] for row in rows]))
            entry.geo_mass = float(np.mean([row.get("geo_mass", 0.0) for row in rows]))
            entry.geo_mass_continents = float(
                np.mean([row.get("geo_mass_continents", 0.0) for row in rows])
            )
            entry.geo_mass_countries = float(
                np.mean([row.get("geo_mass_countries", 0.0) for row in rows])
            )
            entry.geo_mass_directions = float(
                np.mean([row.get("geo_mass_directions", 0.0) for row in rows])
            )
            entry.word_start_rate = float(
                np.mean([1.0 if row.get("is_word_start") else 0.0 for row in rows])
            )
            # 取前 5 个词各自的 top-1，人工扫一眼是否和词相关
            entry.top1_tokens = [
                f"{word}→{token}" for word, token in list(zip(words, top1_by_layer[index]))[:5]
            ]
            results.append(entry)
        return results


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 秩相关（不引入 scipy 依赖）。

    用于回答"lens 置信度随深度上升时，探针误差是否也在上升" ——
    即两个指标是否反向变化。
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")

    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(values.size, dtype=float)
        # 处理并列：取平均秩
        _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            sums = np.zeros(len(counts))
            np.add.at(sums, inverse, ranks)
            ranks = (sums / counts)[inverse]
        return ranks

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float(np.sum(rx**2)) * float(np.sum(ry**2)))
    if denominator == 0:
        return float("nan")
    return float(np.sum(rx * ry) / denominator)
