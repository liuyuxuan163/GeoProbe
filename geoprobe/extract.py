"""隐层激活提取：模型加载、层路径兼容、三种池化、Mock 模式。

**这是整个项目技术上最容易卡住的一环。** 三个坑都在这里：

坑 1：不同模型的"层"属性名不同
--------------------------------
::

    Qwen1 / 老版本:      model.transformer.h[-1]
    Qwen2 / Qwen2.5 / Qwen3 / Llama / Mistral:   model.model.layers[-1]
    GPT-2:               model.transformer.h[-1]
    BERT:                model.encoder.layer[-1]

课程原实验写死了 `model.transformer.h[-1]`，换成 Qwen2 就直接 AttributeError。
本模块用 `resolve_transformer_layers()` 做兼容探测。

坑 2：钩子（hook）如果忘了移除，会持续泄漏显存/内存
---------------------------------------------------
每个 `register_forward_hook` 都必须在 finally 里 `remove()`。
而且要一次注册所有层（为了 layer sweep），不能每层单独跑一遍前向 —— 那样慢 N 倍。

坑 3：`Activations[-1]` 取的是"最后一个 token"，但它不是"地名本身"
---------------------------------------------------------------
如果输入是 `"What are the coordinates of Paris?"`，最后 token 是问号。
它虽然 attend 过全序列，但编码的重点是**任务意图**而不是**实体知识**。
所以本模块提供三种池化：

* `last_token`   —— 原实验做法
* `mean`         —— 全序列平均
* `entity_token` —— **只取地名本身的 token 均值**（推荐，最干净）

Mock 模式的意义
---------------
`MockExtractor` 不加载任何模型，而是**合成**隐层：

* `with_signal=True`  —— 把真实坐标编码进向量（加噪声），探针应该能恢复 → 用于验证"流程正确"
* `with_signal=False` —— 纯噪声，探针应该失败 → 用于验证"基线是有效的"

这样整个 pipeline（探针 → 指标 → 基线 → 报告）**在没有任何模型的情况下也能测**，
CI 不需要下载几百 MB 权重。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config, output_paths
from .wordlists import WordList

__all__ = [
    "resolve_transformer_layers",
    "pool_activations",
    "ActivationExtractor",
    "MockExtractor",
    "build_extractor",
    "extract_dataset",
    "check_memory_budget",
    "estimate_model_memory_gb",
    "find_cached_activations",
    "POOLING_MODES",
]

POOLING_MODES = ("last_token", "mean", "entity_token")


# --------------------------------------------------------------------------- #
# 坑 1：层路径兼容
# --------------------------------------------------------------------------- #


def resolve_transformer_layers(model: Any) -> List[Any]:
    """找出模型的 transformer block 列表（兼容主流模型家族）。

    按优先级尝试几条常见路径。找到就返回，找不到给出**清晰的报错**
    （而不是让调用方在 AttributeError 里猜）。

    Args:
        model: 已加载的 HuggingFace 模型（``AutoModelForCausalLM``）。

    Returns:
        层模块的列表，长度 = 层数。
    """
    candidates = [
        ("model.layers", lambda m: m.model.layers),                    # Qwen2/2.5/3, Llama, Mistral
        ("transformer.h", lambda m: m.transformer.h),                  # GPT-2, Qwen1
        ("model.decoder.layers", lambda m: m.model.decoder.layers),    # OPT
        ("encoder.layer", lambda m: m.encoder.layer),                  # BERT
        ("model.encoder.layer", lambda m: m.model.encoder.layer),      # BERT 变体
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),              # GPT-NeoX
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)

    available = [name for name, _ in candidates]
    raise AttributeError(
        "无法识别模型结构，找不到 transformer 层。\n"
        f"已尝试的路径: {available}\n"
        f"模型类型: {type(model).__name__}\n"
        "请手动确认层路径，或在此函数中补充一条候选。"
    )


# --------------------------------------------------------------------------- #
# 坑 3：池化
# --------------------------------------------------------------------------- #


def pool_activations(
    hidden: np.ndarray,
    mode: str = "entity_token",
    entity_span: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """把一个样本的隐层序列池化成**一个向量**。

    Args:
        hidden: 形状 ``(seq_len, hidden_dim)``。
        mode: ``last_token`` | ``mean`` | ``entity_token``。
        entity_span: ``(start, end)`` 左闭右开。仅 ``entity_token`` 模式需要。

    Returns:
        形状 ``(hidden_dim,)``。
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError(f"期望 (seq_len, hidden_dim)，实际 {hidden.shape}")

    if mode == "last_token":
        return hidden[-1]
    if mode == "mean":
        return hidden.mean(axis=0)
    if mode == "entity_token":
        if entity_span is None:
            # 退化：没有给定实体位置时，退化为 last_token，并留给调用方一个明确信号
            return hidden[-1]
        start, end = entity_span
        start = max(0, min(start, hidden.shape[0] - 1))
        end = max(start + 1, min(end, hidden.shape[0]))
        return hidden[start:end].mean(axis=0)
    raise ValueError(f"未知的池化方式: {mode!r}（可选 {POOLING_MODES}）")


def _entity_span(tokenizer: Any, text: str, full_length: int, has_instruction: bool) -> tuple[int, int]:
    """找出"地名本身"在 token 序列里的位置区间。

    没有任务指令时，地名就在开头（跳过可能的 BOS）；
    有指令时，取指令文本之后的部分。
    """
    if not has_instruction:
        # 裸地名：位置 1 到末尾（位置 0 通常是 BOS）
        return (1, max(2, full_length - 1))
    # 有指令：重新算"指令前缀"的长度
    prefix = text[: text.find("{}")] if "{}" in text else ""
    prefix_ids = tokenizer(prefix, add_special_tokens=False).get("input_ids", [])
    start = len(prefix_ids)
    return (start, max(start + 1, full_length - 1))


# --------------------------------------------------------------------------- #
# 提取器
# --------------------------------------------------------------------------- #


@dataclass
class ExtractResult:
    """一次批量提取的结果。"""

    embeddings: Dict[int, np.ndarray] = field(default_factory=dict)
    """``{layer_idx: (n_words, hidden_dim)}``"""

    layers: List[int] = field(default_factory=list)
    """实际提取到的层号（0-based）"""

    words: List[str] = field(default_factory=list)
    pooling: str = "entity_token"
    model_name: str = ""
    mock: bool = False

    @property
    def n_words(self) -> int:
        return len(self.words)

    def matrix(self, layer: Optional[int] = None) -> np.ndarray:
        """取某一层的完整矩阵；``layer=None`` 时取最后一层。"""
        if not self.embeddings:
            raise ValueError("没有提取到任何层")
        idx = self.layers[-1] if layer is None else int(layer)
        if idx not in self.embeddings:
            raise KeyError(
                f"层 {idx} 不在结果里（已提取 {self.layers}）。\n"
                "常见原因：调用 extract_dataset() 时没有传 layer_indices —— "
                "它默认**只提取最后一层**。\n"
                "做逐层分析（layer sweep）时必须传入全部层号，例如：\n"
                "    extract_dataset(config, lists, layer_indices=list(range(24)))"
            )
        return self.embeddings[idx]

    def save(self, path: str | Path) -> Path:
        """保存为 npz（便于复用，不必每次跑模型）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "words": np.array(self.words, dtype=object),
            "layers": np.array(self.layers, dtype=int),
            "pooling": np.array([self.pooling], dtype=object),
            "model_name": np.array([self.model_name], dtype=object),
            "mock": np.array([self.mock], dtype=bool),
        }
        for idx, matrix in self.embeddings.items():
            payload[f"layer_{idx}"] = matrix.astype(np.float32)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExtractResult":
        data = np.load(Path(path), allow_pickle=True)
        result = cls(
            words=[str(w) for w in data["words"]],
            layers=[int(x) for x in data["layers"]],
            pooling=str(data["pooling"][0]),
            model_name=str(data["model_name"][0]),
            mock=bool(data["mock"][0]),
        )
        for idx in result.layers:
            result.embeddings[idx] = np.asarray(data[f"layer_{idx}"], dtype=np.float32)
        return result


class MockExtractor:
    """不加载模型，合成隐层向量。

    这样做让**整个 pipeline 可以在没有 GPU / 没有网络的情况下被测试** ——
    与 DocPilot 项目里 MockLLM 的思路一致。

    Args:
        dim: 向量维度（模拟 hidden_size）。
        with_signal: 是否把真实坐标编码进向量。
            ``True``  → 探针应该能恢复地理 → 验证"流程正确"
            ``False`` → 纯噪声 → 验证"基线能识破无信号"
        signal_scale: 坐标信号相对噪声的强度。
        noise: 噪声标准差。
        seed: 随机种子（保证可复现）。
    """

    def __init__(
        self,
        dim: int = 32,
        with_signal: bool = True,
        signal_scale: float = 5.0,
        noise: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        self.with_signal = bool(with_signal)
        self.signal_scale = float(signal_scale)
        self.noise = float(noise)
        self.seed = int(seed)
        self.model_name = f"mock(dim={dim},signal={with_signal})"

    def embed(self, word: str, coord: Optional[Sequence[float]] = None) -> np.ndarray:
        """生成一个确定性向量（同一 word 永远同一噪声）。"""
        # 用 word 的哈希作为噪声种子 → 保证同词同向量，且不同词不相关
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big") % (2**63))
        vec = rng.normal(0.0, self.noise, size=self.dim).astype(np.float32)

        if self.with_signal and coord is not None:
            lat, lon = float(coord[0]), float(coord[1])
            # 把经纬度编码到前 2 维（归一化到 [-1,1]），再叠加一点非线性投影到额外维度
            vec[0] += self.signal_scale * (lat / 90.0)
            vec[1] += self.signal_scale * (lon / 180.0)
            if self.dim >= 6:
                # 纬度在赤道附近的非线性项（模拟真实表征的弯曲结构）
                vec[2] += self.signal_scale * 0.5 * np.cos(np.radians(lat))
        return vec


class ActivationExtractor:
    """真实模型提取器：加载 HuggingFace 模型，抓取每层隐层。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_name = str(config.get("model.name"))
        self.dtype_name = str(config.get("model.dtype", "float32"))
        self.device = str(config.get("model.device", "cpu"))
        self.trust_remote_code = bool(config.get("model.trust_remote_code", True))
        self.pooling = str(config.get("extract.pooling", "entity_token"))
        self.use_instruction = bool(config.get("extract.use_instruction", False))
        self.instruction = str(config.get("extract.instruction", "What are the geographic coordinates of {}?"))
        self._model: Any = None
        self._tokenizer: Any = None
        self._layers: List[Any] = []

    # -- 懒加载 --------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 torch 与 transformers 才能提取真实隐层。\n"
                "安装: pip install torch transformers\n"
                "或者用 --mock 模式验证流程（不需要模型）。"
            ) from exc

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.dtype_name, torch.float32)
        if self.device == "cpu" and dtype == torch.float16:
            # CPU 上 fp16 的算子支持不完整（很多 op 未实现），退回 bf16 而不是 fp32 ——
            # bf16 在 CPU 上支持良好，且同样只有一半显存/内存占用。
            dtype = torch.bfloat16
            print("[提示] CPU 不支持稳定的 float16 前向，已自动改用 bfloat16（同样省一半内存）")

        # ★ 载入前先估算内存，不够就提前失败并给出可操作建议（而不是 OOM 崩溃）
        check_memory_budget(self.model_name, str(dtype).replace("torch.", ""), self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=self.trust_remote_code
        )
        self._model.to(self.device)
        self._model.eval()
        self._layers = resolve_transformer_layers(self._model)

    @property
    def num_layers(self) -> int:
        self._ensure_loaded()
        return len(self._layers)

    # -- 单个词的隐层 --------------------------------------------------- #
    def extract_word(self, word: str, layer_indices: Optional[Sequence[int]] = None) -> Dict[int, np.ndarray]:
        """返回 ``{layer_idx: (hidden_dim,)}``。"""
        import torch

        self._ensure_loaded()
        torch_mod = torch
        total = len(self._layers)
        targets = [total - 1] if layer_indices is None else [int(i) for i in layer_indices]
        targets = [t for t in targets if 0 <= t < total]
        if not targets:
            raise ValueError(f"层号越界（模型共 {total} 层）")

        text = self.instruction.format(word) if self.use_instruction else word
        encoded = self._tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        seq_len = int(input_ids.shape[1])

        captured: Dict[int, np.ndarray] = {}

        def make_hook(index: int):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[index] = tensor.detach().float().cpu().numpy()[0]

            return hook

        handles = []
        try:
            for idx in targets:
                handles.append(self._layers[idx].register_forward_hook(make_hook(idx)))
            with torch_mod.no_grad():
                self._model(input_ids=input_ids)
        finally:
            # ★ 必须移除，否则下一次前向还会回调、并造成内存泄漏
            for handle in handles:
                handle.remove()

        span = _entity_span(self._tokenizer, text, seq_len, self.use_instruction)
        out: Dict[int, np.ndarray] = {}
        for idx in targets:
            hidden = captured.get(idx)
            if hidden is None:
                continue
            out[idx] = pool_activations(hidden, self.pooling, span)
        return out

    def extract(
        self,
        words: Sequence[str],
        layer_indices: Optional[Sequence[int]] = None,
        progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """批量提取，返回 ``{layer_idx: (n_words, hidden_dim)}``。"""
        rows: Dict[int, List[np.ndarray]] = {}
        iterator = words
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(words, desc="提取隐层", unit="word")
            except ImportError:
                pass

        for word in iterator:
            per_layer = self.extract_word(word, layer_indices)
            for idx, vec in per_layer.items():
                rows.setdefault(idx, []).append(vec)

        return {idx: np.vstack(vecs) for idx, vecs in rows.items()}


# --------------------------------------------------------------------------- #
# 工厂 + 批量入口
# --------------------------------------------------------------------------- #


def build_extractor(config: Config) -> Any:
    """按配置返回 Mock 或真实提取器。"""
    if bool(config.get("model.mock")):
        return MockExtractor(
            dim=int(config.get("mock.dim", 32) or 32),
            with_signal=bool(config.get("mock.signal", True)),
            signal_scale=float(config.get("mock.signal_scale", 5.0) or 5.0),
            noise=float(config.get("mock.noise", 0.2) or 0.2),
            seed=int(config.get("probe.random_state", 42)),
        )
    return ActivationExtractor(config)


def extract_dataset(
    config: Config,
    lists: Dict[str, WordList],
    layer_indices: Optional[Sequence[int]] = None,
    extractor: Any = None,
) -> Dict[str, ExtractResult]:
    """对三组词表分别提取隐层。

    Mock 模式下，会把``真实坐标``传给 MockExtractor（这样它才能把地理信息编码进去）；
    真实模型模式下不会有这个参数。
    """
    extractor = extractor or build_extractor(config)
    is_mock = isinstance(extractor, MockExtractor)
    pooling = str(config.get("extract.pooling", "entity_token")) if not is_mock else "mock"
    model_name = getattr(extractor, "model_name", str(config.get("model.name")))

    out: Dict[str, ExtractResult] = {}
    for group, word_list in lists.items():
        words = word_list.words
        if is_mock:
            coords = word_list.coords() if word_list.has_truth else [None] * len(words)
            dim = extractor.dim
            mat = np.vstack(
                [extractor.embed(word, coord) for word, coord in zip(words, coords)]
            ).astype(np.float32)
            result = ExtractResult(
                embeddings={0: mat},
                layers=[0],
                words=words,
                pooling="mock",
                model_name=model_name,
                mock=True,
            )
        else:
            mats = extractor.extract(words, layer_indices)
            result = ExtractResult(
                embeddings=mats,
                layers=sorted(mats),
                words=words,
                pooling=pooling,
                model_name=model_name,
                mock=False,
            )
        out[group] = result
    return out


# --------------------------------------------------------------------------- #
# 内存预算估算：把 OOM 崩溃变成"提前失败 + 可操作建议"
# --------------------------------------------------------------------------- #

#: 每参数字节数
_BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0}

#: 加载开销倍数：权重之外还要放激活、注意力缓冲、tokenizer 等。
#: 实测 0.5B fp32 峰值约 2.4 GB（权重 2.0 GB），所以 1.25 倍是偏乐观的下限。
_OVERHEAD_FACTOR = 1.35


def _parse_param_count(model_name: str) -> Optional[float]:
    """从模型名里猜参数量（单位：十亿）。

    支持 ``Qwen2.5-0.5B`` / ``Llama-3.2-1B`` / ``xxx-7b`` 这类常见写法。
    猜不出来返回 ``None``（此时跳过预检，不阻塞用户）。
    """
    import re

    name = model_name.rstrip("/").split("/")[-1]
    # 优先匹配带 B 的（0.5B, 1.5B, 7B），其次匹配纯数字+B 结尾
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def estimate_model_memory_gb(model_name: str, dtype: str = "float32") -> Optional[float]:
    """估算加载某个模型需要的峰值内存/显存（GB）。

    这只是**粗略估算**（参数量 × 每参数字节 × 开销倍数），
    用途是提前给出警告，不是精确预测。返回 ``None`` 表示无法从名字推断参数量。
    """
    params_b = _parse_param_count(model_name)
    if params_b is None:
        return None
    bytes_per_param = _BYTES_PER_PARAM.get(dtype, 4.0)
    return params_b * 1e9 * bytes_per_param * _OVERHEAD_FACTOR / 1024**3


def _available_memory_gb(device: str) -> Optional[float]:
    """当前可用的内存（CPU）或显存（CUDA），单位 GB。"""
    if device.startswith("cuda"):
        try:
            import torch

            free, _total = torch.cuda.mem_get_info()
            return free / 1024**3
        except Exception:  # noqa: BLE001
            return None
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return None


def check_memory_budget(model_name: str, dtype: str, device: str = "cpu") -> None:
    """载入模型前检查内存是否够用。**不够就提前失败，并给出具体建议。**

    为什么要做这件事
    ----------------
    OOM 的失败方式很糟：跑了几分钟、下载了几 GB 权重，然后在某次前向时崩溃，
    留下的报错是 ``RuntimeError: [enforce fail at alloc_cpu.cpp]`` 之类，
    完全看不出"是内存不够、该怎么改"。而且可能把系统拖进交换区，卡死很久。

    提前估算后，失败发生在**下载之前**，并且直接告诉用户三条出路：
    换更小的模型 / 用 bf16 / 关掉其他程序。

    Notes:
        估算猜不出参数量时（``estimate_model_memory_gb`` 返回 ``None``）直接放行 ——
        宁可偶尔 OOM，也不要用错误的估算挡住用户。
    """
    needed = estimate_model_memory_gb(model_name, dtype)
    if needed is None:
        return
    available = _available_memory_gb(device)
    if available is None:
        return

    if needed <= available:
        return

    where = "显存" if device.startswith("cuda") else "内存"
    alt_dtype = "bfloat16" if dtype == "float32" else "更小的模型"
    raise MemoryError(
        f"预计 {where} 不足，已在加载前中止（避免 OOM 崩溃）。\n"
        f"  模型      : {model_name}\n"
        f"  精度      : {dtype}\n"
        f"  预计需要  : {needed:.1f} GB\n"
        f"  当前可用  : {available:.1f} GB（{where}）\n"
        "\n三条出路：\n"
        f"  1. 降低精度：加 --dtype bfloat16（内存减半 → 约 {needed / 2:.1f} GB）\n"
        "  2. 换更小的模型：例如 Qwen/Qwen2.5-0.5B-Instruct 约需 2.7 GB\n"
        "  3. 关掉其他占用内存的程序后重试（浏览器 / IDE / 其他 Python 进程）\n"
        "\n想看每个候选模型的预估需求：python -m geoprobe models"
    )


# --------------------------------------------------------------------------- #
# 缓存查找：文件名带模型标识，防止误用别的模型的激活
# --------------------------------------------------------------------------- #


def find_cached_activations(
    config: Any,
    groups: Optional[Sequence[str]] = None,
) -> tuple[Dict[str, "ExtractResult"], List[Path]]:
    """按当前配置查找缓存的隐层激活。

    查找顺序（前者优先）：

    1. ``activations.<model>.<group>.npz`` —— 当前命名（带模型标识）
    2. ``activations.<group>.npz`` —— 旧命名，**仅在模型名匹配时**才使用

    第 2 条是为了兼容早期跑过的缓存。但会校验文件里记录的 ``model_name``，
    避免把 0.5B 的激活当成 1.5B 的用 —— 那会让整个对比无声地错掉。

    Returns:
        ``(cached, missing)``：找到的字典 + 缺失的文件路径列表。
    """
    groups = list(groups or ("control", "ambiguous", "nongeo"))
    out = config.out_dir()
    template = output_paths(config)["activations"]
    expected_model = str(config.get("model.name"))
    expected_mock = bool(config.get("model.mock"))

    cached: Dict[str, ExtractResult] = {}
    missing: List[Path] = []

    for group in groups:
        new_path = Path(str(template).replace("{group}", group))
        legacy_path = out / f"activations.{group}.npz"

        chosen: Optional[Path] = None
        if new_path.exists():
            chosen = new_path
        elif legacy_path.exists():
            chosen = legacy_path

        if chosen is None:
            missing.append(new_path)
            continue

        result = ExtractResult.load(chosen)
        # 校验：旧命名文件必须确认是同一个模型，否则拒绝使用
        if chosen == legacy_path:
            if result.mock != expected_mock or (
                not expected_mock and result.model_name != expected_model
            ):
                missing.append(new_path)
                continue
        cached[group] = result

    return cached, missing
