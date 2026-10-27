"""配置层：读取 YAML，支持环境变量覆盖。

为什么不用 pydantic / hydra？
-----------------------------
这个项目的配置很浅（三四个小节），引入一个配置框架不划算。
这里用一个点号访问的轻量包装，行为可预测、可单测。

内置了一个极简 YAML 解析器作为 PyYAML 缺失时的兜底 ——
和 DocPilot 项目里的做法一致，好处是"clone 完就能跑"。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["Config", "load_config", "loads_yaml", "model_slug", "output_paths"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "float32",
        "device": "cpu",
        "trust_remote_code": True,
        "mock": False,
    },
    "mock": {"dim": 32, "signal": True, "signal_scale": 5.0, "noise": 0.2},
    "extract": {
        "pooling": "entity_token",
        "layer": None,
        "use_instruction": False,
        "instruction": "What are the geographic coordinates of {}?",
    },
    "probe": {
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "cv_folds": 5,
        "mlp": True,
        "mlp_hidden": 256,
        "mlp_max_iter": 500,
        "random_state": 42,
        # 目标编码：latlon（线性，旧行为）| sincos（经度循环编码，推荐）
        "target_encoding": "latlon",
    },
    "evaluation": {
        "bootstrap_samples": 1000,
        "ci": 0.95,
        "percentiles": [50, 75, 90],
        # 「偏向哪个含义」的可靠性门槛（km）：
        # 只有当**至少一个候选**离预测点这么近时，"偏向"判定才有意义。
        # 否则两个候选都在几千公里外，"谁更近"只是噪声。
        "alt_reliability_km": 2500.0,
    },
    "baselines": {
        "random": True,
        "char_ngram": True,
        "mean": True,
        "permuted": True,
        "char_ngram_range": [1, 3],
    },
    "report": {"out_dir": "reports", "plots": True},
}

# 环境变量覆盖：GEO_PROBE_<SECTION>_<KEY>，例如 GEO_PROBE_MODEL_NAME=xxx
_ENV_PREFIX = "GEO_PROBE_"


def _coerce(raw: str) -> Any:
    """把环境变量字符串转成合适的 Python 类型。"""
    lowered = raw.strip().lower()
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    if lowered in ("none", "null", ""):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """点号访问的配置对象：`config.get("model.name")`。"""

    def __init__(self, data: Dict[str, Any], root: Optional[Path] = None) -> None:
        self._data = data
        self.root = root or PROJECT_ROOT

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        """就地修改一个配置项（供 layer-sweep 这类需要临时覆盖的场景用）。"""
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path_of(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    def out_dir(self) -> Path:
        """报告输出目录。"""
        return self.path_of(str(self.get("report.out_dir", "reports")))

    def as_dict(self) -> Dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    def describe(self) -> str:
        return (
            f"model={self.get('model.name')} "
            f"mock={self.get('model.mock')} "
            f"pooling={self.get('extract.pooling')} "
            f"layer={self.get('extract.layer')}"
        )


# --------------------------------------------------------------------------- #
# 输出文件命名：把「模型」和「编码」都编进文件名
# --------------------------------------------------------------------------- #
#
# **为什么必须这么做（同一个 bug 已经犯了三次）**
# ------------------------------------------------
#   ① `run --mock` 与真实结果都写 `result.json`     → Mock 覆盖真实结果
#   ② latlon 与 sincos 都写 `layer_sweep.json`      → 换编码重跑冲掉上一次
#   ③ 0.5B / 1.5B / 3B 都写 `result.sincos.json`    → 换模型冲掉上一次
#
# 每一次都是"文件名里少了一个区分维度"。
# 现在把**模型标识**与**目标编码**一起编进文件名，从机制上不可能互相覆盖。


def model_slug(config: Config) -> str:
    """把模型名压成可用作文件名的短标识。

    ``Qwen/Qwen2.5-0.5B-Instruct`` → ``qwen2.5-0.5b-instruct``

    Mock 模式下固定返回 ``mock``（没有真实模型）。
    """
    if config.get("model.mock"):
        return "mock"
    raw = str(config.get("model.name") or "unknown")
    # 兼容"本地目录"形式的模型名：Windows 用反斜杠、POSIX 用斜杠。
    # 例如 D:\models\Qwen2.5-3B-Instruct → Qwen2.5-3B-Instruct
    name = re.split(r"[\\/]+", raw.rstrip("\\/"))[-1] or raw
    slug = name.lower()
    # 只保留字母数字与 . -，其余压成单个连字符
    slug = re.sub(r"[^a-z0-9.\-]+", "-", slug).strip("-")
    return slug or "unknown"


def output_paths(config: Config) -> Dict[str, Path]:
    """当前配置对应的全部输出路径。

    命名约定::

        reports/result.<model>.<encoding>.json
        reports/layer_sweep.<model>.<encoding>.json
        reports/report.<model>.<encoding>.md
        reports/activations.<model>.<group>.npz

    这样换模型 / 换编码重跑时，**旧结果不会被覆盖**，可以事后横向对比。
    """
    out = config.out_dir()
    slug = model_slug(config)
    encoding = str(config.get("probe.target_encoding", "latlon"))
    return {
        "result": out / f"result.{slug}.{encoding}.json",
        "layer_sweep": out / f"layer_sweep.{slug}.{encoding}.json",
        "report": out / f"report.{slug}.{encoding}.md",
        "activations": out / f"activations.{slug}.{{group}}.npz",
        "encoding_comparison": out / f"encoding_comparison.{slug}.json",
        "model_slug": slug,
        "encoding": encoding,
    }


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """加载配置：内置默认 ← YAML 文件 ← 环境变量 ← 显式 overrides。"""
    data = _deep_merge({}, DEFAULT_CONFIG)

    candidate = Path(path) if path else (PROJECT_ROOT / "configs" / "default.yaml")
    if candidate.exists():
        data = _deep_merge(data, loads_yaml(candidate.read_text(encoding="utf-8")))

    # 环境变量覆盖
    for env_key, raw in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        parts = env_key[len(_ENV_PREFIX) :].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section in data and isinstance(data[section], dict):
            data[section][key] = _coerce(raw)

    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, PROJECT_ROOT)


# --------------------------------------------------------------------------- #
# YAML：优先用 PyYAML，缺失时用这个极简实现（只覆盖本项目用到的语法）
# --------------------------------------------------------------------------- #

try:  # pragma: no cover
    import yaml as _pyyaml

    _HAS_PYYAML = True
except Exception:  # pragma: no cover
    _pyyaml = None
    _HAS_PYYAML = False


def loads_yaml(text: str, force_mini: bool = False) -> Dict[str, Any]:
    """解析 YAML。`force_mini=True` 可强制走内置实现（便于测试）。"""
    if _HAS_PYYAML and not force_mini:  # pragma: no cover
        data = _pyyaml.safe_load(text)
        return data if isinstance(data, dict) else (data or {})
    return _MiniYaml(text).parse()


class _MiniYaml:
    """极简 YAML：嵌套映射、列表、行内 [] {}、注释、引号、布尔/数字/null。"""

    def __init__(self, text: str) -> None:
        self.lines: list[tuple[int, str]] = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0] if not self._has_quoted_hash(raw) else raw
            if not line.strip():
                continue
            stripped = line.rstrip()
            indent = len(stripped) - len(stripped.lstrip(" "))
            self.lines.append((indent, stripped.strip()))
        self.pos = 0

    @staticmethod
    def _has_quoted_hash(line: str) -> bool:
        return ('"' in line or "'" in line) and "#" in line and line.find("#") > line.find('"')

    def parse(self) -> Dict[str, Any]:
        if not self.lines:
            return {}
        value, _ = self._block(self.lines[0][0])
        return value if isinstance(value, dict) else {}

    def _block(self, indent: int) -> tuple[Any, int]:
        if self.pos >= len(self.lines):
            return None, indent
        current_indent, content = self.lines[self.pos]
        if current_indent < indent:
            return None, current_indent
        if content == "-" or content.startswith("- "):
            return self._list(current_indent)
        return self._mapping(current_indent)

    def _list(self, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while self.pos < len(self.lines):
            current_indent, content = self.lines[self.pos]
            if current_indent < indent or not (content == "-" or content.startswith("- ")):
                break
            payload = content[1:].strip()
            self.pos += 1
            items.append(_scalar(payload) if payload else None)
        return items, indent

    def _mapping(self, indent: int) -> tuple[Dict[str, Any], int]:
        node: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            current_indent, content = self.lines[self.pos]
            if current_indent < indent:
                break
            if ":" not in content:
                raise ValueError(f"YAML 行缺少冒号: {content!r}")
            key, rest = content.split(":", 1)
            self.pos += 1
            rest = rest.strip()
            if rest:
                node[key.strip()] = _scalar(rest)
            elif self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
                value, _ = self._block(self.lines[self.pos][0])
                node[key.strip()] = value
            else:
                node[key.strip()] = None
        return node, indent


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(x) for x in inner.split(",")] if inner else []
    if text.startswith("{") and text.endswith("}"):
        return {}
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text
