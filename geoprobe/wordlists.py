"""词表：三组词（对照 / 歧义 / 非地名）与真实坐标。

**实验设计的核心**
------------------
::

    A 组 control   —— 无歧义地名（Paris, Tokyo, Brazil …）
                     探针只在 A 组上训练，学会"隐层 → 坐标"的映射
    B 组 ambiguous —— 歧义地名（Georgia, Turkey, Chile …）
                     ★ 用 A 组训练好的探针来测，看表征被拉向哪个含义
    C 组 nongeo    —— 非地名常用词（apple, window …）
                     负对照：探针应该给不出有意义的位置

**为什么这个设计干净**

探针只在无歧义词上训练，所以它学到的映射是"**纯粹的地理关系**"。
再用歧义词去测试时，输出反映了模型**自身**对那个词的地理表征偏好 ——
没有被训练过程引导。这就是"探针"和"分类器"的区别。

**B 组的 `alt` 字段是本次改造的关键创新**

歧义词往往有**两个地理含义**：

* `Georgia` → 国家格鲁吉亚（主） vs 美国佐治亚州（备）
* `Washington` → 华盛顿特区（主） vs 华盛顿州（备）
* `Valencia` → 西班牙瓦伦西亚（主） vs 委内瑞拉瓦伦西亚（备）

有了备选坐标，就能回答一个比"误差多大"更有意思的问题：
**模型把预测拉向了哪一个含义？**

数据格式
--------
``data/*.json`` 是列表，每项::

    {
      "word": "Georgia",
      "lat": 42.3154, "lon": 43.3569,
      "label": "格鲁吉亚（国家）",
      "alt_lat": 32.1656, "alt_lon": -82.9001,   # 可选
      "alt_label": "美国佐治亚州",                 # 可选
      "note": "美国语境中佐治亚州更常见"
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["WordList", "WordEntry", "load_wordlists", "GROUPS"]

GROUPS = {
    "control": "A 组 · 无歧义地名（探针训练集）",
    "ambiguous": "B 组 · 歧义地名（用 A 组探针测试）",
    "nongeo": "C 组 · 非地名（负对照）",
}


@dataclass
class WordEntry:
    """一个词的完整信息。"""

    word: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    label: str = ""
    alt_lat: Optional[float] = None
    alt_lon: Optional[float] = None
    alt_label: str = ""
    note: str = ""

    @property
    def has_coord(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def has_alt(self) -> bool:
        return self.alt_lat is not None and self.alt_lon is not None

    def coord(self) -> List[float]:
        if not self.has_coord:
            raise ValueError(f"{self.word!r} 没有真实坐标")
        return [float(self.lat), float(self.lon)]

    def alt_coord(self) -> List[float]:
        if not self.has_alt:
            raise ValueError(f"{self.word!r} 没有备选坐标")
        return [float(self.alt_lat), float(self.alt_lon)]


@dataclass
class WordList:
    """一组词。"""

    name: str
    entries: List[WordEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def words(self) -> List[str]:
        return [e.word for e in self.entries]

    @property
    def has_truth(self) -> bool:
        """这一组是否带真实坐标（C 组没有）。"""
        return bool(self.entries) and all(e.has_coord for e in self.entries)

    def coords(self) -> np.ndarray:
        """形状 ``(n, 2)`` 的真实坐标。"""
        if not self.has_truth:
            raise ValueError(f"{self.name} 组缺少真实坐标，无法计算误差")
        return np.array([e.coord() for e in self.entries], dtype=float)

    def with_alt(self) -> List[WordEntry]:
        """只挑出带备选坐标的条目（用于"偏向哪个含义"分析）。"""
        return [e for e in self.entries if e.has_coord and e.has_alt]

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": GROUPS.get(self.name, ""),
            "n": len(self.entries),
            "has_truth": self.has_truth,
            "n_with_alt": len(self.with_alt()),
        }


def _parse_entry(raw: Dict[str, Any]) -> WordEntry:
    return WordEntry(
        word=str(raw["word"]),
        lat=raw.get("lat"),
        lon=raw.get("lon"),
        label=str(raw.get("label", "")),
        alt_lat=raw.get("alt_lat"),
        alt_lon=raw.get("alt_lon"),
        alt_label=str(raw.get("alt_label", "")),
        note=str(raw.get("note", "")),
    )


def load_wordlists(data_dir: Optional[str | Path] = None) -> Dict[str, WordList]:
    """从 ``data/`` 读取三组词表。

    Args:
        data_dir: 数据目录；默认取包所在项目的 ``data/``。

    Returns:
        ``{"control": WordList, "ambiguous": WordList, "nongeo": WordList}``
    """
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
    names = {"control": "words_control.json", "ambiguous": "words_ambiguous.json", "nongeo": "words_nongeo.json"}

    out: Dict[str, WordList] = {}
    for group, filename in names.items():
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"缺少词表文件: {path}（请先运行 scripts/build_wordlist.py 或检查 data/）")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} 应该是 JSON 数组，实际是 {type(raw).__name__}")
        out[group] = WordList(name=group, entries=[_parse_entry(item) for item in raw])

    _validate(out)
    return out


def _validate(lists: Dict[str, WordList]) -> None:
    """基础校验：防止词表写错导致实验静默失效。"""
    control = lists["control"]
    if len(control) < 10:
        raise ValueError(f"control 组只有 {len(control)} 个词，太少（建议 ≥ 40）")
    if not control.has_truth:
        raise ValueError("control 组必须全部带真实坐标")

    # 重复词检测（跨组重复会让"训练/测试"混淆）
    seen: Dict[str, str] = {}
    for group, word_list in lists.items():
        for entry in word_list.entries:
            key = entry.word.strip().lower()
            if key in seen:
                raise ValueError(f"词 {entry.word!r} 在 {seen[key]} 与 {group} 中重复")
            seen[key] = group

    # 坐标合法性
    for group, word_list in lists.items():
        for entry in word_list.entries:
            if not entry.has_coord:
                continue
            if not (-90 <= entry.lat <= 90):
                raise ValueError(f"{entry.word}: 纬度越界 {entry.lat}")
            if not (-180 <= entry.lon <= 180):
                raise ValueError(f"{entry.word}: 经度越界 {entry.lon}")
