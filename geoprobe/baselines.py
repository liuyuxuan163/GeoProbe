"""基线：证明"隐层真的比廉价特征更有信息"。

**没有基线，这个实验的结论就不成立。**
---------------------------------------
一个常见错误是：跑出 R²=0.8 就宣称"模型编码了地理知识"。
但 R²=0.8 可能来自**伪相关**：

* 以 `-stan` 结尾的地名多在中亚
* 以 `San` 开头的多在西语区
* 字符长度可能与地区命名习惯相关

**纯字符特征**就能捕捉到这些模式。如果不和字符基线比较，
就无法排除"模型只是记住了拼写规律"这一解释。

三个基线
--------
============================  ==================================================
基线                          回答什么问题
============================  ==================================================
``RandomBaseline``            下界。随机向量应该学不到任何东西（R²≈0 或负）。
``CharNgramBaseline``         **最关键**。字符 n-gram TF-IDF → 探针。
                              如果隐层不能明显优于它，"地理知识"的说法就站不住。
``MeanProbe``（在 probe.py）  永远预测训练集均值。R²=0 的参照。
============================  ==================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config

__all__ = ["RandomBaseline", "CharNgramBaseline", "build_baseline_features"]


class RandomBaseline:
    """随机高维向量 → 探针。

    作用：给出"完全没有信息"时的表现。
    **如果真实隐层的表现和它差不多，说明隐层里也没有可用信息。**

    Args:
        dim: 随机向量维度（建议与隐层维度一致，排除"维度差异"这个混淆因素）。
        seed: 随机种子，保证可复现。
    """

    name = "random"

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self.dim = int(dim)
        self.seed = int(seed)

    def transform(self, words: Sequence[str]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, 1.0, size=(len(words), self.dim)).astype(np.float32)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dim": self.dim}


class CharNgramBaseline:
    """字符 n-gram TF-IDF → 探针。

    **这是本项目最重要的基线。** 它只用词表面的字符组成，不使用任何模型。

    如果隐层探针**不能明显优于**它，那么"LLM 编码了地理知识"的结论
    就会被"模型只是记住了拼写规律"所解释，实验就不成立。

    Args:
        ngram_range: 字符 n-gram 的范围，默认 ``(1, 3)``。
            1-gram 捕捉单个字母，3-gram 捕捉 `-stan` 这类后缀模式。
        max_features: 特征数上限，防止维度爆炸。
    """

    name = "char_ngram"

    def __init__(self, ngram_range: Sequence[int] = (1, 3), max_features: int = 2000) -> None:
        self.ngram_range = tuple(int(x) for x in ngram_range)
        self.max_features = int(max_features)
        self._vec: Any = None

    def transform(self, words: Sequence[str]) -> np.ndarray:
        """注意：必须**先在全量词表上 fit**，否则不同组的特征空间不一致。"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._vec is None:
            raise RuntimeError("请先调用 fit_transform 建立词表")
        return self._vec.transform(list(words)).toarray().astype(np.float32)

    def fit_transform(self, words: Sequence[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
        )
        return self._vec.fit_transform(list(words)).toarray().astype(np.float32)

    def describe(self) -> Dict[str, Any]:
        n_features = len(self._vec.get_feature_names_out()) if self._vec is not None else 0
        return {"name": self.name, "ngram_range": list(self.ngram_range), "n_features": n_features}


def build_baseline_features(
    config: Config,
    lists: "Dict[str, Any]",
    hidden_dim: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """为三组词表构造所有基线特征。

    **关键设计：字符 n-gram 的向量化器在"三组词的并集"上 fit。**

    如果每组各自 fit，特征空间不一致，跨组比较就没有意义
    （同一个 n-gram 在不同组里可能对应不同的列）。

    Returns:
        ``{"random": {group: matrix}, "char_ngram": {group: matrix}}``
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    all_words: List[str] = []
    for word_list in lists.values():
        all_words.extend(word_list.words)

    if bool(config.get("baselines.random", True)):
        random_baseline = RandomBaseline(
            dim=int(hidden_dim or 64), seed=int(config.get("probe.random_state", 42))
        )
        out["random"] = {
            group: random_baseline.transform(word_list.words) for group, word_list in lists.items()
        }

    if bool(config.get("baselines.char_ngram", True)):
        rng_range = config.get("baselines.char_ngram_range", [1, 3])
        char_baseline = CharNgramBaseline(ngram_range=rng_range)
        # 在并集上 fit，再分别 transform
        char_baseline.fit_transform(sorted(set(all_words)))
        out["char_ngram"] = {
            group: char_baseline.transform(word_list.words) for group, word_list in lists.items()
        }

    return out
