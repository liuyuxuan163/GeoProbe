"""探针：从隐层表征预测经纬度。

**什么是探针（probe）**
----------------------
如果用一个**简单的模型**能从隐层表征 h 预测出某个属性 y，
就说明该属性在表征中是**可解码的（decodable）**。

本项目用两种探针，目的不同：

===================  ==========================================================
探针                  用途
===================  ==========================================================
``RidgeProbe``        **主探针**。线性 → 只能解码"显式存在"的信息。
                      线性可解码意味着信息"摆在那里"，不需要复杂变换。
``MLPProbe``          **对照**。如果 Ridge 差而 MLP 好 → 信息存在但非线性编码；
                      如果两者都差 → 信息基本不存在。
===================  ==========================================================

**为什么用 Ridge 而不是普通最小二乘（OLS）**
------------------------------------------
隐层维度（Qwen2.5-0.5B 是 896）**远大于样本数**（A 组 108 个词）。
这种"宽数据"下 OLS 会过拟合到完美、甚至无解。
Ridge 的 L2 正则让解唯一且稳定，并靠 `alpha` 控制"记住多少训练集"。

**为什么 alpha 要在训练集内部用 CV 选**
-------------------------------------
如果拿测试集选 alpha，就是**数据泄露** —— 报出来的指标会虚高。
这里用 `RidgeCV`（内部 K 折），测试集从头到尾不参与任何选择。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config
from .metrics import ErrorStats, evaluate_predictions

__all__ = ["BaseProbe", "RidgeProbe", "MLPProbe", "MeanProbe", "build_probe", "ProbeResult"]


class BaseProbe:
    """探针基类：约定 `fit(X, Y)` / `predict(X)`。"""

    name = "base"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "BaseProbe":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


class RidgeProbe(BaseProbe):
    """岭回归探针（主探针）。

    Args:
        alphas: 候选正则强度。**在训练集内部用 K 折交叉验证选最优**，
            绝不使用测试集。
        cv: 折数。
    """

    name = "ridge"

    def __init__(self, alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), cv: int = 5) -> None:
        self.alphas = list(alphas)
        self.cv = int(cv)
        self.model: Any = None
        self.alpha_: Optional[float] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeProbe":
        from sklearn.linear_model import RidgeCV

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        # 折数不能超过样本数
        folds = max(2, min(self.cv, len(X)))
        self.model = RidgeCV(alphas=self.alphas, cv=folds)
        self.model.fit(X, Y)
        self.alpha_ = float(self.model.alpha_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("探针尚未 fit")
        return np.asarray(self.model.predict(np.asarray(X, dtype=np.float64)), dtype=float)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha_, "cv": self.cv}


class MLPProbe(BaseProbe):
    """MLP 探针（对照用）。

    如果显著优于 Ridge，说明信息**存在但非线性编码** —— 这本身是个有价值的发现。
    不优于 Ridge，说明线性已经足够（或信息不存在）。
    """

    name = "mlp"

    def __init__(self, hidden: int = 256, max_iter: int = 500, random_state: int = 42) -> None:
        self.hidden = int(hidden)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.model: Any = None
        self._scaler: Any = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MLPProbe":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        # 隐层维度的量级差异很大，MLP 对尺度敏感 → 必须标准化
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden,),
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.model.fit(Xs, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self._scaler is None:
            raise RuntimeError("探针尚未 fit")
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float64))
        return np.asarray(self.model.predict(Xs), dtype=float)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "hidden": self.hidden}


class MeanProbe(BaseProbe):
    """基线：永远预测训练集的坐标均值。

    它的 Haversine 误差给出了"完全不懂地理"的参照值 ——
    任何探针都必须显著优于它，否则等于没学到东西。
    """

    name = "mean"

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MeanProbe":
        self.mean_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("探针尚未 fit")
        X = np.asarray(X)
        return np.tile(self.mean_, (len(X), 1))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


def build_probe(config: Config, kind: str = "ridge") -> BaseProbe:
    """按配置构造探针。"""
    if kind == "ridge":
        return RidgeProbe(
            alphas=config.get("probe.ridge_alphas", [0.01, 0.1, 1.0, 10.0]),
            cv=int(config.get("probe.cv_folds", 5)),
        )
    if kind == "mlp":
        return MLPProbe(
            hidden=int(config.get("probe.mlp_hidden", 256)),
            max_iter=int(config.get("probe.mlp_max_iter", 500)),
            random_state=int(config.get("probe.random_state", 42)),
        )
    if kind == "mean":
        return MeanProbe()
    raise ValueError(f"未知探针类型: {kind!r}")


@dataclass
class ProbeResult:
    """一次"训练 + 评估"的完整结果。"""

    probe_name: str
    train_group: str
    test_group: str
    stats: ErrorStats
    predictions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    truth: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    words: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.probe_name}@{self.test_group}"


def run_probe(
    probe: BaseProbe,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    test_group: str,
    words_test: Sequence[str],
    Y_test: Optional[np.ndarray] = None,
    config: Optional[Config] = None,
    X_test: Optional[np.ndarray] = None,
) -> ProbeResult:
    """训练探针并在一个测试组上评估。

    Args:
        Y_test: 真实坐标。``None`` 表示该组没有真值（C 组），
            此时只返回预测坐标，误差统计为空。
    """
    probe.fit(X_train, Y_train)
    if X_test is None:
        raise ValueError("必须提供 X_test")

    pred = probe.predict(X_test)
    detail = probe.describe()
    detail["n_train"] = int(len(X_train))

    if Y_test is None:
        return ProbeResult(
            probe_name=probe.name,
            train_group="control",
            test_group=test_group,
            stats=ErrorStats(),
            predictions=pred,
            words=list(words_test),
            detail=detail,
        )

    cfg = config
    stats = evaluate_predictions(
        Y_test,
        pred,
        bootstrap_samples=int(cfg.get("evaluation.bootstrap_samples", 1000)) if cfg else 1000,
        ci=float(cfg.get("evaluation.ci", 0.95)) if cfg else 0.95,
        percentile_list=cfg.get("evaluation.percentiles", [50, 75, 90]) if cfg else [50, 75, 90],
        seed=int(cfg.get("probe.random_state", 42)) if cfg else 42,
    )
    return ProbeResult(
        probe_name=probe.name,
        train_group="control",
        test_group=test_group,
        stats=stats,
        predictions=pred,
        truth=np.asarray(Y_test, dtype=float),
        words=list(words_test),
        detail=detail,
    )
