"""GeoProbe —— LLM 地理知识探针。

研究问题
--------
**LLM 的内部表征里，是否编码了地理坐标？**

做法是 **probing（探针）**：抓取模型的隐层激活，用一个**简单的线性模型**（Ridge 回归）
尝试从激活预测经纬度。如果预测得准，说明地理信息在该层是**线性可解码**的 ——
也就是**显式存在**于表征中，而不是只能靠"输出文本时碰巧答对"。

与课程原实验的三个关键差异
--------------------------
1. **指标换成 Haversine 公里误差**。原实验用 R² 和"度的平方"，没有物理意义；
   而且 R² 对经纬度两维取平均会糊掉信息。公里误差能直接回答"错多远"。
2. **加三个基线**。尤其是**字符 n-gram 基线** —— 没有它就无法排除
   "模型只是记住了拼写模式"这种伪相关。（以 `-stan` 结尾的地名多在中亚，
   纯字符特征就能拿到不低的 R²。）
3. **切口是歧义地名**。Georgia 既是国家也是美国州，Turkey 既是国家也是火鸡。
   探针只在**无歧义地名**上训练，再用歧义词测试，看表征被拉向哪个含义。

模块地图
--------
::

    config.py      配置加载（YAML + 环境变量覆盖）
    metrics.py     Haversine 距离、bootstrap 置信区间、误差统计
    wordlists.py   三组词表（对照 A / 歧义 B / 非地名 C）与真实坐标
    extract.py     模型加载、层路径兼容、隐层提取、三种池化
    targets.py     目标编码（经度循环编码 sin/cos，修复线性编码的缺陷）
    probe.py       Ridge / MLP 探针（含 alpha 网格搜索）
    baselines.py   随机基线、字符 n-gram 基线、均值基线
    experiment.py  实验编排：训练探针 → 三组测试 → 汇总
    report.py      Markdown 报告 + 图表
    cli.py         命令行入口
"""

from .config import Config, load_config  # noqa: F401
from .metrics import ErrorStats, bootstrap_ci, haversine_km  # noqa: F401
from .targets import TargetEncoder  # noqa: F401
from .wordlists import WordList, load_wordlists  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "ErrorStats",
    "bootstrap_ci",
    "haversine_km",
    "TargetEncoder",
    "WordList",
    "load_wordlists",
]
