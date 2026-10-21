# `run_experiment` 逐行注解版

> **适用场景**：你需要在**今天之内**把这个实验讲给别人听（面试、组会、导师汇报）。
>
> **读法**：先看「全局地图」（1 分钟），再看七段注解（每段都有"写错会怎样"），
> 最后背下「一句话总结」。
>
> **对应代码**：`geoprobe/experiment.py` L243–L420（共 178 行，有效代码约 90 行）。
>
> ⚠ 行号会随代码演进而偏移。**以函数名和段落注释（`# ---` / `# ★`）为准**。

---

## 全局地图（1 分钟看懂）

```
run_experiment(config, ...)  →  ExperimentResult
│
├─ S1  L243–264   准备：取出三组词
├─ S2  L266–277   取数据：隐层矩阵 X + 真实坐标 Y
├─ S3  L279–302   配置：目标编码 + 结果容器 + 累积器
├─ S4  L304–325   ★ 主循环：A 组 训练 → 评估        ← 每重复一次
├─ S5  L327–340   ★ 主循环：B 组 用同一个探针测       ← 关键设计
├─ S6  L342–380   汇总：代表值 + 汇总 R² + 含义偏好
└─ S7  L382–420   收尾：负对照 + 四个基线 + 警告
```

**一句话概括整个函数**：

> **在三组词上各取隐层，用 A 组训练线性探针，把同一个探针套到 B、C 组上，
> 用 Haversine 公里误差和置换检验判断"地理信息到底在不在表征里"。**

---

## S1｜准备：取出三组词（L243–264）

```python
243  def run_experiment(
244      config: Config,
245      lists: Optional[Dict[str, WordList]] = None,
246      extract_results: Optional[Dict[str, ExtractResult]] = None,
247      layer: Optional[int] = None,
248      probe_kinds: Sequence[str] = ("ridge",),
249      n_repeats: int = 5,
250      test_size: float = 0.2,
251      run_baselines: bool = True,
252  ) -> ExperimentResult:
253      """跑完整实验。"""
261      lists = lists or load_wordlists(config.path_of("data"))
262      control = lists["control"]
263      ambiguous = lists["ambiguous"]
264      nongeo = lists["nongeo"]
```

### 逐行

| 行 | 做什么 | 为什么 |
|---|---|---|
| L245 | `lists=None` 时自己加载 | 便于测试时注入假词表 |
| L246 | `extract_results=None` 时自己提取 | **这是逐层扫描 bug 的来源**，见下方"陷阱" |
| L247 | `layer=None` → 最后一层 | 默认行为，但**不是最好的层** |
| L249 | `n_repeats=5` | 重复划分 5 次，抵抗单次抽样的运气 |
| L250 | `test_size=0.2` | A 组 108 词，留出约 22 个 |
| L261 | `x or default` 的惯用法 | 传了就用传的，没传就加载默认 |

### 为什么参数这么多

三个"可选注入点"（`lists` / `extract_results` / `layer`）都是为了**同一个目的**：
让这个函数**既能独立跑，也能被复用**。

```python
# 独立跑（命令行默认）
run_experiment(config)

# 逐层扫描：提取一次，循环复用
cached = extract_dataset(config, lists, layer_indices=list(range(24)))
for layer in range(24):
    run_experiment(config, extract_results=cached, layer=layer)   # ← 不重复提取
```

### ⚠ 陷阱：`extract_results=None` 的默认行为

**`extract_dataset()` 默认只提取最后一层**（因为单层实验只需要它）。

所以如果你想要第 0 层却忘了传 `layer_indices`，会炸：

```
KeyError: 层 0 不在结果里（已提取 [23]）
```

**本项目真实踩过这个坑**（`layer-sweep` 早期版本因此完全不可用）。
现在报错信息里直接写了修复方法。

---

## S2｜取数据：隐层矩阵 X + 真实坐标 Y（L266–277）

```python
266      if extract_results is None:
267          extract_results = extract_dataset(config, lists)
269      control_extract = extract_results["control"]
270      ambiguous_extract = extract_results["ambiguous"]
271      nongeo_extract = extract_results["nongeo"]
273      actual_layer = control_extract.layers[-1] if layer is None else int(layer)
274      X_control = control_extract.matrix(actual_layer)      # (108, 896)
275      X_ambiguous = ambiguous_extract.matrix(actual_layer)  # (27, 896)
276      Y_control = control.coords()                          # (108, 2)
277      Y_ambiguous = ambiguous.coords()                      # (27, 2)
```

### 逐行

| 行 | 变量 | 形状 | 含义 |
|---|---|---|---|
| L273 | `actual_layer` | int | 实际用哪一层 |
| L274 | `X_control` | 108 × 896 | A 组的隐层向量（**探针的输入**） |
| L275 | `X_ambiguous` | 27 × 896 | B 组的隐层向量 |
| L276 | `Y_control` | 108 × 2 | A 组真实 `[lat, lon]`（**探针的目标**） |
| L277 | `Y_ambiguous` | 27 × 2 | B 组真实坐标（只用于**评估**，不参与训练） |

### 记法

```
X = 隐层（模型给的）      Y = 真实坐标（数据给的）
大写 X/Y + 形状 (n, d)    这是机器学习里最常见的命名习惯
```

### 为什么 B 组的 Y 只用于评估

因为探针**只在 A 组训练**。`Y_ambiguous` 从头到尾只出现在
`evaluate_predictions(Y_ambiguous, pred_ambiguous)` 里 —— 那是**打分**，不是学习。

**如果 B 组的标签参与了训练，整个实验的结论就失效了** ——
歧义词的表现会被"教"出来，不再是"模型自己怎么理解"。

---

## S3｜配置：目标编码 + 结果容器 + 累积器（L279–302）

```python
279      # ★ 目标编码：经度是循环量，线性编码对它有偏。
282      encoder = TargetEncoder(str(config.get("probe.target_encoding", "latlon")))
283      Y_control_enc = encoder.encode(Y_control)        # latlon: (108,2) / sincos: (108,3)
284      Y_ambiguous_enc = encoder.encode(Y_ambiguous)    # ⚠ 定义了但没用（死代码）
285      reliability_floor = float(config.get("evaluation.alt_reliability_km", 2500.0))
287      result = ExperimentResult(
288          model_name=control_extract.model_name,
289          mock=control_extract.mock,
290          pooling=control_extract.pooling,
291          layer=actual_layer,
292          n_layers=len(control_extract.layers),
293          target_encoding=encoder.mode,
294      )
296      primary_kind = probe_kinds[0]
297      control_repeats = RepeatStats()      # 累积每次重复的中位误差
298      ambiguous_repeats = RepeatStats()
299      all_control_preds: List[np.ndarray] = []   # 累积预测，供汇总 R² 用
300      all_control_truth: List[np.ndarray] = []
301      ambiguous_preds: List[np.ndarray] = []     # 供"多重复平均"用
302      last_probe = None                          # 最后一次重复的探针，供 C 组用
```

### 核心：为什么要 `Y_control_enc`

```
真实坐标 Y_control        = [48.86, 2.35]        ← 巴黎，线性经纬度
编码后 Y_control_enc      = [48.86, sin(2.35°), cos(2.35°)]   ← sincos 模式
```

**探针学习的目标是编码后的值**，不是原始经纬度。

**为什么**：经度是循环量。179° 和 -179° 地理上相邻（差 2°），
线性目标空间里却差 358°，线性模型无法表达这种周期。

### 三个累积器各自的用途

| 累积器 | 用来算什么 | 为什么需要累积 |
|---|---|---|
| `control_repeats.medians` | 中位误差的**均值 ± 标准差** | 单个数字看不出稳定性 |
| `all_control_preds` + `all_control_truth` | **汇总 R²**（L348） | 22 个点算 R² 抽样噪声极大 |
| `ambiguous_preds` | B 组**平均预测**再做逐词分析 | 单次预测抖动大，平均更稳 |
| `last_probe` | C 组负对照 | C 组没有真值，只预测 |

### ⚠ 静默失败的风险：`target_encoding` 打错字

```python
config.get("probe.target_encoding", "latlon")   # 默认值兜底
```

如果配置里写成 `"sin-cos"`，`TargetEncoder` 会抛 `ValueError` —— 这是好的。
但如果你自己写 `config.get(...) or "latlon"`，**空字符串会被当成 falsy 悄悄替换掉**，
结果是你以为在用 sincos、实际在用 latlon。

**所以这里用 `str(...)` 而不是 `or`。**

---

## S4｜主循环 · A 组：训练 → 评估（L304–325）

**这是整个实验的心脏，只有 22 行。**

```python
304      for repeat in range(max(1, int(n_repeats))):
305          seed = int(config.get("probe.random_state", 42)) + repeat
306          train_idx, test_idx = _split_train_test(len(X_control), test_size, seed)
308          probe = build_probe(config, primary_kind)
310          probe.fit(X_control[train_idx], Y_control_enc[train_idx])   # ← 只在 A 组训练
312          # --- A 组留出集：先解码回经纬度，再用 Haversine 评估 ---
313          pred_control = encoder.decode(probe.predict(X_control[test_idx]))
314          stats_control = evaluate_predictions(
315              Y_control[test_idx],
316              pred_control,
317              bootstrap_samples=..., ci=..., percentile_list=..., seed=seed,
321          )
322          control_repeats.medians.append(stats_control.median_km)
323          control_repeats.means.append(stats_control.mean_km)
324          all_control_preds.append(pred_control)
325          all_control_truth.append(Y_control[test_idx])
```

### 逐行

| 行 | 做什么 | 关键点 |
|---|---|---|
| L304 | 重复 `n_repeats` 次 | `max(1, ...)` 保护：至少跑 1 次 |
| L305 | 每次换一个种子 | `42, 43, 44...` → 每次划分不同 |
| L306 | 切分 86 训练 / 22 留出 | **每次划分都不同** |
| L308 | 新建探针 | 每次重复都从零开始，不复用 |
| **L310** | **在编码空间拟合** | `Y_control_enc` 而不是 `Y_control` |
| **L313** | **先解码，再评估** | 把 sincos 预测还原成经纬度 |
| L315 | 用**原始**真实坐标打分 | `Y_control` 而不是 `Y_control_enc` |
| L322–325 | 累积 | 供后面算统计 |

### ⭐ 最关键的一对：L310 与 L313

```python
# 训练：在编码空间
probe.fit(X_control[train_idx], Y_control_enc[train_idx])
#     ↑ 输入是隐层            ↑ 目标是 (lat, sin, cos)

# 预测：先拿编码空间的输出
raw = probe.predict(X_control[test_idx])      # (22, 3)
# 再解码回真实经纬度
pred_control = encoder.decode(raw)            # (22, 2) → [lat, lon]
```

**为什么必须"编码空间训练、真实空间评估"**

| 环节 | 用什么空间 | 原因 |
|---|---|---|
| 训练 | 编码空间（sincos） | 让目标连续，线性模型能拟合 |
| 评估 | 真实空间（经纬度） | Haversine 需要真实经纬度 |

**如果忘了 L313 的解码会怎样**：

```python
evaluate_predictions(Y_control[test_idx], raw)   # ✗ raw 是 3 维（lat, sin, cos）
```

`haversine_errors` 会检查形状并抛 `ValueError: 期望形状 (n, 2)，实际 (22, 3)`。
**这是好的设计** —— 静默地把 `sin` 当经度用会得到完全错误但"看起来合理"的公里数。

### 为什么每次重复都新建探针

```python
for repeat in ...:
    probe = build_probe(...)      # ← 每次新建
    probe.fit(...)                # ← 重新拟合
```

不是浪费。**每个 arm 必须是"只用这次的训练集拟合的探针"**，
否则就变成了拿所有数据训练、再拿其中一部分当测试 —— 数据泄露。

### 为什么报中位数和均值两个

| 指标 | 特点 | 什么时候看 |
|---|---|---|
| **中位数** | 稳健，回答"典型情况错多远" | **主指标** |
| 均值 | 被长尾拉高 | 看长尾有多严重 |

实测：中位 2,311 km、均值 2,937 km（比值 1.2）→ 说明有少数词错得极远。

---

## S5｜主循环 · B 组：用**同一个**探针测（L327–340）

**这 14 行是整个实验设计的核心。**

```python
327          # --- B 组（用同一个探针）---
328          pred_ambiguous = encoder.decode(probe.predict(X_ambiguous))
329          stats_ambiguous = evaluate_predictions(
330              Y_ambiguous,
331              pred_ambiguous,
332              bootstrap_samples=..., ci=..., percentile_list=..., seed=seed,
336          )
337          ambiguous_repeats.medians.append(stats_ambiguous.median_km)
338          ambiguous_repeats.means.append(stats_ambiguous.mean_km)
339          ambiguous_preds.append(pred_ambiguous)
340          last_probe = probe
```

### 逐行

| 行 | 做什么 | **不能改成什么** |
|---|---|---|
| L328 | 用**刚才 A 组训练的** `probe` 预测 B 组 | ✗ 不能在这里 `probe.fit(X_ambiguous, ...)` |
| L330 | 用 B 组**真实的**坐标打分 | ✗ 不能拿 B 组标签参与任何训练 |
| L339 | 累积 B 组预测 | 供"多重复平均"做逐词分析 |
| L340 | 存下这个探针 | C 组负对照要用 |

### ⭐ 一行之差就毁掉整个实验

```python
# ✓ 正确
probe.fit(X_control[train_idx], Y_control_enc[train_idx])
pred_ambiguous = probe.predict(X_ambiguous)      # 直接预测，不重新训练

# ✗ 错误（实验完全失效）
probe.fit(X_ambiguous, Y_ambiguous_enc)          # B 组也参与训练了
pred_ambiguous = probe.predict(X_ambiguous)      # 这是在测"背得熟不熟"
```

**为什么这一行如此关键**

探针只在无歧义地名上训练过，所以它学到的是"**纯粹的地理映射**"。
拿它去测 `Georgia` 时，输出反映的是**模型自己**对这个词的地理表征偏好。

一旦让 `Georgia` 参与训练，探针就被告知了答案 ——
你再也没法区分"模型就是这么理解的"和"探针被教会了"。

> **这就是"探针（probe）"和"分类器（classifier）"的本质区别：**
> **探针不允许在它要检验的东西上训练。**

### `X_ambiguous` 的形状为什么和 `train_idx` 无关

```python
# A 组：要切分，因为既要训练又要测试
probe.fit(X_control[train_idx], ...)
probe.predict(X_control[test_idx])

# B 组：全部用于测试，不切分
probe.predict(X_ambiguous)          # 27 个全测
```

B 组的 27 个词**一个都没参与训练**，所以全部都可以用来测试。

---

## S6｜汇总：代表值 + 汇总 R² + 含义偏好（L342–380）

### S6-a：把最后一次重复当"代表值"（L342–346）

```python
342      # 用最后一次重复的完整统计作为代表值（重复统计另外给出）
343      result.control = stats_control
344      result.ambiguous = stats_ambiguous
345      result.control_repeats = control_repeats
346      result.ambiguous_repeats = ambiguous_repeats
```

`stats_control` 是循环里最后一次赋值的完整 `ErrorStats`，含：
`median_km / mean_km / p90_km / ci_low / ci_high / r2_lat / r2_lon`。

**为什么保留它**：报告里要展示 P90、置信区间这些**单次划分才能算的量**。
而多次重复的稳定性由 `*_repeats` 单独给出。

> ⚠ **注意**：`result.control.r2_lat` 是**单次划分**的值，**不要用它下结论**。
> 下面 S6-b 的 `pooled_r2_*` 才是可信的。

### S6-b：⭐ 汇总 R²（L348–361）

```python
348      # ★ 汇总 R²：把所有重复的留出预测**拼起来**再算。
354      if all_control_preds:
355          pooled_pred = np.vstack(all_control_preds)    # (5×22, 2) = (110, 2)
356          pooled_truth = np.vstack(all_control_truth)
357          result.pooled_r2_lat = _r2_dim(pooled_truth[:, 0], pooled_pred[:, 0])
358          result.pooled_r2_lon = _r2_dim(pooled_truth[:, 1], pooled_pred[:, 1])
359          # ★ 经度专用：圆形 R²（正确处理 180° 环绕）
360          result.pooled_circular_r2_lon = circular_r2(pooled_truth[:, 1], pooled_pred[:, 1])
361          result.pooled_n = int(len(pooled_pred))
```

**`np.vstack` 在这里做什么**

```python
all_control_preds = [ (22,2), (22,2), (22,2), (22,2), (22,2) ]
np.vstack(...)    = (110, 2)     ← 5 次划分的预测纵向拼在一起
```

**为什么要拼**

同一个层、同一编码，只换随机种子，单次 R²(经度) 的 8 个值是：

```
-0.33, 0.84, 0.82, 0.34, 0.12, 0.16, 0.21, 0.82
```

跨度 1.177。**因为每次只有 22 个样本。**

拼起来后 n=110（或 176），R² 才稳定可比。

**为什么经度要用 `circular_r2` 而不是 `_r2_dim`**

```python
_r2_dim          → 残差 = truth - pred          （179 - (-179) = 358 ❌）
circular_r2      → 残差 = 最短角距离             （179 到 -179 = 2 ✅）
```

L358 保留 `_r2_dim` 只是**留作对照**，字段名已标注 `deprecated`。

### S6-c：歧义放大倍数（L363–365）

```python
363      result.ambiguity_ratio, result.ambiguity_comment = ratio_with_comment(
364          ambiguous_repeats.median_mean, control_repeats.median_mean
365      )
```

`ratio_with_comment` 除了算倍数，还返回一句人话判定：

| 倍数 | 判定 |
|---|---|
| ≥ 2.0 | 歧义显著放大 |
| ≥ 1.3 | 轻微放大 |
| ≥ 0.8 | 无明显影响 |
| < 0.8 | 歧义词反而更准 |

**为什么不直接写 `ratio = a / b`**：因为判定阈值应该是**可配置、可单测**的，
不该散落在业务代码里。

### S6-d：⭐ 含义偏好 + 自适应门槛（L367–380）

```python
367      # --- 用平均预测做逐词分析（多重复平均更稳）---
368      mean_ambiguous_pred = np.mean(np.stack(ambiguous_preds), axis=0)
369      result.ambiguous_detail = list_errors(ambiguous.words, Y_ambiguous, mean_ambiguous_pred)
371      # 「偏向判定」的**自适应**可靠性门槛：
377      result.alt_reliability_used = max(reliability_floor, float(control_repeats.median_mean))
378      result.alt_analysis = alt_preference_analysis(
379          ambiguous.entries, mean_ambiguous_pred, reliability_km=result.alt_reliability_used
380      )
```

**L368 为什么先平均**

`ambiguous_preds` 是 5 次重复的预测。直接取第 1 次会很抖；
平均之后更接近"模型的稳定倾向"。

**L377 的自适应门槛 —— 这是本项目修复的一个真实解读陷阱**

```
固定门槛 2500 km  <  模型自身中位误差 3103 km
→ 会把模型定位正常的词也判成"不可靠"
→ 连真正有信号的 Valencia（2990 km）都被误排除
```

所以改成：

```python
alt_reliability_used = max(配置下限, 模型自身中位误差)
```

**判据的自然含义**：*"最近候选至少要和模型平时一样准，才谈得上'偏向'。"*

**效果对比**（同一份数据）：

| | 加门槛前 | 加门槛后 |
|---|---|---|
| 判定可靠 | 20 / 20 | **8 / 20** |
| 偏备选含义 | "6 个" | **1 个** |

那 5 个被排除的（Georgia / Colombia / Santiago / Mexico / Panama），
最近候选都在 4,000 km 以上 —— 所谓"偏向"全是噪声。

---

## S7｜收尾：负对照 + 基线 + 警告（L382–420）

### S7-a：C 组负对照（L382–387）

```python
382      # --- C 组负对照（无真值，只记录预测坐标的离散度）---
383      pred_nongeo = last_probe.predict(nongeo_extract.matrix(actual_layer)) if last_probe else np.zeros((0, 2))
384      result.nongeo_predictions = [
385          {"word": w, "pred_lat": float(p[0]), "pred_lon": float(p[1])}
386          for w, p in zip(nongeo.words, pred_nongeo)
387      ]
```

C 组是 `apple / window / table` 这类**非地名**，没有真值 → 无法算误差。

**它的作用**：看探针在"没有信息"时会不会乱编坐标。

实测结果：

```
经度分布：-0.2 ± 0.3°      ← 几乎完全集中在一个点
纬度分布：14.6 ± 12.4°
```

**探针在无信息时退化输出同一个位置** —— 这证明它没在乱猜。

> **这类"负对照"在实验设计里非常重要**：它检查的是
> "我的方法会不会给任何输入都吐出一个看起来合理的答案"。

### S7-b：四个基线（L389–397）

```python
389      # --- 基线 ---
390      if run_baselines:
391          result.baselines = _run_baselines(config, lists, X_control, Y_control, test_size, encoder)
392          result.baselines.setdefault("hidden", {})["median_km"] = control_repeats.median_mean
393          result.baselines["hidden"]["median_std_km"] = control_repeats.median_std
395      # --- 备选探针（MLP 对照）---
396      if "mlp" in probe_kinds:
397          result.baselines["mlp"] = _run_single_probe(...)
```

L392 用 `setdefault` 是为了**兼容** —— 万一 `_run_baselines` 已经填了 `hidden`，
不覆盖它。

四个基线各自排除什么：

| 基线 | 排除的解释 |
|---|---|
| `mean` | "探针什么都没学到" |
| `random` | "随便给个数也算" |
| **`char_ngram`** | **"模型只是记住拼写规律"** |
| **`permuted`** | **"探针在乱猜也能碰巧对"** |

> **`run_baselines=False` 的用处**：逐层扫描时，只有最后一层需要跑基线 ——
> 基线比主实验慢好几倍（要重复拟合约 4 倍次数）。

### S7-c：警告（L399–418）

```python
399      # --- 警告 ---
411      if not result.signal_ok:
412          result.warnings.append(
413              "⚠ 探针未通过置换检验 …… 此时 B 组的结果不应被解读为「歧义效应」。"
415          )
```

**这是本项目最重要的工程习惯之一**：

> **当结论不可信时，报告要主动写出来，而不是硬给一个漂亮数字。**

`signal_ok` 的判据（在 `ExperimentResult` 里）：

```python
control.median_km < 0.6 × permuted.median_km
```

**为什么是 0.6 而不是 0.9**：实测证明纯噪声特征在"预测均值"基线下也能
"通过"到 0.9 的宽松阈值（因为过拟合后个别点更接近真值，把中位数拉低）。
0.6 配合置换检验才有区分度。

---

## 附：这个函数里的两处死代码

读代码时你可能会困惑，提前说明 —— **都是无害的**：

### ① `Y_ambiguous_enc`（L284）定义了但从未使用

```python
284      Y_ambiguous_enc = encoder.encode(Y_ambiguous)     # 之后再没出现
```

**为什么会有**：作者（我）当初想"B 组也要编码"，但后来发现
B 组只需要 `encoder.decode(probe.predict(...))` —— 编码后的目标根本不用。

**为什么不删**：删了没收益、还可能有人依赖。但它确实会让人误以为
"B 组也参与训练"，**属于该清理的技术债**。

### ② L400 的分支不可达

```python
266      if extract_results is None:
267          extract_results = extract_dataset(config, lists)     # ← 这里已经兜底
...
400      if extract_results is None:                             # ← 永远为 False
401          result.warnings.append("未提供 extract_results，内部已重新提取")
```

到 L400 时 `extract_results` 必然非 None，所以这个警告永远发不出来。

**教学价值**：这是"防御性编程"写多了的典型产物。
**判据是"能不能被测试覆盖"** —— 不可达分支就是死代码。

---

## 一句话总结（背这个）

> `run_experiment` 做三件事：
>
> **① 取 A/B/C 三组的隐层；**
> **② 在 A 组上训练一个线性探针预测经纬度，重复 5 次不同划分；**
> **③ 把同一个探针套到 B 组（歧义地名）和 C 组（非地名）上，**
> **用 Haversine 公里误差、汇总圆形 R²、四重基线（含置换检验）判断结论是否成立。**
>
> **核心约束只有一条：探针绝不在它要检验的东西上训练。**

---

## 动手验证（10 分钟）

把主循环手工重跑一遍，打印中间变量 —— 这是确认"你真的看懂了"的最快方式。

```python
# 保存为 scripts/_walkthrough.py，然后 python scripts/_walkthrough.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geoprobe.config import load_config
from geoprobe.experiment import _split_train_test
from geoprobe.extract import ExtractResult
from geoprobe.metrics import evaluate_predictions
from geoprobe.probe import build_probe
from geoprobe.targets import TargetEncoder
from geoprobe.wordlists import load_wordlists

config = load_config()
lists = load_wordlists()
cached = {g: ExtractResult.load(ROOT / "reports" / f"activations.{g}.npz") for g in lists}

LAYER = 5
encoder = TargetEncoder("sincos")
X = cached["control"].matrix(LAYER)
Y = lists["control"].coords()
Y_enc = encoder.encode(Y)

print(f"X 形状 {X.shape}   Y 形状 {Y.shape}   Y_enc 形状 {Y_enc.shape}")

train_idx, test_idx = _split_train_test(len(X), 0.2, 42)
print(f"训练 {len(train_idx)} 个，留出 {len(test_idx)} 个")

probe = build_probe(config, "ridge").fit(X[train_idx], Y_enc[train_idx])
print(f"选中的 alpha = {probe.alpha_}")

raw = probe.predict(X[test_idx])
pred = encoder.decode(raw)
print(f"原始输出形状 {raw.shape} → 解码后 {pred.shape}")

stats = evaluate_predictions(Y[test_idx], pred, bootstrap_samples=200, seed=42)
print(f"中位误差 {stats.median_km:,.0f} km   95%CI [{stats.ci_low:,.0f}, {stats.ci_high:,.0f}]")
print(f"单次 R²(lat/lon) = {stats.r2_lat:.3f} / {stats.r2_lon:.3f}   ← 单次，不可信")
```

### 实际输出（本机实测）

```
X 形状 (108, 896)   Y 形状 (108, 2)   Y_enc 形状 (108, 3)
训练 86 个，留出 22 个
选中的 alpha = 10.0
原始输出形状 (22, 3) → 解码后 (22, 2)
中位误差 2,156 km   95%CI [1,490, 2,768]
单次 R²(lat/lon) = 0.418 / -0.332   ← 单次，不可信
```

### 这份输出本身就是最好的教学材料

**注意最后一行**：`R²(经度) = -0.332`（**负的**）。

而正式实验报告的圆形 R²(经度) 是 **0.80**。

同一个层、同一个编码、同一份数据，**只差"用哪 22 个词做留出集"**：

```
这一次的 22 个词 → R²(经度) = -0.33
另外 8 次不同的 22 个词 → 在 -0.33 到 0.85 之间跳
汇总 176 个词 → 圆形 R²(经度) = 0.80
```

> **这就是"S6-b 汇总 R²"存在的全部理由。**
> 如果你只跑一次、只看这个 -0.332，你会得出"模型完全不编码经度"的结论。
> **一个负数看起来像铁证，其实只是抽样噪声。**

### 三个要核对的地方

| 检查项 | 应该看到 | 如果不对 |
|---|---|---|
| `Y_enc` 形状 | `(108, 3)` | 若是 `(108, 2)` → 编码器没生效 |
| 原始输出 vs 解码后 | `(22,3)` → `(22,2)` | 若没解码 → 后面 Haversine 会报形状错 |
| `alpha` 值 | 一个具体数（如 10.0） | 若为 `None` → 探针没 fit 成功 |

---

## 最后：三个"如果面试官追问"的回答

### Q1：为什么重复 5 次，而不是跑一次更彻底（比如 5 折交叉验证）？

> 两者目的一样：降低单次划分的运气成分。我选重复随机划分是因为它
> **允许每次留出集大小一致**（都是 22 个），便于把预测拼起来算汇总 R²。
>
> 5 折的话每折留出 21.6 个，而且"折"之间的训练集高度重叠。
> 更重要的是：我在实测中发现**单次划分的 R² 噪声可达 ±0.5**，
> 所以重复次数比"折"的设计更关键 —— 后来我把重复加到 8 次看稳定性。

### Q2：为什么探针用线性（Ridge），不用更强的模型？

> 因为**线性可解码**才是"信息显式存在"的证据。
> 如果必须用深层网络才能解码，那信息可能是网络自己"创造"的，
> 或者只是非线性变换后的副产品。
>
> 我也实现了 MLP 探针作对照（`--with-mlp`）：
> 如果 Ridge 差但 MLP 好，说明信息存在但非线性编码 —— 那是另一种有价值的结论。

### Q3：你怎么知道 2,311 km 这个数字不是过拟合出来的？

> 三层保护：
> **①** 探针只在 86 个词上训练，在 22 个**从未见过**的留出词上评估；
> **②** `alpha` 用训练集**内部**交叉验证选，测试集不参与任何选择；
> **③** 置换检验：把训练集坐标打乱后再拟合，误差升到 5,985 km（2.6 倍）——
> 说明这个精度确实来自"坐标与隐层的对应关系"，而不是模型的拟合能力。

