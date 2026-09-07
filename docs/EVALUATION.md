# DevAssist AI 离线评测说明

## 1. 评测目标

当前评测回答五个有限问题：

1. 在提交的 21 条知识记录中，检索器能否找回人工指定的 relevant ID；
2. 不同检索模式在同一批 20 个可回答问题上的排序有何差异；
3. Agent 能否对 6 个策划型不可回答问题拒答并返回预期 reason code；
4. 返回的引用是否确实来自同一次检索上下文，且元数据和证据片段可回查；
5. 20 个可回答固定题的最终答案引用是否至少覆盖一个人工 relevant ID。

它不评测开放域事实正确率、真实线上满意度、LLM 生成质量、claim-level 蕴含、生产延迟或大规模索引性能。

## 2. 可复现默认设置

默认评测不访问网络，也不下载模型：

- Python 要求：`>=3.12`；
- 知识库：`data/sample/knowledge_base.jsonl`；
- 问题集：`data/evaluation/golden_queries.jsonl`；
- 文档数：21；
- 问题数：26，其中 20 可回答、6 不可回答；
- 默认候选数：20；
- 默认回答 `top_k`：5；
- 置信度阈值：0.36；
- LSA：字符 TF-IDF 2 至 5 gram，`TruncatedSVD(random_state=42)`；
- Reranker：默认加载 `artifacts/reranker/weights.json`；
- 最终模式：`hybrid_rerank`；
- 生成：未配置 LLM 环境变量时使用确定性抽取式生成器。

LSA 是本项目的可复现默认 dense backend。它产生低维稠密向量，但不是 Sentence Transformer，也没有使用预训练语义模型。这个名称和边界在报告、简历及面试中不能模糊。

从仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python scripts/train_reranker.py --blend 0.30
python scripts/evaluate.py --fail-on-threshold --output reports/evaluation.json
```

训练命令会重写权重 artifact 的生成时间，评测命令会重写报告生成时间。比较产物时应忽略 `generated_at` 和可能受机器影响的 `elapsed_seconds`，重点检查权重、指标、失败题与配置是否变化。

为了复现提交报告，必须确保 `DEVASSIST_LLM_BASE_URL` 或 `DEVASSIST_LLM_MODEL` 未配置。若两者同时存在，Service 会调用外部/本地 OpenAI-compatible endpoint；这种运行不再是纯离线确定性评测，也没有独立的 LLM 质量指标。Docker Compose 默认未设置这些变量，仍使用抽取式生成。

当前仓库没有锁文件，NumPy/scikit-learn 只使用版本范围。因此随机种子固定并不保证跨所有未来依赖版本逐位相同；正式发布应增加锁文件并记录 Python、OS、CPU 与依赖版本。

## 3. Reranker 训练 artifact 的评测边界

当前 `weights.json` 来自 40 组 pair：20 正例、20 策划型难负例。训练器对六个特征拟合 `LogisticRegression`，使用平衡类别、`liblinear` 与 `random_state=42`；学习权重只占最终结果的 30%，安全默认值占 70%。Service 和 evaluation runner 都显式加载这个 artifact，缺失时才回退代码默认权重。

artifact 中的 `training_accuracy=1.000` 是在训练输入自身上计算的，不是 validation accuracy。40 pair 与 golden queries 共用 21 条文档和同一组问题主题，也没有按 query family 切分。因此当前报告不能回答“训练后的权重是否优于默认权重”。要回答该问题，至少需要在冻结 held-out 集上同时评测：

1. 代码默认权重；
2. 100% 学习权重；
3. 当前 30% learned / 70% default blend；
4. 多个 blend 比例及其置信区间。

当前 blend 的安全意义是限制小样本系数的影响，不是消除过拟合。

## 4. 实际指标定义

以下定义以 `src/devassist/evaluation/metrics.py` 和 `runner.py` 的当前实现为准。

### 4.1 Recall@5

对每个可回答问题计算“前 5 条中去重后的 relevant ID 数 / 该题全部 relevant ID 数”，再对问题取均值。多证据问题只找回一部分时会得到部分分数，而不是二元命中率。

### 4.2 MRR@10

取前 10 条中第一个 relevant ID 的倒数排名，对问题取均值。未命中记为 0。

### 4.3 nDCG@10

当前 relevant ID 只有二元相关性，DCG 增益统一为 1。它没有实现 `0/1/2` 分级标注；不能把当前 nDCG 描述成 graded relevance 评测。

### 4.4 Version Accuracy

对带版本的可回答问题，要求结果非空，且前 5 条返回结果全部通过当前 `version_is_compatible` 规则；空结果记为 0。语料中标记为 `all` 的版本无关文档在检索层可兼容数字版本，但 Agent 会拒绝用户直接提交的 `latest/any/all` 开放版本标识。该指标验证过滤行为，不验证答案中的每个版本事实。

### 4.5 Refusal Precision / Recall / F1

把 `answerable=false` 视为“应拒答”正类：

- TP：应拒答且实际拒答；
- FP：可回答但实际拒答；
- FN：应拒答但实际回答。

### 4.6 Citation Validity

只对已回答问题计算。引用必须来自当前检索结果，且 ID、标题、URL 一致；压缩后的 evidence 必须是文档正文的子串。这个指标阻止伪造和跨请求引用，但不证明诊断中的每个事实都被引用支持。

### 4.7 Answer Evidence Recall

分母是 20 个 `answerable=true` 的固定问题；若最终回答的引用 ID 与该题 `relevant_ids` 至少有一个交集，则该题记为 1。它比 Citation Validity 多验证一步“引用是否命中人工指定相关资料”，但仍不检查诊断中每个 claim 是否被证据蕴含，也不是开放输入上的答案召回率。

### 4.8 Expected Reason Accuracy

只在提供 `expected_reason` 的题目上计算。当前主要是 6 个不可回答问题；若集合没有此类题，指标安全返回 0 而不会出现除零错误。

## 5. 当前提交报告

`reports/evaluation.json` 的元数据显示：

- 生成时间：`2026-09-06T13:31:28.316179+00:00`；
- 作用域：`curated offline smoke benchmark; not a production claim`；
- 21 条知识记录、26 个问题；
- 加载权重：`artifacts/reranker/weights.json`；
- 本次记录的总耗时：0.308 秒。

耗时没有附带硬件、OS、Python 或依赖版本，且数据极小，因此只能作为一次运行记录，不能写成性能 SLA。

### 5.1 检索消融结果

| 模式 | 可回答题 | Recall@5 | MRR@10 | nDCG@10 | Version Accuracy |
|---|---:|---:|---:|---:|---:|
| BM25 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LSA | 20 | 1.0000 | 0.9750 | 0.9754 | 1.0000 |
| Hybrid | 20 | 1.0000 | 1.0000 | 0.9960 | 1.0000 |
| Hybrid + feature rerank | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

这组结果显示，在当前高度策划的小数据上，BM25 已达到上限；因此不能从该表声称 Hybrid 或 learned-blend reranker 显著优于 BM25。LSA 的 MRR/nDCG 略低，Hybrid 恢复了首位排序，但需要更难、更大的隔离测试集才能判断融合和训练权重是否真正带来泛化收益。

### 5.2 Agent 结果

| 指标 | 当前值 |
|---|---:|
| 评测题数 | 26 |
| 已回答 | 20 |
| 已拒答 | 6 |
| Refusal Precision | 1.0000 |
| Refusal Recall | 1.0000 |
| Refusal F1 | 1.0000 |
| Citation Validity | 1.0000 |
| Answer Evidence Recall | 1.0000 |
| Expected Reason Accuracy | 1.0000 |

报告的 `failures` 当前为空。这只表示这 26 题没有记录到最终分类失败，不表示系统在开放输入上没有失败模式。

### 5.3 独立的进程内性能快照

`reports/benchmark.json` 记录了一次 Windows 11 / Python 3.14.6 开发机运行：21 条文档、200 次诊断调用、20 worker，200 次成功；吞吐 466.080 req/s，mean 26.839 ms、P50 17.762 ms、P95 70.187 ms、P99 137.784 ms，总墙钟时间 0.429 s。

该脚本先预热，再直接并发调用同一进程中的 `DevAssistService.diagnose`。它明确排除 HTTP 序列化与服务器调度、网络、外部 LLM、冷启动和大规模索引成本，也没有记录 CPU/内存。因此这些值只能作为开发机回归快照，不能作为 API 性能、生产吞吐、SLA 或容量规划结论。

## 6. 当前可执行门槛

`scripts/evaluate.py --fail-on-threshold` 实际执行以下门槛：

| 检查 | 当前门槛 |
|---|---:|
| Hybrid + rerank Recall@5 | `>= 0.80` |
| Hybrid + rerank MRR@10 | `>= 0.75` |
| Version Accuracy | `== 1.00` |
| Citation Validity | `== 1.00` |
| Answer Evidence Recall | `>= 0.90` |
| Refusal F1 | `>= 0.80` |

当前报告六项均通过。

`docs/TEST_PLAN.md` 中还列有更严格的拟议门槛和更广的测试矩阵；其中未被 `runner.py` 实现的项目属于目标，不是本次报告已经验证的事实。判断自动化发布是否通过，应以命令实际执行的检查为准，并在后续让计划与代码同步。

## 7. 失败与置信度校准

### 7.1 当前代码中可验证的失败分流

系统不是只比较一个总分阈值，而是先处理高风险失败类型：

| 失败类型 | 当前门控 | 对应 smoke 场景 |
|---|---|---|
| 纯提示注入 | 有安全 warning 且无技术 marker | 索取隐藏提示词 |
| 无相关资料 | 无候选或缺少有效证据锚点 | 酸面包、Django/错误项目 |
| 范围外或开放版本 | 超出语料已验证的 major/minor 范围，使用 `latest/any/all`，或无兼容结果 | PyTorch 2.99、Transformers 99.0 |
| 特定错误无主题证据 | 查询含 OOM 等鲜明错误，但同项目候选未覆盖该错误 | Transformers Trainer OOM hard negative |
| 表面语义相似 | LSA 命中但词法/代码 anchor 不足 | 域外近似词风险 |
| 分数不足 | confidence `< 0.36` | 返回 `LOW_CONFIDENCE` |
| 引用失败 | 无步骤或引用无法回查 | fail closed 为 `NO_EVIDENCE` |

证据门控要求：

```text
raw_signal >= 0.035
AND (body_overlap >= 0.035 OR code_signal > 0)
AND (bm25_raw > 0.15 OR lsa_cosine > 0.045 OR code_signal > 0)
```

通过证据门后，置信度为：

```text
0.40 * rerank_score
+ 0.22 * clipped_bm25
+ 0.14 * lsa_cosine
+ 0.09 * body_overlap
+ 0.07 * code_signal
+ 0.05 * version_compatibility
+ 0.03 * authority
```

结果裁剪到 `[0, 0.99]`，默认回答阈值为 0.36。拒答响应的公开 confidence 被设为 0，因此当前报告不能直接重建所有样例的阈值 ROC 曲线。

### 7.2 对校准历史的诚实说明

仓库没有保存置信度阈值网格搜索、阈值修改前后的失败题、开发集与独立测试集划分，也没有记录 0.36 是如何从候选值中选出的。因此目前只能把这些数值称为“针对 smoke 场景配置的启发式门控”，不能宣称它们经过严格概率校准。

新加入的 Logistic Regression 训练只学习 reranker 的六个特征权重，不学习 Agent 的 0.36 回答阈值，也不把 confidence 校准为概率。`training_accuracy=1.000` 不能替代 refusal calibration。

当前 `failures=[]` 是最终报告状态，不是失败迭代历史。面试时应主动说明这一点；编造“把幻觉率从 X% 降到 Y%”会与仓库证据冲突。

### 7.3 下一轮正确校准流程

1. 先按 URL 家族、Issue、版本对和时间切分 calibration/test，禁止随机切相似问句；
2. 增加高词汇重合的不可回答题、相邻版本冲突和不完整错误日志；
3. 对每题保存未裁剪特征、门控阶段、预测状态与人工标签；
4. 在 calibration 集扫描 evidence gate 和 confidence threshold；
5. 先约束危险误答率，再优化 Refusal F1，而不是追求总体准确率；
6. 固定阈值后只运行一次隔离 test；
7. 发布 PR 曲线、混淆矩阵、按标签切片结果和全部失败题 ID；
8. 语料或权重变化时保留新旧报告，不能静默删除失败题。

建议至少分别报告：可回答题误拒率、不可回答题误答率、版本冲突误答率、安全题误答率，以及阈值附近样例。

## 8. 主要评测威胁

- 数据规模只有 21/26，置信区间很宽；
- 查询和文档由同一目标策划，存在词汇对齐和测试集调参风险；
- 40 个训练 pair 与 golden queries 共享底层文档和主题，没有 held-out 对照；
- 没有来源家族隔离，不能排除近重复或事实泄漏；
- BM25 已在当前数据饱和，无法测出混合检索的增益；
- nDCG 使用二元相关性，没有分级标注；
- Citation Validity 不是 Citation Support；
- 抽取式生成直接读取人工 `summary/resolution`，降低了生成难度；
- 可选 LLM 路径没有进入单独报告，且 citation validity 不会发现其 diagnosis/steps 中未被证据支持的新事实；
- 六个拒答题仍然很少；虽已加入高词汇重合的 Transformers OOM hard negative，仍需更多相邻版本和跨主题难负例；
- 只有 21 文档的预热进程内并发快照，没有 HTTP、10k chunk、内存或长时间稳定性报告；
- 没有人工盲评、标注一致性或真实用户反馈结果；
- 没有依赖锁和运行环境 manifest。

## 9. 合法的结果表述

推荐写法：

> 使用 40 个 pair 训练并与 70% 安全默认值混合的六特征 reranker，在仓库提交的 21 条策划型资料和 26 题离线 smoke benchmark 上完成四模式消融；Hybrid + feature rerank 的 Recall@5/MRR@10 为 1.00，6 个策划型不可回答题全部拒答，Citation Validity 与 Answer Evidence Recall 均为 1.00。训练与评测共享主题，结果只用于管线回归，不代表 held-out 或生产准确率。

不推荐写法：

> 模型准确率 100%，显著优于 BM25，达到生产级效果。

后者同时混淆了模型与规则系统、忽略 BM25 同样达到上限、遗漏样本规模，也把 smoke evaluation 错误外推到真实流量。

## 10. 评测变更审查清单

每次改变语料、分词、权重、阈值或版本规则时，应检查：

- 数据文件数量和 SHA-256 是否记录；
- 所有四个检索模式是否同口径运行；
- 旧报告是否保留或可从版本控制恢复；
- 是否新增或删除问题，原因是什么；
- 是否出现 `failures`；
- 指标改善是否只来自测试题定向修改；
- 拒答改善是否以大量误拒为代价；
- 引用是否只是合法，还是事实也被支持；
- 是否仍能断网运行；
- 是否误把训练内 1.000 写成 held-out accuracy，或误把 30% blend 写成完整模型训练；
- 评测时是否意外设置 LLM 环境变量而改变生成路径；
- 报告中的每个数字能否由脚本重新计算。
