# DevAssist AI 面试讲解指南

## 1. 项目定位

这个项目最有价值的讲法不是“我做了一个 AI 聊天机器人”，而是：

> 我实现了一个版本感知、引用可验证、证据不足会拒答的 Python 技术支持系统。默认方案完全离线，用 BM25 和字符 TF-IDF + LSA 双路召回，再通过 RRF、可解释特征重排、版本过滤和证据门控生成结构化诊断。生成器目前是抽取式，所以仓库不依赖 API Key，也不会把没有证据的内容写进答案。

这句话准确覆盖当前代码，没有把 LSA 包装成预训练 embedding，也没有把固定工作流包装成自主智能体。

## 2. 30 秒版本

> DevAssist AI 解决的是 Python/AI 框架报错资料容易混版本、普通 RAG 容易伪造引用的问题。我用 Python 和 FastAPI 做了完整离线链路：JSONL 清洗切块、BM25 + LSA 混合检索、RRF 与特征重排、版本兼容过滤、置信度拒答和引用回查。默认不调用 LLM，确保断网也能复现。目前 21 条资料、26 题只是 smoke benchmark；它验证工程链路，不代表生产准确率。

## 3. 90 秒版本

> 我选择技术支持场景，是因为它同时要求检索准确、版本隔离和可追溯性。例如 PyTorch 2.5 与 2.6 的 `torch.load` 默认行为不同，拿错版本可能直接给出危险建议。系统启动时从 UTF-8 JSONL 构建 BM25 和字符 TF-IDF + LSA 索引；查询先识别项目、版本、异常名和意图，再做双路召回、RRF 融合和固定特征重排。生成前还有硬门控：纯提示注入、无证据、范围外/开放版本、特定错误无主题证据、不兼容证据和低置信度都会拒答。答案只从资料的 summary 和 resolution 抽取，引用 ID、URL 和 evidence 必须回查到本次检索结果。当前提交的 21 条资料和 26 题 smoke 集上所有门槛通过，但 BM25 也已经饱和，所以我不会声称融合显著优于 baseline。下一步应先做来源隔离的真实难例和阈值校准，再接本地 embedding、Cross-Encoder 或 LLM。

## 4. 面试官最可能追问的问题

### 4.1 为什么不用大模型 API？这还算 AI 项目吗？

建议回答：

> 这是有意做出的第一阶段选择。项目的难点是检索、版本约束、拒答和评测，而不是把 API 调通。LSA 是经典潜在语义分析，Agent 是有限的证据编排系统；它们属于 AI/信息检索方法，但不是生成式大模型。我先建立完全离线、可回归的 baseline，再把神经模型作为可选适配器。这样可以量化模型升级到底带来多少收益，也避免没有网络或模型缓存时项目无法运行。

不要回答“我已经实现了 Sentence Transformer”。当前只在 optional dependency 中声明了它，没有适配器代码。

### 4.2 为什么同时用 BM25 和 LSA？

> BM25 对异常名、API、状态码和路径这种精确 token 很强；字符 TF-IDF + LSA 对拼写变化、子串和一定程度的语义/字符相似更宽容。两路错误模式不同。RRF 先利用排名而不是直接比较不同量纲的原始分数，再混入归一化分数，最后用可解释特征重排。但当前 smoke 数据中 BM25 已达到 1.00，所以融合收益尚未被严格证明，需要更难的隔离测试集。

### 4.3 为什么叫 dense？

> `TruncatedSVD` 把 TF-IDF 投影为低维稠密向量，因此实现名是 `LSADenseIndex`。它不是神经 dense retriever。文档中明确叫 LSA，避免让面试官误以为用了预训练 embedding。

### 4.4 为什么选择 RRF？

> BM25 分数和余弦相似度量纲不同，直接线性相加很敏感。RRF 主要依据名次，天然更稳，某一路没有命中时另一条路仍可工作。项目还保留归一化原始分数，因为在只有 21 条资料时纯 RRF 的区分度有限。当前公式是 0.55 RRF、0.25 BM25、0.20 LSA，不应声称这些权重经过学习；它们是启发式配置。

### 4.5 如何处理版本错误？

> 文档有 `project/version/source_type` 元数据。候选先按项目和来源类型过滤，版本有兼容评分；诊断生成前再次检查请求 major 是否超出知识库覆盖，以及是否存在兼容证据。没有兼容证据就返回 `VERSION_MISMATCH`，不会静默用旧版回答。当前解析只是数字元组，不是完整 PEP 440；生产版会换成 `packaging.version` 和显式版本范围。

### 4.6 如何防止幻觉或伪造引用？

> 当前最强的防线是根本不让生成器自由补充事实。诊断来自首条文档 summary，步骤来自前三条 resolution。引用只能从本次 retrieval results 构造；ID、标题、URL 要与索引完全一致，evidence 必须是正文的规范化子串。这个校验解决引用合法性，不等于 claim-level 支持。接 LLM 前我会先建立事实到证据的标注与 validator。

### 4.7 置信度是概率吗？0.36 怎么来的？

> 不是校准概率，而是固定特征的风险分数。系统先要求词法或代码锚点，防止 LSA 的宽泛相似被当成证据，再组合重排、BM25、LSA、版本和 authority。0.36 是当前 smoke 配置，仓库没有保存阈值扫描或独立 calibration/test，因此我不会说它经过严格概率校准。正确下一步是按来源家族切分 calibration/test，扫描阈值并优先限制危险误答率，然后只在冻结 test 上报告一次。

这是一个关键诚实点。能清楚区分 heuristic score 与 calibrated probability，通常比背出一个漂亮百分比更有说服力。

### 4.8 为什么报告全是 1.00？

> 因为数据集只有 21 条策划资料和 26 道高度对齐的 smoke 题，且 BM25 baseline 已经饱和。1.00 证明的是回归链路在这些固定样例上工作，不证明生产泛化。报告本身也写了 `not a production claim`。真正有意义的下一步是继续增加相邻版本难负例、真实脱敏日志、来源隔离和时间外推集。

不要用“模型准确率 100%”回答。

### 4.9 这算 Agent 吗？

> 它是 bounded agent workflow，不是自主规划 Agent。查询分析器根据意图选择允许的来源类型，随后只有一次检索、一次生成和一次验证。工具集合与调用次数有上界，也不执行用户代码。这样更适合高风险技术支持。以后若加入环境采集或包版本检查，也会使用白名单工具和明确预算。

### 4.10 为什么生成器只做抽取？

> 这是把“检索是否正确”和“生成是否幻觉”拆开的工程策略。抽取式 baseline 先验证数据、检索、版本和引用闭环。它的缺点是语言不自然、跨文档综合弱。只有当 claim-level 评测到位后，才值得接 LLM 并单独度量它增加的收益和风险。

### 4.11 数据是从哪里来的？

> 当前是人工策划的中文技术摘要与解决步骤，每条链接到 PyTorch、Hugging Face 或 FastAPI 官方页面。它不是网页快照，也没有自动抓取记录和逐句出处。三条 `issue` 是场景标签，URL 仍是官方文档，不能说收集了真实 GitHub Issue。正式语料要补抓取时间、来源修订、许可证和事实对齐。

### 4.12 摄取管线有什么工程细节？

> JSONL 支持 UTF-8 与 BOM，错误会定位行号；换行统一为 LF，但 fenced code block 内缩进、Tab、空行和尾随空格都保留。缺 ID 时按规范化内容生成 SHA-256 稳定 ID；去重包含版本，避免相同内容的不同版本被误删。代码块是不可分割单元，超过软上限时宁可保留整个代码块。写文件使用同目录临时文件加 replace，兼容 Windows。

### 4.13 reindex 是完全原子的吗？

> 对当前单进程服务请求是原子切换：先用 `store.read()` 解析并验证新快照，在锁外完整构建 Retriever 和 Agent，再在同一 service 锁内依次替换 store、Retriever 和 Agent；search/diagnose 获取引用时也持有该锁，因此不会读到半构建索引。它仍不是带持久化、回滚日志或跨进程一致性的数据库事务；多实例生产版需要版本化 bundle 与协调发布。

### 4.14 并发和扩展性如何？

> FastAPI 把同步 CPU 工作放到线程池；服务用锁保护索引引用和反馈追加。本地限流、指标和反馈锁都是单进程设计，索引搜索也是内存 O(N)。这适合本地演示，不适合多实例生产。扩展时会把 ANN 索引、反馈、限流和指标移到共享服务，并保留离线 adapter 做测试。

当前另有一次 21 文档、200 次调用、20 worker 的预热进程内快照：200/200 成功、P95 70.187 ms。但它绕过 HTTP、网络和 LLM，只能展示 benchmark 脚本与本机回归基线，不能回答 API 容量或生产 SLA。

### 4.15 安全方面做了什么？

> Pydantic 限制字段、长度、环境包数量和 top_k；规则层检测常见提示注入；疑似 OpenAI/GitHub token 和 key/password 会在持久化反馈前脱敏；系统不执行代码、不访问查询中的 URL、不采用检索文档里的指令。admin reindex 默认关闭，配置至少 16 字符 token 后才启用并同样经过限流。本地限流忽略可伪造的 `X-Forwarded-For`，只按 ASGI 对端计数且限制键数量。局限是规则安全不能覆盖所有攻击，多实例仍需要可信网关和共享限流。

## 5. 现场演示建议

### 5.1 启动前

```bash
python -m pip install -e ".[dev]"
devassist evaluate --output reports/evaluation.json
python scripts/benchmark.py --requests 200 --concurrency 20 --output reports/benchmark.json
devassist serve --host 127.0.0.1 --port 8000
```

演示前确认命令在本机真实执行成功。不要在没有运行的情况下承诺启动时间、并发或 P95。

### 5.2 五分钟演示顺序

1. 精确错误：`Expected all tensors to be on the same device`，展示 BM25 原因和引用；
2. 版本对：分别查询 PyTorch 2.5 与 2.6 的 `torch.load weights_only`，展示不同来源；
3. 语义/混合：切换 `bm25`、`dense`、`hybrid_rerank` 查看 component scores；
4. 域外拒答：询问酸面包，展示 `NO_EVIDENCE`；
5. 未来版本：询问 Transformers 99.0，展示 `VERSION_MISMATCH`；
6. 提示注入：索取隐藏提示词，展示 `UNSAFE_QUERY`；
7. 打开 `reports/evaluation.json`，主动解释 smoke 限制；若展示 `reports/benchmark.json`，同步说明它排除 HTTP/网络/LLM。

### 5.3 代码讲解路径

| 顺序 | 文件 | 讲什么 |
|---:|---|---|
| 1 | `domain.py` / `models.py` | 内部不可变对象与 HTTP Schema 分离 |
| 2 | `retrieval/dense.py` | LSA 离线默认与随机种子 |
| 3 | `retrieval/hybrid.py` | 候选过滤、RRF、版本过滤 |
| 4 | `retrieval/reranker.py` | 可解释固定特征，不冒充训练模型 |
| 5 | `agent.py` | hard gate、confidence 和拒答顺序 |
| 6 | `citations.py` | 引用来源回查和证据子串验证 |
| 7 | `evaluation/runner.py` | 消融、指标和自动门槛 |
| 8 | `ingestion/` | UTF-8、稳定 ID、代码块切分和原子写入 |

## 6. 可写入简历的真实表述

### 6.1 AI 应用/后端方向

- 使用 Python、FastAPI 与 Pydantic 构建版本感知的技术支持服务，实现查询分析、混合检索、结构化拒答、反馈与运行指标接口。
- 实现 BM25 与字符 TF-IDF + LSA 双路召回、RRF 融合及六特征可解释重排；默认断网可运行，无模型下载或外部 API 依赖。
- 建立引用回查与证据门控，阻止未检索 ID、伪造 URL 和无法在原文定位的 evidence 进入回答。
- 构建 UTF-8 JSONL 数据管线，支持 Windows 换行、代码围栏保护、SHA-256 稳定 ID、版本安全去重和原子输出。

### 6.2 带指标的谨慎写法

- 在提交的 21 条策划型资料、26 题离线 smoke benchmark 上完成 BM25/LSA/Hybrid/重排消融；Hybrid + rerank Recall@5、MRR@10 与 Answer Evidence Recall 均为 1.00，6 个策划型拒答题全部正确分类；明确该结果仅用于回归，不外推为生产效果。

如果版面不允许保留“策划型 smoke”限定，就不要写 1.00。

### 6.3 当前不能写的内容

- “微调了 Cross-Encoder/Embedding 模型”；
- “使用 Qdrant/Redis/Elasticsearch 部署”；
- “接入 GPT/Qwen 实现多轮 Agent”；
- “在 10,000 条真实文档上达到生产级性能”；
- “引用支持率 100%”；当前测的是引用合法性，不是 claim 支持率；
- “显著优于 BM25”；当前 BM25 同样饱和；
- “模型准确率 100%”。

## 7. STAR 讲法

### Situation

Python/AI 框架排错资料经常跨版本，普通聊天式 RAG 容易把相似但不兼容的答案混在一起，而且引用可能无法核验。

### Task

做一个无需外部服务也能复现的端到端 baseline，要求检索可解释、版本不静默回退、答案可追溯、无证据时拒答。

### Action

先建立 typed domain 和 JSONL 数据管线，再实现 BM25/LSA 双路召回、RRF、特征重排与版本 gate；把 Agent 限定为一次检索和一次生成；采用抽取式生成隔离幻觉；最后实现四模式消融、拒答和引用合法性评测。

### Result

仓库可以在离线环境运行完整检索和诊断流程；当前 21/26 smoke 数据的自动门槛全部通过，交付前 Windows 11 / Python 3.14.6 本地自动化测试为 205 passed、总覆盖率 90.54%。更重要的是，报告直接暴露 BM25 饱和和样本规模限制，为下一轮难负例、独立校准和神经模型对比建立了基线；本地测试结果不冒充尚未在线执行的 Python 3.12/3.13 CI。

## 8. 设计题：如果再给两周

建议按以下优先级回答：

1. 先扩数据，不先换模型：从真实脱敏问题构建按 URL/Issue/时间隔离的 calibration/test；
2. 增加相邻版本、高词汇重合负例和冲突证据；
3. 保存每题特征，画 risk-coverage 与拒答 PR 曲线，重新冻结阈值；
4. 实现统一 `DenseRetriever`/`Reranker` Protocol；
5. 增加本地 Sentence Transformer 与 Cross-Encoder adapter，并与 LSA 同口径消融；
6. 加 claim-evidence 标注与支持率，再接可选 LLM；
7. 增加依赖锁，保留 Python 3.12/3.13 CI，并把当前 21 文档进程内快照升级为 HTTP、冷启动与 10k chunk benchmark；
8. 最后再考虑向量数据库和分布式组件。

这个顺序体现一个重要判断：当前主要瓶颈是评测可信度，不是模型不够大。

## 9. 面试红线

- 不把 LSA 说成 Transformer embedding；
- 不把抽取式生成说成大模型生成；
- 不把 bounded workflow 说成自主多 Agent；
- 不把 `authority` 人工先验说成权威性模型预测；
- 不把 smoke 结果外推到生产；
- 不隐藏 BM25 已达到同样上限；
- 不声称 0.36 是概率校准阈值；
- 不把官方 URL 当作自动证明所有摘要仍然最新；
- 不展示无法由仓库或报告复核的文档数、并发、延迟或成本数字。

主动说清这些边界不会削弱项目，反而能证明你理解系统评估、数据泄漏和工程可信度。
