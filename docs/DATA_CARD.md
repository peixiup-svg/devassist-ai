# DevAssist AI 数据卡

## 1. 数据集身份

本仓库包含三个用途不同的 JSONL 数据文件，以及一个由训练脚本生成的权重 artifact：

| 文件 | 当前规模 | 用途 |
|---|---:|---|
| `data/sample/knowledge_base.jsonl` | 21 条知识记录 | 本地功能演示、检索索引与 smoke evaluation |
| `data/evaluation/golden_queries.jsonl` | 26 个问题 | 固定的离线 smoke benchmark |
| `data/training/reranker_pairs.jsonl` | 40 个 query-document pair | 训练六特征 Logistic Regression reranker 权重 |
| `artifacts/reranker/weights.json` | 6 个权重及 metadata | Service、评测和 Docker 默认加载的可复现 artifact |

状态：策划型样例数据，不是生产数据，不是公开排行榜数据集，也不是统计上有代表性的真实用户分布。

报告 `reports/evaluation.json` 明确标记其范围为 `curated offline smoke benchmark; not a production claim`。简历、README、演示和面试中必须保留这个限定语。

当前数据没有独立的语义版本号或数据校验和清单。可复现性依赖仓库中这些具体文件；后续正式版本应增加 manifest、文件 SHA-256、构建时间和来源快照时间。

## 2. 知识库组成

### 2.1 按项目

| 项目 | 记录数 | 占比 |
|---|---:|---:|
| PyTorch | 10 | 47.6% |
| Transformers | 6 | 28.6% |
| FastAPI | 5 | 23.8% |
| 合计 | 21 | 100% |

### 2.2 按来源类型标签

| `source_type` | 记录数 |
|---|---:|
| `documentation` | 14 |
| `issue` | 3 |
| `migration` | 3 |
| `changelog` | 1 |

`source_type=issue` 是当前样例中的场景类型标签。这三条记录的 URL 仍指向官方文档页面，而不是可审计的 GitHub Issue 快照，因此不能宣称仓库收录了三个真实 Issue 对话。

### 2.3 按版本覆盖

| 项目与版本 | 记录数 |
|---|---:|
| PyTorch `all` | 8 |
| PyTorch `2.5` | 1 |
| PyTorch `2.6` | 1 |
| Transformers `all` | 4 |
| Transformers `4.30` | 1 |
| Transformers `4.46` | 1 |
| FastAPI `all` | 4 |
| FastAPI `0.100` | 1 |

16 条记录使用 `all`，只有 5 条是显式版本记录。因此当前数据主要测试通用问题，只包含少量版本冲突对：PyTorch `torch.load` 2.5/2.6、Transformers `TrainingArguments` 4.30/4.46，以及 FastAPI/Pydantic v2 迁移。

知识正文长度约为 320 至 464 个字符，均值约 379 个字符。这个规模不代表真实文档切块长度分布。

## 3. 知识记录 Schema

当前 21 条记录都提供以下字段：

| 字段 | 类型 | 含义与注意事项 |
|---|---|---|
| `id` | string | 人工可读稳定 ID；运行时引用主键 |
| `project` | string | `pytorch`、`transformers` 或 `fastapi` |
| `version` | string | 显式版本或 `all`；不是完整版本范围 |
| `source_type` | string | documentation / issue / migration / changelog |
| `title` | string | 人工整理标题 |
| `content` | string | 策划后的技术说明，不是原网页快照 |
| `summary` | string | 抽取式生成器直接使用的诊断摘要 |
| `resolution` | array[string] | 抽取式生成器直接使用的建议步骤 |
| `url` | string | 官方参考页面 URL |
| `authority` | number | `[0,1]` 的人工权威度先验 |
| `tags` | array[string] | 检索补充标签 |

领域模型还支持可选的 `published_at`，但当前样例记录都没有该字段。因而不能仅凭仓库判断资料抓取时间、发布日期或某个网页在何时被核验。

## 4. 来源与创建方式

所有记录链接到 PyTorch、Hugging Face Transformers 或 FastAPI 的公开官方页面。仓库保存的是围绕这些页面人工策划的摘要、问题背景和解决步骤，而非完整网页抓取内容。

当前仓库没有提供：

- 自动抓取脚本执行记录；
- 网页快照、ETag 或内容校验和；
- 每条摘要的撰写者、复核者与复核日期；
- 每条来源的许可证映射；
- 事实与原网页段落的逐句对齐标注。

因此应把 URL 视作“进一步核验入口”，而不是证明摘要在所有时间点都与页面一致。项目自身的 MIT 许可证不会自动改变上游网页内容的许可证；若要发布大规模派生语料，必须单独核查来源条款和再分发权限。

## 5. 评测集组成

26 个问题中：

- 20 个标记为 `answerable=true`；
- 6 个标记为不可回答；
- 问题长度约 14 至 98 个字符，均值约 48.1 个字符；
- 问题以中文为主，混有英文 API、异常信息和参数名。

评测标签可多选，主要覆盖：

- 5 个 `exact-error`；
- 5 个 `version-sensitive`；
- 2 个 `exact-api`；
- 1 个 `multi-evidence`；
- Windows、中英文混合、部署、性能、浏览器、架构和安全等单例；
- 6 个不可回答样例覆盖知识域外、错误项目、未知未来版本、无证据问题、提示注入，以及高词汇重合但主题无证据的 Transformers OOM hard negative。

不可回答题具体用于验证：

- FastAPI 项目过滤下不回答 Django ORM；
- 不回答酸面包制作；
- 不服从索取隐藏提示词的指令；
- PyTorch 项目范围下不回答 TensorFlow/TensorRT；
- 不用 Transformers 4.x 资料猜测 99.0 的 API。

## 6. Reranker 训练 pair 与 artifact

40 条训练记录实际组成 20 个 query 组；每组包含一个正例和一个策划型难负例，因此标签严格平衡为 20/20。字段为：

| 字段 | 类型 | 含义 |
|---|---|---|
| `query` | string | 训练用技术查询 |
| `document_id` | string | 指向同一 21 条知识库中的记录 |
| `label` | 0 / 1 | 相关或不相关标签 |
| `requested_version` | string | 构造 version feature 的请求版本 |

这些 pair 覆盖同一批 PyTorch、Transformers 和 FastAPI 主题，包括相邻版本互为负例。它们是人工策划 pair，不是从生产点击日志、真实反馈或大规模候选中自动挖掘出的 hard negatives。“难负例”只描述配对意图，不代表经过统一难度量化。

训练脚本以六个已有特征拟合平衡 Logistic Regression；将负系数裁为零并归一化，再按 30% 学习值、70% 代码默认值做安全混合。artifact metadata 为：

- 训练 pair：40；
- 正例：20；
- 难负例：20；
- training accuracy：1.000；
- learned blend：0.30；
- 生成时间：`2026-09-04T17:18:38.974784+00:00`。

1.000 是训练集自身准确率，没有 validation/test 切分，也没有置信区间。它只能用作训练管线 smoke check。pair 与 26 题评测共享知识文档、技术主题和大量关键词，不能把评测表现解释成独立的 learned-reranker 泛化证明。

## 7. 预处理与稳定性

`devassist.ingestion` 提供了独立的离线管线：

- UTF-8/UTF-8 BOM JSONL 读取；
- Windows CRLF 到 LF 规范化；
- fenced code block 内容保护；
- 必填字段校验；
- 未提供 ID 时使用 SHA-256 派生稳定 ID；
- 以项目、版本、来源类型、标题、正文计算语义去重指纹；
- Markdown 感知切块；
- 同目录临时文件加原子替换写出。

提交的 21 条样例是运行时直接加载的知识库，不应假定每次启动都会自动重新执行 ingestion。若替换语料，应显式执行 ingest，再运行完整评测。

## 8. 适合用途

这些数据适合：

- 验证服务能在无网络环境启动；
- 验证 BM25、LSA、融合和重排接口；
- 演示精确错误码、少量版本冲突和安全拒答；
- 做单元、集成与端到端 smoke test；
- 验证 Logistic Regression 特征训练、安全 blend、artifact 加载和缺失时默认回退；
- 在面试中解释检索、引用和拒答机制；
- 检查同一提交在不同机器上是否得到一致趋势。

## 9. 不适合用途

这些数据不适合：

- 训练或微调通用 embedding、Cross-Encoder 或 LLM；40 pair 仅支持当前六特征线性 reranker 的管线演示；
- 声称真实用户准确率、线上转化率或客服效率；
- 比较模型在开放域问答中的能力；
- 推断所有 PyTorch、Transformers 或 FastAPI 版本的兼容性；
- 评估代码安全、可执行修复或生产事故处置；
- 证明跨语言检索能力；当前英文资料/中文提问覆盖过少；
- 用作生产 RAG 的完整知识库；
- 将 100% smoke 指标写成无上下文的“模型准确率 100%”。

## 10. 偏差与已知局限

### 10.1 选择偏差

问题和资料由同一项目目标共同策划，查询中经常直接出现文档 ID 对应的异常词、API 或版本。这种高度对齐会显著抬高 BM25 和重排结果。

### 10.2 规模偏差

21 条短资料使候选空间极小。`reports/benchmark.json` 虽记录了 200 请求、20 worker 的预热进程内快照，但它排除 HTTP、网络和 LLM，也没有 10,000 个相似 chunk。当前高指标和性能快照都不能预测大规模下的召回、排序、API 延迟或内存表现。

### 10.3 语言偏差

问题主要为中文夹英文术语，资料也是中文策划摘要。它不构成真正的“中文查询到英文原文”评测，也不能代表其他语言。

### 10.4 版本偏差

大部分资料使用 `all`，版本特异样例太少；没有系统覆盖 patch、release candidate、CUDA build tag、依赖组合或弃用窗口。

### 10.5 负例偏差

6 个不可回答题仍然很少；虽然已加入一个 Transformers OOM 高词汇重合 hard negative，仍缺少结论相反、只有相邻版本可用、多个官方来源冲突、错误日志缺失关键一行等更危险场景。

### 10.6 标注偏差

每题只有 relevant ID 集合，没有分级相关性、事实级 `must_include`、禁止事实、标注人一致性或仲裁记录。实际 nDCG 使用二元相关性。

### 10.7 训练与评测耦合

训练 pair 和 golden queries 不是按 query/source family 隔离构建的，两者共享所有主题和底层文档。artifact 也没有保存 classifier intercept、原始系数、特征矩阵摘要或 held-out 指标。当前权重更接近“受数据提示的安全配置”，不是经独立验证的排序模型。

## 11. 隐私与安全

提交的数据看起来是技术样例，不包含真实用户姓名、邮箱、私有仓库、访问令牌或完整业务日志。但仓库没有自动化 PII 扫描报告，不能把“未观察到”表述为形式化证明。

未来引入真实日志时必须：

1. 在进入标注和存储前脱敏密钥、路径、用户名、主机名与业务数据；
2. 记录合法用途、保留期限和删除机制；
3. 按 URL/事件家族切分，防止同一事故泄漏到训练和测试；
4. 禁止存储可直接执行的凭据与恶意载荷；
5. 对人工标注工具和导出文件实行最小权限。

默认抽取式生成不会把查询或知识片段发送到外部服务。启用 OpenAI-compatible generator 后，原始问题和最多三条检索证据会发送到配置的 endpoint；这会改变数据处理边界。生产启用前必须完成供应商、数据驻留、日志保留和密钥管理审查，并确认脱敏发生在出站请求之前，而不仅是写反馈时。

## 12. 扩充到正式数据集的最低要求

在任何生产指标发布前，至少应完成：

- 100 至 300 个经过复核的问题作为首个冻结测试集；
- calibration 与 test 按来源家族、Issue 和时间隔离；
- reranker pair 按 query family 分组切分 train/validation/test，禁止一正一负被拆到不同集合；
- 每个主要版本至少有正例、相邻版本冲突例和不可回答例；
- 两名标注者独立给出相关等级和事实支持，记录一致性；
- 每条知识记录增加 `retrieved_at`、`source_revision`、许可证、校验和与版本范围；
- 建立近重复检测，禁止问句改写跨集合；
- 保存失败题而不是为提高分数删除；
- 记录真实分布与策划分布的差异；
- 重新计算所有报告，并在报告中显示数据 manifest。

在这些条件达成前，最准确的称呼始终是“可复现的离线 smoke 数据”。
