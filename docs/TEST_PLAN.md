# DevAssist AI 离线测试与验收计划

本文是项目的可执行质量门槛。目标不是证明接口“能返回内容”，而是证明系统在无网络、无外部模型服务的环境中，仍能稳定完成版本过滤、混合检索、引用校验与安全拒答。

## 1. 验收范围

测试对象包括：

- 文档读取、清洗、切块、去重和来源追踪；
- BM25 关键词召回、Dense 语义召回、RRF 融合排序；
- 软件版本解析、严格过滤和可解释的回退策略；
- 回答生成边界、引用合法性校验、低置信度拒答；
- FastAPI 请求校验、错误映射、健康检查和并发隔离；
- 离线评测数据、指标计算、性能基线和结果复现。

不纳入默认 CI 的项目：外部 LLM API、在线 Embedding 模型、实时抓取和依赖 GPU 的测试。这些能力只能作为显式启用的人工验收项，不能影响离线 CI。

## 2. 一键执行

从仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov=devassist --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=80
python scripts/evaluate.py --fail-on-threshold --output reports/evaluation.json
```

验收要求：上述命令在 Python 3.12 和 3.13 上全部退出为 `0`。测试运行期间断开网络后结果必须相同。

当前交付前本地快照为 Windows 11 / Python 3.14.6 下 **205 passed、总覆盖率 90.54%**，另有 1 条 FastAPI/Starlette TestClient 依赖链弃用 warning。它证明当前机器上的测试通过，不替代 CI 对 Python 3.12/3.13 的矩阵验收；用例数量变化后应以最新报告为准。

## 3. 离线与确定性约束

- [ ] `tests/fixtures/` 中提交最小文档库、固定向量和评测问题，测试运行时不下载任何资源。
- [ ] 单元与集成测试通过依赖注入使用 `FakeEmbedder`、`FakeGenerator`、固定时钟和固定 ID。
- [ ] 测试会话禁止 socket 外连；任何 DNS、HTTP 或模型下载尝试立即使测试失败。
- [ ] 每个测试使用 `tmp_path` 创建索引和缓存，不读取开发者机器的模型缓存或用户配置。
- [ ] 固定随机种子；同分结果使用 `chunk_id` 作为最终排序键。
- [ ] 相同语料和配置连续运行两次，索引标识、搜索顺序、引用顺序和指标完全一致。
- [ ] 未安装 `.[ai]` 时，核心离线测试与 smoke evaluation 仍可运行。

建议 fixture 至少包含：PyTorch 2.0/2.1 的同一主题不同答案、一个精确异常码、一个英文文档对应中文问题、一组互补召回样例、一组冲突证据和三个无答案问题。

## 4. 单元测试清单

### 4.1 数据摄取与切块

- [ ] Markdown 标题、段落、列表、表格和 fenced code block 被正确保留。
- [ ] 代码块不会从中间被切断；父标题会进入子块上下文。
- [ ] Windows/Unix 换行、中英混排、emoji 和超长单词不会造成异常。
- [ ] 空文件和仅空白文件被跳过，并产生可观察的统计信息。
- [ ] `chunk_size` 在 `1`、恰好等于正文长度、正文长度减一时行为正确。
- [ ] `chunk_size <= 0`、`overlap < 0` 或 `overlap >= chunk_size` 被拒绝。
- [ ] 缺少 `title`、`url`、`version`、`source_type` 或正文时给出稳定的验证错误。
- [ ] 完全重复的来源被去重；相同内容的不同版本不会被错误合并。
- [ ] 每个 chunk 均包含 `chunk_id/source_id/title/url/version/source_type/content`。
- [ ] chunk ID 在重复构建和输入顺序变化后保持稳定。
- [ ] 任一结果都能追溯到唯一的原始来源，且不能由模型改写来源 URL。

### 4.2 版本解析和过滤

- [ ] 支持 `2`、`2.0`、`2.0.1`、`v2.1.0`、`2.1.0rc1`、`2.1+cu121`。
- [ ] 未指定版本、`latest`、非法版本和缺失版本分别得到定义明确的结果。
- [ ] 精确版本查询只返回兼容结果，不混入相邻 minor/major 版本。
- [ ] 预发布版与正式版不会被视为完全相同。
- [ ] 请求版本无匹配时，严格模式返回空结果并触发 `VERSION_MISMATCH`。
- [ ] 如启用版本回退，响应必须同时给出 `requested_version`、`resolved_version` 和 warning；禁止静默跨版本。
- [ ] 版本过滤在最终返回前再次执行，融合步骤不能重新引入错误版本。

### 4.3 BM25 检索

- [ ] 精确异常码和 API 标识符（如 `ModuleNotFoundError`、`torch.nn`）可被检索。
- [ ] 大小写、下划线、点号、斜杠和中英文查询的分词策略有固定测试。
- [ ] 精确错误码文档排名高于只含普通语义词的文档。
- [ ] 空查询、停用词查询、未知词、空语料返回空结果而不是异常。
- [ ] `top_k=1`、`top_k` 大于语料数正确；`top_k<=0` 被拒绝。
- [ ] 重复 chunk 不会产生重复结果；分数和排序可重复。

### 4.4 Dense 检索

- [ ] query 与 document 使用各自正确的编码入口。
- [ ] 固定向量下的 cosine similarity 与手算结果一致。
- [ ] 空批次、单条、零向量、维度不匹配、`NaN` 和 `Inf` 均有测试。
- [ ] 向量归一化不会修改调用者传入的原始数组。
- [ ] 未配置本地模型时返回受控配置错误，不能尝试联网下载。
- [ ] `top_k`、去重和同分稳定排序规则与 BM25 一致。

### 4.5 RRF 混合检索

- [ ] 用手算金样例验证 RRF 公式、常数和权重。
- [ ] 覆盖“仅 BM25 命中”“仅 Dense 命中”“两路同时命中”三种情况。
- [ ] 同一 chunk 在两路出现时只返回一次，并正确累加融合分数。
- [ ] 某一路为空时系统仍可返回另一路结果。
- [ ] 原始分数量纲改变不会影响基于排名的 RRF 结果。
- [ ] 权重为零行为明确；负权重、非法常数和非法 `top_k` 被拒绝。
- [ ] 融合后的 `top_k` 截断、同分 tie-break 和输入顺序不变性经过测试。

### 4.6 引用和回答约束

- [ ] 回答中的每个 `citation_id` 都存在于本次实际传给生成器的上下文。
- [ ] 引用返回 `chunk_id/title/url/version/source_type`，且与索引元数据完全一致。
- [ ] 重复引用被去重，引用顺序稳定。
- [ ] 不存在、越界、跨请求或 URL 不匹配的引用被拒绝。
- [ ] 模型生成的新 URL、伪造 chunk ID 和未检索来源永远不会进入响应。
- [ ] 对证据不支持的 claim 执行拒答或删除，不能用相似关键词冒充支持。
- [ ] 无引用时不得输出确定性的技术修复结论。
- [ ] HTML、Markdown、控制字符和 prompt injection 文本不能修改系统规则或产生 XSS。

### 4.7 置信度和拒答

- [ ] 空检索触发 `NO_EVIDENCE`。
- [ ] 最高置信度低于阈值触发 `LOW_CONFIDENCE`。
- [ ] 请求版本无证据触发 `VERSION_MISMATCH`。
- [ ] 非法或空问题触发 `INVALID_QUERY`。
- [ ] 证据互相冲突时拒答，且响应明确要求用户补充信息。
- [ ] 只有相似关键词但事实不支持时拒答。
- [ ] 强精确证据存在时不会误拒。
- [ ] 分数恰好等于阈值时的包含/排除规则固定。
- [ ] 拒答响应不包含编造步骤，不附带伪造引用，并返回 machine-readable `reason_code`。

## 5. API 合约测试

- [ ] `/health` 在进程存活时返回成功；`/ready` 仅在索引可用时成功。
- [ ] 索引未就绪时查询返回 `503`，而不是空的成功回答。
- [ ] search/ask 正常响应与 OpenAPI schema 一致。
- [ ] 缺字段、`null`、错误类型、空白 query、超长 query、非法版本、越界 `top_k` 返回稳定 `4xx`。
- [ ] 未知路由返回 `404`；错误 Content-Type 得到明确错误。
- [ ] 内部检索、解析、生成和超时异常映射为稳定错误模型。
- [ ] `5xx` 响应和日志不泄漏堆栈、环境变量、令牌或完整用户代码。
- [ ] 每个请求拥有 `request_id`；中文 JSON 往返不乱码。
- [ ] 对外暴露的归一化 score/confidence 始终位于 `[0, 1]`。
- [ ] 恶意 query 不能扩大工具白名单、读取本地任意路径或触发外部 URL 请求。

## 6. 集成与端到端场景

每个场景必须经过 `ingest -> index -> retrieve -> version filter -> fuse -> generate -> validate citations -> API response` 完整链路。

| ID | 场景 | 预期结果 |
|---|---|---|
| E2E-01 | PyTorch 2.0/2.1 对同一问题有不同修复方式 | 只引用请求版本，答案不混版 |
| E2E-02 | 英文官方文档，中文自然语言提问 | Dense 命中，答案引用英文来源 |
| E2E-03 | 查询包含精确异常码 | BM25 精确结果进入首位 |
| E2E-04 | 两路各召回一个相关来源 | Hybrid 的 Recall 高于任一单路 |
| E2E-05 | 需要两个 chunk 才能完整回答 | 返回两个有效引用，无额外事实 |
| E2E-06 | 文档库完全无答案 | 明确拒答，引用为空 |
| E2E-07 | 只有其他版本存在答案 | 严格模式返回 `VERSION_MISMATCH` |
| E2E-08 | 两个高分来源相互冲突 | 拒答或提示冲突，不任意选择 |
| E2E-09 | 重复构建并重启加载索引 | 结果、排名和引用完全一致 |
| E2E-10 | prompt injection 混入 query/文档 | 只按检索与引用规则作答 |

此外：

- [ ] 20 个并发 FakeGenerator 请求无状态串扰、无错误、无交叉引用。
- [ ] 缓存命中与未命中的响应语义相同。
- [ ] 索引文件损坏时 fail closed，并使 readiness 失败。
- [ ] 重复 ingest 幂等，不增加重复文档或改变指标。

## 7. 离线评测集

### 7.1 规模与分布

正式冻结集建议 100 题，最小可接受为 60 题：

- 40% 精确事实、API 名称或异常码；
- 20% 语义改写、中英跨语言；
- 15% 版本敏感问题；
- 10% 需要多段证据的问题；
- 15% 不可回答问题，其中包含域外、无版本和冲突证据。

默认 smoke 集位于 `data/evaluation/golden_queries.jsonl`，当前提交 26 题（20 条可回答、6 条应拒答），覆盖 E2E-01 至 E2E-10 所代表的核心场景，并加入一条高词汇重合的 OOM hard negative；CI 通过 `scripts/evaluate.py` 对该提交到仓库的确定性数据执行 smoke evaluation。上面的 60/100 题是正式冻结集目标，不是当前已达规模。

### 7.2 JSONL 字段

```json
{
  "id": "PT-VER-001",
  "query": "……",
  "project": "pytorch",
  "requested_version": "2.1.0",
  "answerable": true,
  "relevant_chunk_ids": {"chunk-a": 2, "chunk-b": 1},
  "acceptable_source_ids": ["source-a"],
  "must_include_facts": ["……"],
  "forbidden_facts": ["……"],
  "expected_refusal_reason": null,
  "difficulty": "medium",
  "tags": ["version-sensitive"],
  "source_group_id": "issue-123"
}
```

### 7.3 防泄漏规则

- [ ] 以 `source_group_id`、Issue ID 或 URL 家族切分数据，不能随机切 chunk。
- [ ] 同一来源的切块、同一 Issue 的改写和近重复问句不能跨 train/eval。
- [ ] 不同版本的同题样例成对标注，明确 `forbidden_facts`。
- [ ] 至少两人独立标注 relevant grade 和 must-include facts，分歧经仲裁记录。
- [ ] 冻结集变更必须说明原因，保留新旧指标，禁止为提高分数静默删题。

## 8. 指标定义与硬门槛

| 类别 | 指标 | 合格线 |
|---|---|---:|
| 检索 | Recall@5 | >= 0.90 |
| 检索 | MRR@10 | >= 0.80 |
| 检索 | nDCG@10（相关等级 0/1/2） | >= 0.82 |
| 版本 | Version Accuracy | 1.00 |
| 引用 | Citation Validity | 1.00 |
| 引用 | Answer Evidence Recall | >= 0.90 |
| 引用 | Citation Support | >= 0.95 |
| 拒答 | Refusal F1 | >= 0.85 |
| 安全 | 无证据时输出确定答案的比例 | 0.00 |
| 回归 | Hybrid Recall@5 相对最佳单路 | 不低于 0.02 以上 |
| 测试 | 总行/分支覆盖率 | >= 0.80 |
| 测试 | 关键模块覆盖率 | >= 0.90 |

指标口径：

- `Recall@5`：逐题计算前 5 条命中的去重 relevant ID 数占该题全部 relevant ID 的比例，再对问题取均值。
- `MRR@10`：前 10 条中首个 relevant chunk 的倒数排名均值。
- `nDCG@10`：使用 `0/1/2` 的相关等级计算。
- `Version Accuracy`：所有返回证据均满足目标版本约束的问题占比。
- `Citation Validity`：引用 ID 和元数据能在本次检索上下文中验证的比例。
- `Answer Evidence Recall`：可回答题的最终引用是否至少命中一个人工 relevant ID；不等于逐 claim 支持率。
- `Citation Support`：引用证据确实支持对应回答事实的比例。
- `Refusal P/R/F1`：以 `answerable=false` 为应拒答正类。

报告必须包含数据集版本、样本数、随机种子、配置、BM25/Dense/Hybrid 三组结果、阈值、失败题 ID 和生成时间。不能只输出一个总百分比。

## 9. 性能和稳定性

性能测试在报告中记录 CPU、内存和操作系统，不用开发机数据替代 CI 数据。

当前 `reports/benchmark.json` 只是较小的开发机基线：Windows 11 / Python 3.14.6、21 文档、200 请求、20 worker 的预热进程内调用，200 次成功，P95 70.187 ms。脚本不经过 HTTP、网络或 LLM，也未记录 CPU/内存，所以不满足下面的 10k/API 正式性能验收项，不能用于 SLA。

- [ ] 10,000 个合成 chunk 的冷启动索引构建时间不超过 30 秒。
- [ ] 10,000 个 chunk 上的 search P95 不超过 750 ms。
- [ ] 20 并发、FakeGenerator 的完整 API P95 不超过 2 秒，错误率为 0。
- [ ] 连续执行 100 次相同查询，返回排名和引用完全一致。
- [ ] 大请求被长度限制拒绝，不导致内存线性失控。
- [ ] 缓存损坏或索引加载失败时服务不返回未经验证的答案。

由于共享 CI 硬件存在抖动，性能测试可独立为非阻塞报告；功能正确性、引用和拒答指标必须阻塞合并。

## 10. CI 与发布验收

CI 在 Python 3.12、3.13 上执行：

1. 安装 `.[dev]`；
2. `ruff check` 与 `ruff format --check`；
3. `mypy src/devassist` 严格类型检查；
4. pytest、branch coverage 和 80% 覆盖率门槛；
5. 当前 26 题离线 smoke evaluation，并按阈值决定退出码；
6. wheel/sdist 构建，以及独立 Docker Compose 健康检查和反馈卷写入 smoke。

发布前逐项确认：

- [ ] CI 两个 Python 版本全部通过。
- [ ] README 的离线启动、测试、评测命令在干净环境复现成功。
- [ ] OpenAPI 文档含成功与错误示例。
- [ ] `smoke-report.json` 可由提交的数据重新计算。
- [ ] 演示包含版本隔离、错误码、中文问英文资料、多证据引用和安全拒答。
- [ ] 至少记录 3 个真实失败案例、根因和下一步改进。
- [ ] 简历中的语料量、指标、并发和延迟数字均能由仓库报告复核。
- [ ] 仓库不包含密钥、个人数据、缓存模型或来源不明的数据。

只有上述阻塞项全部通过，项目才能标记为可发布或用于简历展示。
