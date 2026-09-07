# DevAssist AI

> 一个版本感知、引用可验证、证据不足时主动拒答的 Python / AI 技术支持 Demo。

DevAssist AI 接收问题描述、traceback、项目版本和运行环境，在本地知识库中检索 PyTorch、Transformers 与 FastAPI 资料，再返回诊断、排查步骤、置信度和可核验引用。它刻意采用受约束的 `analyse → retrieve → generate → validate` 工作流：没有足够证据、版本不匹配或请求不安全时，不生成确定性修复方案。

当前仓库是一套**可离线复现的工程 Demo**，用于展示数据摄取、轻量监督训练、混合检索、版本过滤、引用校验、拒答策略、FastAPI 服务和评测闭环。仓库包含一个用 40 对标注样本拟合 LogisticRegression、再与安全默认值混合的特征权重训练管线；它不是神经重排模型，也不是生产客服系统或真实用户流量证明。

## 项目价值

普通技术问答很容易把相似但不同版本的资料混在一起。DevAssist AI 重点处理四类风险：

- **版本错配**：请求指定版本后，优先返回兼容资料；无法覆盖时以 `VERSION_MISMATCH` 拒答。
- **关键词与语义互补**：BM25 捕捉错误码、函数名，LSA Dense 补充字符级语义召回，再以加权 RRF 融合。
- **答案可追溯**：回答采用已整理资料中的摘要和解决步骤；引用必须能回溯到本次检索上下文。
- **证据不足不猜测**：通过检索锚点、置信度、版本范围和引用校验共同决定回答或拒答。

本地 Web 工作台还提供示例问题、环境输入、健康状态、运行指标、拒答提示、来源卡片与反馈入口。

## 当前实现边界

| 能力 | 当前真实实现 | 尚未实现 |
|---|---|---|
| 语料 | 21 条人工整理的 JSONL Demo 文档 | 自动抓取、许可证治理、大规模多版本语料 |
| 稀疏检索 | 自实现 BM25 | Elasticsearch / OpenSearch 等生产索引 |
| Dense 检索 | TF-IDF 字符 n-gram + LSA，固定随机种子 | 神经 Embedding、向量数据库 |
| 重排 | 40 对标注 pair 训练 LogisticRegression，再把学习结果按 30% 与 70% 安全默认权重混合；服务默认加载该权重 artifact | 独立验证集、神经 Cross-Encoder、大规模 Learning-to-Rank |
| 生成 | 默认抽取式；可选 OpenAI-compatible 结构化生成，调用失败自动回退抽取式 | LLM 质量评测、事实级 claim 验证、模型微调 |
| Agent | 单轮、有界的规则路由与检索—生成—校验 | 自主多轮规划、外部工具调用 |
| 存储 | 进程内索引；反馈追加到本地 JSONL | Redis、PostgreSQL、分布式持久化 |
| 部署 | 本机 FastAPI、零构建静态页面、非 root Docker 镜像、Compose、健康检查，以及 20 worker 的进程内离线性能快照 | 云部署、Kubernetes、HTTP 压测、多机共享状态、SLA 或真实流量验证 |

因此，项目中的“Dense”“Agent”“训练”和“部署”均按上述工程含义使用：训练的是小型线性分类器所导出的混合特征权重，不是大模型；Docker 可复现也不等于已经上线生产环境。

## 架构

```text
浏览器 / CLI
      │
      ▼
FastAPI + Pydantic
  ├─ 请求校验、固定窗口限流
  ├─ /health、/v1/metrics、/v1/feedback
  └─ 原子式 reindex（新索引构建完成后切换）
      │
      ▼
QueryAnalyzer
  ├─ 清洗与疑似密钥脱敏
  ├─ 项目、版本、异常名和意图识别
  └─ 文档 / Issue / Changelog / Migration 路由
      │
      ▼
HybridRetriever
  ├─ BM25
  ├─ TF-IDF char n-gram + LSA
  ├─ weighted RRF
  ├─ 默认加载训练并安全混合的可解释特征权重
  └─ 项目、来源类型和版本兼容过滤
      │
      ▼
Bounded Agent
  ├─ 证据与置信度门控
  ├─ 默认 ExtractiveGenerator
  ├─ 可选 OpenAI-compatible JSON Generator
  │    └─ 超时、网络或结构错误时回退 ExtractiveGenerator
  ├─ CitationValidator
  └─ ANSWERED / NO_EVIDENCE / LOW_CONFIDENCE /
     VERSION_MISMATCH / UNSAFE_QUERY
```

```text
40 对标注样本（20 positive + 20 hard negative）
      ↓
六维特征：融合分、正文/标题重叠、代码信号、版本、来源权威度
      ↓
LogisticRegression（balanced / liblinear / random_state=42）
      ↓
正系数裁剪与归一化
      ↓
30% 学习权重 + 70% 安全默认权重
      ↓
artifacts/reranker/weights.json → Service 默认加载
```

数据链路由 `ingestion` 模块负责：JSONL 读取 → 字段校验与清洗 → 内容指纹去重 → 稳定切块 → JSONL 输出。离线评测对 BM25、Dense、Hybrid 和 Hybrid + Rerank 使用同一组固定数据做消融；默认评测仍采用抽取式生成，不调用外部 LLM。

## 快速开始

要求 Python 3.12 或更高版本。默认运行不需要网络、GPU、模型 API Key 或外部数据库。

### Windows PowerShell

```powershell
cd outputs/devassist-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
devassist serve --reload
```

如果 PowerShell 不允许执行激活脚本，可以直接使用虚拟环境中的解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m devassist.cli serve --reload
```

### macOS / Linux

```bash
cd outputs/devassist-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
devassist serve --reload
```

启动后访问：

- Web 工作台：<http://127.0.0.1:8000/>
- OpenAPI / Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

也可以不启动 API，直接从 CLI 验证：

```bash
devassist search "torch.cuda.is_available 为什么是 False" --project pytorch --version 2.6
devassist diagnose "RuntimeError: Expected all tensors to be on the same device" --project pytorch --version 2.6
```

源码目录和 Docker 镜像包含演示语料、评测集与已训练权重，是推荐的完整交付方式。`dist/` 中的 wheel 仅打包 Python 代码和静态页面；单独安装 wheel 时，需要通过 `DEVASSIST_ROOT`（或更细粒度的 `DEVASSIST_DATA_PATH`、`DEVASSIST_EVAL_PATH`、`DEVASSIST_RERANKER_WEIGHTS_PATH`）指向这些外部数据文件。

### Docker Compose

仓库已经包含 `Dockerfile` 和 `docker-compose.yml`：

```bash
docker compose up --build
```

容器使用 Python 3.12 slim、非 root 用户和 `/health` 健康检查；Compose 默认仅绑定 `127.0.0.1`，并用具名卷 `devassist-runtime` 持久化反馈，避免宿主机目录所有权冲突。访问地址仍为 <http://127.0.0.1:8000/>。停止服务可运行：

```bash
docker compose down
```

Compose 只证明容器化运行链路；仓库没有据此声称云端上线、生产 SLA 或多副本一致性。

## 在 VS Code 中运行

1. 用 VS Code 打开 `outputs/devassist-ai`，不要打开其上层目录。
2. 通过 **Python: Select Interpreter** 选择 `.venv/Scripts/python.exe`（Windows）或 `.venv/bin/python`（macOS / Linux）。
3. 在集成终端执行 `python -m pip install -e ".[dev]"`。
4. 仓库已经提交 `.vscode/launch.json`、`tasks.json`、`settings.json` 和扩展推荐，无需手工复制配置。
5. 打开 **Run and Debug**，选择以下任一配置：
   - `DevAssist: API (debug)`：启动可断点调试的 FastAPI 服务；
   - `DevAssist: Offline evaluation`：重新生成 smoke 报告；
   - `DevAssist: Offline benchmark`：运行不经过 HTTP/网络/LLM 的进程内性能快照；
   - `DevAssist: Train reranker`：重新训练并写入混合权重 artifact。

也可从 **Terminal → Run Task** 执行环境安装、pytest 或离线评测。在 `src/devassist/api/app.py`、`agent.py`、`generation.py` 或 `retrieval/hybrid.py` 中设置断点，即可跟踪完整请求链路。

## API

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/` | 静态 Web 工作台 |
| `GET` | `/health` | 索引状态、文档数和版本 |
| `POST` | `/v1/search` | 查看 BM25 / Dense / Hybrid / Rerank 检索结果 |
| `POST` | `/v1/diagnose` | 执行受约束诊断工作流 |
| `POST` | `/v1/feedback` | 将赞踩与可选评论写入本地 JSONL |
| `GET` | `/v1/metrics` | 当前进程内请求、拒答、反馈和延迟摘要 |
| `POST` | `/admin/reindex` | 使用 `x-admin-token` 触发重建；未配置时关闭 |

### 诊断请求示例

```bash
curl -X POST http://127.0.0.1:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "query": "PyTorch 2.6 中 torch.load 为什么提示 weights_only 反序列化失败？",
    "project": "pytorch",
    "version": "2.6",
    "environment": {
      "python_version": "3.11.9",
      "cuda_version": "12.1",
      "operating_system": "Windows 11",
      "packages": {"torch": "2.6.0"}
    },
    "top_k": 5
  }'
```

PowerShell 原生写法：

```powershell
$body = @{
  query = "PyTorch 2.6 中 torch.load 为什么提示 weights_only 反序列化失败？"
  project = "pytorch"
  version = "2.6"
  environment = @{
    python_version = "3.11.9"
    cuda_version = "12.1"
    operating_system = "Windows 11"
    packages = @{ torch = "2.6.0" }
  }
  top_k = 5
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/diagnose" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

响应字段包括 `status`、`reason_code`、`diagnosis`、`steps`、`citations`、`confidence`、`missing_information`、`tools_used`、`safety_warnings` 与 `latency_ms`。所有公开请求模型均禁止未知字段；`query` 长度限制为 3–12,000 个字符。

### 检索消融接口

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "device_map='auto' requires Accelerate",
    "project": "transformers",
    "version": "4.46",
    "top_k": 5,
    "mode": "hybrid_rerank"
  }'
```

`mode` 可选 `bm25`、`dense`、`hybrid` 或 `hybrid_rerank`。

## 重排权重训练

仓库提交了 40 对训练样本：20 对正样本和 20 对 hard negative。训练脚本基于六个可解释特征拟合 `LogisticRegression`，将正系数裁剪、归一化后，再按 `blend=0.30` 与原有安全默认权重混合，避免 40 个小样本完全覆盖人工先验。生成的 JSON artifact 会由 `Settings` 和 `DevAssistService` 默认加载。

```bash
python scripts/train_reranker.py
python scripts/evaluate.py --fail-on-threshold --output reports/evaluation.json
```

当前 [`artifacts/reranker/weights.json`](artifacts/reranker/weights.json) 的元数据为：

| 项目 | 值 |
|---|---:|
| Training pairs | 40 |
| Positive / hard negative | 20 / 20 |
| Training accuracy | 1.0000 |
| Learned/default blend | 0.30 / 0.70 |

> [!WARNING]
> **1.0000 是同一批 40 对样本上的训练集拟合准确率，只用于确认训练管线工作，不是 held-out 指标，也不能证明排序泛化能力。** 当前没有独立验证集、交叉验证、统计区间或线上 A/B；真正的质量结论仍需更大且按来源/时间隔离的数据。

如需保留现有 artifact，可把新权重写到其他位置，并通过 `DEVASSIST_RERANKER_WEIGHTS_PATH` 指向它。权重文件不存在时，`FeatureReranker` 会使用代码中的安全默认权重。

## 可选 OpenAI-compatible 生成

默认配置**不会调用 LLM**，使用确定性的 `ExtractiveGenerator`，因此 smoke 评测可离线复现。只有同时设置 `DEVASSIST_LLM_BASE_URL` 和 `DEVASSIST_LLM_MODEL` 时，服务才会启用 OpenAI-compatible `/chat/completions` 结构化生成：

```powershell
$env:DEVASSIST_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
$env:DEVASSIST_LLM_MODEL = "your-local-model"
$env:DEVASSIST_LLM_API_KEY = "optional-token"
devassist serve --reload
```

`base_url` 必须为 HTTPS，或使用 loopback HTTP（`localhost` / `127.0.0.1` / `::1`）。生成器要求返回包含 `diagnosis: string` 和 `steps: string[]` 的 JSON；网络失败、超时、格式错误或空响应时，会自动回退到抽取式生成。引用永远由 `CitationValidator` 从本次检索结果构建，不接受模型自行生成的 URL。

这个可选通路尚未纳入当前 26 条离线 smoke 报告，仓库也没有提供其质量、成本、延迟或稳定性结论。

## 评测与消融

> [!CAUTION]
> **下面的 100% 结果只来自仓库内 21 条人工 curated Demo 文档和 26 条确定性 smoke 问题，不是生产结论。** 其中仅有 20 条可回答问题、6 条应拒答问题，数据规模小、与实现同仓维护，也没有时间切分、真实流量、人工盲评或统计置信区间。它只能证明这些固定样例没有回归，不能证明系统能泛化到开放世界问题。

以下数字原样来自 [`reports/evaluation.json`](reports/evaluation.json)，报告生成时间为 **2026-09-06 13:31:28 UTC**，评测已加载 `artifacts/reranker/weights.json`。知识库按项目分布为 PyTorch 10 条、Transformers 6 条、FastAPI 5 条。

### 检索消融（20 条可回答 smoke case）

| 模式 | Recall@5 | MRR@10 | nDCG@10 | Version Accuracy |
|---|---:|---:|---:|---:|
| BM25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LSA Dense | 1.0000 | 0.9750 | 0.9754 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 0.9960 | 1.0000 |
| Hybrid + Trained/Blended Feature Rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

这个小型集合中 BM25 已经饱和，因此不能据此声称 Hybrid 显著优于 BM25；Dense 与 Hybrid 的轻微差异只用于验证消融代码和排序行为。下一阶段需要更大、更难、按来源和时间隔离的评测集。

### Agent smoke（全部 26 case）

| 指标 | 结果 |
|---|---:|
| 回答 / 拒答 | 20 / 6 |
| Refusal Precision / Recall / F1 | 1.0000 / 1.0000 / 1.0000 |
| Citation Validity | 1.0000 |
| Answer Evidence Recall | 1.0000 |
| Expected Reason Accuracy | 1.0000 |
| 失败 case | 0 |

这里的 `Citation Validity` 只检查引用 ID、标题、URL 和 evidence 是否属于本次检索结果；`Answer Evidence Recall` 只检查 20 条可回答固定题的回答引用是否至少命中一个人工 relevant ID。两者都不等于逐 claim 的事实支持率或人工验证的答案正确率。

### 进程内离线性能快照

以下数字来自 [`reports/benchmark.json`](reports/benchmark.json)：Windows 11、Python 3.14.6、21 条文档、200 次诊断调用、20 worker，200 次均返回 answered。它在同一 Python 进程中直接调用 `DevAssistService`，已预热，并且**不经过 HTTP、网络或 LLM**。

| 指标 | 当前快照 |
|---|---:|
| 成功 / 错误 | 200 / 0 |
| 吞吐 | 466.080 req/s |
| Mean / P50 | 26.839 / 17.762 ms |
| P95 / P99 | 70.187 / 137.784 ms |
| 总墙钟时间 | 0.429 s |

这只是开发机上的小语料、热缓存、进程内并发快照，不能写成 API P95、生产吞吐、SLA 或 10k 文档规模结论；不同 CPU、Python 和依赖版本也可能得到不同数字。

### 复现

```bash
python -m pytest -q
python scripts/evaluate.py --fail-on-threshold --output reports/evaluation.json
python scripts/benchmark.py --requests 200 --concurrency 20 --output reports/benchmark.json
```

交付前在 Windows 11 / Python 3.14.6 的本地运行结果为 **205 passed、总覆盖率 90.54%**（另有 1 条来自 FastAPI/Starlette TestClient 依赖链的弃用 warning）。测试数量会随用例演进，是否通过仍以当前命令退出码为准。评测脚本会重新计算四组检索消融、拒答、引用合法性和回答证据召回，并在阈值未通过时返回非零退出码。更完整的测试目标和正式评测设计见 [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md)。

## 安全设计与限制

已经实现：

- 不执行用户粘贴的代码，也不根据检索文档扩大工具范围。
- 清理 NUL 字符，并对常见 API Key、GitHub Token、密码形式做正则脱敏。
- 识别常见中英文 prompt injection 语句并返回警告；纯恶意指令以 `UNSAFE_QUERY` 拒答。
- 引用只允许来自本次检索结果，且 URL 必须以 `http://` 或 `https://` 开头。
- 静态前端以 `textContent` 渲染服务端文本，并再次限制引用链接协议。
- 可选 LLM 端点要求 HTTPS 或本机 loopback HTTP；检索文本被明确标为不可信证据，模型不能提供引用。
- 可选生成失败时回退到确定性抽取式生成，不绕过原有置信度与引用校验。
- 查询长度、环境包数量、Pydantic schema、`top_k` 与固定窗口请求频率均有限制；限流键数量也有上界。
- `/admin/reindex` 默认关闭；启用时要求至少 16 字符的 `DEVASSIST_ADMIN_TOKEN`，并同样经过限流。
- Docker 镜像使用非 root 用户运行，并配置 `/health` 健康检查。

仍需注意：

- 正则脱敏不是完整 DLP，不应提交真实密钥、私有源码或个人数据；疑似凭据一旦暴露仍应立即轮换。
- 当前限流、指标和索引都在单进程内，重启后会丢失，不适用于多实例部署。
- Demo 没有用户认证、租户隔离、审计系统、内容许可证流水线或恶意语料治理。
- 外部 LLM 会收到当前问题和最多三条检索证据；启用前必须按所选服务商的数据政策评估敏感信息风险。
- 本地限流只使用 ASGI 对端地址并忽略可伪造的 `X-Forwarded-For`；生产反向代理或多实例部署应改用可信网关与共享限流存储。
- curated 文档可能过时，任何高风险修复仍应回到官方原文并在隔离环境验证。

## 配置

所有路径都可以相对仓库根目录或使用绝对路径：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `DEVASSIST_DATA_PATH` | `data/sample/knowledge_base.jsonl` | 知识库 |
| `DEVASSIST_EVAL_PATH` | `data/evaluation/golden_queries.jsonl` | smoke 评测集 |
| `DEVASSIST_FEEDBACK_PATH` | `data/runtime/feedback.jsonl` | 本地反馈文件 |
| `DEVASSIST_RERANKER_WEIGHTS_PATH` | `artifacts/reranker/weights.json` | 训练并混合后的重排权重 |
| `DEVASSIST_TOP_K` | `5` | 默认结果数 |
| `DEVASSIST_CANDIDATE_K` | `20` | 每路候选数 |
| `DEVASSIST_CONFIDENCE_THRESHOLD` | `0.36` | 回答置信度门槛 |
| `DEVASSIST_MAX_QUERY_CHARS` | `12000` | API 查询字符上限 |
| `DEVASSIST_RATE_LIMIT_REQUESTS` | `60` | 窗口内请求数 |
| `DEVASSIST_RATE_LIMIT_WINDOW_SECONDS` | `60` | 限流窗口秒数 |
| `DEVASSIST_ADMIN_TOKEN` | 未设置 | reindex 管理令牌；未设置即关闭接口 |
| `DEVASSIST_LLM_BASE_URL` | 未设置 | OpenAI-compatible base URL；与 model 同时设置才启用 |
| `DEVASSIST_LLM_API_KEY` | 未设置 | 可选 Bearer token |
| `DEVASSIST_LLM_MODEL` | 未设置 | 可选结构化生成模型名 |
| `DEVASSIST_LLM_TIMEOUT_SECONDS` | `20` | 可选生成调用超时 |

## 目录结构

```text
devassist-ai/
├─ .vscode/                              # Debug、任务、测试与扩展配置
├─ artifacts/reranker/weights.json      # 训练后安全混合的默认权重
├─ data/
│  ├─ sample/knowledge_base.jsonl      # 21 条 curated Demo 文档
│  ├─ evaluation/golden_queries.jsonl  # 26 条 smoke 问题
│  └─ training/reranker_pairs.jsonl    # 40 对正/困难负训练样本
├─ docs/
│  ├─ TEST_PLAN.md                     # 正式质量门槛设计
│  └─ RESUME_BULLETS.md                # 可核验的简历表述
├─ reports/
│  ├─ evaluation.json                  # 当前可复现 smoke 报告
│  └─ benchmark.json                   # 进程内离线性能快照
├─ scripts/
│  ├─ evaluate.py                      # 离线消融入口
│  ├─ benchmark.py                     # 不经过 HTTP/网络/LLM 的并发快照
│  └─ train_reranker.py                # LogisticRegression 权重训练
├─ src/devassist/
│  ├─ api/app.py                       # FastAPI 与限流、运维端点
│  ├─ ingestion/                       # 清洗、去重、稳定切块
│  ├─ retrieval/                       # BM25、LSA、RRF、特征重排
│  ├─ training/                        # pair 读取、特征训练与安全 blend
│  ├─ evaluation/                      # 指标与评测 runner
│  ├─ static/index.html                # 零构建响应式前端
│  ├─ analyzer.py                      # 查询分析与路由
│  ├─ agent.py                         # 有界工作流与拒答门控
│  ├─ citations.py                     # 引用构建和合法性校验
│  ├─ generation.py                    # 默认抽取式 + 可选 OpenAI-compatible 生成
│  ├─ security.py                      # 脱敏与注入提示检测
│  └─ service.py                       # 索引、指标与反馈服务
├─ tests/
│  ├─ unit/                            # 自动化单元测试
│  └─ integration/                     # FastAPI 合约与集成测试
├─ Dockerfile                          # 非 root 运行与健康检查
├─ docker-compose.yml                  # 服务与反馈卷配置
└─ pyproject.toml
```

## 扩展路线

1. **先扩大评测可信度**：从公开文档和 resolved Issues 构建 500+ 条按时间、Issue ID 和来源族隔离的冻结集；补充困难负样本、双人标注、Citation Support 和人工盲评。
2. **升级当前轻量训练**：把 40 对训练样本扩展为按来源隔离的 train/validation/test；报告 LogisticRegression 的 held-out 指标，再与神经 Cross-Encoder 比较，并验证相对 BM25 的可复现增益。
3. **评测可选 LLM 通路**：保留结构化响应、失败回退和引用白名单，在 claim 级校验后才输出；比较抽取式基线与不同生成模型的质量、延迟和成本。
4. **建设真实数据管线**：增量抓取官方文档、Changelog 与可信 Issue，记录许可证、发布日期、版本范围、内容哈希和数据血缘。
5. **从容器化走向生产化**：将索引、限流、指标和反馈迁移到持久化组件，加入认证、租户隔离、队列、可观测性、镜像扫描、离线评测发布门禁与并发压测。

在完成并留下可复现证据前，不应把上述路线写成已经实现的成果。
