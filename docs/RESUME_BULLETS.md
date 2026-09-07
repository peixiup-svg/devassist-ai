# DevAssist AI：简历表述与事实边界

本页只使用当前仓库可以复核的事实。项目仍是默认离线、确定性的工程 Demo，但已经包含一个 40 对样本的 LogisticRegression 特征权重训练管线、可选 OpenAI-compatible 结构化生成，以及 Docker/Compose。它不应被写成“生产上线”“服务真实用户”“训练大模型”或“达到生产准确率”。

## 项目标题

**DevAssist AI｜版本感知的 Python / AI 技术支持 Agent（个人工程 Demo）**

技术栈：Python、FastAPI、Pydantic、NumPy、scikit-learn、LogisticRegression、BM25、TF-IDF/LSA、RRF、pytest、Docker、HTML/CSS/JavaScript

## 推荐简历 Bullet

- 设计并实现 DevAssist AI 工程 Demo，面向 PyTorch、Transformers 与 FastAPI 技术问题，以 FastAPI 串联 `查询分析 → 混合检索 → 版本过滤 → 生成 → 引用校验/拒答`；默认使用可离线复现的抽取式生成，并提供 OpenAI-compatible JSON 生成与失败自动回退。
- 构建 **40 对标注样本（20 positive / 20 hard negative）**的轻量训练管线，以六维检索特征拟合 LogisticRegression，将正系数裁剪、归一化后按 **30% 学习权重 + 70% 安全默认权重**生成 artifact，并由服务默认加载；训练集拟合准确率为 100%（仅管线检查，无 held-out 泛化结论）。
- 实现 BM25 与 TF-IDF 字符 n-gram + LSA 双路召回、加权 RRF、训练/安全混合的特征重排及版本兼容过滤；在 **21 条人工 curated 文档、20 条可回答 smoke 问题**上，Hybrid + Rerank 的 Recall@5、MRR@10、nDCG@10 均为 1.0000。
- 建立 **26 条确定性 smoke case（20 条可回答、6 条应拒答）**的离线消融，固定样例中 Refusal F1、Citation Validity 与 Answer Evidence Recall 均为 1.0000；交付前本地自动化测试 **205 passed、覆盖率 90.54%**；实现密钥脱敏、prompt injection 提示、引用白名单、原子式索引切换，并提供非 root Docker 镜像、Compose 健康检查及 VS Code 调试/评测配置。
- 为离线服务增加可复现的进程内并发脚本；一次 Windows 11 / Python 3.14.6、21 文档、200 请求、20 worker 的预热快照得到 200/200 成功、P95 70.187 ms（不含 HTTP/网络/LLM，不作为生产压测或 SLA）。

若版面只能容纳三条，保留前三条，并把第四条中的测试、安全与 Docker 能力合并进第一条；第五条性能数字只有能保留完整测试口径时再使用。所有 smoke、训练和性能数字后都应保留小样本/本机限定。

## 更短版本

- 构建版本感知的 Python 技术支持 Demo，以 FastAPI 串联 BM25 + LSA、RRF、训练/安全混合的特征重排、默认抽取式回答和引用校验，并提供可选 OpenAI-compatible 生成及失败回退。
- 基于 40 对正/困难负样本训练 LogisticRegression 特征权重并按 30:70 与安全默认值混合；训练集拟合准确率 100%（无独立验证集，非泛化指标）。
- 在 21 条 curated 文档与 26 条同仓 smoke case 上完成四组消融；Hybrid + Rerank 的 Recall@5/MRR@10/nDCG@10 和 6 条不可回答样例上的 Refusal F1 均为 1.0000（小型固定回归集，非生产结论），并提供非 root Docker/Compose 运行方式。

## 数字的准确口径

| 可写事实 | 准确含义 |
|---|---|
| 21 条文档 | 人工整理的 Demo JSONL 条目；PyTorch 10、Transformers 6、FastAPI 5 |
| 40 对训练 pair | 20 positive、20 hard negative；与 Demo 语料同仓维护，没有独立验证集 |
| Training accuracy = 1.0000 | LogisticRegression 对这 40 个训练样本的拟合结果；仅为管线检查，不是 held-out 指标 |
| 30% / 70% blend | 学习后正权重占 30%，代码中的安全默认权重占 70%；不是线上 A/B 流量比例 |
| 26 条评测 | 同仓维护的 curated offline smoke；20 条可回答、6 条应拒答 |
| Recall@5 = 1.0000 | 四种模式在 20 条可回答固定样例上均命中；不能证明 Hybrid 优于 BM25 |
| MRR@10 / nDCG@10 = 1.0000 | 仅 Hybrid + 训练/安全混合 Feature Rerank 的当前 smoke 结果；Dense 分别为 0.9750 / 0.9754 |
| Refusal F1 = 1.0000 | 只基于 6 条应拒答案例，样本量很小 |
| Citation Validity = 1.0000 | 引用能映射到本次检索文档且 evidence 为原文片段；不是人工事实正确率 |
| Answer Evidence Recall = 1.0000 | 20 条可回答固定题的答案引用均至少命中一个 relevant ID；不是 claim-level 支持率 |
| 20 worker / P95 70.187 ms | 21 文档、200 次预热后的进程内调用；排除 HTTP、网络和 LLM，仅为单机快照 |
| 205 passed / 90.54% | Windows 11 / Python 3.14.6 的交付前本地测试与总覆盖率；不等于 Python 3.12/3.13 CI 已在线运行 |

所有评测数字可在 `reports/evaluation.json` 复核，并通过以下命令重新生成：

```bash
python scripts/evaluate.py --fail-on-threshold --output reports/evaluation.json
python scripts/benchmark.py --requests 200 --concurrency 20 --output reports/benchmark.json
```

训练 artifact 的口径可在 `artifacts/reranker/weights.json` 复核，并可通过以下命令重建：

```bash
python scripts/train_reranker.py
```

## 当前不能写的说法

不要写：

- “上线并服务 X 名用户”——当前只有本地运行方式，没有真实用户证据。
- “生产环境准确率 100%”——100% 来自 26 条小型 curated smoke case。
- “没有训练任何模型”——不准确；当前确实训练了 LogisticRegression，并把学习结果安全混合为运行时特征权重。
- “训练/微调了 Cross-Encoder 或大语言模型”——不准确；当前没有神经重排模型或 LLM 微调。
- “重排模型泛化准确率 100%”——1.0000 来自同一批 40 对训练样本，没有 held-out 数据，只能称训练集拟合准确率或训练管线检查。
- “使用向量数据库与神经 Embedding”——当前 Dense 是 scikit-learn 的 TF-IDF + LSA，索引在进程内。
- “RAG 显著提升 Recall”——当前 BM25 在固定评测上已经达到 Recall@5 1.0000，没有观察到可声明的 Recall 提升。
- “已上线 Docker/Kubernetes 生产集群”——当前实现的是本地 Docker 镜像与 Compose，不是云端生产部署。
- “LLM 生成质量已达到 X%”——OpenAI-compatible 通路是可选能力，默认关闭，也未纳入当前 26 条离线 smoke 报告。
- “高并发、低延迟、SLA、零故障”——当前只有 21 文档、200 请求的开发机进程内快照，排除了 HTTP/网络/LLM；没有生产压测或在线可靠性数据。
- “防御所有提示词注入与数据泄漏”——目前仅实现有限规则、脱敏和引用约束。

## 面试时的 30 秒说明

> 我做的是一个默认可离线复现的版本感知技术支持 Demo。它先用 BM25 和 TF-IDF/LSA 找候选，经 RRF、版本过滤和特征重排后生成带引用答案；重排权重来自 40 对样本训练的 LogisticRegression，并按 30:70 与安全默认权重混合。默认生成器是抽取式，也支持可选 OpenAI-compatible JSON 生成和失败回退。我用 21 条文档、26 条 smoke case 验证链路与消融，但会明确说明训练与评测集合都很小，不能代表开放问题或生产表现。

这个说明能把项目的工程价值和实验边界同时讲清楚，比把 Demo 包装成生产 AI 系统更可信。
