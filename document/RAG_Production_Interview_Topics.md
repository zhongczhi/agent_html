# 生产环境 RAG 面试题与技术图谱

> 调研来源：网络公开的面试经验、岗位复盘、技术博客与工业实践（截至 2026 年中）。
> 目标：汇总「面试官真正关心」的生产级 RAG 技术与考察点，方便对照复习与体系化准备。

---

## 0. 一句话总览

生产级 RAG 不是「向量数据库 + LLM」就完事。面试官期望候选人能讲清一整套**数据 → 索引 → 检索 → 重排 → 生成 → 评估 → 运维**的工程闭环，并对每一步常见的失败模式与优化手段有自己的判断。

下面按模块汇总。

---

## 1. 基础认知：RAG 范式的演进

| 范式 | 核心思路 | 面试要点 |
|------|----------|----------|
| **Naive RAG** | Index → Retrieve → Generate 三段式 | 适合作为讨论起点，暴露出召回低、答非所问、幻觉等典型问题 |
| **Advanced RAG** | 引入查询改写、分块优化、重排、混合检索等 | 是面试主流考察点 |
| **Modular RAG** | 模块化、可插拔的 RAG 框架 | 强调可组合性与扩展性 |
| **Agentic RAG** | 由 Agent 自主决定「要不要检索、检索什么、用什么工具」 | 2025–2026 年的热点，常作为加分项 |
| **GraphRAG** | 在向量检索之上叠加知识图谱，支持多跳推理 | 适合做关系/全局类问答 |

参考：腾讯云「2026 RAG 系统测试实战新趋势」、腾讯云「RAG 系列 04 — Agentic RAG」、SegmentFault「Advanced RAG 06: 生成结果的相关性低? 快用 Query Rewriting」。

---

## 2. 数据预处理与分块（Chunking）

**为什么面试官必问**：检索失败的根因往往不是模型不好，而是 chunk 没切对。

### 2.1 主流策略
1. **固定长度切分**：实现简单，但易切断语义；通常配合 overlap 缓解。
2. **滑动窗口切分**：通过 chunk_size + overlap 保留上下文连续性；增加存储成本。
3. **结构感知切分**：按 Markdown / HTML / PDF 的标题、章节、表格等结构切（MarkdownHeaderTextSplitter、HTMLHeaderTextSplitter）。
4. **语义切分（Semantic Chunking）**：用嵌入相似度动态识别语义边界。
5. **主题/聚类切分**：HDBSCAN、K-means 等聚合相关段落。
6. **LLM 切分 / Meta-Chunking**：基于 LLM 的边际采样、困惑度等做精细切分。
7. **父子分块（Parent-Document Retriever）**：小 chunk 检索、返回父级完整上下文，平衡精度与上下文长度。
8. **Contextual Retrieval（Anthropic）**：为每个 chunk 生成「上下文说明」前缀，再做嵌入 + BM25，top-20 检索失败率可下降 ~49%。
9. **实体保留切分**：结合 NER + 共指消解，确保关键实体不被切断。

### 2.2 工程关注点
- 先抽样、再大规模实验（LLM 评估成本高）。
- chunk 长度与 overlap 没有银弹，要按语料类型调（FAQ vs 长文档 vs 代码）。
- 同一文档可同时维护多种粒度的索引。

---

## 3. 嵌入（Embedding）与索引

### 3.1 关键概念
- **Bi-Encoder**：query / doc 各自独立编码 → 余弦相似度；速度快、可大规模 ANN。
- **Cross-Encoder**：query + doc 拼接联合编码 → 输出相关性分数；精度高、慢。
- **ColBERT / SPLADE / ColBERTv2**：折中方案，token 级交互但仍可预计算。
- 多向量、多模态嵌入（CLIP / BGE-M3 / Qwen3-Embedding）。

### 3.2 面试常见追问
- 嵌入模型的领域适配（中文场景 BGE / M3E / Qwen3-Embedding）。
- Embedding 维度 vs 召回率 / 存储成本 trade-off。
- 量化、二值化对召回的影响。
- Embedding 模型版本切换时的兼容性与索引迁移。
- 多语种场景下跨语言检索的实现。

---

## 4. 检索（Retrieval）—— RAG 的核心战场

### 4.1 检索方式谱系
| 类型 | 代表 | 优势 | 局限 |
|------|------|------|------|
| 稀疏 | BM25、TF-IDF | 精确术语命中、可解释 | 缺语义 |
| 稠密 | 向量 ANN（FAISS / Milvus / Weaviate / LanceDB） | 语义召回 | 缺关键词 |
| 混合 | BM25 + 向量 → RRF / 线性加权 | 召回稳健 | 工程复杂度高 |
| 关键词过滤 | 元数据过滤 / Self-Query Retriever | 缩小检索空间 | 依赖元数据质量 |
| 知识图谱 | GraphRAG / 实体关系图 | 多跳推理、全局问答 | 构建成本高 |

### 4.2 混合检索（Hybrid Search）必答点
- **融合方式**：结果级 late fusion / 分数级 score fusion / 级联 cascade。
- **倒数排名融合（RRF）**：常用且稳健。
- **为何生产必上混合**：FAQ / 法律 / 专利等术语密集场景，BM25 比向量平均高 12% 准确率；冷启动 + CPU 可跑 + 节省 83% 硬件成本（AutoRAG 实测）。

### 4.3 查询理解与改写（Query Rewriting）
- **多查询检索（FAN-OUT）**：把一个 query 改写成多个变体并行召回。
- **RAG-Fusion**：多个查询结果用 RRF 融合。
- **HyDE（Hypothetical Document Embeddings）**：LLM 先生成「假设答案」，再用假设文档嵌入去检索，弥合语义 gap。
- **Step-Back Prompting**：先抽象出更通用的问题再检索。
- **Query2doc / ITER-RETGEN / Rewrite-Retrieve-Read**。

### 4.4 复杂问题分解
- **多跳问题（Multi-hop QA）**：拆成子问题再分别检索。
- **HopRAG**：用 passage graph + retrieve-reason-prune 做逻辑感知多跳检索。

---

## 5. 重排序（Reranking）与检索后处理

**核心观点**：检索失败 ≠ 找不到相关文档，而是「找到了但没排对」。后处理往往比换 embedding 模型收益更大。

### 5.1 Reranker 家族
- **Cross-Encoder Reranker**：BGE-Reranker、Cohere Rerank、Jina Reranker、Qwen-Ranker。
- **LLM Reranker**：RankGPT、RankZephyr，用 LLM 直接对候选打分。
- **ColBERTv2** 作为 lightweight reranker。

### 5.2 上下文压缩（Contextual Compression）
- 用 LLM 提炼检索到的长文本，剔除无关片段，降低 prompt 长度与噪声。

### 5.3 其他后处理
- **MMR（Maximal Marginal Relevance）**：提升结果多样性。
- **Citation-Enhanced Generation**：强制每个事实带 [1] [2] 引用，幻觉率可降 70%+。
- **低置信度拒答**：「我不知道」比硬答更安全。

### 5.4 常见面试题
> 向量检索 Top-K 已经够相关了，为什么还要 rerank？
>
> 答：余弦相似度排名粒度粗、对长文档短答案定位差；Cross-Encoder 做 query-doc 联合编码，可识别「答案就藏在这一两句话中」的局部相关性。

---

## 6. 生成（Generation）与提示工程

- **Anti-hallucination Prompt**：明确告诉模型「不要编造」「参考资料未提到时直接拒答」。
- **Few-shot + 引用编号**。
- **结构化输出（JSON Schema）**：便于下游消费、做校验、做溯源。
- **多轮对话中的上下文管理**：历史摘要、token 预算控制。
- **流式输出（SSE）** 与首字延迟（TTFT）优化。

---

## 7. 评估体系（Evaluation）—— 必考

### 7.1 检索侧指标
- **Precision@K / Recall@K**：最基础的精度/召回。
- **MRR（Mean Reciprocal Rank）**：首个相关文档的排名倒数。
- **nDCG**：对排名质量做归一化折扣累积增益。
- **Context Precision / Context Recall**（RAGAS 体系）。

### 7.2 生成侧指标
- **Faithfulness（忠实度）**：答案是否能从检索 context 推断出来，对抗幻觉。
- **Answer Relevance**：答案与问题的相关度。
- **Context Entity Recall**：是否抓到问题中的关键实体。
- **Answer Correctness / Semantic Similarity**：相对 ground truth 的语义相似度。

### 7.3 评估框架
- **RAGAS**（开源主流）：question / answer / contexts / ground_truth 四列输入，自动打分。
- **LangSmith / LangChain Evaluators**：在线 A/B、prompt 版本对比。
- **TruLens / DeepEval / Phoenix**：可观测 + 评估一体化。

### 7.4 工业界新趋势（2026）
> 测试重心从「输出正确性」迁移到「过程可证伪性」——
> 检索层 → 重排层 → 生成层每一步都要可独立验证；
> 生成层强制输出「溯源置信度」（Source Confidence Score, SCS），支持反向追溯到原始段落及检索得分。
> —— 腾讯云《2026 RAG 系统测试实战新趋势》

### 7.5 LLM-as-a-Judge
- 使用 GPT-4 / Claude 作为裁判模型。
- 注意 judge model 的偏差与成本。
- 评测集构建（从 chunk 自动生成 Q&A 是常见套路）。

---

## 8. Agentic RAG（2025–2026 热点）

> "Agentic RAG = 传统 RAG + LLM Agent 自主决策。"
> 引入规划、反思、多跳检索、工具调用，把"被动召回 + 生成"升级为主动探索。

### 8.1 经典架构（LangGraph 视角）
- **节点**：classify → retrieve → evaluate → generate / re_route。
- **条件边**：根据上下文质量分决定是否重试、改写或换检索源。

### 8.2 范式分类
1. **Single-Agent RAG**：一个 agent 调度多种检索工具。
2. **Multi-Agent RAG**：Planner + Researcher + Critic + Writer 分工。
3. **Hierarchical RAG**：分层路由 + 反思。
4. **Corrective RAG（CRAG）**：检索质量差时 fallback 到 web 搜索。
5. **Adaptive RAG**：根据问题复杂度动态决定要不要检索。
6. **Graph-based Agentic RAG**：把图谱作为 agent 的工具之一。

### 8.3 面试亮点话术
> 不能「上来就 Agentic」。生产中先用 Naive RAG 跑通闭环，再用评估驱动逐步加 rerank、hybrid、query rewrite；最后才引入 Agent 做多步编排。Agentic 是演进方向，不是 v1 起点。

---

## 9. GraphRAG（多跳/全局问答）

- 微软开源，适合「跨文档主题总结」「实体关系梳理」。
- 流程：文档 → TextUnit → 实体/关系抽取 → Leiden 社区检测 → 社区摘要 → 检索时按社区/路径遍历。
- 在多跳问答任务中可比传统 RAG 准确率提升 ~72%（业界宣传数据，面试可引用并加保留意见）。

---

## 10. 多模态 RAG（PDF / 表格 / 图片）

### 10.1 难点
- PDF 双栏、表格、扫描件、公式、脚注、页眉页脚混杂。
- 简单 pdfplumber / PyPDF2 抽出的文本检索质量差。

### 10.2 工程方案
- **DeepDoc（RAGFlow）**：版面分析 + Table Transformer + OCR 三通道融合。
- **PaddleOCR PP-Structure**：表格 HTML 抽取。
- **MinerU / GROBID**：学术 PDF 结构化。
- **OpenCLIP / BGE-VL**：图文统一向量空间。
- **Unstructured / PyMuPDF**：轻量通用方案。
- 双库：文本向量库 + 图像向量库，分别粗筛 + 多模态精排。

### 10.3 评测
- Salesforce AI 70 万页多模态 RAG 评测（覆盖金融/法律/医疗/制造等 8 大领域）。
- 题目类型：事实检索、比较分析、总结归纳、逻辑推理。

---

## 11. Contextual Retrieval / Anthropic 方案（2024–2025 焦点）

- 为每个 chunk 用小 LLM 生成 50–100 token 的上下文说明，拼在 chunk 前。
- 配合 **Contextual Embeddings + Contextual BM25 + Reranking**，top-20 检索失败率下降 ~49%。
- 注意上下文生成的额外 LLM 成本与延迟，可批处理 + 缓存。

---

## 12. 性能、延迟、成本（生产硬指标）

### 12.1 延迟分解
- Embedding 计算（query 端通常几十 ms）。
- 检索 ANN（Milvus HNSW / IVF）。
- Rerank（Cross-Encoder 几十~几百 ms）。
- LLM 生成（首字 TTFT + 整体 token/s）。

### 12.2 优化手段
- **Embedding 缓存**：常见 query 与 chunk 嵌入缓存。
- **检索结果缓存**：query → top-k doc ids。
- **异步流水线**：检索 / rerank / LLM 流水线并行；Event IO / async pipeline，避免「向量库同步阻塞 LLM」。
- **预过滤（prefilter）**：用元数据先缩小候选集，再做向量检索。
- **量化、蒸馏**：embedding 模型 INT8 量化、Cross-Encoder 蒸馏到小模型。
- **模型分级**：粗筛用便宜模型，复杂问题路由到强模型。
- **流式响应**：SSE 改善用户体验。

### 12.3 成本控制
- Token 计费意识（prompt 中检索 chunk 的占比往往 > 70%）。
- 上下文压缩、稀疏注意力、prompt 模板瘦身。
- 缓存 + 去重 + 路由。

---

## 13. 可观测性、调试与运维

- **LangSmith / Langfuse / Phoenix / Arize**：全链路 trace、prompt 版本管理、A/B。
- 关注指标：
  - 检索：召回率、相关文档排名、Top-K 命中率。
  - 生成：faithfulness、answer relevance、TTFT、token 消耗。
  - 系统：QPS、P99 延迟、错误率、缓存命中率。
- **三阶可证伪测试框架**（2026 趋势）：检索 / 重排 / 生成三层各自可独立验证。
- **数据集管理**：回归测试集、bad case 沉淀、漂移检测。
- **监控告警**：质量分跌破阈值自动告警。

---

## 14. 安全、合规与权限

- **越权防护**：检索前基于用户身份/角色过滤文档范围；片段级 + 文档级双重权限。
- **Prompt Injection 防护**：输入侧 + 输出侧双向校验，识别「忽略上文」「泄漏 system prompt」等攻击。
- **数据隔离**：多租户隔离、跨租户访问控制、PII 检测与脱敏。
- **审计溯源**：每次回答都附引用与检索得分，可回溯到原始文档与版本。
- **红队测试**：对抗性输入、越权尝试、敏感词触发。
- **Citation 强约束**：每个事实必须挂引用源，便于合规审查。

---

## 15. RAG vs Fine-tuning vs Prompt Engineering

| 维度 | Prompt Eng | RAG | Fine-tuning |
|------|-----------|-----|-------------|
| 知识更新 | 弱 | 强（库可热更新） | 弱（要重新训练） |
| 可解释性 | 中 | 高（带引用） | 低 |
| 成本 | 低 | 中 | 高 |
| 适用场景 | 风格/角色 | 知识密集、实时 | 风格/指令/小数据 |
| 常见组合 | FT 改风格 + RAG 加知识 + Prompt 调行为 | | |

面试常考：「为什么不用微调解决？」——答出 RAG 在「知识新鲜度、可解释性、零样本适配」上的优势。

---

## 16. 经典面试题与答题骨架

### Q1. 解释 Naive RAG / Advanced RAG / Modular RAG 的区别。
答：抓住「索引优化、检索前/中/后处理、可插拔模块化」三点。

### Q2. RAG 系统评估的三个关键指标是什么？为什么单独评估检索和生成？
答：Context Relevance、Faithfulness、Answer Relevance。分开评估便于定位瓶颈（召回差 vs 答得差）。

### Q3. 为什么需要 Reranker？Cross-Encoder vs Bi-Encoder？
答：rerank 解决"找到但没排对"；Cross-Encoder 联合编码精度高，Bi-Encoder 独立编码可大规模 ANN。

### Q4. 怎么减少 RAG 的幻觉？
答：检索侧提升召回+相关度；生成侧引用增强 + 反幻觉 prompt + 低置信度拒答；评估侧 faithfulness 监控。

### Q5. Chunking 怎么做？为什么固定长度分块不够？
答：固定切分断语义；推荐结构感知 + 语义切分 + 父子分块；按文档类型选粒度。

### Q6. 混合检索为什么在生产中几乎是默认选项？
答：BM25 抓术语/编号/缩写，向量抓语义；RRF 融合后召回稳健，FAQ/法律/代码场景尤其受益。

### Q7. 怎么设计一个 RAG 系统的评估闭环？
答：构建评测集 → RAGAS 自动指标 + 人工 spot check → 漂移监控 → bad case 反哺分块/检索策略。

### Q8. RAG 系统在生产中怎么观测与调试？
答：LangSmith/Phoenix 全链路 trace；按检索/重排/生成分层埋点；用三阶可证伪测试。

### Q9. Agentic RAG 与传统 RAG 的本质区别？
答：决策权从「固定管线」交给「Agent」。Agent 决定要不要检索、用什么工具、是否改写；代价是复杂度+成本。

### Q10. 多模态 RAG 在生产中最大的坑是什么？
答：PDF 解析质量（双栏/表格/扫描件）、图文向量空间对齐、多模态 LLM 成本。

---

## 17. 工业级 RAG 架构样板

```
┌─────────────┐
│  Identity & │  用户身份 / 角色 / 权限
│  ACL 层     │
└──────┬──────┘
       ▼
┌─────────────┐
│ Query 改写  │  HyDE / Step-back / Multi-query / RAG-Fusion
└──────┬──────┘
       ▼
┌─────────────┐
│ 检索 Router │  向量 / BM25 / 图谱 / SQL / Web 搜索
└──────┬──────┘
       ▼
┌─────────────┐
│ Hybrid 召回 │  多路 → RRF / 加权融合
└──────┬──────┘
       ▼
┌─────────────┐
│  Reranker   │  Cross-Encoder / LLM Reranker
└──────┬──────┘
       ▼
┌─────────────┐
│ 压缩 / 摘要 │  Contextual Compression
└──────┬──────┘
       ▼
┌─────────────┐
│  LLM 生成   │  带引用 / 拒答策略 / 结构化输出
└──────┬──────┘
       ▼
┌─────────────┐
│  评估 & 监控│  RAGAS / LangSmith / 三阶可证伪测试
└─────────────┘
```

横向贯穿：缓存、权限过滤、审计日志、成本统计、漂移检测、bad case 沉淀。

---

## 18. 面试加分项清单

1. 能讲清**检索-重排-生成**三层各自的可证伪测试方法。
2. 能给出**具体的检索失败 case**并说明如何定位是「召回差」「排序差」还是「生成差」。
3. 熟悉**Contextual Retrieval、Parent-Document、Hybrid + RRF、HyDE** 等具体技术的数字收益。
4. 知道 **RAGAS 五大指标**的计算逻辑与局限。
5. 提到 **Agentic RAG 时强调演进路径**，而不是一上来就 AutoGPT。
6. 能聊**多模态/GraphRAG 的适用边界**——什么场景才值得上。
7. 有**权限 / 安全 / 审计**意识（合规场景特别加分）。
8. 提到**Token 经济性**：chunk 数量 vs 上下文长度 vs 召回质量。
9. 聊到**冷启动 + 评估闭环**才是 RAG 项目落地的真正难点。
10. 知道**RAG ≠ 万能**：和 fine-tuning、prompt 工程、工具调用是互补关系。

---

## 19. 一句话记忆口诀

> **Chunk 对、Embed 准、检索广、重排精、压缩狠、引用真、闭环跑、Agent 慎、权限紧、监控勤。**

---

## 20. LLM 幻觉（Hallucination）的检测与消除（面试高频专题）

> 调研来源：知乎《大模型幻觉问题调研》、SegmentFault《Lillian Weng 深度解读 LLM 幻觉》、Anthropic / OpenAI / SelfCheckGPT / CoVe 论文与中文技术博客、CSDN 黑盒 LLM 幻觉抑制 10 大方案综述、hallucination-leaderboard（Vectara HHEM-2.x）、LettuceDetect、AlphaEdit、MEMIT/ROME 系列。
>
> **关键结论：幻觉只能「缓解」，不能「根除」——这是所有方案的共同前提。**

---

### 20.1 幻觉的定义与分类

| 维度 | 类型 | 含义 | 示例 |
|------|------|------|------|
| **按与输入关系** | Context-Conflicting（上下文冲突） | 输出与给定 context 冲突 | 用户描述和朋友 Hill 打篮球，模型改成 Jack |
| | Input-Conflicting（输入冲突） | 输出未遵循用户指令 | 用户问「葡萄牙第三任国王的母亲是谁」，模型错答 |
| | Extrinsic（无中生有） | 凭空编造世界知识 / 引用文献 / 邮箱 / 数字 | 编造不存在的论文标题、人名、数字 |
| **按粒度** | Token 级 | 单 token 错误 | 拼写、数字位 |
| | 句子级 | 整句编造 | "The company grew by 3% in Q2 2023" |
| | 文档级 | 整段虚构内容 | 一篇完整但虚构的摘要 |
| **按可验证性** | Closed-domain | 客观事实可校验 | 历史事件 / 法律条文 |
| | Open-domain | 主观开放难以验证 | 创作 / 闲聊 |

面试要点：**生产系统关注的是 Faithfulness（是否忠实于 context）与 Factualness（是否符合事实）两个轴**。两者用的工具不同——前者靠 RAG + Citation，后者靠外部知识库 / 工具调用 / 自一致性。

---

### 20.2 幻觉产生的根源（高频面试题）

1. **概率采样机制**：自回归生成 + top-k/top-p 采样 → 低概率错误 token 被选中。
2. **训练数据局限**：知识覆盖不全 / 噪声 / 时效滞后。
3. **模型架构特性**：无显式记忆、长程注意力局限、缺乏生成后验证回路。
4. **数据压缩损失**：模型存储的是「关系概率」而非「原文内容」——类似 JPEG 压缩的保真度损失。
5. **SFT 教模型"幻觉"**：当 SFT 数据包含超出模型参数知识的「新知识」时，模型被迫外推，必然产生幻觉。
6. **解码策略**：高 temperature / top-p 会显著放大幻觉。
7. **知识边界模糊**：模型不知道「自己不知道」。

> 引文："LLMs 的能力越强、生成越流畅，用户越难分辨真伪，幻觉的潜在危害越大。" ——《大模型幻觉成因》

---

### 20.3 幻觉的检测（Detection）

#### 20.3.1 黑盒检测（仅看输出）

| 方法 | 思路 | 代表 |
|------|------|------|
| **SelfCheckGPT** | 同一 prompt 多次采样，对比一致性 | EMNLP 2023, 剑桥 |
| **Self-Consistency** | CoT 多路径投票 | Wang et al. 2022 |
| **Semantic Entropy** | 多次采样的语义聚类，簇数越多越可能幻觉 | Nature 2024 |
| **LLM-as-a-Judge** | 用强模型对弱模型打分 | GPT-4 judge |
| **RAGulator** | 训练轻量检测器识别"语义不符 (OOC)" 输出 | arXiv 2411.03920 |
| **LettuceDetect** | 基于 ModernBERT / EuroBERT 的 token 级幻觉检测 | KRLabsOrg |

#### 20.3.2 白盒检测（看模型内部状态）

- **Logit-level Uncertainty**：token 平均负对数概率 → 越高越可能幻觉。
- **Hidden State Probing**：从 hidden states 训练分类器预测幻觉（INSIDE、How to Steer LLM Latents）。
- **MIND**：基于 LLM 内部状态的无监督实时检测（ACL 2024 Findings）。
- **BTProp**：构建 belief tree 做概率传播，AUROC/AUC-PR 较 SOTA 提升 3–9%。

#### 20.3.3 检索增强验证（KALMV）

- **Knowledge-Augmented LM Verification**：训练独立验证器对「问题 + 检索知识 + 生成答案」分类：
  - retrieval error（检索到的不相关）
  - grounding error（生成未基于知识）
  - factual error（生成与事实冲突）
  - 其余标记为正确。

---

### 20.4 幻觉的缓解 / 消除（Mitigation）—— 按干预点分层

这是面试回答的核心框架。**任何方案都要先说明它落在哪个环节**，否则就是空谈。

#### 20.4.1 预训练阶段：Data Cleansing + 知识注入

- 数据清洗、去重、过滤噪声。
- 注入结构化知识（知识图谱）作为额外训练信号。
- 课程学习：分阶段训练提升知识吸收。

> 工业现实：多数公司不会从预训练改起，这一层了解即可。

#### 20.4.2 微调 / 对齐阶段：让模型"知道自己不知道"

| 方法 | 核心思路 | 面试要点 |
|------|----------|----------|
| **高质量 SFT** | 拒绝训练数据超出模型已有知识的样本 | Gekhman 等人发现 SFT 新知识会显著提升幻觉率 |
| **RLHF** | 用人类偏好训练 reward model + PPO | 抑制"胡说"倾向 |
| **DPO / IPO / KTO** | 直接偏好优化，无需 reward model | 工程更简单 |
| **Constitutional AI** | 用原则做 self-critique 生成偏好数据 | Anthropic 路线 |
| **Truthful SFT** | 专门用「真实数据集」微调 | TruthfulQA 类 |

**关键经验**：当用新知识 SFT 时，先评估模型是否已掌握；未掌握的部分改用 RAG 注入而不是 SFT。

#### 20.4.3 解码策略（Inference Decoding）

| 策略 | 作用 | 适用场景 |
|------|------|----------|
| **Greedy / Beam Search** | 选概率最高 token | 事实型问答 |
| **低 temperature（≈0）** | 减少随机性 | QA / 摘要 / 抽取 |
| **Top-K + Top-P** | 限制候选集 | 平衡质量与多样 |
| **Repetition Penalty** | 抑制复读 | 长生成 |
| **Constrained Decoding** | 用 JSON Schema / Grammar 强制结构 | 结构化输出 |
| **Contrastive Decoding** | 加大模型与弱模型分布差异 | 减少通用套话 |

> "事实性任务用 temperature=0 + repetition_penalty，创意任务再调高。"
> —— 面试常考的解码调参经验

#### 20.4.4 Prompt Engineering（成本最低、收益最快）

1. **Anti-Hallucination Prompt**
   - 「仅根据以下参考资料回答；如资料未涉及，请回答『我不知道』」
   - 「不要编造任何参考资料中未出现的信息」
   - 「为每个事实标注 [1] [2] 引用」
2. **Few-shot**：给出「正确引用 + 拒答」示例。
3. **Negative Prompt**：明确告诉模型「不要做 X」。
4. **CoT / Step-Back**：先抽象再回答。
5. **Citation-Enhanced Generation**：强制每个事实带引用，幻觉率可降 70%+。
6. **结构化输出**：要求 JSON Schema，配合 constrained decoding。

#### 20.4.5 RAG（见本文章节 2–7）

RAG 通过外部知识约束模型输出，是工业界抗幻觉的「第一道防线」。关键配置：
- 检索召回率 + 相关度 → faithfulness 的天花板。
- Citation 强制 → 让用户能反向验证。
- 低置信度拒答策略 → "答错不如不答"。

#### 20.4.6 Self-Verification / 自一致性（推理时增强）

| 技术 | 思路 | 代表 |
|------|------|------|
| **Self-Consistency** | 同一问题多路径生成 + 投票 | Wang et al. 2022 |
| **SelfCheckGPT** | 多采样对比不一致 → 视为幻觉 | EMNLP 2023 |
| **CoVe（Chain-of-Verification）** | 生成 → 计划验证问题 → 独立回答 → 修正 | Meta 2023 |
| **Chain-of-Thought** | 显式推理减少跳步 | Wei et al. 2022 |
| **Verification-guided CoT** | Zero-shot 让 LLM 自我打分引导推理 | arXiv 2501.13122 |
| **Self-Reflection** | 让模型对自身输出打分 + 修正 | Shinn et al. 2023 |
| **Lettuce / RAG Truth** | 训练专项 verifier 替代人工 | 开源模型 |

> CoVe 四步法口诀：**Draft → Plan Verifications → Answer Independently → Final Revised Response**。
> 在列表问答、闭卷 QA、长文生成上都能稳定降低幻觉率。

#### 20.4.7 知识编辑（Knowledge Editing）—— 精准修改模型内化的事实

> 适用：模型错误地"记住"了过时 / 错误的事实，需要定点修正而不重训。

| 方法 | 思路 | 局限 |
|------|------|------|
| **Knowledge Neurons** | 识别事实关联的神经元 → 微调激活 | 局限于简单事实 |
| **ROME** | Rank-One Model Editing，定位 MLP 中关键 layer 做闭式编辑 | 单点编辑 |
| **MEMIT** | ROME 的批量版，最小二乘约束 | 大批量编辑 |
| **MEND** | 用小网络学习"如何编辑" | 训练开销 |
| **EMMET** | ROME 的统一视角 + 等式约束批处理 | 2024 |
| **AlphaEdit** | 零空间约束编辑，保全新旧知识 | arXiv 2024 |
| **WISE** | 长期记忆模块 + 路由 | 抗遗忘 |
| **EasyEdit** | 浙大开源工具包，统一评测 reliability / generality / locality | GitHub |

> ⚠️ 关键面试点：**知识编辑对单跳问答有效，对多跳推理效果显著退化**（EMNLP 2025 CAKE）。在生产中，**模型内化 + RAG 外部化**是互补而非替代。

#### 20.4.8 不确定性估计 + 拒答策略

| 方法 | 思路 |
|------|------|
| **Token Logprobs** | 平均对数概率阈值；越低越不确定 |
| **Entropy** | 输出分布的熵 |
| **Semantic Uncertainty** | 多次采样语义聚类熵 |
| **Conformal Prediction** | 用统计保证做覆盖率约束 |
| **QuCo-RAG** | 从预训练语料统计量估计动态检索阈值 |

工业落地：**低置信度拒答（"我不知道"）比硬答更安全**。配合兜底话术 + 转人工，是高风险场景（医疗、法律、金融）的标配。

#### 20.4.9 工具调用 + 结构化输出

- **Tool Use**：模型调用计算器、SQL、API 来"算"而不是"猜"。
- **代码生成 / ReAct**：把推理外包给代码执行。
- **Structured Output（JSON Schema）**：避免格式幻觉。
- **Constrained Decoding**：用 grammar（CFGs / JSON schema）限制 token 候选集。

#### 20.4.10 系统级兜底（生产工程视角）

- **Caching**：相同 query 不再生成。
- **Source Attribution Dashboard**：用户可点击每个事实看来源。
- **A/B + 影子模式**：新模型先做对比再放量。
- **Bad Case 闭环**：用户反馈 → 评测集 → 训练数据。
- **多模型 Ensemble / 路由**：高风险问题路由到更强模型。

---

### 20.5 黑盒 LLM（仅 API 调用）的 10 大幻觉抑制方案（实战）

> 当你只能调用 GPT/Claude 的 API，没有 logits / hidden states 时的「兜底套路」：

1. **Self-Consistency 投票**（SelfCheckGPT 思路）：temperature ≥ 0.7 采样 N=3~5 次，取一致内容。
2. **CoT + 显式推理步骤**：让模型先列步骤再给结论。
3. **Anti-Hallucination Prompt + 引用约束**：强制每事实挂引用。
4. **RAG + 低置信度拒答**：检索不到就拒答。
5. **外部工具校验**：用代码 / SQL / API 验证关键事实。
6. **Few-shot 真实示例**：在 prompt 里塞 3–5 个正确范例。
7. **Domain Knowledge Prompt**：把领域规则直接写在 system prompt 里。
8. **Self-Verification CoVe**：让模型"自己审自己"。
9. **Decompose + Verify**：复杂问题拆解为子问题，分别验证后汇总。
10. **多模型交叉验证（Self-Consistency 升级版）**：GPT 答 + Claude 答 → 比较 → 冲突由第三个裁判判定。

---

### 20.6 行业现状：hallucination-leaderboard（2026）

- Vectara HHEM-2.3 评估：LLM 在文档摘要中的幻觉率分布在 **1.8% – 24%** 之间。
- **Top 5**（2026年1月）：AntGroup Finix-S1-32B（0.6%）、Google Gemini 系列、OpenAI 多款。
- **趋势**：模型规模 ≠ 幻觉率，关键在训练数据质量 + RLHF/DPO 对齐 + 推理时约束。

> 面试可以引："2026 年最强模型摘要幻觉率仍约 1.8%，工业系统需做多级防护而非依赖单模型。"

---

### 20.7 经典面试题

**Q1. LLM 为什么会产生幻觉？**
答：概率生成 + 训练数据压缩 + 解码采样 + 模型不知道自己不知道，四大根源。

**Q2. 如何在不改模型权重的情况下减少幻觉？**
答：RAG + 引用增强 + Anti-Hallucination Prompt + Self-Verification + 工具调用 + 拒答策略，多层叠加。

**Q3. SelfCheckGPT 的核心原理？**
答：模型对同一问题多次采样，真实事实多次结果一致，幻觉结果多次矛盾。一致性低 → 标记幻觉。

**Q4. CoVe 四步法是什么？**
答：Draft（起草）→ Plan Verification Questions（生成验证问题）→ Answer Independently（独立回答避免相互干扰）→ Final Revised Response（修正输出）。

**Q5. 知识编辑（ROME/MEMIT）能否根治幻觉？**
答：不能根治。单跳事实可精确修改，但多跳推理会显著退化；与 RAG 是互补关系。

**Q6. RAG 和 SFT 哪种更能减幻觉？**
答：RAG 更适合动态 / 外部知识，SFT 教风格与指令；用 SFT 学新事实反而会诱发幻觉（Gekhman 等）。两者协同最佳。

**Q7. 如何量化系统幻觉率？**
答：Faithfulness（RAGAS）、HHEM（Vectara）、LettuceDetect（ModernBERT）、LLM-as-Judge。生产中组合使用 + 人工抽检。

**Q8. LLM 输出 logprobs 在生产中怎么用？**
答：作为不确定性信号；阈值以下触发拒答或转人工；监控整体分布漂移。

**Q9. 如果检索召回很好但模型还在幻觉，怎么办？**
答：定位在生成层——加 citation 约束 + 反幻觉 prompt + CoVe 自检 + 限缩 temperature + 必要时切更强模型。

**Q10. 生产环境的"幻觉防御深度"应该怎么搭？**
答：四层防御——
1. 检索层（提升召回 + 引用）
2. 提示层（约束 + 结构化）
3. 验证层（Self-Consistency / CoVe / verifier 模型）
4. 系统层（拒答、转人工、审计）

---

### 20.8 一句话记忆口诀

> **数据清洗要彻底，对齐训练防幻觉；解码低温 + 约束，Prompt 引用是底线；
> 检索召回是天花板，自洽验证多模型；知识编辑补事实，工具调用兜兜转；
> 不确定就拒答，系统兜底保平安。**

---

## 附录：参考资料索引

- 腾讯云《2026 RAG 系统测试实战新趋势》
- 腾讯云《RAG 系列 04 — Agentic RAG》
- 腾讯云《RAG 技术进阶：GraphRAG + 私有数据》
- 蒸汽求职《IBM AI 岗 Offer 复盘：RAG 系统从设计到生产的工程化全流程》
- SegmentFault《Advanced RAG 06: 生成结果的相关性低? 快用 Query Rewriting》
- SegmentFault《万字好文：OpenAI 前安全系统负责人 Lillian Weng 深度解读 LLM 幻觉》
- 知乎《大模型的幻觉问题调研：LLM Hallucination Survey》
- CSDN《LLM 幻觉（Hallucination）缓解技术综述与展望》
- CSDN《A Comprehensive Survey of Hallucination Mitigation Techniques in LLMs》
- CSDN《黑盒 LLM 幻觉抑制：10 大落地方案全解析》
- CSDN《基于验证链 (Chain of Verification) 的大语言模型幻觉问题解决方案》
- CSDN《大模型知识编辑》EasyEdit 教程
- CSDN《RAGulator: 如何识别和缓解大模型所谓的"忠实幻觉"》
- 阿里 RAGulator / LettuceDetect / BTProp / INSIDE / QuCo-RAG 论文
- hallucination-leaderboard（Vectara HHEM-2.x）
- Anthropic Contextual Retrieval（开源实现）
- Salesforce AI 70 万页多模态 RAG 评测
- SelfCheckGPT (EMNLP 2023)、Chain-of-Verification (Meta 2023)
- GitHub: KRLabsOrg/LettuceDetect, EdinburghNLP/awesome-hallucination-detection, kannans/RAG_Techniques, avnlp/vectordb, zjunlp/EasyEdit, SuperBruceJia/Awesome-LLM-Self-Consistency