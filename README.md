# NoteAgent — 本地文档智能问答系统

基于 **RAG（检索增强生成）** 的本地文档问答系统。上传文档 → 向量化 → 自然语言提问 → 带来源引用的回答。

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Chainlit |
| 框架 | LangChain (LCEL) |
| 向量库 | ChromaDB (本地持久化) |
| 嵌入模型 | BAAI/bge-base-zh-v1.5 (本地 GPU 推理) |
| LLM | 智谱 GLM-4-Flash |
| 对话存储 | SQLite |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载嵌入模型
huggingface-cli download BAAI/bge-base-zh-v1.5 --local-dir model/BAAI_bge-base-zh-v1.5

# 3. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入：ZHIPUAI_API_KEY=你的密钥

# 4. 放入文档到 docs/ 目录（支持 .txt / .md / .json）

# 5. 启动
chainlit run app.py
```

## 使用流程

### 对话问答

1. 点击 **构建向量库** 将文档向量化（只需构建一次，后续自动加载）
2. 在对话框输入问题，回答会附带来源引用显示在侧边栏
3. 右侧 ⚙️ 可调整检索策略 (MMR/相似度)、返回条数、分块大小等

### 检索评估

独立的 CLI 脚本，基于 T2Ranking 数据集评估检索质量：

```bash
python evaluate.py                   # 交互模式（推荐）
python evaluate.py --build           # 重建向量库 + 全量评估
python evaluate.py --passages 10000  # 限制段落数
python evaluate.py --max-items 500   # 限制查询数
python evaluate.py --output result.csv  # 导出 CSV/JSON
```

指标：**Recall@k / MRR / NDCG**

### 生成评估

三种方式，按需选用：

```bash
# 方式1（推荐）：收集一次 + 反复评估
python collect_samples.py --max-items 200 --output eval_samples.json
python evaluate_metrics.py --input eval_samples.json --output result_v4.csv
python evaluate_metrics.py --metrics faith,answer_rel ...    # 换指标组合，秒出结果

# 方式2：一键跑完（收集 + 评估）
python collect_samples.py --max-items 200 --eval --eval-output result.csv

# 方式3：旧版脚本（v1-v3 评估用此脚本产出，保留作为历史参考）
python evaluate_generation.py --max-items 200 --output result.csv
```

指标：**Faithfulness / Context Precision / Context Utilization**

## 项目结构

```
NoteAgent/
├── app.py                  # Chainlit 前端入口
├── rag_chain.py            # RAG 核心引擎（嵌入/分块/检索/LLM）
├── database.py             # SQLite 对话持久化
├── config.py               # 全局配置
├── evaluate.py             # 检索评估 CLI
├── evaluate_generation.py  # 生成评估 CLI（旧版，v1-v3 由它产出）
├── collect_samples.py      # 生成评估 — 样本收集
├── evaluate_metrics.py     # 生成评估 — 指标评估 (RAGAS)
├── results/                # 评估结果输出
│   ├── eval_result.csv         # 检索评估结果
│   ├── result_v1.csv ~ v4.csv  # 生成评估结果
│   └── eval_samples.json       # 缓存的评估样本
├── docs/
│   └── t2ranking/          # T2Ranking 评估数据集
├── model/                  # 本地嵌入模型
├── chroma_db/              # 文档向量库（自动生成）
├── data/                   # 对话数据库（自动生成）
└── requirements.txt
```

## 评估结果

### 检索评估（T2Ranking，22,812 条查询，k=10）

| 指标 | 值 |
|------|-----|
| Recall@10 | 0.9464 |
| MRR | 0.8983 |
| NDCG@10 | 0.8133 |

### 生成评估（RAGAS，200 条查询，16 并发，GLM-4-Flash 评判）

#### 评估方法

不需要人工标注。用 T2Ranking 的 200 条查询，每条先走完整 RAG 链（检索 → 生成），然后把 **(问题, 回答, 检索文档)** 三元组交给 RAGAS，由 LLM 自动打分，覆盖三个维度：

- **Faithfulness（忠实度）**：将回答拆解为原子陈述，逐条检验是否能在检索文档中找到依据。衡量"回答有没有编造文档里不存在的东西"（防幻觉）。方向：回答 → 检索文档
- **Context Precision（上下文精度）**：利用 T2Ranking 标注的相关文档，检验相关文档是否排在检索结果前列。方向：问题 → 检索文档（排序质量）
- **Context Utilization（上下文利用率）**：检验检索文档中有多少信息被回答所使用。衡量"检索到的文档有没有被浪费"。方向：检索文档 → 回答

#### 优化过程

| 轮次 | 配置 | Faithfulness | Context Utilization | 分析 |
|------|------|:---:|:---:|------|
| v1 基准 | k=4, 原 prompt | 0.8573 | 0.6981 | 基础效果，但 14% 的回答仍有编造 |
| v2 尝试 | k=4, 加强约束 | 0.6237 | 0.6195 | prompt 过于严厉，回答被压缩成一句话，NLI 模型难以分解验证 |
| v3 改进 | k=3, 积极引导 | 0.8436 | 0.8054 | prompt 从"禁止"改为"引导"，利用率提升 15% |
| **v4 完备** | **k=3, 三指标** | **0.8432** | **0.7923** | 增加 Context Precision，完整覆盖检索+生成评估 |

#### v4 最终结果

200 条查询，3 个指标，16 并发：

| 指标 | 值 | 说明 |
|------|:---:|------|
| Faithfulness | 0.8432 | 84% 的回答内容可在检索文档中找到依据 |
| Context Precision | 0.8765 | 88% 的相关文档排在检索结果前列 |
| Context Utilization | 0.7923 | 79% 的检索文档被答案有效利用 |

**结论**：系统忠实度保持在 0.84 水平，文档排序精准（0.88），利用率也维持在 0.79。完整的 RAG 三角形评估覆盖了检索排序、内容利用和幻觉检测。


## 配置

`.env`:

```bash
ZHIPUAI_API_KEY=你的密钥    # 智谱 API 密钥
EMBEDDING_DEVICE=auto       # auto=自动检测 CUDA，可手动指定 cuda/cpu
```

更多参数见 [config.py](config.py)：分块大小、检索策略、MMR 权重等。
