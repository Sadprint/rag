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

## 项目结构

```
NoteAgent/
├── app.py              # Chainlit 前端入口
├── rag_chain.py         # RAG 核心引擎（嵌入/分块/检索/LLM）
├── database.py          # SQLite 对话持久化
├── config.py            # 全局配置
├── evaluate.py          # 检索评估 CLI
├── docs/
│   ├── t2ranking/       # T2Ranking 评估数据集
│   └── cmrc2018/        # CMRC 2018 数据集
├── model/               # 本地嵌入模型
├── chroma_db/           # 文档向量库（自动生成）
├── data/                # 对话数据库（自动生成）
└── requirements.txt
```

## 评估结果

==========================================================
  T2Ranking 检索评估结果
==========================================================
  查询数:       22,812
  k:            10
  命中 / 未命中: 21,589 / 1,223
  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─  ─
  Recall@10:     0.9464
  MRR:           0.8983
  NDCG@10:       0.8133
==========================================================

## 配置

`.env`:

```bash
ZHIPUAI_API_KEY=你的密钥    # 智谱 API 密钥
EMBEDDING_DEVICE=auto       # auto=自动检测 CUDA，可手动指定 cuda/cpu
```

更多参数见 [config.py](config.py)：分块大小、检索策略、MMR 权重等。
