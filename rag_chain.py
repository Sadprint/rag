"""
基于 LangChain 的链式 RAG 系统
文档加载 → 分块 → 向量化 → 检索 → 生成
"""
import os
import logging
import warnings

# langchain_chroma 在 Windows 上会导致 segfault，暂用 langchain_community 版本
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import config

# 关闭各种进度条和冗余日志
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================
# 1. 嵌入模型 (本地 BGE)
# ============================================================
def _resolve_device():
    """解析嵌入设备"""
    device = config.EMBEDDING_DEVICE
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_embedding_model():
    """加载本地 BGE 嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": _resolve_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# 2. 文档加载
# ============================================================
def load_documents(docs_dir: str = None):
    """从目录加载 .txt / .md / .json 文件"""
    if docs_dir is None:
        docs_dir = str(config.DOCS_DIR)

    txt_loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )
    md_loader = DirectoryLoader(
        docs_dir, glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}, show_progress=False,
    )

    docs = []
    docs.extend(txt_loader.load())
    docs.extend(md_loader.load())

    return docs



# ============================================================
# 3. 文档分块
# ============================================================
def split_documents(docs, chunk_size=None, chunk_overlap=None):
    """使用递归字符分割器分块，中文友好"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        add_start_index=True,  # 记录块在原文中的位置，用于来源引用
    )
    return splitter.split_documents(docs)


# ============================================================
# 4. 向量库 (ChromaDB)
# ============================================================
def create_vectorstore(chunks, embeddings, persist_dir=None):
    """从文档块创建持久化向量库"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir or config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )


def load_vectorstore(embeddings, persist_dir=None):
    """加载已有向量库"""
    return Chroma(
        persist_directory=persist_dir or config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )


def get_retriever(vectorstore, search_type=None, k=None, fetch_k=None, lambda_mult=None):
    """获取检索器，支持 similarity 和 mmr 两种策略"""
    search_kwargs = {"k": k or config.DEFAULT_K}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k or config.MMR_FETCH_K
        search_kwargs["lambda_mult"] = lambda_mult or config.MMR_LAMBDA_MULT

    return vectorstore.as_retriever(
        search_type=search_type or config.DEFAULT_SEARCH_TYPE,
        search_kwargs=search_kwargs,
    )


# ============================================================
# 5. LLM
# ============================================================
def load_llm():
    """加载智谱 GLM LLM"""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.ZHIPUAI_API_KEY,
        base_url=config.ZHIPUAI_BASE_URL,
        temperature=0.3,
    )


# ============================================================
# 6. 问题改写 (Query Rewriting)
# ============================================================
def rewrite_query(user_question, chat_history=None):
    """将用户问题改写为适合检索的独立查询。

    处理两类问题:
    1. 问句形式转换（始终生效）
       "你知道心肌酶组套吗？" → "心肌酶组套 定义 介绍"
       "能不能告诉我什么是RAG？" → "RAG 是什么"
    2. 指代消解（存在对话历史时）
       "它为什么会这样？" → "RAG 系统召回率低的原因是什么"

    改写失败时返回原始问题。
    """
    history = chat_history or []

    # 格式化历史为对话文本（存在时）
    history_section = ""
    if history:
        history_text = ""
        for m in history[-6:]:
            role = "用户" if m["role"] == "user" else "助手"
            history_text += f"{role}: {m['content']}\n"
        history_section = f"""对话历史：
{history_text}"""

    # 有历史时同时做指代消解 + 问句标准化，无历史时只做问句标准化
    base_rules = [
        '将疑问句式转换为保留查询意图的关键词（如「你知道X吗」→「X 介绍 解释」、「X是什么意思」→「X 定义」）',
        '移除口语化填充词（「你知道」「请问」「能不能告诉我」「吗」「呢」「吧」等）',
        '保留核心检索语义，去除礼貌用语和寒暄',
    ]
    anaphora_rules = [
        '将指代词（「它」「这个」「那个」「这样」「这些」）替换为对话历史中提到的具体内容',
        '补全省略的主语、宾语和上下文信息',
    ]

    rules = base_rules + (anaphora_rules if history else base_rules)
    rules_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))

    rewrite_prompt = f"""你的任务是将用户问题改写为适合向量检索的关键词查询。

规则：
{rules_text}
{len(rules) + 1}. 只输出改写后的查询文本，不要加引号或任何解释

{history_section}
用户问题：{user_question}

改写后的查询："""

    try:
        llm = load_llm()
        rewritten = llm.invoke(rewrite_prompt).content.strip()
        return rewritten if rewritten else user_question
    except Exception:
        return user_question


# ============================================================
# 7. 提示词模板
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是知识库助手，必须基于提供的文档内容回答问题。

规则：
1. 每个回答都要有文档依据，从文档中提取关键事实来组织答案
2. 禁止编造、猜测或补充文档中没有的信息
3. 如果文档中完全没有相关信息，回复"文档中未找到相关内容"
4. 回答简洁清晰，重点突出，以用户能理解的方式呈现

以下是文档中检索到的相关内容：

{context}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])


# ============================================================
# 8. 获取/构建向量库 (一键)
# ============================================================
def get_or_create_vectorstore(force_rebuild=False):
    """获取向量库，不存在则构建"""
    embeddings = load_embedding_model()

    if not force_rebuild:
        import os
        if os.path.exists(config.CHROMA_PERSIST_DIR):
            try:
                return load_vectorstore(embeddings)
            except Exception:
                pass  # 加载失败就重建

    docs = load_documents()
    if not docs:
        return None
    chunks = split_documents(docs)
    return create_vectorstore(chunks, embeddings)


# ============================================================
# 9. RAG 链 (LCEL)
# ============================================================
def build_rag_chain(retriever=None, vectorstore=None, search_type=None, k=None, chat_history=None, query=None):
    """构建 LCEL RAG 链。

    chat_history: 最近几轮对话列表 [{'role':'user'/'assistant', 'content':...}, ...]
    query: 改写后的检索查询。提供时用于检索，但 LLM 仍看到原始用户问题。
          不提供时检索和生成使用同一个输入。
    """
    if retriever is None and vectorstore is not None:
        retriever = get_retriever(vectorstore, search_type=search_type, k=k)

    # 如果提供了改写查询，用它做检索，而非原始用户输入
    if query is not None:
        _ret = retriever
        retriever = RunnableLambda(lambda _: _ret.invoke(query))

    llm = load_llm()
    history = chat_history or []

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知")
            start = doc.metadata.get("start_index", "?")
            formatted.append(f"[文档{i}] 来源: {source}, 位置: {start}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: [
                ("human" if m["role"] == "user" else "ai", m["content"])
                for m in history
            ],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ============================================================
# 10. T2Ranking 数据加载
# ============================================================
def load_t2ranking_passages(docs_dir=None):
    """加载 T2Ranking 数据集 passage，返回 List[Document]。
    每个 passage 作为独立文档（不分块），元数据包含 pid 用于评估。"""
    import json as _json

    if docs_dir is None:
        docs_dir = str(config.DOCS_DIR)

    passages_path = os.path.join(docs_dir, "t2ranking", "passages.json")
    if not os.path.exists(passages_path):
        return []

    with open(passages_path, "r", encoding="utf-8") as f:
        raw = _json.load(f)

    docs = []
    for item in raw:
        docs.append(Document(
            page_content=item["text"],
            metadata={
                "source": "t2ranking",
                "pid": item["pid"],
            },
        ))
    return docs


def load_t2ranking_queries(docs_dir=None):
    """加载 T2Ranking 评估查询，返回 [{"qid":..., "query":..., "relevant_pids":[...]}, ...]"""
    import json as _json

    if docs_dir is None:
        docs_dir = str(config.DOCS_DIR)

    queries_path = os.path.join(docs_dir, "t2ranking", "eval_queries.json")
    if not os.path.exists(queries_path):
        return []

    with open(queries_path, "r", encoding="utf-8") as f:
        return _json.load(f)


def create_t2ranking_vectorstore(embeddings, docs_dir=None, batch_size=500,
                                progress_callback=None):
    """从 T2Ranking passages 创建独立向量库（不分割，每条 passage 一个向量）。

    分批嵌入以减少中断风险，进度通过 progress_callback 回调。
    """
    passages = load_t2ranking_passages(docs_dir)
    if not passages:
        raise ValueError("docs/t2ranking/passages.json 不存在，请先下载数据集")

    persist_dir = os.path.join(
        docs_dir or str(config.DOCS_DIR), "t2ranking", "chroma_db"
    )

    total = len(passages)
    vs = None
    for i in range(0, total, batch_size):
        batch = passages[i : i + batch_size]
        if vs is None:
            vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="t2ranking_eval",
            )
        else:
            vs.add_documents(batch)
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
    return vs


def load_t2ranking_vectorstore(embeddings, docs_dir=None):
    """加载已有的 T2Ranking 向量库，空库返回 None"""
    persist_dir = os.path.join(
        docs_dir or str(config.DOCS_DIR), "t2ranking", "chroma_db"
    )
    if not os.path.exists(persist_dir):
        return None
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path=persist_dir)
        collections = client.list_collections()
        for c in collections:
            if c.name == "t2ranking_eval" and c.count() > 0:
                break
        else:
            return None  # 集合不存在或为空
    except Exception:
        pass  # 加载失败，就当不存在

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="t2ranking_eval",
    )


# ============================================================
# 11. 检索 + 来源返回
# ============================================================
def retrieve_with_sources(vectorstore, query, search_type=None, k=None):
    """检索并返回带来源的文档列表，供前端展示"""
    retriever = get_retriever(vectorstore, search_type=search_type, k=k)
    docs = retriever.invoke(query)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "未知"),
            "start_index": doc.metadata.get("start_index", "?"),
        }
        for doc in docs
    ]


def run_rag_query(vectorstore, question, search_type=None, k=None):
    """执行一次 RAG 查询，返回 (answer, contexts)。
    供评估脚本复用，避免重复拼接链逻辑。"""
    chain = build_rag_chain(
        vectorstore=vectorstore,
        search_type=search_type or config.DEFAULT_SEARCH_TYPE,
        k=k or config.DEFAULT_K,
    )
    answer = chain.invoke(question)
    sources = retrieve_with_sources(vectorstore, question, search_type=search_type, k=k)
    contexts = [s["content"] for s in sources]
    return answer, contexts
