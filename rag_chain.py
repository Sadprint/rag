"""
基于 LangChain 的链式 RAG 系统
文档加载 → 分块 → 向量化 → 检索 → 生成
"""
import os
import json
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
from langchain_core.runnables import RunnablePassthrough
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

    # JSON (CMRC 格式)
    json_docs, _ = load_cmrc_json(docs_dir)
    docs.extend(json_docs)

    return docs


def load_cmrc_json(docs_dir: str = None):
    """加载 CMRC 2018 JSON 数据集。
    返回 (documents, qa_pairs)
    - documents: 每条 context_text → Document (metadata: title, context_id, source)
    - qa_pairs: [{"question":..., "answers":[...], "context_id":...}, ...]
    """
    import glob as _glob

    if docs_dir is None:
        docs_dir = str(config.DOCS_DIR)

    target = os.path.join(docs_dir, "**", "*.json")
    json_files = _glob.glob(target, recursive=True)

    documents = []
    qa_pairs = []

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 兼容 {"data": [...]} 和 [...] 两种外层
        if isinstance(raw, dict):
            entries = raw.get("data", [])
        elif isinstance(raw, list):
            entries = raw
        else:
            continue

        for entry in entries:
            context_id = entry.get("context_id", "")
            context_text = entry.get("context_text", "")
            title = entry.get("title", "")
            source = os.path.basename(filepath)

            if not context_text:
                continue

            # 构建 Document
            doc = Document(
                page_content=context_text,
                metadata={
                    "source": source,
                    "context_id": context_id,
                    "title": title,
                },
            )
            documents.append(doc)

            # 提取 QA 对
            for qa in entry.get("qas", []):
                qa_pairs.append({
                    "question": qa.get("query_text", ""),
                    "answers": qa.get("answers", []),
                    "context_id": context_id,
                })

    return documents, qa_pairs


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
# 6. 提示词模板
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是知识库助手，基于提供的文档内容回答问题。

规则：
1. 使用文档中的信息回答，禁止编造
2. 如果文档中没有相关信息，直接说"文档中未找到相关内容"
3. 回答简洁清晰，不要啰嗦

以下是文档中检索到的相关内容：

{context}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])


# ============================================================
# 7. 获取/构建向量库 (一键)
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
# 8. RAG 链 (LCEL)
# ============================================================
def build_rag_chain(retriever=None, vectorstore=None, search_type=None, k=None, chat_history=None):
    """构建 LCEL RAG 链，chat_history 为最近几轮对话列表 [{'role':'user'/'assistant', 'content':...}, ...]"""
    if retriever is None and vectorstore is not None:
        retriever = get_retriever(vectorstore, search_type=search_type, k=k)

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
# 9. T2Ranking 数据加载
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
# 10. 检索 + 来源返回
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
