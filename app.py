"""
本地文档智能问答系统 — Chainlit 前端
基于 LangChain + ChromaDB + BGE + 智谱GLM 的链式 RAG
"""
import sys as _sys
import types as _types
import importlib.machinery as _machinery

# ── onnxruntime mock ──
class _FakeORT(_types.ModuleType):
    def __init__(self, name): super().__init__(name)
    @staticmethod
    def get_available_providers(): return ["CPUExecutionProvider"]
    @staticmethod
    def InferenceSession(*a, **kw): return _types.SimpleNamespace()

def _register(name):
    mod = _FakeORT(name) if name == "onnxruntime" else _types.ModuleType(name)
    mod.__spec__ = _machinery.ModuleSpec(name, loader=None, origin="mock")
    mod.__package__ = name.rpartition(".")[0] or ""
    _sys.modules[name] = mod

_register("onnxruntime")
_register("onnxruntime.capi")
_register("onnxruntime.capi._pybind_state")

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

import chainlit as cl
from pathlib import Path

from rag_chain import (
    load_embedding_model,
    load_documents,
    split_documents,
    create_vectorstore,
    load_vectorstore,
    load_t2ranking_vectorstore,
    build_rag_chain,
    retrieve_with_sources,
    rewrite_query,
)
from database import (
    init_db,
    create_conversation,
    get_conversations,
    update_conversation_title,
    delete_conversation,
    save_message,
    get_messages,
)
import config

init_db()


# ==================== Action Callbacks ====================

@cl.action_callback("build_vs")
async def on_build_vs(action: cl.Action):
    settings = cl.user_session.get("settings")
    msg = cl.Message(content="⏳ 正在读取文档并构建向量库...")
    await msg.send()
    try:
        embeddings = cl.user_session.get("embeddings")
        docs = load_documents()
        if not docs:
            msg.content = "❌ docs 目录下没有找到文档"
            await msg.update()
            return
        chunks = split_documents(docs, settings["chunk_size"], settings["chunk_overlap"])
        vs = create_vectorstore(chunks, embeddings)
        cl.user_session.set("vectorstore", vs)
        vs_source = f"本地文档 ({vs._collection.count():,} 条)"
        cl.user_session.set("vs_source", vs_source)
        msg.content = f"✅ 构建完成！{len(docs)} 个文档 → {len(chunks)} 个向量块"
        await msg.update()
    except Exception as e:
        msg.content = f"❌ 构建失败：{e}"
        await msg.update()


@cl.action_callback("load_vs")
async def on_load_vs(action: cl.Action):
    msg = cl.Message(content="⏳ 正在加载向量库...")
    await msg.send()
    try:
        embeddings = cl.user_session.get("embeddings")
        vs = load_vectorstore(embeddings)
        cl.user_session.set("vectorstore", vs)
        vs_source = f"本地文档 ({vs._collection.count():,} 条)"
        cl.user_session.set("vs_source", vs_source)
        msg.content = "✅ 向量库加载完成"
        await msg.update()
    except Exception as e:
        msg.content = f"❌ 加载失败：{e}"
        await msg.update()


@cl.action_callback("new_conv")
async def on_new_conv(action: cl.Action):
    conv_id = create_conversation()
    cl.user_session.set("conversation_id", conv_id)
    cl.user_session.set("history", [])
    await cl.Message(content="✅ 新对话已创建，开始提问吧").send()


@cl.action_callback("switch_conv")
async def on_switch_conv(action: cl.Action):
    cid = action.payload["cid"]
    cl.user_session.set("conversation_id", cid)
    db_msgs = get_messages(cid)
    cl.user_session.set("history", [
        {"role": m["role"], "content": m["content"], "sources": m.get("sources")}
        for m in db_msgs
    ])
    await cl.Message(content=f"✅ 已切换到对话 #{cid}（{len(db_msgs)} 条历史消息）").send()


@cl.action_callback("delete_conv")
async def on_delete_conv(action: cl.Action):
    cid = action.payload["cid"]
    delete_conversation(cid)
    current = cl.user_session.get("conversation_id")
    if current == cid:
        new_id = create_conversation()
        cl.user_session.set("conversation_id", new_id)
        cl.user_session.set("history", [])
    await cl.Message(content=f"✅ 对话 #{cid} 已删除").send()


@cl.action_callback("manage_conv")
async def on_manage_conv(action: cl.Action):
    conversations = get_conversations()
    current_id = cl.user_session.get("conversation_id")

    actions = []
    for conv in conversations:
        cid = conv["id"]
        title = conv["title"] or "新对话"
        marker = " [当前]" if cid == current_id else ""

        actions.append(cl.Action(
            name="switch_conv",
            payload={"cid": cid},
            label=f"{title}{marker}",
            description=conv.get("updated_at", "")[:16],
        ))
        actions.append(cl.Action(
            name="delete_conv",
            payload={"cid": cid},
            label=f"🗑 删除「{title[:10]}」",
            description="删除对话及其所有消息",
        ))

    actions.append(cl.Action(
        name="new_conv",
        payload={},
        label="➕ 新建对话",
    ))

    await cl.Message(
        content="**💬 对话管理**\n点击切换对话、删除或新建：",
        actions=actions,
    ).send()


# ==================== Chat Lifecycle ====================

@cl.on_chat_start
async def start():
    msg = cl.Message(content="⏳ 正在加载嵌入模型...")
    await msg.send()

    embeddings = load_embedding_model()
    cl.user_session.set("embeddings", embeddings)

    # 优先加载 T2Ranking 向量库（123,354 篇中文段落），其次加载本地文档库
    vs = None
    vs_source = ""
    vs = load_t2ranking_vectorstore(embeddings)
    if vs is not None:
        vs_source = f"T2Ranking ({vs._collection.count():,} 篇段落)"
    else:
        try:
            vs = load_vectorstore(embeddings)
            if vs is not None:
                vs_source = f"本地文档 ({vs._collection.count():,} 条)"
        except Exception:
            pass
    cl.user_session.set("vectorstore", vs)
    cl.user_session.set("vs_source", vs_source)

    # 配置面板
    await cl.ChatSettings([
        cl.input_widget.Select(
            id="search_type", label="检索策略",
            values=["mmr", "similarity"],
            initial_value="mmr",
        ),
        cl.input_widget.Slider(
            id="k", label="返回条数",
            initial=config.DEFAULT_K, min=1, max=10, step=1,
        ),
        cl.input_widget.Slider(
            id="mmr_lambda", label="多样性权重 (仅 MMR 生效)",
            initial=config.MMR_LAMBDA_MULT, min=0.0, max=1.0, step=0.05,
        ),
        cl.input_widget.Slider(
            id="chunk_size", label="分块大小",
            initial=config.CHUNK_SIZE, min=100, max=2000, step=50,
        ),
        cl.input_widget.Slider(
            id="chunk_overlap", label="块间重叠",
            initial=config.CHUNK_OVERLAP, min=0, max=500, step=10,
        ),
    ]).send()

    cl.user_session.set("settings", {
        "search_type": "mmr",
        "k": config.DEFAULT_K,
        "mmr_lambda": config.MMR_LAMBDA_MULT,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    })

    # 初始化对话
    conversations = get_conversations()
    if not conversations:
        create_conversation("新对话")
        conversations = get_conversations()

    cl.user_session.set("conversation_id", conversations[0]["id"])
    cl.user_session.set("history", [])

    msg.content = "✅ 嵌入模型就绪"
    await msg.update()

    # 欢迎消息
    vs_status = f"✅ {vs_source}" if vs is not None else "⚠️ 未加载，请点击按钮"
    await cl.Message(
        content=f"""# 📚 本地文档智能问答

| 组件 | 状态 |
|------|------|
| 向量库 | {vs_status} |
| LLM | 智谱 GLM-4-Flash |
| 嵌入模型 | BAAI/bge-base-zh-v1.5 |

在右侧 ⚙️ 图标中调整检索参数。使用下方按钮管理系统：""",
        actions=[
            cl.Action(name="build_vs", payload={}, label="🔄 构建向量库"),
            cl.Action(name="load_vs", payload={}, label="📂 加载已有向量库"),
            cl.Action(name="new_conv", payload={}, label="➕ 新建对话"),
            cl.Action(name="manage_conv", payload={}, label="💬 管理对话"),
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(updated_settings):
    cl.user_session.set("settings", updated_settings)
    await cl.Message(
        content=f"✅ 设置已更新 · 检索策略={updated_settings['search_type']}, k={updated_settings['k']}"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    vs = cl.user_session.get("vectorstore")
    conv_id = cl.user_session.get("conversation_id")
    history = cl.user_session.get("history")

    save_message(conv_id, "user", message.content)

    if len(history) == 0:
        title = message.content[:20] + ("..." if len(message.content) > 20 else "")
        update_conversation_title(conv_id, title)

    if vs is None:
        await cl.Message(
            content="⚠️ 向量库未加载，请先点击按钮：",
            actions=[
                cl.Action(name="build_vs", payload={}, label="🔄 构建向量库"),
                cl.Action(name="load_vs", payload={}, label="📂 加载已有向量库"),
            ],
        ).send()
        return

    # 问题改写：利用对话历史将指代性问题转为独立检索查询
    rewritten_query = rewrite_query(message.content, history)

    chain = build_rag_chain(
        vectorstore=vs,
        search_type=settings["search_type"],
        k=settings["k"],
        chat_history=history[-6:],
        query=rewritten_query,  # 检索用改写后的查询
    )

    sources = retrieve_with_sources(vs, rewritten_query, search_type=settings["search_type"], k=settings["k"])

    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = chain.invoke(message.content)

    elements = []
    # 问题被改写时展示改写结果（调试用）
    if rewritten_query != message.content:
        elements.append(cl.Text(
            name="查询改写",
            content=f"原始: {message.content}\n改写: {rewritten_query}",
            display="side",
        ))
    if sources:
        for i, s in enumerate(sources, 1):
            source_name = Path(s.get("source", "未知")).name
            src_text = s["content"][:400] + ("..." if len(s["content"]) > 400 else "")
            elements.append(cl.Text(
                name=f"来源{i}: {source_name} (位置 {s.get('start_index', '?')})",
                content=src_text,
                display="side",
            ))

    response_msg.content = full_response
    response_msg.elements = elements
    await response_msg.update()

    save_message(conv_id, "assistant", full_response, sources)

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": full_response, "sources": sources})
    cl.user_session.set("history", history)
