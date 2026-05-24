"""
对话记忆模块 — SQLite 持久化
"""
import json
import sqlite3
import os
from contextlib import contextmanager

import config


@contextmanager
def _get_conn():
    """获取数据库连接，自动提交/关闭"""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """初始化数据库表（不存在则创建）"""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT DEFAULT '新对话',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)


# ========== 对话操作 ==========

def create_conversation(title="新对话"):
    """创建新对话，返回 id"""
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        return cur.lastrowid


def get_conversations():
    """获取所有对话列表，按更新时间倒序"""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        )
        return [dict(row) for row in cur.fetchall()]


def update_conversation_title(conv_id, title):
    """更新对话标题"""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, conv_id),
        )


def delete_conversation(conv_id):
    """删除对话及其所有消息"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


# ========== 消息操作 ==========

def save_message(conv_id, role, content, sources=None):
    """保存一条消息，同时更新对话时间"""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, json.dumps(sources, ensure_ascii=False) if sources else None),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conv_id,),
        )


def get_messages(conv_id):
    """获取某对话的所有消息"""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
    for row in rows:
        if row.get("sources") and isinstance(row["sources"], str):
            row["sources"] = json.loads(row["sources"])
    return rows
