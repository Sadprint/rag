import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
ROOT_DIR = Path(__file__).parent

# ============ 模型配置 ============
# BGE 嵌入模型路径 (相对于项目根目录)
EMBEDDING_MODEL_PATH = str(ROOT_DIR / "model" / "BAAI_bge-base-zh-v1.5")
# 嵌入设备：auto → 自动检测 CUDA，否则手动指定 "cuda" / "cpu"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# ============ 智谱 AI (GLM) API 配置 ============
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
ZHIPUAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"

# ============ 文档处理配置 ============
CHUNK_SIZE = 500           # 分块大小
CHUNK_OVERLAP = 50         # 块间重叠
# 中文友好的分隔符
SEPARATORS = [
    "\n\n",    # 段落
    "\n",      # 行
    "。",      # 中文句号
    "！",      # 感叹号
    "？",      # 问号
    "；",      # 分号
    "，",      # 逗号
    ".",       # 英文句号
    " ",       # 空格
    "",        # 字符级
]

# ============ 检索配置 ============
DEFAULT_K = 4              # 默认检索条数
DEFAULT_SEARCH_TYPE = "mmr"  # similarity 或 mmr
MMR_FETCH_K = 10           # MMR 初筛数量
MMR_LAMBDA_MULT = 0.7      # MMR 多样性权重 (越大越相关, 越小越多样)

# ============ ChromaDB 配置 ============
CHROMA_PERSIST_DIR = str(ROOT_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "local_docs"

# ============ 文档目录 ============
DOCS_DIR = ROOT_DIR / "docs"

# ============ 评估配置 ============
MAX_EVAL_ITEMS = 100  # 每次最多评估条数

# ============ 数据库配置 (SQLite) ============
DATABASE_PATH = str(ROOT_DIR / "data" / "conversations.db")
