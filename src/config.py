# -*- coding: utf-8 -*-
"""
项目配置文件
包含API配置、数据库配置、检索配置等
"""

import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# API 配置
# =============================================================================

# DeepSeek API配置
DEEPSEEK_API_KEY = ""  # DeepSeek API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek API地址
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek聊天模型

# 阿里云百炼Embedding配置
ALIBABA_API_KEY = ""  # 阿里云百炼API密钥
EMBEDDING_MODEL = "text-embedding-v4"  # 阿里云百炼Embedding模型

# 阿里云百炼语音合成(TTS)配置
TTS_ENABLED = True  # 是否启用语音合成功能
TTS_MODEL = "cosyvoice-v3-flash"  # 语音合成模型
TTS_VOICE = "longyingling_v3"  # 音色选择（longanhuan是活泼的女声，适合心理咨询场景）
TTS_RATE = "1"  # 语速（0.5-2.0，1为正常语速）
TTS_OUTPUT_DIR = str(PROJECT_ROOT / "storage" / "audio")  # 音频文件输出目录

# =============================================================================
# 存储配置
# =============================================================================

# ChromaDB配置
CHROMA_DB_PATH = str(PROJECT_ROOT / "storage" / "chroma_db")
COLLECTION_NAME = "psychology_knowledge"

# =============================================================================
# 资源路径配置
# =============================================================================

# 知识库文档目录
KNOWLEDGE_DIR = str(PROJECT_ROOT / "resources" / "knowledge")

# 提示词目录
PROMPTS_DIR = str(PROJECT_ROOT / "resources" / "prompts")

# =============================================================================
# 文本处理配置
# =============================================================================

# 文本分块配置
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# =============================================================================
# 检索配置
# =============================================================================

TOP_K_RESULTS = 6
SIMILARITY_THRESHOLD = 0.15

# =============================================================================
# 对话配置
# =============================================================================

# 对话持续监控配置
MAX_NO_RAG_ROUNDS = 3


