"""
Configuration for the ARLC RAG pipeline.
Extends the starter kit config with additional settings.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Try loading from starter_kit/.env first, then project root
    _sk_env = Path(__file__).resolve().parents[1] / "starter_kit" / ".env"
    _root_env = Path(__file__).resolve().parents[1] / ".env"
    if _sk_env.exists():
        load_dotenv(_sk_env)
    if _root_env.exists():
        load_dotenv(_root_env, override=True)
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _get_int(key: str, default: int) -> int:
    return int(_get(key, str(default)) or str(default))


def _get_optional_int(key: str) -> int | None:
    value = _get(key)
    return int(value) if value else None


# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = PROJECT_ROOT / "index"
STARTER_KIT_DIR = PROJECT_ROOT / "starter_kit"

# Index files
PAGES_JSONL = INDEX_DIR / "pages.jsonl"
CHUNKS_JSONL = INDEX_DIR / "chunks.jsonl"
BM25_INDEX = INDEX_DIR / "bm25.pkl"
FAISS_INDEX = INDEX_DIR / "faiss.index"
FAISS_IDS = INDEX_DIR / "faiss_ids.json"

# --- API Keys ---
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENROUTER_API_KEY = _get("OPENROUTER_API_KEY")
GEMINI_API_KEY = _get("GEMINI_API_KEY")
EVAL_API_KEY = _get("EVAL_API_KEY")
LLM_PROVIDER = _get("LLM_PROVIDER").lower()

# --- Model Settings ---
EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "gemini").lower()
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "models/gemini-embedding-2-preview")
EMBEDDING_BATCH_SIZE = max(1, _get_int("EMBEDDING_BATCH_SIZE", 16))
EMBEDDING_OUTPUT_DIMENSION = _get_optional_int("EMBEDDING_OUTPUT_DIMENSION")
EMBEDDING_REQUEST_TIMEOUT_SECONDS = max(1, _get_int("EMBEDDING_REQUEST_TIMEOUT_SECONDS", 60))
EMBEDDING_MAX_RETRIES = max(1, _get_int("EMBEDDING_MAX_RETRIES", 5))
EMBEDDING_API_BASE = _get("EMBEDDING_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
EMBEDDING_QUERY_TASK_TYPE = _get("EMBEDDING_QUERY_TASK_TYPE", "RETRIEVAL_QUERY")
EMBEDDING_DOCUMENT_TASK_TYPE = _get("EMBEDDING_DOCUMENT_TASK_TYPE", "RETRIEVAL_DOCUMENT")
RERANKER_MODEL = _get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
GENERATION_MODEL = _get("GENERATION_MODEL", "gpt-4.1-mini")
GENERATION_TEMPERATURE = 0.1

# --- Retrieval Settings ---
BM25_TOP_K = 30
SEMANTIC_TOP_K = 30
RRF_K = 60
RERANK_TOP_K = 10          # how many chunks to keep after reranking
RERANK_CANDIDATES = 30     # how many candidates to feed the reranker
ENABLE_RERANKER = _get("ENABLE_RERANKER", "0").lower() in {"1", "true", "yes", "on"}
RERANKER_BATCH_SIZE = max(1, int(_get("RERANKER_BATCH_SIZE", "4") or "4"))
RERANKER_MAX_LENGTH = max(128, int(_get("RERANKER_MAX_LENGTH", "512") or "512"))
RERANKER_USE_FP16 = _get("RERANKER_USE_FP16", "1").lower() in {"1", "true", "yes", "on"}
GENERATION_TOP_K = int(_get("GENERATION_TOP_K", "4") or "4")

# --- Chunking Settings ---
MAX_CHUNK_TOKENS = max(64, _get_int("MAX_CHUNK_TOKENS", 512))
MIN_CHUNK_TOKENS = max(1, _get_int("MIN_CHUNK_TOKENS", 100))
OCR_MIN_CHARS = 50          # if pdfplumber extracts fewer chars, use OCR

# --- Device ---
try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment
    torch = None

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def get_embedding_api_key() -> str:
    """API key for the configured embedding provider."""
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_API_KEY
    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def model_uses_openrouter(model_name: str | None = None) -> bool:
    """Decide whether the selected model should be routed through OpenRouter."""
    model_name = str(model_name or GENERATION_MODEL).strip().lower()

    if LLM_PROVIDER == "openrouter":
        return True
    if LLM_PROVIDER == "openai":
        return False

    return bool(OPENROUTER_API_KEY and "/" in model_name)


def get_llm_api_key(model_name: str | None = None) -> str:
    """API key for LLM generation based on the selected model/provider."""
    if model_uses_openrouter(model_name):
        return OPENROUTER_API_KEY or OPENAI_API_KEY
    return OPENAI_API_KEY or OPENROUTER_API_KEY


def get_llm_api_base(model_name: str | None = None) -> str:
    """API base URL for LLM generation based on the selected model/provider."""
    if model_uses_openrouter(model_name):
        return _get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if OPENAI_API_KEY:
        return _get("OPENAI_API_BASE", "https://api.openai.com/v1")
    return _get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def ensure_dirs():
    """Create required directories if they don't exist."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)