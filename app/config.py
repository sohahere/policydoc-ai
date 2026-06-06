from pathlib import Path


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_docs"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"


# -----------------------------
# Document settings
# -----------------------------

DEFAULT_FILE_NAME = "sample.pdf"


# -----------------------------
# Embedding settings
# -----------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Chunking settings
# -----------------------------
# Selected after evaluating:
# 500/100, 800/150, 1000/200
#
# Final selected config:
# chunk_size=1000, overlap=200
#
# Reason:
# Recall@3 = 1.00
# MRR = 1.00
# Fewer chunks than smaller configs

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# -----------------------------
# Retrieval settings
# -----------------------------

TOP_K = 3


# -----------------------------
# ChromaDB settings
# -----------------------------

CHROMA_COLLECTION_NAME = "policy_documents"

# -----------------------------
# LLM settings
# -----------------------------

HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.2

HARD_EVAL_FILE_NAME = "hard_evaluation.pdf"

CANDIDATE_K = 10
RERANK_TOP_K = 3

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------
# Reranking settings
# -----------------------------

HARD_EVAL_FILE_NAME = "hard_evaluation.pdf"

CANDIDATE_K = 20
RERANK_TOP_K = 3

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------
# Hybrid retrieval settings
# -----------------------------

HYBRID_DENSE_K = 20
HYBRID_BM25_K = 20
HYBRID_FINAL_K = 3

DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5