import re
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_WEIGHT,
    CHROMA_DB_PATH,
    DENSE_WEIGHT,
    EMBEDDING_MODEL_NAME,
    HYBRID_BM25_K,
    HYBRID_DENSE_K,
    HYBRID_FINAL_K,
)


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25 keyword retrieval.
    Lowercases text and keeps alphanumeric tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def get_all_chunks_from_collection(collection):
    """
    Get all stored chunks from ChromaDB so BM25 can search over them.
    """

    data = collection.get(
        include=["documents", "metadatas"]
    )

    chunks = []

    for doc_id, text, metadata in zip(
        data["ids"],
        data["documents"],
        data["metadatas"],
    ):
        chunks.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
            }
        )

    return chunks


def dense_retrieve(collection, embedding_model, question: str, top_k: int):
    """
    Dense semantic retrieval from ChromaDB.
    """

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_results = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk_id = metadata.get("chunk_id")

        dense_results.append(
            {
                "chunk_key": str(chunk_id),
                "text": doc,
                "metadata": metadata,
                "dense_distance": float(distance),
            }
        )

    return dense_results


def bm25_retrieve(all_chunks: list[dict], question: str, top_k: int):
    """
    BM25 keyword retrieval over all chunks.
    """

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(question)

    scores = bm25.get_scores(tokenized_query)

    scored_chunks = []

    for chunk, score in zip(all_chunks, scores):
        scored_chunks.append(
            {
                "chunk_key": str(chunk["metadata"].get("chunk_id")),
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "bm25_score": float(score),
            }
        )

    ranked_chunks = sorted(
        scored_chunks,
        key=lambda item: item["bm25_score"],
        reverse=True,
    )

    return ranked_chunks[:top_k]


def normalize_scores(score_dict: dict, higher_is_better: bool = True):
    """
    Normalize scores into 0-1 range.

    For dense distance:
    lower distance is better, so higher_is_better=False.

    For BM25:
    higher score is better.
    """

    if not score_dict:
        return {}

    values = list(score_dict.values())

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return {
            key: 1.0
            for key in score_dict
        }

    normalized = {}

    for key, value in score_dict.items():
        if higher_is_better:
            normalized[key] = (value - min_value) / (max_value - min_value)
        else:
            normalized[key] = (max_value - value) / (max_value - min_value)

    return normalized


def hybrid_retrieve(
    collection,
    embedding_model,
    question: str,
    dense_k: int = HYBRID_DENSE_K,
    bm25_k: int = HYBRID_BM25_K,
    final_k: int = HYBRID_FINAL_K,
):
    """
    Hybrid retrieval:
    1. Dense retrieval from ChromaDB
    2. BM25 keyword retrieval over stored chunks
    3. Normalize and combine scores
    4. Return final top-k chunks
    """

    all_chunks = get_all_chunks_from_collection(collection)

    dense_results = dense_retrieve(
        collection=collection,
        embedding_model=embedding_model,
        question=question,
        top_k=dense_k,
    )

    bm25_results = bm25_retrieve(
        all_chunks=all_chunks,
        question=question,
        top_k=bm25_k,
    )

    chunk_store = {}

    dense_distances = {}
    bm25_scores = {}

    for item in dense_results:
        key = item["chunk_key"]
        chunk_store[key] = {
            "text": item["text"],
            "metadata": item["metadata"],
            "dense_distance": item["dense_distance"],
            "bm25_score": 0.0,
        }
        dense_distances[key] = item["dense_distance"]

    for item in bm25_results:
        key = item["chunk_key"]

        if key not in chunk_store:
            chunk_store[key] = {
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_distance": None,
                "bm25_score": item["bm25_score"],
            }
        else:
            chunk_store[key]["bm25_score"] = item["bm25_score"]

        bm25_scores[key] = item["bm25_score"]

    normalized_dense = normalize_scores(
        dense_distances,
        higher_is_better=False,
    )

    normalized_bm25 = normalize_scores(
        bm25_scores,
        higher_is_better=True,
    )

    final_results = []

    for key, item in chunk_store.items():
        dense_score = normalized_dense.get(key, 0.0)
        bm25_score = normalized_bm25.get(key, 0.0)

        hybrid_score = (
            DENSE_WEIGHT * dense_score
            + BM25_WEIGHT * bm25_score
        )

        final_results.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_distance": item["dense_distance"],
                "bm25_score": item["bm25_score"],
                "hybrid_score": hybrid_score,
            }
        )

    ranked_results = sorted(
        final_results,
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return ranked_results[:final_k]


def get_hybrid_ready_collection(collection_name: str):
    """
    Utility helper to connect to ChromaDB collection.
    """

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = chroma_client.get_collection(
        name=collection_name
    )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return collection, embedding_model