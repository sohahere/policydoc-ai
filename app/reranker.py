from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME


@lru_cache(maxsize=1)
def load_reranker():
    """
    Load the cross-encoder reranker once and reuse it.
    Without caching, the model may reload for every question.
    """
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_chunks(question: str, chunks: list[dict], top_k: int):
    """
    Rerank candidate chunks using a cross-encoder.

    ChromaDB first retrieves a larger candidate set.
    The cross-encoder then scores each (question, chunk) pair.
    """

    reranker = load_reranker()

    pairs = [
        [question, chunk["text"]]
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return ranked_chunks[:top_k]