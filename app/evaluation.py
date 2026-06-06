import csv
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    PROJECT_ROOT,
    TOP_K,
)


EVAL_FILE_PATH = PROJECT_ROOT / "evaluation" / "eval_questions.json"
REPORT_FILE_PATH = PROJECT_ROOT / "evaluation" / "retrieval_report.csv"


def load_eval_questions(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval():
    print("\nPOLICYDOC AI - RETRIEVAL EVALUATION")
    print("=" * 100)

    eval_questions = load_eval_questions(EVAL_FILE_PATH)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = chroma_client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )

    rows = []
    recall_hits = 0
    reciprocal_ranks = []

    for item in eval_questions:
        question = item["question"]
        expected_pages = item["expected_page_numbers"]

        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_metadatas = results["metadatas"][0]
        retrieved_distances = results["distances"][0]

        retrieved_pages = [
            metadata.get("page_number")
            for metadata in retrieved_metadatas
        ]

        retrieved_chunk_ids = [
            metadata.get("chunk_id")
            for metadata in retrieved_metadatas
        ]

        retrieved_content_types = [
            metadata.get("content_type")
            for metadata in retrieved_metadatas
        ]

        hit = False
        reciprocal_rank = 0.0

        for rank, metadata in enumerate(retrieved_metadatas, start=1):
            page_number = metadata.get("page_number")

            if page_number in expected_pages:
                hit = True
                reciprocal_rank = 1 / rank
                break

        if hit:
            recall_hits += 1

        reciprocal_ranks.append(reciprocal_rank)

        rows.append(
            {
                "question": question,
                "expected_pages": str(expected_pages),
                "retrieved_pages": str(retrieved_pages),
                "retrieved_chunk_ids": str(retrieved_chunk_ids),
                "retrieved_content_types": str(retrieved_content_types),
                "retrieved_distances": str(
                    [round(distance, 4) for distance in retrieved_distances]
                ),
                f"hit_at_{TOP_K}": hit,
                "reciprocal_rank": round(reciprocal_rank, 3),
            }
        )

        print("\nQUESTION:")
        print(question)
        print(f"Expected pages: {expected_pages}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Retrieved chunk IDs: {retrieved_chunk_ids}")
        print(f"Retrieved content types: {retrieved_content_types}")
        print(f"Hit@{TOP_K}: {hit}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")
        print("-" * 100)

    recall_at_k = recall_hits / len(eval_questions)
    mrr = sum(reciprocal_ranks) / len(eval_questions)

    REPORT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE_PATH, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "question",
            "expected_pages",
            "retrieved_pages",
            "retrieved_chunk_ids",
            "retrieved_content_types",
            "retrieved_distances",
            f"hit_at_{TOP_K}",
            "reciprocal_rank",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFINAL RETRIEVAL METRICS")
    print("=" * 100)
    print(f"Recall@{TOP_K}: {recall_at_k:.2f}")
    print(f"MRR: {mrr:.2f}")
    print(f"Report saved to: {REPORT_FILE_PATH}")


if __name__ == "__main__":
    evaluate_retrieval()