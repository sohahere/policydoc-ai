import csv
import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    HARD_EVAL_FILE_NAME,
    PROJECT_ROOT,
    TOP_K,
    UPLOAD_DIR,
)
from loaders import load_document


EVAL_FILE_PATH = PROJECT_ROOT / "evaluation" / "hard_eval_questions.json"
REPORT_FILE_PATH = PROJECT_ROOT / "evaluation" / "hard_base_retrieval_report.csv"
COLLECTION_NAME = "hard_base_policy_documents"


def load_eval_questions(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def reset_collection(chroma_client, collection_name):
    existing_collections = [
        collection.name for collection in chroma_client.list_collections()
    ]

    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)

    return chroma_client.get_or_create_collection(name=collection_name)


def index_hard_pdf():
    file_path = UPLOAD_DIR / HARD_EVAL_FILE_NAME

    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(chunk_texts).tolist()

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    collection = reset_collection(
        chroma_client=chroma_client,
        collection_name=COLLECTION_NAME,
    )

    ids = [
        f"hard_eval_chunk_{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    return collection, embedding_model, len(documents), len(chunks)


def evaluate_base_retrieval():
    print("\nHARD PDF BASE RETRIEVAL EVALUATION")
    print("=" * 100)

    eval_questions = load_eval_questions(EVAL_FILE_PATH)

    collection, embedding_model, document_parts, total_chunks = index_hard_pdf()

    print(f"Document parts loaded: {document_parts}")
    print(f"Chunks indexed: {total_chunks}")
    print(f"Top-k: {TOP_K}")

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
            if metadata.get("page_number") in expected_pages:
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

    print("\nFINAL HARD BASE RETRIEVAL METRICS")
    print("=" * 100)
    print(f"Recall@{TOP_K}: {recall_at_k:.2f}")
    print(f"MRR: {mrr:.2f}")
    print(f"Report saved to: {REPORT_FILE_PATH}")


if __name__ == "__main__":
    evaluate_base_retrieval()