import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CANDIDATE_K,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    HARD_EVAL_FILE_NAME,
    PROJECT_ROOT,
    UPLOAD_DIR,
)
from loaders import load_document


EVAL_FILE_PATH = PROJECT_ROOT / "evaluation" / "hard_eval_questions.json"
COLLECTION_NAME = "debug_candidate_recall_collection"


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


def build_collection():
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
        f"debug_chunk_{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    return collection, embedding_model


def main():
    print("\nCANDIDATE RECALL DEBUG")
    print("=" * 100)
    print(f"Candidate-k: {CANDIDATE_K}")

    eval_questions = load_eval_questions(EVAL_FILE_PATH)
    collection, embedding_model = build_collection()

    candidate_hits = 0

    for item in eval_questions:
        question = item["question"]
        expected_pages = item["expected_page_numbers"]

        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=CANDIDATE_K,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_metadatas = results["metadatas"][0]
        retrieved_pages = [
            metadata.get("page_number")
            for metadata in retrieved_metadatas
        ]
        retrieved_chunk_ids = [
            metadata.get("chunk_id")
            for metadata in retrieved_metadatas
        ]
        retrieved_types = [
            metadata.get("content_type")
            for metadata in retrieved_metadatas
        ]

        hit = any(page in expected_pages for page in retrieved_pages)

        if hit:
            candidate_hits += 1

        print("\nQUESTION:")
        print(question)
        print(f"Expected pages: {expected_pages}")
        print(f"Candidate pages: {retrieved_pages}")
        print(f"Candidate chunk IDs: {retrieved_chunk_ids}")
        print(f"Candidate content types: {retrieved_types}")
        print(f"Candidate hit@{CANDIDATE_K}: {hit}")
        print("-" * 100)

    candidate_recall = candidate_hits / len(eval_questions)

    print("\nFINAL CANDIDATE RECALL")
    print("=" * 100)
    print(f"Candidate Recall@{CANDIDATE_K}: {candidate_recall:.2f}")


if __name__ == "__main__":
    main()