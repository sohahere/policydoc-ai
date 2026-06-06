from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from loaders import load_document


# -----------------------------
# 1. Project paths and settings
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploaded_docs"
    / "sample.pdf"
)

CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


# -----------------------------
# 2. Chunking configurations to test
# -----------------------------

CHUNKING_CONFIGS = [
    {
        "name": "small_chunks",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    {
        "name": "balanced_chunks",
        "chunk_size": 800,
        "chunk_overlap": 150,
    },
    {
        "name": "large_chunks",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
]


# -----------------------------
# 3. Evaluation questions
# -----------------------------
# We evaluate using expected page numbers.
# Why?
# Because the same answer can appear in extracted text OR extracted markdown table.
# At this stage, we care whether retrieval brings evidence from the correct page.

EVALUATION_QUESTIONS = [
    {
        "question": "What is the purpose of this advanced RAG test document?",
        "expected_page_numbers": [1],
    },
    {
        "question": "What are common retrieval evaluation metrics?",
        "expected_page_numbers": [5],
    },
    {
        "question": "What does Recall@k measure?",
        "expected_page_numbers": [5],
    },
    {
        "question": "Why is chunk overlap useful?",
        "expected_page_numbers": [4],
    },
    {
        "question": "How should the system behave when the answer is not in the documents?",
        "expected_page_numbers": [7],
    },
    {
        "question": "What does metadata help with in RAG?",
        "expected_page_numbers": [2],
    },
    {
        "question": "What are the strengths of dense vector search?",
        "expected_page_numbers": [3],
    },
    {
        "question": "What should evaluation include?",
        "expected_page_numbers": [8],
    },
]


# -----------------------------
# 4. Prepare chunks
# -----------------------------

def prepare_chunks(documents, chunk_size, chunk_overlap):
    """
    Remove unsupported/scanned placeholders, split documents,
    and add chunk-level metadata.
    """

    indexable_documents = [
        doc for doc in documents
        if doc.metadata.get("content_type") != "scanned_page"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(indexable_documents)

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    return chunks


# -----------------------------
# 5. Reset ChromaDB collection
# -----------------------------

def reset_collection(chroma_client, collection_name):
    """
    Delete old test collection if it exists, then create a fresh one.
    This avoids duplicate IDs while testing multiple chunk configs.
    """

    existing_collections = [
        collection.name for collection in chroma_client.list_collections()
    ]

    if collection_name in existing_collections:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    return collection


# -----------------------------
# 6. Index chunks into ChromaDB
# -----------------------------

def index_chunks(collection, chunks, embeddings, config_name):
    """
    Store chunk text, embeddings, and metadata in ChromaDB.
    """

    ids = [
        f"{config_name}_chunk_{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )


# -----------------------------
# 7. Evaluate retrieval
# -----------------------------

def evaluate_collection(collection, embedding_model, top_k=3):
    """
    Evaluate whether the retriever returns evidence
    from the expected page within top-k results.

    Metrics:
    - Recall@k: Did expected page appear in top-k?
    - MRR: How high was the first correct page ranked?
    """

    recall_hits = 0
    reciprocal_ranks = []

    for item in EVALUATION_QUESTIONS:
        question = item["question"]
        expected_pages = item["expected_page_numbers"]

        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_metadatas = results["metadatas"][0]

        hit = False
        reciprocal_rank = 0

        for rank, metadata in enumerate(retrieved_metadatas, start=1):
            retrieved_page = metadata.get("page_number")

            if retrieved_page in expected_pages:
                hit = True
                reciprocal_rank = 1 / rank
                break

        if hit:
            recall_hits += 1

        reciprocal_ranks.append(reciprocal_rank)

        print("\nQUESTION:")
        print(question)
        print(f"Expected pages: {expected_pages}")

        print("Retrieved pages:", [
            metadata.get("page_number")
            for metadata in retrieved_metadatas
        ])

        print("Retrieved content types:", [
            metadata.get("content_type")
            for metadata in retrieved_metadatas
        ])

        print(f"Hit@{top_k}: {hit}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")
        print("-" * 100)

    recall_at_k = recall_hits / len(EVALUATION_QUESTIONS)
    mrr = sum(reciprocal_ranks) / len(EVALUATION_QUESTIONS)

    return recall_at_k, mrr


# -----------------------------
# 8. Main experiment runner
# -----------------------------

def main():
    print("\nCHUNK CONFIG RETRIEVAL EVALUATION")
    print("=" * 100)

    print(f"Input file: {FILE_PATH.name}")
    print(f"Top-k: {TOP_K}")

    documents = load_document(FILE_PATH)

    print(f"Document parts loaded: {len(documents)}")

    indexable_documents = [
        doc for doc in documents
        if doc.metadata.get("content_type") != "scanned_page"
    ]

    print(f"Indexable documents: {len(indexable_documents)}")
    print(f"Skipped scanned pages: {len(documents) - len(indexable_documents)}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    final_results = []

    for config in CHUNKING_CONFIGS:
        print("\n" + "=" * 100)
        print(
            f"Testing config: {config['name']} "
            f"(size={config['chunk_size']}, "
            f"overlap={config['chunk_overlap']})"
        )
        print("=" * 100)

        chunks = prepare_chunks(
            documents=documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )

        chunk_texts = [
            chunk.page_content for chunk in chunks
        ]

        embeddings = embedding_model.encode(chunk_texts).tolist()

        collection_name = f"eval_{config['name']}"

        collection = reset_collection(
            chroma_client=chroma_client,
            collection_name=collection_name,
        )

        index_chunks(
            collection=collection,
            chunks=chunks,
            embeddings=embeddings,
            config_name=config["name"],
        )

        recall_at_3, mrr = evaluate_collection(
            collection=collection,
            embedding_model=embedding_model,
            top_k=TOP_K,
        )

        result = {
            "config": config["name"],
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "total_chunks": len(chunks),
            "recall_at_3": recall_at_3,
            "mrr": mrr,
        }

        final_results.append(result)

        print("\nCONFIG RESULT")
        print(f"Total chunks: {len(chunks)}")
        print(f"Recall@3: {recall_at_3:.2f}")
        print(f"MRR: {mrr:.2f}")

    print("\nFINAL COMPARISON")
    print("=" * 100)

    for result in final_results:
        print(
            f"{result['config']} | "
            f"size={result['chunk_size']} | "
            f"overlap={result['chunk_overlap']} | "
            f"chunks={result['total_chunks']} | "
            f"Recall@3={result['recall_at_3']:.2f} | "
            f"MRR={result['mrr']:.2f}"
        )


if __name__ == "__main__":
    main()