import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)


# -----------------------------
# 1. Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# -----------------------------
# 2. Connect to ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DB_PATH)
)

collection = chroma_client.get_collection(
    name=CHROMA_COLLECTION_NAME
)


# -----------------------------
# 3. Ask policy-related test query
# -----------------------------

query = "What is the local meal reimbursement limit?"

query_embedding = embedding_model.encode(query).tolist()


# -----------------------------
# 4. Retrieve top-k chunks
# -----------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=["documents", "metadatas", "distances"],
)


# -----------------------------
# 5. Print retrieved chunks with citation metadata
# -----------------------------

print("\nQUERY:")
print(query)

print("\nTOP RETRIEVED CHUNKS:")

for rank, (doc, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ),
    start=1,
):
    print("\n" + "=" * 100)
    print(f"RANK: {rank}")
    print(f"DISTANCE: {distance:.4f}")
    print(f"FILE: {metadata.get('file_name')}")
    print(f"PAGE: {metadata.get('page_number')}")
    print(f"CHUNK ID: {metadata.get('chunk_id')}")
    print(f"CONTENT TYPE: {metadata.get('content_type')}")
    print("=" * 100)
    print(doc)