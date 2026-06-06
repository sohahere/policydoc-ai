import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_FILE_NAME,
    EMBEDDING_MODEL_NAME,
    UPLOAD_DIR,
)

from loaders import load_document


def index_documents():
    # -----------------------------
    # 1. Input file
    # -----------------------------

    file_path = UPLOAD_DIR / DEFAULT_FILE_NAME

    # -----------------------------
    # 2. Load document
    # -----------------------------

    documents = load_document(file_path)

    print("\nDOCUMENT LOADING COMPLETED")
    print("=" * 100)
    print(f"Input file: {file_path.name}")
    print(f"Document parts returned by loader: {len(documents)}")

    for doc in documents:
        print(doc.metadata)

    # -----------------------------
    # 3. Split documents into chunks
    # -----------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # -----------------------------
    # 4. Add chunk metadata
    # -----------------------------

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    print("\nCHUNKING COMPLETED")
    print("=" * 100)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Total chunks created: {len(chunks)}")

    # -----------------------------
    # 5. Create embeddings
    # -----------------------------

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunk_texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.encode(chunk_texts).tolist()

    print("\nEMBEDDING CREATION COMPLETED")
    print("=" * 100)
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # -----------------------------
    # 6. Connect to ChromaDB
    # -----------------------------

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )

    # -----------------------------
    # 7. Reset old collection data
    # -----------------------------

    existing_count = collection.count()

    if existing_count > 0:
        existing_data = collection.get()
        existing_ids = existing_data["ids"]

        if existing_ids:
            collection.delete(ids=existing_ids)

    print("\nCHROMADB RESET COMPLETED")
    print("=" * 100)
    print(f"Old chunks removed: {existing_count}")

    # -----------------------------
    # 8. Prepare data for ChromaDB
    # -----------------------------

    ids = [
        f"{file_path.stem}_chunk_{chunk.metadata['chunk_id']}"
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    documents_text = [
        chunk.page_content
        for chunk in chunks
    ]

    # -----------------------------
    # 9. Store in ChromaDB
    # -----------------------------

    collection.add(
        ids=ids,
        documents=documents_text,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("\nCHROMADB STORAGE COMPLETED")
    print("=" * 100)
    print(f"Stored chunks: {collection.count()}")
    print(f"Collection name: {CHROMA_COLLECTION_NAME}")
    print(f"ChromaDB path: {CHROMA_DB_PATH}")

    return {
        "file_name": file_path.name,
        "document_parts": len(documents),
        "chunks": len(chunks),
        "embeddings": len(embeddings),
        "collection_name": CHROMA_COLLECTION_NAME,
    }


def main():
    return index_documents()


if __name__ == "__main__":
    main()