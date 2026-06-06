from pathlib import Path
from statistics import mean

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import load_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploaded_docs"
    / "sample.pdf"
)

chunking_configs = [
    {"chunk_size": 500, "chunk_overlap": 100},
    {"chunk_size": 800, "chunk_overlap": 150},
    {"chunk_size": 1000, "chunk_overlap": 200},
]


documents = load_document(FILE_PATH)

indexable_documents = [
    doc for doc in documents
    if doc.metadata.get("content_type") != "scanned_page"
]

print("\nCHUNKING EXPERIMENTS")
print("=" * 100)
print(f"Input file: {FILE_PATH.name}")
print(f"Documents loaded: {len(documents)}")
print(f"Indexable documents: {len(indexable_documents)}")
print(f"Skipped scanned pages: {len(documents) - len(indexable_documents)}")

for config in chunking_configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(indexable_documents)

    for i, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)

    lengths = [len(chunk.page_content) for chunk in chunks]

    print("\n" + "=" * 100)
    print(
        f"CONFIG: chunk_size={config['chunk_size']}, "
        f"overlap={config['chunk_overlap']}"
    )
    print("=" * 100)
    print(f"Total chunks: {len(chunks)}")
    print(f"Min chunk length: {min(lengths)}")
    print(f"Max chunk length: {max(lengths)}")
    print(f"Average chunk length: {mean(lengths):.2f}")

    print("\nSample chunks preview:")

    for chunk in chunks[:4]:
        print("\n" + "-" * 100)
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Length: {chunk.metadata['chunk_length']}")
        print(f"Page: {chunk.metadata.get('page_number')}")
        print(f"Content Type: {chunk.metadata.get('content_type')}")
        print(chunk.page_content[:500])