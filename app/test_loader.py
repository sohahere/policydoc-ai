from pathlib import Path

from loaders import load_document


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploaded_docs"
    / "sample.pdf"
)

documents = load_document(FILE_PATH)

print("\nDOCUMENT LOADING TEST")
print("=" * 80)
print(f"File: {FILE_PATH.name}")
print(f"Total documents returned: {len(documents)}")

for i, doc in enumerate(documents, start=1):
    print("\n" + "=" * 80)
    print(f"DOCUMENT PART {i}")
    print("=" * 80)
    print("Metadata:", doc.metadata)
    print("Preview:")
    print(doc.page_content[:500])