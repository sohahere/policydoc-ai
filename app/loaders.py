from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document


# --------------------------------------------------
# 1. Convert extracted table into markdown
# --------------------------------------------------

def table_to_markdown(table: list) -> str:
    """
    Convert a PDF-extracted table into markdown format.

    Why?
    Raw PDF tables usually come as nested lists.
    Markdown preserves header-value relationships better for RAG.
    """

    if not table:
        return ""

    cleaned_table = [
        [str(cell).strip() if cell is not None else "" for cell in row]
        for row in table
    ]

    headers = cleaned_table[0]
    rows = cleaned_table[1:]

    if not headers:
        return ""

    markdown_lines = []

    # Header
    markdown_lines.append("| " + " | ".join(headers) + " |")

    # Separator
    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Rows
    for row in rows:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[: len(headers)]

        markdown_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(markdown_lines)


# --------------------------------------------------
# 2. Load TXT file
# --------------------------------------------------

def load_txt(file_path: str | Path) -> List[Document]:
    """
    Load a TXT file as one LangChain Document.
    """

    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "txt",
                "content_type": "text",
                "page_number": None,
            },
        )
    ]


# --------------------------------------------------
# 3. Load PDF file
# --------------------------------------------------

def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a PDF file.

    Current v1 support:
    1. Page-wise text extraction
    2. Table extraction
    3. Table-to-markdown conversion

    OCR/scanned PDF support is not implemented in v1.
    """

    file_path = Path(file_path)
    documents: List[Document] = []

    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_tables = page.extract_tables() or []

            # -----------------------------
            # Extract normal page text
            # -----------------------------

            if page_text.strip():
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": "pdf",
                            "content_type": "text",
                            "page_number": page_index,
                        },
                    )
                )

            # -----------------------------
            # Extract tables separately
            # -----------------------------

            for table_index, table in enumerate(page_tables, start=1):
                markdown_table = table_to_markdown(table)

                if markdown_table.strip():
                    documents.append(
                        Document(
                            page_content=markdown_table,
                            metadata={
                                "source": str(file_path),
                                "file_name": file_path.name,
                                "file_type": "pdf",
                                "content_type": "table",
                                "page_number": page_index,
                                "table_index": table_index,
                            },
                        )
                    )

    return documents


# --------------------------------------------------
# 4. Main file-type router
# --------------------------------------------------

def load_document(file_path: str | Path) -> List[Document]:
    """
    Detect file type and call the correct loader.

    Supported in v1:
    - .txt
    - .pdf

    Future work:
    - OCR for scanned PDFs
    - DOCX support
    """

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return load_txt(file_path)

    if suffix == ".pdf":
        return load_pdf(file_path)

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Supported file types are .txt and .pdf"
    )