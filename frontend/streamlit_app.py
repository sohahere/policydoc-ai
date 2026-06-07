import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"

sys.path.append(str(APP_DIR))

from config import DEFAULT_FILE_NAME, UPLOAD_DIR
from index_documents import main as index_documents
from rag_answer import generate_answer


st.set_page_config(
    page_title="PolicyDoc AI",
    page_icon="📄",
    layout="wide",
)


st.title("📄 PolicyDoc AI")
st.subheader("Evaluated RAG Assistant for Company Policy Documents")

st.markdown(
    """
PolicyDoc AI helps users ask questions from company policy and compliance documents.

**Current scope**
- TXT files
- PDF text extraction
- PDF table extraction
- Table-to-markdown conversion
- ChromaDB semantic retrieval
- Hugging Face LLM answer generation
- Source citations
- Missing-answer handling
"""
)


with st.sidebar:
    st.header("Project Settings")
    st.write("**Vector DB:** ChromaDB")
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Chunking:** 1000 chars / 200 overlap")
    st.write("**Top-k:** 3")


uploaded_file = st.file_uploader(
    "Upload a company policy document",
    type=["pdf", "txt"],
)


if uploaded_file is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_DIR / DEFAULT_FILE_NAME

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"Uploaded and saved as `{DEFAULT_FILE_NAME}`")

    if st.button("Index Document"):
        try:
            with st.spinner("Indexing document into ChromaDB..."):
                stats = index_documents()

            st.session_state["document_indexed"] = True
            st.success("Document indexed successfully.")
            st.json(stats)

        except Exception as e:
            st.error("Document indexing failed.")
            st.exception(e)


st.divider()


question = st.text_input(
    "Ask a question from the uploaded policy document",
    placeholder="Example: What is the local meal reimbursement limit?",
)


if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                answer, retrieved_chunks = generate_answer(question)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Sources")

            if not retrieved_chunks:
                st.info("No retrieved sources were returned.")
            else:
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    metadata = chunk.get("metadata", {})

                    source_title = (
                        f"Source {i}: "
                        f"page {metadata.get('page_number')} | "
                        f"chunk {metadata.get('chunk_id')} | "
                        f"{metadata.get('content_type')}"
                    )

                    with st.expander(source_title):
                        st.write(f"**File:** {metadata.get('file_name')}")
                        st.write(f"**Page:** {metadata.get('page_number')}")
                        st.write(f"**Chunk ID:** {metadata.get('chunk_id')}")
                        st.write(f"**Content Type:** {metadata.get('content_type')}")

                        distance = chunk.get("distance")
                        if distance is not None:
                            st.write(f"**Distance:** {distance:.4f}")

                        st.text(chunk.get("text", "")[:1500])

        except Exception as e:
            error_text = str(e)

            if "Collection [policy_documents] does not exist" in error_text:
                st.warning(
                    "Please upload and index a document before asking a question."
                )
            elif "HF_TOKEN not found" in error_text:
                st.warning(
                    "HF_TOKEN is missing. Add it in your .env file locally or as a Space secret on Hugging Face."
                )
            else:
                st.error("Something went wrong while generating the answer.")
                st.exception(e)