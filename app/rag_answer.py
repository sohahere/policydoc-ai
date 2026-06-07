import os
import time

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    APP_VERSION,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    HF_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    PROMPT_FILE_PATH,
    PROMPT_VERSION,
    RAG_RUN_LOG_PATH,
    TOP_K,
)

from exceptions import (
    LLMGenerationError,
    RetrievalError,
    VectorStoreError,
)

from llmops import (
    build_context_from_sources,
    load_prompt_template,
    log_rag_run,
)

from logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def get_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_collection():
    """
    Connect to the existing ChromaDB collection.
    """
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        collection = chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME
        )

        logger.info(
            f"Connected to ChromaDB collection: {CHROMA_COLLECTION_NAME}"
        )

        return collection

    except Exception as error:
        raise VectorStoreError(
            f"Failed to connect to ChromaDB collection: {error}"
        ) from error


def get_hf_client():
    """
    Create Hugging Face Inference Provider client.

    Hugging Face uses an OpenAI-compatible client interface here.
    """
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise LLMGenerationError(
            "HF_TOKEN not found. Add it to your .env file."
        )

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )


def retrieve_context(question: str, top_k: int = TOP_K):
    """
    Retrieve top-k relevant chunks from ChromaDB.
    """
    try:
        logger.info(f"Retrieving context for question: {question}")

        embedding_model = get_embedding_model()
        collection = get_chroma_collection()

        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved_chunks.append(
                {
                    "text": doc,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        return retrieved_chunks

    except Exception as error:
        raise RetrievalError(
            f"Failed to retrieve context from ChromaDB: {error}"
        ) from error


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build prompt using versioned prompt template.

    The prompt template is stored separately inside prompts/
    as a lightweight prompt-versioning practice.
    """
    prompt_template = load_prompt_template(PROMPT_FILE_PATH)

    context = build_context_from_sources(retrieved_chunks)

    prompt = prompt_template.format(
        context=context,
        question=question,
    )

    logger.info(f"Prompt built using version: {PROMPT_VERSION}")

    return prompt


def generate_answer(question: str):
    """
    Complete RAG answer generation flow.

    Steps:
    1. Start timer
    2. Retrieve relevant chunks
    3. Load versioned prompt template
    4. Build grounded prompt
    5. Call Hugging Face LLM
    6. Log RAG run with latency, prompt version, and retrieved sources
    7. Return answer and retrieved chunks
    """
    start_time = time.time()

    try:
        retrieved_chunks = retrieve_context(question)

        prompt = build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        hf_client = get_hf_client()

        logger.info(f"Calling Hugging Face LLM: {HF_MODEL_NAME}")

        response = hf_client.chat.completions.create(
            model=HF_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

        answer = response.choices[0].message.content

        logger.info("LLM answer generated successfully")

        log_rag_run(
            log_path=RAG_RUN_LOG_PATH,
            question=question,
            answer=answer,
            retrieved_sources=retrieved_chunks,
            prompt_version=PROMPT_VERSION,
            app_version=APP_VERSION,
            start_time=start_time,
            extra={
                "model_name": HF_MODEL_NAME,
                "top_k": TOP_K,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
        )

        return answer, retrieved_chunks

    except (RetrievalError, VectorStoreError, LLMGenerationError):
        raise

    except Exception as error:
        raise LLMGenerationError(
            f"Failed to generate answer: {error}"
        ) from error


if __name__ == "__main__":
    question = "What is the local meal reimbursement limit?"

    answer, retrieved_chunks = generate_answer(question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        print(
            f"Source {i}: "
            f"file={metadata.get('file_name')}, "
            f"page={metadata.get('page_number')}, "
            f"chunk_id={metadata.get('chunk_id')}, "
            f"content_type={metadata.get('content_type')}, "
            f"distance={chunk['distance']:.4f}"
        )