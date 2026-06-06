import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL_NAME,
    HF_MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    TOP_K,
)


load_dotenv()


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )

    collection = chroma_client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )

    return collection


def get_hf_client():
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found. Add it to your .env file."
        )

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )


def retrieve_context(question: str, top_k: int = TOP_K):
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

    return retrieved_chunks


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        source_label = (
            f"[Source {i} | "
            f"file={metadata.get('file_name')} | "
            f"page={metadata.get('page_number')} | "
            f"chunk_id={metadata.get('chunk_id')} | "
            f"content_type={metadata.get('content_type')}]"
        )

        context_blocks.append(
            source_label + "\n" + chunk["text"]
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are PolicyDoc AI, a RAG assistant for company policy and compliance documents.

Use ONLY the provided context to answer the question.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "I could not find this information in the provided documents."
3. Keep the answer clear and concise.
4. Include citations using the exact source labels.
5. If the answer comes from a table, mention that it comes from table content.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


def generate_answer(question: str):
    retrieved_chunks = retrieve_context(question)

    prompt = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    hf_client = get_hf_client()

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

    return answer, retrieved_chunks


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