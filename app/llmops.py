import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exceptions import PromptTemplateError
from logger import get_logger


logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Lightweight token estimate.

    This is not exact tokenizer-based counting.
    It is a simple approximation useful for local logging.

    Rough practical approximation:
    1 token ≈ 0.75 words for English-like text.
    """
    if not text:
        return 0

    word_count = len(text.split())
    return int(word_count / 0.75)


def load_prompt_template(prompt_file_path: Path) -> str:
    """
    Load versioned prompt template from the prompts/ folder.
    """
    try:
        if not prompt_file_path.exists():
            raise PromptTemplateError(
                f"Prompt template not found at: {prompt_file_path}"
            )

        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt_template = file.read()

        if "{context}" not in prompt_template or "{question}" not in prompt_template:
            raise PromptTemplateError(
                "Prompt template must contain {context} and {question} placeholders."
            )

        logger.info(f"Prompt template loaded: {prompt_file_path.name}")
        return prompt_template

    except PromptTemplateError:
        raise

    except Exception as error:
        raise PromptTemplateError(
            f"Failed to load prompt template: {error}"
        ) from error


def build_context_from_sources(retrieved_sources: list[dict[str, Any]]) -> str:
    """
    Build a clean context block from retrieved chunks.

    Each source is formatted with citation metadata so the LLM
    can refer to Source 1, Source 2, etc.
    """
    context_parts = []

    for i, source in enumerate(retrieved_sources, start=1):
        metadata = source.get("metadata", {})
        text = source.get("text", "")

        file_name = metadata.get("file_name", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")
        content_type = metadata.get("content_type", "unknown")

        source_header = (
            f"[Source {i} | file={file_name} | page={page_number} | "
            f"chunk_id={chunk_id} | content_type={content_type}]"
        )

        context_parts.append(f"{source_header}\n{text}")

    return "\n\n".join(context_parts)


def log_rag_run(
    log_path: Path,
    question: str,
    answer: str,
    retrieved_sources: list[dict[str, Any]],
    prompt_version: str,
    app_version: str,
    start_time: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save one RAG run as JSONL.

    Each line represents one user query and its system behavior.
    This supports lightweight LLMOps-style tracing.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        end_time = time.time()
        latency_seconds = round(end_time - start_time, 4)

        retrieved_metadata = []

        for source in retrieved_sources:
            metadata = source.get("metadata", {})

            retrieved_metadata.append(
                {
                    "file_name": metadata.get("file_name"),
                    "page_number": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_id"),
                    "content_type": metadata.get("content_type"),
                    "distance": source.get("distance"),
                }
            )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "app_version": app_version,
            "prompt_version": prompt_version,
            "question": question,
            "answer": answer,
            "latency_seconds": latency_seconds,
            "estimated_question_tokens": estimate_tokens(question),
            "estimated_answer_tokens": estimate_tokens(answer),
            "retrieved_source_count": len(retrieved_sources),
            "retrieved_sources": retrieved_metadata,
        }

        if extra:
            record["extra"] = extra

        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            f"RAG run logged | latency={latency_seconds}s | "
            f"sources={len(retrieved_sources)} | prompt_version={prompt_version}"
        )

    except Exception as error:
        # Logging should not crash the main RAG app.
        logger.warning(f"Failed to write RAG run log: {error}")