class PolicyDocAIException(Exception):
    """
    Base exception class for PolicyDoc AI.

    All custom project exceptions inherit from this class.
    This makes debugging easier because project-specific errors
    can be handled separately from generic Python/library errors.
    """

    pass


class UnsupportedFileTypeError(PolicyDocAIException):
    """
    Raised when the uploaded file type is not supported.
    Example: user uploads .jpg or .docx when only .txt and .pdf are supported.
    """

    pass


class DocumentLoadingError(PolicyDocAIException):
    """
    Raised when the system fails to load or parse a document.
    Example: corrupted PDF, unreadable file, extraction failure.
    """

    pass


class ChunkingError(PolicyDocAIException):
    """
    Raised when document chunking fails.
    Example: empty documents, invalid chunk size, splitter failure.
    """

    pass


class EmbeddingCreationError(PolicyDocAIException):
    """
    Raised when embedding creation fails.
    Example: embedding model load failure or empty chunk list.
    """

    pass


class VectorStoreError(PolicyDocAIException):
    """
    Raised when ChromaDB storage or retrieval fails.
    Example: collection error, add/query failure.
    """

    pass


class RetrievalError(PolicyDocAIException):
    """
    Raised when retrieval fails or returns invalid results.
    """

    pass


class PromptTemplateError(PolicyDocAIException):
    """
    Raised when prompt template loading or formatting fails.
    Example: missing prompt file or missing {context}/{question} placeholders.
    """

    pass


class LLMGenerationError(PolicyDocAIException):
    """
    Raised when the LLM API call or answer generation fails.
    Example: missing API token, API timeout, provider error.
    """

    pass