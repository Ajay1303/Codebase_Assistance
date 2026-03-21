"""
Provides a HuggingFace sentence-transformer embedding model.

Model  : all-MiniLM-L6-v2  (~80 MB, downloaded once and cached)
Device : CPU (works everywhere, no GPU required)
Cost   : Completely free — no API key needed
"""
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache so the model is only loaded once per process
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance.

    The model is downloaded from HuggingFace Hub on the first call
    (~80 MB) and reused for all subsequent calls within the same process.

    Returns:
        HuggingFaceEmbeddings object ready for encoding
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' (first time may take ~30s)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded and cached.")

    return _embeddings_instance
