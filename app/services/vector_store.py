"""
Builds and persists FAISS vector stores for each processed repository.

Each repo gets its own sub-directory inside VECTORSTORE_DIR so multiple
repos can be indexed and queried independently.
"""
import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/vectorstores")


def _store_path(repo_name: str) -> str:
    """Returns the filesystem path for a repo's FAISS index directory."""
    return str(Path(VECTORSTORE_DIR) / repo_name)


def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    repo_name: str
) -> FAISS:
    """
    Embeds all chunks and saves a FAISS index to disk.

    Args:
        chunks    : Chunked LangChain Documents (with metadata)
        embeddings: HuggingFace embedding model instance
        repo_name : Used as the directory name for this repo's index

    Returns:
        The in-memory FAISS vectorstore (already saved to disk)
    """
    path = _store_path(repo_name)
    os.makedirs(path, exist_ok=True)

    logger.info(f"Building FAISS index for '{repo_name}' with {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(path)

    logger.info(f"FAISS index saved → {path}")
    return vectorstore


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    repo_name: str
) -> FAISS | None:
    """
    Loads a previously saved FAISS index from disk.

    Args:
        embeddings: Must be the exact same model used during build_vectorstore
        repo_name : Repository whose index to load

    Returns:
        FAISS vectorstore, or None if no index exists for this repo
    """
    path = _store_path(repo_name)

    if not Path(path).exists():
        logger.warning(f"No FAISS index found at '{path}'")
        return None

    logger.info(f"Loading FAISS index from '{path}'")
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # Safe: we wrote this file ourselves
    )
