"""
Splits large code files into smaller overlapping chunks suitable for embedding.

Strategy:
  - Uses RecursiveCharacterTextSplitter with code-aware separators
  - Chunk size  : 1000 characters
  - Chunk overlap: 200 characters (preserves context across boundaries)
  - Separators prioritise splitting at class/function boundaries first
"""
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ordered separators: try to split at logical code boundaries first
CODE_SEPARATORS = [
    "\n\nclass ",      # class definitions
    "\n\ndef ",        # function definitions
    "\n\n",            # blank lines (paragraph breaks)
    "\n",              # single newlines
    " ",               # spaces
    ""                 # character-level fallback
]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits a list of source-file Documents into smaller overlapping chunks.

    Metadata (filename, filepath, repo) is preserved on every chunk so the
    RAG pipeline can cite exact source files in its answers.

    Args:
        documents: List of LangChain Documents (one per source file)

    Returns:
        List of chunked Documents ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CODE_SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        f"Chunking complete: {len(documents)} files → {len(chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    return chunks
