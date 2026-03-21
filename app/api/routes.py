"""
API route definitions for the Codebase Q&A Assistant.

Endpoints:
  POST /api/upload  — Clone a GitHub repo, embed it, store in FAISS
  POST /api/ask     — Ask a natural language question about a processed repo
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.github_loader import clone_repository
from app.services.chunking import chunk_documents
from app.services.embeddings import get_embeddings
from app.services.vector_store import build_vectorstore, load_vectorstore
from app.services.rag_pipeline import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response Models

class UploadRequest(BaseModel):
    repo_url: str  # e.g. "https://github.com/tiangolo/fastapi"


class UploadResponse(BaseModel):
    message: str
    repo_name: str
    files_processed: int
    chunks_created: int


class AskRequest(BaseModel):
    query: str       # Natural language question
    repo_name: str   # Must match the repo_name returned by /upload


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Routes 

@router.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
def upload_repo(request: UploadRequest):
    """
    Clones a public GitHub repository, extracts code files,
    generates embeddings, and stores them in a FAISS vector database.

    - Input : GitHub URL (must be public)
    - Output: Repo name, number of files and chunks processed
    """
    try:
        logger.info(f"Starting ingestion for: {request.repo_url}")

        # Step 1: Clone repo and load code files
        documents, repo_name = clone_repository(request.repo_url)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No valid code files found in this repository."
            )

        # Step 2: Chunk documents
        chunks = chunk_documents(documents)

        # Step 3: Generate embeddings and build FAISS vector store
        embeddings = get_embeddings()
        build_vectorstore(chunks, embeddings, repo_name)

        logger.info(f"Ingestion complete: {len(documents)} files, {len(chunks)} chunks")

        return UploadResponse(
            message="Repository processed successfully.",
            repo_name=repo_name,
            files_processed=len(documents),
            chunks_created=len(chunks)
        )

    except HTTPException:
        raise  # Re-raise HTTP errors as-is
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(request: AskRequest):
    """
    Answers a natural language question about a previously processed repository
    using Retrieval-Augmented Generation (RAG) with Groq Llama 3.

    - Input : Query string + repo_name (from /upload response)
    - Output: LLM-generated answer + list of source file paths
    """
    try:
        logger.info(f"Question received for repo '{request.repo_name}': {request.query}")

        # Load embeddings and existing FAISS store
        embeddings = get_embeddings()
        vectorstore = load_vectorstore(embeddings, request.repo_name)

        if vectorstore is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No vector store found for repo '{request.repo_name}'. "
                    f"Please call /upload first."
                )
            )

        # Run RAG pipeline
        result = answer_question(request.query, vectorstore)

        return AskResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
