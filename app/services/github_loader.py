"""
Handles cloning a public GitHub repository and loading all valid source files.

Supported file types : .py .js .ts .java .cpp .c .go .rs
Ignored directories  : .git node_modules venv dist __pycache__ build .idea
"""
import os
import shutil
import logging
from pathlib import Path
from git import Repo, GitCommandError
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REPOS_DIR = os.getenv("REPOS_DIR", "./data/repos")

# File extensions we want to embed
VALID_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}

# Directories to skip entirely
IGNORED_DIRS = {
    ".git", "node_modules", "venv", "dist",
    "__pycache__", "build", ".idea", ".vscode",
    "env", ".env", "site-packages"
}


def clone_repository(repo_url: str) -> tuple[list[Document], str]:
    """
    Clones a GitHub repository locally and loads all valid source files.

    Args:
        repo_url: Full public GitHub URL, e.g. https://github.com/user/repo

    Returns:
        Tuple of:
          - List of LangChain Document objects (one per file)
          - repo_name string (last segment of the URL)

    Raises:
        ValueError: For invalid URLs, failed clones, or empty repos
    """
    # Basic URL validation
    if not repo_url.startswith("https://github.com/"):
        raise ValueError(
            "Invalid GitHub URL. Must start with https://github.com/"
        )

    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_path = Path(REPOS_DIR) / repo_name

    # Remove any existing clone so we always start fresh
    if clone_path.exists():
        shutil.rmtree(clone_path)

    try:
        logger.info(f"Cloning {repo_url} → {clone_path}")
        Repo.clone_from(repo_url, str(clone_path), depth=1)  # shallow clone = faster
    except GitCommandError as e:
        raise ValueError(
            f"Failed to clone repository. Make sure it's public and the URL is correct. "
            f"Details: {str(e)}"
        )

    documents = _load_code_files(clone_path, repo_name)

    if not documents:
        raise ValueError(
            "Repository has no supported code files "
            f"({', '.join(VALID_EXTENSIONS)})."
        )

    logger.info(f"Loaded {len(documents)} files from '{repo_name}'")
    return documents, repo_name


def _load_code_files(base_path: Path, repo_name: str) -> list[Document]:
    """
    Recursively reads every valid code file under base_path.

    Args:
        base_path : Root of the cloned repository
        repo_name : Repository name (stored in Document metadata)

    Returns:
        List of LangChain Documents, each containing one file's content.
    """
    documents = []

    for file_path in base_path.rglob("*"):
        # Skip directories
        if file_path.is_dir():
            continue

        # Skip files inside ignored directories
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue

        # Skip unsupported extensions
        if file_path.suffix not in VALID_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Skip empty files
            if not content.strip():
                continue

            relative_path = str(file_path.relative_to(base_path))

            documents.append(Document(
                page_content=content,
                metadata={
                    "filename": file_path.name,
                    "filepath": relative_path,
                    "repo": repo_name,
                    "extension": file_path.suffix
                }
            ))

            logger.debug(f"Loaded: {relative_path}")

        except Exception as e:
            logger.warning(f"Skipping {file_path} — could not read: {e}")

    return documents
