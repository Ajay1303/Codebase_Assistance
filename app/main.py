"""
Entry point for the Codebase Q&A Assistant FastAPI application.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Ensure required data directories exist at startup
os.makedirs(os.getenv("REPOS_DIR", "./data/repos"), exist_ok=True)
os.makedirs(os.getenv("VECTORSTORE_DIR", "./data/vectorstores"), exist_ok=True)

app = FastAPI(
    title="Codebase Q&A Assistant",
    description="Ask natural language questions about any public GitHub repository using RAG + Groq Llama 3.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow Streamlit frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all API routes under /api prefix
app.include_router(router, prefix="/api")


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Codebase Q&A Assistant is live!",
        "docs": "/docs"
    }
