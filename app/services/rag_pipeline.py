"""
RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  1. User query → HuggingFace embeddings → similarity search in FAISS
  2. Top-K most relevant code chunks retrieved
  3. Chunks + query sent to Groq Llama 3 via a strict prompt
  4. Answer + source file list returned
"""
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
TOP_K = 4

PROMPT_TEMPLATE = """You are a senior software engineer doing a code review.
Answer questions about a codebase using ONLY the code context provided below.
If the answer is not in the context, say: "I don't know based on the provided code."
Be concise and technical. Reference specific functions, classes, or files when relevant.
Format code snippets with triple backticks.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs):
    """Combine retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def answer_question(query: str, vectorstore: FAISS) -> dict:
    """
    Runs the full RAG pipeline for a user query against a FAISS vectorstore.
    Returns dict with 'answer' (str) and 'sources' (list of file paths).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env file. Get free key at https://console.groq.com")

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key,
        max_tokens=1024
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    # Fetch source docs separately for citations
    source_docs = retriever.invoke(query)

    # Modern LCEL chain — no deprecated RetrievalQA
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(f"Running RAG query: '{query}'")
    answer = chain.invoke(query)

    sources = list({
        doc.metadata.get("filepath", "unknown")
        for doc in source_docs
    })

    logger.info(f"Answer generated. Sources: {sources}")
    return {"answer": answer, "sources": sources}
