"""FastAPI server for RAG chatbot."""
import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys

# MUST load .env FIRST before importing any project modules
from dotenv import load_dotenv

# Try config/.env first, then root .env
env_path = Path(__file__).parent / 'config' / '.env'
if not env_path.exists():
    env_path = Path(__file__).parent / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"[API] Loaded .env from: {env_path}")
else:
    print(f"[API] WARNING: .env file not found at {env_path}")

# Now add src to path and import rag
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from rag import rag_pipeline

app = FastAPI(
    title="Magic: The Gathering RAG Chatbot API",
    description="API for asking questions about Magic rules",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    topk: Optional[int] = 5
    model: Optional[str] = "gpt-4o-mini"


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    answer: str
    topk: int
    model: str


@app.get("/", tags=["Info"])
def root():
    """Root endpoint with API info."""
    return {
        "message": "Magic: The Gathering RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /query": "Ask a question about Magic rules",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """
    Ask a question about Magic: The Gathering rules.
    
    Returns:
        - query: The question asked
        - answer: The AI-generated answer with context
        - topk: Number of context chunks used
        - model: LLM model used
    """
    try:
        # Paths (relative to project root)
        emb_path = "data/embeddings.npy"
        meta_path = "data/metadata.jsonl"
        faiss_index_path = "data/faiss.index"
        
        # Validate files exist
        for fpath in [emb_path, meta_path]:
            if not Path(fpath).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Required file not found: {fpath}. Run setup pipeline first."
                )
        
        # Run RAG pipeline
        answer = rag_pipeline(
            query=request.query,
            emb_path=emb_path,
            meta_path=meta_path,
            faiss_index_path=faiss_index_path,
            topk=request.topk,
            model=request.model
        )
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            topk=request.topk,
            model=request.model
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
