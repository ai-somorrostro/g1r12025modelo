"""FastAPI server for RAG chatbot."""
import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import subprocess

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

# Add CORS middleware to allow web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str
    topk: Optional[int] = 5


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    answer: str
    topk: int


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


@app.post("/setup", tags=["Setup"])
def setup():
    """
    Initialize the RAG pipeline step by step using the existing scripts:
    1. Chunk PDF with chunk_ruler.py
    2. Generate embeddings with embed_chunks.py
    3. Build FAISS index with build_faiss.py
    
    Returns:
        - status: Success or error message with details
    """
    try:
        app_dir = Path(__file__).parent
        data_dir = app_dir / 'data'
        pdf_dir = data_dir / 'pdf'
        
        # Create directories
        data_dir.mkdir(exist_ok=True)
        pdf_dir.mkdir(exist_ok=True)
        
        chunks_path = data_dir / 'chunks_normas.jsonl'
        embeddings_path = data_dir / 'embeddings.npy'
        faiss_index_path = data_dir / 'faiss.index'
        pdf_path = pdf_dir / 'magic_rules_clean.pdf'
        
        steps_completed = []
        
        # Step 1: Chunk the PDF
        print("[SETUP] Step 1: Chunking PDF...")
        if not chunks_path.exists():
            if not pdf_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF file not found: {pdf_path}. Ensure data/pdf/magic_rules_clean.pdf exists."
                )
            
            print(f"  → Running: python src/chunk_ruler.py --input {pdf_path} --output {chunks_path} --definitions-start 275")
            result = subprocess.run(
                [
                    "python",
                    str(app_dir / "src" / "chunk_ruler.py"),
                    "--input", str(pdf_path),
                    "--output", str(chunks_path),
                    "--definitions-start", "275"
                ],
                cwd=str(app_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                error_msg = f"chunk_ruler.py failed: {result.stderr}"
                print(f"  ✗ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
            
            print(f"  ✓ Chunks created: {chunks_path}")
            steps_completed.append(f"Chunked PDF into {chunks_path}")
        else:
            print(f"  ✓ Chunks already exist at {chunks_path}")
            steps_completed.append(f"Chunks already exist: {chunks_path}")
        
        # Count chunks
        with open(chunks_path, 'r') as f:
            chunk_count = sum(1 for _ in f)
        print(f"  → Total chunks: {chunk_count}")
        
        # Step 2: Generate embeddings
        print("[SETUP] Step 2: Generating embeddings...")
        if not embeddings_path.exists():
            print(f"  → Running: python src/embed_chunks.py --input {chunks_path} --out-dir {data_dir}")
            result = subprocess.run(
                [
                    "python",
                    str(app_dir / "src" / "embed_chunks.py"),
                    "--input", str(chunks_path),
                    "--out-dir", str(data_dir)
                ],
                cwd=str(app_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                error_msg = f"embed_chunks.py failed: {result.stderr}"
                print(f"  ✗ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
            
            print(f"  ✓ Embeddings created: {embeddings_path}")
            steps_completed.append(f"Generated embeddings: {embeddings_path}")
        else:
            print(f"  ✓ Embeddings already exist at {embeddings_path}")
            steps_completed.append(f"Embeddings already exist: {embeddings_path}")
        
        # Verify embeddings file
        if embeddings_path.exists():
            emb_shape = np.load(str(embeddings_path)).shape
            print(f"  → Embeddings shape: {emb_shape}")
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Embeddings file was not created: {embeddings_path}"
            )
        
        # Step 3: Build FAISS index
        print("[SETUP] Step 3: Building FAISS index...")
        if not faiss_index_path.exists():
            print(f"  → Running: python src/build_faiss.py --emb {embeddings_path} --output {faiss_index_path}")
            result = subprocess.run(
                [
                    "python",
                    str(app_dir / "src" / "build_faiss.py"),
                    "--emb", str(embeddings_path),
                    "--output", str(faiss_index_path)
                ],
                cwd=str(app_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                error_msg = f"build_faiss.py failed: {result.stderr}"
                print(f"  ✗ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
            
            print(f"  ✓ FAISS index created: {faiss_index_path}")
            steps_completed.append(f"Built FAISS index: {faiss_index_path}")
        else:
            print(f"  ✓ FAISS index already exists at {faiss_index_path}")
            steps_completed.append(f"FAISS index already exists: {faiss_index_path}")
        
        # Final verification
        print("[SETUP] Step 4: Verifying all files...")
        required_files = [chunks_path, embeddings_path, faiss_index_path]
        missing_files = []
        for fpath in required_files:
            if not fpath.exists():
                missing_files.append(str(fpath))
            else:
                print(f"  ✓ {fpath.name} OK ({fpath.stat().st_size} bytes)")
        
        if missing_files:
            raise HTTPException(
                status_code=500,
                detail=f"Setup incomplete. Missing files: {missing_files}"
            )
        
        steps_completed.append("All files verified successfully")
        
        print("[SETUP] ✓ Setup completed successfully!")
        return {
            "status": "success",
            "message": "RAG pipeline initialized successfully",
            "steps": steps_completed,
            "files": {
                "chunks": str(chunks_path),
                "embeddings": str(embeddings_path),
                "faiss_index": str(faiss_index_path)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SETUP] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")




@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """
    Ask a question about Magic: The Gathering rules.
    
    Returns:
        - query: The question asked
        - answer: The AI-generated answer with context
        - topk: Number of context chunks used
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
        
        # Run RAG pipeline (always use gpt-4o-mini)
        answer = rag_pipeline(
            query=request.query,
            emb_path=emb_path,
            meta_path=meta_path,
            faiss_index_path=faiss_index_path,
            topk=request.topk,
            model="gpt-4o-mini"
        )
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            topk=request.topk
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
