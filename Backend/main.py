from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
print(f"🔑 GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
print(f"🔑 API Key (first 20 chars): {os.getenv('GROQ_API_KEY', 'NOT FOUND')[:20]}...")

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreManager
from services.rag_engine import RAGEngine

app = FastAPI(
    title="BIMLO Copilot Télécom API",
    version="3.0.0",
    description="Agentic RAG API powered by LangGraph — routes, retrieves, iterates, analyses."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "*",  # REMOVE IN PRODUCTION
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_processor = DocumentProcessor()
vector_store  = VectorStoreManager()
rag_engine    = RAGEngine(vector_store)

DATA_DIR    = os.getenv("DATA_DIR", "/home/claude/bimlo-copilot/data")
UPLOAD_DIR  = os.path.join(DATA_DIR, "uploads")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer:     str
    sources:    List[dict]
    confidence: float
    route:      Optional[str] = None   # which LangGraph branch was taken
    analytics:  Optional[Any] = None   # populated for analytics queries


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service":  "BIMLO Copilot Télécom API",
        "version":  "3.0.0",
        "status":   "running",
        "graph":    "LangGraph agentic RAG",
        "routes":   ["direct", "rag", "iterative_rag", "analytics"],
        "docs":     "/docs",
        "health":   "/health",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a telecom document (PDF, DOCX, TXT)."""
    try:
        allowed = ['.pdf', '.docx', '.doc', '.txt']
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed:
            raise HTTPException(400, f"Unsupported type. Allowed: {', '.join(allowed)}")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"📄 Saved: {file.filename} ({len(content)} bytes)")

        chunks = doc_processor.process_document(file_path)
        print(f"✂️  {len(chunks)} chunks created")

        doc_id = vector_store.add_document(file.filename, chunks)
        print(f"✅ Indexed: {doc_id}")

        return {
            "status":           "success",
            "filename":         file.filename,
            "document_id":      doc_id,
            "chunks_processed": len(chunks),
            "message":          f"'{file.filename}' processed and indexed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(500, f"Error processing document: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Agentic RAG query endpoint.

    The LangGraph agent automatically selects the best strategy:
    - **direct**        – no retrieval needed (greetings, meta)
    - **rag**           – single retrieval pass
    - **iterative_rag** – multi-round retrieval with query rewriting
    - **analytics**     – structured analytics from documents
    """
    try:
        print(f"\n🔍 Query: {request.query}")
        result = rag_engine.query(request.query, top_k=request.top_k)
        print(f"✅ Done (route={result.get('route')}, confidence={result['confidence']})")
        return QueryResponse(**result)
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(500, f"Error processing query: {e}")


@app.post("/generate-report")
async def generate_report(request: QueryRequest):
    """Generate and persist a structured report."""
    try:
        print(f"\n📊 Report: {request.query}")
        report_data = rag_engine.generate_report(request.query)

        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.json"
        report_path     = os.path.join(REPORTS_DIR, report_filename)

        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Report saved: {report_filename}")
        return {
            "status":      "success",
            "report":      report_data,
            "report_file": report_filename,
            "timestamp":   timestamp,
        }
    except Exception as e:
        print(f"❌ Report error: {e}")
        raise HTTPException(500, f"Error generating report: {e}")


@app.get("/documents")
async def list_documents():
    try:
        docs = vector_store.list_documents()
        return {"status": "success", "documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(500, f"Error listing documents: {e}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        vector_store.delete_document(doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except Exception as e:
        raise HTTPException(500, f"Error deleting document: {e}")


@app.get("/health")
async def health_check():
    try:
        stats = vector_store.get_collection_stats()
        return {
            "status":     "healthy",
            "timestamp":  datetime.now().isoformat(),
            "vector_store": "connected",
            "graph":      "LangGraph agentic RAG",
            "statistics": stats,
        }
    except Exception as e:
        return {"status": "degraded", "timestamp": datetime.now().isoformat(), "error": str(e)}


# ============================================================================
# LIFECYCLE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*60)
    print("🚀 BIMLO Copilot Télécom API v3 — LangGraph Agentic RAG")
    print("="*60)
    print(f"📁 Data dir:    {DATA_DIR}")
    print(f"📁 Uploads:     {UPLOAD_DIR}")
    print(f"📄 Reports:     {REPORTS_DIR}")
    print(f"🕸️  Graph routes: direct | rag | iterative_rag | analytics")
    try:
        s = vector_store.get_collection_stats()
        print(f"📊 Vector store: {s['total_documents']} docs, {s['total_chunks']} chunks")
    except:
        print("⚠️  Vector store stats unavailable")
    print(f"🔑 Groq API: {'✅ configured' if os.getenv('GROQ_API_KEY') else '⚠️  not configured'}")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n👋 BIMLO Copilot API — shutting down\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")