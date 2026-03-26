from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os, mimetypes, uuid, json, asyncio, threading, queue
from datetime import datetime
from dotenv import load_dotenv
from collections import deque

load_dotenv()
print(f"🔑 GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
print(f"🔑 API Key (first 20 chars): {os.getenv('GROQ_API_KEY', 'NOT FOUND')[:20]}...")

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreManager
from services.rag_engine import RAGEngine
from services.suggest import router as suggest_router
from services.voice_transcriber import router as voice_router

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

app.include_router(suggest_router)
app.include_router(voice_router)

DATA_DIR    = os.getenv("DATA_DIR", "/home/claude/bimlo-copilot/data")
UPLOAD_DIR  = os.path.join(DATA_DIR, "uploads")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================================
# SERVER-SIDE CONVERSATION MEMORY
# Each session keeps the last 20 turns (user + assistant) in a deque.
# The frontend only needs to send a session_id — the backend owns history.
# ============================================================================

MAX_HISTORY_TURNS = 20
# session_id -> deque of {role, content} dicts
_sessions: Dict[str, deque] = {}
# session_id -> last route used (for context inheritance in router)
_session_routes: Dict[str, str] = {}
# session_id -> list of {route, query} for each turn (capped at 10)
_session_route_log: Dict[str, list] = {}

def get_history(session_id: str) -> List[dict]:
    return list(_sessions.get(session_id, []))

def get_route_log(session_id: str) -> list:
    return list(_session_route_log.get(session_id, []))

def append_turn(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_HISTORY_TURNS)
    _sessions[session_id].append({"role": role, "content": content})

def log_route(session_id: str, route: str, query: str):
    if session_id not in _session_route_log:
        _session_route_log[session_id] = []
    log = _session_route_log[session_id]
    log.append({"route": route, "query": query[:120]})
    if len(log) > 10:  # keep last 10 turns
        _session_route_log[session_id] = log[-10:]

def clear_history(session_id: str):
    _sessions.pop(session_id, None)
    _session_routes.pop(session_id, None)
    _session_route_log.pop(session_id, None)


# ============================================================================
# MODELS
# ============================================================================

class QueryRequest(BaseModel):
    query:       str
    top_k:       Optional[int] = 5
    session_id:  Optional[str] = None   # omit → new session created automatically
    force_route: Optional[str] = None   # e.g. "graph" — bypasses the LLM router


class QueryResponse(BaseModel):
    answer:     str
    sources:    List[dict]
    confidence: float
    route:      Optional[str] = None
    analytics:  Optional[Any] = None
    session_id: str                    # always returned so frontend can reuse it


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
        "routes":   ["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph"],
        "docs":     "/docs",
        "health":   "/health",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, DOCX, TXT)."""
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
    Agentic RAG query — now session-aware.

    Pass session_id to continue a conversation; omit it to start a new one.
    The server stores and manages the full conversation history.
    """
    try:
        # Resolve or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Get history accumulated so far for this session
        history = get_history(session_id)

        print(f"\n🔍 Query: {request.query} [session={session_id}, history={len(history)} turns]")

        # Run the RAG engine with server-side history
        prev_route = _session_routes.get(session_id, "")
        route_log  = get_route_log(session_id)
        result = rag_engine.query(
            request.query,
            top_k=request.top_k,
            conversation_history=history,
            prev_route=prev_route,
            route_log=route_log,
            force_route=request.force_route,
        )

        # Store this turn in server-side history (clean, no [N] citation markers)
        import re as _re
        clean_answer = _re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
        append_turn(session_id, "user",      request.query)
        append_turn(session_id, "assistant", clean_answer)
        if result.get("route"):
            _session_routes[session_id] = result["route"]
            log_route(session_id, result["route"], request.query)

        print(f"✅ Done (route={result.get('route')}, confidence={result['confidence']})")

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            route=result.get("route"),
            analytics=result.get("analytics"),
            session_id=session_id,
        )
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(500, f"Error processing query: {e}")


@app.post("/query-stream")
async def query_stream(request: QueryRequest):
    """
    Streaming version of /query using Server-Sent Events.

    Emits:
      { "type": "status",   "node": "...", "icon": "...", "message": "..." }
      { "type": "result",   "answer": "...", "sources": [...], ... }
      { "type": "error",    "message": "..." }
    """
    session_id = request.session_id or str(uuid.uuid4())
    history    = get_history(session_id)
    prev_route = _session_routes.get(session_id, "")
    route_log  = get_route_log(session_id)

    # Thread-safe queue: background thread pushes SSE events, async generator reads them
    q: queue.Queue = queue.Queue()
    DONE_SENTINEL = object()

    def status_callback(node: str, icon: str, message: str):
        q.put({"type": "status", "node": node, "icon": icon, "message": message})

    def run_query():
        try:
            result = rag_engine.query(
                request.query,
                top_k=request.top_k,
                conversation_history=history,
                prev_route=prev_route,
                route_log=route_log,
                status_callback=status_callback,
                force_route=request.force_route,
            )
            # Persist session
            import re as _re
            clean_answer = _re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
            append_turn(session_id, "user",      request.query)
            append_turn(session_id, "assistant", clean_answer)
            if result.get("route"):
                _session_routes[session_id] = result["route"]
                log_route(session_id, result["route"], request.query)
            # Strip non-JSON-serializable debug fields before putting on queue
            safe_result = {
                "answer":     result.get("answer", ""),
                "raw_answer": result.get("raw_answer") or result.get("answer", ""),
                "sources":    result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "route":      result.get("route"),
                "analytics":  result.get("analytics"),
            }
            q.put({"type": "result", "session_id": session_id, **safe_result})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(DONE_SENTINEL)

    # Run blocking RAG in a thread so we don't block the event loop
    thread = threading.Thread(target=run_query, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            # Poll queue without blocking the event loop
            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=60))
            except Exception:
                break
            if item is DONE_SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session (e.g. when user starts a new chat)."""
    clear_history(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


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


@app.get("/documents/{doc_id}/content")
async def get_document_content(doc_id: str):
    """Return the raw text content of a stored document for in-app viewing."""
    try:
        docs = vector_store.list_documents()
        doc_meta = next((d for d in docs if d["document_id"] == doc_id), None)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Document not found")

        filename  = doc_meta["filename"]
        file_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {filename}")

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            content = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                raise HTTPException(status_code=500, detail="Could not decode file")

        elif ext == ".pdf":
            try:
                import PyPDF2
                content = ""
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        content += (page.extract_text() or "") + "\n"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

        elif ext in (".docx", ".doc"):
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(file_path)
                content = "\n".join(p.text for p in doc.paragraphs)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {e}")

        else:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

        return {"document_id": doc_id, "filename": filename, "content": content}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Content fetch error: {e}")
        raise HTTPException(500, f"Error reading document: {e}")


@app.get("/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """Return the original uploaded file as binary for in-app PDF/image viewer."""
    try:
        docs = vector_store.list_documents()
        doc_meta = next((d for d in docs if d["document_id"] == doc_id), None)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Document not found")

        filename  = doc_meta["filename"]
        file_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {filename}")

        mime, _ = mimetypes.guess_type(file_path)
        return FileResponse(
            file_path,
            media_type=mime or "application/octet-stream",
            filename=filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise HTTPException(500, f"Error downloading document: {e}")


@app.get("/health")
async def health_check():
    try:
        stats = vector_store.get_collection_stats()
        return {
            "status":       "healthy",
            "timestamp":    datetime.now().isoformat(),
            "vector_store": "connected",
            "graph":        "LangGraph agentic RAG",
            "statistics":   stats,
            "active_sessions": len(_sessions),
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