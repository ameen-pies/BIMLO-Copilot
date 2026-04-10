from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os, mimetypes, uuid, json, asyncio, threading, queue
from datetime import datetime
from dotenv import load_dotenv
from collections import deque
from neo4j_auth import router as auth_router, init_neo4j

load_dotenv()
groq_ok = bool(os.getenv("GROQ_API_KEY"))
cf_ok   = bool(os.getenv("CF_API_KEY"))
print(f"🔑 Groq API: {'✅ configured' if groq_ok else '❌ missing GROQ_API_KEY'}")
print(f"🔑 CF API:   {'✅ configured' if cf_ok   else '⚠️  missing CF_API_KEY (Groq-only mode)'}")

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreManager
from services.rag_engine import RAGEngine
from services.suggest import router as suggest_router
from services.voice_transcriber import router as voice_router
from services.voice_call import router as voice_call_router
from services.autocomplete import router as autocomplete_router
from services.report_agent import (
    router as report_router,
    SharedContext,
)
# SharedContext also exposes set_analytics — call it after every query
# so the report agent can embed charts generated in analytics routes.

# News pipeline — scheduled global cache (every 4 days)
try:
    from news_pipeline import (
        run_news_pipeline,
        pipeline_is_running,
        get_meta,
        get_page,
        get_status as _pipeline_get_status,
        CACHE_DIR as NEWS_CACHE_DIR,
    )
    _news_pipeline_available = True
except ImportError:
    _news_pipeline_available = False
    print("⚠️  news_pipeline not found — /api/news endpoints will return 503")

# News chat agent — standalone, isolated from RAG engine
try:
    from news_chat_agent import router as news_chat_router
    _news_chat_available = True
    print("✅ news_chat_agent loaded — /api/news/chat ready")
except ImportError:
    news_chat_router     = None
    _news_chat_available = False
    print("⚠️  news_chat_agent not found — /api/news/chat will 503")

try:
    from cad_ifc_agent import router as cad_ifc_router
    _cad_ifc_available = True
    print("✅ cad_ifc_agent loaded — /api/cad/upload + /api/cad/query ready")

    # Register this main module under its own name so cad_context_bridge can
    # find append_turn / get_history via sys.modules["main"] at runtime.
    import sys as _sys
    if "main" not in _sys.modules:
        _sys.modules["main"] = _sys.modules[__name__]

except ImportError as _e:
    cad_ifc_router     = None
    _cad_ifc_available = False
    print(f"⚠️  cad_ifc_agent not found — /api/cad endpoints will 503 ({_e})")

app = FastAPI(
    title="BIMLO Copilot Télécom API",
    version="3.0.0",
    description="Agentic RAG API powered by LangGraph — routes, retrieves, iterates, analyses."
)

app.include_router(auth_router)

@app.on_event("startup")
async def startup_event():
    init_neo4j()
    
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

# Give the report agent direct VS access for fallback retrieval when
# SharedContext has no cached chunks for a session (e.g. timing race).
SharedContext.set_vector_store(vector_store)

app.include_router(suggest_router)
app.include_router(voice_router)
app.include_router(voice_call_router)
app.include_router(autocomplete_router)
app.include_router(report_router)
if _news_chat_available:
    app.include_router(news_chat_router)
if _cad_ifc_available:
    app.include_router(cad_ifc_router)

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
    top_k:       Optional[int]  = 5
    session_id:  Optional[str]  = None   # omit → new session created automatically
    force_route: Optional[str]  = None   # e.g. "graph" — bypasses the LLM router
    voice_mode:  Optional[bool] = False  # True → skip citation check, source formatting, iterative loops


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
            voice_mode=request.voice_mode,
        )

        # Store this turn in server-side history (clean, no [N] citation markers)
        import re as _re
        clean_answer = _re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
        append_turn(session_id, "user", request.query)

        if result.get("report_id") and result.get("report_title"):
            from services.report_agent import _reports_store as _rs
            _rpt = _rs.get(result["report_id"], {})
            _content = _rpt.get("content") or ""
            _source_docs = ", ".join(_rpt.get("source_docs") or []) or "unknown"
            _word_count = len(_content.split())
            _section_count = len(_re.findall(r'^#{1,2}\s', _content, _re.M))
            _preview = _content[:400].strip()
            memory_note = (
                f"{clean_answer}\n\n"
                f"[REPORT GENERATED]\n"
                f"Title: {result['report_title']}\n"
                f"Sources: {_source_docs}\n"
                f"Size: {_section_count} sections, {_word_count} words\n"
                f"Opening: {_preview}"
            )
            append_turn(session_id, "assistant", memory_note)
        else:
            append_turn(session_id, "assistant", clean_answer)

        if result.get("route"):
            _session_routes[session_id] = result["route"]
            log_route(session_id, result["route"], request.query)
        # Keep ReportAgent in sync with latest history + retrieved chunks
        SharedContext.set_history(session_id, get_history(session_id))
        SharedContext.set_chunks(session_id, result.get("retrieved_chunks", []))

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
                session_id=session_id,
                voice_mode=request.voice_mode,
            )
            # Persist session
            import re as _re
            clean_answer = _re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
            append_turn(session_id, "user", request.query)

            # If a report was generated, write a structured memory note into history
            # so every future node (direct, rag, etc.) knows what was produced.
            if result.get("report_id") and result.get("report_title"):
                from services.report_agent import _reports_store as _rs
                _rpt = _rs.get(result["report_id"], {})
                _content = _rpt.get("content") or ""
                _source_docs = ", ".join(_rpt.get("source_docs") or []) or "unknown"
                _word_count = len(_content.split())
                _section_count = len(_re.findall(r'^#{1,2}\s', _content, _re.M))
                _preview = _content[:400].strip()
                memory_note = (
                    f"{clean_answer}\n\n"
                    f"[REPORT GENERATED]\n"
                    f"Title: {result['report_title']}\n"
                    f"Sources: {_source_docs}\n"
                    f"Size: {_section_count} sections, {_word_count} words\n"
                    f"Opening: {_preview}"
                )
                append_turn(session_id, "assistant", memory_note)
            else:
                append_turn(session_id, "assistant", clean_answer)

            if result.get("route"):
                _session_routes[session_id] = result["route"]
                log_route(session_id, result["route"], request.query)
            # Keep ReportAgent in sync with latest history + retrieved chunks
            SharedContext.set_history(session_id, get_history(session_id))
            SharedContext.set_chunks(session_id, result.get("retrieved_chunks", []))
            SharedContext.set_analytics(session_id, result.get("analytics"))
            # Strip non-JSON-serializable debug fields before putting on queue
            safe_result = {
                "answer":       result.get("answer", ""),
                "raw_answer":   result.get("raw_answer") or result.get("answer", ""),
                "sources":      result.get("sources", []),
                "confidence":   result.get("confidence", 0.0),
                "route":        result.get("route"),
                "analytics":    result.get("analytics"),
                "report_id":    result.get("report_id"),
                "report_title": result.get("report_title"),
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
                item = await loop.run_in_executor(None, lambda: q.get(timeout=300))
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



@app.post("/title")
async def generate_title(request: dict):
    """
    Generate a short, smart title for a conversation or report.
    Body: { "type": "chat" | "report", "text": "...", "messages": [...] }
    Returns: { "title": "..." }
    """
    from llm_client import call_llm

    title_type = request.get("type", "chat")
    text       = (request.get("text") or "").strip()
    messages   = request.get("messages", [])

    if title_type == "chat":
        # Build a short excerpt from the first few messages
        excerpt = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:150]}"
            for m in messages[:6]
        )
        system = (
            "You generate ultra-short conversation titles. "
            "Rules: 3-6 words, Title Case, NO quotes, NO punctuation at the end. "
            "Capture the TOPIC or ACTION — not the phrasing. "
            "Good: 'Telecom Site Survey Analysis', 'Network Coverage Question', 'Hello and Greeting', 'Revenue Chart Request'. "
            "Bad: 'Make me a report on', 'What is the', 'User asked about'. "
            "Reply with ONLY the title, nothing else."
        )
        prompt = f"Conversation:\n{excerpt}\n\nTitle:"
    else:
        # Report title
        system = (
            "You generate short, specific report titles. "
            "Rules: 4-8 words, Title Case, NO quotes, NO punctuation at the end. "
            "Must name the real subject/entity/metric — never start with 'Report', 'Analysis', 'Summary'. "
            "Good: 'Telecom Site Survey Field Results', 'Network Infrastructure Cost Breakdown', 'Q3 Revenue by Region'. "
            "Bad: 'Report on telecom_site_survey.txt', 'Analysis of the document', 'Summary Report'. "
            "Reply with ONLY the title, nothing else."
        )
        prompt = f"Report request: \"{text}\"\n\nTitle:"

    try:
        raw = call_llm(prompt, system_prompt=system, max_tokens=30, temperature=0.3, task="classify")
        title = raw.strip().strip('"').strip("'").strip()
        if not title or len(title) > 100:
            raise ValueError("bad title")
        return {"title": title}
    except Exception as e:
        print(f"⚠️  /title error: {e}")
        return {"title": ""}


@app.post("/intent/report")
async def report_intent_check(request: dict):
    """
    Smart intent check — returns {wants_report: bool, explicit_docs: [str]}.

    Detection is a 3-layer pipeline:
      Layer 1 — Instant regex (zero latency, catches obvious patterns)
      Layer 2 — LLM classifier with robust JSON parsing
      Layer 3 — Keyword fallback (catches any LLM parse failures)

    Extracts explicit doc mentions so the report agent can target those files.
    """
    from llm_client import call_llm
    import re as _re
    import json as _json
    import ast as _ast

    text = (request.get("text") or "").strip()
    if not text:
        return {"wants_report": False, "explicit_docs": []}

    available_docs: List[str] = request.get("available_docs", [])

    # ── Layer 1: Instant regex fast-path ─────────────────────────────────────
    # Catches the most common patterns across EN/FR without any LLM call.
    # If this fires we still call the LLM — but only for file extraction,
    # skipping the wants_report classification entirely.
    REPORT_RE = _re.compile(
        r'\b(?:'
        # English: action + report
        r'(?:make|create|generate|write|build|produce|prepare|draft|do|give\s+me|get\s+me)'
        r'\s+(?:me\s+)?(?:a\s+|an\s+)?(?:full\s+|detailed\s+|brief\s+)?report'
        r'|report\s+(?:on|about|for|regarding|from)'
        r'|(?:make|create|generate|write|do)\s+(?:me\s+)?(?:a\s+)?(?:summary\s+)?document'
        r'|generate\s+(?:a\s+)?(?:pdf|word\s+doc|summary\s+report)'
        r'|download\s+(?:a\s+)?report'
        # French
        r'|rapport\s+sur'
        r'|fais?\s+(?:moi\s+)?(?:un\s+)?rapport'
        r'|cr[eé]e?\s+(?:un\s+)?rapport'
        r'|g[eé]n[eè]re?\s+(?:un\s+)?rapport'
        r')',
        _re.IGNORECASE,
    )
    regex_hit = bool(REPORT_RE.search(text))

    # ── Layer 2: LLM classifier ───────────────────────────────────────────────
    def _resolve_docs(mentioned: list) -> list:
        """Match partial doc mentions against available_docs."""
        explicit: list = []
        for mention in (mentioned or []):
            ml = mention.lower().strip()
            if not ml:
                continue
            for doc in available_docs:
                dl = doc.lower()
                if ml in dl or dl in ml or _re.search(_re.escape(ml), dl):
                    if doc not in explicit:
                        explicit.append(doc)
        return explicit

    def _parse_llm_json(raw: str) -> Optional[dict]:
        """Robustly extract JSON from LLM response (fences, single quotes, etc.)."""
        if not raw:
            return None
        clean = _re.sub(r"```(?:json)?|```", "", raw).strip()

        # Try standard JSON
        try:
            result = _json.loads(clean)
            if isinstance(result, dict):
                return result
        except (_json.JSONDecodeError, ValueError):
            pass

        # Try extracting first {...} block (LLM sometimes adds preamble)
        m = _re.search(r'\{[\s\S]*?\}', clean)
        if m:
            try:
                result = _json.loads(m.group(0))
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        # Try Python repr (single quotes, True/False/None)
        try:
            result = _ast.literal_eval(clean)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        return None

    try:
        system = (
            "You are an intent and entity classifier for a document Q&A assistant. "
            "Reply with ONLY a raw JSON object — no markdown fences, no explanation whatsoever. "
            'Exact format: {"wants_report": true, "mentioned_files": ["filename.pdf"]} '
            "Set wants_report=true ONLY when the user explicitly asks for a report, document, "
            "summary document, or structured written output to download or read later. "
            "TRUE examples: 'make a report on X', 'generate a report about X', "
            "'create a PDF report', 'write a document summarising Y', 'do a report on file Z'. "
            "FALSE examples: plain questions, 'what is X?', 'show me a chart', 'summarise this' "
            "(quick summary, not a standalone document). "
            "mentioned_files: list any filenames or file references the user typed. "
            "Use partial names too (e.g. 'survey' if they said 'the survey file'). "
            "Return [] if no files mentioned."
        )
        doc_list = ", ".join(available_docs[:15]) if available_docs else "(none uploaded)"
        prompt = (
            f"Available documents: {doc_list}\n\n"
            f'User message: """{text}"""'
        )

        raw_answer = call_llm(
            prompt=prompt,
            system_prompt=system,
            max_tokens=150,
            temperature=0.0,
            task="classify",
        )

        parsed = _parse_llm_json(raw_answer)

        if isinstance(parsed, dict):
            # If regex already confirmed a report request, trust it over LLM
            wants = bool(parsed.get("wants_report", False)) or regex_hit
            explicit_docs = _resolve_docs(parsed.get("mentioned_files", []))
            print(
                f"📋 intent/report → {'✅ YES' if wants else '❌ NO '} "
                f"[regex={'hit' if regex_hit else 'miss'}, llm={parsed.get('wants_report')}] "
                f"| files={explicit_docs} | '{text[:60]}'"
            )
            return {"wants_report": wants, "explicit_docs": explicit_docs}

        # ── Layer 3: Keyword fallback (LLM parse failed) ──────────────────────
        # At this point regex_hit is the most reliable signal.
        if regex_hit:
            # Still try to extract file mentions from the raw answer or the message
            # by matching available_docs directly against the user text
            explicit_docs = _resolve_docs(
                [w for w in _re.split(r'[\s,]+', text) if len(w) > 3]
            )
            print(
                f"📋 intent/report → ✅ YES [regex=hit, llm-parse-failed] "
                f"| files={explicit_docs} | '{text[:60]}'"
            )
            return {"wants_report": True, "explicit_docs": explicit_docs}

        # Also check raw_answer for any YES-like word before giving up
        wants_from_text = bool(_re.search(
            r'\b(?:true|yes|oui|report|document)\b', (raw_answer or ""), _re.I
        ))
        print(
            f"📋 intent/report → {'✅ YES' if wants_from_text else '❌ NO '} "
            f"[regex=miss, llm-parse-failed, text-scan={'hit' if wants_from_text else 'miss'}] "
            f"| '{text[:60]}'"
        )
        return {"wants_report": wants_from_text, "explicit_docs": []}

    except Exception as e:
        # Even on total failure, honour the regex hit
        print(f"⚠️  intent check error: {e} — regex fallback: {regex_hit}")
        return {"wants_report": regex_hit, "explicit_docs": []}


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
        # ── CAD/IFC files live in CadSharedContext, not the vector store ──
        if _cad_ifc_available:
            from cad_ifc_agent import CadSharedContext
            cad_file = CadSharedContext.get_file(doc_id)
            if cad_file:
                # Return a JSON summary as readable text so the viewer renders something
                import json as _json
                summary_text = _json.dumps(cad_file, indent=2, default=str)
                return {"document_id": doc_id, "filename": cad_file.get("filename", doc_id), "content": summary_text}

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
        # ── CAD/IFC files: not stored on disk via this route — return 404 with hint ──
        if _cad_ifc_available:
            from cad_ifc_agent import CadSharedContext
            if CadSharedContext.get_file(doc_id):
                raise HTTPException(
                    status_code=404,
                    detail="CAD/IFC files are streamed from the browser blob URL, not re-downloaded from the server."
                )

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


@app.get("/api/news/meta")
async def news_meta():
    """
    Returns the cache manifest: total_pages, run_at, next_run_at, status.
    Frontend calls this once on mount to know how many pages exist.
    """
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    meta = get_meta()
    if meta is None:
        raise HTTPException(503, "No news cache available yet. The pipeline may still be running.")
    return meta


@app.get("/api/news/pages/{page_num}")
async def news_page(page_num: int):
    """
    Returns a single pre-computed page of articles.
    This is the only endpoint the frontend needs for infinite scroll.
    Response includes has_more so the frontend knows when to stop.
    """
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    if page_num < 0:
        raise HTTPException(400, "page_num must be >= 0")

    data = get_page(page_num)
    if data is None:
        return JSONResponse(
            status_code=404,
            content={"error": "page_not_found", "has_more": False, "page": page_num},
            headers={"Cache-Control": "no-store"},
        )

    # Attach has_more from meta
    meta = get_meta()
    total_pages = meta.get("total_pages", 0) if meta else 0
    data["has_more"] = (page_num + 1) < total_pages

    # Pages are immutable within a cycle — cache aggressively
    return JSONResponse(
        content=data,
        headers={"Cache-Control": "public, max-age=3600, stale-while-revalidate=1800"},
    )


@app.get("/api/news/status")
async def news_status():
    """
    Lightweight polling endpoint — frontend uses this to detect when a fresh
    cycle has finished and show the "New articles available" banner.
    """
    if not _news_pipeline_available:
        return {"running": False, "status": "unavailable"}
    return _pipeline_get_status()


@app.post("/api/news/trigger")
async def news_trigger(background_tasks: BackgroundTasks, force: bool = False):
    """
    Manually trigger a pipeline run (admin / refresh button).
    Returns immediately; the pipeline runs in a FastAPI background task.
    """
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    if pipeline_is_running():
        return {"status": "already_running", "message": "Pipeline is already in progress."}
    background_tasks.add_task(run_news_pipeline, force)
    return {"status": "accepted", "message": "Pipeline triggered. Poll /api/news/status for progress."}


@app.post("/api/news/refresh")
async def news_refresh(background_tasks: BackgroundTasks):
    """Backwards-compatible alias for POST /api/news/trigger?force=true."""
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    if pipeline_is_running():
        return {"status": "already_running"}
    background_tasks.add_task(run_news_pipeline, True)
    return {"status": "accepted", "message": "Force refresh triggered."}


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
            "cad_ifc_agent": "available" if _cad_ifc_available else "unavailable",
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
    print(f"📰 News pipeline: {'✅ available' if _news_pipeline_available else '⚠️  not available'}")
    print(f"🏗️  CAD/IFC agent: {'✅ available' if _cad_ifc_available else '⚠️  not available'}")

    # ── News pipeline scheduler (every 4 days) ────────────────────────────
    if _news_pipeline_available:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            import os as _os

            _scheduler = BackgroundScheduler()
            _scheduler.add_job(
                run_news_pipeline,
                trigger="interval",
                days=int(_os.getenv("NEWS_CYCLE_DAYS", "4")),
                id="news_pipeline_4day",
                replace_existing=True,
            )
            _scheduler.start()

            cycle_days = int(_os.getenv("NEWS_CYCLE_DAYS", "4"))
            print(f"⏰ News scheduler: every {cycle_days} days (APScheduler)")

            # If no cache exists yet, trigger the first run immediately in background
            meta = get_meta()
            if meta is None:
                print("📰 No news cache found — triggering initial pipeline run…")
                import threading as _th
                _th.Thread(target=run_news_pipeline, kwargs={"force": False}, daemon=True).start()
            else:
                print(f"📰 News cache: {meta.get('total_pages', 0)} pages "
                      f"(run_at={meta.get('run_at', 'unknown')})")
        except ImportError:
            print("⚠️  apscheduler not installed — run: pip install apscheduler")
            print("    News pipeline will only run via POST /api/news/trigger")

    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n👋 BIMLO Copilot API — shutting down\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")