from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging

from neo4j_auth import router as auth_router, init_neo4j
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from core.deps import limiter
from core import globals as g
from core.config import settings

logger = logging.getLogger(__name__)

groq_ok = bool(settings.groq_api_key)
cf_ok   = bool(settings.cf_api_key)
CHECK = "\u2705"
CROSS = "\u274c"
groq_status = f"{CHECK} configured" if groq_ok else f"{CROSS} missing GROQ_API_KEY"
cf_status   = f"{CHECK} configured" if cf_ok   else "\u26a0\ufe0f  missing CF_API_KEY (Groq-only mode)"
print(f"\U0001f4e1 Groq API: {groq_status}")
print(f"\U0001f4e1 CF API:   {cf_status}")

from services.document_processor import DocumentProcessor  # noqa: E402
from services.vector_store import VectorStoreManager  # noqa: E402
from services.rag_engine import RAGEngine  # noqa: E402
from services.suggest import router as suggest_router  # noqa: E402
from services.voice_transcriber import router as voice_router  # noqa: E402
from services.voice_call import router as voice_call_router  # noqa: E402
from services.autocomplete import router as autocomplete_router  # noqa: E402
from services.report_agent import router as report_router, SharedContext  # noqa: E402

try:
    from services.ingestion_graph import run_ingestion_pipeline
    g._ingestion_graph_available = True
except ImportError:
    try:
        from ingestion_graph import run_ingestion_pipeline
        g._ingestion_graph_available = True
    except ImportError:
        run_ingestion_pipeline = None
        g._ingestion_graph_available = False
        print("\u26a0\ufe0f  ingestion_graph not found \u2014 falling back to direct indexing")

g.run_ingestion_pipeline = run_ingestion_pipeline

try:
    from news_pipeline import (  # noqa: F401
        run_news_pipeline, pipeline_is_running, get_meta, get_page,  # noqa: F401
        get_status as _pipeline_get_status, CACHE_DIR as NEWS_CACHE_DIR,  # noqa: F401
    )
    g._news_pipeline_available = True
except ImportError:
    g._news_pipeline_available = False
    print("\u26a0\ufe0f  news_pipeline not found \u2014 /api/news endpoints will return 503")

try:
    from news_chat_agent import router as news_chat_router
    g._news_chat_available = True
    print("\u2705 news_chat_agent loaded \u2014 /api/news/chat ready")
except ImportError:
    news_chat_router = None
    g._news_chat_available = False
    print("\u26a0\ufe0f  news_chat_agent not found \u2014 /api/news/chat will 503")

try:
    from cad_ifc_agent import router as cad_ifc_router
    g._cad_ifc_available = True
    print("\u2705 cad_ifc_agent loaded \u2014 /api/cad/upload + /api/cad/query ready")
except ImportError as _e:
    cad_ifc_router = None
    g._cad_ifc_available = False
    print(f"\u26a0\ufe0f  cad_ifc_agent not found \u2014 /api/cad endpoints will 503 ({_e})")

app = FastAPI(
    title="BIMLO Copilot T\u00e9l\u00e9com API",
    version="3.0.0",
    description="Agentic RAG API powered by LangGraph \u2014 routes, retrieves, iterates, analyses.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = settings.allowed_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

@app.middleware("http")
async def limit_upload_size(request, call_next):
    if request.url.path == "/upload":
        cl = request.headers.get("content-length")
        if cl and int(cl) > g.MAX_UPLOAD_SIZE:
            origin = request.headers.get("origin", "*")
            return JSONResponse(
                {"detail": f"File too large. Maximum size is {g.MAX_UPLOAD_SIZE // 1024 // 1024} MB."},
                status_code=413,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                },
            )
    return await call_next(request)

g.doc_processor = DocumentProcessor()
g.vector_store  = VectorStoreManager()
g.rag_engine    = RAGEngine(g.vector_store)

SharedContext.set_vector_store(g.vector_store)

from routers import health, documents, chat, sessions, reports, news, providers  # noqa: E402

app.include_router(auth_router)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(reports.router)
app.include_router(news.router)
app.include_router(suggest_router)
app.include_router(voice_router)
app.include_router(voice_call_router)
app.include_router(autocomplete_router)
app.include_router(report_router)
app.include_router(providers.router)
if g._news_chat_available:
    app.include_router(news_chat_router)
if g._cad_ifc_available:
    app.include_router(cad_ifc_router)

os.makedirs(g.UPLOAD_DIR, exist_ok=True)
os.makedirs(g.REPORTS_DIR, exist_ok=True)

from routers.health import detailed_health_router  # noqa: E402

app.include_router(detailed_health_router)


@app.on_event("startup")
async def startup_event():
    init_neo4j()
    try:
        from observability_otel import init_otel
        init_otel(app)
        print("\u2705 OpenTelemetry initialized")
    except Exception as e:
        print(f"\u26a0\ufe0f  OpenTelemetry init skipped: {e}")
    print("\n" + "="*60)
    print("\U0001f680 BIMLO Copilot T\u00e9l\u00e9com API v3 \u2014 LangGraph Agentic RAG")
    print("="*60)
    print(f"\U0001f4c1 Data dir:    {g.DATA_DIR}")
    print(f"\U0001f4c1 Uploads:     {g.UPLOAD_DIR}")
    print(f"\U0001f4c4 Reports:     {g.REPORTS_DIR}")
    print("\U0001f578\ufe0f  Graph routes: direct | rag | iterative_rag | analytics")
    try:
        s = g.vector_store.get_global_stats()
        print(f"\U0001f4ca Vector store: {s['total_documents']} docs, {s['total_chunks']} chunks")
    except Exception:
        print("\u26a0\ufe0f  Vector store stats unavailable")
    groq_api_label = "\u2705 configured" if settings.groq_api_key else "\u26a0\ufe0f  not configured"
    news_label     = "\u2705 available" if g._news_pipeline_available else "\u26a0\ufe0f  not available"
    cad_label      = "\u2705 available" if g._cad_ifc_available else "\u26a0\ufe0f  not available"
    ingest_label   = "\u2705 LangGraph pipeline" if g._ingestion_graph_available else "\u26a0\ufe0f  fallback (direct)"
    print(f"\U0001f511 Groq API: {groq_api_label}")
    print(f"\U0001f4f0 News pipeline:     {news_label}")
    print(f"\U0001f3d7\ufe0f  CAD/IFC agent:     {cad_label}")
    print(f"\U0001f504 Ingestion graph:   {ingest_label}")

    if g._news_pipeline_available:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from news_pipeline import run_news_pipeline, get_meta

            _scheduler = BackgroundScheduler()
            _scheduler.add_job(
                run_news_pipeline,
                trigger="interval",
                days=settings.news_cycle_days,
                id="news_pipeline_daily",
                replace_existing=True,
            )
            _scheduler.start()

            cycle_days = settings.news_cycle_days
            print(f"\u23f0 News scheduler: every {cycle_days} days (APScheduler)")

            meta = get_meta()
            if meta is None:
                print("\U0001f4f0 No news cache found — triggering initial pipeline run\u2026")
                import threading as _th
                _th.Thread(target=run_news_pipeline, kwargs={"force": False}, daemon=True).start()
            else:
                print(f"\U0001f4f0 News cache: {meta.get('total_pages', 0)} pages "
                      f"(run_at={meta.get('run_at', 'unknown')})")
        except ImportError:
            print("\u26a0\ufe0f  apscheduler not installed — run: pip install apscheduler")
            print("    News pipeline will only run via POST /api/news/trigger")

    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n\U0001f44b BIMLO Copilot API — shutting down\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
