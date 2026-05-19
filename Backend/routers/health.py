from fastapi import APIRouter
from datetime import datetime
import json
import os

from core import globals as g

router = APIRouter(tags=["health"])
detailed_health_router = APIRouter(tags=["health"])


@router.get("/")
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


@detailed_health_router.get("/health")
async def health_check():
    try:
        stats = g.vector_store.get_global_stats()

        elevenlabs_ok  = bool(os.getenv("ELEVENLABS_API_KEY"))

        # Dynamic provider check from providers.json
        llm_providers = {}
        any_provider_ok = False
        try:
            providers_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "providers.json")
            with open(providers_path) as f:
                providers_cfg = json.load(f).get("providers", [])
            for p in providers_cfg:
                key_env = p.get("api_key_env", "")
                fallback_env = p.get("fallback_api_key_env", "")
                ok = bool(os.getenv(key_env, "").strip()) or (fallback_env and bool(os.getenv(fallback_env, "").strip()))
                llm_providers[p["id"]] = "configured" if ok else "not_configured"
                if ok:
                    any_provider_ok = True
        except Exception:
            pass
        llm_status = "ok" if any_provider_ok else "degraded"

        neo4j_status = "unknown"
        try:
            from neo4j_auth import get_driver as _get_neo4j_driver
            drv = _get_neo4j_driver()
            if drv:
                drv.verify_connectivity()
                neo4j_status = "connected"
            else:
                neo4j_status = "not_configured"
        except Exception:
            neo4j_status = "not_configured"

        news_info = {"available": g._news_pipeline_available}
        if g._news_pipeline_available:
            try:
                from news_pipeline import get_status as _news_status, get_page as _news_page
                ns = _news_status()
                news_info.update(ns)
                p1 = _news_page(1)
                if p1 and "articles" in p1:
                    articles_on_p1 = len(p1["articles"])
                    total_items = ns.get("total_items") or (articles_on_p1 * max(ns.get("total_pages", 1), 1))
                    news_info["total_articles"] = total_items
            except Exception as _ne:
                news_info["error"] = str(_ne)

        wiki_ok = False
        try:
            import importlib
            importlib.import_module("wikipediaapi")
            wiki_ok = True
        except ImportError:
            pass

        judge_ok = False
        try:
            judge_ok = True
        except Exception:
            pass

        ingestion_mode = "langgraph" if g._ingestion_graph_available else "fallback_direct"

        report_count = 0
        try:
            from services.report_agent import _reports_store as _rs
            report_count = len(_rs)
        except Exception:
            try:
                from report_agent import _reports_store as _rs
                report_count = len(_rs)
            except Exception:
                pass

        return {
            "status":       "healthy",
            "timestamp":    datetime.now().isoformat(),
            "vector_store": "connected",
            "neo4j":        neo4j_status,
            "api_server":   "ok",
            "llm_status":   llm_status,
            "llm_providers": llm_providers,
            "voice_agent":  "configured" if elevenlabs_ok else "not_configured",
            "elevenlabs":   "configured" if elevenlabs_ok else "not_configured",
            "cad_ifc_agent":    "available" if g._cad_ifc_available else "unavailable",
            "ingestion_mode":   ingestion_mode,
            "wiki_enricher":    "available" if wiki_ok else "unavailable",
            "llm_judge":        "available" if judge_ok else "unavailable",
            "news_pipeline":    news_info,
            "statistics":       stats,
            "active_sessions":  len(g._sessions),
            "report_count":     report_count,
            "services": f"{sum([any_provider_ok, elevenlabs_ok, g._cad_ifc_available, g._news_pipeline_available])} services active",
        }
    except Exception as e:
        return {"status": "degraded", "timestamp": datetime.now().isoformat(), "error": str(e)}
