"""
ingestion_graph.py — LangGraph document ingestion pipeline
══════════════════════════════════════════════════════════════════════════════
Three-node ingestion pipeline (unchanged external API):

  chunk_document  →  index_vector_store  →  ingest_graph_rag  →  END

Each node owns one responsibility, has proper state, error isolation, and
logging.  The pipeline is non-blocking: run_ingestion_pipeline() submits the
graph to a background thread-pool executor and returns immediately.

What changed vs the original:
  ─ classify_doc_intent node: after chunk_document succeeds, a fast heuristic
    pass runs over the filename + first-chunk text to tag the document with a
    routing hint (doc_intent). This is stored in chunk metadata so the RAG
    engine can use it as a retrieval hint without a separate LLM call at query
    time. Uses only the heuristic path — zero LLM cost at ingestion time.
  ─ IngestionState carries doc_intent and doc_language fields.
  ─ index_vector_store passes doc_intent + doc_language through to add_document
    so they land in Chroma metadata.
  ─ Richer completion summary printed at the end.
  ─ _executor.shutdown(wait=False) registered at process exit for clean teardown.

Usage (from main.py) — unchanged:
    from ingestion_graph import run_ingestion_pipeline
    run_ingestion_pipeline(vector_store, doc_id, filename, chunks,
                           session_id=..., user_id=...)
"""

from __future__ import annotations

import atexit
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

# ── Observability ─────────────────────────────────────────────────────────────
try:
    from observability import obs as _obs
    _OBS_AVAILABLE = True
except ImportError:
    _obs = None
    _OBS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════

class IngestionState(TypedDict):
    # inputs
    doc_id:     str
    filename:   str
    chunks:     List[Dict]
    session_id: Optional[str]
    user_id:    Optional[str]

    # set by classify_doc_intent node
    doc_intent:   str            # heuristic intent tag for the whole document
    doc_language: str            # detected language ("en", "fr", "ar", "unknown")

    # set by downstream nodes
    vector_indexed: bool
    graph_ingested: bool
    graph_stats:    Optional[Dict]
    skipped_graph:  bool         # True = graph_rag unavailable, not a failure
    error:          Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — chunk_document
# ══════════════════════════════════════════════════════════════════════════════

def _node_chunk_document(state: IngestionState) -> IngestionState:
    """
    Validate that chunks exist and are non-empty.
    Future: could re-chunk with different parameters, detect language, etc.
    """
    chunks   = state["chunks"]
    filename = state["filename"]

    if not chunks:
        msg = f"[ingestion] '{filename}': no chunks produced — skipping indexing"
        print(f"⚠️  {msg}")
        if _OBS_AVAILABLE:
            _obs.log_ingestion(state["doc_id"], filename, "chunk_document", "failed",
                               details={"reason": "no chunks"})
        return {**state, "error": msg}

    print(
        f"📄 [ingestion:chunk_document] '{filename}' — "
        f"{len(chunks)} chunks ready for indexing"
    )
    if _OBS_AVAILABLE:
        _obs.log_ingestion(
            state["doc_id"], filename, "chunk_document", "ok",
            details={"chunk_count": len(chunks)},
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
        )
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 — classify_doc_intent
# ══════════════════════════════════════════════════════════════════════════════

# ── Heuristic file-type helpers ───────────────────────────────────────────────
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
_CAD_EXTS   = {".ifc", ".ifczip", ".dxf", ".dwg", ".step", ".stp"}

def _infer_doc_intent(filename: str, first_chunk_text: str) -> str:
    """
    Tag the document with a routing intent using filename + first-chunk signals.
    Zero LLM cost — pure pattern matching.

    Returns one of:
        "cad"        — IFC / CAD / BIM model
        "image"      — standalone uploaded image
        "report"     — looks like a study / report / specification
        "contract"   — legal / contract language
        "technical"  — datasheet, manual, spec
        "document"   — generic fallback
    """
    import os as _os
    ext  = _os.path.splitext(filename)[1].lower()
    name = filename.lower()
    text = (first_chunk_text or "").lower()[:600]

    if ext in _CAD_EXTS:
        return "cad"
    if ext in _IMAGE_EXTS:
        return "image"

    # Report / study signals
    _report_kw = [
        "rapport", "report", "étude", "study", "analysis", "analyse",
        "summary", "résumé", "executive", "conclusions", "recommendations",
        "تقرير", "دراسة", "ملخص",
    ]
    if any(k in text or k in name for k in _report_kw):
        return "report"

    # Contract / legal
    _contract_kw = [
        "contract", "contrat", "agreement", "clause", "terms and conditions",
        "legal", "juridique", "liability", "warranty", "عقد", "اتفاقية",
    ]
    if any(k in text or k in name for k in _contract_kw):
        return "contract"

    # Technical datasheet / specification
    _tech_kw = [
        "specification", "spécification", "datasheet", "technical", "technique",
        "standard", "norme", "protocol", "protocole", "procedure", "procédure",
        "manual", "manuel", "مواصفات", "دليل",
    ]
    if any(k in text or k in name for k in _tech_kw):
        return "technical"

    return "document"


def _detect_language(text: str) -> str:
    """
    Lightweight language detection from first 400 chars — no external library needed.
    Returns an ISO 639-1 code or "unknown".
    """
    sample = (text or "").strip()[:400]
    if not sample:
        return "unknown"

    # Count unicode script characters
    arabic = sum(1 for c in sample if "\u0600" <= c <= "\u06ff")
    latin  = sum(1 for c in sample if c.isalpha() and ord(c) < 256)

    if arabic > len(sample) * 0.15:
        return "ar"

    # French markers
    _fr_markers = [
        " le ", " la ", " les ", " des ", " une ", " du ", " en ", " et ",
        " est ", " dans ", " avec ", "é", "è", "ê", "ô", "â",
    ]
    _en_markers = [
        " the ", " is ", " are ", " of ", " in ", " and ", " to ", " for ",
        " that ", " this ", " with ", " from ",
    ]

    fr_score = sum(1 for m in _fr_markers if m in sample.lower())
    en_score = sum(1 for m in _en_markers if m in sample.lower())

    if fr_score > en_score and fr_score >= 2:
        return "fr"
    if en_score >= 2:
        return "en"

    return "unknown"


def _node_classify_doc_intent(state: IngestionState) -> IngestionState:
    """
    Heuristic classification of the document type and language.
    Runs after chunk_document, before vector indexing.
    Tags the state with doc_intent and doc_language so downstream metadata is richer.
    """
    if state.get("error"):
        return state  # chunk_document failed — nothing to classify

    filename = state["filename"]
    chunks   = state["chunks"]

    first_text = chunks[0].get("text", "") if chunks else ""
    doc_intent  = _infer_doc_intent(filename, first_text)
    doc_language = _detect_language(first_text)

    print(
        f"🏷️  [ingestion:classify_doc_intent] '{filename}' → "
        f"intent={doc_intent}, lang={doc_language}"
    )

    # Stamp intent + language onto every chunk's metadata so retrieval can filter
    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        meta.setdefault("doc_intent",   doc_intent)
        meta.setdefault("doc_language", doc_language)

    if _OBS_AVAILABLE:
        _obs.log_ingestion(
            state["doc_id"], filename, "classify_doc_intent", "ok",
            details={"doc_intent": doc_intent, "doc_language": doc_language},
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
        )

    return {**state, "doc_intent": doc_intent, "doc_language": doc_language, "chunks": chunks}


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 — index_vector_store
# ══════════════════════════════════════════════════════════════════════════════

def _make_index_vector_store_node(vector_store):
    """
    Returns a closure node that indexes chunks into ChromaDB.
    Receives the VectorStoreManager at graph-build time so the node
    is a plain function with no class dependency.
    """
    def _node(state: IngestionState) -> IngestionState:
        if state.get("error"):
            return state  # previous node failed — skip

        doc_id     = state["doc_id"]
        filename   = state["filename"]
        chunks     = state["chunks"]
        user_id    = state.get("user_id")
        session_id = state.get("session_id")
        doc_intent = state.get("doc_intent", "document")

        label = f"user_{user_id or 'anon'}_session_{session_id}"
        print(
            f"💾 [ingestion:index_vector_store] indexing '{filename}' "
            f"({len(chunks)} chunks, intent={doc_intent}) → {label}…"
        )
        t0 = time.time()
        try:
            vector_store.add_document(
                filename,
                chunks,
                session_id=session_id,
                user_id=user_id,
                doc_id=doc_id,
            )
            latency_ms = (time.time() - t0) * 1000
            print(
                f"✅ [ingestion:index_vector_store] '{filename}' indexed "
                f"in {time.time()-t0:.1f}s → {label}"
            )
            if _OBS_AVAILABLE:
                _obs.log_ingestion(
                    doc_id, filename, "index_vector_store", "ok",
                    latency_ms=latency_ms,
                    details={"chunk_count": len(chunks), "doc_intent": doc_intent},
                    session_id=session_id, user_id=user_id,
                )
            return {**state, "vector_indexed": True}
        except Exception as e:
            msg = f"[ingestion] vector store indexing failed for '{filename}': {e}"
            print(f"❌ {msg}")
            traceback.print_exc()
            if _OBS_AVAILABLE:
                _obs.log_ingestion(
                    doc_id, filename, "index_vector_store", "failed",
                    details={"error": str(e)},
                    session_id=session_id, user_id=user_id,
                )
            return {**state, "error": msg, "vector_indexed": False}

    return _node


# ══════════════════════════════════════════════════════════════════════════════
# Node 4 — ingest_graph_rag
# ══════════════════════════════════════════════════════════════════════════════

def _node_ingest_graph_rag(state: IngestionState) -> IngestionState:
    """
    Extract entities/relationships from chunks and write them to Neo4j.
    Non-fatal: if graph_rag is unavailable, sets skipped_graph=True.
    If Neo4j is reachable but extraction fails, logs the error but does NOT
    set state["error"] — the document is usable without the knowledge graph.
    """
    if state.get("error"):
        return state  # vector indexing failed — skip graph

    doc_id   = state["doc_id"]
    filename = state["filename"]
    chunks   = state["chunks"]

    try:
        from graph_rag import get_engine as _get_graph_engine
        graph_engine = _get_graph_engine()

        if not graph_engine.available:
            print(f"ℹ️  [ingestion:ingest_graph_rag] Neo4j unavailable — skipping '{filename}'")
            if _OBS_AVAILABLE:
                _obs.log_ingestion(doc_id, filename, "ingest_graph_rag", "skipped",
                                   details={"reason": "Neo4j unavailable"})
            return {**state, "skipped_graph": True, "graph_ingested": False}

        print(f"🕸️  [ingestion:ingest_graph_rag] extracting entities from '{filename}'…")
        _t0 = time.time()
        stats = graph_engine.ingest_chunks(doc_id, filename, chunks)
        latency_ms = (time.time() - _t0) * 1000
        print(
            f"✅ [ingestion:ingest_graph_rag] '{filename}' — "
            f"{stats.get('entities', 0)} entities, "
            f"{stats.get('relationships', 0)} relationships"
        )
        if _OBS_AVAILABLE:
            _obs.log_ingestion(doc_id, filename, "ingest_graph_rag", "ok",
                               latency_ms=latency_ms, details=stats)
        return {**state, "graph_ingested": True, "graph_stats": stats}

    except ImportError:
        print(f"ℹ️  [ingestion:ingest_graph_rag] graph_rag module not found — skipping")
        if _OBS_AVAILABLE:
            _obs.log_ingestion(doc_id, filename, "ingest_graph_rag", "skipped",
                               details={"reason": "graph_rag not installed"})
        return {**state, "skipped_graph": True, "graph_ingested": False}

    except Exception as e:
        # Non-fatal — vector store is already indexed and usable
        print(f"⚠️  [ingestion:ingest_graph_rag] entity extraction failed (non-fatal): {e}")
        if _OBS_AVAILABLE:
            _obs.log_ingestion(doc_id, filename, "ingest_graph_rag", "failed",
                               details={"error": str(e)})
        return {**state, "graph_ingested": False, "graph_stats": {"error": str(e)}}


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_ingestion_graph(vector_store) -> StateGraph:
    """
    Compile the LangGraph ingestion pipeline.
    vector_store is injected here so nodes are pure functions.

    Pipeline:
        chunk_document
            ↓
        classify_doc_intent    ← NEW: heuristic intent tagging (zero LLM cost)
            ↓
        index_vector_store
            ↓
        ingest_graph_rag
            ↓
           END
    """
    workflow = StateGraph(IngestionState)

    workflow.add_node("chunk_document",       _node_chunk_document)
    workflow.add_node("classify_doc_intent",  _node_classify_doc_intent)
    workflow.add_node("index_vector_store",   _make_index_vector_store_node(vector_store))
    workflow.add_node("ingest_graph_rag",     _node_ingest_graph_rag)

    workflow.set_entry_point("chunk_document")
    workflow.add_edge("chunk_document",       "classify_doc_intent")
    workflow.add_edge("classify_doc_intent",  "index_vector_store")
    workflow.add_edge("index_vector_store",   "ingest_graph_rag")
    workflow.add_edge("ingest_graph_rag",     END)

    return workflow.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Thread pool — shared across all upload requests
# ══════════════════════════════════════════════════════════════════════════════

# Max 4 concurrent ingestion jobs; keeps Neo4j + LLM load manageable.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ingestion")

# Clean shutdown: wait=False so the process doesn't hang at exit
atexit.register(lambda: _executor.shutdown(wait=False))


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion_pipeline(
    vector_store,
    doc_id:     str,
    filename:   str,
    chunks:     List[Dict],
    session_id: Optional[str] = None,
    user_id:    Optional[str] = None,
) -> None:
    """
    Submit a document ingestion job to the background thread pool.

    Returns immediately — the /upload endpoint does NOT wait for indexing.
    The LangGraph pipeline runs in the background:

        chunk_document
            ↓
        classify_doc_intent   ← heuristic doc-type + language tagging
            ↓
        index_vector_store    ← ChromaDB
            ↓
        ingest_graph_rag      ← Neo4j (optional, non-fatal)
            ↓
           END

    Args:
        vector_store: VectorStoreManager instance (holds the ChromaDB client)
        doc_id:       UUID string assigned to this document
        filename:     Original filename (used for metadata + Neo4j nodes)
        chunks:       List of chunk dicts from DocumentProcessor
        session_id:   Chat session ID for isolation
        user_id:      User ID for per-user isolation

    NOTE: vector_store.add_document() is called INSIDE the graph.
          Do NOT call it before run_ingestion_pipeline() — it would double-index.
    """
    graph = _build_ingestion_graph(vector_store)

    initial_state: IngestionState = {
        "doc_id":         doc_id,
        "filename":       filename,
        "chunks":         chunks,
        "session_id":     session_id,
        "user_id":        user_id,
        "doc_intent":     "document",  # filled by classify_doc_intent node
        "doc_language":   "unknown",   # filled by classify_doc_intent node
        "vector_indexed": False,
        "graph_ingested": False,
        "graph_stats":    None,
        "skipped_graph":  False,
        "error":          None,
    }

    def _run():
        t0 = time.time()
        try:
            final   = graph.invoke(initial_state)
            elapsed = time.time() - t0

            if final.get("error"):
                print(
                    f"❌ [ingestion_graph] '{filename}' FAILED after {elapsed:.1f}s: "
                    f"{final['error']}"
                )
            else:
                g = final.get("graph_stats") or {}
                _graph_status = (
                    f"✅ {g.get('entities','?')}e/{g.get('relationships','?')}r"
                    if final.get("graph_ingested")
                    else ("⏭ skipped" if final.get("skipped_graph") else "❌")
                )
                print(
                    f"🎉 [ingestion_graph] '{filename}' complete in {elapsed:.1f}s — "
                    f"intent={final.get('doc_intent','?')} "
                    f"lang={final.get('doc_language','?')} "
                    f"vector={'✅' if final.get('vector_indexed') else '❌'} "
                    f"graph={_graph_status}"
                )
        except Exception as e:
            print(f"❌ [ingestion_graph] '{filename}' unhandled exception: {e}")
            traceback.print_exc()

    _executor.submit(_run)
    print(
        f"📬 [ingestion_graph] '{filename}' submitted to background pipeline "
        f"(user_{user_id or 'anon'}_session_{session_id})"
    )