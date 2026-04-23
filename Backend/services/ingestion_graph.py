"""
ingestion_graph.py — LangGraph document ingestion pipeline

Replaces the fire-and-forget threading.Thread() in main.py's /upload endpoint.

Three-node graph:
  chunk_document  →  index_vector_store  →  ingest_graph_rag

Each node owns one responsibility, has proper state, error isolation, and
logging.  The pipeline is non-blocking: main.py invokes it via
run_ingestion_pipeline() which submits the graph to a background thread-pool
executor — the /upload response is returned immediately while ingestion
continues in the background.

State fields intentionally kept minimal:
  - doc_id, filename, chunks: the document being processed
  - vector_indexed: True once ChromaDB add_document() succeeds
  - graph_ingested: True once Neo4j entity extraction succeeds
  - graph_stats: dict returned by graph_rag.ingest_chunks()
  - error: first fatal error message (non-fatal errors logged but don't stop the graph)
  - skipped_graph: True when graph_rag is unavailable (not a failure)

Usage (from main.py):
    from ingestion_graph import run_ingestion_pipeline
    run_ingestion_pipeline(vector_store, doc_id, filename, chunks)
"""

from __future__ import annotations

import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IngestionState(TypedDict):
    # inputs
    doc_id:     str
    filename:   str
    chunks:     List[Dict]
    session_id: Optional[str]

    # outputs set by each node
    vector_indexed:  bool
    graph_ingested:  bool
    graph_stats:     Optional[Dict]
    skipped_graph:   bool   # True = graph_rag unavailable, not a failure
    error:           Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

def _node_chunk_document(state: IngestionState) -> IngestionState:
    """
    Validate that chunks exist and are non-empty.
    In future: could re-chunk with different parameters, detect language, etc.
    """
    chunks = state["chunks"]
    filename = state["filename"]

    if not chunks:
        msg = f"[ingestion] '{filename}': no chunks produced — skipping indexing"
        print(f"⚠️  {msg}")
        return {**state, "error": msg}

    print(
        f"📄 [ingestion:chunk_document] '{filename}' — "
        f"{len(chunks)} chunks ready for indexing"
    )
    return state  # pass through; chunks already produced by DocumentProcessor


def _make_index_vector_store_node(vector_store):
    """
    Returns a closure node that indexes chunks into ChromaDB.
    Receives the VectorStoreManager at graph-build time so the node
    is a plain function with no class dependency.
    """
    def _node(state: IngestionState) -> IngestionState:
        if state.get("error"):
            return state  # previous node failed — skip

        doc_id   = state["doc_id"]
        filename = state["filename"]
        chunks   = state["chunks"]

        print(f"💾 [ingestion:index_vector_store] indexing '{filename}' ({len(chunks)} chunks)…")
        t0 = time.time()
        try:
            # add_document returns the same doc_id we passed in
            vector_store.add_document(
                filename,
                chunks,
                session_id=state.get("session_id"),
            )
            print(
                f"✅ [ingestion:index_vector_store] '{filename}' indexed "
                f"in {time.time()-t0:.1f}s"
            )
            return {**state, "vector_indexed": True}
        except Exception as e:
            msg = f"[ingestion] vector store indexing failed for '{filename}': {e}"
            print(f"❌ {msg}")
            traceback.print_exc()
            return {**state, "error": msg, "vector_indexed": False}

    return _node


def _node_ingest_graph_rag(state: IngestionState) -> IngestionState:
    """
    Extract entities/relationships from chunks and write them to Neo4j.
    Non-fatal: if graph_rag is unavailable, sets skipped_graph=True and
    continues.  If Neo4j is reachable but extraction fails, logs the error
    but does NOT set state["error"] (vector store is already indexed — the
    document is usable without the knowledge graph).
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
            print(f"ℹ️  [ingestion:ingest_graph_rag] Neo4j unavailable — skipping for '{filename}'")
            return {**state, "skipped_graph": True, "graph_ingested": False}

        print(f"🕸️  [ingestion:ingest_graph_rag] extracting entities from '{filename}'…")
        stats = graph_engine.ingest_chunks(doc_id, filename, chunks)
        print(
            f"✅ [ingestion:ingest_graph_rag] '{filename}' — "
            f"{stats.get('entities', 0)} entities, "
            f"{stats.get('relationships', 0)} relationships"
        )
        return {**state, "graph_ingested": True, "graph_stats": stats}

    except ImportError:
        print(f"ℹ️  [ingestion:ingest_graph_rag] graph_rag module not found — skipping")
        return {**state, "skipped_graph": True, "graph_ingested": False}

    except Exception as e:
        # Non-fatal — vector store is already indexed and usable
        print(f"⚠️  [ingestion:ingest_graph_rag] entity extraction failed (non-fatal): {e}")
        return {**state, "graph_ingested": False, "graph_stats": {"error": str(e)}}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_ingestion_graph(vector_store) -> StateGraph:
    """
    Compile a LangGraph StateGraph for the ingestion pipeline.
    vector_store is injected here so nodes are pure functions.
    """
    workflow = StateGraph(IngestionState)

    workflow.add_node("chunk_document",      _node_chunk_document)
    workflow.add_node("index_vector_store",  _make_index_vector_store_node(vector_store))
    workflow.add_node("ingest_graph_rag",    _node_ingest_graph_rag)

    workflow.set_entry_point("chunk_document")
    workflow.add_edge("chunk_document",     "index_vector_store")
    workflow.add_edge("index_vector_store", "ingest_graph_rag")
    workflow.add_edge("ingest_graph_rag",   END)

    return workflow.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Thread pool — shared across all upload requests
# ─────────────────────────────────────────────────────────────────────────────

# Max 4 concurrent ingestion jobs; keeps Neo4j + LLM load manageable.
# Increase if you're on a beefy server and process many large docs in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ingestion")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion_pipeline(
    vector_store,
    doc_id:   str,
    filename: str,
    chunks:   List[Dict],
    session_id: Optional[str] = None,
) -> None:
    """
    Submit a document ingestion job to the background thread pool.

    Returns immediately — the /upload endpoint does NOT wait for indexing.
    The LangGraph pipeline runs in the background:
      chunk_document → index_vector_store → ingest_graph_rag → END

    Args:
        vector_store: VectorStoreManager instance (already holds the ChromaDB client)
        doc_id:       UUID string assigned to this document
        filename:     Original filename (used for metadata + Neo4j nodes)
        chunks:       List of chunk dicts from DocumentProcessor

    NOTE: vector_store.add_document() is called INSIDE the graph now.
          Do NOT call it before run_ingestion_pipeline() — it would double-index.
    """
    graph = _build_ingestion_graph(vector_store)

    initial_state: IngestionState = {
        "doc_id":          doc_id,
        "filename":        filename,
        "chunks":          chunks,
        "session_id":      session_id,
        "vector_indexed":  False,
        "graph_ingested":  False,
        "graph_stats":     None,
        "skipped_graph":   False,
        "error":           None,
    }

    def _run():
        t0 = time.time()
        try:
            final = graph.invoke(initial_state)
            elapsed = time.time() - t0
            if final.get("error"):
                print(
                    f"❌ [ingestion_graph] '{filename}' FAILED after {elapsed:.1f}s: "
                    f"{final['error']}"
                )
            else:
                g = final.get("graph_stats") or {}
                print(
                    f"🎉 [ingestion_graph] '{filename}' complete in {elapsed:.1f}s — "
                    f"vector={'✅' if final.get('vector_indexed') else '❌'} "
                    f"graph={'✅ ' + str(g.get('entities','?')) + 'e/' + str(g.get('relationships','?')) + 'r' if final.get('graph_ingested') else ('⏭ skipped' if final.get('skipped_graph') else '❌')}"
                )
        except Exception as e:
            print(f"❌ [ingestion_graph] '{filename}' unhandled exception: {e}")
            traceback.print_exc()

    _executor.submit(_run)
    print(f"📬 [ingestion_graph] '{filename}' submitted to background pipeline")
