"""
cad_context_bridge.py
─────────────────────────────────────────────────────────────────────────────
Wires the CAD/IFC agent into the shared session memory owned by
core/session_state.py.

cad_ifc_agent.py runs in its own isolated router.  After answering a CAD
question it updates CadSharedContext (its own store) but never tells the
main RAG engine what it found.  This module bridges the gap.

This module provides a single call — `sync_cad_turn_to_main()` — that the
cad_ifc_agent calls at the end of every /api/cad/query response.  It writes:

  1.  Session state (append_turn / log_route) ← so history / direct route knows
  2.  SharedContext.history                      ← so report_agent can reference
  3.  SharedContext.chunks                       ← synthetic chunk from IFC
"""

from __future__ import annotations

import logging
import sys

from core.session_state import append_turn, log_route, get_history

logger = logging.getLogger("cad_context_bridge")


def sync_cad_turn_to_main(
    session_id: str,
    user_query: str,
    assistant_answer: str,
    file_summary: dict,
) -> None:
    """
    Push a completed CAD/IFC turn into the shared session stores.

    Args:
        session_id:        The session this query belongs to.
        user_query:        What the user asked.
        assistant_answer:  What the CAD agent answered.
        file_summary:      The full parsed file summary from CadSharedContext.
    """
    try:
        _write_to_main_sessions(session_id, user_query, assistant_answer)
        _write_to_shared_context(session_id, assistant_answer, file_summary)
        logger.info(f"[cad_bridge] synced turn to main session {session_id[:8]}…")
    except Exception as e:
        # Never crash the CAD agent response — this is best-effort
        logger.warning(f"[cad_bridge] sync failed (non-fatal): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_to_main_sessions(session_id: str, user_query: str, answer: str) -> None:
    """Append this turn to the shared session store."""
    append_turn(session_id, "user",      user_query)
    append_turn(session_id, "assistant", answer)
    log_route(session_id, "cad_ifc", user_query)


def _write_to_shared_context(
    session_id: str,
    answer: str,
    file_summary: dict,
) -> None:
    """Push history + a synthetic CAD chunk into SharedContext."""
    SharedContext = _resolve_shared_context()
    if SharedContext is None:
        logger.debug("[cad_bridge] SharedContext not resolvable — skipping")
        return

    history = get_history(session_id)
    if hasattr(SharedContext, "set_history"):
        SharedContext.set_history(session_id, history)

    chunk = _build_synthetic_chunk(answer, file_summary)
    if hasattr(SharedContext, "set_chunks"):
        SharedContext.set_chunks(session_id, [chunk])


def _resolve_shared_context():
    """Find the SharedContext class via sys.modules (avoids circular imports)."""
    for mod_name in ("services.report_agent", "report_agent"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "SharedContext"):
            return mod.SharedContext
    return None


def _build_synthetic_chunk(answer: str, summary: dict) -> dict:
    """
    Build a dict that looks like a RAG retrieval chunk so SharedContext.chunks
    can carry CAD/IFC data into the report and RAG nodes.
    """
    filename = summary.get("filename", "uploaded file")
    pipeline = summary.get("pipeline", "ifc").upper()
    schema   = summary.get("schema", "")
    storeys  = summary.get("storeys", [])
    elements = summary.get("element_counts") or summary.get("entity_counts") or {}
    materials = summary.get("material_inventory", {})

    # Human-readable element list
    elem_lines = "\n".join(
        f"  - {label}: {count}"
        for label, count in sorted(elements.items(), key=lambda x: -x[1])[:15]
    )
    # Material list
    mat_lines = "\n".join(
        f"  - {mat}: {cnt} elements"
        for mat, cnt in sorted(materials.items(), key=lambda x: -x[1])[:8]
    ) or "  (none detected)"

    # Storey list
    storey_names = ", ".join(s.get("name", "?") for s in storeys[:6]) or "(none)"

    text = (
        f"[CAD/IFC FILE ANALYSIS — {pipeline}]\n"
        f"File: {filename}\n"
        f"Schema: {schema or 'N/A'}\n"
        f"Total elements: {summary.get('total_elements') or summary.get('total_entities') or 0}\n"
        f"Storeys: {storey_names}\n\n"
        f"Element breakdown:\n{elem_lines or '  (none)'}\n\n"
        f"Materials:\n{mat_lines}\n\n"
        f"Last AI answer:\n{answer[:600]}"
    )

    return {
        "text":        text,
        "document_id": f"cad_ifc_{summary.get('filename', 'file')}",
        "filename":    filename,
        "chunk_index": 0,
        "score":       1.0,
        "source":      "cad_ifc_agent",
        "metadata": {
            "pipeline": pipeline,
            "schema":   schema,
            "file_id":  summary.get("file_id", ""),
        },
    }
