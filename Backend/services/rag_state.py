"""
RAG State — AgentState TypedDict and shared constants.

Imported by rag_engine, rag_helpers, rag_nodes, and rag_client.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

# Forward reference for types defined in llm_judge
try:
    from llm_judge import ResponsePlan, ResponseEvaluation
except ImportError:
    from typing import Any as ResponsePlan
    from typing import Any as ResponseEvaluation


class AgentState(TypedDict):
    # input
    query: str
    top_k: int
    conversation_history: List[Dict]  # [{role, content}] prior turns for context

    # routing
    route: Optional[Literal["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph", "report", "image", "cad", "clarify"]]
    force_route: str
    clarification_options: List[str]  # populated by intent classifier for "clarify" route
    full_scope: bool  # True = query needs ALL chunks, not just top_k
    needs_summary: bool  # True = explicit summary request (→ full_doc_node), False = Q&A (→ relevance_node)

    # retrieval
    retrieved_chunks: List[Dict]
    retrieval_iterations: int
    sub_queries: List[str]

    # judge-driven generation
    response_plan: Optional[ResponsePlan]  # NEW: Judge's plan
    response_evaluation: Optional[ResponseEvaluation]  # NEW: Judge's evaluation
    retry_count: int  # NEW: Track retries

    # generation
    context: str
    answer: str
    raw_answer: str
    sources: List[Dict]
    confidence: float
    analytics: Optional[Dict]
    report_id: Optional[str]    # set by report_node
    report_title: Optional[str] # set by report_node
    session_id: str              # passed in from main.py
    user_id: Optional[str]       # passed in from main.py — scopes vector search to per-user collection

    # routing context from previous turn
    prev_route: str
    # full log of {route, query} for this session
    route_log: List[Dict]

    # error
    error: Optional[str]

    # voice call optimisation — skips expensive judge/source/retry nodes
    voice_mode: bool

    # set to True when the user query arrived via voice transcription
    # (frontend appends [voice_input=true] tag — stripped before LLM sees it)
    voice_transcribed: bool

    # user-selected model provider — forwarded to call_llm
    preferred_provider: Optional[str]

    # internal: stores chunks from prior iterative_rag iterations for merge in rerank_merge
    prev_retrieved_chunks: List[Dict]

    # doc scope from intent classifier — resolved filename, positional hint, or ""
    # "" means search all docs; non-empty restricts retrieval to a specific doc
    doc_scope_hint: str

    # filenames of documents explicitly attached to THIS message by the user
    # (populated from pending_doc_ids resolved to filenames by main.py).
    # This is the highest-priority scope signal — overrides the intent classifier.
    attached_doc_filenames: List[str]


# ── Shared constants ─────────────────────────────────────────────────────

MAX_ITER = 3
MAX_RETRIES = 1  # Max retries for quality issues (was 2 — caused up to 7 LLM calls per query)
MIN_CHUNKS = 2
RELEVANCE_THRESHOLD = 0.65
