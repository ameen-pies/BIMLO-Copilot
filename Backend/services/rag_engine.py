"""
Agentic RAG Engine v2 - Judge-Driven (NO HARDCODING)

Key changes from v1:
- NO hardcoded language detection
- NO hardcoded tone/formality rules
- NO if-statements for language/tone
- Judge plans BEFORE generation
- Judge evaluates AFTER generation
- Automatic retry on low quality
- Judge is the brain, not a critic

Flow:
  query → router → retrieve → JUDGE PLANS → generate (following plan) → JUDGE EVALUATES → [retry if needed] → response
"""

from __future__ import annotations

import os
import sys
import re
import json
import requests
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict
from datetime import datetime

# ── Load .env so CF_API_KEY / CF_API_URL are always available ──────────────
try:
    from dotenv import load_dotenv
    # Walk up from this file's directory looking for a .env
    _here = os.path.dirname(os.path.abspath(__file__))
    for _parent in [_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))]:
        _env = os.path.join(_parent, ".env")
        if os.path.exists(_env):
            load_dotenv(_env, override=False)  # override=False: real env vars take priority
            print(f"✅ Loaded .env from {_env}")
            break
except ImportError:
    pass  # dotenv not installed — rely on env vars being set externally

from langgraph.graph import StateGraph, END

# Ensure the directory containing this file is on sys.path so that
# sibling modules (source_agent.py, llm_judge.py) are always importable
# regardless of where uvicorn / Python is launched from.
_services_dir = os.path.dirname(os.path.abspath(__file__))
if _services_dir not in sys.path:
    sys.path.insert(0, _services_dir)

# ── Extracted modules ──────────────────────────────────────────────────────
from rag_state import AgentState, MAX_ITER, MAX_RETRIES, MIN_CHUNKS, RELEVANCE_THRESHOLD
from rag_helpers import (
    _detect_script_language,
    _build_context,
    _build_sources_from_brackets,
    _clean_answer,
    _confidence,
    _is_good_retrieval,
    _resolve_doc_scope,
)
from rag_client import CloudflareClient, GroqClient
from rag_nodes import RAGNodesMixin

# ── Observability (structured pipeline logging) ────────────────────────────────
try:
    from observability import obs as _obs
    _OBS_AVAILABLE = True
except ImportError:
    _obs = None
    _OBS_AVAILABLE = False
    print("⚠️  observability.py not found — structured logging disabled")

# Source Agent — dedicated specialist for source extraction
try:
    from source_agent import build_sources_node
    _SOURCE_AGENT_AVAILABLE = True
except ImportError:
    _SOURCE_AGENT_AVAILABLE = False
    print("⚠️  source_agent.py not found — using legacy _format_sources")

# Graph Agent — chart generation
try:
    from graph_agent import GraphAgent, is_graph_request
    _GRAPH_AGENT_AVAILABLE = True
except ImportError:
    _GRAPH_AGENT_AVAILABLE = False
    print("⚠️  graph_agent.py not found — graph route disabled")

# Import the new judge
try:
    from llm_judge import LLMJudge, ResponsePlan, ResponseEvaluation
except ImportError as e:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from llm_judge import LLMJudge, ResponsePlan, ResponseEvaluation

# Relevance Extractor — map-reduce chunk filtering for full-document Q&A
try:
    from relevance_extractor import extract_relevant_chunks
    _RELEVANCE_EXTRACTOR_AVAILABLE = True
except ImportError:
    _RELEVANCE_EXTRACTOR_AVAILABLE = False
    print("⚠️  relevance_extractor.py not found — full_scope will use full_doc_node")


class RAGEngine(RAGNodesMixin):
    """
    Judge-Driven RAG Engine.

    The LLM Judge is the brain:
    1. Router decides the route (direct/rag/iterative/analytics)
    2. Retrieve gets documents
    3. **JUDGE PLANS** how to respond (language, tone, structure)
    4. Generate follows the plan
    5. **JUDGE EVALUATES** the response
    6. Retry if not acceptable (up to MAX_RETRIES)

    NO hardcoded language detection.
    NO hardcoded formality rules.
    The judge makes ALL decisions.
    """

    def __init__(self, vector_store):
        self.vs     = vector_store
        self.llm    = GroqClient()
        self.judge  = LLMJudge()  # V2 brain
        # Source Agent — dedicated source extraction specialist
        if _SOURCE_AGENT_AVAILABLE:
            self._source_node = build_sources_node(
                api_key=os.getenv("CF_API_KEY", ""),
                # model param kept for SourceAgent API compat (not used by CF worker)
                vector_store=vector_store,   # pass VS so agent can fetch full doc text
            )
        else:
            self._source_node = None
        # Graph Agent — chart generation
        if _GRAPH_AGENT_AVAILABLE:
            self.graph_agent = GraphAgent(
                api_key=os.getenv("CF_API_KEY", ""),
                base_url=os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev"),
            )
        else:
            self.graph_agent = None
        self.graph  = self._build_graph()
        print("🕸️  LangGraph RAG (Judge-Driven + Source Agent + Graph Agent) ready")

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                         #
    # ------------------------------------------------------------------ #

    def query(self, user_query: str, top_k: int = 8, conversation_history: Optional[List[Dict]] = None, prev_route: str = "", route_log: Optional[List[Dict]] = None, status_callback=None, force_route: Optional[str] = None, session_id: str = "", user_id: Optional[str] = None, voice_mode: bool = False, preferred_provider: Optional[str] = None, attached_doc_filenames: Optional[List[str]] = None, cancel_event=None) -> Dict[str, Any]:
        """Main entry point. conversation_history, prev_route, route_log all managed by main.py.

        attached_doc_filenames: filenames of documents explicitly attached to THIS message
            by the user (resolved from pending_doc_ids by main.py before calling here).
        cancel_event: optional threading.Event — when set, long-running nodes (report, graph
            probe) will abort early so the worker thread exits promptly on Stop/disconnect.
            This is the highest-priority scope signal and bypasses the intent classifier.
        """

        # Strip the [voice_input=true] sentinel the frontend appends after transcription.
        # We detect it BEFORE storing anything so the clean query flows everywhere.
        import re as _re_voice
        voice_transcribed = bool(_re_voice.search(r'\[voice_input=true\]', user_query, _re_voice.IGNORECASE))
        if voice_transcribed:
            user_query = _re_voice.sub(r'\s*\[voice_input=true\]', '', user_query).strip()
            print("🎙️  voice_transcribed=True — [voice_input=true] tag stripped from query")

        # Store callback on instance so node wrappers can access it without going through state
        # (state keys starting with _ are stripped by pydantic/langgraph)
        self._status_callback = status_callback or (lambda *_: None)
        self._cancel_event = cancel_event  # threading.Event or None — set by main.py on Stop
        self._voice_mode = voice_mode
        # Store preferred provider so CloudflareClient.complete() can forward it to call_llm
        self._preferred_provider = preferred_provider
        if self.llm:
            self.llm._preferred_provider = preferred_provider

        # In voice mode skip the status-message LLM call — it's a full round-trip
        # just to produce pretty UI strings, which aren't shown during a call anyway.
        if voice_mode:
            self._status_msgs = self._DEFAULT_STATUS_MSGS.copy()
            print("⚡ voice_mode: skipping status-msg LLM pre-call")
        else:
            # Generate contextual status messages in background — ready before nodes need them
            self._status_msgs = self._generate_status_msgs(user_query)

        # If force_route is set, pre-fill the route so the router skips LLM classification
        initial_route = force_route if force_route else None

        initial_state: AgentState = {
            "query": user_query,
            "top_k": top_k,
            "conversation_history": conversation_history or [],
            "prev_route": prev_route,
            "route_log": route_log or [],
            "route": initial_route,
            "force_route": force_route or "",
            "retrieved_chunks": [],
            "prev_retrieved_chunks": [],
            "retrieval_iterations": 0,
            "sub_queries": [],
            "response_plan": None,
            "response_evaluation": None,
            "retry_count": 0,
            "context": "",
            "answer": "",
            "raw_answer": "",
            "sources": [],
            "confidence": 0.0,
            "analytics": None,
            "report_id": None,
            "report_title": None,
            "session_id": session_id,
            "user_id": user_id,
            "error": None,
            "voice_mode": voice_mode,
            "voice_transcribed": voice_transcribed,
            "preferred_provider": preferred_provider,
            "doc_scope_hint": "",
            "full_scope": False,
            "needs_summary": False,
            "clarification_options": [],
            "attached_doc_filenames": attached_doc_filenames or [],
        }

        print(f"\n{'='*80}")
        print(f"🔍 Query: {user_query}")

        # ── Session monitor: track active session ──────────────────────────
        from services.session_monitor import session_monitor
        _user_label = f"user_{(user_id or session_id)[:4]}"
        _initial_route = force_route or "auto"
        session_monitor.start(session_id, _user_label, _initial_route)

        _t0 = time.time()
        try:
            final_state = self.graph.invoke(initial_state)
        finally:
            self._status_callback = None  # always clean up
            self._status_msgs = None
            self._voice_mode = False
        _total_ms = (time.time() - _t0) * 1000

        # ── Observability: log retrieval scores and query end ─────────────────
        if _OBS_AVAILABLE:
            _chunks = final_state.get("retrieved_chunks", [])
            _scores = [c.get("distance", 0.0) for c in _chunks if isinstance(c, dict)]
            if _scores:
                _obs.log_retrieval(
                    session_id=session_id,
                    query=user_query,
                    top_k=top_k,
                    scores=_scores,
                    reranker_used=False,
                    graph_hits=0,
                    latency_ms=_total_ms,
                )
            _eval = final_state.get("response_evaluation")
            _obs.log_query_end(
                session_id=session_id,
                route=final_state.get("route", "unknown"),
                total_latency_ms=_total_ms,
                success=not bool(final_state.get("error")),
                confidence=final_state.get("confidence", 0.0),
                judge_attempts=final_state.get("retry_count", 0) + 1,
                sources_count=len(final_state.get("sources", [])),
                error=str(final_state.get("error") or ""),
            )

        # ── Session monitor: mark complete ─────────────────────────────────
        session_monitor.update(session_id, final_state.get("route", "unknown"), _total_ms)
        session_monitor.complete(
            session_id,
            success=not bool(final_state.get("error")),
            error=str(final_state.get("error")) if final_state.get("error") else None,
        )

        # Build response
        response = {
            "answer": final_state["answer"],
            "raw_answer": final_state.get("raw_answer", final_state["answer"]),
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "route": final_state["route"],
            "analytics": final_state.get("analytics"),
            "report_id": final_state.get("report_id"),
            "report_title": final_state.get("report_title"),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "clarification_options": final_state.get("clarification_options") or [],
            "error": final_state.get("error"),
        }

        # NEW: Include judge's plan and evaluation in debug info
        if final_state.get("response_plan"):
            response["debug_plan"] = final_state["response_plan"].to_dict()
        if final_state.get("response_evaluation"):
            response["debug_evaluation"] = final_state["response_evaluation"].to_dict()

        return response

    # ------------------------------------------------------------------ #
    #  CONTEXTUAL STATUS MESSAGES                                        #
    # ------------------------------------------------------------------ #

    # Default fallbacks — used when LLM generation fails or times out
    _DEFAULT_STATUS_MSGS = {
        "router":           ("🔍", "Understanding your question…"),
        "retrieve_vector":  ("📂", "Searching through documents…"),
        "retrieve_graph":   ("🕸️", "Traversing knowledge graph…"),
        "rerank_merge":     ("🏆", "Re-ranking results for precision…"),
        "retrieve":         ("📂", "Searching through documents…"),
        "check_retrieval":  ("🔎", "Checking search results…"),
        "rewrite_query":   ("✏️",  "Refining the search…"),
        "judge_plan":      ("🧠", "Planning the response…"),
        "synthesise":      ("✍️",  "Generating answer…"),
        "judge_evaluate":  ("⚖️",  "Reviewing quality…"),
        "direct_answer":   ("💬", "Preparing answer…"),
        "transform_node":  ("🔀", "Transforming document…"),
        "analytics_node":  ("📊", "Running analytics…"),
        "define_node":     ("📖", "Looking up definition…"),
        "graph_node":      ("📈", "Building chart from documents…"),
        "cad_node":        ("🏗️", "Analysing CAD/IFC model…"),
        "report_node":     ("📄", "Writing your report…"),
        "image_node":      ("🖼️", "Reading image description…"),
        "clarify_node":    ("🤔", "Preparing clarification…"),
        "full_doc_node":   ("📚", "Reading full document…"),
        "relevance_node":  ("🔍", "Scanning document for relevant content…"),
    }

    def _generate_status_msgs(self, user_query: str) -> Dict[str, tuple]:
        """
        Ask the CF worker to generate short, query-aware status messages for each node.
        Runs synchronously but is called before graph.invoke() so it doesn't add latency
        to the actual answer — it runs while the thread is starting.

        Returns a dict of node_name → (icon, message), falling back to defaults on failure.
        """
        if not self.llm.enabled:
            return self._DEFAULT_STATUS_MSGS.copy()

        q = user_query.strip()[:200]

        prompt = f"""The user asked: "{q}"

Generate short, natural status messages (max 5 words each) for each processing step below.
Messages should feel like they're specifically about THIS query — not generic.

Rules:
- Each message must be 3-6 words, end with "…"
- Reference what the query is actually about when natural
- Keep it natural, like a human narrating what they're doing
- Return ONLY a JSON object with these exact keys, nothing else

Keys and what each step does:
- router: deciding how to handle the query
- retrieve: searching docs for relevant info
- check_retrieval: checking if results are good enough
- rewrite_query: improving the search terms
- judge_plan: planning how to structure the answer
- synthesise: writing the actual answer
- judge_evaluate: checking answer quality
- direct_answer: answering directly without docs
- transform_node: transforming/translating document
- analytics_node: running data analysis
- define_node: explaining a specific term or concept from the documents
- graph_node: extracting data from documents to build a chart

Example for "what is the budget?":
{{"router":"Figuring out your question…","retrieve":"Looking up budget details…","check_retrieval":"Checking what we found…","rewrite_query":"Improving budget search…","judge_plan":"Planning budget summary…","synthesise":"Writing up the numbers…","judge_evaluate":"Double-checking the answer…","direct_answer":"Answering directly…","transform_node":"Transforming document…","analytics_node":"Crunching the numbers…","define_node":"Explaining the term…","graph_node":"Building the chart…"}}

Now generate for: "{q}" """

        try:
            raw = self._chat(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                task="suggest",
            )
            # Parse — handles JSON, Python repr, etc.
            import ast as _ast
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = None
            try:
                parsed = json.loads(clean)
            except Exception:
                try:
                    parsed = _ast.literal_eval(clean)
                except Exception:
                    pass

            if isinstance(parsed, dict):
                icons = {k: v[0] for k, v in self._DEFAULT_STATUS_MSGS.items()}
                result = {}
                for node, default in self._DEFAULT_STATUS_MSGS.items():
                    msg = parsed.get(node, "")
                    if isinstance(msg, str) and 3 <= len(msg) <= 80:
                        # Ensure it ends with …
                        msg = msg.rstrip(".…").rstrip() + "…"
                        result[node] = (default[0], msg)
                    else:
                        result[node] = default
                return result
        except Exception as e:
            print(f"⚠️  Status msg generation failed: {e}")

        return self._DEFAULT_STATUS_MSGS.copy()

    # ------------------------------------------------------------------ #
    #  GRAPH CONSTRUCTION                                                 #
    # ------------------------------------------------------------------ #

    def _is_cancelled(self) -> bool:
        """Check if the user has cancelled the current query."""
        ev = getattr(self, "_cancel_event", None)
        return bool(ev and ev.is_set())

    def _chat(self, *args, **kwargs):
        """Wrapper around self.llm.chat() that checks cancel before each call."""
        if self._is_cancelled():
            raise RuntimeError("query cancelled by user")
        return self.llm.chat(*args, **kwargs)

    def _wrap(self, name: str, fn):
        """Status-aware node wrapper. Fires the status callback before running fn.
        Short-circuits if the user has cancelled the query."""
        default_icon, default_msg = self._DEFAULT_STATUS_MSGS.get(name, ("⚙️", f"{name}…"))
        def _wrapped(state):
            if self._is_cancelled():
                print(f"🛑 [{name}] cancelled — skipping")
                return {**state, "answer": "", "raw_answer": "", "sources": [], "confidence": 0.0}
            cb = getattr(self, "_status_callback", None)
            if callable(cb):
                msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
                icon, msg = msgs.get(name, (default_icon, default_msg))
                cb(name, icon, msg)
            try:
                return fn(state)
            except RuntimeError as e:
                if "query cancelled" in str(e):
                    print(f"🛑 [{name}] cancelled mid-execution")
                    return {**state, "answer": "", "raw_answer": "", "sources": [], "confidence": 0.0}
                raise
        _wrapped.__name__ = name
        return _wrapped

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph routing pipeline.

        Topology (no if/else blocks — every route is a node):

          classify_intent
              ├─ direct      → direct_answer                        → END
              ├─ define      → define_node                          → END
              ├─ image       → image_node                           → END
              └─ *retrieval* → retrieve_vector → retrieve_graph → rerank_merge
                                  ├─ no_docs    → direct_answer     → END
                                  ├─ transform  → transform_node    → END
                                  ├─ graph      → graph_node        → END
                                  ├─ report     → report_node       → END
                                  ├─ analytics  → analytics_node    → END
                                  ├─ rewrite    → rewrite_query ──┐
                                  │                               └→ retrieve_vector (loop)
                                  └─ synthesise → judge_plan → synthesise → judge_evaluate
                                                                  ├─ retry          → synthesise
                                                                  ├─ reroute_direct → direct_answer
                                                                  ├─ analytics      → analytics_node
                                                                  └─ done           → END
        """
        workflow = StateGraph(AgentState)

        # ── Nodes ────────────────────────────────────────────────────────────
        workflow.add_node("classify_intent",  self._wrap("classify_intent",  self._classify_intent_node))
        workflow.add_node("direct_answer",    self._wrap("direct_answer",    self.direct_answer))
        workflow.add_node("retrieve_vector",  self._wrap("retrieve_vector",  self.retrieve_vector))
        workflow.add_node("retrieve_graph",   self._wrap("retrieve_graph",   self.retrieve_graph))
        workflow.add_node("rerank_merge",     self._wrap("rerank_merge",     self.rerank_merge))
        workflow.add_node("rewrite_query",    self._wrap("rewrite_query",    self.rewrite_query))
        workflow.add_node("judge_plan",       self._wrap("judge_plan",       self.judge_plan))
        workflow.add_node("synthesise",       self._wrap("synthesise",       self.synthesise))
        workflow.add_node("judge_evaluate",   self._wrap("judge_evaluate",   self.judge_evaluate))
        workflow.add_node("analytics_node",   self._wrap("analytics_node",   self.analytics_node))
        workflow.add_node("transform_node",   self._wrap("transform_node",   self.transform_node))
        workflow.add_node("define_node",      self._wrap("define_node",      self.define_node))
        workflow.add_node("graph_node",       self._wrap("graph_node",       self.graph_node))
        workflow.add_node("report_node",      self._wrap("report_node",      self.report_node))
        workflow.add_node("image_node",       self._wrap("image_node",       self.image_node))
        workflow.add_node("cad_node",         self._wrap("cad_node",         self.cad_node))
        workflow.add_node("clarify_node",     self._wrap("clarify_node",     self.clarify_node))
        workflow.add_node("full_doc_node",    self._wrap("full_doc_node",    self.full_doc_node))
        workflow.add_node("relevance_node",   self._wrap("relevance_node",   self.relevance_node))

        # no-docs variant — same direct_answer node, flag injected via query prefix
        def _direct_no_docs(state):
            return self.direct_answer({**state, "query": f"__NO_DOCS__:{state['query']}"})
        workflow.add_node("direct_answer_no_docs", _direct_no_docs)

        # ── Entry ─────────────────────────────────────────────────────────────
        workflow.set_entry_point("classify_intent")

        # ── classify_intent → first node (pure dispatch, zero logic) ─────────
        _RETRIEVAL_ROUTES = {"rag", "iterative_rag", "analytics", "transform", "graph", "report"}

        # Routes that have their own full-doc handling — don't divert them
        _FULL_SCOPE_ELIGIBLE = {"rag", "iterative_rag"}

        def _router_dispatch(s):
            route = s.get("route", "rag")
            if route == "direct":   return "direct_answer"
            if route == "define":   return "define_node"
            if route == "image":    return "image_node"
            if route == "cad":      return "cad_node"
            if route == "clarify":  return "clarify_node"
            # full_scope: query needs ALL chunks — divert to map-reduce node
            # Only for rag/iterative_rag. Other routes (transform, analytics, etc.)
            # have their own full-doc handling.
            if s.get("full_scope") and route in _FULL_SCOPE_ELIGIBLE:
                # needs_summary=True → user wants full summary (full_doc_node)
                # needs_summary=False → user wants Q&A across full doc (relevance_node)
                if s.get("needs_summary"):
                    return "full_doc_node"
                if _RELEVANCE_EXTRACTOR_AVAILABLE:
                    return "relevance_node"
                return "full_doc_node"
            if route in _RETRIEVAL_ROUTES: return "retrieve_vector"
            print(f"⚠️  unknown route '{route}' → retrieve_vector")
            return "retrieve_vector"

        workflow.add_conditional_edges(
            "classify_intent",
            _router_dispatch,
            {
                "direct_answer":   "direct_answer",
                "define_node":     "define_node",
                "image_node":      "image_node",
                "cad_node":        "cad_node",
                "clarify_node":    "clarify_node",
                "full_doc_node":   "full_doc_node",
                "relevance_node":  "relevance_node",
                "retrieve_vector": "retrieve_vector",
            },
        )

        # ── Three-stage retrieval pipeline ────────────────────────────────────
        workflow.add_edge("retrieve_vector", "retrieve_graph")
        workflow.add_edge("retrieve_graph",  "rerank_merge")

        def _post_retrieve_dispatch(s):
            chunks = s.get("retrieved_chunks", [])
            route  = s.get("route", "rag")
            if not chunks:
                # Nothing retrieved — check whether session is empty or query just missed
                session_id = s.get("session_id", "")
                user_id    = s.get("user_id")
                try:
                    has_docs = self.vs.has_documents(user_id=user_id, session_id=session_id)
                except Exception:
                    has_docs = True
                if not has_docs:
                    return "direct_answer_no_docs"
                # Query missed but docs exist — fall through to synthesis with no context
            _ROUTE_NODE = {
                "transform": "transform_node",
                "graph":     "graph_node",
                "report":    "report_node",
                "analytics": "analytics_node",
            }
            if route in _ROUTE_NODE:
                return _ROUTE_NODE[route]
            # iterative_rag: keep refining until quality is good or max iterations reached
            if (route == "iterative_rag"
                    and not s.get("voice_mode", False)
                    and s.get("retrieval_iterations", 0) < MAX_ITER
                    and not _is_good_retrieval(chunks)):
                return "rewrite_query"
            return "judge_plan"

        workflow.add_conditional_edges(
            "rerank_merge",
            _post_retrieve_dispatch,
            {
                "direct_answer_no_docs": "direct_answer_no_docs",
                "transform_node":        "transform_node",
                "graph_node":            "graph_node",
                "report_node":           "report_node",
                "analytics_node":        "analytics_node",
                "rewrite_query":         "rewrite_query",
                "judge_plan":            "judge_plan",
            },
        )

        # iterative_rag rewrite loop
        workflow.add_edge("rewrite_query", "retrieve_vector")

        # ── Terminal nodes ────────────────────────────────────────────────────
        for _terminal in ("direct_answer", "direct_answer_no_docs",
                          "transform_node", "define_node",
                          "graph_node", "report_node", "image_node", "cad_node",
                          "clarify_node", "full_doc_node", "relevance_node"):
            workflow.add_edge(_terminal, END)

        # ── Judge-driven synthesis loop ───────────────────────────────────────
        workflow.add_edge("judge_plan",  "synthesise")
        workflow.add_edge("synthesise",  "judge_evaluate")

        workflow.add_conditional_edges(
            "judge_evaluate",
            lambda s: self._should_retry(s),
            {
                "retry":          "synthesise",
                "reroute_direct": "direct_answer",
                "analytics":      "analytics_node",
                "done":           END,
            },
        )

        workflow.add_edge("analytics_node", END)

        return workflow.compile()

    # ------------------------------------------------------------------ #
    #  CLASSIFY INTENT NODE                                               #
    # ------------------------------------------------------------------ #

    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """
        Single-stage intent classifier node — the ONLY node that writes state["route"].

        Calls intent_classifier.classify_intent() which internally runs:
          • An LLM call with chain-of-thought reasoning
          • Falls back to _heuristic_classify() if the LLM is unavailable

        All keyword-matching, bypass logic, and route safety-guards live inside
        intent_classifier.py.  Nothing is duplicated here.
        """
        # Skip when force_route was set by the caller (e.g. test harness)
        if state.get("route"):
            print(f"⚡ classify_intent: force_route={state['route']} — skipping")
            if _OBS_AVAILABLE:
                _obs.log_routing(
                    session_id=state.get("session_id", ""),
                    query=state["query"],
                    route=state["route"],
                    confidence=1.0,
                    forced=True,
                )
            return state

        # Session monitor: track stage
        from services.session_monitor import session_monitor as _sm
        _sm.update(state.get("session_id", ""), "classify_intent")

        query      = state["query"]
        history    = state.get("conversation_history", [])
        route_log  = state.get("route_log", [])
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        attached   = state.get("attached_doc_filenames", [])

        try:
            session_has_docs = self.vs.has_documents(user_id=user_id, session_id=session_id)
        except Exception:
            session_has_docs = False

        try:
            all_docs = self.vs.list_documents(user_id=user_id, session_id=session_id)
        except Exception:
            all_docs = []

        uploaded_filenames = [d.get("filename", "") for d in all_docs if d.get("filename")]

        print(f"📍 classify_intent (has_docs={session_has_docs}) → ", end="")

        try:
            from intent_classifier import classify_intent
            intent = classify_intent(
                query,
                history,
                route_log,
                preferred_provider=state.get("preferred_provider"),
                session_has_docs=session_has_docs,
                uploaded_docs=uploaded_filenames,
                attached_doc_filenames=attached,
            )
        except Exception as e:
            print(f"[intent_classifier error: {e}] → heuristic fallback → ", end="")
            from intent_classifier import _heuristic_classify
            intent = _heuristic_classify(
                query, history, route_log,
                session_has_docs=session_has_docs,
                uploaded_docs=uploaded_filenames,
                attached_doc_filenames=attached,
            )

        route = intent.suggested_route

        # image_query always uses the image node regardless of suggested_route
        if intent.primary_intent == "image_query" and route == "rag":
            route = "image"

        # Guard: cad route requires the query to actually be about CAD/IFC content.
        # If the classifier suggested "cad" but the intent isn't cad_query (e.g. because
        # a CAD file exists in session but the query is about a PDF), fall back to rag.
        if route == "cad" and intent.primary_intent != "cad_query":
            print(f"⚠️  classify_intent: cad route blocked — intent={intent.primary_intent!r} (not cad_query) → rag")
            route = "rag"

        # Guard: full_scope requires session to have documents
        full_scope = getattr(intent, "full_scope", False)
        if full_scope and not session_has_docs:
            print(f"⚠️  classify_intent: full_scope blocked — no docs in session")
            full_scope = False

        print(f"{route} (conf={intent.confidence:.2f}, scope={intent.doc_scope!r}, full_scope={full_scope})")

        if _OBS_AVAILABLE:
            _obs.log_routing(
                session_id=session_id,
                query=query,
                route=route,
                confidence=intent.confidence,
                forced=False,
            )

        return {
            **state,
            "route":                 route,
            "doc_scope_hint":        intent.doc_scope,
            "clarification_options": getattr(intent, "clarification_options", []) or [],
            "full_scope":            full_scope,
            "needs_summary":         getattr(intent, "needs_summary", False),
        }

    def direct_answer(self, state: AgentState) -> AgentState:
        """Handle direct questions without retrieval. Uses fallback plan — no LLM judge call needed."""
        query    = state["query"]
        history  = state.get("conversation_history", [])
        route_log = state.get("route_log", [])
        session_id = state.get("session_id", "")

        # Detect no-docs redirect from check_retrieval
        no_docs = query.startswith("__NO_DOCS__:")
        if no_docs:
            query = query[len("__NO_DOCS__:"):]

        # Even if no_docs flag isn't set, double-check actual session state.
        # This prevents the LLM from being misled by a stale route_log entry
        # that shows a failed RAG attempt from before any documents were uploaded.
        user_id = state.get("user_id")
        if not no_docs:
            try:
                has_docs = self.vs.has_documents(user_id=user_id, session_id=session_id)
            except Exception:
                has_docs = True  # safe default — don't claim no docs if we can't check

        plan = self.judge._fallback_plan(query)

        print(f"💬 Direct answer (language: {plan.target_language}, tone: {plan.target_tone}, no_docs={no_docs})")

        answer = self._generate_direct_answer(
            query, plan, history,
            no_docs=no_docs,
            route_log=route_log,
            session_has_docs=has_docs if not no_docs else False,
        )

        return {
            **state,
            "query": query,   # restore clean query without flag
            "answer": answer,
            "response_plan": plan,
            "confidence": 1.0,
        }

    def _generate_direct_answer(self, query: str, plan: ResponsePlan, history: Optional[List[Dict]] = None, no_docs: bool = False, route_log: Optional[List[Dict]] = None, session_has_docs: bool = True) -> str:
        """Generate a direct answer using conversation history as the primary context."""

        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang
        lang_rule = (
            f"🚨 LANGUAGE REQUIREMENT: You MUST write your ENTIRE answer in {target_lang.upper()}. "
            f"The user asked in {user_lang.upper()}. Every single sentence must be in {target_lang.upper()}. "
            f"This is the most important rule."
        )

        if no_docs:
            system_content = (
                f"You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE — a company specialising in BIM engineering (3D to 7D digital models), Scan to BIM, BIM 4D construction planning, telecom infrastructure studies (rooftop, pylons, calculation notes), and DeepTwin AI digital twins for predictive maintenance. "
                f"Respond in {target_lang.upper()} using a {plan.target_tone} tone. "
                f"{lang_rule} "
                f"The user asked a question that would normally use uploaded documents, but there are no files uploaded in this chat session yet. "
                f"Mention this clearly in your answer, and offer to help further once they upload a document. "
                f"Keep the reply friendly and concise."
            )
        else:
            # Serialize the route log as plain text and give it directly to the LLM.
            # The LLM understands context — no need to hardcode what each route means.
            session_context = ""
            if route_log:
                log_lines = "\n".join(
                    f"- [{e['route']}] {e['query']}" for e in route_log
                )
                session_context = (
                    f"\n\nThis session's action log (what you did and how):\n{log_lines}\n"
                    f"Use this to understand what has already happened in the conversation, "
                    f"regardless of which internal mechanism produced each answer."
                )

            # Inject current document availability so the LLM isn't misled by a
            # stale route_log entry showing a failed RAG attempt from before upload.
            if session_has_docs:
                doc_status_note = (
                    "\n\nIMPORTANT: The user HAS already uploaded documents to this chat session. "
                    "If the action log shows a previous failed document search, that happened "
                    "before the upload — documents ARE now available in this session. "
                    "If the user uploaded an IMAGE file: the system ran a vision LLM on it at upload time "
                    "and stored a full visual description in the vector store. You can answer questions "
                    "about what is in the image — you don't need to say you cannot see images. "
                    "The vision description was already extracted and is searchable."
                )
            else:
                doc_status_note = (
                    "\n\nNote: No documents have been uploaded to this chat session yet."
                )

            system_content = (
                f"You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE — a company specialising in BIM engineering (3D to 7D digital models), Scan to BIM, BIM 4D construction planning, telecom infrastructure studies (rooftop, pylons, calculation notes), and DeepTwin AI digital twins for predictive maintenance. "
                f"You help professionals in construction, BTP, and telecom industries with technical questions and document analysis. "
                f"Respond in {target_lang.upper()} using a {plan.target_tone} tone. "
                f"{lang_rule} "
                f"You are one single assistant — the conversation history and action log below "
                f"are all yours. Use them freely to answer follow-up questions, recall what was "
                f"said or done, and modify previous answers when asked."
                f"{session_context}"
                f"{doc_status_note}"
            )

        messages: List[Dict] = [{"role": "system", "content": system_content}]

        # Inject conversation history so memory recall works correctly
        if history:
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        if self.llm.enabled:
            return self._chat(messages)
        else:
            return f"[Response in {plan.target_language} - LLM unavailable]"

    def _emit(self, node: str, icon: str, message: str):
        """Fire a live status event from inside a node with real runtime context."""
        cb = getattr(self, "_status_callback", None)
        if callable(cb):
            cb(node, icon, message)

    # ------------------------------------------------------------------ #
    #  RETRIEVAL NODES                                                    #
    # ------------------------------------------------------------------ #

    # ─────────────────────────────────────────────────────────────────────
    # RETRIEVAL PIPELINE — three dedicated LangGraph nodes
    #
    #   retrieve_vector  →  retrieve_graph  →  rerank_merge
    #
    # Each has its own state slice, status callback, and error isolation.
    # The old monolithic retrieve() is kept as a private helper only for
    # the iterative_rag loop (rewrite_query → retrieve → check_retrieval).
    # ─────────────────────────────────────────────────────────────────────

    def retrieve_vector(self, state: AgentState) -> AgentState:
        """
        Node 1/3: fetch up to fetch_k candidates from ChromaDB.
        Stores raw results in retrieved_chunks; graph and reranker run next.
        """
        from services.session_monitor import session_monitor as _sm
        _sm.update(state.get("session_id", ""), "retrieve_vector")

        query = state["query"]
        top_k = state["top_k"]
        iteration = state["retrieval_iterations"] + 1

        print(f"🔎 [retrieve_vector] #{iteration} (+{top_k} candidates) → ", end="")
        print(f"[attached={state.get('attached_doc_filenames') or []}] ", end="")

        session_id = state.get("session_id", "")
        user_id = state.get("user_id")
        all_docs = self.vs.list_documents(user_id=user_id, session_id=session_id)

        if state["route"] in ("rag", "iterative_rag", "analytics", "transform", "graph", "report"):
            if not self.vs.has_documents(user_id=user_id, session_id=session_id):
                print(f"⚠️  [retrieve_vector] no documents available for user={user_id!r} session={session_id!r}; skipping retrieval")
                return {
                    **state,
                    "retrieved_chunks": [],
                    "retrieval_iterations": iteration,
                }

        # ── Doc-scope filtering ───────────────────────────────────────────────
        # Priority: 1) intent classifier doc_scope  2) exact filename in query text
        # doc_scope may be an exact filename or a positional hint like "first"/"second".
        filter_dict = None
        doc_scope_hint = state.get("doc_scope_hint", "")

        if doc_scope_hint:
            # Try to match doc_scope against the actual doc list
            resolved = _resolve_doc_scope(
                scope_hint=doc_scope_hint,
                all_docs=all_docs,
                query=query,
            )
            if resolved:
                filter_dict = {"filename": resolved}
                print(f"[scope→{resolved}] ", end="")

        if filter_dict is None:
            # Fallback: exact filename substring match in the raw query text
            for doc in all_docs:
                fname = doc.get("filename", "")
                if fname and fname.lower() in query.lower():
                    filter_dict = {"filename": fname}
                    print(f"[filtered to {fname}] ", end="")
                    break

        # Fetch MORE candidates than top_k so the reranker has room to work
        try:
            from reranker import get_fetch_k
            fetch_k = get_fetch_k(top_k)
        except ImportError:
            fetch_k = top_k

        # ── Multi-doc: search all attached files and merge ────────────────────
        # When multiple docs are attached and no specific file was targeted,
        # search each attached file individually and merge results.
        attached = state.get("attached_doc_filenames") or []
        # Fallback: if frontend didn't send attached list, use all session docs
        if not attached and not filter_dict:
            attached = [d.get("filename") for d in all_docs if d.get("filename")]
        if len(attached) > 1 and filter_dict is None:
            all_chunks: List[Dict] = []
            per_file_k = max(fetch_k // len(attached), 5)
            for fname in attached:
                try:
                    hits = self.vs.search(
                        query,
                        top_k=per_file_k,
                        filter_dict={"filename": fname},
                        user_id=user_id,
                        session_id=session_id,
                    )
                    all_chunks.extend(hits)
                except Exception:
                    pass
            if all_chunks:
                results = all_chunks
                print(f"[multi-doc: {len(attached)} files → {len(results)} chunks] ", end="")
            else:
                results = self.vs.search(query, top_k=fetch_k, user_id=user_id, session_id=session_id)
        else:
            try:
                results = self.vs.search(
                    query,
                    top_k=fetch_k,
                    filter_dict=filter_dict,
                    user_id=user_id,
                    session_id=session_id,
                )
            except Exception:
                results = self.vs.search(query, top_k=fetch_k, user_id=user_id, session_id=session_id)

        print(f"{len(results)} vector chunks")

        return {
            **state,
            "retrieved_chunks": results,
            "retrieval_iterations": iteration,
        }

    def retrieve_graph(self, state: AgentState) -> AgentState:
        """
        Node 2/3: fetch relationship-aware context from the Neo4j knowledge graph.
        Only runs when the query contains graph-traversal signals (topology, connected-to,
        etc.).  Results are APPENDED to retrieved_chunks from the previous node.
        Skipped silently if graph_rag is unavailable or the query has no graph signals.
        """
        query  = state["query"]
        chunks = state["retrieved_chunks"]

        graph_chunks: List[Dict] = []
        try:
            from graph_rag import get_engine as _get_graph_engine, is_graph_query
            if is_graph_query(query):
                self._emit("retrieve_graph", "🕸️", "Traversing knowledge graph…")
                graph_engine = _get_graph_engine()
                if graph_engine.available:
                    graph_chunks = graph_engine.query(query)
                    if graph_chunks:
                        print(f"🕸️  [retrieve_graph] +{len(graph_chunks)} graph chunks")
        except ImportError:
            pass  # graph_rag optional — vector-only mode
        except Exception as e:
            print(f"⚠️  [retrieve_graph] error: {e}")

        if not graph_chunks:
            return state  # nothing to merge, pass through unchanged

        # Merge graph chunks ahead of vector results, deduplicate
        combined = graph_chunks + chunks
        seen: set = set()
        unique: List[Dict] = []
        for c in combined:
            txt = c["text"][:200]
            if txt not in seen:
                seen.add(txt)
                unique.append(c)

        return {**state, "retrieved_chunks": unique}

    def rerank_merge(self, state: AgentState) -> AgentState:
        """
        Node 3/3: cross-encoder re-ranking + graph-chunk boost + iterative merge.

        - Scores every (query, chunk) pair with BAAI/bge-reranker-v2-m3.
        - Graph chunks get +0.3 additive boost AFTER honest scoring, so they
          surface above vector chunks for relationship queries but only when
          genuinely relevant.
        - Merges with chunks from prior retrieval iterations (iterative_rag).
        - Emits status callbacks for files found and visual/table chunks.
        """
        query  = state["query"]
        top_k  = state["top_k"]
        unique = state["retrieved_chunks"]

        # ── Cross-encoder re-ranking ──────────────────────────────────────────
        try:
            from reranker import rerank
            ranked = rerank(query, unique, top_k=top_k * 2)  # extra budget for boost sort
            # Post-rerank additive boost for graph knowledge chunks
            n_graph = 0
            for c in ranked:
                if c.get("_is_graph_chunk") or c.get("metadata", {}).get("source") == "knowledge_graph":
                    boosted = min(1.0, c.get("rerank_score", 0.5) + 0.3)
                    c["rerank_score"] = boosted
                    c["distance"]     = 1.0 - boosted
                    n_graph += 1
            if n_graph:
                ranked = sorted(ranked, key=lambda c: c.get("rerank_score", 0), reverse=True)
            unique = ranked[:top_k]
            if n_graph:
                print(f"🏆 [rerank_merge] {len(unique)} chunks (graph boost: {n_graph})")
        except ImportError:
            unique = unique[:top_k]
        except Exception as e:
            print(f"⚠️  [rerank_merge] reranker error: {e}")
            unique = unique[:top_k]

        # ── Merge with prior iteration chunks (iterative_rag loop) ───────────
        existing = [c for c in state.get("prev_retrieved_chunks", [])]
        if existing:
            all_chunks = existing + unique
            seen2: set = set()
            final: List[Dict] = []
            for c in all_chunks:
                txt = c["text"][:200]
                if txt not in seen2:
                    seen2.add(txt)
                    final.append(c)
            unique = final[:top_k * 2]

        print(f"📦 [rerank_merge] {len(unique)} total chunks ready")

        # ── Status emissions ─────────────────────────────────────────────────
        filenames = list(dict.fromkeys(
            c["metadata"].get("filename", "unknown") for c in unique
        ))
        if filenames:
            if len(filenames) == 1:
                self._emit("rerank_merge", "📄", f"Reading {filenames[0]}…")
            else:
                self._emit("rerank_merge", "📂", f"Found {len(filenames)} files: {', '.join(filenames[:3])}…")

        visual_chunks = [c for c in unique if c["metadata"].get("has_images")]
        table_chunks  = [c for c in unique if c["metadata"].get("has_tables")]
        if visual_chunks:
            self._emit("rerank_merge", "🖼️", f"Found {len(visual_chunks)} chunk(s) with diagram descriptions…")
        if table_chunks:
            self._emit("rerank_merge", "📊", f"Found {len(table_chunks)} chunk(s) with table data…")

        return {
            **state,
            "retrieved_chunks":       unique,
            "prev_retrieved_chunks": unique,  # stored for next iteration if iterative_rag
        }

    # ── Legacy retrieve() — kept for backwards compat; delegates to the three nodes ──

    def rewrite_query(self, state: AgentState) -> AgentState:
        """
        Rewrite query to improve retrieval.
        Uses context about what was already retrieved and what docs exist
        so it can form genuinely different, targeted search terms.
        """
        original = state["query"]
        existing_chunks = state.get("retrieved_chunks", [])

        print(f"🔄 Rewriting query for better retrieval → ", end="")

        if not self.llm.enabled:
            print("(LLM disabled)")
            return state

        # Build context about what we already found (or didn't find)
        retrieved_info = ""
        if existing_chunks:
            filenames = list(dict.fromkeys(
                c.get("metadata", {}).get("filename", "") for c in existing_chunks if c.get("metadata", {}).get("filename")
            ))
            top_excerpt = existing_chunks[0].get("text", "")[:300] if existing_chunks else ""
            retrieved_info = (
                f"\nDocuments found so far: {', '.join(filenames)}"
                f"\nBest retrieved excerpt so far:\n\"{top_excerpt}\""
                f"\nProblem: these results don't fully answer the question."
            )
        else:
            retrieved_info = "\nProblem: no relevant chunks were retrieved at all."

        prompt = f"""You are helping a technical document search system improve its query.

ORIGINAL QUERY: {original}
{retrieved_info}

Your job: rewrite the query to find DIFFERENT, more relevant content.
Strategies:
- Use different keywords or synonyms for the same concept
- Break a complex question into its most searchable core
- Use domain-specific terminology a technical document would use
- If the original is a question, rewrite as the kind of phrase that would appear as an answer in a document

Reply with ONLY the rewritten search query. No explanation."""

        rewritten = self._chat(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
        ).strip().strip('"')

        print(f'"{rewritten}"')
        self._emit("rewrite_query", "✏️", f"Searching for: {rewritten[:60]}…")

        sub_queries = state["sub_queries"] + [rewritten]

        return {
            **state,
            "query": rewritten,
            "sub_queries": sub_queries,
        }

    # ------------------------------------------------------------------ #
    #  JUDGE PLANNING NODE (NEW)                                          #
    # ------------------------------------------------------------------ #

    def judge_plan(self, state: AgentState) -> AgentState:
        """
        Ask the judge to plan how to respond.
        Passes enough context (history + doc content) for the judge to make
        genuinely useful decisions about structure, style, and key points.
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        history = state.get("conversation_history", [])

        if state.get("voice_mode"):
            plan = self.judge._fallback_plan(query)
            plan = ResponsePlan(**{
                **plan.to_dict(),
                "target_tone":         "conversational",
                "response_style":      "concise",
                "max_response_length": "brief",
                "should_cite_sources": False,
            })
            print("⚡ judge_plan: voice_mode preset (conversational/concise/no-cite)")
            return {**state, "response_plan": plan}

        print(f"🧠 Judge planning response → ", end="")

        # Pass more history context — 300 chars per message, last 6 messages
        history_texts = [
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in history[-6:]
        ]

        # Detect user language BEFORE planning so we can cross-check the judge's output
        user_language = _detect_script_language(query)

        plan = self.judge.plan_response(query, retrieved_docs=chunks, conversation_history=history_texts, detected_language=user_language)

        # ── Language override: trust script detection over judge when they disagree ──
        # The judge sometimes lets document language leak into target_language.
        # Script-based detection on the user's query is the ground truth.
        if user_language and plan.target_language != user_language:
            # Only override if the detected language is confident (non-English scripts are always confident)
            # For Latin script (en/fr disambiguation), check if judge has low confidence
            is_non_latin = user_language in ('ar', 'zh', 'ko', 'ru', 'he', 'th', 'hi')
            judge_low_conf = plan.target_language_confidence < 0.8
            if is_non_latin or judge_low_conf:
                print(f"⚠️  Language override: judge said '{plan.target_language}' but query is '{user_language}' — using '{user_language}'")
                plan = ResponsePlan(**{
                    **plan.to_dict(),
                    "target_language": user_language,
                    "target_language_confidence": 0.95,
                })

        print(f"{plan.target_language}/{plan.target_tone}/{plan.response_style} | key_points={len(plan.key_points_to_include)}")

        if plan.key_points_to_include:
            pts = plan.key_points_to_include[:2]
            label = " & ".join(pts) if len(pts) > 1 else pts[0]
            self._emit("judge_plan", "🧠", f"Covering: {label[:60]}…")

        return {
            **state,
            "response_plan": plan,
        }

    # ------------------------------------------------------------------ #
    #  SYNTHESIS NODE (FOLLOWING JUDGE'S PLAN)                            #
    # ------------------------------------------------------------------ #

    def synthesise(self, state: AgentState) -> AgentState:
        """
        Generate the final answer by FOLLOWING the judge's plan.

        The plan tells us:
        - What language to use
        - What tone to use
        - How to structure the response
        - What to include/avoid
        """
        from services.session_monitor import session_monitor as _sm
        _sm.update(state.get("session_id", ""), "synthesise")

        query = state["query"]
        chunks = state["retrieved_chunks"]
        plan = state["response_plan"]
        retry_count = state.get("retry_count", 0)
        evaluation = state.get("response_evaluation")

        if not plan:
            print("❌ No response plan available")
            return {**state, "error": "No response plan"}

        print(f"✍️  Generating (attempt {retry_count + 1}, following plan) → ", end="")

        # Emit the primary file being synthesised from
        if chunks:
            primary_file = chunks[0]["metadata"].get("filename", "")
            if primary_file:
                self._emit("synthesise", "✍️", f"Writing from {primary_file}…")

        # Build context
        context = _build_context(chunks)

        # On retry the plan itself has been updated by _should_retry —
        # the fix instruction is already baked into plan.approach.
        # We pass an empty fix_instruction here so we don't double-inject it.
        voice_transcribed = state.get("voice_transcribed", False)
        user_language = _detect_script_language(query)
        prompt = self._build_synthesis_prompt(query, context, plan, fix_instruction="", voice_transcribed=voice_transcribed, user_language=user_language)

        # Build messages — prepend conversation history so the model can handle
        # modification requests ("make it shorter", "change X to Y") by seeing
        # what it previously said. No truncation — the frontend already caps history.
        history = state.get("conversation_history", [])
        history_msgs: List[Dict] = []
        if history:
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        # Generate answer
        if self.llm.enabled:
            answer = self._chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2500,
            )
        else:
            answer = self._fallback_synthesis(chunks, plan)

        import re as _re

        # ── Kill orphan periods immediately after LLM output ──────────────
        # The model sometimes outputs a sentence on one line and its closing
        # period on the next, e.g. "some text\n.\n- next bullet".
        # Attack it at every possible form before anything else runs:

        # 1. Period on its own line between list items or paragraphs → delete the line
        answer = _re.sub(r'\n[ \t]*\.[ \t]*\n', '\n', answer)
        # 2. Period on its own line at end of string
        answer = _re.sub(r'\n[ \t]*\.[ \t]*$', '', answer.rstrip())
        # 3. Period-only line that follows a bullet line → attach to previous line
        answer = _re.sub(r'([-*+][^\n]+)\n[ \t]*\.[ \t]*\n', r'\1\n', answer)
        # 4. Any remaining line consisting of only a period (with optional spaces)
        answer = _re.sub(r'(?m)^[ \t]*\.[ \t]*$', '', answer)
        # 5. Collapse any triple+ newlines that removals above may have created
        answer = _re.sub(r'\n{3,}', '\n\n', answer)

        # Strip markdown links [text](url) → text
        answer = _re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', answer)

        # Check [N] citation markers
        cited_nums = sorted(set(int(m) for m in _re.findall(r'\[(\d+)\]', answer)))
        print(f"✍️  Citations found: {cited_nums}")
        print(f"   Answer RAW:\n{answer}\n---END RAW---")

        # Emit section headings from the generated answer
        headings = _re.findall(r'^#{1,3}\s+(.+)$', answer, _re.MULTILINE)
        if headings:
            self._emit("synthesise", "📝", f"Sections: {', '.join(h.strip() for h in headings[:3])}…")

        # If LLM produced zero citations, add a single [1] at the end only —
        # do NOT stamp every paragraph, that creates false sources.
        if not cited_nums and chunks:
            answer = answer.rstrip() + " [1]"
            cited_nums = [1]
            print("⚠️  No [N] citations — appended single [1]")

        # Clean the answer first — needed by both source paths below
        clean_answer = _clean_answer(answer)

        # RAG routes always use sources — override judge if needed
        route = state.get("route", "")
        if route in ("rag", "iterative_rag", "analytics") and not plan.should_cite_sources:
            print("⚠️  Judge said no-cite but route is RAG — overriding to True")
            plan = ResponsePlan(**{**plan.to_dict(), "should_cite_sources": True})

        # Build sources only when the judge's plan calls for citations AND not voice_mode.
        # Citations are meaningless spoken aloud, and source extraction is expensive.
        if plan.should_cite_sources and not state.get("voice_mode"):
            if self._source_node and _SOURCE_AGENT_AVAILABLE:
                # Emit which files we're extracting sources from
                source_files = list(dict.fromkeys(
                    c["metadata"].get("filename", "") for c in chunks if c["metadata"].get("filename")
                ))
                for fname in source_files[:3]:
                    self._emit("synthesise", "🔗", f"Extracting sources from {fname}…")
                intermediate = {**state, "answer": clean_answer, "retrieved_chunks": chunks}
                intermediate = self._source_node(intermediate)
                built_sources = intermediate.get("sources", [])
                if not built_sources:
                    built_sources = _build_sources_from_brackets(answer, chunks)
            else:
                built_sources = _build_sources_from_brackets(answer, chunks)
            print(f"📋 Sources built: {len(built_sources)}")
        else:
            built_sources = []
            print(f"📋 Sources skipped (plan: should_cite_sources=False)")

        return {
            **state,
            "context": context,
            "answer": clean_answer,
            "raw_answer": clean_answer,  # same — [N] markers are the citation tokens now
            "sources": built_sources,
            "confidence": _confidence(chunks),
        }

    def _build_synthesis_prompt(self, query: str, context: str, plan: ResponsePlan, fix_instruction: str = "", voice_transcribed: bool = False, user_language: str = "") -> str:
        tone_map = {
            'casual': 'casual and friendly', 'conversational': 'conversational',
            'friendly': 'warm and friendly', 'professional': 'professional', 'technical': 'technical',
        }
        tone = tone_map.get(plan.target_tone, plan.target_tone)

        # Detect the user's language from the query if not provided
        if not user_language:
            user_language = _detect_script_language(query)
        # Use plan.target_language as the authoritative choice (judge may have overridden)
        target_lang = plan.target_language or user_language

        # ── Strong language mirroring instruction ──────────────────────────────
        # This is the #1 issue: the model tends to answer in the document's language
        # rather than the user's language. We need a very explicit, hard-to-ignore rule.
        language_rule = (
            f"\n🚨 LANGUAGE REQUIREMENT (CRITICAL): The user asked you in {user_language.upper()}. "
            f"You MUST write your ENTIRE answer in {target_lang.upper()}. "
            f"The documents below may be in a different language — IGNORE their language. "
            f"Your response language is {target_lang.upper()}, period. "
            f"Every single sentence you write must be in {target_lang.upper()}. "
            f"This is the most important instruction."
        )

        # ── Translate plan fields into concrete formatting instructions ─────────
        # response_style and max_response_length are meaningless unless we translate them
        style_instruction = ""
        if plan.response_style == "concise":
            style_instruction = "Keep the answer tight: 2-4 sentences or a short bullet list. Cut anything that doesn't directly answer the question."
        elif plan.response_style == "bullet_points":
            style_instruction = "Structure the answer as a clean bullet list. Each bullet = one distinct fact or item. No walls of text."
        elif plan.response_style == "narrative":
            style_instruction = "Write in flowing prose paragraphs. No bullet points unless listing 4+ items."
        elif plan.response_style == "technical":
            style_instruction = "Use precise technical language. Preserve exact values, standards codes, and technical terms from the documents. Use bullets for spec lists."
        elif plan.response_style == "detailed":
            style_instruction = "Be thorough. Cover all relevant aspects found in the documents. Use sections if there are 3+ distinct topics."
        else:
            style_instruction = "Match the format to the content: short answer for simple questions, structured for complex ones."

        length_instruction = ""
        if plan.max_response_length == "brief":
            length_instruction = "Length: 1-3 sentences maximum. Be direct."
        elif plan.max_response_length == "comprehensive":
            length_instruction = "Length: Cover the topic fully. Don't omit relevant details from the documents."
        else:
            length_instruction = "Length: As long as needed to fully answer, no longer."

        # ── Key points to cover (from judge's analysis of the docs) ───────────
        key_points_block = ""
        if plan.key_points_to_include:
            pts = "\n".join(f"  - {p}" for p in plan.key_points_to_include[:6])
            key_points_block = f"\nKey aspects to address:\n{pts}\n"

        # ── Approach and correction directives ─────────────────────────────────
        approach_block = ""
        if plan.approach and "CORRECTION REQUIRED" in plan.approach:
            approach_block = f"\n⚠️ CORRECTION REQUIRED: {plan.approach.split('CORRECTION REQUIRED:')[-1].strip()}\n"
        elif plan.approach:
            approach_block = f"\nStrategy: {plan.approach}\n"

        avoid_block = ""
        if plan.things_to_avoid:
            avoid_block = f"\nDo NOT: {'; '.join(plan.things_to_avoid[:5])}\n"

        if fix_instruction:
            approach_block += f"\n{fix_instruction}\n"

        # ── Visual content ─────────────────────────────────────────────────────
        has_visual_chunks = "[IMAGE " in context or "[TABLE " in context or "[UPLOADED IMAGE:" in context
        visual_instruction = ""
        if has_visual_chunks:
            visual_instruction = (
                "\n\nVISUAL & TABLE CONTENT:\n"
                "- Some sources contain [IMAGE on page N: <description>], [TABLE on page N], "
                "or [UPLOADED IMAGE: <filename>] blocks. These are AI-generated descriptions of diagrams, "
                "tables, or user-uploaded images. Treat them as factual content.\n"
                "- IMAGE/DIAGRAM EXPLANATION: When context includes image descriptions, describe what the "
                "diagram shows — explain its components, relationships, flow, and data values. Reference by "
                "page (e.g. 'The diagram on page 3 shows…'). Never say you cannot see images.\n"
                "- TABLE GENERATION: You MAY create markdown tables in your answer when they improve clarity. "
                "Use cases: comparing items, listing specs/properties, summarizing structured data, organizing "
                "steps or phases. Format: `| Header | Header |` with `|---|---|` separator. Keep tables small "
                "(3-7 columns, 3-15 rows). You may also reorganize existing document tables into cleaner format.\n"
                "- Never reproduce the raw [IMAGE] or [TABLE] tags in your answer."
            )

        voice_note = ""
        if voice_transcribed:
            voice_note = (
                "\n\nVOICE INPUT: The user spoke this question — respond naturally and conversationally. "
                "Never mention transcription or voice."
            )

        return f"""You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE (BIM engineering, telecom infrastructure, DeepTwin AI digital twins). You help professionals in construction and telecom analyse technical documents.

Language: {target_lang.upper()} | Tone: {tone}
{language_rule}
{style_instruction}
{length_instruction}{approach_block}{key_points_block}{avoid_block}{visual_instruction}{voice_note}

CORE RULES:
1. Answer DIRECTLY — never start with "Introduction to X", "Overview of X", generic definitions, or preamble. The first sentence must address the question.
2. Extract and use ALL relevant information from every source. Do not stop at the first paragraph. If page 3 or source 4 has the answer, use it.
3. Be SPECIFIC: quote exact values, names, codes, standards, section numbers, dates from the documents. Never generalize when the document has specifics.
4. If the question asks for one specific fact (e.g. "who is the host company?", "what is the wind speed?"), state that fact immediately in the first sentence — then add context if useful.
5. Never invent information. If something is not in the documents, say so clearly.
6. NEVER add unsolicited advice, disclaimers, or generic background that wasn't asked for.

LANGUAGE:
7. 🚨 You MUST write your ENTIRE answer in {target_lang.upper()}. The documents below may be in a different language — do NOT follow their language. Follow the user's language.

FORMATTING:
- Simple/factual questions: answer in 1-3 sentences. No sections needed.
- Multi-part or complex questions: use ## headings for each distinct topic, then bullets or prose under each.
- Use **bold** for key terms, values, and technical codes.
- Bullet lists only when there are 3+ discrete items to enumerate.
- Never use a bullet that is just a citation marker. Never leave a heading empty.

CITATIONS: After every sentence drawn from a document, add [N] where N matches the source number in the headers below. Cite precisely — do not add [1] to every sentence blindly.

USER QUESTION: {query}

DOCUMENTS:
{context}

Remember: Answer in {target_lang.upper()}. The document language is irrelevant. Your ENTIRE answer must be in {target_lang.upper()}."""


    def _fallback_synthesis(self, chunks: List[Dict], plan: ResponsePlan) -> str:
        """
        Fallback when LLM is unavailable.

        Produces a structured, readable digest of the retrieved documents instead of
        dumping raw source text.  The output format mirrors what the LLM would produce
        so the frontend renders it correctly.

        Root cause reminder: this path is only reached when CF_API_KEY is missing or
        invalid.  Fix: set CF_API_KEY in your .env file and restart.
        """
        import re as _re

        if not chunks:
            return "No documents found matching your query."

        lines = [
            "\u26a0\ufe0f *LLM unavailable \u2014 structured summary generated from source documents.*",
            "",
        ]

        for i, chunk in enumerate(chunks[:5], 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown file")
            doc_type = meta.get("doc_type", "")
            text = chunk.get("text", "").strip()

            # Extract first ~2 meaningful sentences (skip short fragments)
            sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
            excerpt = " ".join(sentences[:2])
            if len(excerpt) > 400:
                excerpt = excerpt[:397] + "\u2026"
            if not excerpt:
                excerpt = text[:300] + ("\u2026" if len(text) > 300 else "")

            type_label = f" ({doc_type})" if doc_type and doc_type != "unknown" else ""
            lines.append(f"**Source {i} \u2014 {filename}{type_label}**")
            lines.append(excerpt)
            lines.append("")

        lines.append("---")
        lines.append("*To get a proper AI-generated answer, set your `CF_API_KEY` in the `.env` file and restart.*")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  JUDGE EVALUATION NODE (NEW)                                        #
    # ------------------------------------------------------------------ #

    def judge_evaluate(self, state: AgentState) -> AgentState:
        """
        Ask the judge to evaluate the generated response.
        In voice_mode: skip evaluation entirely — always accept the first synthesis.
        """
        if state.get("voice_mode"):
            print("⚡ judge_evaluate: voice_mode — accepting first synthesis, skipping eval")
            # Create a synthetic passing evaluation so _should_retry returns "done"
            evaluation = self.judge.evaluate_response.__func__  # just need the class
            # Build a minimal passing object without calling the LLM
            from dataclasses import dataclass
            # Re-use the ResponseEvaluation class but with all-pass values
            try:
                passing_eval = ResponseEvaluation(
                    is_acceptable=True,
                    overall_score=0.9,
                    language_correct=True,
                    tone_correct=True,
                    has_hallucination=False,
                    specific_problems=[],
                    how_to_fix="",
                )
            except Exception:
                # Fallback: just don't set response_evaluation and _should_retry will pass
                return state
            return {**state, "response_evaluation": passing_eval}

        query = state["query"]
        answer = state["answer"]
        plan = state["response_plan"]
        chunks = state["retrieved_chunks"]

        if not plan:
            print("❌ No plan to evaluate against")
            return state

        print(f"⚖️  Judge evaluating → ", end="")

        # Get judge's evaluation
        evaluation = self.judge.evaluate_response(query, plan, answer, chunks)

        # Override: only retry if score is genuinely low (< 0.5).
        # The LLM judge tends to be overly strict — a score of 0.6-0.7 is fine.
        # Hallucination is the only hard rejection regardless of score.
        if evaluation.overall_score >= 0.5 and not evaluation.has_hallucination:
            from dataclasses import replace as _replace
            evaluation = type(evaluation)(
                **{**evaluation.__dict__, "is_acceptable": True}
            )

        if evaluation.is_acceptable:
            print(f"✅ ACCEPTED (score: {evaluation.overall_score:.2f})")
            self._emit("judge_evaluate", "✅", f"Answer looks good (score {evaluation.overall_score:.0%})…")
        else:
            print(f"❌ REJECTED (score: {evaluation.overall_score:.2f})")
            print(f"   Issues: {', '.join(evaluation.specific_problems)}")
            print(f"   How to fix: {evaluation.how_to_fix}")
            self._emit("judge_evaluate", "🔁", f"Improving answer: {evaluation.how_to_fix[:60]}…")

        if _OBS_AVAILABLE:
            _obs.log_judge(
                session_id=state.get("session_id", ""),
                attempt=state.get("retry_count", 0) + 1,
                passed=evaluation.is_acceptable,
                score=evaluation.overall_score,
                reason=(", ".join(evaluation.specific_problems) if evaluation.specific_problems
                        else evaluation.how_to_fix),
            )

        return {
            **state,
            "response_evaluation": evaluation,
        }

    def _should_retry(self, state: AgentState) -> str:
        """
        Decide whether to retry based on judge's evaluation.

        On retry: mutate the ResponsePlan to directly encode the judge's fix.
        On repeated failure: check if the route itself was wrong and re-route to direct.

        Returns: 'retry' | 'reroute_direct' | 'analytics' | 'done'
        """
        evaluation = state.get("response_evaluation")
        retry_count = state.get("retry_count", 0)
        route = state["route"]

        def _next(r: str) -> str:
            return "analytics" if r == "analytics" else "done"

        if not evaluation:
            return _next(route)

        if evaluation.is_acceptable:
            return _next(route)

        if retry_count >= MAX_RETRIES:
            print(f"⚠️  Max retries ({MAX_RETRIES}) reached — using best answer so far")
            return _next(route)

        # ── Detect wrong-route situations via LLM ─────────────────────────
        # Skip this extra LLM call in voice_mode — not worth the latency for a
        # spoken answer, and the reroute check is a full round-trip.
        if retry_count == 0 and self.llm.enabled and not state.get("voice_mode"):
            reroute_check = self._chat(
                [{"role": "user", "content": (
                    f"A RAG system routed this query to document search: \"{state['query']}\"\n"
                    f"The judge rejected the answer with this feedback:\n"
                    f"Problems: {', '.join(evaluation.specific_problems or [])}\n"
                    f"Fix: {evaluation.how_to_fix}\n\n"
                    f"Does this feedback suggest the query should NOT have used documents at all "
                    f"and should be answered directly from conversation context instead?\n"
                    f"Reply with YES or NO only."
                )}],
                temperature=0.0,
                max_tokens=5,
                task="classify",
            ).strip().upper()

            if reroute_check.startswith("YES"):
                print(f"🔀 Judge + LLM detected wrong route — switching to direct_answer")
                state["route"] = "direct"
                return "reroute_direct"

        # ── Mutate the plan to encode the judge's fix ─────────────────────
        plan = state.get("response_plan")
        if plan and evaluation.how_to_fix:
            updates = {}
            fix = evaluation.how_to_fix

            # Build correction directives
            correction_parts = []
            lang_correction = ""

            if not evaluation.tone_correct:
                if any(w in fix for w in ("formal", "professional")):
                    updates["target_tone"] = "professional"
                elif any(w in fix for w in ("casual", "friendly", "conversational")):
                    updates["target_tone"] = "conversational"
                elif "technical" in fix:
                    updates["target_tone"] = "technical"

            # ── Language correction ─────────────────────────────────────────
            if not evaluation.language_correct:
                correct_lang = _detect_script_language(state.get("query", ""))
                updates["target_language"] = correct_lang
                lang_correction = f"LANGUAGE ERROR: Your answer was in the wrong language. Rewrite the ENTIRE answer in {correct_lang.upper()}. Every sentence must be in {correct_lang.upper()}."
                correction_parts.append(lang_correction)
                print(f"🌐 Language correction: forcing {correct_lang.upper()} (was wrong language)")

            if any(w in fix for w in ("shorter", "concise", "brief", "too long")):
                updates["max_response_length"] = "brief"
                updates["response_style"] = "concise"
            elif any(w in fix for w in ("longer", "more detail", "expand", "comprehensive")):
                updates["max_response_length"] = "comprehensive"
                updates["response_style"] = "detailed"

            if any(w in fix for w in ("bullet", "list", "structured")):
                updates["response_style"] = "bullet_points"
            elif any(w in fix for w in ("narrative", "prose", "paragraph")):
                updates["response_style"] = "narrative"

            # Combine all fix instructions into approach
            existing_approach = plan.approach or ""
            correction_parts.append(f"CORRECTION REQUIRED: {evaluation.how_to_fix}")
            updates["approach"] = f"{existing_approach}. {' '.join(correction_parts)}"
            updates["things_to_avoid"] = list(plan.things_to_avoid) + evaluation.specific_problems

            if updates:
                state["response_plan"] = ResponsePlan(**{**plan.to_dict(), **updates})
                print(f"🔧 Plan updated for retry: {list(updates.keys())}")

        state["retry_count"] = retry_count + 1
        print(f"🔄 Retrying with improved plan (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
        return "retry"


# ────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("⚠️  This is the RAG engine module")
    print("To use it, import it and provide a vector store:")
    print()
    print("  from rag_engine import RAGEngine")
    print("  engine = RAGEngine(your_vector_store)")
    print("  result = engine.query('your question')")
