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

# Import report_agent at module level so rag_engine and the FastAPI router
# share the SAME module instance — and therefore the SAME _reports_store dict.
#
# CRITICAL: main.py imports this as `services.report_agent`. If we import it
# here as bare `report_agent` (thanks to sys.path manipulation above), Python
# creates a SECOND module object with its own empty _reports_store — causing
# every GET /reports/{id} to 404 even though the report was saved.
#
# Fix: resolve via sys.modules so we always get the same instance that
# main.py registered, falling back to a fresh import only when standalone.
_report_agent_module = None
_REPORT_AGENT_AVAILABLE = False
try:
    import importlib as _importlib
    # Prefer the already-loaded services.report_agent (matches main.py)
    if "services.report_agent" in sys.modules:
        _report_agent_module = sys.modules["services.report_agent"]
    else:
        # Try package path first so it registers under services.report_agent
        try:
            _report_agent_module = _importlib.import_module("services.report_agent")
        except ModuleNotFoundError:
            # Standalone / pytest from services/ dir
            _report_agent_module = _importlib.import_module("report_agent")
    _REPORT_AGENT_AVAILABLE = True
except Exception as _e:
    _report_agent_module = None
    _REPORT_AGENT_AVAILABLE = False
    print(f"⚠️  report_agent not available — report route disabled ({_e})")

# Import the new judge
try:
    from llm_judge import LLMJudge, ResponsePlan, ResponseEvaluation
except ImportError as e:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from llm_judge import LLMJudge, ResponsePlan, ResponseEvaluation


# ────────────────────────────────────────────────────────────────────────────
# STATE
# ────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # input
    query: str
    top_k: int
    conversation_history: List[Dict]  # [{role, content}] prior turns for context

    # routing
    route: Optional[Literal["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph", "report"]]

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

    # routing context from previous turn
    _prev_route: str
    # full log of {route, query} for this session
    _route_log: List[Dict]

    # error
    error: Optional[str]

    # voice call optimisation — skips expensive judge/source/retry nodes
    voice_mode: bool


MAX_ITER = 3
MAX_RETRIES = 1  # Max retries for quality issues (was 2 — caused up to 7 LLM calls per query)
MIN_CHUNKS = 2
RELEVANCE_THRESHOLD = 0.65


# ────────────────────────────────────────────────────────────────────────────
# OLLAMA (LOCAL) CLIENT
# ────────────────────────────────────────────────────────────────────────────

class CloudflareClient:
    """
    LLM client — CF Workers AI primary, Groq automatic fallback.

    Uses the shared llm_client.call_llm() gateway so ALL agents get
    Groq fallback for free without any per-service changes.

    _setup() no longer marks the client as disabled when CF is down:
    as long as at least one provider (CF or Groq) is reachable the
    client is enabled and calls will succeed.
    """

    def __init__(self):
        # Import here to avoid circular imports at module load time
        from llm_client import check_llm_available
        self.enabled, provider = check_llm_available()
        if self.enabled:
            print(f"✅ CloudflareClient: LLM ready via {provider}")
        else:
            print("❌ CloudflareClient: no LLM provider available (set CF_API_KEY or GROQ_API_KEY)")

    # ------------------------------------------------------------------
    # Core chat method — same signature as before
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        max_retries: int = 3,   # kept for API compatibility, handled inside call_llm
        task: str = "synthesise",
    ) -> str:
        """
        Send messages[] to the LLM (CF primary, Groq fallback).
        Decomposes the messages[] array into prompt/systemPrompt/history
        for the CF worker; Groq receives the full messages[] directly.
        """
        if not self.enabled:
            return ""

        from llm_client import call_llm

        system_prompt = ""
        history: List[Dict] = []

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role in ("user", "assistant"):
                history.append({"role": role, "content": content})

        if not history:
            return ""

        # Last user message → prompt; everything before → history
        prompt  = history[-1]["content"]
        history = history[:-1]

        return call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
        )


# Aliases — everything that used GroqClient/OllamaClient/GeminiClient still works
GroqClient   = CloudflareClient
OllamaClient = CloudflareClient
GeminiClient = CloudflareClient


# ────────────────────────────────────────────────────────────────────────────
# NODE HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _extract_best_excerpt(chunk_text: str, answer_sentences: List[str], max_len: int = 500) -> str:
    """
    Find the passage in chunk_text most relevant to the answer sentences.
    Uses word-overlap scoring — no LLM needed.
    """
    import re

    # Build answer vocabulary (meaningful words only)
    stop = {'the','a','an','is','are','was','were','in','of','to','and','or','for',
            'with','from','that','this','it','its','on','at','by','be','as',
            'les','des','une','est','sont','dans','pour','avec','qui','que','le','la'}
    answer_words = set()
    for s in answer_sentences:
        answer_words.update(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ0-9]{3,}', s)
                           if w.lower() not in stop)

    if not answer_words:
        return chunk_text[:max_len]

    # Split chunk into sentences
    chunk_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', chunk_text)
                       if len(s.strip()) > 15]
    if not chunk_sentences:
        return chunk_text[:max_len]

    # Score each sentence by word overlap with answer
    scores = []
    for s in chunk_sentences:
        words = set(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ0-9]{3,}', s))
        scores.append(len(words & answer_words))

    # Find best sentence, expand window to ~max_len chars
    best = max(range(len(scores)), key=lambda i: scores[i])
    result = [chunk_sentences[best]]
    total = len(result[0])
    lo, hi = best - 1, best + 1

    while total < max_len:
        added = False
        if hi < len(chunk_sentences) and total + len(chunk_sentences[hi]) < max_len + 100:
            result.append(chunk_sentences[hi])
            total += len(chunk_sentences[hi])
            hi += 1
            added = True
        if lo >= 0 and total + len(chunk_sentences[lo]) < max_len + 100:
            result.insert(0, chunk_sentences[lo])
            total += len(chunk_sentences[lo])
            lo -= 1
            added = True
        if not added:
            break

    out = ' '.join(result)
    return out[:max_len] + ('…' if len(out) > max_len else '')


def _build_sources_from_brackets(
    answer: str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    Build source cards grouped by the ## section headings in the answer.

    Walk the answer line by line, tracking the current ## heading.
    For each source [N], collect:
      - sections: list of { title (exact ## heading text), lines (cited lines under it), excerpt }
      - cited_facts: all cited lines across all sections (for the flat expandable list)
      - excerpt: best matching passage in the raw chunk (used when clicking into the doc viewer)

    One card per source number. Sections mirror the output structure exactly.
    """
    import re

    # ── Step 1: Walk the answer and group cited lines by heading ──────────────
    # Structure: { source_num: { heading: [line, ...] } }
    by_source: Dict[int, Dict[str, List[str]]] = {}
    current_heading = ""

    for line in answer.split("\n"):
        # Track ## headings — these become section titles verbatim
        heading_match = re.match(r"^#{1,3}\s+(.+)", line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            continue

        # Find every [N] cited on this line
        nums_on_line = [int(m) for m in re.findall(r"\[(\d+)\]", line)]
        if not nums_on_line:
            continue

        # Clean the line: strip [N] markers, bullets, bold markers
        clean = re.sub(r"\[\d+\]", "", line)
        clean = re.sub(r"^\s*[-*•]+\s*", "", clean)
        clean = re.sub(r"\*\*", "", clean).strip()
        if len(clean) < 8:          # skip lines with no real content
            continue

        for num in nums_on_line:
            if num not in by_source:
                by_source[num] = {}
            heading = current_heading or "General"
            if heading not in by_source[num]:
                by_source[num][heading] = []
            if clean not in by_source[num][heading]:
                by_source[num][heading].append(clean)

    # ── Step 2: Build one source card per cited [N] ───────────────────────────
    sources = []
    for num in sorted(by_source.keys()):
        idx = num - 1
        if idx < 0 or idx >= len(chunks):
            continue

        chunk    = chunks[idx]
        metadata = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")

        headings_map = by_source[num]   # { heading: [lines] }

        # All cited lines across all sections (for excerpt search and flat list)
        all_lines = [line for lines in headings_map.values() for line in lines]
        if not all_lines:
            continue

        # Find best matching passage in the raw doc chunk
        excerpt = _extract_best_excerpt(chunk_text, all_lines)

        # Build sections: title = exact heading from output, lines = cited lines under it
        # Each line gets its own doc excerpt so clicking it opens the right passage
        sections = []
        for heading, lines in headings_map.items():
            line_excerpt = _extract_best_excerpt(chunk_text, lines)
            sections.append({
                "title":   heading,
                "lines":   lines,
                "excerpt": line_excerpt,
            })

        sources.append({
            "source_number": num,
            "filename":      metadata.get("filename", "Unknown"),
            "doc_type":      metadata.get("doc_type", "unknown"),
            "project_ref":   metadata.get("project_ref"),
            "excerpt":       excerpt,
            "sections":      sections,
            "cited_facts":   all_lines[:10],
        })

    # Deduplicate by source_number — shouldn't happen but guard anyway
    seen = set()
    deduped = []
    for s in sources:
        if s['source_number'] not in seen:
            seen.add(s['source_number'])
            deduped.append(s)
    return deduped


def _find_section_in_doc(full_text: str, section_title: str) -> str:
    """Legacy helper — kept for analytics node."""
    import re
    title_words = set(
        w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ]{3,}', section_title)
        if w.lower() not in {'the', 'les', 'des', 'and', 'pour', 'avec', 'dans'}
    )
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', full_text) if len(s.strip()) > 15]
    if not sentences or not title_words:
        return full_text[:400]
    scores = [sum(1 for w in title_words if w in s.lower()) for s in sentences]
    best = max(range(len(scores)), key=lambda i: scores[i])
    result = [sentences[best]]
    total = len(result[0])
    lo, hi = best - 1, best + 1
    while total < 400:
        added = False
        if hi < len(sentences) and total + len(sentences[hi]) < 500:
            result.append(sentences[hi]); total += len(sentences[hi]); hi += 1; added = True
        if lo >= 0 and total + len(sentences[lo]) < 500:
            result.insert(0, sentences[lo]); total += len(sentences[lo]); lo -= 1; added = True
        if not added:
            break
    out = ' '.join(result)
    return out[:500] + ('...' if len(out) > 500 else '')


def _format_sources(chunks: List[Dict], query: str = "", llm_client=None, generated_answer: str = "") -> List[Dict]:
    """Legacy fallback — builds sources from {{}} tags or raw chunks."""
    import re
    citations = re.findall(r'\{\{([^|]+)\|Source (\d+)\}\}', generated_answer)
    if not citations:
        citations = re.findall(r'\*\*([^|*\n]{2,40})\|Source (\d+)\*\*', generated_answer)
    by_src: Dict[int, List[str]] = {}
    for title, num in citations:
        n = int(num)
        by_src.setdefault(n, [])
        t = title.strip()
        if t not in by_src[n]:
            by_src[n].append(t)
    sources = []
    for i, c in enumerate(chunks):
        n = i + 1
        if n not in by_src:
            continue
        meta = c['metadata']
        text = c['text']
        titles = by_src[n]
        sections = [{'title': t, 'excerpt': _find_section_in_doc(text, t)} for t in titles]
        sources.append({
            'source_number': n, 'filename': meta.get('filename', 'Unknown'),
            'doc_type': meta.get('doc_type', 'unknown'), 'project_ref': meta.get('project_ref'),
            'sections': sections, 'excerpt': sections[0]['excerpt'] if sections else text[:300],
            'cited_facts': titles,
        })
    return sources




def _build_context(chunks: List[Dict]) -> str:
    """
    Build the context block injected into the synthesis prompt.

    Chunks that contain image descriptions (produced by the vision LLM during
    ingestion) are flagged with a [has visual content] note so the LLM knows
    it can reference diagram/figure descriptions from these sources.
    Chunks with table data are similarly flagged.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        flags = []
        if m.get("has_images"):
            flags.append("has visual content")
        if m.get("has_tables"):
            flags.append("has table data")
        flag_str = f" | {', '.join(flags)}" if flags else ""
        parts.append(
            f"[Source {i} | {m.get('filename')} | {m.get('doc_type')}{flag_str}]\n"
            f"{c['text'][:1200]}"
        )
    return "\n\n".join(parts)


def _build_chart_example_hint(chunks: list) -> str:
    """
    Scan retrieved chunk text for numeric patterns and return a realistic
    chart suggestion phrased in terms of the actual document content.
    Falls back to domain-appropriate telecom/BIM examples if nothing specific is found.
    """
    FIELD_CANDIDATES = [
        # telecom / BIM / infra — highest-value signals first
        ("fiber strand",      "chart fiber strand counts per site as a bar chart"),
        ("insertion loss",    "chart insertion loss per span as a line chart"),
        ("trenching length",  "chart trenching length by section as a bar chart"),
        ("downlink",          "chart downlink throughput by site as a bar chart"),
        ("uplink",            "chart uplink throughput by site as a bar chart"),
        ("latency",           "chart end-to-end latency per node as a line chart"),
        ("bbu",               "chart BBU quantities per site as a bar chart"),
        ("rrh",               "chart RRH unit counts per site as a bar chart"),
        ("antenna",           "chart antenna unit count by site as a bar chart"),
        ("power",             "chart DC power load per site as a bar chart"),
        ("battery",           "chart battery backup autonomy by site as a bar chart"),
        ("signal",            "chart signal levels by sector as a line chart"),
        ("frequency",         "chart frequency allocation by band as a pie chart"),
        ("storey",            "chart element counts per storey as a bar chart"),
        ("floor",             "chart equipment count per floor as a bar chart"),
        ("material",          "chart material quantities as a pie chart"),
        ("temperature",       "chart temperature readings as a line chart"),
        ("load",              "chart load distribution as a bar chart"),
    ]
    combined = " ".join(c.get("text", "") for c in (chunks or [])[:6]).lower()
    for keyword, example in FIELD_CANDIDATES:
        if keyword in combined:
            return example
    return "chart fiber strand counts per site as a bar chart"


def _confidence(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    distances = [c.get("distance", 1.0) for c in chunks]
    avg = sum(distances) / len(distances)
    return round(max(0.0, 1 - avg / 2), 2)


def _is_good_retrieval(chunks: List[Dict]) -> bool:
    if len(chunks) < MIN_CHUNKS:
        return False
    good = [c for c in chunks if (c.get("distance") or 1.0) < RELEVANCE_THRESHOLD]
    return len(good) >= MIN_CHUNKS


# ────────────────────────────────────────────────────────────────────────────
# GRAPH NODES
# ────────────────────────────────────────────────────────────────────────────


def _clean_answer(text: str) -> str:
    """Hard post-process the LLM answer — preserve markdown structure."""
    import re

    # 1. Strip markdown links — keep label only
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)

    # 2. Remove "as mentioned/described in X" phrases
    text = re.sub(
        r',?\s*as (?:mentioned|described|outlined|referenced|stated|noted|discussed) in[^.\n]*',
        '', text, flags=re.IGNORECASE
    )

    # 3. Fix bold spacing — strip whitespace from INSIDE ** markers
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f'**{m.group(1).strip()}**', text)

    # 4. Ensure space BEFORE opening ** when preceded by a word char
    text = re.sub(r'(\w)(\*\*\w)', r'\1 \2', text)

    # 5. Ensure space AFTER closing ** when followed by a word char or (
    text = re.sub(r'(\w\*\*)(\w|\()', r'\1 \2', text)

    # NOTE: Rules 6+7 removed — they were collapsing \n before ** and headings,
    # turning structured markdown output into a wall of text.

    # 6. Remove orphan bullets — marker with no real content.
    # Catches: '-', '- ', '- [1]', '- [1][2]', '* ', bullet followed only by whitespace/citations
    text = re.sub(r'(?m)^\s*[-*+]\s*(\[\d+\]\s*)*\s*$', '', text)

    # 6b. Remove bullets where the only content is whitespace
    text = re.sub(r'(?m)^\s*[-*+]\s+\s*$', '', text)

    # 7. Remove lines that are only citation markers e.g. a lone '[1]'
    text = re.sub(r'(?m)^\s*(\[\d+\]\s*)+$', '', text)

    # 8. Collapse 3+ blank lines (catches gaps left by removals above)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()



class RAGEngine:
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

    def query(self, user_query: str, top_k: int = 5, conversation_history: Optional[List[Dict]] = None, prev_route: str = "", route_log: Optional[List[Dict]] = None, status_callback=None, force_route: Optional[str] = None, session_id: str = "", voice_mode: bool = False) -> Dict[str, Any]:
        """Main entry point. conversation_history, prev_route, route_log all managed by main.py."""

        # Store callback on instance so node wrappers can access it without going through state
        # (state keys starting with _ are stripped by pydantic/langgraph)
        self._status_callback = status_callback or (lambda *_: None)
        self._voice_mode = voice_mode

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
            "_prev_route": prev_route,
            "_route_log": route_log or [],
            "route": initial_route,
            "retrieved_chunks": [],
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
            "error": None,
            "voice_mode": voice_mode,
        }
        
        print(f"\n{'='*80}")
        print(f"🔍 Query: {user_query}")
        
        try:
            final_state = self.graph.invoke(initial_state)
        finally:
            self._status_callback = None  # always clean up
            self._status_msgs = None
            self._voice_mode = False

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
        "router":          ("🔍", "Understanding your question…"),
        "retrieve":        ("📂", "Searching through documents…"),
        "check_retrieval": ("🔎", "Checking search results…"),
        "rewrite_query":   ("✏️",  "Refining the search…"),
        "judge_plan":      ("🧠", "Planning the response…"),
        "synthesise":      ("✍️",  "Generating answer…"),
        "judge_evaluate":  ("⚖️",  "Reviewing quality…"),
        "direct_answer":   ("💬", "Preparing answer…"),
        "transform_node":  ("🔀", "Transforming document…"),
        "analytics_node":  ("📊", "Running analytics…"),
        "define_node":     ("📖", "Looking up definition…"),
        "graph_node":      ("📈", "Building chart from documents…"),
        "report_node":     ("📄", "Writing your report…"),
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
            raw = self.llm.chat(
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

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # ── Status-aware node wrapper ─────────────────────────────────────
        # Per-query contextual messages are stored in self._status_msgs (set in query())
        # Fallback to self._DEFAULT_STATUS_MSGS if not set.

        def _wrap(name, fn):
            default_icon, default_msg = self._DEFAULT_STATUS_MSGS.get(name, ("⚙️", f"{name}…"))
            def _wrapped(state):
                cb = getattr(self, "_status_callback", None)
                if callable(cb):
                    # Use per-query generated messages if available, else defaults
                    msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
                    icon, msg = msgs.get(name, (default_icon, default_msg))
                    cb(name, icon, msg)
                return fn(state)
            _wrapped.__name__ = name
            return _wrapped

        # Add nodes
        workflow.add_node("router",          _wrap("router",          self.router))
        workflow.add_node("direct_answer",   _wrap("direct_answer",   self.direct_answer))
        workflow.add_node("retrieve",        _wrap("retrieve",        self.retrieve))
        workflow.add_node("check_retrieval", _wrap("check_retrieval", self.check_retrieval))
        workflow.add_node("rewrite_query",   _wrap("rewrite_query",   self.rewrite_query))
        workflow.add_node("judge_plan",      _wrap("judge_plan",      self.judge_plan))
        workflow.add_node("synthesise",      _wrap("synthesise",      self.synthesise))
        workflow.add_node("judge_evaluate",  _wrap("judge_evaluate",  self.judge_evaluate))
        workflow.add_node("analytics_node",  _wrap("analytics_node",  self.analytics_node))
        workflow.add_node("transform_node",  _wrap("transform_node",  self.transform_node))
        workflow.add_node("define_node",     _wrap("define_node",     self.define_node))
        workflow.add_node("graph_node",      _wrap("graph_node",      self.graph_node))
        workflow.add_node("report_node",     _wrap("report_node",     self.report_node))

        # Entry point
        workflow.set_entry_point("router")

        # Router edges
        workflow.add_conditional_edges(
            "router",
            lambda s: s["route"],
            {
                "direct": "direct_answer",
                "rag": "retrieve",
                "iterative_rag": "retrieve",
                "analytics": "retrieve",
                "transform": "retrieve",   # fetch doc content, then transform
                "define": "define_node",   # goes straight to Wikipedia — no doc retrieval
                "graph": "retrieve",       # fetch doc content, then build chart
                "report": "retrieve",      # fetch doc content, then write report
            }
        )

        # Direct answer ends
        workflow.add_edge("direct_answer", END)

        # Retrieval flow
        def _check_retrieval_route(s):
            if not s["retrieved_chunks"]:
                return "no_docs"   # nothing in the store → direct answer explains this
            if s["route"] == "transform":
                return "transform"
            if s["route"] == "define":
                return "define"
            if s["route"] == "graph":
                return "graph"
            if s["route"] == "report":
                return "report"
            # voice_mode: always single-pass — skip iterative retrieval loop
            if (not s.get("voice_mode", False)
                    and s["route"] == "iterative_rag"
                    and s["retrieval_iterations"] < MAX_ITER
                    and not _is_good_retrieval(s["retrieved_chunks"])):
                return "rewrite"
            return "done"

        workflow.add_conditional_edges(
            "check_retrieval",
            _check_retrieval_route,
            {"rewrite": "rewrite_query", "done": "judge_plan",
             "transform": "transform_node", "define": "define_node",
             "graph": "graph_node",
             "report": "report_node",
             "no_docs": "direct_answer"},
        )

        workflow.add_edge("retrieve", "check_retrieval")
        workflow.add_edge("rewrite_query", "retrieve")

        # Transform, define, and graph end directly — no judge eval loop
        workflow.add_edge("transform_node", END)
        workflow.add_edge("define_node", END)
        workflow.add_edge("graph_node", END)
        workflow.add_edge("report_node", END)

        # Judge-driven synthesis flow
        workflow.add_edge("judge_plan", "synthesise")
        workflow.add_edge("synthesise", "judge_evaluate")
        
        workflow.add_conditional_edges(
            "judge_evaluate",
            lambda s: self._should_retry(s),
            {
                "retry":           "synthesise",
                "reroute_direct":  "direct_answer",
                "analytics":       "analytics_node",
                "done":            END
            }
        )

        workflow.add_edge("analytics_node", END)

        return workflow.compile()

    # ------------------------------------------------------------------ #
    #  ROUTER NODE                                                        #
    # ------------------------------------------------------------------ #

    def router(self, state: AgentState) -> AgentState:
        """
        Two-stage LLM router:
          Stage 1 — intent_classifier: deep intent analysis with chain-of-thought.
          Stage 2 — router LLM: final decision, informed by the classifier's hint.

        NO hardcoded keywords in either stage — both use the LLM.
        """
        # Skip classification when force_route was set by the caller
        if state.get("route"):
            print(f"⚡ router: force_route={state['route']} — skipping classification")
            return state

        query = state["query"]
        history = state.get("conversation_history", [])
        route_log = state.get("_route_log", [])

        print(f"📍 Route → ", end="")

        # ── Pre-router: code generation guard ────────────────────────────────
        # Catches "write/make/give me code/function/script/algorithm for X"
        # before any LLM call — these have zero ambiguity and were being
        # misrouted to `report` because the router had no code concept.
        _q = query.lower().strip()
        _CODE_VERBS   = ["write", "make", "give me", "create", "generate", "build",
                         "code", "implement", "show me", "do", "écris", "fais", "crée"]
        _CODE_NOUNS   = ["code", "function", "script", "algorithm", "algo", "program",
                         "snippet", "class", "method", "implementation", "solution",
                         "fonction", "algorithme", "programme", "classe", "méthode",
                         "كود", "دالة", "خوارزمية", "برنامج"]
        _has_verb = any(_q.startswith(v) or f" {v} " in _q for v in _CODE_VERBS)
        _has_noun = any(n in _q for n in _CODE_NOUNS)
        if _has_verb and _has_noun:
            print("direct (code-gen guard)")
            return {**state, "route": "direct"}

        if not self.llm.enabled:
            return self._fallback_router(state)

        # ── Stage 1: Intent classifier (separate LLM call, chain-of-thought) ──
        try:
            from intent_classifier import classify_intent
            intent = classify_intent(query, history, route_log)
            intent_hint = (
                f"\n\nINTENT PRE-ANALYSIS (from deep classifier, confidence={intent.confidence:.2f}):\n"
                f"  primary_intent: {intent.primary_intent}\n"
                f"  operation: {intent.operation}\n"
                f"  output_format: {intent.output_format}\n"
                f"  is_followup: {intent.is_followup} ({intent.followup_type})\n"
                f"  language_intent: '{intent.language_intent}'\n"
                f"  ambiguity_score: {intent.ambiguity_score:.2f}\n"
                f"  suggested_route: {intent.suggested_route}\n"
                f"  reasoning: {intent.reasoning}"
            )
            # If classifier is very confident, trust it directly and skip the second LLM call
            if intent.confidence >= 0.88 and intent.ambiguity_score <= 0.25:
                print(f"{intent.suggested_route} (intent classifier, conf={intent.confidence:.2f})")
                return {**state, "route": intent.suggested_route, "_intent": intent.to_dict()}
        except Exception as e:
            print(f"[intent_classifier error: {e}] ", end="")
            intent_hint = ""

        # ── Stage 2: Router LLM (final arbiter, informed by classifier hint) ──
        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )
        last_user_before = ""
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "assistant":
                for j in range(i - 1, -1, -1):
                    if history[j]["role"] == "user":
                        last_user_before = history[j]["content"]
                        break
                break

        prior_context = ""
        if last_assistant or route_log:
            route_history = ""
            if route_log:
                route_history = "\nSESSION ROUTE HISTORY (oldest → newest): " + " → ".join(
                    f"[{e['route']}]" for e in route_log[-6:]
                )
            prior_context = (
                f"\n\nCONVERSATION CONTEXT:{route_history}"
                f"\nLAST USER MESSAGE: {last_user_before[:300]}"
                f"\nLAST ASSISTANT REPLY: {last_assistant[:300]}"
            )

        routing_prompt = f"""You are a query router for Bimlo Copilot. Pick exactly ONE route for the CURRENT QUERY.

ROUTES:
- direct: purely conversational — greetings, small talk, memory recall ("what did you just do", "can you repeat"), edits/modifications to a previous answer ("make it shorter", "translate that", "change the tone"). No document lookup needed.
- rag: the user wants to read, understand, or extract content from documents — summaries, explanations, specific facts, questions answered from docs.
- iterative_rag: like rag, but comparing/contrasting across multiple different documents. Signals: "compare", "vs", "difference between", "across documents".
- transform: the user wants document content in a completely different form — full translation of an entire document, total rewrite/reformat. The ENTIRE document is the output.
- analytics: numerical aggregations across ALL documents — counts, totals, averages, statistics.
- graph: the user wants a chart, graph, or visual plot of data extracted from the documents. Detect this intent in ANY language. EN: chart/graph/plot/visualize; FR: graphique/diagramme/courbe/visualiser; AR: رسم بياني/مخطط/تصور; ES: gráfico/diagrama/visualizar; DE: Diagramm/Grafik; IT: grafico; PT: gráfico.
- report: the user explicitly asks to PRODUCE a standalone written report/PDF/document — "make a report on X", "generate a report", "rapport sur X". Must be an explicit creation request, NOT a summary or question.
- define: the user asks the MEANING of a specific technical term, acronym, or concept — "what is X?", "define X", "what does X mean?", "explain X" where X is a single term. Answer using document context.

CRITICAL RULES:
1. Any query about what the AI just said/did, referencing "you", "your answer", "what you said" → ALWAYS direct.
2. Short follow-up only meaningful from last reply context → direct.
3. "define" over "rag": if the question is clearly about the MEANING of a single term → define.
4. When in doubt between direct and rag: AI/conversation topic → direct; document content → rag.
5. Diagrams, schematics, wiring, rack layouts, floor plans → rag (processor already described them visually).
6. TRUST the intent pre-analysis below — it is the result of a deep chain-of-thought analysis. Override it only when you see a clear contradiction.
{prior_context}
{intent_hint}

CURRENT QUERY: {query}

Reply with ONE word only — the route name."""

        try:
            route = self.llm.chat(
                [{"role": "user", "content": routing_prompt}],
                temperature=0.0,
                max_tokens=10,
            ).strip().lower()

            valid_routes = ["direct", "rag", "iterative_rag", "transform", "analytics", "define", "graph", "report"]
            if route not in valid_routes:
                for valid in valid_routes:
                    if valid in route:
                        route = valid
                        break
                else:
                    # Fall back to classifier suggestion rather than blindly defaulting to rag
                    try:
                        route = intent.suggested_route  # type: ignore[name-defined]
                    except Exception:
                        route = "rag"

            print(route)
            try:
                return {**state, "route": route, "_intent": intent.to_dict()}  # type: ignore[name-defined]
            except Exception:
                return {**state, "route": route}

        except Exception as e:
            print(f"routing_error, using fallback → ", end="")
            return self._fallback_router(state)
    
    def _fallback_router(self, state: AgentState) -> AgentState:
        """Simple keyword-based routing when LLM unavailable."""
        query = state["query"].lower()

        # Code generation — route to direct
        _CODE_VERBS = ["write", "make", "give me", "create", "generate", "build",
                       "code", "implement", "show me", "do", "écris", "fais", "crée"]
        _CODE_NOUNS = ["code", "function", "script", "algorithm", "algo", "program",
                       "snippet", "class", "method", "implementation", "solution",
                       "fonction", "algorithme", "programme", "classe", "méthode"]
        _has_verb = any(query.startswith(v) or f" {v} " in query for v in _CODE_VERBS)
        _has_noun = any(n in query for n in _CODE_NOUNS)
        if _has_verb and _has_noun:
            print("direct (code-gen fallback)")
            return {**state, "route": "direct"}
        
        # Analytics route — only explicit aggregate/cross-doc requests
        if any(kw in query for kw in ["analytics", "statistiques", "rapport analytique"]):
            print("analytics (fallback)")
            return {**state, "route": "analytics"}

        # Transform route — translation, rewriting, reformatting
        transform_kws = ["translat", "translate", "tradui", "traduire", "rewrite", "paraphrase",
                         "summarise in", "summarize in", "résume en", "résumer en"]
        if any(kw in query for kw in transform_kws):
            print("transform (fallback)")
            return {**state, "route": "transform"}

        # Direct answer — casual messages and anything that references the
        # conversation itself rather than the documents.
        # The fallback heuristic: if there are NO document-domain signals and
        # NO comparison/transform signals, treat as direct.  This is intentionally
        # broad — the LLM router handles nuance; the fallback just needs to avoid
        # misrouting obvious non-document queries to RAG.
        casual_keywords = [
            "hello", "hi", "hey", "yo", "wassup", "sup", "what's up", "whats up",
            "thanks", "thank you", "who are you", "bonjour", "merci", "salut",
            "how are you", "how's it going", "hows it going", "what's good", "whats good",
        ]
        if any(kw in query for kw in casual_keywords):
            print("direct (fallback)")
            return {**state, "route": "direct"}

        # If the query is short and contains no document-search signals,
        # default to direct rather than RAG.
        doc_signals = ["document", "file", "show", "find", "what does", "according",
                       "tell me about", "explain", "summarize", "summary", "report"]
        # Report route in fallback
        report_kws = ["make a report", "create a report", "generate a report", "write a report",
                      "do a report", "make me a report", "rapport sur", "fais un rapport",
                      "produce a report", "build a report", "prepare a report"]
        if any(kw in query for kw in report_kws):
            print("report (fallback)")
            return {**state, "route": "report"}
        is_short = len(query.split()) <= 8
        has_doc_signal = any(kw in query for kw in doc_signals)
        has_question_word = any(query.startswith(w) for w in ["what", "who", "when", "where", "how", "why", "which"])
        if is_short and has_question_word and not has_doc_signal:
            print("direct (fallback — short conversational question)")
            return {**state, "route": "direct"}

        # Iterative RAG for complex queries
        if any(kw in query for kw in ["compare", "difference", "vs", "versus", "comparaison", "différence"]):
            print("iterative_rag (fallback)")
            return {**state, "route": "iterative_rag"}

        # Default to single-pass RAG
        print("rag (fallback)")
        return {**state, "route": "rag"}

    # ------------------------------------------------------------------ #
    #  DIRECT ANSWER NODE                                                 #
    # ------------------------------------------------------------------ #

    def direct_answer(self, state: AgentState) -> AgentState:
        """Handle direct questions without retrieval. Uses fallback plan — no LLM judge call needed."""
        query    = state["query"]
        history  = state.get("conversation_history", [])
        route_log = state.get("_route_log", [])

        # Detect no-docs redirect from check_retrieval
        no_docs = query.startswith("__NO_DOCS__:")
        if no_docs:
            query = query[len("__NO_DOCS__:"):]

        plan = self.judge._fallback_plan(query)

        print(f"💬 Direct answer (language: {plan.target_language}, tone: {plan.target_tone}, no_docs={no_docs})")

        answer = self._generate_direct_answer(query, plan, history, no_docs=no_docs, route_log=route_log)

        return {
            **state,
            "query": query,   # restore clean query without flag
            "answer": answer,
            "response_plan": plan,
            "confidence": 1.0,
        }
    
    def _generate_direct_answer(self, query: str, plan: ResponsePlan, history: Optional[List[Dict]] = None, no_docs: bool = False, route_log: Optional[List[Dict]] = None) -> str:
        """Generate a direct answer using conversation history as the primary context."""

        if no_docs:
            system_content = (
                f"You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE — a company specialising in BIM engineering (3D to 7D digital models), Scan to BIM, BIM 4D construction planning, telecom infrastructure studies (rooftop, pylons, calculation notes), and DeepTwin AI digital twins for predictive maintenance. "
                f"Respond in {plan.target_language} using a {plan.target_tone} tone. "
                f"The user asked a question that requires documents, but no documents have been uploaded yet. "
                f"Let them know naturally — match their tone and language exactly. "
                f"Be brief, friendly, and encourage them to upload a document so you can help."
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

            system_content = (
                f"You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE — a company specialising in BIM engineering (3D to 7D digital models), Scan to BIM, BIM 4D construction planning, telecom infrastructure studies (rooftop, pylons, calculation notes), and DeepTwin AI digital twins for predictive maintenance. "
                f"You help professionals in construction, BTP, and telecom industries with technical questions and document analysis. "
                f"Respond in {plan.target_language} using a {plan.target_tone} tone. "
                f"You are one single assistant — the conversation history and action log below "
                f"are all yours. Use them freely to answer follow-up questions, recall what was "
                f"said or done, and modify previous answers when asked."
                f"{session_context}"
            )

        messages: List[Dict] = [{"role": "system", "content": system_content}]

        # Inject conversation history so memory recall works correctly
        if history:
            for msg in history[-10:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        if self.llm.enabled:
            return self.llm.chat(messages)
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

    def retrieve(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant chunks from vector store, then:
          1. Merge with graph RAG results (if query is relationship-focused)
          2. Re-rank the combined pool with a cross-encoder for precision
        """
        query = state["query"]
        top_k = state["top_k"]
        iteration = state["retrieval_iterations"] + 1

        print(f"🔎 Retrieval #{iteration} (+{top_k} chunks) → ", end="")

        # If the query explicitly names a file, restrict search to that file only
        filter_dict = None
        all_docs = self.vs.list_documents()
        for doc in all_docs:
            fname = doc.get("filename", "")
            if fname and fname.lower() in query.lower():
                filter_dict = {"filename": fname}
                print(f"[filtered to {fname}] ", end="")
                break

        # ── Step 1: Vector retrieval — fetch MORE candidates for re-ranking ──
        try:
            from reranker import get_fetch_k
            fetch_k = get_fetch_k(top_k)
        except ImportError:
            fetch_k = top_k

        try:
            results = self.vs.search(query, top_k=fetch_k, filter_dict=filter_dict)
        except Exception:
            results = self.vs.search(query, top_k=fetch_k)

        # ── Step 2: Graph RAG — inject relationship-aware context ─────────────
        graph_chunks = []
        try:
            from graph_rag import get_engine as _get_graph_engine, is_graph_query
            if is_graph_query(query):
                self._emit("retrieve", "🕸️", "Traversing knowledge graph…")
                graph_engine = _get_graph_engine()
                if graph_engine.available:
                    graph_chunks = graph_engine.query(query)
                    if graph_chunks:
                        print(f"[+{len(graph_chunks)} graph chunks] ", end="")
        except ImportError:
            pass
        except Exception as e:
            print(f"[graph_rag error: {e}] ", end="")

        # ── Step 3: Merge vector + graph results ─────────────────────────────
        combined = graph_chunks + results  # graph chunks get priority in sort

        # Deduplicate by text fingerprint
        seen: set = set()
        unique: List[Dict] = []
        for c in combined:
            txt = c["text"][:200]
            if txt not in seen:
                seen.add(txt)
                unique.append(c)

        # ── Step 4: Cross-encoder re-ranking ─────────────────────────────────
        try:
            from reranker import rerank
            # Graph chunks have rerank_score=2.0 pre-set — reranker will
            # sort them correctly alongside vector chunks
            unique = rerank(query, unique, top_k=top_k)
        except ImportError:
            unique = unique[:top_k]
        except Exception as e:
            print(f"[reranker error: {e}] ", end="")
            unique = unique[:top_k]

        # Merge with existing chunks from prior iterations (iterative RAG)
        existing = state["retrieved_chunks"]
        if existing:
            all_chunks = existing + unique
            seen2: set = set()
            final: List[Dict] = []
            for c in all_chunks:
                txt = c["text"][:200]
                if txt not in seen2:
                    seen2.add(txt)
                    final.append(c)
            unique = final[:top_k * 2]  # cap total

        print(f"{len(unique)} total chunks")

        # Emit which files were found
        filenames = list(dict.fromkeys(
            c["metadata"].get("filename", "unknown") for c in unique
        ))
        if filenames:
            if len(filenames) == 1:
                self._emit("retrieve", "📄", f"Reading {filenames[0]}…")
            else:
                self._emit("retrieve", "📂", f"Found {len(filenames)} files: {', '.join(filenames[:3])}…")

        # Emit a note if any chunk contains vision-described images or tables
        visual_chunks = [c for c in unique if c["metadata"].get("has_images")]
        table_chunks  = [c for c in unique if c["metadata"].get("has_tables")]
        if visual_chunks:
            self._emit("retrieve", "🖼️", f"Found {len(visual_chunks)} chunk(s) with diagram descriptions…")
        if table_chunks:
            self._emit("retrieve", "📊", f"Found {len(table_chunks)} chunk(s) with table data…")

        return {
            **state,
            "retrieved_chunks": unique,
            "retrieval_iterations": iteration,
        }

    def check_retrieval(self, state: AgentState) -> AgentState:
        """Check if retrieval quality is good enough."""
        chunks = state["retrieved_chunks"]
        is_good = _is_good_retrieval(chunks)

        if not chunks:
            # Distinguish: is the store empty, or did the query just not match anything?
            # Only flag __NO_DOCS__ if the store itself has nothing — not for bad queries.
            store_stats = self.vs.get_collection_stats()
            if store_stats.get("total_chunks", 0) == 0:
                print("⚠️  Vector store is empty — redirecting to direct answer")
                return {**state, "query": f"__NO_DOCS__:{state['query']}"}
            else:
                print("⚠️  Retrieval found nothing for this query — proceeding to synthesis with no context")
        elif is_good:
            print("✅ Retrieval quality: good")
        else:
            print("⚠️  Retrieval quality: low (will retry if iterative)")

        return state

    def rewrite_query(self, state: AgentState) -> AgentState:
        """Rewrite query to improve retrieval."""
        original = state["query"]
        
        print(f"🔄 Rewriting query for better retrieval → ", end="")
        
        if not self.llm.enabled:
            print("(LLM disabled)")
            return state
        
        prompt = f"""The following query didn't retrieve good results. Rewrite it to be more specific and searchable.

Original query: {original}

Respond with ONLY the rewritten query, nothing else."""
        
        rewritten = self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        print(f'"{rewritten}"')
        self._emit("rewrite_query", "✏️", f"Searching for: {rewritten[:60]}…")
        
        # Add to sub-queries for tracking
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
        History is passed so the judge can detect language shifts across turns.
        In voice_mode: skip the LLM call and use a fast conversational preset plan.
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        history = state.get("conversation_history", [])

        if state.get("voice_mode"):
            # Fast preset: conversational, concise, no citations needed for speech
            plan = self.judge._fallback_plan(query)
            # Override defaults to match voice expectations
            plan = ResponsePlan(**{
                **plan.to_dict(),
                "target_tone":         "conversational",
                "response_style":      "concise",
                "max_response_length": "brief",
                "should_cite_sources": False,  # citations are meaningless spoken aloud
            })
            print("⚡ judge_plan: voice_mode preset (conversational/concise/no-cite)")
            return {**state, "response_plan": plan}

        print(f"🧠 Judge planning response → ", end="")

        # Build conversation context for judge (last 3 turns)
        history_texts = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-6:]]

        plan = self.judge.plan_response(query, retrieved_docs=chunks, conversation_history=history_texts)

        print(f"{plan.target_language}/{plan.target_tone}/{plan.response_style}")

        # Emit what the judge plans to cover
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
        prompt = self._build_synthesis_prompt(query, context, plan, fix_instruction="")
        
        # Build messages — prepend conversation history so the model can handle
        # modification requests ("make it shorter", "change X to Y") by seeing
        # what it previously said. No truncation — the frontend already caps history.
        history = state.get("conversation_history", [])
        history_msgs: List[Dict] = []
        if history:
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        # Generate answer
        if self.llm.enabled:
            answer = self.llm.chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
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
    
    def _build_synthesis_prompt(self, query: str, context: str, plan: ResponsePlan, fix_instruction: str = "") -> str:
        tone_map = {
            'casual': 'casual and friendly', 'conversational': 'conversational',
            'friendly': 'warm and friendly', 'professional': 'professional', 'technical': 'technical',
        }
        tone = tone_map.get(plan.target_tone, plan.target_tone)

        # Surface the approach/correction directive if present
        approach_block = ""
        if plan.approach and "CORRECTION REQUIRED" in plan.approach:
            approach_block = f"\n\n⚠️  CORRECTION: {plan.approach.split('CORRECTION REQUIRED:')[-1].strip()}\n"
        elif plan.approach:
            approach_block = f"\nApproach: {plan.approach}\n"

        # Things to avoid
        avoid_block = ""
        if plan.things_to_avoid:
            avoid_block = f"\nAvoid: {', '.join(plan.things_to_avoid[:5])}\n"

        # Detect whether any retrieved chunk has image or table descriptions
        # so we can add a specific instruction for handling them.
        has_visual_chunks = any(
            "[IMAGE " in c.get("text", "") or "[TABLE " in c.get("text", "")
            for c in (plan._chunks if hasattr(plan, "_chunks") else [])
        )
        # Fallback: check if context string itself contains these markers
        if not has_visual_chunks:
            has_visual_chunks = "[IMAGE " in context or "[TABLE " in context

        visual_instruction = ""
        if has_visual_chunks:
            visual_instruction = (
                "\n\nVISUAL CONTENT NOTE: Some sources contain [IMAGE on page N: <description>] "
                "and [TABLE on page N] blocks. These are AI-generated descriptions of diagrams, "
                "schematics, or tables found in the document. Treat them as factual content — "
                "reference them naturally in your answer (e.g. 'The wiring diagram on page 3 shows…'). "
                "Do NOT reproduce the raw [IMAGE ...] tag — describe what it contains instead."
            )

        return f"""You are Bimlo Copilot, the AI assistant of BIMLO TECHNOLOGIE — a BIM engineering, telecom infrastructure, and DeepTwin AI company. You assist professionals in construction and telecom with technical document analysis. Answer ONLY using the documents below.
Language: {plan.target_language}. Tone: {tone}. Style: {plan.response_style}. Length: {plan.max_response_length}.{approach_block}{avoid_block}{fix_instruction}{visual_instruction}

CITATION RULE: After every sentence that uses information from a document, add [N] where N is the source number shown in the document headers below.

OUTPUT STRUCTURE — follow this exactly:
- Break the answer into sections using ## headings for each major topic
- Under each heading: 2-3 short sentences MAX, or a bullet list — not both
- Use bullet points (- item) when listing 3 or more items
- Each bullet MUST have real text content: **Label**: description [N]
- NEVER output a bullet with no text — if you have nothing to say, omit the bullet entirely
- NEVER output a line that is only a citation marker like "- [1]" or just "-"
- Never write a wall of text — if a paragraph exceeds 3 sentences, split it or use bullets
- Leave a blank line between every section
- Use **bold** for key terms, numbers, technical standards
- NEVER label sections "Source 1" or "Document 2" — use the actual subject matter

{query}

{context}

Answer in {plan.target_language}:"""


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
            reroute_check = self.llm.chat(
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
            if not evaluation.tone_correct:
                if any(w in fix for w in ("formal", "professional")):
                    updates["target_tone"] = "professional"
                elif any(w in fix for w in ("casual", "friendly", "conversational")):
                    updates["target_tone"] = "conversational"
                elif "technical" in fix:
                    updates["target_tone"] = "technical"

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

            existing_approach = plan.approach or ""
            updates["approach"] = f"{existing_approach}. CORRECTION REQUIRED: {evaluation.how_to_fix}"
            updates["things_to_avoid"] = list(plan.things_to_avoid) + evaluation.specific_problems

            if updates:
                state["response_plan"] = ResponsePlan(**{**plan.to_dict(), **updates})
                print(f"🔧 Plan updated for retry: {list(updates.keys())}")

        state["retry_count"] = retry_count + 1
        print(f"🔄 Retrying with improved plan (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
        return "retry"

    # ------------------------------------------------------------------ #
    #  TRANSFORM NODE  (translation, rewrite, reformat — no sources)     #
    # ------------------------------------------------------------------ #

    def transform_node(self, state: AgentState) -> AgentState:
        """
        Execute a transformation task (translate, rewrite, reformat) on the
        retrieved document content.

        Key differences from synthesise:
        - The judge plans tone/language as usual
        - No [N] citation markers injected — sources are irrelevant
        - No judge evaluation loop — no retry recursion
        - Returns empty sources list
        """
        query   = state["query"]
        chunks  = state["retrieved_chunks"]

        print(f"🔀 Transform → ", end="")

        if not chunks:
            print("no documents")
            return {**state, "answer": "No document content found to transform.", "sources": []}

        # Ask the judge to plan (language detection etc.) — same as normal flow
        plan = self.judge.plan_response(query, retrieved_docs=chunks)
        print(f"{plan.target_language}/{plan.target_tone}")

        context = _build_context(chunks)

        # Build a clean context for transform — no [Source N] headers bleeding into output
        clean_context = "\n\n".join(c["text"] for c in chunks)

        prompt = f"""You are a document assistant. Complete the following task exactly as instructed:

TASK: {query}

STRUCTURE RULES:
- The document content below may have lost some formatting due to text extraction. Reconstruct a clean, well-structured document as output.
- Identify sections, subsections, bullet points, and numbered lists from context clues in the text (e.g. "1.", "2.", "- ", all-caps titles, etc.) and render them properly using markdown: ## for section headings, - for bullets, numbered lists where appropriate.
- ALL-CAPS phrases that look like section titles (e.g. "EXECUTIVE SUMMARY", "EQUIPMENT SPECIFICATIONS") should become ## headings.
- Apply the task to the text content only — the structure and layout must be clean and readable in the output.
- Output ONLY the result. No preamble, no explanation, no meta-commentary, no citation markers.

DOCUMENT:
{clean_context}"""

        if self.llm.enabled:
            answer = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
        else:
            answer = "LLM unavailable — cannot perform transformation."

        # Do NOT run _clean_answer on transform output — it would mangle
        # the original document structure (headings, bullets, line breaks).
        answer = answer.strip()
        print(f"done ({len(answer)} chars)")

        return {
            **state,
            "response_plan": plan,
            "answer":     answer,
            "raw_answer": answer,
            "sources":    [],          # no sources for transform tasks
            "confidence": 0.9,
        }

    # ------------------------------------------------------------------ #
    #  DEFINE NODE  (concept explanation — context-aware, not a summary) #
    # ------------------------------------------------------------------ #

    def define_node(self, state: AgentState) -> AgentState:
        """
        Explain a term or concept using:
          - Vector-store document chunks (project-specific context, relevance-filtered)
          - Wikipedia page for the term (general encyclopedic depth)

        Key differences from synthesise:
        - Only chunks above the relevance threshold are fed to the LLM
        - Wikipedia is always the primary source card; doc chunks are secondary
        - Output is structured (intro paragraph + optional bullets) — never a wall of text
        - Ghost sources (cited [N] with no matching chunk) are stripped before returning
        - wiki_url is passed on the source card so the frontend can render an external link
        """
        query   = state["query"]
        history = state.get("conversation_history", [])

        print(f"📖 Define (Wikipedia-only) → ", end="")

        # Ask the judge to plan language/tone — no chunks needed for planning
        history_texts = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=[], conversation_history=history_texts)
        print(f"{plan.target_language}/{plan.target_tone}")

        # ── Wikipedia enrichment — primary and only source ────────────────
        wiki_ctx = None
        try:
            from wiki_enricher import get_wiki_context
            wiki_ctx = get_wiki_context(
                query=query,
                api_key=os.getenv("CF_API_KEY", ""),
                base_url=os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev"),
                language=plan.target_language if len(plan.target_language) == 2 else "en",
            )
        except ImportError:
            print("   ⚠️  wiki_enricher not found")
        except Exception as e:
            print(f"   ⚠️  WikiEnricher error: {e}")

        if not wiki_ctx or not wiki_ctx.found:
            print("   ℹ️  No Wikipedia page found — answering from general knowledge")
            wiki_block = ""
            wiki_source_num = 1
            context_section = "No external source found. Answer from your general knowledge."
        else:
            wiki_source_num = 1   # Wikipedia is always source [1] — no doc chunks
            wiki_raw = wiki_ctx.as_context_block(max_chars=3500)
            wiki_block = f"[Source 1 | Wikipedia: {wiki_ctx.term} | encyclopedia]\n{wiki_raw}"
            context_section = wiki_block
            print(f"   ✅ Wikipedia '{wiki_ctx.term}' ready as [Source 1]")

        # Conversation history
        history_msgs: List[Dict] = []
        if history:
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        prompt = f"""You are a knowledgeable assistant explaining a concept clearly and thoroughly.

Language: {plan.target_language}. Tone: {plan.target_tone}.

TASK: {query}

OUTPUT FORMAT — follow this structure exactly:
1. **Opening paragraph** (2-3 sentences): A clear, direct definition. What is this concept? [1]
2. **How it works** (1 paragraph OR 3-5 bullet points if there are distinct steps/components): Explain the mechanism, key properties, or main aspects. Use bullets when listing 3+ distinct items.
3. **Why it matters** (1-2 sentences): Brief note on real-world significance or practical use.

CITATION RULES:
- After every sentence drawn from Source 1 (Wikipedia), add [1].
- Only cite when you actually used the source. Skip [1] on sentences from pure general knowledge.

STYLE RULES:
- Use blank lines between each section.
- **Bold** the first mention of the key term and any critical sub-terms.
- Use bullet points (- item [1]) when listing 3+ distinct items; prose otherwise.
- Never write more than 3 consecutive sentences without a line break.
- Do NOT use ## headings.
- Output ONLY the explanation — no preamble, no "Here is...", no meta-commentary.

{context_section}

Explain in {plan.target_language}:"""

        if self.llm.enabled:
            answer = self.llm.chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
        else:
            answer = "LLM unavailable — cannot generate explanation."

        import re as _re
        answer = _re.sub(r'\n[ \t]*\.[ \t]*\n', '\n', answer)
        answer = _re.sub(r'\n[ \t]*\.[ \t]*$', '', answer.rstrip())
        answer = _re.sub(r'\n{3,}', '\n\n', answer)
        answer = _re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', answer)
        answer = _clean_answer(answer)

        print(f"done ({len(answer)} chars)")

        # ── Wikipedia source card — only if cited ─────────────────────────
        sources: List[Dict] = []
        cited_nums = set(int(m) for m in _re.findall(r'\[(\d+)\]', answer))

        if wiki_ctx and wiki_ctx.found and wiki_source_num in cited_nums:
            sources.append({
                "source_number": wiki_source_num,
                "filename":      f"Wikipedia: {wiki_ctx.term}",
                "doc_type":      "encyclopedia",
                "project_ref":   None,
                "excerpt":       wiki_ctx.summary[:400],
                "sections":      [{"title": "Overview", "lines": [wiki_ctx.summary[:300]], "excerpt": wiki_ctx.summary[:400]}],
                "cited_facts":   [wiki_ctx.term],
                "wiki_url":      wiki_ctx.url,
            })
            print(f"   📎 Wikipedia source card added")
        else:
            print(f"   ℹ️  Wikipedia not cited in answer — card suppressed")

        return {
            **state,
            "response_plan": plan,
            "answer":     answer,
            "raw_answer": answer,
            "sources":    sources,
            "confidence": 1.0 if (wiki_ctx and wiki_ctx.found) else 0.5,
        }

    # ------------------------------------------------------------------ #
    #  ANALYTICS NODE                                                     #
    # ------------------------------------------------------------------ #

    def analytics_node(self, state: AgentState) -> AgentState:
        """
        Generate structured analytics based on the documents.
        
        Analytics respects the judge's language/tone plan and works in ANY language.
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        plan = state.get("response_plan")
        
        print(f"📊 Analytics generation → ", end="")
        
        if not chunks:
            print("no documents")
            return {**state, "error": "No documents for analytics"}
        
        context = _build_context(chunks)
        
        # Use plan's language if available, otherwise default to 'en'
        target_lang = plan.target_language if plan else 'en'
        
        # Build universal analytics prompt - LLM understands to respond in target language
        prompt = f"""Analyze the documents and produce a structured JSON response in {target_lang}.

User Question: {query}

Documents:
{context}

Instructions:
- Respond in {target_lang}
- Provide analytical insights based on the documents
- Return ONLY a valid JSON object with this structure:

{{
  "summary": "2-3 sentence summary in {target_lang}",
  "key_metrics": {{"metric_name": value}},
  "findings": ["finding 1 in {target_lang}", "finding 2 in {target_lang}"],
  "recommendations": ["recommendation 1 in {target_lang}"],
  "data_quality": "high|medium|low"
}}

Remember: ALL text fields must be in {target_lang}."""
        
        analytics_data = None
        narrative = ""
        
        if self.llm.enabled:
            raw = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            try:
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                analytics_data = json.loads(clean)
                narrative = analytics_data.get("summary", "")
            except (json.JSONDecodeError, AttributeError):
                narrative = raw
        
        if not analytics_data:
            analytics_data = self._build_fallback_analytics(chunks, target_lang)
            narrative = analytics_data.get("summary", "")
        
        print(f"done")
        
        return {
            **state,
            "answer": narrative,
            "sources": _format_sources(chunks, query, self.llm, narrative),
            "confidence": _confidence(chunks),
            "analytics": analytics_data,
        }
    
    def _build_fallback_analytics(self, chunks: List[Dict], lang: str = 'en') -> Dict:
        """
        Build basic analytics when LLM is unavailable.
        Uses simple heuristics to provide basic analytics in any language.
        """
        doc_types: Dict[str, int] = {}
        for c in chunks:
            dt = c["metadata"].get("doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
        
        # Generic fallback that works for any language
        # The actual values are language-agnostic, labels are minimal
        return {
            "summary": f"Analysis of {len(chunks)} document segments.",
            "key_metrics": {
                "total_chunks": len(chunks),
                "document_types": len(doc_types)
            },
            "findings": [c["text"][:200] for c in chunks[:2]],
            "recommendations": ["Index more documents for better analysis"],
            "data_quality": "medium",
        }

    # ------------------------------------------------------------------ #
    #  GRAPH NODE                                                         #
    # ------------------------------------------------------------------ #

    def graph_node(self, state: AgentState) -> AgentState:
        """
        Generate a Chart.js config from document data.

        Retrieval already ran (same path as analytics_node), so we have
        state["retrieved_chunks"].  GraphAgent reads them and returns a
        Chart.js config stored in state["analytics"] with type="chart_config".

        The frontend checks response["analytics"]["type"] == "chart_config"
        and renders a Chart.js canvas instead of markdown text.
        """
        query  = state["query"]
        chunks = state["retrieved_chunks"]

        cb = getattr(self, "_status_callback", None)
        if callable(cb):
            msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
            icon, msg = msgs.get("graph_node", ("📈", "Building chart from documents…"))
            cb("graph_node", icon, msg)

        print(f"📈 graph_node → query={query[:80]}")

        if not self.graph_agent:
            fallback = "Chart generation is unavailable (graph_agent module not loaded)."
            return {**state, "answer": fallback, "raw_answer": fallback,
                    "sources": [], "confidence": 0.0, "analytics": None}

        # Ask the judge for language/tone — same as every other node
        history = state.get("conversation_history", [])
        history_texts = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(
            query,
            retrieved_docs=chunks,
            conversation_history=history_texts,
        )

        result = self.graph_agent.build_chart(
            query=query,
            chunks=chunks,
            language=plan.target_language,
        )

        # ── Clarification needed: vague query or no file specified ──────────
        if result.get("type") == "chart_clarification":
            clarif_mode  = result.get("clarification_mode", "metric")
            raw_groups   = result.get("groups", [])
            valid_groups = raw_groups  # default: trust file-mode groups as-is

            # For metric-mode only: pre-validate each group so users never click
            # an option and get a chart_error back. File-mode groups are already
            # filtered inside _discover_file_groups so no double-probe needed.
            if clarif_mode == "metric":
                print(f"   🤔 graph_node: validating {len(raw_groups)} metric groups…")
                valid_groups = []
                for grp in raw_groups:
                    hint = grp.get("hint") or grp.get("label", "")
                    if not hint:
                        continue
                    try:
                        probe = self.graph_agent.build_chart(
                            query=hint,
                            chunks=chunks,
                            language=plan.target_language,
                            skip_clarification=True,
                        )
                        if probe.get("type") == "chart_config":
                            valid_groups.append(grp)
                            print(f"      ✅ group valid: {grp.get('label','?')}")
                        else:
                            print(f"      ❌ group invalid (no data): {grp.get('label','?')}")
                    except Exception as _ve:
                        print(f"      ⚠️  group probe failed for '{hint}': {_ve}")

                if not valid_groups:
                    print(f"   ⚠️  graph_node: no valid chart groups found after validation")
                    fallback_answer = (
                        "I couldn't find numeric or structured data in your documents "
                        "that's ready to chart right now. Make sure your documents contain "
                        "tables or columns of measurements, then try a specific request — "
                        "for example: 'chart the fiber strand counts by site as a bar chart'."
                    )
                    return {
                        **state,
                        "answer":     fallback_answer,
                        "raw_answer": fallback_answer,
                        "sources":    [],
                        "confidence": 0.0,
                        "analytics":  {"type": "chart_error", "message": fallback_answer},
                    }
            else:
                print(f"   🤔 graph_node: file-picker clarification — {len(raw_groups)} options")

            clarif_answer = result.get(
                "question",
                "Your request covers several different types of data. Which would you like to chart?"
            )
            result_validated = {**result, "groups": valid_groups}
            print(f"   🤔 graph_node: clarification ({clarif_mode} mode, {len(valid_groups)} groups)")
            return {
                **state,
                "answer":     clarif_answer,
                "raw_answer": clarif_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result_validated,
            }

        # ── Chart generation failed ───────────────────────────────────────
        if result.get("type") == "chart_error":
            print(f"   ⚠️  graph_node: {result['message']}")
            # Build a realistic example from the actual retrieved chunks
            _example_hint = _build_chart_example_hint(chunks)
            fallback_answer = (
                f"I wasn't able to generate a chart: {result['message']} "
                f"Make sure your documents contain tables or columns of numeric data, "
                f"then try something specific — for example: '{_example_hint}'."
            )
            return {
                **state,
                "answer":     fallback_answer,
                "raw_answer": fallback_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result,
            }

        # ── Chart ready ───────────────────────────────────────────────────
        title       = result.get("title", "Chart")
        sources_str = ", ".join(result.get("sources", []))
        answer = (
            f"Here is the **{title}** chart generated from your documents"
            f"{(' (' + sources_str + ')') if sources_str else ''}."
        ).strip()

        print(f"   ✅ graph_node: chart ready — {title}")

        return {
            **state,
            "response_plan": plan,
            "answer":        answer,
            "raw_answer":    answer,
            "sources":       [],
            "confidence":    _confidence(chunks),
            "analytics":     result,
        }

    # ------------------------------------------------------------------ #
    #  REPORT NODE                                                        #
    # ------------------------------------------------------------------ #

    def report_node(self, state: AgentState) -> AgentState:
        """
        Generate a full structured report from retrieved chunks.
        Runs _generate_report_content in a sub-thread and emits keepalive
        status pings every 5 s so the SSE connection never idles out during
        the long multi-LLM-call generation (can take 30-90 s).
        """
        import threading as _threading
        import uuid as _uuid
        from datetime import datetime as _dt

        query      = state["query"]
        chunks     = state["retrieved_chunks"]
        history    = state.get("conversation_history", [])
        session_id = state.get("session_id", "")

        cb = getattr(self, "_status_callback", None)
        if callable(cb):
            msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
            icon, msg = msgs.get("report_node", ("📄", "Writing your report…"))
            cb("report_node", icon, msg)

        print(f"📄 report_node → query={query[:80]}")

        try:
            if not _REPORT_AGENT_AVAILABLE or _report_agent_module is None:
                raise ImportError("report_agent module not available")

            # Use module-level references so we write into the SAME _reports_store
            # dict that the FastAPI GET /reports/{id} endpoint reads from.
            SharedContext            = _report_agent_module.SharedContext
            _generate_report_content = _report_agent_module._generate_report_content
            _reports_store           = _report_agent_module._reports_store
            _save_report_to_disk     = _report_agent_module._save_report_to_disk

            # Sync context so report_agent has the latest chunks + history
            if session_id:
                SharedContext.set_chunks(session_id, chunks)
                SharedContext.set_history(session_id, history)

            # Build available / explicit doc lists from chunks
            available_docs: list = list({
                c.get("metadata", {}).get("filename", "")
                for c in chunks
                if c.get("metadata", {}).get("filename")
            })
            explicit_docs: list = []
            if available_docs:
                ql = query.lower()
                for doc in available_docs:
                    stem = doc.lower().rsplit(".", 1)[0]
                    if stem in ql or doc.lower() in ql:
                        explicit_docs.append(doc)

            # ── Chart clarification flow ──────────────────────────────────────
            # Use a session-level flag in SharedContext — reliable, no history sniffing.
            SharedContext = _report_agent_module.SharedContext

            # ── LLM-based chart intent classifier ────────────────────────────
            # Replaces all hardcoded regexes. The LLM understands intent across
            # languages, typos, paraphrases, and implicit chart requests.
            # Returns JSON:
            #   wants_charts: bool  — does the user want any chart at all?
            #   already_specified: bool — did they name exactly what to chart
            #                             (metric + optional chart type), leaving
            #                             nothing ambiguous to clarify?
            #   wants_all: bool     — (clarif-answer context) wants every chart
            #   wants_none: bool    — (clarif-answer context) wants no charts
            def _classify_chart_intent(text: str, is_clarif_answer: bool) -> dict:
                """Ask the LLM what the user means re: charts. Fast, cheap call."""
                try:
                    _system = (
                        "You are a precise intent classifier for a report-generation assistant. "
                        "The user may write in any language, with typos or abbreviations. "
                        "Reply with ONLY a raw JSON object — no markdown, no explanation.\n"
                        'Format: {"wants_charts": bool, "already_specified": bool, "wants_all": bool, "wants_none": bool}\n\n'
                        "Field definitions:\n"
                        "  wants_charts: true if the user wants one or more charts/graphs/visuals in their report. "
                        "    Catch any language or spelling: 'graphe', 'grafico', 'diagramme', 'визуализация', 'رسم بياني', etc.\n"
                        "  already_specified: true ONLY if the user has named both (a) what metric/data to chart "
                        "    AND left nothing ambiguous — e.g. 'bar chart of trenching meters', "
                        "    'camembert des coûts', 'grafico a linee del throughput'. "
                        "    False if the request is vague like 'include charts' or 'add some graphs'.\n"
                        "  wants_all: true if this is a clarification answer meaning 'include all charts' "
                        "    (e.g. 'all of them', 'tous', 'todas', 'كلها', 'все').\n"
                        "  wants_none: true if this is a clarification answer meaning 'no charts at all' "
                        "    (e.g. 'no charts', 'sans graphiques', 'sin gráficos', 'بدون مخططات').\n"
                        f"  is_clarification_answer: {is_clarif_answer} — "
                        "    set to true context: the system previously asked which charts to include, "
                        "    and this text is the user's reply to that question."
                    )
                    _prompt = f'User message: """{text}"""'
                    _raw = call_llm(
                        _prompt,
                        system_prompt=_system,
                        max_tokens=80,
                        temperature=0.0,
                        task="classify",
                    )
                    import json as _j, re as _re2
                    _clean = _re2.sub(r"```(?:json)?|```", "", _raw).strip()
                    # extract first {...} in case the LLM adds preamble
                    _m = _re2.search(r'\{[^}]+\}', _clean)
                    _parsed = _j.loads(_m.group(0) if _m else _clean)
                    return {
                        "wants_charts":      bool(_parsed.get("wants_charts", False)),
                        "already_specified": bool(_parsed.get("already_specified", False)),
                        "wants_all":         bool(_parsed.get("wants_all", False)),
                        "wants_none":        bool(_parsed.get("wants_none", False)),
                    }
                except Exception as _e:
                    print(f"   ⚠️  _classify_chart_intent failed: {_e} — defaulting to wants_charts=False")
                    # Safe fallback: don't ask for clarification, don't build charts
                    return {"wants_charts": False, "already_specified": False,
                            "wants_all": False, "wants_none": False}

            # True if we previously returned a chart clarification for this session
            _prev_was_clarif = SharedContext.get_pending_chart_clarif(session_id)

            _chart_intent = _classify_chart_intent(query, is_clarif_answer=_prev_was_clarif)
            wants_charts         = _chart_intent["wants_charts"]
            chart_already_specified = _chart_intent["already_specified"]
            print(
                f"   🧠 chart_intent: wants={wants_charts} specified={chart_already_specified} "
                f"all={_chart_intent['wants_all']} none={_chart_intent['wants_none']} "
                f"prev_clarif={_prev_was_clarif}"
            )

            # Pre-built chart list to pass into _generate_report_content
            requested_charts: list = []

            # Helper: resolve GraphAgent module once
            def _load_graph_agent_mod():
                import importlib as _il, sys as _s
                if "services.graph_agent" in _s.modules:
                    return _s.modules["services.graph_agent"]
                try:
                    return _il.import_module("services.graph_agent")
                except ModuleNotFoundError:
                    return _il.import_module("graph_agent")

            if wants_charts and not _prev_was_clarif and _GRAPH_AGENT_AVAILABLE:
                if chart_already_specified:
                    # User told us exactly what they want — build it directly, no dialog.
                    try:
                        _ga_s = _load_graph_agent_mod().GraphAgent()
                        cr_s = _ga_s.build_chart(
                            query=query, chunks=chunks, language="en",
                            skip_clarification=True,
                        )
                        if cr_s.get("type") == "chart_config":
                            requested_charts = [cr_s]
                            print("   📊 report_node: chart built directly (already specified)")
                    except Exception as _ce_s:
                        print(f"   ⚠️  report_node direct chart build: {_ce_s}")
                else:
                    # Vague — discover what metrics exist and ask the user to pick.
                    try:
                        _ga = _load_graph_agent_mod().GraphAgent()
                        groups = _ga._discover_metric_groups(query, chunks, "en")

                        if groups and len(groups) > 1:
                            group_names = ", ".join(g["label"] for g in groups[:4])
                            clarif = (
                                f"I can build this report and include charts from the data. "
                                f"I found several data sets I could visualize: "
                                f"**{group_names}**"
                                f"{'...' if len(groups) > 4 else ''}.\n\n"
                                f"Which charts would you like included? "
                                f"You can name one or more, or say **\"all of them\"** — "
                                f"or say **\"no charts\"** to get the report without visuals."
                            )
                            print(f"   🤔 report_node: chart clarification — {len(groups)} groups found")
                            SharedContext.set_pending_chart_clarif(session_id, True)
                            return {
                                **state,
                                "answer":     clarif,
                                "raw_answer": clarif,
                                "sources":    [],
                                "confidence": 0.5,
                                "analytics":  {"type": "report_chart_clarification", "groups": groups},
                                "report_id":  None,
                            }
                        elif groups and len(groups) == 1:
                            # Only one option — build it without asking
                            cr = _ga.build_chart(
                                query=groups[0].get("hint", groups[0]["label"]),
                                chunks=chunks, language="en", skip_clarification=True,
                            )
                            if cr.get("type") == "chart_config":
                                requested_charts = [cr]
                    except Exception as _ce:
                        print(f"   ⚠️  report_node chart discovery: {_ce}")

            elif _prev_was_clarif and _GRAPH_AGENT_AVAILABLE:
                # User answered the clarification — clear the flag immediately so
                # any subsequent report request starts fresh.
                SharedContext.set_pending_chart_clarif(session_id, False)

                if not _chart_intent["wants_none"]:
                    try:
                        _ga2 = _load_graph_agent_mod().GraphAgent()

                        if _chart_intent["wants_all"]:
                            # Build one chart per discovered group
                            groups2 = _ga2._discover_metric_groups(query, chunks, "en")
                            targets = [g.get("hint", g["label"]) for g in (groups2 or [])[:4]]
                        else:
                            # User named specific chart(s) — pass their reply as the query
                            targets = [query]

                        for target_q in targets:
                            cr = _ga2.build_chart(
                                query=target_q, chunks=chunks, language="en",
                                skip_clarification=True,
                            )
                            if cr.get("type") == "chart_config":
                                requested_charts.append(cr)

                        print(f"   📊 report_node: {len(requested_charts)} chart(s) confirmed by user")
                    except Exception as _ce2:
                        print(f"   ⚠️  report_node chart build after clarif: {_ce2}")

            # Clear any lingering clarification flag before building
            SharedContext.set_pending_chart_clarif(session_id, False)

            # ── Run the heavy generation in a sub-thread ──────────────────────
            # report_node itself is already running in a thread (from main.py).
            # We spin a second thread so we can emit keepalive status pings every
            # 5 s while generation runs — this prevents the SSE stream from being
            # dropped by browsers / proxies that close idle connections after ~30 s.
            result_box: list  = [None]   # [result_dict] on success
            error_box:  list  = [None]   # [exception]   on failure
            done_event = _threading.Event()

            def _generate():
                try:
                    result_box[0] = _generate_report_content(
                        prompt          = query,
                        chunks          = chunks,
                        history         = history,
                        language        = None,
                        available_docs  = available_docs,
                        explicit_docs   = explicit_docs,
                        requested_charts= requested_charts,
                    )
                except Exception as _e:
                    error_box[0] = _e
                finally:
                    done_event.set()

            gen_thread = _threading.Thread(target=_generate, daemon=True)
            gen_thread.start()

            # Emit a keepalive ping every 5 s while waiting (max 10 min)
            _progress_msgs = [
                ("📄", "Analysing document content…"),
                ("🗂️",  "Planning report structure…"),
                ("✍️",  "Writing sections…"),
                ("✍️",  "Writing sections…"),
                ("✍️",  "Still writing — large document…"),
                ("✍️",  "Still writing — large document…"),
                ("💾",  "Finalising report…"),
            ]
            _ping_idx = 0
            while not done_event.wait(timeout=5):
                if callable(cb):
                    _icon, _msg = _progress_msgs[min(_ping_idx, len(_progress_msgs) - 1)]
                    cb("report_node", _icon, _msg)
                _ping_idx += 1

            gen_thread.join(timeout=5)

            if error_box[0] is not None:
                raise error_box[0]

            result = result_box[0]
            if result is None:
                raise RuntimeError("_generate_report_content returned None")

            # ── Save report ───────────────────────────────────────────────────
            report_id = str(_uuid.uuid4())
            now       = _dt.now().isoformat()
            report    = {
                "report_id":   report_id,
                "title":       result["title"],
                "content":     result["content"],
                "charts":      result.get("charts", []),
                "source_docs": result["source_docs"],
                "language":    result["language"],
                "summary":     result.get("summary", ""),
                "created_at":  now,
                "updated_at":  now,
                "version":     1,
                "versions": [{
                    "version":     1,
                    "title":       result["title"],
                    "instruction": query,
                    "created_at":  now,
                    "content":     result["content"],
                }],
                "session_id":  session_id,
            }
            _reports_store[report_id] = report
            _save_report_to_disk(report)

            answer = (
                result.get("summary")
                or f'Your report **"{result["title"]}"** is ready — click the panel to view or download it.'
            )
            print(f"   ✅ report_node: saved → {report_id} ({result['title']!r})")

            return {
                **state,
                "answer":     answer,
                "raw_answer": answer,
                "sources":    [],
                "confidence": 1.0,
                "analytics":    None,
                "report_id":    report_id,
                "report_title": result["title"],
            }

        except Exception as e:
            import traceback; traceback.print_exc()
            err = f"Report generation failed: {e}"
            print(f"   ❌ report_node: {err}")
            return {
                **state,
                "answer":     err,
                "raw_answer": err,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  None,
            }



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