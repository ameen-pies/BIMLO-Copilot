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
    route: Optional[Literal["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph", "report", "image"]]

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


MAX_ITER = 3
MAX_RETRIES = 1  # Max retries for quality issues (was 2 — caused up to 7 LLM calls per query)
MIN_CHUNKS = 2
RELEVANCE_THRESHOLD = 0.65


def _detect_script_language(text: str) -> str:
    """
    Quick Unicode-script-based language detection.
    Returns an ISO 639-1 code: 'ar', 'zh', 'ko', 'ru', 'he', 'th', 'hi', or 'en'.
    Falls back to keyword heuristics for French vs English in the Latin range.
    """
    if not text:
        return "en"
    # Arabic
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return 'ar'
    # Cyrillic
    if any('\u0400' <= c <= '\u04FF' for c in text):
        return 'ru'
    # CJK
    if any('\u4E00' <= c <= '\u9FFF' for c in text):
        return 'zh'
    # Korean
    if any('\uAC00' <= c <= '\uD7AF' for c in text):
        return 'ko'
    # Hebrew
    if any('\u0590' <= c <= '\u05FF' for c in text):
        return 'he'
    # Thai
    if any('\u0E00' <= c <= '\u0E7F' for c in text):
        return 'th'
    # Devanagari
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'hi'
    # Latin script — disambiguate French vs English via common words
    import re as _re
    lower_words = set(_re.sub(r'[^a-zàâçéèêëîïôûùüÿ]', ' ', text.lower()).split())
    # French signal words (high precision, whole-word match only)
    french_signals = {'bonjour', 'salut', 'merci', 'svp', 's\'il', 'vous', 'je', 'suis',
                      'nous', 'c\'est', 'dans', 'avec', 'cette', 'être', 'avoir',
                      'faire', 'mais', 'donc', 'où', 'qu\'est-ce', 'comment', 'pourquoi',
                      'quelle', 'quels', 'quelles', 'français', 'fichier', 'pourriez',
                      'donner', 'peut', 'peux', 'mon', 'ton',
                      'son', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'aussi',
                      'tres', 'assez', 'beaucoup', 'encore', 'enfin', 'ensuite',
                      'depuis', 'pendant', 'parce', 'car', 'ni', 'or',
                      'moi', 'toi', 'lui', 'elle', 'eux', 'elles',
                      'ceci', 'cela', 'voici', 'il', 'on', 'ils'}
    # Unambiguous French words that don't appear in English usage
    strong_french = {'bonjour', 'salut', 'merci', 'svp', 'français', 'fichier',
                     'quelle', 'quels', 'quelles', 'pourquoi', 'comment',
                     'pourriez', 'parce', 'ensuite', 'beaucoup', 'je', 'mon', 'ton',
                     'son', 'mes', 'tes', 'ses', 'nos', 'vos', 'moi', 'toi', 'lui',
                     'elle', 'nous', 'leur', 'eux', 'elles', 'ceci', 'cela', 'voici',
                     'voilà', 'il', 'elle', 'on', 'ils', 'elles'}
    if lower_words & strong_french:
        return 'fr'
    # Weaker match — check if a threshold of French signals is present
    if len(lower_words & french_signals) >= 2:
        return 'fr'
    return 'en'


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
            preferred_provider=getattr(self, '_preferred_provider', None),
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
        chunk_text = c['text']
        # Use full text for smaller chunks; cap very large ones to avoid context overflow
        # 3000 chars ~ 750 tokens — safe for most LLMs with 5 chunks
        display_text = chunk_text if len(chunk_text) <= 3000 else chunk_text[:3000]
        parts.append(
            f"[Source {i} | {m.get('filename')} | {m.get('doc_type')}{flag_str}]\n"
            f"{display_text}"
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
    # After reranking, chunks have rerank_score (0-1, higher=better) and distance=(1-score).
    # Before reranking, chunks have raw embedding distance (0-2, lower=better).
    # We accept a chunk as "good" if either:
    #   (a) rerank_score >= 0.35  (post-rerank: model thinks it's relevant)
    #   (b) distance < RELEVANCE_THRESHOLD  (pre-rerank fallback)
    def _chunk_is_good(c: Dict) -> bool:
        rerank_score = c.get("rerank_score")
        if rerank_score is not None:
            return float(rerank_score) >= 0.35
        return (c.get("distance") or 1.0) < RELEVANCE_THRESHOLD
    good = [c for c in chunks if _chunk_is_good(c)]
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



def _resolve_doc_scope(scope_hint: str, all_docs: List[Dict], query: str = "") -> str:
    """
    Resolve a doc_scope hint (from the intent classifier) to an exact filename.

    Handles three cases without hardcoding:
      1. Exact/partial filename match  → return that filename
      2. Positional hint ("first", "second", "1", …) → return docs[idx].filename
      3. Pronoun/generic with one doc  → return the only doc's filename

    Returns "" if scope is ambiguous or can't be resolved, meaning: search all docs.
    """
    if not scope_hint or not all_docs:
        return ""

    filenames = [d.get("filename", "") for d in all_docs if d.get("filename")]
    if not filenames:
        return ""

    scope_lower = scope_hint.lower().strip()

    # ── 1. Exact or substring filename match ──────────────────────────────────
    for fname in filenames:
        if scope_lower == fname.lower() or scope_lower in fname.lower() or fname.lower() in scope_lower:
            return fname

    # ── 2. Positional ordinals ────────────────────────────────────────────────
    _ordinal_map = {
        "first": 0,   "1st": 0,   "1": 0,
        "second": 1,  "2nd": 1,   "2": 1,
        "third": 2,   "3rd": 2,   "3": 2,
        "fourth": 3,  "4th": 3,   "4": 3,
        "premier": 0, "première": 0,
        "deuxième": 1, "second": 1,
        "troisième": 2,
        "الأول": 0, "الأولى": 0,
        "الثاني": 1, "الثانية": 1,
        "الثالث": 2,
    }
    if scope_lower in _ordinal_map:
        idx = _ordinal_map[scope_lower]
        if idx < len(filenames):
            return filenames[idx]

    # ── 3. Generic pronoun with single doc in session ─────────────────────────
    _pronouns = {"it", "this", "that", "the file", "the document",
                 "ce fichier", "ce document", "هذا", "هذه", "له"}
    if scope_lower in _pronouns and len(filenames) == 1:
        return filenames[0]

    return ""


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

    def query(self, user_query: str, top_k: int = 5, conversation_history: Optional[List[Dict]] = None, prev_route: str = "", route_log: Optional[List[Dict]] = None, status_callback=None, force_route: Optional[str] = None, session_id: str = "", user_id: Optional[str] = None, voice_mode: bool = False, preferred_provider: Optional[str] = None, attached_doc_filenames: Optional[List[str]] = None, cancel_event=None) -> Dict[str, Any]:
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
            "attached_doc_filenames": attached_doc_filenames or [],
        }
        
        print(f"\n{'='*80}")
        print(f"🔍 Query: {user_query}")

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
        "report_node":     ("📄", "Writing your report…"),
        "image_node":      ("🖼️", "Reading image description…"),
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

    def _wrap(self, name: str, fn):
        """Status-aware node wrapper. Fires the status callback before running fn."""
        default_icon, default_msg = self._DEFAULT_STATUS_MSGS.get(name, ("⚙️", f"{name}…"))
        def _wrapped(state):
            cb = getattr(self, "_status_callback", None)
            if callable(cb):
                msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
                icon, msg = msgs.get(name, (default_icon, default_msg))
                cb(name, icon, msg)
            return fn(state)
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

        # no-docs variant — same direct_answer node, flag injected via query prefix
        def _direct_no_docs(state):
            return self.direct_answer({**state, "query": f"__NO_DOCS__:{state['query']}"})
        workflow.add_node("direct_answer_no_docs", _direct_no_docs)

        # ── Entry ─────────────────────────────────────────────────────────────
        workflow.set_entry_point("classify_intent")

        # ── classify_intent → first node (pure dispatch, zero logic) ─────────
        _RETRIEVAL_ROUTES = {"rag", "iterative_rag", "analytics", "transform", "graph", "report"}

        def _router_dispatch(s):
            route = s.get("route", "rag")
            if route == "direct":   return "direct_answer"
            if route == "define":   return "define_node"
            if route == "image":    return "image_node"
            if route == "cad":      return "cad_node"
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
                          "graph_node", "report_node", "image_node", "cad_node"):
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

        print(f"{route} (conf={intent.confidence:.2f}, scope={intent.doc_scope!r})")

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
            "route":          route,
            "doc_scope_hint": intent.doc_scope,
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
        query = state["query"]
        top_k = state["top_k"]
        iteration = state["retrieval_iterations"] + 1

        print(f"🔎 [retrieve_vector] #{iteration} (+{top_k} candidates) → ", end="")

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
        
        rewritten = self.llm.chat(
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

        plan = self.judge.plan_response(query, retrieved_docs=chunks, conversation_history=history_texts)

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
            answer = self.llm.chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
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
                "\n\nVISUAL CONTENT: Some sources contain [IMAGE on page N: <description>], [TABLE on page N], "
                "or [UPLOADED IMAGE: <filename>] blocks. "
                "These are AI-generated descriptions of diagrams, tables, or user-uploaded images. "
                "Treat them as factual content — "
                "reference them naturally (e.g. 'The wiring diagram on page 3 shows…' or 'The uploaded image shows…'). "
                "Never reproduce the raw tag."
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

        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        prompt = f"""You are a document assistant. Complete the following task exactly as instructed.

🚨 LANGUAGE: You MUST write your output in {target_lang.upper()}. The document below may be in a different language — IGNORE its language. Your output must be in {target_lang.upper()}.

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

        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        prompt = f"""You are a knowledgeable assistant explaining a concept clearly and thoroughly.

🚨 LANGUAGE REQUIREMENT: You MUST write your ENTIRE answer in {target_lang.upper()}. The source content below may be in a different language — IGNORE its language. Your answer must be in {target_lang.upper()}.

Language: {target_lang.upper()} | Tone: {plan.target_tone}

TASK: {query}

OUTPUT FORMAT — write three sections separated by blank lines, in this exact order:

SECTION A — Definition (2-3 sentences): A clear, direct definition of the concept. [1]

SECTION B — How it works: Explain the mechanism, key properties, or main aspects. Write as a prose paragraph OR as bullet points (- item) when listing 3+ distinct items. Do NOT number items.

SECTION C — Why it matters (1-2 sentences): Brief note on real-world significance or practical use.

Do NOT label the sections with "SECTION A", "SECTION B", etc. — just write the content directly.
Do NOT use any numbered lists (1. 2. 3.) anywhere in your response.

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

Remember: Answer in {target_lang.upper()}. Your ENTIRE answer must be in {target_lang.upper()}."""

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
    #  IMAGE NODE — dedicated route for image content queries            #
    # ------------------------------------------------------------------ #

    def image_node(self, state: AgentState) -> AgentState:
        """
        Handle queries about uploaded images.

        Strategy:
          1. Fetch image chunks from the vector store (doc_type=image).
          2. If chunks contain a real vision description, synthesise the answer.
          3. If chunks only have a placeholder description (vision failed at ingestion),
             OR if no chunks are found yet (ingestion still in flight), attempt a LIVE
             vision call using the raw image bytes from DOCUMENT_FILE_CACHE, update the
             stored chunk text in-place, and then synthesise from the fresh description.
        """
        query      = state["query"]
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        history    = state.get("conversation_history", [])

        self._emit("image_node", "🖼️", "Reading image description…")
        print(f"🖼️  image_node → query={query[:80]!r}")

        # ── Fetch image chunks from the user/session's collection ─────────
        # Respect doc_scope_hint — supports:
        #   ""               → all images in session (doc_type=image filter)
        #   "file.png"       → single scoped image
        #   "a.png|b.png"    → multiple images (pipe-separated, e.g. when user
        #                       asks about both attached images at once)
        _scope_raw = state.get("doc_scope_hint", "")
        _scoped_files = [s.strip() for s in _scope_raw.split("|") if s.strip()] if _scope_raw else []

        all_chunks: list = []

        if _scoped_files:
            # Retrieve chunks for every scoped filename and merge
            print(f"   [image_node: scoped to {_scoped_files}]")
            for _fname in _scoped_files:
                try:
                    _chunks = self.vs.search(
                        query,
                        top_k=10,
                        user_id=user_id,
                        session_id=session_id,
                        filter_dict={"filename": _fname},
                    )
                    all_chunks.extend(_chunks)
                except Exception as _e:
                    print(f"⚠️  image_node: vector search failed for '{_fname}': {_e}")

            # Fallback: unfiltered search, then filter by filename set
            if not all_chunks:
                try:
                    unfiltered = self.vs.search(query, top_k=20, user_id=user_id, session_id=session_id)
                    all_chunks = [c for c in unfiltered if
                                  c.get("metadata", {}).get("filename") in _scoped_files]
                except Exception:
                    pass
        else:
            # No scope — retrieve all image chunks in this session
            try:
                all_chunks = self.vs.search(
                    query,
                    top_k=10,
                    user_id=user_id,
                    session_id=session_id,
                    filter_dict={"doc_type": "image"},
                )
            except Exception as _e:
                print(f"⚠️  image_node: vector search failed: {_e}")
                all_chunks = []

            # Fallback: unfiltered search filtered by doc_type / has_images
            if not all_chunks:
                try:
                    unfiltered = self.vs.search(query, top_k=10, user_id=user_id, session_id=session_id)
                    all_chunks = [c for c in unfiltered if
                                  c.get("metadata", {}).get("doc_type") == "image" or
                                  c.get("metadata", {}).get("has_images")]
                except Exception:
                    pass

        # Deduplicate by chunk id / text hash so merging doesn't double-count
        _seen_texts: set = set()
        _deduped: list = []
        for _c in all_chunks:
            _key = _c.get("id") or _c.get("text", "")[:120]
            if _key not in _seen_texts:
                _seen_texts.add(_key)
                _deduped.append(_c)
        all_chunks = _deduped

        # Group image chunks by filename so a single uploaded image is not
        # treated as multiple images when its description was split into chunks.
        _grouped: dict = {}
        for _c in all_chunks:
            _fname = _c.get("metadata", {}).get("filename", "image")
            _grouped.setdefault(_fname, []).append(_c)

        grouped_chunks: list = []
        for _fname, _items in _grouped.items():
            if len(_items) == 1:
                grouped_chunks.append(_items[0])
                continue

            _items.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", c.get("chunk_id", 0)))
            grouped_text = "\n\n".join(c["text"] for c in _items if c.get("text"))
            merged_meta = _items[0].get("metadata", {}).copy()
            merged_meta["chunk_count"] = len(_items)
            grouped_chunks.append({
                "text": grouped_text,
                "metadata": merged_meta,
                "id": _items[0].get("id"),
                "distance": min((c.get("distance", 1.0) for c in _items), default=1.0),
            })
        all_chunks = grouped_chunks

        # ── Placeholder / missing description detection ───────────────────
        # These strings are what document_processor writes when vision fails.
        _PLACEHOLDER_SIGNALS = (
            "vision description unavailable",
            "could not be read",
            "ask the user to describe",
        )

        def _is_placeholder(chunk: dict) -> bool:
            txt = chunk.get("text", "").lower()
            return any(s in txt for s in _PLACEHOLDER_SIGNALS)

        # Try a live vision call if: no chunks at all, OR all chunks are placeholders
        needs_live_vision = (not all_chunks) or all(_is_placeholder(c) for c in all_chunks)

        if needs_live_vision:
            self._emit("image_node", "🖼️", "Describing image with vision AI…")
            print("🔍 image_node: attempting live vision call (placeholder or missing chunks)")

            # Pull raw bytes from DOCUMENT_FILE_CACHE
            # When multiple images are scoped, attempt a live call for each one.
            _live_chunks: list = []
            try:
                from core.globals import DOCUMENT_FILE_CACHE as _cache

                # Find image entries scoped to this session
                all_image_entries = [
                    (doc_id, entry)
                    for doc_id, entry in _cache.items()
                    if entry.get("is_image") and entry.get("session_id") == session_id
                ]

                if all_image_entries:
                    # Filter to scoped filenames when scope is set
                    if _scoped_files:
                        targeted_entries = [
                            (did, e) for did, e in all_image_entries
                            if e.get("filename") in _scoped_files
                        ]
                        # Preserve order of _scoped_files
                        _fname_to_entry = {e.get("filename"): (did, e) for did, e in targeted_entries}
                        targeted_entries = [
                            _fname_to_entry[fn]
                            for fn in _scoped_files
                            if fn in _fname_to_entry
                        ]
                        if not targeted_entries:
                            targeted_entries = all_image_entries  # fallback: all
                    else:
                        targeted_entries = all_image_entries  # no scope → try all

                    from document_processor import DocumentProcessor as _DP
                    _dp = _DP()

                    for _doc_id, _entry in targeted_entries:
                        _live_fname = _entry["filename"]
                        _raw_bytes  = _entry["bytes"]
                        try:
                            _fresh_desc = _dp.describe_image_bytes(_raw_bytes, _live_fname)
                            if _fresh_desc:
                                print(f"   ✅ Live vision description for '{_live_fname}': {len(_fresh_desc)} chars")
                                _live_chunks.append({
                                    "text": (
                                        f"[UPLOADED IMAGE: {_live_fname}]\n"
                                        f"Visual description:\n{_fresh_desc}"
                                    ),
                                    "metadata": {
                                        "filename":  _live_fname,
                                        "doc_type":  "image",
                                        "has_images": True,
                                        "source":    "live_vision",
                                    },
                                    "distance": 0.05,
                                })
                        except Exception as _ve:
                            print(f"   ⚠️  Live vision call failed for '{_live_fname}': {_ve}")
                else:
                    print("   ⚠️  image_node: no image entries in DOCUMENT_FILE_CACHE for this session")

            except Exception as _ce:
                print(f"   ⚠️  image_node: could not access DOCUMENT_FILE_CACHE: {_ce}")

            if _live_chunks:
                # Use live descriptions; merge with any non-placeholder ingested chunks
                _ingested_ok = [c for c in all_chunks if not _is_placeholder(c)]
                all_chunks = _live_chunks + _ingested_ok
                print(f"   🖼️  image_node: {len(_live_chunks)} live description(s) + {len(_ingested_ok)} indexed chunk(s)")
            elif not all_chunks:
                # Both ingestion and live vision failed — tell the user clearly
                answer = (
                    "I received your image but was unable to generate a visual description. "
                    "This can happen when the vision AI service is temporarily unavailable. "
                    "Please try again in a moment, or describe what you see in the image and I'll help from there."
                )
                print("⚠️  image_node: live vision failed and no chunks — returning helpful error")
                return {**state, "answer": answer, "raw_answer": answer, "sources": [], "confidence": 0.0}
            # else: fall through with the placeholder chunks — synthesis will still run
            # and the LLM prompt below will handle it gracefully

        # ── Build context from image description chunks ───────────────────
        context_parts = []
        for i, chunk in enumerate(all_chunks, 1):
            meta = chunk.get("metadata", {})
            fname = meta.get("filename", "image")
            context_parts.append(f"[Image {i}: {fname}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        # ── Ask the judge for language/tone ───────────────────────────────
        history_texts = [f"{m['role'].upper()}: {m['content'][:200]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=all_chunks, conversation_history=history_texts)

        # ── Synthesise answer from the stored visual description ──────────
        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        prompt = f"""You are analysing images based on their stored visual descriptions.

🚨 LANGUAGE REQUIREMENT: You MUST write your ENTIRE answer in {target_lang.upper()}. The image descriptions below may be in a different language — IGNORE their language. Your answer must be in {target_lang.upper()}.

Language: {target_lang.upper()} | Tone: {plan.target_tone}

The following are AI-generated descriptions of one or more uploaded images:

{context}

User question: {query}

Answer the user's question based ONLY on the image descriptions above.
Be specific and reference what is visible in the image.
If multiple images are present, address each one as relevant.
Do NOT say you cannot see images — you have the visual descriptions.
Remember: Answer in {target_lang.upper()}. Your entire answer must be in {target_lang.upper()}."""

        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        if self.llm.enabled:
            answer = self.llm.chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
        else:
            answer = context  # fallback: return raw description

        answer = _clean_answer(answer)
        print(f"   ✅ image_node: answered from {len(all_chunks)} chunk(s) ({len(answer)} chars)")

        return {
            **state,
            "response_plan":  plan,
            "answer":         answer,
            "raw_answer":     answer,
            "retrieved_chunks": all_chunks,
            "sources":        [],
            "confidence":     0.9,
        }

    # ------------------------------------------------------------------ #
    #  CAD / IFC NODE                                                     #
    # ------------------------------------------------------------------ #

    def cad_node(self, state: AgentState) -> AgentState:
        """
        Handle queries about uploaded CAD/IFC files (DXF, DWG, STEP, IFC).

        Delegates to cad_ifc_agent._build_context_block() + _call_llm_with_judge(),
        which already have their own judge-retry loop and status callbacks.
        Falls back to standard RAG synthesis if cad_ifc_agent is unavailable.
        """
        query      = state["query"]
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        history    = state.get("conversation_history", [])
        doc_scope  = state.get("doc_scope_hint", "")

        self._emit("cad_node", "🏗️", "Parsing CAD/IFC file structure…")
        print(f"🏗️  cad_node → query={query[:80]!r}")

        try:
            from cad_ifc_agent import CadSharedContext, _build_context_block, _call_llm_with_judge, _SYSTEM_PROMPT
            from datetime import datetime as _dt

            # Resolve which file to use: doc_scope_hint > most recent in session
            summary = None
            if doc_scope:
                # Try exact file_id match first, then filename match
                for fid, meta in [(f, CadSharedContext.get_file(f))
                                  for f in [doc_scope]
                                  if CadSharedContext.get_file(f)]:
                    summary = meta
                    break
            if not summary:
                # Fall back to most recently cached file in this session
                all_files = CadSharedContext.list_files()
                # Filter to files belonging to this session if session_id is stored
                session_files = [f for f in all_files if f.get("session_id") == session_id] or all_files
                if session_files:
                    summary = session_files[-1]

            if not summary:
                # No CAD file found — fall back to RAG
                print("⚠️  cad_node: no CAD summary found — falling back to RAG synthesis")
                return self.retrieve_vector({**state, "route": "rag"})

            # Build context block from the parsed CAD/IFC summary
            context = _build_context_block(summary)
            system_prompt = _SYSTEM_PROMPT.format(
                today=_dt.utcnow().strftime("%B %d, %Y")
            ) + f"\n\n{context}"

            # Retrieve conversation history for this session
            cad_history = CadSharedContext.get_history(session_id) or []
            # Also incorporate the RAG conversation history so memory works
            for turn in history[-6:]:
                if not any(t.get("content") == turn.get("content") for t in cad_history):
                    cad_history.append(turn)

            fname    = summary.get("filename", "file")
            pipeline = summary.get("pipeline", "?").upper()
            n_elem   = summary.get("total_elements") or summary.get("total_entities") or 0
            print(f"   📂 {fname} — {n_elem} elements via {pipeline} pipeline")
            self._emit("cad_node", "🔍", f"Loaded `{fname}` — {n_elem} elements")

            answer, judge_score, judge_comment = _call_llm_with_judge(
                system_prompt=system_prompt,
                history=cad_history,
                user_msg=query,
                context=context,
                question=query,
            )

            verdict = "✅" if judge_score >= 0.7 else "⚠️"
            print(f"   {verdict} cad_node judge score={judge_score:.2f} — {judge_comment[:60]}")

            # Store the new turn in CadSharedContext so follow-ups have memory
            CadSharedContext.append_turn(session_id, "user",      query)
            CadSharedContext.append_turn(session_id, "assistant", answer)

            return {
                **state,
                "answer":     answer,
                "raw_answer": answer,
                "sources":    [{"filename": fname, "pipeline": pipeline}],
                "confidence": float(judge_score),
                "analytics":  None,
            }

        except ImportError:
            print("⚠️  cad_node: cad_ifc_agent not installed — falling back to RAG")
            return self.retrieve_vector({**state, "route": "rag"})
        except Exception as e:
            import traceback as _tb; _tb.print_exc()
            err = f"CAD analysis failed: {e}"
            print(f"❌ cad_node: {err}")
            return {
                **state,
                "answer":     err,
                "raw_answer": err,
                "sources":    [],
                "confidence": 0.0,
                "error":      err,
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
        
        # Use plan's language if available, otherwise detect from query
        user_lang = _detect_script_language(query)
        target_lang = (plan.target_language if plan else None) or user_lang

        prompt = f"""Analyze the documents and produce a structured JSON response.

🚨 LANGUAGE: ALL text fields MUST be in {target_lang.upper()}. The documents below may be in a different language — IGNORE their language. ALL summary, findings, and recommendations MUST be written in {target_lang.upper()}.

User Question: {query}

Documents:
{context}

Instructions:
- Respond in {target_lang.upper()}
- Provide analytical insights based on the documents
- Return ONLY a valid JSON object with this structure:

{{
  "summary": "2-3 sentence summary in {target_lang}",
  "key_metrics": {{"metric_name": value}},
  "findings": ["finding 1 in {target_lang}", "finding 2 in {target_lang}"],
  "recommendations": ["recommendation 1 in {target_lang}"],
  "data_quality": "high|medium|low"
}}

Remember: ALL text fields must be in {target_lang.upper()}."""
        
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
                _cancel_ev = getattr(self, "_cancel_event", None)
                for grp in raw_groups:
                    # Stop probing if the user already pressed Stop
                    if _cancel_ev and _cancel_ev.is_set():
                        print("   🛑 graph_node: group probe cancelled by stop event")
                        break
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
                        # Probe failed due to LLM error (e.g. Groq 429) — keep the group
                        # rather than silently dropping it; a transient error doesn't mean
                        # the data doesn't exist.
                        valid_groups.append(grp)
                        print(f"      ⚠️  group probe failed for '{hint}' (kept): {_ve}")

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

        from llm_client import call_llm

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
                        preferred_provider=state.get("preferred_provider"),
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
            _cancel_ev = getattr(self, "_cancel_event", None)
            while not done_event.wait(timeout=5):
                if _cancel_ev and _cancel_ev.is_set():
                    print(f"   🛑 report_node: cancelled by stop event — aborting wait")
                    error_box[0] = RuntimeError("Cancelled by user")
                    done_event.set()
                    break
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