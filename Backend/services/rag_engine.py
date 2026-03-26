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
    route: Optional[Literal["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph"]]

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

    # routing context from previous turn
    _prev_route: str
    # full log of {route, query} for this session
    _route_log: List[Dict]

    # error
    error: Optional[str]


MAX_ITER = 3
MAX_RETRIES = 1  # Max retries for quality issues (was 2 — caused up to 7 LLM calls per query)
MIN_CHUNKS = 2
RELEVANCE_THRESHOLD = 0.65


# ────────────────────────────────────────────────────────────────────────────
# OLLAMA (LOCAL) CLIENT
# ────────────────────────────────────────────────────────────────────────────

class CloudflareClient:
    """
    LLM client for the self-hosted Cloudflare Workers AI proxy.

    Request format  -> POST /
        {
          "prompt":       "<current user turn>",
          "systemPrompt": "<optional system instruction>",
          "history":      [{"role": "user"|"assistant", "content": "..."}],
          "max_tokens":   1200
        }

    Response format <- { "response": "<generated text>" }

    Accepts the same messages[] format used throughout the engine and
    auto-decomposes it into the worker's prompt/systemPrompt/history shape.
    """

    def __init__(self):
        self.api_key  = os.getenv("CF_API_KEY", "")
        self.base_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")
        self.enabled  = self._setup()

    def _setup(self) -> bool:
        if not self.api_key:
            print("❌ CloudflareClient: CF_API_KEY not set")
            return False
        try:
            resp = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=15,
            )
            if resp.status_code == 200:
                print(f"✅ CloudflareClient [Workers AI]: connected -> {self.base_url}")
                return True
            else:
                print(f"❌ CloudflareClient: API returned {resp.status_code}: {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ CloudflareClient: connection failed — {e}")
            return False

    # ------------------------------------------------------------------
    # Core chat method — same signature as the old GroqClient
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        max_retries: int = 3,
        task: str = "synthesise",
    ) -> str:
        """
        Send messages[] to the CF worker.
        The worker expects: { prompt, systemPrompt, history[], max_tokens }
          - system role  → systemPrompt
          - prior turns  → history[]  (worker caps at MAX_HISTORY=10)
          - last user msg → prompt
        """
        if not self.enabled:
            return ""

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

        payload = {
            "prompt":       prompt,
            "systemPrompt": system_prompt,
            "history":      history,
            "max_tokens":   max_tokens,
            "temperature":  temperature,
            "task":         task,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for attempt in range(max_retries):
            try:
                resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    return (data.get("response") or "").strip()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"⏳ CF rate-limit — waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"❌ CF error {resp.status_code}: {resp.text[:200]}")
                    return ""
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"❌ CF request failed: {e}")
                return ""
        return ""


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

    def query(self, user_query: str, top_k: int = 5, conversation_history: Optional[List[Dict]] = None, prev_route: str = "", route_log: Optional[List[Dict]] = None, status_callback=None, force_route: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point. conversation_history, prev_route, route_log all managed by main.py."""

        # Store callback on instance so node wrappers can access it without going through state
        # (state keys starting with _ are stripped by pydantic/langgraph)
        self._status_callback = status_callback or (lambda *_: None)

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
            "error": None,
        }
        
        print(f"\n{'='*80}")
        print(f"🔍 Query: {user_query}")
        
        try:
            final_state = self.graph.invoke(initial_state)
        finally:
            self._status_callback = None  # always clean up
            self._status_msgs = None

        # Build response
        response = {
            "answer": final_state["answer"],
            "raw_answer": final_state.get("raw_answer", final_state["answer"]),
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "route": final_state["route"],
            "analytics": final_state.get("analytics"),
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
            if (s["route"] == "iterative_rag"
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
             "no_docs": "direct_answer"},
        )

        workflow.add_edge("retrieve", "check_retrieval")
        workflow.add_edge("rewrite_query", "retrieve")

        # Transform, define, and graph end directly — no judge eval loop
        workflow.add_edge("transform_node", END)
        workflow.add_edge("define_node", END)
        workflow.add_edge("graph_node", END)

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
        LLM-powered router - intelligently decides whether query needs documents.
        
        NO hardcoded keywords - the LLM decides based on query intent.
        """
        # Skip LLM classification when force_route was set by the caller
        if state.get("route"):
            print(f"⚡ router: force_route={state['route']} — skipping classification")
            return state

        query = state["query"]
        history = state.get("conversation_history", [])
        
        print(f"📍 Route → ", end="")
        
        if not self.llm.enabled:
            # Fallback to simple routing if LLM unavailable
            return self._fallback_router(state)
        
        # Summarise the last assistant answer (if any) so the router can detect
        # follow-up intent ("make it shorter", "translate that") without being
        # confused by the full conversation — we only need the last exchange.
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

        # Build session route history so the model can see the full intent chain,
        # not just the immediately previous turn (which may itself be a direct/conversational hop)
        route_log = state.get("_route_log", [])
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

        routing_prompt = f"""You are a query router. Pick exactly one route for the CURRENT QUERY.

ROUTES:
- direct: purely conversational — greetings, small talk, memory recall ("what did you just do", "what was that", "can you repeat"), edits/modifications to a previous answer ("make it shorter", "translate that", "change the tone"). No document lookup needed.
- rag: the user wants to read, understand, or extract content from documents — summaries, explanations, specific facts, questions answered from docs.
- iterative_rag: like rag, but comparing/contrasting across multiple different documents.
- transform: the user wants document content in a completely different form — full translation, total rewrite/reformat of the entire document.
- analytics: numerical aggregations across ALL documents — counts, totals, averages, statistics.
- graph: the user wants a chart, graph, or visual plot of data extracted from the documents — bar chart, line chart, pie chart, scatter plot, histogram, trend over time, comparison chart, etc. This applies regardless of language — detect the INTENT to visualize data even when the query is in French, Arabic, Spanish, German, Italian, Portuguese, or any other language. Examples: EN: chart/graph/plot/visualize; FR: graphique/diagramme/courbe/visualiser/tracer; AR: رسم بياني/مخطط/تصور; ES: gráfico/diagrama/visualizar; DE: Diagramm/Grafik/visualisieren; IT: grafico/diagramma; PT: gráfico/visualizar.
- define: the user is asking what a specific term, concept, acronym, or niche phrase MEANS — typically triggered by "what is X?", "what does X mean?", "explain X", "define X", where X is a word or short phrase that likely appears in the documents. The answer should explain the concept in depth using the document context, not just summarise what the document says about it.

CRITICAL RULES — read these first:
1. Questions about the CONVERSATION or the AI's own actions → ALWAYS direct. Examples: "what did you just do?", "what did you say?", "summarize what you told me", "what route did you use?", "explain what you just did".
2. Questions referencing "you", "your answer", "what you said/did" → direct.
3. Short follow-up that only makes sense from the last reply context → direct.
4. Use define (not rag) when the question is clearly asking for the meaning or explanation of a single term or concept — not a document summary or fact extraction.
5. When in doubt between direct and rag: if the question is about the AI or the conversation → direct. If it's about document content → rag.
6. Questions about diagrams, schematics, wiring, rack layouts, floor plans, or figures → rag (the document processor has already described these visually, so standard RAG retrieval will find them).
{prior_context}

CURRENT QUERY: {query}

Reply with ONE word only — the route name."""

        try:
            # Router gets NO history messages — only the routing prompt.
            # Passing history here biases the model toward "direct" because it
            # sees the prior conversational turns and thinks the whole session is casual.
            route = self.llm.chat(
                [{"role": "user", "content": routing_prompt}],
                temperature=0.0,
                max_tokens=50,
            ).strip().lower()

            # Validate — rag checked BEFORE transform so ambiguous responses default safely
            valid_routes = ["direct", "rag", "iterative_rag", "transform", "analytics", "define", "graph"]
            if route not in valid_routes:
                # Extract route if LLM added extra text
                for valid in valid_routes:
                    if valid in route:
                        route = valid
                        break
                else:
                    route = "rag"  # Safe fallback
            
            print(route)
            return {**state, "route": route}
            
        except Exception as e:
            print(f"routing_error, using fallback → ", end="")
            return self._fallback_router(state)
    
    def _fallback_router(self, state: AgentState) -> AgentState:
        """Simple keyword-based routing when LLM unavailable."""
        query = state["query"].lower()
        
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
                f"You are Bimlo Copilot, a helpful AI document assistant. "
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
                f"You are Bimlo Copilot, a unified AI document assistant. "
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
        """Retrieve relevant chunks from vector store."""
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

        try:
            results = self.vs.search(query, top_k=top_k, filter_dict=filter_dict)
        except Exception:
            # Some vector stores don't support filter — fall back to unfiltered
            results = self.vs.search(query, top_k=top_k)
        
        # Merge with existing chunks (for iterative RAG)
        existing = state["retrieved_chunks"]
        all_chunks = existing + results
        
        # Deduplicate by text
        seen = set()
        unique = []
        for c in all_chunks:
            txt = c["text"][:200]
            if txt not in seen:
                seen.add(txt)
                unique.append(c)
        
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
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        history = state.get("conversation_history", [])
        
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

        # Build sources only when the judge's plan calls for citations
        if plan.should_cite_sources:
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

        return f"""You are a document assistant. Answer ONLY using the documents below.
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
        
        The judge checks:
        - Did we follow the plan?
        - Correct language?
        - Correct tone?
        - No hallucinations?
        - Good quality?
        
        If not acceptable, we'll retry.
        """
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
        # Ask the LLM whether the judge's feedback suggests the route itself
        # was wrong — no keyword matching, pure LLM judgment.
        if retry_count == 0 and self.llm.enabled:
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

        # ── Clarification needed: query was too vague ─────────────────────
        if result.get("type") == "chart_clarification":
            print(f"   🤔 graph_node: returning clarification options")
            clarif_answer = result.get(
                "question",
                "Your request covers several different types of data. Which would you like to chart?"
            )
            return {
                **state,
                "answer":     clarif_answer,
                "raw_answer": clarif_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result,
            }

        # ── Chart generation failed ───────────────────────────────────────
        if result.get("type") == "chart_error":
            print(f"   ⚠️  graph_node: {result['message']}")
            fallback_answer = (
                f"I wasn't able to generate a chart: {result['message']} "
                f"If your documents contain tables or numerical data, try being "
                f"more specific — for example: 'plot monthly revenue as a bar chart'."
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