"""
Industry Analyst Agent — news_agent.py (DDG edition, improved)
────────────────────────────────────────────────────────────────
Pipeline (LangGraph):
  1. search_news    — DuckDuckGo News with broader, resilient queries
  2. judge_articles — Multi-factor LLM Judge (score 0.3+ instead of 0.50)
  3. enrich_articles — LLM summary + AI-impact on accepted articles
  4. finalize

Public API (used by main.py):
    from news_agent import get_news_briefing
    briefing = get_news_briefing(force=False)

Requirements:
    pip install duckduckgo-search langgraph
"""

import re
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional
import operator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_agent")

# ── Improved search queries — broader, more resilient ────────────────────────

SEARCH_QUERIES = [
    # 5G — one broader query gets more results than 4 narrow ones
    {"query": "5G network 2025",                          "category": "5G"},
    {"query": "5G deployment infrastructure",             "category": "5G"},
    # Fiber
    {"query": "fiber broadband internet 2025",            "category": "Fiber"},
    {"query": "broadband expansion rural",                "category": "Fiber"},
    # Regulation
    {"query": "FCC telecom regulation 2025",              "category": "Regulation"},
    {"query": "spectrum broadband policy",                "category": "Regulation"},
    # Construction
    {"query": "telecom tower construction",               "category": "Construction"},
    {"query": "broadband infrastructure build",           "category": "Construction"},
    # General
    {"query": "telecom carrier ISP news 2025",            "category": "General"},
    {"query": "telecom industry news",                    "category": "General"},
]

MAX_RESULTS_PER_QUERY = 8  # Fewer queries but more results each
CACHE_TTL_HOURS = 6
JUDGE_SCORE_THRESHOLD = 0.30  # More lenient: 0.30+ instead of 0.50


# ── State ──────────────────────────────────────────────────────────────────

class RawArticle(TypedDict):
    title: str
    url: str
    image_url: Optional[str]
    raw_text: str
    category: str
    source: str
    source_url: str
    published_at: str

class EnrichedArticle(TypedDict):
    id: str
    title: str
    source: str
    source_url: str
    article_url: str
    image_url: Optional[str]
    raw_summary: str
    ai_impact: str
    category: str
    published_at: str
    scraped_at: str

class AgentState(TypedDict):
    raw_articles: Annotated[List[RawArticle], operator.add]
    accepted_articles: Annotated[List[RawArticle], operator.add]
    enriched_articles: Annotated[List[EnrichedArticle], operator.add]
    errors: Annotated[List[str], operator.add]
    status: str


# ── Node 1: DuckDuckGo News search (improved) ──────────────────────────────

def search_news(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("📡 NEWS AGENT — Node 1: DDG Search (Improved)")
    print("="*60)

    # Support both old and new package names
    DDGS = None
    try:
        from ddgs import DDGS
    except ImportError:
        pass
    if DDGS is None:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            msg = "ddgs not installed. Run: pip install ddgs"
            print(f"❌ {msg}")
            return {"raw_articles": [], "errors": [msg], "status": "searched"}

    import random

    raw_articles: List[RawArticle] = []
    errors: List[str] = []
    seen_urls: set = set()

    with DDGS() as ddgs:
        for i, entry in enumerate(SEARCH_QUERIES):
            # Respectful delay between queries to avoid DDG rate limits
            if i > 0:
                sleep_sec = random.uniform(3.0, 5.5)
                print(f"     ⏳ Waiting {sleep_sec:.1f}s…")
                time.sleep(sleep_sec)

            print(f"  🔍 Searching: \"{entry['query']}\" [{entry['category']}]")
            results = []
            for attempt in range(3):
                try:
                    results = list(ddgs.news(
                        entry["query"],
                        max_results=MAX_RESULTS_PER_QUERY,
                        safesearch="off",
                        timelimit="m",
                    ))
                    print(f"     → {len(results)} results")
                    break
                except Exception as e:
                    err_str = str(e)
                    if "Ratelimit" in err_str and attempt < 2:
                        wait = random.uniform(8, 15) * (attempt + 1)
                        print(f"     ⏳ Rate limited — retrying in {wait:.0f}s (attempt {attempt+1}/3)…")
                        time.sleep(wait)
                    else:
                        err = f"DDG search failed for '{entry['query']}': {e}"
                        print(f"     ⚠️  {err}")
                        errors.append(err)
                        break

            for r in results:
                url = r.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                try:
                    raw_date = r.get("date", "")
                    if isinstance(raw_date, (int, float)):
                        published_at = datetime.utcfromtimestamp(raw_date).isoformat()
                    elif raw_date:
                        published_at = str(raw_date)
                    else:
                        published_at = datetime.utcnow().isoformat()
                except Exception:
                    published_at = datetime.utcnow().isoformat()

                parts = url.split("/")
                source_url = "/".join(parts[:3]) if len(parts) >= 3 else url

                title = r.get("title", "")[:200]
                image = r.get("image") or None
                print(f"     + [{entry['category']}] {title[:70]}")

                raw_articles.append({
                    "title": title,
                    "url": url,
                    "image_url": image,
                    "raw_text": r.get("body", "")[:2000],
                    "category": entry["category"],
                    "source": r.get("source", "Unknown"),
                    "source_url": source_url,
                    "published_at": published_at,
                })

    print(f"\n✅ DDG search done — {len(raw_articles)} raw articles, {len(errors)} errors")
    return {"raw_articles": raw_articles, "errors": errors, "status": "searched"}



# ── Robust JSON extractor (handles markdown fences, truncation, bad quotes) ──

def _extract_json(raw: str) -> dict:
    """
    Try multiple strategies to extract a JSON object from LLM output.
    Cloudflare Workers AI often wraps output in markdown or truncates it.
    """
    # Strip markdown fences
    text = re.sub(r"```json|```", "", raw).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first {...} block
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract key-value pairs manually (score + reason / raw_summary + ai_impact)
    result = {}
    for key in ("score", "reason", "raw_summary", "ai_impact"):
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1)
        else:
            # Try numeric value (for score)
            num_pattern = rf'"{key}"\s*:\s*([0-9.]+)'
            m2 = re.search(num_pattern, text)
            if m2:
                result[key] = float(m2.group(1))
    if result:
        return result

    raise ValueError(f"Could not extract JSON from: {text[:200]!r}")


# ── Node 2: Multi-factor LLM Judge (improved) ──────────────────────────────

_JUDGE_SYSTEM = (
    "You are a news quality filter for telecom & construction industry briefing. "
    "Use multi-factor scoring: relevance, newsworthiness, and specificity. "
    "Respond ONLY with a single JSON object. No preamble, no markdown."
)

_JUDGE_PROMPT = """Evaluate this article for a professional telecom/construction briefing.

Title: {title}
Snippet: {snippet}
Source: {source}

Score 0.0-1.0 using these factors:
- Relevance (0-0.4): relates to telecom (5G, fiber, broadband, spectrum, carriers, towers, ISPs, networks) OR construction/infrastructure
- Newsworthiness (0-0.3): actual news event, announcement, contract, policy, study, product launch (NOT job ad, homepage, generic list)
- Specificity (0-0.3): concrete details, named entities, actionable info (NOT vague or promotional)

Examples:
- "Verizon deploys 5G in 50 cities" → 0.95 (high relevance, news, specific)
- "5G technology overview" → 0.35 (relevant, not news, vague)
- "Job: Telecom Engineer" → 0.15 (not news)
- "Fiber expansion in rural areas" → 0.85 (relevant, news, specific)

Reply ONLY with this JSON:
{{"score": 0.85, "reason": "one sentence explaining the score"}}"""


def judge_articles(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("⚖️  NEWS AGENT — Node 2: LLM Judge (Multi-factor)")
    print("="*60)

    articles = state["raw_articles"]
    print(f"  Judging {len(articles)} articles…")

    if not articles:
        print("  ⚠️  No articles to judge — DDG returned nothing!")
        return {"accepted_articles": [], "errors": [], "status": "judged"}

    errors: List[str] = []

    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
        if not available:
            print("  ⚠️  No LLM available — accepting all articles (fail-open)")
            return {"accepted_articles": _dedup(articles), "errors": errors, "status": "judged"}
        print(f"  🤖 Using: {provider}")
    except ImportError:
        print("  ⚠️  llm_client not found — accepting all articles")
        return {"accepted_articles": _dedup(articles), "errors": errors, "status": "judged"}

    accepted: List[RawArticle] = []

    for art in articles:
        try:
            prompt = _JUDGE_PROMPT.format(
                title=art["title"],
                snippet=art["raw_text"][:500],
                source=art["source"],
            )
            raw = call_llm(
                prompt=prompt,
                system_prompt=_JUDGE_SYSTEM,
                max_tokens=100,
                temperature=0.0,
                task="evaluate",
            )
            verdict = _extract_json(raw)

            score = float(verdict.get("score", 0.5))
            reason = verdict.get("reason", "")

            if score >= JUDGE_SCORE_THRESHOLD:
                accepted.append(art)
                print(f"  ✅ ACCEPT [{score:.2f}] {art['title'][:65]}")
            else:
                print(f"  ❌ REJECT [{score:.2f}] {art['title'][:65]}")
                if reason:
                    print(f"            {reason}")

        except Exception as e:
            # Fail-open: keep the article on any error
            print(f"  ⚠️  Judge error for '{art['title'][:50]}': {e} → keeping")
            errors.append(f"Judge error for '{art['title'][:50]}': {e}")
            accepted.append(art)

    deduped = _dedup(accepted)
    print(f"\n✅ Judge done — {len(deduped)} accepted, {len(articles) - len(deduped)} dropped")
    return {"accepted_articles": deduped, "errors": errors, "status": "judged"}


def _dedup(articles: List[RawArticle]) -> List[RawArticle]:
    """Remove duplicate articles by normalized title."""
    seen: set = set()
    out: List[RawArticle] = []
    for art in articles:
        key = re.sub(r"[^a-z0-9]", "", art["title"].lower())[:60]
        if key not in seen:
            seen.add(key)
            out.append(art)
    return out


# ── Node 3: LLM Enrichment ─────────────────────────────────────────────────

_ENRICH_SYSTEM = (
    "You are a senior telecom industry analyst. "
    "Given a news article title and snippet, return ONLY a JSON object:\n"
    '{\n'
    '  "raw_summary": "2-sentence factual summary",\n'
    '  "ai_impact":   "2-3 sentences on what this means for telecom/construction professionals — specific and actionable"\n'
    '}\n'
    "No markdown fences, no preamble. Just the JSON."
)


def enrich_articles(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("🤖 NEWS AGENT — Node 3: LLM Enrichment")
    print("="*60)

    articles = state["accepted_articles"]
    print(f"  Enriching {len(articles)} accepted articles…")

    enriched: List[EnrichedArticle] = []
    errors: List[str] = []

    if not articles:
        print("  ⚠️  No accepted articles to enrich!")
        return {"enriched_articles": [], "errors": errors, "status": "enriched"}

    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
        if not available:
            print("  ⚠️  No LLM — returning articles without enrichment")
            for i, art in enumerate(articles):
                enriched.append(_make_unenriched(i, art))
            return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}
        print(f"  🤖 Using: {provider}")
    except ImportError:
        print("  ⚠️  llm_client not found — returning unenriched")
        for i, art in enumerate(articles):
            enriched.append(_make_unenriched(i, art))
        return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}

    for i, art in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {art['title'][:70]}")
        try:
            raw = call_llm(
                prompt=f"Title: {art['title']}\n\nSnippet:\n{art['raw_text']}",
                system_prompt=_ENRICH_SYSTEM,
                max_tokens=400,
                temperature=0.3,
                task="synthesise",
            )
            parsed = _extract_json(raw)

            uid = f"art_{i}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
            enriched.append({
                "id": uid,
                "title": art["title"],
                "source": art["source"],
                "source_url": art["source_url"],
                "article_url": art["url"],
                "image_url": art.get("image_url"),
                "raw_summary": parsed.get("raw_summary", art["raw_text"][:200]),
                "ai_impact": parsed.get("ai_impact", ""),
                "category": art["category"],
                "published_at": art["published_at"],
                "scraped_at": datetime.utcnow().isoformat(),
            })
            print(f"       ✅ enriched")

        except Exception as e:
            err = f"Enrichment failed for '{art['title']}': {e}"
            print(f"       ⚠️  {err} — using unenriched fallback")
            errors.append(err)
            enriched.append(_make_unenriched(i, art))

    print(f"\n✅ Enrichment done — {len(enriched)} articles ready")
    return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}


def _make_unenriched(i: int, art: RawArticle) -> EnrichedArticle:
    uid = f"art_{i}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
    return {
        "id": uid,
        "title": art["title"],
        "source": art["source"],
        "source_url": art["source_url"],
        "article_url": art["url"],
        "image_url": art.get("image_url"),
        "raw_summary": art["raw_text"][:200],
        "ai_impact": "",
        "category": art["category"],
        "published_at": art["published_at"],
        "scraped_at": datetime.utcnow().isoformat(),
    }


# ── Node 4: Finalize ───────────────────────────────────────────────────────

def finalize(state: AgentState) -> dict:
    print("\n" + "="*60)
    print(f"📰 NEWS AGENT — Done: {len(state['enriched_articles'])} articles")
    print("="*60 + "\n")
    return {"status": "done"}


# ── Build graph ────────────────────────────────────────────────────────────

def _build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(AgentState)
    g.add_node("search", search_news)
    g.add_node("judge", judge_articles)
    g.add_node("enrich", enrich_articles)
    g.add_node("finalize", finalize)
    g.set_entry_point("search")
    g.add_edge("search", "judge")
    g.add_edge("judge", "enrich")
    g.add_edge("enrich", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

try:
    _graph = _build_graph()
    print("📰 News agent graph compiled ✅")
except Exception as _graph_err:
    _graph = None
    print(f"❌ Failed to build news graph: {_graph_err}")


# ── In-memory cache ───────────────────────────────────────────────────────

_cache: dict = {}
_cache_time: Optional[datetime] = None


def get_news_briefing(force: bool = False) -> dict:
    """
    Public API for main.py:
        briefing = get_news_briefing(force=False)
    Cached for CACHE_TTL_HOURS. Pass force=True to bypass.
    """
    global _cache, _cache_time

    if _graph is None:
        raise RuntimeError(
            "News agent graph failed to initialise. "
            "Ensure langgraph is installed: pip install langgraph"
        )

    now = datetime.utcnow()
    cache_valid = (
        _cache
        and _cache_time is not None
        and (now - _cache_time) < timedelta(hours=CACHE_TTL_HOURS)
    )
    if cache_valid and not force:
        print("📰 Returning cached briefing")
        return _cache

    print("\n🚀 Starting news agent run…")

    initial_state: AgentState = {
        "raw_articles": [],
        "accepted_articles": [],
        "enriched_articles": [],
        "errors": [],
        "status": "starting",
    }

    result = _graph.invoke(initial_state)

    _cache = {
        "generated_at": now.isoformat(),
        "count": len(result["enriched_articles"]),
        "items": result["enriched_articles"],
        "errors": result["errors"],
    }
    _cache_time = now

    print(f"📰 Briefing cached — {_cache['count']} articles, {len(result['errors'])} errors")
    return _cache