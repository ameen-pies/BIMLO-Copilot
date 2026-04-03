"""
Industry Analyst Agent — news_agent.py
────────────────────────────────────────────────────────────────
Pipeline (LangGraph):
  1. search_news    — DuckDuckGo News
  2. judge_articles — Concurrent LLM Judge (all articles in parallel)
  3. finalize       — Serve immediately, NO blocking enrichment

Enrichment is lazy: call enrich_one(article_id) on-demand when a user
opens a card. This keeps the initial load fast and cheap.

Public API (used by main.py):
    from news_agent import get_news_briefing, enrich_one
    briefing = get_news_briefing(force=False)
    enriched = enrich_one(article_id)           # called by /api/news/enrich/<id>

Requirements:
    pip install duckduckgo-search langgraph
"""

import re
import json
import time
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional
import operator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_agent")

# ── Search queries ────────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    {"query": "5G network 2025",                 "category": "5G"},
    {"query": "5G deployment infrastructure",    "category": "5G"},
    {"query": "fiber broadband internet 2025",   "category": "Fiber"},
    {"query": "broadband expansion rural",       "category": "Fiber"},
    {"query": "FCC telecom regulation 2025",     "category": "Regulation"},
    {"query": "spectrum broadband policy",       "category": "Regulation"},
    {"query": "telecom tower construction",      "category": "Construction"},
    {"query": "broadband infrastructure build",  "category": "Construction"},
    {"query": "telecom carrier ISP news 2025",   "category": "General"},
    {"query": "telecom industry news",           "category": "General"},
]

MAX_RESULTS_PER_QUERY = 8
CACHE_TTL_HOURS       = 6
JUDGE_SCORE_THRESHOLD = 0.30

# How many judge calls to fire in parallel.
# Keep this modest so you don't hammer your LLM provider.
JUDGE_WORKERS = 6


# ── State ─────────────────────────────────────────────────────────────────────

class RawArticle(TypedDict):
    title:        str
    url:          str
    image_url:    Optional[str]
    raw_text:     str
    category:     str
    source:       str
    source_url:   str
    published_at: str

class EnrichedArticle(TypedDict):
    id:           str
    title:        str
    source:       str
    source_url:   str
    article_url:  str
    image_url:    Optional[str]
    raw_summary:  str
    ai_impact:    str
    category:     str
    published_at: str
    scraped_at:   str
    enriched:     bool   # False until enrich_one() has been called

class AgentState(TypedDict):
    raw_articles:      Annotated[List[RawArticle],    operator.add]
    accepted_articles: Annotated[List[RawArticle],    operator.add]
    enriched_articles: Annotated[List[EnrichedArticle], operator.add]
    errors:            Annotated[List[str],            operator.add]
    status:            str


# ── Node 1: DuckDuckGo search ─────────────────────────────────────────────────

def search_news(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("📡 NEWS AGENT — Node 1: DDG Search")
    print("="*60)

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
    errors:       List[str]        = []
    seen_urls:    set              = set()

    with DDGS() as ddgs:
        for i, entry in enumerate(SEARCH_QUERIES):
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
                        print(f"     ⏳ Rate limited — retrying in {wait:.0f}s…")
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

                parts      = url.split("/")
                source_url = "/".join(parts[:3]) if len(parts) >= 3 else url
                title      = r.get("title", "")[:200]
                image      = r.get("image") or None
                print(f"     + [{entry['category']}] {title[:70]}")

                raw_articles.append({
                    "title":        title,
                    "url":          url,
                    "image_url":    image,
                    "raw_text":     r.get("body", "")[:2000],
                    "category":     entry["category"],
                    "source":       r.get("source", "Unknown"),
                    "source_url":   source_url,
                    "published_at": published_at,
                })

    print(f"\n✅ DDG search done — {len(raw_articles)} raw articles, {len(errors)} errors")
    return {"raw_articles": raw_articles, "errors": errors, "status": "searched"}


# ── JSON extractor ────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    text = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    result = {}
    for key in ("score", "reason", "raw_summary", "ai_impact"):
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        m2 = re.search(pattern, text)
        if m2:
            result[key] = m2.group(1)
        else:
            num_pattern = rf'"{key}"\s*:\s*([0-9.]+)'
            m3 = re.search(num_pattern, text)
            if m3:
                result[key] = float(m3.group(1))
    if result:
        return result
    raise ValueError(f"Could not extract JSON from: {text[:200]!r}")


# ── Node 2: Concurrent LLM Judge ──────────────────────────────────────────────

# Trimmed system prompt — fewer tokens = cheaper + faster
_JUDGE_SYSTEM = (
    "Telecom/construction news filter. "
    "Reply ONLY with JSON, no markdown."
)

# Shorter prompt: title + 300 chars of snippet is plenty for a relevance score
_JUDGE_PROMPT = (
    "Score this article 0.0-1.0 for a telecom/construction briefing.\n"
    "Factors: relevance to telecom/infrastructure (0-0.4), "
    "newsworthiness (0-0.3), specificity (0-0.3).\n\n"
    "Title: {title}\n"
    "Snippet: {snippet}\n"
    "Source: {source}\n\n"
    'Reply ONLY: {{"score": 0.85, "reason": "one line"}}'
)


def _judge_one(art: RawArticle, call_llm) -> tuple[RawArticle | None, str | None]:
    """Judge a single article. Returns (article_or_None, error_or_None)."""
    try:
        prompt = _JUDGE_PROMPT.format(
            title=art["title"],
            snippet=art["raw_text"][:300],   # was 500 — 300 is plenty
            source=art["source"],
        )
        raw     = call_llm(
            prompt=prompt,
            system_prompt=_JUDGE_SYSTEM,
            max_tokens=60,          # was 100 — score+reason fit in 60 easily
            temperature=0.0,
            task="evaluate",
        )
        verdict = _extract_json(raw)
        score   = float(verdict.get("score", 0.5))
        reason  = verdict.get("reason", "")

        if score >= JUDGE_SCORE_THRESHOLD:
            print(f"  ✅ ACCEPT [{score:.2f}] {art['title'][:65]}")
            return art, None
        else:
            print(f"  ❌ REJECT [{score:.2f}] {art['title'][:65]}"
                  + (f"\n            {reason}" if reason else ""))
            return None, None

    except Exception as e:
        # Fail-open: keep the article on any error
        err = f"Judge error for '{art['title'][:50]}': {e}"
        print(f"  ⚠️  {err} → keeping")
        return art, err


def judge_articles(state: AgentState) -> dict:
    print("\n" + "="*60)
    print(f"⚖️  NEWS AGENT — Node 2: Concurrent LLM Judge (workers={JUDGE_WORKERS})")
    print("="*60)

    articles = state["raw_articles"]
    print(f"  Judging {len(articles)} articles in parallel…")

    if not articles:
        print("  ⚠️  No articles to judge!")
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

    # Fire all judge calls concurrently instead of one-by-one
    accepted: List[RawArticle] = []
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        futures = {pool.submit(_judge_one, art, call_llm): art for art in articles}
        for future in as_completed(futures):
            art_result, err = future.result()
            if err:
                errors.append(err)
            if art_result is not None:
                accepted.append(art_result)

    deduped = _dedup(accepted)
    print(f"\n✅ Judge done — {len(deduped)} accepted, {len(articles) - len(deduped)} dropped")
    return {"accepted_articles": deduped, "errors": errors, "status": "judged"}


def _dedup(articles: List[RawArticle]) -> List[RawArticle]:
    seen: set = set()
    out:  List[RawArticle] = []
    for art in articles:
        key = re.sub(r"[^a-z0-9]", "", art["title"].lower())[:60]
        if key not in seen:
            seen.add(key)
            out.append(art)
    return out


# ── Node 3: Finalize (no blocking enrichment) ─────────────────────────────────
#
# Articles are served immediately with raw_text as the summary.
# Enrichment (ai_impact + polished summary) happens lazily via enrich_one().

def _make_unenriched(i: int, art: RawArticle) -> EnrichedArticle:
    uid = f"art_{i}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
    return {
        "id":           uid,
        "title":        art["title"],
        "source":       art["source"],
        "source_url":   art["source_url"],
        "article_url":  art["url"],
        "image_url":    art.get("image_url"),
        "raw_summary":  art["raw_text"][:300],
        "ai_impact":    "",
        "category":     art["category"],
        "published_at": art["published_at"],
        "scraped_at":   datetime.utcnow().isoformat(),
        "enriched":     False,
    }


def finalize(state: AgentState) -> dict:
    articles  = state["accepted_articles"]
    items     = [_make_unenriched(i, art) for i, art in enumerate(articles)]
    print("\n" + "="*60)
    print(f"📰 NEWS AGENT — Done: {len(items)} articles ready (unenriched, fast)")
    print("="*60 + "\n")
    return {"enriched_articles": items, "status": "done"}


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(AgentState)
    g.add_node("search",   search_news)
    g.add_node("judge",    judge_articles)
    g.add_node("finalize", finalize)
    g.set_entry_point("search")
    g.add_edge("search",   "judge")
    g.add_edge("judge",    "finalize")
    g.add_edge("finalize", END)
    return g.compile()

try:
    _graph = _build_graph()
    print("📰 News agent graph compiled ✅")
except Exception as _graph_err:
    _graph = None
    print(f"❌ Failed to build news graph: {_graph_err}")


# ── In-memory cache ────────────────────────────────────────────────────────────

_cache:      dict              = {}
_cache_time: Optional[datetime] = None
# article_id → raw RawArticle, kept so enrich_one() can re-use raw_text
_raw_by_id:  dict[str, RawArticle] = {}


def get_news_briefing(force: bool = False) -> dict:
    """
    Public API for main.py. Cached for CACHE_TTL_HOURS.
    Pass force=True to bypass cache and trigger a fresh scrape.
    """
    global _cache, _cache_time, _raw_by_id

    if _graph is None:
        raise RuntimeError(
            "News agent graph failed to initialise. "
            "Ensure langgraph is installed: pip install langgraph"
        )

    now         = datetime.utcnow()
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
        "raw_articles":      [],
        "accepted_articles": [],
        "enriched_articles": [],
        "errors":            [],
        "status":            "starting",
    }

    result = _graph.invoke(initial_state)

    items = result["enriched_articles"]

    # Rebuild raw-article lookup so lazy enrichment can use raw_text
    _raw_by_id = {}
    for item in items:
        # We stored raw_summary from raw_text[:300]; stash the full item for now.
        # enrich_one() will use this when the user opens a card.
        _raw_by_id[item["id"]] = item

    _cache = {
        "generated_at": now.isoformat(),
        "count":        len(items),
        "items":        items,
        "errors":       result["errors"],
    }
    _cache_time = now

    print(f"📰 Briefing cached — {_cache['count']} articles, {len(result['errors'])} errors")
    return _cache


# ── Lazy enrichment ────────────────────────────────────────────────────────────

_ENRICH_SYSTEM = (
    "You are a senior telecom industry analyst. "
    "Given a news article title and snippet, return ONLY a JSON object:\n"
    '{\n'
    '  "raw_summary": "2-sentence factual summary",\n'
    '  "ai_impact":   "2-3 sentences on what this means for telecom/construction professionals"\n'
    '}\n'
    "No markdown fences, no preamble. Just the JSON."
)


def enrich_one(article_id: str) -> Optional[dict]:
    """
    Enrich a single article on-demand (called when a user opens a card).
    Returns the updated article dict, or None if the id is unknown.
    Updates the in-memory cache in-place so subsequent GETs are already enriched.
    """
    item = _raw_by_id.get(article_id)
    if item is None:
        return None

    # Already enriched — return immediately, no LLM call
    if item.get("enriched"):
        return item

    try:
        from llm_client import call_llm, check_llm_available
        available, _ = check_llm_available()
        if not available:
            return item
    except ImportError:
        return item

    try:
        raw = call_llm(
            prompt=f"Title: {item['title']}\n\nSnippet:\n{item['raw_summary']}",
            system_prompt=_ENRICH_SYSTEM,
            max_tokens=350,
            temperature=0.3,
            task="synthesise",
        )
        parsed = _extract_json(raw)

        item["raw_summary"] = parsed.get("raw_summary", item["raw_summary"])
        item["ai_impact"]   = parsed.get("ai_impact", "")
        item["enriched"]    = True

        # Patch the cached briefing in-place
        if _cache and "items" in _cache:
            for cached_item in _cache["items"]:
                if cached_item["id"] == article_id:
                    cached_item.update(item)
                    break

        print(f"  ✅ Enriched on-demand: {item['title'][:65]}")
        return item

    except Exception as e:
        print(f"  ⚠️  On-demand enrich failed for {article_id}: {e}")
        return item