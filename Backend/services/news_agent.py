"""
Industry Analyst Agent — news_agent.py
────────────────────────────────────────────────────────────────
Pipeline (streaming):
  1. search_news  — concurrent DuckDuckGo across all query buckets
  2. judge+enrich — single merged LLM call per article (concurrent)
  3. stream_briefing — generator that yields articles one-by-one as
                       they finish, so the frontend can render immediately

Key changes vs v1:
  • Judge + enrich collapsed into ONE LLM call per article
  • All DDG searches run concurrently (ThreadPoolExecutor)
  • Results stream out via SSE as each article finishes — no waiting
  • Global URL + title dedup across all pages/sessions
  • Rotating extended query set for infinite-scroll "next page" fetches
  • Full cache kept for instant repeat visits within TTL

Public API (used by main.py):
    from news_agent import stream_briefing, get_cached_briefing, get_next_page

Requirements:
    pip install ddgs langgraph
"""

import re
import json
import time
import hashlib
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Generator, Iterator, List, Optional, TypedDict, Annotated
import operator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_agent")


# ── Query buckets ──────────────────────────────────────────────────────────────
# PAGE_0 = initial load queries (shown on first open)
# PAGE_N = extended queries rotated in for infinite scroll pages

PAGE_0_QUERIES = [
    {"query": "5G network deployment 2025",          "category": "5G"},
    {"query": "5G infrastructure rollout",           "category": "5G"},
    {"query": "fiber broadband expansion 2025",      "category": "Fiber"},
    {"query": "rural broadband internet access",     "category": "Fiber"},
    {"query": "FCC telecom regulation 2025",         "category": "Regulation"},
    {"query": "spectrum policy broadband",           "category": "Regulation"},
    {"query": "telecom tower construction permit",   "category": "Construction"},
    {"query": "broadband infrastructure build",      "category": "Construction"},
    {"query": "telecom carrier ISP news 2025",       "category": "General"},
    {"query": "telecom industry quarterly results",  "category": "General"},
    # ── Bimlo core business ───────────────────────────────────────────────────
    {"query": "BIM building information modeling 2025",        "category": "BIM"},
    {"query": "scan to BIM digital twin construction",         "category": "BIM"},
    {"query": "digital twin infrastructure 2025",              "category": "Digital Twin"},
    {"query": "predictive maintenance digital twin AI",        "category": "Digital Twin"},
    {"query": "artificial intelligence construction industry", "category": "AI Construction"},
    {"query": "AI building infrastructure optimization 2025",  "category": "AI Construction"},
]

# Each sub-list = one "infinite scroll page" of fresh queries
EXTENDED_QUERY_PAGES = [
    [
        {"query": "5G mmWave private network",           "category": "5G"},
        {"query": "open RAN vRAN 2025",                  "category": "5G"},
        {"query": "FTTH FTTB fiber to the home",         "category": "Fiber"},
        {"query": "submarine cable internet 2025",       "category": "Fiber"},
        {"query": "net neutrality FCC ruling",           "category": "Regulation"},
        {"query": "telecom merger acquisition 2025",     "category": "Regulation"},
        {"query": "cell tower zoning approval",          "category": "Construction"},
        {"query": "data center power infrastructure",    "category": "Construction"},
        {"query": "Verizon AT&T T-Mobile news",          "category": "General"},
        {"query": "satellite internet Starlink LEO",     "category": "General"},
        {"query": "BIM 4D 5D project planning 2025",     "category": "BIM"},
        {"query": "Revit IFC BIM interoperability",      "category": "BIM"},
        {"query": "digital twin smart city IoT",         "category": "Digital Twin"},
        {"query": "industry 4.0 digital twin factory",  "category": "Digital Twin"},
        {"query": "generative AI architecture design",   "category": "AI Construction"},
        {"query": "AI structural engineering automation","category": "AI Construction"},
    ],
    [
        {"query": "5G standalone core network",          "category": "5G"},
        {"query": "network slicing edge computing 5G",   "category": "5G"},
        {"query": "dark fiber lease IRU deal",           "category": "Fiber"},
        {"query": "broadband subsidy BEAD program",      "category": "Fiber"},
        {"query": "EU telecom regulation digital act",   "category": "Regulation"},
        {"query": "radio frequency interference ruling", "category": "Regulation"},
        {"query": "small cell densification urban",      "category": "Construction"},
        {"query": "underground conduit fiber dig",       "category": "Construction"},
        {"query": "MVNO wholesale agreement 2025",       "category": "General"},
        {"query": "telecom workforce layoff hiring",     "category": "General"},
        {"query": "BIM mandate government infrastructure","category": "BIM"},
        {"query": "point cloud scan to BIM software",   "category": "BIM"},
        {"query": "digital twin energy building performance","category": "Digital Twin"},
        {"query": "real-time digital twin simulation 2025","category": "Digital Twin"},
        {"query": "machine learning site planning construction","category": "AI Construction"},
        {"query": "computer vision construction safety AI",  "category": "AI Construction"},
    ],
    [
        {"query": "C-band 5G spectrum deployment",       "category": "5G"},
        {"query": "millimeter wave 5G enterprise",       "category": "5G"},
        {"query": "ISP fixed wireless broadband rural",  "category": "Fiber"},
        {"query": "middle mile broadband grant",         "category": "Fiber"},
        {"query": "telecom antitrust DOJ FTC 2025",      "category": "Regulation"},
        {"query": "universal service fund USF reform",   "category": "Regulation"},
        {"query": "utility pole attachment rate",        "category": "Construction"},
        {"query": "fiber aerial vs underground cost",    "category": "Construction"},
        {"query": "wholesale bandwidth pricing trend",   "category": "General"},
        {"query": "IoT connectivity smart city 2025",    "category": "General"},
        {"query": "BIM digital construction twin CDE",   "category": "BIM"},
        {"query": "BIM coordination clash detection MEP","category": "BIM"},
        {"query": "digital twin infrastructure lifecycle","category": "Digital Twin"},
        {"query": "AEC digital twin deployment case study","category": "Digital Twin"},
        {"query": "AI predictive maintenance building 2025","category": "AI Construction"},
        {"query": "robotics automation construction site 2025","category": "AI Construction"},
    ],
]

MAX_RESULTS_PER_QUERY = 8
CACHE_TTL_HOURS       = 6
JUDGE_SCORE_THRESHOLD = 0.28
LLM_WORKERS           = 8   # concurrent judge+enrich calls


# ── TypedDicts ─────────────────────────────────────────────────────────────────

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
    enriched:     bool


# ── Global dedup state ─────────────────────────────────────────────────────────
# Tracks every URL + title-fingerprint ever served so no dupe survives
# a page reload or infinite-scroll fetch.

_seen_urls:        set = set()
_seen_fingerprints: set = set()
_dedup_lock = threading.Lock()

def _fingerprint(title: str) -> str:
    """Aggressively normalised 60-char prefix — catches rephrased headlines."""
    return re.sub(r"[^a-z0-9]", "", title.lower())[:60]

def _is_dupe(url: str, title: str) -> bool:
    fp = _fingerprint(title)
    with _dedup_lock:
        if url in _seen_urls or fp in _seen_fingerprints:
            return True
        return False

def _register(url: str, title: str):
    fp = _fingerprint(title)
    with _dedup_lock:
        _seen_urls.add(url)
        _seen_fingerprints.add(fp)

def reset_dedup():
    """Call before a force-refresh so old articles don't block new ones."""
    with _dedup_lock:
        _seen_urls.clear()
        _seen_fingerprints.clear()


# ── JSON extractor ─────────────────────────────────────────────────────────────

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
            num_p = rf'"{key}"\s*:\s*([0-9.]+)'
            m3 = re.search(num_p, text)
            if m3:
                result[key] = float(m3.group(1))
    if result:
        return result
    raise ValueError(f"Could not extract JSON from: {text[:200]!r}")


# ── DDG search (concurrent) ────────────────────────────────────────────────────

def _search_one(entry: dict, ddgs_instance, seen_local: set) -> List[RawArticle]:
    """Search a single query and return raw articles. Thread-safe per ddgs instance."""
    results = []
    for attempt in range(3):
        try:
            results = list(ddgs_instance.news(
                entry["query"],
                max_results=MAX_RESULTS_PER_QUERY,
                safesearch="off",
                timelimit="m",
            ))
            break
        except Exception as e:
            if "Ratelimit" in str(e) and attempt < 2:
                wait = random.uniform(6, 12) * (attempt + 1)
                logger.warning(f"DDG rate limited — retry in {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.warning(f"DDG failed for '{entry['query']}': {e}")
                break

    articles: List[RawArticle] = []
    for r in results:
        url = r.get("url", "")
        if not url or url in seen_local:
            continue
        seen_local.add(url)

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

        articles.append({
            "title":        title,
            "url":          url,
            "image_url":    image,
            "raw_text":     r.get("body", "")[:2000],
            "category":     entry["category"],
            "source":       r.get("source", "Unknown"),
            "source_url":   source_url,
            "published_at": published_at,
        })
    return articles


def fetch_raw_articles(queries: List[dict]) -> List[RawArticle]:
    """
    Run all queries concurrently using multiple DDGS sessions.
    DDG sessions are not thread-safe — each worker gets its own.
    """
    DDGS = None
    try:
        from ddgs import DDGS
    except ImportError:
        pass
    if DDGS is None:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.error("ddgs not installed. Run: pip install ddgs")
            return []

    seen_local: set = set()
    seen_lock = threading.Lock()
    all_articles: List[RawArticle] = []
    results_lock = threading.Lock()

    def worker(entry):
        # Each thread gets its own DDGS session
        with DDGS() as ddgs:
            # Add a small random stagger to avoid burst rate-limits
            time.sleep(random.uniform(0.2, 1.2))
            arts = _search_one(entry, ddgs, seen_local)
            with results_lock:
                all_articles.extend(arts)

    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as pool:
        futures = [pool.submit(worker, q) for q in queries]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.warning(f"Search worker error: {e}")

    logger.info(f"DDG fetch done — {len(all_articles)} raw articles from {len(queries)} queries")
    return all_articles


# ── Merged judge + enrich (one LLM call) ──────────────────────────────────────

_JUDGE_ENRICH_SYSTEM = (
    "You are a senior industry analyst and news editor for BIMLO, a company specialising in "
    "BIM engineering, telecom infrastructure, Digital Twin (DeepTwin), and AI applied to construction. "
    "Given a news article title, snippet, and category, do TWO things in one response:\n"
    "1. Score its relevance 0.0-1.0 for a professional briefing covering: telecom (5G, fiber, towers), "
    "BIM (Building Information Modeling, Scan-to-BIM, IFC, 4D/5D planning), Digital Twins (predictive maintenance, "
    "smart buildings, simulation), AI in construction/infrastructure, and regulation/industry news.\n"
    "2. If score >= 0.28, write a polished 2-sentence summary and a 2-sentence analyst insight.\n\n"
    "Reply ONLY with this JSON, no markdown:\n"
    '{"score": 0.85, "raw_summary": "...", "ai_impact": "..."}\n'
    "If score < 0.28, still return JSON but leave raw_summary and ai_impact as empty strings."
)

_JUDGE_ENRICH_PROMPT = (
    "Category: {category}\n"
    "Title: {title}\n"
    "Snippet: {snippet}\n\n"
    "Score + summarise if relevant."
)


def _judge_and_enrich_one(
    art: RawArticle, call_llm, idx: int
) -> tuple[Optional["EnrichedArticle"], Optional[str]]:
    """Single LLM call: judge relevance AND produce summary+impact together."""
    try:
        prompt = _JUDGE_ENRICH_PROMPT.format(
            category=art["category"],
            title=art["title"],
            snippet=art["raw_text"][:400],
        )
        raw    = call_llm(
            prompt=prompt,
            system_prompt=_JUDGE_ENRICH_SYSTEM,
            max_tokens=300,
            temperature=0.25,
            task="evaluate",
        )
        parsed = _extract_json(raw)
        score  = float(parsed.get("score", 0.5))

        if score < JUDGE_SCORE_THRESHOLD:
            logger.info(f"  ❌ REJECT [{score:.2f}] {art['title'][:65]}")
            return None, None

        uid = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
        item: EnrichedArticle = {
            "id":           uid,
            "title":        art["title"],
            "source":       art["source"],
            "source_url":   art["source_url"],
            "article_url":  art["url"],
            "image_url":    art.get("image_url"),
            "raw_summary":  parsed.get("raw_summary") or art["raw_text"][:300],
            "ai_impact":    parsed.get("ai_impact", ""),
            "category":     art["category"],
            "published_at": art["published_at"],
            "scraped_at":   datetime.utcnow().isoformat(),
            "enriched":     True,
        }
        logger.info(f"  ✅ ACCEPT [{score:.2f}] {art['title'][:65]}")
        return item, None

    except Exception as e:
        # Fail-open: keep the article with raw text
        err = f"LLM error for '{art['title'][:50]}': {e}"
        logger.warning(f"  ⚠️  {err} → keeping raw")
        uid = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
        item = {
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
        return item, err


# ── In-memory cache ────────────────────────────────────────────────────────────

_cache:      dict               = {}
_cache_time: Optional[datetime] = None
_cache_lock  = threading.Lock()

# Tracks which extended query page we've served (for infinite scroll)
_next_page_idx: int = 0
_page_lock = threading.Lock()


# ── Core streaming pipeline ────────────────────────────────────────────────────

def stream_briefing(
    queries: List[dict],
    force_reset_dedup: bool = False,
) -> Iterator[EnrichedArticle]:
    """
    Generator — yields EnrichedArticle dicts one by one as they come out of
    the LLM pool. The FastAPI SSE endpoint iterates this and pushes each
    article to the client immediately.

    Articles are globally deduped before being yielded.
    Results are also accumulated into the cache.
    """
    if force_reset_dedup:
        reset_dedup()

    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
        logger.info(f"LLM provider: {provider if available else 'unavailable (fail-open)'}")
    except ImportError:
        available = False
        call_llm   = None
        logger.warning("llm_client not found — serving raw articles without scoring")

    # 1. Fetch all raw articles (concurrent DDG)
    logger.info(f"🔍 Fetching raw articles for {len(queries)} queries…")
    raw_articles = fetch_raw_articles(queries)

    # Filter out dupes from global seen set BEFORE sending to LLM
    fresh = [a for a in raw_articles if not _is_dupe(a["url"], a["title"])]
    logger.info(f"📋 {len(fresh)} fresh articles (deduped from {len(raw_articles)})")

    if not fresh:
        return

    new_items: List[EnrichedArticle] = []

    if not available or call_llm is None:
        # No LLM — serve everything raw
        for idx, art in enumerate(fresh):
            _register(art["url"], art["title"])
            uid  = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
            item: EnrichedArticle = {
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
            new_items.append(item)
            yield item
    else:
        # 2. Judge + enrich concurrently — yield each as it finishes
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futures = {
                pool.submit(_judge_and_enrich_one, art, call_llm, idx): art
                for idx, art in enumerate(fresh)
            }
            for future in as_completed(futures):
                try:
                    item, err = future.result()
                except Exception as e:
                    logger.warning(f"Worker exception: {e}")
                    continue

                if item is None:
                    continue

                # Register in global dedup now that it's accepted
                art = futures[future]
                _register(art["url"], art["title"])
                new_items.append(item)
                yield item

    # 3. Merge new items into the main cache
    _merge_into_cache(new_items)
    logger.info(f"✅ stream_briefing done — {len(new_items)} articles yielded & cached")


def _merge_into_cache(new_items: List[EnrichedArticle]):
    global _cache, _cache_time
    with _cache_lock:
        existing = _cache.get("items", [])
        existing_ids = {i["id"] for i in existing}
        merged = existing + [i for i in new_items if i["id"] not in existing_ids]
        _cache = {
            "generated_at": (_cache_time or datetime.utcnow()).isoformat(),
            "count":        len(merged),
            "items":        merged,
        }
        if _cache_time is None:
            _cache_time = datetime.utcnow()


def get_cached_briefing() -> dict:
    """Return current cache immediately (for fast repeat loads)."""
    with _cache_lock:
        return dict(_cache)


def cache_is_valid() -> bool:
    with _cache_lock:
        return bool(
            _cache
            and _cache_time is not None
            and (datetime.utcnow() - _cache_time) < timedelta(hours=CACHE_TTL_HOURS)
        )


def get_next_page_queries() -> List[dict]:
    """
    Return the next batch of extended queries for infinite scroll.
    Rotates through EXTENDED_QUERY_PAGES endlessly.
    """
    global _next_page_idx
    with _page_lock:
        queries = EXTENDED_QUERY_PAGES[_next_page_idx % len(EXTENDED_QUERY_PAGES)]
        _next_page_idx += 1
        return queries


def invalidate_cache():
    global _cache, _cache_time, _next_page_idx
    with _cache_lock:
        _cache      = {}
        _cache_time = None
    with _page_lock:
        _next_page_idx = 0
    reset_dedup()


# ── Backwards-compat shim (so existing imports don't break) ───────────────────

def get_news_briefing(force: bool = False) -> dict:
    """
    Blocking version kept for any callers that still use it.
    Drains the stream_briefing generator fully before returning.
    """
    if cache_is_valid() and not force:
        logger.info("📰 Returning cached briefing")
        return get_cached_briefing()

    if force:
        invalidate_cache()

    queries = PAGE_0_QUERIES
    for _ in stream_briefing(queries, force_reset_dedup=force):
        pass  # generator populates the cache as a side-effect

    return get_cached_briefing()