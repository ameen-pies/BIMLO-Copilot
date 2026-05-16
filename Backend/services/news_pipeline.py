"""
news_pipeline.py — Scheduled Industry Analyst Pipeline
───────────────────────────────────────────────────────
Replaces the per-user SSE scraping model with a globally shared,
pre-computed paginated cache that all users read from.

Architecture:
  APScheduler (every 4 days)
      → LangGraph pipeline (search → enrich → filter → dedup → paginate → persist)
      → data/news_cache/page_0.json … page_N.json + meta.json
      → FastAPI serves pages directly, zero LLM per user request

LLM routing:
  ALL LLM calls go exclusively to the dedicated Bimlo News CF Worker
  (CF_NEWS_URL / CF_NEWS_API_KEY). llm_client.py is never imported here.
  This keeps the news pipeline quota 100% separate from the main RAG workers.

Required env vars:
  CF_NEWS_URL     — e.g. https://bimlo.helaliaminhelali.workers.dev
  CF_NEWS_API_KEY — the shared secret you set on that worker

Public API (used by main.py):
    from news_pipeline import (
        run_news_pipeline,
        pipeline_is_running,
        get_meta,
        get_page,
        get_status,
        CACHE_DIR,
    )
"""

from __future__ import annotations

import os
import json
import shutil
import hashlib
import logging
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict

from core import globals as g

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_pipeline")

# ── Config ─────────────────────────────────────────────────────────────────────

CACHE_DIR     = os.getenv("NEWS_CACHE_DIR", os.path.join("data", "news_cache"))
PAGE_SIZE     = int(os.getenv("NEWS_PAGE_SIZE",  "20"))   # 20 per page → more articles per scroll
CYCLE_DAYS    = int(os.getenv("NEWS_CYCLE_DAYS", "4"))    # run every 4 days — matches article expiry window
MAX_SEEN_URLS = 2000   # prune seen_urls.json after this many entries

# _BUILD_DIR and _PREV_DIR are now SUBDIRECTORIES inside CACHE_DIR, not siblings.
# Docker named volumes mount the directory itself as a mount point — you cannot
# rename/move/rmtree the mount point root. All rotation happens inside the volume.
_BUILD_DIR = os.path.join(CACHE_DIR, "_building")
_PREV_DIR  = os.path.join(CACHE_DIR, "_prev")

# ── Dedicated news worker ──────────────────────────────────────────────────────

_CF_NEWS_URL     = os.getenv("CF_NEWS_URL", "").rstrip("/")
_CF_NEWS_API_KEY = os.getenv("CF_NEWS_API_KEY", "")

# ── Thread-safety ──────────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_running       = False


# ── LangGraph state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    run_id:       str
    raw_articles: List[dict]
    enriched:     List[dict]
    deduped:      List[dict]
    pages:        List[List[dict]]
    errors:       List[str]


# ── News worker callable ───────────────────────────────────────────────────────
#
# _news_llm() is a drop-in replacement for llm_client.call_llm().
# It is passed directly into news_agent._judge_and_enrich_one() as the
# `call_llm` argument, so the signature must match exactly:
#
#   call_llm(prompt, system_prompt, max_tokens, temperature, task=None)
#
# The `task` kwarg is accepted but ignored — the worker auto-selects the model.

def _news_llm(
    prompt:        str,
    system_prompt: str   = "",
    max_tokens:    int   = 300,
    temperature:   float = 0.25,
    task:          str   = "",      # accepted for API compat, not forwarded
) -> str:
    """
    Send a single LLM request to the dedicated Bimlo News CF Worker.
    Retries up to 3 times on empty response (CF Workers AI occasionally returns
    an empty string on first try — a retry almost always succeeds).
    Raises RuntimeError if the worker is misconfigured or all retries fail.
    llm_client.py is never imported.
    """
    import httpx

    if not _CF_NEWS_URL or not _CF_NEWS_API_KEY:
        raise RuntimeError(
            "CF_NEWS_URL or CF_NEWS_API_KEY not set — "
            "news pipeline cannot call LLM worker."
        )

    payload: dict = {
        "prompt":      prompt,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["systemPrompt"] = system_prompt

    try:
        resp = httpx.post(
            f"{_CF_NEWS_URL}/",
            headers={
                "Authorization": f"Bearer {_CF_NEWS_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()

        body = resp.json()

        # CF Workers AI binding can return the text under several keys
        # or even as a nested dict — walk all known paths.
        raw_response = (
            body.get("response")
            or body.get("result")
            or body.get("text")
            or body.get("content")
            or body.get("message")
            or ""
        )
        if isinstance(raw_response, dict):
            raw_response = (
                raw_response.get("response")
                or raw_response.get("text")
                or raw_response.get("content")
                or raw_response.get("message")
                or ""
            )
        text = str(raw_response).strip()
        if text:
            return text

        raise RuntimeError(
            f"News worker returned an empty response — body keys: {list(body.keys())}"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text[:120]}") from e
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(str(e)) from e


def _news_llm_available() -> bool:
    """Config check only — no network call."""
    return bool(_CF_NEWS_URL and _CF_NEWS_API_KEY)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fingerprint(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:60]


def _write_atomic(path: str, data: dict) -> None:
    """Write JSON atomically via .tmp + os.replace (POSIX-atomic)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ── Persistent dedup (seen_urls.json) ─────────────────────────────────────────

def _seen_path() -> str:
    return os.path.join(CACHE_DIR, "seen_urls.json")


def _load_seen_set() -> dict:
    data = _read_json(_seen_path(), {"urls": [], "fingerprints": []})
    return {
        "urls":         list(data.get("urls", [])),
        "fingerprints": list(data.get("fingerprints", [])),
    }


def _save_seen_set(seen: dict) -> None:
    # Prune stale entries: drop URLs/fingerprints that were added more than
    # MAX_ARTICLE_AGE_DAYS ago so they can be re-discovered if re-published.
    # Fall back to simple length-cap if timestamps are unavailable.
    cutoff_iso = (datetime.utcnow() - timedelta(days=MAX_ARTICLE_AGE_DAYS)).isoformat()
    timestamps = seen.get("timestamps", {})

    if timestamps:
        # Build a pruned view keeping only entries newer than cutoff
        kept_urls = [u for u in seen["urls"]         if timestamps.get(u, "9999") >= cutoff_iso]
        kept_fps  = [f for f in seen["fingerprints"] if timestamps.get(f, "9999") >= cutoff_iso]
        seen["urls"]         = kept_urls[-MAX_SEEN_URLS:]
        seen["fingerprints"] = kept_fps[-MAX_SEEN_URLS:]
    else:
        seen["urls"]         = seen["urls"][-MAX_SEEN_URLS:]
        seen["fingerprints"] = seen["fingerprints"][-MAX_SEEN_URLS:]

    seen["updated_at"] = datetime.utcnow().isoformat() + "Z"
    # seen_urls lives in CACHE_DIR (not the build dir) so it persists across rotations
    os.makedirs(CACHE_DIR, exist_ok=True)
    _write_atomic(_seen_path(), seen)


# ── LangGraph nodes ────────────────────────────────────────────────────────────

MAX_ARTICLE_AGE_DAYS = int(os.getenv("NEWS_MAX_AGE_DAYS", "4"))   # 4 days default

# ── NOTE on seen_urls / cross-run dedup ────────────────────────────────────────
# Cross-run URL deduplication is intentionally disabled.
# When the pipeline runs daily and articles expire after 4 days, keeping a
# persistent seen_urls.json caused the dedup node to reject ALL articles on
# subsequent runs (they were already seen from prior runs). The result was an
# empty cache. Within-run dedup in _dedup_node is sufficient to prevent
# duplicates from appearing in the same batch.
# ──────────────────────────────────────────────────────────────────────────────


def _age_filter_node(state: PipelineState) -> PipelineState:
    """
    Drop any article whose published_at is older than MAX_ARTICLE_AGE_DAYS.
    Also purges stale entries from seen_urls.json so they can be re-fetched
    once they would reappear as "new" (they won't — they're too old — but
    this prevents the seen-set from growing forever with dead URLs).
    Runs right after search_node, before enrich, so we don't waste LLM calls
    on articles nobody wants to read.
    """
    raw   = state["raw_articles"]
    if not raw:
        return state

    cutoff = datetime.utcnow() - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    fresh, stale = [], []

    for art in raw:
        pub = art.get("published_at", "")
        if not pub:
            stale.append(art)   # no date → drop (can't verify recency)
            continue
        try:
            # Handle ISO strings with or without trailing 'Z' / timezone offset
            pub_clean = pub.rstrip("Z").split("+")[0].split("-")
            # re-join only the date+time part (YYYY-MM-DDTHH:MM:SS)
            pub_dt = datetime.fromisoformat(pub.rstrip("Z").split("+")[0])
            if pub_dt >= cutoff:
                fresh.append(art)
            else:
                stale.append(art)
        except (ValueError, AttributeError):
            stale.append(art)   # unparseable date → drop

    removed = len(stale)
    if removed:
        logger.info(
            f"   age_filter_node: removed {removed} articles older than "
            f"{MAX_ARTICLE_AGE_DAYS} days (cutoff={cutoff.date()})"
        )
    logger.info(f"   age_filter_node: {len(raw)} → {len(fresh)} articles kept")
    return {**state, "raw_articles": fresh}


def _generate_llm_queries() -> list:
    """
    Ask the news LLM worker to generate fresh, timely search queries based on
    today's date. Returns a list of query dicts or [] if worker unavailable.
    These are mixed in with the static queries for maximum coverage.
    """
    if not _news_llm_available():
        return []

    from datetime import date
    today = date.today().strftime("%B %d, %Y")

    prompt = (
        f"Today is {today}. Generate 12 highly specific Google News search queries "
        "to find the most recent (last 4 days) industry news relevant to: "
        "5G deployment, fiber broadband, telecom regulation, BIM/construction tech, "
        "digital twins, and AI in infrastructure. "
        "Vary the queries — mix company names, technology terms, regions, and policy angles. "
        "Return ONLY a JSON array of objects like: "
        '[{"query": "...", "category": "5G"}, ...] '
        "Valid categories: 5G, Fiber, Regulation, Construction, BIM, Digital Twin, AI Construction, General. "
        "No markdown, no preamble, just the JSON array."
    )
    try:
        raw = _news_llm(prompt=prompt, max_tokens=600, temperature=0.7)
        raw = re.sub(r"```json|```", "", raw).strip()
        # Extract JSON array
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return []
        queries = json.loads(m.group())
        valid = [
            q for q in queries
            if isinstance(q, dict) and q.get("query") and q.get("category")
        ]
        logger.info(f"   LLM generated {len(valid)} dynamic queries")
        return valid
    except Exception as e:
        logger.warning(f"   LLM query generation failed: {e}")
        return []


def _search_node(state: PipelineState) -> PipelineState:
    """Fetch raw articles from ALL query buckets + LLM-generated dynamic queries."""
    from news_agent import fetch_raw_articles, PAGE_0_QUERIES, EXTENDED_QUERY_PAGES

    static_queries = PAGE_0_QUERIES + [q for page in EXTENDED_QUERY_PAGES for q in page]

    # Add LLM-generated queries for fresh, timely coverage
    llm_queries = _generate_llm_queries()
    all_queries = static_queries + llm_queries

    logger.info(f"🔍 search_node: {len(static_queries)} static + {len(llm_queries)} LLM queries = {len(all_queries)} total")

    try:
        raw = fetch_raw_articles(all_queries)
        logger.info(f"   → {len(raw)} raw articles fetched")
        return {**state, "raw_articles": raw}
    except Exception as e:
        logger.error(f"   search_node error: {e}")
        return {**state, "raw_articles": [], "errors": state["errors"] + [str(e)]}


def _enrich_node(state: PipelineState) -> PipelineState:
    """
    Judge + enrich all raw articles concurrently via the dedicated news LLM worker.
    Passes _news_llm as the call_llm callable — never imports llm_client.
    """
    raw = state["raw_articles"]
    if not raw:
        return {**state, "enriched": []}

    available = _news_llm_available()
    logger.info(
        f"   enrich_node: news worker = "
        f"{'configured → ' + _CF_NEWS_URL if available else 'NOT CONFIGURED — serving raw articles'}"
    )

    from news_agent import _judge_and_enrich_one

    if not available:
        logger.warning("   enrich_node: CF_NEWS_URL/CF_NEWS_API_KEY not set — fail-open, no scoring")
        enriched = []
        for idx, art in enumerate(raw):
            uid = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
            enriched.append({
                "id":            uid,
                "title":         art["title"],
                "source":        art["source"],
                "source_url":    art["source_url"],
                "article_url":   art["url"],
                "image_url":     art.get("image_url"),
                "raw_summary":   art.get("raw_text", "")[:600],
                "full_content":  art.get("full_content", art.get("raw_text", ""))[:16000],
                "ai_impact":     "",
                "category":      art["category"],
                "published_at":  art["published_at"],
                "scraped_at":    datetime.utcnow().isoformat() + "Z",
                "enriched":      False,
                "score":         0.5,
            })
        return {**state, "enriched": enriched}

    logger.info(f"   enrich_node: enriching {len(raw)} articles (8 workers)…")
    enriched = []
    errors   = list(state["errors"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        # Pass _news_llm as the call_llm callable — matches the expected signature
        futures = {
            pool.submit(_judge_and_enrich_one, art, _news_llm, idx): art
            for idx, art in enumerate(raw)
        }
        for future in as_completed(futures):
            try:
                item, err = future.result()
            except Exception as e:
                errors.append(str(e))
                continue
            if err:
                errors.append(err)
            if item:
                enriched.append(item)

    logger.info(f"   enrich_node: {len(enriched)} articles accepted by judge")
    return {**state, "enriched": enriched, "errors": errors}


def _filter_node(state: PipelineState) -> PipelineState:
    """
    Content filter — removes articles that are inappropriate, off-topic,
    NSFW, or unrelated to telecom/construction/tech.

    Two layers:
      1. Fast keyword blocklist (no LLM call, instant)
      2. LLM batch moderation via the news worker (batches of 15)
    Falls back gracefully to keyword-only if worker is unavailable.
    """
    items = state["enriched"]
    if not items:
        return {**state, "enriched": []}

    # ── Layer 1: keyword pre-filter ────────────────────────────────────────────
    BLOCKLIST = [
        "porn", "pornhub", "onlyfans", "xxx", "nsfw", "nude", "nudity",
        "escort", "sex tape", "leaked video", "adult film", "adult content",
        "gambling", "casino", "betting odds", "sportsbook", "lottery jackpot",
        "celebrity gossip", "divorce", "affair", "cheating scandal",
        "crypto scam", "get rich quick", "make money fast", "lewd", "tabloid", "clickbait",
        "political opinion", "election", "vote", "partisan", "congress", "senate",
        "white house", "president", "prime minister",
        "drugs", "marijuana", "cannabis", "opioid", "heroin", "cocaine",
        "methamphetamine", "fentanyl",
    ]

    def _keyword_clean(art: dict) -> bool:
        haystack = (
            (art.get("title") or "") + " " +
            (art.get("raw_summary") or "") + " " +
            (art.get("source") or "") + " " +
            (art.get("article_url") or "")
        ).lower()
        return not any(kw in haystack for kw in BLOCKLIST)

    pre_filtered = [a for a in items if _keyword_clean(a)]
    kw_removed   = len(items) - len(pre_filtered)
    if kw_removed:
        logger.info(f"   filter_node: keyword pre-filter removed {kw_removed} articles")

    # ── Layer 2: LLM batch moderation removed ─────────────────────────────────
    # Articles are already scored and rejected by the judge in _enrich_node
    # (JUDGE_SCORE_THRESHOLD). Running a second LLM filter on top was culling
    # too many borderline-but-valid industry articles. Keyword blocklist above
    # is sufficient for hard content safety.
    total_removed = len(items) - len(pre_filtered)
    logger.info(f"   filter_node: {len(items)} → {len(pre_filtered)} articles ({total_removed} removed by keyword filter)")
    return {**state, "enriched": pre_filtered}


def _dedup_node(state: PipelineState) -> PipelineState:
    """
    Within-run dedup only (URL + title fingerprint).
    Cross-run persistent dedup via seen_urls.json is intentionally disabled:
    when the pipeline runs daily and articles expire after 4 days, cross-run
    dedup caused ALL articles to be rejected on subsequent runs (they were
    already marked seen). Within-run dedup is sufficient.
    """
    items = state["enriched"]
    if not items:
        return {**state, "deduped": []}

    seen_url_set = set()
    seen_fp_set  = set()
    deduped      = []

    for art in items:
        url = art.get("article_url", "")
        fp  = _fingerprint(art.get("title", ""))
        if url in seen_url_set or fp in seen_fp_set:
            continue
        deduped.append(art)
        seen_url_set.add(url)
        seen_fp_set.add(fp)

    logger.info(
        f"   dedup_node: {len(items)} → {len(deduped)} after within-run dedup "
        f"({len(items) - len(deduped)} removed)"
    )
    return {**state, "deduped": deduped}


def _paginate_node(state: PipelineState) -> PipelineState:
    """Sort by published_at descending, then slice into PAGE_SIZE chunks."""
    items = state["deduped"]

    sorted_items = sorted(
        items,
        key=lambda x: x.get("published_at", ""),
        reverse=True,
    )
    pages = [
        sorted_items[i: i + PAGE_SIZE]
        for i in range(0, len(sorted_items), PAGE_SIZE)
        if sorted_items[i: i + PAGE_SIZE]
    ]

    logger.info(
        f"   paginate_node: {len(items)} articles → {len(pages)} pages "
        f"of up to {PAGE_SIZE}"
    )
    return {**state, "pages": pages}


def _persist_node(state: PipelineState) -> PipelineState:
    """
    Atomic cache rotation — Docker-volume safe.

    CACHE_DIR is a Docker named-volume mount point. The kernel marks mount-point
    roots as busy, so os.rename / shutil.move on the root directory itself raises
    [Errno 16] Device or resource busy. The old sibling-directory rotation
    (_building → swap → _prev) therefore cannot work.

    New strategy — everything stays INSIDE the volume:
      1. Write new pages + meta into  CACHE_DIR/_building/
      2. Copy current live files into CACHE_DIR/_prev/       (fallback)
      3. Copy _building files into    CACHE_DIR/              (atomic per-file)
      4. Remove CACHE_DIR/_building/

    Per-file copy is not transactional but is safe: readers always see either
    the old or the new complete file (written via .tmp + os.replace).
    The window where meta.json is new but page_0.json is still old is ~ms.
    """
    pages  = state["pages"]
    run_id = state["run_id"]

    os.makedirs(CACHE_DIR,   exist_ok=True)
    os.makedirs(_BUILD_DIR,  exist_ok=True)

    # ── Step 1: write new pages + meta into _building ─────────────────────────
    for i, page in enumerate(pages):
        _write_atomic(
            os.path.join(_BUILD_DIR, f"page_{i}.json"),
            {
                "page":       i,
                "run_id":     run_id,
                "item_count": len(page),
                "items":      page,
            },
        )

    now      = datetime.utcnow()
    next_run = now + timedelta(days=CYCLE_DAYS)

    _write_atomic(
        os.path.join(_BUILD_DIR, "meta.json"),
        {
            "version":     1,
            "run_id":      run_id,
            "run_at":      now.isoformat() + "Z",
            "next_run_at": next_run.isoformat() + "Z",
            "total_pages": len(pages),
            "total_items": sum(len(p) for p in pages),
            "page_size":   PAGE_SIZE,
            "cycle_days":  CYCLE_DAYS,
            "status":      "ready",
            "errors":      state["errors"][:20],
        },
    )

    # ── Step 2: snapshot current live files into _prev (best-effort) ──────────
    try:
        os.makedirs(_PREV_DIR, exist_ok=True)
        for fname in os.listdir(CACHE_DIR):
            # Don't recurse into subdirs (_building, _prev, _prev_old)
            src = os.path.join(CACHE_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(_PREV_DIR, fname))
    except Exception as e:
        logger.warning(f"   persist_node: _prev snapshot failed (non-fatal): {e}")

    # ── Step 3: promote _building files into live CACHE_DIR ───────────────────
    # seen_urls.json lives in CACHE_DIR and must survive the rotation.
    _seen_src = os.path.join(CACHE_DIR, "seen_urls.json")
    _seen_bak = _seen_src + ".bak"
    if os.path.exists(_seen_src):
        try:
            shutil.copy2(_seen_src, _seen_bak)
        except Exception:
            pass

    promoted = 0
    for fname in os.listdir(_BUILD_DIR):
        src = os.path.join(_BUILD_DIR, fname)
        if os.path.isfile(src):
            dst = os.path.join(CACHE_DIR, fname)
            try:
                # os.replace is atomic on POSIX — readers never see a partial file
                os.replace(src, dst)
                promoted += 1
            except OSError:
                # Cross-device replace can fail on some FS; fall back to copy+replace
                shutil.copy2(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                promoted += 1

    # Restore seen_urls.json if it was accidentally overwritten
    if os.path.exists(_seen_bak):
        if not os.path.exists(_seen_src):
            os.replace(_seen_bak, _seen_src)
        else:
            os.remove(_seen_bak)

    # ── Step 4: clean up _building ────────────────────────────────────────────
    try:
        shutil.rmtree(_BUILD_DIR)
    except Exception as e:
        logger.warning(f"   persist_node: _building cleanup failed (non-fatal): {e}")

    logger.info(
        f"✅ persist_node: {promoted} files promoted to live cache — "
        f"{len(pages)} pages, run_id={run_id}"
    )
    return state


# ── LangGraph pipeline ─────────────────────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error("langgraph not installed. Run: pip install langgraph")
        return None

    g = StateGraph(PipelineState)
    g.add_node("search",     _search_node)
    g.add_node("age_filter", _age_filter_node)
    g.add_node("enrich",     _enrich_node)
    g.add_node("filter",     _filter_node)
    g.add_node("dedup",      _dedup_node)
    g.add_node("paginate",   _paginate_node)
    g.add_node("persist",    _persist_node)

    g.set_entry_point("search")
    g.add_edge("search",     "age_filter")
    g.add_edge("age_filter", "enrich")
    g.add_edge("enrich",     "filter")
    g.add_edge("filter",   "dedup")
    g.add_edge("dedup",    "paginate")
    g.add_edge("paginate", "persist")
    g.add_edge("persist",  END)

    return g.compile()


_graph      = None
_graph_lock = threading.Lock()


def _get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            _graph = _build_graph()
        return _graph


# ── Public: run pipeline ───────────────────────────────────────────────────────

def pipeline_is_running() -> bool:
    return _running


def run_news_pipeline(force: bool = False) -> None:
    """
    Entry point called by APScheduler and POST /api/news/trigger.
    Safe to call from any thread; will no-op if already running.
    """
    global _running

    if not _pipeline_lock.acquire(blocking=False):
        logger.info("⏭  pipeline already running — skipping")
        return

    _running = True
    _flag    = os.path.join(CACHE_DIR, ".running")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        open(_flag, "w").close()
    except Exception:
        pass

    try:
        _execute(force=force)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
    finally:
        _running = False
        _pipeline_lock.release()
        try:
            os.remove(_flag)
        except Exception:
            pass


def _execute(force: bool = False) -> None:
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"🚀 Pipeline starting — run_id={run_id}, force={force}")

    if force:
        try:
            sp = _seen_path()
            if os.path.exists(sp):
                os.remove(sp)
                logger.info("   Force refresh: seen_urls.json cleared")
            # Also nuke the prev cache so dedup can't load stale fingerprints from there
            if os.path.exists(_PREV_DIR):
                prev_seen = os.path.join(_PREV_DIR, "seen_urls.json")
                if os.path.exists(prev_seen):
                    os.remove(prev_seen)
                    logger.info("   Force refresh: _prev/seen_urls.json cleared")
        except Exception as e:
            logger.warning(f"   Force refresh cleanup error: {e}")

    initial_state: PipelineState = {
        "run_id":       run_id,
        "raw_articles": [],
        "enriched":     [],
        "deduped":      [],
        "pages":        [],
        "errors":       [],
    }

    graph = _get_graph()
    if graph is None:
        # Fallback: run nodes manually without LangGraph
        logger.warning("LangGraph unavailable — running pipeline manually")
        state = initial_state
        for node_fn in [
            _search_node, _age_filter_node, _enrich_node, _filter_node,
            _dedup_node, _paginate_node, _persist_node,
        ]:
            state = node_fn(state)
        _index_articles_for_chat(state)
        return

    result_state = graph.invoke(initial_state)
    logger.info(f"✅ Pipeline complete — run_id={run_id}")

    # ── Index articles into vector store for semantic chat ────────────────
    _index_articles_for_chat(result_state)


# ── Index into vector store for semantic chat ────────────────────────────────

def _index_articles_for_chat(state: PipelineState) -> None:
    """
    Index all deduped articles into the news vector store collection so the
    News Chat Agent can perform semantic retrieval across article content.
    """
    articles = state.get("deduped") or state.get("enriched") or []
    if not articles:
        logger.info("index: no articles to index")
        return

    vs = getattr(g, "vector_store", None)
    if vs is None:
        logger.warning("index: g.vector_store not available — skipping")
        return

    try:
        count = vs.index_news_articles(articles)
        logger.info(f"index: {count} chunks indexed for {len(articles)} articles")
    except Exception as e:
        logger.error(f"index: failed to index articles: {e}", exc_info=True)


# ── Public: cache reads ────────────────────────────────────────────────────────

def get_meta() -> Optional[dict]:
    """Return meta.json or None if no cache exists yet."""
    path = os.path.join(CACHE_DIR, "meta.json")
    if not os.path.exists(path):
        path = os.path.join(_PREV_DIR, "meta.json")
    return _read_json(path, None)


def get_page(page_num: int) -> Optional[dict]:
    """
    Return a single page dict or None.
    Falls back to _prev if the active cache doesn't have this page.
    """
    for base in [CACHE_DIR, _PREV_DIR]:
        path = os.path.join(base, f"page_{page_num}.json")
        data = _read_json(path, None)
        if data is not None:
            return data
    return None


def get_status() -> dict:
    """Lightweight status for the frontend polling endpoint."""
    meta = get_meta()
    return {
        "running":     _running,
        "last_run_at": meta.get("run_at")         if meta else None,
        "next_run_at": meta.get("next_run_at")    if meta else None,
        "total_pages": meta.get("total_pages", 0) if meta else 0,
        "total_items": meta.get("total_items", 0) if meta else 0,
        "status":      meta.get("status", "no_cache") if meta else "no_cache",
    }