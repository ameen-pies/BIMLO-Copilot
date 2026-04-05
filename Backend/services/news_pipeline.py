"""
news_pipeline.py — Scheduled Industry Analyst Pipeline
───────────────────────────────────────────────────────
Replaces the per-user SSE scraping model with a globally shared,
pre-computed paginated cache that all users read from.

Architecture:
  APScheduler (every 4 days)
      → LangGraph pipeline (search → enrich → dedup → paginate → persist)
      → data/news_cache/page_0.json … page_N.json + meta.json
      → FastAPI serves pages directly, zero LLM per user request

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_pipeline")

# ── Config ─────────────────────────────────────────────────────────────────────

CACHE_DIR      = os.getenv("NEWS_CACHE_DIR",  os.path.join("data", "news_cache"))
PAGE_SIZE      = int(os.getenv("NEWS_PAGE_SIZE",  "10"))
CYCLE_DAYS     = int(os.getenv("NEWS_CYCLE_DAYS",  "4"))
MAX_SEEN_URLS  = 2000   # prune seen_urls.json after this many entries

_BUILD_DIR     = CACHE_DIR + "_building"
_PREV_DIR      = CACHE_DIR + "_prev"

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fingerprint(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:60]


def _write_atomic(path: str, data: dict) -> None:
    """Write JSON atomically: write .tmp then os.replace (POSIX-atomic)."""
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
    # Prune to last MAX_SEEN_URLS entries so the file stays lightweight
    seen["urls"]         = seen["urls"][-MAX_SEEN_URLS:]
    seen["fingerprints"] = seen["fingerprints"][-MAX_SEEN_URLS:]
    seen["updated_at"]   = datetime.utcnow().isoformat() + "Z"
    # seen_urls lives in CACHE_DIR (not the build dir) so it persists across rotations
    os.makedirs(CACHE_DIR, exist_ok=True)
    _write_atomic(_seen_path(), seen)


# ── LangGraph nodes ────────────────────────────────────────────────────────────

def _search_node(state: PipelineState) -> PipelineState:
    """Fetch raw articles from ALL query buckets (PAGE_0 + all EXTENDED pages)."""
    from news_agent import fetch_raw_articles, PAGE_0_QUERIES, EXTENDED_QUERY_PAGES

    all_queries = PAGE_0_QUERIES + [q for page in EXTENDED_QUERY_PAGES for q in page]
    logger.info(f"🔍 search_node: running {len(all_queries)} queries…")

    try:
        raw = fetch_raw_articles(all_queries)
        logger.info(f"   → {len(raw)} raw articles fetched")
        return {**state, "raw_articles": raw}
    except Exception as e:
        logger.error(f"   search_node error: {e}")
        return {**state, "raw_articles": [], "errors": state["errors"] + [str(e)]}


def _enrich_node(state: PipelineState) -> PipelineState:
    """Judge + enrich all raw articles concurrently via LLM."""
    raw = state["raw_articles"]
    if not raw:
        return {**state, "enriched": []}

    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
        logger.info(f"   enrich_node: LLM provider = {provider if available else 'unavailable'}")
    except ImportError:
        available = False
        call_llm  = None

    from news_agent import _judge_and_enrich_one

    if not available or call_llm is None:
        # Fail-open: keep all raw articles without scoring
        logger.warning("   enrich_node: no LLM — serving raw articles")
        enriched = []
        for idx, art in enumerate(raw):
            uid = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"
            enriched.append({
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
                "scraped_at":   datetime.utcnow().isoformat() + "Z",
                "enriched":     False,
                "score":        0.5,
            })
        return {**state, "enriched": enriched}

    logger.info(f"   enrich_node: enriching {len(raw)} articles (8 workers)…")
    enriched = []
    errors   = list(state["errors"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_judge_and_enrich_one, art, call_llm, idx): art
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
    LLM-powered content filter — removes articles that are inappropriate,
    off-topic, NSFW, or have nothing to do with telecom/construction/tech industry.
    Falls back to a fast keyword blocklist if LLM is unavailable.
    """
    items = state["enriched"]
    if not items:
        return {**state, "enriched": []}

    # ── Fast keyword pre-filter (catches obvious stuff instantly) ──────────────
    BLOCKLIST = [
        "porn", "pornhub", "onlyfans", "xxx", "nsfw", "nude", "nudity",
        "escort", "sex tape", "leaked video", "adult film", "adult content",
        "gambling", "casino", "betting odds", "sportsbook", "lottery jackpot",
        "celebrity gossip", "divorce", "affair", "cheating scandal",
        "crypto scam", "get rich quick", "make money fast", "lewd", "tabloid", "clickbait",
        "political opinion", "election", "vote", "partisan", "congress", "senate", "white house", "president", "prime minister",
        "drugs", "marijuana", "cannabis", "opioid", "heroin", "cocaine", "methamphetamine", "fentanyl",
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

    # ── LLM batch filter ───────────────────────────────────────────────────────
    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
    except ImportError:
        available = False

    if not available or not pre_filtered:
        logger.info(f"   filter_node: LLM unavailable — keyword filter only, {len(pre_filtered)} articles kept")
        return {**state, "enriched": pre_filtered}

    # Batch articles into groups of 15 to save LLM calls
    BATCH = 15
    accepted = []

    for batch_start in range(0, len(pre_filtered), BATCH):
        batch = pre_filtered[batch_start: batch_start + BATCH]

        lines = "\n".join(
            f"{i+1}. [{a.get('category','')}] {a.get('title','')} | {a.get('source','')}"
            for i, a in enumerate(batch)
        )

        prompt = (
            "You are a strict content moderator for a professional telecom and construction industry news platform.\n"
            "Review the articles below. For each, reply with only its number if it should be REJECTED.\n"
            "Reject articles that are: NSFW/adult/sexual, gambling, celebrity gossip, "
            "tabloid clickbait, crypto scams, political opinion pieces unrelated to telecom/construction, "
            "or completely off-topic from: 5G, fiber, broadband, telecom regulation, construction, "
            "BIM, digital twins, AI in construction/telecom.\n"
            "If an article is borderline but industry-relevant, KEEP it.\n"
            "Reply with ONLY a comma-separated list of rejected numbers, or 'none' if all are fine.\n\n"
            f"Articles:\n{lines}"
        )

        try:
            raw = call_llm(prompt=prompt, system_prompt="You are a content moderation assistant. Be strict but fair.", max_tokens=100, temperature=0.0)
            raw = raw.strip().lower()

            if raw == "none" or not raw:
                rejected_indices = set()
            else:
                rejected_indices = set()
                for part in re.split(r"[,\s]+", raw):
                    part = part.strip().rstrip(".")
                    if part.isdigit():
                        rejected_indices.add(int(part) - 1)  # convert to 0-based

            kept    = [a for i, a in enumerate(batch) if i not in rejected_indices]
            removed = len(batch) - len(kept)
            if removed:
                logger.info(f"   filter_node: LLM rejected {removed} articles in batch starting at {batch_start}")
            accepted.extend(kept)

        except Exception as e:
            logger.warning(f"   filter_node: LLM batch failed ({e}) — keeping batch as-is")
            accepted.extend(batch)

    total_removed = len(items) - len(accepted)
    logger.info(f"   filter_node: {len(items)} → {len(accepted)} articles ({total_removed} total removed)")
    return {**state, "enriched": accepted}

def _dedup_node(state: PipelineState) -> PipelineState:
    """
    Three-layer dedup:
      1. Within-run URL dedup (across all queries in this cycle)
      2. Cross-run persistent dedup (seen_urls.json from previous cycles)
      3. Title fingerprint dedup (catches rephrased headlines)
    """
    items = state["enriched"]
    if not items:
        return {**state, "deduped": []}

    seen = _load_seen_set()
    seen_url_set = set(seen["urls"])
    seen_fp_set  = set(seen["fingerprints"])

    deduped      = []
    new_urls     = []
    new_fps      = []

    # Within-run dedup set (URL only — handles same article from two queries)
    within_run_urls = set()

    for art in items:
        url = art.get("article_url", "")
        fp  = _fingerprint(art.get("title", ""))

        if url in seen_url_set or fp in seen_fp_set or url in within_run_urls:
            continue

        deduped.append(art)
        within_run_urls.add(url)
        new_urls.append(url)
        new_fps.append(fp)

    # Persist the updated seen set
    seen["urls"]         += new_urls
    seen["fingerprints"] += new_fps
    _save_seen_set(seen)

    logger.info(f"   dedup_node: {len(items)} → {len(deduped)} after dedup "
                f"({len(items) - len(deduped)} removed)")
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
        sorted_items[i : i + PAGE_SIZE]
        for i in range(0, len(sorted_items), PAGE_SIZE)
        if sorted_items[i : i + PAGE_SIZE]   # skip empty tail
    ]

    logger.info(f"   paginate_node: {len(items)} articles → {len(pages)} pages "
                f"of up to {PAGE_SIZE}")
    return {**state, "pages": pages}


def _persist_node(state: PipelineState) -> PipelineState:
    """
    Atomic rotation:
      1. Write everything to CACHE_DIR_building/
      2. Move CACHE_DIR          → CACHE_DIR_prev  (fallback during next build)
      3. Move CACHE_DIR_building → CACHE_DIR        (goes live instantly)
    """
    pages  = state["pages"]
    run_id = state["run_id"]

    os.makedirs(_BUILD_DIR, exist_ok=True)

    # Write each page
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

    # Write meta
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
            "errors":      state["errors"][:20],  # keep first 20 errors for diagnostics
        },
    )

    # Atomic rotation
    if os.path.exists(_PREV_DIR):
        shutil.rmtree(_PREV_DIR)
    if os.path.exists(CACHE_DIR):
        # Move seen_urls.json OUT before rotation so it survives
        _seen = os.path.join(CACHE_DIR, "seen_urls.json")
        _seen_tmp = _seen + ".bak"
        if os.path.exists(_seen):
            shutil.copy2(_seen, _seen_tmp)
        os.rename(CACHE_DIR, _PREV_DIR)
        # Restore seen_urls into build dir (it's already written above but let's be safe)
        if os.path.exists(_seen_tmp):
            shutil.copy2(_seen_tmp, os.path.join(_BUILD_DIR, "seen_urls.json"))
            os.remove(_seen_tmp)
    os.rename(_BUILD_DIR, CACHE_DIR)

    logger.info(f"✅ persist_node: cache live — {len(pages)} pages, run_id={run_id}")
    return state


# ── Build and compile the LangGraph pipeline ──────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error("langgraph not installed. Run: pip install langgraph")
        return None

    g = StateGraph(PipelineState)
    g.add_node("search",   _search_node)
    g.add_node("enrich",   _enrich_node)
    g.add_node("filter",   _filter_node)
    g.add_node("dedup",    _dedup_node)
    g.add_node("paginate", _paginate_node)
    g.add_node("persist",  _persist_node)

    g.set_entry_point("search")
    g.add_edge("search",   "enrich")
    g.add_edge("enrich",   "filter")
    g.add_edge("filter",   "dedup")
    g.add_edge("dedup",    "paginate")
    g.add_edge("paginate", "persist")
    from langgraph.graph import END
    g.add_edge("persist",  END)

    return g.compile()


_graph = None
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
    # Write a .running flag so status endpoint can detect in-progress state
    _flag = os.path.join(CACHE_DIR, ".running")
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
        # Wipe the persistent seen set so old articles can reappear
        try:
            sp = _seen_path()
            if os.path.exists(sp):
                os.remove(sp)
                logger.info("   Force refresh: seen_urls.json cleared")
        except Exception:
            pass

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
        for node_fn in [_search_node, _enrich_node, _filter_node, _dedup_node, _paginate_node, _persist_node]:
            state = node_fn(state)
        return

    graph.invoke(initial_state)
    logger.info(f"✅ Pipeline complete — run_id={run_id}")


# ── Public: cache reads ────────────────────────────────────────────────────────

def get_meta() -> Optional[dict]:
    """Return meta.json or None if no cache exists yet."""
    path = os.path.join(CACHE_DIR, "meta.json")
    if not os.path.exists(path):
        # Try previous cycle as fallback
        path = os.path.join(_PREV_DIR, "meta.json")
    return _read_json(path, None)


def get_page(page_num: int) -> Optional[dict]:
    """
    Return a single page dict or None.
    Falls back to _prev if the active cache doesn't have this page
    (e.g. during a rebuild or on first boot).
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
        "last_run_at": meta.get("run_at")      if meta else None,
        "next_run_at": meta.get("next_run_at") if meta else None,
        "total_pages": meta.get("total_pages", 0) if meta else 0,
        "total_items": meta.get("total_items", 0) if meta else 0,
        "status":      meta.get("status", "no_cache") if meta else "no_cache",
    }