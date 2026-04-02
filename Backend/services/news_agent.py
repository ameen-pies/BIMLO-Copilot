"""
Industry Analyst Agent  —  news_agent.py
─────────────────────────────────────────
A LangGraph agent that:
  1. Fetches telecom / construction news via Firecrawl (handles JS, anti-bot, SSL)
     Falls back to direct requests scraping if FIRECRAWL_API_KEY is not set.
  2. Uses the shared llm_client (Groq / Cloudflare Workers AI) to generate
     concise summaries + actionable AI-impact analysis
  3. Returns a structured briefing dict that main.py's /api/news endpoint can serve

Public API (used by main.py):
    from news_agent import get_news_briefing
    briefing = get_news_briefing(force=False)   # synchronous, cache-aware

Requirements:
    pip install langgraph beautifulsoup4 lxml requests certifi firecrawl-py

Environment variables:
    FIRECRAWL_API_KEY  — get a free key at firecrawl.dev (500 free credits/mo)
    GROQ_API_KEY       — (or CF_API_KEY) for LLM enrichment
"""

import os
import re
import json
import time
import hashlib
import logging
import requests
import certifi
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional
import operator

from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ── Source list ────────────────────────────────────────────────────────────

NEWS_SOURCES = [
    {
        "url":        "https://www.rcrwireless.com/category/5g",
        "category":   "5G",
        "source":     "RCR Wireless",
        "source_url": "https://www.rcrwireless.com",
    },
    {
        "url":        "https://www.telecomramblings.com/",
        "category":   "General",
        "source":     "Telecom Ramblings",
        "source_url": "https://www.telecomramblings.com",
    },
    {
        "url":        "https://www.fierce-network.com/wireless",
        "category":   "5G",
        "source":     "Fierce Network",
        "source_url": "https://www.fierce-network.com",
    },
    {
        "url":        "https://www.constructiondive.com/",
        "category":   "Construction",
        "source":     "Construction Dive",
        "source_url": "https://www.constructiondive.com",
    },
    {
        "url":        "https://www.telecompaper.com/news",
        "category":   "General",
        "source":     "Telecompaper",
        "source_url": "https://www.telecompaper.com",
    },
    {
        "url":        "https://broadbandbreakfast.com/",
        "category":   "Regulation",
        "source":     "Broadband Breakfast",
        "source_url": "https://broadbandbreakfast.com",
    },
]

MAX_ARTICLES_PER_SOURCE = 3
CACHE_TTL_HOURS         = 6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── Firecrawl client (lazy init) ───────────────────────────────────────────
# FIX: The firecrawl-py SDK has changed its public API across versions.
#   v0/v1 (<=0.0.16): FirecrawlApp with .scrape_url()
#   v1+   (>=1.0.0) : FirecrawlApp with .scrape()  ← most pip installs today
#   v4 internal SDK : FirecrawlClient (not public)
#
# We auto-detect whichever is installed instead of hardcoding an import path.

_firecrawl_app = None

def _get_firecrawl():
    """Return a working Firecrawl instance if FIRECRAWL_API_KEY is set, else None."""
    global _firecrawl_app
    if _firecrawl_app is not None:
        return _firecrawl_app

    key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not key:
        return None

    try:
        # firecrawl-py >= 1.0 (most common pip install)
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=key)
        # Verify the right method exists before returning
        if not (hasattr(app, "scrape") or hasattr(app, "scrape_url")):
            raise AttributeError("FirecrawlApp has neither .scrape() nor .scrape_url()")
        _firecrawl_app = app
        logger.info("🔥 Firecrawl initialised (FirecrawlApp)")
        return _firecrawl_app
    except ImportError:
        logger.warning("firecrawl-py not installed — falling back to requests. Run: pip install firecrawl-py")
        return None
    except Exception as e:
        logger.warning(f"Firecrawl init failed: {e}")
        return None


# ── State ──────────────────────────────────────────────────────────────────

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

class AgentState(TypedDict):
    raw_articles:      Annotated[List[RawArticle],      operator.add]
    enriched_articles: Annotated[List[EnrichedArticle], operator.add]
    errors:            Annotated[List[str],             operator.add]
    status:            str


# ── Fetch helpers ──────────────────────────────────────────────────────────

def _fetch_html_requests(url: str, timeout: int = 20) -> Optional[str]:
    """Direct requests fetch with certifi SSL bundle and graceful error handling."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=certifi.where())
        if r.status_code in (403, 404):
            logger.warning(f"HTTP {r.status_code} on {url} — skipping")
            return None
        r.raise_for_status()
        return r.text
    except requests.exceptions.SSLError as e:
        logger.warning(f"SSL error on {url}: {e}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout on {url}")
        return None
    except Exception as e:
        logger.warning(f"Fetch failed ({url}): {e}")
        return None


def _fetch_via_firecrawl(url: str) -> Optional[dict]:
    """
    Scrape a single URL with Firecrawl.
    Supports both the old SDK (.scrape_url) and new SDK (.scrape) transparently.
    Returns a normalised plain dict with keys: markdown, html, metadata.
    """
    fc = _get_firecrawl()
    if not fc:
        return None
    try:
        # firecrawl-py >= 1.0.0 uses .scrape(url, formats=[...])
        if hasattr(fc, "scrape"):
            result = fc.scrape(url, formats=["markdown", "html"])
        # firecrawl-py < 1.0.0 uses .scrape_url(url, params={...})
        elif hasattr(fc, "scrape_url"):
            result = fc.scrape_url(url, params={"formats": ["markdown", "html"]})
        else:
            logger.warning("Firecrawl client has no usable scrape method")
            return None

        # Normalise Pydantic model or plain dict
        if hasattr(result, "model_dump"):
            return result.model_dump(exclude_none=True)
        if isinstance(result, dict):
            return result
        return None
    except Exception as e:
        logger.warning(f"Firecrawl failed ({url}): {e}")
        return None


# ── Article parsing ────────────────────────────────────────────────────────

def _extract_article_links(html: str, base_domain: str, listing_url: str) -> List[str]:
    soup  = BeautifulSoup(html, "lxml")
    links = []
    seen  = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("/"):
            href = base_domain.rstrip("/") + href
        if not href.startswith("http"):
            continue
        if (
            base_domain.split("//")[-1].split("/")[0] in href
            and len(href) > len(listing_url) + 5
            and "#" not in href
            and href not in seen
        ):
            seen.add(href)
            links.append(href)

    return links


def _parse_article_html(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    for sel in ["h1", "meta[property='og:title']", "title"]:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get("content", "") or tag.get_text(strip=True)
            if title:
                break

    image_url = None
    og_img = soup.select_one("meta[property='og:image']")
    if og_img:
        image_url = og_img.get("content")
    if not image_url:
        img = soup.select_one("article img[src], .post img[src], main img[src]")
        if img:
            image_url = img.get("src")
    if image_url and (image_url.startswith("data:") or len(image_url) < 12):
        image_url = None

    published_at = datetime.utcnow().isoformat()
    for sel in [
        "meta[property='article:published_time']",
        "meta[name='publishedTime']",
        "time[datetime]",
        "meta[name='date']",
    ]:
        tag = soup.select_one(sel)
        if tag:
            val = tag.get("content") or tag.get("datetime", "")
            if val:
                published_at = val
                break

    for sel in ["article", "main", "body"]:
        container = soup.select_one(sel)
        if container:
            for noise in container.select("nav,footer,aside,script,style,[class*='ad'],[class*='sidebar']"):
                noise.decompose()
            text = container.get_text(separator="\n", strip=True)
            if len(text) > 200:
                break
    else:
        text = soup.get_text(separator="\n", strip=True)

    return {
        "title":        title[:200],
        "image_url":    image_url,
        "published_at": published_at,
        "text":         text[:4000],
    }


def _parse_firecrawl_result(result: dict, fallback_url: str) -> dict:
    """Normalise a Firecrawl scrape result dict into article shape."""
    metadata = result.get("metadata") or {}
    # Handle both old-style camelCase and new snake_case metadata keys
    title = (
        metadata.get("title")
        or metadata.get("og_title")
        or metadata.get("ogTitle")
        or fallback_url.split("/")[-1]
    )
    image_url = (
        metadata.get("og_image")
        or metadata.get("ogImage")
        or metadata.get("image")
    )
    published_at = (
        metadata.get("published_time")
        or metadata.get("publishedTime")
        or metadata.get("article_published_time")
        or metadata.get("articlePublishedTime")
        or datetime.utcnow().isoformat()
    )
    text = result.get("markdown") or result.get("html") or ""

    return {
        "title":        str(title)[:200],
        "image_url":    image_url,
        "published_at": published_at,
        "text":         text[:4000],
    }


# ── LangGraph nodes ────────────────────────────────────────────────────────

def scrape_sources(state: AgentState) -> AgentState:
    """
    Node 1 — scrape listing pages and fetch individual articles.

    Strategy:
      - If FIRECRAWL_API_KEY is set → use Firecrawl for article pages
      - Always fall back to direct requests for listing pages
    """
    raw_articles: List[RawArticle] = []
    errors: List[str] = []
    use_firecrawl = _get_firecrawl() is not None

    if use_firecrawl:
        logger.info("🔥 Using Firecrawl for article fetching")
    else:
        logger.info("🕸️  Firecrawl not configured — using direct requests (set FIRECRAWL_API_KEY for better results)")

    for source in NEWS_SOURCES:
        listing_html = _fetch_html_requests(source["url"])
        if not listing_html:
            errors.append(f"Source listing unreachable: {source['url']}")
            continue

        base_domain = "/".join(source["url"].split("/")[:3])
        links = _extract_article_links(listing_html, base_domain, source["url"])[:MAX_ARTICLES_PER_SOURCE]

        if not links:
            errors.append(f"No article links found on {source['url']}")
            continue

        for link in links:
            parsed = None

            if use_firecrawl:
                fc_result = _fetch_via_firecrawl(link)
                if fc_result:
                    parsed = _parse_firecrawl_result(fc_result, link)

            if not parsed:
                art_html = _fetch_html_requests(link)
                if not art_html:
                    errors.append(f"Article unreachable: {link}")
                    continue
                parsed = _parse_article_html(art_html)

            title = parsed["title"] or link.split("/")[-1].replace("-", " ").title()

            raw_articles.append({
                "title":        title,
                "url":          link,
                "image_url":    parsed["image_url"],
                "raw_text":     parsed["text"],
                "category":     source["category"],
                "source":       source["source"],
                "source_url":   source["source_url"],
                "published_at": parsed["published_at"],
            })

            time.sleep(0.2 if use_firecrawl else 0.5)

    return {"raw_articles": raw_articles, "errors": errors, "status": "scraped"}


def enrich_articles(state: AgentState) -> AgentState:
    """Node 2 — use the shared llm_client to summarise + analyse each article."""
    enriched: List[EnrichedArticle] = []
    errors: List[str] = []

    try:
        from llm_client import call_llm, check_llm_available
        available, provider = check_llm_available()
        if not available:
            errors.append("No LLM provider available (set GROQ_API_KEY or CF_API_KEY) — skipping enrichment")
            return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}
        logger.info(f"🤖 Enriching via {provider}")
    except ImportError:
        errors.append("llm_client module not found — skipping enrichment")
        return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}

    system_prompt = (
        "You are a senior telecom industry analyst. "
        "Given raw article text, return ONLY a JSON object with exactly these two fields:\n"
        '{\n'
        '  "raw_summary": "2-sentence factual summary of the article",\n'
        '  "ai_impact":   "2-3 sentences on what this means for telecom/construction professionals right now — specific, actionable"\n'
        '}\n'
        "No markdown fences, no preamble, just the JSON object."
    )

    for i, art in enumerate(state["raw_articles"]):
        try:
            raw = call_llm(
                prompt=f"Title: {art['title']}\n\nContent:\n{art['raw_text']}",
                system_prompt=system_prompt,
                max_tokens=600,
                temperature=0.3,
                task="synthesise",
            )

            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            uid = f"art_{i}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"

            enriched.append({
                "id":           uid,
                "title":        art["title"],
                "source":       art["source"],
                "source_url":   art["source_url"],
                "article_url":  art["url"],
                "image_url":    art.get("image_url"),
                "raw_summary":  parsed.get("raw_summary", ""),
                "ai_impact":    parsed.get("ai_impact", ""),
                "category":     art["category"],
                "published_at": art["published_at"],
                "scraped_at":   datetime.utcnow().isoformat(),
            })

        except json.JSONDecodeError as e:
            errors.append(f"JSON parse failed for '{art['title']}': {e}")
        except Exception as e:
            errors.append(f"Enrichment failed for '{art['title']}': {e}")

    return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}


def finalize(state: AgentState) -> AgentState:
    return {"status": "done"}


# ── Build graph ────────────────────────────────────────────────────────────
# FIX: Wrap graph construction in a try/except so an import-time failure here
# does NOT crash main.py's startup. The error surfaces cleanly on the first
# API call instead of silently killing the whole server.

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("scrape",   scrape_sources)
    g.add_node("enrich",   enrich_articles)
    g.add_node("finalize", finalize)
    g.set_entry_point("scrape")
    g.add_edge("scrape",   "enrich")
    g.add_edge("enrich",   "finalize")
    g.add_edge("finalize", END)
    return g.compile()

try:
    _graph = _build_graph()
except Exception as _graph_err:
    _graph = None
    logger.error(f"❌ Failed to build LangGraph news graph: {_graph_err}")


# ── In-memory cache ────────────────────────────────────────────────────────

_cache: dict = {}
_cache_time: Optional[datetime] = None


def get_news_briefing(force: bool = False) -> dict:
    """
    Public synchronous API consumed by main.py:

        from news_agent import get_news_briefing
        briefing = get_news_briefing(force=False)

    Returns:
        {
            "generated_at": "ISO-8601",
            "count": int,
            "items": [ EnrichedArticle, ... ],
            "errors": [ str, ... ]
        }

    Cache is valid for CACHE_TTL_HOURS (default 6 h).
    Pass force=True to bypass the cache.
    """
    global _cache, _cache_time

    if _graph is None:
        raise RuntimeError(
            "News agent graph failed to initialise at startup. "
            "Check that langgraph is installed correctly: pip install langgraph"
        )

    now = datetime.utcnow()
    cache_valid = (
        _cache
        and _cache_time is not None
        and (now - _cache_time) < timedelta(hours=CACHE_TTL_HOURS)
    )

    if cache_valid and not force:
        logger.info("📰 Returning cached news briefing")
        return _cache

    logger.info("📰 Running news agent…")

    initial_state: AgentState = {
        "raw_articles":      [],
        "enriched_articles": [],
        "errors":            [],
        "status":            "starting",
    }

    result = _graph.invoke(initial_state)

    _cache = {
        "generated_at": now.isoformat(),
        "count":        len(result["enriched_articles"]),
        "items":        result["enriched_articles"],
        "errors":       result["errors"],
    }
    _cache_time = now

    logger.info(
        f"📰 News briefing ready — {_cache['count']} articles, "
        f"{len(result['errors'])} errors"
    )
    return _cache