"""
Industry Analyst Agent  —  news_agent.py
─────────────────────────────────────────
A LangGraph agent that:
  1. Scrapes major telecom / construction news sites (via requests + BeautifulSoup)
  2. Uses the shared llm_client (Groq / Cloudflare Workers AI) to generate
     concise summaries + actionable AI-impact analysis — no paid API needed
  3. Returns a structured briefing dict that main.py's /api/news endpoint can serve

Public API (used by main.py):
    from news_agent import get_news_briefing
    briefing = get_news_briefing(force=False)   # synchronous, cache-aware

Requirements:
    pip install langgraph beautifulsoup4 lxml requests

Environment variables:
    GROQ_API_KEY   (or CF_API_KEY) — whichever llm_client is already configured
"""

import os
import re
import json
import time
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Optional
import operator

from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

NEWS_SOURCES = [
    {"url": "https://www.rcrwireless.com/category/5g",         "category": "5G",           "source": "RCR Wireless",      "source_url": "https://www.rcrwireless.com"},
    {"url": "https://www.lightreading.com/",                   "category": "General",      "source": "Light Reading",     "source_url": "https://www.lightreading.com"},
    {"url": "https://www.telecomramblings.com/",               "category": "General",      "source": "Telecom Ramblings", "source_url": "https://www.telecomramblings.com"},
    {"url": "https://www.ntia.gov/press-room",                 "category": "Regulation",   "source": "NTIA",              "source_url": "https://www.ntia.gov"},
    {"url": "https://www.fcc.gov/news-events/blog",            "category": "Regulation",   "source": "FCC",               "source_url": "https://www.fcc.gov"},
    {"url": "https://www.constructiondive.com/topic/telecom/", "category": "Construction", "source": "Construction Dive", "source_url": "https://www.constructiondive.com"},
]

MAX_ARTICLES_PER_SOURCE = 3
CACHE_TTL_HOURS         = 6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

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

# ── Helpers ────────────────────────────────────────────────────────────────

def _fetch_html(url: str, timeout: int = 10) -> Optional[str]:
    """GET a URL and return the HTML text, or None on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning(f"Fetch failed ({url}): {e}")
        return None


def _extract_article_links(html: str, base_domain: str, listing_url: str) -> List[str]:
    """
    Pull article-looking hrefs from a listing page.
    Keeps same-domain links that are longer than the listing URL (i.e. actual articles).
    """
    soup  = BeautifulSoup(html, "lxml")
    links = []
    seen  = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        # Normalise relative URLs
        if href.startswith("/"):
            href = base_domain.rstrip("/") + href
        if not href.startswith("http"):
            continue

        # Same domain, longer than the listing URL, not an anchor/query
        if (
            base_domain.split("//")[-1].split("/")[0] in href
            and len(href) > len(listing_url) + 5
            and "#" not in href
            and href not in seen
        ):
            seen.add(href)
            links.append(href)

    return links


def _parse_article(html: str) -> dict:
    """
    Extract title, first image, publish date, and body text from article HTML.
    Returns a dict with keys: title, image_url, published_at, text.
    """
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    for sel in ["h1", "meta[property='og:title']", "title"]:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get("content", "") or tag.get_text(strip=True)
            if title:
                break

    # Image (og:image first, then first <img> with src)
    image_url = None
    og_img = soup.select_one("meta[property='og:image']")
    if og_img:
        image_url = og_img.get("content")
    if not image_url:
        img = soup.select_one("article img[src], .post img[src], main img[src]")
        if img:
            image_url = img.get("src")
    # Filter out tiny icons / SVG data URIs
    if image_url and (image_url.startswith("data:") or len(image_url) < 12):
        image_url = None

    # Published date
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

    # Body text — prefer <article>, fall back to <main>, then <body>
    for sel in ["article", "main", "body"]:
        container = soup.select_one(sel)
        if container:
            # Remove noise
            for noise in container.select("nav, footer, aside, script, style, [class*='ad'], [class*='sidebar']"):
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
        "text":         text[:4000],  # trim for LLM context
    }

# ── LangGraph nodes ────────────────────────────────────────────────────────

def scrape_sources(state: AgentState) -> AgentState:
    """Node 1 — scrape listing pages and individual articles."""
    raw_articles: List[RawArticle] = []
    errors: List[str] = []

    for source in NEWS_SOURCES:
        html = _fetch_html(source["url"])
        if not html:
            errors.append(f"Source unreachable: {source['url']}")
            continue

        base_domain = "/".join(source["url"].split("/")[:3])  # https://domain.com
        links = _extract_article_links(html, base_domain, source["url"])[:MAX_ARTICLES_PER_SOURCE]

        if not links:
            errors.append(f"No article links found on {source['url']}")

        for link in links:
            art_html = _fetch_html(link)
            if not art_html:
                errors.append(f"Article unreachable: {link}")
                continue

            parsed = _parse_article(art_html)
            title  = parsed["title"] or link.split("/")[-1].replace("-", " ").title()

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

            time.sleep(0.4)  # polite crawl rate

    return {"raw_articles": raw_articles, "errors": errors, "status": "scraped"}


def enrich_articles(state: AgentState) -> AgentState:
    """Node 2 — use the shared llm_client (Groq/CF) to summarise + analyse each article."""
    enriched: List[EnrichedArticle] = []
    errors: List[str] = []

    try:
        from llm_client import call_llm, check_llm_available
        if not check_llm_available():
            errors.append("No LLM provider available (set GROQ_API_KEY or CF_API_KEY) — skipping enrichment")
            return {"enriched_articles": enriched, "errors": errors, "status": "enriched"}
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

_graph = _build_graph()

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