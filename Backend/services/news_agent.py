"""
news_agent.py — Bimlo Industry Analyst Agent
=============================================

An agentic news scraper that:
  1. Visits major telecom / construction news sources
  2. Extracts article titles, summaries, and links
  3. Uses call_llm() to generate a professional impact summary for each article
  4. Returns a structured briefing ready for the frontend

Usage
-----
  from news_agent import run_news_agent
  briefing = run_news_agent()   # list[NewsItem]

  Or run standalone:
  python news_agent.py

Schedule
--------
  Run via cron / APScheduler every morning and cache results in a DB or
  flat JSON file. The FastAPI endpoint in main.py can serve the cached data.

Firecrawl upgrade path
----------------------
  Replace _scrape_with_requests() with:
    from firecrawl import FirecrawlApp
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    result = app.scrape_url(url, params={"formats": ["markdown"]})
    content = result.get("markdown", "")
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from llm_client import call_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    id: str
    title: str
    source: str
    source_url: str
    article_url: str
    raw_summary: str          # scraped excerpt
    ai_impact: str            # LLM-generated impact analysis
    category: str             # "5G" | "Fiber" | "Regulation" | "Construction" | "General"
    published_at: str         # ISO-8601
    scraped_at: str           # ISO-8601


# ---------------------------------------------------------------------------
# News sources
# Extend this list freely — just add a dict with the keys below.
# ---------------------------------------------------------------------------

NEWS_SOURCES = [
    {
        "name": "RCR Wireless",
        "url": "https://www.rcrwireless.com/",
        "article_selector": "h3.entry-title a, h2.entry-title a",
        "summary_selector": ".entry-summary p, .excerpt p",
        "category": "5G",
    },
    {
        "name": "Fierce Telecom",
        "url": "https://www.fiercetelecom.com/",
        "article_selector": "h3.title a, h2.title a, .node__title a",
        "summary_selector": ".field--name-body p, .teaser p",
        "category": "General",
    },
    {
        "name": "Light Reading",
        "url": "https://www.lightreading.com/",
        "article_selector": "h2 a, h3 a",
        "summary_selector": "p.summary, .article-teaser p",
        "category": "Fiber",
    },
    {
        "name": "SDxCentral",
        "url": "https://www.sdxcentral.com/",
        "article_selector": "h2.entry-title a, h3.entry-title a",
        "summary_selector": ".entry-excerpt p",
        "category": "General",
    },
    {
        "name": "Telecom Ramblings",
        "url": "https://www.telecomramblings.com/",
        "article_selector": "h2.post-title a, h1.entry-title a",
        "summary_selector": ".entry-content p:first-of-type",
        "category": "Fiber",
    },
]

# How many articles to process per source (keep costs/latency manageable)
MAX_ARTICLES_PER_SOURCE = 3

# Cache file path (swap for Redis/DB in production)
CACHE_PATH = os.getenv("NEWS_CACHE_PATH", "news_cache.json")
CACHE_TTL_SECONDS = 6 * 60 * 60   # 6 hours


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BimloNewsBot/1.0; +https://bimlo.ai/bot)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _scrape_with_requests(url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
    """Fetch a page and return a BeautifulSoup object, or None on failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return None


def _make_absolute(url: str, base: str) -> str:
    if url.startswith("http"):
        return url
    from urllib.parse import urljoin
    return urljoin(base, url)


def _article_id(title: str, source: str) -> str:
    return hashlib.md5(f"{source}:{title}".encode()).hexdigest()[:12]


def _categorize(title: str, source_category: str) -> str:
    title_l = title.lower()
    if any(k in title_l for k in ["5g", "spectrum", "wireless", "mmwave", "ran"]):
        return "5G"
    if any(k in title_l for k in ["fiber", "fibre", "ftth", "fttx", "optical", "cable"]):
        return "Fiber"
    if any(k in title_l for k in ["regulat", "fcc", "policy", "law", "compliance"]):
        return "Regulation"
    if any(k in title_l for k in ["construct", "deploy", "build", "infrastructure", "tower"]):
        return "Construction"
    return source_category


# ---------------------------------------------------------------------------
# LLM impact analysis
# ---------------------------------------------------------------------------

_IMPACT_SYSTEM = (
    "You are Bimlo's Industry Analyst — a telecom expert who writes concise, "
    "actionable impact briefs for network engineers and project managers. "
    "Focus on practical implications for fiber/5G deployments and construction."
)


def _generate_impact(title: str, excerpt: str) -> str:
    """Ask the LLM to produce a 2-sentence impact summary."""
    prompt = (
        f"Article: {title}\n"
        f"Excerpt: {excerpt or '(no excerpt available)'}\n\n"
        "Write exactly 2 sentences: (1) what happened, (2) the practical impact "
        "on telecom infrastructure professionals. Be specific and avoid filler words."
    )
    result = call_llm(
        prompt=prompt,
        system_prompt=_IMPACT_SYSTEM,
        max_tokens=120,
        temperature=0.3,
        task="summarise",
    )
    return result.strip() if result else "Impact analysis unavailable."


# ---------------------------------------------------------------------------
# Per-source scrape
# ---------------------------------------------------------------------------

def _scrape_source(source: dict) -> List[NewsItem]:
    items: List[NewsItem] = []
    soup = _scrape_with_requests(source["url"])
    if soup is None:
        return items

    anchors = soup.select(source["article_selector"])[:MAX_ARTICLES_PER_SOURCE]
    now = datetime.now(timezone.utc).isoformat()

    for a in anchors:
        title = a.get_text(strip=True)
        href  = _make_absolute(a.get("href", ""), source["url"])

        if not title or not href:
            continue

        # Try to get a raw excerpt from the listing page
        parent  = a.find_parent(["article", "div", "li"])
        excerpt = ""
        if parent:
            for sel in source.get("summary_selector", "").split(", "):
                node = parent.select_one(sel.strip())
                if node:
                    excerpt = node.get_text(strip=True)[:400]
                    break

        # Generate AI impact (with a small delay to be polite to the LLM)
        ai_impact = _generate_impact(title, excerpt)
        time.sleep(0.4)

        items.append(NewsItem(
            id           = _article_id(title, source["name"]),
            title        = title,
            source       = source["name"],
            source_url   = source["url"],
            article_url  = href,
            raw_summary  = excerpt,
            ai_impact    = ai_impact,
            category     = _categorize(title, source["category"]),
            published_at = now,   # replace with parsed date if source exposes it
            scraped_at   = now,
        ))

    logger.info(f"[{source['name']}] scraped {len(items)} articles")
    return items


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> Optional[List[dict]]:
    try:
        with open(CACHE_PATH) as f:
            data = json.load(f)
        age = time.time() - data.get("ts", 0)
        if age < CACHE_TTL_SECONDS:
            return data["items"]
    except Exception:
        pass
    return None


def _save_cache(items: List[NewsItem]) -> None:
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({"ts": time.time(), "items": [asdict(i) for i in items]}, f)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_news_agent(force_refresh: bool = False) -> List[NewsItem]:
    """
    Main entry point. Returns a list of NewsItem objects.

    Args:
        force_refresh: Skip cache and re-scrape even if cache is fresh.

    Returns:
        List of NewsItem, sorted newest-scraped first.
    """
    if not force_refresh:
        cached = _load_cache()
        if cached:
            logger.info(f"Returning {len(cached)} cached news items")
            return [NewsItem(**d) for d in cached]

    logger.info("Starting Bimlo Industry Analyst agent…")
    all_items: List[NewsItem] = []

    for source in NEWS_SOURCES:
        try:
            items = _scrape_source(source)
            all_items.extend(items)
        except Exception as e:
            logger.error(f"Source {source['name']} failed: {e}")

    # Deduplicate by id
    seen: set[str] = set()
    unique: List[NewsItem] = []
    for item in all_items:
        if item.id not in seen:
            seen.add(item.id)
            unique.append(item)

    unique.sort(key=lambda x: x.scraped_at, reverse=True)
    _save_cache(unique)
    logger.info(f"Agent complete — {len(unique)} articles collected")
    return unique


# ---------------------------------------------------------------------------
# FastAPI route helper (import in main.py)
# ---------------------------------------------------------------------------

def get_news_briefing(force: bool = False) -> dict:
    """
    Convenience wrapper for the FastAPI route.
    Returns JSON-serialisable dict with metadata + items.
    """
    items = run_news_agent(force_refresh=force)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": [asdict(i) for i in items],
    }


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    briefing = get_news_briefing(force=True)
    print(f"\n✅ {briefing['count']} articles collected at {briefing['generated_at']}\n")
    for item in briefing["items"][:5]:
        print(f"  [{item['category']}] {item['title']}")
        print(f"  → {item['ai_impact']}")
        print(f"  {item['article_url']}\n")
