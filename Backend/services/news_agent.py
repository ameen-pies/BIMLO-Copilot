"""
Industry Analyst Agent — news_agent.py
────────────────────────────────────────────────────────────────
Pipeline (streaming):
  1. search_news  — NewsData.io API (primary) + Google News RSS (fallback)
  2. judge+enrich — keyword scoring per article (concurrent, zero LLM)
  3. stream_briefing — generator that yields articles one-by-one as
                       they finish, so the frontend can render immediately

Why NewsData.io instead of Google News RSS:
  • Real publisher URLs  → thumbnails and article text work first try
  • image_url field      → no scraping, no og:image proxy needed
  • content field        → full article body in the API response itself
  • when filter          → built-in recency (last 2 days)
  • Free tier            → 200 credits/day (plenty for a 4-day pipeline)

Google News RSS is kept as a zero-cost fallback for when the daily
NewsData quota is exhausted or the env var is not set.

Required env vars (NewsData):
  NEWSDATA_API_KEY  — get free key at https://newsdata.io (200 req/day)

Google News RSS needs no API key and still works as fallback.

Public API (used by main.py / news_pipeline.py):
    from news_agent import (
        stream_briefing, get_cached_briefing, get_next_page,
        fetch_raw_articles, PAGE_0_QUERIES, EXTENDED_QUERY_PAGES,
        _judge_and_enrich_one,
    )
"""

from __future__ import annotations

import re
import json
import hashlib
import logging
import threading
import os
import urllib.parse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("news_agent")

# ── NewsData.io config ──────────────────────────────────────────────────────────

_NEWSDATA_API_KEY  = os.getenv("NEWSDATA_API_KEY", "")
_NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"   # /latest endpoint = breaking news

# ── Query buckets ──────────────────────────────────────────────────────────────

PAGE_0_QUERIES = [
    {"query": "5G network deployment",               "category": "5G"},
    {"query": "5G infrastructure rollout",           "category": "5G"},
    {"query": "fiber broadband expansion",           "category": "Fiber"},
    {"query": "rural broadband internet access",     "category": "Fiber"},
    {"query": "FCC telecom regulation",              "category": "Regulation"},
    {"query": "spectrum policy broadband",           "category": "Regulation"},
    {"query": "telecom tower construction",          "category": "Construction"},
    {"query": "broadband infrastructure build",      "category": "Construction"},
    {"query": "telecom carrier ISP news",            "category": "General"},
    {"query": "telecom industry results",            "category": "General"},
    {"query": "BIM building information modeling",           "category": "BIM"},
    {"query": "scan to BIM digital twin construction",       "category": "BIM"},
    {"query": "digital twin infrastructure",                 "category": "Digital Twin"},
    {"query": "predictive maintenance digital twin AI",      "category": "Digital Twin"},
    {"query": "artificial intelligence construction",        "category": "AI Construction"},
    {"query": "AI building infrastructure optimization",     "category": "AI Construction"},
]

EXTENDED_QUERY_PAGES = [
    [
        {"query": "5G mmWave private network",           "category": "5G"},
        {"query": "open RAN vRAN deployment",            "category": "5G"},
        {"query": "FTTH FTTB fiber to the home",         "category": "Fiber"},
        {"query": "submarine cable internet",            "category": "Fiber"},
        {"query": "net neutrality FCC ruling",           "category": "Regulation"},
        {"query": "telecom merger acquisition",          "category": "Regulation"},
        {"query": "cell tower zoning approval",          "category": "Construction"},
        {"query": "data center power infrastructure",    "category": "Construction"},
        {"query": "Verizon AT&T T-Mobile news",          "category": "General"},
        {"query": "satellite internet Starlink LEO",     "category": "General"},
        {"query": "BIM 4D 5D project planning",          "category": "BIM"},
        {"query": "Revit IFC BIM interoperability",      "category": "BIM"},
        {"query": "digital twin smart city IoT",         "category": "Digital Twin"},
        {"query": "industry 4.0 digital twin factory",   "category": "Digital Twin"},
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
        {"query": "MVNO wholesale agreement",            "category": "General"},
        {"query": "telecom workforce layoff hiring",     "category": "General"},
        {"query": "BIM mandate government infrastructure","category": "BIM"},
        {"query": "point cloud scan to BIM software",   "category": "BIM"},
        {"query": "digital twin energy building",        "category": "Digital Twin"},
        {"query": "real-time digital twin simulation",   "category": "Digital Twin"},
        {"query": "machine learning site planning construction","category": "AI Construction"},
        {"query": "computer vision construction safety", "category": "AI Construction"},
    ],
    [
        {"query": "C-band 5G spectrum deployment",       "category": "5G"},
        {"query": "millimeter wave 5G enterprise",       "category": "5G"},
        {"query": "ISP fixed wireless broadband rural",  "category": "Fiber"},
        {"query": "middle mile broadband grant",         "category": "Fiber"},
        {"query": "telecom antitrust DOJ FTC",           "category": "Regulation"},
        {"query": "universal service fund USF reform",   "category": "Regulation"},
        {"query": "utility pole attachment rate",        "category": "Construction"},
        {"query": "fiber aerial underground cost",       "category": "Construction"},
        {"query": "wholesale bandwidth pricing trend",   "category": "General"},
        {"query": "IoT connectivity smart city",         "category": "General"},
        {"query": "BIM digital construction CDE",        "category": "BIM"},
        {"query": "BIM coordination clash detection MEP","category": "BIM"},
        {"query": "digital twin infrastructure lifecycle","category": "Digital Twin"},
        {"query": "AEC digital twin deployment case",    "category": "Digital Twin"},
        {"query": "AI predictive maintenance building",  "category": "AI Construction"},
        {"query": "robotics automation construction site","category": "AI Construction"},
    ],
]

MAX_RESULTS_PER_QUERY = 15
CACHE_TTL_HOURS       = 4
JUDGE_SCORE_THRESHOLD = 0.28
LLM_WORKERS           = 8


# ── TypedDicts ─────────────────────────────────────────────────────────────────

class RawArticle(TypedDict):
    title:        str
    url:          str          # real publisher URL (never a news.google.com redirect)
    image_url:    Optional[str]
    raw_text:     str          # description / snippet
    full_content: str          # full article body when available (from NewsData)
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
    full_content: str          # propagated so chat agent can use it directly
    ai_impact:    str
    category:     str
    published_at: str
    scraped_at:   str
    enriched:     bool


# ── Global dedup state ─────────────────────────────────────────────────────────

_seen_urls:         set = set()
_seen_fingerprints: set = set()
_dedup_lock = threading.Lock()


def _fingerprint(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:60]


def _is_dupe(url: str, title: str) -> bool:
    fp = _fingerprint(title)
    with _dedup_lock:
        return url in _seen_urls or fp in _seen_fingerprints


def _register(url: str, title: str):
    fp = _fingerprint(title)
    with _dedup_lock:
        _seen_urls.add(url)
        _seen_fingerprints.add(fp)


def reset_dedup():
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


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY: NewsData.io API
# ══════════════════════════════════════════════════════════════════════════════

def _newsdata_available() -> bool:
    return bool(_NEWSDATA_API_KEY)


def _parse_newsdata_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD HH:MM:SS' to ISO 8601."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.isoformat()
    except ValueError:
        return date_str


def _newsdata_search(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """
    Fetch articles from NewsData.io /latest endpoint.

    Returns real publisher URLs, image_url straight from the API, and
    full article content (when the publisher allows it) — no scraping.

    API docs: https://newsdata.io/documentation
    Free tier: 200 credits/day, 10 results per request.
    """
    if not _newsdata_available():
        return []

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — pip install requests")
        return []

    params = {
        "apikey":   _NEWSDATA_API_KEY,
        "q":        query_entry["query"],
        "language": "en",
        "size":     10,                    # max per request on free plan
    }

    try:
        resp = requests.get(_NEWSDATA_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"[NewsData] request failed for '{query_entry['query']}': {e}")
        return []

    if data.get("status") != "success":
        logger.warning(f"[NewsData] non-success status: {data.get('status')} — {data.get('results', {})}")
        return []

    articles: List[RawArticle] = []
    for item in (data.get("results") or []):
        url = (item.get("link") or "").strip()
        if not url or url in seen_local:
            continue

        title = (item.get("title") or "")[:200].strip()
        if not title:
            continue

        seen_local.add(url)

        # description is a clean snippet; content is the full body (paywalled sites return "")
        description = re.sub(r"<[^>]+>", " ", item.get("description") or "").strip()
        content_raw = re.sub(r"<[^>]+>", " ", item.get("content")     or "").strip()
        content_raw = re.sub(r"\s{2,}", " ", content_raw)

        # Use content if available and substantial, else fall back to description
        raw_text     = description[:4000]
        full_content = content_raw[:16000] if len(content_raw) > len(description) else description[:4000]

        # image_url comes directly from the API — no scraping needed
        image_url = (item.get("image_url") or "").strip() or None
        if image_url and image_url.startswith("data:"):
            image_url = None

        # published date
        published_at = _parse_newsdata_date(item.get("pubDate") or "")

        # source
        source_id   = item.get("source_id") or ""
        source_name = item.get("source_name") or source_id or _domain_from_url(url)
        source_url  = _origin_from_url(url)

        articles.append({
            "title":        title,
            "url":          url,
            "image_url":    image_url,
            "raw_text":     raw_text,
            "full_content": full_content,
            "category":     query_entry["category"],
            "source":       source_name,
            "source_url":   source_url,
            "published_at": published_at,
        })

    logger.info(f"  [NewsData] '{query_entry['query']}' → {len(articles)} articles")
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK: Google News RSS  (kept for quota exhaustion / no API key)
# ══════════════════════════════════════════════════════════════════════════════

_GNEWS_RSS_BASE = "https://news.google.com/rss/search"

_CATEGORY_RSS: dict = {
    "5G": [
        "https://www.rcrwireless.com/feed",
        "https://www.fiercewireless.com/rss/xml",
    ],
    "Fiber": [
        "https://www.fiercetelecom.com/rss/xml",
        "https://www.telecompetitor.com/feed/",
    ],
    "Regulation": [
        "https://www.fiercetelecom.com/rss/xml",
        "https://www.rcrwireless.com/feed",
    ],
    "Construction": [
        "https://www.constructiondive.com/feeds/news/",
        "https://www.enr.com/rss/news",
    ],
    "General": [
        "https://www.rcrwireless.com/feed",
        "https://www.fiercetelecom.com/rss/xml",
    ],
    "BIM": [
        "https://www.autodesk.com/blogs/aec/feed/",
        "https://www.aecmag.com/feed",
    ],
    "Digital Twin": [
        "https://www.iotforall.com/feed",
    ],
    "AI Construction": [
        "https://www.constructiondive.com/feeds/news/",
        "https://www.enr.com/rss/news",
    ],
}


def _domain_from_url(url: str) -> str:
    try:
        parts = url.split("/")
        return parts[2] if len(parts) >= 3 else url
    except Exception:
        return url


def _origin_from_url(url: str) -> str:
    try:
        parts = url.split("/")
        return "/".join(parts[:3]) if len(parts) >= 3 else url
    except Exception:
        return url


def _fetch_rss(url: str, timeout: int = 10) -> list:
    try:
        import feedparser
        feed = feedparser.parse(url, request_headers={"User-Agent": "BimloRAG/1.0"})
        return feed.entries or []
    except Exception as e:
        logger.warning(f"RSS fetch failed for {url}: {e}")
        return []


def _extract_rss_image(entry) -> Optional[str]:
    """Pull thumbnail from feedparser entry using multiple strategies."""
    def _ok(url: str) -> bool:
        if not url or url.startswith("data:"):
            return False
        if re.search(r'[?&][wh]=\d{1,2}\b', url):
            return False
        return True

    for attr in ("media_thumbnail", "media_content"):
        media = getattr(entry, attr, None) or entry.get(attr)
        if media and isinstance(media, list):
            for m in media:
                u = m.get("url", "")
                if _ok(u):
                    return u

    for enc in getattr(entry, "enclosures", []) or entry.get("enclosures", []):
        if enc.get("type", "").startswith("image/") and _ok(enc.get("href", "")):
            return enc["href"]

    img_pat = re.compile(
        r'<img[^>]+(?:src|data-src)[^>]*=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    html_fields: list = []
    if entry.get("summary"):
        html_fields.append(entry.summary)
    for c in (entry.get("content") or []):
        v = c.get("value", "") if isinstance(c, dict) else ""
        if v:
            html_fields.append(v)

    for html in html_fields:
        for m in img_pat.finditer(html):
            candidate = m.group(1).strip()
            if _ok(candidate):
                return candidate

    return None


# ── Google News redirect resolver ──────────────────────────────────────────────
# When RSS URLs are still news.google.com/rss/articles/CBMi... links,
# resolve them to the real publisher URL before storing.

_url_resolve_cache: dict = {}
_url_resolve_lock = threading.Lock()


def _resolve_gnews_url(url: str) -> str:
    """
    Resolve a Google News redirect URL to the real publisher URL.
    Uses gnewsdecoder first (offline base64 decode), falls back to HTTP follow.
    Result is cached in-process.
    """
    if "news.google.com" not in url:
        return url

    with _url_resolve_lock:
        if url in _url_resolve_cache:
            return _url_resolve_cache[url]

    resolved = url
    try:
        # gnewsdecoder decodes the base64 payload without any HTTP request
        from gnewsdecoder import gnewsdecoder
        result = gnewsdecoder(url)
        if result and result.get("status") == "OK":
            real = result.get("decoded_url", "")
            if real and "google.com" not in real:
                resolved = real
                logger.debug(f"[resolve] gnewsdecoder: {url[:50]} → {resolved[:60]}")
    except Exception:
        pass

    # HTTP-follow fallback
    if resolved == url:
        try:
            import requests
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
                allow_redirects=True,
            )
            if "google.com" not in r.url:
                resolved = r.url
                logger.debug(f"[resolve] HTTP follow: {url[:50]} → {resolved[:60]}")
        except Exception:
            pass

    with _url_resolve_lock:
        _url_resolve_cache[url] = resolved

    return resolved


def _entry_to_raw(entry, query_entry: dict, seen_local: set) -> Optional[RawArticle]:
    """Normalise a feedparser entry into a RawArticle dict with a real URL."""
    url_raw = entry.get("link", "")
    if not url_raw or url_raw in seen_local:
        return None

    # Resolve Google News redirect → real publisher URL
    url = _resolve_gnews_url(url_raw)
    if url in seen_local:
        return None
    seen_local.add(url)
    if url != url_raw:
        seen_local.add(url_raw)  # also mark the redirect as seen

    published_at = ""
    if entry.get("published_parsed"):
        try:
            published_at = datetime.utcfromtimestamp(
                calendar.timegm(entry.published_parsed)
            ).isoformat()
        except Exception:
            pass

    entry_source    = entry.get("source") or {}
    gnews_src_href  = entry_source.get("href", "")
    source_url      = gnews_src_href.rstrip("/") if gnews_src_href else _origin_from_url(url)
    source          = entry_source.get("title") or _domain_from_url(url)
    title           = (entry.get("title") or "")[:200]

    body = ""
    if entry.get("summary"):
        body = entry.summary
    elif entry.get("content"):
        body = entry.content[0].get("value", "")
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s{2,}", " ", body).strip()

    if not title:
        return None

    # Try RSS-embedded image (fast, no HTTP request)
    image_url = _extract_rss_image(entry)

    # If no RSS image, try og:image from the REAL (resolved) URL — not the redirect
    if not image_url and "news.google.com" not in url:
        image_url = _fetch_og_image(url, source_domain=source_url)

    return {
        "title":        title,
        "url":          url,
        "image_url":    image_url,
        "raw_text":     body[:4000],
        "full_content": body[:4000],   # RSS rarely has full body; content is the best we have
        "category":     query_entry["category"],
        "source":       source,
        "source_url":   source_url,
        "published_at": published_at,
    }


# ── og:image fetcher (used by RSS fallback only) ───────────────────────────────

_og_cache:      dict = {}
_og_cache_lock = threading.Lock()

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


def _fetch_og_image(article_url: str, source_domain: str = "", timeout: int = 6) -> Optional[str]:
    """
    Extract og:image / twitter:image from a real publisher URL (never a Google redirect).
    Only called by the RSS fallback path — NewsData.io provides image_url directly.
    """
    if not article_url or article_url.startswith("#") or "news.google.com" in article_url:
        return None

    with _og_cache_lock:
        if article_url in _og_cache:
            return _og_cache[article_url]

    def _is_plausible(url: str) -> bool:
        if not url or url.startswith("data:"):
            return False
        low = url.lower()
        skip = ("sprite", "avatar", "pixel", "tracker", "beacon",
                "1x1", "spacer", "ad.", "doubleclick", "googletagmanager",
                "analytics", "stat.", "counter.")
        if any(p in low for p in skip):
            return False
        if re.search(r'\.(jpg|jpeg|png|webp|gif|avif)(\?|$|#)', low):
            return True
        if re.search(r'(images?|media|photos?|cdn|assets|thumbs?|thumb)[./]', low):
            return True
        if "lh3.googleusercontent.com" in low:
            return True
        return False

    result: Optional[str] = None
    try:
        import requests
        html = ""
        with requests.get(
            article_url,
            headers=_BROWSER_HEADERS,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=16_384, decode_unicode=True):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="ignore")
                html += chunk
                if ("</head>" in html.lower() and len(html) > 4_000) or len(html) > 48_000:
                    break

        meta_patterns = [
            r'<meta[^>]+property=["\']og:image(?::url)?["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::url)?["\']',
            r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']',
        ]
        for pat in meta_patterns:
            m = re.search(pat, html, re.IGNORECASE)
            if m and _is_plausible(m.group(1)):
                result = m.group(1).strip()
                break

        if not result:
            for pat in (
                r'<meta[^>]+itemprop=["\']image["\'][^>]+content=["\']([^"\']+)["\']',
                r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
            ):
                m = re.search(pat, html, re.IGNORECASE)
                if m and _is_plausible(m.group(1)):
                    result = m.group(1).strip()
                    break

        if not result:
            body_html = html
            body_m = re.search(
                r'<(?:article|main|div[^>]+(?:class|id)=["\'][^"\']*(?:article|story|content|post|entry|body)[^"\']*["\'])[^>]*>(.*)',
                html, re.IGNORECASE | re.DOTALL,
            )
            if body_m:
                body_html = body_m.group(1)
            img_pat = re.compile(
                r'<img[^>]+(?:src|data-src|data-lazy-src|data-original)=["\']([^"\']+)["\']',
                re.IGNORECASE,
            )
            for img_m in img_pat.finditer(body_html):
                candidate = img_m.group(1).strip()
                if candidate.startswith("//"):
                    candidate = "https:" + candidate
                elif candidate.startswith("/"):
                    origin_m = re.match(r'(https?://[^/]+)', article_url)
                    if origin_m:
                        candidate = origin_m.group(1) + candidate
                if _is_plausible(candidate):
                    result = candidate
                    break

    except Exception as e:
        logger.debug(f"og:image fetch failed for {article_url[:60]}: {e}")

    with _og_cache_lock:
        _og_cache[article_url] = result

    return result


def _search_gnews_rss(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """Google News RSS with 4-day recency filter — fallback when NewsData is unavailable."""
    q_with_recency = query_entry["query"] + " when:4d"
    params = urllib.parse.urlencode({
        "q":    q_with_recency,
        "hl":   "en-US",
        "gl":   "US",
        "ceid": "US:en",
    })
    url      = f"{_GNEWS_RSS_BASE}?{params}"
    entries  = _fetch_rss(url)
    articles = []
    for e in entries[:MAX_RESULTS_PER_QUERY]:
        art = _entry_to_raw(e, query_entry, seen_local)
        if art:
            articles.append(art)
    return articles


def _search_category_rss(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """Trade publication RSS feeds — last-resort fallback."""
    cat      = query_entry.get("category", "General")
    feeds    = _CATEGORY_RSS.get(cat, _CATEGORY_RSS["General"])
    keywords = query_entry["query"].lower().split()
    articles = []
    for feed_url in feeds:
        if len(articles) >= MAX_RESULTS_PER_QUERY:
            break
        for e in _fetch_rss(feed_url):
            if len(articles) >= MAX_RESULTS_PER_QUERY:
                break
            title_l   = (e.get("title") or "").lower()
            summary_l = (e.get("summary") or "").lower()
            if any(kw in title_l or kw in summary_l for kw in keywords):
                art = _entry_to_raw(e, query_entry, seen_local)
                if art:
                    articles.append(art)
    return articles


def _search_one_query(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """
    Search order:
      1. NewsData.io API      — real URLs, real thumbnails, real content (primary)
      2. Google News RSS      — real URLs after redirect resolution (fallback)
      3. Category trade feeds — keyword-filtered RSS (last resort)
    """
    # 1. NewsData.io (primary)
    if _newsdata_available():
        articles = _newsdata_search(query_entry, seen_local)
        if articles:
            return articles
        logger.info(f"  NewsData empty for '{query_entry['query']}' — trying Google News RSS")

    # 2. Google News RSS (fallback)
    articles = _search_gnews_rss(query_entry, seen_local)
    if articles:
        logger.info(f"  [RSS] '{query_entry['query']}' → {len(articles)} articles")
        return articles

    # 3. Trade feeds (last resort)
    logger.info(f"  GNews empty for '{query_entry['query']}' — trying category feeds")
    articles = _search_category_rss(query_entry, seen_local)
    logger.info(f"  [cat-RSS] '{query_entry['query']}' → {len(articles)} articles")
    return articles


# ── fetch_raw_articles ─────────────────────────────────────────────────────────

def fetch_raw_articles(queries: List[dict]) -> List[RawArticle]:
    """
    Run all queries concurrently and return de-duped raw articles.
    NewsData.io is used when NEWSDATA_API_KEY is set; RSS otherwise.
    No date-rotation suffix added — NewsData /latest handles recency natively.
    """
    all_articles: List[RawArticle] = []
    global_seen: set = set()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_search_one_query, entry, set()): entry
            for entry in queries
        }
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception as e:
                logger.warning(f"Query worker error: {e}")
                continue
            for art in results:
                if art["url"] not in global_seen:
                    global_seen.add(art["url"])
                    all_articles.append(art)

    backend = "NewsData.io" if _newsdata_available() else "RSS"
    logger.info(f"fetch_raw_articles: {len(all_articles)} total unique articles ({backend} backend)")
    return all_articles


# ── Keyword judge + enrich (zero LLM calls) ────────────────────────────────────

_RELEVANCE_KEYWORDS: list = [
    "telecom", "5g", "fiber", "fibre", "broadband", "spectrum", "network",
    "wireless", "isp", "carrier", "operator", "antenna", "cell tower",
    "base station", "radio", "fcc", "ofcom",
    "bim", "digital twin", "construction", "building", "infrastructure",
    "scan to bim", "point cloud", "revit", "ifc", "aec", "structural",
    "artificial intelligence", "machine learning", "predictive", "automation",
    "iot", "smart", "digital", "ai ",
    "regulation", "regulatory", "merger", "acquisition", "subsidy",
    "broadband subsidy", "bead", "deployment", "rollout", "expansion",
]

_CATEGORY_KEYWORDS: dict = {
    "5G": [
        "5g", "mmwave", "millimeter wave", "c-band", "ran ", "oran", "open ran",
        "vran", "small cell", "base station", "network slicing", "private network",
        "massive mimo", "beamforming", "6g", "5g standalone",
    ],
    "Fiber": [
        "fiber", "fibre", "fttp", "fttb", "ftth", "optical", "dark fiber",
        "submarine cable", "broadband", "gigabit", "isp ", "last mile",
        "fiber rollout", "fiber build",
    ],
    "Regulation": [
        "fcc ", "ofcom", "regulation", "regulatory", "spectrum auction",
        "net neutrality", "antitrust", "merger", "acquisition", "bead ",
        "usf ", "universal service", "permit", "zoning", "right of way",
        "spectrum policy", "telecom law",
    ],
    "BIM": [
        "bim ", "building information model", "revit", "ifc ", "archicad",
        "scan to bim", "point cloud", "lidar", "4d bim", "5d bim",
        "clash detection", "mep coordination", "cde ", "bim mandate",
    ],
    "Digital Twin": [
        "digital twin", "predictive maintenance", "smart building",
        "iot sensor", "facility management", "real-time monitoring",
        "smart grid", "smart city", "twin model", "infrastructure twin",
    ],
    "AI Construction": [
        "ai construction", "construction ai", "generative ai", "machine learning site",
        "computer vision construction", "robotics construction",
        "ai structural", "ai safety construction", "contech", "proptech",
    ],
    "Construction": [
        "construction", "building", "infrastructure", "contractor",
        "civil engineering", "structural", "aec ", "architecture",
        "data center construction", "jobsite",
    ],
}

_IMPACT_TEMPLATES: dict = {
    "5G":             "5G deployment and spectrum developments shape telecom infrastructure planning and tower demand.",
    "Fiber":          "Fiber rollout and broadband funding unlock new cabling, ISP, and last-mile infrastructure opportunities.",
    "Regulation":     "Regulatory shifts affect spectrum access, permitting timelines, and compliance requirements.",
    "BIM":            "BIM adoption advances digital construction workflows, coordination efficiency, and project delivery.",
    "Digital Twin":   "Digital twin technology enables predictive maintenance and smarter lifecycle management of infrastructure.",
    "AI Construction":"AI is accelerating safety monitoring, site automation, and design optimisation across the AEC sector.",
    "Construction":   "Construction sector trends influence project economics, digital adoption, and supply chain dynamics.",
    "General":        "Industry development with implications for telecom and construction professionals.",
}


def _kw_relevance_score(text: str) -> float:
    import math
    hits = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text)
    if hits == 0:
        return 0.0
    return min(1.0, math.sqrt(hits / len(_RELEVANCE_KEYWORDS)) * 3.5)


def _kw_detect_category(text: str, hint: str) -> str:
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return hint


def _judge_and_enrich_one(
    art: RawArticle, call_llm, idx: int
) -> tuple[Optional[EnrichedArticle], Optional[str]]:
    """
    Keyword-based judge + enrich — zero LLM calls.
    Propagates full_content so the chat agent can use it directly.
    """
    if not art.get("published_at"):
        logger.debug(f"  ⏭  NO DATE — {art['title'][:65]}")
        return None, None

    text  = f"{art['title']} {art['raw_text']}".lower()
    score = _kw_relevance_score(text)
    if score < JUDGE_SCORE_THRESHOLD:
        logger.info(f"  ❌ REJECT [{score:.2f}] {art['title'][:65]}")
        return None, None

    category = _kw_detect_category(text, art["category"])
    uid      = f"art_{idx}_{hashlib.md5(art['url'].encode()).hexdigest()[:6]}"

    item: EnrichedArticle = {
        "id":           uid,
        "title":        art["title"],
        "source":       art["source"],
        "source_url":   art["source_url"],
        "article_url":  art["url"],
        "image_url":    art.get("image_url"),
        "raw_summary":  art["raw_text"][:600],
        "full_content": art.get("full_content", art["raw_text"])[:16000],
        "ai_impact":    _IMPACT_TEMPLATES.get(category, _IMPACT_TEMPLATES["General"]),
        "category":     category,
        "published_at": art["published_at"],
        "scraped_at":   datetime.utcnow().isoformat(),
        "enriched":     True,
    }
    logger.info(f"  ✅ ACCEPT [{score:.2f}] {art['title'][:65]}")
    return item, None


# ── In-memory cache ────────────────────────────────────────────────────────────

_cache:      dict               = {}
_cache_time: Optional[datetime] = None
_cache_lock  = threading.Lock()

_next_page_idx: int = 0
_page_lock = threading.Lock()


# ── Core streaming pipeline ────────────────────────────────────────────────────

def stream_briefing(
    queries: List[dict],
    force_reset_dedup: bool = False,
) -> Iterator[EnrichedArticle]:
    if force_reset_dedup:
        reset_dedup()

    raw_articles = fetch_raw_articles(queries)
    fresh = [a for a in raw_articles if not _is_dupe(a["url"], a["title"])]
    logger.info(f"📋 {len(fresh)} fresh articles (deduped from {len(raw_articles)})")

    if not fresh:
        return

    new_items: List[EnrichedArticle] = []

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
        futures = {
            pool.submit(_judge_and_enrich_one, art, None, idx): art
            for idx, art in enumerate(fresh)
        }
        for future in as_completed(futures):
            try:
                item, _err = future.result()
            except Exception as e:
                logger.warning(f"Worker exception: {e}")
                continue
            if item is None:
                continue
            art = futures[future]
            _register(art["url"], art["title"])
            new_items.append(item)
            yield item

    _merge_into_cache(new_items)
    logger.info(f"✅ stream_briefing done — {len(new_items)} articles yielded & cached")


def _merge_into_cache(new_items: List[EnrichedArticle]):
    global _cache, _cache_time
    with _cache_lock:
        existing    = _cache.get("items", [])
        existing_ids = {i["id"] for i in existing}
        merged      = existing + [i for i in new_items if i["id"] not in existing_ids]
        _cache = {
            "generated_at": (_cache_time or datetime.utcnow()).isoformat(),
            "count":        len(merged),
            "items":        merged,
        }
        if _cache_time is None:
            _cache_time = datetime.utcnow()


def get_cached_briefing() -> dict:
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


def get_news_briefing(force: bool = False) -> dict:
    if cache_is_valid() and not force:
        logger.info("📰 Returning cached briefing")
        return get_cached_briefing()
    if force:
        invalidate_cache()
    for _ in stream_briefing(PAGE_0_QUERIES, force_reset_dedup=force):
        pass
    return get_cached_briefing()