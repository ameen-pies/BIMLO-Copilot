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
import hashlib
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

MAX_RESULTS_PER_QUERY = 15   # fetch more per query for greater variety
CACHE_TTL_HOURS       = 4    # expire sooner so refreshes feel fresh
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


# ── RSS search backend (replaces SearXNG / DDG) ────────────────────────────────
#
# Google News RSS works from any IP — no API key, no datacenter blocks.
# Category-specific trade feeds are used as fallback when GNews is empty.

import urllib.parse

_GNEWS_RSS_BASE = "https://news.google.com/rss/search"


def _decode_gnews_url(gnews_url: str) -> str:
    """
    Decode a news.google.com/rss/articles/... redirect URL to the real
    publisher article URL — WITHOUT making any HTTP request.

    Google News encodes the destination URL as a base64url segment in the
    path. We decode it and extract the UTF-8 URL that starts at byte 4
    (after a 4-byte length/varint header). Falls back to the original URL
    if decoding fails for any reason.
    """
    if "news.google.com" not in gnews_url:
        return gnews_url
    try:
        import base64
        m = re.search(r'/articles/([A-Za-z0-9_-]+)', gnews_url)
        if not m:
            return gnews_url
        b64 = m.group(1)
        # Pad to a multiple of 4 for standard base64 decoding
        b64 += "=" * (-len(b64) % 4)
        data = base64.urlsafe_b64decode(b64)
        # Real URL is a null-terminated UTF-8 string starting at byte offset 4
        url = data[4:].decode("utf-8", errors="ignore").split("\x00")[0].strip()
        if url.startswith("http"):
            logger.debug(f"Decoded GNews URL: {url[:80]}")
            return url
    except Exception as e:
        logger.debug(f"GNews URL decode failed for {gnews_url[:60]}: {e}")
    return gnews_url

_CATEGORY_RSS: dict = {
    "5G": [
        "https://feeds.feedburner.com/TelecomRamblings",
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
        "https://www.digitaltwinconsortium.org/feed/",
    ],
    "AI Construction": [
        "https://www.constructiondive.com/feeds/news/",
        "https://www.enr.com/rss/news",
    ],
}


def _fetch_rss(url: str, timeout: int = 10) -> list:
    """Return a list of feedparser entry dicts, or [] on any error."""
    try:
        import feedparser
        feed = feedparser.parse(url, request_headers={"User-Agent": "BimloRAG/1.0"})
        return feed.entries or []
    except Exception as e:
        logger.warning(f"RSS fetch failed for {url}: {e}")
        return []


def _extract_rss_image(entry) -> Optional[str]:
    """
    Pull a thumbnail URL from a feedparser entry using every common RSS
    image convention (tried in order of reliability):
      1. <media:content> / <media:thumbnail>  — most trade feeds
      2. <enclosure>                           — podcast-style but used by some news sites
      3. <img> inside the HTML summary         — Google News lh3.googleusercontent.com thumbnails
      4. <img> inside content blocks           — full-content RSS feeds
    Returns None if nothing is found.
    """
    def _ok(url: str) -> bool:
        """Reject data URIs, empty strings, and sub-10px tracking pixels."""
        if not url or url.startswith("data:"):
            return False
        # Block only confirmed tiny pixels (w=1 or h=1 style params)
        if re.search(r'[?&][wh]=\d{1,2}\b', url):
            return False
        return True

    # 1. media:content / media:thumbnail
    for attr in ("media_thumbnail", "media_content"):
        media = getattr(entry, attr, None) or entry.get(attr)
        if media and isinstance(media, list):
            for m in media:
                u = m.get("url", "")
                if _ok(u):
                    return u

    # 2. <enclosure> image tags
    for enc in getattr(entry, "enclosures", []) or entry.get("enclosures", []):
        if enc.get("type", "").startswith("image/") and _ok(enc.get("href", "")):
            return enc["href"]

    # 3 & 4. <img> inside summary and content HTML blocks
    #   Google News embeds lh3.googleusercontent.com/proxy/... thumbnails in the
    #   description — these are real article thumbnails pre-cached by Google.
    img_pat = re.compile(
        r'<img[^>]+(?:src|data-src)[^>]*=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    html_fields: list[str] = []
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


# OG image in-memory cache so repeated pipeline runs don't re-fetch the same URLs
_og_cache: dict = {}
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
    Multi-strategy image extractor — tries several methods in order:
      1. og:image meta tag  (both attribute orders)
      2. twitter:image meta tag
      3. itemprop="image" / link[rel=image_src]
      4. First sizeable <img> in the article body (src / data-src / data-lazy-src)
      5. Clearbit logo for the source domain — guaranteed fallback, never None
         if a source_domain is provided.

    Uses full browser headers (including Sec-Fetch-*) to pass bot-detection
    on most news sites. Streams only up to 48 KB so we never download the page.
    Results are cached in-process to avoid re-fetching the same URL.
    """
    if not article_url or article_url.startswith("#"):
        return _clearbit_logo(source_domain)

    with _og_cache_lock:
        if article_url in _og_cache:
            cached = _og_cache[article_url]
            return cached if cached else _clearbit_logo(source_domain)

    def _is_plausible(url: str) -> bool:
        if not url or url.startswith("data:"):
            return False
        low = url.lower()
        skip = ("logo", "icon", "sprite", "avatar", "badge", "pixel",
                "tracker", "beacon", "1x1", "spacer", "ad.", "doubleclick",
                "googletagmanager", "analytics", "stat.", "counter.")
        if any(p in low for p in skip):
            return False
        # Accept if it has an image extension or a CDN/media path
        if re.search(r'\.(jpg|jpeg|png|webp|gif|avif)(\?|$|#)', low):
            return True
        if re.search(r'(images?|media|photos?|cdn|assets|thumbs?|thumb)[./]', low):
            return True
        # Google's lh3 proxy thumbnails — always legit
        if "lh3.googleusercontent.com" in low:
            return True
        return False

    result: Optional[str] = None

    # ── Strategy 0: trafilatura (best success rate — handles encoding/redirects) ─
    try:
        traf_result = _fetch_og_image_trafilatura(article_url)
        if traf_result and _is_plausible(traf_result):
            result = traf_result
    except Exception as e:
        logger.debug(f"trafilatura pass failed for {article_url[:60]}: {e}")

    # ── Strategies 1-4: manual HTML streaming (fallback if trafilatura misses) ───
    if not result:
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

            # Strategy 1 & 2: og:image / twitter:image (both attribute orderings)
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

            # Strategy 3: itemprop / link rel=image_src
            if not result:
                for pat in (
                    r'<meta[^>]+itemprop=["\']image["\'][^>]+content=["\']([^"\']+)["\']',
                    r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
                ):
                    m = re.search(pat, html, re.IGNORECASE)
                    if m and _is_plausible(m.group(1)):
                        result = m.group(1).strip()
                        break

            # Strategy 4: first sizeable <img> inside the article body
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

    if result:
        logger.debug(f"Image found for {article_url[:60]}: {result[:60]}")

    with _og_cache_lock:
        _og_cache[article_url] = result

    # Strategy 5: Clearbit logo — guaranteed non-None fallback
    return result if result else _clearbit_logo(source_domain)


def _fetch_og_image_trafilatura(url: str) -> Optional[str]:
    """
    Use trafilatura (if installed) to extract og:image / twitter:image /
    schema.org image from a real article URL. Much more robust than manual
    regex — handles encoding, redirects, and common anti-bot patterns.
    Returns None if trafilatura is not installed or extraction fails.
    """
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        meta = trafilatura.extract_metadata(downloaded)
        if meta and getattr(meta, "image", None):
            img = meta.image
            if img and not img.startswith("data:"):
                return img
    except ImportError:
        pass  # trafilatura not installed — fall through to regex method
    except Exception as e:
        logger.debug(f"trafilatura extraction failed for {url[:60]}: {e}")
    return None


def _clearbit_logo(domain: str) -> Optional[str]:
    """
    Return a Clearbit logo URL for the given domain.
    Free tier, no auth, reliably returns a clean branded image.
    Returns None if no domain is supplied.
    e.g. https://logo.clearbit.com/rcrwireless.com
    """
    if not domain:
        return None
    # Strip protocol / path — keep only the bare hostname
    host = re.sub(r'^https?://', '', domain).split("/")[0].strip()
    if not host or "." not in host:
        return None
    return f"https://logo.clearbit.com/{host}"


def _entry_to_raw(entry, query_entry: dict, seen_local: set) -> Optional[dict]:
    """Normalise a feedparser entry into a RawArticle dict."""
    url = entry.get("link", "")
    if not url or url in seen_local:
        return None
    seen_local.add(url)

    published_at = ""
    if entry.get("published_parsed"):
        try:
            import calendar
            published_at = datetime.utcfromtimestamp(
                calendar.timegm(entry.published_parsed)
            ).isoformat()
        except Exception:
            pass

    # For Google News RSS entries the article URL is a news.google.com redirect.
    # feedparser exposes the real publisher domain via entry.source.href — use it
    # for source_url so Clearbit logo lookups hit the right domain.
    entry_source = entry.get("source") or {}
    gnews_source_href = entry_source.get("href", "")  # e.g. "https://rcrwireless.com"

    parts = url.split("/")
    if gnews_source_href:
        source_url = gnews_source_href.rstrip("/")
    else:
        source_url = "/".join(parts[:3]) if len(parts) >= 3 else url

    source = (
        entry_source.get("title")
        or (parts[2] if len(parts) >= 3 else "Unknown")
    )
    title = (entry.get("title") or "")[:200]

    body = ""
    if entry.get("summary"):
        body = entry.summary
    elif entry.get("content"):
        body = entry.content[0].get("value", "")

    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s{2,}", " ", body).strip()

    if not title:
        return None

    # ── Image resolution: RSS media tags → og:image fetch → Clearbit logo ──────
    # Resolve Google News redirect URLs to real article URLs before fetching
    # og:image — hitting news.google.com directly returns no usable meta tags.
    real_url = _decode_gnews_url(url) if "news.google.com" in url else url

    # Step 1: try RSS-embedded images first (fast, no HTTP request)
    image_url = _extract_rss_image(entry)
    # Step 2: fetch og:image from the REAL article page (not the GNews redirect);
    #         Clearbit logo is the final guaranteed fallback inside _fetch_og_image.
    if not image_url:
        image_url = _fetch_og_image(real_url, source_domain=source_url)

    return {
        "title":        title,
        "url":          url,
        "image_url":    image_url,
        "raw_text":     body[:2000],
        "category":     query_entry["category"],
        "source":       source,
        "source_url":   source_url,
        "published_at": published_at,
    }


def _search_gnews_rss(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """Query Google News RSS with a 4-day recency filter — never blocked from datacenter IPs."""
    # Append "when:4d" so Google News only returns articles from the last 4 days.
    # This filters at the source and dramatically reduces stale-article noise.
    q_with_recency = query_entry["query"] + " when:4d"
    params = urllib.parse.urlencode({
        "q":    q_with_recency,
        "hl":   "en-US",
        "gl":   "US",
        "ceid": "US:en",
    })
    url = f"{_GNEWS_RSS_BASE}?{params}"
    entries = _fetch_rss(url)
    articles = []
    for e in entries[:MAX_RESULTS_PER_QUERY]:
        art = _entry_to_raw(e, query_entry, seen_local)
        if art:
            articles.append(art)
    return articles


def _search_category_rss(query_entry: dict, seen_local: set) -> List[RawArticle]:
    """Fallback: keyword-filtered trade publication RSS feeds."""
    cat = query_entry.get("category", "General")
    feeds = _CATEGORY_RSS.get(cat, _CATEGORY_RSS["General"])
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
    """Google News RSS primary → category trade feeds fallback."""
    articles = _search_gnews_rss(query_entry, seen_local)
    if not articles:
        logger.info(f"  GNews empty for '{query_entry['query']}' — trying category feeds")
        articles = _search_category_rss(query_entry, seen_local)
    logger.info(f"  '{query_entry['query']}' → {len(articles)} articles")
    return articles


def _get_dynamic_queries(base_queries: List[dict]) -> List[dict]:
    """
    Append a rotating date suffix to each query so Google News RSS returns
    a different result set on every fetch cycle rather than the same cached page.
    Cycles through: today, yesterday, and 'this week' each time it's called.
    """
    from datetime import date
    today     = date.today()
    yesterday = today - timedelta(days=1)

    # Pick a date-tag that rotates by hour so successive refreshes differ
    hour_slot = datetime.utcnow().hour % 3
    if hour_slot == 0:
        date_tag = today.strftime("%B %d")         # e.g. "May 09"
    elif hour_slot == 1:
        date_tag = yesterday.strftime("%B %d")     # e.g. "May 08"
    else:
        date_tag = today.strftime("%Y")            # e.g. "2025"

    dynamic = []
    for q in base_queries:
        # Only append date tag to queries that don't already end with a year
        query = q["query"]
        if not re.search(r'\b20\d\d\b', query):
            query = f"{query} {date_tag}"
        dynamic.append({**q, "query": query})
    return dynamic


def fetch_raw_articles(queries: List[dict]) -> List[RawArticle]:
    """
    Run all queries concurrently via RSS and return de-duped raw articles.
    Applies date-rotation to queries to avoid identical cached results.
    Drop-in replacement for the old SearXNG / DDG version — no sidecar needed.
    """
    # Apply dynamic date-rotation so each refresh hits a slightly different
    # Google News cache bucket and surfaces genuinely fresh articles.
    queries = _get_dynamic_queries(queries)

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

    logger.info(f"fetch_raw_articles: {len(all_articles)} total unique articles (RSS backend)")
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
        # call_llm implementations differ: some return str, some return dict.
        # Normalise to str before JSON extraction.
        if isinstance(raw, dict):
            raw = (
                raw.get("response")
                or raw.get("text")
                or raw.get("content")
                or raw.get("message")
                or str(raw)
            )
        raw = str(raw)
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

    # 1. Fetch all raw articles (concurrent RSS)
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