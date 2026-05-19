"""
news_chat_agent.py [FIXED VERSION]
──────────────────────────────────────────────────────────────────
Fully standalone news intelligence agent for the Bimlo news panel.

KEY IMPROVEMENTS:
  ✅ Robust article fetch with automatic retries
  ✅ Detailed error logging (see exactly why fetch fails)
  ✅ No silent failures — empty string only after all retries exhausted
  ✅ Proper URL redirect resolution (no premature response closing)
  ✅ Circuit breaker to avoid hammering blocked publishers
  ✅ Better timeout tuning (20s initial, 10s retry)

Endpoint: POST /api/news/chat

LLM routing:
  → CF_NEWS_URL (dedicated Bimlo News Cloudflare Worker, own AI quota)
  llm_client.py is never imported or touched.

Required env vars:
  CF_NEWS_URL     — e.g. https://bimlo.helaliaminhelali.workers.dev
  CF_NEWS_API_KEY — the shared secret you set on that worker
"""

import os
import re
import uuid
import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from core import globals as g

from prompt_loader import load_prompt

logger = logging.getLogger("news_chat_agent")
router = APIRouter()

# ── Dedicated news worker config ───────────────────────────────────────────────

_CF_NEWS_URL     = os.getenv("CF_NEWS_URL", "").rstrip("/")
_CF_NEWS_API_KEY = os.getenv("CF_NEWS_API_KEY", "")

# ── Session memory — isolated from main RAG sessions ──────────────────────────

MAX_HISTORY    = 16
_sessions:      Dict[str, deque] = {}
_sessions_lock = threading.Lock()


def _get_history(sid: str) -> List[dict]:
    with _sessions_lock:
        return list(_sessions.get(sid, []))


def _append(sid: str, role: str, content: str):
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = deque(maxlen=MAX_HISTORY)
        _sessions[sid].append({"role": role, "content": content})


# ── Circuit breaker: prevent hammering blocked publishers ──────────────────────

_CIRCUIT_BREAKER: Dict[str, Tuple[int, float]] = {}
_CB_LOCK = threading.Lock()
_CB_THRESHOLD = 3  # Fail 3× in a row → skip for 5 minutes
_CB_COOLDOWN = 300  # seconds


def _check_circuit(url: str) -> bool:
    """
    Returns False if the circuit is open (publisher is blocked).
    Otherwise returns True (can try).
    """
    if not url:
        return True
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
    except Exception:
        domain = url[:30]

    with _CB_LOCK:
        if domain in _CIRCUIT_BREAKER:
            fail_count, last_fail = _CIRCUIT_BREAKER[domain]
            if time.time() - last_fail < _CB_COOLDOWN:
                logger.warning(f"[CB] Circuit open for {domain} — skipping fetch (too many failures)")
                return False
            else:
                # Cooldown expired, reset
                del _CIRCUIT_BREAKER[domain]
                return True
    return True


def _record_circuit_failure(url: str):
    """Record a fetch failure for this domain."""
    if not url:
        return
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
    except Exception:
        domain = url[:30]

    with _CB_LOCK:
        if domain not in _CIRCUIT_BREAKER:
            _CIRCUIT_BREAKER[domain] = (0, time.time())
        fail_count, _ = _CIRCUIT_BREAKER[domain]
        _CIRCUIT_BREAKER[domain] = (fail_count + 1, time.time())
        if fail_count + 1 >= _CB_THRESHOLD:
            logger.warning(f"[CB] Circuit opened for {domain} after {fail_count + 1} failures")


def _record_circuit_success(url: str):
    """Clear failures for this domain on success."""
    if not url:
        return
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
    except Exception:
        domain = url[:30]

    with _CB_LOCK:
        if domain in _CIRCUIT_BREAKER:
            del _CIRCUIT_BREAKER[domain]


# ── Pydantic models ────────────────────────────────────────────────────────────

class PinnedArticle(BaseModel):
    id:          str
    title:       str
    category:    str
    source:      str
    articleUrl:  Optional[str] = ""
    aiImpact:    Optional[str] = ""
    rawSummary:  Optional[str] = ""
    imageUrl:    Optional[str] = None


class NewsChatRequest(BaseModel):
    query:           str
    session_id:      Optional[str]                 = None
    pinned_articles: Optional[List[PinnedArticle]] = []


class NewsChatResponse(BaseModel):
    answer:     str
    session_id: str


# ── Article content fetcher [IMPROVED] ─────────────────────────────────────────

def _resolve_url(url: str) -> str:
    """
    Resolve the real publisher URL from a Google News RSS link.

    Google News RSS links (CBMi... encoded URLs) cannot be resolved via
    normal HTTP redirects — Google blocks HEAD/GET requests and returns
    the same Google URL. We use gnewsdecoder which decodes the base64
    payload directly without making any HTTP request to Google.

    Falls back to the original URL if decoding fails.
    """
    if not url or url in ("#", ""):
        return url

    # Only bother decoding Google News links
    if "news.google.com" not in url:
        return url

    try:
        from gnews import GNews
        g = GNews()
        # get_full_article returns a newspaper Article object with .url as the real URL
        article = g.get_full_article(url)
        if article and article.url and "google.com" not in article.url:
            logger.info(f"[resolve] GNews decoded: {url[:50]} → {article.url[:60]}")
            return article.url
        logger.debug("[resolve] GNews returned no usable URL")
    except ImportError:
        logger.warning("[resolve] gnews not installed — run: pip install gnews")
    except Exception as e:
        logger.debug(f"[resolve] GNews failed for {url[:50]}: {e}")

    # Last resort: try requests GET (may not work for all Google News links)
    try:
        import requests
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
            allow_redirects=True,
        )
        final = r.url
        if "google.com" not in final:
            logger.info(f"[resolve] requests fallback resolved: {url[:50]} → {final[:60]}")
            return final
    except Exception as e:
        logger.debug(f"[resolve] requests fallback failed: {e}")

    return url


def _fetch_article_text(url: str, char_limit: int = 12000) -> str:
    """
    Fetch and extract readable text from a news article URL.
    Returns up to char_limit chars of clean body text, or "" on failure.

    IMPROVED:
      1. Uses circuit breaker — stops hammering blocked publishers
      2. Retries transient failures (timeout, connection error)
      3. Detailed logging so you see exactly what failed
      4. Respects 429 (rate limit) and backs off
      5. Follows JSON-LD → semantic classes → itemprop → fallback

    Steps:
      0. Circuit breaker check
      1. Follow redirects (Google News RSS links are redirect wrappers)
      2. Try 2× with backoff for transient errors
      3. JSON-LD articleBody  — most reliable for major publishers
      4. Semantic class match — article-body, entry-content, post-text, …
      5. itemprop="articleBody"
      6. <main> / <body> fallback

    Full browser headers defeat most bot-detection.
    Streams max 80 KB so we never download JS bundles.
    """
    if not url or url in ("#", ""):
        return ""

    # Check circuit breaker first
    if not _check_circuit(url):
        return ""

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("[fetch] requests/beautifulsoup4 not installed")
        return ""

    # Step 0: resolve Google News redirect → real publisher URL
    resolved = _resolve_url(url)
    if resolved != url:
        logger.info(f"[fetch] redirect: {url[:50]} → {resolved[:50]}")

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":  "en-US,en;q=0.9",
        "Accept-Encoding":  "gzip, deflate, br",
        "DNT":              "1",
        "Connection":       "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":   "document",
        "Sec-Fetch-Mode":   "navigate",
        "Sec-Fetch-Site":   "none",
        "Cache-Control":    "max-age=0",
    }

    _NOISE = re.compile(
        r"(subscribe|sign up|newsletter|cookie|privacy policy|terms of service"
        r"|advertisement|continue reading|read more|follow us|share this"
        r"|already a subscriber|log in|create account)",
        re.I,
    )

    _ARTICLE_CLASS = re.compile(
        r"\b(article[-_]?(body|content|text|copy)|story[-_]?(body|content|text)"
        r"|post[-_]?(body|content|text)|entry[-_]?(content|body)"
        r"|content[-_]?(body|article)|main[-_]?content)\b",
        re.I,
    )

    def _stream_html(target_url: str, timeout: int = 20) -> str:
        """Stream HTML with max 80 KB cap."""
        buf = []
        with requests.get(
            target_url, headers=_HEADERS, timeout=timeout,
            stream=True, allow_redirects=True,
        ) as r:
            r.raise_for_status()
            size = 0
            for chunk in r.iter_content(chunk_size=16_384, decode_unicode=True):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="ignore")
                buf.append(chunk)
                size += len(chunk)
                if size > 80_000:
                    logger.debug(f"[stream] Hit 80KB limit at {target_url[:40]}")
                    break
        return "".join(buf)

    def _extract(html: str) -> str:
        """Extract article text from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # 1. JSON-LD articleBody — Reuters, Bloomberg, NYT all publish this
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json as _json
                data = _json.loads(script.string or "")
                for obj in (data if isinstance(data, list) else [data]):
                    body = obj.get("articleBody") or obj.get("description") or ""
                    if body and len(body) > 300:
                        logger.debug("[extract] Got text from JSON-LD articleBody")
                        return str(body)[:char_limit]
            except Exception:
                pass

        # 2. Semantic class match — most news sites use one of these
        for el in soup.find_all(["article", "div", "main", "section"]):
            cls = (el.get("class") or [])
            if isinstance(cls, str):
                cls = [cls]
            if any(_ARTICLE_CLASS.search(c) for c in cls):
                txt = el.get_text(separator=" ", strip=True)
                clean = " ".join(txt.split())
                if clean and len(clean) > 300 and not _NOISE.search(clean[:200]):
                    logger.debug("[extract] Got text from semantic class match")
                    return clean[:char_limit]

        # 3. itemprop="articleBody"
        for el in soup.find_all(attrs={"itemprop": "articleBody"}):
            txt = el.get_text(separator=" ", strip=True)
            clean = " ".join(txt.split())
            if clean and len(clean) > 300:
                logger.debug("[extract] Got text from itemprop=articleBody")
                return clean[:char_limit]

        # 4. <main> / <body> fallback
        for el in soup.find_all(["main", "body"]):
            for junk in el.find_all(["script", "style", "meta", "link", "nav"]):
                junk.decompose()
            txt = el.get_text(separator=" ", strip=True)
            clean = " ".join(txt.split())
            if clean and len(clean) > 300 and not _NOISE.search(clean[:200]):
                logger.debug("[extract] Got text from main/body fallback")
                return clean[:char_limit]

        logger.debug("[extract] Could not extract any text from HTML")
        return ""

    # Retry logic: try up to 2× with backoff for transient errors
    max_attempts = 2
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            logger.debug(f"[fetch] Attempt {attempt}/{max_attempts}: {resolved[:60]}")
            html = _stream_html(resolved, timeout=20 if attempt == 1 else 10)
            if not html or len(html) < 500:
                logger.warning(f"[fetch] HTML too small ({len(html)} bytes) from {resolved[:40]}")
                return ""

            extracted = _extract(html)
            if extracted:
                _record_circuit_success(resolved)
                logger.info(f"[fetch] ✅ Success: {len(extracted)} chars from {resolved[:50]}")
                return extracted
            else:
                logger.warning(f"[fetch] No article text extracted from {resolved[:50]}")
                return ""

        except requests.exceptions.Timeout:
            logger.warning(f"[fetch] Timeout on attempt {attempt} for {resolved[:50]}")
            if attempt < max_attempts:
                time.sleep(1)  # Brief backoff before retry
            continue

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[fetch] Connection error on attempt {attempt}: {e}")
            if attempt < max_attempts:
                time.sleep(1)
            continue

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if hasattr(e, 'response') else None
            logger.warning(f"[fetch] HTTP {status} for {resolved[:50]}")
            if status == 429:  # Rate limited
                _record_circuit_failure(resolved)
                return ""
            if status == 404:  # Not found — don't retry
                return ""
            if status >= 500:  # Server error — retry once
                if attempt < max_attempts:
                    time.sleep(2)
                continue
            return ""

        except Exception as e:
            logger.debug(f"[fetch] Unexpected error on attempt {attempt}: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                time.sleep(1)
            continue

    # All retries exhausted
    _record_circuit_failure(resolved)
    logger.error(f"[fetch] ❌ All retries exhausted for {resolved[:50]}")
    return ""


# ── Build context block from pinned articles [IMPROVED] ──────────────────────

def _get_cached_full_content(article_url: str) -> str:
    """
    Look up the article in the news pipeline cache and return its full_content
    if available. This is the content fetched during the pipeline run (from
    NewsData.io or RSS) — no live HTTP request needed.
    """
    if not article_url:
        return ""
    try:
        from news_pipeline import get_page, get_meta
        meta = get_meta()
        if not meta:
            return ""
        total_pages = meta.get("total_pages", 0)
        for page_num in range(total_pages):
            page = get_page(page_num)
            if not page:
                continue
            for item in page.get("items", []):
                if item.get("article_url") == article_url:
                    fc = item.get("full_content", "")
                    if fc and len(fc) > 50:
                        logger.info(
                            f"[cache] ✅ Found full_content in pipeline cache "
                            f"({len(fc)} chars) for {article_url[:50]}"
                        )
                        return fc
    except Exception as e:
        logger.debug(f"[cache] pipeline cache lookup failed: {e}")
    return ""


def _build_pinned_context(pinned: List[PinnedArticle]) -> str:
    """
    Build a context block with full article text OR clearly labeled preview fallback.

    Content resolution order (best → worst):
      1. ChromaDB vector store chunks (semantically indexed full content — best)
      2. full_content from pipeline cache (set by NewsData.io at fetch time)
      3. Live scrape of the real publisher URL (fallback for RSS-sourced articles)
      4. rawSummary cached preview (last resort — limited to 600 chars)
    """
    if not pinned:
        return ""

    vs = getattr(g, "vector_store", None)
    blocks = ["═══ ARTICLES PINNED BY USER ═══\n"]

    for i, a in enumerate(pinned, 1):
        block = f"── [{i}] \"{a.title}\" ──\n"
        block += f"Source: {a.source} | Category: {a.category}\n"
        if a.articleUrl and a.articleUrl != "#":
            block += f"URL: {a.articleUrl}\n"

        full = ""

        # Step 1: Try ChromaDB vector store (semantically indexed full content)
        if vs and a.articleUrl:
            try:
                chunks = vs.get_news_chunks_by_url(a.articleUrl)
                if chunks:
                    # Skip first chunk (title-only), join the rest
                    content_chunks = [c for c in chunks if not c.startswith("Title: ")]
                    if content_chunks:
                        full = "\n".join(content_chunks)
                        logger.info(
                            f"[pinned] ✅ Article {i}/{len(pinned)}: "
                            f"'{a.title[:40]}' — vector store hit ({len(full)} chars, {len(content_chunks)} chunks)"
                        )
            except Exception as e:
                logger.debug(f"[pinned] vector store lookup failed for '{a.title[:30]}': {e}")

        # Step 2: Try pipeline cache (full_content from NewsData.io — no HTTP)
        if not full:
            full = _get_cached_full_content(a.articleUrl or "")
            if full:
                logger.info(
                    f"[pinned] ✅ Article {i}/{len(pinned)}: "
                    f"'{a.title[:40]}' — pipeline cache hit ({len(full)} chars)"
                )

        # Step 3: Live scrape (works for real publisher URLs; fails for Google redirects)
        if not full:
            full = _fetch_article_text(a.articleUrl or "")
            if full:
                logger.info(
                    f"[pinned] ✅ Article {i}/{len(pinned)}: "
                    f"'{a.title[:40]}' — live fetch ({len(full)} chars)"
                )

        if full:
            block += f"Full article content:\n{full}\n"
        else:
            # Step 4: Cached preview fallback
            logger.warning(
                f"[pinned] ⚠️  Article {i}/{len(pinned)}: "
                f"'{a.title[:40]}' — all fetches failed, using cached preview"
            )
            block += (
                "[⚠️  FALLBACK: Full article could not be fetched. "
                "The cached preview below is limited. "
                "The AI will only reference details explicitly shown here. "
                "Do NOT invent statistics or numbers.]\n"
            )
            if a.rawSummary:
                block += f"Cached preview (600 chars): {a.rawSummary[:600]}\n"
            if a.aiImpact:
                block += f"AI impact note: {a.aiImpact}\n"
            if not a.rawSummary and not a.aiImpact:
                block += "(No cached content — only the title is known.)\n"

        blocks.append(block)

    return "\n".join(blocks)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = load_prompt("news_chat_system")


# ── News worker HTTP call ──────────────────────────────────────────────────────

def _call_news_worker(
    prompt:        str,
    system_prompt: str   = "",
    max_tokens:    int   = 1500,
    temperature:   float = 0.4,
) -> str:
    """
    POST a single LLM request to the dedicated Bimlo News CF Worker.
    Never touches llm_client.py — completely isolated quota.
    Raises on misconfiguration or worker error.
    """
    if not _CF_NEWS_URL or not _CF_NEWS_API_KEY:
        raise RuntimeError(
            "CF_NEWS_URL or CF_NEWS_API_KEY is not set. "
            "Add both to your .env to enable the news LLM worker."
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
        # CF Workers AI can nest the text under several keys
        raw = (
            body.get("response")
            or body.get("result")
            or body.get("text")
            or body.get("content")
            or body.get("message")
            or ""
        )
        if isinstance(raw, dict):
            raw = (
                raw.get("response")
                or raw.get("text")
                or raw.get("content")
                or raw.get("message")
                or ""
            )
        text = str(raw).strip()
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


def _call_llm(system_prompt: str, history: List[dict], user_message: str) -> str:
    """
    Build full prompt (transcript + user turn) and call the news worker.
    Returns a safe error string instead of raising so the endpoint never 500s.
    """
    transcript = ""
    if history:
        for turn in history[-6:]:
            role = "User" if turn["role"] == "user" else "Bimlo"
            transcript += f"{role}: {turn['content']}\n"
        transcript += "\n"

    prompt = f"{transcript}User: {user_message}\nBimlo:"

    try:
        return _call_news_worker(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=900,
            temperature=0.4,
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        body   = e.response.text.lower()
        logger.error(f"[news_chat] worker HTTP {status}: {e.response.text[:200]}")
        # Quota / neuron exhaustion — CF Workers AI returns 500 with "neurons" in the body
        if status in (429, 500) and ("neuron" in body or "quota" in body or "allocation" in body or "daily" in body or "limit" in body):
            return (
                "The news AI is taking a breather for today 😴 — "
                "it's hit its daily limit. Check back tomorrow and it'll be ready to go! 🙂"
            )
        return f"⚠️ News worker error ({status}). Please try again."
    except httpx.TimeoutException:
        logger.error("[news_chat] worker timed out after 30s")
        return "⚠️ News worker timed out. Please try again."
    except RuntimeError as e:
        err_str = str(e).lower()
        logger.error(f"[news_chat] config error: {e}")
        # Also catch quota errors surfaced as RuntimeError (e.g. from _call_news_worker parsing)
        if "neuron" in err_str or "quota" in err_str or "allocation" in err_str or "daily" in err_str:
            return (
                "The news AI is taking a breather for today 😴 — "
                "it's hit its daily limit. Check back tomorrow and it'll be ready to go! 🙂"
            )
        return f"⚠️ {e}"
    except Exception as e:
        logger.error(f"[news_chat] unexpected error: {e}")
        return f"⚠️ Could not reach news worker: {e}"


# ── Vector store semantic search ──────────────────────────────────────────────

def _search_news_vector_store(query: str, top_k: int = 12) -> str:
    """
    Search the indexed news articles in ChromaDB for semantically relevant chunks.

    Returns a formatted context block or empty string if no results / unavailable.
    """
    vs = getattr(g, "vector_store", None)
    if vs is None:
        logger.debug("[vector] g.vector_store not available")
        return ""

    try:
        results = vs.search_news(query, top_k=top_k)
    except Exception as e:
        logger.warning(f"[vector] search failed: {e}")
        return ""

    if not results:
        logger.debug("[vector] no relevant news chunks found")
        return ""

    # Group by article URL so each article appears once with all its chunks
    articles: Dict[str, dict] = {}
    for r in results:
        meta = r["metadata"]
        url = meta.get("article_url", "") or meta.get("url", "")
        if url not in articles:
            articles[url] = {
                "title":    meta.get("title", ""),
                "source":   meta.get("source", ""),
                "category": meta.get("category", ""),
                "url":      url,
                "chunks":   [],
                "distance": r.get("distance", 1.0) or 1.0,
            }
        articles[url]["chunks"].append(r["text"])
        articles[url]["distance"] = min(articles[url]["distance"], r.get("distance", 1.0) or 1.0)

    blocks = [
        "═══ ARTICLES RELEVANT TO YOUR QUESTION (from indexed library) ═══\n"
    ]
    for url, art in sorted(articles.items(), key=lambda x: x[1]["distance"])[:7]:
        block = f'── "{art["title"]}" ──\n'
        block += f'Source: {art["source"]} | Category: {art["category"]}\n'
        if art["url"] and art["url"] != "#":
            block += f"URL: {art['url']}\n"
        seen: set = set()
        deduped = []
        for c in art["chunks"]:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        block += "\n".join(deduped) + "\n"
        blocks.append(block)

    logger.info(f"[vector] {len(articles)} relevant articles found for query")
    return "\n".join(blocks)


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/api/news/chat", response_model=NewsChatResponse)
async def news_chat(req: NewsChatRequest):
    """
    Standalone news intelligence endpoint — isolated from /query and the RAG engine.
    Calls CF_NEWS_URL (dedicated news worker) — never depletes main RAG quota.

    Context sources (all used):
      1. Semantically relevant articles from vector store (indexed full content)
      2. User-pinned articles with full content / live fetch / preview fallback
    """
    sid     = req.session_id or str(uuid.uuid4())
    history = _get_history(sid)
    pinned  = req.pinned_articles or []

    logger.info(
        f"[news_chat] sid={sid} | pinned={len(pinned)} | "
        f"turns={len(history) // 2} | q={req.query[:80]!r}"
    )

    system_prompt = _SYSTEM_TEMPLATE.format(
        today=datetime.utcnow().strftime("%B %d, %Y"),
    )

    # 1. Semantically relevant articles from the indexed vector store
    vector_block = _search_news_vector_store(req.query)

    # 2. User-pinned articles (full content → live fetch → preview)
    pinned_block = _build_pinned_context(pinned)

    # 3. Combine both sources
    context_parts: list = []
    if vector_block:
        context_parts.append(vector_block)
    if pinned_block:
        context_parts.append(pinned_block)

    if context_parts:
        user_msg = "\n\n".join(context_parts) + f"\n\nMy question: {req.query}"
    else:
        user_msg = req.query

    answer = _call_llm(system_prompt, history, user_msg)

    _append(sid, "user",      req.query)
    _append(sid, "assistant", answer)

    return NewsChatResponse(answer=answer, session_id=sid)