"""
news_chat_agent.py
──────────────────────────────────────────────────────────────────
Fully standalone news intelligence agent for the Bimlo news panel.

Completely separate from the RAG engine — does NOT touch the vector
store, llm_client.py, or the main RAG quota at all.

What it does:
  1. For any article the user has pinned, fetches the FULL article
     content from its URL via requests + BeautifulSoup so the LLM
     has the real text, not just a 300-char snippet.
  2. Builds a self-contained system prompt with that context and
     calls the DEDICATED NEWS CF WORKER (CF_NEWS_URL) directly.
  3. Maintains per-session conversation history (totally separate
     from _sessions in main.py — no cross-contamination).

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
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

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


# ── Article content fetcher ────────────────────────────────────────────────────

def _fetch_article_text(url: str, char_limit: int = 1500) -> str:
    """
    Fetch and extract readable text from a news article URL.
    Returns up to char_limit chars of clean body text, or "" on failure.
    """
    if not url or url in ("#", ""):
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        body = (
            soup.find("article")
            or soup.find(attrs={"class": re.compile(r"article|story|content|body|post", re.I)})
            or soup.find("main")
            or soup.body
        )

        text = (body or soup).get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:char_limit]

    except ImportError:
        logger.warning("requests/beautifulsoup4 not installed — install them for full article fetch")
        return ""
    except Exception as e:
        logger.warning(f"Could not fetch {url}: {e}")
        return ""


# ── Context builder ────────────────────────────────────────────────────────────

def _build_pinned_context(pinned: List[PinnedArticle]) -> str:
    """
    Rich context block for pinned articles.
    Attempts full HTTP fetch of each article; falls back to cached summary.
    """
    if not pinned:
        return ""

    blocks = ["═══ ARTICLES PINNED BY USER ═══\n"]
    for i, a in enumerate(pinned, 1):
        block  = f"── [{i}] \"{a.title}\" ──\n"
        block += f"Source: {a.source} | Category: {a.category}\n"
        if a.articleUrl and a.articleUrl != "#":
            block += f"URL: {a.articleUrl}\n"

        full = _fetch_article_text(a.articleUrl or "")
        if full:
            block += f"Full article text (fetched live):\n{full}\n"
            logger.info(f"Fetched full text for pinned article '{a.title[:50]}' ({len(full)} chars)")
        else:
            block += (
                "[NOTE: Full article could not be fetched. Only the cached preview below is available. "
                "Do NOT invent or assume any specific numbers, statistics, or details not present in this snippet.]\n"
            )
            if a.rawSummary:
                block += f"Cached preview (300 chars max): {a.rawSummary}\n"
            if a.aiImpact:
                block += f"AI impact note: {a.aiImpact}\n"
            if not a.rawSummary and not a.aiImpact:
                block += "(No cached content available — only the title is known.)\n"
            logger.info(f"Using cached data for pinned article '{a.title[:50]}'")

        blocks.append(block)

    return "\n".join(blocks)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are Bimlo, the AI analyst of BIMLO TECHNOLOGIE — a company specialising in BIM engineering \
(3D–7D digital models), Scan to BIM, BIM 4D construction planning, telecom infrastructure studies \
(rooftop, pylons, calculation notes), and DeepTwin AI digital twins for predictive maintenance. \
Today: {today}.
You are embedded in a live industry news feed covering telecom and construction sectors. \
The user has pinned specific articles for discussion and their content is provided below.

CRITICAL RULES — follow these exactly:
1. Only reference facts, figures, statistics, and details that are explicitly present in the \
provided article text. Never invent, assume, or fill in numbers or data not in the text.
2. If an article shows [NOTE: Full article could not be fetched], only a short cached preview is \
available. Limit your analysis strictly to what the preview states. Do not extrapolate specific \
figures from a headline or snippet.
3. If the user asks for a specific detail (e.g. a percentage, price, or statistic) that is not in \
the provided content, say clearly: "That detail isn't in the article text I have access to — you \
can read the full article at the source link."
4. Analyse through the lens of BIM, telecom infrastructure, and digital construction. Highlight \
implications for BTP/construction professionals and telecom engineers where relevant.
5. Be concise and expert. Cite which article you are drawing from when referencing specific claims.
"""


# ── News worker HTTP call ──────────────────────────────────────────────────────

def _call_news_worker(
    prompt:        str,
    system_prompt: str   = "",
    max_tokens:    int   = 900,
    temperature:   float = 0.4,
) -> str:
    """
    POST a single LLM request to the dedicated Bimlo News CF Worker.
    Never touches llm_client.py — completely isolated quota.
    Raises on misconfiguration or HTTP errors.
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

    text = resp.json().get("response", "").strip()
    if not text:
        raise RuntimeError("News worker returned an empty response")
    return text


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
        logger.error(f"[news_chat] worker HTTP {e.response.status_code}: {e.response.text[:200]}")
        return f"⚠️ News worker error ({e.response.status_code}). Please try again."
    except httpx.TimeoutException:
        logger.error("[news_chat] worker timed out after 30s")
        return "⚠️ News worker timed out. Please try again."
    except RuntimeError as e:
        logger.error(f"[news_chat] config error: {e}")
        return f"⚠️ {e}"
    except Exception as e:
        logger.error(f"[news_chat] unexpected error: {e}")
        return f"⚠️ Could not reach news worker: {e}"


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/api/news/chat", response_model=NewsChatResponse)
async def news_chat(req: NewsChatRequest):
    """
    Standalone news intelligence endpoint — isolated from /query and the RAG engine.
    Calls CF_NEWS_URL (dedicated news worker) — never depletes main RAG quota.
    Only sends pinned article content to the LLM — nothing else.
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

    pinned_block = _build_pinned_context(pinned)

    user_msg = (
        f"{pinned_block}\n\nMy question: {req.query}"
        if pinned_block else req.query
    )

    answer = _call_llm(system_prompt, history, user_msg)

    _append(sid, "user",      req.query)
    _append(sid, "assistant", answer)

    return NewsChatResponse(answer=answer, session_id=sid)