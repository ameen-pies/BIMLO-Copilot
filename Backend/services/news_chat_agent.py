"""
news_chat_agent.py
──────────────────────────────────────────────────────────────────
Fully standalone news intelligence agent for the Bimlo news panel.

Completely separate from the RAG engine — does NOT touch the vector
store or telecom documents at all.

What it does:
  1. Pulls ALL cached articles from news_pipeline (titles, summaries,
     ai_impact, article_url, category, source, published_at).
  2. For any article the user has pinned, fetches the FULL article
     content from its URL via requests + BeautifulSoup so the LLM
     has the real text, not just a 300-char snippet.
  3. Builds a self-contained system prompt with that context and
     calls call_llm(prompt=, system_prompt=) — the correct signature.
  4. Maintains per-session conversation history (totally separate
     from _sessions in main.py — no cross-contamination).

Endpoint: POST /api/news/chat
"""

import re
import uuid
import logging
import threading
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("news_chat_agent")
router = APIRouter()

# ── Session memory — isolated from main RAG sessions ──────────────────────────

MAX_HISTORY = 16
_sessions: Dict[str, deque] = {}
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
    query:            str
    session_id:       Optional[str]                 = None
    pinned_articles:  Optional[List[PinnedArticle]] = []


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
            block += f"Full article text:\n{full}\n"
            logger.info(f"Fetched full text for pinned article '{a.title[:50]}' ({len(full)} chars)")
        else:
            if a.rawSummary:
                block += f"Cached summary: {a.rawSummary}\n"
            if a.aiImpact:
                block += f"Industry impact: {a.aiImpact}\n"
            logger.info(f"Using cached data for pinned article '{a.title[:50]}'")

        blocks.append(block)

    return "\n".join(blocks)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are Bimlo, a telecom industry intelligence analyst. Today: {today}.
You are embedded in a live telecom news feed. The user has selected specific articles for discussion.
Analyse the provided article content deeply. Cite sources naturally. Be concise and expert.
"""


# ── LLM call ───────────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, history: List[dict], user_message: str) -> str:
    """
    Uses call_llm(prompt=, system_prompt=) — the correct llm_client.py signature.
    History is folded into the prompt as a plain transcript.
    """
    try:
        from llm_client import call_llm, check_llm_available

        available, provider = check_llm_available()
        if not available:
            return "⚠️ LLM not configured — please set GROQ_API_KEY."

        transcript = ""
        if history:
            for turn in history[-6:]:
                role = "User" if turn["role"] == "user" else "Bimlo"
                transcript += f"{role}: {turn['content']}\n"
            transcript += "\n"

        prompt = f"{transcript}User: {user_message}\nBimlo:"

        return call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=900,
            temperature=0.4,
        ).strip()

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"⚠️ Error: {e}"


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/api/news/chat", response_model=NewsChatResponse)
async def news_chat(req: NewsChatRequest):
    """
    Standalone news intelligence endpoint — isolated from /query and the RAG engine.
    Only sends pinned article content to the LLM — nothing else.
    """
    sid     = req.session_id or str(uuid.uuid4())
    history = _get_history(sid)
    pinned  = req.pinned_articles or []

    logger.info(
        f"[news_chat] sid={sid} | pinned={len(pinned)} | "
        f"turns={len(history)//2} | q={req.query[:80]!r}"
    )

    # System prompt — no feed dump, just role + date
    system_prompt = _SYSTEM_TEMPLATE.format(
        today=datetime.utcnow().strftime("%B %d, %Y"),
    )

    # Pinned article context (fetches full article text over HTTP)
    pinned_block = _build_pinned_context(pinned)

    # Compose user message
    user_msg = (
        f"{pinned_block}\n\nMy question: {req.query}"
        if pinned_block else req.query
    )

    answer = _call_llm(system_prompt, history, user_msg)

    _append(sid, "user",      req.query)
    _append(sid, "assistant", answer)

    return NewsChatResponse(answer=answer, session_id=sid)