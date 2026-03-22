"""
suggest.py — Contextual next-step suggestion endpoint

Generates 3-4 short follow-up prompt chips based on the last
user query + assistant reply, using the same CloudflareClient
that powers the rest of the RAG engine.

Mount in your main FastAPI app:
    from suggest import router as suggest_router
    app.include_router(suggest_router)
"""

from __future__ import annotations

import os
import re
import json
import time
import requests
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

# ── Load .env so CF_API_KEY is available when running standalone ─────────────
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    for _parent in [_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))]:
        _env = os.path.join(_parent, ".env")
        if os.path.exists(_env):
            load_dotenv(_env, override=False)
            break
except ImportError:
    pass


# ────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ────────────────────────────────────────────────────────────────────────────

class SuggestRequest(BaseModel):
    user_query: str
    assistant_reply: str
    available_docs: List[str] = []   # filenames of uploaded documents


class SuggestResponse(BaseModel):
    suggestions: List[str]


# ────────────────────────────────────────────────────────────────────────────
# CF WORKER CALL  (mirrors CloudflareClient.chat() in rag_engine.py)
# ────────────────────────────────────────────────────────────────────────────

_CF_API_KEY  = os.getenv("CF_API_KEY", "")
_CF_API_URL  = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")

SYSTEM_PROMPT = """You are a smart assistant helping users explore document knowledge.

Given the user's last question, the AI's answer, and the list of available documents, generate follow-up prompt suggestions in TWO categories:

CATEGORY 1 — "contextual" (2-3 chips): Directly about what was just discussed. Based on specific terms, numbers, names, or concepts in the answer. What would a curious person naturally ask next about THIS specific answer?

CATEGORY 2 — "general" (1-2 chips): Broader questions that go beyond the current answer — comparing with other uploaded documents, asking about the overall project, or exploring a related topic not yet covered.

Rules:
- Return ONLY a raw JSON object with two keys. No markdown, no backticks, no explanation.
- Each chip must be 2-5 words, actionable, specific.
- "general" chips should naturally reference other documents or broader scope.

Return exactly this format:
{"contextual": ["chip 1", "chip 2", "chip 3"], "general": ["chip 1", "chip 2"]}"""


def _call_cf(user_query: str, assistant_reply: str, available_docs: List[str], max_retries: int = 2) -> List[str]:
    """
    Hit the CF Worker with a suggestion prompt.
    Returns contextual chips first, then general/cross-doc chips last.
    """
    # Read at call time so .env is guaranteed loaded
    cf_api_key = os.getenv("CF_API_KEY", "")
    cf_api_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")

    if not cf_api_key:
        print("⚠️  suggest: CF_API_KEY not set")
        return []

    docs_section = ""
    if available_docs:
        docs_section = f"\n\nAvailable documents: {', '.join(available_docs[:10])}"

    prompt = (
        f"The user asked: \"{user_query[:300]}\"\n\n"
        f"The AI answered:\n\"{assistant_reply[:600]}\""
        f"{docs_section}\n\n"
        f"Generate follow-up suggestions in the two categories as instructed. "
        f"Return ONLY the JSON object, nothing else."
    )

    payload = {
        "prompt":       prompt,
        "systemPrompt": SYSTEM_PROMPT,
        "history":      [],
        "max_tokens":   150,
        "temperature":  0.7,
        "task":         "suggest",
    }
    headers = {
        "Authorization": f"Bearer {cf_api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(cf_api_url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("response") or ""
                if isinstance(raw, list):
                    # Already parsed as list — treat as flat suggestions
                    return [str(s).strip()[:50] for s in raw if str(s).strip()][:5]
                elif not isinstance(raw, str):
                    raw = str(raw)
                return _parse_suggestions(raw.strip())
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"⚠️  suggest: CF returned {resp.status_code}: {resp.text[:120]}")
                break
        except Exception as e:
            print(f"⚠️  suggest: request failed — {e}")
            break

    return []


def _parse_suggestions(raw: str) -> List[str]:
    """
    Parse the model's structured response into an ordered list:
    contextual chips first, general chips last.
    Handles: proper JSON, Python repr (single quotes/True/False/None), newline list.
    """
    import ast

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    print(f"💡 suggest raw response: {clean[:200]}")

    def _extract(parsed) -> List[str]:
        # Unwrap list-wrapped object: [{...}] → {...}
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if isinstance(parsed, dict):
            contextual = [str(s).strip()[:50] for s in parsed.get("contextual", []) if str(s).strip()]
            general    = [str(s).strip()[:50] for s in parsed.get("general", []) if str(s).strip()]
            return (contextual[:3] + general[:2])[:5]
        if isinstance(parsed, list):
            return [str(s).strip()[:50] for s in parsed if str(s).strip()][:5]
        return []

    # 1. Try standard JSON
    try:
        result = _extract(json.loads(clean))
        if result:
            print(f"✅ suggest parsed (JSON): {result}")
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try Python repr (Llama returns single-quoted dicts)
    try:
        result = _extract(ast.literal_eval(clean))
        if result:
            print(f"✅ suggest parsed (ast): {result}")
            return result
    except (ValueError, SyntaxError):
        pass

    # 3. Last resort: newline/bullet list
    lines = [
        re.sub(r"^[\d\.\-\*\•\s]+", "", line).strip()
        for line in clean.splitlines() if line.strip()
    ]
    result = [l[:50] for l in lines if 2 <= len(l.split()) <= 7][:5]
    if result:
        print(f"✅ suggest parsed (lines): {result}")
    else:
        print(f"⚠️  suggest: could not parse response")
    return result


# ────────────────────────────────────────────────────────────────────────────
# ROUTER
# ────────────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest) -> SuggestResponse:
    """
    POST /suggest
    Body: { "user_query": "...", "assistant_reply": "...", "available_docs": ["file1.pdf", ...] }
    Returns: { "suggestions": ["contextual 1", "contextual 2", ..., "general 1", "general 2"] }
    Contextual chips come first, broader/cross-doc chips come last.
    """
    suggestions = _call_cf(req.user_query, req.assistant_reply, req.available_docs)
    return SuggestResponse(suggestions=suggestions)