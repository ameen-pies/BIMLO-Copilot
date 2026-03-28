"""
autocomplete.py — Instant inline query completion endpoint

Completes the user's partial query in real-time using the same
CloudflareClient / CF Worker that powers the rest of the RAG engine.

Design goals:
  - ⚡ Fast: max_tokens=30, temperature=0, task="classify" (fastest CF path)
  - 🧠 Smart: context-aware — knows available docs and last conversation turn
  - 🔒 Safe: only completes the QUERY, never answers it
  - 🌐 Multilingual: completes in whatever language the user is typing

Mount in your main FastAPI app:
    from services.autocomplete import router as autocomplete_router
    app.include_router(autocomplete_router)

Frontend usage:
    POST /autocomplete
    { "partial": "what is the budget", "session_context": "...", "available_docs": [...] }
    → { "completion": " for Q3?" }   ← only the suffix, not the full string

Streaming variant:
    POST /autocomplete/stream   (same body)
    → text/event-stream  data: <token>\\n\\n  ...  data: [DONE]\\n\\n
"""

from __future__ import annotations

import os
import re
import time
import requests
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class AutocompleteRequest(BaseModel):
    partial: str                        # what the user has typed so far
    session_context: str = ""           # last assistant reply (1-2 sentences max)
    available_docs: List[str] = []      # filenames of uploaded documents
    max_tokens: int = 30                # keep small for speed


class AutocompleteResponse(BaseModel):
    completion: str                     # only the suffix to append, NOT the full string
    full: str                           # partial + completion (convenience)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an autocomplete assistant for a document Q&A chatbot. "
    "Your ONLY job is to complete the user's partial query into a natural, specific question. "
    "Rules:\n"
    "- Output ONLY the completion suffix — the words that come AFTER what the user typed.\n"
    "- Do NOT repeat any words the user already typed.\n"
    "- Do NOT answer the question. Do NOT add explanations.\n"
    "- Keep the completion short: 3–10 words maximum.\n"
    "- Match the user's language exactly (French → French, Arabic → Arabic, etc.).\n"
    "- If the partial is already a complete question, output a single space then nothing (empty).\n"
    "- Never add quotation marks around the completion."
)


# ─────────────────────────────────────────────────────────────────────────────
# CF WORKER CALL
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(partial: str, session_context: str, available_docs: List[str]) -> str:
    docs_hint = ""
    if available_docs:
        docs_hint = f"\nAvailable documents: {', '.join(available_docs[:8])}"

    ctx_hint = ""
    if session_context:
        ctx_hint = f"\nLast assistant reply (for context): {session_context[:300]}"

    return (
        f"Complete this partial query into a natural question.{docs_hint}{ctx_hint}\n\n"
        f'Partial query: "{partial}"\n\n'
        f"Output ONLY the completion suffix (words AFTER what was typed), nothing else:"
    )


def _call_cf_autocomplete(
    partial: str,
    session_context: str,
    available_docs: List[str],
    max_tokens: int = 30,
) -> str:
    """
    Call the LLM for inline autocomplete (CF primary, Groq fallback).
    Returns only the completion suffix. Empty string on any failure.
    """
    from llm_client import call_llm

    if len(partial.strip()) < 3:
        return ""

    prompt = _build_prompt(partial, session_context, available_docs)

    raw = call_llm(
        prompt=prompt,
        system_prompt=_SYSTEM,
        history=[],
        max_tokens=max_tokens,
        temperature=0.0,
        task="classify",
    )
    if not raw:
        return ""
    return _clean_completion(raw.strip(), partial)


def _clean_completion(raw: str, partial: str) -> str:
    """
    Sanitize the model output so the frontend always gets a clean suffix.

    The model occasionally repeats part of the partial or wraps in quotes.
    We strip those so the frontend can blindly append the result.
    """
    if not raw:
        return ""

    # Strip surrounding quotes the model sometimes adds
    completion = raw.strip().strip('"').strip("'").strip()

    # If the model returned the full sentence (partial + completion), strip the partial
    partial_lower = partial.lower().rstrip()
    comp_lower = completion.lower()
    if comp_lower.startswith(partial_lower):
        completion = completion[len(partial):].lstrip()

    # Remove any trailing junk: "..." or just whitespace
    completion = completion.rstrip(".…").strip()

    # Hard length gate — never show a completion longer than 80 chars
    if len(completion) > 80:
        # Truncate at last word boundary
        completion = completion[:80].rsplit(" ", 1)[0]

    # If empty or only punctuation after cleaning, return nothing
    if not completion or re.fullmatch(r'[\s\W]+', completion):
        return ""

    # Ensure the completion starts with a space if the partial doesn't end with one
    if partial and not partial.endswith(" ") and not completion.startswith(" "):
        completion = " " + completion

    return completion


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(req: AutocompleteRequest) -> AutocompleteResponse:
    """
    POST /autocomplete
    Body: { "partial": "what is the budget", "session_context": "...", "available_docs": [...] }
    Returns: { "completion": " for phase 2?", "full": "what is the budget for phase 2?" }

    The frontend should debounce calls (300ms recommended) and only show the
    ghost text when the user has typed ≥ 4 characters without submitting.
    """
    partial = req.partial.strip()

    # Short-circuit: don't call CF for empty or very short input
    if len(partial) < 4:
        return AutocompleteResponse(completion="", full=partial)

    # Don't complete already-complete questions (ends with ? or .)
    if partial.endswith(("?", ".")):
        return AutocompleteResponse(completion="", full=partial)

    t0 = time.time()
    completion = _call_cf_autocomplete(
        partial,
        req.session_context,
        req.available_docs,
        req.max_tokens,
    )
    print(f"✏️  autocomplete ({time.time()-t0:.2f}s): '{partial}' → '{completion}'")

    full = partial + completion if completion else partial
    return AutocompleteResponse(completion=completion, full=full)


@router.post("/autocomplete/stream")
async def autocomplete_stream(req: AutocompleteRequest):
    """
    POST /autocomplete/stream
    Same body as /autocomplete but returns text/event-stream.

    Since the CF Worker doesn't stream, we simulate streaming by splitting
    the completion into words and emitting them with a tiny delay.
    This gives the frontend a typewriter effect for the ghost text.

    SSE format:
        data: word \\n\\n
        ...
        data: [DONE]\\n\\n
    """
    partial = req.partial.strip()

    async def _generate():
        if len(partial) < 4 or partial.endswith(("?", ".")):
            yield "data: [DONE]\n\n"
            return

        completion = _call_cf_autocomplete(
            partial,
            req.session_context,
            req.available_docs,
            req.max_tokens,
        )

        if not completion:
            yield "data: [DONE]\n\n"
            return

        # Emit word by word so the ghost text appears to type itself
        words = completion.split(" ")
        for i, word in enumerate(words):
            token = ((" " if i > 0 else "") + word) if not (i == 0 and completion.startswith(" ")) else (" " + word if i == 0 else " " + word)
            yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )