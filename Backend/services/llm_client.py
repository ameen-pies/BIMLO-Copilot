"""
llm_client.py — Shared LLM gateway for all Bimlo services

Primary:  Cloudflare Workers AI proxy  (CF_API_KEY / CF_API_URL)
Fallback: Groq API                     (GROQ_API_KEY)

Fallback triggers automatically on:
  - CF returning any non-200 status (including 500 quota errors like 4006)
  - CF connection timeout / network error
  - CF_API_KEY not set

All services import `call_llm()` from here instead of each having their
own CF-only implementation. The fallback is therefore universal — every
agent, judge, suggester, and autocompleter gets it for free.
"""

from __future__ import annotations

import os
import time
import requests
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CF_DEFAULT_URL     = "https://bimloapi.medhelaliamin125.workers.dev"
_GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"   # mirrors CF primary
_GROQ_MODEL_FAST    = "llama-3.1-8b-instant"       # mirrors CF fast (max_tokens ≤ 50)


# ---------------------------------------------------------------------------
# Internal: Groq call
# ---------------------------------------------------------------------------

_groq_fallback_logged = False   # suppress repeat fallback lines — logged once per process

def _call_groq(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    reason: str,
) -> str:
    global _groq_fallback_logged
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("⚠️  llm_client: no provider available (CF down, GROQ_API_KEY not set)")
        return ""

    model = _GROQ_MODEL_FAST if max_tokens <= 50 else _GROQ_MODEL_PRIMARY
    if not _groq_fallback_logged:
        print(f"⚡ llm_client: routing via Groq [{model}]")
        _groq_fallback_logged = True

    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(_GROQ_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                return raw if isinstance(raw, str) else str(raw)
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"⚠️  llm_client: Groq {resp.status_code}: {resp.text[:120]}")
                break
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"⚠️  llm_client: Groq request failed — {e}")
    return ""


# ---------------------------------------------------------------------------
# Public: unified LLM call
# ---------------------------------------------------------------------------

_cf_error_logged = False   # log CF errors once, not on every call

def call_llm(
    prompt: str,
    system_prompt: str = "",
    history: Optional[List[Dict]] = None,
    max_tokens: int = 1200,
    temperature: float = 0.3,
    task: str = "synthesise",
) -> str:
    """
    Send a prompt to the LLM. CF Workers AI is tried first; Groq is the
    automatic fallback on any failure.

    Args:
        prompt:        The current user turn / instruction.
        system_prompt: Optional system instruction.
        history:       Prior [{role, content}] turns (capped by worker at 10).
        max_tokens:    Token budget. ≤50 → fast model on both providers.
        temperature:   Sampling temperature.
        task:          Hint for the CF worker (synthesise / plan / classify …).

    Returns:
        Generated text string, or "" if both providers fail.
    """
    cf_key = os.getenv("CF_API_KEY", "").strip()
    cf_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)

    # Build the OpenAI-style messages array (used as fallback to Groq directly)
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            messages.append(h)
    messages.append({"role": "user", "content": prompt})

    # ── Try CF worker ────────────────────────────────────────────────────────
    if cf_key:
        payload = {
            "prompt":       prompt,
            "systemPrompt": system_prompt,
            "history":      history or [],
            "max_tokens":   max_tokens,
            "temperature":  temperature,
            "task":         task,
        }
        headers = {
            "Authorization": f"Bearer {cf_key}",
            "Content-Type":  "application/json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(cf_url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    raw = resp.json().get("response") or ""
                    if isinstance(raw, list):
                        import json as _json
                        raw = _json.dumps(raw)
                    return raw if isinstance(raw, str) else str(raw)
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # 500, 503, quota errors — fall through to Groq
                    global _cf_error_logged
                    if not _cf_error_logged:
                        print(f"⚠️  llm_client: CF {resp.status_code} — falling back to Groq")
                        _cf_error_logged = True
                    reason = f"CF returned {resp.status_code}: {resp.text[:80]}"
                    return _call_groq(messages, max_tokens, temperature, reason)
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return _call_groq(messages, max_tokens, temperature, f"CF exception: {e}")

        # 429 exhausted all retries
        return _call_groq(messages, max_tokens, temperature, "CF rate limited after 3 retries")

    # CF key not set — go straight to Groq
    return _call_groq(messages, max_tokens, temperature, "CF_API_KEY not set")


# ---------------------------------------------------------------------------
# Health check — used by _setup() methods so startup doesn't crash the app
# ---------------------------------------------------------------------------

def check_llm_available() -> tuple[bool, str]:
    """
    Ping whichever provider is reachable.
    Returns (is_available, provider_name).
    Called at startup — never raises.
    """
    cf_key = os.getenv("CF_API_KEY", "").strip()
    cf_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)

    if cf_key:
        try:
            resp = requests.post(
                cf_url,
                headers={"Authorization": f"Bearer {cf_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Cloudflare Workers AI"
            # CF is reachable but quota hit — Groq will take over at call time
            print(f"⚠️  llm_client: CF responded {resp.status_code} — will fall back to Groq")
        except Exception as e:
            print(f"⚠️  llm_client: CF unreachable ({e}) — will fall back to Groq")

    # Check Groq
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        return True, "Groq (fallback)"

    return False, "none"