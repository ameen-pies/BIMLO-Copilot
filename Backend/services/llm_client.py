"""
llm_client.py — Shared LLM gateway for all Bimlo services

Priority chain:
  1. Cloudflare Workers AI — Primary   (CF_API_KEY  / CF_API_URL)
  2. Cloudflare Workers AI — Backup    (CF_BACKUP_API_KEY / CF_BACKUP_URL)
  3. Groq API              — Last resort (GROQ_API_KEY)

Quota-aware cooldown:
  When a CF worker returns a 4006 (daily quota exhausted), it is flagged
  in-process and skipped for 12 hours. After 12 hours it is tried once —
  if it fails again the 12h clock resets; if it succeeds the flag clears.
  This means zero wasted round-trips to a known-dead worker within a session.

Fallback triggers automatically on:
  - CF returning any non-200 status (including 500 quota errors like 4006)
  - CF connection timeout / network error
  - CF_API_KEY not set

Env vars:
  CF_API_KEY         — primary worker bearer token
  CF_API_URL         — primary worker URL       (default: bimloapi.medhelaliamin125.workers.dev)
  CF_BACKUP_URL      — backup worker URL        (default: bimlo.amepies3.workers.dev)
  CF_BACKUP_API_KEY  — backup worker token      (falls back to CF_API_KEY if omitted)
  GROQ_API_KEY       — Groq last-resort key
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
_CF_BACKUP_URL      = "https://bimlo.amepies3.workers.dev/"
_GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL_PRIMARY = "llama-3.3-70b-versatile"
_GROQ_MODEL_FAST    = "llama-3.1-8b-instant"

_COOLDOWN_SECONDS   = 12 * 60 * 60   # 12 hours


# ---------------------------------------------------------------------------
# Quota cooldown tracker
# ---------------------------------------------------------------------------
# Tracks per-worker quota exhaustion. Structure:
#   { "primary": <unix timestamp when cooldown expires>,
#     "backup":  <unix timestamp when cooldown expires> }
# A value of 0 (or missing) means the worker is considered available.

_quota_cooldown: Dict[str, float] = {}


def _is_in_cooldown(label: str) -> bool:
    """Return True if this worker is still within its cooldown window."""
    expires = _quota_cooldown.get(label, 0.0)
    if expires == 0.0:
        return False
    if time.time() < expires:
        remaining_h = (expires - time.time()) / 3600
        print(f"⏸️  llm_client: CF {label} in quota cooldown ({remaining_h:.1f}h remaining) — skipping")
        return True
    # Cooldown expired — clear it and allow one probe attempt
    print(f"🔄 llm_client: CF {label} cooldown expired — probing")
    _quota_cooldown[label] = 0.0
    return False


def _set_cooldown(label: str) -> None:
    """Flag a worker as quota-exhausted for the next 12 hours."""
    expires = time.time() + _COOLDOWN_SECONDS
    _quota_cooldown[label] = expires
    print(f"🚫 llm_client: CF {label} flagged — quota exhausted, skipping for 12h")


def _is_quota_error(response_text: str) -> bool:
    """Detect a Cloudflare 4006 (daily quota exhausted) error in the response body."""
    return "4006" in response_text or "daily free allocation" in response_text.lower()


# ---------------------------------------------------------------------------
# Internal: single CF worker call
# ---------------------------------------------------------------------------

def _call_cf_worker(
    url: str,
    api_key: str,
    payload: dict,
    label: str,
) -> tuple[str | None, str | None]:
    """
    Attempt one CF worker. Respects cooldown — returns (None, reason) immediately
    if the worker is still in its quota cooldown window.

    Returns:
        (text, None)       on success
        (None, reason)     on failure — caller moves to next provider
    """
    if _is_in_cooldown(label):
        return None, f"CF {label} in cooldown"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                raw = resp.json().get("response") or ""
                if isinstance(raw, list):
                    import json as _json
                    raw = _json.dumps(raw)
                # Clear any stale cooldown on success
                _quota_cooldown[label] = 0.0
                return (raw if isinstance(raw, str) else str(raw)), None

            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            else:
                reason = f"CF {label} returned {resp.status_code}: {resp.text[:80]}"
                # 4006 = daily quota exhausted — start cooldown immediately
                if _is_quota_error(resp.text):
                    _set_cooldown(label)
                else:
                    print(f"⚠️  llm_client: {reason}")
                return None, reason

        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            reason = f"CF {label} exception: {e}"
            print(f"⚠️  llm_client: {reason}")
            return None, reason

    reason = f"CF {label} rate-limited after 3 retries"
    print(f"⚠️  llm_client: {reason}")
    return None, reason


# ---------------------------------------------------------------------------
# Internal: Groq call (last resort)
# ---------------------------------------------------------------------------

_groq_fallback_logged = False

def _call_groq(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    reason: str,
) -> str:
    global _groq_fallback_logged
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        print("⚠️  llm_client: no provider available (both CF workers down, GROQ_API_KEY not set)")
        return ""

    model = _GROQ_MODEL_FAST if max_tokens <= 50 else _GROQ_MODEL_PRIMARY
    if not _groq_fallback_logged:
        print(f"⚡ llm_client: both CF workers unavailable — routing via Groq [{model}]")
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

def call_llm(
    prompt: str,
    system_prompt: str = "",
    history: Optional[List[Dict]] = None,
    max_tokens: int = 1200,
    temperature: float = 0.3,
    task: str = "synthesise",
) -> str:
    """
    Send a prompt to the LLM.

    Priority: CF Primary → CF Backup → Groq
    Quota-exhausted workers are skipped for 12h automatically.

    Args:
        prompt:        The current user turn / instruction.
        system_prompt: Optional system instruction.
        history:       Prior [{role, content}] turns (capped by worker at 10).
        max_tokens:    Token budget. ≤50 → fast model on both providers.
        temperature:   Sampling temperature.
        task:          Hint for the CF worker (synthesise / plan / classify …).

    Returns:
        Generated text string, or "" if all three providers fail.
    """
    messages: List[Dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            messages.append(h)
    messages.append({"role": "user", "content": prompt})

    cf_payload = {
        "prompt":       prompt,
        "systemPrompt": system_prompt,
        "history":      history or [],
        "max_tokens":   max_tokens,
        "temperature":  temperature,
        "task":         task,
    }

    # ── 1. CF Primary ────────────────────────────────────────────────────────
    cf_primary_key = os.getenv("CF_API_KEY", "").strip()
    cf_primary_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)

    if cf_primary_key:
        text, reason = _call_cf_worker(cf_primary_url, cf_primary_key, cf_payload, "primary")
        if text is not None:
            return text
        if "cooldown" not in (reason or ""):
            print(f"⚠️  llm_client: CF primary failed ({reason}) — trying backup worker")
    else:
        print("⚠️  llm_client: CF_API_KEY not set — skipping primary, trying backup")

    # ── 2. CF Backup ─────────────────────────────────────────────────────────
    cf_backup_key = os.getenv("CF_BACKUP_API_KEY", cf_primary_key).strip()
    cf_backup_url = os.getenv("CF_BACKUP_URL", _CF_BACKUP_URL)

    if cf_backup_key:
        text, reason = _call_cf_worker(cf_backup_url, cf_backup_key, cf_payload, "backup")
        if text is not None:
            print("✅ llm_client: CF backup worker answered")
            return text
        if "cooldown" not in (reason or ""):
            print(f"⚠️  llm_client: CF backup also failed ({reason}) — falling back to Groq")
    else:
        reason = "no CF key available for backup worker"
        print(f"⚠️  llm_client: {reason}")

    # ── 3. Groq (last resort) ────────────────────────────────────────────────
    return _call_groq(messages, max_tokens, temperature, reason or "both CF workers unavailable")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_llm_available() -> tuple[bool, str]:
    """
    Ping providers in priority order and return the first reachable one.
    Cooldown state is respected — a worker in cooldown is reported as unavailable.
    Returns (is_available, provider_name). Never raises.
    """
    cf_primary_key = os.getenv("CF_API_KEY", "").strip()
    cf_primary_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)

    if cf_primary_key and not _is_in_cooldown("primary"):
        try:
            resp = requests.post(
                cf_primary_url,
                headers={"Authorization": f"Bearer {cf_primary_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Cloudflare Workers AI (primary)"
            if _is_quota_error(resp.text):
                _set_cooldown("primary")
            else:
                print(f"⚠️  llm_client: CF primary responded {resp.status_code} — checking backup")
        except Exception as e:
            print(f"⚠️  llm_client: CF primary unreachable ({e}) — checking backup")

    cf_backup_key = os.getenv("CF_BACKUP_API_KEY", cf_primary_key).strip()
    cf_backup_url = os.getenv("CF_BACKUP_URL", _CF_BACKUP_URL)

    if cf_backup_key and not _is_in_cooldown("backup"):
        try:
            resp = requests.post(
                cf_backup_url,
                headers={"Authorization": f"Bearer {cf_backup_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Cloudflare Workers AI (backup)"
            if _is_quota_error(resp.text):
                _set_cooldown("backup")
            else:
                print(f"⚠️  llm_client: CF backup responded {resp.status_code} — checking Groq")
        except Exception as e:
            print(f"⚠️  llm_client: CF backup unreachable ({e}) — checking Groq")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        return True, "Groq (last resort)"

    return False, "none"