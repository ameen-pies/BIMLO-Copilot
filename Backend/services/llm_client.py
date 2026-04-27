"""
llm_client.py — Shared LLM gateway for all Bimlo services

User-selectable providers:
  "cf_primary"  — Cloudflare Workers AI primary worker
  "cf_backup"   — Cloudflare Workers AI backup worker
  "groq"        — Groq API (llama-3.3-70b-versatile / llama-3.1-8b-instant)
  "nvidia"      — NVIDIA NIM API (minimaxai/minimax-m2.7)

When preferred_provider is set, that provider is tried first. If it fails,
the call falls through to the standard priority chain so responses are never lost.

Standard priority chain (when no preference is set):
  1. Cloudflare Workers AI — Primary   (CF_API_KEY  / CF_API_URL)
  2. Cloudflare Workers AI — Backup    (CF_BACKUP_API_KEY / CF_BACKUP_URL)
  3. Groq API                          (GROQ_API_KEY)

Env vars:
  CF_API_KEY         — primary worker bearer token
  CF_API_URL         — primary worker URL       (default: bimloapi.medhelaliamin125.workers.dev)
  CF_BACKUP_URL      — backup worker URL        (default: bimlo.amepies3.workers.dev)
  CF_BACKUP_API_KEY  — backup worker token      (falls back to CF_API_KEY if omitted)
  GROQ_API_KEY       — Groq last-resort key
  NVIDIA_API_KEY     — NVIDIA NIM API key (nvapi-...)
"""

from __future__ import annotations

import json
import os
import time
import requests
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CF_DEFAULT_URL      = "https://bimloapi.medhelaliamin125.workers.dev"
_CF_BACKUP_URL       = "https://bimlo.amepies3.workers.dev/"
_GROQ_API_URL        = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
_GROQ_MODEL_FAST     = "llama-3.1-8b-instant"
_NVIDIA_API_URL      = "https://integrate.api.nvidia.com/v1/chat/completions"
_NVIDIA_MODEL        = "minimaxai/minimax-m2.7"

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
    Attempt one CF worker and return the response text on success.

    Returns:
        (text, None)       on success
        (None, reason)     on failure — caller moves to next provider
    """
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
                return (raw if isinstance(raw, str) else str(raw)), None

            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            else:
                reason = f"CF {label} returned {resp.status_code}: {resp.text[:80]}"
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
# Internal: NVIDIA NIM call (deepseek-v4-pro)
# ---------------------------------------------------------------------------

def _call_nvidia(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    reason: str,
) -> str:
    """
    Call NVIDIA NIM endpoint (minimaxai/minimax-m2.7) via the OpenAI-compatible REST API.
    Uses NVIDIA_API_KEY env var. Returns "" on failure so the caller can fall through.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not nvidia_key:
        print("⚠️  llm_client: NVIDIA_API_KEY not set — skipping NVIDIA provider")
        return ""

    payload = {
        "model":       _NVIDIA_MODEL,
        "messages":    messages,
        "max_tokens":  min(max_tokens, 8192),  # MiniMax M2.7 output cap
        "temperature": temperature,
        "top_p":       0.95,
        "stream":      False,
    }
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Content-Type":  "application/json",
    }

    masked_key = nvidia_key[:8] + "..." + nvidia_key[-4:]
    print(f"🟢 [llm_client] NVIDIA NIM → model={_NVIDIA_MODEL} | key={masked_key} | reason={reason}")

    for attempt in range(2):  # 2 attempts max — NVIDIA is slow, don't triple-wait
        try:
            # Split timeout: 15s to connect, 60s to read the full response body.
            # A single 90s timeout was hiding slow-connect failures as read hangs.
            resp = requests.post(
                _NVIDIA_API_URL, headers=headers, json=payload,
                timeout=(15, 60),
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]

                # DeepSeek V4 Pro (thinking disabled) returns content in "content".
                # When thinking is accidentally enabled it fills "reasoning_content"
                # and may leave "content" empty — fall back gracefully.
                raw = msg.get("content") or msg.get("reasoning_content") or ""
                raw = raw.strip()

                if not raw:
                    print(f"⚠️  llm_client: NVIDIA responded 200 but content is empty — full msg: {msg}")
                    return ""

                print(f"✅ [llm_client] NVIDIA NIM responded ({len(raw)} chars) — model={_NVIDIA_MODEL}")
                return raw
            elif resp.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"⚠️  llm_client: NVIDIA rate-limited — retrying in {wait}s ({attempt + 1}/2)")
                time.sleep(wait)
            else:
                # Log the FULL error body so failures are never silent
                print(f"❌ llm_client: NVIDIA returned HTTP {resp.status_code} — {resp.text[:300]}")
                break
        except requests.exceptions.ConnectTimeout:
            print(f"❌ llm_client: NVIDIA connect timeout (15s) on attempt {attempt + 1}")
            if attempt < 1:
                time.sleep(2)
        except requests.exceptions.ReadTimeout:
            print(f"❌ llm_client: NVIDIA read timeout (60s) on attempt {attempt + 1} — model may be overloaded")
            break  # don't retry a read timeout — it will just hang again
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
            else:
                print(f"❌ llm_client: NVIDIA request exception — {e}")
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
    preferred_provider: Optional[str] = None,
) -> str:
    """
    Send a prompt to the LLM.

    Priority: preferred_provider (if set) → CF Primary → CF Backup → Groq
    Fallback is always attempted if the preferred provider fails.

    Args:
        prompt:             The current user turn / instruction.
        system_prompt:      Optional system instruction.
        history:            Prior [{role, content}] turns (capped by worker at 10).
        max_tokens:         Token budget. ≤50 → fast model on both providers.
        temperature:        Sampling temperature.
        task:               Hint for the CF worker (synthesise / plan / classify …).
        preferred_provider: One of "cf_primary", "cf_backup", "groq", "nvidia".
                            If set, that provider is tried first; falls back on failure.
                            "nvidia" uses minimaxai/minimax-m2.7 via NVIDIA NIM.

    Returns:
        Generated text string, or "" if all providers fail.
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

    # Resolve CF keys once — used across all CF routing below
    cf_primary_key = os.getenv("CF_API_KEY", "").strip()
    cf_primary_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)
    cf_backup_key  = os.getenv("CF_BACKUP_API_KEY", cf_primary_key).strip()
    cf_backup_url  = os.getenv("CF_BACKUP_URL", _CF_BACKUP_URL)

    print(f"🧠 llm_client: call_llm preferred_provider={preferred_provider or 'auto'} max_tokens={max_tokens} task={task}")
    last_reason: str = "no providers tried"

    # ── User-preferred provider (tried first) ─────────────────────────────────
    if preferred_provider == "cf_primary":
        if cf_primary_key:
            text, reason = _call_cf_worker(cf_primary_url, cf_primary_key, cf_payload, "primary")
            if text is not None:
                return text
            last_reason = reason or "cf_primary failed"
            print(f"⚠️  llm_client: preferred cf_primary failed ({last_reason}) — falling through to backup")
        else:
            last_reason = "CF_API_KEY not set"
            print(f"⚠️  llm_client: preferred cf_primary requested but {last_reason}")

    elif preferred_provider == "cf_backup":
        if cf_backup_key:
            text, reason = _call_cf_worker(cf_backup_url, cf_backup_key, cf_payload, "backup")
            if text is not None:
                return text
            last_reason = reason or "cf_backup failed"
            print(f"⚠️  llm_client: preferred cf_backup failed ({last_reason}) — falling through")
        else:
            last_reason = "no CF backup key available"
            print(f"⚠️  llm_client: preferred cf_backup requested but {last_reason}")

    elif preferred_provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            result = _call_groq(messages, max_tokens, temperature, "user-selected groq")
            if result:
                return result
            last_reason = "groq returned empty"
            print(f"⚠️  llm_client: preferred groq failed — falling through to CF workers")
        else:
            last_reason = "GROQ_API_KEY not set"
            print(f"⚠️  llm_client: preferred groq requested but {last_reason}")

    elif preferred_provider == "nvidia":
        nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if nvidia_key:
            result = _call_nvidia(messages, max_tokens, temperature, "user-selected nvidia")
            if result:
                return result
            last_reason = "nvidia returned empty or failed"
            print(f"⚠️  llm_client: preferred nvidia failed — falling through to CF workers")
        else:
            last_reason = "NVIDIA_API_KEY not set"
            print(f"⚠️  llm_client: preferred nvidia requested but {last_reason}")

    # ── Standard priority fallback chain ──────────────────────────────────────
    # Each provider is skipped if it was already tried as preferred_provider above.
    # This prevents double-calling and keeps the log honest.

    if preferred_provider and preferred_provider not in ("cf_primary", "cf_backup", "groq", "nvidia"):
        print(f"⚠️  llm_client: unknown preferred_provider={preferred_provider!r} — using auto chain")

    # CF Primary
    if preferred_provider != "cf_primary":
        if cf_primary_key:
            text, reason = _call_cf_worker(cf_primary_url, cf_primary_key, cf_payload, "primary")
            if text is not None:
                return text
            last_reason = reason or "cf_primary failed"
            print(f"⚠️  llm_client: CF primary failed ({last_reason}) — trying backup worker")
        else:
            print("⚠️  llm_client: CF_API_KEY not set — skipping primary, trying backup")

    # CF Backup
    if preferred_provider != "cf_backup":
        if cf_backup_key:
            text, reason = _call_cf_worker(cf_backup_url, cf_backup_key, cf_payload, "backup")
            if text is not None:
                print("✅ llm_client: CF backup worker answered")
                return text
            last_reason = reason or "cf_backup failed"
            print(f"⚠️  llm_client: CF backup also failed ({last_reason}) — falling back to Groq")
        else:
            last_reason = "no CF key available for backup worker"
            print(f"⚠️  llm_client: {last_reason}")

    # Groq — skip if already tried as preferred OR if nvidia was preferred (keep nvidia opt-in only)
    if preferred_provider not in ("groq", "nvidia"):
        result = _call_groq(messages, max_tokens, temperature, last_reason)
        if result:
            return result
        last_reason = "groq also failed"
    elif preferred_provider == "groq":
        pass  # already tried above, skip
    # nvidia preferred but failed → still fall through to Groq as last resort
    elif preferred_provider == "nvidia":
        print(f"⚠️  llm_client: NVIDIA preferred but failed — falling back to Groq as last resort")
        result = _call_groq(messages, max_tokens, temperature, last_reason)
        if result:
            return result

    # NVIDIA is NOT in the automatic fallback chain for non-nvidia requests.
    # It is opt-in only via preferred_provider to avoid burning free-tier quota silently.
    return ""


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_llm_available() -> tuple[bool, str]:
    """
    Ping providers in priority order and return the first reachable one.
    Returns (is_available, provider_name). Never raises.
    """
    cf_primary_key = os.getenv("CF_API_KEY", "").strip()
    cf_primary_url = os.getenv("CF_API_URL", _CF_DEFAULT_URL)

    if cf_primary_key:
        try:
            resp = requests.post(
                cf_primary_url,
                headers={"Authorization": f"Bearer {cf_primary_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Cloudflare Workers AI (primary)"
            print(f"⚠️  llm_client: CF primary responded {resp.status_code} — checking backup")
        except Exception as e:
            print(f"⚠️  llm_client: CF primary unreachable ({e}) — checking backup")

    cf_backup_key = os.getenv("CF_BACKUP_API_KEY", cf_primary_key).strip()
    cf_backup_url = os.getenv("CF_BACKUP_URL", _CF_BACKUP_URL)

    if cf_backup_key:
        try:
            resp = requests.post(
                cf_backup_url,
                headers={"Authorization": f"Bearer {cf_backup_key}", "Content-Type": "application/json"},
                json={"prompt": "hi", "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, "Cloudflare Workers AI (backup)"
            print(f"⚠️  llm_client: CF backup responded {resp.status_code} — checking Groq")
        except Exception as e:
            print(f"⚠️  llm_client: CF backup unreachable ({e}) — checking Groq")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        return True, "Groq (last resort)"

    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if nvidia_key:
        return True, "NVIDIA NIM (deepseek-v4-pro)"

    return False, "none"