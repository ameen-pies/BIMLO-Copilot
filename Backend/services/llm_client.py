"""
llm_client.py — Shared LLM gateway for all Bimlo services

User-selectable providers:
  Configured in providers.json via provider_manager.

When preferred_provider is set, that provider is tried first. If it fails,
the call falls through to the standard priority chain so responses are never lost.

Standard priority chain (when no preference is set):
  Driven by provider_manager.get_available_providers() sorted by fallback_order.

Env vars:
  See providers.json for each provider's api_key_env and fallback_api_key_env.
"""

from __future__ import annotations

import time
import requests
from typing import List, Dict, Optional

from provider_manager import provider_manager, ProviderConfig

# ---------------------------------------------------------------------------
# Circuit breaker — per-provider trip state
# ---------------------------------------------------------------------------
# When a provider returns a fatal quota/auth error (not a transient 5xx),
# we trip its breaker so every subsequent call skips it immediately without
# a network round-trip.  Resets automatically after RESET_AFTER_SECONDS so
# the process heals itself if the quota refills (e.g. midnight UTC for CF).

import threading as _threading
import time as _time

_FATAL_CF_CODES = {
    "4006",  # daily quota exhausted
    "4007",  # model not available
    "4012",  # subscription limit
}
_CIRCUIT_RESET_AFTER = 3600  # 1 hour — CF daily quotas refill at midnight UTC

class _CircuitBreaker:
    def __init__(self, name: str):
        self.name       = name
        self._tripped   = False
        self._tripped_at: float = 0.0
        self._lock      = _threading.Lock()

    def trip(self, reason: str) -> None:
        with self._lock:
            if not self._tripped:
                self._tripped    = True
                self._tripped_at = _time.time()
                print(f"🔴 llm_client: circuit breaker OPEN for '{self.name}' — {reason} "
                      f"(will retry after {_CIRCUIT_RESET_AFTER // 60}min)")

    def is_open(self) -> bool:
        with self._lock:
            if not self._tripped:
                return False
            if _time.time() - self._tripped_at > _CIRCUIT_RESET_AFTER:
                self._tripped = False
                print(f"🟢 llm_client: circuit breaker RESET for '{self.name}' — retrying")
                return False
            return True

# Dynamic circuit breakers keyed by provider id
_circuit_breakers: Dict[str, _CircuitBreaker] = {}

def _get_breaker(provider_id: str) -> _CircuitBreaker:
    if provider_id not in _circuit_breakers:
        _circuit_breakers[provider_id] = _CircuitBreaker(provider_id)
    return _circuit_breakers[provider_id]


def _is_fatal_cf_error(response_text: str) -> bool:
    """Return True if the CF error body contains a known fatal (non-transient) code."""
    return any(code in response_text for code in _FATAL_CF_CODES)


class AllProvidersRateLimited(Exception):
    """Raised when every available provider has been rate-limited (429)."""
    pass


# ---------------------------------------------------------------------------
# Internal: single CF worker call
# ---------------------------------------------------------------------------

def _call_cf_worker(
    provider: ProviderConfig,
    payload: dict,
) -> tuple[str | None, str | None]:
    """
    Attempt one CF worker and return the response text on success.

    Returns:
        (text, None)       on success
        (None, reason)     on failure — caller moves to next provider

    If the response contains a fatal quota/auth error code, the breaker
    is tripped so future calls skip this provider immediately.
    """
    url = provider.api_url
    api_key = provider_manager.get_api_key(provider)
    label = provider.name
    breaker = _get_breaker(provider.id)

    # No key configured — skip entirely
    if not api_key:
        return None, f"CF {label} — no API key configured"

    # Fast-path: skip the network call entirely if the breaker is open
    if breaker.is_open():
        reason = f"CF {label} circuit breaker open — skipping"
        return None, reason

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 45))

            if resp.status_code == 200:
                raw = resp.json().get("response") or ""
                if isinstance(raw, list):
                    import json as _json
                    raw = _json.dumps(raw)
                return (raw if isinstance(raw, str) else str(raw)), None

            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt, 2))
                continue

            else:
                reason = f"CF {label} returned {resp.status_code}: {resp.text[:80]}"
                print(f"⚠️  llm_client: {reason}")
                # Trip the breaker for fatal quota/auth errors — no point retrying
                if _is_fatal_cf_error(resp.text):
                    breaker.trip(reason)
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
# Internal: Groq call
# ---------------------------------------------------------------------------

_groq_fallback_logged = False

def _call_groq(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    reason: str,
) -> tuple[str | None, str | None]:
    """
    Call Groq API. Returns (text, None) on success or (None, reason) on failure.
    """
    global _groq_fallback_logged
    provider = provider_manager.get_provider("groq")
    if not provider:
        return None, "groq provider not configured"

    groq_key = provider_manager.get_api_key(provider)
    if not groq_key:
        return None, "GROQ_API_KEY not set"

    model = provider.fast_model if (max_tokens <= 50 and provider.fast_model) else provider.model
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
            resp = requests.post(provider.api_url, headers=headers, json=payload, timeout=(10, 45))
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                return (raw if isinstance(raw, str) else str(raw)), None
            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt, 2))
            else:
                reason_str = f"Groq {resp.status_code}: {resp.text[:120]}"
                print(f"⚠️  llm_client: {reason_str}")
                return None, reason_str
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                reason_str = f"Groq request failed — {e}"
                print(f"⚠️  llm_client: {reason_str}")
                return None, reason_str
    return None, "Groq rate-limited after 3 retries"


# ---------------------------------------------------------------------------
# Internal: NVIDIA NIM call
# ---------------------------------------------------------------------------

def _call_nvidia(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    reason: str,
) -> tuple[str | None, str | None]:
    """
    Call NVIDIA NIM endpoint via the OpenAI-compatible REST API.
    Returns (text, None) on success or (None, reason) on failure.
    """
    provider = provider_manager.get_provider("nvidia")
    if not provider:
        return None, "nvidia provider not configured"

    nvidia_key = provider_manager.get_api_key(provider)
    if not nvidia_key:
        return None, "NVIDIA_API_KEY not set"

    payload = {
        "model":       provider.model,
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
    print(f"🟢 [llm_client] NVIDIA NIM → model={provider.model} | key={masked_key} | reason={reason}")

    for attempt in range(2):  # 2 attempts max — NVIDIA is slow, don't triple-wait
        try:
            # Split timeout: 15s to connect, 60s to read the full response body.
            # A single 90s timeout was hiding slow-connect failures as read hangs.
            resp = requests.post(
                provider.api_url, headers=headers, json=payload,
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
                    return None, "NVIDIA responded 200 but content empty"

                print(f"✅ [llm_client] NVIDIA NIM responded ({len(raw)} chars) — model={provider.model}")
                return raw, None
            elif resp.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"⚠️  llm_client: NVIDIA rate-limited — retrying in {wait}s ({attempt + 1}/2)")
                time.sleep(wait)
            else:
                # Log the FULL error body so failures are never silent
                reason_str = f"NVIDIA returned HTTP {resp.status_code} — {resp.text[:300]}"
                print(f"❌ llm_client: {reason_str}")
                return None, reason_str
        except requests.exceptions.ConnectTimeout:
            print(f"❌ llm_client: NVIDIA connect timeout (15s) on attempt {attempt + 1}")
            if attempt < 1:
                time.sleep(2)
        except requests.exceptions.ReadTimeout:
            reason_str = f"NVIDIA read timeout (60s) on attempt {attempt + 1} — model may be overloaded"
            print(f"❌ llm_client: {reason_str}")
            return None, reason_str  # don't retry a read timeout — it will just hang again
        except Exception as e:
            if attempt < 1:
                time.sleep(1)
            else:
                reason_str = f"NVIDIA request exception — {e}"
                print(f"❌ llm_client: {reason_str}")
                return None, reason_str
    return None, "NVIDIA failed after retries"


# ---------------------------------------------------------------------------
# Internal: OpenRouter call
# ---------------------------------------------------------------------------

def _call_openrouter(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str | None, str | None]:
    """
    Call OpenRouter API. Returns (text, None) on success or (None, reason) on failure.
    """
    provider = provider_manager.get_provider("openrouter")
    if not provider:
        return None, "openrouter provider not configured"

    or_key = provider_manager.get_api_key(provider)
    if not or_key:
        return None, "OPENROUTER_API_KEY not set"

    payload = {
        "model":       provider.model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {or_key}",
        "Content-Type":  "application/json",
    }
    for attempt in range(3):
        try:
            resp = requests.post(provider.api_url, headers=headers, json=payload, timeout=(10, 45))
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                result = raw if isinstance(raw, str) else str(raw)
                if result:
                    print(f"✅ llm_client: OpenRouter responded ({len(result)} chars) — model={provider.model}")
                else:
                    print("⚠️  llm_client: OpenRouter responded 200 but content empty")
                return result, None
            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt, 2))
            else:
                reason_str = f"OpenRouter {resp.status_code}: {resp.text[:120]}"
                print(f"⚠️  llm_client: {reason_str}")
                return None, reason_str
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                reason_str = f"OpenRouter request failed — {e}"
                print(f"⚠️  llm_client: {reason_str}")
                return None, reason_str
    return None, "OpenRouter rate-limited after 3 retries"


# ---------------------------------------------------------------------------
# Internal: dispatch a single provider call
# ---------------------------------------------------------------------------

def _try_provider(
    provider: ProviderConfig,
    messages: List[Dict],
    cf_payload: dict,
    max_tokens: int,
    temperature: float,
    last_reason: str,
) -> tuple[str | None, str | None]:
    """
    Dispatch a call to the appropriate handler based on provider type.
    Returns (text, None) on success or (None, reason) on failure.
    """
    if provider.is_cf_worker:
        return _call_cf_worker(provider, cf_payload)

    # OpenAI-compatible providers
    if provider.id == "groq":
        return _call_groq(messages, max_tokens, temperature, last_reason)
    elif provider.id == "nvidia":
        return _call_nvidia(messages, max_tokens, temperature, last_reason)
    elif provider.id == "openrouter":
        return _call_openrouter(messages, max_tokens, temperature)
    else:
        # Future OpenAI-compatible providers — generic path
        api_key = provider_manager.get_api_key(provider)
        if not api_key:
            return None, f"{provider.api_key_env} not set"

        model = provider.fast_model if (max_tokens <= 50 and provider.fast_model) else provider.model
        payload = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        for attempt in range(3):
            try:
                resp = requests.post(provider.api_url, headers=headers, json=payload, timeout=(10, 45))
                if resp.status_code == 200:
                    raw = resp.json()["choices"][0]["message"]["content"]
                    result = raw if isinstance(raw, str) else str(raw)
                    return result, None
                elif resp.status_code == 429:
                    time.sleep(min(2 ** attempt, 2))
                else:
                    return None, f"{provider.name} {resp.status_code}: {resp.text[:120]}"
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    return None, f"{provider.name} request failed — {e}"
        return None, f"{provider.name} rate-limited after 3 retries"


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

    Priority: preferred_provider (if set) → fallback chain by fallback_order.
    Fallback is always attempted if the preferred provider fails.

    Args:
        prompt:             The current user turn / instruction.
        system_prompt:      Optional system instruction.
        history:            Prior [{role, content}] turns (capped by worker at 10).
        max_tokens:         Token budget. <=50 triggers fast model where available.
        temperature:        Sampling temperature.
        task:               Hint for the CF worker (synthesise / plan / classify ...).
        preferred_provider: Provider id (e.g. "cf_primary", "groq", "nvidia").
                            If set, that provider is tried first; falls back on failure.

    Returns:
        Generated text string, or "" if all providers fail.

    Raises:
        AllProvidersRateLimited if every available provider returns 429.
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

    print(f"🧠 llm_client: call_llm preferred_provider={preferred_provider or 'auto'} max_tokens={max_tokens} task={task}")

    # Get available providers sorted by fallback_order
    available = provider_manager.get_available_providers()
    if not available:
        print("⚠️  llm_client: no providers available (no API keys configured)")
        return ""

    # If preferred_provider is set, move it to the front of the list
    if preferred_provider:
        preferred = provider_manager.get_provider(preferred_provider)
        if preferred and preferred in available:
            available = [preferred] + [p for p in available if p.id != preferred_provider]
        elif preferred:
            # Provider exists in config but has no API key — still try it first
            available = [preferred] + available
            print(f"⚠️  llm_client: preferred provider '{preferred_provider}' has no API key configured")
        else:
            print(f"⚠️  llm_client: unknown preferred_provider={preferred_provider!r} — using auto chain")

    # Try each provider in order
    last_reason: str = "no providers tried"
    all_rate_limited = True
    tried_any = False

    for provider in available:
        # Skip if circuit breaker is open (CF workers only)
        if provider.is_cf_worker:
            breaker = _get_breaker(provider.id)
            if breaker.is_open():
                print(f"⚠️  llm_client: {provider.name} circuit breaker open — skipping")
                continue

        tried_any = True
        text, reason = _try_provider(
            provider, messages, cf_payload, max_tokens, temperature, last_reason,
        )

        if text is not None:
            return text

        last_reason = reason or f"{provider.name} failed"
        print(f"⚠️  llm_client: {provider.name} failed ({last_reason}) — trying next")

        # Track whether ALL failures were rate-limits (429)
        if reason and "rate-limited" not in reason and "429" not in reason:
            all_rate_limited = False

    # All providers exhausted
    if tried_any and all_rate_limited:
        raise AllProvidersRateLimited("All available providers returned 429")
    return ""


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_llm_available() -> tuple[bool, str]:
    """
    Ping providers in priority order and return the first reachable one.
    Returns (is_available, provider_name). Never raises.
    """
    for provider in provider_manager.get_available_providers():
        api_key = provider_manager.get_api_key(provider)
        if not api_key:
            continue

        # Build a minimal probe payload
        if provider.is_cf_worker:
            probe = {"prompt": "hi", "max_tokens": 5}
        else:
            probe = {
                "model": provider.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            }

        try:
            resp = requests.post(
                provider.api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=probe,
                timeout=10,
            )
            if resp.status_code == 200:
                return True, provider.name
            print(f"⚠️  llm_client: {provider.name} responded {resp.status_code} — checking next")
        except Exception as e:
            print(f"⚠️  llm_client: {provider.name} unreachable ({e}) — checking next")

    return False, "none"
