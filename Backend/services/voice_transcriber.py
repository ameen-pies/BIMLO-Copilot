"""
voice_transcriber.py — Voice-to-text transcription endpoint

Accepts an audio blob (webm/ogg/mp4/wav) recorded in the browser,
transcribes it via OpenAI Whisper (or falls back to the CF Worker if
a Whisper-compatible endpoint is configured there), and returns the
transcript as plain text so Chat.tsx can inject it into the input field
exactly like a typed message.

Mount in your main FastAPI app:
    from voice_transcriber import router as voice_router
    app.include_router(voice_router)

Env vars:
    OPENAI_API_KEY      — if set, Whisper API is used (best quality)
    CF_API_KEY          — fallback: your existing CF Worker
    CF_API_URL          — fallback CF Worker base URL
    WHISPER_LANGUAGE    — optional ISO-639-1 code, e.g. "en" (default: auto-detect)
    MAX_AUDIO_MB        — reject uploads larger than this (default: 25)
"""

from __future__ import annotations

import os
import io
import time
import requests
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

# ── Load .env (same pattern as suggest.py) ──────────────────────────────────
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
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

_OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
_CF_API_KEY      = os.getenv("CF_API_KEY", "")
_CF_API_URL      = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")
_WHISPER_LANG    = os.getenv("WHISPER_LANGUAGE", "")          # "" = auto-detect
_MAX_AUDIO_MB    = int(os.getenv("MAX_AUDIO_MB", "25"))

# MIME types the browser's MediaRecorder API can produce
_ALLOWED_MIME = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "video/webm",   # Chrome sometimes uses this for webm audio
}

# Map MIME → file extension expected by Whisper
_MIME_EXT = {
    "audio/webm":  "webm",
    "video/webm":  "webm",
    "audio/ogg":   "ogg",
    "audio/mp4":   "mp4",
    "audio/mpeg":  "mp3",
    "audio/wav":   "wav",
    "audio/x-wav": "wav",
    "audio/x-m4a": "m4a",
}


# ────────────────────────────────────────────────────────────────────────────
# RESPONSE MODEL
# ────────────────────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript: str          # clean text ready to inject into the chat input
    language:   str = ""     # detected language code, if available
    duration_s: float = 0.0  # audio duration returned by Whisper, if available
    backend:    str = ""     # "whisper" | "cf_worker" — for debugging


# ────────────────────────────────────────────────────────────────────────────
# WHISPER TRANSCRIPTION  (OpenAI API)
# ────────────────────────────────────────────────────────────────────────────

def _transcribe_whisper(audio_bytes: bytes, mime_type: str) -> Optional[TranscribeResponse]:
    """
    Call OpenAI Whisper API with the raw audio bytes.
    Returns None if the call fails so the caller can try the CF fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    ext = _MIME_EXT.get(mime_type, "webm")
    filename = f"recording.{ext}"

    files = {
        "file": (filename, io.BytesIO(audio_bytes), mime_type),
    }
    data: dict = {
        "model": "whisper-1",
        "response_format": "verbose_json",   # gives us language + duration
    }
    lang = os.getenv("WHISPER_LANGUAGE", "")
    if lang:
        data["language"] = lang

    try:
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            transcript = (result.get("text") or "").strip()
            if not transcript:
                print("⚠️  Whisper returned empty transcript")
                return None
            print(f"✅ Whisper transcript ({len(transcript)} chars): {transcript[:80]}…")
            return TranscribeResponse(
                transcript=transcript,
                language=result.get("language", ""),
                duration_s=float(result.get("duration", 0)),
                backend="whisper",
            )
        else:
            print(f"⚠️  Whisper API error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"⚠️  Whisper request failed: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# CF WORKER FALLBACK  (mirrors pattern from suggest.py / rag_engine.py)
# ────────────────────────────────────────────────────────────────────────────

def _transcribe_cf(audio_bytes: bytes, mime_type: str) -> Optional[TranscribeResponse]:
    """
    Send audio to the CF Worker's /transcribe task.
    The worker is expected to forward it to a Whisper-compatible model.
    Returns None on failure.
    """
    cf_api_key = os.getenv("CF_API_KEY", "")
    cf_api_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")

    if not cf_api_key:
        print("⚠️  voice_transcriber: no CF_API_KEY set, cannot use CF fallback")
        return None

    ext = _MIME_EXT.get(mime_type, "webm")
    filename = f"recording.{ext}"

    try:
        resp = requests.post(
            cf_api_url,
            headers={"Authorization": f"Bearer {cf_api_key}"},
            files={"audio": (filename, io.BytesIO(audio_bytes), mime_type)},
            data={"task": "transcribe", "language": os.getenv("WHISPER_LANGUAGE", "")},
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            transcript = (result.get("response") or result.get("transcript") or "").strip()
            if not transcript:
                print("⚠️  CF Worker returned empty transcript")
                return None
            print(f"✅ CF transcript ({len(transcript)} chars): {transcript[:80]}…")
            return TranscribeResponse(
                transcript=transcript,
                language=result.get("language", ""),
                duration_s=float(result.get("duration", 0)),
                backend="cf_worker",
            )
        else:
            print(f"⚠️  CF Worker transcribe error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"⚠️  CF Worker transcribe request failed: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# ROUTER
# ────────────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    mime_type: str    = Form("audio/webm"),  # sent by the frontend
) -> TranscribeResponse:
    """
    POST /transcribe
    multipart/form-data:
      audio     — the recorded audio blob
      mime_type — MIME type string (e.g. "audio/webm;codecs=opus")

    Returns: { "transcript": "...", "language": "en", "duration_s": 4.2, "backend": "whisper" }

    Transcription priority:
      1. OpenAI Whisper API  (if OPENAI_API_KEY is set)
      2. CF Worker fallback  (if CF_API_KEY is set)
      3. 503 if neither is available
    """

    # ── Validate MIME ────────────────────────────────────────────────────────
    # Strip codec suffix: "audio/webm;codecs=opus" → "audio/webm"
    base_mime = mime_type.split(";")[0].strip().lower()
    if base_mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{base_mime}'. Allowed: {sorted(_ALLOWED_MIME)}",
        )

    # ── Read & size-check ────────────────────────────────────────────────────
    audio_bytes = await audio.read()
    max_bytes = _MAX_AUDIO_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large ({len(audio_bytes) // (1024*1024)} MB). Max: {_MAX_AUDIO_MB} MB.",
        )
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio blob is empty or too short.")

    print(f"🎙️  Received audio: {len(audio_bytes)} bytes, type: {base_mime}")
    t0 = time.time()

    # ── Try backends in order ────────────────────────────────────────────────
    result = _transcribe_whisper(audio_bytes, base_mime)
    if result is None:
        result = _transcribe_cf(audio_bytes, base_mime)

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Transcription failed. Set OPENAI_API_KEY or ensure CF Worker is reachable.",
        )

    print(f"⏱️  Transcription done in {time.time() - t0:.2f}s via {result.backend}")
    return result
