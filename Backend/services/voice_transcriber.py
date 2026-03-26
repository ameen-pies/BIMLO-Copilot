"""
voice_transcriber.py — Voice-to-text transcription endpoint

Accepts an audio blob (webm/ogg/mp4/wav) recorded in the browser,
transcribes it via Groq's Whisper API (whisper-large-v3-turbo),
and returns the transcript so Chat.tsx can inject it into the input field.

Mount in your main FastAPI app:
    from services.voice_transcriber import router as voice_router
    app.include_router(voice_router)

Env vars:
    GROQ_API_KEY        — required, your Groq API key
    WHISPER_LANGUAGE    — optional ISO-639-1 code e.g. "en" (default: auto-detect)
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

_GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_MODEL   = "whisper-large-v3-turbo"   # fast + accurate, free tier friendly
_MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "25"))

_ALLOWED_MIME = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "video/webm",
}

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
    transcript: str
    language:   str   = ""
    duration_s: float = 0.0
    backend:    str   = ""


# ────────────────────────────────────────────────────────────────────────────
# GROQ WHISPER TRANSCRIPTION
# ────────────────────────────────────────────────────────────────────────────

def _transcribe_groq(audio_bytes: bytes, mime_type: str) -> Optional[TranscribeResponse]:
    """
    Send audio to Groq's Whisper endpoint.
    Groq's API is OpenAI-compatible — same multipart format, different base URL + key.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    print(f"🔑 Groq Whisper: key present={bool(api_key)}, prefix={api_key[:8] if api_key else 'MISSING'}")

    if not api_key:
        print("⚠️  Transcription skipped: GROQ_API_KEY not set")
        return None

    ext = _MIME_EXT.get(mime_type, "webm")
    filename    = f"recording.{ext}"
    upload_mime = f"audio/{ext}"   # strip codec suffix — Groq rejects "audio/webm;codecs=opus"

    files = {
        "file": (filename, io.BytesIO(audio_bytes), upload_mime),
    }
    data: dict = {
        "model":           _GROQ_MODEL,
        "response_format": "verbose_json",
    }
    lang = os.getenv("WHISPER_LANGUAGE", "").strip()
    if lang:
        data["language"] = lang

    try:
        resp = requests.post(
            _GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=30,
        )

        if resp.status_code == 200:
            result     = resp.json()
            transcript = (result.get("text") or "").strip()
            if not transcript:
                print("⚠️  Groq Whisper returned empty transcript")
                return None
            print(f"✅ Groq transcript ({len(transcript)} chars): {transcript[:80]}…")
            return TranscribeResponse(
                transcript=transcript,
                language=result.get("language", ""),
                duration_s=float(result.get("duration", 0)),
                backend="groq_whisper",
            )
        else:
            print(f"⚠️  Groq Whisper error {resp.status_code}: {resp.text[:300]}")
            return None

    except Exception as e:
        print(f"⚠️  Groq Whisper request failed: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# ROUTER
# ────────────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio:     UploadFile = File(...),
    mime_type: str        = Form("audio/webm"),
) -> TranscribeResponse:
    """
    POST /transcribe
    multipart/form-data:
      audio     — the recorded audio blob
      mime_type — MIME type string (e.g. "audio/webm;codecs=opus")

    Returns: { "transcript": "...", "language": "en", "duration_s": 4.2, "backend": "groq_whisper" }
    """

    # Strip codec suffix: "audio/webm;codecs=opus" -> "audio/webm"
    base_mime = mime_type.split(";")[0].strip().lower()
    if base_mime not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{base_mime}'. Allowed: {sorted(_ALLOWED_MIME)}",
        )

    audio_bytes = await audio.read()
    max_bytes   = _MAX_AUDIO_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large ({len(audio_bytes) // (1024*1024)} MB). Max: {_MAX_AUDIO_MB} MB.",
        )
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio blob is empty or too short.")

    print(f"🎙️  Received audio: {len(audio_bytes)} bytes, type: {base_mime}")
    t0 = time.time()

    result = _transcribe_groq(audio_bytes, base_mime)

    if result is None:
        key_set = bool(os.getenv("GROQ_API_KEY", "").strip())
        raise HTTPException(
            status_code=503,
            detail=(
                "Transcription failed: GROQ_API_KEY is not set. "
                "Add GROQ_API_KEY=your_key to your .env file and restart the server."
                if not key_set else
                "Transcription failed: Groq Whisper returned an error — check backend logs for details."
            ),
        )

    print(f"⏱️  Transcription done in {time.time() - t0:.2f}s via {result.backend}")
    return result