"""
voice_call.py — TTS + call-session endpoints for the /call page.

Routes
──────
POST /tts              — convert text to speech via Groq, returns audio/mpeg
POST /call/query       — thin wrapper around /query-stream for the call page
                         (same RAG engine, same session memory, dedicated route)

Groq TTS model: playai-tts  (or playai-tts-arabic for AR)
Voice: Aaliyah-PlayAI  — warm, clear female English voice
"""

from __future__ import annotations

import os
import json
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

router = APIRouter(tags=["voice_call"])

# ── Groq TTS constants ────────────────────────────────────────────────────────
_GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
_GROQ_TTS_URL   = "https://api.groq.com/openai/v1/audio/speech"
_TTS_MODEL      = "playai-tts"
# Aaliyah: warm, clear, natural female English voice from Play.ai
_TTS_VOICE      = "Aaliyah-PlayAI"
_TTS_FORMAT     = "mp3"


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    text:   str
    voice:  Optional[str] = None   # override voice if needed
    speed:  Optional[float] = 1.0  # 0.5–2.0


# ═══════════════════════════════════════════════════════════════════════════════
# TTS  —  POST /tts
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert text to speech using Groq playai-tts.
    Returns audio/mpeg bytes — the frontend plays them directly.

    The response streams back so the first audio bytes arrive quickly
    even for long responses; the browser starts playing immediately.
    """
    if not _GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY not configured — TTS unavailable")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")

    # Hard-cap at 4096 chars (Groq TTS limit)
    if len(text) > 4096:
        text = text[:4093] + "…"

    voice = (req.voice or _TTS_VOICE).strip() or _TTS_VOICE
    speed = max(0.5, min(2.0, req.speed or 1.0))

    payload = {
        "model":           _TTS_MODEL,
        "input":           text,
        "voice":           voice,
        "response_format": _TTS_FORMAT,
        "speed":           speed,
    }

    headers = {
        "Authorization": f"Bearer {_GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    # Stream the response back so playback starts as bytes arrive
    async def _audio_stream():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", _GROQ_TTS_URL,
                                     json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    detail = body.decode(errors="replace")[:300]
                    raise HTTPException(resp.status_code,
                                        f"Groq TTS error: {detail}")
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk

    return StreamingResponse(
        _audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control":               "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AVAILABLE VOICES  —  GET /tts/voices
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/tts/voices")
async def list_voices():
    """Return the curated list of female English voices the call page offers."""
    return {
        "voices": [
            {"id": "Aaliyah-PlayAI",  "label": "Aaliyah",  "description": "Warm, natural — default"},
            {"id": "Arista-PlayAI",   "label": "Arista",   "description": "Clear, professional"},
            {"id": "Celeste-PlayAI",  "label": "Celeste",  "description": "Soft, friendly"},
            {"id": "Deedee-PlayAI",   "label": "Deedee",   "description": "Energetic, upbeat"},
            {"id": "Gail-PlayAI",     "label": "Gail",     "description": "Mature, authoritative"},
            {"id": "Nia-PlayAI",      "label": "Nia",      "description": "Calm, reassuring"},
        ],
        "default": _TTS_VOICE,
    }
