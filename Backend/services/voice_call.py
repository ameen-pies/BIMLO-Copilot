"""
voice_call.py — TTS + voice-call response pipeline.

TTS provider: ElevenLabs (eleven_flash_v2_5)
─────────────────────────────────────────────
Set ELEVENLABS_API_KEY in your .env file.
Groq chat API is still used for answer rewriting (unchanged).

Routes
──────
POST /tts              → MP3 audio
GET  /tts/voices       → available voice options
POST /call/respond     → SSE: answer_audio | done | error
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["voice_call"])

# ── Groq chat (answer rewriting only — NOT used for TTS) ─────────────────────
_GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_LLM_MODEL     = "llama-3.3-70b-versatile"
_TTS_VOICE     = "hannah"

# ── ElevenLabs TTS ────────────────────────────────────────────────────────────
# Required: add  ELEVENLABS_API_KEY=<your key>  to your .env
_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_MODEL   = "eleven_flash_v2_5"   # ~75ms latency, best for real-time conversation

# Maps the voice picker names the frontend knows → ElevenLabs voice IDs
_VOICE_MAP: dict[str, str] = {
    "hannah": "EXAVITQu4vr4xnSDxMaL",  # Sarah  — warm, natural female
    "diana":  "21m00Tcm4TlvDq8ikWAM",  # Rachel — clear, professional female
    "autumn": "AZnzlk1XvdvUeBnXmlld",  # Domi   — soft, calm female
    "austin": "ErXwobaYiN019PkySvjV",  # Antoni — friendly male
    "daniel": "VR6AewLTigWG4xSOukaG",  # Arnold — clear male
    "troy":   "pNInz6obpgDQGcFmaJgB",  # Adam   — deep male
}


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    text:  str
    voice: Optional[str]   = None
    speed: Optional[float] = 1.0


class CallRespondRequest(BaseModel):
    transcript: str
    rag_answer: str
    voice:      Optional[str]  = None
    session_id: Optional[str]  = None
    voice_mode: Optional[bool] = False   # if True, skip the Groq rewrite — answer already conversational


# ══════════════════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════════════════

async def _tts_bytes(text: str, voice: str) -> bytes:
    """
    Fetch TTS from ElevenLabs.
    Returns MP3 bytes. Raises HTTPException on any failure so callers
    always get a proper HTTP error code — never a silent empty body.
    """
    if not text.strip():
        return b""
    if not _ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set.")

    voice_id = _VOICE_MAP.get(voice, _VOICE_MAP["hannah"])
    url      = _ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    payload = {
        "text":            text[:5000],
        "model_id":        _ELEVENLABS_MODEL,
        "output_format":   "mp3_44100_128",
        "voice_settings":  {"stability": 0.5, "similarity_boost": 0.75},
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(
                url,
                headers={
                    "xi-api-key":   _ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept":       "audio/mpeg",
                },
                json=payload,
            )
            r.raise_for_status()
            mp3 = r.content

        print(f"✅ ElevenLabs TTS ({len(mp3):,} bytes, voice={voice}/{voice_id})")
        return mp3

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response else str(e)
        raise HTTPException(e.response.status_code, f"ElevenLabs TTS error: {detail}") from e
    except Exception as e:
        raise HTTPException(503, f"ElevenLabs TTS failed: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS  (Groq chat — answer rewriting only, not TTS)
# ══════════════════════════════════════════════════════════════════════════════

async def _llm(messages: list[dict], max_tokens: int = 300) -> str:
    if not _GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — needed for answer rewriting")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            _GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       _LLM_MODEL,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": 0.7,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s", "", text)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r">\s*", "", text)
    text = re.sub(r"[-*+]\s", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()


async def _rewrite_for_speech(transcript: str, rag_answer: str) -> str:
    clean = _strip_markdown(rag_answer)
    prompt = (
        "Rewrite this answer for natural spoken phone conversation — "
        "like a knowledgeable friend, not a textbook.\n\nRules:"
        "\n- Same language as the answer"
        "\n- No markdown, bullets, numbered lists, or headers"
        "\n- Short sentences; add '…' after commas and breath points"
        "\n- Use contractions where natural"
        "\n- Never open with 'Certainly', 'Of course', 'Great question', 'Sure'"
        "\n- Max 4 sentences. Be concise."
        "\n- Output ONLY the spoken text, nothing else"
        f"\n\nUser asked: {transcript}"
        f"\n\nOriginal answer: {clean}"
    )
    result = await _llm([{"role": "user", "content": prompt}])
    return result.strip().strip('"').strip("'")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    Convert text → MP3 via ElevenLabs TTS.
    Returns a plain Response (not StreamingResponse) so any provider error
    surfaces as a real HTTP error code the frontend can detect.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 4096:
        text = text[:4093] + "…"
    voice = (req.voice or _TTS_VOICE).strip() or _TTS_VOICE

    mp3 = await _tts_bytes(text, voice)
    return Response(
        content=mp3,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
    )


@router.post("/call/respond")
async def call_respond(req: CallRespondRequest):
    """
    Voice-call answer pipeline — SSE stream.

    Events:
      { "type": "answer_audio", "audio": "<base64 MP3>", "text": "…" }
      { "type": "done" }
      { "type": "error", "message": "…" }
    """
    if not _ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not configured.")

    voice = (req.voice or _TTS_VOICE).strip() or _TTS_VOICE

    async def _pipeline():
        # In voice_mode the query prompt already instructs conversational style,
        # so skip the Groq rewrite (saves ~1–2 s). Otherwise rewrite as before.
        spoken = ""
        if req.voice_mode:
            spoken = _strip_markdown(req.rag_answer).strip()
            print("⚡ voice_mode: skipping Groq rewrite")
        else:
            try:
                spoken = await asyncio.wait_for(
                    _rewrite_for_speech(req.transcript, req.rag_answer),
                    timeout=20,
                )
                spoken = spoken.strip().strip('"').strip("'")
            except Exception as e:
                print(f"⚠️  Rewrite failed ({e}), using raw answer")

        if not spoken:
            spoken = _strip_markdown(req.rag_answer).strip()
        if len(spoken) < 4:
            spoken += "."

        print(f"🔊 TTS ({len(spoken)} chars): {spoken[:80]!r}")

        try:
            audio = await _tts_bytes(spoken[:4096], voice)
            b64   = base64.b64encode(audio).decode()
            yield "data: " + json.dumps({"type": "answer_audio", "audio": b64, "text": spoken}) + "\n\n"
        except HTTPException as he:
            yield "data: " + json.dumps({"type": "error", "message": he.detail}) + "\n\n"
        except Exception as e:
            print(f"❌ TTS error: {e}")
            yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        _pipeline(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/tts/voices")
async def list_voices():
    return {
        "voices": [
            {"id": "hannah",  "label": "Hannah",  "description": "Warm, natural — default"},
            {"id": "diana",   "label": "Diana",   "description": "Clear, professional"},
            {"id": "autumn",  "label": "Autumn",  "description": "Soft, calm"},
            {"id": "austin",  "label": "Austin",  "description": "Male, friendly"},
            {"id": "daniel",  "label": "Daniel",  "description": "Male, clear"},
            {"id": "troy",    "label": "Troy",    "description": "Male, deep"},
        ],
        "default": _TTS_VOICE,
    }