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
_ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_MODEL   = "eleven_flash_v2_5"

_VOICE_MAP: dict[str, str] = {
    "hannah": "EXAVITQu4vr4xnSDxMaL",
    "diana":  "21m00Tcm4TlvDq8ikWAM",
    "autumn": "AZnzlk1XvdvUeBnXmlld",
    "austin": "ErXwobaYiN019PkySvjV",
    "daniel": "VR6AewLTigWG4xSOukaG",
    "troy":   "pNInz6obpgDQGcFmaJgB",
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
    voice_mode: Optional[bool] = False


# ══════════════════════════════════════════════════════════════════════════════
# MARKDOWN → PLAIN TEXT  (aggressive, TTS-safe)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_markdown(text: str) -> str:
    """
    Aggressively convert markdown to clean spoken text.
    Order matters — process block-level first, then inline.
    """
    # ── Block-level ──────────────────────────────────────────────────────────
    # Fenced code blocks (```…``` or ~~~…~~~)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"~~~[\s\S]*?~~~", "", text)

    # ATX headings  (# Heading)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Setext headings (underlined with === or ---)
    text = re.sub(r"^[=-]{2,}\s*$", "", text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Blockquotes
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # Ordered lists  (1. 2. 10.)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Unordered lists  (- * +)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # ── Inline ───────────────────────────────────────────────────────────────
    # Bold+italic  ***text*** / ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)

    # Bold  **text** / __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)

    # Italic  *text* / _text_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    # Inline code  `code`
    text = re.sub(r"`+[^`]*`+", "", text)

    # Links  [label](url) → label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Images  ![alt](url) → alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Reference-style links  [label][ref] → label
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)

    # Footnote / citation markers  [1] [^1]
    text = re.sub(r"\[\^?[\w\d]+\]", "", text)

    # HTML tags (sometimes mixed in)
    text = re.sub(r"<[^>]+>", "", text)

    # Pipe-table rows  (| col | col |)
    text = re.sub(r"\|.*\|", "", text)

    # ── Whitespace cleanup ───────────────────────────────────────────────────
    # Collapse multiple blank lines → single space
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def _ensure_plain(text: str) -> str:
    """
    Final safety pass — if ANY markdown-like symbol survives,
    strip it so ElevenLabs doesn't read it aloud.
    """
    # Remove any remaining bare asterisks or underscores used as emphasis
    text = re.sub(r"(?<!\w)[*_]+(?!\w)", "", text)
    text = re.sub(r"(?<!\w)[*_]+", "", text)
    # Remove stray backticks
    text = text.replace("`", "")
    # Remove stray hash symbols at start of words
    text = re.sub(r"(?<!\w)#+\s*", "", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════════════════

async def _tts_bytes(text: str, voice: str) -> bytes:
    if not text.strip():
        return b""
    if not _ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set.")

    voice_id = _VOICE_MAP.get(voice, _VOICE_MAP["hannah"])
    url      = _ELEVENLABS_TTS_URL.format(voice_id=voice_id)

    payload = {
        "text":           text[:5000],
        "model_id":       _ELEVENLABS_MODEL,
        "output_format":  "mp3_44100_128",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
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
# LLM HELPERS  (Groq chat — answer rewriting only)
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


async def _rewrite_for_speech(transcript: str, rag_answer: str) -> str:
    # Strip markdown BEFORE sending to the LLM so it rewrites clean text
    clean = _ensure_plain(_strip_markdown(rag_answer))

    system_prompt = (
        "You convert written answers into natural spoken phone responses. "
        "Your output goes directly into a text-to-speech engine. "
        "Any asterisk, underscore, hash, backtick, dash used as a bullet, "
        "or numbered list will be read aloud verbatim and sound broken — "
        "so NEVER use them. Output plain prose only."
    )

    user_prompt = (
        "Rewrite the answer below as a short, natural spoken response.\n\n"
        "STRICT RULES — violating any will break the voice output:\n"
        "  • ABSOLUTELY NO asterisks (*), underscores (_), hashes (#), backticks (`)\n"
        "  • ABSOLUTELY NO bullet points, numbered lists, or dashes used as bullets\n"
        "  • ABSOLUTELY NO markdown of any kind\n"
        "  • Use the SAME language as the answer (Arabic if Arabic, French if French, etc.)\n"
        "  • Write in short, natural spoken sentences — like talking to a friend\n"
        "  • No filler openers: never start with 'Certainly', 'Of course', 'Sure', 'Great question'\n"
        "  • Max 4 sentences. Be concise.\n"
        "  • Output ONLY the spoken text. Nothing else. No explanations.\n\n"
        f"User asked: {transcript}\n\n"
        f"Answer to rewrite: {clean}"
    )

    result = await _llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )

    # Strip quotes the LLM sometimes wraps around the output
    result = result.strip().strip('"').strip("'")
    # Final safety pass in case the LLM still snuck something in
    return _ensure_plain(result)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/tts")
async def text_to_speech(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 4096:
        text = text[:4093] + "…"
    voice = (req.voice or _TTS_VOICE).strip() or _TTS_VOICE

    # Always strip markdown even for direct TTS calls
    text = _ensure_plain(_strip_markdown(text))

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
        spoken = ""

        if req.voice_mode:
            # voice_mode: skip Groq rewrite but still clean markdown
            spoken = _ensure_plain(_strip_markdown(req.rag_answer)).strip()
            print("⚡ voice_mode: skipping Groq rewrite, markdown stripped")
        else:
            try:
                spoken = await asyncio.wait_for(
                    _rewrite_for_speech(req.transcript, req.rag_answer),
                    timeout=20,
                )
                spoken = spoken.strip().strip('"').strip("'")
            except Exception as e:
                print(f"⚠️  Rewrite failed ({e}), falling back to stripped raw answer")

        # Fallback: raw answer with markdown stripped
        if not spoken:
            spoken = _ensure_plain(_strip_markdown(req.rag_answer)).strip()

        # Final safety pass regardless of path
        spoken = _ensure_plain(spoken)

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