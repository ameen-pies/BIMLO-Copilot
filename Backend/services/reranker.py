"""
reranker.py — ColBERT / Cross-Encoder Re-ranking for BIMLO Copilot

Pipeline:
  1. Vector store returns top-N candidates (fetch_k, default 20)
  2. Cross-encoder scores every (query, chunk) pair for EXACT relevance
  3. Return top-K re-ranked chunks (what the LLM actually sees)

Model: BAAI/bge-reranker-v2-m3
  - Best-in-class multilingual reranker (handles French, Arabic, English)
  - ~560 MB download on first use (cached locally by HuggingFace)
  - Runs on CPU fine for batches of 20 chunks
  - Outputs raw logits (unbounded floats) — we sigmoid-normalize to 0–1
    so scores are human-readable AND compatible with _is_good_retrieval()

Env vars:
  RERANKER_MODEL   — HuggingFace model ID   (default: BAAI/bge-reranker-v2-m3)
  RERANKER_ENABLED — set "0" to disable     (default: enabled if model loads)
  RERANKER_FETCH_K — candidates to fetch    (default: 20)
"""

from __future__ import annotations

import os
import math
import time
from typing import List, Dict, Optional

_RERANKER_MODEL   = os.getenv("RERANKER_MODEL",   "BAAI/bge-reranker-v2-m3")
_RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1").strip() != "0"
_FETCH_K          = int(os.getenv("RERANKER_FETCH_K", "20"))

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load the cross-encoder so startup is never blocked
# ─────────────────────────────────────────────────────────────────────────────

_cross_encoder = None
_load_attempted = False


def _get_cross_encoder():
    global _cross_encoder, _load_attempted
    if _load_attempted:
        return _cross_encoder
    _load_attempted = True

    if not _RERANKER_ENABLED:
        print("ℹ️  reranker: disabled via RERANKER_ENABLED=0")
        return None

    try:
        from sentence_transformers import CrossEncoder
        print(f"⏳ reranker: loading {_RERANKER_MODEL} (first run downloads ~560 MB)…")
        t0 = time.time()
        _cross_encoder = CrossEncoder(_RERANKER_MODEL, max_length=512)
        print(f"✅ reranker: {_RERANKER_MODEL} ready ({time.time()-t0:.1f}s)")
    except ImportError:
        print("⚠️  reranker: sentence-transformers not installed — run: pip install sentence-transformers")
        _cross_encoder = None
    except Exception as e:
        print(f"⚠️  reranker: failed to load model ({e}) — falling back to vector-only retrieval")
        _cross_encoder = None

    return _cross_encoder


def _sigmoid(x: float) -> float:
    """Normalize a raw logit to 0–1 probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Re-rank `chunks` by exact relevance to `query` using a cross-encoder.

    Args:
        query:  The user's question.
        chunks: Raw candidate chunks from the vector store (up to FETCH_K).
        top_k:  How many to return after re-ranking.

    Returns:
        The top_k most relevant chunks, each with:
          - rerank_score : sigmoid-normalized relevance score (0–1)
          - distance     : overwritten with (1 - rerank_score) so that
                           _is_good_retrieval() in rag_engine works correctly
        Falls back to the original order if the model is unavailable.
    """
    if not chunks:
        return chunks

    model = _get_cross_encoder()

    if model is None:
        # Graceful degradation — return as-is, trimmed to top_k
        return chunks[:top_k]

    t0 = time.time()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c["text"][:512]) for c in chunks]

    try:
        raw_scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        print(f"⚠️  reranker.predict failed ({e}) — using original order")
        return chunks[:top_k]

    # Normalize logits → 0–1 via sigmoid, then write back onto each chunk.
    # Also update `distance` (= 1 - score) so _is_good_retrieval() in
    # rag_engine.py can correctly judge quality from reranked results.
    for chunk, raw in zip(chunks, raw_scores):
        score = _sigmoid(float(raw))
        chunk["rerank_score"] = score
        chunk["distance"]     = 1.0 - score   # ← keeps _is_good_retrieval() honest

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    result = ranked[:top_k]

    print(
        f"🏆 reranker: {len(chunks)} → {len(result)} chunks "
        f"(top score: {result[0]['rerank_score']:.3f}, "
        f"bottom: {result[-1]['rerank_score']:.3f}, "
        f"{time.time()-t0:.2f}s)"
    )

    return result


def get_fetch_k(top_k: int) -> int:
    """
    How many candidates to fetch from the vector store before re-ranking.
    Always at least 3× top_k, capped at FETCH_K.
    """
    return max(_FETCH_K, top_k * 3)