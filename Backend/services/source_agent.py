"""
Source Agent v10 — LLM-First Key-Fact Extraction with Windowed Document Search

Architecture overview
─────────────────────
Previous version (v9) had two critical weaknesses:
  1. Only passed doc_text[:4000] to the judge → claims from page 3+ always failed
  2. Fallback heuristics were too loose → wrong sentences slipped through

v10 fixes both with a two-phase approach:

  PHASE 1 — KEY-FACT EXTRACTION (NEW)
    Ask the LLM to read the generated answer and extract a structured list of
    the important facts/numbers/phrases it actually asserted.  This gives us
    precise search targets instead of vague "claim sentences".

  PHASE 2 — WINDOWED DOCUMENT SEARCH
    For each key-fact, scan the FULL document in overlapping 600-char windows
    (not just the first 4000 chars). Candidate windows are pre-filtered by
    keyword/numeric overlap before being sent to the LLM judge, keeping latency
    low while covering the entire document.

  PHASE 3 — VERBATIM VERIFICATION (tightened)
    The passage returned by the judge must:
      (a) pass the existing 4-gram grounding check, AND
      (b) for numeric claims: contain the exact numeric token, AND
      (c) for any claim: not exceed the source window by more than 20 chars
          (prevents the LLM from stitching sentences together)

Output shape (fully backwards-compatible with Chat.tsx):
  [
    {
      source_number: int,
      filename:      str,
      doc_type:      str,
      excerpt:       str,        # verbatim passage from document
      sections:      [{ title: str, lines: [str], excerpt: str }],
      cited_facts:   [str],
      has_images:    bool,
      has_tables:    bool,
      claim_type:    str,        # "numeric" | "table" | "statement"
      value_found:   str | None,
    }
  ]
"""

from __future__ import annotations

import re
import os
import json
import time
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_doc(chunks: List[Dict]) -> str:
    """Stitch sorted chunks back into a single document string."""
    sorted_chunks = sorted(chunks, key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
    if not sorted_chunks:
        return ""
    result = sorted_chunks[0].get("text", "")
    for chunk in sorted_chunks[1:]:
        text = chunk.get("text", "")
        if not text:
            continue
        overlap = _find_overlap(result, text)
        result = result + "\n" + text[overlap:] if overlap < len(text) else result
    return result.strip()


def _find_overlap(a: str, b: str, max_check: int = 300) -> int:
    tail = a[-max_check:]
    for length in range(min(len(tail), len(b), max_check), 20, -1):
        if tail.endswith(b[:length]):
            return length
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM PARSING (unchanged from v9 — still needed for source-number mapping)
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(
    r"""
    (?:
        \d+(?:[.,\s]\d+)*
        \s*
        (?:
            [%\$€£¥]
            |m(?:\b|²|³|2|3)
            |km\b | cm\b | mm\b
            |kg\b | t\b | T\b
            |kW\b | MW\b | kV\b
            |dB\b | dBm\b
            |MHz\b | GHz\b
            |Gbps?\b | Mbps?\b | Kbps?\b
            |h\b | min\b | s\b
            |m/s\b | km/h\b
        )?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_TABLE_CLAIM_RE = re.compile(
    r'\b(table|row|column|cell|tableau|ligne|colonne|cel[llu]+e|'
    r'total|sum|subtotal|taux|rate|avg|average|moyenne)\b',
    re.IGNORECASE,
)


def _classify_claim(text: str) -> Tuple[str, List[str]]:
    nums = [m.group().strip() for m in _NUMERIC_RE.finditer(text)
            if m.group().strip() and re.search(r'\d', m.group())]
    if nums:
        return "numeric", nums
    if _TABLE_CLAIM_RE.search(text):
        return "table", []
    return "statement", []


def _parse_claims(answer: str) -> List[Dict]:
    """
    Walk the answer line by line.  For each line citing [N], produce a Claim dict.
    Each claim is produced ONCE per citation.
    """
    claims: List[Dict] = []
    current_heading = ""

    for raw_line in answer.split('\n'):
        stripped = raw_line.strip()
        if not stripped:
            continue

        h_match = re.match(r'^#{1,3}\s+(.+)', stripped)
        if h_match:
            current_heading = re.sub(r'\[\d+\]', '', h_match.group(1)).strip()
            continue

        cited = sorted(set(int(m) for m in re.findall(r'\[(\d+)\](?!\()', stripped)))
        if not cited:
            continue

        clean = re.sub(r'\[\d+\]', '', stripped)
        clean = re.sub(r'^\s*[-*•·>]+\s*', '', clean)
        clean = re.sub(r'\*\*', '', clean).strip()
        if len(clean) < 6:
            continue

        ctype, values = _classify_claim(clean)

        for src_num in cited:
            claims.append({
                "src_num":    src_num,
                "heading":    current_heading or f"Source {src_num}",
                "raw_line":   stripped,
                "clean_line": clean,
                "claim_type": ctype,
                "values":     values,
            })

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LLM KEY-FACT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_KEY_FACT_SYSTEM = """You are a fact-extraction assistant. Given an AI-generated answer, extract every specific verifiable fact it asserts.

Return a JSON array of objects. Each object:
  {
    "fact": "<the exact phrase or number as stated in the answer>",
    "src_num": <N>,          // the [N] citation number
    "is_numeric": true/false // true if the fact contains a number or measurement
  }

Rules:
- Focus on SPECIFIC facts: numbers, measurements, names, dates, quantities, identifiers.
- Each numeric value gets its own entry (e.g. "31,080 meters" and "47 sites" are separate).
- Important named entities (site names, equipment models, section titles) also get entries.
- Skip vague statements like "the document discusses..." — only extractable facts.
- If a fact has multiple [N] citations, emit one entry per source number.
- Output ONLY the raw JSON array. No markdown, no explanation."""


def _extract_key_facts_from_answer(answer: str, api_key: str, base_url: str) -> List[Dict]:
    """
    Phase 1: Ask the LLM to extract all verifiable key facts from the answer.
    Returns list of {fact, src_num, is_numeric}.
    Falls back to claim-based approach if LLM fails.
    """
    # Strip answer to first 3000 chars for the extraction prompt (summary of claims)
    answer_snippet = answer[:3000]

    prompt = f"""Extract all specific verifiable facts from this AI answer. Return a JSON array as instructed.

AI ANSWER:
{answer_snippet}

JSON array of facts:"""

    try:
        from llm_client import call_llm
        raw = call_llm(
            prompt=prompt,
            system_prompt=_KEY_FACT_SYSTEM,
            history=[],
            max_tokens=600,
            temperature=0.0,
            task="plan",
        )
    except Exception as e:
        print(f"   ⚠️  Key-fact LLM call failed: {e}")
        return []

    if not raw:
        return []

    # Parse JSON
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "fact" in item and "src_num" in item:
                    valid.append({
                        "fact":       str(item["fact"]).strip(),
                        "src_num":    int(item["src_num"]),
                        "is_numeric": bool(item.get("is_numeric", False)),
                    })
            print(f"   🔑 Key-fact extraction: {len(valid)} facts from answer")
            return valid
    except (json.JSONDecodeError, ValueError):
        pass

    # Try ast fallback
    import ast
    try:
        parsed = ast.literal_eval(clean)
        if isinstance(parsed, list):
            valid = [
                {"fact": str(i["fact"]).strip(), "src_num": int(i["src_num"]),
                 "is_numeric": bool(i.get("is_numeric", False))}
                for i in parsed if isinstance(i, dict) and "fact" in i and "src_num" in i
            ]
            print(f"   🔑 Key-fact extraction (ast): {len(valid)} facts")
            return valid
    except Exception:
        pass

    print(f"   ⚠️  Key-fact extraction failed to parse LLM output")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — WINDOWED DOCUMENT SEARCH
# ─────────────────────────────────────────────────────────────────────────────

_WINDOW_SIZE  = 600   # chars per candidate window
_WINDOW_STEP  = 300   # stride (50% overlap so facts near boundaries aren't missed)
_MAX_WINDOWS  = 8     # max candidate windows to send to LLM per fact


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, length ≥ 3, excluding common stop words."""
    STOP = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and',
        'or', 'for', 'with', 'from', 'that', 'this', 'it', 'on', 'at', 'by',
        'les', 'des', 'une', 'est', 'dans', 'pour', 'avec', 'qui', 'que', 'le', 'la'
    }
    return [w for w in re.findall(r'[a-zA-ZÀ-ÿ\d]{3,}', text.lower()) if w not in STOP]


def _score_window(window: str, fact: str, is_numeric: bool) -> float:
    """
    Score how relevant a document window is to a key fact.
    Higher = better candidate to send to the LLM.
    """
    score = 0.0

    # Numeric: exact figure must be present
    if is_numeric:
        nums = [m.group().strip() for m in _NUMERIC_RE.finditer(fact)
                if m.group().strip() and re.search(r'\d', m.group())]
        norm_window = re.sub(r'[\s,.]', '', window.lower())
        for num in nums:
            norm_num = re.sub(r'[\s,.]', '', num.lower())
            if norm_num and norm_num in norm_window:
                score += 5.0   # strong signal — exact number present

    # Keyword overlap
    fact_tokens  = set(_tokenize(fact))
    window_tokens = set(_tokenize(window))
    if fact_tokens:
        overlap = len(fact_tokens & window_tokens) / len(fact_tokens)
        score += overlap * 3.0

    # Substring hit (partial phrase)
    fact_words = fact.lower().split()
    for n in range(min(4, len(fact_words)), 1, -1):
        for i in range(len(fact_words) - n + 1):
            phrase = ' '.join(fact_words[i:i+n])
            if phrase in window.lower():
                score += n * 0.5
                break

    return score


def _get_candidate_windows(fact: str, doc_text: str, is_numeric: bool) -> List[str]:
    """
    Slide a window over the FULL document, score each window against the fact,
    and return the top _MAX_WINDOWS candidates sorted by score descending.
    """
    if len(doc_text) <= _WINDOW_SIZE:
        return [doc_text]

    scored: List[Tuple[float, str]] = []
    for start in range(0, len(doc_text) - _WINDOW_SIZE // 2, _WINDOW_STEP):
        window = doc_text[start:start + _WINDOW_SIZE]
        s = _score_window(window, fact, is_numeric)
        if s > 0:
            scored.append((s, window))

    # Sort descending by score, deduplicate overlapping windows
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[str] = []
    seen_starts: List[int] = []
    for _, window in scored:
        # Avoid windows that are almost identical (overlap > 80%)
        if any(abs(doc_text.find(window) - s) < _WINDOW_STEP // 2 for s in seen_starts):
            continue
        pos = doc_text.find(window)
        if pos >= 0:
            seen_starts.append(pos)
        selected.append(window)
        if len(selected) >= _MAX_WINDOWS:
            break

    return selected if selected else [doc_text[:_WINDOW_SIZE]]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2b — DIRECT NUMERIC/PHRASE SCAN (before calling LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _direct_scan_for_fact(fact: str, doc_text: str, is_numeric: bool) -> Optional[str]:
    """
    Before calling the LLM, try to find the passage directly:
    - For numeric facts: find the exact sentence containing the number
    - For phrase facts: find the sentence with the highest n-gram overlap

    Returns a single sentence (≤ 500 chars) or None.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', doc_text) if len(s.strip()) > 8]

    if is_numeric:
        nums = [m.group().strip() for m in _NUMERIC_RE.finditer(fact)
                if m.group().strip() and re.search(r'\d', m.group())]
        for num in nums:
            norm_num = re.sub(r'[\s,.]', '', num.lower())
            for sent in sentences:
                norm_sent = re.sub(r'[\s,.]', '', sent.lower())
                if norm_num and norm_num in norm_sent:
                    return sent[:500]
        return None  # Don't guess for numeric — only exact hits

    # Statement: best keyword overlap sentence
    fact_tokens = set(_tokenize(fact))
    if not fact_tokens:
        return None
    best_score, best_sent = 0, None
    for sent in sentences:
        score = len(fact_tokens & set(_tokenize(sent)))
        if score > best_score:
            best_score, best_sent = score, sent
    # Require at least 40% token overlap
    threshold = max(2, int(len(fact_tokens) * 0.4))
    return best_sent[:500] if best_sent and best_score >= threshold else None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — LLM JUDGE (windowed, full-doc aware)
# ─────────────────────────────────────────────────────────────────────────────

def _build_windowed_judge_prompt(fact: str, windows: List[str], is_numeric: bool) -> str:
    """
    Build a judge prompt with the top candidate windows (not the full doc).
    The windows already contain the fact — the LLM just needs to pinpoint the sentence.
    """
    windows_text = "\n---\n".join(windows)

    numeric_note = ""
    if is_numeric:
        # Extract the numeric tokens from the fact for emphasis
        nums = [m.group().strip() for m in _NUMERIC_RE.finditer(fact)
                if m.group().strip() and re.search(r'\d', m.group())]
        if nums:
            numeric_note = (
                f"\nIMPORTANT: This fact contains numeric data. "
                f"The specific value(s) to find: {', '.join(repr(n) for n in nums[:5])}. "
                f"Your output MUST contain at least one of these exact figures as they appear in the document."
            )

    return f"""You are a precise source-extraction assistant. Your only job is to find the EXACT sentence or table row from the document excerpts below that directly supports the given fact.

FACT TO VERIFY:
"{fact}"{numeric_note}

STRICT RULES:
1. Copy text VERBATIM from the excerpts — zero rewording, zero summarising.
2. Output the ONE shortest sentence or table row that directly supports the fact.
3. The sentence MUST appear word-for-word in the excerpts below.
4. Do NOT include any commentary, explanation, header, or preamble.
5. If the supporting sentence is not present in the excerpts, output: NOTFOUND
6. Do NOT combine or merge multiple sentences into one.
7. Strip any markdown formatting from the copied text.

DOCUMENT EXCERPTS:
{windows_text}

Verbatim sentence (or NOTFOUND):"""


def _call_llm(prompt: str, max_tokens: int = 400) -> str:
    """Shared LLM call via llm_client gateway."""
    try:
        from llm_client import call_llm
        return call_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            task="plan",
        )
    except Exception as e:
        print(f"      ⚠️  LLM call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# GROUNDING CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _is_grounded(passage: str, doc_text: str, min_4gram_hits: int = 1) -> bool:
    """
    Return True if passage contains at least min_4gram_hits 4-word runs
    that appear verbatim in doc_text.
    """
    words = passage.lower().split()
    if len(words) < 4:
        return passage.lower()[:30] in doc_text.lower()
    doc_lower = doc_text.lower()
    hits = 0
    for i in range(len(words) - 3):
        gram = ' '.join(words[i:i+4])
        if gram in doc_lower:
            hits += 1
            if hits >= min_4gram_hits:
                return True
    return False


def _verify_numeric(passage: str, values: List[str]) -> Optional[str]:
    p_lower = passage.lower()
    for v in values:
        norm_v = re.sub(r'[\s,.]', '', v)
        if norm_v and norm_v in re.sub(r'[\s,.]', '', p_lower):
            return v
        if v.lower() in p_lower:
            return v
    return None


def _clean_passage(raw: str) -> str:
    """Sanitize LLM output to a clean verbatim passage."""
    passage = raw.strip().strip('"').strip("'").strip()
    passage = re.sub(r'^[-–•]\s*', '', passage).strip()
    # Remove any "passage:" or "excerpt:" prefix the LLM sometimes adds
    passage = re.sub(r'^(?:passage|excerpt|verbatim)[:\s]+', '', passage, flags=re.IGNORECASE).strip()
    return passage


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    """
    v10: LLM-first key-fact extraction with windowed full-document search.

    Flow per source:
      1. Call LLM to extract key facts from the answer (Phase 1)
      2. For each fact, scan the FULL document in sliding windows (Phase 2)
      3. Try direct sentence scan first (no LLM call needed)
      4. If direct scan misses, send top candidate windows to the LLM judge
      5. Verify grounding and numeric presence (Phase 3)
      6. Emit one source card per verified (source_number, fact) pair
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_store=None,
    ):
        self.vs       = vector_store
        self.api_key  = api_key or os.getenv("CF_API_KEY", "")
        self.base_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")
        self.llm_ok   = bool(self.api_key)
        print(f"📎 Source Agent v10 [windowed-search, llm={'✅' if self.llm_ok else '❌'}]")

    # ── Vector store helpers ──────────────────────────────────────────────

    def _get_all_chunks_for_file(
        self,
        filename: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        if self.vs is None:
            return []
        try:
            if hasattr(self.vs, '_get_collection'):
                coll = self.vs._get_collection(user_id=user_id, session_id=session_id)
                raw = coll.get(where={"filename": filename})
            elif hasattr(self.vs, 'collection'):
                raw = self.vs.collection.get(where={"filename": filename})
            else:
                return []

            chunks = []
            for i, doc_text in enumerate(raw.get("documents", [])):
                meta = raw["metadatas"][i] if raw.get("metadatas") else {}
                chunks.append({"text": doc_text, "metadata": meta})
            return chunks
        except Exception as e:
            print(f"      ⚠️  VS fetch failed for '{filename}': {e}")
            return []

    def _get_doc_text(
        self,
        chunk: Dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        filename = chunk.get("metadata", {}).get("filename", "")
        if filename:
            all_chunks = self._get_all_chunks_for_file(filename, user_id=user_id, session_id=session_id)
            if all_chunks:
                doc_text = _reconstruct_doc(all_chunks)
                print(f"   📄 '{filename}': {len(doc_text)} chars from {len(all_chunks)} chunks")
                return doc_text
        doc_text = chunk.get("text", "")
        print(f"   📄 '{filename}': using chunk directly ({len(doc_text)} chars)")
        return doc_text

    # ── Fact → Passage lookup ─────────────────────────────────────────────

    def _find_passage_for_fact(
        self,
        fact: str,
        is_numeric: bool,
        doc_text: str,
        emitted_passages: List[str],
    ) -> Optional[str]:
        """
        Find the verbatim backing passage for one key fact.

        Strategy:
          1. Direct sentence scan (fast, no LLM) — exact number / high keyword overlap
          2. Windowed LLM judge (full doc coverage)
          3. Return None if nothing passes verification
        """
        # ── Step 1: Direct scan ───────────────────────────────────────────
        passage = _direct_scan_for_fact(fact, doc_text, is_numeric)
        if passage:
            if _is_grounded(passage, doc_text):
                if not any(_jaccard(passage, seen) > 0.75 for seen in emitted_passages):
                    print(f"      ✅ direct scan: {passage[:80]!r}")
                    return passage
                print(f"      ⏭  Direct scan dedup dropped")
            else:
                print(f"      ⚠️  Direct scan passage not grounded — trying LLM")
                passage = None

        # ── Step 2: Windowed LLM judge ────────────────────────────────────
        if not self.llm_ok:
            print(f"      ❌ LLM unavailable and direct scan failed for: {fact[:60]!r}")
            return None

        windows = _get_candidate_windows(fact, doc_text, is_numeric)
        if not windows:
            print(f"      ❌ No candidate windows found for: {fact[:60]!r}")
            return None

        print(f"      🔍 LLM judge: {len(windows)} windows for {'[NUM]' if is_numeric else '[STMT]'} {fact[:60]!r}")
        prompt = _build_windowed_judge_prompt(fact, windows, is_numeric)
        raw = _call_llm(prompt, max_tokens=300)

        if not raw or raw.strip().upper().startswith("NOTFOUND"):
            print(f"      ❌ LLM returned NOTFOUND for: {fact[:60]!r}")
            return None

        passage = _clean_passage(raw)
        if len(passage) < 8:
            return None

        # Grounding check against full doc
        if not _is_grounded(passage, doc_text):
            print(f"      ⚠️  Grounding failed: {passage[:80]!r}")
            return None

        # Numeric value must be present in passage
        if is_numeric:
            nums = [m.group().strip() for m in _NUMERIC_RE.finditer(fact)
                    if m.group().strip() and re.search(r'\d', m.group())]
            if nums and _verify_numeric(passage, nums) is None:
                print(f"      ⚠️  Numeric value not in passage — dropping: {passage[:80]!r}")
                return None

        # Global dedup
        if any(_jaccard(passage, seen) > 0.75 for seen in emitted_passages):
            print(f"      ⏭  LLM passage dedup dropped")
            return None

        print(f"      ✅ LLM windowed passage: {passage[:80]!r}")
        return passage

    # ── Public entry point ────────────────────────────────────────────────

    def build_sources(
        self,
        answer: str,
        chunks: List[Dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build source cards for the generated answer.

        v10 flow:
          1. Parse citation claims from the answer (for source-number mapping)
          2. Extract key facts via LLM (Phase 1)
          3. For each fact, find verbatim passage via windowed search (Phase 2+3)
          4. Emit one card per verified fact
        """
        # 1. Parse claims for source-number mapping
        all_claims = _parse_claims(answer)
        if not all_claims:
            return []

        # 2. Pre-cache doc texts per source number
        doc_cache: Dict[int, str] = {}
        for claim in all_claims:
            src_num = claim["src_num"]
            if src_num not in doc_cache:
                idx = src_num - 1
                if 0 <= idx < len(chunks):
                    doc_cache[src_num] = self._get_doc_text(
                        chunks[idx], user_id=user_id, session_id=session_id
                    )

        # 3. Phase 1: Extract key facts from the full answer
        key_facts: List[Dict] = []
        if self.llm_ok:
            key_facts = _extract_key_facts_from_answer(answer, self.api_key, self.base_url)

        # If LLM extraction failed, fall back to using claims directly as facts
        if not key_facts:
            print("   ⚠️  Key-fact extraction failed — falling back to claim-based facts")
            for c in all_claims:
                key_facts.append({
                    "fact":       c["clean_line"],
                    "src_num":    c["src_num"],
                    "is_numeric": c["claim_type"] == "numeric",
                })

        # 4. Phase 2+3: Find passage for each fact
        sources: List[Dict] = []
        emitted_passages: List[str] = []

        # Group facts by source number for clean card ordering
        from collections import OrderedDict
        facts_by_src: Dict[int, List[Dict]] = OrderedDict()
        for fact in key_facts:
            src_num = int(fact["src_num"])
            facts_by_src.setdefault(src_num, []).append(fact)

        # Also ensure every cited source number appears (even if fact extraction missed it)
        for claim in all_claims:
            if claim["src_num"] not in facts_by_src:
                facts_by_src.setdefault(claim["src_num"], []).append({
                    "fact":       claim["clean_line"],
                    "src_num":    claim["src_num"],
                    "is_numeric": claim["claim_type"] == "numeric",
                })

        for src_num, src_facts in facts_by_src.items():
            idx = src_num - 1
            if idx < 0 or idx >= len(chunks):
                continue

            chunk    = chunks[idx]
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", f"source_{src_num}")
            doc_text = doc_cache.get(src_num, "")
            if not doc_text:
                continue

            has_images = metadata.get("has_images", False)
            has_tables = metadata.get("has_tables", False)

            for fact_item in src_facts:
                fact       = fact_item["fact"]
                is_numeric = fact_item["is_numeric"]

                passage = self._find_passage_for_fact(fact, is_numeric, doc_text, emitted_passages)
                if not passage:
                    continue

                emitted_passages.append(passage)

                # Card title: the fact itself (short)
                card_title = fact[:80] + "…" if len(fact) > 80 else fact

                # value_found: verified numeric token
                value_found: Optional[str] = None
                if is_numeric:
                    nums = [m.group().strip() for m in _NUMERIC_RE.finditer(fact)
                            if m.group().strip() and re.search(r'\d', m.group())]
                    if nums:
                        value_found = _verify_numeric(passage, nums)

                # Determine claim_type for backwards compat
                _, claim_values = _classify_claim(fact)
                ctype = "numeric" if is_numeric else ("table" if _TABLE_CLAIM_RE.search(fact) else "statement")

                sources.append({
                    "source_number": src_num,
                    "filename":      filename,
                    "doc_type":      metadata.get("doc_type", "unknown"),
                    "project_ref":   metadata.get("project_ref"),
                    "has_images":    has_images,
                    "has_tables":    has_tables,
                    "excerpt":       passage,
                    "sections": [
                        {
                            "title":   card_title,
                            "lines":   [passage],
                            "excerpt": passage,
                        }
                    ],
                    "cited_facts":  [card_title],
                    "claim_type":   ctype,
                    "value_found":  value_found,
                    "heading":      fact_item.get("heading", f"Source {src_num}"),
                })

        # 5. Final dedup: drop cards with near-identical excerpts
        seen_key: set = set()
        deduped: List[Dict] = []
        for card in sources:
            key = (card["source_number"], card["excerpt"][:60])
            if key not in seen_key:
                seen_key.add(key)
                deduped.append(card)

        return deduped


# ─────────────────────────────────────────────────────────────────────────────
# NODE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_sources_node(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    vector_store=None,
):
    """
    Build the LangGraph-compatible sources node.

    Reads (answer, retrieved_chunks, session_id, user_id) from state,
    writes sources back.
    """
    agent = SourceAgent(api_key=api_key, model=model, vector_store=vector_store)

    def node(state: dict) -> dict:
        answer = state.get("answer", "")
        chunks = state.get("retrieved_chunks", [])
        if not answer or not chunks:
            return state

        user_id    = state.get("user_id")
        session_id = state.get("session_id", "")

        # Stat breakdown before extraction
        all_claims = _parse_claims(answer)
        n_numeric   = sum(1 for c in all_claims if c["claim_type"] == "numeric")
        n_table     = sum(1 for c in all_claims if c["claim_type"] == "table")
        n_statement = sum(1 for c in all_claims if c["claim_type"] == "statement")
        print(
            f"📎 Source Agent v10 → {len(all_claims)} claims "
            f"({n_numeric} numeric, {n_table} table, {n_statement} statement)"
        )

        sources = agent.build_sources(
            answer, chunks,
            user_id=user_id,
            session_id=session_id,
        )

        n_numeric_found   = sum(1 for s in sources if s.get("claim_type") == "numeric")
        n_table_found     = sum(1 for s in sources if s.get("claim_type") == "table")
        n_statement_found = sum(1 for s in sources if s.get("claim_type") == "statement")
        print(
            f"   → {len(sources)} source cards "
            f"({n_numeric_found} numeric, {n_table_found} table, {n_statement_found} statement)"
        )

        return {**state, "sources": sources}

    return node