"""
source_agent.py v11 — Atomic Fact-Chip Extraction

Architecture
────────────
The previous approach (v10) produced ONE card per source document showing
a large text blob excerpt. This was nearly useless: a wall of text with no
signal about WHAT the AI actually found.

v11 produces ATOMIC FACT CHIPS — one chip per extracted data point, e.g.:
  • Floors           → 12
  • Total Area       → 18 500 m²
  • Concrete Volume  → 4 200 m³
  • Steel Weight     → 780 tons
  • Base Stations    → 6
  • Fiber Length     → 12.5 km

Each chip carries:
  - label      : the field name ("Concrete Volume")
  - value      : the extracted value ("4 200 m³")
  - raw_line   : the verbatim document sentence this was pulled from
  - is_numeric : True if the value contains a number

Source cards are still grouped per document (one card per file), but the
card body is a set of fact chips — not a paragraph excerpt.

Pipeline
────────
  1. ANSWER SCAN     — regex + LLM extract {label, value, src_num} from AI answer
  2. DOCUMENT VERIFY — for each chip, find the verbatim document line that
                       contains the value (direct scan, no LLM needed for numerics)
  3. EMIT            — one card per document, chips sorted by appearance order

Output shape (backwards-compatible + new fields):
  [
    {
      source_number : int,
      filename      : str,
      doc_type      : str,
      excerpt       : str,          # first chip's raw_line (legacy compat)
      has_images    : bool,
      has_tables    : bool,
      fact_chips    : [             # NEW — atomic fact list
        {
          label      : str,
          value      : str,
          raw_line   : str,
          is_numeric : bool,
        }
      ],
      # legacy fields kept for backward compat
      sections      : [{ title: str, lines: [str], excerpt: str }],
      cited_facts   : [str],
      claim_type    : str,
      value_found   : str | None,
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
# NUMERIC PATTERN  (kept from v10)
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
            |tons?\b | tonnes?\b
        )?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT RECONSTRUCTION  (unchanged from v10)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_doc(chunks: List[Dict]) -> str:
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
# STEP 1 — EXTRACT FACT CHIPS FROM THE AI ANSWER
# ─────────────────────────────────────────────────────────────────────────────

_CHIP_SYSTEM = """You are a fact-chip extraction assistant.

Given an AI-generated answer, extract every specific data point that has BOTH:
  - a label (the field/property name)
  - a value (a concrete number, measurement, or named quantity)

Return a JSON array. Each element:
{
  "label"    : "<concise field name, e.g. 'Floors', 'Steel Weight', 'Fiber Length'>",
  "value"    : "<exact value as stated, e.g. '12', '780 tons', '12.5 km'>",
  "src_num"  : <N>,         // the [N] citation in the answer
  "is_numeric": true/false  // true if value contains a digit
}

Rules:
- Extract ONE entry per distinct (label, value) pair.
- Prefer SHORT labels (1–4 words, title-case).
- Include all numeric measurements, counts, identifiers, and named quantities.
- Skip vague statements ("the system is efficient") — only concrete data points.
- If a value is cited multiple times under different [N], emit once per [N].
- Output ONLY the raw JSON array. No markdown, no explanation."""


def _extract_fact_chips_from_answer(answer: str) -> List[Dict]:
    """
    Phase 1: Ask the LLM to return a flat list of {label, value, src_num, is_numeric}.
    Falls back to regex scan if LLM call fails.
    """
    try:
        from llm_client import call_llm
        raw = call_llm(
            prompt=f"Extract all fact chips from this AI answer.\n\nAI ANSWER:\n{answer[:3000]}\n\nJSON array:",
            system_prompt=_CHIP_SYSTEM,
            history=[],
            max_tokens=800,
            temperature=0.0,
            task="plan",
        )
        clean = re.sub(r"```(?:json)?|```", "", (raw or "")).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "label" in item and "value" in item:
                    valid.append({
                        "label":      str(item["label"]).strip(),
                        "value":      str(item["value"]).strip(),
                        "src_num":    int(item.get("src_num", 1)),
                        "is_numeric": bool(item.get("is_numeric", False)),
                    })
            if valid:
                print(f"   🔑 Chip extraction: {len(valid)} chips from answer (LLM)")
                return valid
    except Exception as e:
        print(f"   ⚠️  Chip LLM call failed: {e}")

    # ── Fallback: regex scan of the answer ───────────────────────────────
    return _regex_extract_chips(answer)


def _regex_extract_chips(answer: str) -> List[Dict]:
    """
    Fallback: walk answer lines, pull (label, value) pairs from patterns like:
      - **Floors**: 12            → label=Floors, value=12
      - Concrete Volume: 4200 m³  → label=Concrete Volume, value=4200 m³
      - 12 floors                 → label=Floors, value=12
    """
    chips: List[Dict] = []
    seen: set = set()

    # Pattern A: "Label: value [N]"
    pat_a = re.compile(
        r'\*{0,2}([\w /()&-]{2,40})\*{0,2}\s*[:–—]\s*'
        r'([\d][\d.,\s]*)(?:\s*(?:m[²³23]?|km|tons?|kg|Mbps?|Gbps?|kW|MW|%|dB))?\b'
        r'(?:.*?\[(\d+)\])?',
        re.IGNORECASE,
    )
    for line in answer.split('\n'):
        for m in pat_a.finditer(line):
            label = re.sub(r'\*+', '', m.group(1)).strip().rstrip(':')
            value = m.group(0).split(':')[-1].split('[')[0].strip()
            src = int(m.group(3)) if m.group(3) else 1
            key = (label.lower(), value)
            if key not in seen and len(label) > 1:
                seen.add(key)
                chips.append({"label": label, "value": value, "src_num": src, "is_numeric": True})

    # Pattern B: catch any numeric mention with a unit near citation [N]
    pat_b = re.compile(
        r'(\d[\d.,\s]*)\s*(m[²³23]?|km|ton[ns]?|tonnes?|kg|Mbps?|Gbps?|kW|MW|%|dBm?|floors?|stations?|sensors?|routers?|switches?|points?)\b'
        r'(?:.*?\[(\d+)\])?',
        re.IGNORECASE,
    )
    for line in answer.split('\n'):
        for m in pat_b.finditer(line):
            val = m.group(1).strip() + " " + m.group(2).strip()
            unit_label = m.group(2).strip().rstrip('s').title()
            src = int(m.group(3)) if m.group(3) else 1
            key = (unit_label.lower(), val)
            if key not in seen:
                seen.add(key)
                chips.append({"label": unit_label, "value": val, "src_num": src, "is_numeric": True})

    print(f"   🔑 Chip extraction: {len(chips)} chips from answer (regex fallback)")
    return chips


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — VERIFY EACH CHIP AGAINST THE SOURCE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip spaces/commas/dots for numeric comparison."""
    return re.sub(r'[\s,.]', '', s.lower())


def _find_raw_line_for_chip(label: str, value: str, is_numeric: bool, doc_text: str) -> Optional[str]:
    """
    Find the SHORTEST document sentence that contains the given value.

    Strategy (no LLM needed for numeric values):
      1. Split doc into sentences.
      2. For numeric chips: exact normalised number match.
      3. For label chips: keyword overlap ≥ 40 %.
      4. Return shortest matching sentence (≤ 300 chars).
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', doc_text) if len(s.strip()) > 4]

    # Extract just the numeric digits from value for comparison
    nums = [m.group().strip() for m in _NUMERIC_RE.finditer(value)
            if m.group().strip() and re.search(r'\d', m.group())]

    if is_numeric and nums:
        candidates = []
        for sent in sentences:
            for num in nums:
                if _normalize(num) in _normalize(sent):
                    candidates.append(sent)
                    break
        if candidates:
            # Prefer shortest sentence that contains the number
            return min(candidates, key=len)[:300]

    # Fallback: label keyword overlap
    label_tokens = set(re.findall(r'[a-zA-Z\d]{3,}', label.lower()))
    value_tokens = set(re.findall(r'[a-zA-Z\d]{3,}', value.lower()))
    search_tokens = label_tokens | value_tokens
    if not search_tokens:
        return None

    best_score, best_sent = 0, None
    for sent in sentences:
        sent_tokens = set(re.findall(r'[a-zA-Z\d]{3,}', sent.lower()))
        score = len(search_tokens & sent_tokens)
        if score > best_score:
            best_score, best_sent = score, sent

    threshold = max(2, int(len(search_tokens) * 0.4))
    return best_sent[:300] if best_sent and best_score >= threshold else None


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY HELPERS  (kept so rag_engine.py doesn't break)
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_CLAIM_RE = re.compile(
    r'\b(table|row|column|cell|total|sum|subtotal|avg|average)\b', re.IGNORECASE
)


def _parse_claims(answer: str) -> List[Dict]:
    """Walk the answer line-by-line, collect {src_num, clean_line, claim_type} per [N] citation."""
    claims: List[Dict] = []
    for raw_line in answer.split('\n'):
        stripped = raw_line.strip()
        cited = sorted(set(int(m) for m in re.findall(r'\[(\d+)\](?!\()', stripped)))
        if not cited:
            continue
        clean = re.sub(r'\[\d+\]', '', stripped)
        clean = re.sub(r'^\s*[-*•·>]+\s*', '', clean)
        clean = re.sub(r'\*\*', '', clean).strip()
        if len(clean) < 6:
            continue
        nums = [m.group().strip() for m in _NUMERIC_RE.finditer(clean)
                if m.group().strip() and re.search(r'\d', m.group())]
        ctype = "numeric" if nums else ("table" if _TABLE_CLAIM_RE.search(clean) else "statement")
        for src_num in cited:
            claims.append({"src_num": src_num, "clean_line": clean, "claim_type": ctype})
    return claims


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    """
    v11: Atomic fact-chip extraction.

    Produces one source card per document, with a `fact_chips` list inside —
    each chip is a {label, value, raw_line} tuple.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_store=None,
    ):
        self.vs       = vector_store
        self.api_key  = api_key or os.getenv("CF_API_KEY", "")
        self.base_url = os.getenv("CF_API_URL", "")
        self.llm_ok   = bool(self.api_key)
        print(f"📎 Source Agent v11 [fact-chips, llm={'✅' if self.llm_ok else '❌'}]")

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
        return chunk.get("text", "")

    # ── Public entry point ────────────────────────────────────────────────

    def build_sources(
        self,
        answer: str,
        chunks: List[Dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        v11 flow:
          1. Extract fact chips from answer (LLM or regex)
          2. Group chips by source number (= document)
          3. For each chip, find the verbatim document sentence
          4. Build one card per document with fact_chips + legacy fields
        """
        if not answer or not chunks:
            return []

        # Step 1: extract chips
        raw_chips = _extract_fact_chips_from_answer(answer)

        # Fallback: if no chips extracted at all, use legacy claim parsing
        if not raw_chips:
            claims = _parse_claims(answer)
            if not claims:
                return []
            for c in claims:
                raw_chips.append({
                    "label":      c["clean_line"][:60],
                    "value":      c["clean_line"],
                    "src_num":    c["src_num"],
                    "is_numeric": c["claim_type"] == "numeric",
                })

        # Step 2: pre-cache doc texts per source number
        doc_cache: Dict[int, str]   = {}
        meta_cache: Dict[int, Dict] = {}
        for src_num in set(c["src_num"] for c in raw_chips):
            idx = src_num - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                doc_cache[src_num]  = self._get_doc_text(chunk, user_id=user_id, session_id=session_id)
                meta_cache[src_num] = chunk.get("metadata", {})

        # Step 3: verify each chip against its document
        verified: Dict[int, List[Dict]] = {}  # src_num → [verified chip]
        seen_lines: List[str] = []

        for chip in raw_chips:
            src_num    = chip["src_num"]
            label      = chip["label"]
            value      = chip["value"]
            is_numeric = chip["is_numeric"]

            doc_text = doc_cache.get(src_num, "")
            if not doc_text:
                continue

            raw_line = _find_raw_line_for_chip(label, value, is_numeric, doc_text)

            # Accept chip even if no raw_line found for non-numeric facts
            # (the value itself IS the evidence)
            if raw_line is None and is_numeric:
                # Try once more with just the digits
                digits_only = re.sub(r'[^\d]', '', value)
                if digits_only:
                    raw_line = _find_raw_line_for_chip(label, digits_only, True, doc_text)

            # Dedup identical lines
            if raw_line and any(_jaccard(raw_line, seen) > 0.85 for seen in seen_lines):
                raw_line = None  # different chip, same sentence — still emit chip but no raw_line

            if raw_line:
                seen_lines.append(raw_line)

            verified.setdefault(src_num, []).append({
                "label":      label,
                "value":      value,
                "raw_line":   raw_line or value,
                "is_numeric": is_numeric,
            })

        # Step 4: build one card per document
        sources: List[Dict] = []
        for src_num, chips in sorted(verified.items()):
            idx = src_num - 1
            if idx < 0 or idx >= len(chunks):
                continue
            metadata   = meta_cache.get(src_num, {})
            filename   = metadata.get("filename", f"source_{src_num}")
            has_images = metadata.get("has_images", False)
            has_tables = metadata.get("has_tables", False)

            # Deduplicate chips within this card by (label, value)
            seen_kv: set = set()
            deduped_chips: List[Dict] = []
            for ch in chips:
                key = (ch["label"].lower(), ch["value"].lower())
                if key not in seen_kv:
                    seen_kv.add(key)
                    deduped_chips.append(ch)

            if not deduped_chips:
                continue

            # Legacy compat: excerpt = first chip's raw_line
            first_excerpt = deduped_chips[0]["raw_line"]

            # Legacy sections: one section per chip (so existing renderer still works)
            sections = [
                {
                    "title":   ch["label"],
                    "lines":   [ch["raw_line"]],
                    "excerpt": ch["raw_line"],
                }
                for ch in deduped_chips
            ]

            sources.append({
                "source_number": src_num,
                "filename":      filename,
                "doc_type":      metadata.get("doc_type", "unknown"),
                "project_ref":   metadata.get("project_ref"),
                "has_images":    has_images,
                "has_tables":    has_tables,
                "excerpt":       first_excerpt,
                # ── NEW ──────────────────────────────────────────────
                "fact_chips":    deduped_chips,
                # ── LEGACY ───────────────────────────────────────────
                "sections":     sections,
                "cited_facts":  [ch["label"] for ch in deduped_chips],
                "claim_type":   "numeric" if any(ch["is_numeric"] for ch in deduped_chips) else "statement",
                "value_found":  deduped_chips[0]["value"] if deduped_chips[0]["is_numeric"] else None,
            })

        print(f"   ✅ v11: {len(sources)} source cards, "
              f"{sum(len(s['fact_chips']) for s in sources)} total chips")
        return sources


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NODE FACTORY  (LangGraph-compatible, same interface as v10)
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

        print(f"📎 Source Agent v11 node → building fact chips for answer ({len(answer)} chars)")
        sources = agent.build_sources(
            answer, chunks,
            user_id=user_id,
            session_id=session_id,
        )
        print(f"   → {len(sources)} cards, "
              f"{sum(len(s.get('fact_chips', [])) for s in sources)} chips total")

        return {**state, "sources": sources}

    return node