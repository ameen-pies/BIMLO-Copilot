"""
Source Agent v9 — Claim-Level Precision Extraction

Architecture overview
─────────────────────
The old approach extracted one excerpt per section heading (coarse-grained).
This version works at the CLAIM level:

  1. PARSE  — walk the generated answer and collect every individual claim
              sentence/bullet that cites [N].  Tag each claim as:
                - "numeric"   if it contains any number, measurement, or value
                - "table"     if it references table/row/column data
                - "statement" for any other claim

  2. JUDGE  — for each claim, build a targeted LLM prompt that asks the judge
              to locate the ONE verbatim passage in the source document (or
              table row) that backs that specific claim.  Numeric claims get
              extra emphasis on finding the exact figure.

  3. VERIFY — strict grounding check: the returned passage must contain at
              least one 4-gram that exists verbatim in the source document.
              Hallucinated passages are dropped silently.

  4. ASSEMBLE — build one source card per unique (filename, claim) pair.
              Each card has:
                sections: [{ title, lines: [verbatim passage], excerpt }]
              The card title is the claim itself (truncated), NOT the heading.
              This gives the frontend one card per backed claim.

Output shape (unchanged — fully backwards-compatible with Chat.tsx):
  [
    {
      source_number: int,
      filename:      str,
      doc_type:      str,
      excerpt:       str,        # verbatim passage from document
      sections:      [{ title: str, lines: [str], excerpt: str }],
      cited_facts:   [str],      # claim text shown in filter
      has_images:    bool,
      has_tables:    bool,
      claim_type:    str,        # "numeric" | "table" | "statement"
      value_found:   str | None, # e.g. "100m" — the specific number verified
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
# CLAIM PARSING
# ─────────────────────────────────────────────────────────────────────────────

# Numeric patterns: integers, decimals, percentages, measurements, currencies
_NUMERIC_RE = re.compile(
    r"""
    (?:
        \d+(?:[.,\s]\d+)*          # integer or decimal with optional separators
        \s*                         # optional space
        (?:                         # optional unit
            [%\$€£¥]               # currency/percent symbols
            |m(?:\b|²|³|2|3)       # metres / m² / m³
            |km\b | cm\b | mm\b    # length
            |kg\b | t\b | T\b      # weight
            |kW\b | MW\b | kV\b    # power/voltage
            |dB\b | dBm\b          # signal
            |MHz\b | GHz\b         # frequency
            |Gbps?\b | Mbps?\b | Kbps?\b  # bandwidth
            |h\b | min\b | s\b     # time
            |m/s\b | km/h\b        # speed
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
    """
    Return (claim_type, [numeric_values_found]).
    claim_type: "numeric" | "table" | "statement"
    numeric_values_found: e.g. ["100m", "2 850 000 €"]
    """
    nums = [m.group().strip() for m in _NUMERIC_RE.finditer(text) if m.group().strip() and re.search(r'\d', m.group())]
    if nums:
        return "numeric", nums
    if _TABLE_CLAIM_RE.search(text):
        return "table", []
    return "statement", []


def _parse_claims(answer: str) -> List[Dict]:
    """
    Walk the answer line by line.  For each line citing [N], produce a Claim dict:

      {
        src_num:    int,         # e.g. 2
        heading:    str,         # current ## heading (context for the judge)
        raw_line:   str,         # full original line including [N] markers
        clean_line: str,         # raw_line stripped of [N] markers and bullets
        claim_type: str,         # "numeric" | "table" | "statement"
        values:     [str],       # numeric tokens if claim_type == "numeric"
      }

    Each claim is produced ONCE per citation — if a line cites [1][3] we emit
    two separate claims so each source gets its own targeted lookup.
    """
    claims: List[Dict] = []
    current_heading = ""

    for raw_line in answer.split('\n'):
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Track headings
        h_match = re.match(r'^#{1,3}\s+(.+)', stripped)
        if h_match:
            current_heading = re.sub(r'\[\d+\]', '', h_match.group(1)).strip()
            continue

        # Find all [N] citations on this line
        cited = sorted(set(int(m) for m in re.findall(r'\[(\d+)\](?!\()', stripped)))
        if not cited:
            continue

        # Clean the line for readability
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
# GROUNDING CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _is_grounded(passage: str, doc_text: str, min_4gram_hits: int = 1) -> bool:
    """
    Return True if `passage` contains at least `min_4gram_hits` 4-word run(s)
    that appear verbatim in `doc_text`.  Protects against hallucinated passages.
    """
    words = passage.lower().split()
    if len(words) < 4:
        # Short passage — fall back to substring check
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
    """
    Return the first numeric value from `values` that is found in `passage`,
    or None if none matched.
    """
    p_lower = passage.lower()
    for v in values:
        # Collapse spaces/separators to catch "2 850 000" == "2850000"
        norm_v = re.sub(r'[\s,.]', '', v)
        if norm_v and norm_v in re.sub(r'[\s,.]', '', p_lower):
            return v
        if v.lower() in p_lower:
            return v
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, api_key: str, base_url: str, max_tokens: int = 600) -> str:
    """Shared CF Worker LLM call with retry and Groq fallback via llm_client."""
    try:
        from llm_client import call_llm
        return call_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            task="plan",
        )
    except ImportError:
        pass

    # Direct CF call fallback
    import requests
    payload = {
        "prompt": prompt,
        "task": "plan",
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                raw = resp.json().get("response") or ""
                return raw.strip() if isinstance(raw, str) else str(raw).strip()
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"      ⚠️  LLM source call {resp.status_code}: {resp.text[:80]}")
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"      ⚠️  LLM source call failed: {e}")
                return ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM → PASSAGE LOOKUP (Judge)
# ─────────────────────────────────────────────────────────────────────────────

def _build_judge_prompt(claim: Dict, doc_text: str) -> str:
    """
    Build a targeted, claim-specific prompt for the LLM judge.
    The judge must return ONLY verbatim text from doc_text that backs this claim.
    """
    doc_snippet = doc_text[:4000]
    clean_claim = claim["clean_line"]
    ctype = claim["claim_type"]
    values = claim.get("values", [])

    # Build the numeric emphasis section
    if ctype == "numeric" and values:
        val_list = ", ".join(f'"{v}"' for v in values[:5])
        numeric_instruction = (
            f"\nIMPORTANT: This claim is about NUMERIC DATA. "
            f"The specific value(s) to verify: {val_list}. "
            f"Your output MUST contain at least one of these exact figures as they appear in the document."
        )
    elif ctype == "table":
        numeric_instruction = (
            "\nIMPORTANT: This claim references TABLE DATA. "
            "Look for the relevant table row(s) in the document. "
            "Copy the table header + the specific row(s) that back this claim."
        )
    else:
        numeric_instruction = ""

    prompt = f"""You are a precise source-extraction assistant. Your only job is to find the EXACT text passage from the document that directly supports the given claim.

CLAIM TO VERIFY:
"{clean_claim}"
{numeric_instruction}

STRICT RULES — violating any rule means output nothing:
1. Copy text VERBATIM from the SOURCE DOCUMENT — zero rewording, zero summarising.
2. Output ONE passage only: the shortest contiguous sentence or table row that directly backs the claim.
3. The passage MUST appear word-for-word in the document below.
4. Do NOT include any commentary, explanation, or preamble — only the raw passage.
5. If the exact supporting text is not present in the document, output: NOTFOUND
6. If the claim references a number, your output must contain that number.
7. Strip any markdown formatting from the copied text.

SOURCE DOCUMENT:
{doc_snippet}

Verbatim passage (or NOTFOUND):"""

    return prompt


def _extract_passage_for_claim(
    claim: Dict,
    doc_text: str,
    api_key: str,
    base_url: str,
) -> Optional[str]:
    """
    Ask the judge to find the verbatim supporting passage for one claim.
    Returns the cleaned, grounded passage, or None if not found/not grounded.
    """
    prompt = _build_judge_prompt(claim, doc_text)
    raw = _call_llm(prompt, api_key, base_url, max_tokens=300)

    if not raw or raw.strip().upper().startswith("NOTFOUND"):
        return None

    # Clean leading/trailing punctuation and quote wrapping
    passage = raw.strip().strip('"').strip("'").strip()
    passage = re.sub(r'^[-–•]\s*', '', passage).strip()

    # Must be substantial
    if len(passage) < 8:
        return None

    # Grounding verification
    if not _is_grounded(passage, doc_text):
        print(f"      ⚠️  Grounding failed, dropping: {passage[:80]!r}")
        return None

    # Extra numeric check: if it's a numeric claim, the value must be in the passage
    if claim["claim_type"] == "numeric" and claim.get("values"):
        verified_val = _verify_numeric(passage, claim["values"])
        if verified_val is None:
            print(f"      ⚠️  Numeric value not found in passage — dropping: {passage[:80]!r}")
            return None

    return passage


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: KEYWORD + NUMERIC SCAN
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_passage(claim: Dict, doc_text: str) -> Optional[str]:
    """
    No-LLM fallback: scan the document sentence by sentence.
    For numeric claims: find the sentence containing the target number(s).
    For table claims: find the table row most related to the claim.
    For statements: find the sentence with the most keyword overlap.
    """
    ctype = claim["claim_type"]
    values = claim.get("values", [])
    clean_claim = claim["clean_line"]

    # Split into sentences (rough)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', doc_text) if len(s.strip()) > 10]

    # ── Numeric: find sentence containing the target value ────────────────
    if ctype == "numeric" and values:
        for sentence in sentences:
            for v in values:
                norm_v = re.sub(r'[\s,.]', '', v)
                norm_s = re.sub(r'[\s,.]', '', sentence.lower())
                if norm_v and norm_v in norm_s:
                    return sentence[:500]
        # No exact value found — return None rather than a wrong sentence
        return None

    # ── Table: find [TABLE on page N] block lines matching claim keywords ─
    if ctype == "table":
        table_blocks = re.findall(r'\[TABLE on page \d+\].*?(?=\[TABLE|\[IMAGE|$)', doc_text, re.DOTALL)
        claim_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', clean_claim))
        best_score, best_row = 0, None
        for block in table_blocks:
            for row in block.split('\n'):
                row_words = set(w.lower() for w in re.findall(r'[a-zA-Z]{3,}', row))
                score = len(claim_words & row_words)
                if score > best_score:
                    best_score, best_row = score, row.strip()
        return best_row if best_score >= 2 else None

    # ── Statement: keyword overlap scoring ───────────────────────────────
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'of', 'to', 'and',
            'or', 'for', 'with', 'from', 'that', 'this', 'it', 'on', 'at', 'by',
            'les', 'des', 'une', 'est', 'dans', 'pour', 'avec', 'qui', 'que', 'le', 'la'}
    claim_words = set(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ]{3,}', clean_claim) if w.lower() not in stop)
    if not claim_words:
        return doc_text[:300].strip()

    best_score, best_sentence = 0, None
    for sentence in sentences:
        s_words = set(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ]{3,}', sentence))
        score = len(claim_words & s_words)
        if score > best_score:
            best_score, best_sentence = score, sentence
    return best_sentence[:500] if best_sentence and best_score >= 2 else None


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
    Claim-level source extraction agent.

    For each claim in the generated answer that cites [N], the agent:
      1. Classifies the claim (numeric / table / statement)
      2. Asks the LLM judge to locate the verbatim backing passage
      3. Verifies the passage is grounded in the actual document
      4. For numeric claims: additionally verifies the target value appears
      5. Emits one source card per (source_number, claim) pair
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
        print(f"📎 Source Agent v9 [claim-level, numeric+table aware, llm={'✅' if self.llm_ok else '❌'}]")

    # ── Vector store helpers ──────────────────────────────────────────────

    def _get_all_chunks_for_file(
        self,
        filename: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch all chunks for a filename from the per-user/session collection."""
        if self.vs is None:
            return []
        try:
            # Use the scoped collection method if available (v3+ VectorStoreManager)
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

    # ── Core extraction ───────────────────────────────────────────────────

    def _get_doc_text(
        self,
        chunk: Dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Reconstruct the full document text for a chunk's source file."""
        filename = chunk.get("metadata", {}).get("filename", "")
        if filename:
            all_chunks = self._get_all_chunks_for_file(filename, user_id=user_id, session_id=session_id)
            if all_chunks:
                doc_text = _reconstruct_doc(all_chunks)
                print(f"   📄 '{filename}': {len(doc_text)} chars from {len(all_chunks)} chunks")
                return doc_text
        # Fallback: use the RAG chunk text directly
        doc_text = chunk.get("text", "")
        print(f"   📄 '{filename}': using chunk directly ({len(doc_text)} chars)")
        return doc_text

    def _process_claim(
        self,
        claim: Dict,
        doc_text: str,
        emitted_passages: List[str],
    ) -> Optional[str]:
        """
        Find the verbatim backing passage for one claim.
        Returns the passage string, or None if not found / already emitted.
        """
        ctype  = claim["claim_type"]
        values = claim.get("values", [])
        clean  = claim["clean_line"]

        print(f"      🔍 [{ctype.upper()}] {clean[:80]!r}")

        # ── LLM judge path ────────────────────────────────────────────────
        if self.llm_ok:
            passage = _extract_passage_for_claim(claim, doc_text, self.api_key, self.base_url)
            if passage:
                # Dedup against already-emitted passages
                if any(_jaccard(passage, seen) > 0.75 for seen in emitted_passages):
                    print(f"      ⏭  Global dedup dropped (jaccard): {passage[:60]!r}")
                    return None
                print(f"      ✅ {ctype} passage found: {passage[:80]!r}")
                return passage
            print(f"      ⚠️  LLM returned nothing — trying fallback")

        # ── Keyword/numeric fallback ──────────────────────────────────────
        passage = _fallback_passage(claim, doc_text)
        if passage and _is_grounded(passage, doc_text):
            if any(_jaccard(passage, seen) > 0.75 for seen in emitted_passages):
                print(f"      ⏭  Fallback dedup dropped: {passage[:60]!r}")
                return None
            print(f"      ✅ fallback passage: {passage[:80]!r}")
            return passage

        print(f"      ❌ No passage found for claim: {clean[:60]!r}")
        return None

    # ── Public entry point ────────────────────────────────────────────────

    def build_sources(
        self,
        answer: str,
        chunks: List[Dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build claim-level source cards for the generated answer.

        Args:
            answer:     The LLM-generated answer with [N] citation markers.
            chunks:     The retrieved_chunks list from the RAG state.
            user_id:    For scoped VS lookup (per-user isolation).
            session_id: For scoped VS lookup (per-session isolation).

        Returns:
            List of source card dicts compatible with Chat.tsx rendering.
            One card per (source_number, claim) pair — not per heading.
        """
        # 1. Parse all claims from the answer
        all_claims = _parse_claims(answer)
        if not all_claims:
            return []

        # 2. Pre-cache doc texts per source number (avoid re-fetching)
        doc_cache: Dict[int, str] = {}
        for claim in all_claims:
            src_num = claim["src_num"]
            if src_num not in doc_cache:
                idx = src_num - 1
                if 0 <= idx < len(chunks):
                    doc_cache[src_num] = self._get_doc_text(
                        chunks[idx], user_id=user_id, session_id=session_id
                    )

        # 3. Process each claim and build source cards
        sources: List[Dict] = []
        emitted_passages: List[str] = []  # global dedup across all cards

        # Group by src_num to emit contiguous cards per source
        from collections import OrderedDict
        claims_by_src: Dict[int, List[Dict]] = OrderedDict()
        for claim in all_claims:
            claims_by_src.setdefault(claim["src_num"], []).append(claim)

        for src_num, src_claims in claims_by_src.items():
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

            # Emit one card per claim
            for claim in src_claims:
                passage = self._process_claim(claim, doc_text, emitted_passages)
                if not passage:
                    continue

                emitted_passages.append(passage)

                # Card title: short form of the claim (≤80 chars)
                card_title = claim["clean_line"]
                if len(card_title) > 80:
                    card_title = card_title[:77] + "…"

                # value_found: the specific numeric value verified in the passage
                value_found: Optional[str] = None
                if claim["claim_type"] == "numeric" and claim.get("values"):
                    value_found = _verify_numeric(passage, claim["values"])

                sources.append({
                    "source_number": src_num,
                    "filename":      filename,
                    "doc_type":      metadata.get("doc_type", "unknown"),
                    "project_ref":   metadata.get("project_ref"),
                    "has_images":    has_images,
                    "has_tables":    has_tables,
                    # Top-level excerpt for "open document" click in card header
                    "excerpt":       passage,
                    # Sections: one section per claim card (title = the claim itself)
                    "sections": [
                        {
                            "title":   card_title,
                            "lines":   [passage],
                            "excerpt": passage,
                        }
                    ],
                    # cited_facts required by Chat.tsx filter (non-empty = show card)
                    "cited_facts":  [card_title],
                    # Extra metadata for frontend enrichment
                    "claim_type":   claim["claim_type"],
                    "value_found":  value_found,
                    "heading":      claim["heading"],
                })

        # 4. Deduplicate source_number cards that have identical passages
        #    (same source cited twice for the same fact in different lines)
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

    The node reads (answer, retrieved_chunks, session_id, user_id) from state
    and writes sources back.  session_id and user_id are passed to the VS
    for scoped document lookup (per-user/session isolation).
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
            f"📎 Source Agent v9 → {len(all_claims)} claims "
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