"""
Source Agent v7 — Per-Fact Verbatim Matching

Key improvements over v6:
  1. Title = the specific cited fact, not just the section heading
     e.g. "Architectural model: 95% complete" not "Project Status Overview"
  2. Per-number splitting: if an answer bullet has multiple key numbers,
     each number is searched independently → separate source lines
  3. Value-first scoring: numbers+units carry 10x weight over keywords
     so paraphrased LLM output still finds the right doc sentence
  4. Window expansion: short best-match sentences are padded with
     surrounding context so excerpts are always readable
  5. Section titles derived from the specific fact text, not headings
"""

from __future__ import annotations

import re
import os
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT FULL DOCUMENT FROM CHUNKS
# ────────────────────────────────────────────────────────────────────────────

def _reconstruct_doc(all_chunks_for_filename: List[Dict]) -> str:
    sorted_chunks = sorted(
        all_chunks_for_filename,
        key=lambda c: c.get("metadata", {}).get("chunk_index", 0)
    )
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


# ────────────────────────────────────────────────────────────────────────────
# VALUE EXTRACTION — numbers with units, LOD codes, antenna types etc.
# ────────────────────────────────────────────────────────────────────────────

_VALUE_RE = re.compile(
    r'(?:'
    r'\d+(?:[.,]\d+)?\s*(?:Gbps|Mbps|kbps|GHz|MHz|kHz|dBm?|ms|km|m²?|%|GB|MB|TB|W|V|A|rpm|°C)'
    r'|\b(?:LOD\s*\d{3}|\d+T\d+R)\b'
    r')',
    re.IGNORECASE,
)

def _extract_values(text: str) -> List[str]:
    return [m.group().strip().lower() for m in _VALUE_RE.finditer(text)]


# ────────────────────────────────────────────────────────────────────────────
# SPLIT ANSWER LINE INTO PER-FACT UNITS
# ────────────────────────────────────────────────────────────────────────────

def _split_into_facts(content: str) -> List[str]:
    """
    If a single line contains multiple distinct measurable values
    (e.g. "1.2Gbps downlink, 150Mbps uplink, 10ms latency") → split it
    so each value gets its own source entry.

    If fewer than 2 values found, return [content] unchanged.
    """
    values = _extract_values(content)
    if len(values) < 2:
        return [content]

    # Split on commas / semicolons / "and"
    clauses = re.split(r'[,;]|\s+and\s+|\s+et\s+', content)
    clauses = [c.strip() for c in clauses if c.strip()]

    facts = []
    used_values: set = set()

    for clause in clauses:
        clause_vals = _extract_values(clause)
        if clause_vals:
            for v in clause_vals:
                if v not in used_values:
                    used_values.add(v)
                    facts.append(clause)
                    break
        elif facts:
            # Context-only clause — merge into previous
            facts[-1] = facts[-1] + ", " + clause

    return facts if facts else [content]


# ────────────────────────────────────────────────────────────────────────────
# PARSE ANSWER INTO SECTIONS
# ────────────────────────────────────────────────────────────────────────────

def _parse_answer_sections(answer: str) -> List[Tuple[str, int, str]]:
    """
    Returns list of (title, src_num, content).

    - title  = specific fact label (NOT just the section heading)
    - src_num = [N] citation number
    - content = cleaned line text used for doc matching
    """
    sections: List[Tuple[str, int, str]] = []
    current_heading: str = ""

    for raw_line in answer.split('\n'):
        line = raw_line.strip()
        if not line:
            continue

        heading_match = re.match(r'^#{1,3}\s+(.+)', line)
        if heading_match:
            current_heading = re.sub(r'\s*\[\d+\](?!\()', '', heading_match.group(1)).strip()
            nums = [int(m) for m in re.findall(r'\[(\d+)\](?!\()', line)]
            if nums:
                content = re.sub(r'\[\d+\](?!\()', '', line).lstrip('#').strip()
                title = _make_fact_title(content, current_heading)
                sections.append((title, nums[0], content))
            continue

        nums = [int(m) for m in re.findall(r'\[(\d+)\](?!\()', line)]
        if not nums:
            continue

        src_num = nums[0]

        # Clean citation markers and bullet chars
        clean = re.sub(r'\[\d+\](?!\()', '', line).strip()
        clean = re.sub(r'^[-*•]\s*', '', clean).strip()
        # Unwrap **Label**: pattern but keep the text
        clean = re.sub(r'^\*\*([^*]+)\*\*:\s*', r'\1: ', clean)

        # Split multi-value lines
        facts = _split_into_facts(clean)

        for fact in facts:
            title = _make_fact_title(fact, current_heading)
            sections.append((title, src_num, fact))

    return sections


def _make_fact_title(content: str, section_heading: str) -> str:
    """
    Build a short, specific title for a source section card.

    Priority:
      1. Bold **Label**: value pattern → "Label: value"
      2. "Label: value" colon pattern → "Label: value snippet"
      3. Content has a measurable value → "context keyword: value"
      4. Truncate content to ~70 chars
      5. Fall back to section_heading
    """
    if not content:
        return section_heading or "Reference"

    # **Label**: rest
    bold_match = re.match(r'\*\*([^*]{2,40})\*\*[:\s]*(.*)', content)
    if bold_match:
        label = bold_match.group(1).strip()
        rest  = bold_match.group(2).strip()
        if rest:
            return f"{label}: {rest[:70]}"
        return label

    # Label: value (already de-bolded)
    colon_match = re.match(r'([A-Za-z][^:]{2,35}):\s*(.+)', content)
    if colon_match:
        label = colon_match.group(1).strip()
        value = colon_match.group(2).strip()
        return f"{label}: {value[:60]}{'…' if len(value) > 60 else ''}"

    # Has measurable value — find the keyword immediately before it
    values = _extract_values(content)
    if values:
        first_val = values[0]
        idx = content.lower().find(first_val.split()[0])
        if idx > 0:
            prefix = content[:idx].strip().rstrip(':,').strip()
            words = prefix.split()
            label = " ".join(words[-4:]) if words else ""
            if label and len(label) > 2:
                return f"{label}: {first_val}"

    # Plain truncation
    stripped = content[:70] + ("…" if len(content) > 70 else "")
    return stripped or section_heading or "Reference"


# ────────────────────────────────────────────────────────────────────────────
# VERBATIM LINE FINDER — value-first scoring
# ────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'the','les','des','and','pour','avec','dans','that','this',
    'are','was','were','has','have','from','will','been','not',
    'par','sur','une','est','qui','que','son','ses','leur',
    'also','each','both','more','than','its','into','they',
    'must','should','shall','about','which','where','when',
}

_MIN_SCORE = 8   # lowered — value matches (weight 10) dominate


def _score_tokens(content: str) -> List[Tuple[str, int]]:
    tokens: List[Tuple[str, int]] = []

    # Full value+unit combos — highest weight
    # Add BOTH the compact form AND the spaced form so "150Mbps" matches "150 Mbps"
    for m in _VALUE_RE.finditer(content):
        val = m.group().strip().lower()
        tokens.append((val, 10))
        # Also add spaced variant: "150mbps" → "150 mbps"
        spaced = re.sub(r'(\d)([a-z])', r'\1 \2', val)
        if spaced != val:
            tokens.append((spaced, 10))
        # Also add just the number part
        num_only = re.match(r'(\d+(?:[.,]\d+)?)', val)
        if num_only:
            tokens.append((num_only.group(1), 6))

    # Bare numbers not already captured
    for m in re.finditer(r'\b\d+(?:[.,]\d+)?\b', content):
        tokens.append((m.group(), 6))

    # Domain acronyms / units standalone
    for m in re.finditer(
        r'\b(?:Gbps|Mbps|kbps|MHz|GHz|kHz|dBm?|ms|km|BBU|MIMO|RAN|BIM|MEP|LOD|RFI|HVAC|FTTC)\b',
        content, re.IGNORECASE
    ):
        tokens.append((m.group().lower(), 4))

    # Long keywords
    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{6,}', content):
        w = m.group().lower()
        if w not in _STOPWORDS:
            tokens.append((w, 2))

    # Medium keywords
    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{4,5}', content):
        w = m.group().lower()
        if w not in _STOPWORDS:
            tokens.append((w, 1))

    return tokens


def _expand_sentence(best_line: str, doc_text: str, max_len: int = 220) -> str:
    """
    If best_line is short (< 60 chars), expand it with surrounding context
    to make the excerpt more readable and self-contained.
    """
    if len(best_line) >= 60:
        return best_line

    idx = doc_text.find(best_line)
    if idx == -1:
        return best_line

    # Expand backward to previous sentence/newline boundary
    start = max(0, idx - 120)
    prefix_text = doc_text[start:idx]
    for sep in ['. ', '\n', ': ']:
        pos = prefix_text.rfind(sep)
        if pos != -1:
            start = start + pos + len(sep)
            break

    # Expand forward to next sentence/newline boundary
    end_start = idx + len(best_line)
    suffix_text = doc_text[end_start:end_start + 120]
    for sep in ['. ', '\n']:
        pos = suffix_text.find(sep)
        if pos != -1:
            end_start = end_start + pos + len(sep)
            break
    else:
        end_start = end_start + len(suffix_text)

    expanded = doc_text[start:end_start].strip()
    if len(expanded) > max_len:
        expanded = expanded[:max_len] + "…"
    return expanded if expanded else best_line


def _find_verbatim_line(section_content: str, doc_text: str) -> Optional[str]:
    """
    Find the document sentence that best matches section_content.
    Returns None if best score < _MIN_SCORE.
    """
    tokens = _score_tokens(section_content)
    if not tokens:
        return None

    raw_segments = re.split(r'\n|(?<=[.!?])\s+', doc_text)
    candidates = [s.strip() for s in raw_segments if len(s.strip()) >= 10]

    if not candidates:
        return doc_text[:250] if doc_text.strip() else None

    def score(line: str) -> int:
        ll = line.lower()
        return sum(w for t, w in tokens if t in ll)

    best_score, best_line = max(((score(l), l) for l in candidates), key=lambda x: x[0])

    if best_score < _MIN_SCORE:
        print(f"      ⛔ score {best_score} < {_MIN_SCORE}, skipping")
        return None

    result = _expand_sentence(best_line, doc_text)
    print(f"      🎯 match (score {best_score}): {result[:100]!r}")
    return result


# ────────────────────────────────────────────────────────────────────────────
# DEDUP
# ────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _is_duplicate(passage: str, seen: List[str], threshold: float = 0.80) -> bool:
    return any(_jaccard(passage, s) > threshold for s in seen) if seen else False


# ────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    """
    Per-fact verbatim source extraction — no LLM calls needed.

    For each cited answer line:
      - Splits multi-value lines into individual facts
      - Finds the best-matching document sentence per fact
      - Titles each source card section with the specific fact, not just heading
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 vector_store=None):
        self.vs = vector_store
        print(f"📎 Source Agent v7 [per-fact verbatim match, min_score={_MIN_SCORE}]")

    def _get_all_chunks_for_file(self, filename: str) -> List[Dict]:
        if self.vs is None:
            return []
        try:
            raw = self.vs.collection.get(where={"filename": filename})
            chunks = []
            for i, doc_text in enumerate(raw.get("documents", [])):
                meta = raw["metadatas"][i] if raw.get("metadatas") else {}
                chunks.append({"text": doc_text, "metadata": meta})
            return chunks
        except Exception as e:
            print(f"      ⚠️  VS fetch failed for '{filename}': {e}")
            return []

    def build_sources(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        sections = _parse_answer_sections(answer)
        if not sections:
            return []

        by_source: Dict[int, List[Tuple[str, str]]] = {}
        for title, src_num, content in sections:
            by_source.setdefault(src_num, []).append((title, content))

        sources = []
        for i, chunk in enumerate(chunks):
            src_num = i + 1
            if src_num not in by_source:
                continue

            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", f"source_{src_num}")

            all_file_chunks = self._get_all_chunks_for_file(filename)
            if all_file_chunks:
                doc_text = _reconstruct_doc(all_file_chunks)
                print(f"   📄 '{filename}': {len(doc_text)} chars from {len(all_file_chunks)} chunks")
            else:
                doc_text = chunk.get("text", "")
                print(f"   📄 '{filename}': using RAG chunk ({len(doc_text)} chars)")

            sec_list = by_source[src_num]
            seen_passages: List[str] = []
            built_sections = []

            for title, section_content in sec_list:
                passage = _find_verbatim_line(section_content, doc_text)

                if passage is None:
                    print(f"      ✗ [{src_num}] '{title[:60]}' — no match above threshold, skipped")
                    continue

                if _is_duplicate(passage, seen_passages):
                    print(f"      ✗ [{src_num}] '{title[:60]}' — duplicate, skipped")
                    continue

                seen_passages.append(passage)
                built_sections.append({"title": title, "excerpt": passage})
                print(f"      ✓ [{src_num}] '{title[:60]}' → {passage[:100]!r}")

            if not built_sections:
                continue

            sources.append({
                "source_number": src_num,
                "filename":      filename,
                "doc_type":      metadata.get("doc_type", "unknown"),
                "project_ref":   metadata.get("project_ref"),
                "sections":      built_sections,
                "excerpt":       built_sections[0]["excerpt"],
                "cited_facts":   [t for t, _ in sec_list],
            })

        return sources


# ────────────────────────────────────────────────────────────────────────────
# NODE FACTORY
# ────────────────────────────────────────────────────────────────────────────

def build_sources_node(api_key: Optional[str] = None, model: Optional[str] = None,
                       vector_store=None):
    agent = SourceAgent(api_key=api_key, model=model, vector_store=vector_store)

    def node(state: dict) -> dict:
        answer = state.get("answer", "")
        chunks = state.get("retrieved_chunks", [])
        if not answer or not chunks:
            return state
        print("📎 Source Agent → extracting from full documents")
        sources = agent.build_sources(answer, chunks)
        n_sec = sum(len(s["sections"]) for s in sources)
        print(f"   → {len(sources)} source(s), {n_sec} section(s) total")
        return {**state, "sources": sources}

    return node