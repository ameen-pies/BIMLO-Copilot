"""
Source Agent v6 — Direct Verbatim Line Matching

How it works:
  1. Parse answer line-by-line: track ## headings as the group title,
     emit one section per cited bullet/sentence (not per paragraph)
  2. Reconstruct each cited document from all stored chunks
  3. For each answer bullet, score every document sentence by weighted
     token overlap (numbers > units/acronyms > long keywords > short keywords)
  4. Only keep matches above a minimum score — discard low-confidence hits
  5. Return the highest-scoring sentence — verbatim from the document
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
# PARSE ANSWER INTO SECTIONS  (one section per cited line)
# ────────────────────────────────────────────────────────────────────────────

def _parse_answer_sections(answer: str) -> List[Tuple[str, int, str]]:
    """
    Walk the answer line by line.
    - ## headings become the current_heading (card title).
    - Each line with a [N] marker becomes ONE section:
        title   = current_heading
        src_num = first [N] on that line
        content = that line stripped of [N] markers and bullet chars
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
                sections.append((current_heading, nums[0], content))
            continue

        nums = [int(m) for m in re.findall(r'\[(\d+)\](?!\()', line)]
        if not nums:
            continue

        src_num = nums[0]

        if current_heading:
            title = current_heading
        else:
            bold_match = re.search(r'\*\*([^*]+)\*\*', line)
            if bold_match:
                title = bold_match.group(1).strip()
            else:
                preview = re.sub(r'\[\d+\](?!\()', '', line)
                preview = re.sub(r'^[-*•]\s*', '', preview).strip()
                title = preview[:60]

        content = re.sub(r'\[\d+\](?!\()', '', line).strip()
        content = re.sub(r'^[-*•]\s*', '', content).strip()
        sections.append((title, src_num, content))

    return sections


# ────────────────────────────────────────────────────────────────────────────
# DIRECT VERBATIM LINE FINDER
# ────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'the','les','des','and','pour','avec','dans','that','this',
    'are','was','were','has','have','from','will','been','not',
    'par','sur','une','est','qui','que','son','ses','leur',
    'also','each','both','more','than','its','into','they',
    'must','should','shall','about','which','where','when',
    'target','value','level','between','install','upgrade',
}

# Minimum score to accept a match — prevents irrelevant doc lines
# from appearing as sources for unrelated answer bullets.
_MIN_SCORE = 20


def _find_verbatim_line(section_content: str, doc_text: str) -> Optional[str]:
    """
    Return the single sentence in doc_text that best matches section_content.
    Returns None if no sentence scores above _MIN_SCORE (i.e. not relevant).

    Splits on both newlines and sentence boundaries so dense chunks don't
    return entire paragraphs.

    Weights:
      numbers              → 4
      units / acronyms     → 3
      long keywords ≥6     → 2
      medium keywords 4-5  → 1
    """
    tokens: List[Tuple[str, int]] = []

    for m in re.finditer(r'\d[\d\s,.]*\d|\d', section_content):
        t = re.sub(r'\s', '', m.group())
        if t:
            tokens.append((t, 4))

    for m in re.finditer(
        r'\b(?:Gbps|Mbps|kbps|MHz|GHz|kHz|dBm?|ms|km|m²?|[A-Z]{2,6})\b',
        section_content, re.IGNORECASE
    ):
        tokens.append((m.group().lower(), 3))

    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{6,}', section_content):
        w = m.group().lower()
        if w not in _STOPWORDS:
            tokens.append((w, 2))

    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{4,5}', section_content):
        w = m.group().lower()
        if w not in _STOPWORDS:
            tokens.append((w, 1))

    # Split into individual sentences (newlines + sentence boundaries)
    raw_segments = re.split(r'\n|(?<=[.!?])\s+', doc_text)
    candidates = [s.strip() for s in raw_segments if len(s.strip()) >= 15]

    if not candidates:
        return doc_text[:250] if doc_text.strip() else None

    if not tokens:
        return None

    def score(line: str) -> int:
        ll = line.lower()
        return sum(w for t, w in tokens if t in ll)

    best_score, best_line = max(((score(l), l) for l in candidates), key=lambda x: x[0])

    if best_score < _MIN_SCORE:
        print(f"      ⛔ score {best_score} < {_MIN_SCORE}, skipping")
        return None

    print(f"      🎯 match (score {best_score}): {best_line[:100]!r}")
    return best_line


# ────────────────────────────────────────────────────────────────────────────
# DEDUP
# ────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _is_duplicate(passage: str, seen: List[str], threshold: float = 0.85) -> bool:
    return any(_jaccard(passage, s) > threshold for s in seen) if seen else False


# ────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    """
    Direct verbatim source extraction — no LLM calls needed.
    Reconstructs each cited document from stored chunks, then for each
    answer bullet finds the document sentence whose tokens best match.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 vector_store=None):
        # api_key / model kept for API compatibility, not used
        self.vs = vector_store
        print(f"📎 Source Agent v6 [direct verbatim match, min_score={_MIN_SCORE}]")

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

                # Skip sections where no relevant match was found
                if passage is None:
                    print(f"      ✗ [{src_num}] '{title}' — no match above threshold, skipped")
                    continue

                # Skip near-duplicate matches
                if _is_duplicate(passage, seen_passages):
                    print(f"      ✗ [{src_num}] '{title}' — duplicate, skipped")
                    continue

                seen_passages.append(passage)
                built_sections.append({"title": title, "excerpt": passage})
                print(f"      ✓ [{src_num}] '{title}' → {passage[:100]!r}")

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