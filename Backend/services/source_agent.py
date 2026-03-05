"""
Source Agent v6 — Direct Verbatim Line Matching

How it works:
  1. Parse the answer line-by-line, one section per cited bullet
  2. Pull ALL chunks for each cited filename from the vector store,
     reconstruct the full document text in reading order
  3. For each answer bullet, score every document line by weighted token
     overlap (numbers > units/acronyms > long keywords > medium keywords)
  4. Return the highest-scoring line — guaranteed to be verbatim from the doc

No LLM calls for extraction. The answer was generated FROM the document,
so its key tokens must appear in it.
"""

from __future__ import annotations

import re
import os
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT FULL DOCUMENT FROM CHUNKS
# ────────────────────────────────────────────────────────────────────────────

def _reconstruct_doc(all_chunks_for_filename: List[Dict]) -> str:
    """
    Given all chunks belonging to one filename, reassemble them into a single
    clean document string in chunk_index order.

    Overlapping text between consecutive chunks is deduplicated by finding the
    longest common suffix/prefix overlap and merging cleanly.
    """
    # Sort by chunk_index so reading order is preserved
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

        # Find overlap between end of result and start of this chunk
        # (chunks were created with overlap, so there is usually some)
        overlap = _find_overlap(result, text)
        result = result + "\n" + text[overlap:] if overlap < len(text) else result

    return result.strip()


def _find_overlap(a: str, b: str, max_check: int = 300) -> int:
    """
    Return the number of characters at the start of `b` that also appear
    at the end of `a` (i.e. the overlap length).  Capped at max_check chars
    to stay fast.
    """
    tail = a[-max_check:]
    # Try decreasing overlap lengths
    for length in range(min(len(tail), len(b), max_check), 20, -1):
        if tail.endswith(b[:length]):
            return length
    return 0


# ────────────────────────────────────────────────────────────────────────────
# PARSE ANSWER INTO SECTIONS
# ────────────────────────────────────────────────────────────────────────────

def _parse_answer_sections(answer: str) -> List[Tuple[str, int, str]]:
    """
    Parse [N] citation markers from the generated answer.
    Splits the answer into paragraphs and assigns each paragraph the source
    number from the first [N] marker found within it.

    Returns list of (title, source_num, section_content).
    The title is derived from the first heading (##) or first bold term in
    the paragraph — falls back to a truncated preview of the content.
    """
    # Split on double newlines to get paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', answer) if p.strip()]
    sections = []

    for para in paragraphs:
        # Find all [N] markers in this paragraph
        nums = [int(m) for m in re.findall(r'\[(\d+)\](?!\()', para)]
        if not nums:
            continue
        source_num = nums[0]

        # Derive a section title
        # 1. ## Heading
        heading_match = re.match(r'^#{1,3}\s+(.+)', para)
        if heading_match:
            title = heading_match.group(1).strip()
        else:
            # 2. First **bold** term
            bold_match = re.search(r'\*\*([^*]+)\*\*', para)
            title = bold_match.group(1).strip() if bold_match else para[:60].strip()

        # Strip citation markers from content for cleaner matching
        content = re.sub(r'\[\d+\](?!\()', '', para).strip()
        sections.append((title, source_num, content))

    return sections


# ────────────────────────────────────────────────────────────────────────────
# DIRECT VERBATIM LINE FINDER  (replaces LLM extraction + old heuristic)
# ────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'the','les','des','and','pour','avec','dans','that','this',
    'are','was','were','has','have','from','will','been','not',
    'par','sur','une','est','qui','que','son','ses','leur',
    'also','each','both','more','than','its','into','they',
    'must','should','shall','about','which','where','when',
    'target','value','level','between','install','upgrade',
}


def _find_verbatim_line(section_content: str, doc_text: str) -> str:
    """
    Find the line in doc_text that is the direct verbatim source for
    section_content.

    The LLM generated section_content FROM doc_text, so the key tokens
    (numbers, units, acronyms, long domain words) MUST appear in one of
    the document's lines.  We score every line by weighted token overlap
    and return the best match — no LLM call needed.

    Scoring weights:
      - Numbers / numeric values  → weight 4  (most discriminating)
      - Units / acronyms          → weight 3
      - Long keywords (≥6 chars)  → weight 2
      - Medium keywords (4-5 chr) → weight 1
    """
    stopwords = _STOPWORDS

    tokens: List[tuple] = []   # (token_str, weight)

    # Numbers — strip internal spaces so "1 .2 Gbps" → "1.2"
    for m in re.finditer(r'\d[\d\s,.]*\d|\d', section_content):
        t = re.sub(r'\s', '', m.group())
        if t:
            tokens.append((t, 4))

    # Units and short acronyms
    for m in re.finditer(
        r'\b(?:Gbps|Mbps|kbps|MHz|GHz|kHz|dBm?|ms|km|m²?|[A-Z]{2,6})\b',
        section_content, re.IGNORECASE
    ):
        tokens.append((m.group().lower(), 3))

    # Long keywords
    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{6,}', section_content):
        w = m.group().lower()
        if w not in stopwords:
            tokens.append((w, 2))

    # Medium keywords
    for m in re.finditer(r'[a-zA-ZÀ-ÿ]{4,5}', section_content):
        w = m.group().lower()
        if w not in stopwords:
            tokens.append((w, 1))

    if not tokens:
        lines = [l.strip() for l in doc_text.split('\n') if len(l.strip()) >= 15]
        return lines[0] if lines else doc_text[:250]

    lines = [l.strip() for l in doc_text.split('\n') if len(l.strip()) >= 15]
    if not lines:
        return doc_text[:250]

    def score_line(line: str) -> int:
        ll = line.lower()
        return sum(w for t, w in tokens if t in ll)

    best_score, best_line = max(((score_line(l), l) for l in lines), key=lambda x: x[0])

    if best_score == 0:
        return lines[0]

    print(f"      🎯 verbatim match (score {best_score}): {best_line[:100]!r}")
    return best_line


# ────────────────────────────────────────────────────────────────────────────
# DEDUP
# ────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _is_duplicate(passage: str, seen: List[str], threshold: float = 0.50) -> bool:
    return any(_jaccard(passage, s) > threshold for s in seen) if seen else False


# ────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    """
    Full-document LLM source extraction.

    Reconstructs the full text of each cited document from all its stored
    chunks, then asks the LLM to find the exact sentence in that full text
    that backs up each answer section.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 vector_store=None):
        # api_key / model kept for API compatibility but no longer used for extraction
        self.vs = vector_store   # VectorStoreManager — can be None
        print(f"📎 Source Agent v6 [full-doc, direct verbatim match]")

    # ── fetch ALL chunks for a given filename from the vector store ────────
    def _get_all_chunks_for_file(self, filename: str) -> List[Dict]:
        """
        Pull every stored chunk for `filename` from the vector store.
        Falls back to an empty list if the VS is unavailable.
        """
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

    # ── build sources ──────────────────────────────────────────────────────
    def build_sources(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """
        Main entry point.  `chunks` are the RAG-retrieved chunks; we use them
        only to map Source N → filename and as a fallback if VS is unavailable.
        """
        sections = _parse_answer_sections(answer)
        if not sections:
            return []

        # Group sections by source number
        by_source: Dict[int, List[Tuple[str, str]]] = {}
        for title, src_num, content in sections:
            by_source.setdefault(src_num, []).append((title, content))

        sources = []
        for i, chunk in enumerate(chunks):
            src_num = i + 1
            if src_num not in by_source:
                continue

            metadata = chunk.get("metadata", {})
            filename  = metadata.get("filename", f"source_{src_num}")

            # ── Reconstruct full document text ─────────────────────────────
            all_file_chunks = self._get_all_chunks_for_file(filename)
            if all_file_chunks:
                doc_text = _reconstruct_doc(all_file_chunks)
                print(f"   📄 '{filename}': reconstructed {len(doc_text)} chars "
                      f"from {len(all_file_chunks)} chunks")
            else:
                # VS unavailable — fall back to the retrieved chunk only
                doc_text = chunk.get("text", "")
                print(f"   📄 '{filename}': using RAG chunk only ({len(doc_text)} chars)")

            sec_list = by_source[src_num]
            seen_passages: List[str] = []
            built_sections = []

            for title, section_content in sec_list:
                # Direct verbatim match — find the doc line whose tokens best
                # match the answer line.  No LLM call, no guesswork.
                passage = _find_verbatim_line(section_content, doc_text)

                # Dedup only at very high similarity so similar bullets
                # (e.g. three throughput lines) aren't collapsed into one
                if _is_duplicate(passage, seen_passages, threshold=0.85):
                    lines = [l.strip() for l in doc_text.split('\n') if len(l.strip()) >= 15]
                    for l in lines:
                        if not _is_duplicate(l, seen_passages, threshold=0.85):
                            passage = l
                            break

                seen_passages.append(passage)
                built_sections.append({"title": title, "excerpt": passage})
                print(f"      ✓ [{src_num}] '{title}': {passage[:120]!r}")

            sources.append({
                "source_number": src_num,
                "filename":      filename,
                "doc_type":      metadata.get("doc_type", "unknown"),
                "project_ref":   metadata.get("project_ref"),
                "sections":      built_sections,
                "excerpt":       built_sections[0]["excerpt"] if built_sections else doc_text[:300],
                "cited_facts":   [t for t, _ in sec_list],
            })

        return sources


# ────────────────────────────────────────────────────────────────────────────
# NODE FACTORY  (drop-in for rag_engine.py)
# ────────────────────────────────────────────────────────────────────────────

def build_sources_node(api_key: Optional[str] = None, model: Optional[str] = None,
                       vector_store=None):
    """
    Returns the callable used by rag_engine.py inside synthesise.

    Pass `vector_store` (a VectorStoreManager instance) so the agent can
    reconstruct full document text.  If omitted it falls back to RAG chunks.
    """
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