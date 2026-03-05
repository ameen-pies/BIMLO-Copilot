"""
Source Agent v5 — Full-Document LLM Extraction

How it works:
  1. From the retrieved chunks, figure out which filenames are cited in the answer
  2. Pull ALL chunks for each cited filename from the vector store (not just the
     retrieval top-k), reconstruct the full document text in reading order
  3. Give the LLM:
       - The full document text, labelled with the filename
       - The specific answer section it generated for that document
  4. Ask it: "copy the shortest sentence from the document that proves this"
  5. Dedup across sections of the same document

The LLM now reads what a human would read — the actual document — not a
RAG chunk fragment.  It also only sees ONE document at a time per call,
so it cannot confuse sources.
"""

from __future__ import annotations

import re
import os
import time
import requests
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
# LLM EXTRACTION — one doc at a time, clear instructions
# ────────────────────────────────────────────────────────────────────────────

def _extract_with_llm(
    filename: str,
    doc_text: str,
    section_title: str,
    section_content: str,
    api_key: str,
    model: str,
) -> str:
    """
    Give the LLM:
      - The full reconstructed document (trimmed to 8000 chars if needed)
      - The specific section the answer wrote about this document
    Ask it to copy the single most specific sentence from the document
    that is the direct source of that section.
    """
    # If doc is very long, trim but keep it large — we want full context
    doc_excerpt = doc_text[:8000]
    if len(doc_text) > 8000:
        doc_excerpt += "\n[... document continues ...]"

    prompt = f"""You are a citation extractor for a document Q&A system.

DOCUMENT: "{filename}"
{doc_excerpt}

---

The assistant wrote this section about the document above:

SECTION TITLE: {section_title}
SECTION TEXT:
{section_content[:600]}

---

Your task:
Find the ONE sentence in the document that is the most direct source for the section text above.
It must be a complete sentence — do not cut mid-sentence.
Copy it VERBATIM from the document. Do not paraphrase. Do not explain.
Output ONLY that sentence. Nothing else.

If multiple sentences are equally relevant, pick the shorter one.
Maximum output: 250 characters."""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                result = resp.json()["choices"][0]["message"]["content"].strip()
                result = result.strip('"\'')

                # Reject meta-commentary
                bad_starts = (
                    "i cannot", "i don't", "there is no", "no matching",
                    "the document", "i was unable", "here is", "the sentence",
                    "this sentence", "based on",
                )
                if len(result) > 15 and not result.lower().startswith(bad_starts):
                    print(f"      🤖 LLM: {result[:120]!r}")
                    return result[:260]

                print(f"      ❌ LLM rejected: {result[:80]!r}")
                return ""

            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"      ❌ HTTP {resp.status_code}: {resp.text[:80]}")
                return ""

        except Exception as e:
            print(f"      ❌ Exception: {e}")
            if attempt < 2:
                time.sleep(1)

    return ""


# ────────────────────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK — anchor-term sentence scoring (no LLM)
# ────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    'the', 'les', 'des', 'and', 'pour', 'avec', 'dans', 'that', 'this',
    'are', 'was', 'were', 'has', 'have', 'from', 'will', 'been', 'not',
    'par', 'sur', 'une', 'est', 'qui', 'que', 'son', 'ses', 'leur',
    'project', 'document', 'system', 'network', 'phase', 'source',
    'also', 'each', 'both', 'more', 'than', 'its', 'into', 'they',
}


def _extract_anchors(text: str) -> List[str]:
    anchors: set[str] = set()
    for m in re.finditer(r'\b\d[\d\s]*(?:[.,]\d+)?(?:\s*(?:dB|km|€|%|ms|PM|PBO|NRO|ONU))?\b', text):
        val = m.group().strip()
        if re.search(r'\d', val) and len(val) >= 2:
            anchors.add(val)
    for m in re.finditer(r'\b[A-Z]{2,6}\b', text):
        anchors.add(m.group())
    for m in re.finditer(r'\b[A-Z][A-Za-z0-9]*[-_.][A-Za-z0-9][-_.A-Za-z0-9]*\b', text):
        anchors.add(m.group())
    for m in re.finditer(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', text):
        anchors.add(m.group())
    for m in re.finditer(r'\b[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜ][a-záàâäéèêëîïôùûü]+-[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜ][a-z]+\b', text):
        anchors.add(m.group())
    return sorted(anchors, key=len, reverse=True)


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r'[ \t]+', ' ', text).strip()
    raw = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜŒÇ\d\"])|(?:\n+)', text)
    return [s.strip() for s in raw if len(s.strip()) >= 20]


def _heuristic_best_sentence(section_content: str, doc_text: str) -> str:
    sentences = _split_sentences(doc_text)
    if not sentences:
        return doc_text[:250]

    anchors = _extract_anchors(section_content)
    if anchors:
        scores = [sum(len(a) for a in anchors if a.lower() in s.lower()) for s in sentences]
        best = max(scores)
        if best > 0:
            return sentences[scores.index(best)]

    content_words = [w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ]{4,}', section_content)
                     if w.lower() not in _STOPWORDS]
    if content_words:
        scores = [sum(1 for w in content_words if w in s.lower()) for s in sentences]
        best = max(scores)
        if best > 0:
            return sentences[scores.index(best)]

    return sentences[0]


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
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model   = model   or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.vs      = vector_store   # VectorStoreManager — can be None
        self.use_llm = bool(self.api_key)
        mode = f"LLM ({self.model})" if self.use_llm else "heuristic"
        print(f"📎 Source Agent v5 [full-doc, {mode}]")

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
                # ── LLM extraction on full doc ─────────────────────────────
                if self.use_llm:
                    passage = _extract_with_llm(
                        filename, doc_text,
                        title, section_content,
                        self.api_key, self.model,
                    )
                    if not passage:
                        print(f"      ⚠️  LLM empty for '{title}', using heuristic")
                        passage = _heuristic_best_sentence(section_content, doc_text)
                else:
                    passage = _heuristic_best_sentence(section_content, doc_text)

                # Dedup: if same sentence picked again, use heuristic on full doc
                if _is_duplicate(passage, seen_passages):
                    sentences = _split_sentences(doc_text)
                    for s in sentences:
                        if not _is_duplicate(s, seen_passages):
                            passage = s
                            break

                seen_passages.append(passage)
                built_sections.append({"title": title, "excerpt": passage})
                print(f"      ✓ '{title}': {passage[:120]!r}")

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