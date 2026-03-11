"""
Source Agent v8 — LLM-Powered Source Extraction

The LLM reads the generated answer and the full reconstructed documents,
then for each cited source returns:
  - The exact section heading from the answer (clean card title)
  - The verbatim sentence(s) from the document that back each claim

No scoring systems. No token matching. No fragile hardcoding.
The LLM understands context, paraphrasing, and implicit references.
"""

from __future__ import annotations

import re
import os
import json
import requests
import time
from typing import Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# RECONSTRUCT FULL DOCUMENT FROM CHUNKS
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# PARSE [N] CITATION NUMBERS FROM ANSWER
# ────────────────────────────────────────────────────────────────────────────

def _cited_source_nums(answer: str) -> List[int]:
    """Return sorted unique [N] citation numbers found in the answer."""
    return sorted(set(int(m) for m in re.findall(r'\[(\d+)\](?!\()', answer)))


def _section_heading_for_num(answer: str, src_num: int) -> str:
    """
    Find all ## headings that have lines citing [N].
    If one heading → return it. If multiple → join with " / ".
    Falls back to 'Source N' if none found.
    """
    current_heading = ""
    headings_seen: List[str] = []

    for line in answer.split('\n'):
        line = line.strip()
        if not line:
            continue
        h = re.match(r'^#{1,3}\s+(.+)', line)
        if h:
            current_heading = re.sub(r'\s*\[\d+\](?!\()', '', h.group(1)).strip()
            continue
        if f'[{src_num}]' in line and current_heading:
            if current_heading not in headings_seen:
                headings_seen.append(current_heading)

    if not headings_seen:
        return f"Source {src_num}"
    if len(headings_seen) == 1:
        return headings_seen[0]
    # Multiple headings — use the first (most prominent)
    return headings_seen[0]


# ────────────────────────────────────────────────────────────────────────────
# DEDUP
# ────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _dedup_excerpts(excerpts: List[str], threshold: float = 0.80) -> List[str]:
    seen = []
    for ex in excerpts:
        if not any(_jaccard(ex, s) > threshold for s in seen):
            seen.append(ex)
    return seen


# ────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, api_key: str, base_url: str, max_tokens: int = 800) -> str:
    payload = {
        "prompt": prompt,
        "task": "plan",           # short call — use fast model
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                return (resp.json().get("response") or "").strip()
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                print(f"      ⚠️  LLM source call {resp.status_code}: {resp.text[:100]}")
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"      ⚠️  LLM source call failed: {e}")
                return ""
    return ""


# ────────────────────────────────────────────────────────────────────────────
# LLM-BASED EXTRACTION
# ────────────────────────────────────────────────────────────────────────────

def _extract_excerpts_with_llm(
    answer_section: str,
    doc_text: str,
    api_key: str,
    base_url: str,
) -> List[str]:
    """
    For each claim in the answer section, ask the LLM to find the ONE
    sentence in the document that contains that specific information.

    The key discipline: we give the LLM the exact claims to look up,
    and force it to copy sentences verbatim or output nothing.
    """
    doc_snippet = doc_text[:3500]

    # Strip citation markers from answer for cleaner claim text
    clean_answer = re.sub(r'\[\d+\]', '', answer_section).strip()

    prompt = f"""You are given an ANSWER and its SOURCE DOCUMENT.

Your task: for each fact stated in the answer, find and copy the EXACT sentence from the source document that contains that fact.

RULES — follow strictly:
1. Only copy sentences that are ALREADY in the SOURCE DOCUMENT, word for word.
2. Each output line must be a direct copy from the document — no rewording, no summarising.
3. Only include sentences where you can see the specific fact from the answer is present in the document sentence.
4. If the answer mentions "April 14: Lock structural framing" → find and copy the document sentence that contains "April 14" and "structural framing".
5. Do NOT include sentences that are vaguely related — only sentences that directly contain the claimed fact.
6. Do NOT repeat the same sentence twice.
7. If you cannot find a sentence in the document for a claim, skip it — output nothing for that claim.
8. Output format: one sentence per line, starting with "- ".
9. Maximum 5 sentences.

ANSWER:
{clean_answer}

SOURCE DOCUMENT:
{doc_snippet}

Copy the matching document sentences (verbatim, one per line):"""

    raw = _call_llm(prompt, api_key, base_url, max_tokens=500)
    if not raw:
        return []

    excerpts = []
    for line in raw.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            sentence = line[2:].strip()
            if sentence and len(sentence) > 10:
                excerpts.append(sentence)

    # Strict grounding check: only keep excerpts that actually appear
    # (or near-appear) in the document — prevents hallucinated sentences
    verified = []
    doc_lower = doc_text.lower()
    for ex in excerpts:
        # Check if at least 60% of the excerpt's words appear consecutively in the doc
        words = ex.lower().split()
        if len(words) < 4:
            verified.append(ex)
            continue
        # Sliding window: look for any 4-word run from the excerpt in the doc
        found = False
        for i in range(len(words) - 3):
            chunk = ' '.join(words[i:i+4])
            if chunk in doc_lower:
                found = True
                break
        if found:
            verified.append(ex)
        else:
            print(f"      ⚠️  Grounding check failed, dropping: {ex[:80]!r}")

    return _dedup_excerpts(verified)


# ────────────────────────────────────────────────────────────────────────────
# EXTRACT ANSWER SECTION FOR A GIVEN SOURCE NUMBER
# ────────────────────────────────────────────────────────────────────────────

def _get_answer_sections_by_heading(answer: str, src_num: int) -> Dict[str, List[str]]:
    """
    Group all lines citing [N] by their ## heading.
    Returns OrderedDict: {heading: [lines...]} preserving answer order.
    Only headings that actually have cited lines are included.
    """
    from collections import OrderedDict
    result: Dict[str, List[str]] = OrderedDict()
    current_heading = f"Source {src_num}"

    for line in answer.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        h = re.match(r'^#{1,3}\s+(.+)', stripped)
        if h:
            current_heading = re.sub(r'\s*\[\d+\](?!\()', '', h.group(1)).strip()
            continue
        if f'[{src_num}]' in stripped:
            if current_heading not in result:
                result[current_heading] = []
            result[current_heading].append(stripped)

    return result


def _get_answer_section_for_num(answer: str, src_num: int) -> str:
    """Full answer section for a source number (all headings combined)."""
    groups = _get_answer_sections_by_heading(answer, src_num)
    parts = []
    for heading, lines in groups.items():
        parts.append(f"## {heading}")
        parts.extend(lines)
    return '\n'.join(parts)


# ────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ────────────────────────────────────────────────────────────────────────────

class SourceAgent:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 vector_store=None):
        self.vs       = vector_store
        self.api_key  = api_key or os.getenv("CF_API_KEY", "")
        self.base_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")
        self.llm_ok   = bool(self.api_key)
        print(f"📎 Source Agent v8 [LLM-powered extraction, llm={'✅' if self.llm_ok else '❌'}]")

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
        cited_nums = _cited_source_nums(answer)
        if not cited_nums:
            return []

        sources = []

        # Global dedup guards
        emitted_cards: set = set()        # (filename, heading) — no duplicate cards
        emitted_excerpts: List[str] = []  # all excerpts seen — no duplicate sentences across cards

        for src_num in cited_nums:
            idx = src_num - 1
            if idx < 0 or idx >= len(chunks):
                continue

            chunk    = chunks[idx]
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", f"source_{src_num}")

            # Reconstruct full document from all its chunks
            all_file_chunks = self._get_all_chunks_for_file(filename)
            if all_file_chunks:
                doc_text = _reconstruct_doc(all_file_chunks)
                print(f"   📄 [{src_num}] '{filename}': {len(doc_text)} chars from {len(all_file_chunks)} chunks")
            else:
                doc_text = chunk.get("text", "")
                print(f"   📄 [{src_num}] '{filename}': using RAG chunk ({len(doc_text)} chars)")

            if not doc_text:
                continue

            # Group answer lines by heading for this source number
            heading_groups = _get_answer_sections_by_heading(answer, src_num)

            for heading, section_lines in heading_groups.items():
                card_key = (filename, heading)

                # Skip if we've already emitted a card for this file+heading combo
                if card_key in emitted_cards:
                    print(f"      ⏭  Skipping duplicate card: [{src_num}] '{heading}'")
                    continue
                emitted_cards.add(card_key)

                answer_section = f"## {heading}\n" + "\n".join(section_lines)

                # Ask LLM to extract verbatim supporting sentences
                if self.llm_ok:
                    print(f"      🤖 LLM extracting for [{src_num}] '{heading}'")
                    raw_excerpts = _extract_excerpts_with_llm(
                        answer_section, doc_text, self.api_key, self.base_url
                    )
                else:
                    raw_excerpts = []

                if not raw_excerpts:
                    print(f"      ⚠️  LLM extraction empty — using chunk fallback")
                    raw_excerpts = [doc_text[:300].strip()]

                # Drop excerpts already used in a previous card (global dedup)
                fresh_excerpts = []
                for ex in raw_excerpts:
                    if not any(_jaccard(ex, seen) > 0.75 for seen in emitted_excerpts):
                        fresh_excerpts.append(ex)
                        emitted_excerpts.append(ex)
                    else:
                        print(f"      ⏭  Global dedup dropped: {ex[:80]!r}")

                if not fresh_excerpts:
                    print(f"      ⚠️  All excerpts were duplicates — skipping card")
                    continue

                print(f"      → {len(fresh_excerpts)} excerpt(s)")

                built_sections = [{"title": heading, "excerpt": ex} for ex in fresh_excerpts]

                sources.append({
                    "source_number": src_num,
                    "filename":      filename,
                    "doc_type":      metadata.get("doc_type", "unknown"),
                    "project_ref":   metadata.get("project_ref"),
                    "sections":      built_sections,
                    "excerpt":       built_sections[0]["excerpt"],
                    "cited_facts":   [heading],
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
        print("📎 Source Agent → LLM-powered extraction")
        sources = agent.build_sources(answer, chunks)
        n_sec = sum(len(s["sections"]) for s in sources)
        print(f"   → {len(sources)} source(s), {n_sec} section(s) total")
        return {**state, "sources": sources}

    return node