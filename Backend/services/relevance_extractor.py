"""
Relevance Extractor — map-reduce chunk filtering for full-document Q&A.

Instead of summarizing all chunks (lossy for targeted queries),
this service batches chunks → LLM selects which are relevant → returns only those.

Used by relevance_node when full_scope=True + route is rag/iterative_rag.
"""

import os
import re
from typing import Callable, Dict, List, Set

_BATCH_SIZE = int(os.getenv("RELEVANCE_BATCH_SIZE", "8"))
_MAX_TOKENS_PER_BATCH = int(os.getenv("RELEVANCE_MAX_TOKENS", "400"))


def _parse_chunk_numbers(raw: str, max_index: int) -> List[int]:
    """Extract chunk numbers from LLM response. Handles '1,3,5' or '1, 3, 5' or '1 3 5'."""
    nums = []
    for m in re.findall(r"\d+", raw):
        n = int(m)
        if 1 <= n <= max_index:  # LLM uses 1-based indexing
            nums.append(n)
    return sorted(set(nums))


def extract_relevant_chunks(
    query: str,
    chunks: List[Dict],
    llm_caller: Callable,
    batch_size: int = _BATCH_SIZE,
    max_tokens: int = _MAX_TOKENS_PER_BATCH,
) -> List[Dict]:
    """
    Map-reduce relevance filtering.

    Map: each batch → LLM picks which chunk numbers are relevant to the query.
    Reduce: merge selected chunks, deduplicate by chunk_index, sort.

    Args:
        query: user's question
        chunks: full list of document chunks (from get_all_chunks)
        llm_caller: callable(messages, temperature, max_tokens) -> str
        batch_size: chunks per LLM call
        max_tokens: max tokens per LLM response

    Returns:
        Filtered list of chunks relevant to the query. Empty list if none found.
    """
    if not chunks:
        return []

    if len(chunks) <= batch_size:
        # Small doc — skip map-reduce, return all (let synthesis handle it)
        return chunks

    n_batches = (len(chunks) + batch_size - 1) // batch_size
    selected_indices: set = set()

    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        # Build numbered chunk list for the LLM
        numbered = "\n\n".join(
            f"[CHUNK {start + i + 1}]\n{c.get('text', '')[:2000]}"
            for i, c in enumerate(batch)
        )

        prompt = f"""Given this user query, list ONLY the chunk numbers that contain relevant information.
Output format: comma-separated numbers only. Example: 1,3,5
If none are relevant, output: 0

USER QUERY: {query}

CHUNKS:
{numbered}"""

        try:
            raw = llm_caller(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            nums = _parse_chunk_numbers(raw, end)
            # Convert 1-based to 0-based
            selected_indices.update(n - 1 for n in nums if n > 0)
        except Exception as e:
            print(f"   ⚠️  relevance_extractor batch {bi+1} error: {e}")

    if not selected_indices:
        # Fallback: return all chunks if extraction found nothing
        print("   ⚠️  relevance_extractor: no relevant chunks found, returning all")
        return chunks

    filtered = [chunks[i] for i in sorted(selected_indices) if i < len(chunks)]

    # Boost image/table chunks when query signals visual/table interest
    filtered = _boost_visual_chunks(query, chunks, filtered, selected_indices)

    print(f"   🔍 relevance_extractor: {len(chunks)} → {len(filtered)} chunks")
    return filtered


# ── Visual/table chunk boosting ──────────────────────────────────────────

_IMAGE_SIGNALS = re.compile(
    r"(?:diagram|figure|image|illustration|schema|draw|chart|picture|photo|"
    r"schéma|image|figure|dessin|photo|صورة|رسم|مخطط)",
    re.IGNORECASE,
)
_TABLE_SIGNALS = re.compile(
    r"(?:table|tableau|tabular|grid|matrix|compare|comparison|versus|vs|"
    r"schéma|tableau|جدول|مقارنة)",
    re.IGNORECASE,
)
_PAGE_PATTERN = re.compile(
    r"(?:page|p\.?|pages?)\s*\d+",
    re.IGNORECASE,
)


def _boost_visual_chunks(
    query: str,
    all_chunks: List[Dict],
    filtered: List[Dict],
    selected_indices: Set[int],
) -> List[Dict]:
    """
    Ensure image/table chunks aren't accidentally dropped by relevance filtering.

    When the query mentions images/diagrams/figures, inject all has_images chunks.
    When the query mentions tables/comparisons, inject all has_tables chunks.
    When the query references a specific page, inject image/table chunks from that page.
    """
    wants_images = bool(_IMAGE_SIGNALS.search(query))
    wants_tables = bool(_TABLE_SIGNALS.search(query))
    wants_page = bool(_PAGE_PATTERN.search(query))

    if not wants_images and not wants_tables and not wants_page:
        return filtered

    boosted_indices = set(selected_indices)
    page_matches = re.findall(r"(?:page|p\.?)\s*(\d+)", query, re.IGNORECASE) if wants_page else []
    text_lower_cache: dict = {}  # cache lowered text per index

    for i, chunk in enumerate(all_chunks):
        if i in boosted_indices:
            continue
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # Inject image chunks when query asks about visuals
        if wants_images and meta.get("has_images") and "[IMAGE " in text.upper():
            boosted_indices.add(i)
            continue

        # Inject table chunks when query asks about tables/comparisons
        if wants_tables and meta.get("has_tables") and "[TABLE " in text.upper():
            boosted_indices.add(i)
            continue

        # Page-specific: inject image/table chunks matching requested page
        if page_matches:
            tl = text_lower_cache.setdefault(i, text.lower())
            for page_num in page_matches:
                if f"page {page_num}" in tl or f"[table on page {page_num}" in tl:
                    if meta.get("has_images") or meta.get("has_tables"):
                        boosted_indices.add(i)
                        break

    if len(boosted_indices) > len(selected_indices):
        filtered = [all_chunks[i] for i in sorted(boosted_indices) if i < len(all_chunks)]
        print(f"   🎨 boosted +{len(boosted_indices) - len(selected_indices)} visual/table chunks")

    return filtered
