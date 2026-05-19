"""
RAG Helpers — Pure helper functions for the RAG engine.

No class dependencies. Only imports constants from rag_state.
"""

from __future__ import annotations

import re
from typing import Dict, List

from rag_state import MIN_CHUNKS, RELEVANCE_THRESHOLD


# ── Language detection ───────────────────────────────────────────────────

def _detect_script_language(text: str) -> str:
    """
    Quick Unicode-script-based language detection.
    Returns an ISO 639-1 code: 'ar', 'zh', 'ko', 'ru', 'he', 'th', 'hi', or 'en'.
    Falls back to keyword heuristics for French vs English in the Latin range.
    """
    if not text:
        return "en"
    # Arabic
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return 'ar'
    # Cyrillic
    if any('\u0400' <= c <= '\u04FF' for c in text):
        return 'ru'
    # CJK
    if any('\u4E00' <= c <= '\u9FFF' for c in text):
        return 'zh'
    # Korean
    if any('\uAC00' <= c <= '\uD7AF' for c in text):
        return 'ko'
    # Hebrew
    if any('\u0590' <= c <= '\u05FF' for c in text):
        return 'he'
    # Thai
    if any('\u0E00' <= c <= '\u0E7F' for c in text):
        return 'th'
    # Devanagari
    if any('\u0900' <= c <= '\u097F' for c in text):
        return 'hi'
    # Latin script — disambiguate French vs English via common words
    lower_words = set(re.sub(r'[^a-zàâçéèêëîïôûùüÿ]', ' ', text.lower()).split())
    # French signal words (high precision, whole-word match only)
    french_signals = {'bonjour', 'salut', 'merci', 'svp', 's\'il', 'vous', 'je', 'suis',
                      'nous', 'c\'est', 'dans', 'avec', 'cette', 'être', 'avoir',
                      'faire', 'mais', 'donc', 'où', 'qu\'est-ce', 'comment', 'pourquoi',
                      'quelle', 'quels', 'quelles', 'français', 'fichier', 'pourriez',
                      'donner', 'peut', 'peux', 'mon', 'ton',
                      'son', 'mes', 'tes', 'ses', 'nos', 'vos', 'leurs', 'aussi',
                      'tres', 'assez', 'beaucoup', 'encore', 'enfin', 'ensuite',
                      'depuis', 'pendant', 'parce', 'car', 'ni', 'or',
                      'moi', 'toi', 'lui', 'elle', 'eux', 'elles',
                      'ceci', 'cela', 'voici', 'il', 'on', 'ils'}
    # Unambiguous French words that don't appear in English usage
    strong_french = {'bonjour', 'salut', 'merci', 'svp', 'français', 'fichier',
                     'quelle', 'quels', 'quelles', 'pourquoi', 'comment',
                     'pourriez', 'parce', 'ensuite', 'beaucoup', 'je', 'mon', 'ton',
                     'son', 'mes', 'tes', 'ses', 'nos', 'vos', 'moi', 'toi', 'lui',
                     'elle', 'nous', 'leur', 'eux', 'elles', 'ceci', 'cela', 'voici',
                     'voilà', 'il', 'elle', 'on', 'ils', 'elles'}
    if lower_words & strong_french:
        return 'fr'
    # Weaker match — check if a threshold of French signals is present
    if len(lower_words & french_signals) >= 2:
        return 'fr'
    return 'en'


# ── Source extraction ────────────────────────────────────────────────────

def _extract_best_excerpt(chunk_text: str, answer_sentences: List[str], max_len: int = 500) -> str:
    """
    Find the passage in chunk_text most relevant to the answer sentences.
    Uses word-overlap scoring — no LLM needed.
    """
    # Build answer vocabulary (meaningful words only)
    stop = {'the','a','an','is','are','was','were','in','of','to','and','or','for',
            'with','from','that','this','it','its','on','at','by','be','as',
            'les','des','une','est','sont','dans','pour','avec','qui','que','le','la'}
    answer_words = set()
    for s in answer_sentences:
        answer_words.update(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ0-9]{3,}', s)
                           if w.lower() not in stop)

    if not answer_words:
        return chunk_text[:max_len]

    # Split chunk into sentences
    chunk_sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', chunk_text)
                       if len(s.strip()) > 15]
    if not chunk_sentences:
        return chunk_text[:max_len]

    # Score each sentence by word overlap with answer
    scores = []
    for s in chunk_sentences:
        words = set(w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ0-9]{3,}', s))
        scores.append(len(words & answer_words))

    # Find best sentence, expand window to ~max_len chars
    best = max(range(len(scores)), key=lambda i: scores[i])
    result = [chunk_sentences[best]]
    total = len(result[0])
    lo, hi = best - 1, best + 1

    while total < max_len:
        added = False
        if hi < len(chunk_sentences) and total + len(chunk_sentences[hi]) < max_len + 100:
            result.append(chunk_sentences[hi])
            total += len(chunk_sentences[hi])
            hi += 1
            added = True
        if lo >= 0 and total + len(chunk_sentences[lo]) < max_len + 100:
            result.insert(0, chunk_sentences[lo])
            total += len(chunk_sentences[lo])
            lo -= 1
            added = True
        if not added:
            break

    out = ' '.join(result)
    return out[:max_len] + ('…' if len(out) > max_len else '')


def _build_sources_from_brackets(
    answer: str,
    chunks: List[Dict],
) -> List[Dict]:
    """
    Build source cards grouped by the ## section headings in the answer.

    Walk the answer line by line, tracking the current ## heading.
    For each source [N], collect:
      - sections: list of { title (exact ## heading text), lines (cited lines under it), excerpt }
      - cited_facts: all cited lines across all sections (for the flat expandable list)
      - excerpt: best matching passage in the raw chunk (used when clicking into the doc viewer)

    One card per source number. Sections mirror the output structure exactly.
    """
    # ── Step 1: Walk the answer and group cited lines by heading ──────────────
    # Structure: { source_num: { heading: [line, ...] } }
    by_source: Dict[int, Dict[str, List[str]]] = {}
    current_heading = ""

    for line in answer.split("\n"):
        # Track ## headings — these become section titles verbatim
        heading_match = re.match(r"^#{1,3}\s+(.+)", line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            continue

        # Find every [N] cited on this line
        nums_on_line = [int(m) for m in re.findall(r"\[(\d+)\]", line)]
        if not nums_on_line:
            continue

        # Clean the line: strip [N] markers, bullets, bold markers
        clean = re.sub(r"\[\d+\]", "", line)
        clean = re.sub(r"^\s*[-*•]+\s*", "", clean)
        clean = re.sub(r"\*\*", "", clean).strip()
        if len(clean) < 8:          # skip lines with no real content
            continue

        for num in nums_on_line:
            if num not in by_source:
                by_source[num] = {}
            heading = current_heading or "General"
            if heading not in by_source[num]:
                by_source[num][heading] = []
            if clean not in by_source[num][heading]:
                by_source[num][heading].append(clean)

    # ── Step 2: Build one source card per cited [N] ───────────────────────────
    sources = []
    for num in sorted(by_source.keys()):
        idx = num - 1
        if idx < 0 or idx >= len(chunks):
            continue

        chunk    = chunks[idx]
        metadata = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")

        headings_map = by_source[num]   # { heading: [lines] }

        # All cited lines across all sections (for excerpt search and flat list)
        all_lines = [line for lines in headings_map.values() for line in lines]
        if not all_lines:
            continue

        # Find best matching passage in the raw doc chunk
        excerpt = _extract_best_excerpt(chunk_text, all_lines)

        # Build sections: title = exact heading from output, lines = cited lines under it
        # Each line gets its own doc excerpt so clicking it opens the right passage
        sections = []
        for heading, lines in headings_map.items():
            line_excerpt = _extract_best_excerpt(chunk_text, lines)
            sections.append({
                "title":   heading,
                "lines":   lines,
                "excerpt": line_excerpt,
            })

        sources.append({
            "source_number": num,
            "filename":      metadata.get("filename", "Unknown"),
            "doc_type":      metadata.get("doc_type", "unknown"),
            "project_ref":   metadata.get("project_ref"),
            "excerpt":       excerpt,
            "sections":      sections,
            "cited_facts":   all_lines[:10],
        })

    # Deduplicate by source_number — shouldn't happen but guard anyway
    seen = set()
    deduped = []
    for s in sources:
        if s['source_number'] not in seen:
            seen.add(s['source_number'])
            deduped.append(s)
    return deduped


def _find_section_in_doc(full_text: str, section_title: str) -> str:
    """Legacy helper — kept for analytics node."""
    title_words = set(
        w.lower() for w in re.findall(r'[a-zA-ZÀ-ÿ]{3,}', section_title)
        if w.lower() not in {'the', 'les', 'des', 'and', 'pour', 'avec', 'dans'}
    )
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', full_text) if len(s.strip()) > 15]
    if not sentences or not title_words:
        return full_text[:400]
    scores = [sum(1 for w in title_words if w in s.lower()) for s in sentences]
    best = max(range(len(scores)), key=lambda i: scores[i])
    result = [sentences[best]]
    total = len(result[0])
    lo, hi = best - 1, best + 1
    while total < 400:
        added = False
        if hi < len(sentences) and total + len(sentences[hi]) < 500:
            result.append(sentences[hi])
            total += len(sentences[hi])
            hi += 1
            added = True
        if lo >= 0 and total + len(sentences[lo]) < 500:
            result.insert(0, sentences[lo])
            total += len(sentences[lo])
            lo -= 1
            added = True
        if not added:
            break
    out = ' '.join(result)
    return out[:500] + ('...' if len(out) > 500 else '')


def _format_sources(chunks: List[Dict], query: str = "", llm_client=None, generated_answer: str = "") -> List[Dict]:
    """Legacy fallback — builds sources from {{}} tags or raw chunks."""
    citations = re.findall(r'\{\{([^|]+)\|Source (\d+)\}\}', generated_answer)
    if not citations:
        citations = re.findall(r'\*\*([^|*\n]{2,40})\|Source (\d+)\*\*', generated_answer)
    by_src: Dict[int, List[str]] = {}
    for title, num in citations:
        n = int(num)
        by_src.setdefault(n, [])
        t = title.strip()
        if t not in by_src[n]:
            by_src[n].append(t)
    sources = []
    for i, c in enumerate(chunks):
        n = i + 1
        if n not in by_src:
            continue
        meta = c['metadata']
        text = c['text']
        titles = by_src[n]
        sections = [{'title': t, 'excerpt': _find_section_in_doc(text, t)} for t in titles]
        sources.append({
            'source_number': n, 'filename': meta.get('filename', 'Unknown'),
            'doc_type': meta.get('doc_type', 'unknown'), 'project_ref': meta.get('project_ref'),
            'sections': sections, 'excerpt': sections[0]['excerpt'] if sections else text[:300],
            'cited_facts': titles,
        })
    return sources


# ── Context building ─────────────────────────────────────────────────────

def _build_context(chunks: List[Dict]) -> str:
    """
    Build the context block injected into the synthesis prompt.

    Chunks that contain image descriptions (produced by the vision LLM during
    ingestion) are flagged with a [has visual content] note so the LLM knows
    it can reference diagram/figure descriptions from these sources.
    Chunks with table data are similarly flagged.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        flags = []
        if m.get("has_images"):
            flags.append("has visual content")
        if m.get("has_tables"):
            flags.append("has table data")
        flag_str = f" | {', '.join(flags)}" if flags else ""
        chunk_text = c['text']
        # Use full text for smaller chunks; cap very large ones to avoid context overflow
        # 3000 chars ~ 750 tokens — safe for most LLMs with 5 chunks
        display_text = chunk_text if len(chunk_text) <= 4000 else chunk_text[:4000]
        # Draw LLM attention to visual/table chunks
        hint = ""
        if m.get("has_images") and "[IMAGE " in display_text:
            hint = "\n[VISUAL: This chunk contains image/diagram descriptions — explain them if the query asks about figures or diagrams]"
        if m.get("has_tables") and "[TABLE " in display_text:
            hint += "\n[TABLE DATA: This chunk contains tabular data — you may reorganize or summarize it in a cleaner table format]"
        parts.append(
            f"[Source {i} | {m.get('filename')} | {m.get('doc_type')}{flag_str}]\n"
            f"{display_text}{hint}"
        )
    return "\n\n".join(parts)


def _build_chart_example_hint(chunks: list) -> str:
    """
    Scan retrieved chunk text for numeric patterns and return a realistic
    chart suggestion phrased in terms of the actual document content.
    Falls back to domain-appropriate telecom/BIM examples if nothing specific is found.
    """
    FIELD_CANDIDATES = [
        # telecom / BIM / infra — highest-value signals first
        ("fiber strand",      "chart fiber strand counts per site as a bar chart"),
        ("insertion loss",    "chart insertion loss per span as a line chart"),
        ("trenching length",  "chart trenching length by section as a bar chart"),
        ("downlink",          "chart downlink throughput by site as a bar chart"),
        ("uplink",            "chart uplink throughput by site as a bar chart"),
        ("latency",           "chart end-to-end latency per node as a line chart"),
        ("bbu",               "chart BBU quantities per site as a bar chart"),
        ("rrh",               "chart RRH unit counts per site as a bar chart"),
        ("antenna",           "chart antenna unit count by site as a bar chart"),
        ("power",             "chart DC power load per site as a bar chart"),
        ("battery",           "chart battery backup autonomy by site as a bar chart"),
        ("signal",            "chart signal levels by sector as a line chart"),
        ("frequency",         "chart frequency allocation by band as a pie chart"),
        ("storey",            "chart element counts per storey as a bar chart"),
        ("floor",             "chart equipment count per floor as a bar chart"),
        ("material",          "chart material quantities as a pie chart"),
        ("temperature",       "chart temperature readings as a line chart"),
        ("load",              "chart load distribution as a bar chart"),
    ]
    combined = " ".join(c.get("text", "") for c in (chunks or [])[:6]).lower()
    for keyword, example in FIELD_CANDIDATES:
        if keyword in combined:
            return example
    return "chart fiber strand counts per site as a bar chart"


# ── Quality checks ───────────────────────────────────────────────────────

def _confidence(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    distances = [c.get("distance", 1.0) for c in chunks]
    avg = sum(distances) / len(distances)
    return round(max(0.0, 1 - avg / 2), 2)


def _is_good_retrieval(chunks: List[Dict]) -> bool:
    if len(chunks) < MIN_CHUNKS:
        return False
    # After reranking, chunks have rerank_score (0-1, higher=better) and distance=(1-score).
    # Before reranking, chunks have raw embedding distance (0-2, lower=better).
    # We accept a chunk as "good" if either:
    #   (a) rerank_score >= 0.35  (post-rerank: model thinks it's relevant)
    #   (b) distance < RELEVANCE_THRESHOLD  (pre-rerank fallback)
    def _chunk_is_good(c: Dict) -> bool:
        rerank_score = c.get("rerank_score")
        if rerank_score is not None:
            return float(rerank_score) >= 0.35
        return (c.get("distance") or 1.0) < RELEVANCE_THRESHOLD
    good = [c for c in chunks if _chunk_is_good(c)]
    return len(good) >= MIN_CHUNKS


# ── Answer cleaning ──────────────────────────────────────────────────────

def _clean_answer(text: str) -> str:
    """Hard post-process the LLM answer — preserve markdown structure."""
    # 1. Strip markdown links — keep label only
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)

    # 2. Remove "as mentioned/described in X" phrases
    text = re.sub(
        r',?\s*as (?:mentioned|described|outlined|referenced|stated|noted|discussed) in[^.\n]*',
        '', text, flags=re.IGNORECASE
    )

    # 3. Fix bold spacing — strip whitespace from INSIDE ** markers
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f'**{m.group(1).strip()}**', text)

    # 4. Ensure space BEFORE opening ** when preceded by a word char
    text = re.sub(r'(\w)(\*\*\w)', r'\1 \2', text)

    # 5. Ensure space AFTER closing ** when followed by a word char or (
    text = re.sub(r'(\w\*\*)(\w|\()', r'\1 \2', text)

    # NOTE: Rules 6+7 removed — they were collapsing \n before ** and headings,
    # turning structured markdown output into a wall of text.

    # 6. Remove orphan bullets — marker with no real content.
    # Catches: '-', '- ', '- [1]', '- [1][2]', '* ', bullet followed only by whitespace/citations
    text = re.sub(r'(?m)^\s*[-*+]\s*(\[\d+\]\s*)*\s*$', '', text)

    # 6b. Remove bullets where the only content is whitespace
    text = re.sub(r'(?m)^\s*[-*+]\s+\s*$', '', text)

    # 7. Remove lines that are only citation markers e.g. a lone '[1]'
    text = re.sub(r'(?m)^\s*(\[\d+\]\s*)+$', '', text)

    # 8. Collapse 3+ blank lines (catches gaps left by removals above)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── Doc scope resolution ─────────────────────────────────────────────────

def _resolve_doc_scope(scope_hint: str, all_docs: List[Dict], query: str = "") -> str:
    """
    Resolve a doc_scope hint (from the intent classifier) to an exact filename.

    Handles three cases without hardcoding:
      1. Exact/partial filename match  → return that filename
      2. Positional hint ("first", "second", "1", …) → return docs[idx].filename
      3. Pronoun/generic with one doc  → return the only doc's filename

    Returns "" if scope is ambiguous or can't be resolved, meaning: search all docs.
    """
    if not scope_hint or not all_docs:
        return ""

    filenames = [d.get("filename", "") for d in all_docs if d.get("filename")]
    if not filenames:
        return ""

    scope_lower = scope_hint.lower().strip()

    # ── 1. Exact or substring filename match ──────────────────────────────────
    for fname in filenames:
        if scope_lower == fname.lower() or scope_lower in fname.lower() or fname.lower() in scope_lower:
            return fname

    # ── 2. Positional ordinals ────────────────────────────────────────────────
    _ordinal_map = {
        "first": 0,   "1st": 0,   "1": 0,
        "second": 1,  "2nd": 1,   "2": 1,
        "third": 2,   "3rd": 2,   "3": 2,
        "fourth": 3,  "4th": 3,   "4": 3,
        "premier": 0, "première": 0,
        "deuxième": 1,
        "troisième": 2,
        "الأول": 0, "الأولى": 0,
        "الثاني": 1, "الثانية": 1,
        "الثالث": 2,
    }
    if scope_lower in _ordinal_map:
        idx = _ordinal_map[scope_lower]
        if idx < len(filenames):
            return filenames[idx]

    # ── 3. Generic pronoun with single doc in session ─────────────────────────
    _pronouns = {"it", "this", "that", "the file", "the document",
                 "ce fichier", "ce document", "هذا", "هذه", "له"}
    if scope_lower in _pronouns and len(filenames) == 1:
        return filenames[0]

    return ""
