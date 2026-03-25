"""
wiki_enricher.py — Wikipedia context enrichment for the define route

When the RAG engine detects a definition/concept query, this module:
  1. Extracts the term being asked about from the user query
  2. Fetches its Wikipedia page (summary + most relevant sections)
  3. Returns clean, structured WikiContext that define_node injects
     alongside the vector-store chunks before calling the LLM

Design principles:
  - Zero hardcoding — term extraction is LLM-driven
  - Graceful degradation — if Wikipedia lookup fails for any reason,
    define_node continues with RAG-only context (no crash, no empty answer)
  - Clean parsing — uses wikipediaapi's section tree, not raw HTML
  - Relevance filtering — only sections that overlap with the query topic
    are included (avoids dumping the entire Wikipedia article)
"""

from __future__ import annotations

import os
import re
import requests
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class WikiSection:
    title: str
    text: str           # cleaned plain text, no markup
    level: int          # 1 = top-level, 2 = subsection, etc.


@dataclass
class WikiContext:
    term: str                           # the resolved Wikipedia page title
    query_term: str                     # the original term extracted from the query
    url: str
    summary: str                        # opening description (~2-3 paragraphs)
    sections: List[WikiSection] = field(default_factory=list)
    found: bool = True

    def as_context_block(self, max_chars: int = 3000) -> str:
        """
        Render as a clean context block to inject into the LLM prompt.
        Format mirrors the [Source N | filename | doc_type] blocks so the
        LLM treats it consistently alongside document chunks.
        """
        if not self.found:
            return ""

        parts = [
            f"[Wikipedia | {self.term} | encyclopedia]",
            self.summary.strip(),
        ]

        budget = max_chars - len(parts[0]) - len(parts[1])
        for sec in self.sections:
            block = f"\n## {sec.title}\n{sec.text.strip()}"
            if budget - len(block) < 0:
                break
            parts.append(block)
            budget -= len(block)

        return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# TERM EXTRACTION  (LLM-driven — no hardcoding)
# ────────────────────────────────────────────────────────────────────────────

def extract_term(query: str, api_key: str, base_url: str) -> str:
    """
    Ask the LLM to pull the single term or concept the user wants explained.
    Falls back to a simple heuristic if the LLM call fails.

    Examples:
      "what is RAG?" → "RAG"
      "explain what a vector embedding is" → "vector embedding"
      "define HVAC in the context of the spec" → "HVAC"
    """
    if not api_key:
        return _heuristic_term(query)

    prompt = (
        f"Extract the single term or concept the user wants explained from this query.\n"
        f"Return ONLY the term itself — no quotes, no explanation, no extra words.\n"
        f"If the query asks about an acronym, return the acronym as written.\n\n"
        f"Query: {query}\n\n"
        f"Term:"
    )

    try:
        resp = requests.post(
            base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "max_tokens": 20, "temperature": 0.0, "task": "classify"},
            timeout=10,
        )
        if resp.status_code == 200:
            term = (resp.json().get("response") or "").strip().strip('"').strip("'")
            # Sanity check — if the model returned a sentence, fall back
            if term and len(term.split()) <= 5:
                return term
    except Exception:
        pass

    return _heuristic_term(query)


def _heuristic_term(query: str) -> str:
    """
    Simple fallback: strip common question prefixes and return the noun phrase.
    Works well for "what is X", "define X", "explain X", "what does X mean".
    """
    q = query.strip().rstrip("?.")
    patterns = [
        r"^(?:what(?:'s| is| are)|define|explain(?:\s+what)?|describe|what does .+ mean[:\s]+|tell me about)\s+(?:a\s+|an\s+|the\s+)?(.+)$",
        r"^(?:what|who|how)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, q, re.IGNORECASE)
        if m:
            term = m.group(1).strip()
            # Strip trailing context phrases like "in the document / in this context"
            term = re.sub(r"\s+(?:in|from|within|according to|based on)\s+.+$", "", term, flags=re.IGNORECASE)
            return term.strip()
    # Last resort: return last 1-3 words (most queries end with the topic)
    words = q.split()
    return " ".join(words[-min(3, len(words)):])


# ────────────────────────────────────────────────────────────────────────────
# SECTION RELEVANCE FILTER
# ────────────────────────────────────────────────────────────────────────────

_STOP = {
    "see also", "references", "further reading", "external links",
    "notes", "footnotes", "bibliography", "citations", "gallery",
}

def _is_relevant_section(section_title: str, query: str, term: str) -> bool:
    """
    Keep sections whose title overlaps with the query/term vocabulary,
    and drop boilerplate navigation sections.
    """
    t = section_title.lower().strip()
    if t in _STOP:
        return False
    # Build vocab from query + term
    vocab = set(re.findall(r'[a-z]{3,}', (query + " " + term).lower()))
    section_words = set(re.findall(r'[a-z]{3,}', t))
    # Keep if there's direct overlap or the section is short/general enough
    if section_words & vocab:
        return True
    # Always keep the first 3 top-level sections (overview, history, how it works)
    return False


def _clean_text(text: str, max_chars: int = 600) -> str:
    """Strip residual wiki markup, collapse whitespace, truncate."""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)      # {{templates}}
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)  # [[link|text]]
    text = re.sub(r"'{2,3}", '', text)              # bold/italic markers
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


# ────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA FETCHER
# ────────────────────────────────────────────────────────────────────────────

class WikiEnricher:
    """
    Thin wrapper around wikipediaapi.Wikipedia.
    Imported lazily so the rest of the engine works even if the package
    isn't installed (define_node degrades gracefully).
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._wiki = None   # lazy init
        self._available = None

    def _ensure_wiki(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import wikipediaapi
            self._wiki = wikipediaapi.Wikipedia(
                user_agent="BimloRAG/1.0 (document-assistant; contact@bimlo.app)",
                language=self.language,
                extract_format=wikipediaapi.ExtractFormat.WIKI,
            )
            self._available = True
        except ImportError:
            print("⚠️  WikiEnricher: wikipediaapi not installed — pip install wikipedia-api")
            self._available = False
        except Exception as e:
            print(f"⚠️  WikiEnricher init failed: {e}")
            self._available = False
        return self._available

    def fetch(self, term: str, query: str, language: str = "en") -> WikiContext:
        """
        Fetch Wikipedia context for `term`.
        Returns WikiContext(found=False) on any failure so callers never crash.
        """
        if language != self.language:
            # Re-init for a different language
            self.language = language
            self._wiki = None
            self._available = None

        if not self._ensure_wiki():
            return WikiContext(term=term, query_term=term, url="", summary="", found=False)

        try:
            page = self._wiki.page(term)
            if not page.exists():
                # Try capitalised variant
                page = self._wiki.page(term.title())
                if not page.exists():
                    print(f"   📖 Wikipedia: no page found for '{term}'")
                    return WikiContext(term=term, query_term=term, url="", summary="", found=False)

            summary = _clean_text(page.summary, max_chars=800)
            url = page.fullurl
            resolved_title = page.title

            # Collect relevant sections
            sections: List[WikiSection] = []
            top_level_count = 0

            def _walk(secs, level: int):
                nonlocal top_level_count
                for sec in secs:
                    if not sec.text.strip():
                        _walk(sec.sections, level + 1)
                        continue
                    if level == 1:
                        top_level_count += 1
                    # Always include first 4 top-level sections; filter the rest
                    keep = (
                        (level == 1 and top_level_count <= 4)
                        or _is_relevant_section(sec.title, query, term)
                    )
                    if keep:
                        sections.append(WikiSection(
                            title=sec.title,
                            text=_clean_text(sec.text, max_chars=500),
                            level=level,
                        ))
                    # Always recurse so we don't miss nested relevant sections
                    _walk(sec.sections, level + 1)

            _walk(page.sections, level=1)

            print(f"   📖 Wikipedia: '{resolved_title}' — {len(summary)} char summary, "
                  f"{len(sections)} section(s) kept")

            return WikiContext(
                term=resolved_title,
                query_term=term,
                url=url,
                summary=summary,
                sections=sections,
                found=True,
            )

        except Exception as e:
            print(f"   ⚠️  WikiEnricher.fetch failed for '{term}': {e}")
            return WikiContext(term=term, query_term=term, url="", summary="", found=False)


# ────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON  (one instance reused across calls)
# ────────────────────────────────────────────────────────────────────────────

_enricher = WikiEnricher()


def get_wiki_context(
    query: str,
    api_key: str,
    base_url: str,
    language: str = "en",
) -> Optional[WikiContext]:
    """
    Public entry point used by define_node.

    1. Extract the term from the query (LLM-driven)
    2. Fetch Wikipedia page for that term
    3. Return WikiContext, or None if nothing useful was found
    """
    term = extract_term(query, api_key, base_url)
    if not term:
        return None

    print(f"   🔎 WikiEnricher: extracted term '{term}' from query")
    ctx = _enricher.fetch(term, query, language=language)
    return ctx if ctx.found else None
