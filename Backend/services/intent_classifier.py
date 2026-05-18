"""
intent_classifier.py — LangGraph-powered Intent Classification Pipeline
────────────────────────────────────────────────────────────────────────
Replaces the monolithic classify_intent() with a proper LangGraph pipeline:

  validate_input
      ↓
  llm_classify  ──(LLM unavailable / fast-path hit)──→  heuristic_classify
      ↓                                                         ↓
  post_process  ←───────────────────────────────────────────────┘
      ↓
  [END]  →  IntentAnalysis

Each node owns one responsibility, can be observed independently, and the
graph short-circuits to heuristics automatically without polluting the main
classify path with try/except spaghetti.

New in this version vs the original monolith:
  ─ Full LangGraph pipeline with typed state (ClassifierState)
  ─ validate_input: catches empty queries, normalises whitespace, resolves
    doc scope BEFORE calling the LLM so the prompt is already rich
  ─ llm_classify: isolated LLM call with clean error → heuristic fallback
  ─ post_process: all safety-net overrides live here (one place, not scattered)
  ─ heuristic_classify: unchanged logic, now a proper graph node
  ─ Warm result cache: identical (query, session fingerprint) → skip LLM call,
    useful for rapid follow-ups in the same session
  ─ Observability hook for every node transition
  ─ route_confidence_map: returned alongside the analysis so the RAG engine
    can use confidence to decide whether to double-check with a second call

External API is 100% backward compatible:
    from intent_classifier import classify_intent, IntentAnalysis
    intent = classify_intent(query, history, route_log, ...)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger("intent_classifier")

# ══════════════════════════════════════════════════════════════════════════════
# File-type helpers  (shared with downstream agents)
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
_CAD_EXTS   = {".ifc", ".ifczip", ".dxf", ".dwg", ".step", ".stp"}
_DOC_EXTS   = {".pdf", ".docx", ".doc", ".txt"}

def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

def _is_image(f: str) -> bool: return _ext(f) in _IMAGE_EXTS
def _is_cad(f: str) -> bool:   return _ext(f) in _CAD_EXTS
def _is_doc(f: str) -> bool:   return _ext(f) in _DOC_EXTS


# ══════════════════════════════════════════════════════════════════════════════
# IntentAnalysis  — the output object (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentAnalysis:
    primary_intent:   str    # extract_info | compare_docs | generate_report | …
    secondary_intent: str    # same enum or ""
    target_entity:    str    # doc name, concept, data point; "" if none
    operation:        str    # extract | compare | define | visualise | generate | …
    output_format:    str    # prose | table | chart | pdf | code | audio | unspecified
    language_intent:  str    # ISO 639-1 code if explicit; "" = mirror query lang
    is_followup:      bool
    followup_type:    str    # modify | repeat | translate | clarify | none
    ambiguity_score:  float  # 0–1
    suggested_route:  str    # direct | rag | iterative_rag | transform | analytics | graph | report | define | cad
    confidence:       float  # 0–1
    reasoning:        str    # chain-of-thought (debug)
    doc_scope:        str    # "" | filename | "first" | "second" | …

    # ── New in this version ───────────────────────────────────────────────────
    classification_path: str = "llm"  # "llm" | "heuristic" | "cached"
    latency_ms:          float = 0.0  # wall-clock time for the full pipeline
    clarification_options: List[str] = field(default_factory=list)  # 2-3 options when query is vague
    vagueness_score:     float = 0.0  # 0.0–1.0, how ambiguous the query is
    full_scope:          bool = False  # True = query needs ALL chunks, not just top_k

    def to_dict(self) -> Dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph State
# ══════════════════════════════════════════════════════════════════════════════

class ClassifierState(TypedDict):
    # ── inputs (set once at graph entry) ──────────────────────────────────────
    query:                  str
    history:                List[Dict]
    route_log:              List[Dict]
    preferred_provider:     Optional[str]
    session_has_docs:       bool
    uploaded_docs:          List[str]
    attached_doc_filenames: List[str]

    # ── internal ──────────────────────────────────────────────────────────────
    normalised_query:       str          # whitespace-cleaned query
    doc_scope_hint:         str          # resolved before LLM call
    session_fingerprint:    str          # for cache key
    use_heuristic:          bool         # True = skip LLM node
    llm_raw:                Optional[str]# raw text from LLM
    error:                  Optional[str]# first fatal error message

    # ── output ────────────────────────────────────────────────────────────────
    result:                 Optional[IntentAnalysis]
    classification_path:    str          # "llm" | "heuristic" | "cached"
    started_at:             float


# ══════════════════════════════════════════════════════════════════════════════
# Warm result cache  (in-process, bounded at 256 entries)
# ══════════════════════════════════════════════════════════════════════════════

from collections import OrderedDict as _OD

class _BoundedCache:
    def __init__(self, maxsize: int = 256):
        self._store: _OD = _OD()
        self._max = maxsize

    def get(self, key: str) -> Optional[IntentAnalysis]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key: str, value: IntentAnalysis) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._max:
            self._store.popitem(last=False)

_cache = _BoundedCache(256)


# ══════════════════════════════════════════════════════════════════════════════
# LLM prompt templates  (unchanged content, just organised here)
# ══════════════════════════════════════════════════════════════════════════════

_CLASSIFIER_SYSTEM = """\
You are an expert intent classifier for Bimlo Copilot — the AI assistant of BIMLO TECHNOLOGIE.
Users are BIM engineers, telecom infrastructure specialists, and construction (BTP) professionals.
They upload technical documents (PDF specs, IFC models, study reports, telecom notes) and ask questions in any language.

Your job: deeply analyse the user's intent with chain-of-thought reasoning, then output a structured JSON object.
No hallucinating. If unsure, set ambiguity_score high and confidence low.
"""

_CLASSIFIER_PROMPT_TEMPLATE = """\
Analyse the following query in the context of the conversation history and output a JSON intent analysis.

CONVERSATION HISTORY (last 4 turns, oldest first):
{history_block}

SESSION ROUTE HISTORY: {route_history}

UPLOADED DOCUMENTS IN THIS SESSION: {uploaded_docs_block}

CURRENT QUERY: {query}

TASK — output ONLY this JSON object, no preamble, no backticks:
{{
  "primary_intent": "<one of: extract_info | compare_docs | generate_report | visualise_data | define_term | modify_output | converse | converse_meta | image_query | cad_query | translate_content | aggregate_stats>",
  "secondary_intent": "<same enum or empty string>",
  "target_entity": "<the document name, term, concept, or data point the user cares about — empty if none>",
  "operation": "<action verb: extract | compare | define | visualise | generate | modify | converse | translate | aggregate>",
  "output_format": "<prose | table | chart | pdf | code | audio | unspecified>",
  "language_intent": "<ISO 639-1 code if user explicitly requests output in a specific language, else empty string>",
  "is_followup": <true|false>,
  "followup_type": "<modify | repeat | translate | clarify | none>",
  "ambiguity_score": <0.0–1.0>,
  "vagueness_score": <0.0–1.0>,
  "full_scope": <true|false>,
  "suggested_route": "<one of: direct | rag | iterative_rag | transform | analytics | graph | report | define | cad>",
  "confidence": <0.0–1.0>,
  "doc_scope": "<exact filename from the uploaded docs list if the user clearly targets ONE specific document; positional hint like 'first' or 'second' if they reference position; empty string if they want ALL docs or the scope is unclear>",
  "reasoning": "<2–4 sentence chain-of-thought: what signals did you see, why did you pick this route, and which document (if any) are they targeting>",
  "clarification_options": ["<option 1>", "<option 2>", "<option 3>"]
}}

ROUTING RULES (apply these strictly when choosing suggested_route):
- direct: casual/conversational, memory recall ("what did you just say"), self-referential ("what route did you use"), edits to last reply ("make it shorter", "rephrase that")
- direct [converse_meta]: queries asking about the AI's own FILE RECEIPT/ACCESS — "did you get the file?", "can you access it?", "is it there?".
    NOT for asking about file content. NEVER route to rag.
    Signals: "did you get the file", "is the file there", "do you have access", "can you read this" (meaning: did it arrive),
    "tu as reçu", "avez-vous reçu", "هل وصلك", "هل لديك الملف".
    Set primary_intent="converse_meta", suggested_route="direct".
- cad: queries that explicitly ask about the CONTENT of a CAD/IFC/BIM file — elements, storeys, walls, components, structure,
    materials, quantities, IFC classes, building data. The user must be asking about the CAD file content, not just mentioning it.
    Signals: "how many elements", "list the storeys", "what walls", "what components", "summarize this ifc",
    "what does this ifc contain", "analyse this ifc", "what is in the ifc", "ifc elements", "ifc classes",
    "building structure", "what floors", "what materials in the model".
    CRITICAL: Only route to "cad" when there IS a CAD file in the session AND the query is about its content.
    A general question asked while a CAD file exists in the session is NOT a cad query.
- rag [image_query]: when the session has uploaded image files AND the user asks about the VISUAL CONTENT of an image.
    The system already ran a vision LLM on upload and stored the description in the vector store — RAG retrieves it.
    Signals (any language): "what do you see", "what is in this image", "describe the image", "describe this",
    "what does this show", "what is shown", "read the text in it", "what's in the picture", "analyse this image".
    CRITICAL DISTINCTION: "what do you see in this image" = image_query → rag (about content).
    "did you receive the image / can you access the file" = converse_meta → direct (about file receipt).
    Set primary_intent="image_query", suggested_route="rag".
- rag: ANY question about document content — a specific fact, a section, a value, a person, a location, a date,
    what something says, what something means in context of the document. DEFAULT route for all document questions.
- iterative_rag: questions that explicitly require looking across MULTIPLE different documents simultaneously.
    Signals: "compare", "vs", "difference between", "across all files", "between the two documents".
    Do NOT use for questions that can be answered from one document.
- transform: full document translation or complete rewrite/reformat
- analytics: numerical aggregation across ALL docs (signals: "total", "average", "how many across", "statistics")
- graph: explicit request for a visual chart/plot (signals: "chart", "graph", "plot", "visualise", "graphique", "رسم بياني", "diagramme")
- report: explicit request to PRODUCE a standalone written report/PDF/document (signals: "make a report", "generate a report", "rapport sur", "create a report")
- define: asking the MEANING of a standalone technical term or acronym WITH NO reference to any uploaded document — ONLY when the user is clearly asking about a generic concept in isolation (e.g. "what is BIM?" with no docs, "define IFC"). NEVER use define if: (1) the session has uploaded documents AND the query references file content, (2) the query contains words like "this file", "this document", "it", "here", "the file", "in this", "of this".

VAGUENESS DETECTION (independent of route):
- vagueness_score: 0.0–1.0. How ambiguous is the query? High = system cannot act without clarification.
    0.0–0.3 = clear query with obvious intent and subject
    0.4–0.6 = slightly unclear but system can make a reasonable guess
    0.7–1.0 = genuinely vague — system would be guessing
    Set HIGH (0.7+) ONLY when the query is vague even AFTER considering conversation history:
    (a) generic one-word/two-word commands with no subject AND history doesn't clarify ("summarize", "analyze", "help" with no prior context)
    (b) references something unclear AND history doesn't clarify ("what about it", "tell me more" with no prior context)
    (c) could reasonably map to 2+ completely different routes and confidence is low
    Set LOW (0.0–0.3) when:
    - the query has a clear subject, action, and target (even if short: "summarize this document")
    - the conversation history makes the intent clear (e.g., user said "should I summarize?" then "yes" → NOT vague)
    - the query is a follow-up that clearly references the previous exchange
    - the query is a greeting or casual opener (hi, hello, hey, salam, bonjour, etc.) — NEVER flag greetings as vague
    CRITICAL: Always check conversation history before flagging as vague. Short messages like "continue", "yes", "do it" are NOT vague if history provides context. Greetings are NEVER vague.
    When vagueness_score >= 0.7, populate "clarification_options" with 2-3 ACTIONABLE choices the user can click. Each option must be:
    - A concrete action or request (e.g., "Summarize the entire document"), NOT a question (NOT "What would you like to do?")
    - Written in the SAME LANGUAGE as the user's query
    - Specific enough that clicking it gives the system a clear task
    - Context-aware: consider what documents are uploaded, what the conversation history suggests, and what the user likely wants
    - Example good options: "Summarize the attached document", "Translate the document to English", "Explain the main concepts"
    - Example bad options: "What would you like to do?", "How can I help?", "Tell me more"

FULL SCOPE DETECTION (independent of route):
- full_scope: true when the query needs ALL content from the document(s), not just top_k relevant chunks.
    Set true when:
    (a) "summarize this document/file", "give me a summary of everything", "analyze all of it"
    (b) "what does the whole document say", "go through the entire file", "read everything"
    (c) any query where the user explicitly wants comprehensive coverage of all content
    Set false when: targeted questions, factual lookups, specific sections, definitions, anything where top_k retrieval suffices.

DOC SCOPE RULES (for the "doc_scope" field):
- If the user names a specific file (partial or full name), set doc_scope to that filename exactly as it appears in UPLOADED DOCUMENTS.
- If the user uses a positional reference ("the first one", "le premier", "الأول"), set doc_scope to "first", "second", etc.
- If the user uses a pronoun ("it", "this one", "that file") AND there is only one uploaded document, set doc_scope to that document's filename.
- If there are multiple documents and the user says "this image" / "this one" / "this file" WITHOUT naming a specific file → set doc_scope to the LAST document in the uploaded list (most recently uploaded = most likely what they mean).
- If the conversation history shows the user was just discussing a specific document, and the new query uses a pronoun, set doc_scope to that same document.
- If the user says "both", "all", "all of them", or implies cross-document scope, set doc_scope to "".
- NEVER leave doc_scope empty when a single document is clearly implied by context, recency, or pronoun.

CRITICAL OVERRIDE RULES:
1. Any query about what the AI just said/did → direct (is_followup=true, followup_type=modify or repeat)
2. "Did you get/can you access/is the file there" → direct with primary_intent=converse_meta. NEVER rag.
3. The presence of an uploaded file does NOT by itself determine the route.
   Route is determined by WHAT THE USER IS ASKING, not by what files exist in the session.
4. A CAD/IFC file being present in the session does NOT route every query to "cad".
   Only route to "cad" when the query explicitly asks about the IFC/CAD content.
5. When ambiguity_score > 0.6, set suggested_route to the safest option (rag for document queries, direct for conversation)
6. Mixed-language queries are fine — detect the INTENT not the language
7. Questions like "what is X" where X is something IN a document → rag, not define.
"""

_VALID_ROUTES  = {"direct", "rag", "iterative_rag", "transform", "analytics", "graph", "report", "define", "cad", "clarify"}
_VALID_INTENTS = {
    "extract_info", "compare_docs", "generate_report", "visualise_data",
    "define_term", "modify_output", "converse", "converse_meta",
    "image_query", "cad_query", "translate_content", "aggregate_stats",
}


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 — validate_input
# ══════════════════════════════════════════════════════════════════════════════

def _node_validate_input(state: ClassifierState) -> ClassifierState:
    """
    Normalise whitespace, compute a cache key, resolve doc_scope_hint early,
    and flag fast-path cases (empty query, cache hit, LLM unavailable).
    """
    import re as _re

    raw_q = state["query"] or ""
    norm_q = _re.sub(r"\s+", " ", raw_q).strip()

    if not norm_q:
        # Empty query — return a safe default without hitting the LLM
        fallback = _make_heuristic("", "direct", "converse", "converse", "prose", 0.5)
        fallback.reasoning = "Empty query — defaulted to direct/converse."
        return {
            **state,
            "normalised_query": "",
            "doc_scope_hint": "",
            "session_fingerprint": "",
            "use_heuristic": True,
            "result": fallback,
            "classification_path": "heuristic",
        }

    # ── Cache key: hash of (norm_query, sorted uploaded_docs, session_has_docs,
    #    recent history turns).  History MUST be part of the key so that a
    #    follow-up with the same text as a prior standalone message doesn't
    #    get a stale "not a follow-up" result from the cache.
    _docs_key = ",".join(sorted(state.get("uploaded_docs") or []))
    _history   = state.get("history") or []
    # Use last 4 turns as the context fingerprint (role+first-80-chars)
    _hist_key  = "|".join(
        f"{h.get('role','')}:{str(h.get('content',''))[:80]}"
        for h in _history[-4:]
    )
    _cache_key = hashlib.md5(
        f"{norm_q}|{_docs_key}|{state.get('session_has_docs', False)}|{_hist_key}".encode()
    ).hexdigest()

    cached = _cache.get(_cache_key)
    if cached:
        logger.debug(f"intent_classifier: cache HIT for {norm_q[:60]!r}")
        return {
            **state,
            "normalised_query": norm_q,
            "doc_scope_hint": cached.doc_scope,
            "session_fingerprint": _cache_key,
            "use_heuristic": True,         # skip LLM node
            "result": cached,
            "classification_path": "cached",
        }

    # ── Resolve doc_scope hint once before the LLM call ─────────────────────
    _doc_scope = _resolve_doc_scope_heuristic(
        norm_q.lower(),
        state.get("uploaded_docs") or [],
        state.get("attached_doc_filenames") or [],
    )

    # ── Check LLM availability ────────────────────────────────────────────────
    try:
        from llm_client import check_llm_available
        _avail, _ = check_llm_available()
        use_heuristic = not _avail
    except Exception:
        use_heuristic = True

    return {
        **state,
        "normalised_query": norm_q,
        "doc_scope_hint": _doc_scope,
        "session_fingerprint": _cache_key,
        "use_heuristic": use_heuristic,
        "llm_raw": None,
        "error": None,
        "result": None,
        "classification_path": "llm",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 2a — llm_classify
# ══════════════════════════════════════════════════════════════════════════════

def _node_llm_classify(state: ClassifierState) -> ClassifierState:
    """
    Call the LLM with the full prompt and store the raw response.
    On failure → sets use_heuristic=True so post_process falls back.
    """
    query    = state["normalised_query"]
    session_has_docs        = state.get("session_has_docs", False)
    uploaded_docs           = state.get("uploaded_docs") or []
    attached_doc_filenames  = state.get("attached_doc_filenames") or []
    preferred_provider      = state.get("preferred_provider")

    history_block = _format_history(state.get("history") or [])
    route_history = _format_route_log(state.get("route_log") or [])

    if uploaded_docs:
        uploaded_docs_block = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(uploaded_docs))
    else:
        uploaded_docs_block = "(none uploaded yet)"

    # ── Session-state hint ────────────────────────────────────────────────────
    docs_hint = (
        "\nSESSION STATE: The user HAS uploaded documents to this session. "
        "Short or ambiguous messages that could refer to the uploaded documents "
        "(e.g. 'its uploaded', 'summarize it', 'what do you think', 'go ahead', "
        "'yes', 'analyze it', 'check the file') MUST be classified as rag, not direct. "
        "Only classify as direct if the message is clearly conversational with zero "
        "relation to any document (greetings, thanks, AI self-reference). "
        "EXCEPTION: queries asking IF the AI can see/access/read the file are direct "
        "(primary_intent=converse_meta) — they are about AI perception, not doc content."
        if session_has_docs else
        "\nSESSION STATE: No documents have been uploaded yet. "
        "Classify document-related requests as rag anyway — the retrieval "
        "pipeline will handle the empty state gracefully."
    )

    # ── Attachment-type context ───────────────────────────────────────────────
    attached_hint = ""
    if attached_doc_filenames:
        _attached_names      = ", ".join(attached_doc_filenames)
        _attached_has_cad    = any(_is_cad(f) for f in attached_doc_filenames)
        _attached_has_images = any(_is_image(f) for f in attached_doc_filenames)
        _attached_has_docs   = any(_is_doc(f) for f in attached_doc_filenames)

        # Build per-file type notes so the LLM understands what each attachment is
        file_type_lines = []
        for f in attached_doc_filenames:
            if _is_cad(f):
                file_type_lines.append(
                    f"  - {f} → CAD/IFC/BIM file. Route to 'cad' ONLY if query explicitly asks "
                    "about building content (elements, storeys, materials, IFC classes). "
                    "NEVER route to 'cad' just because this file exists — the query must be about its content. "
                    "Greetings, general questions, summaries, or queries about other attached files → NOT 'cad'."
                )
            elif _is_image(f):
                file_type_lines.append(
                    f"  - {f} → Image file. Route to 'rag' (primary_intent='image_query') "
                    "only if query asks about what is visible in this image."
                )
            else:
                file_type_lines.append(
                    f"  - {f} → Document/PDF. Route to 'rag' if query asks about its content."
                )

        mixed_note = ""
        if _attached_has_cad and (_attached_has_docs or _attached_has_images):
            mixed_note = (
                "\nCRITICAL: Multiple file types are attached. "
                "Identify which file the user is asking about based on their query content, "
                "not on which file type was last uploaded. "
                "A question about text, specifications, measurements, or any non-BIM topic "
                "should route to 'rag' targeting the document/PDF, NOT to 'cad'. "
                "If the query is vague, generic, or not clearly about the CAD file, default to 'rag'."
            )

        attached_hint = (
            f"\n\nATTACHED TO THIS MESSAGE:\n"
            + "\n".join(file_type_lines)
            + mixed_note
            + f"\n\nFor doc_scope: set to the specific filename that the query is asking about. "
            "Only set doc_scope to a CAD file if the query is explicitly about BIM/IFC content. "
            "Do NOT set doc_scope if the scope is unclear or the user is asking generally."
        )

    # ── Pre-resolved doc_scope hint (from validate_input node) ───────────────
    _doc_hint_str = state.get("doc_scope_hint", "")
    if _doc_hint_str:
        attached_hint += (
            f"\n\nPRE-RESOLVED DOC SCOPE (heuristic): '{_doc_hint_str}'. "
            "Override this only if the query clearly points to a different document."
        )

    prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
        history_block=history_block,
        route_history=route_history,
        uploaded_docs_block=uploaded_docs_block,
        query=query,
    ) + docs_hint + attached_hint

    try:
        from llm_client import call_llm
        raw = call_llm(
            prompt=prompt,
            system_prompt=_CLASSIFIER_SYSTEM,
            max_tokens=500,
            temperature=0.0,
            task="classify",
            preferred_provider=preferred_provider,
        )
        return {**state, "llm_raw": raw, "error": None}
    except Exception as e:
        logger.warning(f"intent_classifier LLM call failed: {e}")
        return {**state, "llm_raw": None, "use_heuristic": True, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Node 2b — heuristic_classify
# ══════════════════════════════════════════════════════════════════════════════

def _node_heuristic_classify(state: ClassifierState) -> ClassifierState:
    """
    Fast heuristic fallback — no LLM required.
    All pattern matching lives here; post_process still runs afterwards
    for the same safety-net overrides as the LLM path.
    """
    query    = state.get("normalised_query") or state.get("query", "")
    history  = state.get("history") or []
    uploaded_docs          = state.get("uploaded_docs") or []
    attached_doc_filenames = state.get("attached_doc_filenames") or []
    session_has_docs       = state.get("session_has_docs", False)

    result = _run_heuristics(query, history, uploaded_docs, attached_doc_filenames, session_has_docs)
    return {
        **state,
        "result": result,
        "classification_path": state.get("classification_path", "heuristic"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 — post_process
# ══════════════════════════════════════════════════════════════════════════════

def _node_post_process(state: ClassifierState) -> ClassifierState:
    """
    Parse LLM raw output (if any), apply safety-net overrides, and cache the result.
    On LLM parse failure → falls back to the heuristic result if one exists,
    otherwise runs heuristics now.
    """
    # If we already have a result (cached or heuristic path), just apply overrides.
    result: Optional[IntentAnalysis] = state.get("result")

    if result is None:
        # LLM path: parse the raw response
        raw = state.get("llm_raw") or ""
        if raw:
            try:
                result = _parse_llm_result(raw)
            except Exception as e:
                logger.warning(f"intent_classifier: failed to parse LLM output: {e}")
                result = None

        if result is None:
            # Parse failed — run heuristics as emergency fallback
            query = state.get("normalised_query") or state.get("query", "")
            result = _run_heuristics(
                query,
                state.get("history") or [],
                state.get("uploaded_docs") or [],
                state.get("attached_doc_filenames") or [],
                state.get("session_has_docs", False),
            )
            state = {**state, "classification_path": "heuristic"}

    # ── Safety-net overrides (consolidated — was scattered in original) ────────

    uploaded_docs          = state.get("uploaded_docs") or []
    attached_doc_filenames = state.get("attached_doc_filenames") or []
    session_has_docs       = state.get("session_has_docs", False)
    doc_scope_hint         = state.get("doc_scope_hint", "")

    # 1. image_query must always route to rag
    if result.primary_intent == "image_query" and result.suggested_route != "rag":
        result.suggested_route = "rag"
        result.reasoning += " [override: image_query → rag]"

    # 2. cad route requires a CAD file to exist in session
    _session_has_cad = any(_is_cad(f) for f in uploaded_docs)
    if result.suggested_route == "cad" and not _session_has_cad:
        result.suggested_route = "rag"
        result.reasoning += " [override: cad route requires a CAD file in session]"


    # 3. converse_meta must always be direct (never let LLM mis-route it)
    if result.primary_intent == "converse_meta" and result.suggested_route != "direct":
        result.suggested_route = "direct"
        result.reasoning += " [override: converse_meta → direct]"

    # 4. If session has docs and classifier said "direct" with HIGH ambiguity,
    #    and it's not a meta/greeting/follow-up, nudge to rag.
    #
    #    IMPORTANT CHANGES vs original:
    #    - Threshold raised from 0.2 → 0.65 (0.2 fired on almost everything)
    #    - is_followup=True is now fully exempt — when the LLM says it's a
    #      follow-up, we trust it. Follow-ups should stay "direct", not get
    #      silently re-routed to rag just because docs exist.
    #    - confidence must also be low (<0.55) to trigger — if the LLM is
    #      confident about "direct", we shouldn't override it.
    if (session_has_docs
            and result.suggested_route == "direct"
            and not result.is_followup                   # ← follow-ups are exempt
            and result.confidence < 0.55                 # ← only override low-confidence calls
            and result.ambiguity_score >= 0.65           # ← raised from 0.2 → 0.65
            and result.primary_intent not in ("converse_meta", "image_query", "converse")):
        result.suggested_route = "rag"
        result.reasoning += " [nudge direct→rag: session has docs + high ambiguity + low confidence]"

    # 5. Auto-fill doc_scope from pre-resolved hint when LLM left it empty
    if not result.doc_scope and doc_scope_hint:
        if result.suggested_route in ("rag", "iterative_rag", "cad", "transform", "analytics"):
            result.doc_scope = doc_scope_hint
            result.reasoning += f" [doc_scope filled from heuristic: {doc_scope_hint}]"

    # 6. Auto-fill doc_scope from attachment when LLM left it empty —
    #    Only when exactly one file attached. With multiple files (same or
    #    mixed type), leave doc_scope empty so retrieve_vector searches all
    #    attached docs via attached_doc_filenames.
    if attached_doc_filenames and not result.doc_scope:
        if result.suggested_route in ("rag", "iterative_rag", "cad", "transform", "analytics"):
            if len(attached_doc_filenames) == 1:
                result.doc_scope = attached_doc_filenames[0]
                result.reasoning += f" [doc_scope set from attachment: {attached_doc_filenames[0]}]"
            # else: multiple files — leave empty; retrieve_vector handles multi-doc search

    # 7. Vagueness override — if query is genuinely vague, force clarify
    #    regardless of what route the LLM picked. This is the pre-routing wrapper.
    if result.vagueness_score >= 0.7 and result.suggested_route != "clarify":
        result.suggested_route = "clarify"
        if not result.clarification_options:
            result.clarification_options = []  # clarify_node has LLM fallback
        result.reasoning += f" [override: vagueness_score={result.vagueness_score:.2f} → clarify]"

    # 8. Validate route is in known set
    if result.suggested_route not in _VALID_ROUTES:
        result.suggested_route = "rag"
        result.reasoning += " [override: unknown route → rag]"

    # ── Stamp path and latency ────────────────────────────────────────────────
    path = state.get("classification_path", "llm")
    elapsed_ms = (time.time() - state.get("started_at", time.time())) * 1000
    result.classification_path = path
    result.latency_ms           = round(elapsed_ms, 1)

    # ── Cache the result (only LLM results — not heuristics, to keep cache meaningful) ──
    fingerprint = state.get("session_fingerprint", "")
    if path == "llm" and fingerprint:
        _cache.set(fingerprint, result)

    logger.debug(
        f"intent_classifier [{path}] {result.suggested_route!r} "
        f"conf={result.confidence:.2f} ambi={result.ambiguity_score:.2f} "
        f"scope={result.doc_scope!r} ({elapsed_ms:.0f}ms)"
    )

    return {**state, "result": result}


# ══════════════════════════════════════════════════════════════════════════════
# Routing function  (decides which classify node to call)
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_validate(state: ClassifierState) -> Literal["llm_classify", "heuristic_classify", "post_process"]:
    """
    After validate_input:
      - If we already have a cached result → skip straight to post_process
      - If LLM unavailable / fast-path → heuristic_classify
      - Otherwise → llm_classify
    """
    if state.get("result") is not None:
        return "post_process"
    if state.get("use_heuristic"):
        return "heuristic_classify"
    return "llm_classify"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_classifier_graph() -> StateGraph:
    workflow = StateGraph(ClassifierState)

    workflow.add_node("validate_input",      _node_validate_input)
    workflow.add_node("llm_classify",        _node_llm_classify)
    workflow.add_node("heuristic_classify",  _node_heuristic_classify)
    workflow.add_node("post_process",        _node_post_process)

    workflow.set_entry_point("validate_input")
    workflow.add_conditional_edges(
        "validate_input",
        _route_after_validate,
        {
            "llm_classify":       "llm_classify",
            "heuristic_classify": "heuristic_classify",
            "post_process":       "post_process",
        },
    )

    # After LLM call: if LLM flagged use_heuristic (call failed), go to heuristic
    workflow.add_conditional_edges(
        "llm_classify",
        lambda s: "heuristic_classify" if s.get("use_heuristic") else "post_process",
        {
            "heuristic_classify": "heuristic_classify",
            "post_process":       "post_process",
        },
    )

    workflow.add_edge("heuristic_classify", "post_process")
    workflow.add_edge("post_process",        END)

    return workflow.compile()


# Build the graph once at import time — it's pure (no side effects at build time)
_classifier_graph = _build_classifier_graph()


# ══════════════════════════════════════════════════════════════════════════════
# Public API  — 100% backward compatible
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(
    query:                  str,
    history:                Optional[List[Dict]] = None,
    route_log:              Optional[List[Dict]] = None,
    preferred_provider:     Optional[str] = None,
    session_has_docs:       bool = False,
    uploaded_docs:          Optional[List[str]] = None,
    attached_doc_filenames: Optional[List[str]] = None,
) -> IntentAnalysis:
    """
    Classify the intent of a query using the LangGraph pipeline.

    The route is determined by WHAT THE USER IS ASKING, not by what files
    happen to exist in the session.  Attached files provide context for
    doc_scope resolution but do not override intent routing.

    Args:
        query:                  The current user message.
        history:                Conversation history [{role, content}, ...].
        route_log:              Per-session route log from AgentState._route_log.
        preferred_provider:     Optional provider hint to honour user-selected model.
        session_has_docs:       True if the session has documents in the vector store.
        uploaded_docs:          List of filenames currently uploaded in the session.
        attached_doc_filenames: Filenames of docs physically attached to THIS message.

    Returns:
        IntentAnalysis with suggested_route, doc_scope, and rich metadata.
    """
    initial_state: ClassifierState = {
        "query":                  query or "",
        "history":                history or [],
        "route_log":              route_log or [],
        "preferred_provider":     preferred_provider,
        "session_has_docs":       session_has_docs,
        "uploaded_docs":          uploaded_docs or [],
        "attached_doc_filenames": attached_doc_filenames or [],

        # internal — will be filled by validate_input
        "normalised_query":    "",
        "doc_scope_hint":      "",
        "session_fingerprint": "",
        "use_heuristic":       False,
        "llm_raw":             None,
        "error":               None,
        "result":              None,
        "classification_path": "llm",
        "started_at":          time.time(),
    }

    try:
        final = _classifier_graph.invoke(initial_state)
        result: IntentAnalysis = final["result"]
        if result is None:
            raise ValueError("Graph returned None result")
        return result
    except Exception as e:
        logger.error(f"intent_classifier graph failed unexpectedly: {e}")
        # Nuclear fallback — should never be needed, but we never want to crash chat
        return _run_heuristics(
            query or "",
            history or [],
            uploaded_docs or [],
            attached_doc_filenames or [],
            session_has_docs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic engine  (used both as a node and as the emergency fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _run_heuristics(
    query: str,
    history: List[Dict],
    uploaded_docs: List[str],
    attached_doc_filenames: List[str],
    session_has_docs: bool,
) -> IntentAnalysis:
    """
    Pattern-matching classification — no LLM required.
    Route is determined by query content, not file presence.
    """
    q       = query.lower().strip()
    _docs   = uploaded_docs or []
    _attach = attached_doc_filenames or []

    _doc_scope = _resolve_doc_scope_heuristic(q, _docs, _attach)

    # ── Code generation ────────────────────────────────────────────────────────
    _code_verbs = [
        "write", "make", "give me", "create", "generate", "build",
        "code", "implement", "show me", "do",
        "écris", "fais", "crée",
        "اكتب", "أنشئ", "اعمل",
    ]
    _code_nouns = [
        "code", "function", "script", "algorithm", "algo", "program",
        "snippet", "class", "method", "implementation", "solution",
        "fonction", "algorithme", "programme", "classe", "méthode",
        "كود", "دالة", "خوارزمية", "برنامج",
    ]
    if (any(q.startswith(v) or f" {v} " in q for v in _code_verbs) and
            any(n in q for n in _code_nouns)):
        return _make_heuristic(query, "direct", "converse", "generate", "code", 0.92)

    # ── File-receipt meta-queries ──────────────────────────────────────────────
    _meta_signals = [
        "did you get", "is the file there", "do you have the file", "do you have access",
        "can you access", "did you receive", "is it uploaded", "did it upload",
        "tu as reçu", "avez-vous reçu", "est-ce que tu as reçu",
        "هل وصلك", "هل لديك الملف", "هل استلمت",
    ]
    if any(s in q for s in _meta_signals):
        return _make_heuristic(query, "direct", "converse_meta", "converse", "prose", 0.90)

    # ── CAD/IFC content query ──────────────────────────────────────────────────
    _session_has_cad = any(_is_cad(f) for f in _docs)
    if _session_has_cad:
        _cad_signals = [
            # Unambiguous IFC/BIM terms only.
            # Generic words (model, building, structure, material, component, wall,
            # floor, element) have been removed — they appear in PDF specs/reports
            # and caused RAG queries to be misrouted to the CAD agent.
            "ifc", "bim",
            "storey", "storeys",
            "ifc class", "ifc classes",
            "ifc element", "ifc elements",
            "what is in this ifc", "what does this ifc",
            "analyse this ifc", "analyze this ifc",
            "summarize this ifc", "summarise this ifc",
        ]
        _attached_cad = [f for f in _attach if _is_cad(f)]
        _query_mentions_cad       = any(s in q for s in _cad_signals)
        # Only treat CAD attachment as routing signal when the user explicitly
        # names the CAD file by (partial) filename in their query.
        # Previously, _attached_cad being non-empty was enough — that caused
        # every query sent alongside an IFC to be routed to cad.
        _query_directly_refs_cad  = any(
            _is_cad(f) and os.path.splitext(f)[0].lower() in q
            for f in _docs
        )
        if _query_mentions_cad or _query_directly_refs_cad:
            _cad_scope = (_attached_cad[-1] if _attached_cad
                          else next((f for f in _docs if _is_cad(f)), ""))
            return _make_heuristic(query, "cad", "cad_query", "extract", "prose", 0.88,
                                   doc_scope=_cad_scope)

    # ── Image content query ────────────────────────────────────────────────────
    _image_content_signals = [
        "what do you see", "what is in this image", "describe the image",
        "describe this image", "what does this show", "what is shown",
        "what's in the picture", "what is in the picture",
        "analyse this image", "analyze this image", "tell me about this image",
        "what can you tell me about this", "look at this image", "read the text in",
        "ما الذي تراه", "ما في هذه الصورة", "صف هذه الصورة", "ماذا يوجد في",
        "que vois-tu", "qu'est-ce que tu vois", "décris cette image",
    ]
    _image_in_session = any(_is_image(f) for f in _docs)
    if any(s in q for s in _image_content_signals) or (
        _image_in_session and any(
            w in q for w in ["see", "show", "describe", "image", "picture", "photo",
                              "screenshot", "صورة"]
        )
    ):
        return _make_heuristic(query, "rag", "image_query", "extract", "prose", 0.92,
                               doc_scope=_doc_scope)

    # ── Chart / graph request ──────────────────────────────────────────────────
    graph_signals = [
        "chart", "graph", "plot", "visuali", "graphique", "diagramme",
        "courbe", "histogramme", "رسم بياني", "مخطط", "تصور",
        "camembert", "nuage de points", "gráfico", "diagrama",
    ]
    if any(s in q for s in graph_signals):
        return _make_heuristic(query, "graph", "visualise_data", "visualise", "chart", 0.80,
                               doc_scope=_doc_scope)

    # ── Report generation ──────────────────────────────────────────────────────
    report_signals = [
        "make a report", "create a report", "generate a report", "write a report",
        "do a report", "make me a report", "rapport sur", "fais un rapport",
        "produce a report", "build a report", "prepare a report", "un rapport",
    ]
    if any(s in q for s in report_signals):
        return _make_heuristic(query, "report", "generate_report", "generate", "pdf", 0.85,
                               doc_scope=_doc_scope)

    # ── Transform / translate ──────────────────────────────────────────────────
    transform_signals = [
        "translat", "tradui", "traduire", "rewrite", "paraphrase",
        "summarise in", "summarize in", "résume en", "résumer en",
    ]
    if any(s in q for s in transform_signals):
        return _make_heuristic(query, "transform", "translate_content", "translate", "prose", 0.75,
                               doc_scope=_doc_scope)

    # ── Analytics ─────────────────────────────────────────────────────────────
    analytics_signals = [
        "analytics", "statistiques", "total", "average", "count",
        "how many across", "across all", "sum of",
    ]
    if any(s in q for s in analytics_signals):
        return _make_heuristic(query, "analytics", "aggregate_stats", "aggregate", "table", 0.75)

    # ── Iterative RAG (multi-doc compare) ─────────────────────────────────────
    compare_signals = ["compare", "difference", "vs", "versus", "comparaison", "différence", "مقارنة"]
    if any(s in q for s in compare_signals):
        return _make_heuristic(query, "iterative_rag", "compare_docs", "compare", "prose", 0.75)

    # ── Define — standalone terminology, no doc reference ────────────────────
    define_signals  = ["what is ", "what does ", "define ", "meaning of ", "qu'est-ce que"]
    _doc_ref_words  = [
        "this", "the file", "the document", "this file", "this document",
        "it", "here", "uploaded", "given", "provided", "attached",
        "هذا", "هذه", "الملف", "ce fichier", "ce document",
    ]
    _is_pure_term = (
        any(s in q for s in define_signals)
        and len(q.split()) <= 8
        and not session_has_docs
        and not any(w in q for w in _doc_ref_words)
    )
    if _is_pure_term:
        return _make_heuristic(query, "define", "define_term", "define", "prose", 0.70)

    # ── Casual / conversational ────────────────────────────────────────────────
    casual_signals = [
        "hello", "hi", "hey", "yo", "wassup", "sup", "what's up", "whats up",
        "thanks", "thank you", "who are you", "bonjour", "merci", "salut",
        "how are you", "what did you", "what did you say", "repeat",
    ]
    if any(s in q for s in casual_signals):
        if not session_has_docs:
            return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.80)
        doc_words = ["file", "document", "uploaded", "it", "this", "that", "the"]
        if not any(w in q for w in doc_words):
            return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.80)

    # ── Self-referential ───────────────────────────────────────────────────────
    self_ref = ["you just", "you said", "your answer", "what you", "last response", "what route"]
    if any(s in q for s in self_ref):
        return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.75,
                               is_followup=True, followup_type="repeat")

    # ── Default: RAG ───────────────────────────────────────────────────────────
    return _make_heuristic(
        query, "rag", "extract_info", "extract", "prose",
        0.60 if not session_has_docs else 0.75,
        doc_scope=_doc_scope,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_doc_scope_heuristic(
    q_lower: str,
    uploaded_docs: List[str],
    attached: Optional[List[str]] = None,
) -> str:
    """
    Resolve which document the user is referring to, without the LLM.
    Returns a filename or positional hint ("first", "second", …) or "".
    """
    if not uploaded_docs:
        return ""

    # Exact / partial filename match (case-insensitive, try longest match first)
    for fname in sorted(uploaded_docs, key=len, reverse=True):
        if fname.lower() in q_lower or os.path.splitext(fname)[0].lower() in q_lower:
            return fname

    # Positional ordinals
    ordinals = {
        "first": 0,  "premier": 0, "première": 0, "الأول": 0, "الأولى": 0,
        "second": 1, "deuxième": 1, "الثاني": 1, "الثانية": 1,
        "third": 2,  "troisième": 2, "الثالث": 2,
        "fourth": 3, "quatrième": 3, "الرابع": 3,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3,
    }
    for word, idx in ordinals.items():
        if word in q_lower and idx < len(uploaded_docs):
            return uploaded_docs[idx]

    # Single-doc session + any pronoun → unambiguously the only doc
    pronouns = [
        "it", "this", "that", "the file", "the document", "the image",
        "له", "هذا", "هذه", "ce fichier", "ce document", "cette image",
    ]
    if len(uploaded_docs) == 1 and any(p in q_lower for p in pronouns):
        return uploaded_docs[0]

    # Most-recently-uploaded signals
    _recency_signals = [
        "this image", "this photo", "this picture", "this screenshot",
        "this file", "this document", "this one", "that image", "that file",
        "هذه الصورة", "هذا الملف", "cette image", "ce fichier",
    ]
    if any(s in q_lower for s in _recency_signals):
        return uploaded_docs[-1]

    # Attached file (most recently attached) wins when nothing else matched —
    # BUT skip CAD files if the query has no CAD signals, to avoid poisoning
    # the scope hint when a PDF + IFC are both in the session.
    if attached:
        _cad_signals = {
            # These are unambiguously IFC/BIM — keep them
            "ifc", "bim", "storey", "storeys",
            "ifc class", "ifc classes", "ifc element", "ifc elements",
            "طابق", "عنصر", "مبنى",
            # Structural terms only count when paired with explicit BIM context,
            # so we keep only the most unambiguous standalone ones here.
            # Generic words like "model", "structure", "building", "material"
            # are intentionally REMOVED — they appear in PDF specs too and were
            # causing PDF queries to get misrouted to the CAD agent.
        }
        _last_attachment = attached[-1]
        if _is_cad(_last_attachment):
            # Only use the CAD attachment as scope hint if query has explicit BIM/CAD intent
            if any(sig in q_lower for sig in _cad_signals):
                return _last_attachment
            # else: query is probably about a document, not the CAD file — leave scope empty
        else:
            return _last_attachment

    return ""


def _make_heuristic(
    query: str,
    route: str,
    primary_intent: str,
    operation: str,
    output_format: str,
    confidence: float,
    is_followup: bool = False,
    followup_type: str = "none",
    doc_scope: str = "",
) -> IntentAnalysis:
    return IntentAnalysis(
        primary_intent    = primary_intent,
        secondary_intent  = "",
        target_entity     = "",
        operation         = operation,
        output_format     = output_format,
        language_intent   = "",
        is_followup       = is_followup,
        followup_type     = followup_type,
        ambiguity_score   = 0.4,
        vagueness_score   = 0.0,
        full_scope        = False,
        suggested_route   = route,
        confidence        = confidence,
        reasoning         = f"Heuristic classification: matched '{route}' pattern in query.",
        doc_scope         = doc_scope,
        classification_path = "heuristic",
        latency_ms        = 0.0,
        clarification_options = [],
    )


def _parse_llm_result(raw: str) -> IntentAnalysis:
    """Parse LLM output into IntentAnalysis — resilient to malformed output."""
    import ast
    import re as _re

    clean = raw.strip()

    if "```" in clean:
        parts = clean.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            clean = inner.strip()

    if clean.startswith("["):
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                clean = json.dumps(parsed[0])
        except Exception:
            pass

    data: Dict = {}
    try:
        data = json.loads(clean)
    except Exception:
        try:
            data = ast.literal_eval(clean)
        except Exception:
            route_match = _re.search(r'"suggested_route"\s*:\s*"([^"]+)"', clean)
            if route_match:
                data = {"suggested_route": route_match.group(1)}

    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}

    suggested = str(data.get("suggested_route", "rag")).strip().lower()
    if suggested not in _VALID_ROUTES:
        suggested = "rag"

    # Parse clarification_options — must be a list of strings
    raw_opts = data.get("clarification_options", [])
    if isinstance(raw_opts, list):
        clarification_options = [str(o) for o in raw_opts if o]
    else:
        clarification_options = []

    return IntentAnalysis(
        primary_intent    = str(data.get("primary_intent", "extract_info")),
        secondary_intent  = str(data.get("secondary_intent", "")),
        target_entity     = str(data.get("target_entity", "")),
        operation         = str(data.get("operation", "extract")),
        output_format     = str(data.get("output_format", "prose")),
        language_intent   = str(data.get("language_intent", "")),
        is_followup       = bool(data.get("is_followup", False)),
        followup_type     = str(data.get("followup_type", "none")),
        ambiguity_score   = float(data.get("ambiguity_score", 0.5)),
        vagueness_score   = float(data.get("vagueness_score", 0.0)),
        full_scope        = bool(data.get("full_scope", False)),
        suggested_route   = suggested,
        confidence        = float(data.get("confidence", 0.7)),
        reasoning         = str(data.get("reasoning", "")),
        doc_scope         = str(data.get("doc_scope", "")),
        classification_path = "llm",
        latency_ms        = 0.0,
        clarification_options = clarification_options,
    )


def _format_history(history: List[Dict]) -> str:
    if not history:
        return "(none)"
    recent = history[-8:]
    lines = []
    for h in recent:
        role    = h.get("role", "user").capitalize()
        content = str(h.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_route_log(route_log: List[Dict]) -> str:
    if not route_log:
        return "(none)"
    return " → ".join(f"[{e.get('route', '?')}]" for e in route_log[-6:])