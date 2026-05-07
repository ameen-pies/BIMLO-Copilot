"""
intent_classifier.py — Deep Intent Classifier for Bimlo Copilot
────────────────────────────────────────────────────────────────
A dedicated, chain-of-thought intent analysis layer that sits
UPSTREAM of the LLM router.

Unlike the router (which outputs a single route word), this classifier
outputs a rich IntentAnalysis object with:
  - primary_intent   : what the user actually wants
  - secondary_intent : any secondary goal (e.g. translate + summarise)
  - target_entity    : the doc/concept/term they care about
  - operation        : the verb of the action (extract, compare, define…)
  - output_format    : what they want back (prose, table, chart, PDF…)
  - language_intent  : language switch requested explicitly?
  - is_followup      : is this continuing from the last turn?
  - followup_type    : modify | repeat | translate | clarify | none
  - ambiguity_score  : 0–1 how ambiguous the query is
  - suggested_route  : best matching RAG route (for router to use as hint)
  - confidence       : 0–1 classifier confidence
  - reasoning        : chain-of-thought trace (debug only)

The router in rag_engine.py calls `classify_intent()` and uses
`suggested_route` + `IntentAnalysis` to make a more informed routing
decision, reducing misroutes significantly.

Usage:
    from intent_classifier import classify_intent, IntentAnalysis
    intent = classify_intent(query, history, route_log)
    # intent.suggested_route → "rag" | "graph" | "report" | etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger("intent_classifier")


# ── Intent analysis dataclass ──────────────────────────────────────────────────

@dataclass
class IntentAnalysis:
    primary_intent:    str        # e.g. "extract_info", "compare_docs", "generate_report"
    secondary_intent:  str        # e.g. "translate", "summarise", "" if none
    target_entity:     str        # doc name, term, or concept the user cares about
    operation:         str        # action verb: extract | compare | define | visualise | generate | modify | converse
    output_format:     str        # prose | table | chart | pdf | json | code | audio
    language_intent:   str        # "" = mirror query lang; else explicit target lang code e.g. "fr"
    is_followup:       bool
    followup_type:     str        # modify | repeat | translate | clarify | none
    ambiguity_score:   float      # 0–1
    suggested_route:   str        # direct | rag | iterative_rag | transform | analytics | graph | report | define
    confidence:        float      # 0–1
    reasoning:         str        # chain-of-thought (debug)
    # ── Document scope resolution ──────────────────────────────────────────────
    # When the user targets a specific document (by name, position, or pronoun),
    # this field carries the resolved reference so retrieve_vector can filter.
    # Examples: "contract.pdf", "the first one", "the report", "it", ""
    # Empty string means: no specific document targeted → search all docs.
    doc_scope:         str        # "" | exact filename | positional hint e.g. "first"

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Prompt ─────────────────────────────────────────────────────────────────────

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
  "primary_intent": "<one of: extract_info | compare_docs | generate_report | visualise_data | define_term | modify_output | converse | converse_meta | translate_content | aggregate_stats>",
  "secondary_intent": "<same enum or empty string>",
  "target_entity": "<the document name, term, concept, or data point the user cares about — empty if none>",
  "operation": "<action verb: extract | compare | define | visualise | generate | modify | converse | translate | aggregate>",
  "output_format": "<prose | table | chart | pdf | code | audio | unspecified>",
  "language_intent": "<ISO 639-1 code if user explicitly requests output in a specific language, else empty string>",
  "is_followup": <true|false>,
  "followup_type": "<modify | repeat | translate | clarify | none>",
  "ambiguity_score": <0.0–1.0>,
  "suggested_route": "<one of: direct | rag | iterative_rag | transform | analytics | graph | report | define>",
  "confidence": <0.0–1.0>,
  "doc_scope": "<exact filename from the uploaded docs list if the user clearly targets ONE specific document; positional hint like 'first' or 'second' if they reference position; empty string if they want ALL docs or the scope is unclear>",
  "reasoning": "<2–4 sentence chain-of-thought: what signals did you see, why did you pick this route, and which document (if any) are they targeting>"
}}

ROUTING RULES (apply these strictly when choosing suggested_route):
- direct: casual/conversational, memory recall ("what did you just say"), self-referential ("what route did you use"), edits to last reply ("make it shorter", "rephrase that")
- direct [converse_meta]: queries that ask about the AI's own awareness or perception of an uploaded file — NOT about its content.
    Signals (any language): "do you see this", "can you read this", "did you get the file", "is the file there",
    "do you have access", "هل ترى هذا", "tu vois ce fichier", "tu peux lire ça", "avez-vous reçu",
    "peux-tu voir", "est-ce que tu as reçu", "vois-tu le document".
    These are meta-questions about the AI's state — NEVER route them to rag.
    Set primary_intent="converse_meta", suggested_route="direct".
- rag: ANY question about document content — a specific fact, a section, a value, a person, a location, a date, what something says, what something means in context of the document, how something works according to the document. This is the DEFAULT route for all document questions.
- iterative_rag: questions that explicitly require looking across MULTIPLE different documents simultaneously — comparisons, differences, aggregations across files. Signals: "compare", "vs", "difference between", "across all files", "between the two documents". Do NOT use for questions that can be answered from one document.
- transform: full document translation or complete rewrite/reformat (user wants the whole doc in another form)
- analytics: numerical aggregation across ALL docs (signals: "total", "average", "how many across", "statistics")
- graph: explicit request for a visual chart/plot (signals: "chart", "graph", "plot", "visualise", "graphique", "رسم بياني", "diagramme")
- report: explicit request to PRODUCE a standalone written report/PDF/document (signals: "make a report", "generate a report", "rapport sur", "create a report")
- define: asking the MEANING of a standalone technical term or acronym WITH NO reference to any uploaded document — ONLY when the user is clearly asking about a generic concept in isolation (e.g. "what is BIM?" with no docs, "define IFC"). NEVER use define if: (1) the session has uploaded documents AND the query references file content, (2) the query contains words like "this file", "this document", "it", "here", "the file", "in this", "of this". "What is the host company of this file?" → rag. "What is BIM?" with no docs → define.

DOC SCOPE RULES (for the "doc_scope" field):
- If the user names a specific file (partial or full name), set doc_scope to that filename exactly as it appears in UPLOADED DOCUMENTS.
- If the user uses a positional reference ("the first one", "the first file", "le premier", "الأول"), set doc_scope to "first", "second", etc.
- If the user uses a pronoun ("it", "this one", "that file") AND there is only one uploaded document, set doc_scope to that document's filename.
- If there are multiple documents and the pronoun is ambiguous, set doc_scope to "" (search all).
- If the user says "both", "all", "all of them", or implies cross-document scope, set doc_scope to "".

CRITICAL OVERRIDE RULES:
1. Any query about what the AI just said/did → direct (is_followup=true, followup_type=modify or repeat)
2. Any query asking IF the AI can see/read/access a file → direct with primary_intent=converse_meta. NEVER rag.
3. "translate" alone on a short phrase → transform; "translate [specific term]" with explanation → define
4. When ambiguity_score > 0.6, set suggested_route to the safest option (rag for document queries, direct for conversation)
5. Mixed-language queries are fine — detect the INTENT not the language
6. When multiple docs are uploaded and the user clearly targets only ONE (by name or position) → set doc_scope accordingly so retrieval is scoped correctly.
7. Questions like "what is X" where X is something IN a document (company name, person, zone, standard) → rag, not define.
"""


# ── Main classifier ────────────────────────────────────────────────────────────

def classify_intent(
    query: str,
    history: Optional[List[Dict]] = None,
    route_log: Optional[List[Dict]] = None,
    preferred_provider: Optional[str] = None,
    session_has_docs: bool = False,
    uploaded_docs: Optional[List[str]] = None,
) -> IntentAnalysis:
    """
    Classify the intent of a query using the LLM.
    Falls back to heuristic classification if the LLM is unavailable.

    Args:
        query:              The current user message.
        history:            Conversation history [{role, content}, ...].
        route_log:          Per-session route log from AgentState._route_log.
        preferred_provider: Optional provider hint to honor user-selected model.
        session_has_docs:   True if the session has documents in the vector store.
                            When True, ambiguous/short messages default to rag.
        uploaded_docs:      List of filenames currently uploaded in the session.
                            Used for doc_scope resolution (which doc is being targeted).

    Returns:
        IntentAnalysis with suggested_route, doc_scope, and rich metadata.
    """
    try:
        from llm_client import call_llm, check_llm_available
        available, _ = check_llm_available()
        if not available:
            return _heuristic_classify(query, history, route_log, session_has_docs, uploaded_docs)

        history_block = _format_history(history or [])
        route_history = _format_route_log(route_log or [])

        # Format the uploaded doc list so the LLM can reason about doc_scope
        if uploaded_docs:
            uploaded_docs_block = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(uploaded_docs))
        else:
            uploaded_docs_block = "(none uploaded yet)"

        # Build a session-state hint so the LLM knows whether docs exist
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

        prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
            history_block=history_block,
            route_history=route_history,
            uploaded_docs_block=uploaded_docs_block,
            query=query,
        ) + docs_hint

        raw = call_llm(
            prompt=prompt,
            system_prompt=_CLASSIFIER_SYSTEM,
            max_tokens=500,
            temperature=0.0,
            task="classify",
            preferred_provider=preferred_provider,
        )

        result = _parse_result(raw)

        # Post-parse safety net: if session has docs and classifier said "direct"
        # but ambiguity is non-trivial, override to rag — UNLESS it's a meta-awareness
        # query (converse_meta), which must always stay direct.
        if (session_has_docs
                and result.suggested_route == "direct"
                and result.ambiguity_score >= 0.2
                and result.primary_intent != "converse_meta"):
            result.suggested_route = "rag"
            result.reasoning += " [overridden direct→rag: session has docs and query is ambiguous]"

        return result

    except Exception as e:
        logger.warning(f"intent_classifier LLM call failed: {e}")
        return _heuristic_classify(query, history, route_log, session_has_docs, uploaded_docs)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_history(history: List[Dict]) -> str:
    if not history:
        return "(none)"
    recent = history[-8:]  # last 4 turns (user+assistant = 2 msgs each)
    lines = []
    for h in recent:
        role = h.get("role", "user").capitalize()
        content = str(h.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_route_log(route_log: List[Dict]) -> str:
    if not route_log:
        return "(none)"
    return " → ".join(f"[{e.get('route', '?')}]" for e in route_log[-6:])


def _parse_result(raw: str) -> IntentAnalysis:
    """Parse LLM output into IntentAnalysis — resilient to malformed output."""
    import ast, re as _re

    clean = raw.strip()

    # Strip markdown fences
    if "```" in clean:
        parts = clean.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            clean = inner.strip()

    # Unwrap list-wrapped objects
    if clean.startswith("["):
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                clean = json.dumps(parsed[0])
        except Exception:
            pass

    data = {}
    try:
        data = json.loads(clean)
    except Exception:
        try:
            data = ast.literal_eval(clean)
        except Exception:
            # Extract what we can with regex
            route_match = _re.search(r'"suggested_route"\s*:\s*"([^"]+)"', clean)
            if route_match:
                data = {"suggested_route": route_match.group(1)}

    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}

    valid_routes = {"direct", "rag", "iterative_rag", "transform", "analytics", "graph", "report", "define"}
    suggested = str(data.get("suggested_route", "rag")).strip().lower()
    if suggested not in valid_routes:
        suggested = "rag"

    return IntentAnalysis(
        primary_intent   = str(data.get("primary_intent", "extract_info")),
        secondary_intent = str(data.get("secondary_intent", "")),
        target_entity    = str(data.get("target_entity", "")),
        operation        = str(data.get("operation", "extract")),
        output_format    = str(data.get("output_format", "prose")),
        language_intent  = str(data.get("language_intent", "")),
        is_followup      = bool(data.get("is_followup", False)),
        followup_type    = str(data.get("followup_type", "none")),
        ambiguity_score  = float(data.get("ambiguity_score", 0.5)),
        suggested_route  = suggested,
        confidence       = float(data.get("confidence", 0.7)),
        reasoning        = str(data.get("reasoning", "")),
        doc_scope        = str(data.get("doc_scope", "")),
    )


def _heuristic_classify(
    query: str,
    history: Optional[List[Dict]],
    route_log: Optional[List[Dict]],
    session_has_docs: bool = False,
    uploaded_docs: Optional[List[str]] = None,
) -> IntentAnalysis:
    """
    Fast heuristic fallback — no LLM required.
    Mirrors the logic in _fallback_router but returns an IntentAnalysis.
    When session_has_docs=True, short/ambiguous messages default to rag.
    """
    q = query.lower().strip()

    # ── Meta-awareness: "do you see this file?", "can you read it?", etc. ──────
    # These ask about the AI's perception — never route to rag regardless of docs.
    _meta_signals = [
        "do you see", "can you see", "can you read", "do you have access",
        "did you get", "is the file there", "do you have the file",
        "tu vois", "tu peux voir", "tu peux lire", "tu as reçu", "avez-vous reçu",
        "peux-tu voir", "peux-tu lire", "est-ce que tu vois", "est-ce que tu as",
        "هل ترى", "هل يمكنك رؤية", "هل تستطيع قراءة", "هل وصلك",
    ]
    if any(s in q for s in _meta_signals):
        return _make_heuristic(query, "direct", "converse_meta", "converse", "prose", 0.90,
                               doc_scope="")

    # ── Doc scope: heuristic resolution when one doc clearly targeted ──────────
    _doc_scope = _resolve_doc_scope_heuristic(q, uploaded_docs or [])

    # Code generation — always direct, before anything else
    _code_verbs = ["write", "make", "give me", "create", "generate", "build",
                   "code", "implement", "show me", "do", "écris", "fais", "crée"]
    _code_nouns = ["code", "function", "script", "algorithm", "algo", "program",
                   "snippet", "class", "method", "implementation", "solution",
                   "fonction", "algorithme", "programme", "classe", "méthode",
                   "كود", "دالة", "خوارزمية", "برنامج"]
    if (any(q.startswith(v) or f" {v} " in q for v in _code_verbs) and
            any(n in q for n in _code_nouns)):
        return _make_heuristic(query, "direct", "converse", "generate", "code", 0.92)

    # Chart / graph
    graph_signals = [
        "chart", "graph", "plot", "visuali", "graphique", "diagramme",
        "courbe", "histogramme", "رسم بياني", "مخطط", "تصور",
        "gráfico", "diagrama", "visualizar", "diagramm", "grafik",
    ]
    if any(s in q for s in graph_signals):
        return _make_heuristic(query, "graph", "visualise_data", "visualise", "chart", 0.8,
                               doc_scope=_doc_scope)

    # Report generation
    report_signals = [
        "make a report", "create a report", "generate a report", "write a report",
        "do a report", "make me a report", "rapport sur", "fais un rapport",
        "produce a report", "build a report", "prepare a report", "un rapport",
    ]
    if any(s in q for s in report_signals):
        return _make_heuristic(query, "report", "generate_report", "generate", "pdf", 0.85,
                               doc_scope=_doc_scope)

    # Transform / translate
    transform_signals = [
        "translat", "tradui", "traduire", "rewrite", "paraphrase",
        "summarise in", "summarize in", "résume en", "résumer en",
    ]
    if any(s in q for s in transform_signals):
        return _make_heuristic(query, "transform", "translate_content", "translate", "prose", 0.75,
                               doc_scope=_doc_scope)

    # Analytics
    analytics_signals = ["analytics", "statistiques", "total", "average", "count", "how many across"]
    if any(s in q for s in analytics_signals):
        return _make_heuristic(query, "analytics", "aggregate_stats", "aggregate", "table", 0.75)

    # Iterative RAG
    compare_signals = ["compare", "difference", "vs", "versus", "comparaison", "différence", "مقارنة"]
    if any(s in q for s in compare_signals):
        return _make_heuristic(query, "iterative_rag", "compare_docs", "compare", "prose", 0.75)

    # Define — ONLY for standalone terminology questions with no doc context.
    # Never trigger define when session has docs — "what is the host company of this file"
    # is a RAG question, not a Wikipedia lookup.
    define_signals = ["what is ", "what does ", "define ", "meaning of ", "qu'est-ce que"]
    _doc_ref_words = ["this", "the file", "the document", "this file", "this document",
                      "it", "here", "uploaded", "given", "provided", "attached",
                      "هذا", "هذه", "الملف", "ce fichier", "ce document"]
    _is_pure_term = (
        any(s in q for s in define_signals)
        and len(q.split()) <= 8
        and not session_has_docs                          # no docs → Wikipedia is fine
        and not any(w in q for w in _doc_ref_words)      # no doc reference words
    )
    if _is_pure_term:
        return _make_heuristic(query, "define", "define_term", "define", "prose", 0.7)

    # Direct / conversational — only clear greetings/thanks, never doc-context
    casual_signals = [
        "hello", "hi", "hey", "yo", "wassup", "sup", "what's up", "whats up",
        "thanks", "thank you", "who are you", "bonjour", "merci", "salut",
        "how are you", "what did you", "what did you say", "repeat",
    ]
    if any(s in q for s in casual_signals):
        # Even casual signals shouldn't override when docs exist and query looks doc-related
        if not session_has_docs:
            return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.8)
        # With docs present: only truly casual (no doc words) → direct
        doc_words = ["file", "document", "uploaded", "it", "this", "that", "the"]
        if not any(w in q for w in doc_words):
            return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.8)

    # Self-referential (references the AI or conversation)
    self_ref = ["you just", "you said", "your answer", "what you", "last response", "what route"]
    if any(s in q for s in self_ref):
        return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.75,
                               is_followup=True, followup_type="repeat")

    # Default → RAG (always rag when session has docs; otherwise rag anyway as safest)
    _reasoning = (
        "Heuristic classification: session has documents — defaulting to rag for ambiguous query."
        if session_has_docs else
        "Heuristic classification: matched 'rag' pattern in query."
    )
    return _make_heuristic(query, "rag", "extract_info", "extract", "prose",
                           0.6 if not session_has_docs else 0.75, doc_scope=_doc_scope)


def _resolve_doc_scope_heuristic(q: str, uploaded_docs: List[str]) -> str:
    """
    Heuristic doc-scope resolver: returns a filename or positional hint if the
    query clearly targets one specific document, else returns "".

    No LLM needed — exact filename substring match + ordinal word detection.
    The LLM path in the full classifier handles ambiguous/natural-language cases.
    """
    if not uploaded_docs:
        return ""

    # Exact / partial filename match (case-insensitive)
    for fname in uploaded_docs:
        if fname.lower() in q:
            return fname

    # Positional ordinals — map word → 0-based index
    ordinals = {
        "first": 0,  "premier": 0, "première": 0, "الأول": 0,  "الأولى": 0,
        "second": 1, "deuxième": 1, "الثاني": 1, "الثانية": 1,
        "third": 2,  "troisième": 2, "الثالث": 2,
        "fourth": 3, "quatrième": 3, "الرابع": 3,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3,
    }
    for word, idx in ordinals.items():
        if word in q and idx < len(uploaded_docs):
            return uploaded_docs[idx]

    # Single-doc session + pronoun → unambiguously the only doc
    if len(uploaded_docs) == 1:
        pronouns = ["it", "this", "that", "the file", "the document", "له", "هذا", "هذه", "ce fichier", "ce document"]
        if any(p in q for p in pronouns):
            return uploaded_docs[0]

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
        primary_intent   = primary_intent,
        secondary_intent = "",
        target_entity    = "",
        operation        = operation,
        output_format    = output_format,
        language_intent  = "",
        is_followup      = is_followup,
        followup_type    = followup_type,
        ambiguity_score  = 0.4,
        suggested_route  = route,
        confidence       = confidence,
        reasoning        = f"Heuristic classification: matched '{route}' pattern in query.",
        doc_scope        = doc_scope,
    )