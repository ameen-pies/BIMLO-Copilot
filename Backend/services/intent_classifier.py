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

CURRENT QUERY: {query}

TASK — output ONLY this JSON object, no preamble, no backticks:
{{
  "primary_intent": "<one of: extract_info | compare_docs | generate_report | visualise_data | define_term | modify_output | converse | translate_content | aggregate_stats>",
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
  "reasoning": "<2–4 sentence chain-of-thought: what signals did you see, why did you pick this route>"
}}

ROUTING RULES (apply these strictly when choosing suggested_route):
- direct: casual/conversational, memory recall ("what did you just say"), self-referential ("what route did you use"), edits to last reply ("make it shorter", "rephrase that")
- rag: information extraction or question answering FROM documents (most common)
- iterative_rag: comparison or cross-document analysis (signals: "compare", "vs", "difference between", "across")
- transform: full document translation or complete rewrite/reformat (user wants the whole doc in another form)
- analytics: numerical aggregation across ALL docs (signals: "total", "average", "how many across", "statistics")
- graph: explicit request for a visual chart/plot (signals: "chart", "graph", "plot", "visualise", "graphique", "رسم بياني", "diagramme")
- report: explicit request to PRODUCE a standalone written report/PDF/document (signals: "make a report", "generate a report", "rapport sur", "create a report")
- define: asking the MEANING of a specific technical term or acronym (signals: "what is X", "define X", "what does X mean", "explain X")

CRITICAL OVERRIDE RULES:
1. Any query about what the AI just said/did → direct (is_followup=true, followup_type=modify or repeat)
2. "translate" alone on a short phrase → transform; "translate [specific term]" with explanation → define
3. When ambiguity_score > 0.6, set suggested_route to the safest option (rag for document queries, direct for conversation)
4. Mixed-language queries are fine — detect the INTENT not the language
"""


# ── Main classifier ────────────────────────────────────────────────────────────

def classify_intent(
    query: str,
    history: Optional[List[Dict]] = None,
    route_log: Optional[List[Dict]] = None,
) -> IntentAnalysis:
    """
    Classify the intent of a query using the LLM.
    Falls back to heuristic classification if the LLM is unavailable.

    Args:
        query:     The current user message.
        history:   Conversation history [{role, content}, ...].
        route_log: Per-session route log from AgentState._route_log.

    Returns:
        IntentAnalysis with suggested_route and rich metadata.
    """
    try:
        from llm_client import call_llm, check_llm_available
        available, _ = check_llm_available()
        if not available:
            return _heuristic_classify(query, history, route_log)

        history_block = _format_history(history or [])
        route_history = _format_route_log(route_log or [])

        prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(
            history_block=history_block,
            route_history=route_history,
            query=query,
        )

        raw = call_llm(
            prompt=prompt,
            system_prompt=_CLASSIFIER_SYSTEM,
            max_tokens=500,
            temperature=0.0,
            task="classify",
        )

        return _parse_result(raw)

    except Exception as e:
        logger.warning(f"intent_classifier LLM call failed: {e}")
        return _heuristic_classify(query, history, route_log)


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
    )


def _heuristic_classify(
    query: str,
    history: Optional[List[Dict]],
    route_log: Optional[List[Dict]],
) -> IntentAnalysis:
    """
    Fast heuristic fallback — no LLM required.
    Mirrors the logic in _fallback_router but returns an IntentAnalysis.
    """
    q = query.lower().strip()

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
        return _make_heuristic(query, "graph", "visualise_data", "visualise", "chart", 0.8)

    # Report generation
    report_signals = [
        "make a report", "create a report", "generate a report", "write a report",
        "do a report", "make me a report", "rapport sur", "fais un rapport",
        "produce a report", "build a report", "prepare a report", "un rapport",
    ]
    if any(s in q for s in report_signals):
        return _make_heuristic(query, "report", "generate_report", "generate", "pdf", 0.85)

    # Transform / translate
    transform_signals = [
        "translat", "tradui", "traduire", "rewrite", "paraphrase",
        "summarise in", "summarize in", "résume en", "résumer en",
    ]
    if any(s in q for s in transform_signals):
        return _make_heuristic(query, "transform", "translate_content", "translate", "prose", 0.75)

    # Analytics
    analytics_signals = ["analytics", "statistiques", "total", "average", "count", "how many across"]
    if any(s in q for s in analytics_signals):
        return _make_heuristic(query, "analytics", "aggregate_stats", "aggregate", "table", 0.75)

    # Iterative RAG
    compare_signals = ["compare", "difference", "vs", "versus", "comparaison", "différence", "مقارنة"]
    if any(s in q for s in compare_signals):
        return _make_heuristic(query, "iterative_rag", "compare_docs", "compare", "prose", 0.75)

    # Define
    define_signals = ["what is ", "what does ", "define ", "explain ", "meaning of ", "qu'est-ce que"]
    if any(s in q for s in define_signals) and len(q.split()) <= 10:
        return _make_heuristic(query, "define", "define_term", "define", "prose", 0.7)

    # Direct / conversational
    casual_signals = [
        "hello", "hi", "hey", "yo", "wassup", "sup", "what's up", "whats up",
        "thanks", "thank you", "who are you", "bonjour", "merci", "salut",
        "how are you", "what did you", "what did you say", "repeat",
    ]
    if any(s in q for s in casual_signals):
        return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.8)

    # Self-referential (references the AI or conversation)
    self_ref = ["you just", "you said", "your answer", "what you", "last response", "what route"]
    if any(s in q for s in self_ref):
        return _make_heuristic(query, "direct", "converse", "converse", "prose", 0.75, is_followup=True, followup_type="repeat")

    # Default → RAG
    return _make_heuristic(query, "rag", "extract_info", "extract", "prose", 0.6)


def _make_heuristic(
    query: str,
    route: str,
    primary_intent: str,
    operation: str,
    output_format: str,
    confidence: float,
    is_followup: bool = False,
    followup_type: str = "none",
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
    )