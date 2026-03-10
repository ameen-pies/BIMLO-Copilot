"""
LLM-as-a-Judge v3: Route-Aware Brain

Key addition over v2:
- ResponsePlan now includes `recommended_route` and `doc_relevance_score`
- The judge sees both the query AND the retrieved docs in plan_response()
  and decides whether the docs are actually useful enough to cite.
- If doc_relevance_score < DOC_RELEVANCE_THRESHOLD, the engine can
  downgrade "rag" → "direct" so it doesn't hallucinate citations from
  irrelevant chunks.

Everything else (language, tone, structure, evaluation) unchanged from v2.
"""

import os
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    for _parent in [_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))]:
        _env = os.path.join(_parent, ".env")
        if os.path.exists(_env):
            load_dotenv(_env, override=False)
            break
except ImportError:
    pass

# If retrieved docs score below this, the judge recommends answering directly
# (no citations, no RAG hallucination).  Tune 0–1; 0.35 is a conservative default.
DOC_RELEVANCE_THRESHOLD = float(os.getenv("DOC_RELEVANCE_THRESHOLD", "0.35"))


@dataclass
class ResponsePlan:
    """The judge's plan for how to respond."""

    # ── Language ──────────────────────────────────────────────────────────
    target_language: str            # ISO 639-1: 'en', 'fr', 'ar', ...
    target_language_confidence: float

    # ── Tone / style ──────────────────────────────────────────────────────
    target_tone: str                # casual | friendly | professional | technical | conversational
    tone_reasoning: str
    response_style: str             # concise | detailed | bullet_points | narrative | technical
    should_cite_sources: bool
    max_response_length: str        # brief | medium | comprehensive

    # ── Content guidance ──────────────────────────────────────────────────
    key_points_to_include: List[str]
    things_to_avoid: List[str]
    approach: str
    reasoning: str

    # ── NEW: Route recommendation ─────────────────────────────────────────
    recommended_route: str          # direct | rag | iterative_rag | analytics | transform
    doc_relevance_score: float      # 0–1 — how relevant are the retrieved docs to the query
    route_reasoning: str            # one-line explanation

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResponseEvaluation:
    """Post-generation evaluation of the response."""

    plan_adherence_score: float
    language_correct: bool
    tone_correct: bool
    has_hallucination: bool
    hallucination_details: Optional[str]
    uses_sources_correctly: bool
    overall_score: float
    is_acceptable: bool
    issues: List[str]
    specific_problems: List[str]
    how_to_fix: str
    reasoning: str

    def to_dict(self) -> Dict:
        return asdict(self)


class LLMJudge:
    """
    The brain of the RAG system. (V3 - route-aware)

    New vs v2:
    - plan_response() now always receives retrieved_docs so it can assess relevance.
    - ResponsePlan gains recommended_route + doc_relevance_score.
    - The RAG engine uses doc_relevance_score to decide whether to actually cite
      sources or fall back to a direct/conversational answer.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.model   = model   or os.getenv("GROQ_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled  = self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup(self) -> bool:
        if not self.api_key:
            print("⚠️  LLM Judge - GROQ_API_KEY not set, using fallback")
            return False
        try:
            resp = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"🧠 LLM Judge v3 [{self.model}]: connected (DOC_RELEVANCE_THRESHOLD={DOC_RELEVANCE_THRESHOLD})")
                return True
            else:
                print(f"❌ LLM Judge: API returned {resp.status_code}: {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ LLM Judge: connection failed — {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────────

    def plan_response(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[str]] = None,
        current_route: Optional[str] = None,      # NEW: hint from the router
    ) -> ResponsePlan:
        """
        Plan how to respond — now includes route recommendation.

        The judge sees the retrieved docs and decides:
        1. Are they actually relevant? (doc_relevance_score)
        2. Should we use them? (recommended_route)
        3. How should we present the answer? (language/tone/style)
        """
        if not self.enabled:
            return self._fallback_plan(user_query, retrieved_docs, current_route)

        prompt = self._build_planning_prompt(user_query, retrieved_docs, conversation_history, current_route)

        try:
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=700)
            return self._parse_plan(raw)
        except Exception as e:
            print(f"❌ Judge planning failed: {e}")
            return self._fallback_plan(user_query, retrieved_docs, current_route)

    def evaluate_response(
        self,
        user_query: str,
        plan: ResponsePlan,
        generated_response: str,
        retrieved_docs: Optional[List[Dict]] = None,
    ) -> ResponseEvaluation:
        """Evaluate the generated response against the plan."""
        if not self.enabled:
            return self._fallback_evaluation()

        prompt = self._build_evaluation_prompt(user_query, plan, generated_response, retrieved_docs)

        try:
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=300)
            return self._parse_evaluation(raw)
        except Exception as e:
            print(f"❌ Judge evaluation failed: {e}")
            return self._fallback_evaluation()

    # ── Prompt builders ───────────────────────────────────────────────────

    def _build_planning_prompt(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Dict]],
        conversation_history: Optional[List[str]],
        current_route: Optional[str],
    ) -> str:
        docs_section = ""
        if retrieved_docs:
            docs_formatted = "\n\n".join([
                f"Doc {i+1} (distance={doc.get('distance', '?'):.3f} if isinstance(doc.get('distance'), float) else doc.get('distance', '?')):\n{doc.get('text', '')[:350]}..."
                for i, doc in enumerate(retrieved_docs[:5])
            ])
            docs_section = f"\n\nRETRIEVED DOCUMENTS (lower distance = more relevant):\n{docs_formatted}"
        else:
            docs_section = "\n\nRETRIEVED DOCUMENTS: none"

        history_section = ""
        if conversation_history:
            history_section = "\n\nRECENT CONVERSATION:\n" + "\n".join(conversation_history[-3:])

        route_hint = f"\n\nROUTER CHOSE: {current_route} (you may override this in recommended_route)" if current_route else ""

        return f"""You are a response planner for a RAG system. Analyze the query AND the retrieved documents, then output a JSON plan.

QUERY: {user_query}{route_hint}{docs_section}{history_section}

=== ROUTE DECISION ===
Assess whether the retrieved documents are actually relevant to the query.
doc_relevance_score: 0.0 = completely off-topic, 1.0 = perfectly on-topic.
- Score > 0.5 → documents are useful → recommended_route = rag (or keep router's choice)
- Score 0.35–0.5 → marginally relevant → recommended_route = rag but set should_cite_sources = false
- Score < 0.35 → documents are irrelevant/empty → recommended_route = direct (answer from LLM knowledge, no citations)

ROUTE OPTIONS:
- direct: no docs needed — greetings, general knowledge, or docs are off-topic
- rag: answer from documents with citations
- iterative_rag: multi-document comparison
- analytics: aggregate stats across all docs
- transform: translate/rewrite the document itself

=== LANGUAGE RULE ===
- Mirror the query language unless user explicitly requests another language.
- should_cite_sources = false ONLY for: translation, rewriting, paraphrasing, summarising into another language, casual greetings, OR when doc_relevance_score < 0.35.
- OVERRIDE: if the response will include any specific numbers, measurements with units, dates, percentages, currency values, or version numbers from documents — should_cite_sources MUST be true, no exceptions.

Respond ONLY with this JSON (no markdown, no preamble):
{{
  "target_language": "<ISO 639-1>",
  "target_language_confidence": 0.0,
  "target_tone": "casual|friendly|professional|technical|conversational",
  "tone_reasoning": "one sentence",
  "response_style": "concise|detailed|bullet_points|narrative|technical",
  "should_cite_sources": true,
  "max_response_length": "brief|medium|comprehensive",
  "key_points_to_include": ["point 1"],
  "things_to_avoid": ["thing 1"],
  "approach": "one sentence",
  "reasoning": "one sentence",
  "recommended_route": "direct|rag|iterative_rag|analytics|transform",
  "doc_relevance_score": 0.0,
  "route_reasoning": "one sentence explaining why docs are or aren't useful here"
}}"""

    def _build_evaluation_prompt(
        self,
        user_query: str,
        plan: ResponsePlan,
        generated_response: str,
        retrieved_docs: Optional[List[Dict]],
    ) -> str:
        docs_section = ""
        if retrieved_docs:
            docs_formatted = "\n\n".join([
                f"Document {i+1}:\n{doc.get('text', '')[:400]}..."
                for i, doc in enumerate(retrieved_docs[:5])
            ])
            docs_section = f"\n\nSOURCE DOCUMENTS:\n{docs_formatted}"

        return f"""You are a response evaluator. Score this RAG response.

QUERY: {user_query}
PLAN: language={plan.target_language}, tone={plan.target_tone}, style={plan.response_style}, route={plan.recommended_route}
RESPONSE: {generated_response}
{docs_section}

Check ALL of the following:
1. Correct language?
2. Correct tone?
3. Any hallucinations (claims not supported by source documents)?
4. Does response respect the recommended_route?
5. CRITICAL — if the response contains ANY specific numbers, measurements with units (e.g. 22 kPa, 400 MHz, 3.5 km), dates, percentages, currency amounts, or version numbers, those MUST have a [N] citation marker. Flag has_hallucination=true and is_acceptable=false if specific data appears without a citation.

Respond ONLY with this JSON:
{{
  "plan_adherence_score": 0.0,
  "language_correct": true,
  "tone_correct": true,
  "has_hallucination": false,
  "hallucination_details": null,
  "uses_sources_correctly": true,
  "overall_score": 0.0,
  "is_acceptable": true,
  "issues": [],
  "specific_problems": [],
  "how_to_fix": "",
  "reasoning": "one sentence"
}}"""

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        resp = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise Exception(f"Groq API {resp.status_code}: {resp.text[:200]}")

    # ── Parsers ───────────────────────────────────────────────────────────

    def _parse_plan(self, raw_json: str) -> ResponsePlan:
        clean = self._clean_json(raw_json)
        data  = json.loads(clean)

        return ResponsePlan(
            target_language=data["target_language"],
            target_language_confidence=float(data["target_language_confidence"]),
            target_tone=data["target_tone"],
            tone_reasoning=data["tone_reasoning"],
            response_style=data["response_style"],
            should_cite_sources=data["should_cite_sources"],
            max_response_length=data["max_response_length"],
            key_points_to_include=data.get("key_points_to_include", []),
            things_to_avoid=data.get("things_to_avoid", []),
            approach=data["approach"],
            reasoning=data["reasoning"],
            # New fields — graceful defaults if older judge response missing them
            recommended_route=data.get("recommended_route", "rag"),
            doc_relevance_score=float(data.get("doc_relevance_score", 0.5)),
            route_reasoning=data.get("route_reasoning", ""),
        )

    def _parse_evaluation(self, raw_json: str) -> ResponseEvaluation:
        clean = self._clean_json(raw_json)
        data  = json.loads(clean)

        return ResponseEvaluation(
            plan_adherence_score=float(data["plan_adherence_score"]),
            language_correct=data["language_correct"],
            tone_correct=data["tone_correct"],
            has_hallucination=data["has_hallucination"],
            hallucination_details=data.get("hallucination_details"),
            uses_sources_correctly=data["uses_sources_correctly"],
            overall_score=float(data["overall_score"]),
            is_acceptable=data["is_acceptable"],
            issues=data.get("issues", []),
            specific_problems=data.get("specific_problems", []),
            how_to_fix=data.get("how_to_fix", ""),
            reasoning=data.get("reasoning", ""),
        )

    def _clean_json(self, raw: str) -> str:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return clean.strip()

    # ── Fallbacks ─────────────────────────────────────────────────────────

    def _fallback_plan(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Dict]] = None,
        current_route: Optional[str] = None,
    ) -> ResponsePlan:
        """
        Heuristic plan when LLM is unavailable.
        Estimates doc relevance from cosine distances returned by the vector store.
        """
        # Script-based language detection
        if any('\u0600' <= c <= '\u06FF' for c in user_query):
            lang = 'ar'
        elif any('\u0400' <= c <= '\u04FF' for c in user_query):
            lang = 'ru'
        elif any('\u4E00' <= c <= '\u9FFF' for c in user_query):
            lang = 'zh'
        elif any('\uAC00' <= c <= '\uD7AF' for c in user_query):
            lang = 'ko'
        elif any('\u0590' <= c <= '\u05FF' for c in user_query):
            lang = 'he'
        else:
            lang = 'en'

        # Estimate relevance from distances
        relevance = 0.5
        if retrieved_docs:
            distances = [d.get("distance", 1.0) for d in retrieved_docs if d.get("distance") is not None]
            if distances:
                avg_dist = sum(distances) / len(distances)
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to relevance score 0–1
                relevance = round(max(0.0, 1.0 - avg_dist), 2)

        if relevance < DOC_RELEVANCE_THRESHOLD:
            recommended_route = "direct"
            should_cite = False
            route_reasoning = f"Fallback: avg distance implies low relevance (score={relevance})"
        else:
            recommended_route = current_route or "rag"
            should_cite = True
            route_reasoning = f"Fallback: docs appear relevant (score={relevance})"

        return ResponsePlan(
            target_language=lang,
            target_language_confidence=0.6,
            target_tone='conversational',
            tone_reasoning='Fallback mode',
            response_style='detailed',
            should_cite_sources=should_cite,
            max_response_length='medium',
            key_points_to_include=['Answer the query'],
            things_to_avoid=['Hallucinations'],
            approach='Provide a helpful response based on available documents' if should_cite else 'Answer from general knowledge',
            reasoning='Fallback plan — LLM judge not available',
            recommended_route=recommended_route,
            doc_relevance_score=relevance,
            route_reasoning=route_reasoning,
        )

    def _fallback_evaluation(self) -> ResponseEvaluation:
        return ResponseEvaluation(
            plan_adherence_score=0.7,
            language_correct=True,
            tone_correct=True,
            has_hallucination=False,
            hallucination_details=None,
            uses_sources_correctly=True,
            overall_score=0.7,
            is_acceptable=True,
            issues=[],
            specific_problems=[],
            how_to_fix='',
            reasoning='Fallback evaluation — LLM judge not available',
        )


if __name__ == "__main__":
    judge = LLMJudge()

    print("\n" + "="*80)
    print("TEST: Low-relevance query (should recommend direct route)")
    print("="*80)

    query = "yo whatcha think about the documents i uploaded"
    # Simulate low-relevance retrieved docs (high distances)
    fake_docs = [
        {"text": "Technical specification for valve pressure ratings...", "distance": 0.91},
        {"text": "Installation manual chapter 3 section 4...", "distance": 0.88},
    ]
    plan = judge.plan_response(query, retrieved_docs=fake_docs, current_route="rag")

    print(f"\nQuery: {query}")
    print(f"\n🎯 RESPONSE PLAN:")
    print(f"   Language:          {plan.target_language}")
    print(f"   Tone:              {plan.target_tone}")
    print(f"   Recommended route: {plan.recommended_route}")
    print(f"   Doc relevance:     {plan.doc_relevance_score}")
    print(f"   Route reasoning:   {plan.route_reasoning}")
    print(f"   Should cite:       {plan.should_cite_sources}")
    print(f"   Overall:           {plan.reasoning}")
    print("="*80)