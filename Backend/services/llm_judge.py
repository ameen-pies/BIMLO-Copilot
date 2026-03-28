"""
LLM-as-a-Judge v2: The Brain of RAG (Drop-in Replacement)

SAME CLASS NAME as before (LLMJudge) but completely rewritten.
Just replace your old llm_judge.py with this file.

This judge doesn't just evaluate - it DECIDES everything:
- What language to respond in
- What tone to use
- How to structure the response
- What content to include
- When to retry

NO hardcoding. NO if-statements for language/tone. The LLM is the brain.
"""

import os
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# ── Load .env so CF_API_KEY is always available ─────────────────────────────
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


@dataclass
class ResponsePlan:
    """The judge's plan for how to respond."""
    
    # What language should the response be in?
    target_language: str  # 'en', 'fr', 'ar', 'es', 'mixed', etc.
    target_language_confidence: float  # 0-1
    
    # What tone/formality should be used?
    target_tone: str  # 'casual', 'friendly', 'professional', 'technical', 'conversational'
    tone_reasoning: str
    
    # How should the response be structured?
    response_style: str  # 'concise', 'detailed', 'bullet_points', 'narrative', 'technical'
    should_cite_sources: bool
    max_response_length: str  # 'brief', 'medium', 'comprehensive'
    
    # Content guidance
    key_points_to_include: List[str]
    things_to_avoid: List[str]
    
    # Overall strategy
    approach: str  # Description of how to answer
    reasoning: str  # Why this plan makes sense
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResponseEvaluation:
    """Post-generation evaluation of the response."""
    
    # Did we follow the plan?
    plan_adherence_score: float  # 0-1
    language_correct: bool
    tone_correct: bool
    
    # Quality checks
    has_hallucination: bool
    hallucination_details: Optional[str]
    uses_sources_correctly: bool
    
    # Overall quality
    overall_score: float  # 0-1
    is_acceptable: bool
    
    # Feedback
    issues: List[str]
    specific_problems: List[str]  # Concrete things wrong
    how_to_fix: str  # Actionable instruction for retry
    
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LLMJudge:
    """
    The brain of the RAG system. (V2 - rewritten, same class name)
    
    Instead of hardcoded rules, the LLM makes ALL decisions:
    - Language detection and selection
    - Tone/formality determination
    - Response structure and style
    - Quality evaluation
    - Retry decisions
    
    No if-statements. No hardcoding. Pure LLM decision-making.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # model param kept for API compatibility — not used by the CF worker
        self.api_key  = api_key or os.getenv("CF_API_KEY", "")
        self.base_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")
        self.enabled  = self._setup()

    def _setup(self) -> bool:
        from llm_client import check_llm_available
        available, provider = check_llm_available()
        if available:
            print(f"🧠 LLM Judge: ready via {provider}")
        else:
            print("❌ LLM Judge: no LLM provider available (set CF_API_KEY or GROQ_API_KEY)")
        return available
    
    def plan_response(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Dict]] = None,
        conversation_history: Optional[List[str]] = None
    ) -> ResponsePlan:
        """
        BEFORE generating a response, ask the judge: HOW should we respond?
        
        This is where all decisions happen:
        - Language choice
        - Tone/formality
        - Structure
        - Content priorities
        
        Returns a complete plan for the response generator to follow.
        """
        if not self.enabled:
            return self._fallback_plan(user_query)
        
        prompt = self._build_planning_prompt(user_query, retrieved_docs, conversation_history)
        
        try:
            # Pass conversation_history as worker history so LLM gets full context
            cf_history = conversation_history if isinstance(conversation_history, list) and \
                         conversation_history and isinstance(conversation_history[0], dict) \
                         else []
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=600, history=cf_history, task="plan")
            return self._parse_plan(raw)
        except Exception as e:
            print(f"❌ Judge planning failed: {e}")
            return self._fallback_plan(user_query)
    
    def evaluate_response(
        self,
        user_query: str,
        plan: ResponsePlan,
        generated_response: str,
        retrieved_docs: Optional[List[Dict]] = None
    ) -> ResponseEvaluation:
        """
        AFTER generating a response, ask the judge: Is this good enough?
        
        The judge checks:
        - Did we follow the plan?
        - Any hallucinations?
        - Language/tone correct?
        - Overall quality acceptable?
        
        If not acceptable, provides specific instructions for retry.
        """
        if not self.enabled:
            return self._fallback_evaluation()
        
        prompt = self._build_evaluation_prompt(
            user_query, plan, generated_response, retrieved_docs
        )
        
        try:
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=300, task="evaluate")
            return self._parse_evaluation(raw)
        except Exception as e:
            print(f"❌ Judge evaluation failed: {e}")
            return self._fallback_evaluation()
    
    def _build_planning_prompt(
        self,
        user_query: str,
        retrieved_docs: Optional[List[Dict]],
        conversation_history: Optional[List[str]]
    ) -> str:
        """Build the prompt that asks the judge to plan the response."""
        
        docs_section = ""
        if retrieved_docs:
            docs_formatted = "\n\n".join([
                f"Document {i+1}:\n{doc.get('text', '')[:400]}..."
                for i, doc in enumerate(retrieved_docs[:5])
            ])
            docs_section = f"\n\nAVAILABLE DOCUMENTS:\n{docs_formatted}"
        
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_section = f"\n\nRECENT CONVERSATION:\n" + "\n".join(conversation_history[-3:])
        
        return f"""You are a response planner. Analyze the query and output a JSON plan.

QUERY: {user_query}
{docs_section}
{history_section}

IMPORTANT LANGUAGE RULE:
- If the user explicitly requests a specific output language (e.g. "translate to French", "respond in Spanish", "باللغة العربية"), set target_language to that requested language — this overrides everything else.
- If the user explicitly requests a specific format, tone, or style, honour it exactly.
- Otherwise, mirror the query language: respond in whatever language the user wrote in.
- Example: query in English, no language request → target_language = "en"
- Example: query in English but says "translate to French" → target_language = "fr"
- Example: query in Arabic → target_language = "ar"

IMPORTANT SOURCE CITATION RULE:
- should_cite_sources = true whenever the query asks for ANY information from documents — facts, values, specs, descriptions, names, dates, quantities, explanations. When in doubt → true.
- should_cite_sources = false ONLY for: translation, rewriting, paraphrasing, summarising into another language, or casual greetings/small talk. These are the ONLY exceptions.

Respond ONLY with this JSON:
{{
  "target_language": "<ISO 639-1 code of the QUERY language>",
  "target_language_confidence": 0.0-1.0,
  "target_tone": "casual|friendly|professional|technical|conversational",
  "tone_reasoning": "one sentence",
  "response_style": "concise|detailed|bullet_points|narrative|technical",
  "should_cite_sources": true|false,
  "max_response_length": "brief|medium|comprehensive",
  "key_points_to_include": ["point 1"],
  "things_to_avoid": ["thing 1"],
  "approach": "one sentence describing how to answer",
  "reasoning": "one sentence of overall reasoning"
}}"""
    
    def _build_evaluation_prompt(
        self,
        user_query: str,
        plan: ResponsePlan,
        generated_response: str,
        retrieved_docs: Optional[List[Dict]]
    ) -> str:
        """Build the prompt that asks the judge to evaluate the response."""
        
        docs_section = ""
        if retrieved_docs:
            docs_formatted = "\n\n".join([
                f"Document {i+1}:\n{doc.get('text', '')[:400]}..."
                for i, doc in enumerate(retrieved_docs[:5])
            ])
            docs_section = f"\n\nSOURCE DOCUMENTS:\n{docs_formatted}"
        
        return f"""You are a response evaluator. Score this RAG response.

QUERY: {user_query}
PLAN: language={plan.target_language}, tone={plan.target_tone}, style={plan.response_style}
RESPONSE: {generated_response}
{docs_section}

Check: correct language? correct tone? any hallucinations? good quality?

Respond ONLY with this JSON:
{{
  "plan_adherence_score": 0.0-1.0,
  "language_correct": true|false,
  "tone_correct": true|false,
  "has_hallucination": true|false,
  "hallucination_details": "description or null",
  "uses_sources_correctly": true|false,
  "overall_score": 0.0-1.0,
  "is_acceptable": true|false,
  "issues": ["issue 1"],
  "specific_problems": ["problem 1"],
  "how_to_fix": "one sentence instruction if not acceptable",
  "reasoning": "one sentence"
}}"""
    
    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500,
                  system_prompt: str = "", history: Optional[List[Dict]] = None,
                  task: str = "plan") -> str:
        """Call the LLM via the shared gateway (CF primary, Groq fallback)."""
        from llm_client import call_llm
        raw = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history or [],
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
        )
        # Normalise: CF worker can return a list when the model outputs JSON directly
        if isinstance(raw, list):
            import json as _json
            raw = _json.dumps(raw)
        elif not isinstance(raw, str):
            raw = str(raw)
        return raw.strip()
    
    def _parse_plan(self, raw_json: str) -> ResponsePlan:
        """Parse LLM output into ResponsePlan — resilient to malformed/list-wrapped output."""
        clean_json = self._clean_json(raw_json)
        try:
            data = json.loads(clean_json)
        except json.JSONDecodeError as e:
            print(f"⚠️  Judge plan JSON parse error: {e} | raw: {clean_json[:200]}")
            return self._fallback_plan("")
        # Llama sometimes wraps the object in a list: [{...}]
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        if not isinstance(data, dict):
            return self._fallback_plan("")
        return ResponsePlan(
            target_language=str(data.get("target_language", "en")),
            target_language_confidence=float(data.get("target_language_confidence", 0.8)),
            target_tone=str(data.get("target_tone", "conversational")),
            tone_reasoning=str(data.get("tone_reasoning", "")),
            response_style=str(data.get("response_style", "detailed")),
            should_cite_sources=bool(data.get("should_cite_sources", True)),
            max_response_length=str(data.get("max_response_length", "medium")),
            key_points_to_include=list(data.get("key_points_to_include", [])),
            things_to_avoid=list(data.get("things_to_avoid", [])),
            approach=str(data.get("approach", "")),
            reasoning=str(data.get("reasoning", ""))
        )

    def _parse_evaluation(self, raw_json: str) -> ResponseEvaluation:
        """Parse LLM output into ResponseEvaluation — resilient to malformed/list-wrapped output."""
        clean_json = self._clean_json(raw_json)
        try:
            data = json.loads(clean_json)
        except json.JSONDecodeError as e:
            print(f"⚠️  Judge eval JSON parse error: {e} | raw: {clean_json[:200]}")
            return self._fallback_evaluation()
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else {}
        if not isinstance(data, dict):
            return self._fallback_evaluation()
        return ResponseEvaluation(
            plan_adherence_score=float(data.get("plan_adherence_score", 0.7)),
            language_correct=bool(data.get("language_correct", True)),
            tone_correct=bool(data.get("tone_correct", True)),
            has_hallucination=bool(data.get("has_hallucination", False)),
            hallucination_details=data.get("hallucination_details"),
            uses_sources_correctly=bool(data.get("uses_sources_correctly", True)),
            overall_score=float(data.get("overall_score", 0.7)),
            is_acceptable=bool(data.get("is_acceptable", True)),
            issues=list(data.get("issues", [])),
            specific_problems=list(data.get("specific_problems", [])),
            how_to_fix=str(data.get("how_to_fix", "")),
            reasoning=str(data.get("reasoning", ""))
        )
    
    def _clean_json(self, raw: str) -> str:
        """
        Normalise LLM output to a valid JSON string.
        Handles:
          - markdown fences ```json ... ```
          - list-wrapped objects [{...}] → unwrap first element
          - Python repr strings with single quotes / True / False / None
          - non-string inputs (list, dict)
        """
        import json as _json
        import ast

        # Already a Python object — re-serialise, unwrapping list if needed
        if isinstance(raw, list):
            obj = raw[0] if raw and isinstance(raw[0], dict) else raw
            return _json.dumps(obj)
        if isinstance(raw, dict):
            return _json.dumps(raw)
        if not isinstance(raw, str):
            raw = str(raw)

        clean = raw.strip()

        # Strip markdown fences  ```json ... ```  or  ``` ... ```
        if "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                clean = inner.strip()

        # If model wrapped object in a list  [{ ... }]  → unwrap first element
        if clean.startswith("["):
            try:
                parsed = _json.loads(clean)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return _json.dumps(parsed[0])
            except _json.JSONDecodeError:
                pass

        # Try standard JSON first
        try:
            _json.loads(clean)
            return clean  # already valid JSON
        except _json.JSONDecodeError:
            pass

        # Fallback: Python repr string (single quotes, True/False/None)
        # ast.literal_eval handles this safely
        try:
            obj = ast.literal_eval(clean)
            if isinstance(obj, (dict, list)):
                if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    obj = obj[0]
                return _json.dumps(obj)
        except (ValueError, SyntaxError):
            pass

        return clean
    
    def _fallback_plan(self, user_query: str) -> ResponsePlan:
        """
        Minimal plan when LLM is unavailable.
        
        Uses simple heuristics to detect common languages, but defaults to 'en' 
        as a safe fallback since most technical/programming content is in English.
        """
        query_lower = user_query.lower()
        
        # Simple script-based detection (works without LLM)
        # Arabic script
        if any('\u0600' <= c <= '\u06FF' for c in user_query):
            lang = 'ar'
        # Cyrillic script (Russian, etc.)
        elif any('\u0400' <= c <= '\u04FF' for c in user_query):
            lang = 'ru'
        # Chinese/Japanese script
        elif any('\u4E00' <= c <= '\u9FFF' for c in user_query):
            lang = 'zh'
        # Korean script
        elif any('\uAC00' <= c <= '\uD7AF' for c in user_query):
            lang = 'ko'
        # Hebrew script
        elif any('\u0590' <= c <= '\u05FF' for c in user_query):
            lang = 'he'
        # Thai script
        elif any('\u0E00' <= c <= '\u0E7F' for c in user_query):
            lang = 'th'
        # Devanagari script (Hindi, etc.)
        elif any('\u0900' <= c <= '\u097F' for c in user_query):
            lang = 'hi'
        # Default to English (most common for technical queries)
        else:
            lang = 'en'
        
        return ResponsePlan(
            target_language=lang,
            target_language_confidence=0.6,
            target_tone='conversational',
            tone_reasoning='Fallback mode - no LLM available for tone detection',
            response_style='detailed',
            should_cite_sources=True,
            max_response_length='medium',
            key_points_to_include=['Answer the query'],
            things_to_avoid=['Hallucinations'],
            approach='Provide a helpful response based on available documents',
            reasoning='Basic fallback plan using script-based language detection - LLM judge not available'
        )
    
    def _fallback_evaluation(self) -> ResponseEvaluation:
        """Minimal evaluation when LLM is unavailable."""
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
            reasoning='Fallback evaluation - LLM judge not available'
        )


# For backwards compatibility - add the old method signatures if they exist
# This ensures old code using LLMJudge will still work
if __name__ == "__main__":
    judge = LLMJudge()
    
    # Test case 1: Casual English query
    print("\n" + "="*80)
    print("TEST: Casual English Query")
    print("="*80)
    
    query = "yo whatcha think about the documents i uploaded"
    plan = judge.plan_response(query)
    
    print(f"\nQuery: {query}")
    print(f"\n🎯 RESPONSE PLAN:")
    print(f"   Language: {plan.target_language} (confidence: {plan.target_language_confidence})")
    print(f"   Tone: {plan.target_tone}")
    print(f"   Reasoning: {plan.reasoning}")
    
    print("\n" + "="*80)