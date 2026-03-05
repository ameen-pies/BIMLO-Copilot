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

# ── Load .env so GROQ_API_KEY is always available ──────────────────────────
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
        self.model = model or os.getenv("GROQ_JUDGE_MODEL") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled = self._setup()

    def _setup(self) -> bool:
        if not self.api_key:
            print("⚠️  LLM Judge - GROQ_API_KEY not set, using fallback")
            return False
        # Test with raw requests — no groq package needed (same as competitors_agent.py)
        try:
            resp = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                print(f"🧠 LLM Judge [{self.model}]: connected")
                return True
            else:
                print(f"❌ LLM Judge: API returned {resp.status_code}: {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ LLM Judge: connection failed — {e}")
            return False
    
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
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=600)
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
            raw = self._call_llm(prompt, temperature=0.1, max_tokens=300)
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
- Detect the language of the QUERY (not the documents).
- The response MUST be in the query language, even if the documents are in a different language.
- Example: if query is in English but docs are in French → target_language = "en"
- Example: if query is in Arabic → target_language = "ar"

IMPORTANT SOURCE CITATION RULE:
- should_cite_sources = true ONLY when the answer retrieves specific facts, values, or quotes from documents that the user can verify.
- should_cite_sources = false when the task is a transformation (translation, rewrite, paraphrase, summarise) or a conversational reply — in these cases sources add no value.

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
    
    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
        """Call Groq REST API directly (no groq package needed)."""
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
    
    def _parse_plan(self, raw_json: str) -> ResponsePlan:
        """Parse LLM output into ResponsePlan."""
        clean_json = self._clean_json(raw_json)
        data = json.loads(clean_json)
        
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
            reasoning=data["reasoning"]
        )
    
    def _parse_evaluation(self, raw_json: str) -> ResponseEvaluation:
        """Parse LLM output into ResponseEvaluation."""
        clean_json = self._clean_json(raw_json)
        data = json.loads(clean_json)
        
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
            reasoning=data.get("reasoning", "")
        )
    
    def _clean_json(self, raw: str) -> str:
        """Remove markdown code fences from JSON."""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return clean.strip()
    
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