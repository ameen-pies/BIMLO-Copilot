"""
RAG Client — LLM client class with automatic provider fallback.

CloudflareClient uses the shared llm_client.call_llm() gateway
so all agents get Groq fallback for free.
"""

from __future__ import annotations

from typing import Dict, List


class CloudflareClient:
    """
    LLM client — CF Workers AI primary, Groq automatic fallback.

    Uses the shared llm_client.call_llm() gateway so ALL agents get
    Groq fallback for free without any per-service changes.

    _setup() no longer marks the client as disabled when CF is down:
    as long as at least one provider (CF or Groq) is reachable the
    client is enabled and calls will succeed.
    """

    def __init__(self):
        # Import here to avoid circular imports at module load time
        from llm_client import check_llm_available
        self.enabled, provider = check_llm_available()
        if self.enabled:
            print(f"✅ CloudflareClient: LLM ready via {provider}")
        else:
            print("❌ CloudflareClient: no LLM provider available (set CF_API_KEY or GROQ_API_KEY)")

    # ------------------------------------------------------------------
    # Core chat method — same signature as before
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 1200,
        max_retries: int = 3,   # kept for API compatibility, handled inside call_llm
        task: str = "synthesise",
    ) -> str:
        """
        Send messages[] to the LLM (CF primary, Groq fallback).
        Decomposes the messages[] array into prompt/systemPrompt/history
        for the CF worker; Groq receives the full messages[] directly.
        """
        if not self.enabled:
            return ""

        from llm_client import call_llm

        system_prompt = ""
        history: List[Dict] = []

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role in ("user", "assistant"):
                history.append({"role": role, "content": content})

        if not history:
            return ""

        # Last user message → prompt; everything before → history
        prompt  = history[-1]["content"]
        history = history[:-1]

        return call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            task=task,
            preferred_provider=getattr(self, '_preferred_provider', None),
        )


# Aliases — everything that used GroqClient/OllamaClient/GeminiClient still works
GroqClient   = CloudflareClient
OllamaClient = CloudflareClient
GeminiClient = CloudflareClient
