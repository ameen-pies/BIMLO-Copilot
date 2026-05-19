"""
RAG Nodes — Specialized LangGraph node handlers as a mixin class.

RAGEngine inherits RAGNodesMixin to get all specialized node methods.
These methods depend on self.vs, self.llm, self._chat, self._emit, self.judge —
all provided by the RAGEngine base.
"""

from __future__ import annotations

import os
import re
import json
import threading
from typing import Any, Dict, List
from datetime import datetime

from rag_state import AgentState, MAX_RETRIES
from rag_helpers import (
    _detect_script_language,
    _build_context,
    _build_sources_from_brackets,
    _clean_answer,
    _confidence,
    _build_chart_example_hint,
    _format_sources,
)

try:
    from llm_judge import ResponsePlan
except ImportError:
    from typing import Any as ResponsePlan

try:
    from relevance_extractor import extract_relevant_chunks
    _RELEVANCE_EXTRACTOR_AVAILABLE = True
except ImportError:
    _RELEVANCE_EXTRACTOR_AVAILABLE = False

try:
    from graph_agent import GraphAgent, is_graph_request
    _GRAPH_AGENT_AVAILABLE = True
except ImportError:
    _GRAPH_AGENT_AVAILABLE = False

try:
    import importlib as _importlib
    if "services.report_agent" in __import__('sys').modules:
        _report_agent_module = __import__('sys').modules["services.report_agent"]
    else:
        try:
            _report_agent_module = _importlib.import_module("services.report_agent")
        except ModuleNotFoundError:
            _report_agent_module = _importlib.import_module("report_agent")
    _REPORT_AGENT_AVAILABLE = True
except Exception:
    _report_agent_module = None
    _REPORT_AGENT_AVAILABLE = False


class RAGNodesMixin:
    """Mixin providing all specialized node methods for RAGEngine."""

    _MAP_REDUCE_BATCH_SIZE = 8

    def transform_node(self, state: AgentState) -> AgentState:
        """
        Execute a transformation task (translate, rewrite, reformat) on the
        FULL document content — not top_k RAG chunks.

        If user specifies a page number, only that page is used.
        If user names a document, all chunks for that document are loaded.
        """
        import re as _re

        query   = state["query"]
        chunks  = state["retrieved_chunks"]

        print(f"🔀 Transform → ", end="")

        if not chunks:
            print("no documents")
            return {**state, "answer": "No document content found to transform.", "sources": []}

        # ── Load FULL document instead of top_k RAG chunks ──────────────
        # Transform tasks need the actual document content, not semantically
        # similar snippets. Load all chunks for the target document.
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        doc_scope  = state.get("doc_scope_hint", "")

        # Determine target filename: prefer doc_scope, fallback to first chunk's filename
        target_filename = doc_scope or chunks[0].get("metadata", {}).get("filename", "")

        if target_filename and self.vs:
            full_chunks = self.vs.get_all_chunks(target_filename, user_id=user_id, session_id=session_id)
            if full_chunks:
                chunks = full_chunks
                print(f"[full doc: {target_filename} → {len(chunks)} chunks] ", end="")

        # ── Page filtering ──────────────────────────────────────────────
        # If user asks for a specific page (e.g. "page 5", "p. 3", "pages 2-4"),
        # filter chunks to only include content from that page.
        page_match = _re.search(
            r'(?:page|p\.?|pages?|page\s*)\s*(\d+)(?:\s*[-–]\s*(\d+))?',
            query, _re.IGNORECASE
        )
        if page_match:
            page_start = int(page_match.group(1))
            page_end = int(page_match.group(2)) if page_match.group(2) else page_start
            page_chunks = []
            for c in chunks:
                text = c.get("text", "")
                # Look for page references in chunk text: [TABLE on page X], [IMAGE on page X], etc.
                page_refs = _re.findall(r'(?:page|p\.?)\s*(\d+)', text, _re.IGNORECASE)
                if page_refs:
                    if any(page_start <= int(p) <= page_end for p in page_refs):
                        page_chunks.append(c)
                # Skip chunks with no page markers — they're not from the target page
            if page_chunks:
                chunks = page_chunks
                print(f"[filtered to page {page_start}" + (f"-{page_end}" if page_end != page_start else "") + f" → {len(chunks)} chunks] ", end="")
            else:
                print(f"[page {page_start} filter: no chunks matched, using chunk_index fallback] ", end="")
                # Fallback: use chunk position to estimate page (rough heuristic)
                total = len(chunks)
                per_page = max(1, total // 10)  # assume ~10 pages
                idx_start = (page_start - 1) * per_page
                idx_end = page_end * per_page
                chunks = chunks[idx_start:idx_end]
                if not chunks:
                    chunks = chunks[:per_page]  # at least first page worth

        if not chunks:
            print("no matching content")
            return {**state, "answer": "No document content found for the specified page.", "sources": []}

        # Ask the judge to plan (language detection etc.) — same as normal flow
        plan = self.judge.plan_response(query, retrieved_docs=chunks)
        print(f"{plan.target_language}/{plan.target_tone}")

        # Build a clean context for transform — no [Source N] headers bleeding into output
        clean_context = "\n\n".join(c["text"] for c in chunks)

        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        # Override: detect explicit language request in query ("in english", "en anglais", etc.)
        _LANG_OVERRIDES = {
            r'\b(?:in|to|into)\s+english\b': 'en',
            r'\b(?:in|to|into)\s+french\b': 'fr',
            r'\b(?:in|to|into)\s+arabic\b': 'ar',
            r'\b(?:in|to|into)\s+spanish\b': 'es',
            r'\ben\s+anglais\b': 'en',
            r'\ben\s+fran[çc]ais\b': 'fr',
            r'\ben\s+arabe\b': 'ar',
            r'\bفي\s+الإنجليزية\b': 'en',
            r'\bبالإنجليزية\b': 'en',
            r'\bà\s+l\'anglais\b': 'en',
        }
        for pattern, lang_code in _LANG_OVERRIDES.items():
            if _re.search(pattern, query, _re.IGNORECASE):
                target_lang = lang_code
                print(f"[lang override: {lang_code}] ", end="")
                break

        # Scale max_tokens with document size — full docs need more room
        doc_len = len(clean_context)
        max_tokens = min(max(2000, doc_len // 2), 8000)

        # ── Map-reduce for large documents ──────────────────────────────
        # If the document is too large for a single LLM call, batch it.
        # Transform is different from summarize: we concatenate results, not merge.
        MAP_REDUCE_THRESHOLD = 20000  # chars (~5000 tokens)
        BATCH_SIZE = 8  # chunks per batch

        def _build_transform_prompt(context: str) -> str:
            return f"""You are a document assistant. Complete the following task exactly as instructed.

🚨 LANGUAGE: You MUST write your output in {target_lang.upper()}. The document below may be in a different language — IGNORE its language. Your output must be in {target_lang.upper()}.

TASK: {query}

STRUCTURE RULES:
- The document content below may have lost some formatting due to text extraction. Reconstruct a clean, well-structured document as output.
- Identify sections, subsections, bullet points, and numbered lists from context clues in the text (e.g. "1.", "2.", "- ", all-caps titles, etc.) and render them properly using markdown: ## for section headings, - for bullets, numbered lists where appropriate.
- ALL-CAPS phrases that look like section titles (e.g. "EXECUTIVE SUMMARY", "EQUIPMENT SPECIFICATIONS") should become ## headings.
- Apply the task to the text content only — the structure and layout must be clean and readable in the output.
- Preserve ALL content from the document. Do not summarize or skip sections.
- Output ONLY the result. No preamble, no explanation, no meta-commentary, no citation markers.

DOCUMENT:
{context}

REMEMBER: Your entire output MUST be in {target_lang.upper()}. Do not keep any content in the original document language."""

        if doc_len > MAP_REDUCE_THRESHOLD and len(chunks) > BATCH_SIZE:
            # Map-reduce: transform each batch, concatenate results
            n_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[map-reduce: {len(chunks)} chunks → {n_batches} batches] ", end="")

            batch_results: List[str] = []
            for bi in range(n_batches):
                batch = chunks[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
                batch_context = "\n\n".join(c["text"] for c in batch)
                self._emit("transform_node", "📄", f"Transforming batch {bi + 1}/{n_batches}…")

                if self.llm.enabled:
                    batch_answer = self._chat(
                        [{"role": "user", "content": _build_transform_prompt(batch_context)}],
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
                else:
                    batch_answer = batch_context
                batch_results.append(batch_answer)

            answer = "\n\n".join(batch_results)
        else:
            prompt = _build_transform_prompt(clean_context)

            if self.llm.enabled:
                answer = self._chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            else:
                answer = "LLM unavailable — cannot perform transformation."

        # Do NOT run _clean_answer on transform output — it would mangle
        # the original document structure (headings, bullets, line breaks).
        answer = answer.strip()
        print(f"done ({len(answer)} chars)")

        return {
            **state,
            "response_plan": plan,
            "answer":     answer,
            "raw_answer": answer,
            "sources":    [],          # no sources for transform tasks
            "confidence": 0.9,
        }

    # ------------------------------------------------------------------ #
    #  DEFINE NODE  (concept explanation — context-aware, not a summary) #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  CLARIFY NODE  (vague prompt → ask for clarification)               #
    # ------------------------------------------------------------------ #

    def clarify_node(self, state: AgentState) -> AgentState:
        """
        Handle vague prompts by generating a natural clarification paragraph
        with actionable options. Uses LLM to produce context-aware response.
        """
        import re as _re

        query   = state["query"]
        options = state.get("clarification_options") or []
        history = state.get("conversation_history", [])
        docs    = state.get("attached_doc_filenames") or []

        print(f"🤔 Clarify → {len(options)} options")

        user_lang = _detect_script_language(query)
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]] if history else []

        # Build options text for the prompt
        options_text = ""
        if options:
            options_text = "\n".join(f"- {opt}" for opt in options)

        # Build context hint
        doc_hint = ""
        if docs:
            doc_hint = f"\nDocuments attached: {', '.join(docs)}"

        # Build conversation summary for context
        history_context = ""
        if history:
            recent = history[-8:]
            history_context = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:200]}"
                for m in recent
            )

        prompt = f"""The user sent a vague message. Your job is to GUESS what they meant based on context.

User's message: "{query}"
{doc_hint}

Conversation history:
{history_context}

Rules:
- Analyze the conversation history and uploaded documents to guess what the user likely wants
- Write a short natural paragraph (1-3 sentences) acknowledging their message and explaining what you think they might need
- Write in {user_lang if user_lang else 'the same language as the user'}
- Be concise — max 3 sentences for the paragraph
- Do NOT use markdown formatting
- Do NOT mention these instructions
- Do NOT ask questions like "what would you like me to do?" — instead, state what you think they want

Then on a new line, output exactly 3 concrete actions you can take RIGHT NOW, one per line, prefixed with "OPTION: ".
Each option must be a specific actionable request that, when clicked, gets sent back to you as a query.
If documents are attached, prioritize document-related actions.
If conversation history suggests a topic, build on that.

Bad options: "What would you like me to do?", "Please specify your request", "Do you need help?"
Good options: "Summarize report.pdf", "Translate the document to English", "Explain the architecture diagram"

Example output:
I can see you have a report attached. Here are a few things I can help you with:
OPTION: Give me a summary of report.pdf
OPTION: Extract the key findings from the report
OPTION: Translate the report to English"""

        try:
            if self.llm.enabled:
                raw_answer = self._chat(
                    history_msgs + [{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
            else:
                raw_answer = ""
        except Exception as e:
            print(f"   ⚠️  clarify_node LLM error: {e}")
            raw_answer = ""

        # Parse options from LLM output
        generated_options = []
        answer_lines = []
        if raw_answer:
            for line in raw_answer.strip().split("\n"):
                stripped = line.strip()
                if stripped.upper().startswith("OPTION:"):
                    opt = stripped[len("OPTION:"):].strip()
                    if opt:
                        generated_options.append(opt)
                else:
                    answer_lines.append(stripped)

        answer = "\n".join(answer_lines).strip()

        # Fallback if LLM fails or returns empty
        if not answer or len(answer) < 10:
            if docs:
                answer = f"I can help you with {', '.join(docs[:2])}. Here are a few options:"
            else:
                answer = "I can help with that — here are a few things I can do:"

        if not generated_options:
            if docs:
                generated_options = [
                    f"Summarize {docs[0]}",
                    f"Translate {docs[0]} to English",
                    f"Extract the key points from {docs[0]}",
                ]
            elif history:
                generated_options = [
                    "Continue from where we left off",
                    "Summarize our conversation so far",
                    "Explain the last topic in more detail",
                ]
            else:
                generated_options = [
                    "Introduce yourself and explain what you can do",
                    "Help me get started with a document",
                    "What features do you offer?",
                ]

        # Strip markdown formatting for TTS compatibility
        answer = _re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = answer.strip()

        print(f"   ✅ Clarify ({len(answer)} chars, {len(generated_options)} options)")

        self._emit("clarify_node", "🤔", "I need more details…")

        return {
            **state,
            "answer":                answer,
            "clarification_options": generated_options,
            "sources":               [],
            "confidence":            0.9,
        }

    # ------------------------------------------------------------------ #
    #  FULL DOCUMENT NODE  (map-reduce over entire document)               #
    # ------------------------------------------------------------------ #

    _MAP_REDUCE_BATCH_SIZE = 8  # chunks per batch

    def full_doc_node(self, state: AgentState) -> AgentState:
        """
        Retrieve ALL chunks for attached documents and synthesize via map-reduce.

        Used when the user asks to summarize/analyze the entire document and
        the intent classifier routes to 'full_doc'.

        Flow:
        1. Get all chunks for each attached doc (bypasses top_k)
        2. If total chunks ≤ batch size → single-shot synthesis (no map-reduce)
        3. Otherwise: synthesize each batch independently, then merge summaries
        """
        query     = state["query"]
        session_id = state.get("session_id")
        user_id   = state.get("user_id")
        history   = state.get("conversation_history", [])

        # Detect attached docs
        attached = state.get("attached_doc_filenames") or []
        all_docs = self.vs.list_documents(user_id=user_id, session_id=session_id)

        if not attached:
            attached = [d.get("filename") for d in all_docs if d.get("filename")]

        if not attached:
            print("⚠️  full_doc_node: no documents found → falling back to direct")
            return self.direct_answer(state)

        # Gather ALL chunks for attached docs
        all_chunks: List[Dict] = []
        for fname in attached:
            try:
                doc_chunks = self.vs.get_all_chunks(fname, user_id=user_id, session_id=session_id)
                all_chunks.extend(doc_chunks)
                print(f"   📄 {fname}: {len(doc_chunks)} chunks")
            except Exception as e:
                print(f"   ⚠️  {fname}: get_all_chunks error: {e}")

        if not all_chunks:
            print("⚠️  full_doc_node: zero chunks retrieved → falling back to direct")
            return self.direct_answer(state)

        # Sort by chunk_index to maintain document order
        all_chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

        total = len(all_chunks)
        batch_size = self._MAP_REDUCE_BATCH_SIZE
        user_language = _detect_script_language(query)

        print(f"📚 full_doc_node: {total} chunks across {len(attached)} doc(s)")

        # ── Single-shot path (small doc) ────────────────────────────────
        if total <= batch_size:
            print(f"   → single-shot synthesis ({total} chunks)")
            context = _build_context(all_chunks)
            plan = ResponsePlan(
                key_points=[],
                sub_queries=[query],
                target_language=user_language or "en",
                response_style="comprehensive",
                should_cite_sources=True,
            )
            prompt = self._build_synthesis_prompt(query, context, plan, fix_instruction="", user_language=user_language)
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]] if history else []

            if self.llm.enabled:
                answer = self._chat(history_msgs + [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=3000)
            else:
                answer = self._fallback_synthesis(all_chunks, plan)

            self._emit("full_doc_node", "📚", f"Analyzed {total} chunks from {len(attached)} doc(s)")

            return {
                **state,
                "response_plan": plan,
                "answer":        answer,
                "raw_answer":    answer,
                "sources":       _build_sources_from_brackets(answer, all_chunks),
                "confidence":    0.9,
            }

        # ── Map-reduce path (large doc) ─────────────────────────────────
        n_batches = (total + batch_size - 1) // batch_size
        print(f"   → map-reduce: {n_batches} batches of ≤{batch_size}")

        batch_summaries: List[str] = []
        for bi in range(n_batches):
            batch = all_chunks[bi * batch_size : (bi + 1) * batch_size]
            context = _build_context(batch)
            self._emit("full_doc_node", "📖", f"Processing batch {bi + 1}/{n_batches}…")

            prompt = f"""Summarize the key information from this section of a document.
Be thorough — extract all important facts, numbers, names, and relationships.
Write in {user_language or 'the same language as the document'}.

USER QUERY: {query}

SECTION:
{context}"""

            try:
                if self.llm.enabled:
                    summary = self._chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=800)
                else:
                    summary = f"[Batch {bi+1}: {len(batch)} chunks]"
                batch_summaries.append(summary)
            except Exception as e:
                print(f"   ⚠️  batch {bi+1} synthesis error: {e}")
                batch_summaries.append(f"[Batch {bi+1}: synthesis failed]")

        # ── Merge phase ─────────────────────────────────────────────────
        self._emit("full_doc_node", "🔀", "Merging summaries…")
        combined = "\n\n---\n\n".join(
            f"SECTION {i+1}:\n{s}" for i, s in enumerate(batch_summaries)
        )

        merge_prompt = f"""You have {n_batches} section summaries from a document. Merge them into a single, comprehensive answer to the user's query.

Rules:
- Be thorough — include all important facts, numbers, and relationships from every section
- Remove redundancy but keep unique details from each section
- Use [N] citation markers to reference sources
- Write in {user_language or 'the same language as the user'}
- Structure with clear headings if the content is long

USER QUERY: {query}

SECTION SUMMARIES:
{combined}"""

        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]] if history else []

        if self.llm.enabled:
            final_answer = self._chat(history_msgs + [{"role": "user", "content": merge_prompt}], temperature=0.2, max_tokens=3000)
        else:
            final_answer = combined

        # Post-processing (same as synthesise_node)
        import re as _re
        final_answer = _re.sub(r'\n[ \t]*\.[ \t]*\n', '\n', final_answer)
        final_answer = _re.sub(r'\n{3,}', '\n\n', final_answer)
        final_answer = _re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', final_answer)

        cited_nums = sorted(set(int(m) for m in _re.findall(r'\[(\d+)\]', final_answer)))
        if not cited_nums and all_chunks:
            final_answer = final_answer.rstrip() + " [1]"
            cited_nums = [1]

        print(f"   ✅ full_doc_node: {len(final_answer)} chars, citations={cited_nums}")

        self._emit("full_doc_node", "📚", f"Analyzed all {total} chunks from {len(attached)} doc(s)")

        return {
            **state,
            "answer":     final_answer,
            "raw_answer": final_answer,
            "sources":    _build_sources_from_brackets(final_answer, all_chunks),
            "confidence": 0.9,
        }

    def relevance_node(self, state: AgentState) -> AgentState:
        """
        Full-document Q&A via relevance extraction.

        Unlike full_doc_node (which summarizes everything), this node:
        1. Loads ALL chunks for attached documents
        2. Uses map-reduce to FILTER only chunks relevant to the query
        3. Synthesizes answer from the filtered set

        This preserves answer precision while eliminating top-k blindness.
        """
        query      = state["query"]
        session_id = state.get("session_id")
        user_id    = state.get("user_id")
        history    = state.get("conversation_history", [])

        # Detect attached docs
        attached = state.get("attached_doc_filenames") or []
        all_docs = self.vs.list_documents(user_id=user_id, session_id=session_id)

        if not attached:
            attached = [d.get("filename") for d in all_docs if d.get("filename")]

        if not attached:
            print("⚠️  relevance_node: no documents found → falling back to direct")
            return self.direct_answer(state)

        # Gather ALL chunks
        all_chunks: List[Dict] = []
        for fname in attached:
            try:
                doc_chunks = self.vs.get_all_chunks(fname, user_id=user_id, session_id=session_id)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                print(f"   ⚠️  {fname}: get_all_chunks error: {e}")

        if not all_chunks:
            print("⚠️  relevance_node: zero chunks → falling back to direct")
            return self.direct_answer(state)

        all_chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))

        # Map-reduce relevance extraction
        self._emit("relevance_node", "🔍", f"Scanning {len(all_chunks)} chunks for relevance…")
        filtered = extract_relevant_chunks(query, all_chunks, self._chat)

        # Synthesize from filtered chunks
        self._emit("relevance_node", "✍️", f"Generating answer from {len(filtered)} relevant chunks…")
        context = _build_context(filtered)
        user_language = _detect_script_language(query)
        plan = ResponsePlan(
            key_points=[],
            sub_queries=[query],
            target_language=user_language or "en",
            response_style="comprehensive",
            should_cite_sources=True,
        )
        prompt = self._build_synthesis_prompt(query, context, plan, fix_instruction="", user_language=user_language)
        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]] if history else []

        if self.llm.enabled:
            answer = self._chat(history_msgs + [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=3000)
        else:
            answer = self._fallback_synthesis(filtered, plan)

        print(f"   ✅ relevance_node: {len(all_chunks)} → {len(filtered)} chunks, {len(answer)} chars")

        return {
            **state,
            "answer":     answer,
            "raw_answer": answer,
            "sources":    _build_sources_from_brackets(answer, filtered),
            "confidence": 0.9,
        }

    def define_node(self, state: AgentState) -> AgentState:
        """
        Explain a term or concept using:
          - Vector-store document chunks (project-specific context, relevance-filtered)
          - Wikipedia page for the term (general encyclopedic depth)

        Key differences from synthesise:
        - Only chunks above the relevance threshold are fed to the LLM
        - Wikipedia is always the primary source card; doc chunks are secondary
        - Output is structured (intro paragraph + optional bullets) — never a wall of text
        - Ghost sources (cited [N] with no matching chunk) are stripped before returning
        - wiki_url is passed on the source card so the frontend can render an external link
        """
        query   = state["query"]
        history = state.get("conversation_history", [])

        print(f"📖 Define (Wikipedia-only) → ", end="")

        # Ask the judge to plan language/tone — no chunks needed for planning
        history_texts = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=[], conversation_history=history_texts)
        print(f"{plan.target_language}/{plan.target_tone}")

        # ── Wikipedia enrichment — primary and only source ────────────────
        wiki_ctx = None
        try:
            from wiki_enricher import get_wiki_context
            wiki_ctx = get_wiki_context(
                query=query,
                api_key=os.getenv("CF_API_KEY", ""),
                base_url=os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev"),
                language=plan.target_language if len(plan.target_language) == 2 else "en",
            )
        except ImportError:
            print("   ⚠️  wiki_enricher not found")
        except Exception as e:
            print(f"   ⚠️  WikiEnricher error: {e}")

        if not wiki_ctx or not wiki_ctx.found:
            print("   ℹ️  No Wikipedia page found — answering from general knowledge")
            wiki_block = ""
            wiki_source_num = 1
            context_section = "No external source found. Answer from your general knowledge."
        else:
            wiki_source_num = 1   # Wikipedia is always source [1] — no doc chunks
            wiki_raw = wiki_ctx.as_context_block(max_chars=3500)
            wiki_block = f"[Source 1 | Wikipedia: {wiki_ctx.term} | encyclopedia]\n{wiki_raw}"
            context_section = wiki_block
            print(f"   ✅ Wikipedia '{wiki_ctx.term}' ready as [Source 1]")

        # Conversation history
        history_msgs: List[Dict] = []
        if history:
            history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        prompt = f"""You are a knowledgeable assistant explaining a concept clearly and thoroughly.

🚨 LANGUAGE REQUIREMENT: You MUST write your ENTIRE answer in {target_lang.upper()}. The source content below may be in a different language — IGNORE its language. Your answer must be in {target_lang.upper()}.

Language: {target_lang.upper()} | Tone: {plan.target_tone}

TASK: {query}

OUTPUT FORMAT — write three sections separated by blank lines, in this exact order:

SECTION A — Definition (2-3 sentences): A clear, direct definition of the concept. [1]

SECTION B — How it works: Explain the mechanism, key properties, or main aspects. Write as a prose paragraph OR as bullet points (- item) when listing 3+ distinct items. Do NOT number items.

SECTION C — Why it matters (1-2 sentences): Brief note on real-world significance or practical use.

Do NOT label the sections with "SECTION A", "SECTION B", etc. — just write the content directly.
Do NOT use any numbered lists (1. 2. 3.) anywhere in your response.

CITATION RULES:
- After every sentence drawn from Source 1 (Wikipedia), add [1].
- Only cite when you actually used the source. Skip [1] on sentences from pure general knowledge.

STYLE RULES:
- Use blank lines between each section.
- **Bold** the first mention of the key term and any critical sub-terms.
- Use bullet points (- item [1]) when listing 3+ distinct items; prose otherwise.
- Never write more than 3 consecutive sentences without a line break.
- Do NOT use ## headings.
- Output ONLY the explanation — no preamble, no "Here is...", no meta-commentary.

{context_section}

Remember: Answer in {target_lang.upper()}. Your ENTIRE answer must be in {target_lang.upper()}."""

        if self.llm.enabled:
            answer = self._chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1200,
            )
        else:
            answer = "LLM unavailable — cannot generate explanation."

        import re as _re
        answer = _re.sub(r'\n[ \t]*\.[ \t]*\n', '\n', answer)
        answer = _re.sub(r'\n[ \t]*\.[ \t]*$', '', answer.rstrip())
        answer = _re.sub(r'\n{3,}', '\n\n', answer)
        answer = _re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', answer)
        answer = _clean_answer(answer)

        print(f"done ({len(answer)} chars)")

        # ── Wikipedia source card — only if cited ─────────────────────────
        sources: List[Dict] = []
        cited_nums = set(int(m) for m in _re.findall(r'\[(\d+)\]', answer))

        if wiki_ctx and wiki_ctx.found and wiki_source_num in cited_nums:
            sources.append({
                "source_number": wiki_source_num,
                "filename":      f"Wikipedia: {wiki_ctx.term}",
                "doc_type":      "encyclopedia",
                "project_ref":   None,
                "excerpt":       wiki_ctx.summary[:400],
                "sections":      [{"title": "Overview", "lines": [wiki_ctx.summary[:300]], "excerpt": wiki_ctx.summary[:400]}],
                "cited_facts":   [wiki_ctx.term],
                "wiki_url":      wiki_ctx.url,
            })
            print(f"   📎 Wikipedia source card added")
        else:
            print(f"   ℹ️  Wikipedia not cited in answer — card suppressed")

        return {
            **state,
            "response_plan": plan,
            "answer":     answer,
            "raw_answer": answer,
            "sources":    sources,
            "confidence": 1.0 if (wiki_ctx and wiki_ctx.found) else 0.5,
        }

    # ------------------------------------------------------------------ #
    #  IMAGE NODE — dedicated route for image content queries            #
    # ------------------------------------------------------------------ #

    def image_node(self, state: AgentState) -> AgentState:
        """
        Handle queries about uploaded images.

        Strategy:
          1. Fetch image chunks from the vector store (doc_type=image).
          2. If chunks contain a real vision description, synthesise the answer.
          3. If chunks only have a placeholder description (vision failed at ingestion),
             OR if no chunks are found yet (ingestion still in flight), attempt a LIVE
             vision call using the raw image bytes from DOCUMENT_FILE_CACHE, update the
             stored chunk text in-place, and then synthesise from the fresh description.
        """
        query      = state["query"]
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        history    = state.get("conversation_history", [])

        self._emit("image_node", "🖼️", "Reading image description…")
        print(f"🖼️  image_node → query={query[:80]!r}")

        # ── Fetch image chunks from the user/session's collection ─────────
        # Respect doc_scope_hint — supports:
        #   ""               → all images in session (doc_type=image filter)
        #   "file.png"       → single scoped image
        #   "a.png|b.png"    → multiple images (pipe-separated, e.g. when user
        #                       asks about both attached images at once)
        _scope_raw = state.get("doc_scope_hint", "")
        _scoped_files = [s.strip() for s in _scope_raw.split("|") if s.strip()] if _scope_raw else []

        all_chunks: list = []

        if _scoped_files:
            # Retrieve chunks for every scoped filename and merge
            print(f"   [image_node: scoped to {_scoped_files}]")
            for _fname in _scoped_files:
                try:
                    _chunks = self.vs.search(
                        query,
                        top_k=10,
                        user_id=user_id,
                        session_id=session_id,
                        filter_dict={"filename": _fname},
                    )
                    all_chunks.extend(_chunks)
                except Exception as _e:
                    print(f"⚠️  image_node: vector search failed for '{_fname}': {_e}")

            # Fallback: unfiltered search, then filter by filename set
            if not all_chunks:
                try:
                    unfiltered = self.vs.search(query, top_k=20, user_id=user_id, session_id=session_id)
                    all_chunks = [c for c in unfiltered if
                                  c.get("metadata", {}).get("filename") in _scoped_files]
                except Exception:
                    pass
        else:
            # No scope — retrieve all image chunks in this session
            try:
                all_chunks = self.vs.search(
                    query,
                    top_k=10,
                    user_id=user_id,
                    session_id=session_id,
                    filter_dict={"doc_type": "image"},
                )
            except Exception as _e:
                print(f"⚠️  image_node: vector search failed: {_e}")
                all_chunks = []

            # Fallback: unfiltered search filtered by doc_type / has_images
            if not all_chunks:
                try:
                    unfiltered = self.vs.search(query, top_k=10, user_id=user_id, session_id=session_id)
                    all_chunks = [c for c in unfiltered if
                                  c.get("metadata", {}).get("doc_type") == "image" or
                                  c.get("metadata", {}).get("has_images")]
                except Exception:
                    pass

        # Deduplicate by chunk id / text hash so merging doesn't double-count
        _seen_texts: set = set()
        _deduped: list = []
        for _c in all_chunks:
            _key = _c.get("id") or _c.get("text", "")[:120]
            if _key not in _seen_texts:
                _seen_texts.add(_key)
                _deduped.append(_c)
        all_chunks = _deduped

        # Group image chunks by filename so a single uploaded image is not
        # treated as multiple images when its description was split into chunks.
        _grouped: dict = {}
        for _c in all_chunks:
            _fname = _c.get("metadata", {}).get("filename", "image")
            _grouped.setdefault(_fname, []).append(_c)

        grouped_chunks: list = []
        for _fname, _items in _grouped.items():
            if len(_items) == 1:
                grouped_chunks.append(_items[0])
                continue

            _items.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", c.get("chunk_id", 0)))
            grouped_text = "\n\n".join(c["text"] for c in _items if c.get("text"))
            merged_meta = _items[0].get("metadata", {}).copy()
            merged_meta["chunk_count"] = len(_items)
            grouped_chunks.append({
                "text": grouped_text,
                "metadata": merged_meta,
                "id": _items[0].get("id"),
                "distance": min((c.get("distance", 1.0) for c in _items), default=1.0),
            })
        all_chunks = grouped_chunks

        # ── Placeholder / missing description detection ───────────────────
        # These strings are what document_processor writes when vision fails.
        _PLACEHOLDER_SIGNALS = (
            "vision description unavailable",
            "could not be read",
            "ask the user to describe",
        )

        def _is_placeholder(chunk: dict) -> bool:
            txt = chunk.get("text", "").lower()
            return any(s in txt for s in _PLACEHOLDER_SIGNALS)

        # Try a live vision call if: no chunks at all, OR all chunks are placeholders
        needs_live_vision = (not all_chunks) or all(_is_placeholder(c) for c in all_chunks)

        if needs_live_vision:
            self._emit("image_node", "🖼️", "Describing image with vision AI…")
            print("🔍 image_node: attempting live vision call (placeholder or missing chunks)")

            # Pull raw bytes from DOCUMENT_FILE_CACHE
            # When multiple images are scoped, attempt a live call for each one.
            _live_chunks: list = []
            try:
                from core.globals import DOCUMENT_FILE_CACHE as _cache

                # Find image entries scoped to this session
                all_image_entries = [
                    (doc_id, entry)
                    for doc_id, entry in _cache.items()
                    if entry.get("is_image") and entry.get("session_id") == session_id
                ]

                if all_image_entries:
                    # Filter to scoped filenames when scope is set
                    if _scoped_files:
                        targeted_entries = [
                            (did, e) for did, e in all_image_entries
                            if e.get("filename") in _scoped_files
                        ]
                        # Preserve order of _scoped_files
                        _fname_to_entry = {e.get("filename"): (did, e) for did, e in targeted_entries}
                        targeted_entries = [
                            _fname_to_entry[fn]
                            for fn in _scoped_files
                            if fn in _fname_to_entry
                        ]
                        if not targeted_entries:
                            targeted_entries = all_image_entries  # fallback: all
                    else:
                        targeted_entries = all_image_entries  # no scope → try all

                    from document_processor import DocumentProcessor as _DP
                    _dp = _DP()

                    for _doc_id, _entry in targeted_entries:
                        _live_fname = _entry["filename"]
                        _raw_bytes  = _entry["bytes"]
                        try:
                            _fresh_desc = _dp.describe_image_bytes(_raw_bytes, _live_fname)
                            if _fresh_desc:
                                print(f"   ✅ Live vision description for '{_live_fname}': {len(_fresh_desc)} chars")
                                _live_chunks.append({
                                    "text": (
                                        f"[UPLOADED IMAGE: {_live_fname}]\n"
                                        f"Visual description:\n{_fresh_desc}"
                                    ),
                                    "metadata": {
                                        "filename":  _live_fname,
                                        "doc_type":  "image",
                                        "has_images": True,
                                        "source":    "live_vision",
                                    },
                                    "distance": 0.05,
                                })
                        except Exception as _ve:
                            print(f"   ⚠️  Live vision call failed for '{_live_fname}': {_ve}")
                else:
                    print("   ⚠️  image_node: no image entries in DOCUMENT_FILE_CACHE for this session")

            except Exception as _ce:
                print(f"   ⚠️  image_node: could not access DOCUMENT_FILE_CACHE: {_ce}")

            if _live_chunks:
                # Use live descriptions; merge with any non-placeholder ingested chunks
                _ingested_ok = [c for c in all_chunks if not _is_placeholder(c)]
                all_chunks = _live_chunks + _ingested_ok
                print(f"   🖼️  image_node: {len(_live_chunks)} live description(s) + {len(_ingested_ok)} indexed chunk(s)")
            elif not all_chunks:
                # Both ingestion and live vision failed — tell the user clearly
                answer = (
                    "I received your image but was unable to generate a visual description. "
                    "This can happen when the vision AI service is temporarily unavailable. "
                    "Please try again in a moment, or describe what you see in the image and I'll help from there."
                )
                print("⚠️  image_node: live vision failed and no chunks — returning helpful error")
                return {**state, "answer": answer, "raw_answer": answer, "sources": [], "confidence": 0.0}
            # else: fall through with the placeholder chunks — synthesis will still run
            # and the LLM prompt below will handle it gracefully

        # ── Build context from image description chunks ───────────────────
        context_parts = []
        for i, chunk in enumerate(all_chunks, 1):
            meta = chunk.get("metadata", {})
            fname = meta.get("filename", "image")
            context_parts.append(f"[Image {i}: {fname}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        # ── Ask the judge for language/tone ───────────────────────────────
        history_texts = [f"{m['role'].upper()}: {m['content'][:200]}" for m in history[-6:]]
        plan = self.judge.plan_response(query, retrieved_docs=all_chunks, conversation_history=history_texts)

        # ── Synthesise answer from the stored visual description ──────────
        user_lang = _detect_script_language(query)
        target_lang = plan.target_language or user_lang

        prompt = f"""You are an expert image analyst. You are looking at an image uploaded by the user.

🚨 LANGUAGE REQUIREMENT: You MUST write your ENTIRE answer in {target_lang.upper()}. The image details below may be in a different language — IGNORE their language. Your answer must be in {target_lang.upper()}.

Language: {target_lang.upper()} | Tone: {plan.target_tone}

Here is what you can see in the image:

{context}

User question: {query}

Answer as if you are directly looking at the image. Say "I can see..." or "The image shows..." — never mention descriptions, metadata, or that you're reading text about the image.
Be specific and reference what is visible. If multiple images are present, address each one.
Remember: Answer in {target_lang.upper()}."""

        history_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        if self.llm.enabled:
            answer = self._chat(
                history_msgs + [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
        else:
            answer = context  # fallback: return raw description

        answer = _clean_answer(answer)
        print(f"   ✅ image_node: answered from {len(all_chunks)} chunk(s) ({len(answer)} chars)")

        return {
            **state,
            "response_plan":  plan,
            "answer":         answer,
            "raw_answer":     answer,
            "retrieved_chunks": all_chunks,
            "sources":        [],
            "confidence":     0.9,
        }

    # ------------------------------------------------------------------ #
    #  CAD / IFC NODE                                                     #
    # ------------------------------------------------------------------ #

    def cad_node(self, state: AgentState) -> AgentState:
        """
        Handle queries about uploaded CAD/IFC files (DXF, DWG, STEP, IFC).

        Delegates to cad_ifc_agent._build_context_block() + _call_llm_with_judge(),
        which already have their own judge-retry loop and status callbacks.
        Falls back to standard RAG synthesis if cad_ifc_agent is unavailable.
        """
        query      = state["query"]
        session_id = state.get("session_id", "")
        user_id    = state.get("user_id")
        history    = state.get("conversation_history", [])
        doc_scope  = state.get("doc_scope_hint", "")

        self._emit("cad_node", "🏗️", "Parsing CAD/IFC file structure…")
        print(f"🏗️  cad_node → query={query[:80]!r}")

        try:
            from cad_ifc_agent import CadSharedContext, _build_context_block, _call_llm_with_judge, _SYSTEM_PROMPT
            from datetime import datetime as _dt

            # Resolve which file to use: doc_scope_hint > most recent in session
            summary = None
            if doc_scope:
                # Try exact file_id match first, then filename match
                for fid, meta in [(f, CadSharedContext.get_file(f))
                                  for f in [doc_scope]
                                  if CadSharedContext.get_file(f)]:
                    summary = meta
                    break
            if not summary:
                # Fall back to most recently cached file in this session
                all_files = CadSharedContext.list_files()
                # Filter to files belonging to this session if session_id is stored
                session_files = [f for f in all_files if f.get("session_id") == session_id] or all_files
                if session_files:
                    summary = session_files[-1]

            if not summary:
                # No CAD file found — fall back to RAG
                print("⚠️  cad_node: no CAD summary found — falling back to RAG synthesis")
                return self.retrieve_vector({**state, "route": "rag"})

            # Build context block from the parsed CAD/IFC summary
            context = _build_context_block(summary)
            system_prompt = _SYSTEM_PROMPT.format(
                today=_dt.utcnow().strftime("%B %d, %Y")
            ) + f"\n\n{context}"

            # Retrieve conversation history for this session
            cad_history = CadSharedContext.get_history(session_id) or []
            # Also incorporate the RAG conversation history so memory works
            for turn in history[-6:]:
                if not any(t.get("content") == turn.get("content") for t in cad_history):
                    cad_history.append(turn)

            fname    = summary.get("filename", "file")
            pipeline = summary.get("pipeline", "?").upper()
            n_elem   = summary.get("total_elements") or summary.get("total_entities") or 0
            print(f"   📂 {fname} — {n_elem} elements via {pipeline} pipeline")
            self._emit("cad_node", "🔍", f"Loaded `{fname}` — {n_elem} elements")

            answer, judge_score, judge_comment = _call_llm_with_judge(
                system_prompt=system_prompt,
                history=cad_history,
                user_msg=query,
                context=context,
                question=query,
            )

            verdict = "✅" if judge_score >= 0.7 else "⚠️"
            print(f"   {verdict} cad_node judge score={judge_score:.2f} — {judge_comment[:60]}")

            # Store the new turn in CadSharedContext so follow-ups have memory
            CadSharedContext.append_turn(session_id, "user",      query)
            CadSharedContext.append_turn(session_id, "assistant", answer)

            return {
                **state,
                "answer":     answer,
                "raw_answer": answer,
                "sources":    [{"filename": fname, "pipeline": pipeline}],
                "confidence": float(judge_score),
                "analytics":  None,
            }

        except ImportError:
            print("⚠️  cad_node: cad_ifc_agent not installed — falling back to RAG")
            return self.retrieve_vector({**state, "route": "rag"})
        except Exception as e:
            import traceback as _tb; _tb.print_exc()
            err = f"CAD analysis failed: {e}"
            print(f"❌ cad_node: {err}")
            return {
                **state,
                "answer":     err,
                "raw_answer": err,
                "sources":    [],
                "confidence": 0.0,
                "error":      err,
            }

    # ------------------------------------------------------------------ #
    #  ANALYTICS NODE                                                     #
    # ------------------------------------------------------------------ #

    def analytics_node(self, state: AgentState) -> AgentState:
        """
        Generate structured analytics based on the documents.

        Analytics respects the judge's language/tone plan and works in ANY language.
        """
        query = state["query"]
        chunks = state["retrieved_chunks"]
        plan = state.get("response_plan")

        print(f"📊 Analytics generation → ", end="")

        if not chunks:
            print("no documents")
            return {**state, "error": "No documents for analytics"}

        context = _build_context(chunks)

        # Use plan's language if available, otherwise detect from query
        user_lang = _detect_script_language(query)
        target_lang = (plan.target_language if plan else None) or user_lang

        prompt = f"""Analyze the documents and produce a structured JSON response.

🚨 LANGUAGE: ALL text fields MUST be in {target_lang.upper()}. The documents below may be in a different language — IGNORE their language. ALL summary, findings, and recommendations MUST be written in {target_lang.upper()}.

User Question: {query}

Documents:
{context}

Instructions:
- Respond in {target_lang.upper()}
- Provide analytical insights based on the documents
- Return ONLY a valid JSON object with this structure:

{{
  "summary": "2-3 sentence summary in {target_lang}",
  "key_metrics": {{"metric_name": value}},
  "findings": ["finding 1 in {target_lang}", "finding 2 in {target_lang}"],
  "recommendations": ["recommendation 1 in {target_lang}"],
  "data_quality": "high|medium|low"
}}

Remember: ALL text fields must be in {target_lang.upper()}."""

        analytics_data = None
        narrative = ""

        if self.llm.enabled:
            raw = self._chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            try:
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                analytics_data = json.loads(clean)
                narrative = analytics_data.get("summary", "")
            except (json.JSONDecodeError, AttributeError):
                narrative = raw

        if not analytics_data:
            analytics_data = self._build_fallback_analytics(chunks, target_lang)
            narrative = analytics_data.get("summary", "")

        print(f"done")

        return {
            **state,
            "answer": narrative,
            "sources": _format_sources(chunks, query, self.llm, narrative),
            "confidence": _confidence(chunks),
            "analytics": analytics_data,
        }

    def _build_fallback_analytics(self, chunks: List[Dict], lang: str = 'en') -> Dict:
        """
        Build basic analytics when LLM is unavailable.
        Uses simple heuristics to provide basic analytics in any language.
        """
        doc_types: Dict[str, int] = {}
        for c in chunks:
            dt = c["metadata"].get("doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1

        # Generic fallback that works for any language
        # The actual values are language-agnostic, labels are minimal
        return {
            "summary": f"Analysis of {len(chunks)} document segments.",
            "key_metrics": {
                "total_chunks": len(chunks),
                "document_types": len(doc_types)
            },
            "findings": [c["text"][:200] for c in chunks[:2]],
            "recommendations": ["Index more documents for better analysis"],
            "data_quality": "medium",
        }

    # ------------------------------------------------------------------ #
    #  GRAPH NODE                                                         #
    # ------------------------------------------------------------------ #

    def graph_node(self, state: AgentState) -> AgentState:
        """
        Generate a Chart.js config from document data.

        Retrieval already ran (same path as analytics_node), so we have
        state["retrieved_chunks"].  GraphAgent reads them and returns a
        Chart.js config stored in state["analytics"] with type="chart_config".

        The frontend checks response["analytics"]["type"] == "chart_config"
        and renders a Chart.js canvas instead of markdown text.
        """
        query  = state["query"]
        chunks = state["retrieved_chunks"]

        # Load ALL chunks for attached docs — graph needs complete datasets
        attached = state.get("attached_doc_filenames", [])
        session_id = state.get("session_id", "")
        user_id = state.get("user_id")
        if attached and self.vs:
            all_chunks = []
            for fname in attached:
                try:
                    doc_chunks = self.vs.get_all_chunks(fname, user_id=user_id, session_id=session_id)
                    all_chunks.extend(doc_chunks)
                    print(f"   📊 graph: loaded {len(doc_chunks)} chunks from {fname}")
                except Exception as e:
                    print(f"   ⚠️  graph: get_all_chunks error for {fname}: {e}")
            if all_chunks:
                chunks = all_chunks

        cb = getattr(self, "_status_callback", None)
        if callable(cb):
            msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
            icon, msg = msgs.get("graph_node", ("📈", "Building chart from documents…"))
            cb("graph_node", icon, msg)

        print(f"📈 graph_node → query={query[:80]}")

        if not self.graph_agent:
            fallback = "Chart generation is unavailable (graph_agent module not loaded)."
            return {**state, "answer": fallback, "raw_answer": fallback,
                    "sources": [], "confidence": 0.0, "analytics": None}

        # Ask the judge for language/tone — same as every other node
        history = state.get("conversation_history", [])
        history_texts = [f"{m['role'].upper()}: {m['content'][:150]}" for m in history[-6:]]
        plan = self.judge.plan_response(
            query,
            retrieved_docs=chunks,
            conversation_history=history_texts,
        )

        # Detect if this query is a clarification follow-up (user picked a chart option)
        was_clarification_response = (
            state.get("force_route") == "graph"
            and any(m.get("role") == "assistant" for m in history[-2:])
        )

        result = self.graph_agent.build_chart(
            query=query,
            chunks=chunks,
            language=plan.target_language,
            skip_clarification=was_clarification_response,
        )

        # ── Clarification needed: vague query or no file specified ──────────
        if result.get("type") == "chart_clarification":
            clarif_mode  = result.get("clarification_mode", "metric")
            raw_groups   = result.get("groups", [])
            valid_groups = raw_groups  # default: trust file-mode groups as-is

            # For metric-mode only: pre-validate each group so users never click
            # an option and get a chart_error back. File-mode groups are already
            # filtered inside _discover_file_groups so no double-probe needed.
            if clarif_mode == "metric":
                print(f"   🤔 graph_node: validating {len(raw_groups)} metric groups…")
                valid_groups = []
                _cancel_ev = getattr(self, "_cancel_event", None)
                for grp in raw_groups:
                    # Stop probing if the user already pressed Stop
                    if _cancel_ev and _cancel_ev.is_set():
                        print("   🛑 graph_node: group probe cancelled by stop event")
                        break
                    hint = grp.get("hint") or grp.get("label", "")
                    if not hint:
                        continue
                    try:
                        probe = self.graph_agent.build_chart(
                            query=hint,
                            chunks=chunks,
                            language=plan.target_language,
                            skip_clarification=True,
                        )
                        if probe.get("type") == "chart_config":
                            valid_groups.append(grp)
                            print(f"      ✅ group valid: {grp.get('label','?')}")
                        else:
                            print(f"      ❌ group invalid (no data): {grp.get('label','?')}")
                    except Exception as _ve:
                        # Probe failed due to LLM error (e.g. Groq 429) — keep the group
                        # rather than silently dropping it; a transient error doesn't mean
                        # the data doesn't exist.
                        valid_groups.append(grp)
                        print(f"      ⚠️  group probe failed for '{hint}' (kept): {_ve}")

                if not valid_groups:
                    print(f"   ⚠️  graph_node: no valid chart groups found after validation")
                    fallback_answer = (
                        "I couldn't find numeric or structured data in your documents "
                        "that's ready to chart right now. Make sure your documents contain "
                        "tables or columns of measurements, then try a specific request — "
                        "for example: 'chart the fiber strand counts by site as a bar chart'."
                    )
                    return {
                        **state,
                        "answer":     fallback_answer,
                        "raw_answer": fallback_answer,
                        "sources":    [],
                        "confidence": 0.0,
                        "analytics":  {"type": "chart_error", "message": fallback_answer},
                    }
            else:
                print(f"   🤔 graph_node: file-picker clarification — {len(raw_groups)} options")

            clarif_answer = result.get(
                "question",
                "Your request covers several different types of data. Which would you like to chart?"
            )
            result_validated = {**result, "groups": valid_groups}
            print(f"   🤔 graph_node: clarification ({clarif_mode} mode, {len(valid_groups)} groups)")
            return {
                **state,
                "answer":     clarif_answer,
                "raw_answer": clarif_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result_validated,
            }

        # ── Chart generation failed ───────────────────────────────────────
        if result.get("type") == "chart_error":
            print(f"   ⚠️  graph_node: {result['message']}")
            # Build a realistic example from the actual retrieved chunks
            _example_hint = _build_chart_example_hint(chunks)
            fallback_answer = (
                f"I wasn't able to generate a chart: {result['message']} "
                f"Make sure your documents contain tables or columns of numeric data, "
                f"then try something specific — for example: '{_example_hint}'."
            )
            return {
                **state,
                "answer":     fallback_answer,
                "raw_answer": fallback_answer,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  result,
            }

        # ── Chart ready ───────────────────────────────────────────────────
        title       = result.get("title", "Chart")
        sources_str = ", ".join(result.get("sources", []))
        answer = (
            f"Here is the **{title}** chart generated from your documents"
            f"{(' (' + sources_str + ')') if sources_str else ''}."
        ).strip()

        print(f"   ✅ graph_node: chart ready — {title}")

        return {
            **state,
            "response_plan": plan,
            "answer":        answer,
            "raw_answer":    answer,
            "sources":       [],
            "confidence":    _confidence(chunks),
            "analytics":     result,
        }

    # ------------------------------------------------------------------ #
    #  REPORT NODE                                                        #
    # ------------------------------------------------------------------ #

    def report_node(self, state: AgentState) -> AgentState:
        """
        Generate a full structured report from retrieved chunks.
        Runs _generate_report_content in a sub-thread and emits keepalive
        status pings every 5 s so the SSE connection never idles out during
        the long multi-LLM-call generation (can take 30-90 s).
        """
        import threading as _threading
        import uuid as _uuid
        from datetime import datetime as _dt

        query      = state["query"]
        chunks     = state["retrieved_chunks"]
        history    = state.get("conversation_history", [])
        session_id = state.get("session_id", "")

        from llm_client import call_llm

        cb = getattr(self, "_status_callback", None)
        if callable(cb):
            msgs = getattr(self, "_status_msgs", None) or self._DEFAULT_STATUS_MSGS
            icon, msg = msgs.get("report_node", ("📄", "Writing your report…"))
            cb("report_node", icon, msg)

        print(f"📄 report_node → query={query[:80]}")

        try:
            if not _REPORT_AGENT_AVAILABLE or _report_agent_module is None:
                raise ImportError("report_agent module not available")

            # Use module-level references so we write into the SAME _reports_store
            # dict that the FastAPI GET /reports/{id} endpoint reads from.
            SharedContext            = _report_agent_module.SharedContext
            _generate_report_content = _report_agent_module._generate_report_content
            _reports_store           = _report_agent_module._reports_store
            _save_report_to_disk     = _report_agent_module._save_report_to_disk

            # Sync context so report_agent has the latest chunks + history
            if session_id:
                SharedContext.set_chunks(session_id, chunks)
                SharedContext.set_history(session_id, history)

            # Build available / explicit doc lists from chunks
            available_docs: list = list({
                c.get("metadata", {}).get("filename", "")
                for c in chunks
                if c.get("metadata", {}).get("filename")
            })
            explicit_docs: list = []
            if available_docs:
                ql = query.lower()
                for doc in available_docs:
                    stem = doc.lower().rsplit(".", 1)[0]
                    if stem in ql or doc.lower() in ql:
                        explicit_docs.append(doc)

            # ── Chart clarification flow ──────────────────────────────────────
            # Use a session-level flag in SharedContext — reliable, no history sniffing.
            SharedContext = _report_agent_module.SharedContext

            # ── LLM-based chart intent classifier ────────────────────────────
            # Replaces all hardcoded regexes. The LLM understands intent across
            # languages, typos, paraphrases, and implicit chart requests.
            # Returns JSON:
            #   wants_charts: bool  — does the user want any chart at all?
            #   already_specified: bool — did they name exactly what to chart
            #                             (metric + optional chart type), leaving
            #                             nothing ambiguous to clarify?
            #   wants_all: bool     — (clarif-answer context) wants every chart
            #   wants_none: bool    — (clarif-answer context) wants no charts
            def _classify_chart_intent(text: str, is_clarif_answer: bool) -> dict:
                """Ask the LLM what the user means re: charts. Fast, cheap call."""
                try:
                    _system = (
                        "You are a precise intent classifier for a report-generation assistant. "
                        "The user may write in any language, with typos or abbreviations. "
                        "Reply with ONLY a raw JSON object — no markdown, no explanation.\n"
                        'Format: {"wants_charts": bool, "already_specified": bool, "wants_all": bool, "wants_none": bool}\n\n'
                        "Field definitions:\n"
                        "  wants_charts: true if the user wants one or more charts/graphs/visuals in their report. "
                        "    Catch any language or spelling: 'graphe', 'grafico', 'diagramme', 'визуализация', 'رسم بياني', etc.\n"
                        "  already_specified: true ONLY if the user has named both (a) what metric/data to chart "
                        "    AND left nothing ambiguous — e.g. 'bar chart of trenching meters', "
                        "    'camembert des coûts', 'grafico a linee del throughput'. "
                        "    False if the request is vague like 'include charts' or 'add some graphs'.\n"
                        "  wants_all: true if this is a clarification answer meaning 'include all charts' "
                        "    (e.g. 'all of them', 'tous', 'todas', 'كلها', 'все').\n"
                        "  wants_none: true if this is a clarification answer meaning 'no charts at all' "
                        "    (e.g. 'no charts', 'sans graphiques', 'sin gráficos', 'بدون مخططات').\n"
                        f"  is_clarification_answer: {is_clarif_answer} — "
                        "    set to true context: the system previously asked which charts to include, "
                        "    and this text is the user's reply to that question."
                    )
                    _prompt = f'User message: """{text}"""'
                    _raw = call_llm(
                        _prompt,
                        system_prompt=_system,
                        max_tokens=80,
                        temperature=0.0,
                        task="classify",
                        preferred_provider=state.get("preferred_provider"),
                    )
                    import json as _j, re as _re2
                    _clean = _re2.sub(r"```(?:json)?|```", "", _raw).strip()
                    # extract first {...} in case the LLM adds preamble
                    _m = _re2.search(r'\{[^}]+\}', _clean)
                    _parsed = _j.loads(_m.group(0) if _m else _clean)
                    return {
                        "wants_charts":      bool(_parsed.get("wants_charts", False)),
                        "already_specified": bool(_parsed.get("already_specified", False)),
                        "wants_all":         bool(_parsed.get("wants_all", False)),
                        "wants_none":        bool(_parsed.get("wants_none", False)),
                    }
                except Exception as _e:
                    print(f"   ⚠️  _classify_chart_intent failed: {_e} — defaulting to wants_charts=False")
                    # Safe fallback: don't ask for clarification, don't build charts
                    return {"wants_charts": False, "already_specified": False,
                            "wants_all": False, "wants_none": False}

            # True if we previously returned a chart clarification for this session
            _prev_was_clarif = SharedContext.get_pending_chart_clarif(session_id)

            _chart_intent = _classify_chart_intent(query, is_clarif_answer=_prev_was_clarif)
            wants_charts         = _chart_intent["wants_charts"]
            chart_already_specified = _chart_intent["already_specified"]
            print(
                f"   🧠 chart_intent: wants={wants_charts} specified={chart_already_specified} "
                f"all={_chart_intent['wants_all']} none={_chart_intent['wants_none']} "
                f"prev_clarif={_prev_was_clarif}"
            )

            # Pre-built chart list to pass into _generate_report_content
            requested_charts: list = []

            # Helper: resolve GraphAgent module once
            def _load_graph_agent_mod():
                import importlib as _il, sys as _s
                if "services.graph_agent" in _s.modules:
                    return _s.modules["services.graph_agent"]
                try:
                    return _il.import_module("services.graph_agent")
                except ModuleNotFoundError:
                    return _il.import_module("graph_agent")

            if wants_charts and not _prev_was_clarif and _GRAPH_AGENT_AVAILABLE:
                if chart_already_specified:
                    # User told us exactly what they want — build it directly, no dialog.
                    try:
                        _ga_s = _load_graph_agent_mod().GraphAgent()
                        cr_s = _ga_s.build_chart(
                            query=query, chunks=chunks, language="en",
                            skip_clarification=True,
                        )
                        if cr_s.get("type") == "chart_config":
                            requested_charts = [cr_s]
                            print("   📊 report_node: chart built directly (already specified)")
                    except Exception as _ce_s:
                        print(f"   ⚠️  report_node direct chart build: {_ce_s}")
                else:
                    # Vague — discover what metrics exist and ask the user to pick.
                    try:
                        _ga = _load_graph_agent_mod().GraphAgent()
                        groups = _ga._discover_metric_groups(query, chunks, "en")

                        if groups and len(groups) > 1:
                            group_names = ", ".join(g["label"] for g in groups[:4])
                            clarif = (
                                f"I can build this report and include charts from the data. "
                                f"I found several data sets I could visualize: "
                                f"**{group_names}**"
                                f"{'...' if len(groups) > 4 else ''}.\n\n"
                                f"Which charts would you like included? "
                                f"You can name one or more, or say **\"all of them\"** — "
                                f"or say **\"no charts\"** to get the report without visuals."
                            )
                            print(f"   🤔 report_node: chart clarification — {len(groups)} groups found")
                            SharedContext.set_pending_chart_clarif(session_id, True)
                            return {
                                **state,
                                "answer":     clarif,
                                "raw_answer": clarif,
                                "sources":    [],
                                "confidence": 0.5,
                                "analytics":  {"type": "report_chart_clarification", "groups": groups},
                                "report_id":  None,
                            }
                        elif groups and len(groups) == 1:
                            # Only one option — build it without asking
                            cr = _ga.build_chart(
                                query=groups[0].get("hint", groups[0]["label"]),
                                chunks=chunks, language="en", skip_clarification=True,
                            )
                            if cr.get("type") == "chart_config":
                                requested_charts = [cr]
                    except Exception as _ce:
                        print(f"   ⚠️  report_node chart discovery: {_ce}")

            elif _prev_was_clarif and _GRAPH_AGENT_AVAILABLE:
                # User answered the clarification — clear the flag immediately so
                # any subsequent report request starts fresh.
                SharedContext.set_pending_chart_clarif(session_id, False)

                if not _chart_intent["wants_none"]:
                    try:
                        _ga2 = _load_graph_agent_mod().GraphAgent()

                        if _chart_intent["wants_all"]:
                            # Build one chart per discovered group
                            groups2 = _ga2._discover_metric_groups(query, chunks, "en")
                            targets = [g.get("hint", g["label"]) for g in (groups2 or [])[:4]]
                        else:
                            # User named specific chart(s) — pass their reply as the query
                            targets = [query]

                        for target_q in targets:
                            cr = _ga2.build_chart(
                                query=target_q, chunks=chunks, language="en",
                                skip_clarification=True,
                            )
                            if cr.get("type") == "chart_config":
                                requested_charts.append(cr)

                        print(f"   📊 report_node: {len(requested_charts)} chart(s) confirmed by user")
                    except Exception as _ce2:
                        print(f"   ⚠️  report_node chart build after clarif: {_ce2}")

            # Clear any lingering clarification flag before building
            SharedContext.set_pending_chart_clarif(session_id, False)

            # ── Run the heavy generation in a sub-thread ──────────────────────
            # report_node itself is already running in a thread (from main.py).
            # We spin a second thread so we can emit keepalive status pings every
            # 5 s while generation runs — this prevents the SSE stream from being
            # dropped by browsers / proxies that close idle connections after ~30 s.
            result_box: list  = [None]   # [result_dict] on success
            error_box:  list  = [None]   # [exception]   on failure
            done_event = _threading.Event()

            def _generate():
                try:
                    result_box[0] = _generate_report_content(
                        prompt          = query,
                        chunks          = chunks,
                        history         = history,
                        language        = None,
                        available_docs  = available_docs,
                        explicit_docs   = explicit_docs,
                        requested_charts= requested_charts,
                    )
                except Exception as _e:
                    error_box[0] = _e
                finally:
                    done_event.set()

            gen_thread = _threading.Thread(target=_generate, daemon=True)
            gen_thread.start()

            # Emit a keepalive ping every 5 s while waiting (max 10 min)
            _progress_msgs = [
                ("📄", "Analysing document content…"),
                ("🗂️",  "Planning report structure…"),
                ("✍️",  "Writing sections…"),
                ("✍️",  "Writing sections…"),
                ("✍️",  "Still writing — large document…"),
                ("✍️",  "Still writing — large document…"),
                ("💾",  "Finalising report…"),
            ]
            _ping_idx = 0
            _cancel_ev = getattr(self, "_cancel_event", None)
            while not done_event.wait(timeout=5):
                if _cancel_ev and _cancel_ev.is_set():
                    print(f"   🛑 report_node: cancelled by stop event — aborting wait")
                    error_box[0] = RuntimeError("Cancelled by user")
                    done_event.set()
                    break
                if callable(cb):
                    _icon, _msg = _progress_msgs[min(_ping_idx, len(_progress_msgs) - 1)]
                    cb("report_node", _icon, _msg)
                _ping_idx += 1

            gen_thread.join(timeout=5)

            if error_box[0] is not None:
                raise error_box[0]

            result = result_box[0]
            if result is None:
                raise RuntimeError("_generate_report_content returned None")

            # ── Save report ───────────────────────────────────────────────────
            report_id = str(_uuid.uuid4())
            now       = _dt.now().isoformat()
            report    = {
                "report_id":   report_id,
                "title":       result["title"],
                "content":     result["content"],
                "charts":      result.get("charts", []),
                "source_docs": result["source_docs"],
                "language":    result["language"],
                "summary":     result.get("summary", ""),
                "created_at":  now,
                "updated_at":  now,
                "version":     1,
                "versions": [{
                    "version":     1,
                    "title":       result["title"],
                    "instruction": query,
                    "created_at":  now,
                    "content":     result["content"],
                }],
                "session_id":  session_id,
            }
            _reports_store[report_id] = report
            _save_report_to_disk(report)

            answer = (
                result.get("summary")
                or f'Your report **"{result["title"]}"** is ready — click the panel to view or download it.'
            )
            print(f"   ✅ report_node: saved → {report_id} ({result['title']!r})")

            return {
                **state,
                "answer":     answer,
                "raw_answer": answer,
                "sources":    [],
                "confidence": 1.0,
                "analytics":    None,
                "report_id":    report_id,
                "report_title": result["title"],
            }

        except Exception as e:
            import traceback; traceback.print_exc()
            err = f"Report generation failed: {e}"
            print(f"   ❌ report_node: {err}")
            return {
                **state,
                "answer":     err,
                "raw_answer": err,
                "sources":    [],
                "confidence": 0.0,
                "analytics":  None,
            }
