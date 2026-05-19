from fastapi import APIRouter, Header, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from typing import Optional
import uuid
import json
import asyncio
import threading
import queue
import re

from models.query import QueryRequest, QueryResponse
from core.globals import (
    rag_engine, _session_user_context, _session_routes,
    _cancel_events, _session_file_stack,
)
from core.session_state import (
    get_history, get_route_log, append_turn, log_route,
    commit_pending_docs,
)
from services.report_agent import SharedContext
from services.llm_client import AllProvidersRateLimited
from prompt_loader import load_prompt

router = APIRouter(tags=["chat"])


def _generate_chat_title(session_id: str, messages: list) -> str:
    """
    Generate a chat title after the first exchange.
    Returns the title string, or empty string on failure.
    """
    from llm_client import call_llm

    excerpt = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:150]}"
        for m in messages[:6]
    )
    system = load_prompt("title_chat_system")
    prompt = f"Conversation:\n{excerpt}\n\nTitle:"

    try:
        raw = call_llm(prompt, system_prompt=system, max_tokens=30, temperature=0.3, task="classify")
        title = raw.strip().strip('"').strip("'").strip()
        if not title or len(title) > 100:
            return ""
        return title
    except Exception as e:
        print(f"⚠️  title generation failed: {e}")
        # Fallback: clean first user message
        for m in messages:
            if m.get("role") == "user":
                fallback = re.sub(r'[#*_`~]', '', str(m.get("content", "")))[:50].strip()
                if len(fallback) > 40:
                    fallback = fallback[:40].rsplit(' ', 1)[0] + "…"
                return fallback
        return ""


def _update_neo4j_title(session_id: str, title: str, frontend_conv_id: str = None):
    """Update conversation title in Neo4j."""
    try:
        from neo4j_auth import _run as _neo4j_run
        from core.globals import _conversation_cache
        _conv_id = frontend_conv_id or _conversation_cache.get(session_id)
        if not _conv_id:
            _rows = _neo4j_run(
                "MATCH (c:Conversation {session_id: $sid}) RETURN c.id AS id LIMIT 1",
                {"sid": session_id},
            )
            _conv_id = _rows[0]["id"] if _rows else None
        if _conv_id:
            _neo4j_run(
                "MATCH (c:Conversation {id: $conv_id}) SET c.title = $title, c.updated_at = $now",
                {"conv_id": _conv_id, "title": title, "now": __import__('datetime').datetime.utcnow().isoformat()},
            )
            print(f"✅ Neo4j title updated: {title!r}")
    except Exception as e:
        print(f"⚠️  Neo4j title update failed: {e}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        from neo4j_auth import optional_user
        user = optional_user(authorization)
        user_id = user["user_id"] if user else None

        if user:
            _session_user_context[session_id] = user["user_id"]

        history = get_history(session_id)

        pending_doc_ids = request.pending_doc_ids or []
        if pending_doc_ids:
            print(f"\U0001f4ce /query: {len(pending_doc_ids)} pending doc(s) — ensuring indexed before search")
            commit_pending_docs(pending_doc_ids, session_id, user_id)

        for _fname in (request.attached_doc_filenames or []):
            _stack = _session_file_stack.setdefault(session_id, [])
            if _fname in _stack:
                _stack.remove(_fname)
            _stack.append(_fname)

        print(f"\U0001f50d Query: {request.query} [session={session_id}, history={len(history)} turns, preferred_provider={request.preferred_provider!r}]")

        effective_history = history
        if request.voice_mode:
            voice_system = {
                "role": "system",
                "content": (
                    "VOICE MODE — your answer will be read aloud by a text-to-speech engine. "
                    "Rules you MUST follow:\n"
                    "- No markdown whatsoever: no *, **, _, __, #, `, bullets, numbered lists, or tables\n"
                    "- Write in short, natural spoken sentences — like a knowledgeable friend on the phone\n"
                    "- Max 4 sentences. Be concise and direct.\n"
                    "- Same language as the user's question\n"
                    "- Never open with 'Certainly', 'Of course', 'Sure', or 'Great question'"
                ),
            }
            effective_history = [voice_system] + list(history)

        prev_route = _session_routes.get(session_id, "")
        route_log  = get_route_log(session_id)

        _attached = list(request.attached_doc_filenames or [])

        result = rag_engine.query(
            request.query,
            top_k=request.top_k,
            conversation_history=effective_history,
            prev_route=prev_route,
            route_log=route_log,
            force_route=request.force_route,
            session_id=session_id,
            user_id=user_id,
            voice_mode=request.voice_mode,
            preferred_provider=request.preferred_provider,
            attached_doc_filenames=_attached,
        )

        clean_answer = re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
        append_turn(session_id, "user", request.query)

        if result.get("report_id") and result.get("report_title"):
            from services.report_agent import _reports_store as _rs
            _rpt = _rs.get(result["report_id"], {})
            if result["report_id"] in _rs and not _rpt.get("session_id"):
                _rs[result["report_id"]]["session_id"] = session_id
            _content = _rpt.get("content") or ""
            _source_docs = ", ".join(_rpt.get("source_docs") or []) or "unknown"
            _word_count = len(_content.split())
            _section_count = len(re.findall(r'^#{1,2}\s', _content, re.M))
            _preview = _content[:400].strip()
            memory_note = (
                f"{clean_answer}\n\n"
                f"[REPORT GENERATED]\n"
                f"Title: {result['report_title']}\n"
                f"Sources: {_source_docs}\n"
                f"Size: {_section_count} sections, {_word_count} words\n"
                f"Opening: {_preview}"
            )
            append_turn(session_id, "assistant", memory_note)
        else:
            append_turn(session_id, "assistant", clean_answer)

        if result.get("route"):
            _session_routes[session_id] = result["route"]
            log_route(session_id, result["route"], request.query)

        current_history = get_history(session_id)
        SharedContext.set_history(session_id, current_history)
        SharedContext.set_chunks(session_id, result.get("retrieved_chunks", []))

        # Generate title on first exchange
        chat_title = ""
        user_msg_count = sum(1 for m in current_history if m.get("role") == "user")
        if user_msg_count == 1 and not result.get("report_id"):
            chat_title = _generate_chat_title(session_id, current_history)
            if chat_title:
                _update_neo4j_title(session_id, chat_title)

        print(f"\u2705 Done (route={result.get('route')}, confidence={result['confidence']})")

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            route=result.get("route"),
            analytics=result.get("analytics"),
            session_id=session_id,
            chat_title=chat_title,
        )
    except AllProvidersRateLimited:
        raise HTTPException(429, detail={"error": "rate_limited", "message": "All LLM providers are rate-limited. Please change model or retry later."})
    except Exception as e:
        print(f"\u274c Query error: {e}")
        raise HTTPException(500, f"Error processing query: {e}")


@router.post("/query-stream/cancel")
async def cancel_stream(body: dict):
    sid = body.get("session_id", "")
    if sid and sid in _cancel_events:
        _cancel_events[sid].set()
        print(f"\U0001f6d1 Cancel requested for session {sid}")
    return {"status": "ok"}


@router.post("/query-stream")
async def query_stream(
    http_request: FastAPIRequest,
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    session_id = request.session_id or str(uuid.uuid4())

    from neo4j_auth import optional_user
    user = optional_user(authorization)
    user_id = user["user_id"] if user else None
    if user:
        _session_user_context[session_id] = user["user_id"]

    frontend_conv_id = request.conversation_id or None

    history    = get_history(session_id)
    prev_route = _session_routes.get(session_id, "")
    route_log  = get_route_log(session_id)

    effective_history = history
    if request.voice_mode:
        voice_system = {
            "role": "system",
            "content": (
                "VOICE MODE — your answer will be read aloud by a text-to-speech engine. "
                "Rules you MUST follow:\n"
                "- No markdown whatsoever: no *, **, _, __, #, `, bullets, numbered lists, or tables\n"
                "- Write in short, natural spoken sentences — like a knowledgeable friend on the phone\n"
                "- Max 4 sentences. Be concise and direct.\n"
                "- Same language as the user's question\n"
                "- Never open with 'Certainly', 'Of course', 'Sure', or 'Great question'"
            ),
        }
        effective_history = [voice_system] + list(history)

    print(f"\U0001f527 /query-stream preferred_provider={request.preferred_provider!r}")
    q: queue.Queue = queue.Queue()
    DONE_SENTINEL = object()

    cancel_event = threading.Event()
    _cancel_events[session_id] = cancel_event

    def status_callback(node: str, icon: str, message: str):
        if cancel_event.is_set():
            return
        print(f"[SSE status] node={node} icon={icon} message={message}")
        q.put({"type": "status", "node": node, "icon": icon, "message": message})

    def run_query():
        try:
            _pending = request.pending_doc_ids or []
            if _pending:
                print(f"\U0001f4ce /query-stream: {len(_pending)} pending doc(s) — ensuring indexed")
                commit_pending_docs(_pending, session_id, user_id)

            if cancel_event.is_set():
                print(f"\U0001f6d1 /query-stream: cancelled before LLM call (session={session_id})")
                q.put(DONE_SENTINEL)
                return

            _attached = list(request.attached_doc_filenames or [])
            for _fname in _attached:
                _stack_s = _session_file_stack.setdefault(session_id, [])
                if _fname in _stack_s:
                    _stack_s.remove(_fname)
                _stack_s.append(_fname)

            result = rag_engine.query(
                request.query,
                top_k=request.top_k,
                conversation_history=effective_history,
                prev_route=prev_route,
                route_log=route_log,
                status_callback=status_callback,
                force_route=request.force_route,
                session_id=session_id,
                user_id=user_id,
                voice_mode=request.voice_mode,
                preferred_provider=request.preferred_provider,
                attached_doc_filenames=_attached,
                cancel_event=cancel_event,
            )

            if cancel_event.is_set():
                print(f"\U0001f6d1 /query-stream: result discarded (cancelled) session={session_id}")
                q.put(DONE_SENTINEL)
                return

            print(f"\u2705 /query-stream done route={result.get('route')!r} preferred_provider={request.preferred_provider!r}")
            clean_answer = re.sub(r'\s*\[\d+\]', '', result["answer"]).strip()
            append_turn(session_id, "user", request.query, frontend_conv_id)

            if result.get("report_id") and result.get("report_title"):
                from services.report_agent import _reports_store as _rs
                _rpt        = _rs.get(result["report_id"], {})
                if result["report_id"] in _rs and not _rpt.get("session_id"):
                    _rs[result["report_id"]]["session_id"] = session_id
                _content    = _rpt.get("content") or ""
                _source_docs = ", ".join(_rpt.get("source_docs") or []) or "unknown"
                _word_count  = len(_content.split())
                _section_count = len(re.findall(r'^#{1,2}\s', _content, re.M))
                _preview     = _content[:400].strip()
                memory_note  = (
                    f"{clean_answer}\n\n"
                    f"[REPORT GENERATED]\n"
                    f"Title: {result['report_title']}\n"
                    f"Sources: {_source_docs}\n"
                    f"Size: {_section_count} sections, {_word_count} words\n"
                    f"Opening: {_preview}"
                )
                append_turn(session_id, "assistant", memory_note, frontend_conv_id)

                try:
                    from neo4j_auth import _run as _neo4j_run
                    from core.globals import _conversation_cache
                    from datetime import datetime
                    _now     = datetime.utcnow().isoformat()
                    _conv_id = frontend_conv_id or _conversation_cache.get(session_id)
                    if not _conv_id:
                        _rows = _neo4j_run(
                            "MATCH (c:Conversation {session_id: $sid}) RETURN c.id AS id LIMIT 1",
                            {"sid": session_id},
                        )
                        _conv_id = _rows[0]["id"] if _rows else None
                    if _conv_id:
                        _neo4j_run(
                            """
                            MERGE (r:Report {id: $report_id})
                            SET r.title         = $title,
                                r.query         = $query,
                                r.source_docs   = $source_docs,
                                r.word_count    = $word_count,
                                r.section_count = $section_count,
                                r.preview       = $preview,
                                r.session_id    = $session_id,
                                r.created_at    = $now
                            WITH r
                            MATCH (c:Conversation {id: $conv_id})
                            MERGE (c)-[:HAS_REPORT]->(r)
                            """,
                            {
                                "report_id":     result["report_id"],
                                "title":         result["report_title"],
                                "query":         request.query[:300],
                                "source_docs":   _source_docs,
                                "word_count":    _word_count,
                                "section_count": _section_count,
                                "preview":       _preview[:400],
                                "session_id":    session_id,
                                "conv_id":       _conv_id,
                                "now":           _now,
                            },
                        )
                        print(f"\u2601\ufe0f  Neo4j: inline Report '{result['report_title']}' → Conversation {_conv_id}")
                except Exception as _ne:
                    print(f"\u26a0\ufe0f  Neo4j inline report save failed (non-fatal): {_ne}")
            else:
                append_turn(session_id, "assistant", clean_answer, frontend_conv_id)

            if result.get("route"):
                _session_routes[session_id] = result["route"]
                log_route(session_id, result["route"], request.query)

            current_history = get_history(session_id)
            SharedContext.set_history(session_id, current_history)
            SharedContext.set_chunks(session_id, result.get("retrieved_chunks", []))
            SharedContext.set_analytics(session_id, result.get("analytics"))

            # Generate title on first exchange (1 user + 1 assistant = 2 messages)
            chat_title = ""
            user_msg_count = sum(1 for m in current_history if m.get("role") == "user")
            if user_msg_count == 1 and not result.get("report_id"):
                chat_title = _generate_chat_title(session_id, current_history)
                if chat_title:
                    _update_neo4j_title(session_id, chat_title, frontend_conv_id)

            safe_result = {
                "answer":       result.get("answer", ""),
                "raw_answer":   result.get("raw_answer") or result.get("answer", ""),
                "sources":      result.get("sources", []),
                "confidence":   result.get("confidence", 0.0),
                "route":        result.get("route"),
                "analytics":    result.get("analytics"),
                "report_id":    result.get("report_id"),
                "report_title": result.get("report_title"),
                "clarification_options": result.get("clarification_options") or [],
                "chat_title":   chat_title,
            }
            q.put({"type": "result", "session_id": session_id, **safe_result})
        except AllProvidersRateLimited:
            if not cancel_event.is_set():
                q.put({"type": "error", "error_type": "rate_limited", "message": "All LLM providers are rate-limited. Please change model or retry later."})
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            if not cancel_event.is_set():
                q.put({"type": "error", "message": f"Server error: {e}"})
        finally:
            q.put(DONE_SENTINEL)
            _cancel_events.pop(session_id, None)

    thread = threading.Thread(target=run_query, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            if await http_request.is_disconnected():
                cancel_event.set()
                print(f"\U0001f6d1 /query-stream: client disconnected (session={session_id})")
                break

            try:
                item = await loop.run_in_executor(None, lambda: q.get(timeout=5))
            except Exception:
                yield ": keepalive\n\n"
                continue
            if item is DONE_SENTINEL:
                break
            if not cancel_event.is_set():
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
