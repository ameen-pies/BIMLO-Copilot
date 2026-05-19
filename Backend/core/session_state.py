from collections import deque
from typing import List, Optional
import time
import os
import tempfile

from core.globals import (
    _sessions, _session_routes, _session_route_log,
    _session_user_context, _conversation_cache,
    _session_file_stack,
    DOCUMENT_FILE_CACHE, MAX_HISTORY_TURNS,
    doc_processor, vector_store,
)


def get_history(session_id: str) -> List[dict]:
    if session_id in _sessions:
        return list(_sessions[session_id])
    try:
        from neo4j_auth import _run as neo4j_run
        rows = neo4j_run(
            """
            MATCH (c:Conversation {session_id: $sid})-[r:CONTAINS]->(m:Message)
            RETURN m.role AS role, m.content AS content
            ORDER BY r.index ASC
            """,
            {"sid": session_id},
        )
        if rows:
            _sessions[session_id] = deque(
                [{"role": r["role"], "content": r["content"]} for r in rows],
                maxlen=MAX_HISTORY_TURNS,
            )
            return list(_sessions[session_id])
    except Exception as e:
        print(f"\u26a0\ufe0f  Neo4j history reload failed: {e}")
    return []


def get_route_log(session_id: str) -> list:
    return list(_session_route_log.get(session_id, []))


def _auto_save_to_neo4j(session_id: str, role: str, content: str, frontend_conv_id: str = None):
    try:
        from neo4j_auth import _run as neo4j_run
        from datetime import datetime as _dt
        import uuid as _uuid_mod

        user_id = _session_user_context.get(session_id)
        now     = _dt.utcnow().isoformat()

        if user_id:
            conv_id = frontend_conv_id
            if not conv_id:
                return
            neo4j_run(
                """
                MATCH (u:User {id: $user_id})
                MERGE (c:Conversation {id: $conv_id})
                ON CREATE SET c.session_id = $session_id,
                              c.title      = "Chat",
                              c.preview    = "",
                              c.chat_type  = "rag",
                              c.created_at = $now,
                              c.updated_at = $now
                ON MATCH  SET c.session_id = $session_id
                MERGE (u)-[:HAS_CONVERSATION]->(c)
                """,
                {"user_id": user_id, "conv_id": conv_id, "session_id": session_id, "now": now},
            )
            return

        conv_id = frontend_conv_id or _conversation_cache.get(session_id)

        if not conv_id:
            rows = neo4j_run(
                "MATCH (c:Conversation {session_id: $sid}) RETURN c.id AS id LIMIT 1",
                {"sid": session_id},
            )
            if rows:
                conv_id = rows[0]["id"]

        if not conv_id:
            conv_id = str(_uuid_mod.uuid4())

        _conversation_cache[session_id] = conv_id

        neo4j_run(
            """
            MERGE (c:Conversation {id: $conv_id})
            ON CREATE SET c.session_id = $session_id,
                          c.title      = "Chat",
                          c.preview    = "",
                          c.chat_type  = "rag",
                          c.created_at = $now,
                          c.updated_at = $now
            ON MATCH  SET c.session_id = $session_id,
                          c.updated_at = $now
            """,
            {"conv_id": conv_id, "session_id": session_id, "now": now},
        )

        msg_id     = str(_uuid_mod.uuid4())
        ts         = _dt.utcnow().isoformat()
        count_rows = neo4j_run(
            "MATCH (:Conversation {id: $conv_id})-[r:CONTAINS]->(m:Message) RETURN COUNT(r) AS cnt",
            {"conv_id": conv_id},
        )
        index = count_rows[0]["cnt"] if count_rows else 0
        neo4j_run(
            """
            MATCH (c:Conversation {id: $conv_id})
            CREATE (msg:Message {
                id:        $msg_id,
                role:      $role,
                content:   $content,
                timestamp: $ts,
                payload:   ""
            })
            CREATE (c)-[:CONTAINS {index: $index}]->(msg)
            SET c.updated_at = $ts
            """,
            {"conv_id": conv_id, "msg_id": msg_id, "role": role,
             "content": content, "ts": ts, "index": index},
        )
        print(f"\U0001f4be Auto-saved (anon): {role} message to conversation {conv_id}")
    except Exception as e:
        print(f"\u26a0\ufe0f  Auto-save to Neo4j failed (non-fatal): {e}")


def append_turn(session_id: str, role: str, content: str, frontend_conv_id: str = None):
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_HISTORY_TURNS)
    _sessions[session_id].append({"role": role, "content": content})

    import threading as _thr
    _thr.Thread(
        target=_auto_save_to_neo4j,
        args=(session_id, role, content, frontend_conv_id),
        daemon=True,
    ).start()


def log_route(session_id: str, route: str, query: str):
    if session_id not in _session_route_log:
        _session_route_log[session_id] = []
    log = _session_route_log[session_id]
    log.append({"route": route, "query": query[:120]})
    if len(log) > 10:
        _session_route_log[session_id] = log[-10:]


def clear_history(session_id: str):
    _sessions.pop(session_id, None)
    _session_routes.pop(session_id, None)
    _session_route_log.pop(session_id, None)
    _session_file_stack.pop(session_id, None)


def commit_pending_docs(pending_doc_ids: List[str], session_id: str, user_id: Optional[str]):
    if not pending_doc_ids:
        return

    for _pending_id in pending_doc_ids:
        try:
            _existing = vector_store.list_documents(user_id=user_id, session_id=session_id)
            if any(d["document_id"] == _pending_id for d in _existing):
                print(f"\u2705 Doc {_pending_id} already indexed in session {session_id}")
                continue
        except Exception:
            pass

        _cached = DOCUMENT_FILE_CACHE.get(_pending_id)
        if not _cached:
            print(f"\u26a0\ufe0f  Doc {_pending_id} not in file cache — ingestion pipeline may still be running")
            continue

        if _cached.get("pipeline_submitted"):
            _waited = 0
            _poll_interval = 0.25
            _max_wait      = 30.0
            _found = False
            while _waited < _max_wait:
                try:
                    _check = vector_store.list_documents(user_id=user_id, session_id=session_id)
                    if any(d["document_id"] == _pending_id for d in _check):
                        print(f"\u2705 Doc {_pending_id} indexed by pipeline after {_waited:.1f}s wait")
                        _found = True
                        break
                except Exception:
                    pass
                time.sleep(_poll_interval)
                _waited += _poll_interval
            if not _found:
                print(
                    f"\u26a0\ufe0f  Doc {_pending_id} ('{_cached['filename']}') still not in Chroma after "
                    f"{_max_wait}s — checking one more time before falling back to direct index."
                )
                try:
                    _check = vector_store.list_documents(user_id=user_id, session_id=session_id)
                    if any(d["document_id"] == _pending_id for d in _check):
                        print(f"\u2705 Doc {_pending_id} found on final check")
                        continue
                except Exception:
                    pass
                _ext = _cached.get('doc_type', '.txt')
                # Skip CAD/IFC files — they're handled by the CAD agent pipeline,
                # not the standard document processor.
                _cad_exts = {'.ifc', '.dxf', '.dwg', '.step', '.stp'}
                if _ext.lower() in _cad_exts:
                    print(f"⏭️  Skipping '{_cached['filename']}': CAD file handled by CAD agent")
                    continue
                print(f"\u23f3 Pipeline timed out — indexing '{_cached['filename']}' directly")
                _is_image = _cached.get('is_image', False)
                if _is_image:
                    _chunks = doc_processor.process_image_bytes(_cached['bytes'], _cached['filename'])
                else:
                    with tempfile.NamedTemporaryFile(suffix=_ext, delete=False) as _tmp:
                        _tmp.write(_cached['bytes'])
                        _tmp_path = _tmp.name
                    try:
                        _chunks = doc_processor.process_document(_tmp_path)
                    finally:
                        os.unlink(_tmp_path)
                if not _chunks:
                    print(f"⚠️  Skipping '{_cached['filename']}': file is empty or unreadable (0 chunks)")
                    continue
                for _idx, _chunk in enumerate(_chunks):
                    if "metadata" not in _chunk:
                        _chunk["metadata"] = {}
                    _chunk["metadata"]["chunk_index"] = _idx
                    _chunk["metadata"]["document_id"] = _pending_id
                    if _is_image:
                        _chunk["metadata"]["doc_type"]   = "image"
                        _chunk["metadata"]["has_images"] = True
                vector_store.add_document(
                    _cached['filename'],
                    _chunks,
                    session_id=session_id,
                    user_id=user_id,
                    doc_id=_pending_id,
                )
                print(f"\u2705 Directly indexed '{_cached['filename']}' ({len(_chunks)} chunks) after pipeline timeout")
            continue

        print(f"\u23f3 Committing pending doc '{_cached['filename']}' ({_pending_id})…")
        try:
            _ext = _cached.get('doc_type', '.txt')
            _is_image = _cached.get('is_image', False)

            if _is_image:
                _chunks = doc_processor.process_image_bytes(_cached['bytes'], _cached['filename'])
            else:
                with tempfile.NamedTemporaryFile(suffix=_ext, delete=False) as _tmp:
                    _tmp.write(_cached['bytes'])
                    _tmp_path = _tmp.name
                try:
                    _chunks = doc_processor.process_document(_tmp_path)
                finally:
                    os.unlink(_tmp_path)

            for _idx, _chunk in enumerate(_chunks):
                if "metadata" not in _chunk:
                    _chunk["metadata"] = {}
                _chunk["metadata"]["chunk_index"] = _idx
                _chunk["metadata"]["document_id"] = _pending_id
                if _is_image:
                    _chunk["metadata"]["doc_type"]   = "image"
                    _chunk["metadata"]["has_images"] = True

            vector_store.add_document(
                _cached['filename'],
                _chunks,
                session_id=session_id,
                user_id=user_id,
                doc_id=_pending_id,
            )
            print(f"\u2705 Committed '{_cached['filename']}' ({len(_chunks)} chunks) → session {session_id}")
        except Exception as _err:
            print(f"\u274c Failed to commit pending doc {_pending_id}: {_err}")
