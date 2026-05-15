from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse, Response
from typing import Dict, List, Optional, Any
from io import BytesIO
import os, uuid, json, hashlib
from datetime import datetime

from core import globals as g

router = APIRouter(prefix="", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
    session_id: Optional[str] = None,
):
    try:
        text_allowed = ['.pdf', '.docx', '.doc', '.txt']
        image_allowed = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'}
        cad_allowed = {'.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp'}
        ext = os.path.splitext(file.filename)[1].lower()
        is_cad   = ext in cad_allowed
        is_image = ext in image_allowed
        if ext not in text_allowed and not is_cad and not is_image:
            raise HTTPException(400, f"Unsupported type. Allowed: {', '.join(text_allowed + list(image_allowed) + list(cad_allowed))}")

        content = await file.read()
        if len(content) > g.MAX_UPLOAD_SIZE:
            raise HTTPException(
                413,
                f"File too large: {len(content)/1024/1024:.1f} MB "
                f"> {g.MAX_UPLOAD_SIZE/1024/1024} MB"
            )
        print(f"\U0001f4c4 Received: {file.filename} ({len(content)} bytes)")

        import uuid as _uuid_mod
        session_id = session_id or str(_uuid_mod.uuid4())
        content_hash = hashlib.sha256(content).hexdigest()

        if session_id:
            try:
                from neo4j_auth import _run as neo4j_run
                rows = neo4j_run(
                    """
                    MATCH (c:Conversation {session_id: $session_id})-[:USED_DOCUMENT]->(d:Document)
                    WHERE d.content_hash = $content_hash
                    RETURN d.id AS document_id,
                           d.filename AS filename,
                           d.doc_type AS doc_type,
                           d.chunk_count AS chunk_count
                    LIMIT 1
                    """,
                    {"session_id": session_id, "content_hash": content_hash},
                )
                if rows:
                    existing = rows[0]
                    print(f"\u26a0\ufe0f  Duplicate upload detected for session {session_id}: returning existing document {existing['document_id']}")
                    _stack = g._session_file_stack.setdefault(session_id, [])
                    if existing['filename'] in _stack:
                        _stack.remove(existing['filename'])
                    _stack.append(existing['filename'])
                    return {
                        "status": "success",
                        "filename": existing['filename'],
                        "document_id": existing['document_id'],
                        "session_id": session_id,
                        "chunks_processed": existing['chunk_count'] or 0,
                        "cad_summary": None,
                        "is_image": ext in image_allowed,
                        "message": f"Duplicate file upload skipped; returning existing document '{existing['filename']}'.",
                    }
            except Exception as _dup_err:
                print(f"\u26a0\ufe0f  Duplicate file lookup failed: {_dup_err}")

        doc_id = str(_uuid_mod.uuid4())
        g.DOCUMENT_FILE_CACHE[doc_id] = {
            'filename': file.filename,
            'bytes': content,
            'content_type': g._guess_mime_type(file.filename),
            'session_id': session_id,
            'doc_type': ext,
            'is_image': is_image,
            'content_hash': content_hash,
        }
        chunks = []
        cad_summary = None

        if is_cad:
            if not g._cad_ifc_available:
                raise HTTPException(400, f"CAD/IFC support is not available on this server.")
            try:
                from cad_ifc_agent import _parse_file, CadSharedContext

                summary = _parse_file(file.filename, content)
                summary['file_id'] = doc_id
                CadSharedContext.cache_file(doc_id, summary)
                cad_summary = summary
                print(f"\u2705 CAD/IFC parsed: {file.filename} (doc_id={doc_id})")
            except Exception as cad_err:
                print(f"\u274c CAD upload error: {cad_err}")
                raise HTTPException(500, f"Error processing CAD/IFC document: {cad_err}")
        elif is_image:
            print(f"\U0001f5bc\ufe0f  Image upload detected: {file.filename} ({len(content)} bytes)")
            try:
                chunks = g.doc_processor.process_image_bytes(content, file.filename)
                for idx, chunk in enumerate(chunks):
                    if "metadata" not in chunk:
                        chunk["metadata"] = {}
                    chunk["metadata"]["chunk_index"] = idx
                    chunk["metadata"]["document_id"] = doc_id
                    chunk["metadata"]["doc_type"]    = "image"
                    chunk["metadata"]["has_images"]  = True
                print(f"\U0001f5bc\ufe0f  Image processed: {len(chunks)} chunk(s) for '{file.filename}'")
            except Exception as img_err:
                print(f"\u274c Image processing error: {img_err}")
                raise HTTPException(500, f"Error processing image: {img_err}")
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                chunks = g.doc_processor.process_document(tmp_path)
                for idx, chunk in enumerate(chunks):
                    if "metadata" not in chunk:
                        chunk["metadata"] = {}
                    chunk["metadata"]["chunk_index"] = idx
                    chunk["metadata"]["document_id"] = doc_id
            finally:
                os.unlink(tmp_path)

            print(f"\u2702\ufe0f  {len(chunks)} chunks created")

        from neo4j_auth import optional_user, _run as neo4j_run
        user = optional_user(authorization)
        user_id = user["user_id"] if user else None

        if g._ingestion_graph_available:
            g.run_ingestion_pipeline(g.vector_store, doc_id, file.filename, chunks, session_id=session_id, user_id=user_id)
            g.DOCUMENT_FILE_CACHE[doc_id]["pipeline_submitted"] = True
            print(f"\U0001f4ec Ingestion pipeline submitted for '{file.filename}' (doc_id={doc_id}, user={user_id or 'anonymous'}, session={session_id})")
        else:
            g.vector_store.add_document(file.filename, chunks, session_id=session_id, user_id=user_id)
            print(f"\u2705 Indexed (direct fallback): {doc_id} \u2192 user_{user_id or 'anon'}_session_{session_id}")
            try:
                from graph_rag import get_engine as _get_graph_engine
                import threading as _threading
                _graph_engine = _get_graph_engine()
                if _graph_engine.available:
                    def _ingest_fallback():
                        _graph_engine.ingest_chunks(doc_id, file.filename, chunks)
                    _threading.Thread(target=_ingest_fallback, daemon=True).start()
            except Exception as _ge:
                print(f"\u26a0\ufe0f  graph_rag ingestion skipped (fallback): {_ge}")

        try:
            from datetime import datetime as _dt
            _now = _dt.utcnow().isoformat()
            neo4j_run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.filename      = $filename,
                    d.doc_type      = $doc_type,
                    d.chunk_count   = $chunk_count,
                    d.uploaded_at   = $now,
                    d.session_id    = $session_id,
                    d.content_hash  = $content_hash
                """,
                {
                    "doc_id":      doc_id,
                    "filename":    file.filename,
                    "doc_type":    ext.lstrip("."),
                    "chunk_count": len(chunks),
                    "now":         _now,
                    "session_id":  session_id or "",
                    "content_hash": g.DOCUMENT_FILE_CACHE[doc_id].get('content_hash'),
                },
            )
            if session_id:
                neo4j_run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Conversation {session_id: $session_id})
                    MERGE (c)-[:USED_DOCUMENT]->(d)
                    """,
                    {"doc_id": doc_id, "session_id": session_id},
                )
            if user:
                neo4j_run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MATCH (u:User {id: $user_id})
                    MERGE (u)-[:UPLOADED]->(d)
                    """,
                    {"doc_id": doc_id, "user_id": user["user_id"]},
                )
                print(f"\u2601\ufe0f  Neo4j: '{file.filename}' \u2192 user {user['username']} | session {session_id}")
            else:
                print(f"\u2601\ufe0f  Neo4j: '{file.filename}' \u2192 anonymous session {session_id}")
        except Exception as neo_err:
            print(f"\u26a0\ufe0f  Neo4j doc save failed (non-fatal): {neo_err}")

        _stack = g._session_file_stack.setdefault(session_id, [])
        if file.filename not in _stack:
            _stack.append(file.filename)
        else:
            _stack.remove(file.filename)
            _stack.append(file.filename)

        return {
            "status":           "success",
            "filename":         file.filename,
            "document_id":      doc_id,
            "session_id":       session_id,
            "chunks_processed": len(chunks),
            "cad_summary":      cad_summary,
            "is_image":         is_image,
            "message":          f"'{file.filename}' processed and indexed successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"\u274c Upload error: {e}")
        raise HTTPException(500, f"Error processing document: {e}")


@router.get("/documents")
async def list_documents(
    session_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    try:
        if not session_id:
            return {"status": "success", "documents": [], "total": 0, "scoped": False}

        try:
            from neo4j_auth import _run as neo4j_run
            rows = neo4j_run(
                """
                MATCH (c:Conversation {session_id: $sid})-[:USED_DOCUMENT]->(d:Document)
                RETURN d.id        AS document_id,
                       d.filename  AS filename,
                       d.doc_type  AS doc_type,
                       d.chunk_count AS chunk_count,
                       d.uploaded_at AS timestamp
                ORDER BY d.uploaded_at DESC
                """,
                {"sid": session_id},
            )
            seen = set()
            docs = []
            for r in rows:
                if r["document_id"] in seen:
                    continue
                seen.add(r["document_id"])
                docs.append({
                    "document_id": r["document_id"],
                    "filename":    r["filename"] or "unknown",
                    "doc_type":    r["doc_type"] or "unknown",
                    "chunk_count": r["chunk_count"] or 0,
                    "timestamp":   r["timestamp"] or "",
                })
            return {"status": "success", "documents": docs, "total": len(docs), "scoped": True}
        except Exception as neo_err:
            print(f"\u26a0\ufe0f  Neo4j session-doc lookup failed: {neo_err}")
            return {"status": "success", "documents": [], "total": 0, "scoped": False}
    except Exception as e:
        raise HTTPException(500, f"Error listing documents: {e}")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    session_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    if not session_id:
        raise HTTPException(status_code=403, detail="Session ID required")

    try:
        from neo4j_auth import optional_user, _run as neo4j_run
        user = optional_user(authorization)
        user_id = user["user_id"] if user else None

        rows = neo4j_run(
            """
            MATCH (c:Conversation {session_id: $sid})-[:USED_DOCUMENT]->(d:Document {id: $doc_id})
            RETURN d.doc_type AS doc_type
            """,
            {"sid": session_id, "doc_id": doc_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_type = (rows[0].get("doc_type") or "").lower()
        normalized_doc_type = (doc_type or "").lstrip('.').lower()
        if g._cad_ifc_available and normalized_doc_type in {'ifc', 'ifczip', 'dxf', 'dwg', 'step', 'stp'}:
            try:
                from cad_ifc_agent import CadSharedContext
                CadSharedContext.delete_file(doc_id)
            except Exception:
                pass

        g.vector_store.delete_document(doc_id, user_id=user_id, session_id=session_id)

        neo4j_run(
            """
            MATCH (c:Conversation {session_id: $sid})-[r:USED_DOCUMENT]->(d:Document {id: $doc_id})
            DELETE r
            """,
            {"sid": session_id, "doc_id": doc_id},
        )
        g.DOCUMENT_FILE_CACHE.pop(doc_id, None)
        neo4j_run(
            "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
            {"doc_id": doc_id},
        )

        return {"status": "success", "message": f"Document {doc_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error deleting document: {e}")


@router.get("/documents/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    session_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None)
):
    if not session_id:
        raise HTTPException(status_code=403, detail="Session ID required")

    print(f"\U0001f4c4 [content] doc_id={doc_id} session_id={session_id}")

    try:
        from neo4j_auth import optional_user
        user = optional_user(authorization)
        user_id = user["user_id"] if user else None

        if g._cad_ifc_available:
            from cad_ifc_agent import CadSharedContext
            cad_file = CadSharedContext.get_file(doc_id)
            if cad_file:
                import json as _json
                return {"document_id": doc_id, "filename": cad_file.get("filename", doc_id), "content": _json.dumps(cad_file, indent=2, default=str)}

        def _assemble(results, fallback_filename):
            pairs = sorted(
                zip(results["metadatas"], results["documents"]),
                key=lambda x: x[0].get("chunk_index", x[0].get("chunk_id", 0)),
            )
            fname = pairs[0][0].get("filename", fallback_filename) if pairs else fallback_filename
            return fname, "\n\n".join(t for _, t in pairs)

        collection = g.vector_store._get_collection(user_id=user_id, session_id=session_id)
        results = collection.get(where={"document_id": doc_id})
        doc_results = []
        if results and results.get("ids"):
            for meta, text in zip(results.get("metadatas", []), results.get("documents", [])):
                doc_results.append({"metadata": meta, "text": text})
        if doc_results:
            print(f"\u2705 [content] {len(doc_results)} chunks via document_id for user_{user_id or 'anon'}_session_{session_id}")
            fname = doc_results[0]["metadata"].get("filename", doc_id)
            content = "\n\n".join(r["text"] for r in doc_results)
            return {"document_id": doc_id, "filename": fname, "content": content}

        fallback_filename = None
        cached = g.DOCUMENT_FILE_CACHE.get(doc_id)
        if cached:
            fallback_filename = cached.get("filename")
            print(f"\U0001f50d [content] DOCUMENT_FILE_CACHE hit: filename={fallback_filename!r}")

        if not fallback_filename:
            try:
                from neo4j_auth import _run as neo4j_run
                rows = neo4j_run(
                    "MATCH (d:Document {id:$doc_id}) RETURN d.filename AS filename LIMIT 1",
                    {"doc_id": doc_id},
                )
                if rows:
                    fallback_filename = rows[0]["filename"]
                    print(f"\U0001f50d [content] Neo4j filename lookup: {fallback_filename!r}")
            except Exception as ne:
                print(f"\u26a0\ufe0f  [content] Neo4j lookup failed: {ne}")

        if fallback_filename:
            collection = g.vector_store._get_collection(user_id=user_id, session_id=session_id)
            results2 = collection.get(where={"filename": fallback_filename})
            print(f"\U0001f50d [content] by filename={fallback_filename!r} \u2192 {len(results2.get('ids') or [])} chunks")
            if results2 and results2.get("ids"):
                meta0 = results2["metadatas"][0] if results2.get("metadatas") else {}
                stored_sid = meta0.get("session_id", "")
                print(f"\U0001f50d [content] stored session_id={stored_sid!r} requested={session_id!r}")
                if stored_sid == session_id:
                    fname, content = _assemble(results2, fallback_filename)
                    print(f"\u2705 [content] {len(results2['ids'])} chunks via filename for '{fname}'")
                    return {"document_id": doc_id, "filename": fname, "content": content}
                print(f"\u26a0\ufe0f  [content] filename-based session mismatch")

        print(f"\u274c [content] no chunks found anywhere for doc_id={doc_id}")
        raise HTTPException(status_code=404, detail="Document not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"\u274c [content] exception: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Error reading document: {e}")


@router.get("/documents/{doc_id}/download")
async def download_document(doc_id: str, session_id: Optional[str] = None):
    if not session_id:
        raise HTTPException(status_code=403, detail="Session ID required")

    try:
        if g._cad_ifc_available:
            from cad_ifc_agent import CadSharedContext
            if CadSharedContext.get_file(doc_id):
                from neo4j_auth import _run as neo4j_run
                rows = neo4j_run(
                    """
                    MATCH (c:Conversation {session_id: $sid})-[:USED_DOCUMENT]->(d:Document {id: $doc_id})
                    RETURN d.id AS id
                    """,
                    {"sid": session_id, "doc_id": doc_id},
                )
                if not rows:
                    raise HTTPException(status_code=404, detail="Document not found")
                raise HTTPException(
                    status_code=404,
                    detail="CAD/IFC files are streamed from the browser blob URL, not re-downloaded from the server."
                )

        cached_file = g.DOCUMENT_FILE_CACHE.get(doc_id)
        if cached_file and cached_file.get('session_id') == session_id:
            if cached_file.get('content_type') in {'application/pdf', 'image/png', 'image/jpeg', 'image/webp', 'image/gif'}:
                return StreamingResponse(
                    BytesIO(cached_file['bytes']),
                    media_type=cached_file['content_type'],
                    headers={
                        'Content-Disposition': f'inline; filename="{cached_file["filename"]}"',
                    },
                )

        collection = g.vector_store._get_collection(user_id=None, session_id=session_id)
        results = collection.get(where={"document_id": doc_id})
        if not results or not results.get("ids"):
            raise HTTPException(status_code=404, detail="Document not found")

        metadata = results["metadatas"][0] if results.get("metadatas") else {}
        if metadata.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Document not found")

        pairs = sorted(
            zip(results["metadatas"], results["documents"]),
            key=lambda x: x[0].get("chunk_index", 0),
        )
        filename = pairs[0][0].get("filename", doc_id) if pairs else doc_id
        content  = "\n\n".join(text for _, text in pairs)

        return Response(
            content=content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}.txt"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"\u274c Download error: {e}")
        raise HTTPException(500, f"Error downloading document: {e}")


@router.get("/documents/{doc_id}/image")
async def get_image_document(
    doc_id: str,
    session_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    if not session_id:
        raise HTTPException(status_code=403, detail="session_id required")

    cached = g.DOCUMENT_FILE_CACHE.get(doc_id)
    if cached and cached.get("session_id") == session_id and cached.get("is_image"):
        mime = cached.get("content_type") or "image/jpeg"
        return StreamingResponse(
            BytesIO(cached["bytes"]),
            media_type=mime,
            headers={"Content-Disposition": f'inline; filename="{cached["filename"]}"'},
        )

    try:
        from neo4j_auth import _run as neo4j_run
        rows = neo4j_run(
            """
            MATCH (c:Conversation {session_id: $sid})-[:USED_DOCUMENT]->(d:Document {id: $doc_id})
            WHERE d.doc_type IN ['png','jpg','jpeg','webp','gif','bmp','tiff','tif']
            RETURN d.id AS id
            """,
            {"sid": session_id, "doc_id": doc_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Image document not found")
    except HTTPException:
        raise
    except Exception:
        pass

    cached = g.DOCUMENT_FILE_CACHE.get(doc_id)
    if cached and cached.get("is_image"):
        mime = cached.get("content_type") or "image/jpeg"
        return StreamingResponse(
            BytesIO(cached["bytes"]),
            media_type=mime,
            headers={"Content-Disposition": f'inline; filename="{cached["filename"]}"'},
        )

    raise HTTPException(status_code=404, detail="Image bytes not available (server may have restarted)")
