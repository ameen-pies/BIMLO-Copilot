from fastapi import APIRouter, Header, HTTPException
from typing import Optional, Dict, List
import re, json, ast, os, uuid
from datetime import datetime

from models.query import QueryRequest
from core.globals import (
    _session_user_context, _session_routes, _conversation_cache,
    rag_engine, REPORTS_DIR,
)
from core.session_state import get_history, append_turn, get_route_log, log_route
from services.report_agent import SharedContext

router = APIRouter(tags=["reports"])


@router.post("/title")
async def generate_title(request: dict):
    from llm_client import call_llm

    title_type = request.get("type", "chat")
    text       = (request.get("text") or "").strip()
    messages   = request.get("messages", [])

    if title_type == "chat":
        excerpt = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:150]}"
            for m in messages[:6]
        )
        system = (
            "You generate ultra-short conversation titles. "
            "Rules: 3-6 words, Title Case, NO quotes, NO punctuation at the end. "
            "Capture the TOPIC or ACTION — not the literal phrasing of the user. "
            "For greetings in any language, summarise the tone/language, e.g. 'Casual French Greeting', 'Polite Good Morning', 'Friendly Hello'. "
            "For questions, name the subject: 'Network Coverage Question', 'Revenue Chart Request', 'Telecom Site Survey Analysis'. "
            "NEVER echo the user's words back verbatim. NEVER start with 'User' or 'What'. "
            "Reply with ONLY the title, nothing else."
        )
        prompt = f"Conversation:\n{excerpt}\n\nTitle:"
    else:
        system = (
            "You generate short, specific report titles. "
            "Rules: 4-8 words, Title Case, NO quotes, NO punctuation at the end. "
            "Must name the real subject/entity/metric — never start with 'Report', 'Analysis', 'Summary'. "
            "Good: 'Telecom Site Survey Field Results', 'Network Infrastructure Cost Breakdown', 'Q3 Revenue by Region'. "
            "Bad: 'Report on telecom_site_survey.txt', 'Analysis of the document', 'Summary Report'. "
            "Reply with ONLY the title, nothing else."
        )
        prompt = f"Report request: \"{text}\"\n\nTitle:"

    try:
        raw = call_llm(prompt, system_prompt=system, max_tokens=30, temperature=0.3, task="classify")
        title = raw.strip().strip('"').strip("'").strip()
        if not title or len(title) > 100:
            raise ValueError("bad title")
        return {"title": title}
    except Exception as e:
        print(f"\u26a0\ufe0f  /title error: {e}")
        return {"title": ""}


@router.post("/intent/report")
async def report_intent_check(request: dict):
    from llm_client import call_llm

    text = (request.get("text") or "").strip()
    if not text:
        return {"wants_report": False, "explicit_docs": []}

    available_docs: List[str] = request.get("available_docs", [])

    REPORT_RE = re.compile(
        r'\b(?:'
        r'(?:make|create|generate|write|build|produce|prepare|draft|do|give\s+me|get\s+me)'
        r'\s+(?:me\s+)?(?:a\s+|an\s+)?(?:full\s+|detailed\s+|brief\s+)?report'
        r'|report\s+(?:on|about|for|regarding|from)'
        r'|(?:make|create|generate|write|do)\s+(?:me\s+)?(?:a\s+)?(?:summary\s+)?document'
        r'|generate\s+(?:a\s+)?(?:pdf|word\s+doc|summary\s+report)'
        r'|download\s+(?:a\s+)?report'
        r'|rapport\s+sur'
        r'|fais?\s+(?:moi\s+)?(?:un\s+)?rapport'
        r'|cr[eé]e?\s+(?:un\s+)?rapport'
        r'|g[eé]n[eè]re?\s+(?:un\s+)?rapport'
        r')',
        re.IGNORECASE,
    )
    regex_hit = bool(REPORT_RE.search(text))

    def _resolve_docs(mentioned: list) -> list:
        explicit: list = []
        for mention in (mentioned or []):
            ml = mention.lower().strip()
            if not ml:
                continue
            for doc in available_docs:
                dl = doc.lower()
                if ml in dl or dl in ml or re.search(re.escape(ml), dl):
                    if doc not in explicit:
                        explicit.append(doc)
        return explicit

    def _parse_llm_json(raw: str) -> Optional[dict]:
        if not raw:
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            result = json.loads(clean)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        m = re.search(r'\{[\s\S]*?\}', clean)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        try:
            result = ast.literal_eval(clean)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        return None

    try:
        system = (
            "You are an intent and entity classifier for a document Q&A assistant. "
            "Reply with ONLY a raw JSON object — no markdown fences, no explanation whatsoever. "
            'Exact format: {"wants_report": true, "mentioned_files": ["filename.pdf"]} '
            "Set wants_report=true ONLY when the user explicitly asks for a report, document, "
            "summary document, or structured written output to download or read later. "
            "TRUE examples: 'make a report on X', 'generate a report about X', "
            "'create a PDF report', 'write a document summarising Y', 'do a report on file Z'. "
            "FALSE examples: plain questions, 'what is X?', 'show me a chart', 'summarise this' "
            "(quick summary, not a standalone document). "
            "mentioned_files: list any filenames or file references the user typed. "
            "Use partial names too (e.g. 'survey' if they said 'the survey file'). "
            "Return [] if no files mentioned."
        )
        doc_list = ", ".join(available_docs[:15]) if available_docs else "(none uploaded)"
        prompt = (
            f"Available documents: {doc_list}\n\n"
            f'User message: """{text}"""'
        )

        raw_answer = call_llm(
            prompt=prompt,
            system_prompt=system,
            max_tokens=150,
            temperature=0.0,
            task="classify",
        )

        parsed = _parse_llm_json(raw_answer)

        if isinstance(parsed, dict):
            wants = bool(parsed.get("wants_report", False)) or regex_hit
            explicit_docs = _resolve_docs(parsed.get("mentioned_files", []))
            wants_label = "\u2705 YES" if wants else "\u274c NO "
            regex_label = "hit" if regex_hit else "miss"
            print(
                f"\U0001f4cb intent/report \u2192 {wants_label} "
                f"[regex={regex_label}, llm={parsed.get('wants_report')}] "
                f"| files={explicit_docs} | '{text[:60]}'"
            )
            return {"wants_report": wants, "explicit_docs": explicit_docs}

        if regex_hit:
            explicit_docs = _resolve_docs(
                [w for w in re.split(r'[\s,]+', text) if len(w) > 3]
            )
            print(
                f"\U0001f4cb intent/report \u2192 '\u2705 YES' [regex=hit, llm-parse-failed] "
                f"| files={explicit_docs} | '{text[:60]}'"
            )
            return {"wants_report": True, "explicit_docs": explicit_docs}

        wants_from_text = bool(re.search(
            r'\b(?:true|yes|oui|report|document)\b', (raw_answer or ""), re.I
        ))
        wants_label = "\u2705 YES" if wants_from_text else "\u274c NO "
        scan_label = "hit" if wants_from_text else "miss"
        print(
            f"\U0001f4cb intent/report \u2192 {wants_label} "
            f"[regex=miss, llm-parse-failed, text-scan={scan_label}] "
            f"| '{text[:60]}'"
        )
        return {"wants_report": wants_from_text, "explicit_docs": []}

    except Exception as e:
        print(f"\u26a0\ufe0f  intent check error: {e} — regex fallback: {regex_hit}")
        return {"wants_report": regex_hit, "explicit_docs": []}


@router.post("/generate-report")
async def generate_report(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        from neo4j_auth import optional_user
        user    = optional_user(authorization)
        user_id = user["user_id"] if user else None
        if user:
            _session_user_context[session_id] = user["user_id"]

        frontend_conv_id = request.conversation_id or None

        print(f"\U0001f4ca Report: {request.query!r} [session={session_id}]")

        report_data = rag_engine.generate_report(request.query)

        report_id = report_data.get("report_id") or str(uuid.uuid4())
        try:
            from services.report_agent import _reports_store as _rs
            if report_id in _rs and not _rs[report_id].get("session_id"):
                _rs[report_id]["session_id"] = session_id
        except Exception:
            pass

        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.json"
        report_path     = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\u2705 Report saved to disk: {report_filename}")

        content      = report_data.get("content") or report_data.get("answer") or ""
        title        = report_data.get("title")   or request.query[:80]
        source_docs  = ", ".join(report_data.get("source_docs") or []) or "unknown"
        word_count   = len(content.split())
        section_count = len(re.findall(r'^#{1,2}\s', content, re.M))
        preview      = content[:400].strip()

        memory_note = (
            f"[REPORT GENERATED]\n"
            f"Title: {title}\n"
            f"Sources: {source_docs}\n"
            f"Size: {section_count} sections, {word_count} words\n"
            f"Opening: {preview}"
        )

        append_turn(session_id, "user",      request.query,  frontend_conv_id)
        append_turn(session_id, "assistant", memory_note,    frontend_conv_id)
        SharedContext.set_history(session_id, get_history(session_id))

        try:
            from neo4j_auth import _run as neo4j_run
            now = datetime.utcnow().isoformat()

            conv_id = frontend_conv_id or _conversation_cache.get(session_id)
            if not conv_id:
                rows = neo4j_run(
                    "MATCH (c:Conversation {session_id: $sid}) RETURN c.id AS id LIMIT 1",
                    {"sid": session_id},
                )
                conv_id = rows[0]["id"] if rows else None

            if not conv_id:
                conv_id = str(uuid.uuid4())
                _conversation_cache[session_id] = conv_id

            if user_id:
                neo4j_run(
                    """
                    MATCH (u:User {id: $user_id})
                    MERGE (c:Conversation {id: $conv_id})
                    ON CREATE SET c.session_id = $session_id,
                                  c.title      = $title,
                                  c.preview    = $preview,
                                  c.chat_type  = 'rag',
                                  c.created_at = $now,
                                  c.updated_at = $now
                    ON MATCH  SET c.updated_at  = $now
                    MERGE (u)-[:HAS_CONVERSATION]->(c)
                    """,
                    {"user_id": user_id, "conv_id": conv_id,
                     "session_id": session_id, "title": title,
                     "preview": preview[:120], "now": now},
                )
            else:
                neo4j_run(
                    """
                    MERGE (c:Conversation {id: $conv_id})
                    ON CREATE SET c.session_id = $session_id,
                                  c.title      = $title,
                                  c.preview    = $preview,
                                  c.chat_type  = 'rag',
                                  c.created_at = $now,
                                  c.updated_at = $now
                    ON MATCH  SET c.updated_at  = $now
                    """,
                    {"conv_id": conv_id, "session_id": session_id,
                     "title": title, "preview": preview[:120], "now": now},
                )

            neo4j_run(
                """
                MERGE (r:Report {id: $report_id})
                SET r.title        = $title,
                    r.query        = $query,
                    r.source_docs  = $source_docs,
                    r.word_count   = $word_count,
                    r.section_count = $section_count,
                    r.preview      = $preview,
                    r.filename     = $filename,
                    r.session_id   = $session_id,
                    r.created_at   = $now
                WITH r
                MATCH (c:Conversation {id: $conv_id})
                MERGE (c)-[:HAS_REPORT]->(r)
                """,
                {
                    "report_id":     report_id,
                    "title":         title,
                    "query":         request.query[:300],
                    "source_docs":   source_docs,
                    "word_count":    word_count,
                    "section_count": section_count,
                    "preview":       preview[:400],
                    "filename":      report_filename,
                    "session_id":    session_id,
                    "conv_id":       conv_id,
                    "now":           now,
                },
            )
            print(f"\u2601\ufe0f  Neo4j: Report '{title}' \u2192 Conversation {conv_id}")
        except Exception as neo_err:
            print(f"\u26a0\ufe0f  Neo4j report save failed (non-fatal): {neo_err}")

        return {
            "status":      "success",
            "report":      report_data,
            "report_id":   report_id,
            "report_file": report_filename,
            "session_id":  session_id,
            "timestamp":   timestamp,
        }
    except Exception as e:
        print(f"\u274c Report error: {e}")
        raise HTTPException(500, f"Error generating report: {e}")
