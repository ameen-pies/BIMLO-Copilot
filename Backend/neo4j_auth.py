"""
neo4j_auth.py — User authentication & session persistence via Neo4j

Graph schema:
  (:User {id, email, username, password_hash, created_at, last_seen})
  (:Conversation {id, title, preview, created_at, updated_at, session_id})
  (:Message {id, role, content, timestamp})
  (:Document {id, filename, doc_type, uploaded_at, chunk_count})

  (:User)-[:HAS_CONVERSATION]->(:Conversation)
  (:Conversation)-[:CONTAINS {index}]->(:Message)
  (:User)-[:UPLOADED]->(:Document)
  (:Conversation)-[:USED_DOCUMENT]->(:Document)

Mount in main.py:
    from neo4j_auth import router as auth_router, init_neo4j
    app.include_router(auth_router)
    # and in startup call init_neo4j() ONCE
"""

from __future__ import annotations

import os
import uuid
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()  # must run before os.getenv() calls below

import logging
# Suppress Neo4j notification spam (INFO/WARNING about missing props/relations
# on empty databases — harmless, just noisy during first-run)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

try:
    from neo4j import GraphDatabase, exceptions as neo4j_exc
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    print("⚠️  neo4j driver not installed — run: pip install neo4j")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — set these in your .env file
# ─────────────────────────────────────────────────────────────────────────────
#
#  LOCAL Neo4j Desktop / Community Edition:
#    NEO4J_URI      = bolt://127.0.0.1:7687     ← use bolt://, NOT neo4j://
#    NEO4J_DATABASE = neo4j                      ← Community only has ONE db
#
#  Neo4j AuraDB (cloud free tier):
#    NEO4J_URI      = neo4j+s://<your-id>.databases.neo4j.io
#    NEO4J_DATABASE = neo4j                      ← AuraDB free also uses "neo4j"
#
# ─────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# ⚠️  IMPORTANT: Community Edition ONLY supports the default "neo4j" database.
# Named databases (like "users") require Enterprise or AuraDB Pro.
# On Community, keep this as "neo4j" and use labels to separate data.
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print(f"🔍 NEO4J DEBUG — URI={NEO4J_URI} USER={NEO4J_USER} PASS={NEO4J_PASSWORD[:3]}***")

# Token TTL
TOKEN_TTL_HOURS = 72

# ── In-memory cache (fast path) ───────────────────────────────────────────────
# Tokens are ALSO persisted to Neo4j as (:Token) nodes so they survive restarts.
# On a cache miss we hit Neo4j once, then warm the cache.
_active_tokens: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J DRIVER SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError("neo4j package not installed — run: pip install neo4j")
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # Connection pool settings — keeps things snappy
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
        )
    return _driver


def _run(cypher: str, params: dict = None, database: str = NEO4J_DATABASE):
    """Execute a Cypher query and return a list of record dicts.
    Raises HTTPException(503) if Neo4j is unreachable so the API responds
    with a clear error instead of hanging forever.
    """
    try:
        driver = get_driver()
        with driver.session(database=database) as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]
    except RuntimeError as e:
        raise HTTPException(503, f"Database driver error: {e}")
    except Exception as e:
        # Surface the real Neo4j error so you can debug it
        err_msg = str(e)
        print(f"❌ Neo4j query error: {err_msg}")
        print(f"   Query: {cypher[:120]}")
        raise HTTPException(503, f"Database error: {err_msg}")


def _setup_constraints():
    """Create uniqueness constraints on first startup."""
    constraints = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        "CREATE CONSTRAINT conv_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT msg_id IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT token_val IF NOT EXISTS FOR (t:Token) REQUIRE t.token IS UNIQUE",
    ]
    for c in constraints:
        try:
            _run(c)
        except HTTPException:
            pass  # constraint may already exist — safe to ignore
    print("✅ Neo4j: constraints ready")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == digest
    except Exception:
        return False


def _issue_token(user_id: str, username: str, email: str) -> str:
    token      = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL_HOURS * 3600
    payload    = {"user_id": user_id, "username": username, "email": email, "expires_at": expires_at}
    # Warm in-memory cache
    _active_tokens[token] = payload
    # Persist to Neo4j so the token survives server restarts
    try:
        _run(
            """
            MATCH (u:User {id: $user_id})
            CREATE (t:Token {
                token:      $token,
                expires_at: $expires_at,
                created_at: $now
            })
            MERGE (u)-[:HAS_TOKEN]->(t)
            """,
            {"user_id": user_id, "token": token,
             "expires_at": expires_at, "now": datetime.utcnow().isoformat()},
        )
    except Exception as e:
        print(f"⚠️  Token persist failed (auth still works this session): {e}")
    return token


def _revoke_token(token: str):
    _active_tokens.pop(token, None)
    try:
        _run("MATCH (t:Token {token: $token}) DETACH DELETE t", {"token": token})
    except Exception:
        pass


def _resolve_token(token: str) -> Optional[Dict]:
    """Return token payload if valid and not expired, else None.
    Checks in-memory cache first; falls back to Neo4j on a cache miss
    (e.g. after a server restart) so existing sessions stay valid.
    """
    now = time.time()

    # ── Fast path: in-memory cache ────────────────────────────────────────
    payload = _active_tokens.get(token)
    if payload:
        if now > payload["expires_at"]:
            _active_tokens.pop(token, None)
            try:
                _run("MATCH (t:Token {token: $token}) DETACH DELETE t", {"token": token})
            except Exception:
                pass
            return None
        return payload

    # ── Cold path: Neo4j lookup (server just restarted) ───────────────────
    try:
        rows = _run(
            """
            MATCH (u:User)-[:HAS_TOKEN]->(t:Token {token: $token})
            RETURN u.id AS user_id, u.username AS username, u.email AS email,
                   t.expires_at AS expires_at
            """,
            {"token": token},
        )
        if not rows:
            return None
        row = rows[0]
        if now > row["expires_at"]:
            _run("MATCH (t:Token {token: $token}) DETACH DELETE t", {"token": token})
            return None
        # Warm the cache so subsequent requests skip Neo4j
        payload = {
            "user_id":    row["user_id"],
            "username":   row["username"],
            "email":      row["email"],
            "expires_at": row["expires_at"],
        }
        _active_tokens[token] = payload
        return payload
    except Exception as e:
        print(f"⚠️  Token Neo4j lookup failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY: optional auth (doesn't raise — returns None for guests)
# ─────────────────────────────────────────────────────────────────────────────

def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[Dict]:
    print(f"🔐 AUTH — header: {repr(authorization)[:80]}")
    if not authorization or not authorization.startswith("Bearer "):
        print(f"   → REJECTED: no/missing Bearer header")
        return None
    token = authorization.split(" ", 1)[1]
    result = _resolve_token(token)
    print(f"   → token={token[:16]}... in_memory={token in _active_tokens} resolved={result is not None}")
    return result


def require_user(authorization: Optional[str] = Header(default=None)) -> Dict:
    user = optional_user(authorization)
    if not user:
        print(f"❌ 401 — header was: {repr(authorization)[:80]}")
        raise HTTPException(status_code=401, detail="Authentication required")
    print(f"✅ AUTH OK — user={user.get('username')}")
    return user


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email:    str
    username: str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class AuthResponse(BaseModel):
    token:    str
    user_id:  str
    username: str
    email:    str

class SaveConversationRequest(BaseModel):
    conversation_id: str
    session_id:      str
    title:           str
    preview:         str
    messages:        List[Dict[str, Any]]
    doc_ids:         List[str] = []

class SaveDocumentRequest(BaseModel):
    doc_id:      str
    filename:    str
    doc_type:    str = "unknown"
    chunk_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    email    = req.email.strip().lower()
    username = req.username.strip()
    password = req.password

    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address")

    # Check email uniqueness
    existing = _run(
        "MATCH (u:User {email: $email}) RETURN u.id AS id LIMIT 1",
        {"email": email},
    )
    if existing:
        raise HTTPException(409, "Email already registered")

    user_id       = str(uuid.uuid4())
    password_hash = _hash_password(password)
    now           = datetime.utcnow().isoformat()

    _run(
        """
        CREATE (u:User {
            id:            $id,
            email:         $email,
            username:      $username,
            password_hash: $password_hash,
            created_at:    $now,
            last_seen:     $now
        })
        """,
        {"id": user_id, "email": email, "username": username,
         "password_hash": password_hash, "now": now},
    )

    token = _issue_token(user_id, username, email)
    print(f"✅ auth: new user '{username}' ({email})")
    return AuthResponse(token=token, user_id=user_id, username=username, email=email)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    email = req.email.strip().lower()

    rows = _run(
        "MATCH (u:User {email: $email}) RETURN u.id AS id, u.username AS username, u.password_hash AS ph",
        {"email": email},
    )
    if not rows:
        raise HTTPException(401, "Invalid email or password")

    row = rows[0]
    if not _verify_password(req.password, row["ph"]):
        raise HTTPException(401, "Invalid email or password")

    # Update last_seen
    _run(
        "MATCH (u:User {id: $id}) SET u.last_seen = $now",
        {"id": row["id"], "now": datetime.utcnow().isoformat()},
    )

    token = _issue_token(row["id"], row["username"], email)
    print(f"✅ auth: login '{row['username']}'")
    return AuthResponse(token=token, user_id=row["id"], username=row["username"], email=email)


@router.post("/logout")
def logout(user: Dict = Depends(require_user), authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        _revoke_token(authorization.split(" ", 1)[1])
    return {"ok": True}


class GoogleTokenRequest(BaseModel):
    access_token: str
    email:        str
    name:         str
    sub:          str   # Google user ID


@router.post("/google-token", response_model=AuthResponse)
def google_token_auth(req: GoogleTokenRequest):
    """
    Accept a Google OAuth2 access token + basic user info from the frontend.
    Verifies the token is valid by hitting Google's tokeninfo endpoint,
    then find-or-create the user in Neo4j.
    """
    import urllib.request, json as _json

    # Verify the access token is genuine
    try:
        url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={req.access_token}"
        with urllib.request.urlopen(url, timeout=8) as r:
            info = _json.loads(r.read())
        if info.get("error_description"):
            raise HTTPException(401, f"Invalid Google token: {info['error_description']}")
        # Make sure the email matches what the frontend sent
        if info.get("email", "").lower() != req.email.strip().lower():
            raise HTTPException(401, "Token email mismatch")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, f"Google token verification failed: {e}")

    email    = req.email.strip().lower()
    now      = datetime.utcnow().isoformat()
    name     = req.name or email.split("@")[0]
    username = "".join(c for c in name if c.isalnum() or c in "_-")[:30] or "user"

    rows = _run(
        "MATCH (u:User {email: $email}) RETURN u.id AS id, u.username AS username",
        {"email": email},
    )

    if rows:
        user_id  = rows[0]["id"]
        username = rows[0]["username"]
        _run("MATCH (u:User {id: $id}) SET u.last_seen = $now", {"id": user_id, "now": now})
        print(f"✅ auth/google-token: existing user '{username}' ({email})")
    else:
        user_id = str(uuid.uuid4())
        base = username
        suffix = 0
        while True:
            check = _run("MATCH (u:User {username: $u}) RETURN u.id LIMIT 1", {"u": username})
            if not check:
                break
            suffix += 1
            username = f"{base}{suffix}"

        _run(
            """
            CREATE (u:User {
                id: $id, email: $email, username: $username,
                password_hash: '', google_auth: true,
                created_at: $now, last_seen: $now
            })
            """,
            {"id": user_id, "email": email, "username": username, "now": now},
        )
        print(f"✅ auth/google-token: new user '{username}' ({email})")

    token = _issue_token(user_id, username, email)
    return AuthResponse(token=token, user_id=user_id, username=username, email=email)


@router.get("/me")
def me(user: Dict = Depends(require_user)):
    rows = _run(
        "MATCH (u:User {id: $id}) RETURN u.username AS username, u.email AS email, u.created_at AS created_at",
        {"id": user["user_id"]},
    )
    if not rows:
        raise HTTPException(404, "User not found")
    return {**rows[0], "user_id": user["user_id"]}


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/conversations/save")
def save_conversation(req: SaveConversationRequest, user: Dict = Depends(require_user)):
    """Upsert conversation + messages using execute_write (neo4j driver 5.x compatible)."""
    now = datetime.utcnow().isoformat()

    msgs_data = [
        {
            "msg_id":  msg.get("id") or str(uuid.uuid4()),
            "role":    msg.get("role", "user"),
            "content": msg.get("content", ""),
            "ts":      str(msg.get("timestamp", now)),
            "idx":     idx,
        }
        for idx, msg in enumerate(req.messages)
    ]

    def _work(tx):
        # Upsert conversation + link to user
        tx.run(
            """
            MATCH (u:User {id: $user_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c:Conversation {id: $conv_id})
            SET   c.title      = $title,
                  c.preview    = $preview,
                  c.session_id = $session_id,
                  c.updated_at = $now,
                  c.created_at = CASE WHEN c.created_at IS NULL THEN $now ELSE c.created_at END
            """,
            conv_id=req.conversation_id,
            title=req.title or "",
            preview=req.preview or "",
            session_id=req.session_id or "",
            now=now,
            user_id=user["user_id"],
        )
        # Wipe old messages
        tx.run(
            "MATCH (:Conversation {id: $conv_id})-[:CONTAINS]->(m:Message) DETACH DELETE m",
            conv_id=req.conversation_id,
        )
        # Bulk insert new messages
        if msgs_data:
            tx.run(
                """
                MATCH (c:Conversation {id: $conv_id})
                UNWIND $msgs AS m
                CREATE (msg:Message {
                    id:        m.msg_id,
                    role:      m.role,
                    content:   m.content,
                    timestamp: m.ts
                })
                CREATE (c)-[:CONTAINS {index: m.idx}]->(msg)
                """,
                conv_id=req.conversation_id,
                msgs=msgs_data,
            )

    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(_work)
        return {"ok": True, "conversation_id": req.conversation_id}
    except Exception as e:
        print(f"❌ save_conversation error: {e}")
        raise HTTPException(503, f"Database error: {e}")


@router.get("/conversations")
def list_conversations(user: Dict = Depends(require_user)):
    """Return all conversations for the logged-in user, newest first."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
        RETURN c.id AS id, c.title AS title, c.preview AS preview,
               c.created_at AS created_at, c.updated_at AS updated_at,
               c.session_id AS session_id
        ORDER BY c.updated_at DESC
        """,
        {"user_id": user["user_id"]},
    )
    return rows


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: Dict = Depends(require_user)):
    """Return a single conversation with all its messages."""
    conv = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation {id: $conv_id})
        RETURN c.id AS id, c.title AS title, c.session_id AS session_id
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    if not conv:
        raise HTTPException(404, "Conversation not found")

    messages = _run(
        """
        MATCH (:Conversation {id: $conv_id})-[r:CONTAINS]->(m:Message)
        RETURN m.id AS id, m.role AS role, m.content AS content, m.timestamp AS timestamp
        ORDER BY r.index ASC
        """,
        {"conv_id": conversation_id},
    )
    return {**conv[0], "messages": messages}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: Dict = Depends(require_user)):
    """Delete a conversation and all its messages."""
    _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation {id: $conv_id})
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Message)
        DETACH DELETE c, m
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT ASSOCIATION
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/documents/save")
def save_document(req: SaveDocumentRequest, user: Dict = Depends(require_user)):
    """Record an uploaded document against the user's account."""
    now = datetime.utcnow().isoformat()
    _run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d.filename    = $filename,
            d.doc_type    = $doc_type,
            d.chunk_count = $chunk_count,
            d.uploaded_at = $now
        WITH d
        MATCH (u:User {id: $user_id})
        MERGE (u)-[:UPLOADED]->(d)
        """,
        {
            "doc_id":      req.doc_id,
            "filename":    req.filename,
            "doc_type":    req.doc_type,
            "chunk_count": req.chunk_count,
            "now":         now,
            "user_id":     user["user_id"],
        },
    )
    return {"ok": True}


@router.get("/documents")
def list_documents(user: Dict = Depends(require_user)):
    """Return all documents the user has uploaded."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:UPLOADED]->(d:Document)
        RETURN d.id AS id, d.filename AS filename, d.doc_type AS doc_type,
               d.chunk_count AS chunk_count, d.uploaded_at AS uploaded_at
        ORDER BY d.uploaded_at DESC
        """,
        {"user_id": user["user_id"]},
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP HOOK — call from main.py ONCE
# ─────────────────────────────────────────────────────────────────────────────

def init_neo4j():
    """Call this once at app startup to verify connection and create constraints."""
    if not _NEO4J_AVAILABLE:
        print("⚠️  neo4j driver missing — run: pip install neo4j")
        print("    Auth endpoints will return 503 until the package is installed.")
        return
    try:
        get_driver().verify_connectivity()
        _setup_constraints()
        # Clean up expired tokens left over from previous runs
        try:
            now = time.time()
            result = _run(
                "MATCH (t:Token) WHERE t.expires_at < $now DETACH DELETE t RETURN count(t) AS n",
                {"now": now},
            )
            n = result[0]["n"] if result else 0
            if n:
                print(f"🧹 Neo4j: purged {n} expired token(s)")
        except Exception as e:
            print(f"⚠️  Token cleanup failed (non-fatal): {e}")
        print(f"✅ Neo4j connected — {NEO4J_URI} / db:{NEO4J_DATABASE}")
    except Exception as e:
        print(f"❌ Neo4j connection FAILED: {e}")
        print("─" * 60)
        print("  Checklist:")
        print("  1. Is Neo4j running? (neo4j start  or  check Desktop)")
        print(f"  2. Is the URI correct? Current: {NEO4J_URI}")
        print(f"     • Local:  bolt://127.0.0.1:7687")
        print(f"     • AuraDB: neo4j+s://<id>.databases.neo4j.io")
        print(f"  3. Is NEO4J_PASSWORD set correctly in .env?")
        print(f"  4. Community Edition? Set NEO4J_DATABASE=neo4j in .env")
        print("─" * 60)