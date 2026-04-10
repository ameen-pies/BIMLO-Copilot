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
    from neo4j_auth import router as auth_router
    app.include_router(auth_router)
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
from pydantic import BaseModel, EmailStr

try:
    from neo4j import GraphDatabase, exceptions as neo4j_exc
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    print("⚠️  neo4j driver not installed — run: pip install neo4j")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "users")

# Simple in-memory token store: token -> {user_id, expires_at, username, email}
# For production swap with Redis or a proper JWT flow.
_active_tokens: Dict[str, Dict] = {}
TOKEN_TTL_HOURS = 72


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J DRIVER SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError("neo4j package not installed")
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def _run(cypher: str, params: dict = None, database: str = NEO4J_DATABASE):
    """Execute a Cypher query and return a list of record dicts."""
    driver = get_driver()
    with driver.session(database=database) as session:
        result = session.run(cypher, params or {})
        return [dict(r) for r in result]


def _setup_constraints():
    """Create uniqueness constraints on first startup."""
    constraints = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        "CREATE CONSTRAINT conv_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT msg_id IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    ]
    for c in constraints:
        try:
            _run(c)
        except Exception:
            pass  # constraint may already exist
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
    token = secrets.token_urlsafe(32)
    _active_tokens[token] = {
        "user_id":    user_id,
        "username":   username,
        "email":      email,
        "expires_at": time.time() + TOKEN_TTL_HOURS * 3600,
    }
    return token


def _revoke_token(token: str):
    _active_tokens.pop(token, None)


def _resolve_token(token: str) -> Optional[Dict]:
    """Return token payload if valid and not expired, else None."""
    payload = _active_tokens.get(token)
    if not payload:
        return None
    if time.time() > payload["expires_at"]:
        _active_tokens.pop(token, None)
        return None
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY: optional auth (doesn't raise — returns None for guests)
# ─────────────────────────────────────────────────────────────────────────────

def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[Dict]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    return _resolve_token(token)


def require_user(authorization: Optional[str] = Header(default=None)) -> Dict:
    user = optional_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
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
    """Upsert a full conversation (create or update by conversation_id)."""
    now = datetime.utcnow().isoformat()

    # Upsert Conversation node
    _run(
        """
        MERGE (c:Conversation {id: $conv_id})
        SET   c.title      = $title,
              c.preview    = $preview,
              c.session_id = $session_id,
              c.updated_at = $now
        ON CREATE SET c.created_at = $now
        WITH c
        MATCH (u:User {id: $user_id})
        MERGE (u)-[:HAS_CONVERSATION]->(c)
        """,
        {
            "conv_id":    req.conversation_id,
            "title":      req.title,
            "preview":    req.preview,
            "session_id": req.session_id,
            "now":        now,
            "user_id":    user["user_id"],
        },
    )

    # Delete old messages and rewrite (simpler than diffing)
    _run(
        """
        MATCH (:Conversation {id: $conv_id})-[:CONTAINS]->(m:Message)
        DETACH DELETE m
        """,
        {"conv_id": req.conversation_id},
    )

    # Insert messages
    for idx, msg in enumerate(req.messages):
        _run(
            """
            MATCH (c:Conversation {id: $conv_id})
            CREATE (m:Message {
                id:        $msg_id,
                role:      $role,
                content:   $content,
                timestamp: $ts
            })
            CREATE (c)-[:CONTAINS {index: $idx}]->(m)
            """,
            {
                "conv_id": req.conversation_id,
                "msg_id":  msg.get("id", str(uuid.uuid4())),
                "role":    msg.get("role", "user"),
                "content": msg.get("content", ""),
                "ts":      msg.get("timestamp", now),
                "idx":     idx,
            },
        )

    return {"ok": True, "conversation_id": req.conversation_id}


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
    # Verify ownership
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
# STARTUP HOOK — call from main.py
# ─────────────────────────────────────────────────────────────────────────────

def init_neo4j():
    """Call this once at app startup."""
    if not _NEO4J_AVAILABLE:
        print("⚠️  neo4j driver missing — auth endpoints will 500")
        return
    try:
        get_driver().verify_connectivity()
        _setup_constraints()
        print(f"✅ Neo4j connected — {NEO4J_URI} / db:{NEO4J_DATABASE}")
    except Exception as e:
        print(f"⚠️  Neo4j connection failed: {e}")
        print("    Auth routes registered but will return 500 until Neo4j is reachable.")
