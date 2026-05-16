"""
neo4j_auth.py â€” User authentication & session persistence via Neo4j

Graph schema:
  (:User {id, email, username, password_hash, created_at, last_seen})
  (:Conversation {id, title, preview, created_at, updated_at, session_id, chat_type})
  (:Message {id, role, content, timestamp, payload})
  (:Document {id, filename, doc_type, uploaded_at, chunk_count})
  (:Report {id, title, query, source_docs, word_count, section_count,
            preview, filename, session_id, created_at})

  (:User)-[:HAS_CONVERSATION]->(:Conversation)
  (:Conversation)-[:CONTAINS {index}]->(:Message)
  (:Conversation)-[:HAS_REPORT]->(:Report)
  (:User)-[:UPLOADED]->(:Document)
  (:Conversation)-[:USED_DOCUMENT]->(:Document)

Mount in main.py:
    from neo4j_auth import router as auth_router, init_neo4j
    app.include_router(auth_router)
    # and in startup call init_neo4j() ONCE
"""

import os
import uuid
import hashlib
import secrets
import time
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from cachetools import LRUCache

from fastapi import APIRouter, HTTPException, Depends, Header, Cookie, Response, Request
from core.deps import limiter
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()  # must run before os.getenv() calls below

from core.config import settings

import logging
logger = logging.getLogger("auth")
# Suppress Neo4j notification spam (INFO/WARNING about missing props/relations
# on empty databases â€” harmless, just noisy during first-run)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

try:
    from neo4j import GraphDatabase, exceptions as neo4j_exc
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    print("âš ï¸  neo4j driver not installed â€” run: pip install neo4j")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIG  â€” set these in your .env file
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#  LOCAL Neo4j Desktop / Community Edition:
#    NEO4J_URI      = bolt://127.0.0.1:7687     â† use bolt://, NOT neo4j://
#    NEO4J_DATABASE = neo4j                      â† Community only has ONE db
#
#  Neo4j AuraDB (cloud free tier):
#    NEO4J_URI      = neo4j+s://<your-id>.databases.neo4j.io
#    NEO4J_DATABASE = neo4j                      â† AuraDB free also uses "neo4j"
#
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

NEO4J_URI      = settings.neo4j_uri
NEO4J_USER     = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
# âš ï¸  IMPORTANT: Community Edition ONLY supports the default "neo4j" database.
# Named databases (like "users") require Enterprise or AuraDB Pro.
# On Community, keep this as "neo4j" and use labels to separate data.
NEO4J_DATABASE = settings.neo4j_database

print(f"ðŸ” NEO4J DEBUG â€” URI={NEO4J_URI} USER={NEO4J_USER} PASS={NEO4J_PASSWORD[:3]}***")

# Token TTL
TOKEN_TTL_HOURS = 72
ACCESS_TOKEN_TTL_MINUTES = 15
REFRESH_TOKEN_TTL_DAYS = 7

# â”€â”€ In-memory cache (fast path) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tokens are ALSO persisted to Neo4j as (:Token) nodes so they survive restarts.
# On a cache miss we hit Neo4j once, then warm the cache.
# Bounded LRU: prevents unbounded memory growth from stale tokens.
_active_tokens: Dict[str, Dict] = LRUCache(maxsize=5000)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NEO4J DRIVER SINGLETON
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        if not _NEO4J_AVAILABLE:
            raise RuntimeError("neo4j package not installed â€” run: pip install neo4j")
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # Connection pool settings â€” keeps things snappy
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
        print(f"âŒ Neo4j query error: {err_msg}")
        print(f"   Query: {cypher[:120]}")
        raise HTTPException(503, f"Database error: {err_msg}")


def _setup_constraints():
    """Create uniqueness constraints on first startup."""
    constraints = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        "CREATE CONSTRAINT conv_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT news_conv_id IF NOT EXISTS FOR (c:NewsConversation) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT msg_id IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT token_val IF NOT EXISTS FOR (t:Token) REQUIRE t.token IS UNIQUE",
        "CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE",
    ]
    for c in constraints:
        try:
            _run(c)
        except HTTPException:
            pass  # constraint may already exist â€” safe to ignore
    print("âœ… Neo4j: constraints ready")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _hash_password(password: str) -> str:
    """Hash a password with bcrypt (work factor 12)."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12)
    ).decode("utf-8")


def _verify_password(password: str, stored: str) -> bool:
    """
    Verify a password against the stored string.
    Automatically handles legacy SHA-256 hashes (format "salt:digest")
    as well as bcrypt hashes.
    """
    try:
        # Compatibility with existing SHA-256 hashes (migration path)
        if ":" in stored and len(stored.split(":")[0]) == 32:
            salt, digest = stored.split(":", 1)
            return hashlib.sha256((salt + password).encode()).hexdigest() == digest
        # Modern bcrypt verification
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return False


def _issue_token(user_id: str, username: str, email: str, role: str = "user",
                 token_type: str = "access", ttl_seconds: Optional[int] = None) -> str:
    if ttl_seconds is None:
        ttl_seconds = ACCESS_TOKEN_TTL_MINUTES * 60 if token_type == "access" else REFRESH_TOKEN_TTL_DAYS * 86400
    token      = secrets.token_urlsafe(32)
    expires_at = time.time() + ttl_seconds
    payload    = {"user_id": user_id, "username": username, "email": email,
                  "expires_at": expires_at, "role": role, "type": token_type}
    _active_tokens[token] = payload
    try:
        _run(
            """
            MATCH (u:User {id: $user_id})
            CREATE (t:Token {
                token:      $token,
                expires_at: $expires_at,
                type:       $token_type,
                created_at: $now
            })
            MERGE (u)-[:HAS_TOKEN]->(t)
            """,
            {"user_id": user_id, "token": token,
             "expires_at": expires_at, "token_type": token_type,
             "now": datetime.utcnow().isoformat()},
        )
    except Exception as e:
        print(f"âš ï¸  Token persist failed (auth still works this session): {e}")
    return token


def _issue_token_pair(user_id: str, username: str, email: str, role: str = "user") -> tuple:
    """Issue both access (15 min) and refresh (7 day) tokens, return (access, refresh)."""
    access  = _issue_token(user_id, username, email, role, token_type="access")
    refresh = _issue_token(user_id, username, email, role, token_type="refresh",
                           ttl_seconds=REFRESH_TOKEN_TTL_DAYS * 86400)
    return access, refresh


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

    # â”€â”€ Fast path: in-memory cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Cold path: Neo4j lookup (server just restarted) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        rows = _run(
            """
            MATCH (u:User)-[:HAS_TOKEN]->(t:Token {token: $token})
            RETURN u.id AS user_id, u.username AS username, u.email AS email,
                   u.role AS role, t.expires_at AS expires_at, t.type AS type
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
            "role":       row.get("role") or "user",
            "type":       row.get("type") or "access",
        }
        _active_tokens[token] = payload
        return payload
    except Exception as e:
        print(f"âš ï¸  Token Neo4j lookup failed: {e}")
        return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DEPENDENCY: optional auth (doesn't raise â€” returns None for guests)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def optional_user(
    authorization: Optional[str] = Header(default=None),
    bimlo_token: Optional[str] = Cookie(default=None),
) -> Optional[Dict]:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    elif isinstance(bimlo_token, str) and bimlo_token:
        token = bimlo_token
    if not token:
        return None
    result = _resolve_token(token)
    logger.debug("auth: %s", "ok" if result else "rejected")
    return result


def require_user(
    authorization: Optional[str] = Header(default=None),
    bimlo_token: Optional[str] = Cookie(default=None),
) -> Dict:
    user = optional_user(authorization, bimlo_token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PYDANTIC MODELS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SignupRequest(BaseModel):
    email:    str
    username: str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class AuthResponse(BaseModel):
    token:        str
    user_id:      str
    username:     str
    email:        str
    avatar_url:   str = ""
    display_name: str = ""
    role:         str = "user"

class SaveConversationRequest(BaseModel):
    conversation_id: str
    session_id:      str
    title:           str
    preview:         str
    messages:        List[Dict[str, Any]]
    doc_ids:         List[str] = []
    chat_type:       str = "rag"   # "rag" | "news"


class SaveNewsChatRequest(BaseModel):
    conversation_id: str
    session_id:      str
    title:           str
    preview:         str
    messages:        List[Dict[str, Any]]

class SaveDocumentRequest(BaseModel):
    doc_id:      str
    filename:    str
    doc_type:    str = "unknown"
    chunk_count: int = 0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ROUTER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse)
@limiter.limit("3/minute")
def signup(request: Request, req: SignupRequest, response: Response):
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
            role:          'user',
            created_at:    $now,
            last_seen:     $now
        })
        """,
        {"id": user_id, "email": email, "username": username,
         "password_hash": password_hash, "now": now},
    )

    access_token, refresh_token = _issue_token_pair(user_id, username, email, role="user")
    response.set_cookie(
        key="bimlo_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=REFRESH_TOKEN_TTL_DAYS * 86400,
    )
    print(f"âœ… auth: new user '{username}' ({email})")
    return AuthResponse(token=access_token, user_id=user_id, username=username, email=email, role="user")


@router.post("/login", response_model=AuthResponse)
@limiter.limit("5/minute")
def login(request: Request, req: LoginRequest, response: Response):
    email = req.email.strip().lower()

    rows = _run(
        "MATCH (u:User {email: $email}) RETURN u.id AS id, u.username AS username, u.password_hash AS ph, coalesce(u.role, 'user') AS role",
        {"email": email},
    )
    if not rows:
        raise HTTPException(401, "Invalid email or password")

    row = rows[0]
    if not _verify_password(req.password, row["ph"]):
        raise HTTPException(401, "Invalid email or password")

    # â”€â”€ Migration: if password is still stored as SHA-256, re-hash with bcrypt â”€â”€
    stored_hash = row["ph"]
    if ":" in stored_hash and len(stored_hash.split(":")[0]) == 32:
        new_hash = _hash_password(req.password)
        _run(
            "MATCH (u:User {id: $id}) SET u.password_hash = $h",
            {"id": row["id"], "h": new_hash},
        )

    # Update last_seen
    _run(
        "MATCH (u:User {id: $id}) SET u.last_seen = $now",
        {"id": row["id"], "now": datetime.utcnow().isoformat()},
    )

    role = row.get("role") or "user"
    access_token, refresh_token = _issue_token_pair(row["id"], row["username"], email, role=role)
    response.set_cookie(
        key="bimlo_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=REFRESH_TOKEN_TTL_DAYS * 86400,
    )
    print(f"âœ… auth: login '{row['username']}' (role={role})")
    return AuthResponse(token=access_token, user_id=row["id"], username=row["username"], email=email, role=role)


@router.post("/logout")
def logout(
    response: Response,
    user: Dict = Depends(require_user),
    authorization: Optional[str] = Header(default=None),
    bimlo_token: Optional[str] = Cookie(default=None),
):
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    elif isinstance(bimlo_token, str) and bimlo_token:
        token = bimlo_token
    if token:
        _revoke_token(token)
    response.delete_cookie("bimlo_token")
    return {"ok": True}


@router.post("/refresh")
def refresh_token(
    response: Response,
    bimlo_token: Optional[str] = Cookie(default=None),
):
    if not bimlo_token:
        raise HTTPException(401, "No refresh token provided")
    payload = _resolve_token(bimlo_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid refresh token")
    _revoke_token(bimlo_token)
    new_access, new_refresh = _issue_token_pair(
        payload["user_id"], payload["username"],
        payload["email"], payload.get("role", "user"),
    )
    response.set_cookie(
        key="bimlo_token",
        value=new_refresh,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=REFRESH_TOKEN_TTL_DAYS * 86400,
    )
    return {"access_token": new_access, "token_type": "bearer"}


@router.delete("/delete-account")
def delete_account(
    response: Response,
    user: Dict = Depends(require_user),
    authorization: Optional[str] = Header(default=None),
    bimlo_token: Optional[str] = Cookie(default=None),
):
    """Permanently delete the authenticated user and all their data."""
    user_id = user["user_id"]
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    elif isinstance(bimlo_token, str) and bimlo_token:
        token = bimlo_token
    if token:
        _revoke_token(token)
    response.delete_cookie("bimlo_token")
    _run(
        """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[:HAS_TOKEN]->(t:Token)
        OPTIONAL MATCH (u)-[:HAS_CONVERSATION]->(c:Conversation)
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Message)
        OPTIONAL MATCH (u)-[:UPLOADED]->(d:Document)
        DETACH DELETE u, t, c, m, d
        """,
        {"user_id": user_id},
    )
    print(f"ðŸ—‘ï¸  auth: deleted account user_id={user_id}")
    return {"ok": True}


class GoogleTokenRequest(BaseModel):
    access_token: str
    email:        str
    name:         str
    sub:          str        # Google user ID
    picture:      str = ""  # Google profile picture URL


@router.post("/google-token", response_model=AuthResponse)
def google_token_auth(req: GoogleTokenRequest, response: Response):
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
        "MATCH (u:User {email: $email}) RETURN u.id AS id, u.username AS username, coalesce(u.role, 'user') AS role",
        {"email": email},
    )

    if rows:
        user_id  = rows[0]["id"]
        username = rows[0]["username"]
        role     = rows[0].get("role") or "user"   # â† preserve existing role (e.g. "admin")
        _run("MATCH (u:User {id: $id}) SET u.last_seen = $now, u.picture = $picture, u.display_name = $display_name",
             {"id": user_id, "now": now, "picture": req.picture, "display_name": req.name})
        print(f"âœ… auth/google-token: existing user '{username}' ({email}) role={role}")
    else:
        user_id = str(uuid.uuid4())
        role    = "user"
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
                picture: $picture,
                display_name: $display_name,
                role: 'user',
                created_at: $now, last_seen: $now
            })
            """,
            {"id": user_id, "email": email, "username": username,
             "picture": req.picture, "display_name": req.name, "now": now},
        )
        print(f"âœ… auth/google-token: new user '{username}' ({email})")

    access_token, refresh_token = _issue_token_pair(user_id, username, email, role=role)
    response.set_cookie(
        key="bimlo_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=REFRESH_TOKEN_TTL_DAYS * 86400,
    )
    return AuthResponse(token=access_token, user_id=user_id, username=username, email=email, avatar_url=req.picture, display_name=req.name, role=role)


@router.get("/me")
def me(user: Dict = Depends(require_user)):
    now = datetime.utcnow().isoformat()
    rows = _run(
        """
        MATCH (u:User {id: $id})
        SET u.last_seen = $now
        RETURN u.username AS username, u.email AS email, u.created_at AS created_at,
               coalesce(u.role, 'user') AS role,
               coalesce(u.picture, u.avatar_url, '') AS avatar_url
        """,
        {"id": user["user_id"], "now": now},
    )
    if not rows:
        raise HTTPException(404, "User not found")
    return {**rows[0], "user_id": user["user_id"], "token": ""}


@router.post("/heartbeat")
def heartbeat(user: Dict = Depends(require_user)):
    """Lightweight ping to keep last_seen fresh. Called every 60s from the frontend."""
    now = datetime.utcnow().isoformat()
    _run(
        "MATCH (u:User {id: $id}) SET u.last_seen = $now",
        {"id": user["user_id"], "now": now},
    )
    return {"ok": True}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PUBLIC CONTACT FORM
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ContactRequest(BaseModel):
    name:    str
    email:   str
    subject: str
    message: str

@router.post("/contact")
def contact(req: ContactRequest):
    """
    Public contact form â€” no auth required.
    Forwards the message to BIMLO's inbox via SMTP.
    Uses the same SMTP env vars as the admin email sender.
    """
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.utils import formataddr

    smtp_host  = settings.smtp_host
    smtp_port  = settings.smtp_port
    smtp_user  = settings.smtp_user
    smtp_pass  = settings.smtp_pass
    smtp_from  = settings.smtp_from or smtp_user or "noreply@bimlo.local"
    bimlo_inbox = settings.contact_to or smtp_user or smtp_from

    body_text = (
        f"Name:    {req.name}\n"
        f"Email:   {req.email}\n"
        f"Subject: {req.subject}\n\n"
        f"{req.message}"
    )

    body_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f1117;padding:32px 16px;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;">
        <tr>
          <td style="background:#161b27;border-radius:14px 14px 0 0;padding:24px 28px;border-bottom:1px solid #1e2535;">
            <div style="font-size:18px;font-weight:800;color:#f1f5f9;letter-spacing:-0.01em;">New Contact Message</div>
            <div style="font-size:11px;color:#64748b;margin-top:3px;">via BIMLO Copilot contact form</div>
          </td>
        </tr>
        <tr>
          <td style="background:#161b27;padding:24px 28px;">
            <table cellpadding="0" cellspacing="0" width="100%">
              <tr><td style="padding-bottom:12px;">
                <span style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">From</span><br>
                <span style="font-size:14px;color:#f1f5f9;">{req.name}</span>
                <span style="font-size:13px;color:#64748b;margin-left:8px;">&lt;{req.email}&gt;</span>
              </td></tr>
              <tr><td style="padding-bottom:20px;border-bottom:1px solid #1e2535;">
                <span style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Subject</span><br>
                <span style="font-size:14px;color:#f1f5f9;">{req.subject}</span>
              </td></tr>
              <tr><td style="padding-top:20px;">
                <span style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.05em;">Message</span><br>
                <div style="font-size:14px;line-height:1.75;color:#cbd5e1;margin-top:8px;">{req.message.replace(chr(10), '<br>')}</div>
              </td></tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="background:#0d1117;border-radius:0 0 14px 14px;padding:16px 28px;border-top:1px solid #1e2535;">
            <div style="font-size:11px;color:#334155;">Reply directly to {req.email} to respond.</div>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    try:
        if smtp_host and smtp_user:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Contact] {req.subject}"
            msg["From"]    = formataddr(("BIMLO Copilot", smtp_from))
            msg["To"]      = bimlo_inbox
            msg["Reply-To"] = req.email
            msg.attach(MIMEText(body_text, "plain"))
            msg.attach(MIMEText(body_html, "html"))
            ctx = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, bimlo_inbox, msg.as_string())
            print(f"ðŸ“§ contact: message from {req.email} delivered to {bimlo_inbox}")
        else:
            print(f"ðŸ“§ [SMTP not configured] Contact from {req.email}: {req.subject}")
    except Exception as e:
        print(f"âš ï¸  contact: failed to send: {e}")
        raise HTTPException(500, "Failed to send message. Please try again later.")

    return {"ok": True}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONVERSATION PERSISTENCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/conversations/save")
def save_conversation(req: SaveConversationRequest, user: Dict = Depends(require_user)):
    """
    Upsert conversation + messages.
    Each message payload (sources, analytics, reportId, thinkingSteps, etc.)
    is JSON-serialised into Message.payload so nothing is lost.
    """
    import json as _json
    now = datetime.utcnow().isoformat()

    msgs_data = []
    for idx, msg in enumerate(req.messages):
        # Collect every field that isn't the core text fields into the payload
        payload_dict = {k: v for k, v in msg.items()
                        if k not in ("id", "role", "content", "timestamp")}
        msgs_data.append({
            "msg_id":  msg.get("id") or str(uuid.uuid4()),
            "role":    msg.get("role", "user"),
            "content": msg.get("content", ""),
            "ts":      str(msg.get("timestamp", now)),
            "idx":     idx,
            "payload": _json.dumps(payload_dict, default=str) if payload_dict else "",
        })

    def _work(tx):
        # Upsert conversation node first (separate from relationship to avoid
        # concurrent MERGE race with uniqueness constraints)
        tx.run(
            """
            MERGE (c:Conversation {id: $conv_id})
            SET   c.title      = $title,
                  c.preview    = $preview,
                  c.session_id = $session_id,
                  c.chat_type  = $chat_type,
                  c.updated_at = $now,
                  c.created_at = CASE WHEN c.created_at IS NULL THEN $now ELSE c.created_at END
            """,
            conv_id=req.conversation_id,
            title=req.title or "",
            preview=req.preview or "",
            session_id=req.session_id or "",
            chat_type=req.chat_type,
            now=now,
        )
        # Link to user
        tx.run(
            """
            MATCH (u:User {id: $user_id})
            MATCH (c:Conversation {id: $conv_id})
            MERGE (u)-[:HAS_CONVERSATION]->(c)
            """,
            conv_id=req.conversation_id,
            user_id=user["user_id"],
        )
        # Wipe old messages (full replace on every save)
        tx.run(
            "MATCH (:Conversation {id: $conv_id})-[:CONTAINS]->(m:Message) DETACH DELETE m",
            conv_id=req.conversation_id,
        )
        # Bulk insert messages with rich payload
        if msgs_data:
            tx.run(
                """
                MATCH (c:Conversation {id: $conv_id})
                UNWIND $msgs AS m
                CREATE (msg:Message {
                    id:        m.msg_id,
                    role:      m.role,
                    content:   m.content,
                    timestamp: m.ts,
                    payload:   m.payload
                })
                CREATE (c)-[:CONTAINS {index: m.idx}]->(msg)
                """,
                conv_id=req.conversation_id,
                msgs=msgs_data,
            )

    import time as _time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            driver = get_driver()
            with driver.session(database=NEO4J_DATABASE) as session:
                session.execute_write(_work)
            print(f"✅ save_conversation for user={user['user_id']} conv={req.conversation_id} messages={len(msgs_data)} session_id={req.session_id}")
            return {"ok": True, "conversation_id": req.conversation_id}
        except Exception as e:
            err_str = str(e)
            if "ConstraintValidationFailed" in err_str and attempt < max_retries - 1:
                _time.sleep(0.1 * (attempt + 1))
                continue
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
    """Return a single conversation with all its messages + rich payload."""
    import json as _json
    conv = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation {id: $conv_id})
        RETURN c.id AS id, c.title AS title, c.session_id AS session_id,
               c.chat_type AS chat_type
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    if not conv:
        raise HTTPException(404, "Conversation not found")

    messages = _run(
        """
        MATCH (:Conversation {id: $conv_id})-[r:CONTAINS]->(m:Message)
        RETURN m.id AS id, m.role AS role, m.content AS content,
               m.timestamp AS timestamp, m.payload AS payload
        ORDER BY r.index ASC
        """,
        {"conv_id": conversation_id},
    )
    print(f"ðŸ”„ get_conversation for user={user['user_id']} conv={conversation_id} found_messages={len(messages)}")

    enriched = []
    for m in messages:
        msg = dict(m)
        raw = msg.pop("payload", "") or ""
        try:
            msg["payload"] = _json.loads(raw) if raw else {}
        except Exception:
            msg["payload"] = {}
        enriched.append(msg)

    return {**conv[0], "messages": enriched}


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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DOCUMENT ASSOCIATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NEWS CHAT CONVERSATION PERSISTENCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/news-conversations/save")
def save_news_conversation(req: SaveNewsChatRequest, user: Dict = Depends(require_user)):
    """
    Persist a news chat conversation to Neo4j as a NewsConversation node.
    Messages include full payload (pinnedArticles, etc.).
    """
    import json as _json
    now = datetime.utcnow().isoformat()

    msgs_data = [
        {
            "msg_id":  msg.get("id") or str(uuid.uuid4()),
            "role":    msg.get("role", "user"),
            "content": msg.get("content", ""),
            "ts":      str(msg.get("timestamp", now)),
            "idx":     idx,
            "payload": _json.dumps(
                {k: v for k, v in msg.items()
                 if k not in ("id", "role", "content", "timestamp")},
                default=str,
            ),
        }
        for idx, msg in enumerate(req.messages)
    ]

    def _work(tx):
        tx.run(
            """
            MERGE (c:NewsConversation {id: $conv_id})
            SET   c.title      = $title,
                  c.preview    = $preview,
                  c.session_id = $session_id,
                  c.updated_at = $now,
                  c.created_at = CASE WHEN c.created_at IS NULL THEN $now ELSE c.created_at END
            """,
            conv_id=req.conversation_id,
            title=req.title or "News Chat",
            preview=req.preview or "",
            session_id=req.session_id or "",
            now=now,
        )
        tx.run(
            """
            MATCH (u:User {id: $user_id})
            MATCH (c:NewsConversation {id: $conv_id})
            MERGE (u)-[:HAS_NEWS_CONVERSATION]->(c)
            """,
            conv_id=req.conversation_id,
            user_id=user["user_id"],
        )
        tx.run(
            "MATCH (:NewsConversation {id: $conv_id})-[:CONTAINS]->(m:Message) DETACH DELETE m",
            conv_id=req.conversation_id,
        )
        if msgs_data:
            tx.run(
                """
                MATCH (c:NewsConversation {id: $conv_id})
                UNWIND $msgs AS m
                CREATE (msg:Message {
                    id:        m.msg_id,
                    role:      m.role,
                    content:   m.content,
                    timestamp: m.ts,
                    payload:   m.payload
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
        print(f"âŒ save_news_conversation error: {e}")
        raise HTTPException(503, f"Database error: {e}")


@router.get("/news-conversations")
def list_news_conversations(user: Dict = Depends(require_user)):
    """Return all news chat conversations for the user, newest first."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_NEWS_CONVERSATION]->(c:NewsConversation)
        RETURN c.id AS id, c.title AS title, c.preview AS preview,
               c.created_at AS created_at, c.updated_at AS updated_at,
               c.session_id AS session_id
        ORDER BY c.updated_at DESC
        """,
        {"user_id": user["user_id"]},
    )
    return rows


@router.get("/news-conversations/{conversation_id}")
def get_news_conversation(conversation_id: str, user: Dict = Depends(require_user)):
    """Return a single news chat with all messages and their payloads."""
    import json as _json
    conv = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_NEWS_CONVERSATION]->(c:NewsConversation {id: $conv_id})
        RETURN c.id AS id, c.title AS title, c.session_id AS session_id
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    if not conv:
        raise HTTPException(404, "News conversation not found")

    messages = _run(
        """
        MATCH (:NewsConversation {id: $conv_id})-[r:CONTAINS]->(m:Message)
        RETURN m.id AS id, m.role AS role, m.content AS content,
               m.timestamp AS timestamp, m.payload AS payload
        ORDER BY r.index ASC
        """,
        {"conv_id": conversation_id},
    )
    enriched = []
    for m in messages:
        msg = dict(m)
        raw = msg.pop("payload", "") or ""
        try:
            msg["payload"] = _json.loads(raw) if raw else {}
        except Exception:
            msg["payload"] = {}
        enriched.append(msg)

    return {**conv[0], "messages": enriched}


@router.delete("/news-conversations/{conversation_id}")
def delete_news_conversation(conversation_id: str, user: Dict = Depends(require_user)):
    """Delete a news chat conversation and all its messages."""
    _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_NEWS_CONVERSATION]->(c:NewsConversation {id: $conv_id})
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Message)
        DETACH DELETE c, m
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    return {"ok": True}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# REPORT PERSISTENCE
#
# Graph additions:
#   (:Report {id, title, query, source_docs, word_count, section_count,
#             preview, filename?, session_id, created_at})
#   (:Conversation)-[:HAS_REPORT]->(:Report)
#
# Reports are scoped to a Conversation so they always belong to the chat
# that triggered them. Users can list, load, and delete their reports.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/conversations/{conversation_id}/reports")
def list_reports_for_conversation(
    conversation_id: str,
    user: Dict = Depends(require_user),
):
    """Return all reports generated inside a specific conversation."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation {id: $conv_id})
        MATCH (c)-[:HAS_REPORT]->(r:Report)
        RETURN r.id           AS id,
               r.title        AS title,
               r.query        AS query,
               r.source_docs  AS source_docs,
               r.word_count   AS word_count,
               r.section_count AS section_count,
               r.preview      AS preview,
               r.filename     AS filename,
               r.created_at   AS created_at
        ORDER BY r.created_at DESC
        """,
        {"user_id": user["user_id"], "conv_id": conversation_id},
    )
    return rows


@router.get("/reports")
def list_all_reports(user: Dict = Depends(require_user)):
    """Return every report across all conversations for the user, newest first."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
        MATCH (c)-[:HAS_REPORT]->(r:Report)
        RETURN r.id            AS id,
               r.title         AS title,
               r.query         AS query,
               r.source_docs   AS source_docs,
               r.word_count    AS word_count,
               r.section_count AS section_count,
               r.preview       AS preview,
               r.filename      AS filename,
               r.session_id    AS session_id,
               r.created_at    AS created_at,
               c.id            AS conversation_id,
               c.title         AS conversation_title
        ORDER BY r.created_at DESC
        """,
        {"user_id": user["user_id"]},
    )
    return rows


@router.get("/reports/{report_id}")
def get_report(report_id: str, user: Dict = Depends(require_user)):
    """Return a single report â€” only if it belongs to a conversation owned by the user."""
    rows = _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
        MATCH (c)-[:HAS_REPORT]->(r:Report {id: $report_id})
        RETURN r.id            AS id,
               r.title         AS title,
               r.query         AS query,
               r.source_docs   AS source_docs,
               r.word_count    AS word_count,
               r.section_count AS section_count,
               r.preview       AS preview,
               r.filename      AS filename,
               r.session_id    AS session_id,
               r.created_at    AS created_at,
               c.id            AS conversation_id
        """,
        {"user_id": user["user_id"], "report_id": report_id},
    )
    if not rows:
        raise HTTPException(404, "Report not found")
    return rows[0]


@router.delete("/reports/{report_id}")
def delete_report(report_id: str, user: Dict = Depends(require_user)):
    """Delete a report node (does NOT delete the conversation or messages)."""
    _run(
        """
        MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
        MATCH (c)-[:HAS_REPORT]->(r:Report {id: $report_id})
        DETACH DELETE r
        """,
        {"user_id": user["user_id"], "report_id": report_id},
    )
    return {"ok": True}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STARTUP HOOK â€” call from main.py ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def init_neo4j():
    """Call this once at app startup to verify connection and create constraints.
    Retries up to 15 times with 10s delay to handle slow Neo4j startup in Docker.
    """
    if not _NEO4J_AVAILABLE:
        print("âš ï¸  neo4j driver missing â€” run: pip install neo4j")
        print("    Auth endpoints will return 503 until the package is installed.")
        return

    max_retries = 15
    retry_delay = 10  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            global _driver
            _driver = None  # reset driver so we get a fresh connection each retry
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
                    print(f"ðŸ§¹ Neo4j: purged {n} expired token(s)")
            except Exception as e:
                print(f"âš ï¸  Token cleanup failed (non-fatal): {e}")
            print(f"âœ… Neo4j connected â€” {NEO4J_URI} / db:{NEO4J_DATABASE}")
            _seed_admin()
            return  # success â€” exit retry loop
        except Exception as e:
            if attempt < max_retries:
                print(f"â³ Neo4j not ready (attempt {attempt}/{max_retries}) â€” retrying in {retry_delay}s... ({e})")
                time.sleep(retry_delay)
            else:
                print(f"âŒ Neo4j connection FAILED after {max_retries} attempts: {e}")
                print("â”€" * 60)
                print("  Checklist:")
                print("  1. Is Neo4j running? (neo4j start  or  check Desktop)")
                print(f"  2. Is the URI correct? Current: {NEO4J_URI}")
                print(f"     â€¢ Local:  bolt://127.0.0.1:7687")
                print(f"     â€¢ AuraDB: neo4j+s://<id>.databases.neo4j.io")
                print(f"  3. Is NEO4J_PASSWORD set correctly in .env?")
                print(f"  4. Community Edition? Set NEO4J_DATABASE=neo4j in .env")
                print("â”€" * 60)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ADMIN: seed default admin account + helper
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _seed_admin() -> None:
    """Create the default admin account (admin / admin) if it doesn't exist."""
    try:
        existing = _run(
            "MATCH (u:User {email: 'admin@bimlo.local'}) RETURN u.id AS id LIMIT 1"
        )
        if not existing:
            admin_id = str(uuid.uuid4())
            pw_hash  = _hash_password("admin")
            now      = datetime.utcnow().isoformat()
            _run(
                """
                CREATE (u:User {
                    id:            $id,
                    email:         'admin@bimlo.local',
                    username:      'admin',
                    password_hash: $ph,
                    role:          'admin',
                    created_at:    $now,
                    last_seen:     $now
                })
                """,
                {"id": admin_id, "ph": pw_hash, "now": now},
            )
            print("âœ… admin account seeded (email: admin@bimlo.local, password: admin)")
        else:
            # Ensure existing admin has role=admin
            _run(
                "MATCH (u:User {email: 'admin@bimlo.local'}) SET u.role = 'admin'",
            )
    except Exception as e:
        print(f"âš ï¸  Admin seed failed (non-fatal): {e}")


def require_admin(
    authorization: Optional[str] = Header(default=None),
    bimlo_token: Optional[str] = Cookie(default=None),
) -> Dict:
    user = optional_user(authorization, bimlo_token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ADMIN ENDPOINTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AdminUpdateUserRequest(BaseModel):
    username:  Optional[str] = None
    email:     Optional[str] = None
    password:  Optional[str] = None
    role:      Optional[str] = None   # "user" | "admin"


class AdminSendEmailRequest(BaseModel):
    user_ids: List[str]
    subject:  str
    body:     str


@router.get("/admin/users")
def admin_list_users(admin: Dict = Depends(require_admin)):
    """List all users with stats."""
    rows = _run(
        """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:HAS_CONVERSATION]->(c:Conversation)
        OPTIONAL MATCH (u)-[:UPLOADED]->(d:Document)
        RETURN u.id          AS user_id,
               u.username    AS username,
               u.email       AS email,
               coalesce(u.role, 'user') AS role,
               u.created_at  AS created_at,
               u.last_seen   AS last_seen,
               coalesce(u.picture, u.avatar_url, '') AS avatar_url,
               count(DISTINCT c) AS conversation_count,
               count(DISTINCT d) AS document_count
        ORDER BY u.created_at DESC
        """
    )
    return [
        {
            "user_id":            r["user_id"],
            "username":           r["username"],
            "email":              r["email"],
            "role":               r["role"],
            "created_at":         r["created_at"],
            "last_seen":          r["last_seen"],
            "avatar_url":         r.get("avatar_url") or "",
            "conversation_count": r["conversation_count"] or 0,
            "document_count":     r["document_count"] or 0,
        }
        for r in rows
    ]


@router.get("/admin/stats")
def admin_stats(admin: Dict = Depends(require_admin)):
    """KPI stats for the dashboard."""
    users = _run(
        """
        MATCH (u:User)
        RETURN count(u) AS total,
               count(CASE WHEN u.role = 'admin' THEN 1 END) AS admins,
               count(CASE WHEN datetime(u.last_seen) > datetime() - duration({hours: 1}) THEN 1 END) AS active_1h,
               count(CASE WHEN datetime(u.last_seen) > datetime() - duration({days: 1}) THEN 1 END) AS active_24h,
               count(CASE WHEN datetime(u.created_at) > datetime() - duration({days: 7}) THEN 1 END) AS new_7d
        """
    )
    convs = _run("MATCH (c:Conversation) RETURN count(c) AS total")
    docs  = _run("MATCH (d:Document) RETURN count(d) AS total")
    rpts  = _run("MATCH (r:Report) RETURN count(r) AS total")
    u = users[0] if users else {}
    return {
        "total_users":    u.get("total", 0),
        "admin_users":    u.get("admins", 0),
        "active_1h":      u.get("active_1h", 0),
        "active_24h":     u.get("active_24h", 0),
        "new_users_7d":   u.get("new_7d", 0),
        "total_conversations": convs[0]["total"] if convs else 0,
        "total_documents":     docs[0]["total"] if docs else 0,
        "total_reports":       rpts[0]["total"] if rpts else 0,
    }


@router.patch("/admin/users/{user_id}")
def admin_update_user(
    user_id: str,
    req: AdminUpdateUserRequest,
    admin: Dict = Depends(require_admin),
):
    """Update any user's credentials or role."""
    rows = _run("MATCH (u:User {id: $id}) RETURN u.id AS id", {"id": user_id})
    if not rows:
        raise HTTPException(404, "User not found")

    sets = []
    params: Dict[str, Any] = {"id": user_id}
    if req.username:
        sets.append("u.username = $username"); params["username"] = req.username.strip()
    if req.email:
        sets.append("u.email = $email"); params["email"] = req.email.strip().lower()
    if req.password:
        sets.append("u.password_hash = $ph"); params["ph"] = _hash_password(req.password)
    if req.role in ("user", "admin"):
        sets.append("u.role = $role"); params["role"] = req.role
    if not sets:
        raise HTTPException(400, "Nothing to update")

    _run(f"MATCH (u:User {{id: $id}}) SET {', '.join(sets)}", params)

    # Invalidate all active tokens for this user so next login picks up new creds
    token_rows = _run(
        "MATCH (u:User {id: $id})-[:HAS_TOKEN]->(t:Token) RETURN t.token AS token",
        {"id": user_id},
    )
    for tr in token_rows:
        _active_tokens.pop(tr["token"], None)
    _run("MATCH (u:User {id: $id})-[:HAS_TOKEN]->(t:Token) DETACH DELETE t", {"id": user_id})

    updated = _run(
        "MATCH (u:User {id: $id}) RETURN u.username AS username, u.email AS email, coalesce(u.role, 'user') AS role",
        {"id": user_id},
    )
    print(f"âœï¸  admin: updated user {user_id}")
    return updated[0] if updated else {"ok": True}


@router.delete("/admin/users/{user_id}")
def admin_delete_user(user_id: str, admin: Dict = Depends(require_admin)):
    """Permanently delete a user and all their data."""
    if user_id == admin["user_id"]:
        raise HTTPException(400, "Cannot delete your own account via admin panel")
    rows = _run("MATCH (u:User {id: $id}) RETURN u.id AS id", {"id": user_id})
    if not rows:
        raise HTTPException(404, "User not found")

    # Revoke tokens from memory cache
    token_rows = _run(
        "MATCH (u:User {id: $id})-[:HAS_TOKEN]->(t:Token) RETURN t.token AS token",
        {"id": user_id},
    )
    for tr in token_rows:
        _active_tokens.pop(tr["token"], None)

    _run(
        """
        MATCH (u:User {id: $id})
        OPTIONAL MATCH (u)-[:HAS_TOKEN]->(t:Token)
        OPTIONAL MATCH (u)-[:HAS_CONVERSATION]->(c:Conversation)
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Message)
        OPTIONAL MATCH (u)-[:UPLOADED]->(d:Document)
        DETACH DELETE u, t, c, m, d
        """,
        {"id": user_id},
    )
    print(f"ðŸ—‘ï¸  admin: deleted user {user_id}")
    return {"ok": True, "deleted_user_id": user_id}


@router.post("/admin/send-email")
def admin_send_email(req: AdminSendEmailRequest, admin: Dict = Depends(require_admin)):
    """
    Send an email to selected users.
    Uses SMTP settings from env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM.
    Falls back to logging if SMTP is not configured.
    """
    import smtplib, ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = settings.smtp_host
    smtp_port = settings.smtp_port
    smtp_user = settings.smtp_user
    smtp_pass = settings.smtp_pass
    smtp_from = settings.smtp_from or smtp_user or "noreply@bimlo.local"

    # Fetch emails for target user_ids
    rows = _run(
        "MATCH (u:User) WHERE u.id IN $ids RETURN u.email AS email, u.username AS username",
        {"ids": req.user_ids},
    )
    if not rows:
        raise HTTPException(404, "No users found for the given IDs")

    # Styled HTML email template with BIMLO Copilot branding
    def build_html(body_text: str, username: str) -> str:
        escaped = body_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0f1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f1117;padding:32px 16px;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;">
        <!-- Header -->
        <tr>
          <td style="background:#161b27;border-radius:14px 14px 0 0;padding:24px 28px;border-bottom:1px solid #1e2535;">
            <div style="font-size:18px;font-weight:800;color:#f1f5f9;letter-spacing:-0.01em;">BIMLO Copilot</div>
            <div style="font-size:11px;color:#64748b;margin-top:3px;">AI-Powered BIM Assistant</div>
          </td>
        </tr>
        <!-- Body -->
        <tr>
          <td style="background:#161b27;padding:28px 28px 24px;">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:16px;">Hi {username},</div>
            <div style="font-size:14px;line-height:1.75;color:#cbd5e1;">{escaped}</div>
          </td>
        </tr>
        <!-- Footer -->
        <tr>
          <td style="background:#0d1117;border-radius:0 0 14px 14px;padding:16px 28px;border-top:1px solid #1e2535;">
            <div style="font-size:11px;color:#334155;">
              Sent by BIMLO Copilot admin &middot; You're receiving this because you have an account with us.
            </div>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    from email.utils import formataddr

    sent, failed = [], []
    for r in rows:
        to_email = r["email"]
        username = r.get("username", "there")
        try:
            if smtp_host and smtp_user:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = req.subject
                msg["From"]    = formataddr(("BIMLO Copilot", smtp_from))
                msg["To"]      = to_email
                msg.attach(MIMEText(req.body, "plain"))
                msg.attach(MIMEText(build_html(req.body, username), "html"))
                ctx = ssl.create_default_context()
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.ehlo()
                    server.starttls(context=ctx)
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(smtp_from, to_email, msg.as_string())
                sent.append(to_email)
                print(f"ðŸ“§ admin: sent email to {to_email}")
            else:
                # SMTP not configured â€” log instead
                print(f"ðŸ“§ [SMTP not configured] Would send to {to_email}: {req.subject}")
                sent.append(to_email)  # treat as sent for UI purposes
        except Exception as e:
            print(f"âš ï¸  admin: email to {to_email} failed: {e}")
            failed.append(to_email)

    return {"sent": sent, "failed": failed}


@router.get("/admin/logs")
def admin_logs(admin: Dict = Depends(require_admin), limit: int = 200):
    """
    Return recent system log entries collected in memory.
    The /admin/logs/stream endpoint (SSE) is the live version.
    """
    return {"logs": list(_log_buffer)[-limit:]}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# IN-MEMORY LOG BUFFER (captured from stdout for admin panel)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import sys
import threading
from collections import deque
from datetime import datetime as _dt

_log_buffer: deque = deque(maxlen=500)
_log_subscribers: list = []   # SSE response queues
_log_lock = threading.Lock()


class _LogCapture(object):
    """Tee: write to original stdout AND push to _log_buffer + SSE subscribers."""
    def __init__(self, original):
        self._orig = original

    def write(self, text):
        self._orig.write(text)
        stripped = text.rstrip("\n")
        if stripped:
            entry = {"ts": _dt.utcnow().isoformat(), "msg": stripped}
            with _log_lock:
                _log_buffer.append(entry)
                dead = []
                for q in _log_subscribers:
                    try:
                        q.put_nowait(entry)
                    except Exception:
                        dead.append(q)
                for d in dead:
                    _log_subscribers.remove(d)

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# Install once
if not isinstance(sys.stdout, _LogCapture):
    sys.stdout = _LogCapture(sys.stdout)


@router.get("/admin/logs/stream")
async def admin_logs_stream(admin: Dict = Depends(require_admin)):
    """SSE stream of live log lines for the admin dashboard."""
    import asyncio, queue as _queue
    from fastapi.responses import StreamingResponse as _SR

    q: _queue.Queue = _queue.Queue(maxsize=200)
    with _log_lock:
        _log_subscribers.append(q)
        # Flush last 50 buffered lines immediately so the panel isn't empty
        for entry in list(_log_buffer)[-50:]:
            try:
                q.put_nowait(entry)
            except Exception:
                pass

    async def _gen():
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    entry = await loop.run_in_executor(None, lambda: q.get(timeout=30))
                    yield f"data: {json.dumps(entry)}\n\n"
                except Exception:
                    yield "data: {}\n\n"  # keepalive ping
        finally:
            with _log_lock:
                try:
                    _log_subscribers.remove(q)
                except ValueError:
                    pass

    return _SR(_gen(), media_type="text/event-stream",
               headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@router.get("/admin/pipeline-logs")
def admin_pipeline_logs(
    limit: int = 300,
    event: str = None,          # optional filter: routing|judge|retrieval|â€¦
    user: dict = Depends(require_admin),
):
    """
    Return the last `limit` structured pipeline log entries from
    observability.py's JSONL file.
 
    Query params:
      limit  â€” max entries to return (default 300, max 2000)
      event  â€” filter by event type (routing, judge, retrieval, ingestion,
                query_end, alert, latency)
 
    Response:
      { "entries": [ {...}, â€¦ ], "total": N, "log_file": "â€¦" }
    """
    try:
        from observability import obs
        limit = min(max(1, limit), 2000)
        entries = obs.tail_logs(n=limit, event_filter=event or None)
        return {
            "entries":  entries,
            "total":    len(entries),
            "log_file": str(obs.LOG_FILE) if hasattr(obs, "LOG_FILE") else "unknown",
        }
    except ImportError:
        raise HTTPException(503, "observability module not available â€” ensure observability.py is in the Python path")
    except Exception as e:
        raise HTTPException(500, f"Failed to read pipeline logs: {e}")
 
 
@router.get("/admin/pipeline-logs/stats")
def admin_pipeline_logs_stats(user: dict = Depends(require_admin)):
    """
    Return aggregate statistics computed from the full pipeline log file.
 
    Response:
      {
        "event_counts":    { "routing": N, "judge": N, â€¦ },
        "judge_pass":      N,
        "judge_fail":      N,
        "judge_pass_rate": 0.xx | null,
        "log_file":        "/path/to/bimlo_structured.jsonl"
      }
    """
    try:
        from observability import obs
        stats = obs.get_stats()
        return stats
    except ImportError:
        raise HTTPException(503, "observability module not available")
    except Exception as e:
        raise HTTPException(500, f"Failed to compute pipeline stats: {e}")
