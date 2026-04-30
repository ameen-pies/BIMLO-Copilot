"""
auth.py — Authentication façade for BIMLO Copilot
══════════════════════════════════════════════════════════════════════════════
The report references "auth.py" as the authentication module that exposes
/auth/register, /auth/login, /auth/logout, and the JWT + RBAC layer.

All of that logic lives inside neo4j_auth.py.
This file is the thin shim that:
  1. Re-exports everything the rest of the codebase needs from neo4j_auth
  2. Keeps backward compat if anything imports from "auth" directly
  3. Makes the project structure match the report exactly

PLACE THIS FILE AT: backend/auth.py   (same directory as neo4j_auth.py)

Nothing here needs to change — just keep neo4j_auth.py up to date.
══════════════════════════════════════════════════════════════════════════════
"""

from neo4j_auth import (
    # FastAPI router (mount this in main.py — already done via auth_router)
    router,

    # Neo4j lifecycle
    init_neo4j,
    get_driver,
    _run,

    # Auth helpers (used by other services)
    require_user,
    require_admin,
    optional_user,

    # Token utilities
    _issue_token,
    _revoke_token,
    _resolve_token,

    # Pydantic models (imported by tests and other modules)
    AuthResponse,
    SignupRequest,
    LoginRequest,
    SaveConversationRequest,
    SaveDocumentRequest,

    # In-memory token cache (tests read this to verify token state)
    _active_tokens,
)

__all__ = [
    "router",
    "init_neo4j",
    "get_driver",
    "_run",
    "require_user",
    "require_admin",
    "optional_user",
    "_issue_token",
    "_revoke_token",
    "_resolve_token",
    "AuthResponse",
    "SignupRequest",
    "LoginRequest",
    "SaveConversationRequest",
    "SaveDocumentRequest",
    "_active_tokens",
]
