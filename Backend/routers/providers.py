"""
providers.py — API endpoint for LLM provider list
──────────────────────────────────────────────────
GET /providers — returns provider list for frontend dropdown.
"""

from fastapi import APIRouter
from services.provider_manager import provider_manager

router = APIRouter()


@router.get("/providers")
async def list_providers():
    return {"providers": provider_manager.get_frontend_providers()}
