from fastapi import APIRouter

from core.session_state import get_history, clear_history

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "messages": history}


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    clear_history(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}
