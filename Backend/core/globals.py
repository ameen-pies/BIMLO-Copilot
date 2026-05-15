from collections import deque
from cachetools import TTLCache, LRUCache
from typing import Dict, List, Optional, Any
import os, threading
from core.config import settings

MAX_UPLOAD_SIZE = int(settings.max_upload_mb) * 1024 * 1024
MAX_HISTORY_TURNS = 20

DOCUMENT_FILE_CACHE = TTLCache(maxsize=500, ttl=1800)

_sessions = LRUCache(maxsize=2000)
_session_routes = LRUCache(maxsize=2000)
_session_route_log = LRUCache(maxsize=2000)
_session_user_context = LRUCache(maxsize=2000)
_conversation_cache = LRUCache(maxsize=2000)
_cancel_events: Dict[str, threading.Event] = {}
_session_file_stack: Dict[str, List[str]] = {}

doc_processor = None
vector_store = None
rag_engine = None
run_ingestion_pipeline = None
_guess_mime_type_func = None

_cad_ifc_available = False
_ingestion_graph_available = False
_news_pipeline_available = False
_news_chat_available = False

DATA_DIR = settings.data_dir
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


def _guess_mime_type(filename: str) -> str:
    import mimetypes
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    ext = os.path.splitext(filename)[1].lower()
    return {
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc':  'application/msword',
        '.txt':  'text/plain',
        '.ifc':  'application/octet-stream',
        '.ifczip': 'application/octet-stream',
        '.dxf':  'application/octet-stream',
        '.dwg':  'application/octet-stream',
        '.step': 'application/octet-stream',
        '.stp':  'application/octet-stream',
    }.get(ext, 'application/octet-stream')
