from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import re, logging

from core.globals import _news_pipeline_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/meta")
async def news_meta():
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    from news_pipeline import get_meta
    meta = get_meta()
    if meta is None:
        raise HTTPException(503, "No news cache available yet. The pipeline may still be running.")
    return meta


@router.get("/pages/{page_num}")
async def news_page(page_num: int):
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    if page_num < 0:
        raise HTTPException(400, "page_num must be >= 0")

    from news_pipeline import get_page, get_meta

    data = get_page(page_num)
    if data is None:
        return JSONResponse(
            status_code=404,
            content={"error": "page_not_found", "has_more": False, "page": page_num},
            headers={"Cache-Control": "no-store"},
        )

    meta = get_meta()
    total_pages = meta.get("total_pages", 0) if meta else 0
    data["has_more"] = (page_num + 1) < total_pages

    return JSONResponse(
        content=data,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/status")
async def news_status():
    if not _news_pipeline_available:
        return {"running": False, "status": "unavailable"}
    from news_pipeline import get_status as _pipeline_get_status
    return _pipeline_get_status()


@router.post("/trigger")
async def news_trigger(background_tasks: BackgroundTasks, force: bool = False):
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    from news_pipeline import pipeline_is_running, run_news_pipeline
    if pipeline_is_running():
        return {"status": "already_running", "message": "Pipeline is already in progress."}
    background_tasks.add_task(run_news_pipeline, force)
    return {"status": "accepted", "message": "Pipeline triggered. Poll /api/news/status for progress."}


@router.post("/refresh")
async def news_refresh(background_tasks: BackgroundTasks):
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    from news_pipeline import pipeline_is_running, run_news_pipeline
    if pipeline_is_running():
        return {"status": "already_running"}
    background_tasks.add_task(run_news_pipeline, True)
    return {"status": "accepted", "message": "Force refresh triggered."}


@router.get("/soft-refresh")
async def news_soft_refresh():
    if not _news_pipeline_available:
        raise HTTPException(503, "News pipeline not available.")
    from news_pipeline import get_meta
    meta = get_meta()
    if meta is None:
        raise HTTPException(503, "No news cache available yet.")
    return meta


@router.get("/thumb")
async def news_thumb(url: str):
    if not url or url == "#":
        return {"image_url": None}

    from news_pipeline import get_meta, get_page

    if _news_pipeline_available:
        try:
            meta = get_meta()
            if meta:
                for page_num in range(meta.get("total_pages", 0)):
                    page = get_page(page_num)
                    if not page:
                        continue
                    for item in page.get("items", []):
                        if item.get("article_url") == url:
                            cached_img = item.get("image_url")
                            if cached_img:
                                logger.debug(f"thumb: cache hit for {url[:50]}")
                                return {"image_url": cached_img}
                            break
                    else:
                        continue
                    break
        except Exception as _ce:
            logger.debug(f"thumb: cache lookup error: {_ce}")

    if "news.google.com" in url:
        return {"image_url": None}

    _HDRS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def _is_ok(candidate: str) -> bool:
        if not candidate or candidate.startswith("data:"):
            return False
        low = candidate.lower()
        junk = ("sprite", "avatar", "pixel", "tracker", "beacon",
                "1x1", "spacer", "ad.", "doubleclick", "analytics", "stat.")
        if any(p in low for p in junk):
            return False
        if re.search(r"[?&][wh]=\d{1,2}", low):
            return False
        return True

    image_url = None
    try:
        import requests as _req
        html = ""
        with _req.get(url, headers=_HDRS, timeout=8, stream=True, allow_redirects=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=16_384, decode_unicode=True):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="ignore")
                html += chunk
                if ("</head>" in html.lower() and len(html) > 4_000) or len(html) > 48_000:
                    break

        meta_pats = [
            r"""<meta[^>]+property=["']og:image(?::url)?["'][^>]+content=["']([^"']+)["']""",
            r"""<meta[^>]+content=["']([^"']+)["'][^>]+property=["']og:image(?::url)?["']""",
            r"""<meta[^>]+name=["']twitter:image(?::src)?["'][^>]+content=["']([^"']+)["']""",
            r"""<meta[^>]+content=["']([^"']+)["'][^>]+name=["']twitter:image(?::src)?["']""",
        ]
        for pat in meta_pats:
            m = re.search(pat, html, re.IGNORECASE)
            if m and _is_ok(m.group(1)):
                image_url = m.group(1).strip()
                break

        if not image_url:
            body_m = re.search(r"<(?:article|main)[^>]*>(.*)", html, re.IGNORECASE | re.DOTALL)
            body_html = body_m.group(1) if body_m else html
            img_pat = re.compile(
                r"""<img[^>]+(?:src|data-src|data-lazy-src)=["']([^"']+)["']""",
                re.IGNORECASE,
            )
            for img_m in img_pat.finditer(body_html):
                candidate = img_m.group(1).strip()
                if candidate.startswith("//"):
                    candidate = "https:" + candidate
                elif candidate.startswith("/"):
                    origin_m = re.match(r"(https?://[^/]+)", url)
                    if origin_m:
                        candidate = origin_m.group(1) + candidate
                if not _is_ok(candidate):
                    continue
                low = candidate.lower()
                if (
                    re.search(r"\.(jpg|jpeg|png|webp|gif|avif)(\?|$|#)", low)
                    or re.search(r"(images?|media|photos?|cdn|assets|thumbs?)[./]", low)
                ):
                    image_url = candidate
                    break

    except Exception as _e:
        logger.debug(f"thumb proxy failed for {url[:60]}: {_e}")

    return {"image_url": image_url or None}
