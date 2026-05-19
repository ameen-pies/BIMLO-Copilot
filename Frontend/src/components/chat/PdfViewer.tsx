import React, { useState, useRef, useCallback, useEffect } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { HIGHLIGHT_COLOR } from "@/utils/chatUtils";

declare global {
  interface Window {
    pdfjsLib: any;
  }
}

interface PdfViewerProps {
  blobUrl: string;
  highlightText: string | null;
  highlightLines: string[] | null;
  highlightKey: number;
}

function loadPdfJs(): Promise<any> {
  return new Promise((resolve, reject) => {
    if (window.pdfjsLib) { resolve(window.pdfjsLib); return; }
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      resolve(window.pdfjsLib);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export const PdfViewer: React.FC<PdfViewerProps> = ({ blobUrl, highlightText, highlightLines, highlightKey }) => {
  const outerRef   = useRef<HTMLDivElement>(null);
  const innerRef   = useRef<HTMLDivElement>(null);
  const pdfDocRef  = useRef<any>(null);
  const [status, setStatus]   = useState<"loading" | "ready" | "error">("loading");
  const [zoom, setZoom]       = useState(1.0);
  const [zoomMode, setZoomMode] = useState(false);
  const zoomRef    = useRef(1.0);

  const highlightsRef = useRef<string[]>([]);
  highlightsRef.current = [...new Set(
    [...(highlightLines ?? []), ...(highlightText ? [highlightText] : [])].filter(Boolean)
  )];

  const findMatchingItems = (
    items: Array<{ str: string; transform: number[] }>,
    needle: string
  ): number[] => {
    if (!needle.trim()) return [];
    const cleanNeedle = needle.replace(/\*\*/g, '').replace(/\[\d+\]/g, '').trim();
    const needleLo = cleanNeedle.toLowerCase().replace(/\s+/g, " ");
    // Use up to 120 chars — long enough to be unique, short enough for OCR gaps
    const probe = needleLo.slice(0, Math.min(120, needleLo.length)).trim();
    const hit = new Set<number>();

    // Strategy A: direct substring match on joined text items
    for (const sep of [" ", ""]) {
      const joined   = items.map(i => i.str).join(sep);
      const joinedLo = joined.toLowerCase();
      const idx = joinedLo.indexOf(probe);
      if (idx === -1) continue;
      let charCount = 0;
      for (let i = 0; i < items.length; i++) {
        const start = charCount, end = charCount + items[i].str.length;
        charCount += items[i].str.length + sep.length;
        if (end > idx && start < idx + probe.length) hit.add(i);
      }
      if (hit.size > 0) break;
    }
    if (hit.size > 0) return [...hit];

    // Strategy B: word-based fallback — find items containing 3+ significant words
    const sigWords = needleLo.split(/\s+/).filter(w => w.length >= 4);
    if (sigWords.length >= 2) {
      const need = Math.min(3, sigWords.length);
      for (let i = 0; i < items.length; i++) {
        const itemLo = items[i].str.toLowerCase();
        const matches = sigWords.filter(w => itemLo.includes(w)).length;
        if (matches >= need) hit.add(i);
      }
    }
    return [...hit];
  };

  const viewportsRef = useRef<Map<number, any>>(new Map());

  const drawOverlay = useCallback(async (pageNum: number) => {
    const pdf = pdfDocRef.current;
    const inner = innerRef.current;
    if (!pdf || !inner) return;
    const wrapper = inner.querySelector(`[data-page="${pageNum}"]`) as HTMLDivElement | null;
    if (!wrapper) return;
    const overlay = wrapper.querySelector("canvas.hl-overlay") as HTMLCanvasElement | null;
    if (!overlay) return;
    const ctx = overlay.getContext("2d")!;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const needles = highlightsRef.current;
    if (!needles.length) return;
    const viewport = viewportsRef.current.get(pageNum);
    if (!viewport) return;
    const page = await pdf.getPage(pageNum);
    const tc   = await page.getTextContent();
    const items = tc.items as Array<{ str: string; transform: number[] }>;
    ctx.save();
    ctx.fillStyle = HIGHLIGHT_COLOR;
    for (const needle of needles) {
      for (const idx of findMatchingItems(items, needle)) {
        const item = items[idx] as any;
        const [sx, , , sy, tx, ty] = item.transform;
        const [cx, cy] = viewport.convertToViewportPoint(tx, ty);
        const itemW = typeof item.width === "number" && item.width > 0
          ? item.width * viewport.scale
          : Math.abs(sx) * viewport.scale;
        const height = Math.abs(sy) * viewport.scale;
        ctx.fillRect(cx, cy - height * 0.9, itemW, height * 1.1);
      }
    }
    ctx.restore();
  }, []);

  const renderPage = useCallback(async (pageNum: number) => {
    const pdf   = pdfDocRef.current;
    const inner = innerRef.current;
    const outer = outerRef.current;
    if (!pdf || !inner || !outer) return;

    try {
      const page     = await pdf.getPage(pageNum);
      const DPR      = Math.min(window.devicePixelRatio || 1, 2);
      const availW   = outer.clientWidth - 32;
      const baseVP   = page.getViewport({ scale: 1 });
      const baseScale = availW / baseVP.width;
      const viewport = page.getViewport({ scale: baseScale * DPR });
      viewportsRef.current.set(pageNum, viewport);

      const cssW = Math.floor(baseScale * baseVP.width);
      const cssH = Math.floor(baseScale * baseVP.height);
      const pxW  = Math.floor(viewport.width);
      const pxH  = Math.floor(viewport.height);

      let wrapper = inner.querySelector(`[data-page="${pageNum}"]`) as HTMLDivElement | null;
      let pgCanvas: HTMLCanvasElement;
      let hlCanvas: HTMLCanvasElement;

      if (!wrapper) {
        wrapper = document.createElement("div");
        wrapper.dataset.page  = String(pageNum);
        wrapper.style.cssText = `position:relative;margin:0 auto 12px;box-shadow:0 1px 8px rgba(0,0,0,.25);background:#fff;flex-shrink:0;`;
        pgCanvas = document.createElement("canvas");
        pgCanvas.className = "pg-canvas";
        pgCanvas.style.cssText = "position:absolute;top:0;left:0;";
        hlCanvas = document.createElement("canvas");
        hlCanvas.className = "hl-overlay";
        hlCanvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;";
        wrapper.appendChild(pgCanvas);
        wrapper.appendChild(hlCanvas);
        const siblings     = Array.from(inner.querySelectorAll("[data-page]"));
        const insertBefore = siblings.find(s => Number((s as HTMLElement).dataset.page) > pageNum);
        if (insertBefore) inner.insertBefore(wrapper, insertBefore);
        else inner.appendChild(wrapper);
      } else {
        pgCanvas = wrapper.querySelector("canvas.pg-canvas")!;
        hlCanvas = wrapper.querySelector("canvas.hl-overlay")!;
      }

      wrapper.style.width  = cssW + "px";
      wrapper.style.height = cssH + "px";
      pgCanvas.width  = pxW;  pgCanvas.height  = pxH;
      pgCanvas.style.width  = cssW + "px"; pgCanvas.style.height = cssH + "px";
      hlCanvas.width  = pxW;  hlCanvas.height  = pxH;
      hlCanvas.style.width  = cssW + "px"; hlCanvas.style.height = cssH + "px";

      const ctx = pgCanvas.getContext("2d")!;
      ctx.clearRect(0, 0, pxW, pxH);
      await page.render({ canvasContext: ctx, viewport }).promise;
      await drawOverlay(pageNum);
    } catch { /* skip */ }
  }, [drawOverlay]);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    zoomRef.current = 1.0;
    setZoom(1.0);
    setZoomMode(false);
    naturalSizeRef.current = null;
    translateRef.current   = { x: 0, y: 0 };
    if (innerRef.current) {
      innerRef.current.style.transform = "";
      innerRef.current.style.width     = "";
      innerRef.current.style.height    = "";
    }
    viewportsRef.current.clear();

    (async () => {
      try {
        const lib = await loadPdfJs();
        const pdf = await lib.getDocument(blobUrl).promise;
        if (cancelled) return;
        pdfDocRef.current = pdf;
        setStatus("ready");
        for (let i = 1; i <= pdf.numPages; i++) {
          if (cancelled) return;
          await renderPage(i);
        }
      } catch (e) { console.error("[PdfViewer] load error:", e); if (!cancelled) setStatus("error"); }
    })();
    return () => { cancelled = true; };
  }, [blobUrl, renderPage]);

  useEffect(() => {
    const el = outerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if (status !== "ready" || !pdfDocRef.current) return;
      if (innerRef.current) {
        innerRef.current.style.transform = "";
        innerRef.current.style.width     = "";
        innerRef.current.style.height    = "";
      }
      naturalSizeRef.current = null;
      translateRef.current   = { x: 0, y: 0 };
      zoomRef.current = 1.0;
      setZoom(1.0);
      const pdf = pdfDocRef.current;
      (async () => {
        for (let i = 1; i <= pdf.numPages; i++) await renderPage(i);
      })();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [status, renderPage]);

  useEffect(() => {
    if (status !== "ready" || !pdfDocRef.current) return;
    for (let i = 1; i <= pdfDocRef.current.numPages; i++) drawOverlay(i);
  }, [highlightKey, status, drawOverlay]);

  useEffect(() => {
    if (status !== "ready" || !pdfDocRef.current) return;
    const needles = highlightsRef.current;
    if (!needles.length || !innerRef.current || !outerRef.current) return;
    setTimeout(async () => {
      const pdf   = pdfDocRef.current;
      const inner = innerRef.current;
      const outer = outerRef.current;
      if (!pdf || !inner || !outer) return;
      const wrappers = Array.from(inner.querySelectorAll("[data-page]")) as HTMLDivElement[];
      for (const wrapper of wrappers) {
        const pageNum = Number(wrapper.dataset.page);
        const page    = await pdf.getPage(pageNum);
        const tc      = await page.getTextContent();
        const spaced  = (tc.items as any[]).map((i: any) => i.str).join(" ").toLowerCase();
        const nospace = (tc.items as any[]).map((i: any) => i.str).join("").toLowerCase();
        const hit = needles.some(n => {
          const nl = n.toLowerCase().replace(/\s+/g, " ").trim();
          return spaced.includes(nl) || nospace.includes(nl.replace(/\s+/g, ""));
        });
        if (!hit) continue;

        const z = zoomRef.current;
        const pageOffsetTop = wrapper.offsetTop * z + (translateRef.current.y % 1);
        const targetTy = -(wrapper.offsetTop * z) + outer.clientHeight / 2 - (wrapper.offsetHeight * z) / 2;
        const maxTy = 0;
        const minTy = -(inner.scrollHeight * z - outer.clientHeight);
        const clampedTy = Math.min(maxTy, Math.max(minTy, targetTy));

        const startTy = translateRef.current.y;
        const startTx = translateRef.current.x;
        const duration = 350;
        const startTime = performance.now();
        const animate = (now: number) => {
          const t = Math.min(1, (now - startTime) / duration);
          const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
          const newTy = startTy + (clampedTy - startTy) * ease;
          translateRef.current = { x: startTx, y: newTy };
          if (inner) inner.style.transform = `translate(${startTx}px, ${newTy}px) scale(${z})`;
          if (t < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
        break;
      }
    }, 250);
  }, [highlightKey, status]);

  const naturalSizeRef = useRef<{ w: number; h: number } | null>(null);
  const translateRef   = useRef({ x: 0, y: 0 });

  const applyZoom = useCallback((newZoom: number, focalX: number, focalY: number) => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    const oldZoom = zoomRef.current;
    const ratio   = newZoom / oldZoom;

    if (!naturalSizeRef.current) {
      naturalSizeRef.current = { w: inner.offsetWidth, h: inner.offsetHeight };
    }
    const { w: natW, h: natH } = naturalSizeRef.current;

    const tx = translateRef.current.x;
    const ty = translateRef.current.y;

    const originX = focalX - tx;
    const originY = focalY - ty;

    const newTx = focalX - originX * ratio;
    const newTy = focalY - originY * ratio;

    translateRef.current = { x: newTx, y: newTy };
    zoomRef.current = newZoom;

    inner.style.transformOrigin = "0 0";
    inner.style.transform       = `translate(${newTx}px, ${newTy}px) scale(${newZoom})`;
    inner.style.width           = `${natW}px`;
    inner.style.height          = `${natH}px`;

    setZoom(newZoom);
  }, []);

  const applyZoomRef   = useRef(applyZoom);
  useEffect(() => { applyZoomRef.current = applyZoom; }, [applyZoom]);

  const isPanningRef   = useRef(false);
  const panStartRef    = useRef({ x: 0, y: 0 });
  const panTranslateRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const el = outerRef.current;
    if (!el || status !== "ready") return;

    let pending: { deltaY: number; deltaX: number; clientX: number; clientY: number; isZoom: boolean } | null = null;
    let rafId = 0;

    const flush = () => {
      if (!pending) return;
      const { deltaY, deltaX, clientX, clientY, isZoom } = pending;
      pending = null;

      if (isZoom) {
        const factor  = deltaY > 0 ? 0.88 : 1.0 / 0.88;
        const newZoom = Math.min(4, Math.max(0.3, zoomRef.current * factor));
        const rect    = el.getBoundingClientRect();
        applyZoomRef.current(newZoom, clientX - rect.left, clientY - rect.top);
      } else {
        const inner = innerRef.current;
        if (!inner) return;
        if (!naturalSizeRef.current) {
          naturalSizeRef.current = { w: inner.offsetWidth, h: inner.offsetHeight };
        }
        const { w: natW, h: natH } = naturalSizeRef.current;
        const z = zoomRef.current;
        const maxX = -(natW * z - el.clientWidth);
        const maxY = -(natH * z - el.clientHeight);
        const newTx = Math.min(0, Math.max(maxX, translateRef.current.x - deltaX));
        const newTy = Math.min(0, Math.max(maxY, translateRef.current.y - deltaY));
        translateRef.current = { x: newTx, y: newTy };
        inner.style.transform = `translate(${newTx}px, ${newTy}px) scale(${z})`;
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const isZoom = e.ctrlKey || e.metaKey;
      if (pending && pending.isZoom !== isZoom) {
        cancelAnimationFrame(rafId);
        flush();
      }
      pending = {
        deltaY:  (pending?.deltaY  ?? 0) + e.deltaY,
        deltaX:  (pending?.deltaX  ?? 0) + e.deltaX,
        clientX: e.clientX,
        clientY: e.clientY,
        isZoom,
      };
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(flush);
    };

    el.addEventListener("wheel", onWheel, { passive: false, capture: true });
    return () => {
      el.removeEventListener("wheel", onWheel, { capture: true } as any);
      cancelAnimationFrame(rafId);
    };
  }, [status]);

  useEffect(() => {
    const el = outerRef.current;
    if (!el || status !== "ready") return;

    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      if ((e.target as HTMLElement).closest("button")) return;
      isPanningRef.current = true;
      panStartRef.current  = { x: e.clientX, y: e.clientY };
      panTranslateRef.current = { ...translateRef.current };
      el.style.cursor = "grabbing";
      e.preventDefault();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isPanningRef.current) return;
      const inner = innerRef.current;
      if (!inner || !naturalSizeRef.current) return;
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      const z = zoomRef.current;
      const { w: natW, h: natH } = naturalSizeRef.current;
      const maxX = -(natW * z - el.clientWidth);
      const maxY = -(natH * z - el.clientHeight);
      const newTx = Math.min(0, Math.max(maxX, panTranslateRef.current.x + dx));
      const newTy = Math.min(0, Math.max(maxY, panTranslateRef.current.y + dy));
      translateRef.current = { x: newTx, y: newTy };
      inner.style.transform = `translate(${newTx}px, ${newTy}px) scale(${z})`;
    };

    const onMouseUp = () => {
      if (!isPanningRef.current) return;
      isPanningRef.current = false;
      el.style.cursor = zoomMode ? (zoomRef.current >= 4 ? "zoom-out" : "zoom-in") : "grab";
    };

    el.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup",   onMouseUp);
    return () => {
      el.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup",   onMouseUp);
    };
  }, [status, zoomMode]);

  const toolbarZoom = useCallback((newZoom: number) => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    if (newZoom === 1) {
      translateRef.current   = { x: 0, y: 0 };
      naturalSizeRef.current = null;
      zoomRef.current        = 1;
      inner.style.transform  = "";
      inner.style.width      = "";
      inner.style.height     = "";
      setZoom(1);
      return;
    }

    applyZoom(newZoom, outer.clientWidth / 2, outer.clientHeight / 2);
  }, [applyZoom]);

  const handleContainerClick = (e: React.MouseEvent) => {
    if (!zoomMode) return;
    const outer = outerRef.current;
    if (!outer) return;
    const rect  = outer.getBoundingClientRect();
    const factor = e.altKey ? 0.8 : 1.25;
    const newZoom = Math.min(4, Math.max(0.3, zoomRef.current * factor));
    applyZoom(newZoom, e.clientX - rect.left, e.clientY - rect.top);
  };

  return (
    <div className="flex flex-col h-full" style={{ minHeight: 0 }}>
      {status === "loading" && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 text-muted-foreground">
          <Loader2 className="h-7 w-7 animate-spin text-primary" />
          <p className="text-sm">Rendering PDF…</p>
        </div>
      )}
      {status === "error" && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-6">
          <AlertCircle className="h-8 w-8 text-destructive/70" />
          <p className="text-sm text-muted-foreground">Could not render PDF.</p>
        </div>
      )}
      {status === "ready" && (
        <>
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-border bg-background/80 shrink-0 gap-2">
            <span className="text-[11px] text-muted-foreground/60 hidden sm:block">Ctrl+scroll to zoom</span>
            <div className="flex items-center gap-1 ms-auto">
              <button
                onClick={() => setZoomMode(m => !m)}
                title={zoomMode ? "Exit zoom mode" : "Click-to-zoom mode"}
                className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${zoomMode ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                  <circle cx="6" cy="6" r="4.5"/>
                  <line x1="9.5" y1="9.5" x2="13" y2="13"/>
                  {zoomMode ? <><line x1="4" y1="6" x2="8" y2="6"/><line x1="6" y1="4" x2="6" y2="8"/></> : <line x1="4" y1="6" x2="8" y2="6"/>}
                </svg>
              </button>
              <button onClick={() => toolbarZoom(Math.max(0.3, zoomRef.current / 1.25))} disabled={zoom <= 0.3} className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-30 transition-colors font-bold text-sm select-none">−</button>
              <button onClick={() => toolbarZoom(1)} className="px-1.5 h-6 text-[11px] text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors tabular-nums min-w-[2.8rem] text-center">{Math.round(zoom * 100)}%</button>
              <button onClick={() => toolbarZoom(Math.min(4, zoomRef.current * 1.25))} disabled={zoom >= 4} className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-30 transition-colors font-bold text-sm select-none">+</button>
            </div>
          </div>
          <div
            ref={outerRef}
            onClick={handleContainerClick}
            className="flex-1 overflow-hidden scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40"
            style={{
              background: "hsl(var(--muted)/0.3)",
              cursor: zoomMode ? (zoom >= 4 ? "zoom-out" : "zoom-in") : "grab",
              minHeight: 0,
            }}
          >
            <div
              ref={innerRef}
              style={{ transformOrigin: "0 0", display: "flex", flexDirection: "column", alignItems: "center", padding: "16px" }}
            />
          </div>
        </>
      )}
    </div>
  );
};
