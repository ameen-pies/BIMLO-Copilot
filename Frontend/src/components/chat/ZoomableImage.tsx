import React from "react";
import { useTranslation } from "react-i18next";

interface ZoomableImageProps {
  src: string;
  alt: string;
  compact?: boolean;
}

export const ZoomableImage: React.FC<ZoomableImageProps> = ({ src, alt, compact = false }) => {
  const { t } = useTranslation();
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [scale, setScale]   = React.useState(1);
  const [offset, setOffset] = React.useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = React.useState(false);
  const dragStart = React.useRef<{ mx: number; my: number; ox: number; oy: number } | null>(null);

  const clampOffset = React.useCallback((x: number, y: number, s: number) => {
    const c = containerRef.current;
    if (!c) return { x, y };
    const maxX = ((s - 1) * c.clientWidth)  / 2;
    const maxY = ((s - 1) * c.clientHeight) / 2;
    return {
      x: Math.max(-maxX, Math.min(maxX, x)),
      y: Math.max(-maxY, Math.min(maxY, y)),
    };
  }, []);

  const onWheel = React.useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setScale(s => {
      const next = Math.min(8, Math.max(1, s - e.deltaY * 0.002));
      if (next === 1) setOffset({ x: 0, y: 0 });
      else setOffset(o => clampOffset(o.x, o.y, next));
      return next;
    });
  }, [clampOffset]);

  const onMouseDown = React.useCallback((e: React.MouseEvent) => {
    if (scale <= 1) return;
    e.preventDefault();
    setIsDragging(true);
    dragStart.current = { mx: e.clientX, my: e.clientY, ox: offset.x, oy: offset.y };
  }, [scale, offset]);

  const onMouseMove = React.useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragStart.current) return;
    const dx = e.clientX - dragStart.current.mx;
    const dy = e.clientY - dragStart.current.my;
    setOffset(clampOffset(dragStart.current.ox + dx, dragStart.current.oy + dy, scale));
  }, [isDragging, scale, clampOffset]);

  const onMouseUp = React.useCallback(() => {
    setIsDragging(false);
    dragStart.current = null;
  }, []);

  const lastTouches = React.useRef<{ dist: number; cx: number; cy: number } | null>(null);

  const onTouchStart = React.useCallback((e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const [a, b] = [e.touches[0], e.touches[1]];
      const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      lastTouches.current = { dist, cx: (a.clientX + b.clientX) / 2, cy: (a.clientY + b.clientY) / 2 };
    } else if (e.touches.length === 1 && scale > 1) {
      dragStart.current = { mx: e.touches[0].clientX, my: e.touches[0].clientY, ox: offset.x, oy: offset.y };
      setIsDragging(true);
    }
  }, [scale, offset]);

  const onTouchMove = React.useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    if (e.touches.length === 2 && lastTouches.current) {
      const [a, b] = [e.touches[0], e.touches[1]];
      const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      const ratio = dist / lastTouches.current.dist;
      setScale(s => {
        const next = Math.min(8, Math.max(1, s * ratio));
        if (next === 1) setOffset({ x: 0, y: 0 });
        else setOffset(o => clampOffset(o.x, o.y, next));
        return next;
      });
      lastTouches.current = { dist, cx: (a.clientX + b.clientX) / 2, cy: (a.clientY + b.clientY) / 2 };
    } else if (e.touches.length === 1 && isDragging && dragStart.current) {
      const dx = e.touches[0].clientX - dragStart.current.mx;
      const dy = e.touches[0].clientY - dragStart.current.my;
      setOffset(clampOffset(dragStart.current.ox + dx, dragStart.current.oy + dy, scale));
    }
  }, [isDragging, scale, clampOffset]);

  const onTouchEnd = React.useCallback(() => {
    setIsDragging(false);
    dragStart.current  = null;
    lastTouches.current = null;
  }, []);

  const reset = () => { setScale(1); setOffset({ x: 0, y: 0 }); };

  return (
    <div className="flex flex-col h-full w-full">
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden relative flex items-center justify-center bg-transparent"
        style={{ cursor: scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default', touchAction: 'none' }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        <img
          src={src}
          alt={alt}
          draggable={false}
          className="rounded-lg shadow-lg select-none"
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            transform: `scale(${scale}) translate(${offset.x / scale}px, ${offset.y / scale}px)`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.15s ease',
            userSelect: 'none',
            pointerEvents: 'none',
          }}
        />
      </div>

      <div className={`flex items-center justify-between gap-2 px-3 ${compact ? 'py-1' : 'py-2'} border-t border-border/30`}>
        <span className="text-xs text-muted-foreground truncate max-w-[50%]">{alt}</span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setScale(s => { const n = Math.max(0.25, s - 0.25); if (n < 1) setOffset({x:0,y:0}); return n; })}
            className="rounded px-2 py-0.5 text-xs bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
            title={t("chat.zoom_out")}
          >
            <ZoomOut size={14} />
          </button>
          <button
            title={t("chat.zoom_in")}
            onClick={() => setScale(s => Math.min(4, s + 0.25))}
          >
            <ZoomIn size={14} />
          </button>
          {scale !== 1 && (
            <button
              title={t("chat.reset_zoom")}
              onClick={() => { setScale(1); setOffset({ x: 0, y: 0 }); }}
            >{t("chat.reset_zoom")}</button>
          )}
        </div>
      </div>
    </div>
  );
};
