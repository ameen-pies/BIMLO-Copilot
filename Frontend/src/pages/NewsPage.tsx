import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, ExternalLink, Zap,
  Radio, Cable, Scale, HardHat, Newspaper, Sun, Moon, Loader2,
} from "lucide-react";

// ── Category config ────────────────────────────────────────────────────────

const CATEGORY_META: Record<string, { label: string; Icon: React.ElementType; color: string }> = {
  "5G":           { label: "5G",           Icon: Radio,     color: "#3b9eff" },
  "Fiber":        { label: "Fiber",        Icon: Cable,     color: "#34d399" },
  "Regulation":   { label: "Regulation",   Icon: Scale,     color: "#fbbf24" },
  "Construction": { label: "Construction", Icon: HardHat,   color: "#a78bfa" },
  "General":      { label: "General",      Icon: Newspaper, color: "#94a3b8" },
};

const ALL_FILTERS = ["All", "5G", "Fiber", "Regulation", "Construction", "General"] as const;

const FALLBACK_GRADIENTS: Record<string, string> = {
  "5G":           "linear-gradient(135deg, #0f2540 0%, #1a4a7a 100%)",
  "Fiber":        "linear-gradient(135deg, #0d2a1f 0%, #1a5c3a 100%)",
  "Regulation":   "linear-gradient(135deg, #2a1f06 0%, #5c440a 100%)",
  "Construction": "linear-gradient(135deg, #1f0d2a 0%, #3d1a5c 100%)",
  "General":      "linear-gradient(135deg, #1a1a2e 0%, #2d2d4e 100%)",
};

const PAGE_SIZE = 5;

// ── Card size variants ─────────────────────────────────────────────────────
//
// The pattern repeats every 8 cards.  Each entry is [colSpan, rowSpan].
// On a 4-column grid this keeps things balanced:
//   featured  (2×2) → big anchor card
//   wide      (2×1) → horizontal emphasis
//   tall      (1×2) → vertical emphasis
//   normal    (1×1) → default
//
// The pattern is fixed so the grid NEVER reflows existing cards
// when new ones arrive — they always append at the tail.

type CardSize = "normal" | "wide" | "tall" | "featured";

const SIZE_PATTERN: CardSize[] = [
  "featured", // 0 → 2×2
  "normal",   // 1 → 1×1
  "tall",     // 2 → 1×2
  "normal",   // 3 → 1×1
  "wide",     // 4 → 2×1
  "normal",   // 5 → 1×1
  "tall",     // 6 → 1×2
  "normal",   // 7 → 1×1
];

function getSize(index: number): CardSize {
  return SIZE_PATTERN[index % SIZE_PATTERN.length];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m    = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function normalize(item: any) {
  return {
    ...item,
    articleUrl:  item.article_url  ?? item.articleUrl  ?? "#",
    imageUrl:    item.image_url    ?? item.imageUrl    ?? null,
    aiImpact:    item.ai_impact    ?? item.aiImpact    ?? "",
    sourceUrl:   item.source_url   ?? item.sourceUrl   ?? "#",
    publishedAt: item.published_at ?? item.publishedAt ?? new Date().toISOString(),
  };
}

// ── SSE helper ─────────────────────────────────────────────────────────────

type StreamCleanup = () => void;

function openStream(
  url: string,
  onItem:  (item: any) => void,
  onDone:  (count: number) => void,
  onError: (msg: string) => void,
): StreamCleanup {
  const es = new EventSource(url);

  es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "article") {
        onItem(normalize(msg.item));
      } else if (msg.type === "done") {
        onDone(msg.count ?? 0);
        es.close();
      } else if (msg.type === "error") {
        onError(msg.message ?? "Unknown stream error");
        es.close();
      }
    } catch {
      // malformed frame — ignore
    }
  };

  es.onerror = () => {
    onError("Stream connection lost");
    es.close();
  };

  return () => es.close();
}

// ── IntersectionObserver infinite scroll ──────────────────────────────────

function useInfiniteScroll(onLoadMore: () => void, enabled: boolean) {
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!enabled) return;
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) onLoadMore(); },
      { rootMargin: "0px 0px 200px 0px", threshold: 0 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [enabled, onLoadMore]);

  return sentinelRef;
}

// ── Grid News Card ─────────────────────────────────────────────────────────

function NewsCard({ item, revealed, theme, size }: {
  item: any;
  revealed: boolean;
  theme: "light" | "dark";
  size: CardSize;
}) {
  const meta = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href = item.articleUrl && item.articleUrl !== "#" ? item.articleUrl : item.sourceUrl;
  const [imgError, setImgError] = useState(false);
  const hasImage = item.imageUrl && !imgError;
  const hasAiImpact = !!item.aiImpact;

  const isFeatured = size === "featured";
  const isWide     = size === "wide";
  const isTall     = size === "tall" || isFeatured;

  // Col / row span values (grid handles the sizing, not the card itself)
  const colSpan = (size === "wide" || size === "featured") ? 2 : 1;
  const rowSpan = (size === "tall" || size === "featured") ? 2 : 1;

  // Title font size scales with card importance
  const titleSize = isFeatured ? "1.08rem" : isWide ? "0.92rem" : "0.82rem";
  const titleClamp = isFeatured ? 4 : isTall ? 4 : isWide ? 2 : 3;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        gridColumn: `span ${colSpan}`,
        gridRow:    `span ${rowSpan}`,
        display: "block",
        position: "relative",
        borderRadius: 14,
        overflow: "hidden",
        textDecoration: "none",
        cursor: "pointer",
        opacity: revealed ? 1 : 0,
        transform: revealed
          ? "translateY(0) scale(1)"
          : "translateY(20px) scale(0.97)",
        transition: "opacity 0.38s ease, transform 0.38s ease, box-shadow 0.22s ease",
        border: `1px solid ${theme === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}`,
        // No self-contained height — the grid row height drives it
      }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1.012)";
        el.style.zIndex = "10";
        el.style.boxShadow = "0 16px 48px rgba(0,0,0,0.35)";
        const ov = el.querySelector(".card-ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.95) 0%,rgba(0,0,0,0.55) 50%,rgba(0,0,0,0.15) 100%)";
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1)";
        el.style.zIndex = "1";
        el.style.boxShadow = "none";
        const ov = el.querySelector(".card-ov") as HTMLElement;
        if (ov) ov.style.background = overlayGradient;
      }}
    >
      {/* Background */}
      {hasImage
        ? <img
            src={item.imageUrl}
            alt=""
            onError={() => setImgError(true)}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          />
        : <div style={{ position: "absolute", inset: 0, background: FALLBACK_GRADIENTS[item.category] ?? FALLBACK_GRADIENTS["General"] }} />
      }

      {/* Overlay */}
      <div
        className="card-ov"
        style={{
          position: "absolute", inset: 0,
          background: overlayGradient,
          transition: "background 0.22s ease",
        }}
      />

      {/* Category badge */}
      <div style={{
        position: "absolute", top: 12, left: 12,
        display: "inline-flex", alignItems: "center", gap: "0.28rem",
        fontSize: "0.62rem", fontWeight: 700, color: "#fff",
        background: meta.color + "cc", borderRadius: 999,
        padding: "0.18rem 0.55rem",
        backdropFilter: "blur(6px)",
        letterSpacing: "0.02em",
      }}>
        <meta.Icon size={9} />{meta.label}
      </div>

      {/* External link icon */}
      <ExternalLink
        size={12}
        style={{ position: "absolute", top: 13, right: 13, color: "rgba(255,255,255,0.45)" }}
      />

      {/* Featured accent bar */}
      {isFeatured && (
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0,
          height: 3,
          background: `linear-gradient(90deg, ${meta.color}, transparent)`,
        }} />
      )}

      {/* Content */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        padding: isFeatured ? "1.2rem 1.1rem" : "0.85rem 0.95rem",
      }}>
        {/* Source + time */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.38rem" }}>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.55)", fontWeight: 600 }}>
            {item.source}
          </span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)" }}>·</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.38)" }}>
            {timeAgo(item.publishedAt)}
          </span>
        </div>

        {/* Title */}
        <h3 style={{
          margin: 0,
          fontSize: titleSize,
          fontWeight: 700,
          color: "#fff",
          lineHeight: 1.35,
          display: "-webkit-box",
          WebkitLineClamp: titleClamp,
          WebkitBoxOrient: "vertical" as const,
          overflow: "hidden",
        }}>
          {item.title}
        </h3>

        {/* AI Impact — show on featured & tall */}
        {hasAiImpact && (isFeatured || isTall) && (
          <p style={{
            margin: "0.48rem 0 0",
            fontSize: isFeatured ? "0.74rem" : "0.68rem",
            color: "rgba(255,255,255,0.6)",
            lineHeight: 1.5,
            display: "-webkit-box",
            WebkitLineClamp: isFeatured ? 3 : 2,
            WebkitBoxOrient: "vertical" as const,
            overflow: "hidden",
          }}>
            <Zap
              size={9}
              style={{ display: "inline", marginRight: 3, color: "#fbbf24", verticalAlign: "middle" }}
            />
            {item.aiImpact}
          </p>
        )}

        {/* Wide card: show summary inline if available */}
        {hasAiImpact && isWide && !isFeatured && (
          <p style={{
            margin: "0.38rem 0 0",
            fontSize: "0.68rem",
            color: "rgba(255,255,255,0.55)",
            lineHeight: 1.45,
            display: "-webkit-box",
            WebkitLineClamp: 1,
            WebkitBoxOrient: "vertical" as const,
            overflow: "hidden",
          }}>
            <Zap size={9} style={{ display: "inline", marginRight: 3, color: "#fbbf24", verticalAlign: "middle" }} />
            {item.aiImpact}
          </p>
        )}
      </div>
    </a>
  );
}

// Static overlay gradient so it's stable across renders
const overlayGradient =
  "linear-gradient(to top,rgba(0,0,0,0.88) 0%,rgba(0,0,0,0.40) 55%,rgba(0,0,0,0.08) 100%)";

// ── Skeleton Card ──────────────────────────────────────────────────────────

function SkeletonCard({ theme, size }: { theme: "light" | "dark"; size: CardSize }) {
  const colSpan = (size === "wide" || size === "featured") ? 2 : 1;
  const rowSpan = (size === "tall" || size === "featured") ? 2 : 1;

  return (
    <div style={{
      gridColumn: `span ${colSpan}`,
      gridRow:    `span ${rowSpan}`,
      borderRadius: 14,
      background: theme === "dark" ? "#1a1a1a" : "#e8e8e8",
      position: "relative",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: theme === "dark"
          ? "linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.04) 50%,transparent 100%)"
          : "linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.6) 50%,transparent 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.6s ease-in-out infinite",
      }} />
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────

const NewsPage = () => {
  const navigate = useNavigate();

  const [theme, setTheme] = useState<"light" | "dark">(() =>
    (typeof window !== "undefined" && (localStorage.getItem("theme") as "light" | "dark")) || "light"
  );
  const toggleTheme = () => {
    const t = theme === "light" ? "dark" : "light";
    setTheme(t); localStorage.setItem("theme", t);
  };

  const [filter, setFilter] = useState<string>("All");

  const [allArticles, setAllArticles] = useState<any[]>([]);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  // revealedSet tracks which *visible* indices have been faded in
  const [revealedSet, setRevealedSet] = useState<Set<number>>(new Set());
  const prevVisibleCount = useRef(0);

  const [streaming, setStreaming]       = useState(false);
  const [fetchingMore, setFetchingMore] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const seenIds = useRef<Set<string>>(new Set());
  const allArticlesLengthRef = useRef(0);

  useEffect(() => {
    document.documentElement.classList.add("scrollbar-thin");
    return () => document.documentElement.classList.remove("scrollbar-thin");
  }, []);

  // ── Append article, skip dupes ─────────────────────────────────────────

  const addArticle = useCallback((item: any) => {
    const id = item.id ?? item.articleUrl ?? item.title ?? Math.random().toString();
    if (seenIds.current.has(id)) return;
    seenIds.current.add(id);
    setAllArticles(prev => [...prev, item]);
  }, []);

  // ── Auto-advance visibleCount during streaming ────────────────────────

  useEffect(() => {
    if (!streaming) return;
    const newLen = allArticles.length;
    if (newLen <= allArticlesLengthRef.current) return;
    allArticlesLengthRef.current = newLen;
    const shouldShow = Math.ceil(newLen / PAGE_SIZE) * PAGE_SIZE;
    setVisibleCount(prev => Math.max(prev, shouldShow));
  }, [allArticles.length, streaming]);

  // ── Initial stream on mount ────────────────────────────────────────────

  useEffect(() => {
    setStreaming(true);
    allArticlesLengthRef.current = 0;
    const cleanup = openStream(
      "http://localhost:8000/api/news/stream",
      addArticle,
      () => setStreaming(false),
      (err) => { console.error("Initial stream error:", err); setStreaming(false); },
    );
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Refresh ────────────────────────────────────────────────────────────

  const handleRefresh = useCallback(() => {
    if (isRefreshing || streaming) return;
    setIsRefreshing(true);
    setAllArticles([]);
    setVisibleCount(PAGE_SIZE);
    setRevealedSet(new Set());
    prevVisibleCount.current = 0;
    allArticlesLengthRef.current = 0;
    seenIds.current.clear();
    setStreaming(true);
    const cleanup = openStream(
      "http://localhost:8000/api/news/stream?force=true",
      addArticle,
      () => { setStreaming(false); setIsRefreshing(false); },
      (err) => { console.error("Refresh error:", err); setStreaming(false); setIsRefreshing(false); },
    );
    return cleanup;
  }, [isRefreshing, streaming, addArticle]);

  // ── Derived lists ──────────────────────────────────────────────────────

  const filtered = filter === "All" ? allArticles : allArticles.filter(a => a.category === filter);
  const visible  = filtered.slice(0, visibleCount);

  // ── Staggered card reveal ──────────────────────────────────────────────

  useEffect(() => {
    const start = prevVisibleCount.current;
    const end   = visible.length;
    if (end <= start) { prevVisibleCount.current = end; return; }
    prevVisibleCount.current = end;
    for (let i = start; i < end; i++) {
      setTimeout(() => {
        setRevealedSet(prev => { const s = new Set(prev); s.add(i); return s; });
      }, (i - start) * 55);
    }
  }, [visible.length]);

  // Reset reveal on filter change
  useEffect(() => {
    setRevealedSet(new Set());
    prevVisibleCount.current = 0;
    setVisibleCount(PAGE_SIZE);
  }, [filter]);

  // ── Infinite scroll ────────────────────────────────────────────────────

  const fetchMoreGuard = useRef(false);

  const loadMore = useCallback(() => {
    if (fetchMoreGuard.current || streaming) return;

    if (visibleCount < filtered.length) {
      setVisibleCount(c => c + PAGE_SIZE);
      return;
    }

    if (fetchingMore) return;
    fetchMoreGuard.current = true;
    setFetchingMore(true);

    openStream(
      "http://localhost:8000/api/news/next-page",
      (item) => {
        addArticle(item);
        setVisibleCount(c => c + 1);
      },
      () => {
        setFetchingMore(false);
        fetchMoreGuard.current = false;
      },
      (err) => {
        console.error("Next-page stream error:", err);
        setFetchingMore(false);
        fetchMoreGuard.current = false;
      },
    );
  }, [visibleCount, filtered.length, streaming, fetchingMore, addArticle]);

  const sentinelRef = useInfiniteScroll(loadMore, !streaming && !fetchingMore);

  const isInitialLoading = streaming && allArticles.length === 0;

  // ── Grid styles ────────────────────────────────────────────────────────
  //
  // 4 equal columns, auto rows at 160px. Cards spanning 2 rows = 160*2 + gap = 336px.
  // The grid is append-only — new items go to the end, nothing reflows.

  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gridAutoRows: "160px",
    gap: "0.75rem",
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: theme === "dark" ? "#0a0a0a" : "#f5f5f5",
      padding: "2rem 2.5rem 3rem",
      transition: "background 0.15s ease",
      color: theme === "dark" ? "#ffffff" : "#000000",
    }}>
      <style>{`
        @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
        @keyframes spin    { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }

        /* Responsive: collapse to 2 cols on tablet */
        @media (max-width: 1024px) {
          .news-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        /* Responsive: single col on mobile, no span */
        @media (max-width: 640px) {
          .news-grid { grid-template-columns: 1fr !important; grid-auto-rows: 200px !important; }
          .news-grid > * { grid-column: span 1 !important; grid-row: span 1 !important; }
        }
      `}</style>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button
            onClick={() => navigate(-1)}
            style={{
              display: "inline-flex", alignItems: "center",
              color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              background: "none", border: "none", cursor: "pointer", padding: 0, opacity: 0.6,
            }}
          >
            <ArrowLeft size={15} />
          </button>
          <div>
            <h1 style={{
              margin: 0, fontSize: "1.35rem", fontWeight: 800,
              color: theme === "dark" ? "#fff" : "#000", lineHeight: 1.2,
            }}>
              Industry Briefing
            </h1>
            <p style={{
              margin: "0.1rem 0 0", fontSize: "0.72rem",
              color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", opacity: 0.65,
            }}>
              Live telecom &amp; construction news
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
          {!isInitialLoading && allArticles.length > 0 && (
            <span style={{
              fontSize: "0.65rem",
              color: theme === "dark" ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.35)",
              display: "flex", alignItems: "center", gap: "0.3rem",
            }}>
              {visible.length} / {filtered.length}
              {streaming && (
                <Loader2 size={10} style={{ animation: "spin 1s linear infinite", color: "#3b9eff" }} />
              )}
            </span>
          )}
          <button
            onClick={handleRefresh}
            disabled={isRefreshing || streaming}
            style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              width: 36, height: 36, borderRadius: 8,
              background: theme === "dark" ? "#1a1a1a" : "#e8e8e8",
              border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              cursor: isRefreshing || streaming ? "not-allowed" : "pointer",
              transition: "all 0.15s",
              color: theme === "dark" ? "#94a3b8" : "#666",
              opacity: isRefreshing || streaming ? 0.5 : 1,
            }}
            title="Refresh news"
          >
            <RefreshCw size={16} style={{ animation: isRefreshing ? "spin 0.8s linear infinite" : "none" }} />
          </button>
          <button
            onClick={toggleTheme}
            style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              width: 36, height: 36, borderRadius: 8,
              background: theme === "dark" ? "#1a1a1a" : "#e8e8e8",
              border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              cursor: "pointer", transition: "all 0.15s",
              color: theme === "dark" ? "#fbbf24" : "#3b9eff",
            }}
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      {/* ── Filter pills ──────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.38rem", marginBottom: "1.35rem" }}>
        {ALL_FILTERS.map(cat => {
          const meta   = CATEGORY_META[cat];
          const active = filter === cat;
          const color  = meta?.color ?? "#3b9eff";
          return (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              style={{
                display: "inline-flex", alignItems: "center", gap: "0.25rem",
                fontSize: "0.68rem", fontWeight: 600, padding: "0.22rem 0.7rem",
                borderRadius: 999, cursor: "pointer",
                border: `1px solid ${active ? color + "88" : theme === "dark" ? "#333" : "#ddd"}`,
                color: active ? color : theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
                background: active ? color + "18" : "transparent",
                transition: "all 0.12s",
              }}
            >
              {meta?.Icon && <meta.Icon size={9} />}{cat}
            </button>
          );
        })}
      </div>

      {/* ── Live indicator ────────────────────────────────────────────── */}
      {streaming && allArticles.length > 0 && (
        <div style={{
          display: "flex", alignItems: "center", gap: "0.45rem",
          fontSize: "0.68rem", fontWeight: 500, marginBottom: "0.8rem",
          color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.38)",
        }}>
          <Loader2 size={11} style={{ animation: "spin 1s linear infinite", flexShrink: 0 }} />
          Analysing more articles…
        </div>
      )}

      {/* ── Grid ─────────────────────────────────────────────────────── */}
      {isInitialLoading ? (
        <div className="news-grid" style={gridStyle}>
          {SIZE_PATTERN.map((size, i) => (
            <SkeletonCard key={i} theme={theme} size={size} />
          ))}
        </div>
      ) : visible.length === 0 ? (
        <div style={{
          textAlign: "center", padding: "5rem 0",
          color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)",
        }}>
          <Newspaper size={30} style={{ margin: "0 auto 0.6rem", display: "block" }} />
          <p style={{ margin: 0, fontSize: "0.82rem" }}>No articles in this category yet.</p>
          {!streaming && (
            <button
              onClick={handleRefresh}
              style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff", background: "none", border: "none", cursor: "pointer" }}
            >
              Refresh →
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="news-grid" style={gridStyle}>
            {visible.map((item, i) => (
              <NewsCard
                key={item.id ?? item.articleUrl ?? i}
                item={item}
                revealed={revealedSet.has(i)}
                theme={theme}
                size={getSize(i)}
              />
            ))}
            {fetchingMore && SIZE_PATTERN.slice(0, 4).map((size, i) => (
              <SkeletonCard key={`skel-${i}`} theme={theme} size={size} />
            ))}
          </div>

          {/* Sentinel */}
          <div ref={sentinelRef} style={{ height: 1, marginTop: "0.5rem" }} />

          {fetchingMore && (
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "center",
              gap: "0.45rem", padding: "1rem 0",
              fontSize: "0.7rem",
              color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)",
            }}>
              <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} />
              Loading more articles…
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default NewsPage;