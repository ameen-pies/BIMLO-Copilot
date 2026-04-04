import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, ExternalLink, Zap,
  Radio, Cable, Scale, HardHat, Newspaper, Sun, Moon, Loader2,
  RefreshCcw,
} from "lucide-react";

// ── Config ─────────────────────────────────────────────────────────────────

const API_BASE  = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const PAGE_SIZE = 10; // must match server-side NEWS_PAGE_SIZE

// ── Category config ─────────────────────────────────────────────────────────

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

// ── Card size variants ──────────────────────────────────────────────────────

type CardSize = "normal" | "wide" | "tall" | "featured";

const SIZE_PATTERN: CardSize[] = [
  "featured", "normal", "tall", "normal",
  "wide",     "normal", "tall", "normal",
];

function getSize(index: number): CardSize {
  return SIZE_PATTERN[index % SIZE_PATTERN.length];
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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

// ── Static overlay gradient ──────────────────────────────────────────────────
const overlayGradient =
  "linear-gradient(to top,rgba(0,0,0,0.88) 0%,rgba(0,0,0,0.40) 55%,rgba(0,0,0,0.08) 100%)";

// ── IntersectionObserver infinite scroll ────────────────────────────────────

function useInfiniteScroll(onLoadMore: () => void, enabled: boolean) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!enabled) return;
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) onLoadMore(); },
      { rootMargin: "0px 0px 300px 0px", threshold: 0 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [enabled, onLoadMore]);
  return sentinelRef;
}

// ── Grid News Card ───────────────────────────────────────────────────────────

function NewsCard({ item, revealed, theme, size }: {
  item: any; revealed: boolean; theme: "light" | "dark"; size: CardSize;
}) {
  const meta  = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href  = item.articleUrl && item.articleUrl !== "#" ? item.articleUrl : item.sourceUrl;
  const [imgError, setImgError] = useState(false);
  const hasImage    = item.imageUrl && !imgError;
  const hasAiImpact = !!item.aiImpact;
  const isFeatured  = size === "featured";
  const isWide      = size === "wide";
  const isTall      = size === "tall" || isFeatured;
  const colSpan     = (size === "wide" || size === "featured") ? 2 : 1;
  const rowSpan     = (size === "tall" || size === "featured") ? 2 : 1;
  const titleSize   = isFeatured ? "1.08rem" : isWide ? "0.92rem" : "0.82rem";
  const titleClamp  = isFeatured ? 4 : isTall ? 4 : isWide ? 2 : 3;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        gridColumn: `span ${colSpan}`,
        gridRow:    `span ${rowSpan}`,
        display: "block", position: "relative",
        borderRadius: 14, overflow: "hidden",
        textDecoration: "none", cursor: "pointer",
        opacity: revealed ? 1 : 0,
        transform: revealed ? "translateY(0) scale(1)" : "translateY(20px) scale(0.97)",
        transition: "opacity 0.38s ease, transform 0.38s ease, box-shadow 0.22s ease",
        border: `1px solid ${theme === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}`,
      }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1.012)";
        el.style.zIndex    = "10";
        el.style.boxShadow = "0 16px 48px rgba(0,0,0,0.35)";
        const ov = el.querySelector(".card-ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.95) 0%,rgba(0,0,0,0.55) 50%,rgba(0,0,0,0.15) 100%)";
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1)";
        el.style.zIndex    = "1";
        el.style.boxShadow = "none";
        const ov = el.querySelector(".card-ov") as HTMLElement;
        if (ov) ov.style.background = overlayGradient;
      }}
    >
      {/* Background */}
      {hasImage
        ? <img
            src={item.imageUrl} alt=""
            onError={() => setImgError(true)}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          />
        : <div style={{ position: "absolute", inset: 0, background: FALLBACK_GRADIENTS[item.category] ?? FALLBACK_GRADIENTS["General"] }} />
      }

      {/* Overlay */}
      <div className="card-ov" style={{ position: "absolute", inset: 0, background: overlayGradient, transition: "background 0.22s ease" }} />

      {/* Category badge */}
      <div style={{
        position: "absolute", top: 12, left: 12,
        display: "inline-flex", alignItems: "center", gap: "0.28rem",
        fontSize: "0.62rem", fontWeight: 700, color: "#fff",
        background: meta.color + "cc", borderRadius: 999,
        padding: "0.18rem 0.55rem", backdropFilter: "blur(6px)",
        letterSpacing: "0.02em",
      }}>
        <meta.Icon size={9} />{meta.label}
      </div>

      {/* External link icon */}
      <ExternalLink size={12} style={{ position: "absolute", top: 13, right: 13, color: "rgba(255,255,255,0.45)" }} />

      {/* Featured accent bar */}
      {isFeatured && (
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg, ${meta.color}, transparent)` }} />
      )}

      {/* Content */}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: isFeatured ? "1.2rem 1.1rem" : "0.85rem 0.95rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.38rem" }}>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.55)", fontWeight: 600 }}>{item.source}</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)" }}>·</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.38)" }}>{timeAgo(item.publishedAt)}</span>
        </div>
        <h3 style={{
          margin: 0, fontSize: titleSize, fontWeight: 700, color: "#fff", lineHeight: 1.35,
          display: "-webkit-box", WebkitLineClamp: titleClamp,
          WebkitBoxOrient: "vertical" as const, overflow: "hidden",
        }}>
          {item.title}
        </h3>
        {hasAiImpact && (isFeatured || isTall) && (
          <p style={{
            margin: "0.48rem 0 0", fontSize: isFeatured ? "0.74rem" : "0.68rem",
            color: "rgba(255,255,255,0.6)", lineHeight: 1.5,
            display: "-webkit-box", WebkitLineClamp: isFeatured ? 3 : 2,
            WebkitBoxOrient: "vertical" as const, overflow: "hidden",
          }}>
            <Zap size={9} style={{ display: "inline", marginRight: 3, color: "#fbbf24", verticalAlign: "middle" }} />
            {item.aiImpact}
          </p>
        )}
        {hasAiImpact && isWide && !isFeatured && (
          <p style={{
            margin: "0.38rem 0 0", fontSize: "0.68rem", color: "rgba(255,255,255,0.55)", lineHeight: 1.45,
            display: "-webkit-box", WebkitLineClamp: 1,
            WebkitBoxOrient: "vertical" as const, overflow: "hidden",
          }}>
            <Zap size={9} style={{ display: "inline", marginRight: 3, color: "#fbbf24", verticalAlign: "middle" }} />
            {item.aiImpact}
          </p>
        )}
      </div>
    </a>
  );
}

// ── Skeleton Card ────────────────────────────────────────────────────────────

function SkeletonCard({ theme, size }: { theme: "light" | "dark"; size: CardSize }) {
  const colSpan = (size === "wide" || size === "featured") ? 2 : 1;
  const rowSpan = (size === "tall" || size === "featured") ? 2 : 1;
  return (
    <div style={{
      gridColumn: `span ${colSpan}`, gridRow: `span ${rowSpan}`,
      borderRadius: 14,
      background: theme === "dark" ? "#1a1a1a" : "#e8e8e8",
      position: "relative", overflow: "hidden",
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

// ── Main Page ────────────────────────────────────────────────────────────────

const NewsPage = () => {
  const navigate = useNavigate();

  // ── Theme ──────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    (typeof window !== "undefined" && (localStorage.getItem("theme") as "light" | "dark")) || "dark"
  );
  const toggleTheme = () => {
    const t = theme === "light" ? "dark" : "light";
    setTheme(t);
    localStorage.setItem("theme", t);
  };

  // ── Filter ─────────────────────────────────────────────────────────────
  const [filter, setFilter] = useState<string>("All");

  // ── Article storage ────────────────────────────────────────────────────
  const [allArticles, setAllArticles]     = useState<any[]>([]);
  const [visibleCount, setVisibleCount]   = useState(PAGE_SIZE);
  const [revealedSet, setRevealedSet]     = useState<Set<number>>(new Set());
  const prevVisibleCount                  = useRef(0);
  const renderedIds                       = useRef<Set<string>>(new Set());

  // ── Pagination state ───────────────────────────────────────────────────
  const [currentPage, setCurrentPage]     = useState(0);
  const [totalPages, setTotalPages]       = useState(0);
  const [hasMore, setHasMore]             = useState(true);

  // ── Loading states ─────────────────────────────────────────────────────
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [fetchingMore, setFetchingMore]           = useState(false);
  const [isRefreshing, setIsRefreshing]           = useState(false);

  // ── "New articles available" freshness banner ──────────────────────────
  const [newRunAvailable, setNewRunAvailable] = useState(false);
  const knownRunAt                             = useRef<string | null>(null);

  const loadMoreGuard = useRef(false);

  // ── Scroll listener ────────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.classList.add("scrollbar-thin");
    return () => document.documentElement.classList.remove("scrollbar-thin");
  }, []);

  // ── Add articles deduped ───────────────────────────────────────────────
  const addArticles = useCallback((items: any[]) => {
    const fresh = items
      .map(normalize)
      .filter(a => {
        const id = a.id ?? a.articleUrl ?? a.title ?? String(Math.random());
        if (renderedIds.current.has(id)) return false;
        renderedIds.current.add(id);
        return true;
      });
    if (fresh.length === 0) return;
    setAllArticles(prev => [...prev, ...fresh]);
  }, []);

  // ── Fetch a single page from the cache ────────────────────────────────
  const fetchPage = useCallback(async (pageNum: number): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/news/pages/${pageNum}`);
      if (!res.ok) return false;
      const data = await res.json();
      addArticles(data.items ?? []);
      setCurrentPage(pageNum);
      setHasMore(data.has_more ?? false);
      return true;
    } catch (e) {
      console.error(`fetchPage(${pageNum}) error:`, e);
      return false;
    }
  }, [addArticles]);

  // ── Initial load ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function init() {
      setIsInitialLoading(true);
      try {
        // Fetch manifest to know total pages
        const metaRes = await fetch(`${API_BASE}/api/news/meta`);
        if (!metaRes.ok) throw new Error("No cache available yet");
        const meta = await metaRes.json();
        if (!cancelled) {
          setTotalPages(meta.total_pages ?? 0);
          knownRunAt.current = meta.run_at ?? null;
        }
        // Fetch first page
        await fetchPage(0);
      } catch (e) {
        console.error("Initial load error:", e);
        // If the cache isn't ready yet, poll until it is
        if (!cancelled) pollUntilReady();
      } finally {
        if (!cancelled) setIsInitialLoading(false);
      }
    }

    init();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll status until the pipeline produces a first cache (first-boot case)
  const pollUntilReady = useCallback(() => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/news/status`);
        const data = await res.json();
        if (!data.running && data.total_pages > 0) {
          clearInterval(interval);
          // Cache is now ready — reload
          const metaRes = await fetch(`${API_BASE}/api/news/meta`);
          const meta    = await metaRes.json();
          setTotalPages(meta.total_pages ?? 0);
          knownRunAt.current = meta.run_at ?? null;
          await fetchPage(0);
          setIsInitialLoading(false);
        }
      } catch (_) { /* keep polling */ }
    }, 8000);
    return () => clearInterval(interval);
  }, [fetchPage]);

  // ── Poll for new cycle every 60s ───────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/news/status`);
        const data = await res.json();
        if (
          knownRunAt.current !== null &&
          data.last_run_at   !== null &&
          data.last_run_at   !== knownRunAt.current
        ) {
          setNewRunAvailable(true);
        }
      } catch (_) { /* ignore */ }
    }, 60_000);
    return () => clearInterval(interval);
  }, []);

  // ── Hard reset: reload from page 0 with fresh cache ───────────────────
  const resetAndReload = useCallback(async () => {
    renderedIds.current.clear();
    setAllArticles([]);
    setVisibleCount(PAGE_SIZE);
    setRevealedSet(new Set());
    prevVisibleCount.current = 0;
    setCurrentPage(0);
    setHasMore(true);
    setNewRunAvailable(false);
    setIsInitialLoading(true);

    try {
      const metaRes = await fetch(`${API_BASE}/api/news/meta`);
      const meta    = await metaRes.json();
      setTotalPages(meta.total_pages ?? 0);
      knownRunAt.current = meta.run_at ?? null;
      await fetchPage(0);
    } catch (e) {
      console.error("resetAndReload error:", e);
    } finally {
      setIsInitialLoading(false);
    }
  }, [fetchPage]);

  // ── Refresh button: trigger pipeline + poll until done ─────────────────
  const handleRefresh = useCallback(async () => {
    if (isRefreshing || fetchingMore) return;
    setIsRefreshing(true);
    try {
      await fetch(`${API_BASE}/api/news/trigger?force=true`, { method: "POST" });
      // Poll status until running=false, then reload
      const poll = setInterval(async () => {
        try {
          const res  = await fetch(`${API_BASE}/api/news/status`);
          const data = await res.json();
          if (!data.running) {
            clearInterval(poll);
            setIsRefreshing(false);
            await resetAndReload();
          }
        } catch (_) { /* keep polling */ }
      }, 5000);
    } catch (e) {
      console.error("Refresh trigger error:", e);
      setIsRefreshing(false);
    }
  }, [isRefreshing, fetchingMore, resetAndReload]);

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

  // ── Infinite scroll: load more articles ───────────────────────────────
  const loadMore = useCallback(async () => {
    if (loadMoreGuard.current || fetchingMore || isInitialLoading) return;

    // First exhaust what's already in allArticles (just advance visibleCount)
    if (visibleCount < filtered.length) {
      setVisibleCount(c => c + PAGE_SIZE);
      return;
    }

    // Then fetch the next server page
    if (!hasMore) return;
    const nextPage = currentPage + 1;
    if (nextPage >= totalPages) { setHasMore(false); return; }

    loadMoreGuard.current = true;
    setFetchingMore(true);
    try {
      const ok = await fetchPage(nextPage);
      if (ok) setVisibleCount(c => c + PAGE_SIZE);
    } finally {
      setFetchingMore(false);
      loadMoreGuard.current = false;
    }
  }, [
    loadMoreGuard, fetchingMore, isInitialLoading,
    visibleCount, filtered.length,
    hasMore, currentPage, totalPages, fetchPage,
  ]);

  const sentinelRef = useInfiniteScroll(loadMore, !isInitialLoading && !fetchingMore && (hasMore || visibleCount < filtered.length));

  // ── Grid styles ────────────────────────────────────────────────────────
  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gridAutoRows: "160px",
    gap: "0.75rem",
    gridAutoFlow: "dense",
  };

  // ── Render ─────────────────────────────────────────────────────────────
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
        @media (max-width: 1024px) { .news-grid { grid-template-columns: repeat(2, 1fr) !important; } }
        @media (max-width: 640px)  {
          .news-grid { grid-template-columns: 1fr !important; grid-auto-rows: 200px !important; }
          .news-grid > * { grid-column: span 1 !important; grid-row: span 1 !important; }
        }
      `}</style>

      {/* ── Header ────────────────────────────────────────────────────── */}
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
            <h1 style={{ margin: 0, fontSize: "1.35rem", fontWeight: 800, color: theme === "dark" ? "#fff" : "#000", lineHeight: 1.2 }}>
              Industry Briefing
            </h1>
            <p style={{ margin: "0.1rem 0 0", fontSize: "0.72rem", color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", opacity: 0.65 }}>
              Telecom &amp; construction intelligence
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
          {!isInitialLoading && allArticles.length > 0 && (
            <span style={{ fontSize: "0.65rem", color: theme === "dark" ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.35)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
              {visible.length} / {filtered.length}
              {fetchingMore && <Loader2 size={10} style={{ animation: "spin 1s linear infinite", color: "#3b9eff" }} />}
            </span>
          )}
          <button
            onClick={handleRefresh}
            disabled={isRefreshing || fetchingMore}
            title="Refresh news"
            style={{
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              width: 36, height: 36, borderRadius: 8,
              background: theme === "dark" ? "#1a1a1a" : "#e8e8e8",
              border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              cursor: isRefreshing || fetchingMore ? "not-allowed" : "pointer",
              transition: "all 0.15s",
              color: theme === "dark" ? "#94a3b8" : "#666",
              opacity: isRefreshing || fetchingMore ? 0.5 : 1,
            }}
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

      {/* ── "New articles available" banner ────────────────────────────── */}
      {newRunAvailable && (
        <div
          onClick={resetAndReload}
          style={{
            display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem",
            padding: "0.55rem 1rem", marginBottom: "1rem",
            borderRadius: 10, cursor: "pointer",
            background: theme === "dark" ? "#0d2a1f" : "#d1fae5",
            border: `1px solid ${theme === "dark" ? "#1a5c3a" : "#6ee7b7"}`,
            fontSize: "0.75rem", fontWeight: 600,
            color: theme === "dark" ? "#34d399" : "#065f46",
            transition: "opacity 0.15s",
          }}
        >
          <RefreshCcw size={13} />
          Fresh articles are ready — click to load them
        </div>
      )}

      {/* ── Refreshing indicator ──────────────────────────────────────── */}
      {isRefreshing && (
        <div style={{
          display: "flex", alignItems: "center", gap: "0.45rem",
          fontSize: "0.68rem", fontWeight: 500, marginBottom: "0.8rem",
          color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.38)",
        }}>
          <Loader2 size={11} style={{ animation: "spin 1s linear infinite", flexShrink: 0 }} />
          Running Industry Analyst Agent… this takes a few minutes
        </div>
      )}

      {/* ── Filter pills ─────────────────────────────────────────────── */}
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
          <p style={{ margin: 0, fontSize: "0.82rem" }}>
            {totalPages === 0
              ? "The Industry Analyst Agent is collecting articles. Check back in a few minutes."
              : "No articles in this category."}
          </p>
          {totalPages === 0 && (
            <p style={{ margin: "0.4rem 0 0", fontSize: "0.68rem", opacity: 0.5 }}>
              Polls automatically every 8 seconds…
            </p>
          )}
          {totalPages > 0 && (
            <button
              onClick={() => setFilter("All")}
              style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff", background: "none", border: "none", cursor: "pointer" }}
            >
              Show all →
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

          {/* Sentinel for IntersectionObserver */}
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

          {!hasMore && !fetchingMore && allArticles.length > 0 && (
            <p style={{
              textAlign: "center", padding: "1.2rem 0",
              fontSize: "0.65rem",
              color: theme === "dark" ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.25)",
            }}>
              You've reached the end of this cycle's briefing.
            </p>
          )}
        </>
      )}
    </div>
  );
};

export default NewsPage;