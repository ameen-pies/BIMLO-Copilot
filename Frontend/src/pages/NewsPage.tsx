import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, ExternalLink, Zap,
  Radio, Cable, Scale, HardHat, Newspaper, Sun, Moon,
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

const SIZE_PATTERN = ["wide", "narrow", "narrow"] as const;
const PAGE_SIZE = 5;

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
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

// ── IntersectionObserver infinite-scroll hook ──────────────────────────────
//
// Attaches an observer to a sentinel <div> at the bottom of the list.
// When it enters the viewport, onLoadMore is called once.
// Re-arms automatically after loadingMore flips back to false.

function useInfiniteScroll(
  onLoadMore: () => void,
  enabled: boolean,
) {
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!enabled) return;
    const el = sentinelRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          onLoadMore();
        }
      },
      {
        // Start loading a little before the sentinel is fully in view
        rootMargin: "0px 0px 120px 0px",
        threshold: 0,
      },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [enabled, onLoadMore]);

  return sentinelRef;
}

// ── Cards ──────────────────────────────────────────────────────────────────

const _enrichRequested = new Set<string>();

async function requestEnrich(articleId: string, onDone: (enriched: any) => void) {
  if (!articleId || _enrichRequested.has(articleId)) return;
  _enrichRequested.add(articleId);
  try {
    const res = await fetch(`http://localhost:8000/api/news/enrich/${articleId}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      onDone(data);
    }
  } catch {
    // silently ignore — card stays unenriched
  }
}

function NewsCard({ item: initialItem, size, revealed, theme, onEnriched }: {
  item: any; size: "wide" | "narrow"; revealed: boolean; theme: "light" | "dark";
  onEnriched?: (id: string, enriched: any) => void;
}) {
  const [item, setItem] = useState(initialItem);
  const meta = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href = item.articleUrl && item.articleUrl !== "#" ? item.articleUrl : item.sourceUrl;
  const [imgError, setImgError] = useState(false);
  const hasImage = item.imageUrl && !imgError;

  useEffect(() => { setItem(initialItem); }, [initialItem]);

  const handleMouseEnter = (e: React.MouseEvent<HTMLAnchorElement>) => {
    (e.currentTarget as HTMLElement).style.transform = "scale(1.015)";
    (e.currentTarget as HTMLElement).style.zIndex = "2";
    const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
    if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.93) 0%,rgba(0,0,0,0.52) 55%,rgba(0,0,0,0.12) 100%)";

    if (!item.enriched && item.id) {
      requestEnrich(item.id, (enriched) => {
        const normalized = {
          ...enriched,
          articleUrl:  enriched.article_url  ?? enriched.articleUrl  ?? "#",
          imageUrl:    enriched.image_url    ?? enriched.imageUrl    ?? null,
          aiImpact:    enriched.ai_impact    ?? enriched.aiImpact    ?? "",
          sourceUrl:   enriched.source_url   ?? enriched.sourceUrl   ?? "#",
          publishedAt: enriched.published_at ?? enriched.publishedAt ?? new Date().toISOString(),
        };
        setItem(normalized);
        onEnriched?.(item.id, normalized);
      });
    }
  };

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        gridColumn: size === "wide" ? "span 2" : "span 1",
        position: "relative", display: "block",
        height: size === "wide" ? 260 : 200,
        borderRadius: 12, overflow: "hidden",
        textDecoration: "none", cursor: "pointer", zIndex: 1,
        opacity: revealed ? 1 : 0,
        transform: revealed ? "translateY(0) scale(1)" : "translateY(18px) scale(0.97)",
        transition: "opacity 0.38s ease, transform 0.38s ease",
        border: `1px solid ${theme === "dark" ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"}`,
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.transform = "scale(1)";
        (e.currentTarget as HTMLElement).style.zIndex = "1";
        const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.84) 0%,rgba(0,0,0,0.36) 55%,transparent 100%)";
      }}
    >
      {hasImage
        ? <img src={item.imageUrl} alt="" onError={() => setImgError(true)}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
        : <div style={{ position: "absolute", inset: 0, background: FALLBACK_GRADIENTS[item.category] ?? FALLBACK_GRADIENTS["General"] }} />
      }

      <div className="ov" style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(to top,rgba(0,0,0,0.84) 0%,rgba(0,0,0,0.36) 55%,transparent 100%)",
        transition: "background 0.22s ease",
      }} />

      <div style={{
        position: "absolute", top: 10, left: 10,
        display: "inline-flex", alignItems: "center", gap: "0.28rem",
        fontSize: "0.63rem", fontWeight: 700, color: "#fff",
        background: meta.color + "cc", borderRadius: 999, padding: "0.18rem 0.5rem",
        backdropFilter: "blur(6px)",
      }}>
        <meta.Icon size={9} />
        {meta.label}
      </div>

      <ExternalLink size={12} style={{ position: "absolute", top: 12, right: 12, color: "rgba(255,255,255,0.5)" }} />

      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: size === "wide" ? "1rem 1.15rem" : "0.8rem 0.95rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>{item.source}</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)" }}>·</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.4)" }}>{timeAgo(item.publishedAt)}</span>
        </div>
        <h3 style={{
          margin: 0, fontSize: size === "wide" ? "0.98rem" : "0.8rem",
          fontWeight: 700, color: "#fff", lineHeight: 1.35,
          display: "-webkit-box", WebkitLineClamp: size === "wide" ? 2 : 3,
          WebkitBoxOrient: "vertical" as const, overflow: "hidden",
        }}>
          {item.title}
        </h3>
        {size === "wide" && item.aiImpact && (
          <p style={{
            margin: "0.45rem 0 0", fontSize: "0.7rem", color: "rgba(255,255,255,0.6)", lineHeight: 1.5,
            display: "-webkit-box", WebkitLineClamp: 2,
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

function SkeletonCard({ size, theme }: { size: "wide" | "narrow"; theme: "light" | "dark" }) {
  return (
    <div style={{
      gridColumn: size === "wide" ? "span 2" : "span 1",
      height: size === "wide" ? 260 : 200,
      borderRadius: 12,
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

// ── Sentinel / load-more indicator ────────────────────────────────────────

function LoadMoreSentinel({
  sentinelRef,
  loadingMore,
  hasMore,
  theme,
}: {
  sentinelRef: React.RefObject<HTMLDivElement>;
  loadingMore: boolean;
  hasMore: boolean;
  theme: "light" | "dark";
}) {
  return (
    <div
      ref={sentinelRef}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "1.5rem 0",
        gap: "0.5rem",
        fontSize: "0.72rem",
        fontWeight: 500,
        color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)",
        // Keep it in the layout so IntersectionObserver can see it
        minHeight: 40,
      }}
    >
      {loadingMore && (
        <>
          <RefreshCw size={13} style={{ animation: "spin 0.7s linear infinite" }} />
          Loading more…
        </>
      )}
      {!loadingMore && hasMore && (
        // Invisible placeholder — observer watches this
        <span style={{ opacity: 0, userSelect: "none" }}>·</span>
      )}
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

  const [filter, setFilter]           = useState<string>("All");
  const [allFetched, setAllFetched]   = useState<any[]>([]);
  const [visible, setVisible]         = useState<any[]>([]);
  const [revealedSet, setRevealedSet] = useState<Set<number>>(new Set());
  const [isLoading, setIsLoading]     = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore]         = useState(false);
  const prevCount = useRef(0);

  useEffect(() => {
    document.documentElement.classList.add("scrollbar-thin");
    return () => document.documentElement.classList.remove("scrollbar-thin");
  }, []);

  const fetchNews = useCallback(async (force = false) => {
    try {
      const res  = await fetch(`http://localhost:8000/api/news?force=${force}`);
      const data = await res.json();
      if (data.items) return (data.items as any[]).map(normalize);
    } catch (e) {
      console.error("Failed to fetch news:", e);
    }
    return [];
  }, []);

  const applyFilter = useCallback((items: any[], f: string) =>
    f === "All" ? items : items.filter(i => i.category === f), []);

  // Initial load
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setIsLoading(true);
      setVisible([]); setRevealedSet(new Set()); prevCount.current = 0;
      const items = await fetchNews(false);
      if (cancelled) return;
      setAllFetched(items);
      const filtered = applyFilter(items, filter);
      setVisible(filtered.slice(0, PAGE_SIZE));
      setHasMore(filtered.length > PAGE_SIZE);
      setIsLoading(false);
    })();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-filter on tab change
  useEffect(() => {
    setRevealedSet(new Set()); prevCount.current = 0;
    const filtered = applyFilter(allFetched, filter);
    setVisible(filtered.slice(0, PAGE_SIZE));
    setHasMore(filtered.length > PAGE_SIZE);
  }, [filter, allFetched, applyFilter]);

  // Load next batch when sentinel enters viewport
  const loadMore = useCallback(() => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    const filtered = applyFilter(allFetched, filter);
    const next     = filtered.slice(0, visible.length + PAGE_SIZE);
    // Small delay so skeleton cards are visible and the UX feels deliberate
    setTimeout(() => {
      setVisible(next);
      setHasMore(next.length < filtered.length);
      setLoadingMore(false);
    }, 500);
  }, [loadingMore, hasMore, allFetched, filter, visible.length, applyFilter]);

  // Observer is enabled only when there's more to load and we're not already loading
  const sentinelRef = useInfiniteScroll(loadMore, hasMore && !loadingMore && !isLoading);

  // Refresh — force re-fetch from backend
  const handleRefresh = useCallback(async () => {
    if (isRefreshing || isLoading) return;
    setIsRefreshing(true);
    setVisible([]); setRevealedSet(new Set()); prevCount.current = 0;
    const items    = await fetchNews(true);
    setAllFetched(items);
    const filtered = applyFilter(items, filter);
    setVisible(filtered.slice(0, PAGE_SIZE));
    setHasMore(filtered.length > PAGE_SIZE);
    setIsRefreshing(false);
  }, [isRefreshing, isLoading, fetchNews, filter, applyFilter]);

  // Staggered card reveal
  useEffect(() => {
    const start = prevCount.current;
    const end   = visible.length;
    prevCount.current = end;
    if (end <= start) return;
    for (let i = start; i < end; i++) {
      setTimeout(() => {
        setRevealedSet(prev => { const s = new Set(prev); s.add(i); return s; });
      }, (i - start) * 55);
    }
  }, [visible.length]);

  const totalFiltered = applyFilter(allFetched, filter).length;

  return (
    <div style={{
      minHeight: "100vh",
      background: theme === "dark" ? "#0a0a0a" : "#ffffff",
      padding: "2rem 2.5rem 3rem",
      transition: "background 0.15s ease",
      color: theme === "dark" ? "#ffffff" : "#000000",
    }}>
      <style>{`
        @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
        @keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
        @media (max-width: 768px) {
          .news-grid { grid-template-columns: 1fr !important; }
          .news-grid > * { grid-column: span 1 !important; height: 180px !important; }
        }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button onClick={() => navigate(-1)} style={{
            display: "inline-flex", alignItems: "center",
            color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
            background: "none", border: "none", cursor: "pointer", padding: 0, opacity: 0.6,
          }}>
            <ArrowLeft size={15} />
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.35rem", fontWeight: 800, color: theme === "dark" ? "#fff" : "#000", lineHeight: 1.2 }}>
              Industry Briefing
            </h1>
            <p style={{ margin: "0.1rem 0 0", fontSize: "0.72rem", color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", opacity: 0.65 }}>
              Live telecom &amp; construction news
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
          {!isLoading && visible.length > 0 && (
            <span style={{ fontSize: "0.65rem", color: theme === "dark" ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.35)" }}>
              {visible.length} / {totalFiltered}
            </span>
          )}
          <button onClick={handleRefresh} disabled={isRefreshing || isLoading} style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 36, height: 36, borderRadius: 8,
            background: theme === "dark" ? "#1a1a1a" : "#f0f0f0",
            border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
            cursor: isRefreshing || isLoading ? "not-allowed" : "pointer",
            transition: "all 0.15s", color: theme === "dark" ? "#94a3b8" : "#666",
            opacity: isRefreshing || isLoading ? 0.5 : 1,
          }} title="Refresh news">
            <RefreshCw size={16} style={{ animation: isRefreshing ? "spin 0.8s linear infinite" : "none" }} />
          </button>
          <button onClick={toggleTheme} style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 36, height: 36, borderRadius: 8,
            background: theme === "dark" ? "#1a1a1a" : "#f0f0f0",
            border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
            cursor: "pointer", transition: "all 0.15s",
            color: theme === "dark" ? "#fbbf24" : "#3b9eff",
          }} title="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      {/* Filter pills */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.38rem", marginBottom: "1.35rem" }}>
        {ALL_FILTERS.map(cat => {
          const meta = CATEGORY_META[cat]; const active = filter === cat; const color = meta?.color ?? "#3b9eff";
          return (
            <button key={cat} onClick={() => setFilter(cat)} style={{
              display: "inline-flex", alignItems: "center", gap: "0.25rem",
              fontSize: "0.68rem", fontWeight: 600, padding: "0.22rem 0.7rem",
              borderRadius: 999, cursor: "pointer",
              border: `1px solid ${active ? color + "88" : theme === "dark" ? "#333" : "#ddd"}`,
              color: active ? color : theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              background: active ? color + "18" : "transparent", transition: "all 0.12s",
            }}>
              {meta?.Icon && <meta.Icon size={9} />}{cat}
            </button>
          );
        })}
      </div>

      {/* Grid */}
      {isLoading ? (
        <div className="news-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.8rem", gridAutoFlow: "dense" }}>
          {[...Array(6)].map((_, i) => <SkeletonCard key={i} size={SIZE_PATTERN[i % 3]} theme={theme} />)}
        </div>
      ) : visible.length === 0 ? (
        <div style={{ textAlign: "center", padding: "5rem 0", color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }}>
          <Newspaper size={30} style={{ margin: "0 auto 0.6rem", display: "block" }} />
          <p style={{ margin: 0, fontSize: "0.82rem" }}>No articles in this category.</p>
          <button onClick={handleRefresh} style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff", background: "none", border: "none", cursor: "pointer" }}>
            Refresh →
          </button>
        </div>
      ) : (
        <>
          <div className="news-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.8rem", gridAutoFlow: "dense" }}>
            {visible.map((item, i) => (
              <NewsCard
                key={item.id ?? i}
                item={item}
                size={SIZE_PATTERN[i % 3]}
                revealed={revealedSet.has(i)}
                theme={theme}
                onEnriched={(id, enriched) => {
                  setAllFetched(prev => prev.map(a => a.id === id ? { ...a, ...enriched } : a));
                  setVisible(prev => prev.map(a => a.id === id ? { ...a, ...enriched } : a));
                }}
              />
            ))}
            {loadingMore && [...Array(PAGE_SIZE)].map((_, i) => (
              <SkeletonCard key={`skel-${i}`} size={SIZE_PATTERN[(visible.length + i) % 3]} theme={theme} />
            ))}
          </div>

          {/* Sentinel lives outside the grid so the observer doesn't fight with grid layout */}
          <LoadMoreSentinel
            sentinelRef={sentinelRef}
            loadingMore={loadingMore}
            hasMore={hasMore}
            theme={theme}
          />

          {!hasMore && visible.length > 0 && (
            <div style={{
              textAlign: "center", padding: "0.5rem 0 0",
              fontSize: "0.65rem", letterSpacing: "0.08em",
              color: theme === "dark" ? "rgba(255,255,255,0.22)" : "rgba(0,0,0,0.22)",
            }}>
              — end of briefing —
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default NewsPage;