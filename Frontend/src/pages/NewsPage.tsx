import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, ExternalLink, Zap,
  Radio, Cable, Scale, HardHat, Newspaper, Sun, Moon,
} from "lucide-react";

// ── Category config ────────────────────────────────────────────────────────

const CATEGORY_META: Record<string, { label: string; Icon: React.ElementType; color: string }> = {
  "5G": { label: "5G", Icon: Radio, color: "#3b9eff" },
  "Fiber": { label: "Fiber", Icon: Cable, color: "#34d399" },
  "Regulation": { label: "Regulation", Icon: Scale, color: "#fbbf24" },
  "Construction": { label: "Construction", Icon: HardHat, color: "#a78bfa" },
  "General": { label: "General", Icon: Newspaper, color: "#94a3b8" },
};

const ALL_FILTERS = ["All", "5G", "Fiber", "Regulation", "Construction", "General"] as const;

const FALLBACK_GRADIENTS: Record<string, string> = {
  "5G": "linear-gradient(135deg, #0f2540 0%, #1a4a7a 100%)",
  "Fiber": "linear-gradient(135deg, #0d2a1f 0%, #1a5c3a 100%)",
  "Regulation": "linear-gradient(135deg, #2a1f06 0%, #5c440a 100%)",
  "Construction": "linear-gradient(135deg, #1f0d2a 0%, #3d1a5c 100%)",
  "General": "linear-gradient(135deg, #1a1a2e 0%, #2d2d4e 100%)",
};

const SIZE_PATTERN = ["wide", "narrow", "narrow"] as const;
const PAGE_SIZE = 12;

// Spring config
const SPRING_RESISTANCE = 0.18;
const SPRING_TRIGGER_PX = 80;
const SPRING_STIFFNESS = 0.12;

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// ── Spring pull-zone hook ──────────────────────────────────────────────────

function useSpringPull(
  onTrigger: () => void,
  enabled: boolean,
): React.RefObject<HTMLDivElement> {
  const pullRef = useRef<HTMLDivElement>(null);
  const springY = useRef(0);
  const targetY = useRef(0);
  const rafId = useRef<number>(0);
  const triggered = useRef(false);
  const isSettling = useRef(false);

  const animate = useCallback(() => {
    const diff = targetY.current - springY.current;
    springY.current += diff * SPRING_STIFFNESS;

    if (pullRef.current) {
      pullRef.current.style.height = `${Math.max(0, springY.current)}px`;
      const progress = Math.min(springY.current / SPRING_TRIGGER_PX, 1);
      pullRef.current.style.opacity = String(progress);
    }

    if (Math.abs(diff) > 0.5) {
      rafId.current = requestAnimationFrame(animate);
    } else {
      springY.current = targetY.current;
      if (pullRef.current) {
        pullRef.current.style.height = `${Math.max(0, targetY.current)}px`;
      }
      isSettling.current = false;
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const onScroll = () => {
      const scrolled = window.scrollY;
      const totalH = document.body.scrollHeight;
      const windowH = window.innerHeight;
      const distFromBot = totalH - scrolled - windowH;

      if (distFromBot < 40) {
        const excess = Math.max(0, 40 - distFromBot);
        targetY.current = excess * SPRING_RESISTANCE * 100;

        if (!isSettling.current) {
          isSettling.current = true;
          rafId.current = requestAnimationFrame(animate);
        }

        if (excess > SPRING_TRIGGER_PX && !triggered.current) {
          triggered.current = true;
          onTrigger();
          setTimeout(() => {
            targetY.current = 0;
            triggered.current = false;
            if (!isSettling.current) {
              isSettling.current = true;
              rafId.current = requestAnimationFrame(animate);
            }
          }, 400);
        }
      } else {
        if (targetY.current > 0) {
          targetY.current = 0;
          triggered.current = false;
          if (!isSettling.current) {
            isSettling.current = true;
            rafId.current = requestAnimationFrame(animate);
          }
        }
      }
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(rafId.current);
    };
  }, [enabled, onTrigger, animate]);

  return pullRef as React.RefObject<HTMLDivElement>;
}

// ── Cards ──────────────────────────────────────────────────────────────────

interface NewsCardProps {
  item: any;
  size: "wide" | "narrow";
  revealed: boolean;
}

function NewsCard({ item, size, revealed }: NewsCardProps) {
  const meta = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href = item.articleUrl && item.articleUrl !== "#" ? item.articleUrl : item.sourceUrl;
  const [imgError, setImgError] = useState(false);
  const hasImage = item.imageUrl && !imgError;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        gridColumn: size === "wide" ? "span 2" : "span 1",
        position: "relative",
        display: "block",
        height: size === "wide" ? 260 : 200,
        borderRadius: 12,
        overflow: "hidden",
        textDecoration: "none",
        cursor: "pointer",
        zIndex: 1,
        opacity: revealed ? 1 : 0,
        transform: revealed ? "translateY(0) scale(1)" : "translateY(18px) scale(0.97)",
        transition: "opacity 0.38s ease, transform 0.38s ease",
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.transform = "scale(1.015)";
        (e.currentTarget as HTMLElement).style.zIndex = "2";
        const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.93) 0%,rgba(0,0,0,0.52) 55%,rgba(0,0,0,0.12) 100%)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.transform = "scale(1)";
        (e.currentTarget as HTMLElement).style.zIndex = "1";
        const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.84) 0%,rgba(0,0,0,0.36) 55%,transparent 100%)";
      }}
    >
      {hasImage ? (
        <img
          src={item.imageUrl}
          alt=""
          onError={() => setImgError(true)}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
      ) : (
        <div style={{ position: "absolute", inset: 0, background: FALLBACK_GRADIENTS[item.category] ?? FALLBACK_GRADIENTS["General"] }} />
      )}

      <div className="ov" style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(to top,rgba(0,0,0,0.84) 0%,rgba(0,0,0,0.36) 55%,transparent 100%)",
        transition: "background 0.22s ease",
      }} />

      <div style={{
        position: "absolute", top: 10, left: 10,
        display: "inline-flex", alignItems: "center", gap: "0.28rem",
        fontSize: "0.63rem", fontWeight: 700, color: "#fff",
        background: meta.color + "cc",
        borderRadius: 999, padding: "0.18rem 0.5rem",
        backdropFilter: "blur(6px)",
      }}>
        <meta.Icon size={9} />
        {meta.label}
      </div>

      <ExternalLink size={12} style={{ position: "absolute", top: 12, right: 12, color: "rgba(255,255,255,0.5)" }} />

      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        padding: size === "wide" ? "1rem 1.15rem" : "0.8rem 0.95rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>{item.source}</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)" }}>·</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.4)" }}>{timeAgo(item.publishedAt)}</span>
        </div>

        <h3 style={{
          margin: 0,
          fontSize: size === "wide" ? "0.98rem" : "0.8rem",
          fontWeight: 700, color: "#fff", lineHeight: 1.35,
          display: "-webkit-box",
          WebkitLineClamp: size === "wide" ? 2 : 3,
          WebkitBoxOrient: "vertical" as const,
          overflow: "hidden",
        }}>
          {item.title}
        </h3>

        {size === "wide" && item.aiImpact && (
          <p style={{
            margin: "0.45rem 0 0",
            fontSize: "0.7rem", color: "rgba(255,255,255,0.6)", lineHeight: 1.5,
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

function SkeletonCard({ size }: { size: "wide" | "narrow" }) {
  return (
    <div style={{
      gridColumn: size === "wide" ? "span 2" : "span 1",
      height: size === "wide" ? 260 : 200,
      borderRadius: 12,
      background: "#e0e0e0",
      position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.05) 50%,transparent 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.6s ease-in-out infinite",
      }} />
    </div>
  );
}

function SpringPullZone({
  pullRef,
  loadingMore,
}: {
  pullRef: React.RefObject<HTMLDivElement>;
  loadingMore: boolean;
}) {
  return (
    <div
      ref={pullRef}
      style={{
        height: 0,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity: 0,
        transition: "none",
      }}
    >
      <div style={{
        display: "flex", alignItems: "center", gap: "0.5rem",
        fontSize: "0.7rem", color: "#666", opacity: 0.6,
      }}>
        <RefreshCw
          size={12}
          style={{ animation: loadingMore ? "spin 0.8s linear infinite" : "none" }}
        />
        {loadingMore ? "Loading more…" : "Pull to load more"}
      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────

const NewsPage = () => {
  const navigate = useNavigate();
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem("theme") as "light" | "dark") || "light";
    }
    return "light";
  });

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
  };

  const [filter, setFilter] = useState<string>("All");
  const [offset, setOffset] = useState(0);
  const [allItems, setAllItems] = useState<any[]>([]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [revealedSet, setRevealedSet] = useState<Set<number>>(new Set());
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const prevVisibleCount = useRef(0);

  // TODO: Connect your news agent API here
  // Replace this with your actual API call to fetch news from your agent
  useEffect(() => {
    const fetchNews = async () => {
      setIsLoading(true);
      try {
        // Example: Replace with your actual endpoint
        // const response = await fetch(`/api/news?category=${filter}&offset=${offset}`);
        // const data = await response.json();
        // setAllItems(data.items);
        // setHasMore(data.hasMore);

        // For now, this is a placeholder
        setAllItems([]);
        setHasMore(false);
      } catch (error) {
        console.error("Failed to fetch news:", error);
      }
      setIsLoading(false);
    };
    fetchNews();
  }, [filter, offset]);

  // Reveal cards with stagger
  useEffect(() => {
    const start = prevVisibleCount.current;
    const end = allItems.length;
    prevVisibleCount.current = end;

    if (end <= start) return;

    for (let i = start; i < end; i++) {
      const delay = (i - start) * 60;
      setTimeout(() => {
        setRevealedSet(prev => {
          const newSet = new Set(prev);
          newSet.add(i);
          return newSet;
        });
      }, delay);
    }
  }, [allItems.length]);

  // Reset when filter changes
  useEffect(() => {
    setOffset(0);
    setAllItems([]);
    setRevealedSet(new Set());
    prevVisibleCount.current = 0;
  }, [filter]);

  // Load more
  const loadMore = useCallback(() => {
    if (!hasMore || loadingMore) return;
    setLoadingMore(true);
    setTimeout(() => {
      setOffset(prev => prev + PAGE_SIZE);
      setLoadingMore(false);
    }, 350);
  }, [hasMore, loadingMore]);

  // Refresh
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    setAllItems([]);
    setOffset(0);
    setRevealedSet(new Set());
    prevVisibleCount.current = 0;
    
    // TODO: Call your news agent refresh endpoint here
    // Example: await fetch('/api/news/refresh', { method: 'POST' });
    
    setTimeout(() => {
      setIsRefreshing(false);
      setIsLoading(true);
    }, 500);
  }, []);

  // Spring pull hook
  const pullRef = useSpringPull(loadMore, hasMore && !isLoading && !loadingMore) as React.RefObject<HTMLDivElement>;

  return (
    <div style={{
      minHeight: "100vh",
      background: theme === "dark" ? "#0a0a0a" : "#ffffff",
      padding: "2rem 2.5rem 3rem",
      transition: "background 0.15s ease",
      opacity: 1,
      animation: "fadeIn 0.5s ease-out",
      color: theme === "dark" ? "#ffffff" : "#000000",
    }}>
      <style>{`
        @keyframes shimmer {
          0%   { background-position: -200% 0 }
          100% { background-position:  200% 0 }
        }
        @keyframes spin {
          from { transform: rotate(0deg) }
          to   { transform: rotate(360deg) }
        }
        @keyframes fadeIn {
          from { opacity: 0 }
          to { opacity: 1 }
        }
        @media (max-width: 768px) {
          .news-grid { grid-template-columns: 1fr !important; }
          .news-grid > * { grid-column: span 1 !important; height: 180px !important; }
        }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button
            onClick={() => navigate(-1)}
            style={{
              display: "inline-flex", alignItems: "center",
              color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              textDecoration: "none", opacity: 0.6,
              background: "none", border: "none", cursor: "pointer", padding: 0,
            }}
          >
            <ArrowLeft size={15} />
          </button>
          <div>
            <h1 style={{
              margin: 0,
              fontSize: "1.35rem",
              fontWeight: 800,
              color: theme === "dark" ? "#ffffff" : "#000000",
              lineHeight: 1.2,
            }}>
              Industry Briefing
            </h1>
            <p style={{
              margin: "0.1rem 0 0",
              fontSize: "0.72rem",
              color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              opacity: 0.65,
            }}>
              Live telecom &amp; construction news
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
          {!isLoading && allItems.length > 0 && (
            <span style={{
              fontSize: "0.65rem",
              color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              opacity: 0.4,
            }}>
              {allItems.length}
            </span>
          )}
          
          <button
            onClick={handleRefresh}
            disabled={isRefreshing || isLoading}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              height: 36,
              borderRadius: 8,
              background: theme === "dark" ? "#1a1a1a" : "#f0f0f0",
              border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              cursor: isRefreshing || isLoading ? "not-allowed" : "pointer",
              transition: "all 0.15s ease",
              color: theme === "dark" ? "#94a3b8" : "#666",
              opacity: isRefreshing || isLoading ? 0.5 : 1,
            }}
            title="Refresh news"
          >
            <RefreshCw
              size={16}
              style={{ animation: isRefreshing ? "spin 0.8s linear infinite" : "none" }}
            />
          </button>

          <button
            onClick={toggleTheme}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              height: 36,
              borderRadius: 8,
              background: theme === "dark" ? "#1a1a1a" : "#f0f0f0",
              border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              cursor: "pointer",
              transition: "all 0.15s ease",
              color: theme === "dark" ? "#fbbf24" : "#3b9eff",
            }}
            title="Toggle theme"
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      {/* Filter pills */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.38rem", marginBottom: "1.35rem", alignItems: "center" }}>
        {ALL_FILTERS.map(cat => {
          const meta = CATEGORY_META[cat];
          const active = filter === cat;
          const color = meta?.color ?? "#3b9eff";
          return (
            <button key={cat} onClick={() => setFilter(cat)} style={{
              display: "inline-flex", alignItems: "center", gap: "0.25rem",
              fontSize: "0.68rem", fontWeight: 600,
              padding: "0.22rem 0.7rem", borderRadius: 999, cursor: "pointer",
              border: `1px solid ${active ? color + "88" : theme === "dark" ? "#333" : "#ddd"}`,
              color: active ? color : theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)",
              background: active ? color + "18" : "transparent",
              transition: "all 0.12s",
            }}>
              {meta?.Icon && <meta.Icon size={9} />}
              {cat}
            </button>
          );
        })}
      </div>

      {/* Bento grid with dense layout */}
      {isLoading && allItems.length === 0 ? (
        <div className="news-grid" style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "0.8rem",
          gridAutoFlow: "dense",
        }}>
          {[...Array(6)].map((_, i) => (
            <SkeletonCard key={i} size={SIZE_PATTERN[i % 3]} />
          ))}
        </div>
      ) : allItems.length === 0 ? (
        <div style={{ textAlign: "center", padding: "5rem 0", color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", opacity: 0.45 }}>
          <Newspaper size={30} style={{ margin: "0 auto 0.6rem", display: "block" }} />
          <p style={{ margin: 0, fontSize: "0.82rem" }}>No articles in this category. Connect your news agent to start fetching articles.</p>
          <button onClick={() => setFilter("All")} style={{
            marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff",
            background: "none", border: "none", cursor: "pointer",
          }}>
            View all →
          </button>
        </div>
      ) : (
        <>
          <div className="news-grid" style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "0.8rem",
            gridAutoFlow: "dense",
          }}>
            {allItems.map((item, i) => (
              <NewsCard
                key={item.id}
                item={item}
                size={SIZE_PATTERN[i % 3]}
                revealed={revealedSet.has(i)}
              />
            ))}

            {loadingMore && [...Array(PAGE_SIZE)].map((_, i) => (
              <SkeletonCard key={`skel-${i}`} size={SIZE_PATTERN[(allItems.length + i) % 3]} />
            ))}
          </div>

          <SpringPullZone pullRef={pullRef} loadingMore={loadingMore} />

          {!hasMore && !loadingMore && allItems.length > PAGE_SIZE && (
            <div style={{
              textAlign: "center", padding: "2rem 0 0",
              fontSize: "0.65rem", color: theme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)", opacity: 0.3,
              letterSpacing: "0.08em",
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
