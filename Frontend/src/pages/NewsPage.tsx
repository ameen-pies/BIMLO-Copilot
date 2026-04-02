import { useState, useEffect } from "react";
import { ArrowLeft, RefreshCw, ExternalLink, Zap, Radio, Cable, Scale, HardHat, Newspaper } from "lucide-react";
import { Link } from "react-router-dom";
import { getCachedBriefing, prefetchNews, type NewsItem } from "@/lib/newsCache";

// ── Category config ────────────────────────────────────────────────────────

const CATEGORY_META: Record<string, { label: string; Icon: React.ElementType; color: string }> = {
  "5G":           { label: "5G",          Icon: Radio,     color: "#3b9eff" },
  "Fiber":        { label: "Fiber",        Icon: Cable,     color: "#34d399" },
  "Regulation":   { label: "Regulation",   Icon: Scale,     color: "#fbbf24" },
  "Construction": { label: "Construction", Icon: HardHat,   color: "#a78bfa" },
  "General":      { label: "General",      Icon: Newspaper, color: "#94a3b8" },
};

const ALL_FILTERS = ["All", "5G", "Fiber", "Regulation", "Construction", "General"] as const;

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

const FALLBACK_GRADIENTS: Record<string, string> = {
  "5G":           "linear-gradient(135deg, #0f2540 0%, #1a4a7a 100%)",
  "Fiber":        "linear-gradient(135deg, #0d2a1f 0%, #1a5c3a 100%)",
  "Regulation":   "linear-gradient(135deg, #2a1f06 0%, #5c440a 100%)",
  "Construction": "linear-gradient(135deg, #1f0d2a 0%, #3d1a5c 100%)",
  "General":      "linear-gradient(135deg, #1a1a2e 0%, #2d2d4e 100%)",
};

// Repeating 3-col bento: wide(2), narrow(1), narrow(1)
const SIZE_PATTERN = ["wide", "narrow", "narrow"];

function NewsCard({ item, size }: { item: NewsItem; size: "wide" | "narrow" }) {
  const meta = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href = item.article_url && item.article_url !== "#" ? item.article_url : item.source_url;
  const [imgError, setImgError] = useState(false);
  const hasImage = item.image_url && !imgError;

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
        transition: "transform 0.2s ease",
        zIndex: 1,
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.transform = "scale(1.015)";
        (e.currentTarget as HTMLElement).style.zIndex = "2";
        const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top, rgba(0,0,0,0.93) 0%, rgba(0,0,0,0.52) 55%, rgba(0,0,0,0.12) 100%)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.transform = "scale(1)";
        (e.currentTarget as HTMLElement).style.zIndex = "1";
        const ov = (e.currentTarget as HTMLElement).querySelector(".ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top, rgba(0,0,0,0.84) 0%, rgba(0,0,0,0.36) 55%, transparent 100%)";
      }}
    >
      {/* Background */}
      {hasImage ? (
        <img
          src={item.image_url!}
          alt=""
          onError={() => setImgError(true)}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
      ) : (
        <div style={{ position: "absolute", inset: 0, background: FALLBACK_GRADIENTS[item.category] ?? FALLBACK_GRADIENTS["General"] }} />
      )}

      {/* Overlay */}
      <div className="ov" style={{
        position: "absolute", inset: 0,
        background: "linear-gradient(to top, rgba(0,0,0,0.84) 0%, rgba(0,0,0,0.36) 55%, transparent 100%)",
        transition: "background 0.22s ease",
      }} />

      {/* Category badge — top left */}
      <div style={{
        position: "absolute", top: 10, left: 10,
        display: "inline-flex", alignItems: "center", gap: "0.28rem",
        fontSize: "0.63rem", fontWeight: 700,
        color: "#fff",
        background: meta.color + "cc",
        borderRadius: 999, padding: "0.18rem 0.5rem",
        backdropFilter: "blur(6px)",
      }}>
        <meta.Icon size={9} />
        {meta.label}
      </div>

      {/* External link — top right */}
      <ExternalLink size={12} style={{
        position: "absolute", top: 12, right: 12,
        color: "rgba(255,255,255,0.5)",
      }} />

      {/* Bottom text */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        padding: size === "wide" ? "1rem 1.15rem" : "0.8rem 0.95rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.35rem" }}>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>{item.source}</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)" }}>·</span>
          <span style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.4)" }}>{timeAgo(item.scraped_at)}</span>
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

        {size === "wide" && item.ai_impact && (
          <p style={{
            margin: "0.45rem 0 0",
            fontSize: "0.7rem", color: "rgba(255,255,255,0.6)", lineHeight: 1.5,
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical" as const,
            overflow: "hidden",
          }}>
            <Zap size={9} style={{ display: "inline", marginRight: 3, color: "#fbbf24", verticalAlign: "middle" }} />
            {item.ai_impact}
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
      background: "hsl(var(--secondary))",
      position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.6s ease-in-out infinite",
      }} />
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

const NewsPage = () => {
  const [items, setItems]           = useState<NewsItem[]>(getCachedBriefing()?.items ?? []);
  const [loading, setLoading]       = useState(items.length === 0);
  const [filter, setFilter]         = useState<string>("All");
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    if (items.length > 0) return;
    prefetchNews().then(b => { setItems(b.items); setLoading(false); });
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    const b = await prefetchNews();
    setItems(b.items);
    setRefreshing(false);
  };

  const filtered = filter === "All" ? items : items.filter(i => i.category === filter);

  return (
    <div style={{ minHeight: "100vh", background: "transparent", padding: "2rem 2.5rem 3rem" }}>
      <style>{`
        @keyframes shimmer {
          0%   { background-position: -200% 0 }
          100% { background-position:  200% 0 }
        }
        @media (max-width: 768px) {
          .news-grid { grid-template-columns: 1fr !important; }
          .news-grid > * { grid-column: span 1 !important; height: 180px !important; }
        }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <Link to="/" style={{
            display: "inline-flex", alignItems: "center",
            color: "hsl(var(--muted-foreground))", textDecoration: "none", opacity: 0.6,
          }}>
            <ArrowLeft size={15} />
          </Link>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.35rem", fontWeight: 800, color: "hsl(var(--foreground))", lineHeight: 1.2 }}>
              Industry Briefing
            </h1>
            <p style={{ margin: "0.1rem 0 0", fontSize: "0.72rem", color: "hsl(var(--muted-foreground))", opacity: 0.65 }}>
              Live telecom &amp; construction news
            </p>
          </div>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing || loading}
          style={{
            display: "inline-flex", alignItems: "center", gap: "0.35rem",
            fontSize: "0.7rem", fontWeight: 600,
            color: "hsl(var(--muted-foreground))",
            background: "hsl(var(--secondary))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8, padding: "0.38rem 0.8rem",
            cursor: refreshing || loading ? "not-allowed" : "pointer",
            opacity: refreshing || loading ? 0.5 : 1,
          }}
        >
          <RefreshCw size={11} style={{ animation: refreshing ? "spin 1s linear infinite" : "none" }} />
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
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
              border: `1px solid ${active ? color + "88" : "hsl(var(--border))"}`,
              color: active ? color : "hsl(var(--muted-foreground))",
              background: active ? color + "18" : "transparent",
              transition: "all 0.12s",
            }}>
              {meta?.Icon && <meta.Icon size={9} />}
              {cat}
            </button>
          );
        })}
        {!loading && (
          <span style={{ marginLeft: "auto", fontSize: "0.65rem", color: "hsl(var(--muted-foreground))", opacity: 0.45 }}>
            {filtered.length} articles
          </span>
        )}
      </div>

      {/* Bento grid */}
      {loading ? (
        <div className="news-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.8rem" }}>
          {[...Array(6)].map((_, i) => (
            <SkeletonCard key={i} size={SIZE_PATTERN[i % 3] as "wide" | "narrow"} />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div style={{ textAlign: "center", padding: "5rem 0", color: "hsl(var(--muted-foreground))", opacity: 0.45 }}>
          <Newspaper size={30} style={{ margin: "0 auto 0.6rem", display: "block" }} />
          <p style={{ margin: 0, fontSize: "0.82rem" }}>No articles in this category.</p>
          <button onClick={() => setFilter("All")} style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff", background: "none", border: "none", cursor: "pointer" }}>
            View all →
          </button>
        </div>
      ) : (
        <div className="news-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.8rem" }}>
          {filtered.map((item, i) => (
            <NewsCard key={item.id} item={item} size={SIZE_PATTERN[i % 3] as "wide" | "narrow"} />
          ))}
        </div>
      )}
    </div>
  );
};

export default NewsPage;