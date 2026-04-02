import { useState, useEffect } from "react";
import { Radio, Cable, Scale, HardHat, Newspaper, ExternalLink, Zap, Clock, ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import Navbar from "@/components/Navbar";
import BackgroundManager from "@/components/BackgroundManager";
import { getCachedBriefing, prefetchNews, type NewsItem } from "@/lib/newsCache";

// ── Category config ────────────────────────────────────────────────────────

const CATEGORY_META: Record<string, { label: string; Icon: React.ElementType; color: string }> = {
  "5G":           { label: "5G & Wireless", Icon: Radio,     color: "hsl(200 90% 55%)" },
  "Fiber":        { label: "Fiber",          Icon: Cable,     color: "hsl(160 70% 45%)" },
  "Regulation":   { label: "Regulation",     Icon: Scale,     color: "hsl(38 90% 55%)"  },
  "Construction": { label: "Construction",   Icon: HardHat,   color: "hsl(280 60% 60%)" },
  "General":      { label: "General",        Icon: Newspaper, color: "hsl(var(--primary))" },
};

const ALL_FILTERS = ["All", "5G", "Fiber", "Regulation", "Construction", "General"] as const;

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60)  return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24)  return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// ── Bento card sizes cycle: big, small, small, big, small, small … ─────────
// "big" spans 2 cols on desktop, "small" spans 1.
const SIZE_PATTERN = ["big", "small", "small", "big", "small", "small"];

function NewsCard({ item, big }: { item: NewsItem; big: boolean }) {
  const meta = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href = item.article_url && item.article_url !== "#" ? item.article_url : item.source_url;

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        gridColumn: big ? "span 2" : "span 1",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        padding: big ? "1.75rem" : "1.25rem",
        borderRadius: 18,
        border: "1px solid hsl(var(--border))",
        background: "linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--secondary)) 100%)",
        textDecoration: "none",
        position: "relative",
        overflow: "hidden",
        cursor: "pointer",
        transition: "border-color 0.2s, transform 0.15s",
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.borderColor = meta.color + "66";
        (e.currentTarget as HTMLElement).style.transform = "translateY(-2px)";
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.borderColor = "hsl(var(--border))";
        (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
      }}
    >
      {/* top accent bar */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: `linear-gradient(90deg, ${meta.color}, transparent)`,
      }} />

      {/* category + time */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem" }}>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: "0.3rem",
          fontSize: "0.68rem", fontWeight: 700, letterSpacing: "0.04em",
          color: meta.color,
          background: meta.color + "1a",
          border: `1px solid ${meta.color}33`,
          borderRadius: 999, padding: "0.18rem 0.6rem",
        }}>
          <meta.Icon size={10} />
          {meta.label}
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontSize: "0.68rem", color: "hsl(var(--muted-foreground))" }}>
          <Clock size={9} />
          {timeAgo(item.scraped_at)}
        </span>
      </div>

      {/* title */}
      <h3 style={{
        margin: 0,
        fontSize: big ? "1.05rem" : "0.9rem",
        fontWeight: 700,
        lineHeight: 1.4,
        color: "hsl(var(--foreground))",
      }}>
        {item.title}
      </h3>

      {/* AI impact — only on big cards or as truncated on small */}
      {big ? (
        <div style={{
          background: "hsl(var(--primary) / 0.06)",
          border: "1px solid hsl(var(--primary) / 0.12)",
          borderRadius: 10, padding: "0.65rem 0.9rem",
          display: "flex", gap: "0.5rem", alignItems: "flex-start",
        }}>
          <Zap size={12} style={{ color: "hsl(var(--primary))", flexShrink: 0, marginTop: 2 }} />
          <p style={{ margin: 0, fontSize: "0.8rem", color: "hsl(var(--foreground)/0.75)", lineHeight: 1.55 }}>
            {item.ai_impact}
          </p>
        </div>
      ) : (
        <p style={{
          margin: 0, fontSize: "0.78rem", color: "hsl(var(--muted-foreground))",
          lineHeight: 1.5,
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical" as const,
          overflow: "hidden",
        }}>
          {item.raw_summary}
        </p>
      )}

      {/* source + read link */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "auto" }}>
        <span style={{ fontSize: "0.68rem", color: "hsl(var(--muted-foreground))" }}>{item.source}</span>
        <span style={{
          display: "inline-flex", alignItems: "center", gap: "0.25rem",
          fontSize: "0.68rem", fontWeight: 600, color: meta.color,
        }}>
          Read <ExternalLink size={9} />
        </span>
      </div>
    </a>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────

const NewsPage = () => {
  const [items, setItems]   = useState<NewsItem[]>(getCachedBriefing()?.items ?? []);
  const [loading, setLoading] = useState(items.length === 0);
  const [filter, setFilter] = useState<string>("All");

  useEffect(() => {
    if (items.length > 0) return; // already have data from cache
    prefetchNews().then(b => {
      setItems(b.items);
      setLoading(false);
    });
  }, []);

  const filtered = filter === "All" ? items : items.filter(i => i.category === filter);

  return (
    <div className="min-h-screen bg-background overflow-x-hidden" style={{ animation: "bgFadeIn 0.4s ease-out both" }}>
      <style>{`
        @keyframes bgFadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes shimmer { 0%,100% { opacity: 0.4 } 50% { opacity: 0.8 } }
      `}</style>
      <BackgroundManager />
      <Navbar />

      <div className="container mx-auto px-6 pt-28 pb-24 max-w-6xl">

        {/* Header */}
        <div style={{ marginBottom: "2rem" }}>
          <Link to="/" style={{
            display: "inline-flex", alignItems: "center", gap: "0.4rem",
            fontSize: "0.78rem", color: "hsl(var(--muted-foreground))",
            textDecoration: "none", marginBottom: "1.25rem",
          }}>
            <ArrowLeft size={13} /> Back
          </Link>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", flexWrap: "wrap", gap: "0.75rem" }}>
            <div>
              <h1 style={{ margin: 0, fontSize: "2rem", fontWeight: 800, color: "hsl(var(--foreground))", lineHeight: 1.2 }}>
                Industry Briefing
              </h1>
              <p style={{ margin: "0.4rem 0 0", fontSize: "0.875rem", color: "hsl(var(--muted-foreground))" }}>
                AI-curated telecom news with impact analysis
              </p>
            </div>
            <span style={{
              display: "inline-flex", alignItems: "center", gap: "0.35rem",
              fontSize: "0.72rem", fontWeight: 600,
              color: "hsl(var(--primary))",
              background: "hsl(var(--primary) / 0.08)",
              border: "1px solid hsl(var(--primary) / 0.18)",
              borderRadius: 999, padding: "0.3rem 0.8rem",
            }}>
              <Zap size={11} /> {items.length} articles
            </span>
          </div>
        </div>

        {/* Filter pills */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem", marginBottom: "1.75rem" }}>
          {ALL_FILTERS.map(cat => {
            const meta = CATEGORY_META[cat];
            const active = filter === cat;
            const color = meta?.color ?? "hsl(var(--primary))";
            return (
              <button
                key={cat}
                onClick={() => setFilter(cat)}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.3rem",
                  fontSize: "0.72rem", fontWeight: 600,
                  padding: "0.28rem 0.8rem", borderRadius: 999, cursor: "pointer",
                  border: `1px solid ${active ? color : "hsl(var(--border))"}`,
                  color: active ? color : "hsl(var(--muted-foreground))",
                  background: active ? color + "1a" : "hsl(var(--secondary))",
                  transition: "all 0.15s",
                }}
              >
                {meta?.Icon && <meta.Icon size={10} />}
                {cat}
              </button>
            );
          })}
        </div>

        {/* Bento grid */}
        {loading ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} style={{
                gridColumn: i % 3 === 0 ? "span 2" : "span 1",
                height: i % 3 === 0 ? 200 : 160,
                borderRadius: 18,
                background: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                animation: "shimmer 1.5s ease-in-out infinite",
              }} />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "5rem 0", color: "hsl(var(--muted-foreground))" }}>
            <Newspaper size={36} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
            <p style={{ margin: 0 }}>No articles in this category.</p>
            <button onClick={() => setFilter("All")} style={{ marginTop: "0.75rem", fontSize: "0.82rem", color: "hsl(var(--primary))", background: "none", border: "none", cursor: "pointer" }}>
              View all →
            </button>
          </div>
        ) : (
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "1rem",
          }}>
            {filtered.map((item, i) => (
              <NewsCard
                key={item.id}
                item={item}
                big={SIZE_PATTERN[i % SIZE_PATTERN.length] === "big"}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default NewsPage;