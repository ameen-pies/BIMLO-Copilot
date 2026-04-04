import { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, ExternalLink, Zap,
  Radio, Cable, Scale, HardHat, Newspaper, Sun, Moon, Loader2,
  RefreshCcw, X, Sparkles, MessageSquare,
} from "lucide-react";

// ── Config ─────────────────────────────────────────────────────────────────

const API_BASE  = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const PAGE_SIZE = 10;

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

// ── Chat types ───────────────────────────────────────────────────────────────

interface PinnedArticle {
  id: string;
  title: string;
  category: string;
  source: string;
  aiImpact?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  loading?: boolean;
  pinnedArticles?: PinnedArticle[];
}

// ── Grid News Card ───────────────────────────────────────────────────────────

function NewsCard({ item, revealed, theme, size, onPin, isPinned, chatOpen }: {
  item: any;
  revealed: boolean;
  theme: "light" | "dark";
  size: CardSize;
  onPin: (item: any) => void;
  isPinned: boolean;
  chatOpen: boolean;
}) {
  const meta  = CATEGORY_META[item.category] ?? CATEGORY_META["General"];
  const href  = item.articleUrl && item.articleUrl !== "#" ? item.articleUrl : item.sourceUrl;
  const [imgError, setImgError] = useState(false);
  const hasImage    = item.imageUrl && !imgError;
  const hasAiImpact = !!item.aiImpact;

  // Collapse wide/featured spans when chat pushes grid to 3 cols to avoid overflow
  const effectiveSize: CardSize = chatOpen && (size === "wide" || size === "featured") ? "normal" : size;
  const isFeatured  = effectiveSize === "featured";
  const isWide      = effectiveSize === "wide";
  const isTall      = effectiveSize === "tall" || isFeatured;
  const colSpan     = (effectiveSize === "wide" || effectiveSize === "featured") ? 2 : 1;
  const rowSpan     = (effectiveSize === "tall" || effectiveSize === "featured") ? 2 : 1;
  const titleSize   = isFeatured ? "1.08rem" : isWide ? "0.92rem" : "0.82rem";
  const titleClamp  = isFeatured ? 4 : isTall ? 4 : isWide ? 2 : 3;

  return (
    <div
      style={{
        gridColumn: `span ${colSpan}`,
        gridRow:    `span ${rowSpan}`,
        position: "relative",
        borderRadius: 14, overflow: "hidden",
        cursor: "default",
        opacity: revealed ? 1 : 0,
        transform: revealed ? "translateY(0) scale(1)" : "translateY(20px) scale(0.97)",
        transition: "opacity 0.38s ease, transform 0.38s ease, box-shadow 0.22s ease",
        border: isPinned
          ? `1.5px solid ${meta.color}88`
          : `1px solid ${theme === "dark" ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}`,
        boxShadow: isPinned ? `0 0 0 2px ${meta.color}22, inset 0 0 0 1px ${meta.color}22` : "none",
      }}
      onMouseEnter={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1.012)";
        el.style.zIndex    = "10";
        el.style.boxShadow = isPinned
          ? `0 0 0 2px ${meta.color}55, 0 16px 48px rgba(0,0,0,0.35)`
          : "0 16px 48px rgba(0,0,0,0.35)";
        const ov = el.querySelector(".card-ov") as HTMLElement;
        if (ov) ov.style.background = "linear-gradient(to top,rgba(0,0,0,0.95) 0%,rgba(0,0,0,0.55) 50%,rgba(0,0,0,0.15) 100%)";
      }}
      onMouseLeave={e => {
        const el = e.currentTarget as HTMLElement;
        el.style.transform = "scale(1)";
        el.style.zIndex    = "1";
        el.style.boxShadow = isPinned ? `0 0 0 2px ${meta.color}22` : "none";
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

      {/* Top-right: pin + external link */}
      <div style={{ position: "absolute", top: 10, right: 10, display: "flex", gap: "0.35rem", alignItems: "center" }}>
        <button
          onClick={() => onPin(item)}
          title={isPinned ? "Remove from chat" : "Mention in Bimlo chat"}
          style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 24, height: 24, borderRadius: 6,
            background: isPinned ? meta.color + "ee" : "rgba(0,0,0,0.5)",
            backdropFilter: "blur(8px)",
            border: isPinned ? `1px solid ${meta.color}` : "1px solid rgba(255,255,255,0.18)",
            color: "#fff", cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          {isPinned ? <X size={10} strokeWidth={2.5} /> : <MessageSquare size={10} strokeWidth={2} />}
        </button>
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          onClick={e => e.stopPropagation()}
          style={{ display: "flex" }}
        >
          <ExternalLink size={12} style={{ color: "rgba(255,255,255,0.45)" }} />
        </a>
      </div>

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
    </div>
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

// ── News Chat Panel ──────────────────────────────────────────────────────────

function NewsChatPanel({
  theme,
  pinnedArticles,
  onRemovePin,
  onClose,
}: {
  theme: "light" | "dark";
  pinnedArticles: PinnedArticle[];
  onRemovePin: (id: string) => void;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hey! I'm Bimlo. Tap the 💬 icon on any article to pin it here, then ask me anything — trends, impact, comparisons. What's on your mind?",
    },
  ]);
  const [input, setInput] = useState("");
  const [sessionId] = useState(() => Math.random().toString(36).slice(2));
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const dark = theme === "dark";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const buildPrompt = (userInput: string): string => {
    if (pinnedArticles.length === 0) return userInput;
    const ctx = pinnedArticles.map((a, i) =>
      `[${i + 1}] "${a.title}" — ${a.category} · ${a.source}${a.aiImpact ? `\n    AI Impact: ${a.aiImpact}` : ""}`
    ).join("\n");
    return `You are Bimlo, a telecom industry intelligence expert. The user has pinned these articles for context:\n\n${ctx}\n\nUser: ${userInput}\n\nAnswer as Bimlo — concise, expert, reference the articles by number when relevant. No fluff.`;
  };

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      pinnedArticles: pinnedArticles.length > 0 ? [...pinnedArticles] : undefined,
    };
    const loadingMsg: ChatMessage = {
      id: Date.now().toString() + "-l",
      role: "assistant",
      content: "",
      loading: true,
    };

    setMessages(prev => [...prev, userMsg, loadingMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: buildPrompt(text),
          session_id: sessionId,
          top_k: 3,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const answer = data.answer ?? "Sorry, no response received.";
      setMessages(prev => prev.map(m => m.loading ? { ...m, content: answer, loading: false } : m));
    } catch {
      setMessages(prev => prev.map(m => m.loading
        ? { ...m, content: "Something went wrong. Please try again.", loading: false }
        : m
      ));
    } finally {
      setIsLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, isLoading, pinnedArticles, sessionId]);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div style={{
      width: 350,
      flexShrink: 0,
      display: "flex",
      flexDirection: "column",
      height: "calc(100vh - 5rem)",
      position: "sticky",
      top: "2rem",
      borderRadius: 18,
      overflow: "hidden",
      background: dark ? "#0c0d18" : "#f0f1f8",
      border: `1px solid ${dark ? "rgba(59,158,255,0.13)" : "rgba(59,158,255,0.2)"}`,
      boxShadow: dark
        ? "0 0 0 1px rgba(59,158,255,0.05), 0 32px 80px rgba(0,0,0,0.6)"
        : "0 8px 40px rgba(0,0,0,0.1)",
    }}>

      {/* ── Panel header ──────────────────────────────────────────────── */}
      <div style={{
        padding: "0.85rem 1rem 0.7rem",
        background: dark
          ? "linear-gradient(160deg, #101528 0%, #0c0d18 100%)"
          : "linear-gradient(160deg, #e6e8f5 0%, #f0f1f8 100%)",
        borderBottom: `1px solid ${dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)"}`,
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <div style={{
              width: 30, height: 30, borderRadius: 9,
              background: "linear-gradient(135deg, #1d6cf6, #7c3aed)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 4px 12px rgba(29,108,246,0.35)",
              flexShrink: 0,
            }}>
              <Sparkles size={14} color="#fff" />
            </div>
            <div>
              <div style={{ fontSize: "0.8rem", fontWeight: 700, color: dark ? "#fff" : "#111", lineHeight: 1.2 }}>Ask Bimlo</div>
              <div style={{ fontSize: "0.58rem", color: dark ? "rgba(255,255,255,0.38)" : "rgba(0,0,0,0.38)" }}>News Intelligence</div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              width: 26, height: 26, borderRadius: 7, border: "none",
              background: dark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.06)",
              color: dark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.4)",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer",
            }}
          ><X size={12} /></button>
        </div>

        {/* Pinned article chips */}
        <div style={{ marginTop: "0.55rem", minHeight: 20 }}>
          {pinnedArticles.length === 0 ? (
            <span style={{ fontSize: "0.59rem", color: dark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.3)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
              <MessageSquare size={9} />
              Tap <MessageSquare size={9} /> on a card to add context
            </span>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.28rem" }}>
              {pinnedArticles.map(a => {
                const m = CATEGORY_META[a.category] ?? CATEGORY_META["General"];
                return (
                  <span
                    key={a.id}
                    style={{
                      display: "inline-flex", alignItems: "center", gap: "0.22rem",
                      fontSize: "0.58rem", fontWeight: 600,
                      color: m.color,
                      background: m.color + "18",
                      border: `1px solid ${m.color}40`,
                      borderRadius: 999,
                      padding: "0.14rem 0.35rem 0.14rem 0.45rem",
                      maxWidth: 148,
                    }}
                  >
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 110 }}>{a.title}</span>
                    <button
                      onClick={() => onRemovePin(a.id)}
                      style={{ background: "none", border: "none", padding: 0, cursor: "pointer", color: m.color, display: "flex", lineHeight: 1, flexShrink: 0 }}
                    ><X size={8} /></button>
                  </span>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Messages ──────────────────────────────────────────────────── */}
      <div style={{
        flex: 1, overflowY: "auto",
        padding: "0.85rem 0.9rem",
        display: "flex", flexDirection: "column", gap: "0.65rem",
        scrollbarWidth: "thin",
        scrollbarColor: dark ? "rgba(255,255,255,0.08) transparent" : "rgba(0,0,0,0.08) transparent",
      }}>
        {messages.map(msg => (
          <div key={msg.id} style={{ display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start", gap: "0.22rem" }}>
            {/* Pinned chips above user message */}
            {msg.role === "user" && msg.pinnedArticles && msg.pinnedArticles.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.22rem", justifyContent: "flex-end", maxWidth: "90%" }}>
                {msg.pinnedArticles.map(a => {
                  const m = CATEGORY_META[a.category] ?? CATEGORY_META["General"];
                  return (
                    <span key={a.id} style={{
                      fontSize: "0.54rem", fontWeight: 600,
                      color: m.color, background: m.color + "15",
                      border: `1px solid ${m.color}30`, borderRadius: 999,
                      padding: "0.1rem 0.38rem",
                      maxWidth: 110, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{a.title}</span>
                  );
                })}
              </div>
            )}
            <div style={{
              maxWidth: "90%",
              padding: msg.role === "user" ? "0.5rem 0.72rem" : "0.6rem 0.78rem",
              borderRadius: msg.role === "user" ? "13px 13px 3px 13px" : "3px 13px 13px 13px",
              background: msg.role === "user"
                ? "linear-gradient(135deg, #1d6cf6 0%, #7c3aed 100%)"
                : dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
              border: msg.role === "assistant"
                ? `1px solid ${dark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)"}`
                : "none",
              fontSize: "0.73rem", lineHeight: 1.55,
              color: msg.role === "user" ? "#fff" : dark ? "rgba(255,255,255,0.82)" : "rgba(0,0,0,0.78)",
              whiteSpace: "pre-wrap",
            }}>
              {msg.loading ? (
                <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
                  {[0, 1, 2].map(i => (
                    <span key={i} style={{
                      width: 5, height: 5, borderRadius: "50%",
                      background: "#3b9eff", display: "inline-block",
                      animation: `dotBounce 1.2s ease-in-out ${i * 0.18}s infinite`,
                    }} />
                  ))}
                </span>
              ) : msg.content}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Input ─────────────────────────────────────────────────────── */}
      <div style={{
        padding: "0.6rem 0.7rem 0.7rem",
        borderTop: `1px solid ${dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)"}`,
        background: dark ? "rgba(0,0,0,0.25)" : "rgba(255,255,255,0.6)",
        flexShrink: 0,
      }}>
        <div style={{
          display: "flex", alignItems: "flex-end", gap: "0.45rem",
          background: dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
          border: `1px solid ${dark ? "rgba(255,255,255,0.09)" : "rgba(0,0,0,0.09)"}`,
          borderRadius: 12, padding: "0.48rem 0.48rem 0.48rem 0.7rem",
        }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              pinnedArticles.length > 0
                ? `Ask about ${pinnedArticles.length === 1 ? "this article" : `these ${pinnedArticles.length} articles`}…`
                : "Ask about the news…"
            }
            rows={1}
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              resize: "none", fontSize: "0.73rem", lineHeight: 1.5,
              color: dark ? "#fff" : "#000", fontFamily: "inherit",
              maxHeight: 76, overflowY: "auto", scrollbarWidth: "none",
            }}
            onInput={e => {
              const el = e.target as HTMLTextAreaElement;
              el.style.height = "auto";
              el.style.height = Math.min(el.scrollHeight, 76) + "px";
            }}
          />
          {/* Ask Bimlo — icon-only gradient button */}
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            title="Ask Bimlo"
            style={{
              width: 30, height: 30, borderRadius: 9, border: "none", flexShrink: 0,
              background: input.trim() && !isLoading
                ? "linear-gradient(135deg, #1d6cf6 0%, #7c3aed 100%)"
                : dark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)",
              cursor: input.trim() && !isLoading ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center",
              opacity: input.trim() && !isLoading ? 1 : 0.38,
              transition: "all 0.15s",
              boxShadow: input.trim() && !isLoading ? "0 4px 12px rgba(29,108,246,0.4)" : "none",
            }}
          >
            {isLoading
              ? <Loader2 size={13} color="rgba(255,255,255,0.6)" style={{ animation: "spin 1s linear infinite" }} />
              : <Sparkles size={13} color={input.trim() ? "#fff" : dark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.3)"} />
            }
          </button>
        </div>
        <p style={{
          margin: "0.28rem 0 0", fontSize: "0.54rem", textAlign: "center",
          color: dark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.22)",
        }}>
          Enter · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

const NewsPage = () => {
  const navigate = useNavigate();

  const [theme, setTheme] = useState<"light" | "dark">(() =>
    (typeof window !== "undefined" && (localStorage.getItem("theme") as "light" | "dark")) || "dark"
  );
  const toggleTheme = () => {
    const t = theme === "light" ? "dark" : "light";
    setTheme(t); localStorage.setItem("theme", t);
  };

  const [filter, setFilter]               = useState<string>("All");
  const [allArticles, setAllArticles]     = useState<any[]>([]);
  const [visibleCount, setVisibleCount]   = useState(PAGE_SIZE);
  const [revealedSet, setRevealedSet]     = useState<Set<number>>(new Set());
  const prevVisibleCount                  = useRef(0);
  const renderedIds                       = useRef<Set<string>>(new Set());
  const [currentPage, setCurrentPage]     = useState(0);
  const [totalPages, setTotalPages]       = useState(0);
  const [hasMore, setHasMore]             = useState(true);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [fetchingMore, setFetchingMore]   = useState(false);
  const [isRefreshing, setIsRefreshing]   = useState(false);
  const [newRunAvailable, setNewRunAvailable] = useState(false);
  const knownRunAt                        = useRef<string | null>(null);
  const loadMoreGuard                     = useRef(false);

  // ── Chat state ─────────────────────────────────────────────────────────
  const [chatOpen, setChatOpen]             = useState(false);
  const [pinnedArticles, setPinnedArticles] = useState<PinnedArticle[]>([]);

  const handlePinArticle = useCallback((item: any) => {
    const id = item.id ?? item.articleUrl ?? item.title ?? String(Math.random());
    setPinnedArticles(prev => {
      if (prev.find(a => a.id === id)) return prev.filter(a => a.id !== id);
      return [...prev.slice(-4), { id, title: item.title, category: item.category, source: item.source, aiImpact: item.aiImpact }];
    });
    setChatOpen(true);
  }, []);

  const handleRemovePin = useCallback((id: string) => {
    setPinnedArticles(prev => prev.filter(a => a.id !== id));
  }, []);

  useEffect(() => {
    document.documentElement.classList.add("scrollbar-thin");
    return () => document.documentElement.classList.remove("scrollbar-thin");
  }, []);

  const addArticles = useCallback((items: any[]) => {
    const fresh = items.map(normalize).filter(a => {
      const id = a.id ?? a.articleUrl ?? a.title ?? String(Math.random());
      if (renderedIds.current.has(id)) return false;
      renderedIds.current.add(id); return true;
    });
    if (fresh.length === 0) return;
    setAllArticles(prev => [...prev, ...fresh]);
  }, []);

  const fetchPage = useCallback(async (pageNum: number): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/news/pages/${pageNum}`);
      if (!res.ok) return false;
      const data = await res.json();
      addArticles(data.items ?? []);
      setCurrentPage(pageNum);
      setHasMore(data.has_more ?? false);
      return true;
    } catch (e) { console.error(`fetchPage(${pageNum}) error:`, e); return false; }
  }, [addArticles]);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      setIsInitialLoading(true);
      try {
        const metaRes = await fetch(`${API_BASE}/api/news/meta`);
        if (!metaRes.ok) throw new Error("No cache yet");
        const meta = await metaRes.json();
        if (!cancelled) { setTotalPages(meta.total_pages ?? 0); knownRunAt.current = meta.run_at ?? null; }
        await fetchPage(0);
      } catch (e) {
        console.error("Initial load error:", e);
        if (!cancelled) pollUntilReady();
      } finally {
        if (!cancelled) setIsInitialLoading(false);
      }
    }
    init();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollUntilReady = useCallback(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/news/status`);
        const data = await res.json();
        if (!data.running && data.total_pages > 0) {
          clearInterval(interval);
          const metaRes = await fetch(`${API_BASE}/api/news/meta`);
          const meta = await metaRes.json();
          setTotalPages(meta.total_pages ?? 0);
          knownRunAt.current = meta.run_at ?? null;
          await fetchPage(0);
          setIsInitialLoading(false);
        }
      } catch (_) {}
    }, 8000);
    return () => clearInterval(interval);
  }, [fetchPage]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/news/status`);
        const data = await res.json();
        if (knownRunAt.current !== null && data.last_run_at !== null && data.last_run_at !== knownRunAt.current)
          setNewRunAvailable(true);
      } catch (_) {}
    }, 60_000);
    return () => clearInterval(interval);
  }, []);

  const resetAndReload = useCallback(async () => {
    renderedIds.current.clear();
    setAllArticles([]); setVisibleCount(PAGE_SIZE); setRevealedSet(new Set());
    prevVisibleCount.current = 0; setCurrentPage(0); setHasMore(true);
    setNewRunAvailable(false); setIsInitialLoading(true);
    try {
      const metaRes = await fetch(`${API_BASE}/api/news/meta`);
      const meta = await metaRes.json();
      setTotalPages(meta.total_pages ?? 0); knownRunAt.current = meta.run_at ?? null;
      await fetchPage(0);
    } catch (e) { console.error("resetAndReload error:", e); }
    finally { setIsInitialLoading(false); }
  }, [fetchPage]);

  const handleRefresh = useCallback(async () => {
    if (isRefreshing || fetchingMore) return;
    setIsRefreshing(true);
    try {
      await fetch(`${API_BASE}/api/news/trigger?force=true`, { method: "POST" });
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/news/status`);
          const data = await res.json();
          if (!data.running) { clearInterval(poll); setIsRefreshing(false); await resetAndReload(); }
        } catch (_) {}
      }, 5000);
    } catch (e) { console.error("Refresh trigger error:", e); setIsRefreshing(false); }
  }, [isRefreshing, fetchingMore, resetAndReload]);

  const filtered = filter === "All" ? allArticles : allArticles.filter(a => a.category === filter);
  const visible  = filtered.slice(0, visibleCount);

  useEffect(() => {
    const start = prevVisibleCount.current, end = visible.length;
    if (end <= start) { prevVisibleCount.current = end; return; }
    prevVisibleCount.current = end;
    for (let i = start; i < end; i++) {
      setTimeout(() => { setRevealedSet(prev => { const s = new Set(prev); s.add(i); return s; }); }, (i - start) * 55);
    }
  }, [visible.length]);

  useEffect(() => { setRevealedSet(new Set()); prevVisibleCount.current = 0; setVisibleCount(PAGE_SIZE); }, [filter]);

  const loadMore = useCallback(async () => {
    if (loadMoreGuard.current || fetchingMore || isInitialLoading) return;
    if (visibleCount < filtered.length) { setVisibleCount(c => c + PAGE_SIZE); return; }
    if (!hasMore) return;
    const nextPage = currentPage + 1;
    if (nextPage >= totalPages) { setHasMore(false); return; }
    loadMoreGuard.current = true; setFetchingMore(true);
    try { const ok = await fetchPage(nextPage); if (ok) setVisibleCount(c => c + PAGE_SIZE); }
    finally { setFetchingMore(false); loadMoreGuard.current = false; }
  }, [loadMoreGuard, fetchingMore, isInitialLoading, visibleCount, filtered.length, hasMore, currentPage, totalPages, fetchPage]);

  const sentinelRef = useInfiniteScroll(loadMore, !isInitialLoading && !fetchingMore && (hasMore || visibleCount < filtered.length));

  const gridCols = chatOpen ? 3 : 4;
  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
    gridAutoRows: "160px",
    gap: "0.75rem",
    gridAutoFlow: "dense",
  };

  const pinnedIds = new Set(pinnedArticles.map(a => a.id));

  return (
    <div style={{
      minHeight: "100vh",
      background: theme === "dark" ? "#07080f" : "#f5f4fb",
      padding: "2rem 2.5rem 3rem",
      transition: "background 0.15s ease",
      color: theme === "dark" ? "#ffffff" : "#000000",
    }}>
      <style>{`
        @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
        @keyframes spin    { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
        @keyframes dotBounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40%            { transform: translateY(-5px); opacity: 1; }
        }
        @media (max-width: 1024px) { .news-grid { grid-template-columns: repeat(2, 1fr) !important; } }
        @media (max-width: 640px) {
          .news-grid { grid-template-columns: 1fr !important; grid-auto-rows: 200px !important; }
          .news-grid > * { grid-column: span 1 !important; grid-row: span 1 !important; }
        }
      `}</style>

      {/* ── Header ────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button onClick={() => navigate(-1)} style={{ display: "inline-flex", alignItems: "center", color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", background: "none", border: "none", cursor: "pointer", padding: 0, opacity: 0.6 }}>
            <ArrowLeft size={15} />
          </button>
          <div>
            <h1 style={{ margin: 0, fontSize: "1.35rem", fontWeight: 800, color: theme === "dark" ? "#fff" : "#000", lineHeight: 1.2 }}>Industry Briefing</h1>
            <p style={{ margin: "0.1rem 0 0", fontSize: "0.72rem", color: theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", opacity: 0.65 }}>Telecom &amp; construction intelligence</p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
          {!isInitialLoading && allArticles.length > 0 && (
            <span style={{ fontSize: "0.65rem", color: theme === "dark" ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.35)", display: "flex", alignItems: "center", gap: "0.3rem" }}>
              {visible.length} / {filtered.length}
              {fetchingMore && <Loader2 size={10} style={{ animation: "spin 1s linear infinite", color: "#3b9eff" }} />}
            </span>
          )}

          {/* Ask Bimlo toggle */}
          <button
            onClick={() => setChatOpen(o => !o)}
            title={chatOpen ? "Close chat" : "Ask Bimlo about the news"}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.4rem",
              height: 36, padding: "0 0.9rem", borderRadius: 9,
              background: chatOpen
                ? "linear-gradient(135deg, #1d6cf6 0%, #7c3aed 100%)"
                : theme === "dark" ? "#1a1a1a" : "#e8e8e8",
              border: chatOpen ? "1px solid rgba(124,58,237,0.4)" : `1px solid ${theme === "dark" ? "#333" : "#ddd"}`,
              color: chatOpen ? "#fff" : theme === "dark" ? "#94a3b8" : "#666",
              cursor: "pointer", fontSize: "0.7rem", fontWeight: 600,
              transition: "all 0.18s",
              boxShadow: chatOpen ? "0 4px 16px rgba(29,108,246,0.3)" : "none",
            }}
          >
            <Sparkles size={13} />
            {chatOpen ? "Close" : "Ask Bimlo"}
            {pinnedArticles.length > 0 && !chatOpen && (
              <span style={{
                background: "#3b9eff", color: "#fff",
                borderRadius: 999, fontSize: "0.55rem", fontWeight: 700,
                padding: "0.05rem 0.38rem", lineHeight: 1.5,
              }}>{pinnedArticles.length}</span>
            )}
          </button>

          <button onClick={handleRefresh} disabled={isRefreshing || fetchingMore} title="Refresh news" style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: 8, background: theme === "dark" ? "#1a1a1a" : "#e8e8e8", border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`, cursor: isRefreshing || fetchingMore ? "not-allowed" : "pointer", transition: "all 0.15s", color: theme === "dark" ? "#94a3b8" : "#666", opacity: isRefreshing || fetchingMore ? 0.5 : 1 }}>
            <RefreshCw size={16} style={{ animation: isRefreshing ? "spin 0.8s linear infinite" : "none" }} />
          </button>
          <button onClick={toggleTheme} style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: 8, background: theme === "dark" ? "#1a1a1a" : "#e8e8e8", border: `1px solid ${theme === "dark" ? "#333" : "#ddd"}`, cursor: "pointer", transition: "all 0.15s", color: theme === "dark" ? "#fbbf24" : "#3b9eff" }} title="Toggle theme">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      {/* ── Banners ───────────────────────────────────────────────────── */}
      {newRunAvailable && (
        <div onClick={resetAndReload} style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem", padding: "0.55rem 1rem", marginBottom: "1rem", borderRadius: 10, cursor: "pointer", background: theme === "dark" ? "#0d2a1f" : "#d1fae5", border: `1px solid ${theme === "dark" ? "#1a5c3a" : "#6ee7b7"}`, fontSize: "0.75rem", fontWeight: 600, color: theme === "dark" ? "#34d399" : "#065f46" }}>
          <RefreshCcw size={13} /> Fresh articles are ready — click to load them
        </div>
      )}
      {isRefreshing && (
        <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.68rem", fontWeight: 500, marginBottom: "0.8rem", color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.38)" }}>
          <Loader2 size={11} style={{ animation: "spin 1s linear infinite", flexShrink: 0 }} />
          Running Industry Analyst Agent… this takes a few minutes
        </div>
      )}

      {/* ── Filter pills ─────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.38rem", marginBottom: "1.35rem" }}>
        {ALL_FILTERS.map(cat => {
          const meta = CATEGORY_META[cat], active = filter === cat, color = meta?.color ?? "#3b9eff";
          return (
            <button key={cat} onClick={() => setFilter(cat)} style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem", fontSize: "0.68rem", fontWeight: 600, padding: "0.22rem 0.7rem", borderRadius: 999, cursor: "pointer", border: `1px solid ${active ? color + "88" : theme === "dark" ? "#333" : "#ddd"}`, color: active ? color : theme === "dark" ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)", background: active ? color + "18" : "transparent", transition: "all 0.12s" }}>
              {meta?.Icon && <meta.Icon size={9} />}{cat}
            </button>
          );
        })}
      </div>

      {/* ── Main layout: grid + chat panel ───────────────────────────── */}
      <div style={{ display: "flex", gap: "1.2rem", alignItems: "flex-start" }}>

        {/* Grid */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {isInitialLoading ? (
            <div className="news-grid" style={gridStyle}>
              {SIZE_PATTERN.map((size, i) => <SkeletonCard key={i} theme={theme} size={size} />)}
            </div>
          ) : visible.length === 0 ? (
            <div style={{ textAlign: "center", padding: "5rem 0", color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }}>
              <Newspaper size={30} style={{ margin: "0 auto 0.6rem", display: "block" }} />
              <p style={{ margin: 0, fontSize: "0.82rem" }}>
                {totalPages === 0 ? "The Industry Analyst Agent is collecting articles. Check back in a few minutes." : "No articles in this category."}
              </p>
              {totalPages === 0 && <p style={{ margin: "0.4rem 0 0", fontSize: "0.68rem", opacity: 0.5 }}>Polls automatically every 8 seconds…</p>}
              {totalPages > 0 && <button onClick={() => setFilter("All")} style={{ marginTop: "0.5rem", fontSize: "0.75rem", color: "#3b9eff", background: "none", border: "none", cursor: "pointer" }}>Show all →</button>}
            </div>
          ) : (
            <>
              <div className="news-grid" style={gridStyle}>
                {visible.map((item, i) => {
                  const id = item.id ?? item.articleUrl ?? item.title ?? String(i);
                  return (
                    <NewsCard
                      key={item.id ?? item.articleUrl ?? i}
                      item={item}
                      revealed={revealedSet.has(i)}
                      theme={theme}
                      size={getSize(i)}
                      onPin={handlePinArticle}
                      isPinned={pinnedIds.has(id)}
                      chatOpen={chatOpen}
                    />
                  );
                })}
                {fetchingMore && SIZE_PATTERN.slice(0, 4).map((size, i) => <SkeletonCard key={`skel-${i}`} theme={theme} size={size} />)}
              </div>
              <div ref={sentinelRef} style={{ height: 1, marginTop: "0.5rem" }} />
              {fetchingMore && (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.45rem", padding: "1rem 0", fontSize: "0.7rem", color: theme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }}>
                  <Loader2 size={12} style={{ animation: "spin 1s linear infinite" }} /> Loading more articles…
                </div>
              )}
              {!hasMore && !fetchingMore && allArticles.length > 0 && (
                <p style={{ textAlign: "center", padding: "1.2rem 0", fontSize: "0.65rem", color: theme === "dark" ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.25)" }}>
                  You've reached the end of this cycle's briefing.
                </p>
              )}
            </>
          )}
        </div>

        {/* Chat panel — slides in from right */}
        <div style={{
          width: chatOpen ? 350 : 0,
          overflow: "hidden",
          opacity: chatOpen ? 1 : 0,
          transform: chatOpen ? "translateX(0)" : "translateX(20px)",
          transition: "width 0.32s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease, transform 0.25s ease",
          flexShrink: 0,
          pointerEvents: chatOpen ? "auto" : "none",
        }}>
          {chatOpen && (
            <NewsChatPanel
              theme={theme}
              pinnedArticles={pinnedArticles}
              onRemovePin={handleRemovePin}
              onClose={() => setChatOpen(false)}
            />
          )}
        </div>

      </div>
    </div>
  );
};

export default NewsPage;