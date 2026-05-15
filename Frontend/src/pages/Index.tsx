import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Zap, FileText, MessageSquare, Network, Newspaper, Radio, Cable, Scale, HardHat, TrendingUp, Box, Layers, BrainCircuit, Mail, Send, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import RotatingWords from "@/components/RotatingWords";
import Logo from "@/components/Logo";
import { ElegantShape } from "@/components/ui/shape-landing-hero";
import CardSwap, { Card } from "@/components/CardSwap";
import AuthModal, { AuthUser } from "@/components/AuthModal";
import { useAuth } from "@/context/AuthContext";
import { useNavigate } from "react-router-dom";
import { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from "react-i18next";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.5, ease: "easeOut" as const },
  }),
};

const CATEGORY_META: Record<string, { label: string; Icon: React.ElementType; color: string; gradient: string }> = {
  "5G":             { label: "5G",           Icon: Radio,        color: "#60a5fa", gradient: "linear-gradient(135deg,#0f1a2e,#1a3050)" },
  "Fiber":          { label: "Fiber",        Icon: Cable,        color: "#93c5fd", gradient: "linear-gradient(135deg,#0a1628,#162540)" },
  "Regulation":     { label: "Regulation",   Icon: Scale,        color: "#bfdbfe", gradient: "linear-gradient(135deg,#0d1e36,#1a2e4a)" },
  "Construction":   { label: "Construction", Icon: HardHat,      color: "#7dd3fc", gradient: "linear-gradient(135deg,#0b1a2c,#142540)" },
  "General":        { label: "General",      Icon: Newspaper,    color: "#94a3b8", gradient: "linear-gradient(135deg,#111827,#1f2937)" },
  "BIM":            { label: "BIM",          Icon: Box,          color: "#38bdf8", gradient: "linear-gradient(135deg,#0c1f2e,#0f3a52)" },
  "Digital Twin":   { label: "Digital Twin", Icon: Layers,       color: "#818cf8", gradient: "linear-gradient(135deg,#0f0c2e,#1a1552)" },
  "AI Construction":{ label: "AI",           Icon: BrainCircuit, color: "#a78bfa", gradient: "linear-gradient(135deg,#160c2e,#2a1050)" },
};

function timeAgo(iso: string) {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

interface Article {
  id: string;
  title: string;
  category: string;
  source: string;
  article_url?: string;
  articleUrl?: string;
  published_at?: string;
  image_url?: string;
  imageUrl?: string;
}

// ── Single ticker card ───────────────────────────────────────────────────────
const TickerCard = ({ article, onLoginRequired, isLoggedIn }: { article: Article; onLoginRequired: () => void; isLoggedIn: boolean }) => {
  const meta  = CATEGORY_META[article.category] ?? CATEGORY_META["General"];
  const Icon  = meta.Icon;
  const href  = article.article_url  ?? article.articleUrl  ?? "#";
  const img   = article.image_url    ?? article.imageUrl    ?? null;
  const [imgErr, setImgErr] = useState(false);

  return (
    <a
      href={undefined}
      onClick={e => { e.preventDefault(); if (isLoggedIn) { window.open(href, "_blank", "noopener,noreferrer"); } else { onLoginRequired(); } }}
      className="ticker-card"
    >
      {/* image or gradient fallback */}
      <div className="ticker-card-img">
        {img && !imgErr
          ? <img src={img} alt="" onError={() => setImgErr(true)} />
          : <div className="ticker-card-img-fallback" style={{ background: meta.gradient }} />
        }
        <span className="ticker-card-cat-badge" style={{ background: meta.color + "22", color: meta.color, borderColor: meta.color + "55" }}>
          <Icon size={10} style={{ display: "inline", marginRight: 3 }} />
          {meta.label}
        </span>
      </div>

      {/* body */}
      <div className="ticker-card-body">
        <p className="ticker-card-title">{article.title}</p>
        <div className="ticker-card-footer">
          <span className="ticker-card-source">{article.source}</span>
          <span className="ticker-card-time">{timeAgo(article.published_at ?? "")}</span>
        </div>
      </div>
    </a>
  );
};

// ── Trending section ─────────────────────────────────────────────────────────
const TrendingSection = ({ onLoginRequired, isLoggedIn }: { onLoginRequired: () => void; isLoggedIn: boolean }) => {
  const { t } = useTranslation();
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading,  setLoading]  = useState(true);
  const trackRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    (async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/news/pages/0`, { cache: "no-store" });
        if (!res.ok) throw new Error("no cache");
        const data = await res.json();
        setArticles(data.items ?? []);
      } catch {
        setArticles([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <section className="trending-section">
      <div id="trending" style={{ position: "relative", top: "-90px" }} />

      {/* centered header */}
      <motion.div
        className="trending-header"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        custom={0}
      >
        <h2 className="trending-title">
          {t("landing.trending_title_prefix")}
          <Logo className="trending-logo" />
          {t("landing.trending_title_suffix")}
        </h2>
        <p className="trending-sub">{t("landing.trending_sub")}</p>
      </motion.div>

      {/* ticker */}
      {loading ? (
        <div className="ticker-skeleton-row">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="ticker-skeleton-card" style={{ animationDelay: `${i * 0.12}s` }} />
          ))}
        </div>
      ) : articles.length === 0 ? (
        <p className="ticker-empty">{t("landing.trending_empty")}</p>
      ) : (
        <motion.div
          className="ticker-wrap"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.15 }}
        >
          <div
            ref={trackRef}
            className="ticker-track"
          >
            {[...articles, ...articles].map((a, i) => (
              <TickerCard key={`${a.id}-${i}`} article={a} onLoginRequired={onLoginRequired} isLoggedIn={isLoggedIn} />
            ))}
          </div>
        </motion.div>
      )}

      {/* CTA */}
      <motion.div
        className="trending-cta"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        variants={fadeUp}
        custom={2}
      >
        <Button
          size="lg"
          onClick={() => isLoggedIn ? navigate("/news") : onLoginRequired()}
          className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold text-base px-10 h-12 gap-2"
        >
          <TrendingUp className="h-4 w-4" />
          {t("landing.open_briefing")}
          <ArrowRight className="h-4 w-4" />
        </Button>
      </motion.div>
    </section>
  );
};

// ── Contact form ─────────────────────────────────────────────────────────────
const API_CONTACT = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const ContactForm = () => {
  const { t } = useTranslation();
  const [name,    setName]    = useState("");
  const [email,   setEmail]   = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [sent,    setSent]    = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !email.trim() || !subject.trim() || !message.trim()) return;
    setSending(true); setError(null);
    try {
      const res = await fetch(`${API_CONTACT}/auth/contact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, subject, message }),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Failed");
      setSent(true);
    } catch (err: any) {
      setError(err.message ?? "Something went wrong. Please try again.");
    } finally {
      setSending(false);
    }
  }

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "10px 14px", borderRadius: 10, fontSize: 14,
    background: "hsl(var(--muted)/0.35)", border: "1px solid hsl(var(--border))",
    color: "hsl(var(--foreground))", outline: "none", boxSizing: "border-box",
    transition: "border-color 0.15s",
    fontFamily: "inherit",
  };

  if (sent) return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center gap-4 py-16 text-center">
      <div style={{ width: 56, height: 56, borderRadius: "50%", background: "#22c55e18", border: "1px solid #22c55e40",
        display: "flex", alignItems: "center", justifyContent: "center" }}>
        <CheckCircle size={26} color="#22c55e" />
      </div>
      <h3 className="font-heading text-xl font-bold text-foreground">{t("landing.form_success_title")}</h3>
      <p className="text-muted-foreground text-sm max-w-xs">{t("landing.form_success_desc")}</p>
      <button onClick={() => { setSent(false); setName(""); setEmail(""); setSubject(""); setMessage(""); }}
        className="text-sm text-primary hover:underline mt-2">{t("landing.form_send_another")}</button>
    </motion.div>
  );

  return (
    <motion.form onSubmit={handleSubmit}
      initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={1}
      style={{
        background: "hsl(var(--card))", border: "1px solid hsl(var(--border))",
        borderRadius: 20, padding: "32px 36px", display: "flex", flexDirection: "column", gap: 18,
      }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", textTransform: "uppercase", letterSpacing: "0.05em" }}>{t("landing.form_name")}</label>
          <input value={name} onChange={e => setName(e.target.value)} placeholder={t("landing.form_name_placeholder")} required style={inputStyle}
            onFocus={e => (e.currentTarget.style.borderColor = "hsl(var(--primary))")}
            onBlur={e  => (e.currentTarget.style.borderColor = "hsl(var(--border))")} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", textTransform: "uppercase", letterSpacing: "0.05em" }}>{t("landing.form_email")}</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder={t("landing.form_email_placeholder")} required style={inputStyle}
            onFocus={e => (e.currentTarget.style.borderColor = "hsl(var(--primary))")}
            onBlur={e  => (e.currentTarget.style.borderColor = "hsl(var(--border))")} />
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", textTransform: "uppercase", letterSpacing: "0.05em" }}>{t("landing.form_subject")}</label>
        <input value={subject} onChange={e => setSubject(e.target.value)} placeholder={t("landing.form_subject_placeholder")} required style={inputStyle}
          onFocus={e => (e.currentTarget.style.borderColor = "hsl(var(--primary))")}
          onBlur={e  => (e.currentTarget.style.borderColor = "hsl(var(--border))")} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label style={{ fontSize: 11, fontWeight: 700, color: "hsl(var(--muted-foreground))", textTransform: "uppercase", letterSpacing: "0.05em" }}>{t("landing.form_message")}</label>
        <textarea value={message} onChange={e => setMessage(e.target.value)} placeholder={t("landing.form_message_placeholder")} required rows={5}
          style={{ ...inputStyle, resize: "vertical", lineHeight: 1.7 }}
          onFocus={e => (e.currentTarget.style.borderColor = "hsl(var(--primary))")}
          onBlur={e  => (e.currentTarget.style.borderColor = "hsl(var(--border))")} />
      </div>
      {error && (
        <div style={{ fontSize: 13, color: "#ef4444", background: "#ef444412", border: "1px solid #ef444430",
          borderRadius: 8, padding: "8px 12px" }}>{error}</div>
      )}
      <button type="submit" disabled={sending || !name.trim() || !email.trim() || !subject.trim() || !message.trim()}
        style={{
          display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
          padding: "11px 24px", borderRadius: 10, fontSize: 14, fontWeight: 700, border: "none",
          cursor: sending ? "not-allowed" : "pointer",
          background: "linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary)/0.8))",
          color: "hsl(var(--primary-foreground))",
          opacity: (!name.trim() || !email.trim() || !subject.trim() || !message.trim()) ? 0.5 : 1,
          transition: "opacity 0.15s",
          boxShadow: "0 4px 14px hsl(var(--primary)/0.3)",
        }}>
        {sending
          ? <><span style={{ width: 14, height: 14, border: "2px solid currentColor", borderTopColor: "transparent", borderRadius: "50%", display: "inline-block", animation: "spin 0.7s linear infinite" }} /> {t("landing.form_sending")}</>
          : <><Send size={14} /> {t("landing.form_send")}</>
        }
      </button>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </motion.form>
  );
};

// ── Main page ────────────────────────────────────────────────────────────────
const Index = () => {
  const { t } = useTranslation();
  const { isLoggedIn } = useAuth();
  const [showScrollArrow, setShowScrollArrow] = useState(true);
  const [isDark, setIsDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );
  const [loginModalOpen, setLoginModalOpen] = useState(false);
  const openLogin = useCallback(() => setLoginModalOpen(true), []);

  useEffect(() => {
    const observer = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains("dark"))
    );
    observer.observe(document.documentElement, { attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const h = () => setShowScrollArrow(window.scrollY < 100);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);

  useEffect(() => {
    document.documentElement.style.scrollBehavior = "smooth";
    document.documentElement.classList.add("scrollbar-thin");
    return () => {
      document.documentElement.style.scrollBehavior = "auto";
      document.documentElement.classList.remove("scrollbar-thin");
    };
  }, []);

  return (
    <>
      <style>{`
        @keyframes bgFadeIn    { from{opacity:0} to{opacity:1} }
        @keyframes liquidFadeIn{ from{opacity:0} to{opacity:0.5} }
        .animate-fade-in { animation: liquidFadeIn 1.2s ease-out 0.3s both; }

        .swap-card {
          background: linear-gradient(135deg,hsl(var(--card)) 0%,hsl(var(--secondary)) 100%);
          border:1px solid hsl(var(--border)) !important;
          border-radius:20px !important; padding:2rem;
          display:flex; flex-direction:column; gap:1rem;
          box-shadow:0 8px 32px rgba(0,0,0,0.25);
        }
        .swap-card-icon {
          width:48px; height:48px; border-radius:12px;
          background:hsl(var(--accent));
          display:flex; align-items:center; justify-content:center; margin-bottom:0.5rem;
        }
        .swap-card h3 { font-size:1.1rem; font-weight:700; color:hsl(var(--foreground)); margin:0; }
        .swap-card p  { font-size:0.875rem; color:hsl(var(--muted-foreground)); line-height:1.6; margin:0; }
        .swap-card .tag {
          display:inline-flex; align-items:center; gap:0.4rem;
          font-size:0.75rem; font-weight:600; color:hsl(var(--primary));
          background:hsl(var(--primary)/0.1); border:1px solid hsl(var(--primary)/0.2);
          border-radius:999px; padding:0.25rem 0.75rem; width:fit-content;
        }

        /* ── Ensure all page content sits above geometric background ── */
        .content-above-bg {
          position: relative;
          z-index: 1;
        }

        /* ── Trending section ── */
        .trending-section {
          padding: 72px 0 96px;
          overflow: hidden;
          position: relative;
          z-index: 1;
        }
        .trending-header {
          text-align: center;
          margin-bottom: 44px;
          padding: 0 24px;
        }
        .trending-title {
          font-family: var(--font-heading, inherit);
          font-size: clamp(1.35rem, 2.5vw, 1.85rem);
          font-weight: 800;
          color: hsl(var(--foreground));
          display: inline-flex;
          align-items: center;
          gap: 9px;
          margin: 0 0 8px;
          line-height: 1;
        }
        .trending-logo {
          width: 30px; height: 30px;
          border-radius: 50%;
          display: inline-block;
          vertical-align: middle;
        }
        .trending-sub {
          font-size: 0.875rem;
          color: hsl(var(--muted-foreground));
          margin: 0;
        }

        /* ticker */
        .ticker-wrap {
          -webkit-mask-image: linear-gradient(to right, transparent 0%, black 7%, black 93%, transparent 100%);
          mask-image: linear-gradient(to right, transparent 0%, black 7%, black 93%, transparent 100%);
          overflow: hidden;
        }
        @keyframes slideLeft {
          from { transform: translateX(0); }
          to   { transform: translateX(-50%); }
        }
        .ticker-track {
          display: flex;
          gap: 18px;
          width: max-content;
          padding: 12px 0 20px;
          animation: slideLeft 40s linear infinite;
          will-change: transform;
        }

        /* card */
        .ticker-card {
          flex-shrink: 0;
          width: 260px;
          background: hsl(var(--card));
          border: 1px solid hsl(var(--border));
          border-radius: 16px;
          overflow: hidden;
          text-decoration: none;
          display: flex;
          flex-direction: column;
          transition:
            transform 0.32s cubic-bezier(0.34, 1.56, 0.64, 1),
            box-shadow 0.28s ease,
            border-color 0.22s ease;
          cursor: pointer;
        }
        .ticker-card:hover {
          transform: translateY(-6px) scale(1.025);
          box-shadow: 0 20px 48px rgba(0,0,0,0.22), 0 0 0 1px hsl(var(--primary)/0.18);
          border-color: hsl(var(--primary)/0.35);
        }

        /* image area */
        .ticker-card-img {
          position: relative;
          width: 100%;
          height: 136px;
          overflow: hidden;
          flex-shrink: 0;
        }
        .ticker-card-img img {
          width: 100%; height: 100%;
          object-fit: cover;
          display: block;
          transition: transform 0.4s ease;
        }
        .ticker-card:hover .ticker-card-img img { transform: scale(1.05); }
        .ticker-card-img-fallback {
          width: 100%; height: 100%;
        }
        .ticker-card-cat-badge {
          position: absolute;
          bottom: 8px; left: 8px;
          font-size: 0.62rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.07em;
          padding: 3px 8px;
          border-radius: 999px;
          border: 1px solid;
          backdrop-filter: blur(6px);
          display: flex;
          align-items: center;
          line-height: 1;
        }

        /* body */
        .ticker-card-body {
          padding: 12px 14px 14px;
          display: flex;
          flex-direction: column;
          gap: 8px;
          flex: 1;
        }
        .ticker-card-title {
          font-size: 0.8rem;
          font-weight: 650;
          color: hsl(var(--foreground));
          line-height: 1.42;
          margin: 0;
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .ticker-card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: auto;
        }
        .ticker-card-source { font-size: 0.67rem; color: hsl(var(--muted-foreground)); font-weight: 500; }
        .ticker-card-time   { font-size: 0.64rem; color: hsl(var(--muted-foreground)); white-space: nowrap; }

        /* skeleton */
        @keyframes pulse { 0%,100%{opacity:.3} 50%{opacity:.75} }
        .ticker-skeleton-row {
          display: flex;
          gap: 18px;
          padding: 0 48px;
          overflow: hidden;
        }
        .ticker-skeleton-card {
          flex-shrink: 0;
          width: 260px;
          height: 220px;
          border-radius: 16px;
          background: hsl(var(--border));
          animation: pulse 1.4s ease-in-out infinite;
        }

        .ticker-empty {
          text-align: center;
          color: hsl(var(--muted-foreground));
          font-size: 0.875rem;
          padding: 24px;
        }

        .trending-cta {
          display: flex;
          justify-content: center;
          margin-top: 44px;
        }

        /* ── Ticker pauses on hover via CSS (no JS flicker) ── */
        .ticker-wrap:hover .ticker-track {
          animation-play-state: paused;
        }
      `}</style>

      <div
        className="min-h-screen bg-background overflow-x-hidden"
        style={{ animation: "bgFadeIn 0.6s ease-out both", ...(isDark && { background: "#07080f" }), transition: "background 0.15s ease" }}
      >
        {/* ── Login required modal ── */}
        <AuthModal
          open={loginModalOpen}
          onClose={() => setLoginModalOpen(false)}
          onSuccess={(user: AuthUser) => { setLoginModalOpen(false); /* handle authed user here */ }}
        />

        {/* Geometric background layer */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
          <div className={`absolute inset-0 ${isDark
            ? 'bg-gradient-to-br from-blue-500/[0.06] via-transparent to-indigo-500/[0.06]'
            : 'bg-gradient-to-br from-blue-500/[0.10] via-transparent to-indigo-500/[0.10]'} blur-3xl`} />
          <div className={`absolute inset-0 ${isDark
            ? 'bg-gradient-to-tr from-sky-500/[0.03] via-transparent to-blue-600/[0.03]'
            : 'bg-gradient-to-tr from-sky-500/[0.07] via-transparent to-blue-600/[0.07]'} blur-3xl`} />
          <div className="absolute inset-0 overflow-hidden">
            <ElegantShape
              isDark={isDark}
              delay={0.3}
              width={600}
              height={140}
              rotate={12}
              gradient={isDark ? "from-blue-500/[0.12]" : "from-blue-500/[0.25]"}
              className="left-[-10%] md:left-[-5%] top-[15%] md:top-[20%]"
            />
            <ElegantShape
              isDark={isDark}
              delay={0.5}
              width={500}
              height={120}
              rotate={-15}
              gradient={isDark ? "from-indigo-500/[0.12]" : "from-indigo-500/[0.25]"}
              className="right-[-5%] md:right-[0%] top-[70%] md:top-[75%]"
            />
            <ElegantShape
              isDark={isDark}
              delay={0.4}
              width={300}
              height={80}
              rotate={-8}
              gradient={isDark ? "from-sky-500/[0.12]" : "from-sky-500/[0.25]"}
              className="left-[5%] md:left-[10%] bottom-[5%] md:bottom-[10%]"
            />
            <ElegantShape
              isDark={isDark}
              delay={0.6}
              width={200}
              height={60}
              rotate={20}
              gradient={isDark ? "from-violet-500/[0.12]" : "from-violet-500/[0.25]"}
              className="right-[15%] md:right-[20%] top-[10%] md:top-[15%]"
            />
            <ElegantShape
              isDark={isDark}
              delay={0.7}
              width={150}
              height={40}
              rotate={-25}
              gradient={isDark ? "from-cyan-400/[0.12]" : "from-cyan-500/[0.25]"}
              className="left-[20%] md:left-[25%] top-[5%] md:top-[10%]"
            />
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-background/80 pointer-events-none" />
        </div>
        <Navbar />

        {/* Hero */}
        <section className="content-above-bg relative pt-40 pb-24 px-6 min-h-screen flex items-center overflow-visible">
          <div className="absolute top-20 left-1/4 w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full bg-primary/8 blur-3xl pointer-events-none" />
          <div className="container mx-auto relative z-10 flex flex-col items-center text-center">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
              <motion.span
                animate={{ y: [0, 4, 0] }}
                transition={{ y: { duration: 3.5, repeat: Infinity, ease: "easeInOut" } }}
                className="inline-flex items-center gap-2 mb-6 px-5 py-2 rounded-full text-sm font-medium relative overflow-hidden
                bg-gradient-to-r from-blue-500/10 via-indigo-500/10 to-transparent
                backdrop-blur-[3px] border-2 border-white/20 dark:border-white/10
                shadow-[0_8px_32px_0_rgba(59,130,246,0.15)]
                after:absolute after:inset-0 after:rounded-full after:bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.12),transparent_70%)]
                text-foreground/80">
                <span className="relative z-10">{t("landing.badge")}</span>
              </motion.span>
            </motion.div>
            <motion.h1
              className="font-heading text-5xl sm:text-6xl lg:text-7xl font-bold text-foreground leading-tight max-w-4xl flex flex-col items-center"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.15 }}
            >
              <span>{t("landing.hero_title_1")}</span>
              <motion.span className="flex items-center justify-center gap-3 overflow-visible flex-nowrap" layout transition={{ duration: 0.3, ease: "easeInOut" }}>
                {t("landing.hero_title_2")} <RotatingWords />
              </motion.span>
            </motion.h1>
            <motion.p className="text-lg text-muted-foreground max-w-2xl mb-10"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.3 }}>
              {t("landing.hero_desc")}
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.45 }}
              className="flex flex-col items-center gap-10"
            >
              <Link to="/chat">
                <Button size="lg" className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold text-base px-8 h-12 gap-2">
                  {t("landing.start_conversation")} <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <motion.div
                initial={{ opacity: 0 }}
                animate={showScrollArrow ? { opacity: 1 } : { opacity: 0 }}
                transition={{ duration: 0.4 }}
                style={{ pointerEvents: showScrollArrow ? "auto" : "none" }}
              >
                <motion.div
                  animate={{ y: [0, 8, 0] }}
                  transition={{ y: { duration: 1.5, repeat: Infinity, ease: "easeInOut" } }}
                  className="cursor-pointer text-3xl text-primary"
                  onClick={() => {
                    const el = document.getElementById("trending");
                    if (el) {
                      const pos = el.getBoundingClientRect().top + window.pageYOffset - 64.2;
                      window.scrollTo({ top: pos, behavior: "smooth" });
                    }
                  }}
                >↓</motion.div>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Trending — live infinite ticker (2nd, right after hero) */}
        <TrendingSection onLoginRequired={openLogin} isLoggedIn={isLoggedIn} />

        {/* Features */}
        <section className="content-above-bg pt-0 pb-24 px-6 mt-8">
          <div id="features" style={{ position: "relative", top: "-70px" }} />
          <div className="container mx-auto">
            <div className="flex flex-col lg:flex-row items-center gap-16">
              <div className="flex-1">
                <motion.h2 className="font-heading text-3xl sm:text-4xl font-bold text-foreground mb-4"
                  initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={0}>
                  {t("landing.features_title")}
                </motion.h2>
                <motion.p className="text-foreground/75 max-w-xl mb-8"
                  initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={1}>
                  {t("landing.features_desc")}
                </motion.p>
                <div className="flex flex-col gap-3">
                  {([
                    { icon: MessageSquare, key: "intelligent_conversations" },
                    { icon: FileText,      key: "document_analysis" },
                    { icon: Network,       key: "network_expertise" },
                    { icon: Zap,           key: "instant_answers" },
                    { icon: Newspaper,     key: "industry_news" },
                  ] as const).map((f, i) => (
                    <motion.div key={f.key} className="flex items-center gap-3"
                      initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i + 2}>
                      <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <f.icon className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-sm font-medium text-foreground/80 whitespace-nowrap"
                        style={{ textShadow: "0 0 12px hsl(var(--primary)/0.6), 0 0 24px hsl(var(--primary)/0.3)" }}>
                        {t(`landing.feature_${f.key}`)}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </div>

              <motion.div className="flex-1 hidden lg:flex items-center justify-center"
                initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 }}
                style={{ height: 500, position: "relative", marginTop: "-60px" }}>
                <CardSwap width={480} height={320} cardDistance={60} verticalDistance={70} delay={4000} easing="elastic">
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><MessageSquare size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> {t("landing.card_tag_chat")}</span>
                    <h3>{t("landing.card_chat_title")}</h3>
                    <p>{t("landing.card_chat_desc")}</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><FileText size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> {t("landing.card_tag_doc")}</span>
                    <h3>{t("landing.card_doc_title")}</h3>
                    <p>{t("landing.card_doc_desc")}</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Network size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> {t("landing.card_tag_expertise")}</span>
                    <h3>{t("landing.card_network_title")}</h3>
                    <p>{t("landing.card_network_desc")}</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Zap size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> {t("landing.card_tag_answers")}</span>
                    <h3>{t("landing.card_instant_title")}</h3>
                    <p>{t("landing.card_instant_desc")}</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Newspaper size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> {t("landing.card_tag_briefing")}</span>
                    <h3>{t("landing.card_news_title")}</h3>
                    <p>{t("landing.card_news_desc")}</p>
                  </Card>
                </CardSwap>
              </motion.div>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="content-above-bg py-8 px-6 bg-secondary/50 backdrop-blur-xl mb-24 mt-16">
          <div id="how-it-works" style={{ position: "relative", top: "-90px" }} />
          <div className="container mx-auto text-center">
            <motion.h2 className="font-heading text-3xl sm:text-4xl font-bold text-foreground mb-6"
              initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={0}>
              {t("landing.how_title")}
            </motion.h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto pt-4 pb-8">
              {[
                { step: "01", titleKey: "step_1_title", descKey: "step_1_desc" },
                { step: "02", titleKey: "step_2_title", descKey: "step_2_desc" },
                { step: "03", titleKey: "step_3_title", descKey: "step_3_desc" },
              ].map((item, i) => (
                <motion.div key={item.step} className="text-center"
                  initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i + 1}>
                  <span className="text-5xl font-heading font-bold text-gradient-blue">{item.step}</span>
                  <h3 className="font-heading font-semibold text-foreground mt-3 mb-1.5">{t(`landing.${item.titleKey}`)}</h3>
                  <p className="text-sm text-muted-foreground">{t(`landing.${item.descKey}`)}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Contact */}
        <section className="content-above-bg py-24 px-6">
          <div id="contact" style={{ position: "relative", top: "-90px" }} />
          <div className="container mx-auto max-w-2xl">
            <motion.div className="text-center mb-12"
              initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={0}>
              <h2 className="font-heading text-3xl sm:text-4xl font-bold text-foreground mb-3">{t("landing.contact_title")}</h2>
              <p className="text-muted-foreground">{t("landing.contact_desc")}</p>
            </motion.div>
            <ContactForm />
          </div>
        </section>

        {/* Footer */}
        <footer className="content-above-bg py-12 px-6 border-t border-border">
          <div className="container mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
            <span className="font-heading font-semibold text-foreground">Bimlo Copilot</span>
            <span className="text-sm text-muted-foreground">{t("landing.footer_copyright")}</span>
          </div>
        </footer>
      </div>
    </>
  );
};

export default Index;