import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Zap, FileText, MessageSquare, Network, Newspaper, Radio, Cable, Scale, HardHat, TrendingUp, Box, Layers, BrainCircuit } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import RotatingWords from "@/components/RotatingWords";
import BackgroundManager from "@/components/BackgroundManager";
import CardSwap, { Card } from "@/components/CardSwap";
import AuthModal, { AuthUser } from "@/components/AuthModal";
import { useAuth } from "@/context/AuthContext";
import { useNavigate } from "react-router-dom";
import { useState, useEffect, useRef, useCallback } from "react";

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
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading,  setLoading]  = useState(true);
  const trackRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    (async () => {
      try {
        const res  = await fetch(`${API_BASE}/api/news/pages/0`);
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
          Trending on&nbsp;
          <img src="/favicon.svg" alt="Bimlo" className="trending-logo" />
          &nbsp;this week
        </h2>
        <p className="trending-sub">Top stories shaping the telecom industry right now</p>
      </motion.div>

      {/* ticker */}
      {loading ? (
        <div className="ticker-skeleton-row">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="ticker-skeleton-card" style={{ animationDelay: `${i * 0.12}s` }} />
          ))}
        </div>
      ) : articles.length === 0 ? (
        <p className="ticker-empty">Articles are being collected — check back in a few minutes.</p>
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
          Open full industry briefing
          <ArrowRight className="h-4 w-4" />
        </Button>
      </motion.div>
    </section>
  );
};

// ── Main page ────────────────────────────────────────────────────────────────
const Index = () => {
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

        /* ── Ensure all page content sits above BackgroundManager ── */
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

        <BackgroundManager />
        <Navbar />

        {/* Hero */}
        <section className="content-above-bg relative pt-40 pb-24 px-6 min-h-screen flex items-center overflow-visible">
          <div className="absolute top-20 left-1/4 w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full bg-primary/8 blur-3xl pointer-events-none" />
          <div className="container mx-auto relative z-10 flex flex-col items-center text-center">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
              <span className="inline-block mb-6 px-4 py-1.5 rounded-full bg-accent text-accent-foreground text-sm font-medium border border-primary/10">
                AI-Powered Telecom Assistant
              </span>
            </motion.div>
            <motion.h1
              className="font-heading text-5xl sm:text-6xl lg:text-7xl font-bold text-foreground leading-normal max-w-4xl -mb-4 flex flex-col items-center"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.15 }}
            >
              <span>Your intelligent partner</span>
              <motion.span className="flex items-center justify-center gap-3 py-2 overflow-visible" layout transition={{ duration: 0.3, ease: "easeInOut" }}>
                for <RotatingWords />
              </motion.span>
            </motion.h1>
            <motion.p className="text-lg text-muted-foreground max-w-2xl mb-10"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.3 }}>
              Bimlo Copilot helps telecom professionals analyze documents, plan networks,
              and make data-driven decisions — powered by specialized AI.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.45 }}
              className="flex flex-col items-center gap-10"
            >
              <Link to="/chat">
                <Button size="lg" className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold text-base px-8 h-12 gap-2">
                  Start a conversation <ArrowRight className="h-4 w-4" />
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
                  Built for telecom professionals
                </motion.h2>
                <motion.p className="text-foreground/75 max-w-xl mb-8"
                  initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={1}>
                  Everything you need to accelerate your telecom projects — from document analysis to network planning and real-time decision support.
                </motion.p>
                <div className="flex flex-col gap-3">
                  {[
                    { icon: MessageSquare, title: "Intelligent Conversations" },
                    { icon: FileText,      title: "Document Analysis" },
                    { icon: Network,       title: "Network Expertise" },
                    { icon: Zap,           title: "Instant Answers" },
                    { icon: Newspaper,     title: "Industry News Briefings" },
                  ].map((f, i) => (
                    <motion.div key={f.title} className="flex items-center gap-3"
                      initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i + 2}>
                      <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <f.icon className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-sm font-medium text-foreground/80 whitespace-nowrap"
                        style={{ textShadow: "0 0 12px hsl(var(--primary)/0.6), 0 0 24px hsl(var(--primary)/0.3)" }}>
                        {f.title}
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
                    <span className="tag"><Zap size={11} /> Intelligent Chat</span>
                    <h3>Intelligent Conversations</h3>
                    <p>Chat naturally about telecom infrastructure with context-aware AI that understands your projects.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><FileText size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> Doc Analysis</span>
                    <h3>Document Analysis</h3>
                    <p>Upload PDFs, specs, and reports for instant AI-powered insights and citations.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Network size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> Deep Expertise</span>
                    <h3>Network Expertise</h3>
                    <p>Deep knowledge of fiber, 5G, and optical network deployments built right in.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Zap size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> Instant Answers</span>
                    <h3>Instant Answers</h3>
                    <p>Get fast, accurate technical guidance for complex telecom decisions powered by specialized AI.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon"><Newspaper size={22} color="hsl(var(--accent-foreground))" /></div>
                    <span className="tag"><Zap size={11} /> Daily Briefing</span>
                    <h3>Industry News</h3>
                    <p>AI-curated telecom news with expert impact analysis, delivered fresh every morning.</p>
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
              How it works
            </motion.h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto pt-4 pb-8">
              {[
                { step: "01", title: "Upload Documents", desc: "Add your technical PDFs, specs, and reports." },
                { step: "02", title: "Ask Questions",    desc: "Chat naturally about your telecom projects." },
                { step: "03", title: "Get Insights",     desc: "Receive accurate, context-aware answers instantly." },
              ].map((item, i) => (
                <motion.div key={item.step} className="text-center"
                  initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i + 1}>
                  <span className="text-5xl font-heading font-bold text-gradient-blue">{item.step}</span>
                  <h3 className="font-heading font-semibold text-foreground mt-3 mb-1.5">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="content-above-bg py-12 px-6 border-t border-border">
          <div className="container mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
            <span className="font-heading font-semibold text-foreground">Bimlo Copilot</span>
            <span className="text-sm text-muted-foreground">© 2026 Bimlo. All rights reserved.</span>
          </div>
        </footer>
      </div>
    </>
  );
};

export default Index;