import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Zap, FileText, MessageSquare, Network, Newspaper } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import RotatingWords from "@/components/RotatingWords";
import BackgroundManager from "@/components/BackgroundManager";
import CardSwap, { Card } from "@/components/CardSwap";
import { useState, useEffect } from "react";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.5, ease: "easeOut" as const },
  }),
};

const Index = () => {
  const [showScrollArrow, setShowScrollArrow] = useState(true);
  const [isDark, setIsDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const observer = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains("dark"))
    );
    observer.observe(document.documentElement, { attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleScroll = () => setShowScrollArrow(window.scrollY < 100);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    document.documentElement.style.scrollBehavior = 'smooth';
    document.documentElement.classList.add('scrollbar-thin');
    return () => {
      document.documentElement.style.scrollBehavior = 'auto';
      document.documentElement.classList.remove('scrollbar-thin');
    };
  }, []);

  return (
    <>
      <style>{`
        @keyframes bgFadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes liquidFadeIn {
          from { opacity: 0; }
          to   { opacity: 0.5; }
        }
        .animate-fade-in {
          animation: liquidFadeIn 1.2s ease-out 0.3s both;
        }
        .swap-card {
          background: linear-gradient(135deg, hsl(var(--card)) 0%, hsl(var(--secondary)) 100%);
          border: 1px solid hsl(var(--border)) !important;
          border-radius: 20px !important;
          padding: 2rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        }
        .swap-card-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          background: hsl(var(--accent));
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 0.5rem;
        }
        .swap-card h3 {
          font-size: 1.1rem;
          font-weight: 700;
          color: hsl(var(--foreground));
          margin: 0;
        }
        .swap-card p {
          font-size: 0.875rem;
          color: hsl(var(--muted-foreground));
          line-height: 1.6;
          margin: 0;
        }
        .swap-card .tag {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          font-size: 0.75rem;
          font-weight: 600;
          color: hsl(var(--primary));
          background: hsl(var(--primary) / 0.1);
          border: 1px solid hsl(var(--primary) / 0.2);
          border-radius: 999px;
          padding: 0.25rem 0.75rem;
          width: fit-content;
        }
      `}</style>

      <div className="min-h-screen bg-background overflow-x-hidden" style={{ animation: "bgFadeIn 0.6s ease-out both", ...(isDark && { background: "#07080f" }), transition: "background 0.15s ease" }}>
        <BackgroundManager />
        <Navbar />

        {/* Hero — centered */}
        <section className="relative pt-40 pb-24 px-6 min-h-screen flex items-center overflow-hidden">
          <div className="absolute top-20 left-1/4 w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl pointer-events-none" />
          <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full bg-primary/8 blur-3xl pointer-events-none" />

          <div className="container mx-auto relative z-10 flex flex-col items-center text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <span className="inline-block mb-6 px-4 py-1.5 rounded-full bg-accent text-accent-foreground text-sm font-medium border border-primary/10">
                AI-Powered Telecom Assistant
              </span>
            </motion.div>

            <motion.h1
              className="font-heading text-5xl sm:text-6xl lg:text-7xl font-bold text-foreground leading-tight max-w-4xl mb-6 flex flex-col items-center"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.15 }}
            >
              <span>Your intelligent partner</span>
              <motion.span
                className="flex items-center justify-center gap-3"
                layout
                transition={{ duration: 0.3, ease: "easeInOut" }}
              >
                for <RotatingWords />
              </motion.span>
            </motion.h1>

            <motion.p
              className="text-lg text-muted-foreground max-w-2xl mb-10"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              Bimlo Copilot helps telecom professionals analyze documents, plan networks,
              and make data-driven decisions — powered by specialized AI.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.45 }}
              className="flex flex-col items-center gap-10"
            >
              <div className="flex flex-col sm:flex-row items-center gap-4">
                <Link to="/chat">
                  <Button
                    size="lg"
                    className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold text-base px-8 h-12 gap-2"
                  >
                    Start a conversation
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link to="/news">
                  <Button
                    size="lg"
                    variant="outline"
                    className="font-heading font-semibold text-base px-8 h-12 gap-2 border-primary/20 hover:border-primary/50 hover:bg-primary/5 transition-all"
                  >
                    <Newspaper className="h-4 w-4" />
                    Industry Briefing
                  </Button>
                </Link>
              </div>

              <motion.div
                initial={{ opacity: 0 }}
                animate={{
                  opacity: showScrollArrow ? 1 : 0,
                  y: [0, 10, 0]
                }}
                transition={{
                  opacity: { duration: 0.3 },
                  y: { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
                }}
                className="cursor-pointer text-3xl text-primary"
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              >
                ↓
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Features — text left, CardSwap right */}
        <section className="pt-0 pb-24 px-6 mt-8">
          <div id="features" style={{ position: "relative", top: "-110px" }} />
          <div className="container mx-auto">
            <div className="flex flex-col lg:flex-row items-center gap-16">

              {/* Left: text + feature list */}
              <div className="flex-1">
                <motion.h2
                  className="font-heading text-3xl sm:text-4xl font-bold text-foreground mb-4"
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  variants={fadeUp}
                  custom={0}
                >
                  Built for telecom professionals
                </motion.h2>
                <motion.p
                  className="text-foreground/75 max-w-xl mb-8"
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  variants={fadeUp}
                  custom={1}
                >
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
                    <motion.div
                      key={f.title}
                      className="flex items-center gap-3"
                      initial="hidden"
                      whileInView="visible"
                      viewport={{ once: true }}
                      variants={fadeUp}
                      custom={i + 2}
                    >
                      <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                        <f.icon className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-sm font-medium text-foreground/80 whitespace-nowrap"
                        style={{ textShadow: "0 0 12px hsl(var(--primary) / 0.6), 0 0 24px hsl(var(--primary) / 0.3)" }}>
                        {f.title}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Right: CardSwap */}
              <motion.div
                className="flex-1 hidden lg:flex items-center justify-center"
                initial={{ opacity: 0, x: 40 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 }}
                style={{ height: 500, position: 'relative', marginTop: '-60px' }}
              >
                <CardSwap
                  width={480}
                  height={320}
                  cardDistance={60}
                  verticalDistance={70}
                  delay={4000}
                  easing="elastic"
                >
                  <Card customClass="swap-card">
                    <div className="swap-card-icon">
                      <MessageSquare size={22} color="hsl(var(--accent-foreground))" />
                    </div>
                    <span className="tag"><Zap size={11} /> Intelligent Chat</span>
                    <h3>Intelligent Conversations</h3>
                    <p>Chat naturally about telecom infrastructure with context-aware AI that understands your projects.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon">
                      <FileText size={22} color="hsl(var(--accent-foreground))" />
                    </div>
                    <span className="tag"><Zap size={11} /> Doc Analysis</span>
                    <h3>Document Analysis</h3>
                    <p>Upload PDFs, specs, and reports for instant AI-powered insights and citations.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon">
                      <Network size={22} color="hsl(var(--accent-foreground))" />
                    </div>
                    <span className="tag"><Zap size={11} /> Deep Expertise</span>
                    <h3>Network Expertise</h3>
                    <p>Deep knowledge of fiber, 5G, and optical network deployments built right in.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon">
                      <Zap size={22} color="hsl(var(--accent-foreground))" />
                    </div>
                    <span className="tag"><Zap size={11} /> Instant Answers</span>
                    <h3>Instant Answers</h3>
                    <p>Get fast, accurate technical guidance for complex telecom decisions powered by specialized AI.</p>
                  </Card>
                  <Card customClass="swap-card">
                    <div className="swap-card-icon">
                      <Newspaper size={22} color="hsl(var(--accent-foreground))" />
                    </div>
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
        <section className="py-8 px-6 bg-secondary/50 backdrop-blur-xl mb-24 mt-16">
          <div id="how-it-works" style={{ position: "relative", top: "400px" }} />
          <div className="container mx-auto text-center">
            <motion.h2
              className="font-heading text-3xl sm:text-4xl font-bold text-foreground mb-6"
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={fadeUp}
              custom={0}
            >
              How it works
            </motion.h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto pt-4 pb-8">
              {[
                { step: "01", title: "Upload Documents", desc: "Add your technical PDFs, specs, and reports." },
                { step: "02", title: "Ask Questions",    desc: "Chat naturally about your telecom projects." },
                { step: "03", title: "Get Insights",     desc: "Receive accurate, context-aware answers instantly." },
              ].map((item, i) => (
                <motion.div
                  key={item.step}
                  className="text-center"
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  variants={fadeUp}
                  custom={i + 1}
                >
                  <span className="text-5xl font-heading font-bold text-gradient-blue">{item.step}</span>
                  <h3 className="font-heading font-semibold text-foreground mt-3 mb-1.5">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="py-12 px-6 border-t border-border">
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