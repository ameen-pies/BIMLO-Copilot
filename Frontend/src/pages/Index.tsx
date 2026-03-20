import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Zap, FileText, MessageSquare, Network } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import RotatingWords from "@/components/RotatingWords";
import BackgroundManager from "@/components/BackgroundManager";
import { useState, useEffect } from "react";

const features = [
  {
    icon: MessageSquare,
    title: "Intelligent Conversations",
    desc: "Chat naturally about telecom infrastructure with context-aware AI responses.",
  },
  {
    icon: FileText,
    title: "Document Analysis",
    desc: "Upload PDFs, specs, and reports for instant AI-powered insights.",
  },
  {
    icon: Network,
    title: "Network Expertise",
    desc: "Deep knowledge of fiber, 5G, and optical network deployments.",
  },
  {
    icon: Zap,
    title: "Instant Answers",
    desc: "Get fast, accurate technical guidance for complex telecom decisions.",
  },
];

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

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollArrow(window.scrollY < 100);
    };

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
    <div className="min-h-screen bg-background">
      <BackgroundManager />
      <Navbar />

      {/* Hero */}
      <section className="relative pt-40 pb-24 px-6 overflow-hidden min-h-screen flex items-center">
        {/* Subtle gradient orbs */}
        <div className="absolute top-20 left-1/4 w-[500px] h-[500px] rounded-full bg-primary/5 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full bg-primary/8 blur-3xl pointer-events-none" />

        <div className="container mx-auto text-center relative z-10">
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
            className="font-heading text-5xl sm:text-6xl lg:text-7xl font-bold text-foreground leading-tight max-w-4xl mx-auto mb-6 text-center flex flex-col items-center"
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
            className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10"
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
            className="flex flex-col items-center justify-center gap-16"
          >
            <Link to="/chat">
              <Button
                size="lg"
                className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold text-base px-8 h-12 gap-2"
              >
                Start a conversation
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>

            {/* Bouncing down arrow */}
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
              onClick={() => {
                document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              ↓
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="pt-0 pb-48 px-6">
        <div className="container mx-auto">
          <motion.h2
            className="font-heading text-3xl sm:text-4xl font-bold text-center text-foreground mb-2"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            custom={0}
          >
            Built for telecom professionals
          </motion.h2>
          <motion.p
            className="text-muted-foreground text-center max-w-xl mx-auto mb-8"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            custom={1}
          >
            Everything you need to accelerate your telecom projects.
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                className="group p-6 rounded-2xl glass-surface border border-white/10 hover:border-primary/30 hover:shadow-blue transition-all duration-300 relative overflow-hidden"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeUp}
                custom={i + 2}
              >
                <div className="relative z-10">
                  <div className="h-12 w-12 rounded-xl bg-accent flex items-center justify-center mb-4 group-hover:bg-hero-gradient transition-colors duration-300">
                    <f.icon className="h-6 w-6 text-accent-foreground group-hover:text-primary-foreground transition-colors duration-300" />
                  </div>
                  <h3 className="font-heading font-semibold text-foreground mb-2">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="py-8 px-6 bg-secondary/50 backdrop-blur-xl mb-24">
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
              { step: "02", title: "Ask Questions", desc: "Chat naturally about your telecom projects." },
              { step: "03", title: "Get Insights", desc: "Receive accurate, context-aware answers instantly." },
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
  );
};

export default Index;