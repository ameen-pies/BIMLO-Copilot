import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import ThemeToggle from "@/components/ThemeToggle";
import Logo from "@/components/Logo";
import { useAuth } from "@/context/AuthContext";
import { useState, useRef, useEffect } from "react";

// ── Reusable profile bubble used by Navbar, Chat, CallPage, NewsPage ─────────
export interface ProfileUser { username: string; email: string; }

export const ProfileBubble = ({
  user,
  onLogout,
  align = "right",
}: {
  user: ProfileUser;
  onLogout: () => void;
  align?: "right" | "left";
}) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const initial = user.username[0].toUpperCase();

  return (
    <div ref={ref} style={{ position: "relative" }}>
      {/* Avatar button */}
      <button
        onClick={() => setOpen(o => !o)}
        title={user.username}
        style={{
          width: 34, height: 34, borderRadius: "50%",
          background: open
            ? "linear-gradient(135deg,#6366f1,#3b82f6)"
            : "linear-gradient(135deg,#3b82f6,#6366f1)",
          border: open ? "2px solid rgba(99,102,241,0.6)" : "2px solid rgba(99,102,241,0.25)",
          cursor: "pointer", color: "#fff",
          fontSize: 13, fontWeight: 700,
          display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: open
            ? "0 0 0 3px rgba(99,102,241,0.2), 0 8px 24px rgba(99,102,241,0.3)"
            : "0 0 0 2px rgba(99,102,241,0.15)",
          transition: "all 0.18s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        {initial}
      </button>

      {/* Dropdown */}
      <div
        style={{
          position: "absolute",
          top: "calc(100% + 10px)",
          [align === "right" ? "right" : "left"]: 0,
          minWidth: 210,
          background: "hsl(var(--card))",
          border: "1px solid hsl(var(--border))",
          borderRadius: 16,
          boxShadow: "0 8px 32px rgba(0,0,0,0.18), 0 0 0 1px rgba(0,0,0,0.06)",
          zIndex: 200,
          overflow: "hidden",
          opacity: open ? 1 : 0,
          transform: open ? "scale(1) translateY(0)" : "scale(0.94) translateY(-6px)",
          pointerEvents: open ? "auto" : "none",
          transition: "opacity 0.18s ease, transform 0.18s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        {/* Header band */}
        <div style={{
          padding: "14px 16px 12px",
          background: "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(99,102,241,0.08))",
          borderBottom: "1px solid rgba(99,102,241,0.12)",
        }}>
          {/* Avatar large */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 38, height: 38, borderRadius: "50%",
              background: "linear-gradient(135deg,#3b82f6,#6366f1)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 15, fontWeight: 800, color: "#fff",
              boxShadow: "0 4px 12px rgba(99,102,241,0.35)",
              flexShrink: 0,
            }}>
              {initial}
            </div>
            <div style={{ minWidth: 0 }}>
              <p style={{ margin: 0, fontSize: 13, fontWeight: 700, color: "hsl(var(--foreground))", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {user.username}
              </p>
              <p style={{ margin: "2px 0 0", fontSize: 11, color: "rgba(148,163,184,0.8)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {user.email}
              </p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div style={{ padding: "6px 0 6px" }}>
          <button
            onClick={() => { setOpen(false); onLogout(); }}
            style={{
              width: "100%", textAlign: "left",
              padding: "9px 16px",
              background: "transparent", border: "none", cursor: "pointer",
              fontSize: 13, fontWeight: 500,
              color: "#f87171",
              display: "flex", alignItems: "center", gap: 8,
              transition: "background 0.12s",
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(248,113,113,0.08)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
            </svg>
            Log out
          </button>
        </div>
      </div>
    </div>
  );
};

const scrollTo = (id: string, extraOffset = 0) => (e: React.MouseEvent) => {
  e.preventDefault();
  const el = document.getElementById(id);
  if (el) {
    const pos = el.getBoundingClientRect().top + window.pageYOffset - 64 + extraOffset;
    window.scrollTo({ top: pos, behavior: 'smooth' });
  }
};

const Navbar = () => {
  const { currentUser, showAuthModal, logout } = useAuth();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50" style={{
      backdropFilter: 'blur(12px)',
      WebkitBackdropFilter: 'blur(12px)',
      backgroundColor: 'hsl(var(--background) / 0.55)',
      borderBottom: '1px solid hsl(var(--border) / 0.3)',
    }}>
      <div className="container mx-auto flex items-center justify-between h-16 px-6">
        <Link to="/" className="flex items-center gap-2">
          <Logo className="h-8 w-8" />
          <span className="font-heading text-xl font-bold text-foreground">
            Bimlo Copilot
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          <a href="#trending" onClick={scrollTo('trending', 0)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Trending
          </a>
          <a href="#features" onClick={scrollTo('features', 0)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Features
          </a>
          <a href="#how-it-works" onClick={scrollTo('how-it-works', 0)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            How it works
          </a>
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Documentation
          </a>
        </div>

        <div className="flex items-center gap-2">
          {currentUser ? (
            /* ── Logged-in: theme + profile bubble ── */
            <>
              <ThemeToggle />
              <ProfileBubble user={currentUser} onLogout={logout} align="right" />
            </>
          ) : (
            /* ── Guest: Log in + Launch Copilot + Theme ── */
            <>
              <button
                onClick={() => showAuthModal()}
                style={{
                  fontSize: 13, fontWeight: 600, padding: "6px 14px",
                  borderRadius: 8, border: "1px solid rgba(96,165,250,0.35)",
                  background: "rgba(96,165,250,0.08)", color: "#60a5fa",
                  cursor: "pointer", transition: "background 0.15s",
                }}
                onMouseEnter={e => { (e.currentTarget).style.background = "rgba(96,165,250,0.18)"; }}
                onMouseLeave={e => { (e.currentTarget).style.background = "rgba(96,165,250,0.08)"; }}
              >
                Log in
              </button>
              <Link to="/chat">
                <Button className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold">
                  Launch Copilot
                </Button>
              </Link>
              <ThemeToggle />
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;