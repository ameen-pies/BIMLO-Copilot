import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import ThemeToggle from "@/components/ThemeToggle";
import Logo from "@/components/Logo";
import { useAuth } from "@/context/AuthContext";

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
          <ThemeToggle />

          {currentUser ? (
            /* ── Logged-in: avatar + hover dropdown ── */
            <div className="relative group">
              <button
                title={currentUser.username}
                style={{
                  width: 34, height: 34, borderRadius: "50%",
                  background: "linear-gradient(135deg,#3b82f6,#6366f1)",
                  border: "none", cursor: "pointer", color: "#fff",
                  fontSize: 13, fontWeight: 700,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "0 0 0 2px rgba(99,102,241,0.3)",
                }}
              >
                {currentUser.username[0].toUpperCase()}
              </button>
              <div
                className="absolute right-0 mt-2 invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-all duration-150"
                style={{
                  background: "var(--color-card,#1e293b)",
                  border: "1px solid rgba(100,116,139,0.25)",
                  borderRadius: 12, padding: "8px 0", minWidth: 170, zIndex: 100,
                  boxShadow: "0 12px 32px rgba(0,0,0,0.3)",
                }}
              >
                <div style={{ padding: "6px 14px 10px", borderBottom: "1px solid rgba(100,116,139,0.15)" }}>
                  <p style={{ fontSize: 12, fontWeight: 600, margin: 0 }}>{currentUser.username}</p>
                  <p style={{ fontSize: 11, color: "#64748b", margin: "2px 0 0" }}>{currentUser.email}</p>
                </div>
                <button
                  onClick={logout}
                  style={{
                    width: "100%", textAlign: "left", padding: "9px 14px",
                    background: "transparent", border: "none", cursor: "pointer",
                    fontSize: 13, color: "#f87171",
                  }}
                >
                  Log out
                </button>
              </div>
            </div>
          ) : (
            /* ── Guest: Log in + Launch Copilot ── */
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
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;