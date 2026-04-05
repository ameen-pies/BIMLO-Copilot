import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import ThemeToggle from "@/components/ThemeToggle";
import Logo from "@/components/Logo";

const scrollTo = (id: string, extraOffset = 0) => (e: React.MouseEvent) => {
  e.preventDefault();
  const el = document.getElementById(id);
  if (el) {
    const pos = el.getBoundingClientRect().top + window.pageYOffset - 64 + extraOffset;
    window.scrollTo({ top: pos, behavior: 'smooth' });
  }
};

const Navbar = () => {
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
          <a href="#features" onClick={scrollTo('features', 64)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Features
          </a>
          <a href="#how-it-works" onClick={scrollTo('how-it-works', -160)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            How it works
          </a>
          <a href="#trending" onClick={scrollTo('trending', -20)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Trending
          </a>
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
            Documentation
          </a>
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Link to="/chat">
            <Button className="bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity font-heading font-semibold">
              Launch Copilot
            </Button>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;