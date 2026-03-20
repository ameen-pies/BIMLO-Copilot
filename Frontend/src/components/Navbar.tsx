import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import ThemeToggle from "@/components/ThemeToggle";
import Logo from "@/components/Logo";

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
          <a 
            href="#features" 
            onClick={(e) => {
              e.preventDefault();
              const element = document.getElementById('features');
              if (element) {
                const navbarHeight = 64; // h-16 = 64px
                const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
                const offsetPosition = elementPosition - navbarHeight - 160; // Extra 160px padding
                window.scrollTo({
                  top: offsetPosition,
                  behavior: 'smooth'
                });
              }
            }}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Features
          </a>
          <a 
            href="#how-it-works" 
            onClick={(e) => {
              e.preventDefault();
              const element = document.getElementById('how-it-works');
              if (element) {
                const navbarHeight = 64; // h-16 = 64px
                const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
                const offsetPosition = elementPosition - navbarHeight - 160; // Extra 160px padding
                window.scrollTo({
                  top: offsetPosition,
                  behavior: 'smooth'
                });
              }
            }}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            How it works
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