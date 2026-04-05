import { useLayoutEffect, useRef, useState, useEffect } from 'react';
import { gsap } from 'gsap';
import { ArrowUpRight } from 'lucide-react';
import './CardNav.css';

interface NavLink {
  label: string;
  ariaLabel?: string;
  href?: string;
  onSelect?: () => void;
  active?: boolean;
}

interface NavItem {
  label: string;
  bgColor: string;
  textColor: string;
  links?: NavLink[];
}

interface CardNavProps {
  logo?: string;
  logoAlt?: string;
  items: NavItem[];
  className?: string;
  ease?: string;
  baseColor?: string;
  menuColor?: string;
  buttonBgColor?: string;
  buttonTextColor?: string;
  theme?: 'light' | 'dark';
  filterMode?: boolean;
  filterLabel?: string;
}

const CardNav = ({
  logo,
  logoAlt = 'Logo',
  items,
  className = '',
  ease = 'power3.out',
  baseColor = '#fff',
  menuColor,
  buttonBgColor,
  buttonTextColor,
  theme = 'light',
  filterMode = false,
  filterLabel = 'Filter',
}: CardNavProps) => {
  const [isHamburgerOpen, setIsHamburgerOpen] = useState(false);
  const [isExpanded, setIsExpanded]           = useState(false);
  const navRef   = useRef<HTMLElement>(null);
  const cardsRef = useRef<HTMLDivElement[]>([]);
  const tlRef    = useRef<gsap.core.Timeline | null>(null);

  const dark = theme === 'dark';
  const resolvedBase      = baseColor !== '#fff' ? baseColor : dark ? '#0c0d18' : '#ffffff';
  const resolvedMenuColor = menuColor ?? (dark ? 'rgba(255,255,255,0.75)' : '#000');
  const resolvedBtnBg     = buttonBgColor ?? (dark ? '#1a1a2e' : '#111');
  const resolvedBtnText   = buttonTextColor ?? '#fff';

  const calculateWidth = () => {
    const navEl = navRef.current;
    if (!navEl) return 400;
    const contentEl = navEl.querySelector('.card-nav-content') as HTMLElement | null;
    const topBar = navEl.querySelector('.card-nav-top') as HTMLElement | null;
    const topW = topBar?.offsetWidth ?? 120;
    if (contentEl) {
      const prev = { vis: contentEl.style.visibility, pe: contentEl.style.pointerEvents, w: contentEl.style.width };
      contentEl.style.visibility = 'visible';
      contentEl.style.pointerEvents = 'auto';
      contentEl.style.width = 'auto';
      contentEl.offsetWidth;
      const natural = contentEl.scrollWidth;
      Object.assign(contentEl.style, { visibility: prev.vis, pointerEvents: prev.pe, width: prev.w });
      return topW + natural + 8;
    }
    return 400;
  };

  const createTimeline = () => {
    const navEl = navRef.current;
    if (!navEl) return null;
    const topBar = navEl.querySelector('.card-nav-top') as HTMLElement | null;
    const collapsedW = topBar?.offsetWidth ?? 120;
    gsap.set(navEl, { width: collapsedW, overflow: 'hidden' });
    gsap.set(cardsRef.current, { x: 30, opacity: 0 });
    const tl = gsap.timeline({ paused: true });
    tl.to(navEl, { width: calculateWidth, duration: 0.4, ease });
    tl.to(cardsRef.current, { x: 0, opacity: 1, duration: 0.35, ease, stagger: 0.07 }, '-=0.15');
    return tl;
  };

  useLayoutEffect(() => {
    const tl = createTimeline();
    tlRef.current = tl;
    return () => { tl?.kill(); tlRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ease, items, theme]);

  useLayoutEffect(() => {
    const handleResize = () => {
      if (!tlRef.current) return;
      if (isExpanded) {
        gsap.set(navRef.current, { width: calculateWidth() });
        tlRef.current.kill();
        const tl = createTimeline();
        if (tl) { tl.progress(1); tlRef.current = tl; }
      } else {
        tlRef.current.kill();
        const tl = createTimeline();
        if (tl) tlRef.current = tl;
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isExpanded]);

  // close on outside click
  useEffect(() => {
    if (!isExpanded) return;
    const handler = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setIsHamburgerOpen(false);
        tlRef.current?.eventCallback('onReverseComplete', () => setIsExpanded(false));
        tlRef.current?.reverse();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [isExpanded]);

  const toggleMenu = () => {
    if (!isExpanded) {
      // Build fresh timeline every open — avoids stale state after reverse
      const tl = createTimeline();
      if (!tl) return;
      tlRef.current?.kill();
      tlRef.current = tl;
      setIsHamburgerOpen(true);
      setIsExpanded(true);
      tl.play();
    } else {
      setIsHamburgerOpen(false);
      tlRef.current?.eventCallback('onReverseComplete', () => setIsExpanded(false));
      tlRef.current?.reverse();
    }
  };

  const setCardRef = (i: number) => (el: HTMLDivElement | null) => {
    if (el) cardsRef.current[i] = el;
  };

  return (
    <div
      className={`card-nav-container ${className}`}
    >
      <nav
        ref={navRef}
        className={`card-nav ${isExpanded ? 'open' : ''}`}
        style={{
          backgroundColor: resolvedBase,
          border: dark ? '1px solid rgba(255,255,255,0.08)' : '0.5px solid rgba(0,0,0,0.1)',
          boxShadow: dark ? '0 4px 24px rgba(0,0,0,0.5)' : '0 4px 12px rgba(0,0,0,0.1)',
        }}
      >
        {/* Top bar */}
        <div className="card-nav-top">
          <div
            className={`hamburger-menu ${isHamburgerOpen ? 'open' : ''}`}
            onClick={toggleMenu}
            role="button"
            aria-label={isExpanded ? 'Close filter menu' : 'Open filter menu'}
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && toggleMenu()}
            style={{ color: resolvedMenuColor }}
          >
            <div className="hamburger-line" />
            <div className="hamburger-line" />
          </div>

          {filterMode ? (
            <span style={{
              fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.04em',
              color: resolvedMenuColor, whiteSpace: 'nowrap', pointerEvents: 'none',
            }}>
              {filterLabel}
            </span>
          ) : logo ? (
            <div className="logo-container">
              <img src={logo} alt={logoAlt} className="logo" />
            </div>
          ) : null}

          {!filterMode && (
            <button
              type="button"
              className="card-nav-cta-button"
              style={{ backgroundColor: resolvedBtnBg, color: resolvedBtnText }}
            >
              Get Started
            </button>
          )}
        </div>

        {/* Cards */}
        <div className="card-nav-content" aria-hidden={!isExpanded}>
          {(items || []).slice(0, 4).map((item, idx) => (
            <div
              key={`${item.label}-${idx}`}
              className="nav-card"
              ref={setCardRef(idx)}
              style={{ backgroundColor: item.bgColor, color: item.textColor }}
            >
              <div className="nav-card-label">{item.label}</div>
              <div className="nav-card-links">
                {item.links?.map((lnk, i) => (
                  <a
                    key={`${lnk.label}-${i}`}
                    className="nav-card-link"
                    href={lnk.href ?? '#'}
                    aria-label={lnk.ariaLabel}
                    onClick={e => {
                      if (lnk.onSelect) {
                        e.preventDefault();
                        lnk.onSelect();
                        setIsHamburgerOpen(false);
                        tlRef.current?.eventCallback('onReverseComplete', () => setIsExpanded(false));
                        tlRef.current?.reverse();
                      }
                    }}
                    style={{
                      fontWeight: lnk.active ? 700 : undefined,
                      textDecoration: lnk.active ? 'underline' : undefined,
                      textUnderlineOffset: '3px',
                    }}
                  >
                    <ArrowUpRight className="nav-card-link-icon" size={14} aria-hidden="true" />
                    {lnk.label}
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      </nav>
    </div>
  );
};

export default CardNav;