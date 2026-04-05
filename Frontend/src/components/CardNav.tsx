import { useLayoutEffect, useRef, useState, useEffect } from 'react';
import { gsap } from 'gsap';
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

  // Measure the top bar's natural width without GSAP inline styles
  const getCollapsedWidth = () => {
    const navEl = navRef.current;
    if (!navEl) return 140;
    const topBar = navEl.querySelector('.card-nav-top') as HTMLElement | null;
    if (!topBar) return 140;
    const saved = navEl.style.width;
    navEl.style.width = 'max-content';
    const w = Math.ceil(topBar.getBoundingClientRect().width);
    navEl.style.width = saved;
    return w;
  };

  // Measure expanded width by temporarily letting nav shrink-wrap
  const getExpandedWidth = () => {
    const navEl = navRef.current;
    if (!navEl) return 400;
    const contentEl = navEl.querySelector('.card-nav-content') as HTMLElement | null;
    if (!contentEl) return getCollapsedWidth() + 200;

    const savedNavW   = navEl.style.width;
    const savedNavOvf = navEl.style.overflow;
    const savedVis    = contentEl.style.visibility;
    const savedPe     = contentEl.style.pointerEvents;
    const savedW      = contentEl.style.width;

    navEl.style.width             = 'max-content';
    navEl.style.overflow          = 'visible';
    contentEl.style.visibility    = 'visible';
    contentEl.style.pointerEvents = 'auto';
    contentEl.style.width         = 'auto';
    navEl.offsetWidth; // force reflow

    const measured = Math.ceil(navEl.getBoundingClientRect().width);

    navEl.style.width             = savedNavW;
    navEl.style.overflow          = savedNavOvf;
    contentEl.style.visibility    = savedVis;
    contentEl.style.pointerEvents = savedPe;
    contentEl.style.width         = savedW;

    return measured;
  };

  const buildTimeline = (resetPositions = true) => {
    const navEl = navRef.current;
    if (!navEl) return null;
    const collapsedW = getCollapsedWidth();
    if (resetPositions) {
      gsap.set(navEl, { width: collapsedW, overflow: 'hidden' });
      gsap.set(cardsRef.current, { x: 30, opacity: 0 });
    }
    const tl = gsap.timeline({ paused: true });
    tl.to(navEl, { width: () => getExpandedWidth(), duration: 0.4, ease });
    tl.to(cardsRef.current, { x: 0, opacity: 1, duration: 0.3, ease, stagger: 0.05 }, '-=0.15');
    return tl;
  };

  // Only rebuild timeline when collapsed — prevents jump when filterLabel changes while closing
  useLayoutEffect(() => {
    if (isExpanded) return;
    const tl = buildTimeline();
    tlRef.current = tl;
    return () => { tl?.kill(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ease, items, theme, isExpanded]);

  useLayoutEffect(() => {
    const handleResize = () => {
      if (isExpanded) {
        gsap.set(navRef.current, { width: getExpandedWidth() });
        tlRef.current?.kill();
        const tl = buildTimeline(false);
        if (tl) { tl.progress(1); tlRef.current = tl; }
      } else {
        tlRef.current?.kill();
        const tl = buildTimeline();
        if (tl) tlRef.current = tl;
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isExpanded]);

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
      const tl = buildTimeline();
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

  // Select a filter: call onSelect, then close — don't rebuild timeline
  const handleLinkSelect = (onSelect: () => void) => {
    onSelect();
    setIsHamburgerOpen(false);
    if (tlRef.current) {
      tlRef.current.eventCallback('onReverseComplete', () => setIsExpanded(false));
      tlRef.current.reverse();
    }
  };

  const setCardRef = (i: number) => (el: HTMLDivElement | null) => {
    if (el) cardsRef.current[i] = el;
  };

  return (
    <div className={`card-nav-container ${className}`}>
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

        {/* Filter links */}
        <div className="card-nav-content" aria-hidden={!isExpanded}>
          {(items || []).map((item, idx) => (
            <div
              key={`${item.label}-${idx}`}
              className="nav-card"
              ref={setCardRef(idx)}
            >
              {item.links?.map((lnk, i) => (
                <a
                  key={`${lnk.label}-${i}`}
                  className={`nav-card-link${lnk.active ? ' nav-card-link--active' : ''}`}
                  href={lnk.href ?? '#'}
                  aria-label={lnk.ariaLabel}
                  style={{ color: resolvedMenuColor }}
                  onClick={e => {
                    if (lnk.onSelect) {
                      e.preventDefault();
                      handleLinkSelect(lnk.onSelect);
                    }
                  }}
                >
                  {lnk.label}
                </a>
              ))}
            </div>
          ))}
        </div>
      </nav>
    </div>
  );
};

export default CardNav;