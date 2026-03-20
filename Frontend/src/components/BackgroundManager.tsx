import React, { useState } from 'react';
import NetworkBackground from '@/components/NetworkBackground';
import LiquidEther from '@/components/LiquidEther';

/**
 * BackgroundManager — re-rolls on every full page load.
 * Stores the pick in sessionStorage so SPA navigation (going to /chat
 * and back) keeps the same background without re-rolling.
 *
 * How: on mount we check if this React tree was just freshly mounted
 * (module-level flag `didInit` is false on first mount after a page load).
 * If fresh → roll a new id and save it. If re-mount (HMR / StrictMode
 * double-invoke) → reuse the saved id.
 */

const BACKGROUND_IDS = ['network', 'liquid-ether'] as const;
type BgId = typeof BACKGROUND_IDS[number];

const SESSION_KEY = 'bimlo_bg';

// Module-level: reset to false on every true page load (JS module re-evaluation)
let pickedThisLoad = false;

function getOrRoll(): BgId {
  if (!pickedThisLoad) {
    // First mount after a real page load — always pick fresh
    pickedThisLoad = true;
    const idx = Math.floor(Math.random() * BACKGROUND_IDS.length);
    const id = BACKGROUND_IDS[idx];
    sessionStorage.setItem(SESSION_KEY, id);
    return id;
  }
  // Re-mount (StrictMode, HMR, SPA nav) — reuse stored pick
  const stored = sessionStorage.getItem(SESSION_KEY) as BgId | null;
  if (stored && BACKGROUND_IDS.includes(stored)) return stored;
  return BACKGROUND_IDS[0];
}

function renderBg(id: BgId): React.ReactNode {
  if (id === 'liquid-ether') {
    return (
      <div
        className="fixed inset-0 pointer-events-none"
        style={{ zIndex: 0, opacity: 0.6 }}
      >
        <LiquidEther
          colors={['#5227FF', '#FF9FFC', '#B19EEF']}
          mouseForce={9}
          cursorSize={120}
          isViscous
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce
          autoDemo
          autoSpeed={0.5}
          autoIntensity={2.2}
          takeoverDuration={0.25}
          autoResumeDelay={3000}
          autoRampDuration={0.6}
        />
      </div>
    );
  }
  return <NetworkBackground />;
}

const BackgroundManager = () => {
  const [id] = useState<BgId>(getOrRoll);
  return <>{renderBg(id)}</>;
};

export default BackgroundManager;