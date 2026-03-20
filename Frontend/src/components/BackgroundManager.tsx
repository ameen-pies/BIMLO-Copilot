import React, { useState, useEffect } from 'react';
import LineWaves from '@/components/LineWaves';

const BackgroundManager = () => {
  const [opacity, setOpacity] = useState(0);
  const [isDark, setIsDark] = useState(
    () => document.documentElement.classList.contains('dark')
  );

  useEffect(() => {
    let r1: number, r2: number;
    r1 = requestAnimationFrame(() => {
      r2 = requestAnimationFrame(() => setOpacity(1));
    });
    return () => { cancelAnimationFrame(r1); cancelAnimationFrame(r2); };
  }, []);

  // Track theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const forwardMove = (e: MouseEvent) => {
      const canvas = document.querySelector<HTMLCanvasElement>('.line-waves-container canvas');
      if (!canvas) return;
      canvas.dispatchEvent(new MouseEvent('mousemove', {
        clientX: e.clientX,
        clientY: e.clientY,
        bubbles: true,
      }));
    };
    window.addEventListener('mousemove', forwardMove);
    return () => window.removeEventListener('mousemove', forwardMove);
  }, []);

  // Dark mode: light blues on dark bg
  // Light mode: deeper blues/indigos on white bg, lower brightness so they're visible but not garish
  const colors = isDark
    ? { c1: '#3b82f6', c2: '#60a5fa', c3: '#93c5fd', brightness: 0.18, vignette: 0.75 }
    : { c1: '#1d4ed8', c2: '#2563eb', c3: '#3b82f6', brightness: 0.12, vignette: 0.55 };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        willChange: 'opacity',
        opacity: opacity * 0.85,
        transition: 'opacity 0.8s ease-out',
      }}
    >
      {/* Softens center behind hero text */}
      <div style={{
        position: 'absolute',
        inset: 0,
        background: `radial-gradient(ellipse 70% 50% at 50% 40%, hsl(var(--background) / ${colors.vignette}) 0%, transparent 100%)`,
        zIndex: 1,
      }} />
      <LineWaves
        speed={0.25}
        innerLineCount={28}
        outerLineCount={32}
        warpIntensity={0.9}
        rotation={-45}
        edgeFadeWidth={0}
        colorCycleSpeed={0.6}
        brightness={colors.brightness}
        color1={colors.c1}
        color2={colors.c2}
        color3={colors.c3}
        enableMouseInteraction
        mouseInfluence={1.6}
      />
    </div>
  );
};

export default BackgroundManager;