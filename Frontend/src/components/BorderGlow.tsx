import { useRef, useCallback } from 'react';
import './BorderGlow.css';

const BorderGlow = ({
  children,
  className = '',
  glowColor = '214 100 65',
  backgroundColor = 'transparent',
  borderRadius = 16,
  glowIntensity = 1.5,
  colors = ['#60a5fa', '#3b82f6', '#93c5fd'],
  // kept for API compatibility, unused
  edgeSensitivity,
  glowRadius,
  coneSpread,
  animated,
  fillOpacity,
}) => {
  const cardRef = useRef(null);

  const buildGlowColor = (opacity) => {
    const [h, s, l] = glowColor.split(' ');
    return `hsl(${h}deg ${s}% ${l}% / ${Math.min(opacity * glowIntensity, 100)}%)`;
  };

  return (
    <div
      ref={cardRef}
      className={`simple-glow-card ${className}`}
      style={{
        '--glow-color':    buildGlowColor(100),
        '--glow-color-60': buildGlowColor(60),
        '--glow-color-40': buildGlowColor(40),
        '--glow-color-20': buildGlowColor(20),
        '--glow-color-10': buildGlowColor(10),
        '--card-bg': backgroundColor,
        '--border-radius': `${borderRadius}px`,
        '--gradient-base': `linear-gradient(${colors[0]} 0 100%)`,
        '--gradient-one':  `radial-gradient(at 20% 50%, ${colors[0]} 0px, transparent 60%)`,
        '--gradient-two':  `radial-gradient(at 80% 50%, ${colors[1]} 0px, transparent 60%)`,
        '--gradient-three':`radial-gradient(at 50% 50%, ${colors[2]} 0px, transparent 60%)`,
      } as React.CSSProperties}
    >
      <div className="simple-glow-inner">
        {children}
      </div>
    </div>
  );
};

export default BorderGlow;