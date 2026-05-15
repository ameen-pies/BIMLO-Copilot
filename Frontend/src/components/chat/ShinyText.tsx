import React from "react";
import { motion, useMotionValue, useAnimationFrame, useTransform } from "framer-motion";

export const ShinyText = ({
  text,
  speed = 2.5,
  color = "hsl(var(--muted-foreground) / 0.45)",
  shineColor = "hsl(var(--foreground) / 0.85)",
  spread = 120,
  className = "",
}: {
  text: string;
  speed?: number;
  color?: string;
  shineColor?: string;
  spread?: number;
  className?: string;
}) => {
  const progress = useMotionValue(0);
  const elapsedRef = React.useRef(0);
  const lastTimeRef = React.useRef<number | null>(null);
  const animationDuration = speed * 1000;

  useAnimationFrame(time => {
    if (lastTimeRef.current === null) { lastTimeRef.current = time; return; }
    const delta = time - lastTimeRef.current;
    lastTimeRef.current = time;
    elapsedRef.current += delta;
    const cycleTime = elapsedRef.current % animationDuration;
    progress.set((cycleTime / animationDuration) * 100);
  });

  const backgroundPosition = useTransform(progress, p => `${150 - p * 2}% center`);

  return (
    <motion.span
      className={className}
      style={{
        backgroundImage: `linear-gradient(${spread}deg, ${color} 0%, ${color} 35%, ${shineColor} 50%, ${color} 65%, ${color} 100%)`,
        backgroundSize: "200% auto",
        WebkitBackgroundClip: "text",
        backgroundClip: "text",
        WebkitTextFillColor: "transparent",
        display: "inline-block",
        backgroundPosition,
      }}
    >
      {text}
    </motion.span>
  );
};
