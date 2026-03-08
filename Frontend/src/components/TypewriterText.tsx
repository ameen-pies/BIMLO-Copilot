import { useEffect, useRef, useState } from "react";

interface TypewriterTextProps {
  text: string;
  speed?: number; // chars per interval tick (default 1)
  onComplete?: () => void;
  render: (partial: string) => React.ReactNode;
}

/**
 * Typewriter that keeps running even when the user switches tabs.
 *
 * Root cause of the freeze: browsers throttle/suspend setTimeout when the
 * tab is hidden (Page Visibility API + background timer throttling).
 *
 * Fix: listen for `visibilitychange`. When the tab becomes visible again,
 * immediately jump the displayed text to wherever the real-time cursor
 * should be, based on elapsed wall-clock time — so no characters are lost
 * and it never "freezes" visually.
 */
const TypewriterText: React.FC<TypewriterTextProps> = ({
  text,
  speed = 1,
  onComplete,
  render,
}) => {
  const [displayed, setDisplayed] = useState("");
  const indexRef    = useRef(0);       // how many chars we've shown
  const startRef    = useRef<number>(Date.now()); // wall-clock start time
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const doneRef     = useRef(false);
  const TICK_MS     = 16; // ~60 fps tick rate
  const CHARS_PER_TICK = Math.max(1, speed); // chars to advance per tick

  // How many chars should be visible right now based on elapsed time?
  const expectedIndex = () => {
    const elapsed = Date.now() - startRef.current;
    const ticks = elapsed / TICK_MS;
    return Math.min(Math.floor(ticks * CHARS_PER_TICK), text.length);
  };

  const advance = () => {
    if (doneRef.current) return;

    const target = expectedIndex();
    if (target >= text.length) {
      setDisplayed(text);
      doneRef.current = true;
      if (intervalRef.current) clearInterval(intervalRef.current);
      onComplete?.();
      return;
    }

    if (target > indexRef.current) {
      indexRef.current = target;
      setDisplayed(text.slice(0, target));
    }
  };

  // When tab becomes visible again, snap to the correct position immediately
  const onVisibilityChange = () => {
    if (!document.hidden) {
      advance();
    }
  };

  useEffect(() => {
    // Reset on new text
    indexRef.current  = 0;
    doneRef.current   = false;
    startRef.current  = Date.now();
    setDisplayed("");

    intervalRef.current = setInterval(advance, TICK_MS);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return <>{render(displayed)}</>;
};

export default TypewriterText;