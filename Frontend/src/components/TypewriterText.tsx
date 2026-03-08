import { useState, useEffect, useRef } from "react";

interface TypewriterTextProps {
  text: string;
  speed?: number;  // ms per token — same as original (speed={10} = 10ms/token = fast)
  onComplete?: () => void;
  render?: (partial: string) => React.ReactNode;
}

function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const re = /(\{\{[^}]*\}\}|\[[^\]]*\]\([^)]*\)|[*_]{1,2}[^*_\n]+[*_]{1,2}|`[^`]+`|[\s\S])/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    tokens.push(m[1]);
  }
  return tokens;
}

/**
 * TypewriterText — survives tab switches.
 *
 * speed = ms per token (same contract as original — speed={10} is fast).
 *
 * Uses requestAnimationFrame + wall-clock time instead of chained setTimeout.
 * Browsers throttle/pause setTimeout on hidden tabs; rAF + performance.now()
 * means when you switch back, the next frame catches up instantly.
 */
const TypewriterText = ({ text, speed = 20, onComplete, render }: TypewriterTextProps) => {
  const [displayedText, setDisplayedText] = useState("");
  const startTimeRef  = useRef<number>(0);
  const rafRef        = useRef<number>(0);
  const doneRef       = useRef(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    const tokens = tokenize(text);
    startTimeRef.current = performance.now();
    doneRef.current = false;
    setDisplayedText("");

    const tick = (now: number) => {
      if (doneRef.current) return;

      const elapsed = now - startTimeRef.current;
      const target  = Math.min(Math.floor(elapsed / speed), tokens.length);

      setDisplayedText(tokens.slice(0, target).join(""));

      if (target >= tokens.length) {
        doneRef.current = true;
        onCompleteRef.current?.();
        return;
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [text, speed]);

  if (render) return <>{render(displayedText)}</>;
  return <span>{displayedText}</span>;
};

export default TypewriterText;