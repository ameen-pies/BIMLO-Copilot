import { useState, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";

const words = [
  "fiber networks",
  "5G infrastructure",
  "optical deployments",
  "telecom projects",
  "network planning",
];

const RotatingWords = () => {
  const [index, setIndex] = useState(0);
  const [width, setWidth] = useState<number | "auto">("auto");
  const measureRefs = useRef<(HTMLSpanElement | null)[]>([]);

  useEffect(() => {
    // Measure all words after mount and pick widest
    const widths = measureRefs.current.map((el) => el?.offsetWidth ?? 0);
    // Set initial width to current word's width
    if (widths[0]) setWidth(widths[0] + 8);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % words.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const el = measureRefs.current[index];
    if (el) setWidth(el.offsetWidth + 8); // +8px buffer prevents clipping
  }, [index]);

  return (
    <motion.span
      className="relative inline-block overflow-visible"
      animate={{ width }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      style={{ display: 'inline-block', paddingTop: '0.15em', paddingBottom: '0.15em' }}
    >
      {/* Hidden elements to pre-measure all words */}
      {words.map((word, i) => (
        <span
          key={word}
          ref={(el) => { measureRefs.current[i] = el; }}
          className="absolute opacity-0 pointer-events-none text-gradient-blue whitespace-nowrap font-heading text-5xl sm:text-6xl lg:text-7xl font-bold"
          style={{ lineHeight: 1.3 }}
          aria-hidden="true"
        >
          {word}
        </span>
      ))}

      <AnimatePresence mode="wait">
        <motion.span
          key={words[index]}
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -30, opacity: 0 }}
          transition={{ duration: 0.45, ease: "easeInOut" }}
          className="text-gradient-blue inline-block whitespace-nowrap"
          style={{ lineHeight: 1.3, paddingBottom: '0.1em' }}
        >
          {words[index]}
        </motion.span>
      </AnimatePresence>
    </motion.span>
  );
};

export default RotatingWords;