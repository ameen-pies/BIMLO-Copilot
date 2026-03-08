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
  const measureRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % words.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Measure immediately when index changes
    if (measureRef.current) {
      const newWidth = measureRef.current.offsetWidth;
      setWidth(newWidth);
    }
  }, [index]);

  return (
    <motion.span 
      className="relative inline-block"
      animate={{ width }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      style={{ display: 'inline-block' }}
    >
      {/* Hidden element to measure width */}
      <span 
        ref={measureRef}
        className="absolute opacity-0 pointer-events-none text-gradient-blue whitespace-nowrap font-heading text-5xl sm:text-6xl lg:text-7xl font-bold"
        aria-hidden="true"
      >
        {words[index]}
      </span>
      
      <AnimatePresence mode="wait">
        <motion.span
          key={words[index]}
          initial={{ y: 30, opacity: 0, filter: "blur(4px)" }}
          animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
          exit={{ y: -30, opacity: 0, filter: "blur(4px)" }}
          transition={{ duration: 0.45, ease: "easeInOut" }}
          className="text-gradient-blue inline-block whitespace-nowrap"
        >
          {words[index]}
        </motion.span>
      </AnimatePresence>
    </motion.span>
  );
};

export default RotatingWords;