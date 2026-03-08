import { motion } from "framer-motion";

const TypingIndicator = () => {
  return (
    <div className="flex gap-1 items-center py-1">
      {[0, 1, 2].map((index) => (
        <motion.div
          key={index}
          className="w-1.5 h-1.5 rounded-full bg-muted-foreground/60"
          animate={{
            y: [0, -6, 0],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            ease: "easeInOut",
            delay: index * 0.15,
            repeatDelay: 0.5,
          }}
        />
      ))}
    </div>
  );
};

export default TypingIndicator;