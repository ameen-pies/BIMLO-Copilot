import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, Search, ScrollText, Pencil, Check, Sparkles, ChevronDown } from "lucide-react";
import { ThinkingStep } from "@/pages/chatTypes";
import { ShinyText } from "./ShinyText";
import TypingIndicator from "@/components/TypingIndicator";

interface ThinkingTraceProps {
  steps: ThinkingStep[];
  isExpanded: boolean;
  onToggle: () => void;
  showDots?: boolean;
  isLoading?: boolean;
}

export const ThinkingTrace: React.FC<ThinkingTraceProps> = ({
  steps,
  isExpanded,
  onToggle,
  showDots = false,
  isLoading = false,
}) => {
  if (steps.length === 0) return null;

  const cls = "h-3 w-3 shrink-0 text-muted-foreground/35";

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-1.5"
      >
        <AnimatePresence mode="wait">
          {steps.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4, transition: { duration: 0.25 } }}
              transition={{ duration: 0.2 }}
              className="flex flex-col gap-1"
            >
              <button
                onClick={onToggle}
                className="flex items-center gap-1.5 group w-fit"
              >
                <motion.span
                  className="inline-block h-1.5 w-1.5 rounded-full bg-primary/50 shrink-0"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
                />
                <motion.span
                  key={steps[steps.length - 1]?.message}
                  initial={{ opacity: 0, y: 2 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                  className="text-[11px] italic leading-none"
                >
                  <ShinyText
                    text={steps[steps.length - 1]?.message ?? "Thinking…"}
                    speed={2.2}
                  />
                </motion.span>
                <motion.span
                  animate={{ rotate: isExpanded ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                  className="opacity-0 group-hover:opacity-60 transition-opacity"
                >
                  <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
                </motion.span>
              </button>
              <AnimatePresence>
                {isExpanded && steps.length > 1 && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.18 }}
                    className="overflow-hidden"
                  >
                    <div className="flex flex-col gap-0.5 pl-3 border-l border-border/25 ml-[2px]">
                      {steps.slice(0, -1).map((step, i) => {
                        const icon =
                          step.node === "retrieve"       ? <FileText className={cls} /> :
                          step.node === "rewrite_query"  ? <Search className={cls} /> :
                          step.node === "judge_plan"     ? <ScrollText className={cls} /> :
                          step.node === "synthesise"     ? <Pencil className={cls} /> :
                          step.node === "judge_evaluate" ? <Check className={cls} /> :
                                                           <Sparkles className={cls} />;
                        return (
                          <motion.div
                            key={`${step.node}-${i}`}
                            initial={{ opacity: 0, x: -4 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.12, delay: i * 0.03 }}
                            className="flex items-center gap-1.5"
                          >
                            {icon}
                            <ShinyText text={step.message} speed={3.5} className="text-[10px] leading-snug" />
                          </motion.div>
                        );
                      })}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
        {showDots && (
          <div className="flex gap-3 items-center">
            <div className="bg-secondary rounded-2xl rounded-bl-md px-4 py-3 inline-block">
              <TypingIndicator />
            </div>
          </div>
        )}
      </motion.div>
    );
  }

  return (
    <div className="flex flex-col gap-1 pl-0 mb-1">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 group/think w-fit"
      >
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary/30 shrink-0" />
        <span className="text-[11px] text-muted-foreground/40 italic leading-none">
          {steps[steps.length - 1]?.message ?? "Thought process"}
        </span>
        <motion.span
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="opacity-0 group-hover/think:opacity-60 transition-opacity"
        >
          <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
        </motion.span>
      </button>
      <AnimatePresence>
        {isExpanded && steps.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-0.5 pl-3 border-l border-border/25 ml-[2px]">
              {steps.map((step, i) => {
                const icon =
                  step.node === "retrieve"       ? <FileText className={cls} /> :
                  step.node === "rewrite_query"  ? <Search className={cls} /> :
                  step.node === "judge_plan"     ? <ScrollText className={cls} /> :
                  step.node === "synthesise"     ? <Pencil className={cls} /> :
                  step.node === "judge_evaluate" ? <Check className={cls} /> :
                                                   <Sparkles className={cls} />;
                return (
                  <div key={`${step.node}-${i}`} className="flex items-center gap-1.5">
                    {icon}
                    <span className="text-[10px] text-muted-foreground/35 leading-snug">{step.message}</span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
