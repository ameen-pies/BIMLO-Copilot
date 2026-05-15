import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import Logo from "@/components/Logo";
import { getGreetingBucket } from "@/utils/chatUtils";

const GREETING_COUNTS: Record<string, number> = {
  night: 7, morning: 8, afternoon: 8, evening: 8,
};

function getGreetingKey(): string {
  const bucket = getGreetingBucket();
  const index = Math.floor(Math.random() * GREETING_COUNTS[bucket]);
  return `chat.greetings.${bucket}.${index}`;
}

export const WelcomeSplash: React.FC<{ visible: boolean }> = ({ visible }) => {
  const { t } = useTranslation();
  const [greetingKey] = useState(getGreetingKey);
  return (
  <AnimatePresence>
    {visible && (
      <motion.div
        key="welcome-splash"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0, transition: { duration: 0.2 } }}
        transition={{ duration: 0.3 }}
        className="absolute inset-0 flex flex-col items-center pointer-events-none select-none z-10"
        style={{ justifyContent: "center", paddingBottom: "140px" }}
      >
        <div className="absolute inset-0 -z-10 flex items-center justify-center">
          <div className="h-48 w-48 rounded-full bg-primary/6 blur-3xl" />
        </div>
        <div className="flex items-center gap-4">
          <Logo className="h-12 w-12 opacity-90" />
          <div className="flex flex-col gap-0.5">
            <h1 className="text-3xl font-semibold tracking-tight text-foreground/90 leading-tight">
              {t(greetingKey)}
            </h1>
            <p className="text-sm text-muted-foreground/50 leading-relaxed">
              {t("common.welcome_splash")}
            </p>
          </div>
        </div>
      </motion.div>
    )}
  </AnimatePresence>
  );
};
