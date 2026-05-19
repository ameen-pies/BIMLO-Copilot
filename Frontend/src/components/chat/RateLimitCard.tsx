import React from "react";

interface RateLimitCardProps {
  onRetry: () => void;
  onOpenModelSelector?: () => void;
}

export default function RateLimitCard({ onRetry, onOpenModelSelector }: RateLimitCardProps) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm max-w-lg">
      <span className="text-xl mt-0.5">⚠️</span>
      <div className="flex-1 space-y-2">
        <p className="font-medium text-amber-200">Rate limit reached</p>
        <p className="text-amber-100/70 text-xs">
          All LLM providers are currently rate-limited. Please switch to a different model or try again later.
        </p>
        <div className="flex gap-2 pt-1">
          <button
            onClick={onRetry}
            className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-100 text-xs font-medium transition-colors"
          >
            Retry
          </button>
          {onOpenModelSelector && (
            <button
              onClick={onOpenModelSelector}
              className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-amber-100/70 text-xs font-medium transition-colors"
            >
              Change model
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
