import React, { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { FileText, BarChart2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MD_COMPONENTS } from "./MDComponents";
import {
  splitIntoSentences,
  INSIGHT_LABELS,
  INSIGHT_ICONS,
  buildInitialCfg,
  makeGradientPlugin,
  applyThemeToChart,
  downloadChartPng,
} from "@/utils/chatUtils";

export interface ChartGroup {
  label:        string;
  description:  string;
  hint:         string;
  source_file?: string;
}

export interface ChartAnalytics {
  type: "chart_config" | "chart_clarification" | "report_chart_clarification" | "chart_error";
  chart_js?:            Record<string, any>;
  chart_type?:          string;
  title?:               string;
  description?:         string;
  interpretation?:      string;
  sources?:             string[];
  question?:            string;
  groups?:              ChartGroup[];
  clarification_mode?:  "file" | "metric";
  message?:             string;
}

export interface ChartMessageProps {
  analytics: ChartAnalytics;
  answer:    string;
}

export const ChartInsights: React.FC<{ text: string }> = ({ text }) => {
  const sentences = splitIntoSentences(text);
  const isSingle  = sentences.length === 1;

  return (
    <div className="border-t border-primary/10 px-3 pt-3 pb-3 flex flex-col gap-2.5">
      {isSingle ? (
        <p className="text-[12px] text-muted-foreground leading-relaxed">{text}</p>
      ) : (
        sentences.map((sentence, idx) => (
          <div key={idx} className="flex gap-2.5 items-start">
            <span className="text-primary/60 mt-0.5">
              {INSIGHT_ICONS[idx % INSIGHT_ICONS.length]}
            </span>
            <div className="flex flex-col gap-0.5 min-w-0">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-primary/50 leading-none">
                {INSIGHT_LABELS[idx % INSIGHT_LABELS.length]}
              </span>
              <p className="text-[12px] text-muted-foreground leading-relaxed">
                {sentence}
              </p>
            </div>
          </div>
        ))
      )}
    </div>
  );
};

export const ChartMessage: React.FC<ChartMessageProps> = ({ analytics, answer }) => {
  const { t } = useTranslation();
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const [interpExpanded, setInterpExpanded] = useState(false);

  const getIsDark = () => document.documentElement.classList.contains("dark");

  useEffect(() => {
    if (!canvasRef.current || !analytics.chart_js) return;
    import("chart.js/auto").then(({ Chart }) => {
      if (chartInstanceRef.current) { chartInstanceRef.current.destroy(); chartInstanceRef.current = null; }
      const isDark  = getIsDark();
      const cfg     = buildInitialCfg(analytics.chart_js!, isDark);
      const gradPlugin = makeGradientPlugin(analytics.chart_js!, isDark);
      chartInstanceRef.current = new Chart(canvasRef.current!, { ...cfg, plugins: [gradPlugin] } as any);
    });
    return () => { chartInstanceRef.current?.destroy(); chartInstanceRef.current = null; };
  }, [analytics]);

  useEffect(() => {
    let rafId = 0;
    const observer = new MutationObserver(mutations => {
      if (!mutations.some(m => m.attributeName === "class")) return;
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const chart = chartInstanceRef.current;
        if (!chart || !analytics.chart_js) return;
        const isDark = getIsDark();
        chart.config.plugins = [makeGradientPlugin(analytics.chart_js!, isDark)];
        applyThemeToChart(chart, analytics.chart_js!, isDark);
      });
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => { observer.disconnect(); cancelAnimationFrame(rafId); };
  }, [analytics]);

  if (analytics.type === "chart_error") {
    return (
      <div className="leading-relaxed text-sm text-foreground">
        {answer && (
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{answer}</ReactMarkdown>
        )}
      </div>
    );
  }

  const hasInterp = Boolean(analytics.interpretation);
  const pngName   = `${(analytics.title ?? "chart").replace(/\s+/g, "_")}.png`;

  return (
    <div className="flex flex-col gap-4 w-full">

      {answer && (
        <div className="leading-relaxed text-sm text-foreground">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{answer}</ReactMarkdown>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-semibold text-foreground leading-snug flex-1 truncate">
            {analytics.title || analytics.chart_js?.options?.plugins?.title?.text || ""}
          </span>
          <div className="flex items-center gap-1.5 shrink-0 ml-auto">
            {analytics.sources && analytics.sources.length > 0 && (
              <span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-muted text-muted-foreground whitespace-nowrap">
                {analytics.sources[0]}{analytics.sources.length > 1 ? ` +${analytics.sources.length - 1}` : ""}
              </span>
            )}
            <button
              onClick={() => downloadChartPng(chartInstanceRef.current, pngName, getIsDark())}
              title={t("chat.download_chart_png")}
              className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted border border-transparent hover:border-border transition-all"
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              PNG
            </button>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-inner overflow-hidden">
          <canvas ref={canvasRef} />
          {(() => {
            const cfg    = analytics.chart_js;
            const labels = cfg?.data?.labels as string[] | undefined;
            const ds0    = cfg?.data?.datasets?.[0];
            if (!labels || labels.length === 0) return null;
            const isPie  = cfg?.type === "pie" || cfg?.type === "doughnut";
            const isDarkNow = typeof window !== "undefined" && document.documentElement.classList.contains("dark");
            const palette = isDarkNow
              ? ["#818cf8","#34d399","#fbbf24","#fb7185","#60a5fa","#f472b4","#a78bfa","#2dd4bf"]
              : ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#ec4899","#8b5cf6","#14b8a6"];
            return (
              <div className="flex flex-wrap justify-center gap-x-4 gap-y-1.5 mt-3 px-1">
                {labels.map((label: string, i: number) => {
                  const color = isPie
                    ? palette[i % palette.length]
                    : (Array.isArray(ds0?.backgroundColor)
                        ? ds0.backgroundColor[i % ds0.backgroundColor.length]
                        : ds0?.borderColor ?? palette[i % palette.length]);
                  return (
                    <span key={i} className="flex items-center gap-1.5 text-[11px] font-normal text-muted-foreground">
                      <span style={{ background: color, width: 8, height: 8, borderRadius: "50%", display: "inline-block", flexShrink: 0 }} />
                      {label}
                    </span>
                  );
                })}
              </div>
            );
          })()}
        </div>

        {hasInterp && (
          <div className="rounded-lg border border-primary/20 bg-primary/5 overflow-hidden">
            <button
              onClick={() => setInterpExpanded(v => !v)}
              className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium text-primary/80 hover:text-primary transition-colors"
            >
              <span className="flex items-center gap-1.5">
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.347a3.252 3.252 0 01-4.78 0l-.347-.347z" />
                </svg>
                Key insights
              </span>
              <svg className={`h-3.5 w-3.5 transition-transform ${interpExpanded ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {interpExpanded && <ChartInsights text={analytics.interpretation ?? ""} />}
          </div>
        )}
      </div>
    </div>
  );
};

export interface ChartClarificationProps {
  analytics: ChartAnalytics;
  onSelect:  (hint: string) => void;
}

export const ChartClarification: React.FC<ChartClarificationProps> = ({ analytics, onSelect }) => {
  const groups = analytics.groups ?? [];
  const isFilePicker = analytics.clarification_mode === "file";
  const [selected, setSelected] = React.useState<string | null>(null);

  if (selected !== null) {
    return (
      <p className="text-sm text-muted-foreground italic">
        Got it — building chart for <span className="text-foreground font-medium">{selected}</span>…
      </p>
    );
  }

  const fileGroups: Record<string, ChartGroup[]> = {};
  if (isFilePicker) {
    groups.forEach(g => {
      const file = g.source_file ?? "Other";
      if (!fileGroups[file]) fileGroups[file] = [];
      fileGroups[file].push(g);
    });
  }

  return (
    <div className="flex flex-col gap-3 w-full">
      <p className="text-sm leading-relaxed text-foreground">
        {analytics.question ?? "I found several types of data in your documents. Which would you like to chart?"}
      </p>

      {isFilePicker ? (
        <div className="flex flex-col gap-3">
          {Object.entries(fileGroups).map(([filename, items], fi) => (
            <div key={fi} className="flex flex-col gap-1.5">
              <div className="flex items-center gap-1.5 px-0.5">
                <FileText className="h-3 w-3 text-primary/50 shrink-0" />
                <span className="text-[11px] font-semibold text-primary/60 uppercase tracking-wide truncate">
                  {filename}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {items.map((g, i) => (
                  <motion.button
                    key={i}
                    initial={{ opacity: 0, scale: 0.94 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (fi * 3 + i) * 0.04 }}
                    onClick={() => { setSelected(g.label); onSelect(g.hint); }}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border hover:border-primary/50 bg-muted/40 hover:bg-primary/10 text-left transition-all group/opt"
                  >
                    <BarChart2 className="h-3 w-3 text-primary/40 shrink-0 group-hover/opt:text-primary transition-colors" />
                    <span className="text-[12px] font-medium text-foreground group-hover/opt:text-primary transition-colors">
                      {g.label}
                    </span>
                  </motion.button>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {groups.map((g, i) => (
            <motion.button
              key={i}
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
              onClick={() => { setSelected(g.label); onSelect(g.hint); }}
              className="flex flex-col items-start gap-0.5 px-3 py-2 rounded-xl border border-border hover:border-primary/50 bg-muted/40 hover:bg-primary/10 text-left transition-all group/opt max-w-[220px]"
            >
              <span className="text-[13px] font-medium text-foreground group-hover/opt:text-primary transition-colors leading-snug">
                {g.label}
              </span>
              {g.description && (
                <span className="text-[11px] text-muted-foreground leading-snug line-clamp-2">
                  {g.description}
                </span>
              )}
            </motion.button>
          ))}
        </div>
      )}
    </div>
  );
};
