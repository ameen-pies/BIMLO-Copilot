import React, { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { ChartRecord, ReportRecord } from "@/pages/chatTypes";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChartInsights } from "./ChartMessage";
import {
  buildInitialCfg,
  makeGradientPlugin,
  applyThemeToChart,
  downloadChartPng,
} from "@/utils/chatUtils";

const ReportInlineChart = ({ chart }: { chart: ChartRecord }) => {
  const { t } = useTranslation();
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const chartRef   = useRef<any>(null);
  const [interpExpanded, setInterpExpanded] = React.useState(false);

  useEffect(() => {
    if (!canvasRef.current) return;
    const win = window as any;
    if (!win.Chart) return;
    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
    try {
      const isDark  = document.documentElement.classList.contains("dark");
      const rawCfg  = chart.chart_js as Record<string, any>;
      const config  = buildInitialCfg(rawCfg, isDark);
      config.plugins = [makeGradientPlugin(rawCfg, isDark)];
      chartRef.current = new win.Chart(canvasRef.current, config);
    } catch (e) { console.error("Report chart render error:", e); }
    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [chart.chart_id]);

  useEffect(() => {
    let rafId = 0;
    const observer = new MutationObserver(mutations => {
      if (!mutations.some(m => m.attributeName === "class")) return;
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const c = chartRef.current;
        if (!c || !chart.chart_js) return;
        const isDark = document.documentElement.classList.contains("dark");
        c.config.plugins = [makeGradientPlugin(chart.chart_js as any, isDark)];
        applyThemeToChart(c, chart.chart_js as any, isDark);
      });
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => { observer.disconnect(); cancelAnimationFrame(rafId); };
  }, [chart]);

  const hasInterp = Boolean(chart.interpretation);

  return (
    <div className="my-4 rounded-xl border border-border bg-card overflow-hidden shadow-sm">
      <div className="flex items-center justify-between px-4 pt-3 pb-1 gap-2">
        <p className="text-[13px] font-semibold text-foreground leading-snug flex-1 truncate">
          {chart.title}
        </p>
        <button
          onClick={() => {
            const win = window as any;
            const c = chartRef.current;
            if (!c || !win) return;
            const pngName = `${(chart.title ?? "chart").replace(/\s+/g, "_")}.png`;
            downloadChartPng(c, pngName, document.documentElement.classList.contains("dark"));
          }}
          title={t("chat.download_chart_png")}
          className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted border border-transparent hover:border-border transition-all shrink-0"
        >
          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          {t("chat.chart_png")}
        </button>
      </div>

      <div className="px-4 pb-3">
        <div className="rounded-lg border border-border bg-card p-3 shadow-inner overflow-hidden">
          <div style={{ position: "relative", height: 220 }}>
            <canvas ref={canvasRef} />
          </div>
        </div>
      </div>

      {hasInterp && (
        <div className="mx-4 mb-3 rounded-lg border border-primary/20 bg-primary/5 overflow-hidden">
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
          {interpExpanded && <ChartInsights text={chart.interpretation ?? ""} />}
        </div>
      )}
    </div>
  );
};

const renderReportContent = (record: ReportRecord) => {
  const chartMap: Record<string, ChartRecord> = {};
  for (const c of (record.charts ?? [])) chartMap[c.chart_id] = c;
  const CHART_RE = /<!-- CHART:([a-zA-Z0-9_-]+) -->/g;
  const segments: Array<{ type: "text" | "chart"; value: string }> = [];
  let last = 0; let m: RegExpExecArray | null;
  const content = record.content;
  while ((m = CHART_RE.exec(content)) !== null) {
    if (m.index > last) segments.push({ type: "text", value: content.slice(last, m.index) });
    segments.push({ type: "chart", value: m[1] });
    last = m.index + m[0].length;
  }
  if (last < content.length) segments.push({ type: "text", value: content.slice(last) });

  return segments.map((seg, i) => {
    if (seg.type === "chart") {
      const chart = chartMap[seg.value];
      return chart ? <ReportInlineChart key={`rc-${i}`} chart={chart} /> : null;
    }
    return (
      <ReactMarkdown
        key={`rm-${i}`}
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="text-base font-bold mt-5 mb-2 text-foreground">{children}</h1>,
          h2: ({ children }) => <h2 className="text-sm font-bold mt-4 mb-1.5 text-foreground border-b border-border pb-1">{children}</h2>,
          h3: ({ children }) => <h3 className="text-sm font-semibold mt-3 mb-1 text-foreground">{children}</h3>,
          p: ({ children }) => <p className="mb-2 text-foreground/90 leading-relaxed text-xs">{children}</p>,
          ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5 text-foreground/90 text-xs">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5 text-foreground/90 text-xs">{children}</ol>,
          li: ({ children }) => <li className="ms-2">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
          em: ({ children }) => <em className="italic text-foreground/80">{children}</em>,
          hr: () => <hr className="border-border my-4" />,
          blockquote: ({ children }) => <blockquote className="border-s-2 border-primary/40 ps-3 italic text-muted-foreground my-2 text-xs">{children}</blockquote>,
          code: ({ inline, node, children }: { inline?: boolean; node?: any; children?: React.ReactNode }) => {
            const codeStr = String(children ?? "");
            const isInline = inline === true || (inline === undefined && !codeStr.includes("\n"));
            if (isInline) return <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono text-primary/90 whitespace-nowrap">{children}</code>;
            const codeText = codeStr.replace(/\n$/, "");
            const classList: string[] = node?.properties?.className ?? [];
            const lang = (classList.find((c: string) => c.startsWith("language-")) ?? "").replace("language-", "");
            const handleCopy = () => { navigator.clipboard.writeText(codeText).catch(() => {}); const btn = document.activeElement as HTMLElement; if (btn) { const orig = btn.textContent; btn.textContent = "Copied!"; setTimeout(() => { btn.textContent = orig; }, 1500); } };
            const handleDownload = () => { const blob = new Blob([codeText], { type: "text/plain" }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = `code.${lang || "txt"}`; a.click(); URL.revokeObjectURL(url); };
            return (
              <div className="relative my-3 rounded-lg overflow-hidden border border-border/50">
                <div className="flex items-center justify-between px-3 py-1.5 bg-muted/80 border-b border-border/40">
                  <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-wide">{lang || "code"}</span>
                  <div className="flex items-center gap-1.5">
                    <button onClick={handleDownload} className="text-[10px] text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-muted transition-colors">↓ Download</button>
                    <button onClick={handleCopy} className="text-[10px] text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-muted transition-colors">Copy</button>
                  </div>
                </div>
                <pre className="bg-muted/40 p-3 text-xs font-mono overflow-x-auto whitespace-pre leading-relaxed m-0"><code>{codeText}</code></pre>
              </div>
            );
          },
          table: ({ children }) => <div className="overflow-x-auto my-3"><table className="text-xs border-collapse w-full">{children}</table></div>,
          th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-semibold text-start text-xs">{children}</th>,
          td: ({ children }) => <td className="border border-border px-2 py-1 text-xs">{children}</td>,
        }}
      >
        {seg.value}
      </ReactMarkdown>
    );
  });
};

export { ReportInlineChart, renderReportContent };
