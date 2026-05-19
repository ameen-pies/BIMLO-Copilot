import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink, ImageIcon } from "lucide-react";
import { Message, FactChip } from "@/pages/chatTypes";
import { WikiThumbnail } from "@/components/chat/WikiThumbnail";

const SourcesPanel = ({ msg, onOpenDocument }: { msg: Message; onOpenDocument: (filename: string, excerpt: string) => void }) => {
  const { t } = useTranslation();
  if (msg.role !== "assistant" || !msg.sources?.length) return null;

  const seenNums = new Set<number>();
  const citedSources = msg.sources.filter((source) => {
    const hasChips = (source as any).fact_chips?.length > 0;
    const hasLegacy = (source as any).cited_facts?.length > 0;
    if (!hasChips && !hasLegacy) return false;
    if (seenNums.has(source.source_number)) return false;
    seenNums.add(source.source_number);
    return true;
  });
  if (citedSources.length === 0) return null;

  return (
    <div className="mt-3 space-y-2">
      {citedSources.map((source) => {
        const docExcerpt: string = (source as any).excerpt ?? "";
        const sourceKey = `${msg.id}-${source.source_number}`;
        const isWiki = !!(source as any).wiki_url;
        const wikiUrl: string = (source as any).wiki_url ?? "";
        const hasImages = !!(source as any).has_images;
        const hasTables = !!(source as any).has_tables;

        const factChips: FactChip[] = (() => {
          const raw = (source as any).fact_chips;
          if (Array.isArray(raw) && raw.length > 0) return raw as FactChip[];
          const secs = (source as any).sections ?? [];
          return secs.map((sec: any) => ({
            label: sec.title ?? source.filename,
            value: sec.excerpt ?? (sec.lines?.[0] ?? ""),
            raw_line: sec.excerpt ?? (sec.lines?.[0] ?? ""),
            is_numeric: false,
          })).filter((c: FactChip) => c.value);
        })();

        if (isWiki) {
          return (
            <a
              key={source.source_number}
              id={`source-${msg.id}-${source.source_number}`}
              href={wikiUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-blue-500/25 overflow-hidden scroll-mt-4 flex flex-col hover:border-blue-500/50 transition-colors group/wiki no-underline"
            >
              <div className="flex items-center gap-2 px-3 py-2 bg-blue-500/8 group-hover/wiki:bg-blue-500/12 transition-colors">
                <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-blue-500/20 text-blue-400 font-bold text-[10px] shrink-0">
                  {source.source_number}
                </span>
                <span className="flex-1 min-w-0">
                  <span className="flex items-center gap-1.5">
                    <span className="text-[12px] font-semibold text-foreground truncate">{source.filename}</span>
                    <ExternalLink className="h-3 w-3 shrink-0 text-blue-400/70 group-hover/wiki:text-blue-400 transition-colors" />
                  </span>
                  <span className="text-[10px] text-blue-400/50 truncate block">wikipedia.org</span>
                </span>
              </div>
              {docExcerpt && (
                <div className="flex items-start gap-3 px-3 py-2 border-t border-blue-500/10 bg-background/40">
                  <p className="text-[11px] text-muted-foreground/60 leading-relaxed line-clamp-3 flex-1">{docExcerpt}</p>
                  <WikiThumbnail wikiUrl={wikiUrl} />
                </div>
              )}
            </a>
          );
        }

        return (
          <div
            key={source.source_number}
            id={`source-${msg.id}-${source.source_number}`}
            className="rounded-xl border border-primary/20 overflow-hidden scroll-mt-4"
          >
            <div className="flex items-center gap-2 px-3 py-2 bg-primary/8">
              <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-primary/20 text-primary font-bold text-[10px] shrink-0">
                {source.source_number}
              </span>
              <span className="flex-1 min-w-0 flex items-center gap-1.5 flex-wrap">
                <span
                  title={source.filename}
                  className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[9px] font-mono font-medium shrink-0 max-w-[200px] cursor-pointer hover:opacity-80 transition-opacity"
                  style={{ color: "#60a5fa", background: "color-mix(in srgb, #3b82f6 12%, transparent)", border: "1px solid color-mix(in srgb, #3b82f6 20%, transparent)" }}
                  data-open-doc
                  onClick={(e) => { e.stopPropagation(); onOpenDocument(source.filename, docExcerpt); }}
                >
                  <svg width="9" height="9" viewBox="0 0 12 12" fill="none" className="shrink-0">
                    <path d="M2 2h5l3 3v5a1 1 0 01-1 1H2a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                    <path d="M7 2v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                  </svg>
                  <span className="truncate">{source.filename}</span>
                </span>
                {factChips.length > 0 && (
                  <span className="text-[9px] text-muted-foreground/50 shrink-0">
                    {factChips.length} fact{factChips.length !== 1 ? "s" : ""}
                  </span>
                )}
                {hasImages && (
                  <span title={t("chat.contains_diagrams")} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-violet-500/15 text-violet-400 text-[9px] font-medium shrink-0">
                    <ImageIcon className="h-2.5 w-2.5" />
                    visual
                  </span>
                )}
                {hasTables && (
                  <span title={t("chat.contains_tables")} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-teal-500/15 text-teal-400 text-[9px] font-medium shrink-0">
                    table
                  </span>
                )}
              </span>
            </div>

            {factChips.length > 0 && (
              <div className="px-3 py-2.5 flex flex-wrap gap-1.5 border-t border-primary/10 bg-background/40">
                {factChips.map((chip, ci) => (
                  <button
                    key={ci}
                    title={chip.raw_line}
                    data-open-doc
                    onClick={() => onOpenDocument(source.filename, chip.raw_line)}
                    className="group/chip inline-flex items-baseline gap-1 rounded-lg border transition-all hover:border-primary/40 hover:bg-primary/5 active:scale-95"
                    style={{
                      padding: "3px 8px 3px 6px",
                      background: "color-mix(in srgb, hsl(var(--primary)) 4%, transparent)",
                      borderColor: "color-mix(in srgb, hsl(var(--primary)) 16%, transparent)",
                    }}
                  >
                    <span className="text-[9px] font-medium text-muted-foreground/60 uppercase tracking-wide leading-none group-hover/chip:text-muted-foreground transition-colors whitespace-nowrap">
                      {chip.label}
                    </span>
                    <span className="text-[8px] text-muted-foreground/30 leading-none">·</span>
                    <span className={`text-[11px] font-semibold leading-none whitespace-nowrap truncate max-w-[180px] ${chip.is_numeric ? "text-primary" : "text-foreground/80"}`} title={chip.value}>
                      {chip.value}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export { SourcesPanel };
