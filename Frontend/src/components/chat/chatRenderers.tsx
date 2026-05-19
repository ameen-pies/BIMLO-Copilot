import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MD_COMPONENTS } from "./MDComponents";
import {
  collapseInlineBreaks,
  injectChips,
  parseSegments,
  findExcerptRange,
} from "@/utils/chatUtils";
import { Source } from "@/services/api";

export function renderContent(
  text: string,
  msgId: string,
  sources: Source[] | undefined,
  onSourceClick: (sourceNum: number, msgId: string) => void,
): React.ReactNode {
  text = text
    .replace(/(?:^|\n)[ \t]*[-*+][ \t]*(\[\d+\][ \t]*)*(?=\n|$)/gm, '')
    .replace(/(?:^|\n)[ \t]*(\[\d+\][ \t]*)+(?=\n|$)/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  const citedNums = Array.from(new Set(
    Array.from(text.matchAll(/\[(\d+)\](?!\()/g)).map(m => parseInt(m[1]))
  ));
  const hasSources = sources && sources.length > 0 && citedNums.length > 0;

  if (!hasSources) {
    const cleanText = collapseInlineBreaks(text.replace(/\s*\[\d+\](?!\()/g, '').replace(/  +/g, ' ').trim());
    return (
      <div className="leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
          {cleanText}
        </ReactMarkdown>
      </div>
    );
  }

  const segments = parseSegments(text, citedNums[0]);

  return (
    <div className="leading-relaxed">
      {segments.map((seg, si) => {
        const cleanSeg = seg.text.replace(/  +/g, ' ').trim();
        if (!cleanSeg) return null;

        const components: React.ComponentProps<typeof ReactMarkdown>["components"] = {
          ...MD_COMPONENTS,
          p: ({ children }) => (
            <span className="inline leading-relaxed">
              {injectChips(children, `p-${si}`)}
            </span>
          ),
          li: ({ children }) => (
            <li className="ms-2 leading-relaxed">
              {injectChips(children, `li-${si}`)}
            </li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">
              {injectChips(children, `str-${si}`)}
            </strong>
          ),
        };

        return (
          <ReactMarkdown key={si} remarkPlugins={[remarkGfm]} components={components}>
            {cleanSeg}
          </ReactMarkdown>
        );
      })}
    </div>
  );
}

export interface ViewerState {
  doc: import("@/services/api").Document;
  content: string | null;
  loading: boolean;
  error: string | null;
  highlightText: string | null;
  highlightLines: string[] | null;
  highlightKey: number;
  blobUrl?: string;
  mediaType?: "pdf" | "image" | "txt" | "cad";
  cadSummary?: Record<string, unknown>;
  ifcBlobUrl?: string;
}

export function renderDocumentContent(
  content: string,
  highlightText: string | null,
  highlightRef: React.RefObject<HTMLElement>,
  highlightKey: number,
  highlightLines?: string[] | null,
): React.ReactNode {
  const plain = <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground/85">{content}</pre>;
  const excerpts = (highlightLines && highlightLines.length > 0) ? highlightLines : highlightText ? [highlightText] : null;
  if (!excerpts) return plain;

  type R = [number, number];
  const raw: R[] = [];
  for (const ex of excerpts) {
    if (!ex.trim()) continue;
    const r = findExcerptRange(content, ex);
    if (r) raw.push(r);
  }
  if (raw.length === 0) return plain;

  raw.sort((a, b) => a[0] - b[0]);
  const merged: R[] = [raw[0]];
  for (let i = 1; i < raw.length; i++) {
    const prev = merged[merged.length - 1];
    if (raw[i][0] <= prev[1] + 1) merged[merged.length - 1] = [prev[0], Math.max(prev[1], raw[i][1])];
    else merged.push(raw[i]);
  }

  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  merged.forEach(([start, end], idx) => {
    if (start > cursor) nodes.push(content.slice(cursor, start));
    nodes.push(
      <mark key={`hl-${highlightKey}-${idx}`}
        ref={idx === 0 ? (highlightRef as React.RefObject<HTMLElement>) : undefined}
        className="doc-highlight doc-highlight-active"
        style={{ color: 'inherit', display: 'inline', boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone' } as React.CSSProperties}>
        {content.slice(start, end)}
      </mark>
    );
    cursor = end;
  });
  if (cursor < content.length) nodes.push(content.slice(cursor));
  return <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground/85">{nodes}</pre>;
}
