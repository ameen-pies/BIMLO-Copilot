import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, FileText, X, User, ArrowLeft, Plus, Loader2, AlertCircle, ChevronDown, ChevronUp, ExternalLink, ScrollText, Eye, Square, ThumbsUp, ThumbsDown, RotateCcw, Pencil, Check, Copy, ImageIcon, Search, MessageSquare, Clock, SortAsc, FolderOpen, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import ThemeToggle from "@/components/ThemeToggle";
import TypingIndicator from "@/components/TypingIndicator";
import TypewriterText from "@/components/TypewriterText";
import Logo from "@/components/Logo";
import api, { Document, Source } from "@/services/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import BorderGlow from "@/components/BorderGlow";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rawAnswer?: string;
  sources?: Source[];
  confidence?: number;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messages: Message[];
}

// ---------------------------------------------------------------------------
// Markdown rendering helpers
// ---------------------------------------------------------------------------

/**
 * Returns true if the text contains block-level markdown that needs full
 * ReactMarkdown rendering (lists, headings, code blocks, tables, blockquotes).
 * Plain sentences — even with **bold** or *italic* — are inline-safe.
 */
function hasBlockMarkdown(text: string): boolean {
  return /^[\s]*([-*+]|\d+\.)\s/m.test(text)   // lists
    || /^#{1,6}\s/m.test(text)                  // headings
    || /^```/m.test(text)                        // fenced code
    || /^\|.+\|/m.test(text)                     // tables
    || /^>/m.test(text);                         // blockquotes
}

/** Strip markdown links, keeping label text only. */
function stripLinks(text: string): string {
  return text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
}

/**
 * Render inline bold/italic/code inside a plain string without wrapping in <p>.
 * Handles **bold**, *italic*, and `code` only.
 */
function renderInlineMarkdown(text: string): React.ReactNode[] {
  // Token: **bold**, *italic*, `code`, or plain char runs
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[1];
    if (tok.startsWith("**")) {
      parts.push(<strong key={i++} className="font-semibold text-foreground">{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("*")) {
      parts.push(<em key={i++} className="italic text-foreground/80">{tok.slice(1, -1)}</em>);
    } else {
      parts.push(<code key={i++} className="bg-muted px-1 py-0.5 rounded text-xs font-mono">{tok.slice(1, -1)}</code>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

const MD_COMPONENTS: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  // Suppress the wrapping <p> so block segments don't inject extra line breaks
  // around inline citation buttons sitting next to them.
  p: ({ children }) => <span className="inline leading-relaxed">{children}</span>,
  ul: ({ children }) => <ul className="list-disc list-inside my-2 space-y-1 block">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside my-2 space-y-1 block">{children}</ol>,
  li: ({ children }) => <li className="ml-2 leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  em: ({ children }) => <em className="italic text-foreground/80">{children}</em>,
  h1: ({ children }) => <h1 className="text-base font-bold mb-2 mt-4 block">{children}</h1>,
  h2: ({ children }) => <h2 className="text-sm font-bold mb-1.5 mt-3 first:mt-0 block">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0 block">{children}</h3>,
  code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
    inline ? (
      <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono">{children}</code>
    ) : (
      <pre className="bg-muted rounded-lg p-3 text-xs font-mono overflow-x-auto my-3 whitespace-pre-wrap block">
        <code>{children}</code>
      </pre>
    ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-primary/40 pl-3 italic text-muted-foreground my-2 block">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-border my-4" />,
  a: ({ href, children }) => (
    <span className="inline-flex items-baseline gap-px">
      <span>{children}</span>
      {href && href !== "#" && (
        <a href={href} target="_blank" rel="noopener noreferrer" title={href}
          className="inline-flex items-center ml-0.5 text-primary/50 hover:text-primary transition-colors"
          onClick={(e) => e.stopPropagation()}>
          <ExternalLink className="h-2.5 w-2.5 translate-y-[-1px]" strokeWidth={2.5} />
        </a>
      )}
    </span>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-3">
      <table className="text-xs border-collapse w-full">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-semibold text-left">{children}</th>,
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
  br: () => <br />,
};

/**
 * Render a non-citation text segment.
 *
 * Strategy:
 * - Split on double-newlines to get paragraphs.
 * - Each paragraph that only contains inline content (bold/italic/plain) is
 *   rendered as an inline <span> — this prevents ReactMarkdown from injecting
 *   a block <p> that forces a line-break around adjacent citation buttons.
 * - Paragraphs with real block elements (lists, headings, etc.) fall back to
 *   full ReactMarkdown so those render correctly.
 * - A blank line between paragraphs gets a small spacer div.
 */
function renderTextSegment(raw: string, segKey: number): React.ReactNode {
  const cleaned = stripLinks(raw);
  const paragraphs = cleaned.split(/\n\n+/);

  return (
    <span key={segKey}>
      {paragraphs.map((para, pIdx) => (
        <span key={pIdx}>
          {hasBlockMarkdown(para) ? (
            // Block content: full markdown, rendered as a block element
            <span className="block">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                {para.trim()}
              </ReactMarkdown>
            </span>
          ) : (
            // Inline content: render bold/italic ourselves, no wrapping <p>
            <span>{renderInlineMarkdown(para)}</span>
          )}
          {pIdx < paragraphs.length - 1 && <span className="block h-2" />}
        </span>
      ))}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Spec-number chip highlighting
// ---------------------------------------------------------------------------

// Units ordered longest→shortest within each group so the regex alternation
// always tries the most-specific match first.  Case-sensitive for ambiguous
// SI prefixes (M=Mega vs m=milli/meters, K=kilo prefix vs k=kilo, G vs g).
const UNIT_PATTERN = [
  // Data-rate — must come before bare Gbit/Mbit/kbit and bare G/M/k
  'Gbps','Mbps','kbps',
  // Frequency
  'GHz','MHz','kHz','Hz',
  // Data size
  'Gbit\\/s','Mbit\\/s','kbit\\/s','Gbit','Mbit','kbit','GB','MB','KB','TB',
  // Time — ms/ns/µs before bare 's', 'min' before 'm'
  'ms','ns','µs','min','hrs?','hours?',
  // Power / electrical — kW/MW before bare W; kHz already above
  'kW','MW','dBm','dB','GHz','MHz','kHz','Hz','V','A','Ω',
  // Weight
  'kg','mg',
  // Temperature
  '°C','°F',
  // Distance — km before m; cm/mm before m
  'km','cm','mm','m²','m³','m',
  // Area
  'ha','acres?',
  // Imperial distance
  'mi','ft','in',
  // Currency
  '€','\\$','£','¥','CHF','EUR','USD',
  // Other
  '%','ppm','rpm',
  // Bare SI letters — only after all compound units above
  'K','W','g','s',
  // Count nouns
  'sites?','users?','nodes?','ports?',
].join('|');
// Case-SENSITIVE so "M" (Mega) ≠ "m" (meters), "K" ≠ "k", etc.
const UNIT_RE = new RegExp(`^\\s*(?:${UNIT_PATTERN})(?![a-zA-Z])`);

function isSpecNumber(raw: string, after: string): boolean {
  if (UNIT_RE.test(after.slice(0, 14))) return true;
  if (/^\d+:\d+$/.test(raw.trim())) return true;
  if (/^\d+-\d+$/.test(raw.trim())) return false;
  const digits = raw.replace(/[\s,.]/g, '');
  if (digits.length >= 4 && /[\s,.]/.test(raw)) return true;
  return false;
}

/** Wrap spec-numbers in visual-only highlight chips (not clickable). */
function highlightNumbers(
  text: string,
  keyPrefix: string,
): React.ReactNode[] {
  const numRe = /(\d[\d\s,.:]*)/g;
  const result: React.ReactNode[] = [];
  let last = 0, ni = 0, m: RegExpExecArray | null;
  while ((m = numRe.exec(text)) !== null) {
    const raw = m[1];
    const after = text.slice(m.index + m[0].length);
    if (!isSpecNumber(raw.trim(), after)) continue;
    if (m.index > last) result.push(text.slice(last, m.index));
    const unitMatch = after.match(new RegExp(`^\\s*(?:${UNIT_PATTERN})(?![a-zA-Z])`));
    const unit = unitMatch ? unitMatch[0] : '';
    const displayVal = raw.trim() + unit;
    result.push(
      <span
        key={`${keyPrefix}-n${ni++}`}
        className="inline-flex items-center px-1 py-0 rounded bg-primary/15 text-primary font-semibold text-[0.82em] border border-primary/20 mx-0.5"
      >
        {displayVal}
      </span>
    );
    last = m.index + m[0].length + unit.length;
  }
  if (last < text.length) result.push(text.slice(last));
  return result;
}

/** Recursively walk React children injecting visual number chips. */
function injectChips(
  children: React.ReactNode,
  keyPrefix: string,
): React.ReactNode {
  if (typeof children === 'string') {
    const parts = highlightNumbers(children, keyPrefix);
    return parts.length === 1 && typeof parts[0] === 'string' ? children : <>{parts}</>;
  }
  if (Array.isArray(children)) {
    return <>{children.map((c, i) => injectChips(c, `${keyPrefix}-${i}`))}</>;
  }
  return children;
}

// ---------------------------------------------------------------------------
// Main content renderer
// ---------------------------------------------------------------------------

/**
 * Parse the raw answer into segments, each bound to a source number.
 *
 * Key insight: [N] appears AFTER the text it cites (end-of-sentence).
 * So the correct source for a piece of text is the [N] that comes
 * immediately AFTER it — i.e. we look AHEAD, not behind.
 *
 * Algorithm:
 *   1. Find all [N] positions in the raw string.
 *   2. For each character position, its source = the [N] immediately
 *      to its right (lookahead). If none exists to the right, use the
 *      last [N] seen (fallback).
 *   3. Split the string on [N] tokens, and each text chunk gets the
 *      source of the [N] that closes it.
 *
 * This means every number chip in a sentence gets attributed to the
 * correct [N] that ends that sentence, not the previous one.
 */
function parseSegments(raw: string, fallbackSource: number): { text: string; source: number }[] {
  const tokens = raw.split(/(\[\d+\](?!\())/);
  const segments: { text: string; source: number }[] = [];

  // Walk tokens: text tok → buffer it; [N] tok → flush buffer with this N as source
  let buf = '';
  let lastSource = fallbackSource;

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const citeMatch = tok.match(/^\[(\d+)\]$/);
    if (citeMatch) {
      const n = parseInt(citeMatch[1]);
      // Flush whatever text was buffered — it belongs to this [N]
      if (buf.trim()) {
        segments.push({ text: buf, source: n });
      }
      buf = '';
      lastSource = n;
    } else {
      buf += tok;
    }
  }

  // Trailing text after the last [N] stays attributed to lastSource
  if (buf.trim()) {
    segments.push({ text: buf, source: lastSource });
  }

  return segments;
}

/**
 * Parse the raw answer into source groups for the new pill/card UI.
 *
 * For each unique [N] citation, find:
 *   - The nearest ## heading above it (section title)
 *   - All the bullet lines / sentences that cite [N]
 *
 * Returns a map: sourceNum → { sectionTitle, lines[] }
 */
function parseSourceGroups(raw: string, sources: Source[]): Map<number, { sectionTitle: string; lines: string[] }> {
  const result = new Map<number, { sectionTitle: string; lines: string[] }>();

  // Split into lines, track current heading
  const rawLines = raw.split('\n');
  let currentHeading = '';

  for (const line of rawLines) {
    // Detect ## headings
    const headingMatch = line.match(/^#{1,3}\s+(.+)/);
    if (headingMatch) {
      currentHeading = headingMatch[1].trim();
      continue;
    }

    // Find all [N] citations in this line
    const citations = Array.from(line.matchAll(/\[(\d+)\](?!\()/g)).map(m => parseInt(m[1]));
    if (citations.length === 0) continue;

    // Clean the line — remove [N] markers and markdown bold/bullet
    const cleanLine = line
      .replace(/\[\d+\](?!\()/g, '')
      .replace(/^\s*[-*•]\s*/, '')
      .replace(/\*\*/g, '')
      .trim();

    if (!cleanLine) continue;

    // Add this line under each cited source
    for (const n of citations) {
      if (!result.has(n)) {
        result.set(n, { sectionTitle: currentHeading, lines: [] });
      }
      const group = result.get(n)!;
      // Update heading if this is the first line or heading changed
      if (group.lines.length === 0) {
        group.sectionTitle = currentHeading;
      }
      if (!group.lines.includes(cleanLine)) {
        group.lines.push(cleanLine);
      }
    }
  }

  return result;
}


function renderContent(
  text: string,
  msgId: string,
  sources: Source[] | undefined,
  onSourceClick: (sourceNum: number, msgId: string) => void,
): React.ReactNode {
  // Strip orphan bullets FIRST — lines whose only content after the dash is [N] markers or whitespace.
  // Must happen before citedNums so we don't count markers that only existed on orphan lines.
  text = text
    .replace(/(?:^|\n)[ \t]*[-*+][ \t]*(\[\d+\][ \t]*)*(?=\n|$)/gm, '')
    .replace(/(?:^|\n)[ \t]*(\[\d+\][ \t]*)+(?=\n|$)/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  // Compute citedNums from the CLEANED text so hasSources reflects what actually renders
  const citedNums = Array.from(new Set(
    Array.from(text.matchAll(/\[(\d+)\](?!\()/g)).map(m => parseInt(m[1]))
  ));
  const hasSources = sources && sources.length > 0 && citedNums.length > 0;

  // No sources — plain markdown render
  if (!hasSources) {
    const cleanText = text.replace(/\s*\[\d+\](?!\()/g, '').replace(/  +/g, ' ').trim();
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
            <li className="ml-2 leading-relaxed">
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




// ---------------------------------------------------------------------------
// Excerpt highlighting
// ---------------------------------------------------------------------------

/**
 * Normalize a string for fuzzy comparison:
 * collapse whitespace, strip punctuation differences.
 */
function norm(s: string): string {
  return s.toLowerCase().replace(/[\s\u00A0]+/g, ' ').trim();
}

/** Escape HTML special chars so raw document text is safe before we inject <mark> */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Try to highlight `fact` inside `excerpt` using three strategies,
 * returning an HTML string with <mark> tags around every match found.
 *
 * Strategy 1 — exact (case-insensitive, whitespace-normalised)
 * Strategy 2 — numeric tokens: extract all digit runs from the fact and
 *               highlight any match of those numbers in the excerpt.
 * Strategy 3 — keyword: split fact into words ≥4 chars, find each in text.
 */
function highlightFact(excerpt: string, fact: string): string {
  const MARK_OPEN  = '<mark class="bg-primary/25 text-primary font-semibold px-0.5 rounded">';
  const MARK_CLOSE = '</mark>';

  // ── Strategy 1: exact match (normalised whitespace, case-insensitive) ──
  const escapedFact = fact.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
                          .replace(/\s+/g, '\\s+');   // flexible whitespace
  const exactRe = new RegExp(`(${escapedFact})`, 'gi');
  if (exactRe.test(excerpt)) {
    return excerpt.replace(exactRe, `${MARK_OPEN}$1${MARK_CLOSE}`);
  }

  // ── Strategy 2: numeric tokens (great for amounts like "2 850 000 €") ──
  // Extract all digit sequences from the fact
  const numbers = fact.match(/\d[\d\s.,]*/g)
    ?.map(n => n.replace(/\s/g, '').replace(',', '.'))   // "2 850 000" → "2850000"
    .filter(n => n.length >= 2) ?? [];

  if (numbers.length > 0) {
    // Build a regex that matches the digits ignoring space-separators in the source
    // e.g. "2850000" matches "2 850 000", "2.850.000", "2850000"
    let modified = excerpt;
    let anyHit = false;
    for (const num of numbers) {
      // Allow optional spaces or dots between every digit group
      const flexNum = num.split('').join('[\\s.,]?').replace(/[.,]/g, '[.,]?');
      const numRe = new RegExp(`(${flexNum}(?:\\s*[€$%]|\\s*(?:EUR|USD))?)`, 'gi');
      if (numRe.test(modified)) {
        modified = modified.replace(numRe, `${MARK_OPEN}$1${MARK_CLOSE}`);
        anyHit = true;
      }
    }
    if (anyHit) return modified;
  }

  // ── Strategy 3: significant keywords (≥4 chars, not stopwords) ──
  const stopwords = new Set(['avec','dans','pour','les','des','une','sur','par','que','qui','est','son','ses','leur','leurs','this','that','with','from','have','been','they','their','the','and','for','are','was','were']);
  const keywords = norm(fact)
    .split(/\s+/)
    .filter(w => w.length >= 4 && !stopwords.has(w))
    .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

  if (keywords.length === 0) return excerpt;

  let modified = excerpt;
  let anyHit = false;
  for (const kw of keywords) {
    const kwRe = new RegExp(`(${kw})`, 'gi');
    if (kwRe.test(modified)) {
      modified = modified.replace(kwRe, `${MARK_OPEN}$1${MARK_CLOSE}`);
      anyHit = true;
    }
  }
  return anyHit ? modified : excerpt;
}

/**
 * Render the excerpt as plain escaped HTML — no highlight marks.
 * Highlighting happens dynamically via handleNumberClick injecting
 * a temporary <mark> into the card's DOM, keeping full alignment.
 */
function buildHighlightedExcerpt(excerpt: string, _facts: string[]): string {
  const clean = excerpt.replace(/<[^>]*>/g, '');
  return escapeHtml(clean);
}

// ---------------------------------------------------------------------------
// Document Viewer
// ---------------------------------------------------------------------------

interface ViewerState {
  doc: Document;
  content: string | null;
  loading: boolean;
  error: string | null;
  highlightText: string | null;
  highlightLines: string[] | null;
  highlightKey: number;
  blobUrl?: string;          // object URL for PDF/image local preview
  mediaType?: "pdf" | "image" | "txt";  // what kind of viewer to show
}

interface DocumentViewerProps {
  state: ViewerState;
  onClose: () => void;
}

function findExcerptRange(content: string, excerpt: string): [number, number] | null {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const clean = (s: string) => s.replace(/\*\*/g, '').replace(/\[\d+\]/g, '').trim();
  const needle = clean(excerpt);
  if (!needle) return null;

  // Strategy 1: exact match with flexible whitespace
  const exactRe = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+'), 'i');
  const exactM = exactRe.exec(content);
  if (exactM) return [exactM.index, exactM.index + exactM[0].length];

  // Strategy 2: sliding window of 5 consecutive words — finds partial matches
  const words = needle.split(/\s+/).filter(Boolean);
  const windowSize = Math.min(5, words.length);
  for (let start = 0; start <= words.length - windowSize; start++) {
    const probe = words.slice(start, start + windowSize);
    const re = new RegExp('(' + probe.map(esc).join('[\\s.,;:-]{0,6}') + ')', 'i');
    const m = re.exec(content);
    if (m) return [m.index, m.index + m[0].length];
  }

  // Strategy 3: two longest significant words together
  const stop = new Set(['the','and','for','are','was','with','this','that','from','have','been','they','will','has']);
  const sig = words.filter(w => w.length >= 4 && !stop.has(w.toLowerCase())).sort((a,b) => b.length - a.length);
  if (sig.length >= 2) {
    const re = new RegExp('(' + esc(sig[0]) + '[\\s\\S]{0,30}' + esc(sig[1]) + '|'  + esc(sig[1]) + '[\\s\\S]{0,30}' + esc(sig[0]) + ')', 'i');
    const m = re.exec(content);
    if (m) return [m.index, m.index + m[0].length];
  }
  if (sig.length >= 1) {
    const fm = new RegExp(esc(sig[0]), 'i').exec(content);
    if (fm) return [fm.index, fm.index + fm[0].length];
  }
  return null;
}

function renderDocumentContent(
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
        style={{ background: 'none', color: 'inherit', display: 'inline', boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone' } as React.CSSProperties}>
        {content.slice(start, end)}
      </mark>
    );
    cursor = end;
  });
  if (cursor < content.length) nodes.push(content.slice(cursor));
  return <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground/85">{nodes}</pre>;
}

// ---------------------------------------------------------------------------
// PDF.js viewer with highlight support
// ---------------------------------------------------------------------------

declare global {
  interface Window {
    pdfjsLib: any;
  }
}

interface PdfViewerProps {
  blobUrl: string;
  highlightText: string | null;
  highlightLines: string[] | null;
  highlightKey: number;
}

function loadPdfJs(): Promise<any> {
  return new Promise((resolve, reject) => {
    if (window.pdfjsLib) { resolve(window.pdfjsLib); return; }
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      resolve(window.pdfjsLib);
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

const HIGHLIGHT_COLOR = "rgba(253, 224, 71, 0.45)"; // yellow-300 at 45%

const PdfViewer: React.FC<PdfViewerProps> = ({ blobUrl, highlightText, highlightLines, highlightKey }) => {
  // outerRef: the scrollable viewport div
  // innerRef: the CSS-transformed content div (never re-rendered on zoom)
  const outerRef   = useRef<HTMLDivElement>(null);
  const innerRef   = useRef<HTMLDivElement>(null);
  const pdfDocRef  = useRef<any>(null);
  const [status, setStatus]   = useState<"loading" | "ready" | "error">("loading");
  const [zoom, setZoom]       = useState(1.0);   // React state only for toolbar display
  const [zoomMode, setZoomMode] = useState(false);
  const zoomRef    = useRef(1.0);  // always-current zoom, no closure staleness

  const highlightsRef = useRef<string[]>([]);
  highlightsRef.current = [...new Set(
    [...(highlightLines ?? []), ...(highlightText ? [highlightText] : [])].filter(Boolean)
  )];

  // ── Text-match helper ──────────────────────────────────────────────────────
  const findMatchingItems = (
    items: Array<{ str: string; transform: number[] }>,
    needle: string
  ): number[] => {
    if (!needle.trim()) return [];
    // Clean markdown artifacts from the needle before searching
    const cleanNeedle = needle.replace(/\*\*/g, '').replace(/\[\d+\]/g, '').trim();
    const needleLo = cleanNeedle.toLowerCase().replace(/\s+/g, " ");
    // Try first 60 chars of needle for tighter matching — long excerpts rarely match verbatim
    const probe = needleLo.slice(0, 60).trim();
    const hit = new Set<number>();
    for (const sep of [" ", ""]) {
      const joined   = items.map(i => i.str).join(sep);
      const joinedLo = joined.toLowerCase();
      // Only use the FIRST occurrence — avoids highlighting every mention on the page
      const idx = joinedLo.indexOf(probe);
      if (idx === -1) continue;
      let charCount = 0;
      for (let i = 0; i < items.length; i++) {
        const start = charCount, end = charCount + items[i].str.length;
        charCount += items[i].str.length + sep.length;
        if (end > idx && start < idx + probe.length) hit.add(i);
      }
      if (hit.size > 0) break;
    }
    return [...hit];
  };

  // ── Draw highlight overlay for one page ───────────────────────────────────
  const viewportsRef = useRef<Map<number, any>>(new Map());

  const drawOverlay = useCallback(async (pageNum: number) => {
    const pdf = pdfDocRef.current;
    const inner = innerRef.current;
    if (!pdf || !inner) return;
    const wrapper = inner.querySelector(`[data-page="${pageNum}"]`) as HTMLDivElement | null;
    if (!wrapper) return;
    const overlay = wrapper.querySelector("canvas.hl-overlay") as HTMLCanvasElement | null;
    if (!overlay) return;
    const ctx = overlay.getContext("2d")!;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const needles = highlightsRef.current;
    if (!needles.length) return;
    const viewport = viewportsRef.current.get(pageNum);
    if (!viewport) return;
    const page = await pdf.getPage(pageNum);
    const tc   = await page.getTextContent();
    const items = tc.items as Array<{ str: string; transform: number[] }>;
    ctx.save();
    ctx.fillStyle = HIGHLIGHT_COLOR;
    for (const needle of needles) {
      for (const idx of findMatchingItems(items, needle)) {
        const item = items[idx] as any;
        const [sx, , , sy, tx, ty] = item.transform;
        // Convert bottom-left corner of item from PDF coords to canvas pixel coords
        const [cx, cy] = viewport.convertToViewportPoint(tx, ty);
        // item.width is the actual text run width in PDF user units (most reliable)
        // fall back to font-size-based estimate if not present
        const itemW = typeof item.width === "number" && item.width > 0
          ? item.width * viewport.scale
          : Math.abs(sx) * viewport.scale;
        const height = Math.abs(sy) * viewport.scale;
        ctx.fillRect(cx, cy - height * 0.9, itemW, height * 1.1);
      }
    }
    ctx.restore();
  }, []);

  // ── Render a single page (canvas at 2× for crisp text) ────────────────────
  const renderPage = useCallback(async (pageNum: number) => {
    const pdf   = pdfDocRef.current;
    const inner = innerRef.current;
    const outer = outerRef.current;
    if (!pdf || !inner || !outer) return;

    try {
      const page     = await pdf.getPage(pageNum);
      const DPR      = Math.min(window.devicePixelRatio || 1, 2); // 2× max
      const availW   = outer.clientWidth - 32;
      const baseVP   = page.getViewport({ scale: 1 });
      const baseScale = availW / baseVP.width;
      // Render at base scale × DPR for crispness; CSS will scale back down via width/height
      const viewport = page.getViewport({ scale: baseScale * DPR });
      viewportsRef.current.set(pageNum, viewport);

      const cssW = Math.floor(baseScale * baseVP.width);
      const cssH = Math.floor(baseScale * baseVP.height);
      const pxW  = Math.floor(viewport.width);
      const pxH  = Math.floor(viewport.height);

      // Build or reuse wrapper
      let wrapper = inner.querySelector(`[data-page="${pageNum}"]`) as HTMLDivElement | null;
      let pgCanvas: HTMLCanvasElement;
      let hlCanvas: HTMLCanvasElement;

      if (!wrapper) {
        wrapper = document.createElement("div");
        wrapper.dataset.page  = String(pageNum);
        wrapper.style.cssText = `position:relative;margin:0 auto 12px;box-shadow:0 1px 8px rgba(0,0,0,.25);background:#fff;flex-shrink:0;`;
        pgCanvas = document.createElement("canvas");
        pgCanvas.className = "pg-canvas";
        pgCanvas.style.cssText = "position:absolute;top:0;left:0;";
        hlCanvas = document.createElement("canvas");
        hlCanvas.className = "hl-overlay";
        hlCanvas.style.cssText = "position:absolute;top:0;left:0;pointer-events:none;";
        wrapper.appendChild(pgCanvas);
        wrapper.appendChild(hlCanvas);
        const siblings     = Array.from(inner.querySelectorAll("[data-page]"));
        const insertBefore = siblings.find(s => Number((s as HTMLElement).dataset.page) > pageNum);
        if (insertBefore) inner.insertBefore(wrapper, insertBefore);
        else inner.appendChild(wrapper);
      } else {
        pgCanvas = wrapper.querySelector("canvas.pg-canvas")!;
        hlCanvas = wrapper.querySelector("canvas.hl-overlay")!;
      }

      // Set CSS size (layout size — zoom via CSS transform on innerRef, not here)
      wrapper.style.width  = cssW + "px";
      wrapper.style.height = cssH + "px";
      // Canvas pixel size = CSS size × DPR for sharpness
      pgCanvas.width  = pxW;  pgCanvas.height  = pxH;
      pgCanvas.style.width  = cssW + "px"; pgCanvas.style.height = cssH + "px";
      hlCanvas.width  = pxW;  hlCanvas.height  = pxH;
      hlCanvas.style.width  = cssW + "px"; hlCanvas.style.height = cssH + "px";

      const ctx = pgCanvas.getContext("2d")!;
      ctx.clearRect(0, 0, pxW, pxH);
      await page.render({ canvasContext: ctx, viewport }).promise;
      await drawOverlay(pageNum);
    } catch { /* skip */ }
  }, [drawOverlay]);

  // ── Load PDF — render all pages once, never again ─────────────────────────
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    zoomRef.current = 1.0;
    setZoom(1.0);
    setZoomMode(false);
    naturalSizeRef.current = null;
    translateRef.current   = { x: 0, y: 0 };
    // Reset inner transform
    if (innerRef.current) {
      innerRef.current.style.transform = "";
      innerRef.current.style.width     = "";
      innerRef.current.style.height    = "";
    }
    viewportsRef.current.clear();

    (async () => {
      try {
        const lib = await loadPdfJs();
        const pdf = await lib.getDocument(blobUrl).promise;
        if (cancelled) return;
        pdfDocRef.current = pdf;
        setStatus("ready");
        // Render pages sequentially
        for (let i = 1; i <= pdf.numPages; i++) {
          if (cancelled) return;
          await renderPage(i);
        }
      } catch { if (!cancelled) setStatus("error"); }
    })();
    return () => { cancelled = true; };
  }, [blobUrl, renderPage]);

  // ── Resize — re-render pages (layout width changed) ───────────────────────
  useEffect(() => {
    const el = outerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if (status !== "ready" || !pdfDocRef.current) return;
      // Reset zoom transform first so available width is accurate
      if (innerRef.current) {
        innerRef.current.style.transform = "";
        innerRef.current.style.width     = "";
        innerRef.current.style.height    = "";
      }
      naturalSizeRef.current = null;
      translateRef.current   = { x: 0, y: 0 };
      zoomRef.current = 1.0;
      setZoom(1.0);
      const pdf = pdfDocRef.current;
      (async () => {
        for (let i = 1; i <= pdf.numPages; i++) await renderPage(i);
      })();
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [status, renderPage]);

  // ── Highlight change ───────────────────────────────────────────────────────
  useEffect(() => {
    if (status !== "ready" || !pdfDocRef.current) return;
    for (let i = 1; i <= pdfDocRef.current.numPages; i++) drawOverlay(i);
  }, [highlightKey, status, drawOverlay]);

  // ── Scroll to highlight ────────────────────────────────────────────────────
  // scrollIntoView is broken here because outerRef uses overflow-hidden + CSS transform.
  // Instead: find the matching page wrapper, compute its offsetTop inside innerRef,
  // then animate translateRef so that page is centered in the viewport.
  useEffect(() => {
    if (status !== "ready" || !pdfDocRef.current) return;
    const needles = highlightsRef.current;
    if (!needles.length || !innerRef.current || !outerRef.current) return;
    setTimeout(async () => {
      const pdf   = pdfDocRef.current;
      const inner = innerRef.current;
      const outer = outerRef.current;
      if (!pdf || !inner || !outer) return;
      const wrappers = Array.from(inner.querySelectorAll("[data-page]")) as HTMLDivElement[];
      for (const wrapper of wrappers) {
        const pageNum = Number(wrapper.dataset.page);
        const page    = await pdf.getPage(pageNum);
        const tc      = await page.getTextContent();
        const spaced  = (tc.items as any[]).map((i: any) => i.str).join(" ").toLowerCase();
        const nospace = (tc.items as any[]).map((i: any) => i.str).join("").toLowerCase();
        const hit = needles.some(n => {
          const nl = n.toLowerCase().replace(/\s+/g, " ").trim();
          return spaced.includes(nl) || nospace.includes(nl.replace(/\s+/g, ""));
        });
        if (!hit) continue;

        // Compute the page wrapper's top offset relative to innerRef
        const z = zoomRef.current;
        const pageOffsetTop = wrapper.offsetTop * z + (translateRef.current.y % 1); // scaled
        // We want the page centered in the outer viewport
        const targetTy = -(wrapper.offsetTop * z) + outer.clientHeight / 2 - (wrapper.offsetHeight * z) / 2;
        const maxTy = 0;
        const minTy = -(inner.scrollHeight * z - outer.clientHeight);
        const clampedTy = Math.min(maxTy, Math.max(minTy, targetTy));

        // Animate smoothly by stepping toward target
        const startTy = translateRef.current.y;
        const startTx = translateRef.current.x;
        const duration = 350;
        const startTime = performance.now();
        const animate = (now: number) => {
          const t = Math.min(1, (now - startTime) / duration);
          const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
          const newTy = startTy + (clampedTy - startTy) * ease;
          translateRef.current = { x: startTx, y: newTy };
          if (inner) inner.style.transform = `translate(${startTx}px, ${newTy}px) scale(${z})`;
          if (t < 1) requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
        break;
      }
    }, 250);
  }, [highlightKey, status]);

  // ── Core zoom function ────────────────────────────────────────────────────
  // Use translate+scale so the focal point stays exactly under the mouse.
  // No scroll math — we track a cumulative translate offset instead.
  const naturalSizeRef = useRef<{ w: number; h: number } | null>(null);
  const translateRef   = useRef({ x: 0, y: 0 });

  const applyZoom = useCallback((newZoom: number, focalX: number, focalY: number) => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    const oldZoom = zoomRef.current;
    const ratio   = newZoom / oldZoom;

    if (!naturalSizeRef.current) {
      naturalSizeRef.current = { w: inner.offsetWidth, h: inner.offsetHeight };
    }
    const { w: natW, h: natH } = naturalSizeRef.current;

    // Current translate
    const tx = translateRef.current.x;
    const ty = translateRef.current.y;

    // Focal point relative to the scaled+translated content origin
    const originX = focalX - tx;
    const originY = focalY - ty;

    // New translate: scale around the focal point
    const newTx = focalX - originX * ratio;
    const newTy = focalY - originY * ratio;

    translateRef.current = { x: newTx, y: newTy };
    zoomRef.current = newZoom;

    inner.style.transformOrigin = "0 0";
    inner.style.transform       = `translate(${newTx}px, ${newTy}px) scale(${newZoom})`;
    inner.style.width           = `${natW}px`;
    inner.style.height          = `${natH}px`;

    setZoom(newZoom);
  }, []);

  // ── All wheel events handled manually — no native overflow scroll ──────────
  // We use overflow-hidden on the outer div so the browser NEVER gets a chance
  // to scroll natively. Every wheel event (plain scroll OR ctrl+zoom) goes
  // through our handler, eliminating the scroll-then-zoom race entirely.
  const applyZoomRef   = useRef(applyZoom);
  useEffect(() => { applyZoomRef.current = applyZoom; }, [applyZoom]);

  // ── Mouse pan state ────────────────────────────────────────────────────
  const isPanningRef   = useRef(false);
  const panStartRef    = useRef({ x: 0, y: 0 });
  const panTranslateRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const el = outerRef.current;
    if (!el || status !== "ready") return;

    // rAF-batched state so multiple wheel events per frame collapse into one DOM write
    let pending: { deltaY: number; deltaX: number; clientX: number; clientY: number; isZoom: boolean } | null = null;
    let rafId = 0;

    const flush = () => {
      if (!pending) return;
      const { deltaY, deltaX, clientX, clientY, isZoom } = pending;
      pending = null;

      if (isZoom) {
        const factor  = deltaY > 0 ? 0.88 : 1.0 / 0.88;
        const newZoom = Math.min(4, Math.max(0.3, zoomRef.current * factor));
        const rect    = el.getBoundingClientRect();
        applyZoomRef.current(newZoom, clientX - rect.left, clientY - rect.top);
      } else {
        // Plain scroll — shift the translate
        const inner = innerRef.current;
        if (!inner) return;
        // Snapshot natural size on demand if reset cleared it
        if (!naturalSizeRef.current) {
          naturalSizeRef.current = { w: inner.offsetWidth, h: inner.offsetHeight };
        }
        const { w: natW, h: natH } = naturalSizeRef.current;
        const z = zoomRef.current;
        const maxX = -(natW * z - el.clientWidth);
        const maxY = -(natH * z - el.clientHeight);
        const newTx = Math.min(0, Math.max(maxX, translateRef.current.x - deltaX));
        const newTy = Math.min(0, Math.max(maxY, translateRef.current.y - deltaY));
        translateRef.current = { x: newTx, y: newTy };
        inner.style.transform = `translate(${newTx}px, ${newTy}px) scale(${z})`;
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const isZoom = e.ctrlKey || e.metaKey;
      if (pending && pending.isZoom !== isZoom) {
        // Mode switched mid-batch — flush immediately before accumulating
        cancelAnimationFrame(rafId);
        flush();
      }
      pending = {
        deltaY:  (pending?.deltaY  ?? 0) + e.deltaY,
        deltaX:  (pending?.deltaX  ?? 0) + e.deltaX,
        clientX: e.clientX,
        clientY: e.clientY,
        isZoom,
      };
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(flush);
    };

    el.addEventListener("wheel", onWheel, { passive: false, capture: true });
    return () => {
      el.removeEventListener("wheel", onWheel, { capture: true } as any);
      cancelAnimationFrame(rafId);
    };
  }, [status]);

  // ── Mouse pan (click + drag) ───────────────────────────────────────────────
  useEffect(() => {
    const el = outerRef.current;
    if (!el || status !== "ready") return;

    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;           // left button only
      if ((e.target as HTMLElement).closest("button")) return; // don't hijack toolbar
      isPanningRef.current = true;
      panStartRef.current  = { x: e.clientX, y: e.clientY };
      panTranslateRef.current = { ...translateRef.current };
      el.style.cursor = "grabbing";
      e.preventDefault();
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isPanningRef.current) return;
      const inner = innerRef.current;
      if (!inner || !naturalSizeRef.current) return;
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      const z = zoomRef.current;
      const { w: natW, h: natH } = naturalSizeRef.current;
      const maxX = -(natW * z - el.clientWidth);
      const maxY = -(natH * z - el.clientHeight);
      const newTx = Math.min(0, Math.max(maxX, panTranslateRef.current.x + dx));
      const newTy = Math.min(0, Math.max(maxY, panTranslateRef.current.y + dy));
      translateRef.current = { x: newTx, y: newTy };
      inner.style.transform = `translate(${newTx}px, ${newTy}px) scale(${z})`;
    };

    const onMouseUp = () => {
      if (!isPanningRef.current) return;
      isPanningRef.current = false;
      el.style.cursor = zoomMode ? (zoomRef.current >= 4 ? "zoom-out" : "zoom-in") : "grab";
    };

    el.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup",   onMouseUp);
    return () => {
      el.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup",   onMouseUp);
    };
  }, [status, zoomMode]);

  // ── Toolbar buttons ────────────────────────────────────────────────────────
  const toolbarZoom = useCallback((newZoom: number) => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    if (newZoom === 1) {
      // Full reset — wipe transform and all offsets back to initial fit
      translateRef.current   = { x: 0, y: 0 };
      naturalSizeRef.current = null;
      zoomRef.current        = 1;
      inner.style.transform  = "";
      inner.style.width      = "";
      inner.style.height     = "";
      setZoom(1);
      return;
    }

    applyZoom(newZoom, outer.clientWidth / 2, outer.clientHeight / 2);
  }, [applyZoom]);

  // ── Click-to-zoom ──────────────────────────────────────────────────────────
  const handleContainerClick = (e: React.MouseEvent) => {
    if (!zoomMode) return;
    const outer = outerRef.current;
    if (!outer) return;
    const rect  = outer.getBoundingClientRect();
    const factor = e.altKey ? 0.8 : 1.25;
    const newZoom = Math.min(4, Math.max(0.3, zoomRef.current * factor));
    applyZoom(newZoom, e.clientX - rect.left, e.clientY - rect.top);
  };

  return (
    <div className="flex flex-col h-full" style={{ minHeight: 0 }}>
      {status === "loading" && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 text-muted-foreground">
          <Loader2 className="h-7 w-7 animate-spin text-primary" />
          <p className="text-sm">Rendering PDF…</p>
        </div>
      )}
      {status === "error" && (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-6">
          <AlertCircle className="h-8 w-8 text-destructive/70" />
          <p className="text-sm text-muted-foreground">Could not render PDF.</p>
        </div>
      )}
      {status === "ready" && (
        <>
          {/* Zoom toolbar */}
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-border bg-background/80 shrink-0 gap-2">
            <span className="text-[11px] text-muted-foreground/60 hidden sm:block">Ctrl+scroll to zoom</span>
            <div className="flex items-center gap-1 ml-auto">
              <button
                onClick={() => setZoomMode(m => !m)}
                title={zoomMode ? "Exit zoom mode" : "Click-to-zoom mode"}
                className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${zoomMode ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                  <circle cx="6" cy="6" r="4.5"/>
                  <line x1="9.5" y1="9.5" x2="13" y2="13"/>
                  {zoomMode ? <><line x1="4" y1="6" x2="8" y2="6"/><line x1="6" y1="4" x2="6" y2="8"/></> : <line x1="4" y1="6" x2="8" y2="6"/>}
                </svg>
              </button>
              <button onClick={() => toolbarZoom(Math.max(0.3, zoomRef.current / 1.25))} disabled={zoom <= 0.3} className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-30 transition-colors font-bold text-sm select-none">−</button>
              <button onClick={() => toolbarZoom(1)} className="px-1.5 h-6 text-[11px] text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors tabular-nums min-w-[2.8rem] text-center">{Math.round(zoom * 100)}%</button>
              <button onClick={() => toolbarZoom(Math.min(4, zoomRef.current * 1.25))} disabled={zoom >= 4} className="h-6 w-6 flex items-center justify-center rounded text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-30 transition-colors font-bold text-sm select-none">+</button>
            </div>
          </div>
          {/* Scrollable outer viewport */}
          <div
            ref={outerRef}
            onClick={handleContainerClick}
            className="flex-1 overflow-hidden scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40"
            style={{
              background: "hsl(var(--muted)/0.3)",
              cursor: zoomMode ? (zoom >= 4 ? "zoom-out" : "zoom-in") : "grab",
              minHeight: 0,
            }}
          >
            {/* Inner div — CSS scaled, never wiped during zoom */}
            <div
              ref={innerRef}
              style={{ transformOrigin: "0 0", display: "flex", flexDirection: "column", alignItems: "center", padding: "16px" }}
            />
          </div>
        </>
      )}
    </div>
  );
};


const DocumentViewer: React.FC<DocumentViewerProps> = ({ state, onClose }) => {
  const highlightRef = useRef<HTMLElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Scroll to highlighted passage whenever it changes
  // rAF ensures the ref is painted before we read its position
  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (!highlightRef.current || !scrollContainerRef.current) return;
      const container = scrollContainerRef.current;
      const el = highlightRef.current;
      const elTop = el.getBoundingClientRect().top - container.getBoundingClientRect().top;
      const targetScroll = container.scrollTop + elTop - container.clientHeight / 2 + el.offsetHeight / 2;
      container.scrollTo({ top: targetScroll, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(raf);
  }, [state.highlightKey, state.content]);

  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 480, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
      className="border-l border-border bg-background flex flex-col overflow-hidden shrink-0 h-screen"
    >
      {/* Header */}
      <div className="h-14 border-b border-border flex items-center gap-3 px-4 shrink-0">
        {state.mediaType === 'image' ? <ImageIcon className="h-4 w-4 text-primary shrink-0" /> : <ScrollText className="h-4 w-4 text-primary shrink-0" />}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-foreground truncate">{state.doc.filename}</p>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{state.doc.doc_type}</p>
        </div>
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground shrink-0" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Highlight badge */}
      {state.highlightText && state.mediaType !== 'image' && (
        <div className="px-4 py-2 bg-primary/8 border-b border-primary/20 flex items-start gap-2">
          <span className="text-[10px] font-semibold text-primary uppercase tracking-wide shrink-0 mt-0.5">Cited passage</span>
          <p className="text-[11px] text-primary/80 leading-relaxed line-clamp-2 flex-1">{state.highlightText}</p>
        </div>
      )}

      {/* Content */}
      <div ref={scrollContainerRef} className={`flex-1 overflow-hidden scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40 ${state.mediaType === 'pdf' ? 'p-0' : state.mediaType === 'image' ? 'p-2 overflow-y-auto' : 'p-4 overflow-y-auto'}`}>
        {state.loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm">Loading document…</p>
          </div>
        )}

        {state.error && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
            <AlertCircle className="h-8 w-8 text-destructive/70" />
            <p className="text-sm text-muted-foreground">{state.error}</p>
          </div>
        )}

        {!state.loading && !state.error && (
          state.mediaType === 'image' ? (
            <div className="flex flex-col items-center justify-center h-full p-4 gap-4">
              <img
                src={state.blobUrl}
                alt={state.doc.filename}
                className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                style={{ maxHeight: 'calc(100vh - 180px)' }}
              />
              <p className="text-xs text-muted-foreground">{state.doc.filename}</p>
            </div>
          ) : state.mediaType === 'pdf' ? (
            state.blobUrl ? (
              <PdfViewer
                blobUrl={state.blobUrl}
                highlightText={state.highlightText}
                highlightLines={state.highlightLines}
                highlightKey={state.highlightKey}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
                <FileText className="h-10 w-10 text-muted-foreground/40" />
                <p className="text-sm text-muted-foreground">PDF preview available after upload.</p>
                <p className="text-xs text-muted-foreground/60">The file could not be fetched from the server.</p>
              </div>
            )
          ) : state.content !== null ? (
            renderDocumentContent(state.content, state.highlightText, highlightRef, state.highlightKey, state.highlightLines)
          ) : null
        )}
      </div>
    </motion.aside>
  );
};

// ---------------------------------------------------------------------------
// Robust query helper — bypasses api.query() to guarantee correct payload
// ---------------------------------------------------------------------------

/** Serialize any caught value to a readable string (handles Error, object, string, etc.) */
function serializeError(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  if (err && typeof err === "object") {
    // FastAPI 422 detail is often { detail: [...] }
    const e = err as Record<string, unknown>;
    if (e.detail) {
      if (typeof e.detail === "string") return e.detail;
      if (Array.isArray(e.detail)) {
        return e.detail
          .map((d: unknown) =>
            d && typeof d === "object"
              ? `${(d as Record<string, unknown>).loc ?? ""}: ${(d as Record<string, unknown>).msg ?? JSON.stringify(d)}`
              : String(d)
          )
          .join("; ");
      }
      return JSON.stringify(e.detail);
    }
    if (e.message && typeof e.message === "string") return e.message;
    return JSON.stringify(err);
  }
  return String(err);
}

/**
 * Direct POST to /query with a correctly-formed JSON body.
 * Falls back gracefully if api.query() has a broken serialisation or
 * wrong field name — this always sends { "query": "...", "top_k": 5 }.
 * Passes conversation_history so the backend can use prior turns as context.
 */
async function queryBackend(
  question: string,
  sessionId: string | null,
  signal?: AbortSignal,
): Promise<{ answer: string; raw_answer?: string; sources: Source[]; confidence: number; session_id: string }> {
  const base =
    (typeof import.meta !== "undefined" && (import.meta as Record<string, unknown>).env
      ? ((import.meta as Record<string, { VITE_API_URL?: string }>).env.VITE_API_URL ?? "")
      : "") || "http://localhost:8000";

  const res = await fetch(`${base}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: question,
      top_k: 5,
      ...(sessionId ? { session_id: sessionId } : {}),
    }),
    signal,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = serializeError(body);
    } catch { /* ignore */ }
    throw new Error(detail);
  }

  return res.json();
}

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm your AI document assistant. Upload your documents and ask me anything — I'll find the relevant information and cite my sources.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string>("default");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [historySearch, setHistorySearch] = useState("");
  const [historySort, setHistorySort] = useState<"newest" | "oldest">("newest");
  const [docsPanelOpen, setDocsPanelOpen] = useState(false);
  const docsPanelOpenRef = useRef(false);
  docsPanelOpenRef.current = docsPanelOpen;
  const [convsPanelOpen, setConvsPanelOpen] = useState(false);
  const convsPanelOpenRef = useRef(false);
  convsPanelOpenRef.current = convsPanelOpen;
  const [bubbleDoc, setBubbleDoc] = useState<Document | null>(null);
  const [bubbleViewer, setBubbleViewer] = useState<ViewerState | null>(null);
  const bubbleViewerRef = useRef<ViewerState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike" | null>>({});
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragDepth, setDragDepth] = useState(0);
  const [isDraggingInput, setIsDraggingInput] = useState(false);
  const [inputDragDepth, setInputDragDepth] = useState(0);
  const [openSourceKey, setOpenSourceKey] = useState<string | null>(null);
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  // Map document_id → local object URL (for PDF/image preview without re-downloading)
  const blobUrlMapRef = useRef<Map<string, { url: string; type: "pdf" | "image" | "txt" }>>(new Map());
  const bubbleHighlightRef = useRef<HTMLElement>(null);
  const bubbleScrollRef    = useRef<HTMLDivElement>(null);
  bubbleViewerRef.current  = bubbleViewer; // always-fresh mirror, no stale closure
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isHoveringVoice, setIsHoveringVoice] = useState(false);
  const { toast } = useToast();

  // ── Document viewer helpers ──────────────────────────────────────────────

  const getApiBase = () =>
    ((typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000");

  // Fetch a raw file from the backend and cache it as a blob URL
  const fetchBlobUrl = async (doc: Document): Promise<{ url: string; type: "pdf" | "image" | "txt" }> => {
    const cached = blobUrlMapRef.current.get(doc.document_id);
    if (cached) return cached;

    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
    const type: "pdf" | "image" | "txt" = IMAGE_EXTS.includes(`.${ext}`) ? "image" : ext === "pdf" ? "pdf" : "txt";

    const base = getApiBase();
    const res = await fetch(`${base}/documents/${doc.document_id}/download`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    blobUrlMapRef.current.set(doc.document_id, { url, type });
    return { url, type };
  };

  const openBubbleDoc = async (doc: Document, highlightText: string | null = null, highlightLines: string[] | null = null) => {
    // Read from ref — never stale, even in async callbacks
    const current = bubbleViewerRef.current;

    // Always ensure the panel is open (works whether panel was open or closed)
    setDocsPanelOpen(true);

    // Same doc already open and loaded — just move the highlight, no reload
    if (current && current.doc.document_id === doc.document_id && !current.loading && !current.error) {
      setBubbleViewer(prev => prev ? { ...prev, highlightText, highlightLines, highlightKey: (prev.highlightKey ?? 0) + 1 } : prev);
      return;
    }

    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
    const isPdfOrImage = ext === 'pdf' || IMAGE_EXTS.includes(`.${ext}`);
    setBubbleDoc(doc);

    if (isPdfOrImage) {
      const cached = blobUrlMapRef.current.get(doc.document_id);
      if (cached) {
        setBubbleViewer({ doc, content: null, loading: false, error: null, highlightText, highlightLines, highlightKey: 0, blobUrl: cached.url, mediaType: cached.type });
      } else {
        setBubbleViewer({ doc, content: null, loading: true, error: null, highlightText, highlightLines, highlightKey: 0, mediaType: ext === 'pdf' ? 'pdf' : 'image' });
        try {
          const { url, type } = await fetchBlobUrl(doc);
          setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, loading: false, blobUrl: url, mediaType: type } : p);
        } catch {
          setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, loading: false, error: "Could not load file." } : p);
        }
      }
    } else {
      setBubbleViewer({ doc, content: null, loading: true, error: null, highlightText, highlightLines, highlightKey: 0, mediaType: 'txt' });
      try {
        const res = await (api as any).getDocumentContent(doc.document_id);
        setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, content: res.content, loading: false } : p);
      } catch {
        setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, loading: false, error: "Could not load document." } : p);
      }
    }
  };

  const openDocumentViewer = async (doc: Document, highlightText: string | null = null, highlightLines: string[] | null = null) => {
    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
    const isPdfOrImage = ext === 'pdf' || IMAGE_EXTS.includes(`.${ext}`);

    if (isPdfOrImage) {
      // Show loading state immediately, then fetch blob in background
      const cached = blobUrlMapRef.current.get(doc.document_id);
      if (cached) {
        setViewer({ doc, content: null, loading: false, error: null, highlightText, highlightLines, highlightKey: 0, blobUrl: cached.url, mediaType: cached.type });
      } else {
        setViewer({ doc, content: null, loading: true, error: null, highlightText, highlightLines, highlightKey: 0, mediaType: ext === 'pdf' ? 'pdf' : 'image' });
        try {
          const { url, type } = await fetchBlobUrl(doc);
          setViewer(p => p && p.doc.document_id === doc.document_id
            ? { ...p, loading: false, blobUrl: url, mediaType: type }
            : p
          );
        } catch (e) {
          setViewer(p => p && p.doc.document_id === doc.document_id
            ? { ...p, loading: false, error: "Could not load file. Make sure the backend exposes GET /documents/{id}/download." }
            : p
          );
        }
      }
      return;
    }

    // Text document — fetch text content as before
    setViewer(prev => {
      const alreadyLoaded = prev && prev.doc.document_id === doc.document_id && prev.content !== null;
      if (alreadyLoaded) {
        return { ...prev!, highlightText, highlightLines, highlightKey: (prev!.highlightKey ?? 0) + 1 };
      }
      return { doc, content: null, loading: true, error: null, highlightText, highlightLines, highlightKey: 0, mediaType: 'txt' };
    });
    setViewer(prev => {
      if (prev && prev.doc.document_id === doc.document_id && prev.content !== null) return prev;
      (async () => {
        try {
          const res = await (api as any).getDocumentContent(doc.document_id);
          const fullContent = res.content as string;
          setViewer(p => {
            if (!p || p.doc.document_id !== doc.document_id) return p;
            return { ...p, content: fullContent, loading: false, error: null };
          });
        } catch {
          setViewer(p => p && p.doc.document_id === doc.document_id
            ? { ...p, loading: false, error: "Could not load document content. Make sure the backend exposes GET /documents/{id}/content." }
            : p
          );
        }
      })();
      return prev;
    });
  };

  // Find which line in fullText contains numValue, return that line
  const findLineForNumber = (fullText: string, numValue: string): string | null => {
    const norm = (s: string) => s.replace(/[\s,]/g, '').toLowerCase();
    const needle = norm(numValue);
    const lines = fullText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    return lines.find(l => norm(l).includes(needle)) ?? null;
  };

  const openDocumentAtExcerpt = (filename: string, excerpt: string) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    openBubbleDoc(doc, excerpt, null);
  };

  const openDocumentAtExcerpts = (filename: string, excerpts: string[]) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    openBubbleDoc(doc, excerpts[0] ?? null, excerpts);
  };

  // Open viewer and highlight the line containing a specific number value
  const openDocumentAtNumber = async (filename: string, numValue: string, fallbackExcerpt: string) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    // If same doc already loaded in bubble, find the right line and update highlight only
    const current = bubbleViewerRef.current;
    if (current && current.doc.document_id === doc.document_id && current.content) {
      const line = findLineForNumber(current.content, numValue) ?? fallbackExcerpt;
      setBubbleViewer(prev => prev ? { ...prev, highlightText: line, highlightLines: null, highlightKey: (prev.highlightKey ?? 0) + 1 } : prev);
      return;
    }
    // Otherwise load fresh into bubble — different doc
    setBubbleDoc(doc);
    setDocsPanelOpen(true);
    setBubbleViewer({ doc, content: null, loading: true, error: null, highlightText: fallbackExcerpt, highlightLines: null, highlightKey: 0, mediaType: 'txt' });
    try {
      const res = await (api as any).getDocumentContent(doc.document_id);
      const fullContent = res.content as string;
      const line = findLineForNumber(fullContent, numValue) ?? fallbackExcerpt;
      setBubbleViewer(p => p && p.doc.document_id === doc.document_id
        ? { ...p, content: fullContent, loading: false, error: null, highlightText: line, highlightLines: null, highlightKey: (p.highlightKey ?? 0) + 1 }
        : p
      );
    } catch {
      setBubbleViewer(p => p && p.doc.document_id === doc.document_id
        ? { ...p, loading: false, error: "Could not load document content." }
        : p
      );
    }
  };

  // Auto-resize edit textarea
  useEffect(() => {
    const ta = editTextareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 300)}px`;
  }, [editDraft]);

  // Focus edit textarea when entering edit mode
  useEffect(() => {
    if (editingMsgId && editTextareaRef.current) {
      const ta = editTextareaRef.current;
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
    }
  }, [editingMsgId]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Close docs panel when clicking outside — registered once, reads ref so never stale
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!docsPanelOpenRef.current) return;
      const target = e.target as HTMLElement;
      if (!target.closest("[data-docs-panel]") && !target.closest("[data-open-doc]")) setDocsPanelOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []); // empty deps — ref keeps it fresh

  // Scroll bubble text viewer to highlighted mark when highlight changes
  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (!bubbleHighlightRef.current || !bubbleScrollRef.current) return;
      const el = bubbleHighlightRef.current;
      const container = bubbleScrollRef.current;
      const elTop = el.getBoundingClientRect().top - container.getBoundingClientRect().top;
      container.scrollTo({ top: container.scrollTop + elTop - container.clientHeight / 2 + el.offsetHeight / 2, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(raf);
  }, [bubbleViewer?.highlightKey]);

  // Close convs panel when clicking outside — registered once, reads ref so never stale
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!convsPanelOpenRef.current) return;
      const target = e.target as HTMLElement;
      if (!target.closest("[data-convs-panel]")) setConvsPanelOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  // ==========================================================================
  // DOCUMENT MANAGEMENT
  // ==========================================================================

  const loadDocuments = async () => {
    try {
      const response = await api.listDocuments();
      setDocuments(response.documents);
    } catch (error) {
      console.error("Failed to load documents:", error);
      toast({
        title: "Error loading documents",
        description: serializeError(error),
        variant: "destructive",
      });
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragDepth(prev => prev + 1);
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragDepth(prev => {
      const newDepth = prev - 1;
      if (newDepth === 0) {
        setIsDragging(false);
      }
      return newDepth;
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];
  const ALLOWED_EXTS = ['.pdf', '.docx', '.doc', '.txt', ...IMAGE_EXTS];

  const _processUploadFile = async (file: File) => {
    const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const isImage = IMAGE_EXTS.includes(fileExt);
    const isPdf   = fileExt === '.pdf';

    if (!ALLOWED_EXTS.includes(fileExt)) {
      toast({ title: "Invalid file type", description: `${file.name}: supported formats are PDF, DOCX, TXT, PNG, JPG, WEBP, GIF`, variant: "destructive" });
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast({ title: "File too large", description: `${file.name}: maximum file size is 50MB`, variant: "destructive" });
      return;
    }

    setIsUploading(true);
    try {
      const response = await api.uploadDocument(file);
      const docId = response.document_id ?? response.id ?? response.filename;

      // Store a local object URL so the viewer can open it instantly without re-downloading
      const blobUrl = URL.createObjectURL(file);
      blobUrlMapRef.current.set(docId, {
        url: blobUrl,
        type: isImage ? "image" : isPdf ? "pdf" : "txt",
      });

      toast({ title: "Uploaded", description: response.filename });
      await loadDocuments();
    } catch (error) {
      toast({ title: "Upload failed", description: `${file.name}: ${serializeError(error)}`, variant: "destructive" });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setDragDepth(0);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    for (const file of files) {
      await _processUploadFile(file);
    }
  };

  const handleInputDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInputDragDepth(prev => prev + 1);
    setIsDraggingInput(true);
  };

  const handleInputDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInputDragDepth(prev => {
      const next = prev - 1;
      if (next === 0) setIsDraggingInput(false);
      return next;
    });
  };

  const handleInputDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleInputDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingInput(false);
    setInputDragDepth(0);
    await handleDrop(e);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      await _processUploadFile(file);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeDocument = async (documentId: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      await api.deleteDocument(documentId);
      
      toast({
        title: "Document deleted",
        description: "Document removed successfully",
      });

      await loadDocuments();
    } catch (error) {
      toast({
        title: "Delete failed",
        description: serializeError(error),
        variant: "destructive",
      });
    }
  };

  // ==========================================================================
  // MESSAGE HANDLING
  // ==========================================================================

  const startNewConversation = () => {
    const newId = Date.now().toString();
    setActiveConvId(newId);
    setSessionId(null);
    setMessages([{
      id: "welcome-" + newId,
      role: "assistant",
      content: "Hello! I'm Bimlo Copilot. How can I help you with your documents today?",
      timestamp: new Date(),
    }]);
  };

  const loadConversation = (conv: Conversation) => {
    setActiveConvId(conv.id);
    setMessages(conv.messages);
  };

  const deleteConversation = (convId: string) => {
    setConversations(prev => prev.filter(c => c.id !== convId));
    if (convId === activeConvId) startNewConversation();
  };

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    // Collapse all sources when sending new message
    setOpenSourceKey(null);

    // Create user message
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: trimmedInput,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      abortControllerRef.current = new AbortController();
      const response = await queryBackend(trimmedInput, sessionId, abortControllerRef.current.signal);
      if (response.session_id) setSessionId(response.session_id);

      // Create assistant message with sources
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        rawAnswer: response.raw_answer ?? response.answer,
        sources: response.sources,
        confidence: response.confidence,
        timestamp: new Date(),
      };

      setMessages((prev) => {
        const updated = [...prev, assistantMsg];
        // Save/update conversation in history
        const title = trimmedInput.length > 50 ? trimmedInput.slice(0, 50) + "…" : trimmedInput;
        const preview = response.answer.replace(/\[.*?\]/g, "").replace(/#{1,3}\s/g, "").slice(0, 80) + "…";
        setConversations(convs => {
          const existing = convs.find(c => c.id === activeConvId);
          if (existing) {
            return convs.map(c => c.id === activeConvId ? { ...c, messages: updated, preview, timestamp: new Date() } : c);
          }
          return [{ id: activeConvId, title, preview, timestamp: new Date(), messages: updated, sessionId } as any, ...convs];
        });
        return updated;
      });
      setTypingMessageId(assistantMsg.id);

    } catch (error) {
      // Ignore abort errors (user clicked Stop)
      if (error instanceof Error && error.name === "AbortError") {
        setIsLoading(false);
        return;
      }
      const msg = serializeError(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${msg}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      toast({ title: "Query failed", description: msg, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setTypingMessageId(null);
  };

  const handleEditCancel = () => {
    setEditingMsgId(null);
    setEditDraft("");
  };

  const handleEditSubmit = async (msgId: string) => {
    const newContent = editDraft.trim();
    if (!newContent) return;

    // Stop any in-flight request immediately
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setTypingMessageId(null);
    setEditingMsgId(null);
    setEditDraft("");

    // Find the user message index and drop everything from it onward
    const msgIndex = messages.findIndex(m => m.id === msgId);
    if (msgIndex === -1) return;

    const updatedUserMsg: Message = {
      ...messages[msgIndex],
      content: newContent,
      timestamp: new Date(),
    };

    // Keep all messages before the edited one, then the updated user message
    setMessages(prev => [...prev.slice(0, msgIndex), updatedUserMsg]);
    setIsLoading(true);

    try {
      abortControllerRef.current = new AbortController();
      const response = await queryBackend(newContent, sessionId, abortControllerRef.current.signal);
      if (response.session_id) setSessionId(response.session_id);

      const assistantMsg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.answer,
        rawAnswer: response.raw_answer ?? response.answer,
        sources: response.sources,
        confidence: response.confidence,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMsg]);
      setTypingMessageId(assistantMsg.id);
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        setIsLoading(false);
        return;
      }
      const msg = serializeError(error);
      const errorMsg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${msg}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
      toast({ title: "Query failed", description: msg, variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRedo = async (msgId: string) => {
    const msgIndex = messages.findIndex(m => m.id === msgId);
    if (msgIndex < 1) return;
    const prevUserMsg = [...messages].slice(0, msgIndex).reverse().find(m => m.role === "user");
    if (!prevUserMsg) return;
    setMessages(prev => prev.filter(m => m.id !== msgId));
    setIsLoading(true);
    try {
      abortControllerRef.current = new AbortController();
      const response = await queryBackend(prevUserMsg.content, sessionId, abortControllerRef.current.signal);
      if (response.session_id) setSessionId(response.session_id);
      const redoMsg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.answer,
        rawAnswer: response.raw_answer ?? response.answer,
        sources: response.sources,
        confidence: response.confidence,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, redoMsg]);
      setTypingMessageId(redoMsg.id);
    } catch (error) {
      // Silently ignore abort; surface real errors
      if (!(error instanceof Error && error.name === "AbortError")) {
        const msg = serializeError(error);
        toast({ title: "Regeneration failed", description: msg, variant: "destructive" });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSourceClick = (sourceNum: number, msgId: string) => {
    // Scroll to the source card
    setTimeout(() => {
      const sourceCard = document.getElementById(`source-${msgId}-${sourceNum}`);
      sourceCard?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      if (sourceCard) {
        sourceCard.style.transition = "box-shadow 0.25s ease";
        sourceCard.style.boxShadow = "0 0 0 2px hsl(var(--primary) / 0.7)";
        setTimeout(() => { sourceCard.style.boxShadow = ""; }, 1800);
      }
    }, 150);

    // Open document viewer at the full excerpt
    const msg = messages.find(m => m.id === msgId);
    const source = msg?.sources?.find(s => s.source_number === sourceNum);
    if (source) {
      openDocumentAtExcerpt(source.filename, source.excerpt ?? "");
    }
  };

  /**
   * Called when a number chip is clicked.
   * Opens the source panel, scrolls to the source card, then finds the specific
   * number inside the excerpt and pulses only that <mark> element.
   */
  const handleNumberClick = (sourceNum: number, msgId: string, numValue: string) => {
    // 1. Expand the sources panel and scroll to the right source card
    setOpenSourceKey(`${msgId}-${sourceNum}`);

    const msg = messages.find(m => m.id === msgId);
    const source = msg?.sources?.find(s => s.source_number === sourceNum);

    // 2. Find the exact line containing this number.
    //    Priority: search the full document content already loaded in the viewer
    //    (most reliable). Fall back to searching the excerpt if doc not loaded yet.
    let highlightSentence: string | null = null;

    const findLineWithNumber = (text: string, numVal: string): string | null => {
      const norm = (s: string) => s.replace(/[\s,]/g, '').toLowerCase();
      const needle = norm(numVal);
      // Try each line
      const lines = text.split(/\n/).map(l => l.trim()).filter(l => l.length > 0);
      return lines.find(l => norm(l).includes(needle)) ?? null;
    };

    // First try: full document content already in bubble viewer state
    if (bubbleViewerRef.current && bubbleViewerRef.current.doc.filename === source?.filename && bubbleViewerRef.current.content) {
      highlightSentence = findLineWithNumber(bubbleViewerRef.current.content, numValue);
    }

    // Second try: excerpt text
    if (!highlightSentence && source?.excerpt) {
      highlightSentence = findLineWithNumber(source.excerpt, numValue);
    }

    // Last resort: use the full excerpt so the viewer at least opens at the right spot
    if (!highlightSentence) {
      highlightSentence = source?.excerpt ?? null;
    }

    setTimeout(() => {
      const sourceCard = document.getElementById(`source-${msgId}-${sourceNum}`);
      if (!sourceCard) return;
      sourceCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

      // 3. Pulse the card border
      sourceCard.style.transition = "box-shadow 0.2s ease";
      sourceCard.style.boxShadow = "0 0 0 2px hsl(var(--primary) / 0.6)";
      setTimeout(() => { sourceCard.style.boxShadow = ""; }, 1800);
    }, 150);

    // 5. Open document viewer — search full doc content for the exact line with this number
    if (source) {
      openDocumentAtNumber(source.filename, numValue, highlightSentence ?? source.excerpt ?? "");
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div 
      className="h-screen flex bg-background relative overflow-hidden"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center pointer-events-none"
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="border-2 border-dashed border-primary rounded-2xl p-12 bg-card/50"
            >
              <div className="text-center">
                <FileText className="h-16 w-16 text-primary mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-foreground mb-2">Drop files here</h3>
                <p className="text-sm text-muted-foreground">
                  Upload PDF, DOCX, TXT, or images (PNG, JPG)
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-14 border-b border-border flex items-center px-4 gap-2 shrink-0">
          <Link to="/">
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <Logo className="h-7 w-7" />
            <span className="font-heading font-semibold text-sm text-foreground">Bimlo Copilot</span>
          </div>
          <div className="ml-auto flex items-center gap-2">

            {/* ── Conversations bubble ── */}
            <div className="relative" data-convs-panel>
              <button
                onClick={() => setConvsPanelOpen(p => !p)}
                className={`relative flex items-center gap-2 pl-3 pr-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
                  convsPanelOpen
                    ? "bg-primary text-primary-foreground border-primary shadow-sm"
                    : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
                }`}
              >
                <MessageSquare className="h-3.5 w-3.5 shrink-0" />
                <span>Conversations</span>
                {conversations.length > 0 && (
                  <span className={`inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-bold ${
                    convsPanelOpen ? "bg-primary-foreground/20 text-primary-foreground" : "bg-primary/15 text-primary"
                  }`}>
                    {conversations.length}
                  </span>
                )}
              </button>

              <AnimatePresence>
                {convsPanelOpen && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: -6 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.9, y: -6 }}
                    transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
                    className="absolute right-0 top-full mt-2 w-72 bg-background border border-border rounded-2xl shadow-2xl overflow-hidden z-50 flex flex-col"
                    style={{ transformOrigin: "top right", maxHeight: 480 }}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
                      <div className="flex items-center gap-2">
                        <MessageSquare className="h-4 w-4 text-primary" />
                        <span className="text-sm font-semibold text-foreground">Conversations</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => { startNewConversation(); setConvsPanelOpen(false); }}
                          className="flex items-center gap-1 text-[11px] text-primary hover:text-primary/80 px-2 py-1 rounded-lg hover:bg-primary/10 transition-colors font-medium"
                        >
                          <Plus className="h-3 w-3" /> New
                        </button>
                        <button onClick={() => setConvsPanelOpen(false)} className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors">
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* Search + sort */}
                    <div className="px-3 pt-2.5 pb-2 space-y-2 shrink-0">
                      <div className="relative">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/50 pointer-events-none" />
                        <input
                          value={historySearch}
                          onChange={e => setHistorySearch(e.target.value)}
                          placeholder="Search conversations…"
                          className="w-full pl-7 pr-3 py-1.5 text-xs bg-muted/50 rounded-lg border border-transparent focus:border-primary/30 focus:bg-background focus:outline-none text-foreground placeholder:text-muted-foreground/50 transition-all"
                        />
                      </div>
                      <button
                        onClick={() => setHistorySort(s => s === "newest" ? "oldest" : "newest")}
                        className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground px-1 py-0.5 rounded transition-colors"
                      >
                        <Clock className="h-3 w-3" />
                        {historySort === "newest" ? "Newest first" : "Oldest first"}
                        <SortAsc className={`h-3 w-3 transition-transform duration-200 ${historySort === "oldest" ? "rotate-180" : ""}`} />
                      </button>
                    </div>

                    {/* List */}
                    <div className="flex-1 overflow-y-auto px-2 pb-2 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40">
                      {conversations.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-10 gap-2 text-center px-4">
                          <div className="h-9 w-9 rounded-xl bg-muted/60 flex items-center justify-center">
                            <MessageSquare className="h-4 w-4 text-muted-foreground/40" />
                          </div>
                          <p className="text-xs text-muted-foreground font-medium">No conversations yet</p>
                          <p className="text-[11px] text-muted-foreground/60">Start chatting to build history</p>
                        </div>
                      ) : (
                        <div className="space-y-0.5">
                          {[...conversations]
                            .filter(c => historySearch === "" || c.title.toLowerCase().includes(historySearch.toLowerCase()) || c.preview.toLowerCase().includes(historySearch.toLowerCase()))
                            .sort((a, b) => historySort === "newest"
                              ? b.timestamp.getTime() - a.timestamp.getTime()
                              : a.timestamp.getTime() - b.timestamp.getTime()
                            )
                            .map(conv => (
                              <div
                                key={conv.id}
                                className={`group relative flex flex-col gap-0.5 px-3 py-2.5 rounded-xl cursor-pointer transition-all ${
                                  conv.id === activeConvId
                                    ? "bg-primary/10"
                                    : "hover:bg-muted/60"
                                }`}
                                onClick={() => { loadConversation(conv); setConvsPanelOpen(false); }}
                              >
                                <div className="flex items-start justify-between gap-2">
                                  <p className="text-[12px] font-semibold text-foreground leading-snug truncate flex-1">{conv.title}</p>
                                  <div className="shrink-0 flex items-center mt-0.5">
                                    <span className="text-[10px] text-muted-foreground/60 tabular-nums group-hover:hidden">
                                      {conv.timestamp.toLocaleDateString([], { month: "short", day: "numeric" })}
                                    </span>
                                    <button
                                      onClick={e => { e.stopPropagation(); deleteConversation(conv.id); }}
                                      className="hidden group-hover:flex items-center justify-center p-1 rounded-md text-muted-foreground/40 hover:text-destructive hover:bg-destructive/10 transition-all"
                                    >
                                      <Trash2 className="h-3 w-3" />
                                    </button>
                                  </div>
                                </div>
                                <p className="text-[11px] text-muted-foreground leading-snug line-clamp-2">{conv.preview}</p>
                              </div>
                            ))
                          }
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* ── Documents bubble ── */}
            <div className="relative" data-docs-panel>
              <button
                onClick={() => { const next = !docsPanelOpenRef.current; docsPanelOpenRef.current = next; setDocsPanelOpen(next); }}
                className={`relative flex items-center gap-2 pl-3 pr-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
                  docsPanelOpen
                    ? "bg-primary text-primary-foreground border-primary shadow-sm"
                    : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
                }`}
                title="Documents"
              >
                <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                <span>Documents</span>
                {documents.length > 0 && (
                  <span className={`inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-bold ${
                    docsPanelOpen ? "bg-primary-foreground/20 text-primary-foreground" : "bg-primary/15 text-primary"
                  }`}>
                    {documents.length}
                  </span>
                )}
              </button>

              {/* Always-mounted — CSS transitions only, React state inside never destroyed */}
              <div
                className="absolute right-0 top-full mt-2 bg-background border border-border rounded-2xl shadow-2xl z-50 flex flex-col"
                style={{
                  transformOrigin: "top right",
                  width: bubbleDoc ? 520 : 288,
                  height: bubbleDoc ? 640 : "auto",
                  maxHeight: bubbleDoc ? 640 : 400,
                  overflow: bubbleDoc ? "hidden" : "visible",
                  transition: "width 0.22s cubic-bezier(0.4,0,0.2,1), height 0.22s cubic-bezier(0.4,0,0.2,1), opacity 0.18s, transform 0.18s",
                  opacity: docsPanelOpen ? 1 : 0,
                  transform: docsPanelOpen ? "scale(1) translateY(0)" : "scale(0.92) translateY(-6px)",
                  pointerEvents: docsPanelOpen ? "auto" : "none",
                  minHeight: 0,
                }}
              >
                    {/* ── Doc list view ── */}
                    {!bubbleDoc && (
                      <>
                        <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
                          <div className="flex items-center gap-2">
                            <FolderOpen className="h-4 w-4 text-primary" />
                            <span className="text-sm font-semibold text-foreground">Documents</span>
                            <span className="text-xs text-muted-foreground">({documents.length})</span>
                          </div>
                          <button onClick={() => setDocsPanelOpen(false)} className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors">
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>

                        <div className="max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                          {documents.length === 0 && !isUploading ? (
                            <div className="flex flex-col items-center justify-center py-10 gap-2 px-4 text-center">
                              <div className="h-9 w-9 rounded-xl bg-muted/60 flex items-center justify-center">
                                <FileText className="h-4 w-4 text-muted-foreground/40" />
                              </div>
                              <p className="text-xs text-muted-foreground font-medium">No documents yet</p>
                              <p className="text-[11px] text-muted-foreground/60">Upload PDFs, images or text files</p>
                            </div>
                          ) : (
                            <div className="p-2 space-y-0.5">
                              {isUploading && (
                                <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary/5">
                                  <Loader2 className="h-4 w-4 text-primary animate-spin shrink-0" />
                                  <span className="text-xs text-muted-foreground">Uploading…</span>
                                </div>
                              )}
                              {documents.map(doc => {
                                const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
                                const isImg = IMAGE_EXTS.includes(`.${ext}`);
                                return (
                                  <div
                                    key={doc.document_id}
                                    className="group flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-muted/60 cursor-pointer transition-colors"
                                    onClick={() => openBubbleDoc(doc)}
                                  >
                                    <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                                      {isImg ? <ImageIcon className="h-3.5 w-3.5 text-primary" /> : <FileText className="h-3.5 w-3.5 text-primary" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-xs font-medium text-foreground truncate">{doc.filename}</p>
                                      <p className="text-[10px] text-muted-foreground capitalize">{doc.doc_type}</p>
                                    </div>
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                                      <Eye className="h-3 w-3 text-muted-foreground/50" />
                                      <button
                                        onClick={e => { e.stopPropagation(); removeDocument(doc.document_id); }}
                                        className="p-0.5 rounded text-muted-foreground/50 hover:text-destructive transition-colors"
                                      >
                                        <X className="h-3 w-3" />
                                      </button>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>

                        <div className="p-3 border-t border-border shrink-0">
                          <input ref={fileInputRef} type="file" accept=".pdf,.txt,.docx,.doc,.png,.jpg,.jpeg,.webp,.gif" className="hidden" onChange={handleFileUpload} disabled={isUploading} />
                          <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                            className="w-full flex items-center justify-center gap-2 py-2 rounded-xl bg-primary/10 hover:bg-primary/15 text-primary text-xs font-medium transition-colors disabled:opacity-50"
                          >
                            <Plus className="h-3.5 w-3.5" />
                            Upload document
                          </button>
                        </div>
                      </>
                    )}

                    {/* ── Expanded document viewer ── */}
                    {bubbleDoc && bubbleViewer && (
                      <>
                        {/* Viewer header */}
                        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border shrink-0">
                          <button
                            onClick={() => { setBubbleDoc(null); setBubbleViewer(null); }}
                            className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors shrink-0"
                            title="Back to documents"
                          >
                            <ArrowLeft className="h-3.5 w-3.5" />
                          </button>
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-semibold text-foreground truncate">{bubbleDoc.filename}</p>
                            <p className="text-[10px] text-muted-foreground capitalize">{bubbleDoc.doc_type}</p>
                          </div>
                          <button onClick={() => { setDocsPanelOpen(false); setBubbleDoc(null); setBubbleViewer(null); }} className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors shrink-0">
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>

                        {/* Viewer content */}
                        <div className="flex-1 overflow-hidden" style={{ minHeight: 0 }}>
                          {bubbleViewer.loading && (
                            <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
                              <Loader2 className="h-6 w-6 animate-spin text-primary" />
                              <p className="text-xs">Loading…</p>
                            </div>
                          )}
                          {bubbleViewer.error && (
                            <div className="flex flex-col items-center justify-center h-full gap-2 px-6 text-center">
                              <AlertCircle className="h-6 w-6 text-destructive/70" />
                              <p className="text-xs text-muted-foreground">{bubbleViewer.error}</p>
                            </div>
                          )}
                          {!bubbleViewer.loading && !bubbleViewer.error && (
                            bubbleViewer.mediaType === 'image' ? (
                              <div className="flex items-center justify-center h-full p-3">
                                <img src={bubbleViewer.blobUrl} alt={bubbleDoc.filename} className="max-w-full max-h-full object-contain rounded-lg" />
                              </div>
                            ) : bubbleViewer.mediaType === 'pdf' && bubbleViewer.blobUrl ? (
                              <PdfViewer
                                blobUrl={bubbleViewer.blobUrl}
                                highlightText={bubbleViewer.highlightText ?? null}
                                highlightLines={bubbleViewer.highlightLines ?? null}
                                highlightKey={bubbleViewer.highlightKey ?? 0}
                              />
                            ) : bubbleViewer.content !== null ? (
                              <div ref={bubbleScrollRef} className="h-full overflow-y-auto p-3 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                                {renderDocumentContent(bubbleViewer.content, bubbleViewer.highlightText, bubbleHighlightRef, bubbleViewer.highlightKey ?? 0, bubbleViewer.highlightLines ?? null)}
                              </div>
                            ) : null
                          )}
                        </div>
                      </>
                    )}
                </div>
            </div>
            <ThemeToggle />
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex items-start ${msg.role === "user" ? "justify-end gap-2" : "gap-3"}`}
              >
                {msg.role === "assistant" && (
                  <Logo className="h-8 w-8 shrink-0 mt-0.5" />
                )}
                {msg.role === "user" && (
                  <div className="order-last h-8 w-8 rounded-lg bg-muted flex items-center justify-center shrink-0 mt-0.5">
                    <User className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
                <div className={`space-y-2 ${msg.role === "user" ? "max-w-[80%] flex flex-col items-end" : "max-w-[80%]"}`}>
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-md w-fit"
                        : "bg-secondary text-secondary-foreground rounded-bl-md w-fit"
                    }`}
                  >
                    {msg.role === "assistant" && msg.id === typingMessageId ? (
                      <TypewriterText
                        text={msg.rawAnswer ?? msg.content}
                        speed={10}
                        onComplete={() => setTypingMessageId(null)}
                        render={(partial) => (
                          <div className="leading-relaxed">{renderContent(partial, msg.id, msg.sources, handleSourceClick)}</div>
                        )}
                      />
                    ) : msg.role === "assistant" ? (
                      <div className="leading-relaxed">{renderContent(msg.rawAnswer ?? msg.content, msg.id, msg.sources, handleSourceClick)}</div>
                    ) : editingMsgId === msg.id ? (
                      /* ── Inline edit mode ── */
                      <div className="flex flex-col gap-2 -mx-1">
                        <textarea
                          ref={editTextareaRef}
                          value={editDraft}
                          onChange={(e) => setEditDraft(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              handleEditSubmit(msg.id);
                            }
                            if (e.key === "Escape") {
                              handleEditCancel();
                            }
                          }}
                          rows={1}
                          style={{ maxHeight: "300px" }}
                          className="w-full resize-none bg-primary-foreground/10 text-primary-foreground placeholder:text-primary-foreground/50 rounded-lg px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary-foreground/40 leading-relaxed overflow-y-auto scrollbar-thin"
                        />
                        <div className="flex items-center gap-1.5 justify-end">
                          <button
                            onClick={handleEditCancel}
                            className="flex items-center gap-1 text-[11px] text-primary-foreground/60 hover:text-primary-foreground px-2 py-1 rounded-md hover:bg-primary-foreground/10 transition-colors"
                          >
                            <X className="h-3 w-3" />
                            Cancel
                          </button>
                          <button
                            onClick={() => handleEditSubmit(msg.id)}
                            disabled={!editDraft.trim()}
                            className="flex items-center gap-1 text-[11px] bg-primary-foreground/15 hover:bg-primary-foreground/25 text-primary-foreground px-2 py-1 rounded-md transition-colors disabled:opacity-40"
                          >
                            <Check className="h-3 w-3" />
                            Send
                          </button>
                        </div>
                      </div>
                    ) : (
                      /* ── Normal user message ── */
                      <span>{msg.content}</span>
                    )}
                  </div>

                  {/* Timestamp + action bar */}
                  <div className={`flex items-center gap-2 px-1 ${msg.role === "user" ? "justify-end" : "justify-start"} group/msgbar`}>
                    {/* User message actions — edit + copy, shown on hover */}
                    {msg.role === "user" && editingMsgId !== msg.id && (
                      <div className="flex items-center gap-0.5 opacity-0 group-hover/msgbar:opacity-100 transition-opacity">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(msg.content);
                            setCopiedMsgId(msg.id);
                            setTimeout(() => setCopiedMsgId(null), 1500);
                          }}
                          className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground transition-colors"
                          title="Copy message"
                        >
                          {copiedMsgId === msg.id
                            ? <Check className="h-3 w-3 text-primary" />
                            : <Copy className="h-3 w-3" />
                          }
                        </button>
                        <button
                          onClick={() => {
                            setEditingMsgId(msg.id);
                            setEditDraft(msg.content);
                          }}
                          className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground transition-colors"
                          title="Edit message"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                      </div>
                    )}
                    <span className="text-[10px] text-muted-foreground/50">
                      {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                    {msg.role === "assistant" && msg.id !== typingMessageId && (
                      <div className="flex items-center gap-1 ml-1">
                        <button
                          onClick={() => setFeedback(prev => ({ ...prev, [msg.id]: prev[msg.id] === "like" ? null : "like" }))}
                          className={`p-1 rounded-md transition-colors ${feedback[msg.id] === "like" ? "text-primary" : "text-muted-foreground/40 hover:text-primary"}`}
                          title="Good response"
                        >
                          <ThumbsUp className="h-3 w-3" />
                        </button>
                        <button
                          onClick={() => setFeedback(prev => ({ ...prev, [msg.id]: prev[msg.id] === "dislike" ? null : "dislike" }))}
                          className={`p-1 rounded-md transition-colors ${feedback[msg.id] === "dislike" ? "text-destructive" : "text-muted-foreground/40 hover:text-destructive"}`}
                          title="Bad response"
                        >
                          <ThumbsDown className="h-3 w-3" />
                        </button>
                        <button
                          onClick={() => handleRedo(msg.id)}
                          disabled={isLoading}
                          className="p-1 rounded-md text-muted-foreground/40 hover:text-foreground transition-colors disabled:opacity-30"
                          title="Regenerate response"
                        >
                          <RotateCcw className="h-3 w-3" />
                        </button>
                      </div>
                    )}
                  </div>

                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (() => {
                    // Deduplicate by source_number — backend may emit dupes if same chunk cited multiple times
                    const seenNums = new Set<number>();
                    const citedSources = msg.sources.filter(s => {
                      if (!(s as any).cited_facts?.length) return false;
                      if (seenNums.has(s.source_number)) return false;
                      seenNums.add(s.source_number);
                      return true;
                    });
                    if (citedSources.length === 0) return null;

                    return (
                      <div className="mt-3 space-y-2">
                        {citedSources.map((source) => {
                          // sections: [{ title, lines, excerpt }] — grouped by ## heading from the output
                          // Normalize shape, then merge sections with the same title
                          const rawSections: Array<{ title: string; lines: string[]; excerpt: string }> =
                            ((source as any).sections ?? []).map((s: any) => ({
                              title:   s.title   ?? "",
                              lines:   Array.isArray(s.lines) ? s.lines : (s.excerpt ? [s.excerpt] : []),
                              excerpt: s.excerpt ?? "",
                            }));
                          // Merge duplicate titles: combine their lines, keep first excerpt
                          const sectionsMap = new Map<string, { lines: string[]; excerpt: string }>();
                          rawSections.forEach(sec => {
                            const key = sec.title.trim();
                            if (!sectionsMap.has(key)) {
                              sectionsMap.set(key, { lines: [...sec.lines], excerpt: sec.excerpt });
                            } else {
                              const existing = sectionsMap.get(key)!;
                              sec.lines.forEach(l => { if (!existing.lines.includes(l)) existing.lines.push(l); });
                            }
                          });
                          const sections = Array.from(sectionsMap.entries()).map(([title, v]) => ({ title, ...v }));
                          const docExcerpt: string = (source as any).excerpt ?? "";
                          const sourceKey = `${msg.id}-${source.source_number}`;
                          const isExpanded = openSourceKey === sourceKey;

                          return (
                            <div
                              key={source.source_number}
                              id={`source-${msg.id}-${source.source_number}`}
                              className="rounded-xl border border-primary/20 overflow-hidden scroll-mt-4"
                            >
                              {/* Card header */}
                              <div
                                className="flex items-center gap-2 px-3 py-2 bg-primary/8 hover:bg-primary/12 transition-colors cursor-pointer"
                                onClick={() => setOpenSourceKey(k => k === sourceKey ? null : sourceKey)}
                              >
                                <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-primary/20 text-primary font-bold text-[10px] shrink-0">
                                  {source.source_number}
                                </span>
                                <span className="flex-1 min-w-0">
                                  <span className="flex items-center gap-1.5">
                                    <span className="text-[12px] font-semibold text-foreground truncate">
                                      {sections[0]?.title || source.filename}
                                    </span>
                                    <ExternalLink
                                      className="h-3 w-3 shrink-0 text-primary hover:text-primary/60 transition-colors cursor-pointer"
                                      data-open-doc
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        openDocumentAtExcerpt(source.filename, docExcerpt);
                                      }}
                                    />
                                  </span>
                                  <span className="text-[10px] text-muted-foreground truncate block">{source.filename}</span>
                                </span>
                                <span className="text-[10px] text-muted-foreground shrink-0 flex items-center gap-1">
                                  <Eye className="h-3 w-3" />
                                  {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                                </span>
                              </div>

                              {/* Expandable — sections mirror ## headings from the output exactly */}
                              <AnimatePresence>
                                {isExpanded && sections.filter(s => s.lines.length > 0).length > 0 && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.18 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="bg-background/60">
                                      {sections.filter(s => s.lines.length > 0).map((sec, si) => (
                                        <div key={si} className="border-t border-primary/10">
                                          {/* Lines cited under this heading */}
                                          {sec.lines.map((line, li) => (
                                            <button
                                              key={li}
                                              className="w-full text-left px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground hover:bg-primary/5 transition-colors flex items-start gap-2 group/line"
                                              data-open-doc
                                              onClick={() => openDocumentAtExcerpt(source.filename, line)}
                                            >
                                              <span className="w-1 h-1 rounded-full bg-primary/40 mt-1.5 shrink-0 group-hover/line:bg-primary transition-colors" />
                                              <span className="leading-relaxed">{line}</span>
                                            </button>
                                          ))}
                                        </div>
                                      ))}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>

              </motion.div>
            ))}

            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <Logo className="h-8 w-8 shrink-0" />
                <div className="bg-secondary rounded-2xl rounded-bl-md px-4 py-3">
                  <TypingIndicator />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border p-4 overflow-hidden shadow-[0_-4px_24px_0_rgba(0,0,0,0.06)] bg-background/80 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto">
            <BorderGlow
              edgeSensitivity={30}
              glowColor="214 100 65"
              backgroundColor="transparent"
              borderRadius={16}
              glowRadius={40}
              glowIntensity={1.5}
              coneSpread={25}
              animated={false}
              colors={['#60a5fa', '#3b82f6', '#93c5fd']}
              className="w-full"
            >
            <div
              className="relative rounded-2xl bg-card shadow-md"
              onDragEnter={handleInputDragEnter}
              onDragOver={handleInputDragOver}
              onDragLeave={handleInputDragLeave}
              onDrop={handleInputDrop}
            >
              {/* Drag-over overlay — same style as the page-level one */}
              <AnimatePresence>
                {isDraggingInput && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 z-10 rounded-2xl bg-background/80 backdrop-blur-sm flex items-center justify-center pointer-events-none"
                  >
                    <motion.div
                      initial={{ scale: 0.9 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0.9 }}
                      className="border-2 border-dashed border-primary rounded-2xl p-6 bg-card/50"
                    >
                      <div className="text-center">
                        <FileText className="h-8 w-8 text-primary mx-auto mb-2" />
                        <h3 className="text-sm font-semibold text-foreground mb-1">Drop files here</h3>
                        <p className="text-xs text-muted-foreground">PDF · DOCX · TXT · PNG · JPG</p>
                      </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Actual input row — hidden behind overlay when dragging */}
              <div className={`flex items-end gap-2 px-3 py-2.5 transition-opacity duration-150 ${isDraggingInput ? "opacity-0 pointer-events-none" : "opacity-100"}`}>
                {/* Plus / attach button */}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors mb-0.5"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                >
                  <Plus className="h-5 w-5" />
                </Button>

                {/* Auto-expanding textarea */}
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask a question about your documents..."
                  rows={1}
                  disabled={isLoading}
                  style={{ maxHeight: "160px" }}
                  className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed py-1 overflow-y-auto scrollbar-thin"
                />

                {/* Voice wave button */}
                <button
                  type="button"
                  onMouseEnter={() => setIsHoveringVoice(true)}
                  onMouseLeave={() => setIsHoveringVoice(false)}
                  className="h-8 w-8 shrink-0 flex items-center justify-center rounded-lg text-muted-foreground hover:text-primary transition-colors mb-0.5 group"
                  title="Voice message"
                >
                  <svg width="22" height="18" viewBox="0 0 22 18" fill="none" xmlns="http://www.w3.org/2000/svg" className="overflow-visible">
                    {[
                      { x: 1,  baseH: 4,  hoverH: 6,  delay: "0ms"   },
                      { x: 4,  baseH: 8,  hoverH: 14, delay: "60ms"  },
                      { x: 7,  baseH: 12, hoverH: 18, delay: "120ms" },
                      { x: 10, baseH: 16, hoverH: 18, delay: "180ms" },
                      { x: 13, baseH: 12, hoverH: 18, delay: "120ms" },
                      { x: 16, baseH: 8,  hoverH: 14, delay: "60ms"  },
                      { x: 19, baseH: 4,  hoverH: 6,  delay: "0ms"   },
                    ].map((bar, i) => (
                      <rect
                        key={i}
                        x={bar.x}
                        y={isHoveringVoice ? (9 - bar.hoverH / 2) : (9 - bar.baseH / 2)}
                        width="2"
                        rx="1"
                        height={isHoveringVoice ? bar.hoverH : bar.baseH}
                        fill="currentColor"
                        style={{
                          transition: `y 0.55s cubic-bezier(0.34,1.56,0.64,1) ${bar.delay}, height 0.55s cubic-bezier(0.34,1.56,0.64,1) ${bar.delay}`,
                        }}
                      />
                    ))}
                  </svg>
                </button>

                {/* Send / Stop button */}
                {isLoading ? (
                  <Button
                    size="icon"
                    className="h-8 w-8 bg-destructive/90 hover:bg-destructive text-white transition-colors shrink-0 rounded-lg mb-0.5"
                    onClick={handleStop}
                    title="Stop generating"
                  >
                    <Square className="h-3.5 w-3.5 fill-current" />
                  </Button>
                ) : (
                  <Button
                    size="icon"
                    className="h-8 w-8 bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity shrink-0 rounded-lg mb-0.5"
                    onClick={handleSend}
                    disabled={!input.trim()}
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
            </BorderGlow>
          </div>
          <p className="text-center text-[11px] text-muted-foreground/50 mt-2 select-none">
            This agent can make mistakes — be sure to verify results.
          </p>
        </div>
      </div>

      {/* Document Viewer Panel */}
      <AnimatePresence>
        {viewer && (
          <DocumentViewer
            state={viewer}
            onClose={() => setViewer(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default Chat;