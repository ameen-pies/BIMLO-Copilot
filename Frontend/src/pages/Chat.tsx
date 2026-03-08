import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, FileText, X, User, ArrowLeft, Plus, Loader2, AlertCircle, ChevronDown, ChevronUp, ExternalLink, ScrollText, Eye } from "lucide-react";
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

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rawAnswer?: string;   // original answer with {{}} citation tags (for source matching)
  sources?: Source[];
  confidence?: number;
  timestamp: Date;
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
  h2: ({ children }) => <h2 className="text-sm font-bold mb-1.5 mt-3 block">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 block">{children}</h3>,
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
}

interface DocumentViewerProps {
  state: ViewerState;
  onClose: () => void;
}

function findExcerptRange(content: string, excerpt: string): [number, number] | null {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const exactRe = new RegExp(excerpt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+'), 'i');
  const exactM = exactRe.exec(content);
  if (exactM) {
    const ls = content.lastIndexOf('\n', exactM.index) + 1;
    const le = content.indexOf('\n', exactM.index + exactM[0].length);
    return [ls, le === -1 ? content.length : le];
  }
  const words = excerpt.trim().split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    const m = new RegExp(`(${words.map(esc).join('[\\s\\S]{0,8}')})`, 'i').exec(content);
    if (m) {
      const ls = content.lastIndexOf('\n', m.index) + 1;
      const le = content.indexOf('\n', m.index + m[0].length);
      return [ls, le === -1 ? content.length : le];
    }
  }
  const stop = new Set(['the','and','for','are','was','with','this','that','from','have','been','they']);
  const longest = words.filter(w => w.length >= 4 && !stop.has(w.toLowerCase())).sort((a,b) => b.length - a.length)[0];
  if (longest) {
    const fm = new RegExp(esc(longest), 'i').exec(content);
    if (fm) {
      const ls = content.lastIndexOf('\n', fm.index) + 1;
      const le = content.indexOf('\n', fm.index + fm[0].length);
      return [ls, le === -1 ? content.length : le];
    }
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

const DocumentViewer: React.FC<DocumentViewerProps> = ({ state, onClose }) => {
  const highlightRef = useRef<HTMLElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Scroll to highlighted passage whenever it changes
  useEffect(() => {
    if (!highlightRef.current || !scrollContainerRef.current) return;
    const container = scrollContainerRef.current;
    const el = highlightRef.current;
    const elTop = el.getBoundingClientRect().top - container.getBoundingClientRect().top;
    const targetScroll = container.scrollTop + elTop - container.clientHeight / 2 + el.offsetHeight / 2;
    container.scrollTo({ top: targetScroll, behavior: "smooth" });
  }, [state.highlightKey, state.content]);

  const ext = state.doc.filename.split('.').pop()?.toLowerCase() ?? '';
  const isTxt = ext === 'txt';

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
        <ScrollText className="h-4 w-4 text-primary shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-foreground truncate">{state.doc.filename}</p>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{state.doc.doc_type}</p>
        </div>
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground shrink-0" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Highlight badge */}
      {state.highlightText && (
        <div className="px-4 py-2 bg-primary/8 border-b border-primary/20 flex items-start gap-2">
          <span className="text-[10px] font-semibold text-primary uppercase tracking-wide shrink-0 mt-0.5">Cited passage</span>
          <p className="text-[11px] text-primary/80 leading-relaxed line-clamp-2 flex-1">{state.highlightText}</p>
        </div>
      )}

      {/* Content */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40">
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

        {!state.loading && !state.error && state.content !== null && (
          isTxt ? (
            renderDocumentContent(state.content, state.highlightText, highlightRef, state.highlightKey, state.highlightLines)
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
              <FileText className="h-10 w-10 text-muted-foreground/40" />
              <p className="text-sm text-muted-foreground">
                In-app preview for <span className="font-semibold">.{ext}</span> files coming soon.
              </p>
              <p className="text-xs text-muted-foreground/60">Only .txt files are rendered inline for now.</p>
            </div>
          )
        )}
      </div>
    </motion.aside>
  );
};

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
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragDepth, setDragDepth] = useState(0);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isHoveringVoice, setIsHoveringVoice] = useState(false);
  const { toast } = useToast();

  // ── Document viewer helpers ──────────────────────────────────────────────

  const openDocumentViewer = async (doc: Document, highlightText: string | null = null, highlightLines: string[] | null = null) => {
    setViewer(prev => {
      const alreadyLoaded = prev && prev.doc.document_id === doc.document_id && prev.content !== null;
      if (alreadyLoaded) {
        return { ...prev!, highlightText, highlightLines, highlightKey: (prev!.highlightKey ?? 0) + 1 };
      }
      return { doc, content: null, loading: true, error: null, highlightText, highlightLines, highlightKey: 0 };
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
    openDocumentViewer(doc, excerpt, null);
  };

  const openDocumentAtExcerpts = (filename: string, excerpts: string[]) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    openDocumentViewer(doc, excerpts[0] ?? null, excerpts);
  };

  // Open viewer and highlight the line containing a specific number value
  const openDocumentAtNumber = async (filename: string, numValue: string, fallbackExcerpt: string) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    if (viewer && viewer.doc.document_id === doc.document_id && viewer.content) {
      const line = findLineForNumber(viewer.content, numValue) ?? fallbackExcerpt;
      setViewer(prev => prev ? { ...prev, highlightText: line, highlightLines: null, highlightKey: (prev.highlightKey ?? 0) + 1 } : prev);
      return;
    }
    setViewer({ doc, content: null, loading: true, error: null, highlightText: fallbackExcerpt, highlightLines: null, highlightKey: 0 });
    try {
      const res = await (api as any).getDocumentContent(doc.document_id);
      const fullContent = res.content as string;
      const line = findLineForNumber(fullContent, numValue) ?? fallbackExcerpt;
      setViewer(p => p && p.doc.document_id === doc.document_id
        ? { ...p, content: fullContent, loading: false, error: null, highlightText: line, highlightLines: null, highlightKey: (p.highlightKey ?? 0) + 1 }
        : p
      );
    } catch {
      setViewer(p => p && p.doc.document_id === doc.document_id
        ? { ...p, loading: false, error: "Could not load document content." }
        : p
      );
    }
  };

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
        description: error instanceof Error ? error.message : "Unknown error",
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

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setDragDepth(0);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    // Process each file
    for (const file of files) {
      // Validate file type
      const allowedTypes = ['.pdf', '.docx', '.doc', '.txt'];
      const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      
      if (!allowedTypes.includes(fileExt)) {
        toast({
          title: "Invalid file type",
          description: `${file.name}: Please upload ${allowedTypes.join(', ')} files only`,
          variant: "destructive",
        });
        continue;
      }

      // Validate file size (max 10MB)
      const maxSize = 10 * 1024 * 1024;
      if (file.size > maxSize) {
        toast({
          title: "File too large",
          description: `${file.name}: Maximum file size is 10MB`,
          variant: "destructive",
        });
        continue;
      }

      setIsUploading(true);

      try {
        const response = await api.uploadDocument(file);
        
        toast({
          title: "Document uploaded successfully",
          description: `${response.filename} processed (${response.chunks_processed} chunks)`,
        });

        await loadDocuments();
      } catch (error) {
        toast({
          title: "Upload failed",
          description: `${file.name}: ${error instanceof Error ? error.message : "Unknown error"}`,
          variant: "destructive",
        });
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    
    // Validate file type
    const allowedTypes = ['.pdf', '.docx', '.doc', '.txt'];
    const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!allowedTypes.includes(fileExt)) {
      toast({
        title: "Invalid file type",
        description: `Please upload ${allowedTypes.join(', ')} files only`,
        variant: "destructive",
      });
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      toast({
        title: "File too large",
        description: "Maximum file size is 10MB",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);

    try {
      const response = await api.uploadDocument(file);
      
      toast({
        title: "Document uploaded successfully",
        description: `${response.filename} processed (${response.chunks_processed} chunks)`,
      });

      // Reload documents list
      await loadDocuments();

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
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
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };

  // ==========================================================================
  // MESSAGE HANDLING
  // ==========================================================================

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    // Check if documents are uploaded
    if (documents.length === 0) {
      toast({
        title: "No documents uploaded",
        description: "Please upload at least one document before asking questions",
        variant: "destructive",
      });
      return;
    }

    // Collapse all sources when sending new message
    setExpandedSources({});

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
      // Query the RAG system
      const response = await api.query(trimmedInput);

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

      setMessages((prev) => [...prev, assistantMsg]);
      setTypingMessageId(assistantMsg.id);

    } catch (error) {
      // Create error message
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again.`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMsg]);

      toast({
        title: "Query failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
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
    setExpandedSources((prev) => ({ ...prev, [`${msgId}-${sourceNum}`]: true }));

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

    // First try: full document content already in viewer state
    if (viewer && viewer.doc.filename === source?.filename && viewer.content) {
      highlightSentence = findLineWithNumber(viewer.content, numValue);
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
                  Upload PDF, DOCX, DOC, or TXT files
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="border-r border-border bg-secondary/30 flex flex-col overflow-hidden shrink-0"
          >
            <div className="p-4 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                <span className="font-heading font-semibold text-foreground text-sm">
                  Source Documents ({documents.length})
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-foreground"
                onClick={() => setSidebarOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40">
              {documents.length === 0 && !isUploading && (
                <div className="text-center py-12">
                  <FileText className="h-10 w-10 text-muted-foreground/40 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground">No documents uploaded yet</p>
                  <p className="text-xs text-muted-foreground/60 mt-1">Upload PDFs, .txt, or .docx files</p>
                </div>
              )}

              {isUploading && (
                <div className="text-center py-8">
                  <Loader2 className="h-8 w-8 text-primary mx-auto mb-2 animate-spin" />
                  <p className="text-sm text-muted-foreground">Uploading document...</p>
                </div>
              )}

              {documents.map((doc) => (
                <motion.div
                  key={doc.document_id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-start gap-3 p-3 rounded-lg bg-card border border-border group hover:border-primary/30 hover:bg-primary/5 transition-colors cursor-pointer"
                  onClick={() => openDocumentViewer(doc, null)}
                >
                  <div className="h-9 w-9 rounded-lg bg-accent flex items-center justify-center shrink-0">
                    <FileText className="h-4 w-4 text-accent-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate" title={doc.filename}>
                      {doc.filename}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Type: {doc.doc_type}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <span className="opacity-0 group-hover:opacity-60 transition-opacity">
                      <Eye className="h-3.5 w-3.5 text-muted-foreground" />
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity"
                      onClick={(e) => { e.stopPropagation(); removeDocument(doc.document_id); }}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="p-4 border-t border-border">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.docx,.doc"
                className="hidden"
                onChange={handleFileUpload}
                disabled={isUploading}
              />
              <Button
                variant="outline"
                className="w-full gap-2 text-sm font-medium"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4" />
                    Upload Documents
                  </>
                )}
              </Button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-14 border-b border-border flex items-center px-4 gap-3 shrink-0">
          <Link to="/">
            <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          {!sidebarOpen && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
              onClick={() => setSidebarOpen(true)}
            >
              <FileText className="h-4 w-4" />
            </Button>
          )}
          <div className="flex items-center gap-2">
            <Logo className="h-7 w-7" />
            <span className="font-heading font-semibold text-sm text-foreground">Bimlo Copilot</span>
          </div>
          <div className="ml-auto">
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
                className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
              >
                {msg.role === "assistant" && (
                  <Logo className="h-8 w-8 shrink-0 mt-0.5" />
                )}
                <div className="max-w-[80%] space-y-2">
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-md"
                        : "bg-secondary text-secondary-foreground rounded-bl-md"
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
                    ) : (
                      msg.content
                    )}
                  </div>

                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (() => {
                    const citationBase = msg.rawAnswer ?? msg.content;
                    const citedNums = new Set(
                      Array.from(citationBase.matchAll(/\[(\d+)\](?!\()/g)).map(m => parseInt(m[1]))
                    );
                    // Only show sources that are actually cited in the answer
                    const citedSources = msg.sources.filter(s => citedNums.has(s.source_number));
                    if (citedSources.length === 0) return null;

                    return (
                      <div className="mt-3 space-y-2">
                        {citedSources.map((source) => {
                          // sections = verbatim doc sentences from source agent, one per cited bullet
                          const sections: Array<{ title: string; excerpt: string }> = (source as any).sections ?? [];
                          // Deduplicate titles — the shared ## heading appears on every section
                          const uniqueTitles = Array.from(new Set(sections.map(s => s.title).filter(Boolean)));
                          const cardTitle = uniqueTitles[0] ?? source.filename;
                          const excerpts = sections.map(s => s.excerpt).filter(Boolean);
                          const isExpanded = expandedSources[`${msg.id}-${source.source_number}`];

                          return (
                            <div
                              key={source.source_number}
                              id={`source-${msg.id}-${source.source_number}`}
                              className="rounded-xl border border-primary/20 overflow-hidden scroll-mt-4"
                            >
                              {/* Pill header — expand/collapse only, no document highlight */}
                              <div
                                className="flex items-center gap-2 px-3 py-2 bg-primary/8 hover:bg-primary/12 transition-colors cursor-pointer"
                                onClick={() => setExpandedSources(prev => ({
                                  ...prev,
                                  [`${msg.id}-${source.source_number}`]: !prev[`${msg.id}-${source.source_number}`]
                                }))}
                              >
                                <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-primary/20 text-primary font-bold text-[10px] shrink-0">
                                  {source.source_number}
                                </span>
                                <span className="flex-1 min-w-0">
                                  <span className="flex items-center gap-1.5">
                                    <span className="text-[12px] font-semibold text-foreground truncate">{cardTitle}</span>
                                    <ExternalLink
                                      className="h-3 w-3 shrink-0 text-primary hover:text-primary/60 transition-colors cursor-pointer"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        // Open viewer and highlight all excerpts as separate spans
                                        openDocumentAtExcerpts(source.filename, excerpts.length > 0 ? excerpts : [source.excerpt ?? ""]);
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

                              {/* Expandable excerpts — each is a verbatim doc sentence */}
                              <AnimatePresence>
                                {isExpanded && excerpts.length > 0 && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: "auto", opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.18 }}
                                    className="overflow-hidden"
                                  >
                                    <div className="divide-y divide-primary/10 bg-background/60">
                                      {excerpts.map((excerpt, li) => (
                                        <button
                                          key={li}
                                          className="w-full text-left px-3 py-2 text-[11px] text-muted-foreground hover:text-foreground hover:bg-primary/5 transition-colors flex items-start gap-2 group/line"
                                          onClick={() => openDocumentAtExcerpt(source.filename, excerpt)}
                                        >
                                          <span className="w-1 h-1 rounded-full bg-primary/40 mt-1.5 shrink-0 group-hover/line:bg-primary transition-colors" />
                                          <span className="leading-relaxed">{excerpt}</span>
                                        </button>
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
                {msg.role === "user" && (
                  <div className="h-8 w-8 rounded-lg bg-muted flex items-center justify-center shrink-0 mt-0.5">
                    <User className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
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
        <div className="border-t border-border p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-2 rounded-2xl border border-input bg-card px-3 py-2.5 focus-within:ring-2 focus-within:ring-ring focus-within:border-transparent transition-all">
              {/* Plus / attach button — front */}
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
                style={{ maxHeight: "200px" }}
                className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed py-1 overflow-y-auto"
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

              {/* Send button */}
              <Button
                size="icon"
                className="h-8 w-8 bg-hero-gradient text-primary-foreground shadow-blue hover:opacity-90 transition-opacity shrink-0 rounded-lg mb-0.5"
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
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