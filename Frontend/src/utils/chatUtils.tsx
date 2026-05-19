import React from "react";
import { Source } from "@/services/api";

// ── Markdown helpers ──────────────────────────────────────────────

/**
 * Returns true if the text contains block-level markdown that needs full
 * ReactMarkdown rendering (lists, headings, code blocks, tables, blockquotes).
 * Plain sentences — even with **bold** or *italic* — are inline-safe.
 */
export function hasBlockMarkdown(text: string): boolean {
  return /^[\s]*([-*+]|\d+\.)\s/m.test(text)   // lists
    || /^#{1,6}\s/m.test(text)                  // headings
    || /^```/m.test(text)                        // fenced code
    || /^\|.+\|/m.test(text)                     // tables
    || /^>/m.test(text);                         // blockquotes
}

/** Strip markdown links, keeping label text only. */
export function stripLinks(text: string): string {
  return text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
}

/**
 * Collapse "soft-broken" inline code tokens that the LLM outputs on separate lines.
 * e.g.  "`Level 1`,\n`Level 2`,\n`Level 4`"  →  "`Level 1`, `Level 2`, `Level 4`"
 * This prevents remark from treating each backtick item as a separate paragraph.
 */
export function collapseInlineBreaks(text: string): string {
  // Join lines where previous line ends with a backtick+comma and next starts with a backtick
  return text
    .replace(/(`)([,;]?\s*)\n(\s*`)/g, "$1$2 $3")
    // Also collapse lines that are ONLY a comma/comma-space (orphan commas between code tokens)
    .replace(/`,\s*\n\s*`/g, "`, `");
}

/**
 * Render inline bold/italic/code inside a plain string without wrapping in <p>.
 * Handles **bold**, *italic*, and `code` only.
 */
export function renderInlineMarkdown(text: string): React.ReactNode[] {
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

/**
 * Strategy:
 * - Split on double-newlines to get paragraphs.
 * - Each paragraph that only contains inline content (bold/italic/plain) is
 *   rendered as an inline <span> — this prevents ReactMarkdown from injecting
 *   a block <p> that forces a line-break around adjacent citation buttons.
 * - Paragraphs with real block elements (lists, headings, etc.) fall back to
 *   full ReactMarkdown so those render correctly.
 * - A blank line between paragraphs gets a small spacer div.
 */
export function renderTextSegment(raw: string, segKey: number): React.ReactNode {
  const cleaned = collapseInlineBreaks(stripLinks(raw));
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

// ── Number / spec highlighting ────────────────────────────────────

// Units ordered longest→shortest within each group so the regex alternation
// always tries the most-specific match first.  Case-sensitive for ambiguous
// SI prefixes (M=Mega vs m=milli/meters, K=kilo prefix vs k=kilo, G vs g).
export const UNIT_PATTERN = [
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
export const UNIT_RE = new RegExp(`^\\s*(?:${UNIT_PATTERN})(?![a-zA-Z])`);

export function isSpecNumber(raw: string, after: string): boolean {
  if (UNIT_RE.test(after.slice(0, 14))) return true;
  if (/^\d+:\d+$/.test(raw.trim())) return true;
  if (/^\d+-\d+$/.test(raw.trim())) return false;
  const digits = raw.replace(/[\s,.]/g, '');
  if (digits.length >= 4 && /[\s,.]/.test(raw)) return true;
  return false;
}

/** Wrap spec-numbers in visual-only highlight chips (not clickable). */
export function highlightNumbers(
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
export function injectChips(
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

// ── Source parsing ────────────────────────────────────────────────

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
export function parseSegments(raw: string, fallbackSource: number): { text: string; source: number }[] {
  const tokens = raw.split(/(\[\d+\](?!\())/);
  const segments: { text: string; source: number }[] = [];

  let buf = '';
  let lastSource = fallbackSource;

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const citeMatch = tok.match(/^\[(\d+)\]$/);
    if (citeMatch) {
      const n = parseInt(citeMatch[1]);
      if (buf.trim()) {
        segments.push({ text: buf, source: n });
      }
      buf = '';
      lastSource = n;
    } else {
      buf += tok;
    }
  }

  if (buf.trim()) {
    segments.push({ text: buf, source: lastSource });
  }

  // Strip leading orphan punctuation left by citation splitting.
  // e.g. ".\n- next bullet" → "- next bullet"
  const cleaned = segments.map(seg => ({
    ...seg,
    text: seg.text.replace(/^[\s.,;:!?]+/, ''),
  })).filter(seg => seg.text.trim().length > 0);

  return mergeAdjacentSameSourceSegments(cleaned);
}

export function mergeAdjacentSameSourceSegments(segments: { text: string; source: number }[]) {
  if (segments.length <= 1) return segments;
  const merged: { text: string; source: number }[] = [segments[0]];

  for (let i = 1; i < segments.length; i += 1) {
    const current = segments[i];
    const previous = merged[merged.length - 1];
    if (current.source === previous.source) {
      previous.text += current.text;
    } else {
      merged.push(current);
    }
  }

  return merged;
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
export function parseSourceGroups(raw: string, sources: Source[]): Map<number, { sectionTitle: string; lines: string[] }> {
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

// ── Fact / excerpt helpers ────────────────────────────────────────

/**
 * Normalize a string for fuzzy comparison:
 * collapse whitespace, strip punctuation differences.
 */
export function norm(s: string): string {
  return s.toLowerCase().replace(/[\s\u00A0]+/g, ' ').trim();
}

/** Escape HTML special chars so raw document text is safe before we inject <mark> */
export function escapeHtml(s: string): string {
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
export function highlightFact(excerpt: string, fact: string): string {
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
export function buildHighlightedExcerpt(excerpt: string, _facts: string[]): string {
  const clean = excerpt.replace(/<[^>]*>/g, '');
  return escapeHtml(clean);
}

export function findExcerptRange(content: string, excerpt: string): [number, number] | null {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const clean = (s: string) => s.replace(/\*\*/g, '').replace(/\[\d+\]/g, '').trim();
  const needle = clean(excerpt);
  if (!needle) return null;

  // Strategy 1: exact match with flexible whitespace
  const exactRe = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+'), 'i');
  const exactM = exactRe.exec(content);
  if (exactM) return [exactM.index, exactM.index + exactM[0].length];

  // Strategy 2: sliding window of 8 consecutive words — finds partial matches
  const words = needle.split(/\s+/).filter(Boolean);
  const windowSize = Math.min(8, words.length);
  for (let start = 0; start <= words.length - windowSize; start++) {
    const probe = words.slice(start, start + windowSize);
    const re = new RegExp('(' + probe.map(esc).join('[\\s.,;:-]{0,3}') + ')', 'i');
    const m = re.exec(content);
    if (m) return [m.index, m.index + m[0].length];
  }

  // Strategy 3: two longest significant words together (tighter gap)
  const stop = new Set(['the','and','for','are','was','with','this','that','from','have','been','they','will','has']);
  const sig = words.filter(w => w.length >= 4 && !stop.has(w.toLowerCase())).sort((a,b) => b.length - a.length);
  if (sig.length >= 2) {
    const re = new RegExp('(' + esc(sig[0]) + '[\\s\\S]{0,15}' + esc(sig[1]) + '|'  + esc(sig[1]) + '[\\s\\S]{0,15}' + esc(sig[0]) + ')', 'i');
    const m = re.exec(content);
    if (m) return [m.index, m.index + m[0].length];
  }
  if (sig.length >= 1) {
    const fm = new RegExp(esc(sig[0]), 'i').exec(content);
    if (fm) return [fm.index, fm.index + fm[0].length];
  }
  return null;
}

export const HIGHLIGHT_COLOR = "rgba(253, 224, 71, 0.45)"; // yellow-300 at 45%

// ── Trie (autocomplete) ─────────────────────────────────────────────

export interface TrieNode {
  children: Map<string, TrieNode>;
  freq: number;      // times this exact word was inserted
  isEnd: boolean;
}

export function makeTrie(): TrieNode {
  return { children: new Map(), freq: 0, isEnd: false };
}

export function trieInsert(root: TrieNode, word: string, weight = 1) {
  let node = root;
  for (const ch of word) {
    if (!node.children.has(ch)) node.children.set(ch, makeTrie());
    node = node.children.get(ch)!;
  }
  node.isEnd = true;
  node.freq += weight;
}

/** Walk to the node matching `prefix`, or null if not found. */
export function trieFind(root: TrieNode, prefix: string): TrieNode | null {
  let node = root;
  for (const ch of prefix) {
    if (!node.children.has(ch)) return null;
    node = node.children.get(ch)!;
  }
  return node;
}

/** Collect up to `limit` completions from a subtree, sorted by freq desc. */
export function trieCollect(node: TrieNode, prefix: string, limit: number): { word: string; freq: number }[] {
  const out: { word: string; freq: number }[] = [];
  const dfs = (n: TrieNode, buf: string) => {
    if (out.length >= limit * 3) return; // over-collect then sort
    if (n.isEnd) out.push({ word: buf, freq: n.freq });
    // Visit higher-freq children first (greedy DFS)
    const sorted = [...n.children.entries()].sort((a, b) => b[1].freq - a[1].freq);
    for (const [ch, child] of sorted) dfs(child, buf + ch);
  };
  dfs(node, prefix);
  return out.sort((a, b) => b.freq - a.freq).slice(0, limit);
}

// ── Misc ──────────────────────────────────────────────────────────

/** Serialize any caught value to a readable string (handles Error, object, string, etc.) */
export function serializeError(err: unknown): string {
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

/** Tokenise any text into lowercase words ≥3 chars, stripping markdown/punctuation. */
export function tokenise(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/```[\s\S]*?```/g, "")   // strip code blocks
    .replace(/[^a-z0-9'\s-]/g, " ")
    .split(/\s+/)
    .filter(w => w.length >= 3 && !/^\d+$/.test(w));
}

// ── Chart helpers ────────────────────────────────────────────────

export const INSIGHT_LABELS = ["Observation", "Trend", "Implication", "Note"];
export const INSIGHT_ICONS  = [
  // Bar-chart icon
  <svg key="obs" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  // Trending-up icon
  <svg key="trend" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>,
  // Lightbulb icon
  <svg key="impl" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.347a3.252 3.252 0 01-4.78 0l-.347-.347z" /></svg>,
  // Info icon
  <svg key="note" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
];

export function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by a space + capital letter
  // Handles abbreviations poorly, but good enough for 2-4 sentence interpretations
  const raw = text
    .replace(/([.!?])\s+(?=[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜ"'])/g, "$1\n")
    .split("\n")
    .map(s => s.trim())
    .filter(s => s.length > 10);
  return raw.length >= 2 ? raw : [text]; // fallback: treat whole text as one block
}

export const STROKES_DARK  = ["#818cf8","#34d399","#fbbf24","#fb7185","#60a5fa","#f472b4","#a78bfa","#2dd4bf"];
export const STROKES_LIGHT = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#ec4899","#8b5cf6","#14b8a6"];

// Gradient stop pairs [top, bottom] per palette
// Top stop is near-fully opaque; bottom stop is used at ~80% of height,
// then the plugin adds a hard rgba(0,0,0,0) stop at 1.0 for a clean fade.
export const GRAD_STOPS_DARK: [string, string][] = [
  ["rgba(129,140,248,0.95)", "rgba(99,102,241,0.22)"],
  ["rgba(52,211,153,0.95)",  "rgba(16,185,129,0.22)"],
  ["rgba(251,191,36,0.95)",  "rgba(245,158,11,0.22)"],
  ["rgba(251,113,133,0.95)", "rgba(239,68,68,0.22)"],
  ["rgba(96,165,250,0.95)",  "rgba(59,130,246,0.22)"],
  ["rgba(244,114,182,0.95)", "rgba(236,72,153,0.22)"],
  ["rgba(167,139,250,0.95)", "rgba(139,92,246,0.22)"],
  ["rgba(45,212,191,0.95)",  "rgba(20,184,166,0.22)"],
];
export const GRAD_STOPS_LIGHT: [string, string][] = [
  ["rgba(99,102,241,0.90)",  "rgba(99,102,241,0.15)"],
  ["rgba(16,185,129,0.90)",  "rgba(16,185,129,0.15)"],
  ["rgba(245,158,11,0.90)",  "rgba(245,158,11,0.15)"],
  ["rgba(239,68,68,0.90)",   "rgba(239,68,68,0.15)"],
  ["rgba(59,130,246,0.90)",  "rgba(59,130,246,0.15)"],
  ["rgba(236,72,153,0.90)",  "rgba(236,72,153,0.15)"],
  ["rgba(139,92,246,0.90)",  "rgba(139,92,246,0.15)"],
  ["rgba(20,184,166,0.90)",  "rgba(20,184,166,0.15)"],
];

// ── Gradient plugin — recreates gradients each draw using the chart's own ctx ─
// This is the correct pattern: gradients are tied to the canvas they're built on,
// so they must be created inside beforeDraw where `chart.ctx` and `chart.chartArea`
// are guaranteed to be the right canvas and have valid dimensions.
//
// Gradient direction: top = highly opaque, bottom = fully transparent, for all types.
export function makeGradientPlugin(rawCfg: Record<string, any>, isDark: boolean) {
  return {
    id: "dynamicGradients",
    beforeDraw(chart: any) {
      const { ctx, chartArea } = chart;
      if (!chartArea) return;
      const { top, bottom, left, right } = chartArea;
      const h          = bottom - top;
      const w          = right  - left;
      const chartType  = rawCfg.type as string;
      const isPie      = chartType === "pie" || chartType === "doughnut";
      const isLine     = chartType === "line" || chartType === "scatter";
      const isRadar    = chartType === "radar";
      const isPolar    = chartType === "polarArea";
      const isHBar     = chartType === "bar" && (rawCfg.options?.indexAxis === "y");
      const gradStops  = isDark ? GRAD_STOPS_DARK  : GRAD_STOPS_LIGHT;
      const strokes    = isDark ? STROKES_DARK      : STROKES_LIGHT;
      const nDatasets  = chart.data.datasets.length;

      chart.data.datasets.forEach((ds: any, i: number) => {
        const si = i % gradStops.length;

        // ── Line / Scatter — strong fill gradient top→bottom ────────────────
        if (isLine) {
          const grad = ctx.createLinearGradient(0, top, 0, bottom);
          grad.addColorStop(0,   gradStops[si][0]);   // highly opaque at top
          grad.addColorStop(0.6, gradStops[si][1]);   // fades to low at 60%
          grad.addColorStop(1,   "rgba(0,0,0,0)");    // fully transparent at bottom
          ds.backgroundColor = grad;
          return;
        }

        // ── Pie / Doughnut — radial-style diagonal gradient per slice ────────
        if (isPie) {
          const nSlices = (chart.data.labels ?? []).length;
          ds.backgroundColor = Array.from({ length: nSlices }, (_: any, j: number) => {
            const g = ctx.createLinearGradient(left + w * 0.1, top + h * 0.1, left + w * 0.9, top + h * 0.9);
            g.addColorStop(0,   gradStops[j % gradStops.length][0]);
            g.addColorStop(1,   "rgba(0,0,0,0)");
            return g;
          });
          ds.borderColor  = Array.from({ length: nSlices }, (_: any, j: number) => strokes[j % strokes.length]);
          ds.borderWidth  = 2;
          ds.hoverOffset  = 8;
          return;
        }

        // ── Polar Area — top-to-bottom gradient per segment ─────────────────
        if (isPolar) {
          const nSlices = (chart.data.labels ?? []).length;
          ds.backgroundColor = Array.from({ length: nSlices }, (_: any, j: number) => {
            const g = ctx.createLinearGradient(0, top, 0, bottom);
            g.addColorStop(0,   gradStops[j % gradStops.length][0]);
            g.addColorStop(1,   "rgba(0,0,0,0)");
            return g;
          });
          ds.borderColor = Array.from({ length: nSlices }, (_: any, j: number) => strokes[j % strokes.length]);
          ds.borderWidth = 1.5;
          return;
        }

        // ── Radar — direct color, no gradient ───────────────────────────────
        if (isRadar) {
          ds.backgroundColor = strokes[si] + "22";
          ds.borderColor     = strokes[si];
          ds.borderWidth     = 2.5;
          ds.pointBackgroundColor = strokes[si];
          ds.pointBorderColor     = isDark ? "#1e293b" : "#ffffff";
          ds.pointBorderWidth     = 2;
          return;
        }

        // ── Horizontal bar — left→right gradient ────────────────────────────
        if (isHBar) {
          const grad = ctx.createLinearGradient(left, 0, right, 0);
          grad.addColorStop(0,   gradStops[si][0]);
          grad.addColorStop(0.6, gradStops[si][1]);
          grad.addColorStop(1,   "rgba(0,0,0,0)");
          ds.backgroundColor = grad;
          ds.borderColor     = strokes[si];
          ds.borderWidth     = 1.5;
          ds.borderRadius    = 0;
          return;
        }

        // ── Vertical bar (default) — top→bottom gradient ────────────────────
        ds.backgroundColor = gradStops[si][0];
        ds.borderColor     = "transparent";
        ds.borderWidth     = 0;
        ds.borderRadius    = nDatasets === 1 ? 7 : 6;
        ds.borderSkipped   = false;
      });
    },
  };
}

// ── Apply theme colors to an existing live chart via chart.update() ───────────
// Using update() instead of destroy+recreate gives a smooth CSS-like transition.
export function applyThemeToChart(chart: any, rawCfg: Record<string, any>, isDark: boolean) {
  const chartType  = rawCfg.type as string;
  const isLine     = chartType === "line" || chartType === "scatter";
  const strokes    = isDark ? STROKES_DARK : STROKES_LIGHT;

  const textColor  = isDark ? "#f1f5f9"                : "#0f172a";
  const mutedColor = isDark ? "#94a3b8"                : "#64748b";
  const tooltipBg  = isDark ? "rgba(15,23,42,0.97)"   : "rgba(255,255,255,0.98)";
  const tooltipTxt = isDark ? "#f1f5f9"               : "#0f172a";
  const tooltipSub = isDark ? "#cbd5e1"               : "#475569";
  const tooltipBrd = isDark ? "rgba(129,140,248,0.5)" : "rgba(99,102,241,0.3)";
  const gridColor  = isDark ? "rgba(148,163,184,0.10)": "rgba(100,116,139,0.09)";

  // Dataset strokes (gradients are handled by the plugin on next draw)
  chart.data.datasets.forEach((ds: any, i: number) => {
    const si = i % strokes.length;
    if (isLine) {
      ds.borderColor          = strokes[si];
      ds.pointBackgroundColor = strokes[si];
      ds.pointBorderColor     = isDark ? "#1e293b" : "#ffffff";
    }
  });

  // Scales
  const scales = chart.options.scales ?? {};
  for (const axis of Object.values(scales) as any[]) {
    if (!axis || typeof axis !== "object") continue;
    if (axis.title) axis.title.color = mutedColor;
    if (axis.ticks) axis.ticks.color = mutedColor;
    if (axis.grid)  axis.grid.color  = gridColor;
  }

  // Plugins — legend is rendered in React DOM, not by Chart.js
  const pl = chart.options.plugins ?? {};
  pl.legend = { display: false };
  if (pl.title)           pl.title.color         = textColor;
  if (pl.tooltip) {
    pl.tooltip.backgroundColor = tooltipBg;
    pl.tooltip.titleColor      = tooltipTxt;
    pl.tooltip.bodyColor       = tooltipSub;
    pl.tooltip.borderColor     = tooltipBrd;
  }

  chart.update("none"); // "none" = instant, no animation — smooth visual swap
}

// ── Build initial Chart.js config (no canvas-bound gradients — plugin handles them) ─
export function buildInitialCfg(raw: Record<string, any>, isDark: boolean): Record<string, any> {
  const cfg        = JSON.parse(JSON.stringify(raw));
  const chartType  = cfg.type as string;
  const isPie      = chartType === "pie" || chartType === "doughnut";
  const isLine     = chartType === "line" || chartType === "scatter";
  const strokes    = isDark ? STROKES_DARK : STROKES_LIGHT;

  const textColor  = isDark ? "#f1f5f9"                : "#0f172a";
  const mutedColor = isDark ? "#94a3b8"                : "#64748b";
  const tooltipBg  = isDark ? "rgba(15,23,42,0.97)"   : "rgba(255,255,255,0.98)";
  const tooltipTxt = isDark ? "#f1f5f9"               : "#0f172a";
  const tooltipSub = isDark ? "#cbd5e1"               : "#475569";
  const tooltipBrd = isDark ? "rgba(129,140,248,0.5)" : "rgba(99,102,241,0.3)";
  const gridColor  = isDark ? "rgba(148,163,184,0.10)": "rgba(100,116,139,0.09)";

  // Placeholder colors — the gradient plugin overwrites these on first draw
  cfg.data.datasets = (cfg.data.datasets ?? []).map((ds: any, i: number) => {
    const si = i % strokes.length;
    if (isLine) {
      return { ...ds, borderColor: strokes[si], backgroundColor: "transparent", fill: true, tension: 0.42, borderWidth: 2.5, pointRadius: 4, pointHoverRadius: 7, pointBackgroundColor: strokes[si], pointBorderColor: isDark ? "#1e293b" : "#ffffff", pointBorderWidth: 2 };
    }
    if (isPie) {
      return { ...ds, backgroundColor: strokes.slice(0, (cfg.data.labels ?? []).length), borderColor: "transparent", borderWidth: 2, hoverOffset: 8 };
    }
    return { ...ds, backgroundColor: strokes[si], borderColor: "transparent", borderWidth: 0, borderRadius: 7, borderSkipped: false };
  });

  cfg.options         = cfg.options         ?? {};
  cfg.options.plugins = cfg.options.plugins ?? {};
  cfg.options.scales  = cfg.options.scales  ?? {};

  for (const axis of Object.values(cfg.options.scales) as any[]) {
    if (!axis || typeof axis !== "object") continue;
    if (axis.title)  axis.title.color = mutedColor;
    if (axis.ticks)  { axis.ticks.color = mutedColor; axis.ticks.font = { ...(axis.ticks.font ?? {}), size: 11 }; }
    if (axis.grid)   { axis.grid.color = gridColor; axis.grid.drawBorder = false; axis.grid.drawTicks = false; }
    if (axis.border) axis.border = { display: false };
  }

  const pl = cfg.options.plugins;
  pl.legend  = { display: false };  // legend rendered as React DOM below the canvas
  // Disable canvas title — rendered as a React <span> above the canvas instead,
  // so it automatically picks up Tailwind theme classes with no JS required.
  pl.title   = { display: false };
  pl.tooltip = { ...(pl.tooltip ?? {}), backgroundColor: tooltipBg, titleColor: tooltipTxt, bodyColor: tooltipSub, borderColor: tooltipBrd, borderWidth: 1, padding: { x: 12, y: 10 }, cornerRadius: 10, boxPadding: 4 };

  const cbs = pl?.tooltip?.callbacks;
  if (cbs?.label && typeof cbs.label === "string") {
    try {   cbs.label = new Function("return " + cbs.label)(); } catch { delete cbs.label; }
  }

  cfg.options.animation = {
    duration: 750, easing: "easeOutQuart",
    delay: (ctx: any) => (ctx.type === "data" && ctx.mode === "default") ? ctx.dataIndex * 30 + ctx.datasetIndex * 60 : 0,
  };
  if (isLine) {
    cfg.options.animations = { y: { duration: 750, easing: "easeOutQuart", from: (ctx: any) => { const s = ctx.chart?.scales?.y; return s ? s.getPixelForValue(0) : ctx.chart?.height ?? 0; } } };
  }

  cfg.options.responsive          = true;
  cfg.options.maintainAspectRatio = true;
  return cfg;
}

// ── PNG download — snapshot the live canvas with a solid background ───────────
// We draw the bg fill directly onto the live canvas (then restore), which avoids
// the off-screen canvas cross-contamination issue entirely.
export function downloadChartPng(chart: any, filename: string, isDark: boolean) {
  if (!chart) return;
  const canvas = chart.canvas as HTMLCanvasElement;
  const ctx    = canvas.getContext("2d")!;
  const bg     = isDark ? "#0f172a" : "#ffffff";

  // Save → fill bg behind existing pixels → export → restore
  ctx.save();
  ctx.globalCompositeOperation = "destination-over";
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  const link     = document.createElement("a");
  link.download  = filename;
  link.href      = canvas.toDataURL("image/png", 1.0);
  link.click();
}

// ── Greeting ────────────────────────────────────────────────────

export type GreetingBucket = "night" | "morning" | "afternoon" | "evening";

export function getGreetingBucket(): GreetingBucket {
  const h = new Date().getHours();
  if (h < 5) return "night";
  if (h < 12) return "morning";
  if (h < 18) return "afternoon";
  return "evening";
}
