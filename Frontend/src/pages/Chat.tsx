import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, FileText, X, User, ArrowLeft, Plus, Loader2, AlertCircle, ChevronDown, ChevronUp, ExternalLink, ScrollText, Eye, Square, ThumbsUp, ThumbsDown, RotateCcw, Pencil, Check, Copy, ImageIcon, Search, MessageSquare, Clock, SortAsc, FolderOpen, Trash2, Sparkles, Bell, BellOff, BookOpen, BarChart2, ChevronRight, RefreshCw, Phone } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link, useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import ThemeToggle from "@/components/ThemeToggle";
import TypingIndicator from "@/components/TypingIndicator";
import TypewriterText from "@/components/TypewriterText";
import Logo from "@/components/Logo";
import api, { Document, Source } from "@/services/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import BorderGlow from "@/components/BorderGlow";
import { useAuth } from "@/context/AuthContext";
import { ProfileBubble } from "@/components/Navbar";

interface ThinkingStep { node: string; icon: string; message: string; ts: number; }

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  rawAnswer?: string;
  sources?: Source[];
  confidence?: number;
  timestamp: Date;
  thinkingSteps?: ThinkingStep[];
  // Voice message fields
  voiceBlobUrl?: string;   // object URL for the recorded audio
  voiceDuration?: number;  // seconds
  voiceTranscript?: string; // the transcription text
  voiceWaveform?: number[]; // captured amplitude samples (0-1) during recording
  interrupted?: true;       // response was stopped mid-generation
  callCard?: { duration: number; startedAt: Date }; // voice call summary card
  attachedDocIds?: string[];               // doc IDs staged at send time (shown in bubble)
  analytics?: Record<string, any> | null;  // chart_config or chart_error from graph_node
  reportId?: string | null;                // report card rendered below the response
  reportTitle?: string | null;             // title from SSE event — no fetch needed
  reportMeta?: { word_count: number; section_count: number; source_docs: string[]; version: number } | null;
  reportGenerating?: boolean;              // true while the report is being created
  navAction?: { path: string; label: string; icon: string } | null; // inline nav buttons baked into the bubble
}

interface Conversation {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messages: Message[];
}

// ---------------------------------------------------------------------------
// Report types
// ---------------------------------------------------------------------------

interface ChartRecord {
  section_id:     string;
  chart_id:       string;
  chart_js:       Record<string, unknown>;
  title:          string;
  description:    string;
  interpretation: string;
}

interface VersionInfo {
  version:     number;
  title:       string;
  instruction: string;
  created_at:  string;
}

interface ReportRecord {
  report_id:   string;
  title:       string;
  content:     string;
  summary?:    string;   // conversational summary returned by the agent
  charts:      ChartRecord[];
  source_docs: string[];
  created_at:  string;
  updated_at:  string;
  version:     number;
  versions:    VersionInfo[];
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
 * Collapse "soft-broken" inline code tokens that the LLM outputs on separate lines.
 * e.g.  "`Level 1`,\n`Level 2`,\n`Level 4`"  →  "`Level 1`, `Level 2`, `Level 4`"
 * This prevents remark from treating each backtick item as a separate paragraph.
 */
function collapseInlineBreaks(text: string): string {
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
  li: ({ children }) => {
    // Flatten children: unwrap any wrapping <p> spans that remark injects inside
    // list items — they cause inline `code` tokens to appear on their own line.
    const flatChildren = React.Children.map(children, (child: any) => {
      if (child?.props?.className === "inline leading-relaxed") return child.props.children;
      return child;
    });
    const text = typeof children === "string" ? children : Array.isArray(children) ? children.join("") : String(children ?? "");
    if (!text.trim()) return null;
    return <li className="ml-2 leading-relaxed">{flatChildren ?? children}</li>;
  },
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  em: ({ children }) => <em className="italic text-foreground/80">{children}</em>,
  h1: ({ children }) => <h1 className="text-base font-bold mb-2 mt-4 block">{children}</h1>,
  h2: ({ children }) => <h2 className="text-sm font-bold mb-1.5 mt-3 first:mt-0 block">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mb-1 mt-2 first:mt-0 block">{children}</h3>,
  code: ({ inline, node, children }: { inline?: boolean; node?: any; children?: React.ReactNode }) => {
    const codeStr = String(children ?? "");
    const isInline = inline === true || (inline === undefined && !codeStr.includes("\n"));
    if (isInline) {
      return <code className="bg-primary/10 px-1.5 py-0.5 rounded text-[11px] font-mono text-primary whitespace-nowrap border border-primary/15">{children}</code>;
    }
    const codeText = codeStr.replace(/\n$/, "");
    const classList: string[] = node?.properties?.className ?? [];
    const langClass = classList.find((c: string) => c.startsWith("language-")) ?? "";
    const lang = langClass.replace("language-", "");

    // ── Language → { label, accentColor, fileExt } ──────────────────────
    // accentColor: vivid enough to read on BOTH dark AND light backgrounds.
    // All chosen to be AA-contrast compliant against both #0f172a and #ffffff.
    const LANG_META: Record<string, { label: string; accent: string; ext: string }> = {
      python:     { label: "Python",     accent: "#2563eb", ext: "py"   },
      py:         { label: "Python",     accent: "#2563eb", ext: "py"   },
      javascript: { label: "JavaScript", accent: "#b45309", ext: "js"   },
      js:         { label: "JavaScript", accent: "#b45309", ext: "js"   },
      typescript: { label: "TypeScript", accent: "#4338ca", ext: "ts"   },
      ts:         { label: "TypeScript", accent: "#4338ca", ext: "ts"   },
      tsx:        { label: "TSX",        accent: "#0e7490", ext: "tsx"  },
      jsx:        { label: "JSX",        accent: "#047857", ext: "jsx"  },
      bash:       { label: "Bash",       accent: "#15803d", ext: "sh"   },
      sh:         { label: "Shell",      accent: "#15803d", ext: "sh"   },
      json:       { label: "JSON",       accent: "#c2410c", ext: "json" },
      css:        { label: "CSS",        accent: "#be185d", ext: "css"  },
      html:       { label: "HTML",       accent: "#b91c1c", ext: "html" },
      sql:        { label: "SQL",        accent: "#6d28d9", ext: "sql"  },
      yaml:       { label: "YAML",       accent: "#0f766e", ext: "yaml" },
      yml:        { label: "YAML",       accent: "#0f766e", ext: "yml"  },
      rust:       { label: "Rust",       accent: "#c2410c", ext: "rs"   },
      go:         { label: "Go",         accent: "#0369a1", ext: "go"   },
      java:       { label: "Java",       accent: "#b91c1c", ext: "java" },
      cpp:        { label: "C++",        accent: "#4338ca", ext: "cpp"  },
      c:          { label: "C",          accent: "#4338ca", ext: "c"    },
      csharp:     { label: "C#",         accent: "#6d28d9", ext: "cs"   },
      cs:         { label: "C#",         accent: "#6d28d9", ext: "cs"   },
      php:        { label: "PHP",        accent: "#6d28d9", ext: "php"  },
      ruby:       { label: "Ruby",       accent: "#b91c1c", ext: "rb"   },
      swift:      { label: "Swift",      accent: "#c2410c", ext: "swift"},
      kotlin:     { label: "Kotlin",     accent: "#6d28d9", ext: "kt"   },
      r:          { label: "R",          accent: "#1d4ed8", ext: "r"    },
      markdown:   { label: "Markdown",   accent: "#374151", ext: "md"   },
      md:         { label: "Markdown",   accent: "#374151", ext: "md"   },
      xml:        { label: "XML",        accent: "#c2410c", ext: "xml"  },
      ifc:        { label: "IFC",        accent: "#15803d", ext: "ifc"  },
      text:       { label: "Text",       accent: "#374151", ext: "txt"  },
      txt:        { label: "Text",       accent: "#374151", ext: "txt"  },
    };
    const meta = LANG_META[lang.toLowerCase()] ?? { label: lang || "Code", accent: "#374151", ext: lang || "txt" };

    // ── Smart filename inference ───────────────────────────────────────────
    // Scan the first ~20 lines for def/class/function/const declarations
    // and use the first match as the base filename.
    const inferFilename = (): string => {
      const lines = codeText.split("\n").slice(0, 20);
      // Patterns: Python def/class, JS/TS function/const/class, Rust fn/struct, Go func, Java/C# class
      const patterns = [
        /^(?:export\s+)?(?:async\s+)?(?:function\s+)([\w$]+)/,      // function foo
        /^(?:export\s+)?(?:default\s+)?class\s+([\w$]+)/,            // class Foo
        /^(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*=/,           // const foo =
        /^def\s+([\w]+)/,                                             // Python def
        /^class\s+([\w]+)/,                                           // Python class
        /^(?:pub\s+)?fn\s+([\w]+)/,                                   // Rust fn
        /^func\s+([\w]+)/,                                            // Go func
        /^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:class|interface|enum)\s+([\w]+)/, // Java/C#
        /^(?:sub|function)\s+([\w]+)/i,                               // VB/general
      ];
      for (const line of lines) {
        const trimmed = line.trim();
        for (const pat of patterns) {
          const m = trimmed.match(pat);
          if (m && m[1]) {
            // Convert CamelCase → snake_case for py, keep as-is for others
            const name = meta.ext === "py"
              ? m[1].replace(/([A-Z])/g, (c, i) => (i > 0 ? "_" : "") + c.toLowerCase())
              : m[1];
            return `${name}.${meta.ext}`;
          }
        }
      }
      return `snippet.${meta.ext}`;
    };
    const downloadFilename = inferFilename();

    // ── Token colours — dual pairs: [darkModeHex, lightModeHex] ──────────
    // We detect dark mode once per render using matchMedia.
    // All colours chosen for AA contrast on their respective backgrounds.
    const isDark = typeof window !== "undefined"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        || document.documentElement.classList.contains("dark")
      : true;

    const C = {
      base:     isDark ? "#cbd5e1" : "#1e293b",   // default code text
      comment:  isDark ? "#64748b" : "#6b7280",   // comments (muted, italic)
      string:   isDark ? "#10b981" : "#047857",   // strings (green family)
      number:   isDark ? "#f97316" : "#b45309",   // numbers (amber family)
      kw_ctrl:  isDark ? "#c084fc" : "#7c3aed",   // control flow (purple)
      kw_decl:  isDark ? "#60a5fa" : "#1d4ed8",   // declarations (blue)
      kw_lit:   isDark ? "#f97316" : "#b45309",   // literals (amber)
      type:     isDark ? "#34d399" : "#047857",   // types (teal)
      builtin:  isDark ? "#fbbf24" : "#b45309",   // builtins (amber)
      classname:isDark ? "#34d399" : "#0f766e",   // class names (teal)
      dunder:   isDark ? "#94a3b8" : "#64748b",   // __dunder__ (slate)
      operator: isDark ? "#f87171" : "#b91c1c",   // operators (red)
      bracket:  isDark ? "#94a3b8" : "#475569",   // brackets (slate)
      punct:    isDark ? "#64748b" : "#6b7280",   // punctuation (muted)
      decorator:isDark ? "#c084fc" : "#7c3aed",   // decorators (purple)
    };

    // ── Token-based syntax highlighter ────────────────────────────────────
    function tokenize(code: string, language: string): React.ReactNode[] {
      const l = language.toLowerCase();

      const isLineComment = (s: string) =>
        (["python","py","r"].includes(l) && s.startsWith("#")) ||
        (["javascript","js","typescript","ts","tsx","jsx","java","cpp","c","csharp","cs","go","rust","swift","kotlin","php"].includes(l) && s.startsWith("//")) ||
        (["sql"].includes(l) && s.startsWith("--")) ||
        (["bash","sh"].includes(l) && s.startsWith("#"));

      const lines = code.split("\n");
      const result: React.ReactNode[] = [];
      let inBlockComment = false;
      let inDocstring = false;
      let docstringChar = "";

      lines.forEach((line, li) => {
        if (li > 0) result.push("\n");

        const cs = (color: string, content: string, key: string, italic = false) => (
          <span key={key} style={{ color, ...(italic ? { fontStyle: "italic" } : {}) }}>{content}</span>
        );

        // Block comment /* */ languages
        if (["javascript","js","typescript","ts","tsx","jsx","java","cpp","c","csharp","cs","go","rust","swift","kotlin","php","css"].includes(l)) {
          if (inBlockComment) {
            const endIdx = line.indexOf("*/");
            if (endIdx !== -1) {
              result.push(cs(C.comment, line.slice(0, endIdx + 2), `bc-${li}`, true));
              inBlockComment = false;
              const rest = line.slice(endIdx + 2);
              if (rest.trim()) result.push(...tokenizeLine(rest, l, li));
            } else {
              result.push(cs(C.comment, line, `bc-${li}`, true));
            }
            return;
          }
          const bcStart = line.indexOf("/*");
          if (bcStart !== -1 && !line.slice(0, bcStart).includes('"') && !line.slice(0, bcStart).includes("'")) {
            const bcEnd = line.indexOf("*/", bcStart + 2);
            if (bcEnd !== -1) {
              if (bcStart > 0) result.push(...tokenizeLine(line.slice(0, bcStart), l, li));
              result.push(cs(C.comment, line.slice(bcStart, bcEnd + 2), `bc2-${li}`, true));
              const rest = line.slice(bcEnd + 2);
              if (rest.trim()) result.push(...tokenizeLine(rest, l, li));
              return;
            } else {
              if (bcStart > 0) result.push(...tokenizeLine(line.slice(0, bcStart), l, li));
              result.push(cs(C.comment, line.slice(bcStart), `bc3-${li}`, true));
              inBlockComment = true;
              return;
            }
          }
        }

        // Python docstrings
        if (["python","py"].includes(l)) {
          if (inDocstring) {
            const endIdx = line.indexOf(docstringChar);
            if (endIdx !== -1) {
              result.push(cs(C.comment, line.slice(0, endIdx + 3), `ds-${li}`, true));
              inDocstring = false;
              const rest = line.slice(endIdx + 3);
              if (rest.trim()) result.push(...tokenizeLine(rest, l, li));
            } else {
              result.push(cs(C.comment, line, `ds-${li}`, true));
            }
            return;
          }
          for (const q of ['"""', "'''"]) {
            const startIdx = line.indexOf(q);
            if (startIdx !== -1) {
              const endIdx = line.indexOf(q, startIdx + 3);
              if (endIdx !== -1) {
                if (startIdx > 0) result.push(...tokenizeLine(line.slice(0, startIdx), l, li));
                result.push(cs(C.comment, line.slice(startIdx, endIdx + 3), `ds2-${li}`, true));
                const rest = line.slice(endIdx + 3);
                if (rest.trim()) result.push(...tokenizeLine(rest, l, li));
              } else {
                if (startIdx > 0) result.push(...tokenizeLine(line.slice(0, startIdx), l, li));
                result.push(cs(C.comment, line.slice(startIdx), `ds3-${li}`, true));
                inDocstring = true;
                docstringChar = q;
              }
              return;
            }
          }
        }

        // Full-line comment
        const trimmed = line.trimStart();
        if (isLineComment(trimmed)) {
          result.push(cs(C.comment, line, `cm-${li}`, true));
          return;
        }

        // Inline comment detection
        let commentStart = -1;
        if (["python","py","r","bash","sh"].includes(l)) {
          let inStr = false; let strChar = "";
          for (let ci = 0; ci < line.length; ci++) {
            const ch = line[ci];
            if (!inStr && (ch === '"' || ch === "'")) { inStr = true; strChar = ch; }
            else if (inStr && ch === strChar && line[ci-1] !== "\\") { inStr = false; }
            else if (!inStr && ch === "#") { commentStart = ci; break; }
          }
        } else if (["javascript","js","typescript","ts","tsx","jsx","java","cpp","c","csharp","cs","go","rust","swift","kotlin","php"].includes(l)) {
          let inStr = false; let strChar = "";
          for (let ci = 0; ci < line.length - 1; ci++) {
            const ch = line[ci]; const ch2 = line[ci+1];
            if (!inStr && (ch === '"' || ch === "'" || ch === "`")) { inStr = true; strChar = ch; }
            else if (inStr && ch === strChar && line[ci-1] !== "\\") { inStr = false; }
            else if (!inStr && ch === "/" && ch2 === "/") { commentStart = ci; break; }
          }
        } else if (l === "sql") {
          const idx = line.indexOf("--");
          if (idx !== -1) commentStart = idx;
        }

        if (commentStart !== -1) {
          result.push(...tokenizeLine(line.slice(0, commentStart), l, li));
          result.push(cs(C.comment, line.slice(commentStart), `icm-${li}`, true));
        } else {
          result.push(...tokenizeLine(line, l, li));
        }
      });

      return result;
    }

    function tokenizeLine(line: string, _l: string, li: number): React.ReactNode[] {
      const TOKEN_RE = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d+\.?\d*([eE][+-]?\d+)?\b)|([\w$]+)|(=>|->|::|[=!<>]=|&&|\|\||[+\-*/%&|^~<>!?:@,;.[\](){}])/g;
      const result: React.ReactNode[] = [];
      let lastIdx = 0;
      let m: RegExpExecArray | null;
      TOKEN_RE.lastIndex = 0;

      const controlWords = new Set(["if","else","elif","for","while","do","switch","case","break","continue","return","yield","pass","try","except","finally","catch","throw","raise","with","async","await","in","of","is","not","and","or","new","delete","typeof","instanceof","void","import","export","from","as","default","match","goto","using","include","require"]);
      const typeWords    = new Set(["int","str","float","bool","list","dict","tuple","set","bytes","any","Any","Optional","Union","List","Dict","Tuple","Set","Type","Callable","string","number","boolean","object","array","void","never","unknown","bigint","symbol","char","double","long","short","byte","uint","i32","i64","u32","u64","f32","f64","usize","isize","Vec","HashMap","Option","Result","Box","Arc","Rc","String","integer","real"]);
      const builtinWords = new Set(["print","len","range","type","isinstance","hasattr","getattr","setattr","enumerate","zip","map","filter","sorted","reversed","sum","min","max","abs","round","open","input","repr","super","property","staticmethod","classmethod","console","Math","JSON","Object","Array","Promise","Error","Date","RegExp","Symbol","Buffer","process","setTimeout","setInterval","clearTimeout","clearInterval","fetch","document","window","navigator"]);
      const declWords    = new Set(["class","def","fn","func","fun","pub","priv","mod","use","type","struct","impl","trait","interface","enum","abstract","override","extends","implements","let","const","var","declare","namespace","module","package","where","self","this","super"]);
      const literalWords = new Set(["None","True","False","null","undefined","true","false","NaN","Infinity"]);

      const col = (color: string, content: string, key: string, bold = false) => (
        <span key={key} style={{ color, ...(bold ? { fontWeight: 600 } : {}) }}>{content}</span>
      );

      while ((m = TOKEN_RE.exec(line)) !== null) {
        if (m.index > lastIdx) result.push(<span key={`t-${li}-${lastIdx}`} style={{ color: C.base }}>{line.slice(lastIdx, m.index)}</span>);
        lastIdx = m.index + m[0].length;
        const [full, strTok, numTok, , wordTok, opTok] = m;

        if (strTok !== undefined) {
          result.push(col(C.string, full, `s-${li}-${m.index}`));
        } else if (numTok !== undefined) {
          result.push(col(C.number, full, `n-${li}-${m.index}`));
        } else if (wordTok !== undefined) {
          if (controlWords.has(wordTok))       result.push(col(C.kw_ctrl,  full, `kc-${li}-${m.index}`, true));
          else if (declWords.has(wordTok))     result.push(col(C.kw_decl,  full, `kd-${li}-${m.index}`, true));
          else if (literalWords.has(wordTok))  result.push(col(C.kw_lit,   full, `kl-${li}-${m.index}`));
          else if (typeWords.has(wordTok))     result.push(col(C.type,     full, `kt-${li}-${m.index}`));
          else if (builtinWords.has(wordTok))  result.push(col(C.builtin,  full, `kb-${li}-${m.index}`));
          else if (/^[A-Z]/.test(wordTok))    result.push(col(C.classname, full, `cls-${li}-${m.index}`));
          else if (/^_/.test(wordTok))         result.push(col(C.dunder,   full, `dun-${li}-${m.index}`));
          else                                 result.push(<span key={`id-${li}-${m.index}`} style={{ color: C.base }}>{full}</span>);
        } else if (opTok !== undefined) {
          if (full === "@") {
            result.push(col(C.decorator, full, `at-${li}-${m.index}`));
          } else if (["=>","->","::","=","==","!=","<","<=",">",">=","&&","||","!","+","-","*","/","%","&","|","^","~","?"].includes(full)) {
            result.push(col(C.operator, full, `op-${li}-${m.index}`));
          } else if (["(",")","{","}","[","]"].includes(full)) {
            result.push(col(C.bracket, full, `br-${li}-${m.index}`));
          } else {
            result.push(col(C.punct, full, `pu-${li}-${m.index}`));
          }
        }
      }
      if (lastIdx < line.length) result.push(<span key={`tail-${li}-${lastIdx}`} style={{ color: C.base }}>{line.slice(lastIdx)}</span>);
      return result;
    }

    const HIGHLIGHTED_LANGS = new Set([
      "python","py","javascript","js","typescript","ts","tsx","jsx",
      "bash","sh","sql","css","json","html","xml","rust","go","java",
      "cpp","c","csharp","cs","php","ruby","swift","kotlin","r","yaml","yml",
    ]);
    const shouldHighlight = HIGHLIGHTED_LANGS.has(lang.toLowerCase());

    const [copied, setCopied] = React.useState(false);
    const handleCopy = () => {
      navigator.clipboard.writeText(codeText).catch(() => {});
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    };
    const handleDownload = () => {
      const blob = new Blob([codeText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = downloadFilename;
      a.click();
      URL.revokeObjectURL(url);
    };

    // Distinct bg colours for header vs body — works in both modes
    const headerBg = isDark ? "rgba(15,23,42,0.85)" : "rgba(241,245,249,1)";
    const bodyBg   = isDark ? "rgba(2,6,23,0.92)"   : "rgba(248,250,252,1)";
    const borderCol= isDark ? "rgba(51,65,85,0.6)"  : "rgba(203,213,225,0.8)";

    return (
      <div className="relative my-3 rounded-xl overflow-hidden shadow-md" style={{ border: `1px solid ${borderCol}` }}>
        {/* Header bar */}
        <div className="flex items-center justify-between px-3 py-2" style={{ background: headerBg, borderBottom: `1px solid ${borderCol}` }}>
          <div className="flex items-center gap-2">
            {/* Language accent dot */}
            <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: meta.accent }} />
            {/* Language label */}
            <span className="text-[11px] font-semibold font-mono tracking-wide" style={{ color: meta.accent }}>
              {meta.label}
            </span>
            {/* Inferred filename — dim, separated by a divider */}
            <span style={{ color: isDark ? "#475569" : "#94a3b8" }} className="text-[10px] select-none">·</span>
            <span className="text-[10px] font-mono" style={{ color: isDark ? "#64748b" : "#94a3b8" }}>{downloadFilename}</span>
          </div>
          <div className="flex items-center gap-0.5">
            <button
              onClick={handleDownload}
              title={`Download ${downloadFilename}`}
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-md transition-colors font-mono"
              style={{ color: isDark ? "#64748b" : "#94a3b8" }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = isDark ? "#e2e8f0" : "#1e293b"; (e.currentTarget as HTMLElement).style.background = isDark ? "rgba(51,65,85,0.5)" : "rgba(203,213,225,0.5)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = isDark ? "#64748b" : "#94a3b8"; (e.currentTarget as HTMLElement).style.background = ""; }}
            >
              <svg width="10" height="10" viewBox="0 0 12 12" fill="none" className="shrink-0">
                <path d="M6 1v7M3 5l3 3 3-3M1 10h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Download
            </button>
            <button
              onClick={handleCopy}
              title="Copy code"
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-md transition-colors font-mono"
              style={{
                color: copied ? meta.accent : isDark ? "#64748b" : "#94a3b8",
                background: copied ? `${meta.accent}20` : undefined,
              }}
              onMouseEnter={e => { if (!copied) { (e.currentTarget as HTMLElement).style.color = isDark ? "#e2e8f0" : "#1e293b"; (e.currentTarget as HTMLElement).style.background = isDark ? "rgba(51,65,85,0.5)" : "rgba(203,213,225,0.5)"; }}}
              onMouseLeave={e => { if (!copied) { (e.currentTarget as HTMLElement).style.color = isDark ? "#64748b" : "#94a3b8"; (e.currentTarget as HTMLElement).style.background = ""; }}}
            >
              {copied ? (
                <>
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" className="shrink-0">
                    <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" className="shrink-0">
                    <rect x="4" y="4" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 4V2a1 1 0 00-1-1H2a1 1 0 00-1 1v5a1 1 0 001 1h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  Copy
                </>
              )}
            </button>
          </div>
        </div>
        {/* Code body — distinct from header AND from surrounding chat bg */}
        <pre className="p-4 text-[12px] font-mono overflow-x-auto whitespace-pre leading-[1.7] m-0"
          style={{ background: bodyBg }}>
          <code style={{ color: C.base }}>
            {shouldHighlight ? tokenize(codeText, lang) : codeText}
          </code>
        </pre>
      </div>
    );
  },
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
  // This happens because "[N]." splits into segment="." + segment="next text"
  return segments.map(seg => ({
    ...seg,
    text: seg.text.replace(/^[\s.,;:!?]+/, ''),
  })).filter(seg => seg.text.trim().length > 0);
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
  mediaType?: "pdf" | "image" | "txt" | "cad";  // what kind of viewer to show
  cadSummary?: Record<string, unknown>;           // parsed CAD/IFC summary from upload
  ifcBlobUrl?: string;       // IFC blob URL for CAD→IFC 3D rendering
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
// Xeokit IFC/CAD 3D Viewer
// ---------------------------------------------------------------------------

declare global {
  interface Window {
    pdfjsLib: any;
    XKT_VERSION?: string;
  }
}

interface XeokitViewerProps {
  blobUrl: string;
  filename: string;
  pipeline: string; // 'ifc' | 'cad'
}

// Cache the dynamic-import promise so we only load once
let _xeokitPromise: Promise<{ Viewer: any; WebIFCLoaderPlugin: any; WebIFC: any }> | null = null;

function loadXeokit(): Promise<{ Viewer: any; WebIFCLoaderPlugin: any; WebIFC: any }> {
  if (_xeokitPromise) return _xeokitPromise;
  _xeokitPromise = (async () => {
    // Load both in parallel — xeokit ES module + web-ifc ES module
    const [mod, WebIFC] = await Promise.all([
      import(/* @vite-ignore */ "https://esm.sh/@xeokit/xeokit-sdk@2.6.1"),
      import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/web-ifc-api.js"),
    ]);
    if (!mod.Viewer)             throw new Error("xeokit: Viewer not exported");
    if (!mod.WebIFCLoaderPlugin) throw new Error("xeokit: WebIFCLoaderPlugin not exported");
    return { Viewer: mod.Viewer, WebIFCLoaderPlugin: mod.WebIFCLoaderPlugin, WebIFC };
  })();
  _xeokitPromise.catch(() => { _xeokitPromise = null; });
  return _xeokitPromise;
}

function XeokitViewer({ blobUrl, filename, pipeline }: XeokitViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errMsg, setErrMsg] = useState("");

  useEffect(() => {
    let destroyed = false;

    async function init() {
      try {
        setStatus("loading");

        // 1. Load xeokit via dynamic ES import (reliable, no UMD script-tag issues)
        const { Viewer, WebIFCLoaderPlugin, WebIFC } = await loadXeokit();

        if (destroyed) return;
        if (!canvasRef.current) throw new Error("Canvas ref is null");

        // 2. Create viewer
        const viewer = new Viewer({
          canvasElement: canvasRef.current,
          transparent: true,
        });
        viewerRef.current = viewer;

        viewer.camera.eye  = [10, 10, 10];
        viewer.camera.look = [0, 0, 0];
        viewer.camera.up   = [0, 1, 0];

        // 3. Load model based on type
        const ext = filename.split(".").pop()?.toLowerCase() ?? "";
        const isIfc = ["ifc", "ifczip"].includes(ext);

        if (isIfc) {
          // wasmPath: web-ifc WASM files fetched at runtime by the plugin
          // web-ifc must be loaded as a side-effect import (it exposes a global via its own wasm init)
          // WebIFC is the namespace module, IfcAPI must be instantiated + Init()d before passing in
          const ifcAPI = new WebIFC.IfcAPI();
          ifcAPI.SetWasmPath("https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/");
          await ifcAPI.Init();

          if (destroyed) return;

          const loader = new WebIFCLoaderPlugin(viewer, {
            WebIFC,
            IfcAPI: ifcAPI,
          });
          const model = loader.load({ id: "model", src: blobUrl, edges: true, performance: true });
          model.on("loaded", () => {
            if (destroyed) return;
            viewer.cameraFlight.flyTo(model);
            setStatus("ready");
          });
          model.on("error", (e: any) => {
            if (!destroyed) { setErrMsg(typeof e === "string" ? e : (e?.message ?? "Model load error")); setStatus("error"); }
          });
        } else if (["dxf"].includes(ext)) {
          // DXF: parse and draw 2D geometry on canvas using a lightweight approach
          if (destroyed) return;
          const resp = await fetch(blobUrl);
          const text = await resp.text();
          if (destroyed) return;

          // Parse LINE entities from DXF text
          const lines: {x1:number,y1:number,x2:number,y2:number}[] = [];
          const entityRe = /^\s*0\s*\nLINE([\s\S]*?)(?=\n\s*0\s*\n)/gm;
          const getVal = (block: string, code: number) => {
            const m = new RegExp(`\n\s*${code}\s*\n\s*([\d.+\-eE]+)`).exec(block);
            return m ? parseFloat(m[1]) : 0;
          };
          let em: RegExpExecArray | null;
          while ((em = entityRe.exec(text)) !== null) {
            lines.push({ x1: getVal(em[1], 10), y1: getVal(em[1], 20), x2: getVal(em[1], 11), y2: getVal(em[1], 21) });
          }

          if (destroyed) return;
          const canvas = canvasRef.current!;
          const ctx = canvas.getContext("2d")!;
          canvas.width  = canvas.offsetWidth  || 600;
          canvas.height = canvas.offsetHeight || 320;

          if (lines.length === 0) {
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#6b7280";
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("DXF loaded — no LINE entities to preview", canvas.width/2, canvas.height/2);
            ctx.fillText("AI analysis works via chat below", canvas.width/2, canvas.height/2 + 20);
            setStatus("ready");
            return;
          }

          // Fit-to-view
          const minX = Math.min(...lines.map(l=>Math.min(l.x1,l.x2)));
          const maxX = Math.max(...lines.map(l=>Math.max(l.x1,l.x2)));
          const minY = Math.min(...lines.map(l=>Math.min(l.y1,l.y2)));
          const maxY = Math.max(...lines.map(l=>Math.max(l.y1,l.y2)));
          const pad = 24;
          const scaleX = (canvas.width  - pad*2) / (maxX - minX || 1);
          const scaleY = (canvas.height - pad*2) / (maxY - minY || 1);
          const scale  = Math.min(scaleX, scaleY);
          const tx = (x: number) => pad + (x - minX) * scale;
          const ty = (y: number) => canvas.height - pad - (y - minY) * scale;

          ctx.fillStyle = "#0f0f1a";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = "#60a5fa";
          ctx.lineWidth = 0.7;
          ctx.globalAlpha = 0.8;
          ctx.beginPath();
          for (const l of lines) {
            ctx.moveTo(tx(l.x1), ty(l.y1));
            ctx.lineTo(tx(l.x2), ty(l.y2));
          }
          ctx.stroke();
          if (!destroyed) setStatus("ready");
        } else {
          throw new Error(`.${ext} files: convert to IFC for 3D view, or DXF for 2D preview. AI analysis works via chat.`);
        }
      } catch (e: any) {
        if (!destroyed) { setErrMsg(e?.message ?? String(e)); setStatus("error"); }
      }
    }

    init();

    return () => {
      destroyed = true;
      if (viewerRef.current) {
        try { viewerRef.current.destroy(); } catch {}
        viewerRef.current = null;
      }
    };
  }, [blobUrl, filename]);

  // Block page scroll when pointer is inside the 3D canvas
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const block = (e: WheelEvent) => e.preventDefault();
    el.addEventListener("wheel", block, { passive: false });
    return () => el.removeEventListener("wheel", block);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full bg-black/80 rounded-lg overflow-hidden" style={{ height: 320 }}>
      <canvas
        ref={canvasRef}
        className="w-full h-full block"
        style={{ touchAction: "none" }}
      />
      {status === "loading" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <p className="text-xs text-white/70">Loading 3D model…</p>
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/70 p-4 text-center">
          <AlertCircle className="h-6 w-6 text-destructive" />
          <p className="text-xs text-white/80">{errMsg || "3D viewer unavailable"}</p>
          <p className="text-[10px] text-white/40">AI analysis still works via chat ↓</p>
        </div>
      )}
      {status === "ready" && (
        <div className="absolute bottom-2 right-2 flex gap-1">
          <span className="text-[9px] text-white/30 bg-black/40 px-1.5 py-0.5 rounded">
            drag · scroll · right-drag
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// PDF.js viewer with highlight support
// ---------------------------------------------------------------------------

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
        {state.mediaType === 'image' ? <ImageIcon className="h-4 w-4 text-primary shrink-0" /> : state.mediaType === 'cad' ? <FileText className="h-4 w-4 text-primary shrink-0" /> : <ScrollText className="h-4 w-4 text-primary shrink-0" />}
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
          state.mediaType === 'cad' ? (
            <div className="p-4 space-y-4 overflow-y-auto h-full">
              {/* ── Xeokit 3D Viewer ── */}
              {state.cadSummary && (
                (() => {
                  const ps = state.cadSummary as any;
                  const pl = (ps.pipeline ?? '').toLowerCase();
                  const fn = (state.doc?.filename ?? ps.filename ?? 'model.ifc');
                  const ext = fn.split('.').pop()?.toLowerCase() ?? '';

                  // Prefer converted IFC blob for 3D; fall back to native IFC blob
                  const viewerBlobUrl = state.ifcBlobUrl ?? (pl === 'ifc' ? state.blobUrl : null);
                  const viewerFilename = state.ifcBlobUrl
                    ? fn.replace(/\.(dxf|dwg|step|stp|rvt|nwd|nwc|dgn|skp|fbx|obj|stl|sat|iges|igs|prt|sldprt|catpart|3ds|dae|rfa|rte)$/i, '.ifc') || 'model.ifc'
                    : fn;

                  if (viewerBlobUrl) {
                    return (
                      <XeokitViewer
                        blobUrl={viewerBlobUrl}
                        filename={viewerFilename}
                        pipeline={pl}
                      />
                    );
                  }

                  // DXF without converted IFC — still show canvas 2D via original blob
                  if (ext === 'dxf' && state.blobUrl) {
                    return (
                      <XeokitViewer
                        blobUrl={state.blobUrl}
                        filename={fn}
                        pipeline={pl}
                      />
                    );
                  }

                  // No 3D available — show info panel
                  return (
                    <div className="w-full rounded-lg bg-muted/30 border border-border flex flex-col items-center justify-center gap-2 p-6 text-center" style={{ height: 200 }}>
                      <svg className="h-8 w-8 text-muted-foreground/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" /></svg>
                      <p className="text-xs text-muted-foreground/60">3D preview not available for this format</p>
                      <p className="text-[10px] text-muted-foreground/40">AI analysis works via chat ↓</p>
                    </div>
                  );
                })()
              )}
              {/* CAD/IFC Summary Panel */}
              {state.cadSummary ? (() => {
                const s = state.cadSummary as any;
                const pipeline = (s.pipeline ?? 'cad').toUpperCase();
                const isIfc = pipeline === 'IFC';
                return (
                  <div className="space-y-3">
                    {/* Header badge */}
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-primary/15 text-primary border border-primary/30">
                        {pipeline}
                      </span>
                      {s.schema && <span className="text-[10px] text-muted-foreground">Schema: {s.schema}</span>}
                      {s.dimension && <span className="text-[10px] text-muted-foreground">{s.dimension}</span>}
                    </div>

                    {/* UX hint */}
                    {s.ux_hint && (
                      <div className="bg-primary/8 border border-primary/20 rounded-lg px-3 py-2">
                        <p className="text-xs text-primary/80 leading-relaxed">{s.ux_hint}</p>
                      </div>
                    )}

                    {/* Parse errors */}
                    {s.parse_errors && s.parse_errors.length > 0 && (
                      <div className="bg-destructive/8 border border-destructive/20 rounded-lg px-3 py-2">
                        <p className="text-[10px] font-semibold text-destructive uppercase tracking-wide mb-1">Parse Warnings</p>
                        {s.parse_errors.map((e: string, i: number) => (
                          <p key={i} className="text-xs text-destructive/80">{e}</p>
                        ))}
                      </div>
                    )}

                    {/* Totals */}
                    <div className="grid grid-cols-2 gap-2">
                      {s.total_elements != null && (
                        <div className="bg-muted/50 rounded-lg p-3">
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Elements</p>
                          <p className="text-xl font-bold text-foreground">{s.total_elements.toLocaleString()}</p>
                        </div>
                      )}
                      {s.total_entities != null && (
                        <div className="bg-muted/50 rounded-lg p-3">
                          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Entities</p>
                          <p className="text-xl font-bold text-foreground">{s.total_entities.toLocaleString()}</p>
                        </div>
                      )}
                    </div>

                    {/* Storeys */}
                    {isIfc && s.storeys && s.storeys.length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Building Storeys</p>
                        <div className="space-y-1">
                          {s.storeys.map((st: any, i: number) => (
                            <div key={i} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1.5 text-xs">
                              <span className="text-foreground font-medium">{st.name ?? `Storey ${i+1}`}</span>
                              {st.elevation != null && <span className="text-muted-foreground">{typeof st.elevation === 'number' ? st.elevation.toFixed(2) : st.elevation} m</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Element counts */}
                    {s.element_counts && Object.keys(s.element_counts).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Element Types</p>
                        <div className="space-y-1">
                          {Object.entries(s.element_counts as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .map(([label, count]) => (
                              <div key={label} className="flex items-center gap-2">
                                <div className="flex-1 flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                  <span className="text-foreground capitalize">{label}</span>
                                  <span className="text-primary font-semibold">{count}</span>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {/* Entity counts (CAD) */}
                    {s.entity_counts && Object.keys(s.entity_counts).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Entity Types</p>
                        <div className="space-y-1">
                          {Object.entries(s.entity_counts as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .slice(0, 15)
                            .map(([label, count]) => (
                              <div key={label} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                <span className="text-foreground">{label}</span>
                                <span className="text-primary font-semibold">{count}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {/* Layers (CAD/DXF) */}
                    {s.layers && s.layers.length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Layers ({s.layers.length})</p>
                        <div className="space-y-1 max-h-48 overflow-y-auto">
                          {s.layers.map((layer: any, i: number) => (
                            <div key={i} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                              <span className="text-foreground font-mono">{layer.name ?? layer}</span>
                              {layer.entity_count != null && <span className="text-muted-foreground">{layer.entity_count} entities</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Material inventory */}
                    {s.material_inventory && Object.keys(s.material_inventory).length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">Materials</p>
                        <div className="space-y-1">
                          {Object.entries(s.material_inventory as Record<string, number>)
                            .sort(([,a],[,b]) => b - a)
                            .map(([mat, count]) => (
                              <div key={mat} className="flex items-center justify-between bg-muted/40 rounded px-2 py-1 text-xs">
                                <span className="text-foreground">{mat}</span>
                                <span className="text-primary font-semibold">{count}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })() : (
                <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
                  <FileText className="h-10 w-10 text-muted-foreground/40" />
                  <p className="text-sm text-muted-foreground">No summary available for this file.</p>
                </div>
              )}
            </div>
          ) : state.mediaType === 'image' ? (
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

// ---------------------------------------------------------------------------
// Word-completion engine — Directed Acyclic Word Graph (DAWG / trie)
// Mirrors the fast-autocomplete DWG approach, zero dependencies.
// Vocabulary is built dynamically from message history + document names.
// ---------------------------------------------------------------------------

interface TrieNode {
  children: Map<string, TrieNode>;
  freq: number;      // times this exact word was inserted
  isEnd: boolean;
}

function makeTrie(): TrieNode {
  return { children: new Map(), freq: 0, isEnd: false };
}

function trieInsert(root: TrieNode, word: string, weight = 1) {
  let node = root;
  for (const ch of word) {
    if (!node.children.has(ch)) node.children.set(ch, makeTrie());
    node = node.children.get(ch)!;
  }
  node.isEnd = true;
  node.freq += weight;
}

/** Walk to the node matching `prefix`, or null if not found. */
function trieFind(root: TrieNode, prefix: string): TrieNode | null {
  let node = root;
  for (const ch of prefix) {
    if (!node.children.has(ch)) return null;
    node = node.children.get(ch)!;
  }
  return node;
}

/** Collect up to `limit` completions from a subtree, sorted by freq desc. */
function trieCollect(node: TrieNode, prefix: string, limit: number): { word: string; freq: number }[] {
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

/** Tokenise any text into lowercase words ≥3 chars, stripping markdown/punctuation. */
function tokenise(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/```[\s\S]*?```/g, "")   // strip code blocks
    .replace(/[^a-z0-9'\s-]/g, " ")
    .split(/\s+/)
    .filter(w => w.length >= 3 && !/^\d+$/.test(w));
}

/**
 * Given what the user has typed so far, scan all user messages for a sentence
 * that starts with the same prefix.  Returns the remainder of that sentence
 * (the ghost-text suffix), or "" if nothing matches.
 *
 * Strategy:
 *  1. Split every user message into sentences.
 *  2. Normalise each sentence to lowercase, collapsed whitespace.
 *  3. If a sentence starts with `typed` (and is strictly longer), return the
 *     suffix — i.e. the part the user hasn't typed yet.
 *  4. Prefer shorter suffixes (less presumptuous) and deduplicate.
 */
/**
 * AI-powered inline ghost-text completion.
 * Calls the backend /autocomplete endpoint (backed by the CF LLM worker)
 * and returns the suggested suffix to ghost after the cursor.
 *
 * Debouncing is handled by the caller — this just does the fetch.
 * Returns "" on any failure so the UI degrades gracefully.
 */
async function fetchAICompletion(
  typed: string,
  recentMessages: { role: string; content: string }[],
  apiBase: string,
  availableDocs: string[] = [],
): Promise<string> {
  if (typed.trim().length < 4) return ""; // guard on trimmed length, but send raw `typed` to backend

  // Use only the last assistant reply as context — keeps the prompt tiny for speed
  const lastAssistant = [...recentMessages].reverse().find(m => m.role === "assistant");
  const sessionContext = lastAssistant ? lastAssistant.content.slice(0, 300) : "";

  try {
    const res = await fetch(`${apiBase}/autocomplete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        partial:         typed,
        session_context: sessionContext,
        available_docs:  availableDocs.slice(0, 8),
        max_tokens:      28,
      }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return "";
    const data = await res.json();
    const completion = data.completion ?? "";
    if (!completion || completion.trim().length === 0 || completion.length > 80) return "";
    return completion; // preserve leading space — backend sets it intentionally
  } catch {
    return "";
  }
}

// Fetches and displays a Wikipedia page thumbnail via the public REST API
const WikiThumbnail: React.FC<{ wikiUrl: string }> = ({ wikiUrl }) => {
  const [imgSrc, setImgSrc] = React.useState<string | null>(null);

  React.useEffect(() => {
    // Extract the page title from the URL e.g. https://en.wikipedia.org/wiki/Photosynthesis
    const match = wikiUrl.match(/\/wiki\/([^#?]+)/);
    if (!match) return;
    const title = decodeURIComponent(match[1]);
    // Detect language subdomain e.g. fr.wikipedia.org
    const langMatch = wikiUrl.match(/^https?:\/\/([a-z]{2})\.wikipedia/);
    const lang = langMatch ? langMatch[1] : "en";

    fetch(
      `https://${lang}.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`
    )
      .then(r => r.json())
      .then(data => {
        const url = data?.thumbnail?.source || data?.originalimage?.source;
        if (url) setImgSrc(url);
      })
      .catch(() => {});
  }, [wikiUrl]);

  if (!imgSrc) return null;
  return (
    <img
      src={imgSrc}
      alt=""
      className="h-16 w-16 object-cover rounded-lg shrink-0 opacity-90"
    />
  );
};

// ---------------------------------------------------------------------------
// Voice Message Bubble
// ---------------------------------------------------------------------------

interface VoiceMessageBubbleProps {
  blobUrl: string;
  duration: number;
  transcript: string | undefined;
  isExpanded: boolean;
  onToggleTranscript: () => void;
  waveform?: number[];
  timestamp: Date;
}

const VoiceMessageBubble: React.FC<VoiceMessageBubbleProps> = ({
  blobUrl,
  duration,
  transcript,
  isExpanded,
  onToggleTranscript,
  waveform,
  timestamp,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [speed, setSpeed] = useState<0.5 | 1 | 2 | 4>(1);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const animFrameRef = useRef<number>(0);

  const SPEEDS: Array<0.5 | 1 | 2 | 4> = [0.5, 1, 2, 4];
  const cycleSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length] as 0.5 | 1 | 2 | 4;
    setSpeed(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  // Build the bubble waveform from captured samples.
  // We resample to exactly BAR_COUNT bars regardless of how many samples were captured.
  const BAR_COUNT = 52;
  const bubbleBars = React.useMemo(() => {
    if (!waveform || waveform.length === 0) {
      // Fallback: gentle organic shape so it never looks empty
      return Array.from({ length: BAR_COUNT }, (_, i) => {
        const t = i / (BAR_COUNT - 1);
        return 0.15 + 0.45 * Math.sin(t * Math.PI) + 0.1 * Math.sin(t * Math.PI * 4);
      });
    }
    // Resample: group source samples into BAR_COUNT buckets, take the peak in each
    return Array.from({ length: BAR_COUNT }, (_, i) => {
      const start = Math.floor((i / BAR_COUNT) * waveform.length);
      const end   = Math.ceil(((i + 1) / BAR_COUNT) * waveform.length);
      const slice = waveform.slice(start, end);
      return slice.length > 0 ? Math.max(...slice) : 0;
    });
  }, [waveform]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(() => {});
    }
  };

  useEffect(() => {
    const audio = new Audio(blobUrl);
    audio.playbackRate = speed;
    audioRef.current = audio;

    const tick = () => {
      if (!audioRef.current) return;
      const dur = audioRef.current.duration || duration;
      const ct  = audioRef.current.currentTime;
      setCurrentTime(ct);
      setProgress(dur > 0 ? ct / dur : 0);
      animFrameRef.current = requestAnimationFrame(tick);
    };

    const onPlay  = () => { setIsPlaying(true); tick(); };
    const onPause = () => { setIsPlaying(false); cancelAnimationFrame(animFrameRef.current); };
    const onEnded = () => { setIsPlaying(false); setProgress(0); setCurrentTime(0); cancelAnimationFrame(animFrameRef.current); };

    audio.addEventListener("play",  onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      audio.removeEventListener("play",  onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.pause();
      audio.src = "";
    };
  }, [blobUrl]);

  const isPending = transcript === undefined;

  // ── Waveform scrubber (click + drag to seek) ──────────────────────────────
  const waveformRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  const seekTo = (clientX: number) => {
    const el = waveformRef.current;
    const audio = audioRef.current;
    if (!el || !audio) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const dur = audio.duration || duration;
    if (isFinite(dur) && dur > 0) {
      audio.currentTime = frac * dur;
      setProgress(frac);
      setCurrentTime(frac * dur);
    }
  };

  const handleWaveMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    seekTo(e.clientX);
    const onMove = (ev: MouseEvent) => { if (isDraggingRef.current) seekTo(ev.clientX); };
    const onUp   = () => { isDraggingRef.current = false; window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return (
    <div className="flex flex-col items-end gap-1 w-fit">
      {/* Transcript — above the bubble */}
      <AnimatePresence>
        {isExpanded && transcript && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden w-full"
          >
            <p className="text-[11px] text-muted-foreground/60 leading-relaxed pl-3 border-l border-border/25 italic text-right">
              {transcript}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transcript toggle — above bubble, right-aligned */}
      <button
        onClick={onToggleTranscript}
        className="flex items-center gap-1.5 group/transcript w-fit pr-1"
        disabled={isPending}
      >
        {isPending ? (
          <span className="text-[11px] text-muted-foreground/40 italic leading-none flex items-center gap-1">
            <Loader2 className="h-2.5 w-2.5 animate-spin" />
            Transcribing…
          </span>
        ) : (
          <span className="text-[11px] text-muted-foreground/40 italic leading-none">
            {isExpanded ? "Hide transcript" : "Show transcript"}
          </span>
        )}
        {!isPending && (
          <motion.span
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="opacity-0 group-hover/transcript:opacity-60 transition-opacity"
          >
            <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
          </motion.span>
        )}
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-muted-foreground/30 shrink-0" />
      </button>

      {/* Voice bubble */}
      <div className="bg-primary text-primary-foreground rounded-2xl rounded-br-md px-3 py-2.5 flex items-center gap-2.5 w-fit">
        {/* Play / pause button */}
        <button
          onClick={togglePlay}
          className="h-8 w-8 rounded-full bg-primary-foreground/20 hover:bg-primary-foreground/30 flex items-center justify-center shrink-0 transition-colors"
        >
          {isPlaying ? (
            <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor">
              <rect x="0" y="0" width="4" height="14" rx="1.5" />
              <rect x="8" y="0" width="4" height="14" rx="1.5" />
            </svg>
          ) : (
            <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor">
              <path d="M1 1.5v11l10-5.5L1 1.5z" />
            </svg>
          )}
        </button>

        {/* Waveform scrubber + time */}
        <div className="flex flex-col gap-1" style={{ width: "160px" }}>
          {/* Clickable / draggable waveform */}
          <div
            ref={waveformRef}
            onMouseDown={handleWaveMouseDown}
            className="flex items-center gap-[1.5px] h-[22px] cursor-pointer select-none"
            title="Click or drag to seek"
          >
            {bubbleBars.map((h, i) => {
              const frac = i / (bubbleBars.length - 1);
              const isPast = frac <= progress;
              const minH = 2;
              const maxH = 20;
              const barH = minH + h * (maxH - minH);
              return (
                <div
                  key={i}
                  style={{
                    height: `${barH}px`,
                    flex: "1 1 0",
                    borderRadius: "1px",
                    background: isPast ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.32)",
                    transition: "background 0.06s ease",
                  }}
                />
              );
            })}
          </div>
          {/* Time display */}
          <span className="text-[10px] text-primary-foreground/55 tabular-nums">
            {isPlaying ? formatTime(currentTime) : formatTime(duration)}
          </span>
        </div>

        {/* Speed toggle button */}
        <button
          onClick={cycleSpeed}
          className="shrink-0 h-7 min-w-[32px] px-1 rounded-md bg-primary-foreground/15 hover:bg-primary-foreground/25 transition-colors flex items-center justify-center"
          title="Change playback speed"
        >
          <span className="text-[11px] font-semibold text-primary-foreground/80 leading-none tabular-nums">
            {speed === 0.5 ? "×½" : speed === 1 ? "×1" : speed === 2 ? "×2" : "×4"}
          </span>
        </button>
      </div>

      {/* Timestamp — sits directly under the blue pill */}
      <span className="text-[10px] text-muted-foreground/50 tabular-nums">
        {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </span>
    </div>
  );
};


// ---------------------------------------------------------------------------
// ChartMessage + ChartClarification — graph_node response renderers
// ---------------------------------------------------------------------------

interface ChartGroup {
  label:        string;
  description:  string;
  hint:         string;
  source_file?: string;
}

interface ChartAnalytics {
  type: "chart_config" | "chart_clarification" | "report_chart_clarification" | "chart_error";
  // chart_config
  chart_js?:            Record<string, any>;
  chart_type?:          string;
  title?:               string;
  description?:         string;
  interpretation?:      string;
  sources?:             string[];
  // chart_clarification
  question?:            string;
  groups?:              ChartGroup[];
  clarification_mode?:  "file" | "metric";
  // chart_error
  message?:             string;
}

interface ChartMessageProps {
  analytics: ChartAnalytics;
  answer:    string;
}

// ── Chart insights renderer ──────────────────────────────────────────────────
/**
 * Splits the plain-prose interpretation into individual sentences and renders
 * them as labelled insight rows: "Observation", "Trend", "Implication".
 * Falls back gracefully if the text can't be split.
 */
const INSIGHT_LABELS = ["Observation", "Trend", "Implication", "Note"];
const INSIGHT_ICONS  = [
  // Bar-chart icon
  <svg key="obs" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
  // Trending-up icon
  <svg key="trend" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>,
  // Lightbulb icon
  <svg key="impl" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.347.347a3.252 3.252 0 01-4.78 0l-.347-.347z" /></svg>,
  // Info icon
  <svg key="note" className="h-3.5 w-3.5 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
];

function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by a space + capital letter
  // Handles abbreviations poorly, but good enough for 2-4 sentence interpretations
  const raw = text
    .replace(/([.!?])\s+(?=[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜ"'])/g, "$1\n")
    .split("\n")
    .map(s => s.trim())
    .filter(s => s.length > 10);
  return raw.length >= 2 ? raw : [text]; // fallback: treat whole text as one block
}

const ChartInsights: React.FC<{ text: string }> = ({ text }) => {
  const sentences = splitIntoSentences(text);
  const isSingle  = sentences.length === 1;

  return (
    <div className="border-t border-primary/10 px-3 pt-3 pb-3 flex flex-col gap-2.5">
      {isSingle ? (
        // Graceful fallback — single block with no label
        <p className="text-[12px] text-muted-foreground leading-relaxed">{text}</p>
      ) : (
        sentences.map((sentence, idx) => (
          <div key={idx} className="flex gap-2.5 items-start">
            {/* Icon */}
            <span className="text-primary/60 mt-0.5">
              {INSIGHT_ICONS[idx % INSIGHT_ICONS.length]}
            </span>
            {/* Label + text */}
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

// ── Color palettes ────────────────────────────────────────────────────────────
// Gradients are created at draw-time inside a Chart.js plugin using the chart's
// own canvas context. This is the ONLY correct approach — gradients built from a
// different canvas (e.g. an off-screen one) are canvas-bound and corrupt when
// used in another context or serialized to PNG.
const STROKES_DARK  = ["#818cf8","#34d399","#fbbf24","#fb7185","#60a5fa","#f472b4","#a78bfa","#2dd4bf"];
const STROKES_LIGHT = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#ec4899","#8b5cf6","#14b8a6"];

// Gradient stop pairs [top, bottom] per palette
// Top stop is near-fully opaque; bottom stop is used at ~80% of height,
// then the plugin adds a hard rgba(0,0,0,0) stop at 1.0 for a clean fade.
const GRAD_STOPS_DARK: [string, string][] = [
  ["rgba(129,140,248,0.95)", "rgba(99,102,241,0.22)"],
  ["rgba(52,211,153,0.95)",  "rgba(16,185,129,0.22)"],
  ["rgba(251,191,36,0.95)",  "rgba(245,158,11,0.22)"],
  ["rgba(251,113,133,0.95)", "rgba(239,68,68,0.22)"],
  ["rgba(96,165,250,0.95)",  "rgba(59,130,246,0.22)"],
  ["rgba(244,114,182,0.95)", "rgba(236,72,153,0.22)"],
  ["rgba(167,139,250,0.95)", "rgba(139,92,246,0.22)"],
  ["rgba(45,212,191,0.95)",  "rgba(20,184,166,0.22)"],
];
const GRAD_STOPS_LIGHT: [string, string][] = [
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
function makeGradientPlugin(rawCfg: Record<string, any>, isDark: boolean) {
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

        // ── Radar — gradient fill ────────────────────────────────────────────
        if (isRadar) {
          const grad = ctx.createLinearGradient(0, top, 0, bottom);
          grad.addColorStop(0,   gradStops[si][0]);
          grad.addColorStop(1,   "rgba(0,0,0,0)");
          ds.backgroundColor = grad;
          ds.borderColor     = strokes[si];
          ds.borderWidth     = 2;
          ds.pointBackgroundColor = strokes[si];
          return;
        }

        // ── Horizontal Bar — left (opaque) → right (transparent) ────────────
        if (isHBar) {
          if (nDatasets === 1) {
            const nBars = (chart.data.labels ?? []).length;
            ds.backgroundColor = Array.from({ length: nBars }, (_: any, j: number) => {
              const g = ctx.createLinearGradient(left, 0, right, 0);
              g.addColorStop(0,   gradStops[j % gradStops.length][0]);
              g.addColorStop(1,   "rgba(0,0,0,0)");
              return g;
            });
          } else {
            const g = ctx.createLinearGradient(left, 0, right, 0);
            g.addColorStop(0,   gradStops[si][0]);
            g.addColorStop(1,   "rgba(0,0,0,0)");
            ds.backgroundColor = g;
          }
          ds.borderColor   = "transparent";
          ds.borderWidth   = 0;
          ds.borderRadius  = nDatasets === 1 ? 7 : 6;
          ds.borderSkipped = false;
          return;
        }

        // ── Vertical Bar (default) — top (opaque) → bottom (transparent) ────
        if (nDatasets === 1) {
          const nBars = (chart.data.labels ?? []).length;
          ds.backgroundColor = Array.from({ length: nBars }, (_: any, j: number) => {
            const g = ctx.createLinearGradient(0, top, 0, bottom);
            g.addColorStop(0,   gradStops[j % gradStops.length][0]);
            g.addColorStop(0.8, gradStops[j % gradStops.length][1]);
            g.addColorStop(1,   "rgba(0,0,0,0)");
            return g;
          });
        } else {
          const g = ctx.createLinearGradient(0, top, 0, bottom);
          g.addColorStop(0,   gradStops[si][0]);
          g.addColorStop(0.8, gradStops[si][1]);
          g.addColorStop(1,   "rgba(0,0,0,0)");
          ds.backgroundColor = g;
        }
        ds.borderColor   = "transparent";
        ds.borderWidth   = 0;
        ds.borderRadius  = nDatasets === 1 ? 7 : 6;
        ds.borderSkipped = false;
      });
    },
  };
}

// ── Apply theme colors to an existing live chart via chart.update() ───────────
// Using update() instead of destroy+recreate gives a smooth CSS-like transition.
function applyThemeToChart(chart: any, rawCfg: Record<string, any>, isDark: boolean) {
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
function buildInitialCfg(raw: Record<string, any>, isDark: boolean): Record<string, any> {
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
    try { /* eslint-disable-next-line no-new-func */ cbs.label = new Function("return " + cbs.label)(); } catch { delete cbs.label; }
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
function downloadChartPng(chart: any, filename: string, isDark: boolean) {
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

// ── Chart renderer ──────────────────────────────────────────────────────────
const ChartMessage: React.FC<ChartMessageProps> = ({ analytics, answer }) => {
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const [interpExpanded, setInterpExpanded] = useState(false);

  const getIsDark = () => document.documentElement.classList.contains("dark");

  // Initial build (only when analytics changes — NOT on theme change)
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

  // Smooth theme update — debounced via rAF so every ChartMessage on screen
  // batches into a single paint frame instead of firing one-after-another.
  // Root cause of the 2fps lag: N charts × synchronous update() calls.
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

      {/* ── 1. Narrative RAG answer ── */}
      {answer && (
        <div className="leading-relaxed text-sm text-foreground">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>{answer}</ReactMarkdown>
        </div>
      )}

      {/* ── 2. Chart card ── */}
      <div className="flex flex-col gap-2">
        {/* Title + action row — pure React DOM, so Tailwind theme classes apply automatically */}
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
              title="Download chart as PNG"
              className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted border border-transparent hover:border-border transition-all"
            >
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              PNG
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div className="rounded-xl border border-border bg-card p-4 shadow-inner overflow-hidden">
          <canvas ref={canvasRef} />
          {/* ── React-rendered legend — immune to Chart.js theme bugs ── */}
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

        {/* Key insights — collapsible */}
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

// ── Clarification picker ────────────────────────────────────────────────────
interface ChartClarificationProps {
  analytics: ChartAnalytics;
  onSelect:  (hint: string) => void;
}

const ChartClarification: React.FC<ChartClarificationProps> = ({ analytics, onSelect }) => {
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

  // For file-mode: group pills under their filename headers
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

// ── Welcome splash — shown when no messages yet ──────────────────────────────

const GREETINGS: Record<"night" | "morning" | "afternoon" | "evening", string[]> = {
  night: [
    "Burning the midnight oil?",
    "Still at it this late?",
    "The night is young.",
    "Couldn't sleep either?",
    "Late night deep dive.",
    "Quiet hours, sharp focus.",
    "Night owl mode: on.",
  ],
  morning: [
    "Good morning.",
    "Rise and grind.",
    "Early bird energy.",
    "Morning, let's get to it.",
    "Fresh start, fresh mind.",
    "Coffee ready? Let's go.",
    "New day, new answers.",
    "Morning momentum.",
  ],
  afternoon: [
    "Good afternoon.",
    "Afternoon grind.",
    "Halfway through the day.",
    "Post-lunch clarity.",
    "Keep the momentum going.",
    "Afternoon deep dive.",
    "Still got hours left.",
    "Productive afternoon ahead.",
  ],
  evening: [
    "Good evening.",
    "Wrapping up the day?",
    "Evening wind-down.",
    "End of day, still curious.",
    "Evening focus session.",
    "One more thing to figure out.",
    "Quiet evening, good thinking.",
    "Sunset productivity.",
  ],
};

function getGreeting(): string {
  const h = new Date().getHours();
  const bucket: keyof typeof GREETINGS =
    h < 5 ? "night" : h < 12 ? "morning" : h < 18 ? "afternoon" : "evening";
  const opts = GREETINGS[bucket];
  return opts[Math.floor(Math.random() * opts.length)];
}

// Stable per-mount greeting so it doesn't re-roll on re-render
const SPLASH_GREETING = getGreeting();

const WelcomeSplash: React.FC<{ visible: boolean }> = ({ visible }) => (
  <AnimatePresence>
    {visible && (
      <motion.div
        key="welcome-splash"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0, transition: { duration: 0.2 } }}
        transition={{ duration: 0.3 }}
        className="absolute inset-0 flex flex-col items-center pointer-events-none select-none z-10"
        style={{ justifyContent: "center", paddingBottom: "160px" }}
      >
        {/* Subtle glow */}
        <div className="absolute inset-0 -z-10 flex items-center justify-center">
          <div className="h-48 w-48 rounded-full bg-primary/6 blur-3xl" />
        </div>

        {/* Logo + text inline — no nested motion, no scale */}
        <div className="flex items-center gap-4">
          <Logo className="h-12 w-12 opacity-90" />
          <div className="flex flex-col gap-0.5">
            <h1 className="text-3xl font-semibold tracking-tight text-foreground/90 leading-tight">
              {SPLASH_GREETING}
            </h1>
            <p className="text-sm text-muted-foreground/50 leading-relaxed">
              Upload a document and ask me anything.
            </p>
          </div>
        </div>
      </motion.div>
    )}
  </AnimatePresence>
);

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [pendingDocIds, setPendingDocIds] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  // Keep refs in sync so async callbacks (and loadConversation) always read
  // the latest values — not stale closure captures from a previous render.
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  const messagesRef = useRef<Message[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const activeConvIdRef = useRef<string>("");
  useEffect(() => { activeConvIdRef.current = activeConvId; }, [activeConvId]);
  const [pendingConvIds, setPendingConvIds] = useState<Record<string, boolean>>({});
  const markPendingConversation = useCallback((convId: string) => {
    setPendingConvIds(prev => ({ ...prev, [convId]: true }));
  }, []);
  const clearPendingConversation = useCallback((convId: string) => {
    setPendingConvIds(prev => {
      if (!prev[convId]) return prev;
      const next = { ...prev };
      delete next[convId];
      return next;
    });
  }, []);
  const updateConversationMessages = useCallback(
    (convId: string, updatedMessages: Message[], preview: string, title?: string) => {
      setConversations(prev => {
        const existing = prev.find(c => c.id === convId);
        if (existing) {
          return prev.map(c =>
            c.id === convId
              ? {
                  ...c,
                  messages: updatedMessages,
                  preview: preview || c.preview,
                  timestamp: new Date(),
                  title: title || c.title,
                }
              : c
          );
        }
        return [
          {
            id: convId,
            title: title || "Untitled",
            preview,
            timestamp: new Date(),
            messages: updatedMessages,
            sessionId: sessionIdRef.current,
          } as any,
          ...prev,
        ];
      });
      if (activeConvIdRef.current === convId) {
        setMessages(updatedMessages);
      }
    },
    []
  );
  const [historySearch, setHistorySearch] = useState("");
  const [historySort, setHistorySort] = useState<"newest" | "oldest">("newest");
  const [docsPanelOpen, setDocsPanelOpen] = useState(false);
  const docsPanelOpenRef = useRef(false);
  docsPanelOpenRef.current = docsPanelOpen;
  const [convsPanelOpen, setConvsPanelOpen] = useState(false);
  const convsPanelOpenRef = useRef(false);
  convsPanelOpenRef.current = convsPanelOpen;
  const [reportsPanelOpen, setReportsPanelOpen] = useState(false);
  const reportsPanelOpenRef = useRef(false);
  reportsPanelOpenRef.current = reportsPanelOpen;

  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [activeReport, setActiveReport] = useState<ReportRecord | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isPatchingReport, setIsPatchingReport] = useState(false);
  const [reportEditInstruction, setReportEditInstruction] = useState("");
  const [reportEditMode, setReportEditMode] = useState(false);
  const [downloadingReportId, setDownloadingReportId] = useState<string | null>(null);
  const [deletingReportId, setDeletingReportId] = useState<string | null>(null);
  const reportsPanelRef = useRef<HTMLDivElement>(null);
  // Version history
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [restoringVersion, setRestoringVersion] = useState<number | null>(null);
  // Previewed version content — null means show the live activeReport content
  const [previewedVersion, setPreviewedVersion] = useState<{ version: number; content: string; charts: ChartRecord[]; title: string } | null>(null);
  const reportEditInputRef = useRef<HTMLTextAreaElement>(null);
  const [bubbleDoc, setBubbleDoc] = useState<Document | null>(null);
  const [bubbleViewer, setBubbleViewer] = useState<ViewerState | null>(null);
  const bubbleViewerRef = useRef<ViewerState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [convLoading, setConvLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { currentUser, showAuthModal, logout } = useAuth();
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike" | null>>({});
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [typingConvId, setTypingConvId] = useState<string | null>(null);
  const typingConvIdRef = useRef<string | null>(null);
  useEffect(() => { typingConvIdRef.current = typingConvId; }, [typingConvId]);
  const [openSourceKey, setOpenSourceKey] = useState<string | null>(null);
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  // Map document_id → local object URL (for PDF/image preview without re-downloading)
  const blobUrlMapRef = useRef<Map<string, { url: string; type: "pdf" | "image" | "txt" | "cad"; cadSummary?: Record<string, unknown>; ifcBlobUrl?: string }>>(new Map());
  const bubbleHighlightRef = useRef<HTMLElement>(null);
  const bubbleScrollRef    = useRef<HTMLDivElement>(null);
  bubbleViewerRef.current  = bubbleViewer; // always-fresh mirror, no stale closure
  const fileInputRef = useRef<HTMLInputElement>(null); // kept for CAD upload only (no local picker UI)
  const docFileInputRef = useRef<HTMLInputElement>(null); // regular doc upload
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounterRef = useRef(0); // track nested drag enter/leave
  const [isChatDragOver, setIsChatDragOver] = useState(false);
  const chatDragCounterRef = useRef(0); // track nested drag enter/leave for chat area

  const handleFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    const sid = sessionIdRef.current ?? sessionId;
    setIsUploading(true);
    for (const file of arr) {
      // Optimistic placeholder in docs panel
      const placeholderId = `uploading-${Date.now()}-${file.name}`;
      setDocuments(prev => [...prev, {
        document_id: placeholderId,
        filename: file.name,
        doc_type: "uploading",
        timestamp: new Date().toISOString(),
      } as any]);
      try {
        const res = await api.uploadDocument(file, sid ?? undefined);
        setPendingDocIds(prev => [...prev, res.document_id]);
        setDocuments(prev => prev
          .filter(d => d.document_id !== placeholderId)
          .concat([{ document_id: res.document_id, filename: res.filename, doc_type: file.name.split(".").pop() ?? "doc", timestamp: new Date().toISOString() }])
        );
        toast({ title: "File uploaded", description: `${res.filename} ready` });
      } catch (err) {
        setDocuments(prev => prev.filter(d => d.document_id !== placeholderId));
        toast({ title: "Upload failed", description: String(err), variant: "destructive" });
      }
    }
    setIsUploading(false);
  };
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputAreaRef = useRef<HTMLDivElement>(null);
  const [notifyBottomOffset, setNotifyBottomOffset] = useState(125);
  const [isHoveringVoice, setIsHoveringVoice] = useState(false);
  const [isDark, setIsDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const observer = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains("dark"))
    );
    observer.observe(document.documentElement, { attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  const { toast } = useToast();
  const navigate = useNavigate();

  // ── Nav confirmation state ────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e: CustomEvent) => {
      const { duration, startedAt, convId } = e.detail as {
        duration: number;
        startedAt: string;
        convId: string;
      };
      if (convId !== activeConvId) return;
      const card: Message = {
        id: `call-${Date.now()}`,
        role: "user",
        content: "",
        timestamp: new Date(),
        callCard: { duration, startedAt: new Date(startedAt) },
      };
      setMessages(prev => [...prev, card]);
    };
    window.addEventListener("call-ended", handler as EventListener);
    return () => window.removeEventListener("call-ended", handler as EventListener);
  }, [activeConvId]);

  // ── Voice recording ───────────────────────────────────────────────────────
  type VoiceState = "idle" | "recording" | "transcribing";
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [showSilenceWarning, setShowSilenceWarning] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef   = useRef<Blob[]>([]);
  const recordingStartRef = useRef<number>(0);
  // Web Audio API analyser for live waveform
  const audioCtxRef         = useRef<AudioContext | null>(null);
  const analyserRef         = useRef<AnalyserNode | null>(null);
  const animFrameRef        = useRef<number>(0);
  const maxDurationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [waveformBars, setWaveformBars]         = useState<number[]>(Array(48).fill(0));
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Captured amplitude samples (one per ~50 ms) — used to build the bubble waveform
  const waveformSamplesRef = useRef<number[]>([]);
  // Track which voice message bubbles have transcription expanded
  const [expandedTranscripts, setExpandedTranscripts] = useState<Set<string>>(new Set());

  const handleVoiceClick = useCallback(async () => {
    // ── Stop recording ───────────────────────────────────────────────────────
    if (voiceState === "recording") {
      mediaRecorderRef.current?.stop();
      return;
    }

    // ── Start recording ──────────────────────────────────────────────────────
    if (voiceState !== "idle") return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/ogg";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recordingStartRef.current = Date.now();

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Stop all mic tracks immediately
        stream.getTracks().forEach((t) => t.stop());

        // ── Clean up Web Audio + timers ───────────────────────────────────
        cancelAnimationFrame(animFrameRef.current);
        analyserRef.current = null;
        audioCtxRef.current?.close().catch(() => {});
        audioCtxRef.current = null;
        if (elapsedTimerRef.current) { clearInterval(elapsedTimerRef.current); elapsedTimerRef.current = null; }
        if (maxDurationTimerRef.current) { clearTimeout(maxDurationTimerRef.current); maxDurationTimerRef.current = null; }
        setWaveformBars(Array(28).fill(0));
        setRecordingElapsed(0);

        const duration = (Date.now() - recordingStartRef.current) / 1000;
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        audioChunksRef.current = [];

        if (blob.size < 100) {
          setVoiceState("idle");
          return;
        }

        setVoiceState("transcribing");

        // Create a local object URL for the audio blob so we can play it
        const blobUrl = URL.createObjectURL(blob);
        // Snapshot amplitude samples captured during recording
        const capturedSamples = [...waveformSamplesRef.current];
        waveformSamplesRef.current = [];

        // Immediately add a voice message bubble (transcript will fill in async)
        const voiceMsgId = Date.now().toString();
        const voiceMsg: Message = {
          id: voiceMsgId,
          role: "user",
          content: "", // will be filled with transcript once available
          voiceBlobUrl: blobUrl,
          voiceDuration: duration,
          voiceTranscript: undefined, // pending
          voiceWaveform: capturedSamples,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, voiceMsg]);

        try {
          const formData = new FormData();
          formData.append("audio", blob, "recording.webm");
          formData.append("mime_type", mimeType);

          const res = await fetch(`${getApiBase()}/transcribe`, {
            method: "POST",
            body: formData,
          });

          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error((err as Record<string, string>).detail ?? `HTTP ${res.status}`);
          }

          const data = await res.json() as { transcript: string };
          const transcript = data.transcript?.trim();

          if (transcript) {
            // Update the voice bubble with transcript and set content for RAG
            const updatedVoiceMsg: Message = { ...voiceMsg, content: transcript, voiceTranscript: transcript };
            const voiceMessages = messages.some(m => m.id === voiceMsgId)
              ? messages.map(m => m.id === voiceMsgId ? updatedVoiceMsg : m)
              : [...messages, updatedVoiceMsg];
            setMessages(voiceMessages);
            // Wrap the transcript so the agent knows this was spoken, not typed
            const voiceQuery = `[Voice message — the user said this aloud, not typed it]\n\n${transcript}`;
            // Now run the RAG query using the transcript as the question
            setIsLoading(true);
            if (!notifyDismissed) setShowNotifyBanner(true);
            try {
              const convId = ensureActiveConversationId();
              const assistantMsg = await runStreamingQuery(voiceQuery);
              if (!assistantMsg) return;
              const updated = [...voiceMessages, assistantMsg];
              const title = transcript.length > 50 ? transcript.slice(0, 50) + "…" : transcript;
              const preview = assistantMsg.content.replace(/\[.*?\]/g, "").replace(/#{1,3}\s/g, "").slice(0, 80) + "…";
              setMessages(updated);
              setConversations(convs => {
                const existing = convs.find(c => c.id === convId);
                if (existing) return convs.map(c => c.id === convId ? { ...c, messages: updated, preview, timestamp: new Date() } : c);
                return [{ id: convId, title, preview, timestamp: new Date(), messages: updated, sessionId } as any, ...convs];
              });
              // Persist voice reply to DB
              saveConversationToDB(convId, sessionIdRef.current ?? "", title, preview, updated);
              setTypingMessageId(assistantMsg.id);
              fetchSuggestions(transcript, assistantMsg.content);
              fireNotification();
            } catch (error) {
              const msg = serializeError(error);
              const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: `Sorry, I encountered an error: ${msg}. Please try again.`,
                timestamp: new Date(),
              };
              setMessages(prev => [...prev, errorMsg]);
              toast({ title: "Query failed", description: msg, variant: "destructive" });
            }
          } else {
            // No speech detected — mark transcript as empty
            setMessages(prev => prev.map(m =>
              m.id === voiceMsgId
                ? { ...m, voiceTranscript: "(no speech detected)" }
                : m
            ));
            toast({ title: "Nothing detected", description: "No speech was recognised — please try again.", variant: "destructive" });
          }
        } catch (err) {
          console.error("Transcription error:", err);
          setMessages(prev => prev.map(m =>
            m.id === voiceMsgId
              ? { ...m, voiceTranscript: "(transcription failed)" }
              : m
          ));
          toast({ title: "Transcription failed", description: err instanceof Error ? err.message : "Could not reach the transcription service.", variant: "destructive" });
        } finally {
          setVoiceState("idle");
        }
      };

      recorder.start();
      setVoiceState("recording");
      setRecordingElapsed(0);

      // ── Web Audio API — live waveform analyser ────────────────────────────
      try {
        const audioCtx = new AudioContext();
        audioCtxRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.78;
        source.connect(analyser);
        analyserRef.current = analyser;

        const BAR_COUNT = 48;
        const freqData = new Uint8Array(analyser.frequencyBinCount);
        waveformSamplesRef.current = [];
        let lastSampleTime = 0;
        let lastHeardTime = performance.now();
        let silenceToastShown = false;

        const tick = (timestamp: number) => {
          analyser.getByteFrequencyData(freqData);
          // Map frequency bins to bar heights (0–1), focus on vocal range (0–60% of bins)
          const bars = Array.from({ length: BAR_COUNT }, (_, i) => {
            const binIdx = Math.floor((i / BAR_COUNT) * (freqData.length * 0.6));
            const raw = freqData[binIdx] / 255;
            const centreFactor = 1 - Math.abs((i / (BAR_COUNT - 1)) - 0.5) * 0.35;
            return Math.min(1, raw * centreFactor * 1.4);
          });
          setWaveformBars(bars);

          // Capture amplitude sample every ~50 ms for bubble waveform
          if (timestamp - lastSampleTime >= 50) {
            lastSampleTime = timestamp;
            const avg = bars.reduce((s, v) => s + v, 0) / bars.length;
            waveformSamplesRef.current.push(avg);

            // Silence detection — if avg amplitude above threshold, reset the clock
            if (avg > 0.04) {
              lastHeardTime = performance.now();
              if (silenceToastShown) {
                silenceToastShown = false;
                setShowSilenceWarning(false);
              }
            } else if (!silenceToastShown && performance.now() - lastHeardTime > 3000) {
              silenceToastShown = true;
              setShowSilenceWarning(true);
            }
          }

          animFrameRef.current = requestAnimationFrame(tick);
        };
        animFrameRef.current = requestAnimationFrame(tick);
      } catch {
        // Web Audio unavailable — waveform stays flat, recording still works
      }

      // ── Elapsed timer ─────────────────────────────────────────────────────
      elapsedTimerRef.current = setInterval(() => {
        setRecordingElapsed(s => s + 1);
      }, 1000);

      // ── 60-second hard stop ───────────────────────────────────────────────
      maxDurationTimerRef.current = setTimeout(() => {
        mediaRecorderRef.current?.stop();
      }, 60_000);

    } catch (err) {
      console.error("Microphone error:", err);
      toast({ title: "Microphone access denied", description: "Allow microphone access in your browser settings.", variant: "destructive" });
      setVoiceState("idle");
    }
  }, [voiceState, toast]);

  // ── Thinking / progress steps ────────────────────────────────────────────
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [thinkingExpanded, setThinkingExpanded] = useState(true);

  // ── Notify-when-done ─────────────────────────────────────────────────────
  const [notifyEnabled, setNotifyEnabled]       = useState(false);
  const [showNotifyBanner, setShowNotifyBanner] = useState(false);
  const [notifyDismissed, setNotifyDismissed]   = useState(false);
  // Ref mirror so fireNotification never reads a stale closure value.
  const notifyEnabledRef = useRef(false);
  useEffect(() => { notifyEnabledRef.current = notifyEnabled; }, [notifyEnabled]);

  // Audio is created lazily on first user gesture to satisfy browser autoplay policy.
  // We keep a ref to the Audio object (null until first interaction) and a flag
  // that tracks whether it has been "unlocked" (played silently once).
  const beepAudioRef    = useRef<HTMLAudioElement | null>(null);
  const audioUnlocked   = useRef(false);

  // Call once on any user gesture to create + silently unlock the audio element.
  const ensureAudio = useCallback(() => {
    if (beepAudioRef.current) return beepAudioRef.current;
    const audio = new Audio("/beep.wav");
    audio.preload = "auto";
    beepAudioRef.current = audio;
    // Unlock: play at zero volume then immediately pause — satisfies autoplay policy.
    if (!audioUnlocked.current) {
      audio.volume = 0;
      audio.play().then(() => {
        audio.pause();
        audio.currentTime = 0;
        audio.volume = 1;
        audioUnlocked.current = true;
      }).catch(() => {});
    }
    return audio;
  }, []);

  const playBeep = useCallback(() => {
    const audio = ensureAudio();
    audio.currentTime = 0;
    audio.volume = 1;
    audio.play().catch(() => {});
  }, [ensureAudio]);

  // Uses ref so it never has a stale value of notifyEnabled, regardless of
  // when/where it's called from inside async callbacks like handleSend.
  const fireNotification = useCallback(() => {
    if (!notifyEnabledRef.current) return;
    // Always play a single ding — whether the tab is visible or not.
    playBeep();
    // If the tab is hidden, also send a browser notification (if permitted).
    if (document.visibilityState !== "visible" && Notification.permission === "granted") {
      new Notification("Done!", { body: "Your answer is ready.", icon: "/favicon.svg" });
    }
  }, [playBeep]); // no longer depends on notifyEnabled state — reads ref instead

  // ── Filename autocomplete ────────────────────────────────────────────────
  const [autocomplete, setAutocomplete] = useState<{
    query: string;
    triggerPos: number;
    results: Document[];
    activeIdx: number;
  } | null>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);
  const [autocompletePos, setAutocompletePos] = useState<{ top: number; left: number; width: number } | null>(null);

  // ── Word-completion trie (DWG) ───────────────────────────────────────────
  const trieRef = useRef<TrieNode>(makeTrie());
  // Ghost-text inline suggestion: the suffix to append after the current word
  const [wordSuffix, setWordSuffix] = useState<string>("");
  // Debounce timer for AI ghost-text — avoids a fetch on every keystroke
  const ghostDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Rebuild trie whenever messages or documents change.
  useEffect(() => {
    const root = makeTrie();
    // Seed from all message content (user + assistant), weighted by role
    for (const msg of messages) {
      const weight = msg.role === "user" ? 3 : 1; // user phrasing more relevant
      for (const word of tokenise(msg.content)) trieInsert(root, word, weight);
    }
    // Seed from document filenames (high weight — user types these often)
    for (const doc of documents) {
      for (const word of tokenise(doc.filename)) trieInsert(root, word, 5);
      // Also insert full filename tokens as-is (e.g. "report2024")
      const slug = doc.filename.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9]/g, "");
      if (slug.length >= 3) trieInsert(root, slug, 5);
    }
    trieRef.current = root;
  }, [messages, documents]);

  // ── Contextual next-step suggestions ────────────────────────────────────
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [expandingSuggestion, setExpandingSuggestion] = useState<string | null>(null);

  // ── Document viewer helpers ──────────────────────────────────────────────

  const IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];
  const CAD_EXTS = ['.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp', '.rvt', '.nwd', '.nwc', '.dgn', '.skp', '.3dm', '.fbx', '.obj', '.stl', '.sat', '.iges', '.igs', '.prt', '.sldprt', '.catpart', '.3ds', '.dae', '.rfa', '.rte'];
  const ALLOWED_EXTS = ['.pdf', '.docx', '.doc', '.txt', ...IMAGE_EXTS, ...CAD_EXTS];

  const getApiBase = () =>
    ((typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000");

  /**
   * After the first assistant reply, call the backend /title endpoint to get a
   * short, specific conversation title and update the sidebar entry.
   * Fires once per conversation (only while title still looks auto-generated).
   */
  const generateSmartTitle = useCallback(async (
    convId: string,
    messages: Message[],
    currentTitle: string,
  ) => {
    const userMsgs = messages.filter(m => m.role === "user");
    if (userMsgs.length < 1) return;
    const firstUserText = userMsgs[0]?.content ?? "";
    // Re-title whenever the title still looks like a raw first-message slice
    // (either exact match or truncated version). Be permissive — smart titles
    // are cheap and always better than raw user phrasing.
    const norm = (s: string) => s.trim().toLowerCase().replace(/…$/, "").slice(0, 50);
    const looksRaw = norm(currentTitle) === norm(firstUserText);
    if (!looksRaw) return;

    try {
      const res = await fetch(`${getApiBase()}/title`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "chat",
          messages: messages.slice(0, 6).map(m => ({ role: m.role, content: m.content.slice(0, 150) })),
        }),
        signal: AbortSignal.timeout(8000),
      });
      if (!res.ok) return;
      const data = await res.json();
      const smartTitle = (data.title ?? "").trim();
      if (!smartTitle || smartTitle.length > 80) return;
      setConversations(convs => convs.map(c => c.id === convId ? { ...c, title: smartTitle } : c));
      // Update title in DB
      const _conv = conversations.find(c => c.id === convId);
      if (_conv) {
        const messagesToSave = (_conv.messages && _conv.messages.length > 0)
          ? _conv.messages
          : messages;
        saveConversationToDB(convId, (_conv as any).sessionId ?? sessionIdRef.current ?? "", smartTitle, _conv.preview, messagesToSave);
      }
    } catch {
      // Silent fail — title stays as-is
    }
  }, []);

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
    const isCadExt = ['.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp'].includes(`.${ext}`);
    setBubbleDoc(doc);

    // ── CAD/IFC: use the locally cached blob + summary, never hit /content ──
    if (isCadExt) {
      const cached = blobUrlMapRef.current.get(doc.document_id);
      setBubbleViewer({
        doc,
        content: null,
        loading: false,
        error: null,
        highlightText,
        highlightLines,
        highlightKey: 0,
        mediaType: 'cad',
        blobUrl: cached?.url,
        cadSummary: cached?.cadSummary,
        ifcBlobUrl: cached?.ifcBlobUrl,
      });
      return;
    }

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
    const isCadExt = ['.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp'].includes(`.${ext}`);

    if (isCadExt) {
      const cached = blobUrlMapRef.current.get(doc.document_id);
      setViewer({
        doc,
        content: null,
        loading: false,
        error: null,
        highlightText,
        highlightLines,
        highlightKey: 0,
        mediaType: 'cad',
        cadSummary: cached?.cadSummary,
        blobUrl: cached?.url,
        ifcBlobUrl: cached?.ifcBlobUrl,
      });
      return;
    }

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

  // Track input area height so notify banner floats the right distance above it
  useEffect(() => {
    const el = inputAreaRef.current;
    if (!el) return;
    const update = () => setNotifyBottomOffset(el.offsetHeight + 8);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [suggestions, suggestionsLoading]);

  // Auto-scroll to bottom when new messages arrive (skip on very first message to avoid jump)
  useEffect(() => {
    if (messages.length <= 1) return;
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

  // Close reports panel when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!reportsPanelOpenRef.current) return;
      const target = e.target as HTMLElement;
      if (!target.closest("[data-reports-panel]")) {
        reportsPanelOpenRef.current = false;
        setReportsPanelOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Focus report edit textarea when edit mode opens
  useEffect(() => {
    if (reportEditMode && reportEditInputRef.current) {
      reportEditInputRef.current.focus();
    }
  }, [reportEditMode]);

  // Load documents + conversations on mount
  useEffect(() => {
    loadDocuments();
    loadReports();
    loadConversationsFromDB();
  }, []);

  // Retry after a short delay — AuthContext hydrates async, so the first
  // call above may have no token yet. This ensures conversations load on
  // the splash screen without requiring the user to click "New Chat" first.
  useEffect(() => {
    const t = setTimeout(() => loadConversationsFromDB(), 600);
    return () => clearTimeout(t);
  }, []);

  // Re-load conversations + docs when user logs in or out
  useEffect(() => {
    if (currentUser) {
      // Logged in — fetch their conversations and session-scoped docs
      loadConversationsFromDB();
      loadDocuments(sessionIdRef.current);
    } else {
      // Logged out — clear everything
      setConversations([]);
      setDocuments([]);
      startNewConversation();
    }
  }, [currentUser?.user_id]); // user_id stable ref avoids re-runs on object recreation






  // ==========================================================================
  // DOCUMENT MANAGEMENT
  // ==========================================================================

  const loadDocuments = async (sid?: string | null) => {
    try {
      // Pass session_id so the backend returns only this session's documents.
      // Falls back to full list if session_id is not yet known (first load).
      const effectiveSid = sid ?? sessionIdRef.current;
      const response = await api.listDocuments(effectiveSid ?? undefined);
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


  // ── DB conversation persistence helpers ──────────────────────────────────

  const getAuthHeader = (): Record<string, string> => {
    // currentUser.token is always the source of truth (AuthContext stores full object in bimlo_auth)
    const token = currentUser?.token ?? (() => {
      try { return JSON.parse(localStorage.getItem("bimlo_auth") ?? "{}").token ?? ""; } catch { return ""; }
    })();
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const loadConversationsFromDB = async () => {
    // Use getAuthHeader() which falls back to localStorage, so this works even
    // before currentUser has hydrated in the React context.
    const authH = getAuthHeader();
    if (!authH.Authorization) {
      console.warn("loadConversationsFromDB skipped: no auth token available");
      return;
    }
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/auth/conversations`, {
        headers: { ...(authH as any), "Content-Type": "application/json" },
      });
      if (!res.ok) {
        const errorText = await res.text().catch(() => "<no body>");
        console.warn("loadConversationsFromDB failed:", res.status, errorText);
        return;
      }
      const rows: Array<{ id: string; title: string; preview: string; updated_at: string; session_id: string }> = await res.json();
      console.log(`✅ loadConversationsFromDB loaded ${rows.length} conversations`, rows.map(r => ({ id: r.id, session_id: r.session_id })));
      // Keep any in-memory conversation state (messages/pending typing) while
      // refreshing the sidebar list. This prevents the user from losing a
      // pending message when the DB list is reloaded mid-generation.
      setConversations(prev => {
        const existingById = new Map(prev.map(c => [c.id, c]));
        const refreshed = rows.map(r => {
          const existing = existingById.get(r.id);
          return {
            id: r.id,
            title: r.title || "Untitled",
            preview: r.preview || "",
            timestamp: new Date(r.updated_at || Date.now()),
            messages: existing?.messages?.length ? existing.messages : [],
            sessionId: r.session_id,
          } as any;
        });
        // Preserve any local-only conversations that have not yet been synced.
        const localOnly = prev.filter(c => !rows.some(r => r.id === c.id));
        return [...localOnly, ...refreshed];
      });
    } catch (e) {
      console.error("loadConversationsFromDB failed:", e);
    }
  };

  const saveConversationToDB = async (
    convId: string,
    sid: string,
    title: string,
    preview: string,
    msgs: Message[],
  ) => {
    const authHeader = getAuthHeader();
    if (!authHeader.Authorization) {
      console.warn("saveConversationToDB skipped: no auth token available", { convId, title, preview, messageCount: msgs.length });
      return;
    }

    try {
      const base = getApiBase();
      console.log("💾 saveConversationToDB", { convId, sessionId: sid, title, preview, messageCount: msgs.length, currentUser: !!currentUser });
      const res = await fetch(`${base}/auth/conversations/save`, {
        method: "POST",
        headers: { ...(authHeader as any), "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: convId,
          session_id: sid,
          title,
          preview,
          chat_type: "rag",
          messages: msgs.map(m => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : m.timestamp,
            // Rich payload — everything beyond core text fields
            sources:          m.sources          ?? undefined,
            confidence:       m.confidence       ?? undefined,
            analytics:        m.analytics        ?? undefined,
            reportId:         m.reportId         ?? undefined,
            reportTitle:      m.reportTitle      ?? undefined,
            reportMeta:       m.reportMeta       ?? undefined,
            thinkingSteps:    m.thinkingSteps    ?? undefined,
            voiceTranscript:  m.voiceTranscript  ?? undefined,
            callCard:         m.callCard         ?? undefined,
            attachedDocIds:   m.attachedDocIds   ?? undefined,
            navAction:        m.navAction        ?? undefined,
            interrupted:      m.interrupted      ?? undefined,
          })),
        }),
      });
      if (!res.ok) {
        const errorText = await res.text().catch(() => "<no body>");
        console.error("saveConversationToDB failed response:", res.status, errorText);
      } else {
        console.log("✅ saveConversationToDB succeeded", { convId, messageCount: msgs.length });
      }
    } catch (e) {
      console.error("saveConversationToDB failed:", e);
    }
  };

  // ==========================================================================
  // REPORT MANAGEMENT
  // ==========================================================================

  const loadReports = async () => {
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/reports`);
      if (res.ok) {
        const data: ReportRecord[] = await res.json();
        setReports(data);
      }
    } catch (err) {
      console.error("Failed to load reports:", err);
    }
  };

  const handleGenerateReport = async (prompt: string, sid?: string, explicitDocs: string[] = []) => {
    const resolvedSid = sid || sessionIdRef.current || sessionId;
    if (!resolvedSid || isGeneratingReport) return;
    setIsGeneratingReport(true);

    // Show loading skeleton on the last assistant message immediately
    setMessages(prev => {
      const msgs = [...prev];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") {
          msgs[i] = { ...msgs[i], reportGenerating: true, reportId: null };
          break;
        }
      }
      return msgs;
    });

    try {
      const base = getApiBase();
      const res = await fetch(`${base}/reports`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          session_id:     resolvedSid,
          available_docs: documents.map(d => d.filename),
          explicit_docs:  explicitDocs,
          include_charts: true,
          language:       "en",
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const report: ReportRecord = await res.json();
      setReports(prev => [report, ...prev]);
      setActiveReport(report);
      reportsPanelOpenRef.current = true;
      setReportsPanelOpen(true);
      // Replace the last assistant message content with the agent's summary,
      // then attach the reportId so the card renders below it.
      const summaryText = report.summary?.trim()
        || `The report "${report.title}" has been generated with ${(report.content.match(/^#{1,2}\s/gm) || []).length} sections.`;
      setMessages(prev => {
        const msgs = [...prev];
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === "assistant") {
            msgs[i] = {
              ...msgs[i],
              content:          summaryText,
              rawAnswer:        summaryText,
              reportId:         report.report_id,
              reportGenerating: false,
            };
            break;
          }
        }
        setConversations(convs => convs.map(conv => {
          if (conv.id !== activeConvId) return conv;
          return { ...conv, messages: msgs };
        }));
        return msgs;
      });
      toast({ title: "Report ready", description: report.title });
    } catch (err) {
      // Clear the loading skeleton on failure
      setMessages(prev => {
        const msgs = [...prev];
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === "assistant") {
            msgs[i] = { ...msgs[i], reportGenerating: false };
            break;
          }
        }
        return msgs;
      });
      console.error("Report generation failed:", err);
      toast({ title: "Report generation failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handlePatchReport = async () => {
    if (!activeReport || !reportEditInstruction.trim() || isPatchingReport) return;
    setIsPatchingReport(true);
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/reports/${activeReport.report_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction: reportEditInstruction.trim(),
          session_id:  sessionId ?? "",
          language:    "en",
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const updated: ReportRecord = await res.json();
      setActiveReport(updated);
      setReports(prev => prev.map(r => r.report_id === updated.report_id ? updated : r));
      setReportEditInstruction("");
      setReportEditMode(false);
      toast({ title: "Report updated", description: `v${updated.version}` });
    } catch (err) {
      console.error("Patch failed:", err);
      toast({ title: "Edit failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setIsPatchingReport(false);
    }
  };

  const handleDeleteReport = async (reportId: string) => {
    setDeletingReportId(reportId);
    try {
      const base = getApiBase();
      await fetch(`${base}/reports/${reportId}`, { method: "DELETE" });
      setReports(prev => prev.filter(r => r.report_id !== reportId));
      if (activeReport?.report_id === reportId) {
        setActiveReport(null);
        setReportEditMode(false);
      }
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeletingReportId(null);
    }
  };

  const handleDownloadReport = async (report: ReportRecord, fmt: "pdf" | "md" = "pdf") => {
    setDownloadingReportId(report.report_id);
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/reports/${report.report_id}/download?fmt=${fmt}`);
      if (!res.ok) throw new Error("Download failed");
      const blob     = await res.blob();
      const url      = URL.createObjectURL(blob);
      const a        = document.createElement("a");
      a.href         = url;
      const safeName = report.title.replace(/\s+/g, "_").slice(0, 40);
      // If server fell back from pdf to md (Content-Type check)
      const ct       = res.headers.get("Content-Type") ?? "";
      const ext      = ct.includes("pdf") ? "pdf" : ct.includes("markdown") ? "md" : fmt;
      a.download     = `${safeName}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast({ title: "Download failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setDownloadingReportId(null);
    }
  };

  const handleRestoreVersion = async (report: ReportRecord, version: number) => {
    setRestoringVersion(version);
    try {
      const base = getApiBase();
      const res  = await fetch(`${base}/reports/${report.report_id}/restore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version }),
      });
      if (!res.ok) throw new Error("Restore failed");
      const updated: ReportRecord = await res.json();
      setActiveReport(updated);
      setReports(prev => prev.map(r => r.report_id === updated.report_id ? updated : r));
      setShowVersionHistory(false);
      toast({ title: `Restored to v${version}`, description: `Now at v${updated.version}` });
    } catch (err) {
      toast({ title: "Restore failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setRestoringVersion(null);
    }
  };

  // ── ReportCard — compact single-row card rendered below the assistant bubble ──
  // Accepts either a full ReportRecord OR inline SSE data (title + meta) so it
  // never needs to wait on a secondary fetch or React state propagation.
  const ReportCard = ({
    report,
    inlineTitle,
    inlineMeta,
    isLoading: cardLoading = false,
  }: {
    report: ReportRecord | null;
    inlineTitle?: string | null;
    inlineMeta?: { word_count: number; section_count: number; source_docs: string[]; version: number } | null;
    isLoading?: boolean;
  }) => {
    const isDownloading = report ? downloadingReportId === report.report_id : false;

    // Resolve display values — prefer full report, fall back to inline SSE data
    const title        = report?.title ?? inlineTitle ?? null;
    const wordCount    = report ? report.content.split(/\s+/).length : (inlineMeta?.word_count ?? 0);
    const sectionCount = report ? (report.content.match(/^#{1,2}\s/gm) || []).length : (inlineMeta?.section_count ?? 0);
    const sourceDocs   = report?.source_docs ?? inlineMeta?.source_docs ?? [];
    const version      = report?.version ?? inlineMeta?.version ?? 1;

    // Show spinner only if we have no title at all yet (truly still building)
    if (cardLoading && !title) {
      return (
        <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-border bg-muted/30">
          <Loader2 className="h-3.5 w-3.5 text-primary animate-spin shrink-0" />
          <span className="text-xs text-muted-foreground italic">Building report…</span>
        </div>
      );
    }

    // We have at least a title — render the card (download may be disabled until report is ready)
    return (
      <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-border bg-muted/20 hover:bg-muted/40 hover:border-primary/30 transition-all duration-150 group">
        <ScrollText className="h-5 w-5 text-primary/70 shrink-0" />

        {/* Title + meta — click to open panel */}
        <div
          className="flex-1 min-w-0 cursor-pointer"
          onClick={async () => {
            if (report) {
              setActiveReport(report);
            } else if (inlineTitle) {
              // Report not in state yet — try fetching by ID when user clicks
              const fullReport = reports.find(r => r.title === inlineTitle) ?? null;
              if (fullReport) setActiveReport(fullReport);
            }
            reportsPanelOpenRef.current = true;
            setReportsPanelOpen(true);
          }}
        >
          <span className="text-sm font-semibold text-foreground truncate block group-hover:text-primary transition-colors">
            {title}
          </span>
          <span className="text-xs text-muted-foreground/70 mt-0.5 block">
            {sectionCount > 0 && `${sectionCount} sections · `}{wordCount > 0 ? `${wordCount.toLocaleString()} words` : ""}
            {version > 1 && ` · v${version}`}
            {sourceDocs.length > 0 && ` · ${sourceDocs.length} source${sourceDocs.length === 1 ? "" : "s"}`}
          </span>
        </div>

        {/* Download buttons — only enabled once we have the full report object */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded transition-colors disabled:opacity-40"
            onClick={e => { e.stopPropagation(); if (report) handleDownloadReport(report, "md"); }}
            disabled={!report || isDownloading}
            title={report ? "Download Markdown" : "Report loading…"}
          >
            MD
          </button>
          <button
            className="text-xs font-medium text-primary hover:text-primary/70 px-2 py-1 rounded transition-colors disabled:opacity-40 flex items-center gap-1"
            onClick={e => { e.stopPropagation(); if (report) handleDownloadReport(report, "pdf"); }}
            disabled={!report || isDownloading}
            title={report ? "Download PDF" : "Report loading…"}
          >
            {isDownloading && <Loader2 className="h-3 w-3 animate-spin" />}
            PDF
          </button>
        </div>
      </div>
    );
  };

  // ── AssistantContent — renders main markdown response, strips legacy report markers ──
  const AssistantContent = ({
    raw, msgId, sources, onSourceClick,
  }: {
    raw: string;
    msgId: string;
    sources: Source[] | undefined;
    onSourceClick: (n: number, id: string) => void;
  }) => {
    const cleanRaw = raw.replace(/__REPORT_CARD__:[a-zA-Z0-9_-]+/g, "").trim();
    return (
      <div className="leading-relaxed">
        {renderContent(cleanRaw, msgId, sources, onSourceClick)}
      </div>
    );
  };

  // Inline chart renderer — draws a Chart.js chart into a canvas
  const ReportInlineChart = ({ chart }: { chart: ChartRecord }) => {
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

    // Re-theme when dark/light mode changes
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
        {/* Header row */}
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
            title="Download chart as PNG"
            className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted border border-transparent hover:border-border transition-all shrink-0"
          >
            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            PNG
          </button>
        </div>

        {/* Canvas */}
        <div className="px-4 pb-3">
          <div className="rounded-lg border border-border bg-card p-3 shadow-inner overflow-hidden">
            <div style={{ position: "relative", height: 220 }}>
              <canvas ref={canvasRef} />
            </div>
          </div>
        </div>

        {/* Key insights — same collapsible panel as ChartMessage */}
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

  // Render report markdown content, injecting charts at placeholder positions
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
            li: ({ children }) => <li className="ml-2">{children}</li>,
            strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
            em: ({ children }) => <em className="italic text-foreground/80">{children}</em>,
            hr: () => <hr className="border-border my-4" />,
            blockquote: ({ children }) => <blockquote className="border-l-2 border-primary/40 pl-3 italic text-muted-foreground my-2 text-xs">{children}</blockquote>,
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
            th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-semibold text-left text-xs">{children}</th>,
            td: ({ children }) => <td className="border border-border px-2 py-1 text-xs">{children}</td>,
          }}
        >
          {seg.value}
        </ReactMarkdown>
      );
    });
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
    setActiveConvId("");
    setSessionId(null);
    sessionIdRef.current = null;
    setMessages([]);
  };

  const ensureActiveConversationId = () => {
    if (activeConvId) return activeConvId;
    const newId = Date.now().toString();
    setActiveConvId(newId);
    activeConvIdRef.current = newId;  // sync update so runStreamingQuery reads it immediately
    return newId;
  };

  const loadConversation = async (conv: Conversation) => {
    console.log("🔄 loadConversation called for:", conv.id, conv.title);
    // Deduplicate: skip if already on this conversation and messages are loaded
    const currentMessages = messagesRef.current;
    const currentActiveConvId = activeConvIdRef.current;
    if (currentMessages.length > 0 && conv.id === currentActiveConvId) {
      console.log("⏭️ Skipping: already on this conversation with messages loaded");
      return;
    }

    const sid = (conv as any).sessionId ?? null;
    console.log("📋 Conversation details:", { id: conv.id, hasMessages: conv.messages?.length ?? 0, hasSessionId: !!sid });

    // 1. In-memory fast path (already loaded this session)
    if (conv.messages && conv.messages.length > 0) {
      console.log("⚡ Using fast path: messages already in memory");
      setActiveConvId(conv.id);
      setSessionId(null);
      sessionIdRef.current = null;
      setMessages(conv.messages);
      setConvLoading(false);
      if (sid) { setSessionId(sid); sessionIdRef.current = sid; loadDocuments(sid); }
      setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
      return;
    }

    // 2. Fetch from DB — auth conversations endpoint (has full payload)
    const base = getApiBase();
    // Build auth header: prefer live currentUser token, then bimlo_auth in localStorage,
    // then fall back to the legacy "token" key so existing sessions survive a page reload.
    const rawToken = currentUser?.token ?? (() => {
      try {
        return (
          JSON.parse(localStorage.getItem("bimlo_auth") ?? "{}").token ||
          localStorage.getItem("token") ||
          ""
        );
      } catch { return ""; }
    })();
    const authHeaders: Record<string, string> = {
      "Content-Type": "application/json",
      ...(rawToken ? { Authorization: `Bearer ${rawToken}` } : {}),
    };

    if (!rawToken) {
      console.warn("loadConversation skipped: no auth token available", { convId: conv.id, title: conv.title, sessionId: sid });
      return;
    }
    console.log("🔐 loadConversation auth info", { hasToken: !!rawToken, currentUser: !!currentUser, sessionId: sid });

    // Commit to loading this conversation only after we know we have a token
    setActiveConvId(conv.id);
    setSessionId(null);
    sessionIdRef.current = null;
    setConvLoading(true);
    setMessages([]);

    // Helper: something went wrong — reset so user isn't stuck on a blank screen
    const abortLoad = () => {
      setConvLoading(false);
      setActiveConvId("");
      activeConvIdRef.current = "";
      setMessages([]);
    };

    try {
      const res = await fetch(`${base}/auth/conversations/${conv.id}`, { headers: authHeaders });
      if (res.ok) {
        const data = await res.json();
        console.log("✅ loadConversation DB response", { conversationId: conv.id, messageCount: (data.messages ?? []).length, sessionId: data.session_id });
        const restored: Message[] = (data.messages ?? []).map((m: any) => ({
          // Spread the raw message first (captures any flat rich fields the backend returns)
          ...m,
          // Then spread payload if the backend nests rich fields there
          ...(m.payload ?? {}),
          // Always ensure these core fields are correct types and can't be overwritten
          id:        m.id ?? Date.now().toString(),
          role:      m.role as "user" | "assistant",
          content:   m.content ?? "",
          timestamp: new Date(m.timestamp ?? (m.payload?.timestamp) ?? Date.now()),
        }));
        setMessages(restored);
        setConvLoading(false);
        setConversations(prev => prev.map(c => c.id === conv.id ? { ...c, messages: restored } : c));
        if (data.session_id) { setSessionId(data.session_id); sessionIdRef.current = data.session_id; loadDocuments(data.session_id); }
        setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
        return;
      }
      const errorText = await res.text().catch(() => "<no body>");
      console.warn("loadConversation DB fetch non-ok:", res.status, errorText);
      // DB fetch failed, try fallback path if available
      if (!sid) {
        // No sessionId fallback available, abort
        abortLoad();
        return;
      }
    } catch (e) {
      console.error("loadConversation from DB failed:", e);
      abortLoad();
      return;
    }

    // 3. Fallback — session history endpoint (text-only, no payload)
    if (sid) {
      try {
        const res = await fetch(`${base}/sessions/${sid}/history`);
        if (res.ok) {
          const data = await res.json();
          const turns: Array<{ role: string; content: string }> = data.messages ?? [];
          if (turns.length > 0) {
            const restored: Message[] = turns.map((m, i) => ({
              id:        `${conv.id}-${i}`,
              role:      m.role as "user" | "assistant",
              content:   m.content,
              timestamp: new Date(),
            }));
            setMessages(restored);
            setConversations(prev => prev.map(c => c.id === conv.id ? { ...c, messages: restored } : c));
            setSessionId(sid); sessionIdRef.current = sid; loadDocuments(sid);
            setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
            setConvLoading(false);
            return;
          }
        }
      } catch (e) { console.error("session history fetch failed:", e); }
    }

    // All paths exhausted with no messages — reset so user isn't stuck on blank
    abortLoad();
  };

  const deleteConversation = async (convId: string) => {
    setConversations(prev => prev.filter(c => c.id !== convId));
    if (convId === activeConvId) startNewConversation();
    if (currentUser) {
      try {
        const base = getApiBase();
        await fetch(`${base}/auth/conversations/${convId}`, {
          method: "DELETE",
          headers: getAuthHeader() as any,
        });
      } catch (e) { console.error("deleteConversation from DB failed:", e); }
    }
  };

  // ── Shared streaming query runner ────────────────────────────────────────
  // Used by handleSend, handleRedo, and handleEditSubmit so they all get
  // thinking steps and the SSE stream.
  const runStreamingQuery = useCallback(async (query: string, forceRoute?: string) => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);

    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";

      abortControllerRef.current = new AbortController();

      // Build auth header so the backend links auto-saved messages to the user account
      const rawToken = currentUser?.token ?? (() => {
        try { return JSON.parse(localStorage.getItem("bimlo_auth") ?? "{}").token ?? ""; } catch { return ""; }
      })();
      const streamHeaders: Record<string, string> = { "Content-Type": "application/json" };
      if (rawToken) streamHeaders["Authorization"] = `Bearer ${rawToken}`;

      const res = await fetch(`${base}/query-stream`, {
        method: "POST",
        headers: streamHeaders,
        body: JSON.stringify({
          query,
          top_k: 5,
          ...(sessionId ? { session_id: sessionId } : {}),
          ...(forceRoute ? { force_route: forceRoute } : {}),
          // Tell the backend to save messages under the frontend's conversation ID
          // so loadConversation finds them without a mismatch.
          ...(activeConvIdRef.current ? { conversation_id: activeConvIdRef.current } : {}),
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(serializeError(body) || `HTTP ${res.status}`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let resultMsg: Message | null = null;
      const accumulatedSteps: ThinkingStep[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "status") {
              const step: ThinkingStep = { node: event.node, icon: event.icon, message: event.message, ts: Date.now() };
              accumulatedSteps.push(step);
              setThinkingSteps(prev => [...prev, step]);
            } else if (event.type === "result") {
              if (event.session_id) { setSessionId(event.session_id); sessionIdRef.current = event.session_id; loadDocuments(event.session_id); }
              const reportId: string | undefined = event.report_id;
              // If the router sent us to report_node, schedule a background fetch for the
              // reports panel — but the card itself uses inline SSE data, no fetch needed.
              if (reportId) {
                // Background fetch — populates reports panel & enables downloads.
                // Retry up to 3× with backoff since the server may not have saved yet.
                const _fetchReport = (attempt = 0) => {
                  fetch(`${base}/reports/${reportId}`)
                    .then(r => r.ok ? r.json() : Promise.reject(r.status))
                    .then(report => {
                      setReports(prev => [report, ...prev.filter((r: ReportRecord) => r.report_id !== reportId)]);
                      setActiveReport(report);
                      reportsPanelOpenRef.current = true;
                      setReportsPanelOpen(true);
                      toast({ title: "Report ready", description: report.title });
                    })
                    .catch(() => { if (attempt < 5) setTimeout(() => _fetchReport(attempt + 1), 3000 * (attempt + 1)); });
                };
                // Small initial delay — report save happens synchronously before SSE fires,
                // but disk write may lag slightly. Start at 500ms.
                setTimeout(() => _fetchReport(), 500);
              }
              resultMsg = {
                id: Date.now().toString(),
                role: "assistant",
                content: event.answer,
                rawAnswer: event.raw_answer ?? event.answer,
                sources: event.sources,
                confidence: event.confidence,
                analytics: event.analytics ?? null,
                reportId: reportId ?? null,
                reportTitle: event.report_title ?? null,
                reportMeta: event.report_meta ?? null,
                timestamp: new Date(),
              };
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          } catch { /* malformed SSE line */ }
        }
      }

      // Persist steps onto the message so they render after generation
      if (resultMsg) {
        resultMsg = { ...resultMsg, thinkingSteps: accumulatedSteps };
      }
      setThinkingSteps([]);
      return resultMsg;
    } catch (error) {
      setThinkingSteps([]);
      if (error instanceof Error && error.name === "AbortError") return null;
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  /**
   * Stream a report directly from /report-stream.
   * Emits the same status/result SSE events as /query-stream so the same
   * thinking-steps UI lights up. Returns a Message with reportId attached.
   */
  const runReportStream = useCallback(async (
    query: string,
    explicitDocs: string[],
  ): Promise<Message | null> => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);

    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";

      abortControllerRef.current = new AbortController();

      const res = await fetch(`${base}/report-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt:         query,
          session_id:     sessionIdRef.current ?? "",
          available_docs: documents.map(d => d.filename),
          explicit_docs:  explicitDocs,
          include_charts: true,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(serializeError(body) || `HTTP ${res.status}`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let resultMsg: Message | null = null;
      const accumulatedSteps: ThinkingStep[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "status") {
              const step: ThinkingStep = { node: event.node, icon: event.icon, message: event.message, ts: Date.now() };
              accumulatedSteps.push(step);
              setThinkingSteps(prev => [...prev, step]);
            } else if (event.type === "result") {
              if (event.session_id) { setSessionId(event.session_id); sessionIdRef.current = event.session_id; loadDocuments(event.session_id); }
              const reportId: string | undefined = event.report_id;
              if (reportId) {
                const _fetchReport2 = (attempt = 0) => {
                  fetch(`${base}/reports/${reportId}`)
                    .then(r => r.ok ? r.json() : Promise.reject(r.status))
                    .then(report => {
                      setReports(prev => [report, ...prev.filter((r: ReportRecord) => r.report_id !== reportId)]);
                      setActiveReport(report);
                      reportsPanelOpenRef.current = true;
                      setReportsPanelOpen(true);
                      toast({ title: "Report ready", description: report.title });
                    })
                    .catch(() => { if (attempt < 5) setTimeout(() => _fetchReport2(attempt + 1), 3000 * (attempt + 1)); });
                };
                setTimeout(() => _fetchReport2(), 500);
              }
              resultMsg = {
                id:          Date.now().toString(),
                role:        "assistant",
                content:     event.answer,
                rawAnswer:   event.raw_answer ?? event.answer,
                sources:     event.sources ?? [],
                confidence:  event.confidence ?? 1.0,
                analytics:   null,
                reportId:    reportId ?? null,
                reportTitle: event.report_title ?? null,
                reportMeta:  event.report_meta ?? null,
                timestamp:   new Date(),
              };
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          } catch { /* malformed SSE line */ }
        }
      }

      if (resultMsg) {
        resultMsg = { ...resultMsg, thinkingSteps: accumulatedSteps };
      }
      setThinkingSteps([]);
      return resultMsg;
    } catch (error) {
      setThinkingSteps([]);
      if (error instanceof Error && error.name === "AbortError") return null;
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, documents]);

  // ── CAD/IFC dedicated query runner ─────────────────────────────────────────
  // Calls /api/cad/query SSE stream — real status events from the backend.
  const runCadQuery = useCallback(async (query: string, fileId: string): Promise<Message | null> => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);

    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";

      abortControllerRef.current = new AbortController();

      const res = await fetch(`${base}/api/cad/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id:    fileId,
          query,
          session_id: sessionIdRef.current ?? undefined,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as any).detail ?? `HTTP ${res.status}`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let resultMsg: Message | null = null;
      const accumulatedSteps: ThinkingStep[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "status") {
              const step: ThinkingStep = { node: event.node, icon: event.icon, message: event.message, ts: Date.now() };
              accumulatedSteps.push(step);
              setThinkingSteps(prev => [...prev, step]);
            } else if (event.type === "result") {
              if (event.session_id) { setSessionId(event.session_id); sessionIdRef.current = event.session_id; loadDocuments(event.session_id); }
              resultMsg = {
                id:        Date.now().toString(),
                role:      "assistant",
                content:   event.answer,
                rawAnswer: event.answer,
                sources:   [],
                confidence: event.judge_score ?? 1.0,
                timestamp: new Date(),
              };
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          } catch { /* malformed SSE line */ }
        }
      }

      if (resultMsg) resultMsg = { ...resultMsg, thinkingSteps: accumulatedSteps };
      setThinkingSteps([]);
      return resultMsg;
    } catch (error) {
      setThinkingSteps([]);
      if (error instanceof Error && error.name === "AbortError") return null;
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  const inputOverrideRef = useRef<string | null>(null);

  // Sends a specific text string directly, bypassing the input state flush delay.
  // Used by expandSuggestion to auto-send immediately after expansion.
  const handleSendWithText = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    inputOverrideRef.current = trimmed;
    setInput(trimmed);
    // handleSend will pick up inputOverrideRef.current synchronously
    setTimeout(() => handleSendDirectRef.current?.(), 0);
  }, [isLoading]);

  // Stable ref that always points to the latest handleSend — avoids stale closures.
  const handleSendDirectRef = useRef<(() => Promise<void>) | null>(null);

  const handleSend = async () => {
    const rawInput = inputOverrideRef.current ?? input;
    inputOverrideRef.current = null;
    const trimmedInput = rawInput.trim();
    if (!trimmedInput || isLoading) return;

    // ── AUTH GATE — show popup if not logged in ──────────────────────────
    if (!currentUser) {
      showAuthModal(() => handleSend());
      return;
    }
    // ─────────────────────────────────────────────────────────────────────

    // ── Nav intent — detects topic interest and suggests the dedicated page naturally ──
    const _softNav = (() => {
      const q = trimmedInput.toLowerCase();

      const softPages = [
        {
          path: "/news",
          label: "News",
          icon: "📰",
          signals: [
            "news","latest news","headlines","breaking","article","articles",
            "feed","press","media","actualit","journal","nouvelles",
            "recent update","what's happening","what is happening","current event",
            "today's update","industry news","sector news","market news",
            "go to news","take me to news","open news","news page","show news",
          ],
          teaser: () =>
            `I can give you a quick take here, but honestly our **News page** has a dedicated agent built specifically for this — it tracks live updates, lets you filter by topic, and goes way deeper than I can in a chat window.\n\nWant me to take you there, or would you rather I answer from what I know right now?`,
        },
        {
          path: "/call",
          label: "Voice Call",
          icon: "📞",
          signals: [
            "voice","speak","talk","audio conversation","phone","hear me",
            "verbal","call with","speak with","voice mode","conversation vocale",
            "go to call","take me to call","open call","call page","start a call","make a call",
          ],
          teaser: () =>
            `Sounds like you'd prefer a voice conversation! There's a **Voice Call page** set up exactly for that — real-time audio with the agent, no typing needed.\n\nWant to head there, or are you good staying in chat?`,
        },
      ];

      for (const page of softPages) {
        if (page.signals.some(s => q.includes(s))) {
          return page;
        }
      }
      return null;
    })();

    if (_softNav) {
      const userId   = Date.now().toString();
      const assistId = (Date.now() + 1).toString();
      const userMsg: Message = {
        id: userId,
        role: "user",
        content: trimmedInput,
        timestamp: new Date(),
      };
      const assistantMsg: Message = {
        id: assistId,
        role: "assistant",
        content: _softNav.teaser(),
        rawAnswer: _softNav.teaser(),
        navAction: { path: _softNav.path, label: _softNav.label, icon: _softNav.icon },
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, userMsg, assistantMsg]);
      setInput("");
      if (!notifyDismissed) setShowNotifyBanner(true);
      // Typewriter effect — same as every other assistant reply
      setTypingMessageId(assistId);
      return;
    }

    ensureAudio();
    setOpenSourceKey(null);
    setSuggestions([]);
    setShowSilenceWarning(false);

    const convId = ensureActiveConversationId();
    const originMessages = messages;
    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: trimmedInput,
      timestamp: new Date(),
      attachedDocIds: pendingDocIds.length > 0 ? [...pendingDocIds] : undefined,
    };
    const userRawTitle = trimmedInput.length > 50 ? trimmedInput.slice(0, 50) + "…" : trimmedInput;
    const updatedUserMessages = [...originMessages, userMsg];
    setMessages(prev => [...prev, userMsg]);
    updateConversationMessages(convId, updatedUserMessages, "", userRawTitle);
    setInput("");
    setPendingDocIds([]);
    if (!notifyDismissed) setShowNotifyBanner(true);
    markPendingConversation(convId);
    setTypingConvId(convId);

    try {
      // ── Route: CAD/IFC agent if a CAD file is the active context ──────────
      // Priority 1: a CAD doc explicitly attached to this message (pendingDocIds)
      // Priority 2: user mentions the file by name or uses 3D/BIM keywords → use last uploaded CAD file
      const CAD_DOC_TYPES = new Set(["ifc", "cad", "dxf", "dwg", "step", "stp"]);

      const CAD_INTENT_KEYWORDS = [
        /\b(ifc|dxf|dwg|step|stp|bim|3d model|3d file|building model|ifc file|cad file)\b/i,
        /\b(wall|slab|beam|column|storey|floor|level|door|window|railing|roof|stair|footing|material|element)\b/i,
        /\b(how many|count|total|list|breakdown|summary|analyze|analyse|show me|tell me about).{0,30}\b(element|component|building|structure|model|file)\b/i,
      ];
      const cadKeywordMatch = CAD_INTENT_KEYWORDS.some(re => re.test(trimmedInput));

      const cadDocFilenames = documents
        .filter(d => CAD_DOC_TYPES.has((d.doc_type ?? "").toLowerCase()))
        .map(d => d.filename?.replace(/\.[^.]+$/, "").toLowerCase())
        .filter(Boolean);
      const mentionsFilename = cadDocFilenames.some(fn => fn && trimmedInput.toLowerCase().includes(fn));

      const activeCadFileId: string | null = (() => {
        // Priority 1: explicitly attached CAD doc
        const attachedCad = documents.find(
          d =>
            userMsg.attachedDocIds?.includes(d.document_id) &&
            CAD_DOC_TYPES.has((d.doc_type ?? "").toLowerCase())
        );
        if (attachedCad) return attachedCad.document_id;

        // Priority 2: BIM keyword or filename mention → route to last uploaded CAD file
        const lastId = (window as any).__lastCadFileId as string | undefined;
        if (lastId && (cadKeywordMatch || mentionsFilename)) {
          const hasCachedDoc = documents.some(
            d => d.document_id === lastId && CAD_DOC_TYPES.has((d.doc_type ?? "").toLowerCase())
          );
          if (hasCachedDoc) return lastId;
        }
        return null;
      })();

      const assistantMsg = activeCadFileId
        ? await runCadQuery(trimmedInput, activeCadFileId)
        : await runStreamingQuery(trimmedInput);
      if (!assistantMsg) return;
      const updatedMessages: Message[] = [...originMessages, userMsg, assistantMsg];
      const rawTitle = trimmedInput.length > 50 ? trimmedInput.slice(0, 50) + "…" : trimmedInput;
      const preview  = assistantMsg.content.replace(/\[.*?\]/g, "").replace(/#{1,3}\s/g, "").slice(0, 80) + "…";
      updateConversationMessages(convId, updatedMessages, preview, rawTitle);
      // Persist to DB (non-blocking)
      saveConversationToDB(convId, sessionIdRef.current ?? "", rawTitle, preview, updatedMessages);
      setTypingMessageId(assistantMsg.id);
      fetchSuggestions(trimmedInput, assistantMsg.content);
      fireNotification();
      if (activeConvIdRef.current !== convId) {
        toast({ title: "Response complete", description: "A response finished in another conversation.", });
      }
      // Smart conversation title:
      // • Report route → call /title with the prompt so the sidebar shows
      //   "Telecom Site Survey Field Results" instead of the raw user message.
      // • All other routes → call generateSmartTitle with the first few messages.
      if (assistantMsg.reportId) {
        try {
          const titleRes = await fetch(`${(typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000"}/title`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "report", text: trimmedInput }),
            signal: AbortSignal.timeout(8000),
          });
          if (titleRes.ok) {
            const { title: reportConvTitle } = await titleRes.json();
            if (reportConvTitle?.trim()) {
              setConversations(convs => convs.map(c => c.id === convId ? { ...c, title: reportConvTitle.trim() } : c));
            }
          }
        } catch { /* silent — title stays as raw input */ }
      } else {
        const currentTitle = conversations.find(c => c.id === convId)?.title ?? rawTitle;
        generateSmartTitle(convId, updatedMessages, currentTitle);
      }
    } catch (error) {
      const msg = serializeError(error);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${msg}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
      toast({ title: "Query failed", description: msg, variant: "destructive" });
    } finally {
      clearPendingConversation(convId);
      if (typingConvIdRef.current === convId) {
        setTypingConvId(null);
      }
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setTypingMessageId(null);
    setTypingConvId(null);
    setThinkingSteps([]);
    // Add a subtle interrupted indicator as an assistant message
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant") return prev;
      return [...prev, {
        id: Date.now().toString(),
        role: "assistant" as const,
        content: "⏸ Response stopped.",
        timestamp: new Date(),
        interrupted: true,
      }];
    });
  };

  // ── Filename autocomplete ─────────────────────────────────────────────────
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setInput(val);

    // Always clear ghost immediately; never show on empty input
    setWordSuffix("");
    if (ghostDebounceRef.current) clearTimeout(ghostDebounceRef.current);

    if (!val) {
      setAutocomplete(null);
      setAutocompletePos(null);
      return;
    }

    const caret = e.target.selectionStart ?? val.length;
    const before = val.slice(0, caret);
    const atIdx = before.lastIndexOf("@");
    if (atIdx !== -1) {
      const fragment = before.slice(atIdx + 1).toLowerCase();
      if (!fragment.includes(" ")) {
        const results = documents.filter(d =>
          d.filename.toLowerCase().includes(fragment)
        ).slice(0, 6);
        // Compute position from textarea
        const rect = e.target.getBoundingClientRect();
        setAutocompletePos({
          top: rect.top,        // will render above using transform
          left: rect.left,
          width: rect.width,
        });
        setAutocomplete({ query: fragment, triggerPos: atIdx, results, activeIdx: 0 });
        setWordSuffix("");
        return;
      }
    }
    setAutocomplete(null);
    setAutocompletePos(null);

    // ── AI ghost-text completion ─────────────────────────────────────────
    const trimmed = before.trim();
    if (trimmed.length >= 4 && !isLoading) {
      ghostDebounceRef.current = setTimeout(async () => {
        const base = (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000";
        const docNames = documents.map((d: { filename: string }) => d.filename);
        const suffix = await fetchAICompletion(before, messages, base, docNames);
        setWordSuffix(suffix);
      }, 280);
    }
  };

  const applyAutocomplete = (doc: Document) => {
    if (!autocomplete) return;
    const before = input.slice(0, autocomplete.triggerPos);
    const after = input.slice(autocomplete.triggerPos + 1 + autocomplete.query.length);
    setInput(before + doc.filename + after);
    setAutocomplete(null);
    setAutocompletePos(null);
    textareaRef.current?.focus();
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (autocomplete && autocomplete.results.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setAutocomplete(a => a ? { ...a, activeIdx: (a.activeIdx + 1) % a.results.length } : a);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setAutocomplete(a => a ? { ...a, activeIdx: (a.activeIdx - 1 + a.results.length) % a.results.length } : a);
        return;
      }
      if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        applyAutocomplete(autocomplete.results[autocomplete.activeIdx]);
        return;
      }
      if (e.key === "Escape") {
        setAutocomplete(null);
        setAutocompletePos(null);
        return;
      }
    }
    // ── Word-completion: Tab/→ accepts, Escape dismisses ──────────────────────
    if (wordSuffix) {
      if (e.key === "Tab" || e.key === "ArrowRight") {
        e.preventDefault();
        setInput(prev => prev + wordSuffix);
        setWordSuffix("");
        requestAnimationFrame(() => {
          const ta = textareaRef.current;
          if (ta) { const end = ta.value.length; ta.setSelectionRange(end, end); }
        });
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setWordSuffix("");
        return;
      }
      // Any navigation or editing key clears the ghost
      if (e.key === " " || e.key === "Enter" || e.key === "Backspace" ||
          e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "ArrowDown" ||
          e.key === "Home" || e.key === "End") {
        setWordSuffix("");
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Contextual suggestions (via backend /suggest → CF Worker) ───────────
  const fetchSuggestions = useCallback(async (userQuery: string, assistantReply: string) => {
    // Set loading FIRST so the skeleton appears immediately
    setSuggestionsLoading(true);
    setSuggestions([]);
    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";
      const res = await fetch(`${base}/suggest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_query:      userQuery.slice(0, 400),
          assistant_reply: assistantReply.slice(0, 800),
          available_docs:  documents.map(d => d.filename),
        }),
      });
      if (!res.ok) {
        console.warn(`[suggestions] /suggest returned ${res.status}`);
        setSuggestionsLoading(false);
        return;
      }
      const data = await res.json();
      const chips: string[] = Array.isArray(data.suggestions) ? data.suggestions : [];
      setSuggestions(chips.slice(0, 4).map((s: string) => String(s).slice(0, 50)));
    } catch (err) {
      console.warn("[suggestions] fetch failed:", err);
    } finally {
      setSuggestionsLoading(false);
    }
  }, []);

  // ── Expand a suggestion pill into a full polished prompt ─────────────────
  // Keeps the clicked pill visible with a spinner while the backend expands
  // the label, then auto-sends the resulting prompt so the user sees it work.
  const expandSuggestion = useCallback(async (label: string, isGeneral: boolean) => {
    setExpandingSuggestion(label);
    // Do NOT clear suggestions yet — keep the pill visible while loading
    let expandedPrompt = label;
    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";

      const lastAssistant = [...messages].reverse().find(m => m.role === "assistant");
      const lastUser      = [...messages].reverse().find(m => m.role === "user");

      const res = await fetch(`${base}/expand-suggestion`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label,
          is_general:      isGeneral,
          last_user_query: lastUser?.content?.slice(0, 300) ?? "",
          last_ai_reply:   lastAssistant?.content?.slice(0, 600) ?? "",
          available_docs:  documents.map(d => d.filename),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        expandedPrompt = data.prompt ?? label;
      }
    } catch {
      // fallback to raw label
    }
    // Hide pills, populate textarea, focus it
    setSuggestions([]);
    setExpandingSuggestion(null);
    setInput(expandedPrompt);
    textareaRef.current?.focus();
  }, [messages, documents]);

  const handleEditCancel = () => {
    setEditingMsgId(null);
    setEditDraft("");
  };

  const handleEditSubmit = async (msgId: string) => {
    const newContent = editDraft.trim();
    if (!newContent) return;

    const originConvId = activeConvIdRef.current;
    const originMessages = messages;

    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setTypingMessageId(null);
    setEditingMsgId(null);
    setEditDraft("");

    const msgIndex = originMessages.findIndex(m => m.id === msgId);
    if (msgIndex === -1) return;

    const updatedUserMsg: Message = { ...originMessages[msgIndex], content: newContent, timestamp: new Date() };
    if (activeConvIdRef.current === originConvId) {
      setMessages(prev => [...prev.slice(0, msgIndex), updatedUserMsg]);
    }

    try {
      setTypingConvId(originConvId);
      const assistantMsg = await runStreamingQuery(newContent);
      if (!assistantMsg) return;
      const existingConv = conversations.find(c => c.id === originConvId);
      const updatedMessages = [...originMessages.slice(0, msgIndex), updatedUserMsg, assistantMsg];
      updateConversationMessages(
        originConvId,
        updatedMessages,
        existingConv?.preview ?? "",
        existingConv?.title
      );
      setTypingMessageId(assistantMsg.id);
    } catch (error) {
      const msg = serializeError(error);
      const errorMsg: Message = { id: Date.now().toString(), role: "assistant", content: `Sorry: ${msg}`, timestamp: new Date() };
      if (activeConvIdRef.current === originConvId) {
        setMessages(prev => [...prev, errorMsg]);
      }
      toast({ title: "Query failed", description: msg, variant: "destructive" });
    } finally {
      if (typingConvIdRef.current === originConvId) {
        setTypingConvId(null);
      }
    }
  };

  const handleRedo = async (msgId: string) => {
    const originConvId = activeConvIdRef.current;
    const originMessages = messages;

    const msgIndex = originMessages.findIndex(m => m.id === msgId);
    if (msgIndex < 1) return;
    const prevUserMsg = [...originMessages].slice(0, msgIndex).reverse().find(m => m.role === "user");
    if (!prevUserMsg) return;

    setThinkingSteps([]);
    setThinkingExpanded(true);
    const cleanedMessages = originMessages.filter(m => m.id !== msgId);
    if (activeConvIdRef.current === originConvId) {
      setMessages(cleanedMessages);
    }

    try {
      setTypingConvId(originConvId);
      const existingConv = conversations.find(c => c.id === originConvId);
      const redoMsg = await runStreamingQuery(prevUserMsg.content);
      if (!redoMsg) return;
      const updatedMessages = [...cleanedMessages, redoMsg];
      updateConversationMessages(
        originConvId,
        updatedMessages,
        existingConv?.preview ?? "",
        existingConv?.title
      );
      if (activeConvIdRef.current === originConvId) {
        setTypingMessageId(redoMsg.id);
      }
      fireNotification();
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        const msg = serializeError(error);
        toast({ title: "Regeneration failed", description: msg, variant: "destructive" });
      }
    } finally {
      if (typingConvIdRef.current === originConvId) {
        setTypingConvId(null);
      }
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
      style={{ ...(isDark && { background: "#07080f" }), transition: "background 0.15s ease" }}
      onDragEnter={e => { e.preventDefault(); dragCounterRef.current++; setIsDragOver(true); }}
      onDragLeave={e => { e.preventDefault(); dragCounterRef.current--; if (dragCounterRef.current === 0) setIsDragOver(false); }}
      onDragOver={e => e.preventDefault()}
      onDrop={e => { e.preventDefault(); dragCounterRef.current = 0; setIsDragOver(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
    >

      {/* ── Drag-and-drop overlay (page-level) ── */}
      <AnimatePresence>
        {isDragOver && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="absolute inset-0 z-[9999] bg-background/50 backdrop-blur-[3px] flex items-center justify-center pointer-events-none"
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
              className="flex flex-col items-center gap-2 px-8 py-6 rounded-2xl border border-primary/30 bg-card/90 shadow-2xl"
            >
              <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                <Plus className="h-5 w-5 text-primary" />
              </div>
              <p className="text-sm font-medium text-foreground">Drop to upload</p>
              <p className="text-[11px] text-muted-foreground/60">PDF · DOCX · TXT · PNG · JPG · WEBP · IFC · DWG · DXF · STEP</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-14 border-b border-border flex items-center px-4 gap-2 shrink-0 bg-background">
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
                onClick={() => {
                  const next = !convsPanelOpen;
                  setConvsPanelOpen(next);
                  if (next) loadConversationsFromDB();
                }}
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
                                  {pendingConvIds[conv.id] && conv.id !== activeConvId ? (
                                    <span className="mr-2 inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
                                  ) : null}
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

            {/* ── Reports bubble ── */}
            {reports.length > 0 && (
            <div
              className="relative z-[9999]"
              data-reports-panel
              ref={reportsPanelRef}
            >
                {/* Pill button */}
                <button
                  onClick={() => {
                    const next = !reportsPanelOpenRef.current;
                    reportsPanelOpenRef.current = next;
                    setReportsPanelOpen(next);
                    if (!next) { setActiveReport(null); setReportEditMode(false); setPreviewedVersion(null); }
                  }}
                  className={`relative flex items-center gap-2 pl-3 pr-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
                    reportsPanelOpen
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
                  }`}
                  title="Generated Reports"
                >
                  <BookOpen className="h-3.5 w-3.5 shrink-0" />
                  <span>Reports</span>
                  <span className={`inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-bold ${
                    reportsPanelOpen ? "bg-primary-foreground/20 text-primary-foreground" : "bg-primary/15 text-primary"
                  }`}>
                    {reports.length}
                  </span>
                  {isGeneratingReport && (
                    <Loader2 className="h-3 w-3 animate-spin ml-0.5" />
                  )}
                </button>
                <div
                  className="absolute right-0 top-full mt-2 bg-background border border-border rounded-2xl shadow-2xl z-[9999] flex flex-col"
                  style={{
                    transformOrigin: "top right",
                    width:     activeReport ? 540 : 300,
                    height:    activeReport ? 620 : "auto",
                    maxHeight: activeReport ? 620 : 440,
                    overflow:  activeReport ? "hidden" : "visible",
                    transition: "width 0.22s cubic-bezier(0.4,0,0.2,1), height 0.22s cubic-bezier(0.4,0,0.2,1), opacity 0.18s, transform 0.18s",
                    opacity:       reportsPanelOpen ? 1 : 0,
                    transform:     reportsPanelOpen ? "scale(1) translateY(0)" : "scale(0.92) translateY(-6px)",
                    pointerEvents: reportsPanelOpen ? "auto" : "none",
                    minHeight: 0,
                  }}
                >
                  {/* ── List view ── */}
                  {!activeReport && (
                    <>
                      <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
                        <div className="flex items-center gap-2">
                          <BookOpen className="h-4 w-4 text-primary" />
                          <span className="text-sm font-semibold text-foreground">Generated Reports</span>
                          <span className="text-xs text-muted-foreground">({reports.length})</span>
                        </div>
                        <button onClick={() => { reportsPanelOpenRef.current = false; setReportsPanelOpen(false); }} className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors">
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>

                      <div className="max-h-72 overflow-y-auto scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                        <div className="p-2 space-y-0.5">
                          {reports.map(report => (
                            <div
                              key={report.report_id}
                              className="group flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-muted/60 cursor-pointer transition-colors"
                              onClick={async () => {
                                // List items are slim (no charts array) — fetch full record before opening
                                try {
                                  const res = await fetch(`${getApiBase()}/reports/${report.report_id}`);
                                  if (res.ok) setActiveReport(await res.json());
                                  else setActiveReport(report);
                                } catch { setActiveReport(report); }
                              }}
                            >
                              <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                                {(report.charts?.length ?? 0) > 0
                                  ? <BarChart2 className="h-3.5 w-3.5 text-primary" />
                                  : <FileText className="h-3.5 w-3.5 text-primary" />
                                }
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-xs font-medium text-foreground truncate">{report.title}</p>
                                <p className="text-[10px] text-muted-foreground flex items-center gap-1.5">
                                  {new Date(report.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                                  {report.version > 1 && <span className="text-primary/60">v{report.version}</span>}
                                  {(report.charts?.length ?? 0) > 0 && (
                                    <span className="inline-flex items-center gap-0.5 text-primary/70">
                                      <BarChart2 className="h-2.5 w-2.5" />{report.charts.length}
                                    </span>
                                  )}
                                </p>
                              </div>
                              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all shrink-0">
                                <Eye className="h-3 w-3 text-muted-foreground/40" />
                                <button
                                  onClick={e => { e.stopPropagation(); handleDownloadReport(report, "pdf"); }}
                                  className="p-0.5 rounded text-muted-foreground/50 hover:text-primary transition-colors"
                                  title="Download"
                                >
                                  {downloadingReportId === report.report_id
                                    ? <Loader2 className="h-3 w-3 animate-spin" />
                                    : <ScrollText className="h-3 w-3" />
                                  }
                                </button>
                                <button
                                  onClick={e => { e.stopPropagation(); handleDeleteReport(report.report_id); }}
                                  className="p-0.5 rounded text-muted-foreground/50 hover:text-destructive transition-colors"
                                  title="Delete"
                                >
                                  {deletingReportId === report.report_id
                                    ? <Loader2 className="h-3 w-3 animate-spin" />
                                    : <Trash2 className="h-3 w-3" />
                                  }
                                </button>
                              </div>
                              <ChevronRight className="h-3 w-3 text-muted-foreground/30 shrink-0" />
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Footer hint */}
                      <div className="px-4 py-3 border-t border-border shrink-0">
                        <p className="text-[10px] text-muted-foreground/60 text-center flex items-center justify-center gap-1">
                          <Sparkles className="h-2.5 w-2.5" />
                          Ask the AI to "generate a report on…" to create one
                        </p>
                      </div>
                    </>
                  )}

                  {/* ── Expanded report viewer ── */}
                  {activeReport && (
                    <>
                      {/* Viewer header */}
                      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border shrink-0">
                        <button
                          onClick={() => { setActiveReport(null); setReportEditMode(false); setReportEditInstruction(""); setShowVersionHistory(false); setPreviewedVersion(null); }}
                          className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors shrink-0"
                          title="Back to reports"
                        >
                          <ArrowLeft className="h-3.5 w-3.5" />
                        </button>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-foreground truncate">
                            {previewedVersion ? previewedVersion.title : activeReport.title}
                          </p>
                          <p className="text-[10px] text-muted-foreground flex items-center gap-1.5">
                            {new Date(activeReport.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
                            {previewedVersion ? (
                              <span className="text-amber-500/80 font-medium">viewing v{previewedVersion.version}</span>
                            ) : (
                              activeReport.version > 1 && <span className="text-primary/60">v{activeReport.version}</span>
                            )}
                            {isPatchingReport && (
                              <span className="inline-flex items-center gap-1 text-primary animate-pulse">
                                <RefreshCw className="h-2.5 w-2.5 animate-spin" />editing…
                              </span>
                            )}
                          </p>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          {/* Version history toggle */}
                          {activeReport.versions && activeReport.versions.length > 1 && (
                            <button
                              onClick={() => setShowVersionHistory(v => !v)}
                              className={`p-1.5 rounded-lg transition-colors text-[10px] flex items-center gap-1 ${showVersionHistory ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-muted/50"}`}
                              title="Version history"
                            >
                              <Clock className="h-3.5 w-3.5" />
                              <span className="hidden sm:inline">v{activeReport.version}</span>
                            </button>
                          )}
                          {/* PDF download */}
                          <button
                            onClick={() => handleDownloadReport(activeReport, "pdf")}
                            className="p-1.5 rounded-lg text-muted-foreground hover:text-primary hover:bg-muted/50 transition-colors"
                            title="Download PDF"
                          >
                            {downloadingReportId === activeReport.report_id
                              ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              : <ScrollText className="h-3.5 w-3.5" />
                            }
                          </button>
                          <button
                            onClick={() => handleDeleteReport(activeReport.report_id)}
                            className="p-1.5 rounded-lg text-muted-foreground hover:text-destructive hover:bg-muted/50 transition-colors"
                            title="Delete report"
                          >
                            {deletingReportId === activeReport.report_id
                              ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              : <Trash2 className="h-3.5 w-3.5" />
                            }
                          </button>
                          <button
                            onClick={() => { reportsPanelOpenRef.current = false; setReportsPanelOpen(false); setActiveReport(null); setReportEditMode(false); setShowVersionHistory(false); setPreviewedVersion(null); }}
                            className="p-1.5 rounded-lg text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>

                      {/* Source doc tags */}
                      {(activeReport.source_docs?.length ?? 0) > 0 && (
                        <div className="flex flex-wrap gap-1 px-4 py-2 border-b border-border shrink-0">
                          {activeReport.source_docs.slice(0, 5).map(doc => (
                            <span key={doc} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-medium">
                              <FileText className="h-2.5 w-2.5" />
                              {doc.length > 22 ? doc.slice(0, 20) + "…" : doc}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* ── Version history drawer ── */}
                      {showVersionHistory && activeReport.versions && activeReport.versions.length > 1 && (
                        <div className="border-b border-border shrink-0">
                          {/* Header */}
                          <div className="px-4 py-2 flex items-center justify-between bg-muted/30">
                            <span className="text-[10px] font-semibold text-foreground/70 uppercase tracking-widest flex items-center gap-1.5">
                              <Clock className="h-3 w-3 text-primary" />
                              Version History
                            </span>
                            <div className="flex items-center gap-2">
                              {previewedVersion && (
                                <button
                                  onClick={() => setPreviewedVersion(null)}
                                  className="text-[10px] text-primary hover:text-primary/80 font-medium flex items-center gap-1 transition-colors"
                                >
                                  <ArrowLeft className="h-2.5 w-2.5" />
                                  Back to current
                                </button>
                              )}
                              <span className="text-[9px] text-muted-foreground/60 bg-muted px-1.5 py-0.5 rounded-full">
                                {activeReport.versions.length} versions
                              </span>
                            </div>
                          </div>

                          {/* Horizontal scrollable version cards */}
                          <div className="flex gap-2 px-4 py-3 overflow-x-auto scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                            {[...activeReport.versions].reverse().map(snap => {
                              const isLive       = snap.version === activeReport.version;
                              const isPreviewing = previewedVersion?.version === snap.version;
                              const isActive     = isPreviewing || (isLive && !previewedVersion);
                              const isRestoring  = restoringVersion === snap.version;
                              return (
                                <button
                                  key={snap.version}
                                  disabled={isRestoring}
                                  onClick={async () => {
                                    if (isRestoring) return;
                                    // Already viewing this card — no-op
                                    if (isActive) return;
                                    // Clicking the live version while previewing something else → go back to live
                                    if (isLive) { setPreviewedVersion(null); return; }
                                    // Fetch + preview a past version
                                    try {
                                      const res = await fetch(`${getApiBase()}/reports/${activeReport.report_id}/versions/${snap.version}`);
                                      if (res.ok) {
                                        const full = await res.json();
                                        setPreviewedVersion({ version: snap.version, content: full.content, charts: full.charts ?? [], title: full.title });
                                      }
                                    } catch { /* ignore */ }
                                  }}
                                  className={[
                                    "shrink-0 flex flex-col items-start gap-1 px-3 py-2.5 rounded-xl border text-left w-44",
                                    "transition-all duration-150",
                                    isActive
                                      ? "bg-primary/10 border-primary/40 shadow-sm ring-1 ring-primary/20"
                                      : "bg-muted/40 border-border/60 hover:bg-muted/70 hover:border-border",
                                    isRestoring ? "opacity-50 cursor-wait" : "cursor-pointer",
                                  ].join(" ")}
                                >
                                  {/* Badge row */}
                                  <div className="flex items-center gap-1.5 w-full">
                                    <span className={[
                                      "inline-flex items-center justify-center h-5 w-5 rounded-full text-[9px] font-bold shrink-0",
                                      isActive
                                        ? "bg-primary text-primary-foreground"
                                        : "bg-muted-foreground/15 text-muted-foreground border border-border",
                                    ].join(" ")}>
                                      {snap.version}
                                    </span>
                                    <span className={[
                                      "text-[11px] font-semibold truncate flex-1",
                                      isActive ? "text-primary" : "text-foreground/70",
                                    ].join(" ")}>
                                      {isLive ? "Current" : `v${snap.version}`}
                                    </span>
                                    {isRestoring
                                      ? <Loader2 className="h-3 w-3 animate-spin text-primary shrink-0" />
                                      : isLive && (
                                        <span className="shrink-0 text-[8px] bg-primary/15 text-primary px-1.5 py-0.5 rounded-full font-medium">live</span>
                                      )
                                    }
                                  </div>

                                  {/* Title */}
                                  <p className="text-[10px] text-muted-foreground leading-snug line-clamp-1 w-full text-left">
                                    {snap.title}
                                  </p>

                                  {/* Date */}
                                  <p className="text-[9px] text-muted-foreground/60 leading-snug">
                                    {new Date(snap.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                                  </p>

                                  {/* Edit instruction snippet */}
                                  {snap.instruction && (
                                    <p className="text-[9px] text-muted-foreground/50 italic leading-snug line-clamp-2 w-full text-left">
                                      "{snap.instruction.slice(0, 50)}{snap.instruction.length > 50 ? "…" : ""}"
                                    </p>
                                  )}

                                  {/* Restore button — only for non-live, non-restoring */}
                                  {!isLive && !isRestoring && (
                                    <button
                                      onClick={e => { e.stopPropagation(); handleRestoreVersion(activeReport, snap.version); }}
                                      className={[
                                        "mt-1 w-full text-[9px] font-medium px-2 py-1 rounded-lg border transition-colors",
                                        isActive
                                          ? "border-primary/30 text-primary hover:bg-primary/15"
                                          : "border-border/50 text-muted-foreground/60 hover:text-primary hover:border-primary/30",
                                      ].join(" ")}
                                      title="Make this the active version"
                                    >
                                      ↩ Restore
                                    </button>
                                  )}
                                </button>
                              );
                            })}
                          </div>

                          {/* Preview mode banner */}
                          {previewedVersion && (
                            <div className="mx-4 mb-3 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/8 border border-amber-500/20">
                              <Eye className="h-3 w-3 shrink-0 text-amber-500/70" />
                              <span className="text-[10px] font-medium flex-1 text-amber-600 dark:text-amber-400">
                                Previewing v{previewedVersion.version} — read-only
                              </span>
                              <button
                                onClick={() => setPreviewedVersion(null)}
                                className="text-[10px] text-amber-600/80 dark:text-amber-400/80 underline underline-offset-2 hover:opacity-100 opacity-70 transition-opacity"
                              >
                                Return to current
                              </button>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Report content */}
                      <div className={`flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent min-h-0 px-4 py-3 transition-opacity duration-200 ${previewedVersion ? "opacity-80" : ""}`}>
                        {renderReportContent(previewedVersion
                          ? { ...activeReport, content: previewedVersion.content, charts: previewedVersion.charts, title: previewedVersion.title }
                          : activeReport
                        )}
                      </div>

                      {/* Edit bar */}
                      <div className="px-3 py-2.5 border-t border-border shrink-0">
                        {!reportEditMode ? (
                          <button
                            onClick={() => setReportEditMode(true)}
                            className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground px-2 py-1 rounded-lg hover:bg-muted/60 transition-colors w-full justify-center"
                          >
                            <Pencil className="h-3 w-3" />
                            Edit report
                          </button>
                        ) : (
                          <div className="flex items-end gap-2 w-full bg-muted/40 rounded-xl border border-border px-3 py-2">
                            <textarea
                              ref={reportEditInputRef}
                              value={reportEditInstruction}
                              onChange={e => setReportEditInstruction(e.target.value)}
                              onKeyDown={e => {
                                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handlePatchReport(); }
                                if (e.key === "Escape") { setReportEditMode(false); setReportEditInstruction(""); }
                              }}
                              placeholder='e.g. "Add a conclusion" or "Edit the budget section to include Q3 data"'
                              rows={2}
                              className="flex-1 resize-none bg-transparent text-xs text-foreground placeholder:text-muted-foreground/50 focus:outline-none"
                            />
                            <div className="flex items-center gap-1 shrink-0">
                              <button
                                onClick={() => { setReportEditMode(false); setReportEditInstruction(""); }}
                                className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
                              >
                                <X className="h-3 w-3" />
                              </button>
                              <button
                                onClick={handlePatchReport}
                                disabled={!reportEditInstruction.trim() || isPatchingReport}
                                className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-primary text-primary-foreground text-[11px] font-medium disabled:opacity-50 transition-colors hover:bg-primary/90"
                              >
                                {isPatchingReport ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                                Apply
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

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
                                const isUploadingDoc = doc.doc_type === "uploading";
                                return (
                                  <div
                                    key={doc.document_id}
                                    className={`group flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors ${isUploadingDoc ? 'opacity-80 cursor-default' : 'hover:bg-muted/60 cursor-pointer'}`}
                                    onClick={() => { if (!isUploadingDoc) openBubbleDoc(doc); }}
                                  >
                                    <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                                      {isUploadingDoc ? <Loader2 className="h-3.5 w-3.5 text-primary animate-spin" /> : isImg ? <ImageIcon className="h-3.5 w-3.5 text-primary" /> : <FileText className="h-3.5 w-3.5 text-primary" />}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                      <p className="text-xs font-medium text-foreground truncate">{doc.filename}</p>
                                      <p className="text-[10px] text-muted-foreground capitalize">{isUploadingDoc ? 'uploading...' : doc.doc_type}</p>
                                    </div>
                                    {!isUploadingDoc && (
                                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                                        <Eye className="h-3 w-3 text-muted-foreground/50" />
                                        <button
                                          onClick={e => { e.stopPropagation(); removeDocument(doc.document_id); }}
                                          className="p-0.5 rounded text-muted-foreground/50 hover:text-destructive transition-colors"
                                        >
                                          <X className="h-3 w-3" />
                                        </button>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
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
                            ) : bubbleViewer.mediaType === 'cad' ? (
                              <div className="h-full overflow-y-auto p-3 space-y-3 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent">
                                {bubbleViewer.cadSummary && (() => {
                                  const ps = bubbleViewer.cadSummary as any;
                                  const pl = (ps.pipeline ?? '').toLowerCase();
                                  const fn = bubbleDoc.filename;
                                  const ext = fn.split('.').pop()?.toLowerCase() ?? '';
                                  const viewerBlobUrl = bubbleViewer.ifcBlobUrl ?? (pl === 'ifc' ? bubbleViewer.blobUrl : null);
                                  const viewerFilename = bubbleViewer.ifcBlobUrl
                                    ? fn.replace(/\.(dxf|dwg|step|stp|rvt|nwd|nwc|dgn|skp|fbx|obj|stl|sat|iges|igs|prt|sldprt|catpart|3ds|dae|rfa|rte)$/i, '.ifc') || 'model.ifc'
                                    : fn;
                                  if (viewerBlobUrl) {
                                    return <XeokitViewer blobUrl={viewerBlobUrl} filename={viewerFilename} pipeline={pl} />;
                                  }
                                  if (ext === 'dxf' && bubbleViewer.blobUrl) {
                                    return <XeokitViewer blobUrl={bubbleViewer.blobUrl} filename={fn} pipeline={pl} />;
                                  }
                                  return (
                                    <div className="w-full rounded-lg bg-muted/30 border border-border flex flex-col items-center justify-center gap-2 p-4 text-center" style={{ height: 160 }}>
                                      <svg className="h-6 w-6 text-muted-foreground/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" /></svg>
                                      <p className="text-[11px] text-muted-foreground/60">3D preview unavailable — AI analysis via chat ↓</p>
                                    </div>
                                  );
                                })()}
                                {bubbleViewer.cadSummary && (() => {
                                  const s = bubbleViewer.cadSummary as any;
                                  const pipeline = (s.pipeline ?? 'cad').toUpperCase();
                                  const isIfc = pipeline === 'IFC';
                                  return (
                                    <div className="space-y-2 text-xs">
                                      <div className="flex items-center gap-2">
                                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-primary/15 text-primary border border-primary/30">{pipeline}</span>
                                        {s.schema && <span className="text-muted-foreground">Schema: {s.schema}</span>}
                                        {s.dimension && <span className="text-muted-foreground">{s.dimension}</span>}
                                      </div>
                                      {s.ux_hint && <p className="text-primary/80 bg-primary/8 border border-primary/20 rounded px-2 py-1.5 leading-relaxed">{s.ux_hint}</p>}
                                      {s.total_elements != null && <div className="bg-muted/50 rounded p-2"><p className="text-[10px] text-muted-foreground uppercase tracking-wide">Elements</p><p className="text-lg font-bold">{s.total_elements.toLocaleString()}</p></div>}
                                      {isIfc && s.storeys?.length > 0 && (
                                        <div>
                                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">Storeys</p>
                                          {s.storeys.map((st: any, i: number) => (
                                            <div key={i} className="flex justify-between bg-muted/40 rounded px-2 py-1 mb-0.5">
                                              <span className="font-medium">{st.name ?? `Storey ${i+1}`}</span>
                                              {st.elevation != null && <span className="text-muted-foreground">{typeof st.elevation === 'number' ? st.elevation.toFixed(2) : st.elevation} m</span>}
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                      {s.element_counts && Object.keys(s.element_counts).length > 0 && (
                                        <div>
                                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">Element Types</p>
                                          {Object.entries(s.element_counts as Record<string,number>).sort(([,a],[,b])=>b-a).map(([label,count])=>(
                                            <div key={label} className="flex justify-between bg-muted/40 rounded px-2 py-1 mb-0.5">
                                              <span className="capitalize">{label}</span>
                                              <span className="text-primary font-semibold">{count}</span>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                      {s.layers?.length > 0 && (
                                        <div>
                                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">Layers ({s.layers.length})</p>
                                          {s.layers.slice(0,12).map((layer: any, i: number) => (
                                            <div key={i} className="flex justify-between bg-muted/40 rounded px-2 py-1 mb-0.5 font-mono">
                                              <span>{layer.name ?? layer}</span>
                                              {layer.entity_count != null && <span className="text-muted-foreground">{layer.entity_count}</span>}
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  );
                                })()}
                              </div>
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
            {/* ── Call pill ── */}
            <button
              type="button"
              onClick={() => {
                // Pass the current session so CallPage shares history with this chat
                navigate("/call", {
                  state: {
                    sessionId: sessionIdRef.current,
                    convId: activeConvId,
                  },
                });
              }}
              className="flex items-center gap-1.5 pl-3 pr-3.5 py-1.5 rounded-full text-xs font-medium border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500/60 transition-all"
            >
              <Phone className="h-3.5 w-3.5 shrink-0" />
              <span>Call</span>
            </button>
            <ThemeToggle />
            {currentUser && (
              <ProfileBubble user={currentUser} onLogout={logout} align="right" />
            )}
            {!currentUser && (
              <button
                onClick={() => showAuthModal()}
                style={{
                  fontSize: 12, fontWeight: 600, padding: "5px 12px",
                  borderRadius: 8, border: "1px solid rgba(96,165,250,0.35)",
                  background: "rgba(96,165,250,0.08)", color: "#60a5fa",
                  cursor: "pointer",
                }}
              >
                Log in
              </button>
            )}
          </div>
        </header>

        {/* Silence warning banner */}
        <AnimatePresence>
          {showSilenceWarning && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="fixed top-14 left-0 right-0 flex justify-center pt-2 z-50 pointer-events-none"
            >
              <div className="pointer-events-auto inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-950 border border-red-500/50 text-red-400 text-xs font-medium shadow-sm">
                <span>😢</span>
                <span>We're having trouble hearing you — check your mic is connected and unmuted.</span>
                <button onClick={() => setShowSilenceWarning(false)} className="ml-1 hover:text-red-300 transition-colors">
                  <X className="h-3 w-3" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent hover:scrollbar-thumb-muted-foreground/40 relative"
          onDragEnter={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current++; setIsChatDragOver(true); }}
          onDragLeave={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current--; if (chatDragCounterRef.current === 0) setIsChatDragOver(false); }}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current = 0; setIsChatDragOver(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
        >
          <AnimatePresence>
            {isChatDragOver && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="absolute inset-0 z-50 bg-background/60 backdrop-blur-[2px] flex items-center justify-center pointer-events-none rounded-lg"
              >
                <motion.div
                  initial={{ scale: 0.92, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.92, opacity: 0 }}
                  transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
                  className="flex flex-col items-center gap-2 px-8 py-6 rounded-2xl border border-primary/30 bg-card/90 shadow-2xl"
                >
                  <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <Plus className="h-5 w-5 text-primary" />
                  </div>
                  <p className="text-sm font-medium text-foreground">Drop to upload</p>
                  <p className="text-[11px] text-muted-foreground/60">PDF · DOCX · TXT · PNG · JPG · WEBP · IFC · DWG · DXF · STEP</p>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
          <WelcomeSplash visible={messages.length === 0 && !convLoading && !activeConvId} />
          <AnimatePresence mode="wait">
            {convLoading ? (
              <motion.div
                key="conv-loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none"
              >
                <Loader2 className="h-6 w-6 animate-spin text-primary/50" />
              </motion.div>
            ) : (
              <motion.div
                key={activeConvId || "empty"}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                className="max-w-3xl mx-auto space-y-6"
              >
          {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`relative z-10 flex items-end ${msg.role === "user" ? "justify-end gap-2" : "gap-3 items-start"}`}
              >

                <div className={`group/msg relative ${msg.role === "user" ? "max-w-[80%] flex flex-col items-end gap-0.5" : "max-w-[80%] space-y-2"}`}>
                  {/* Persisted thinking steps — shown above the bubble for completed assistant messages */}
                  {msg.role === "assistant" && msg.thinkingSteps && msg.thinkingSteps.length > 0 && (() => {
                    const steps = msg.thinkingSteps;
                    const stepKey = `thinking-${msg.id}`;
                    const isOpen = openSourceKey === stepKey;
                    const cls = "h-3 w-3 shrink-0 text-muted-foreground/35";
                    return (
                      <div className="flex flex-col gap-1 pl-0 mb-1">
                        <button
                          onClick={() => setOpenSourceKey(k => k === stepKey ? null : stepKey)}
                          className="flex items-center gap-1.5 group/think w-fit"
                        >
                          <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary/30 shrink-0" />
                          <span className="text-[11px] text-muted-foreground/40 italic leading-none">
                            {steps[steps.length - 1]?.message ?? "Thought process"}
                          </span>
                          <motion.span
                            animate={{ rotate: isOpen ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="opacity-0 group-hover/think:opacity-60 transition-opacity"
                          >
                            <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
                          </motion.span>
                        </button>
                        <AnimatePresence>
                          {isOpen && steps.length > 0 && (
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
                  })()}
                  {/* ── Voice message — rendered directly, bypasses group/bubble wrapper ── */}
                  {msg.callCard ? (
                    /* ── Call card — Messenger-style "You called" summary ── */
                    <div className="flex flex-col items-end gap-1">
                      <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-2xl rounded-br-md bg-emerald-500/12 border border-emerald-500/25 w-fit">
                        <div className="flex items-center justify-center h-7 w-7 rounded-full bg-emerald-500/20 shrink-0">
                          <Phone className="h-3.5 w-3.5 text-emerald-400" />
                        </div>
                        <div className="flex flex-col gap-0.5">
                          <span className="text-[13px] font-medium text-emerald-300 leading-none">Voice call</span>
                          <span className="text-[11px] text-emerald-400/60 tabular-nums">
                            {Math.floor(msg.callCard.duration / 60)}:{String(msg.callCard.duration % 60).padStart(2, "0")}
                            {" · "}
                            {msg.callCard.startedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : msg.voiceBlobUrl ? (
                    <VoiceMessageBubble
                      blobUrl={msg.voiceBlobUrl}
                      duration={msg.voiceDuration ?? 0}
                      transcript={msg.voiceTranscript}
                      isExpanded={expandedTranscripts.has(msg.id)}
                      waveform={msg.voiceWaveform}
                      timestamp={msg.timestamp}
                      onToggleTranscript={() =>
                        setExpandedTranscripts(prev => {
                          const next = new Set(prev);
                          if (next.has(msg.id)) next.delete(msg.id);
                          else next.add(msg.id);
                          return next;
                        })
                      }
                    />
                  ) : (
                  <>
                  {/* Attached docs — shown above the bubble for user messages */}
                  {msg.role === "user" && msg.attachedDocIds && msg.attachedDocIds.length > 0 && (
                    <div className="flex flex-wrap justify-end gap-1.5 mb-1">
                      {msg.attachedDocIds.map(docId => {
                        const doc = documents.find(d => d.document_id === docId);
                        if (!doc) return null;
                        const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
                        const isImg = ['png','jpg','jpeg','webp','gif'].includes(ext);
                        const isIfc = ['ifc','ifczip'].includes(ext);
                        const isCad = ['dwg','dxf','step','stp'].includes(ext);
                        const isPdf = ext === 'pdf';
                        const cached = blobUrlMapRef.current.get(doc.document_id);
                        return (
                          <button
                            key={docId}
                            onClick={() => openBubbleDoc(doc)}
                            className="group flex items-center gap-2 pl-1.5 pr-3 py-1.5 rounded-xl bg-card border border-border hover:border-primary/40 hover:bg-primary/5 transition-all text-left shadow-sm"
                            title={doc.filename}
                          >
                            <div className="w-7 h-7 rounded-md overflow-hidden bg-muted/60 flex items-center justify-center shrink-0 border border-border/50">
                              {isImg && cached?.url ? (
                                <img src={cached.url} alt="" className="w-full h-full object-cover" />
                              ) : isIfc ? (
                                <span className="text-[8px] font-bold text-primary">IFC</span>
                              ) : isCad ? (
                                <span className="text-[8px] font-bold text-orange-400">{ext.toUpperCase()}</span>
                              ) : isPdf ? (
                                <span className="text-[8px] font-bold text-red-400">PDF</span>
                              ) : (
                                <FileText className="h-3 w-3 text-muted-foreground/60" />
                              )}
                            </div>
                            <span className="text-[10px] text-foreground/70 font-medium max-w-[90px] truncate group-hover:text-foreground transition-colors">
                              {(() => { const base = ext ? doc.filename.slice(0, doc.filename.length - ext.length - 1) : doc.filename; return base.length > 8 ? base.slice(0, 8) + '...' + (ext ? '.' + ext : '') : doc.filename; })()}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                  <div
                    className={`group/bubble relative z-10 px-4 rounded-2xl text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "py-3 bg-primary text-primary-foreground rounded-br-md w-fit break-words min-w-0 max-w-full"
                        : msg.interrupted
                          ? "px-3 py-2 bg-transparent border border-dashed border-muted-foreground/20 rounded-bl-md w-fit"
                          : msg.analytics?.type === "chart_clarification" || msg.analytics?.type === "report_chart_clarification"
                            ? "pt-3 pb-3 bg-secondary text-secondary-foreground rounded-bl-md w-fit"
                            : "py-3 bg-secondary text-secondary-foreground rounded-bl-md w-fit"
                    }`}
                  >
                  {/* Copy button — floats right and sticks as you scroll long answers */}
                  {msg.role === "assistant" && !msg.interrupted && msg.analytics?.type !== "chart_clarification" && msg.analytics?.type !== "report_chart_clarification" && msg.analytics?.type !== "chart_config" && msg.analytics?.type !== "chart_error" && (
                    <button
                      onClick={() => {
                        const text = msg.rawAnswer ?? msg.content;
                        navigator.clipboard.writeText(
                          text.replace(/\[\d+\]/g, "").replace(/#{1,3}\s/g, "").trim()
                        );
                        setCopiedMsgId(msg.id);
                        setTimeout(() => setCopiedMsgId(null), 1500);
                      }}
                      className={`sticky top-2 float-right ml-2 -mr-1 flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[11px] font-medium transition-all duration-150 z-10 ${
                        copiedMsgId === msg.id
                          ? "opacity-100 bg-primary/15 text-primary"
                          : "opacity-0 group-hover/msg:opacity-100 bg-muted/80 text-muted-foreground hover:text-foreground hover:bg-muted"
                      }`}
                      title="Copy response"
                    >
                      {copiedMsgId === msg.id
                        ? <><Check className="h-3 w-3" /><span>Copied</span></>
                        : <><Copy className="h-3 w-3" /><span>Copy</span></>}
                    </button>
                  )}

                    {msg.role === "assistant" && msg.id === typingMessageId ? (
                      <TypewriterText
                        text={msg.rawAnswer ?? msg.content}
                        speed={10}
                        onComplete={() => setTypingMessageId(null)}
                        render={(partial) => (
                          <div className="leading-relaxed">{renderContent(partial, msg.id, msg.sources, handleSourceClick)}</div>
                        )}
                      />
                    ) : msg.role === "assistant" && msg.interrupted ? (
                      /* ── Interrupted / stopped indicator ── */
                      <span className="flex items-center gap-1.5 text-[12px] text-muted-foreground/45 italic select-none">
                        <Square className="h-2.5 w-2.5 fill-muted-foreground/30 text-muted-foreground/30 shrink-0" />
                        Response stopped
                      </span>
                    ) : msg.role === "assistant" && msg.analytics?.type === "chart_clarification" ? (
                      <ChartClarification
                        analytics={msg.analytics as any}
                        onSelect={async (hint) => {
                          if (isLoading) return;
                          const userMsg: Message = {
                            id: Date.now().toString(),
                            role: "user",
                            content: hint,
                            timestamp: new Date(),
                          };
                          setMessages(prev => [...prev, userMsg]);
                          setOpenSourceKey(null);
                          setSuggestions([]);
                          if (!notifyDismissed) setShowNotifyBanner(true);
                          try {
                            const assistantMsg = await runStreamingQuery(hint, "graph");
                            if (!assistantMsg) return;
                            setMessages(prev => [...prev, assistantMsg]);
                            setTypingMessageId(assistantMsg.id);
                            fetchSuggestions(hint, assistantMsg.content);
                            fireNotification();
                          } catch (error) {
                            const errMsg: Message = {
                              id: (Date.now() + 1).toString(),
                              role: "assistant",
                              content: `Sorry, I encountered an error: ${serializeError(error)}`,
                              timestamp: new Date(),
                            };
                            setMessages(prev => [...prev, errMsg]);
                          }
                        }}
                      />
                    ) : msg.role === "assistant" && msg.analytics?.type === "report_chart_clarification" ? (
                      <ChartClarification
                        analytics={{
                          ...msg.analytics as any,
                          question: (msg.analytics as any).question
                            ?? "Which charts would you like included in the report?",
                          groups: [
                            ...((msg.analytics as any).groups ?? []),
                            { label: "All of them", description: "Include every chart found", hint: "all of them" },
                            { label: "No charts", description: "Build the report without visuals", hint: "no charts" },
                          ],
                        }}
                        onSelect={async (hint) => {
                          if (isLoading) return;
                          const userMsg: Message = {
                            id: Date.now().toString(),
                            role: "user",
                            content: hint,
                            timestamp: new Date(),
                          };
                          setMessages(prev => [...prev, userMsg]);
                          setOpenSourceKey(null);
                          setSuggestions([]);
                          if (!notifyDismissed) setShowNotifyBanner(true);
                          try {
                            const assistantMsg = await runStreamingQuery(hint, "report");
                            if (!assistantMsg) return;
                            setMessages(prev => [...prev, assistantMsg]);
                            setTypingMessageId(assistantMsg.id);
                            fetchSuggestions(hint, assistantMsg.content);
                            fireNotification();
                          } catch (error) {
                            const errMsg: Message = {
                              id: (Date.now() + 1).toString(),
                              role: "assistant",
                              content: `Sorry, I encountered an error: ${serializeError(error)}`,
                              timestamp: new Date(),
                            };
                            setMessages(prev => [...prev, errMsg]);
                          }
                        }}
                      />
                    ) : msg.role === "assistant" && (msg.analytics?.type === "chart_config" || msg.analytics?.type === "chart_error") ? (
                      <ChartMessage analytics={msg.analytics as any} answer={msg.content} />
                    ) : msg.role === "assistant" ? (
                      <div>
                        <AssistantContent
                          raw={msg.rawAnswer ?? msg.content}
                          msgId={msg.id}
                          sources={msg.sources}
                          onSourceClick={handleSourceClick}
                        />
                        {/* Inline nav action buttons — baked into the bubble for soft-nav replies */}
                        {msg.navAction && (
                          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-white/10">
                            <button
                              onClick={() => { setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, navAction: null } : m)); navigate(msg.navAction!.path); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:opacity-90 transition-opacity"
                            >
                              <ChevronRight className="h-3 w-3" />
                              Take me to {msg.navAction.icon} {msg.navAction.label}
                            </button>
                            <button
                              onClick={() => {
                                const replyId = Date.now().toString();
                                setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, navAction: null } : m));
                                setMessages(prev => [...prev, { id: replyId, role: "assistant", content: "No problem! Ask away — I'll do my best right here.", rawAnswer: "No problem! Ask away — I'll do my best right here.", timestamp: new Date() }]);
                                setTypingMessageId(replyId);
                              }}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 text-white/70 hover:text-white text-xs font-medium transition-colors"
                            >
                              <X className="h-3 w-3" />
                              Stay & answer here
                            </button>
                          </div>
                        )}
                      </div>
                    
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
                  </> )} {/* end voice ? ... : <> */}

                  {/* Timestamp + action bar — hidden for voice messages (timestamp is inside VoiceMessageBubble) */}
                  {!msg.voiceBlobUrl && (
                  <div className={`flex items-center gap-2 ${msg.role === "user" ? "justify-end" : "justify-start px-1"} group/msgbar`}>
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
                      {(msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp)).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                    {msg.role === "assistant" && msg.id !== typingMessageId && !msg.interrupted && (
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
                  )} {/* end !msg.voiceBlobUrl */}

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

                          const isWiki = !!(source as any).wiki_url;
                          const wikiUrl: string = (source as any).wiki_url ?? "";

                          // ── Wikipedia source card ──────────────────────
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
                                      <span className="text-[12px] font-semibold text-foreground truncate">
                                        {source.filename}
                                      </span>
                                      <ExternalLink className="h-3 w-3 shrink-0 text-blue-400/70 group-hover/wiki:text-blue-400 transition-colors" />
                                    </span>
                                    <span className="text-[10px] text-blue-400/50 truncate block">wikipedia.org</span>
                                  </span>
                                  <span className="text-[10px] text-blue-400/40 shrink-0">
                                    <ExternalLink className="h-3 w-3" />
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

                          // ── Regular document source card ───────────────
                          const hasImages = !!(source as any).has_images;
                          const hasTables = !!(source as any).has_tables;
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
                                  <span className="flex items-center gap-1.5 flex-wrap">
                                    <span className="text-[12px] font-semibold text-foreground truncate">
                                      {sections[0]?.title || source.filename}
                                    </span>
                                    {/* Filename badge — blue, inline, shown when title differs from filename */}
                                    {sections[0]?.title && sections[0].title !== source.filename && (
                                      <span
                                        title={source.filename}
                                        className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-md text-[9px] font-mono font-medium shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
                                        style={{ color: "#60a5fa", background: "color-mix(in srgb, #3b82f6 12%, transparent)", border: "1px solid color-mix(in srgb, #3b82f6 20%, transparent)" }}
                                        data-open-doc
                                        onClick={(e) => { e.stopPropagation(); openDocumentAtExcerpt(source.filename, docExcerpt); }}
                                      >
                                        <svg width="9" height="9" viewBox="0 0 12 12" fill="none" className="shrink-0">
                                          <path d="M2 2h5l3 3v5a1 1 0 01-1 1H2a1 1 0 01-1-1V3a1 1 0 011-1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                                          <path d="M7 2v3h3" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                                        </svg>
                                        {source.filename}
                                      </span>
                                    )}
                                    {hasImages && (
                                      <span title="Contains diagram/figure descriptions" className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-violet-500/15 text-violet-400 text-[9px] font-medium shrink-0">
                                        <ImageIcon className="h-2.5 w-2.5" />
                                        visual
                                      </span>
                                    )}
                                    {hasTables && (
                                      <span title="Contains table data" className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-teal-500/15 text-teal-400 text-[9px] font-medium shrink-0">
                                        table
                                      </span>
                                    )}
                                    <ExternalLink
                                      className="h-3 w-3 shrink-0 text-primary hover:text-primary/60 transition-colors cursor-pointer"
                                      data-open-doc
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        openDocumentAtExcerpt(source.filename, docExcerpt);
                                      }}
                                    />
                                  </span>
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

                  {/* ── Report card — shown below source cards ── */}
                  {msg.role === "assistant" && (msg.reportId || msg.reportGenerating || msg.reportTitle) && (() => {
                    // Resolve the full report object from state (for downloads/panel).
                    // The card renders immediately using inline SSE data (reportTitle/reportMeta)
                    // and doesn't wait on this lookup.
                    const rpt: ReportRecord | null =
                      msg.reportId ? (reports.find(r => r.report_id === msg.reportId) ?? (activeReport?.report_id === msg.reportId ? activeReport : null)) : null;
                    return (
                      <div className="mt-2">
                        <ReportCard
                          report={rpt}
                          inlineTitle={msg.reportTitle}
                          inlineMeta={msg.reportMeta}
                          isLoading={msg.reportGenerating === true && !msg.reportTitle}
                        />
                      </div>
                    );
                  })()}
                </div>

              </motion.div>
            ))}

            {isLoading && typingConvId === activeConvId && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col gap-1.5"
              >
                {/* Thinking steps above logo+dots */}
                <AnimatePresence>
                  {thinkingSteps.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="flex flex-col gap-1"
                    >
                      <button
                        onClick={() => setThinkingExpanded(v => !v)}
                        className="flex items-center gap-1.5 group w-fit"
                      >
                        <motion.span
                          className="inline-block h-1.5 w-1.5 rounded-full bg-primary/50 shrink-0"
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
                        />
                        <motion.span
                          key={thinkingSteps[thinkingSteps.length - 1]?.message}
                          initial={{ opacity: 0, y: 2 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.18 }}
                          className="text-[11px] text-muted-foreground/50 italic leading-none"
                        >
                          {thinkingSteps[thinkingSteps.length - 1]?.message ?? "Thinking…"}
                        </motion.span>
                        <motion.span
                          animate={{ rotate: thinkingExpanded ? 180 : 0 }}
                          transition={{ duration: 0.2 }}
                          className="opacity-0 group-hover:opacity-60 transition-opacity"
                        >
                          <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
                        </motion.span>
                      </button>
                      <AnimatePresence>
                        {thinkingExpanded && thinkingSteps.length > 1 && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.18 }}
                            className="overflow-hidden"
                          >
                            <div className="flex flex-col gap-0.5 pl-3 border-l border-border/25 ml-[2px]">
                              {thinkingSteps.slice(0, -1).map((step, i) => {
                                const cls = "h-3 w-3 shrink-0 text-muted-foreground/35";
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
                                    <span className="text-[10px] text-muted-foreground/35 leading-snug">{step.message}</span>
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
                {/* Logo + dots always same row */}
                <div className="flex gap-3 items-center">
                  <div className="bg-secondary rounded-2xl rounded-bl-md px-4 py-3 inline-block">
                    <TypingIndicator />
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Notification banner — fixed, always floats above the input area regardless of layout */}
        <AnimatePresence>
          {showNotifyBanner && messages.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.2 }}
              className="fixed left-0 right-0 flex justify-center z-30 pointer-events-none"
              style={{ bottom: notifyBottomOffset + 4 }}
            >
              <div className="pointer-events-auto inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-background/95 backdrop-blur-sm shadow-md text-xs text-muted-foreground">
                <Bell className="h-3 w-3 text-primary shrink-0" />
                <span>Notify when done?</span>
                <button
                  onClick={() => {
                    Notification.requestPermission().then(p => {
                      if (p === "granted") setNotifyEnabled(true);
                    });
                    setShowNotifyBanner(false);
                    setNotifyDismissed(true);
                  }}
                  className="font-medium text-primary hover:text-primary/80 transition-colors"
                >
                  Allow
                </button>
                <span className="text-border">·</span>
                <button
                  onClick={() => {
                    setShowNotifyBanner(false);
                    setNotifyDismissed(true);
                  }}
                  className="hover:text-foreground transition-colors"
                >
                  No thanks
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input */}
        <div
          ref={inputAreaRef}
          className={`overflow-hidden transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] ${messages.length > 0 ? "relative border-t border-border pt-3 pb-4 px-4 shadow-[0_-4px_24px_0_rgba(0,0,0,0.06)] bg-background/80 backdrop-blur-sm" : "absolute left-0 right-0 px-4 pt-3 pb-4 z-20 bg-transparent"}`}
          style={messages.length === 0 ? { bottom: "50%", transform: "translateY(calc(50% + 80px))" } : {}}
          onDragEnter={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current++; setIsChatDragOver(true); }}
          onDragLeave={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current--; if (chatDragCounterRef.current === 0) setIsChatDragOver(false); }}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); e.stopPropagation(); chatDragCounterRef.current = 0; setIsChatDragOver(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
        >
          <div className="max-w-3xl mx-auto">

            {/* ── Contextual suggestion chips ── */}
            <AnimatePresence>
              {(suggestions.length > 0 || suggestionsLoading || expandingSuggestion !== null) && !isLoading && messages.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 4 }}
                  transition={{ duration: 0.2 }}
                  className="flex items-center justify-center gap-2 flex-wrap mb-3 mt-0"
                >
                  {suggestionsLoading && suggestions.length === 0 ? (
                    <>
                      <Sparkles className="h-3.5 w-3.5 text-primary/50 shrink-0" />
                      {[80, 96, 72, 88].map((w, i) => (
                        <div
                          key={i}
                          className="h-7 rounded-full bg-muted animate-pulse"
                          style={{ width: `${w}px` }}
                        />
                      ))}
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-3.5 w-3.5 text-primary/50 shrink-0" />
                      {suggestions.map((s, i) => {
                        const isGeneral = i >= suggestions.length - 2 && suggestions.length >= 3;
                        const isExpanding = expandingSuggestion === s;
                        return (
                          <motion.button
                            key={s}
                            initial={{ opacity: 0, scale: 0.92 }}
                            animate={isExpanding
                              ? { opacity: 1, scale: 1.04 }
                              : { opacity: expandingSuggestion !== null ? 0.35 : 1, scale: 1 }
                            }
                            transition={{ delay: isExpanding ? 0 : i * 0.06, duration: 0.18 }}
                            disabled={expandingSuggestion !== null}
                            onClick={() => expandSuggestion(s, isGeneral)}
                            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all duration-200 whitespace-nowrap inline-flex items-center gap-1.5 ${
                              isExpanding
                                ? "border-primary/60 bg-primary/15 text-primary shadow-sm"
                                : expandingSuggestion !== null
                                ? "cursor-not-allowed border-border bg-card text-muted-foreground"
                                : isGeneral
                                ? "border-dashed border-border/70 bg-transparent hover:bg-muted hover:border-primary/30 text-muted-foreground/60 hover:text-muted-foreground cursor-pointer"
                                : "border-border bg-card hover:bg-muted hover:border-primary/40 text-muted-foreground hover:text-foreground cursor-pointer"
                            }`}
                          >
                            {isExpanding && (
                              <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                            )}
                            {s}
                          </motion.button>
                        );
                      })}
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── Uploaded files tab strip ── */}
            <AnimatePresence>
              {(pendingDocIds.length > 0 || isUploading) && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 4 }}
                  transition={{ duration: 0.18 }}
                  className="flex items-center gap-1.5 flex-wrap mb-2"
                >
                  {/* Loading cards for in-progress uploads */}
                  {documents.filter(d => d.doc_type === "uploading").map(doc => (
                    <div
                      key={doc.document_id}
                      className="flex items-center gap-1.5 pl-1 pr-2.5 py-1 rounded-lg bg-card border border-primary/30 text-left shadow-sm opacity-80"
                      title={doc.filename}
                    >
                      <div className="w-8 h-8 rounded-md overflow-hidden bg-muted/60 flex items-center justify-center shrink-0 border border-border/50">
                        <Loader2 className="h-3.5 w-3.5 text-primary animate-spin" />
                      </div>
                      <span className="text-[11px] text-muted-foreground font-medium max-w-[90px] truncate">
                        {(() => { const e = doc.filename.split('.').pop() ?? ''; const base = e ? doc.filename.slice(0, doc.filename.length - e.length - 1) : doc.filename; return base.length > 8 ? base.slice(0, 8) + '...' + (e ? '.' + e : '') : doc.filename; })()}
                      </span>
                    </div>
                  ))}
                  {pendingDocIds.slice(0, 8).map(docId => {
                    const doc = documents.find(d => d.document_id === docId);
                    if (!doc) return null;
                    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
                    const isImg = ['png','jpg','jpeg','webp','gif'].includes(ext);
                    const isIfc = ['ifc','ifczip'].includes(ext);
                    const isCad = ['dwg','dxf','step','stp'].includes(ext);
                    const isPdf = ext === 'pdf';
                    const cached = blobUrlMapRef.current.get(doc.document_id);
                    return (
                      <button
                        key={doc.document_id}
                        onClick={() => openBubbleDoc(doc)}
                        className="group flex items-center gap-1.5 pl-1 pr-2.5 py-1 rounded-lg bg-card border border-border hover:border-primary/40 hover:bg-primary/5 transition-all text-left shadow-sm"
                        title={doc.filename}
                      >
                        {/* mini preview */}
                        <div className="w-8 h-8 rounded-md overflow-hidden bg-muted/60 flex items-center justify-center shrink-0 border border-border/50">
                          {isImg && cached?.url ? (
                            <img src={cached.url} alt="" className="w-full h-full object-cover" />
                          ) : isIfc ? (
                            <span className="text-[9px] font-bold text-primary">IFC</span>
                          ) : isCad ? (
                            <span className="text-[9px] font-bold text-orange-400">{ext.toUpperCase()}</span>
                          ) : isPdf ? (
                            <span className="text-[9px] font-bold text-red-400">PDF</span>
                          ) : (
                            <FileText className="h-3.5 w-3.5 text-muted-foreground/60" />
                          )}
                        </div>
                        <span className="text-[11px] text-foreground/80 font-medium max-w-[90px] truncate group-hover:text-foreground transition-colors">
                          {(() => { const base = ext ? doc.filename.slice(0, doc.filename.length - ext.length - 1) : doc.filename; return base.length > 8 ? base.slice(0, 8) + '...' + (ext ? '.' + ext : '') : doc.filename; })()}
                        </span>
                      </button>
                    );
                  })}
                  {pendingDocIds.length > 8 && (
                    <span className="text-[10px] text-muted-foreground/50 px-1">+{pendingDocIds.length - 8} more</span>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── Input box ── */}
            <div className="relative">

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
              >
                {/* ── Type-bar drag overlay ── */}
                <AnimatePresence>
                  {isChatDragOver && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      className="absolute inset-0 z-20 rounded-2xl bg-primary/8 border-2 border-dashed border-primary/40 flex items-center justify-center pointer-events-none"
                    >
                      <div className="flex items-center gap-2">
                        <Plus className="h-4 w-4 text-primary" />
                        <span className="text-xs font-medium text-primary">Drop files here</span>
                        <span className="text-[10px] text-primary/60">PDF · DOCX · TXT · PNG · JPG · IFC · DWG</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Actual input row */}
                <div className={"flex items-center gap-2 px-3 py-2.5"}>

                  {/* Auto-expanding textarea with ghost-text word completion — hidden while recording */}
                  {voiceState === "recording" ? (
                    /* ── Live waveform visualizer — replaces the textarea during recording ── */
                    <div className="flex-1 flex items-center gap-2 py-0">
                      {/* Delete recording button */}
                      <button
                        type="button"
                        onClick={() => {
                          // Cancel recording without transcribing
                          if (maxDurationTimerRef.current) { clearTimeout(maxDurationTimerRef.current); maxDurationTimerRef.current = null; }
                          if (elapsedTimerRef.current) { clearInterval(elapsedTimerRef.current); elapsedTimerRef.current = null; }
                          cancelAnimationFrame(animFrameRef.current);
                          analyserRef.current = null;
                          audioCtxRef.current?.close().catch(() => {});
                          audioCtxRef.current = null;
                          waveformSamplesRef.current = [];
                          setWaveformBars(Array(48).fill(0));
                          setRecordingElapsed(0);
                          // Forcibly stop the recorder without triggering onstop → transcribe flow
                          const recorder = mediaRecorderRef.current;
                          if (recorder) {
                            recorder.onstop = () => {};  // override handler
                            recorder.stream?.getTracks().forEach(t => t.stop());
                            try { recorder.stop(); } catch {}
                            mediaRecorderRef.current = null;
                          }
                          setVoiceState("idle");
                        }}
                        className="h-7 w-7 shrink-0 flex items-center justify-center rounded-full text-destructive/60 hover:text-destructive hover:bg-destructive/10 transition-colors"
                        title="Discard recording"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>

                      {/* Waveform bars */}
                      <div className="flex-1 flex items-center gap-[1.5px] h-8">
                        {waveformBars.map((level, i) => {
                          const minH = 2;
                          const maxH = 28;
                          const h = minH + level * (maxH - minH);
                          const opacity = 0.3 + level * 0.7;
                          return (
                            <div
                              key={i}
                              style={{
                                height: `${h}px`,
                                opacity,
                                flex: "1 1 0",
                                borderRadius: "1px",
                                background: "hsl(var(--primary))",
                                transition: "height 0.06s ease-out, opacity 0.06s ease-out",
                              }}
                            />
                          );
                        })}
                      </div>

                      {/* Elapsed timer */}
                      <span className="text-[11px] text-primary/70 font-mono tabular-nums shrink-0 w-8 text-right">
                        {String(Math.floor(recordingElapsed / 60)).padStart(2, "0")}:{String(recordingElapsed % 60).padStart(2, "0")}
                      </span>
                    </div>
                  ) : (
                  <div className="flex-1 relative flex items-center gap-1.5">
                    {/* Upload plus button — sits at the left edge, inline with placeholder */}
                    <button
                      type="button"
                      onClick={() => docFileInputRef.current?.click()}
                      disabled={isLoading || isUploading}
                      className="shrink-0 h-6 w-6 flex items-center justify-center rounded-md text-muted-foreground/50 hover:text-primary hover:bg-primary/8 transition-colors disabled:opacity-30"
                      title="Upload file"
                    >
                      {isUploading
                        ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        : <Plus className="h-3.5 w-3.5" />
                      }
                    </button>
                    <input
                      ref={docFileInputRef}
                      type="file"
                      accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg,.webp,.gif,.ifc,.ifczip,.dxf,.dwg,.step,.stp,.rvt,.nwd,.nwc,.dgn,.skp,.3dm,.fbx,.obj,.stl,.sat,.iges,.igs,.prt,.sldprt,.catpart,.3ds,.dae,.rfa,.rte"
                      multiple
                      className="hidden"
                      onChange={e => { if (e.target.files) handleFiles(e.target.files); e.target.value = ""; }}
                    />
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={handleInputChange}
                      onKeyDown={handleInputKeyDown}
                      onBlur={() => {
                        setTimeout(() => { setAutocomplete(null); setAutocompletePos(null); }, 150);
                        setWordSuffix("");
                      }}
                      placeholder="Ask a question… type @ to mention a file"
                      rows={1}
                      disabled={isLoading}
                      style={{ maxHeight: "160px" }}
                      className="flex-1 resize-none bg-transparent text-sm text-foreground placeholder:text-muted-foreground/60 placeholder:align-middle focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed leading-5 py-1.5 overflow-y-auto scrollbar-thin self-center"
                    />
                    {/* Ghost-text suffix — never shown on empty input */}
                    {wordSuffix && input.length > 0 && !autocomplete && (
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute left-0 top-0 w-full h-full flex items-center text-sm leading-5 whitespace-pre-wrap break-words select-none px-0 py-1.5"
                        style={{ color: "transparent" }}
                      >
                        <span>{input}</span>
                        <span className="text-muted-foreground/40">{wordSuffix}</span>
                      </span>
                    )}
                  </div>
                  )}

                  {/* Notify toggle */}
                  {notifyDismissed && (
                    <button
                      type="button"
                      onClick={() => {
                        if (!notifyEnabled) {
                          Notification.requestPermission().then(p => { if (p === "granted") setNotifyEnabled(true); });
                        } else {
                          setNotifyEnabled(false);
                        }
                      }}
                      title={notifyEnabled ? "Notifications on — click to disable" : "Enable notifications"}
                      className={"h-8 w-8 shrink-0 flex items-center justify-center rounded-lg transition-colors mb-0.5 " + (notifyEnabled ? "text-primary" : "text-muted-foreground/40 hover:text-muted-foreground")}
                    >
                      {notifyEnabled ? <Bell className="h-4 w-4" /> : <BellOff className="h-4 w-4" />}
                    </button>
                  )}

                  {/* Voice / Stop-recording button */}
                  <button
                    type="button"
                    onClick={handleVoiceClick}
                    onMouseEnter={() => setIsHoveringVoice(true)}
                    onMouseLeave={() => setIsHoveringVoice(false)}
                    disabled={voiceState === "transcribing" || isLoading}
                    className={[
                      "h-8 w-8 shrink-0 flex items-center justify-center rounded-lg transition-colors group",
                      voiceState === "recording"
                        ? "text-destructive hover:text-destructive/80 hover:bg-destructive/10"
                        : voiceState === "transcribing"
                        ? "text-primary opacity-60 cursor-wait"
                        : "text-muted-foreground hover:text-primary",
                    ].join(" ")}
                    title={
                      voiceState === "recording"
                        ? "Stop recording"
                        : voiceState === "transcribing"
                        ? "Transcribing…"
                        : "Voice message"
                    }
                  >
                    {voiceState === "transcribing" ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : voiceState === "recording" ? (
                      /* Stop icon — solid red square */
                      <Square className="h-3.5 w-3.5 fill-current" />
                    ) : (
                      <svg width="22" height="18" viewBox="0 0 22 18" fill="none" xmlns="http://www.w3.org/2000/svg" className="overflow-visible">
                        {[
                          { x: 1,  baseH: 4,  hoverH: 6,  delay: "0ms"   },
                          { x: 4,  baseH: 8,  hoverH: 14, delay: "60ms"  },
                          { x: 7,  baseH: 12, hoverH: 18, delay: "120ms" },
                          { x: 10, baseH: 16, hoverH: 18, delay: "180ms" },
                          { x: 13, baseH: 12, hoverH: 18, delay: "120ms" },
                          { x: 16, baseH: 8,  hoverH: 14, delay: "60ms"  },
                          { x: 19, baseH: 4,  hoverH: 6,  delay: "0ms"   },
                        ].map((bar, i) => {
                          const activeH = isHoveringVoice ? bar.hoverH : bar.baseH;
                          return (
                            <rect
                              key={i}
                              x={bar.x}
                              y={9 - activeH / 2}
                              width="2"
                              rx="1"
                              height={activeH}
                              fill="currentColor"
                              style={{
                                transition: `y 0.55s cubic-bezier(0.34,1.56,0.64,1) ${bar.delay}, height 0.55s cubic-bezier(0.34,1.56,0.64,1) ${bar.delay}`,
                              }}
                            />
                          );
                        })}
                      </svg>
                    )}
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

      {/* ── Filename autocomplete — fixed portal, never clipped ── */}
      <AnimatePresence>
        {autocomplete && autocomplete.results.length > 0 && autocompletePos && (
          <motion.div
            ref={autocompleteRef}
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            style={{
              position: "fixed",
              bottom: `calc(100vh - ${autocompletePos.top}px + 8px)`,
              left: autocompletePos.left,
              width: autocompletePos.width,
              zIndex: 9999,
            }}
            className="bg-card border border-border rounded-xl shadow-2xl overflow-hidden"
          >
            <div className="px-3 py-1.5 border-b border-border/60 flex items-center gap-1.5">
              <FileText className="h-3 w-3 text-primary/60" />
              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
                Documents
              </span>
              <span className="text-[10px] text-muted-foreground/50 ml-auto">
                ↑↓ · Tab to select
              </span>
            </div>
            {autocomplete.results.map((doc, idx) => (
              <button
                key={doc.document_id}
                onMouseDown={(e) => {
                  e.preventDefault();
                  applyAutocomplete(doc);
                }}
                onMouseEnter={() =>
                  setAutocomplete(a => a ? { ...a, activeIdx: idx } : a)
                }
                className={`w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                  idx === autocomplete.activeIdx
                    ? "bg-primary/10 text-foreground"
                    : "hover:bg-muted text-foreground/80"
                }`}
              >
                <FileText className={`h-3.5 w-3.5 shrink-0 ${idx === autocomplete.activeIdx ? "text-primary" : "text-muted-foreground"}`} />
                <span className="text-sm flex-1 truncate">{doc.filename}</span>
                <span className="text-[10px] text-muted-foreground/50 shrink-0 capitalize">
                  {(doc.doc_type ?? "").replace(/_/g, " ")}
                </span>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Chat;