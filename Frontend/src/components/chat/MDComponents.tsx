import React from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ExternalLink } from "lucide-react";

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

export const MD_COMPONENTS: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  p: ({ children }) => <span className="inline leading-relaxed">{children}</span>,
  ul: ({ children }) => <ul className="list-disc list-inside my-2 space-y-1 block">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside my-2 space-y-1 block">{children}</ol>,
  li: ({ children }) => {
    const flatChildren = React.Children.map(children, (child: any) => {
      if (child?.props?.className === "inline leading-relaxed") return child.props.children;
      return child;
    });
    const text = typeof children === "string" ? children : Array.isArray(children) ? children.join("") : String(children ?? "");
    if (!text.trim()) return null;
    return <li className="ms-2 leading-relaxed">{flatChildren ?? children}</li>;
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
    const meta = LANG_META[lang.toLowerCase()] ?? { label: lang || "Code", accent: "#374151", ext: lang || "txt" };

    const inferFilename = (): string => {
      const lines = codeText.split("\n").slice(0, 20);
      const patterns = [
        /^(?:export\s+)?(?:async\s+)?(?:function\s+)([\w$]+)/,
        /^(?:export\s+)?(?:default\s+)?class\s+([\w$]+)/,
        /^(?:export\s+)?(?:const|let|var)\s+([\w$]+)\s*=/,
        /^def\s+([\w]+)/,
        /^class\s+([\w]+)/,
        /^(?:pub\s+)?fn\s+([\w]+)/,
        /^func\s+([\w]+)/,
        /^(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:class|interface|enum)\s+([\w]+)/,
        /^(?:sub|function)\s+([\w]+)/i,
      ];
      for (const line of lines) {
        const trimmed = line.trim();
        for (const pat of patterns) {
          const m = trimmed.match(pat);
          if (m && m[1]) {
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

    const isDark = typeof window !== "undefined"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        || document.documentElement.classList.contains("dark")
      : true;

    const C = {
      base:     isDark ? "#cbd5e1" : "#1e293b",
      comment:  isDark ? "#64748b" : "#6b7280",
      string:   isDark ? "#10b981" : "#047857",
      number:   isDark ? "#f97316" : "#b45309",
      kw_ctrl:  isDark ? "#c084fc" : "#7c3aed",
      kw_decl:  isDark ? "#60a5fa" : "#1d4ed8",
      kw_lit:   isDark ? "#f97316" : "#b45309",
      type:     isDark ? "#34d399" : "#047857",
      builtin:  isDark ? "#fbbf24" : "#b45309",
      classname:isDark ? "#34d399" : "#0f766e",
      dunder:   isDark ? "#94a3b8" : "#64748b",
      operator: isDark ? "#f87171" : "#b91c1c",
      bracket:  isDark ? "#94a3b8" : "#475569",
      punct:    isDark ? "#64748b" : "#6b7280",
      decorator:isDark ? "#c084fc" : "#7c3aed",
    };

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

        const trimmed = line.trimStart();
        if (isLineComment(trimmed)) {
          result.push(cs(C.comment, line, `cm-${li}`, true));
          return;
        }

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

    const headerBg = isDark ? "rgba(15,23,42,0.85)" : "rgba(241,245,249,1)";
    const bodyBg   = isDark ? "rgba(2,6,23,0.92)"   : "rgba(248,250,252,1)";
    const borderCol= isDark ? "rgba(51,65,85,0.6)"  : "rgba(203,213,225,0.8)";

    return (
      <div className="relative my-3 rounded-xl overflow-hidden shadow-md" style={{ border: `1px solid ${borderCol}` }}>
        <div className="flex items-center justify-between px-3 py-2" style={{ background: headerBg, borderBottom: `1px solid ${borderCol}` }}>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: meta.accent }} />
            <span className="text-[11px] font-semibold font-mono tracking-wide" style={{ color: meta.accent }}>
              {meta.label}
            </span>
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
    <blockquote className="border-s-2 border-primary/40 ps-3 italic text-muted-foreground my-2 block">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-border my-4" />,
  a: ({ href, children }) => (
    <span className="inline-flex items-baseline gap-px">
      <span>{children}</span>
      {href && href !== "#" && (
        <a href={href} target="_blank" rel="noopener noreferrer" title={href}
          className="inline-flex items-center ms-0.5 text-primary/50 hover:text-primary transition-colors"
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
  th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-semibold text-start">{children}</th>,
  td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
  br: () => <br />,
};
