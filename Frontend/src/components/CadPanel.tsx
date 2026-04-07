/**
 * CadPanel.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * CAD / IFC Agent panel — drop into Chat.tsx alongside the existing docs /
 * reports panels.
 *
 * Usage in Chat.tsx
 * ─────────────────
 * 1. Import this file:
 *      import CadPanel from "@/components/CadPanel";
 *
 * 2. Add state next to the other panel state vars (around line 2433):
 *      const [cadPanelOpen, setCadPanelOpen]       = useState(false);
 *      const cadPanelOpenRef = useRef(false);
 *      cadPanelOpenRef.current = cadPanelOpen;
 *
 * 3. Add a toolbar button to open the panel (copy the "Reports" button style):
 *      <button onClick={() => setCadPanelOpen(v => !v)} title="CAD / IFC files">
 *        <Building2 className="h-4 w-4" />
 *      </button>
 *
 * 4. Mount the panel just before the closing </div> of the layout root:
 *      <CadPanel
 *        open={cadPanelOpen}
 *        onClose={() => setCadPanelOpen(false)}
 *        apiBase={getApiBase()}
 *        sessionId={sessionId}
 *        onAnswer={(answer) => {
 *          // inject answer as assistant message
 *          const msg: Message = {
 *            id: Date.now().toString(),
 *            role: "assistant",
 *            content: answer,
 *            timestamp: new Date(),
 *          };
 *          setMessages(prev => [...prev, msg]);
 *        }}
 *      />
 *
 * 5. Add "Building2" to the lucide-react import line in Chat.tsx.
 *
 * That's it — no backend changes needed; the endpoints are already wired.
 */

import React, {
  useState,
  useRef,
  useEffect,
  useCallback,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Upload,
  Loader2,
  Send,
  Trash2,
  Building2,
  ChevronDown,
  AlertCircle,
  FileText,
  Layers,
  Box,
  BarChart2,
} from "lucide-react";
import { Button } from "@/components/ui/button";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface CadFile {
  file_id:   string;
  filename:  string;
  file_type: string;
  pipeline:  string;
  cached_at: string;
  // upload response extras (only set right after upload)
  ux_hint?:        string;
  total_elements?: number;
  total_entities?: number;
  dimension?:      string;
  storeys?:        any[];
  element_counts?: Record<string, number>;
  entity_counts?:  Record<string, number>;
  layers?:         any[];
  material_inventory?: Record<string, any>;
  parse_errors?:   string[];
}

interface CadMessage {
  id:      string;
  role:    "user" | "assistant";
  content: string;
}

interface CadPanelProps {
  open:      boolean;
  onClose:   () => void;
  apiBase:   string;
  sessionId: string | null;
  /** Called whenever a CAD answer arrives — inject it into the main chat. */
  onAnswer?: (answer: string, fileId: string) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const CAD_EXTS = [".ifc", ".ifczip", ".dwg", ".dxf", ".step", ".stp"];

function isCADFile(name: string): boolean {
  const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
  return CAD_EXTS.includes(ext);
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// ─────────────────────────────────────────────────────────────────────────────
// File summary card
// ─────────────────────────────────────────────────────────────────────────────

const FileSummaryCard: React.FC<{ file: CadFile }> = ({ file }) => {
  const [expanded, setExpanded] = useState(false);

  const isPipeline = (p: string) => file.pipeline === p;
  const isBIM = isPipeline("ifc");

  const elementRows = isBIM
    ? Object.entries(file.element_counts ?? {}).slice(0, 8)
    : Object.entries(file.entity_counts  ?? {}).slice(0, 8);

  const hasDetails =
    (file.storeys && file.storeys.length > 0) ||
    elementRows.length > 0 ||
    (file.layers && file.layers.length > 0) ||
    (file.material_inventory && Object.keys(file.material_inventory).length > 0);

  return (
    <div className="rounded-xl border border-border bg-muted/30 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          {isBIM
            ? <Building2 className="h-4 w-4 text-primary" />
            : <Layers     className="h-4 w-4 text-primary" />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-medium text-foreground truncate">{file.filename}</p>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
            {file.pipeline.toUpperCase()}
            {file.dimension ? ` · ${file.dimension}` : ""}
            {file.total_elements ? ` · ${file.total_elements.toLocaleString()} elements` : ""}
            {file.total_entities ? ` · ${file.total_entities.toLocaleString()} entities` : ""}
          </p>
        </div>
        {hasDetails && (
          <button
            onClick={() => setExpanded(v => !v)}
            className="text-muted-foreground/50 hover:text-muted-foreground transition-colors"
          >
            <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.18 }}>
              <ChevronDown className="h-3.5 w-3.5" />
            </motion.span>
          </button>
        )}
      </div>

      {/* UX hint */}
      {file.ux_hint && (
        <p className="px-3 pb-2 text-[11px] text-muted-foreground/70 leading-snug italic">
          {file.ux_hint}
        </p>
      )}

      {/* Expanded details */}
      <AnimatePresence>
        {expanded && hasDetails && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 flex flex-col gap-2 border-t border-border/40 pt-2">
              {/* Storeys */}
              {file.storeys && file.storeys.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                    Storeys ({file.storeys.length})
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {file.storeys.slice(0, 10).map((s: any, i) => (
                      <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                        {typeof s === "string" ? s : s.name ?? JSON.stringify(s)}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Element / entity counts */}
              {elementRows.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                    {isBIM ? "Elements" : "Entities"}
                  </p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                    {elementRows.map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between">
                        <span className="text-[10px] text-muted-foreground capitalize truncate">{k}</span>
                        <span className="text-[10px] font-medium text-foreground tabular-nums">{(v as number).toLocaleString()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Layers */}
              {file.layers && file.layers.length > 0 && (
                <div>
                  <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                    Layers ({file.layers.length})
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {file.layers.slice(0, 12).map((l: any, i) => (
                      <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono">
                        {typeof l === "string" ? l : l.name ?? JSON.stringify(l)}
                      </span>
                    ))}
                    {file.layers.length > 12 && (
                      <span className="text-[10px] text-muted-foreground/50">+{file.layers.length - 12} more</span>
                    )}
                  </div>
                </div>
              )}

              {/* Parse errors */}
              {file.parse_errors && file.parse_errors.length > 0 && (
                <div className="flex items-start gap-1.5 rounded-lg bg-yellow-500/10 border border-yellow-500/20 px-2 py-1.5">
                  <AlertCircle className="h-3 w-3 text-yellow-400 shrink-0 mt-0.5" />
                  <p className="text-[10px] text-yellow-400 leading-snug">{file.parse_errors[0]}</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Panel
// ─────────────────────────────────────────────────────────────────────────────

const CadPanel: React.FC<CadPanelProps> = ({
  open,
  onClose,
  apiBase,
  sessionId,
  onAnswer,
}) => {
  // ── State ──────────────────────────────────────────────────────────────────
  const [files, setFiles]                   = useState<CadFile[]>([]);
  const [activeFile, setActiveFile]         = useState<CadFile | null>(null);
  const [messages, setMessages]             = useState<CadMessage[]>([]);
  const [input, setInput]                   = useState("");
  const [isUploading, setIsUploading]       = useState(false);
  const [isQuerying, setIsQuerying]         = useState(false);
  const [uploadError, setUploadError]       = useState<string | null>(null);
  const [isDragging, setIsDragging]         = useState(false);
  const [dragDepth, setDragDepth]           = useState(0);
  const [deletingId, setDeletingId]         = useState<string | null>(null);

  const fileInputRef   = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef    = useRef<HTMLTextAreaElement>(null);

  // ── Load cached files on mount / open ─────────────────────────────────────
  const loadFiles = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/cad/files`);
      if (!res.ok) return;
      const data = await res.json();
      setFiles(data.files ?? []);
    } catch { /* silent */ }
  }, [apiBase]);

  useEffect(() => {
    if (open) loadFiles();
  }, [open, loadFiles]);

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Upload ─────────────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (file: File) => {
    setUploadError(null);

    if (!isCADFile(file.name)) {
      setUploadError(`Unsupported format. Accepted: ${CAD_EXTS.join(", ")}`);
      return;
    }

    setIsUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${apiBase}/api/cad/upload`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data: CadFile = await res.json();
      setFiles(prev => [data, ...prev.filter(f => f.file_id !== data.file_id)]);
      setActiveFile(data);
      setMessages([]);
    } catch (e: any) {
      setUploadError(e?.message ?? "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [apiBase]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = "";
  };

  // ── Drag & drop ────────────────────────────────────────────────────────────
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    setDragDepth(d => { if (d === 0) setIsDragging(true); return d + 1; });
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragDepth(d => { if (d === 1) setIsDragging(false); return d - 1; });
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setDragDepth(0);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  };

  // ── Delete ─────────────────────────────────────────────────────────────────
  const deleteFile = async (fileId: string) => {
    setDeletingId(fileId);
    try {
      await fetch(`${apiBase}/api/cad/files/${fileId}`, { method: "DELETE" });
      setFiles(prev => prev.filter(f => f.file_id !== fileId));
      if (activeFile?.file_id === fileId) {
        setActiveFile(null);
        setMessages([]);
      }
    } catch { /* silent */ } finally {
      setDeletingId(null);
    }
  };

  // ── Query ──────────────────────────────────────────────────────────────────
  const handleQuery = useCallback(async () => {
    if (!input.trim() || !activeFile || isQuerying) return;

    const userMsg: CadMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsQuerying(true);

    try {
      const res = await fetch(`${apiBase}/api/cad/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query:      userMsg.content,
          file_id:    activeFile.file_id,
          session_id: sessionId ?? undefined,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: CadMessage = {
        id:      `${Date.now()}-ans`,
        role:    "assistant",
        content: data.answer,
      };
      setMessages(prev => [...prev, assistantMsg]);
      onAnswer?.(data.answer, activeFile.file_id);
    } catch (e: any) {
      setMessages(prev => [...prev, {
        id:      `${Date.now()}-err`,
        role:    "assistant",
        content: `⚠️ ${e?.message ?? "Query failed."}`,
      }]);
    } finally {
      setIsQuerying(false);
    }
  }, [input, activeFile, isQuerying, apiBase, sessionId, onAnswer]);

  // ── Key handler ────────────────────────────────────────────────────────────
  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  // ── Auto-resize textarea ───────────────────────────────────────────────────
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, [input]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ x: "100%", opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: "100%", opacity: 0 }}
          transition={{ type: "spring", stiffness: 320, damping: 32 }}
          className="fixed top-0 right-0 bottom-0 z-40 w-[380px] max-w-[92vw] bg-background border-l border-border flex flex-col shadow-2xl"
          onDragEnter={handleDragEnter}
          onDragOver={e => e.preventDefault()}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* Drag overlay */}
          <AnimatePresence>
            {isDragging && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center rounded-none pointer-events-none"
              >
                <div className="border-2 border-dashed border-primary rounded-2xl px-8 py-10 flex flex-col items-center gap-3">
                  <Building2 className="h-8 w-8 text-primary" />
                  <p className="text-sm font-semibold text-foreground">Drop CAD / IFC file</p>
                  <p className="text-[11px] text-muted-foreground">{CAD_EXTS.join(" · ")}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Header ── */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
            <div className="flex items-center gap-2">
              <Building2 className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold text-foreground">CAD / IFC Agent</span>
              {files.length > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-medium">
                  {files.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Upload button */}
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors disabled:opacity-50"
              >
                {isUploading
                  ? <Loader2 className="h-3 w-3 animate-spin" />
                  : <Upload   className="h-3 w-3" />}
                {isUploading ? "Uploading…" : "Upload"}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept={CAD_EXTS.join(",")}
                className="hidden"
                onChange={handleFileInput}
              />
              <button
                onClick={onClose}
                className="h-7 w-7 flex items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* ── Body ── */}
          <div className="flex-1 overflow-hidden flex flex-col min-h-0">

            {/* Error banner */}
            <AnimatePresence>
              {uploadError && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden px-4 pt-3"
                >
                  <div className="flex items-start gap-2 rounded-lg bg-destructive/10 border border-destructive/20 px-3 py-2.5">
                    <AlertCircle className="h-3.5 w-3.5 text-destructive shrink-0 mt-0.5" />
                    <p className="text-[11px] text-destructive leading-snug">{uploadError}</p>
                    <button onClick={() => setUploadError(null)} className="ml-auto text-destructive/60 hover:text-destructive">
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Split: file list (top) + chat (bottom) */}
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">

              {/* ── File list ── */}
              {files.length > 0 && (
                <div className="shrink-0 max-h-[45%] overflow-y-auto px-4 pt-3 pb-2 flex flex-col gap-2 border-b border-border/50">
                  {files.map(f => (
                    <div
                      key={f.file_id}
                      className={`relative group cursor-pointer transition-all ${activeFile?.file_id === f.file_id ? "ring-1 ring-primary/40 rounded-xl" : ""}`}
                      onClick={() => {
                        setActiveFile(f);
                        setMessages([]);
                      }}
                    >
                      <FileSummaryCard file={f} />
                      {/* Delete button */}
                      <button
                        onClick={e => { e.stopPropagation(); deleteFile(f.file_id); }}
                        disabled={deletingId === f.file_id}
                        className="absolute top-2 right-2 h-5 w-5 hidden group-hover:flex items-center justify-center rounded bg-background/80 text-muted-foreground/60 hover:text-destructive transition-colors"
                      >
                        {deletingId === f.file_id
                          ? <Loader2 className="h-2.5 w-2.5 animate-spin" />
                          : <Trash2  className="h-2.5 w-2.5" />}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* ── Empty state ── */}
              {files.length === 0 && !isUploading && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 px-6 text-center">
                  <div className="h-14 w-14 rounded-2xl bg-muted flex items-center justify-center">
                    <Building2 className="h-7 w-7 text-muted-foreground/40" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-foreground">No CAD / IFC files yet</p>
                    <p className="text-[11px] text-muted-foreground/60 mt-1 leading-relaxed">
                      Upload an IFC, DXF, DWG or STEP file to start analysing it with AI.
                    </p>
                  </div>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium bg-primary text-primary-foreground hover:opacity-90 transition-opacity"
                  >
                    <Upload className="h-4 w-4" />
                    Upload file
                  </button>
                  <p className="text-[10px] text-muted-foreground/40">{CAD_EXTS.join(" · ")}</p>
                </div>
              )}

              {isUploading && files.length === 0 && (
                <div className="flex-1 flex items-center justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-primary/60" />
                </div>
              )}

              {/* ── Chat area ── */}
              {activeFile && (
                <div className="flex-1 flex flex-col min-h-0">
                  {/* Active file indicator */}
                  <div className="shrink-0 flex items-center gap-2 px-4 py-2 bg-primary/5 border-b border-border/40">
                    <Box className="h-3 w-3 text-primary shrink-0" />
                    <span className="text-[11px] text-primary font-medium truncate">{activeFile.filename}</span>
                    <span className="text-[10px] text-muted-foreground/50 uppercase shrink-0">{activeFile.pipeline}</span>
                  </div>

                  {/* Messages */}
                  <div className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-3 scroll-smooth">
                    {messages.length === 0 && (
                      <div className="flex flex-col gap-2 mt-2">
                        <p className="text-[11px] text-muted-foreground/50 text-center">
                          Ask anything about this file
                        </p>
                        {/* Suggested prompts */}
                        {[
                          activeFile.pipeline === "ifc"
                            ? "What types of elements are in this model?"
                            : "What layers does this drawing have?",
                          activeFile.pipeline === "ifc"
                            ? "How many storeys does the building have?"
                            : "What is the overall dimension of the design?",
                          "What materials are used?",
                        ].map((q, i) => (
                          <button
                            key={i}
                            onClick={() => setInput(q)}
                            className="text-left text-[11px] px-3 py-2 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 text-muted-foreground hover:text-foreground transition-all"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}

                    {messages.map(msg => (
                      <motion.div
                        key={msg.id}
                        initial={{ opacity: 0, y: 4 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.15 }}
                        className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-2xl px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap ${
                            msg.role === "user"
                              ? "bg-primary text-primary-foreground rounded-br-md"
                              : "bg-secondary text-foreground rounded-bl-md"
                          }`}
                        >
                          {msg.content}
                        </div>
                      </motion.div>
                    ))}

                    {isQuerying && (
                      <div className="flex justify-start">
                        <div className="bg-secondary rounded-2xl rounded-bl-md px-3 py-2 flex items-center gap-1.5">
                          {[0, 1, 2].map(i => (
                            <motion.span
                              key={i}
                              className="h-1.5 w-1.5 rounded-full bg-muted-foreground/40"
                              animate={{ opacity: [0.3, 1, 0.3] }}
                              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    <div ref={messagesEndRef} />
                  </div>

                  {/* Input */}
                  <div className="shrink-0 px-3 pb-3 pt-2 border-t border-border/40">
                    <div className="flex items-end gap-2 rounded-xl bg-card border border-border px-3 py-2">
                      <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKey}
                        placeholder="Ask about this file…"
                        rows={1}
                        className="flex-1 resize-none bg-transparent text-[13px] text-foreground placeholder:text-muted-foreground/40 outline-none leading-relaxed overflow-hidden"
                        style={{ minHeight: "24px", maxHeight: "120px" }}
                        disabled={isQuerying}
                      />
                      <button
                        onClick={handleQuery}
                        disabled={!input.trim() || isQuerying}
                        className="h-7 w-7 shrink-0 flex items-center justify-center rounded-lg bg-primary text-primary-foreground disabled:opacity-40 hover:opacity-90 transition-opacity"
                      >
                        {isQuerying
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <Send    className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default CadPanel;
