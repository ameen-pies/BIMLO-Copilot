import React, { useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Send, FileText, X, User, ArrowLeft, Plus, Loader2, AlertCircle,
  ChevronDown, ChevronUp, ExternalLink, ScrollText, Eye, Square,
  ThumbsUp, ThumbsDown, RotateCcw, Pencil, Check, Copy, ImageIcon,
  Search, MessageSquare, Clock, SortAsc, FolderOpen, Trash2, Sparkles,
  Bell, BellOff, BookOpen, BarChart2, ChevronRight, RefreshCw, Phone,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import ThemeToggle from "@/components/ThemeToggle";
import LangToggle from "@/components/LangToggle";
import TypewriterText from "@/components/TypewriterText";
import Logo from "@/components/Logo";
import { ProfileBubble } from "@/components/Navbar";
import { ZoomableImage } from "@/components/chat/ZoomableImage";
import { PdfViewer } from "@/components/chat/PdfViewer";
import { XeokitViewer } from "@/components/chat/XeokitViewer";
import { renderDocumentContent, ViewerState } from "@/components/chat/chatRenderers";
import { renderReportContent } from "@/components/chat/ChartRenderer";
import { Conversation, ReportRecord } from "@/pages/chatTypes";
import { Document } from "@/services/api";
import { IMAGE_EXTS } from "@/pages/chatTypes";
import { AuthUser } from "@/context/AuthContext";

interface ChatHeaderProps {
  activeConversation: Conversation | undefined;
  convsPanelOpen: boolean;
  setConvsPanelOpen: React.Dispatch<React.SetStateAction<boolean>>;
  loadConversationsFromDB: () => Promise<void>;
  conversations: Conversation[];
  startNewConversation: () => void;
  historySearch: string;
  setHistorySearch: React.Dispatch<React.SetStateAction<string>>;
  historySort: "newest" | "oldest";
  setHistorySort: React.Dispatch<React.SetStateAction<"newest" | "oldest">>;
  activeConvId: string;
  loadConversation: (conv: Conversation) => void;
  pendingConvIds: Record<string, boolean>;
  deleteConversation: (convId: string) => Promise<void>;
  reports: ReportRecord[];
  reportsPanelOpenRef: React.MutableRefObject<boolean>;
  setReportsPanelOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setActiveReport: React.Dispatch<React.SetStateAction<ReportRecord | null>>;
  setReportEditMode: React.Dispatch<React.SetStateAction<boolean>>;
  previewedVersion: { version: number; content: string; charts: any[]; title: string } | null;
  setPreviewedVersion: React.Dispatch<React.SetStateAction<{ version: number; content: string; charts: any[]; title: string } | null>>;
  reportsPanelOpen: boolean;
  isGeneratingReport: boolean;
  handleDownloadReport: (report: ReportRecord, format: "pdf") => void;
  downloadingReportId: string | null;
  handleDeleteReport: (reportId: string) => void;
  deletingReportId: string | null;
  activeReport: ReportRecord | null;
  showVersionHistory: boolean;
  setShowVersionHistory: React.Dispatch<React.SetStateAction<boolean>>;
  isPatchingReport: boolean;
  reportEditMode: boolean;
  reportEditInstruction: string;
  setReportEditInstruction: React.Dispatch<React.SetStateAction<string>>;
  reportEditInputRef: React.RefObject<HTMLTextAreaElement>;
  handlePatchReport: () => void;
  restoringVersion: number | null;
  handleRestoreVersion: (report: ReportRecord, version: number) => void;
  docsPanelOpenRef: React.MutableRefObject<boolean>;
  setDocsPanelOpen: React.Dispatch<React.SetStateAction<boolean>>;
  docsPanelOpen: boolean;
  documents: Document[];
  bubbleDoc: Document | null;
  setBubbleDoc: React.Dispatch<React.SetStateAction<Document | null>>;
  setBubbleViewer: React.Dispatch<React.SetStateAction<ViewerState | null>>;
  bubbleViewer: ViewerState | null;
  openBubbleDoc: (doc: Document) => void;
  confirmingDeleteId: string | null;
  setConfirmingDeleteId: React.Dispatch<React.SetStateAction<string | null>>;
  removeDocument: (documentId: string) => Promise<void>;
  isUploading: boolean;
  bubbleScrollRef: React.RefObject<HTMLDivElement>;
  bubbleHighlightRef: React.RefObject<HTMLDivElement>;
  sessionIdRef: React.MutableRefObject<string | null>;
  currentUser: AuthUser | null;
  showAuthModal: (onSuccess?: (() => void) | null) => void;
  logout: () => void;
  onRenameConversation?: (convId: string, newTitle: string) => void;
}

export function ChatHeader({
  activeConversation,
  convsPanelOpen,
  setConvsPanelOpen,
  loadConversationsFromDB,
  conversations,
  startNewConversation,
  historySearch,
  setHistorySearch,
  historySort,
  setHistorySort,
  activeConvId,
  loadConversation,
  pendingConvIds,
  deleteConversation,
  reports,
  reportsPanelOpenRef,
  setReportsPanelOpen,
  setActiveReport,
  setReportEditMode,
  previewedVersion,
  setPreviewedVersion,
  reportsPanelOpen,
  isGeneratingReport,
  handleDownloadReport,
  downloadingReportId,
  handleDeleteReport,
  deletingReportId,
  activeReport,
  showVersionHistory,
  setShowVersionHistory,
  isPatchingReport,
  reportEditMode,
  reportEditInstruction,
  setReportEditInstruction,
  reportEditInputRef,
  handlePatchReport,
  restoringVersion,
  handleRestoreVersion,
  docsPanelOpenRef,
  setDocsPanelOpen,
  docsPanelOpen,
  documents,
  bubbleDoc,
  setBubbleDoc,
  setBubbleViewer,
  bubbleViewer,
  openBubbleDoc,
  confirmingDeleteId,
  setConfirmingDeleteId,
  removeDocument,
  isUploading,
  bubbleScrollRef,
  bubbleHighlightRef,
  sessionIdRef,
  currentUser,
  showAuthModal,
  logout,
  onRenameConversation,
}: ChatHeaderProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [editingTitle, setEditingTitle] = useState(false);
  const [editTitleValue, setEditTitleValue] = useState("");
  const editTitleRef = useRef<HTMLInputElement>(null);
  const getApiBase = () =>
    ((typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000");

  return (
    <header className="h-14 border-b border-border flex items-center px-4 gap-2 shrink-0 bg-background relative">
      <Link to="/">
        <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
        </Button>
      </Link>
      <div className="flex items-center gap-2 min-w-0">
        <Logo className="h-7 w-7" />
        <span className="font-heading font-semibold text-sm text-foreground">Bimlo Copilot</span>
      </div>
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 max-w-[calc(100%-18rem)] w-full text-center px-4 pointer-events-none hidden md:block">
        {editingTitle ? (
          <input
            ref={editTitleRef}
            style={{ pointerEvents: "auto" }}
            value={editTitleValue}
            onChange={e => setEditTitleValue(e.target.value)}
            onBlur={() => {
              const trimmed = editTitleValue.trim();
              if (trimmed && activeConversation && onRenameConversation) {
                onRenameConversation(activeConversation.id, trimmed);
              }
              setEditingTitle(false);
            }}
            onKeyDown={e => {
              if (e.key === "Enter") { (e.target as HTMLInputElement).blur(); }
              if (e.key === "Escape") { setEditingTitle(false); }
            }}
            maxLength={80}
            className="text-sm font-semibold text-foreground bg-transparent border-b border-primary/40 outline-none text-center w-full max-w-[18rem] mx-auto px-1 py-0.5"
          />
        ) : (
          <p
            className="text-sm font-semibold text-foreground truncate mx-auto max-w-[18rem] cursor-pointer hover:text-primary/80 transition-colors pointer-events-auto"
            onClick={() => {
              if (!activeConversation) return;
              setEditTitleValue(activeConversation.title || "");
              setEditingTitle(true);
              setTimeout(() => editTitleRef.current?.select(), 10);
            }}
            title={t("chat.edit_title", "Click to rename")}
          >
            <TypewriterText
              key={(activeConversation?.id ?? "new-conversation") + "-" + (activeConversation?.title ?? "")}
              text={activeConversation?.initialTitle ?? activeConversation?.title ?? "New conversation"}
              speed={28}
              render={partial => <span className="inline-block align-middle">{partial}</span>}
            />
          </p>
        )}
      </div>
      <div className="ms-auto flex items-center gap-2">

        {/* ── Conversations bubble ── */}
        <div className="relative" data-convs-panel>
          <button
            onClick={() => {
              const next = !convsPanelOpen;
              setConvsPanelOpen(next);
              if (next) loadConversationsFromDB();
            }}
            className={`relative flex items-center gap-2 ps-3 pe-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
              convsPanelOpen
                ? "bg-primary text-primary-foreground border-primary shadow-sm"
                : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5 shrink-0" />
            <span>{t("chat.conversations")}</span>
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
                className="absolute end-0 top-full mt-2 w-72 bg-background border border-border rounded-2xl shadow-2xl overflow-hidden z-[70] flex flex-col"
                style={{ transformOrigin: "top right", maxHeight: 480 }}
              >
                {/* Header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4 text-primary" />
                    <span className="text-sm font-semibold text-foreground">{t("chat.conversations")}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => { startNewConversation(); setConvsPanelOpen(false); }}
                      className="flex items-center gap-1 text-[11px] text-primary hover:text-primary/80 px-2 py-1 rounded-lg hover:bg-primary/10 transition-colors font-medium"
                    >
                      <Plus className="h-3 w-3" /> {t("chat.new")}
                    </button>
                    <button onClick={() => setConvsPanelOpen(false)} className="p-1 rounded-md text-muted-foreground/50 hover:text-foreground hover:bg-muted/50 transition-colors">
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Search + sort */}
                <div className="px-3 pt-2.5 pb-2 space-y-2 shrink-0">
                  <div className="relative">
                    <Search className="absolute start-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/50 pointer-events-none" />
                    <input
                      value={historySearch}
                      onChange={e => setHistorySearch(e.target.value)}
                      placeholder={t("chat.search_conversations")}
                      className="w-full ps-7 pe-3 py-1.5 text-xs bg-muted/50 rounded-lg border border-transparent focus:border-primary/30 focus:bg-background focus:outline-none text-foreground placeholder:text-muted-foreground/50 transition-all"
                    />
                  </div>
                  <button
                    onClick={() => setHistorySort(s => s === "newest" ? "oldest" : "newest")}
                    className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground px-1 py-0.5 rounded transition-colors"
                  >
                    <Clock className="h-3 w-3" />
                    {historySort === "newest" ? t("chat.newest_first") : t("chat.oldest_first")}
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
                      <p className="text-xs text-muted-foreground font-medium">{t("chat.no_conversations_yet")}</p>
                      <p className="text-[11px] text-muted-foreground/60">{t("chat.start_chatting_hint")}</p>
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
                              {pendingConvIds[conv.id] ? (
                                <span className="me-2 inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
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
          ref={reportsPanelOpenRef}
        >
            {/* Pill button */}
            <button
              onClick={() => {
                const next = !reportsPanelOpenRef.current;
                reportsPanelOpenRef.current = next;
                setReportsPanelOpen(next);
                if (!next) { setActiveReport(null); setReportEditMode(false); setPreviewedVersion(null); }
              }}
              className={`relative flex items-center gap-2 ps-3 pe-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
                reportsPanelOpen
                  ? "bg-primary text-primary-foreground border-primary shadow-sm"
                  : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
              }`}
              title={t("chat.generated_reports")}
            >
              <BookOpen className="h-3.5 w-3.5 shrink-0" />
              <span>{t("chat.reports")}</span>
              <span className={`inline-flex items-center justify-center h-4 min-w-4 px-1 rounded-full text-[10px] font-bold ${
                reportsPanelOpen ? "bg-primary-foreground/20 text-primary-foreground" : "bg-primary/15 text-primary"
              }`}>
                {reports.length}
              </span>
              {isGeneratingReport && (
                <Loader2 className="h-3 w-3 animate-spin ms-0.5" />
              )}
            </button>
            <div
              className="absolute end-0 top-full mt-2 bg-background border border-border rounded-2xl shadow-2xl z-[9999] flex flex-col"
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
                      <span className="text-sm font-semibold text-foreground">{t("chat.generated_reports")}</span>
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
                              title={t("chat.download_pdf")}
                            >
                              {downloadingReportId === report.report_id
                                ? <Loader2 className="h-3 w-3 animate-spin" />
                                : <ScrollText className="h-3 w-3" />
                              }
                            </button>
                            <button
                              onClick={e => { e.stopPropagation(); handleDeleteReport(report.report_id); }}
                              className="p-0.5 rounded text-muted-foreground/50 hover:text-destructive transition-colors"
                              title={t("chat.delete_report")}
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
                      {t("chat.report_hint")}
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
                      title={t("chat.back_to_reports")}
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
                          <span className="text-amber-500/80 font-medium">{t("chat.previewing_version", { version: previewedVersion.version })}</span>
                        ) : (
                          activeReport.version > 1 && <span className="text-primary/60">v{activeReport.version}</span>
                        )}
                        {isPatchingReport && (
                          <span className="inline-flex items-center gap-1 text-primary animate-pulse">
                            <RefreshCw className="h-2.5 w-2.5 animate-spin" />{t("chat.editing")}
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
                          title={t("chat.version_history")}
                        >
                          <Clock className="h-3.5 w-3.5" />
                          <span className="hidden sm:inline">v{activeReport.version}</span>
                        </button>
                      )}
                      {/* PDF download */}
                      <button
                        onClick={() => handleDownloadReport(activeReport, "pdf")}
                        className="p-1.5 rounded-lg text-muted-foreground hover:text-primary hover:bg-muted/50 transition-colors"
                          title={t("chat.download_pdf")}
                      >
                        {downloadingReportId === activeReport.report_id
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          : <ScrollText className="h-3.5 w-3.5" />
                        }
                      </button>
                      <button
                        onClick={() => handleDeleteReport(activeReport.report_id)}
                        className="p-1.5 rounded-lg text-muted-foreground hover:text-destructive hover:bg-muted/50 transition-colors"
                        title={t("chat.delete_report")}
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
                          {t("chat.version_history")}
                        </span>
                        <div className="flex items-center gap-2">
                          {previewedVersion && (
                            <button
                              onClick={() => setPreviewedVersion(null)}
                              className="text-[10px] text-primary hover:text-primary/80 font-medium flex items-center gap-1 transition-colors"
                            >
                              <ArrowLeft className="h-2.5 w-2.5" />
                              {t("chat.back_to_current")}
                            </button>
                          )}
                          <span className="text-[9px] text-muted-foreground/60 bg-muted px-1.5 py-0.5 rounded-full">
                            {t("chat.versions_count", { count: activeReport.versions.length })}
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
                                "shrink-0 flex flex-col items-start gap-1 px-3 py-2.5 rounded-xl border text-start w-44",
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
                                  {isLive ? t("chat.current") : `v${snap.version}`}
                                </span>
                                {isRestoring
                                  ? <Loader2 className="h-3 w-3 animate-spin text-primary shrink-0" />
                                  : isLive && (
                                    <span className="shrink-0 text-[8px] bg-primary/15 text-primary px-1.5 py-0.5 rounded-full font-medium">{t("chat.live")}</span>
                                  )
                                }
                              </div>

                              {/* Title */}
                              <p className="text-[10px] text-muted-foreground leading-snug line-clamp-1 w-full text-start">
                                {snap.title}
                              </p>

                              {/* Date */}
                              <p className="text-[9px] text-muted-foreground/60 leading-snug">
                                {new Date(snap.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                              </p>

                              {/* Edit instruction snippet */}
                              {snap.instruction && (
                                <p className="text-[9px] text-muted-foreground/50 italic leading-snug line-clamp-2 w-full text-start">
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
                                  ↩ {t("chat.restore")}
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
                            {t("chat.previewing_version", { version: previewedVersion.version })}
                          </span>
                          <button
                            onClick={() => setPreviewedVersion(null)}
                            className="text-[10px] text-amber-600/80 dark:text-amber-400/80 underline underline-offset-2 hover:opacity-100 opacity-70 transition-opacity"
                          >
                            {t("chat.return_to_current")}
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
                        {t("chat.edit_report")}
                      </button>
                    ) : (
                      <div className="flex items-center gap-2 w-full bg-muted/40 rounded-xl border border-border px-3 py-2">
                        <textarea
                          ref={reportEditInputRef}
                          value={reportEditInstruction}
                          onChange={e => setReportEditInstruction(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handlePatchReport(); }
                            if (e.key === "Escape") { setReportEditMode(false); setReportEditInstruction(""); }
                          }}
                          placeholder={t("chat.edit_hint")}
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
                            {t("chat.apply")}
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
            className={`relative flex items-center gap-2 ps-3 pe-3.5 py-1.5 rounded-full text-xs font-medium transition-all border ${
              docsPanelOpen
                ? "bg-primary text-primary-foreground border-primary shadow-sm"
                : "bg-muted/60 hover:bg-muted text-muted-foreground hover:text-foreground border-border"
            }`}
            title={t("chat.documents")}
          >
            <FolderOpen className="h-3.5 w-3.5 shrink-0" />
            <span>{t("chat.documents")}</span>
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
            className="absolute end-0 top-full mt-2 bg-background border border-border rounded-2xl shadow-2xl z-[70] flex flex-col"
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
                        <span className="text-sm font-semibold text-foreground">{t("chat.documents")}</span>
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
                          <p className="text-xs text-muted-foreground font-medium">{t("chat.no_documents_yet")}</p>
                          <p className="text-[11px] text-muted-foreground/60">{t("chat.upload_hint")}</p>
                        </div>
                      ) : (
                        <div className="p-2 space-y-0.5">
                          {isUploading && (
                            <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary/5">
                              <Loader2 className="h-4 w-4 text-primary animate-spin shrink-0" />
                              <span className="text-xs text-muted-foreground">{t("chat.uploading_dots")}</span>
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
                                  <p className="text-[10px] text-muted-foreground capitalize group-hover:opacity-0 transition-opacity">{isUploadingDoc ? t("chat.uploading_dots") : doc.doc_type}</p>
                                </div>
                                {!isUploadingDoc && (
                                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
                                    <Eye className="h-3 w-3 text-muted-foreground/50" />
                                    {confirmingDeleteId === doc.document_id ? (
                                      <div className="flex items-center gap-1">
                                        <button
                                          onClick={e => { e.stopPropagation(); removeDocument(doc.document_id); }}
                                          className="p-0.5 rounded-full bg-emerald-500 text-emerald-50 hover:bg-emerald-400 transition-colors"
                                          title={t("chat.confirm_delete_doc")}
                                        >
                                          <Check className="h-3 w-3" />
                                        </button>
                                        <button
                                          onClick={e => { e.stopPropagation(); setConfirmingDeleteId(null); }}
                                          className="p-0.5 rounded-full bg-muted border border-border text-muted-foreground hover:text-foreground hover:bg-muted/80 transition-colors"
                                          title={t("chat.cancel_delete")}
                                        >
                                          <X className="h-3 w-3" />
                                        </button>
                                      </div>
                                    ) : (
                                      <button
                                        onClick={e => { e.stopPropagation(); setConfirmingDeleteId(doc.document_id); }}
                                        className="p-0.5 rounded text-muted-foreground/50 hover:text-destructive transition-colors"
                                        title={t("chat.delete_document")}
                                      >
                                        <X className="h-3 w-3" />
                                      </button>
                                    )}
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
                        title={t("chat.back_to_documents")}
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
                          <p className="text-xs">{t("chat.loading_document")}</p>
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
                          <div className="flex items-center justify-center h-full p-1 overflow-hidden">
                            <ZoomableImage src={bubbleViewer.blobUrl!} alt={bubbleDoc?.filename ?? ''} compact />
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
          className="flex items-center gap-1.5 ps-3 pe-3.5 py-1.5 rounded-full text-xs font-medium border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500/60 transition-all"
        >
          <Phone className="h-3.5 w-3.5 shrink-0" />
          <span>{t("nav.call")}</span>
        </button>
        <LangToggle />
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
  );
}
