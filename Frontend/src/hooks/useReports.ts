import { useState, useRef, useCallback, useEffect } from "react";
import { ReportRecord, Message, createUniqueId } from "@/pages/chatTypes";
import { serializeError } from "@/utils/chatUtils";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys, useReportsQuery } from "@/services/queries";

export function useReports(options: {
  toast: (props: { title: string; description?: string; variant?: "default" | "destructive" }) => void;
  sessionIdRef: React.MutableRefObject<string | null>;
  sessionId: string | null;
  documents: { document_id: string; filename: string; doc_type: string }[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setConversations: React.Dispatch<React.SetStateAction<any[]>>;
  activeConvId: string;
  messages: Message[];
}) {
  const { toast, sessionIdRef, sessionId, documents, setMessages, setConversations, activeConvId, messages } = options;

  const queryClient = useQueryClient();
  const reportsQuery = useReportsQuery(sessionIdRef.current || sessionId);

  const [reports, setReports] = useState<ReportRecord[]>(() => reportsQuery.data ?? []);
  useEffect(() => {
    if (reportsQuery.data) {
      setReports(reportsQuery.data);
    }
  }, [reportsQuery.data]);

  const [activeReport, setActiveReport] = useState<ReportRecord | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isPatchingReport, setIsPatchingReport] = useState(false);
  const [reportEditInstruction, setReportEditInstruction] = useState("");
  const [reportEditMode, setReportEditMode] = useState(false);
  const [downloadingReportId, setDownloadingReportId] = useState<string | null>(null);
  const [deletingReportId, setDeletingReportId] = useState<string | null>(null);
  const reportsPanelRef = useRef<HTMLDivElement>(null);
  const [reportsPanelOpen, setReportsPanelOpen] = useState(false);
  const reportsPanelOpenRef = useRef(false);
  reportsPanelOpenRef.current = reportsPanelOpen;
  const [showVersionHistory, setShowVersionHistory] = useState(false);
  const [restoringVersion, setRestoringVersion] = useState<number | null>(null);
  const [previewedVersion, setPreviewedVersion] = useState<{ version: number; content: string; charts: any[]; title: string } | null>(null);
  const reportEditInputRef = useRef<HTMLTextAreaElement>(null);

  const getApiBase = () =>
    ((typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000");

  const loadReports = async (sid?: string | null) => {
    const resolvedSid = sid ?? sessionIdRef.current ?? sessionId;
    if (!resolvedSid) { setReports([]); return; }
    queryClient.invalidateQueries({ queryKey: queryKeys.reports(resolvedSid) });
  };

  const handleGenerateReport = async (prompt: string, sid?: string, explicitDocs: string[] = []) => {
    const resolvedSid = sid || sessionIdRef.current || sessionId;
    if (!resolvedSid || isGeneratingReport) return;
    setIsGeneratingReport(true);

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
        setConversations(convs => convs.map((conv: any) => {
          if (conv.id !== activeConvId) return conv;
          return { ...conv, messages: msgs };
        }));
        return msgs;
      });
    } catch (err) {
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
    } catch (err) {
      toast({ title: "Restore failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setRestoringVersion(null);
    }
  };

  const fetchSuggestions = useCallback(async (
    userQuery: string,
    assistantReply: string,
    setSuggestions: React.Dispatch<React.SetStateAction<string[]>>,
    setSuggestionsLoading: React.Dispatch<React.SetStateAction<boolean>>,
  ) => {
    setSuggestionsLoading(true);
    setSuggestions([]);
    try {
      const base = getApiBase();
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
  }, [documents]);

  const expandSuggestion = useCallback(async (
    label: string,
    isGeneral: boolean,
    setExpandingSuggestion: React.Dispatch<React.SetStateAction<string | null>>,
  ): Promise<string> => {
    setExpandingSuggestion(label);
    let expandedPrompt = label;
    try {
      const base = getApiBase();
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
    }
    setExpandingSuggestion(null);
    return expandedPrompt;
  }, [messages, documents]);

  return {
    reports, setReports,
    activeReport, setActiveReport,
    loadReports,
    handleGenerateReport,
    handlePatchReport,
    handleDeleteReport,
    handleDownloadReport,
    handleRestoreVersion,
    isGeneratingReport, setIsGeneratingReport,
    isPatchingReport, setIsPatchingReport,
    reportEditInstruction, setReportEditInstruction,
    reportEditMode, setReportEditMode,
    downloadingReportId, setDownloadingReportId,
    deletingReportId, setDeletingReportId,
    reportsPanelRef,
    reportsPanelOpen, setReportsPanelOpen,
    reportsPanelOpenRef,
    showVersionHistory, setShowVersionHistory,
    restoringVersion, setRestoringVersion,
    previewedVersion, setPreviewedVersion,
    reportEditInputRef,
    fetchSuggestions,
    expandSuggestion,
  };
}
