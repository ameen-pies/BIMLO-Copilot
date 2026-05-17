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
import {
  ThinkingStep,
  Message,
  FactChip,
  Conversation,
  ChartRecord,
  VersionInfo,
  ReportRecord,
  createUniqueId,
} from "@/pages/chatTypes";
import {
  hasBlockMarkdown,
  stripLinks,
  collapseInlineBreaks,
  renderInlineMarkdown,
  renderTextSegment,
  UNIT_PATTERN,
  UNIT_RE,
  isSpecNumber,
  highlightNumbers,
  injectChips,
  parseSegments,
  mergeAdjacentSameSourceSegments,
  parseSourceGroups,
  norm,
  escapeHtml,
  highlightFact,
  buildHighlightedExcerpt,
  findExcerptRange,
  HIGHLIGHT_COLOR,
  serializeError,
  TrieNode,
  makeTrie,
  trieInsert,
  trieFind,
  trieCollect,
  tokenise,
  INSIGHT_LABELS,
  INSIGHT_ICONS,
  splitIntoSentences,
  STROKES_DARK,
  STROKES_LIGHT,
  GRAD_STOPS_DARK,
  GRAD_STOPS_LIGHT,
  makeGradientPlugin,
  applyThemeToChart,
  buildInitialCfg,
  downloadChartPng,
} from "@/utils/chatUtils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import BorderGlow from "@/components/BorderGlow";
import { useAuth } from "@/context/AuthContext";
import { ProfileBubble } from "@/components/Navbar";
import { ShinyText } from "@/components/chat/ShinyText";
import { ZoomableImage } from "@/components/chat/ZoomableImage";
import { PdfViewer } from "@/components/chat/PdfViewer";
import { XeokitViewer } from "@/components/chat/XeokitViewer";
import { DocumentViewer } from "@/components/chat/DocumentViewer";
import { WelcomeSplash } from "@/components/chat/WelcomeSplash";
import { WikiThumbnail } from "@/components/chat/WikiThumbnail";
import { VoiceMessageBubble } from "@/components/chat/VoiceMessageBubble";
import {
  ChartMessage,
  ChartClarification,
  ChartInsights,
  ChartAnalytics,
  ChartGroup,
} from "@/components/chat/ChartMessage";
import {
  renderContent,
  renderDocumentContent,
  ViewerState,
} from "@/components/chat/chatRenderers";
import { queryBackend, fetchAICompletion } from "@/components/chat/queryHelpers";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ReportCard } from "@/components/chat/ReportCard";
import { ReportInlineChart, renderReportContent } from "@/components/chat/ChartRenderer";
import { ThinkingTrace } from "@/components/chat/ThinkingTrace";
import { useNotifications } from "@/hooks/useNotifications";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { AutocompletePanel } from "@/components/chat/AutocompletePanel";
import { ChatHeader } from "@/components/chat/ChatHeader";


























export function useChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [pendingDocIds, setPendingDocIds] = useState<string[]>([]);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [duplicateBanner, setDuplicateBanner] = useState<string | null>(null);
  const duplicateBannerTimeoutRef = useRef<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    return () => {
      if (duplicateBannerTimeoutRef.current) {
        window.clearTimeout(duplicateBannerTimeoutRef.current);
      }
    };
  }, []);

  const createPreviewUrlForFile = (file: File) => {
    const ext = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`;
    const isImage = IMAGE_EXTS.includes(ext);
    const isPdf = ext === '.pdf';
    const isCad = CAD_EXTS.includes(ext);
    if (!isImage && !isPdf && !isCad) return null;
    const url = URL.createObjectURL(file);
    return {
      url,
      type: isPdf ? 'pdf' : isImage ? 'image' : 'cad' as const,
      ifcBlobUrl: ext === '.ifc' ? url : undefined,
    };
  };

  const revokePreviewUrl = (documentId: string) => {
    const cached = blobUrlMapRef.current.get(documentId);
    if (!cached) return;
    if (cached.url) URL.revokeObjectURL(cached.url);
    if (cached.ifcBlobUrl && cached.ifcBlobUrl !== cached.url) URL.revokeObjectURL(cached.ifcBlobUrl);
    blobUrlMapRef.current.delete(documentId);
  };

  const removePendingAttachment = (documentId: string) => {
    setPendingDocIds(prev => prev.filter(id => id !== documentId));
    setConfirmingDeleteId(null);
  };
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
  const activeConversation = conversations.find(c => c.id === activeConvId);
  // Tracks IDs of conversations that have been deleted so in-flight responses
  // don't re-insert them into the sidebar.
  const deletedConvIdsRef = useRef<Set<string>>(new Set());
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
      // Never re-insert a conversation that was explicitly deleted by the user
      if (deletedConvIdsRef.current.has(convId)) return;
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
                  title: c.titleLocked ? c.title : (title || c.title),
                  initialTitle: c.initialTitle ?? title ?? c.title,
                }
              : c
          );
        }
        // Double-check the set inside the updater too (state closure timing)
        if (deletedConvIdsRef.current.has(convId)) return prev;
        // Never create a new sidebar entry with a blank, whitespace-only, or
        // trivially short title — this is the root cause of the ghost "chat"
        // conversation that appears whenever the user types a short phrase and
        // the first updateConversationMessages call fires before the assistant
        // has replied. We only insert when we have a real, non-trivial title.
        const resolvedTitle = (title || "").trim();
        if (!resolvedTitle || resolvedTitle.length < 2) return prev;
        return [
          {
            id: convId,
            title: resolvedTitle || "Untitled",
            initialTitle: resolvedTitle || "Untitled",
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
  // ── Model selector ──────────────────────────────────────────────────────
  type ModelProvider = "cf_primary" | "cf_backup" | "groq" | "nvidia";
  const [selectedModel, setSelectedModel] = useState<ModelProvider>("cf_primary");
  // Ref always tracks latest selectedModel so runStreamingQuery's useCallback
  // can read it without a stale closure — avoids adding selectedModel to the
  // dep array which would break in-flight streams on every model switch.
  const selectedModelRef = useRef<ModelProvider>("cf_primary");
  useEffect(() => { selectedModelRef.current = selectedModel; }, [selectedModel]);

  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    console.log("[Chat] selected model:", selectedModel);
  }, [selectedModel]);

  // Close model dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const MODEL_OPTIONS: { id: ModelProvider; label: string; shortLabel: string; desc: string; color: string }[] = [
    { id: "cf_primary",  label: "Cloudflare Primary",  shortLabel: "CF Primary",  desc: "Primary Cloudflare Workers AI",    color: "#f97316" },
    { id: "cf_backup",   label: "Cloudflare Backup",   shortLabel: "CF Backup",   desc: "Backup Cloudflare Workers AI",     color: "#3b82f6" },
    { id: "groq",        label: "Groq",                shortLabel: "Groq",        desc: "Groq (llama-3.3-70b-versatile)",   color: "#10b981" },
    { id: "nvidia",      label: "MiniMax M2.7",         shortLabel: "MiniMax",     desc: "MiniMax M2.7 via NVIDIA NIM",      color: "#76b900" },
  ];

  const isLoading_ref = useRef(false);
  const [isLoading, setIsLoading] = useState(false);
  // thinkingVisible stays true briefly after isLoading → false so the
  // AnimatePresence exit animation has time to run before unmounting.
  const [thinkingVisible, setThinkingVisible] = useState(false);
  const thinkingLingerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep thinking panel visible for 350ms after loading ends so exit anim plays
  React.useEffect(() => {
    if (isLoading) {
      if (thinkingLingerRef.current) clearTimeout(thinkingLingerRef.current);
      setThinkingVisible(true);
    } else {
      thinkingLingerRef.current = setTimeout(() => setThinkingVisible(false), 350);
    }
    return () => { if (thinkingLingerRef.current) clearTimeout(thinkingLingerRef.current); };
  }, [isLoading]);
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
  const pageDragCounterRef = useRef(0); // capture-phase count for page-wide overlay
  const [isChatDragOver, setIsChatDragOver] = useState(false);
  const chatDragCounterRef = useRef(0); // track nested drag enter/leave for chat area

  // Safety reset: if the user drags out of the browser window, clear the overlay
  useEffect(() => {
    const reset = () => {
      dragCounterRef.current = 0;
      pageDragCounterRef.current = 0;
      setIsDragOver(false);
      chatDragCounterRef.current = 0;
      setIsChatDragOver(false);
    };
    const handleWindowDragLeave = (e: DragEvent) => { if (!e.relatedTarget) reset(); };
    window.addEventListener('dragend', reset);
    document.addEventListener('dragleave', handleWindowDragLeave);
    return () => {
      window.removeEventListener('dragend', reset);
      document.removeEventListener('dragleave', handleWindowDragLeave);
    };
  }, []);

  const handleFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    let sid = sessionIdRef.current ?? sessionId;
    if (!sid) {
      sid = crypto.randomUUID?.() ?? `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setSessionId(sid);
      sessionIdRef.current = sid;
    }
    setIsUploading(true);
    for (const file of arr) {
      if (documents.some(d => d.filename.toLowerCase() === file.name.toLowerCase())) {
        if (duplicateBannerTimeoutRef.current) window.clearTimeout(duplicateBannerTimeoutRef.current);
        setDuplicateBanner(`Duplicate file detected 🙂`);
        duplicateBannerTimeoutRef.current = window.setTimeout(() => setDuplicateBanner(null), 4500);
        continue;
      }
      // Optimistic placeholder in docs panel
      const placeholderId = `uploading-${Date.now()}-${file.name}`;
      setDocuments(prev => [...prev, {
        document_id: placeholderId,
        filename: file.name,
        doc_type: "uploading",
        timestamp: new Date().toISOString(),
      } as any]);
      try {
        const res = await api.uploadDocument(file, sid);
        const returnedSid = res.session_id ?? sid;
        const sessionChanged = returnedSid !== sid;
        if (sessionChanged) {
          sid = returnedSid;
          setSessionId(returnedSid);
          sessionIdRef.current = returnedSid;
          // loadDocuments replaces the full list from the server — which already
          // includes the newly uploaded doc — so we must NOT also manually concat
          // below, or the doc appears twice in the panel.
          await loadDocuments(returnedSid);
        }
        const ext = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`;
        const preview = createPreviewUrlForFile(file);
        if (preview) {
          blobUrlMapRef.current.set(res.document_id, {
            url: preview.url,
            type: preview.type,
            cadSummary: res.cad_summary,
            ifcBlobUrl: preview.ifcBlobUrl,
          });
        } else if (res.cad_summary) {
          blobUrlMapRef.current.set(res.document_id, {
            url: blobUrlMapRef.current.get(res.document_id)?.url ?? '',
            type: 'cad',
            cadSummary: res.cad_summary,
          });
        }
        // If the server confirmed it processed this as an image, ensure the blob map
        // has the correct type so the viewer opens as an image (not text).
        if (res.is_image && !blobUrlMapRef.current.has(res.document_id)) {
          const objectUrl = URL.createObjectURL(file);
          blobUrlMapRef.current.set(res.document_id, { url: objectUrl, type: 'image' });
        }
        setPendingDocIds(prev => [...prev, res.document_id]);
        // Only manually update the doc list when the session did NOT change.
        // If it changed, loadDocuments() above already refreshed the full list from
        // the server (which includes this doc), so adding it again would duplicate it.
        if (!sessionChanged) {
          setDocuments(prev => {
            const withoutPlaceholder = prev.filter(d => d.document_id !== placeholderId);
            const newDoc = {
              document_id: res.document_id,
              filename: res.filename,
              doc_type: file.name.split(".").pop() ?? "doc",
              timestamp: new Date().toISOString(),
            } as any;
            const merged = [...withoutPlaceholder.filter(d => d.document_id !== res.document_id), newDoc];
            return merged;
          });
        }
        // File uploaded successfully, no toast needed
        // toast({ title: "File uploaded", description: `${res.filename} ready` });
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
  const transcribeAbortRef = useRef<AbortController | null>(null);
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

        // Snapshot any documents waiting to be attached before transcription
        const snapshotDocIds = [...pendingDocIds];
        // Immediately add a voice message bubble (transcript will fill in async)
        const voiceMsgId = createUniqueId("msg-");
        const voiceMsg: Message = {
          id: voiceMsgId,
          role: "user",
          content: "", // will be filled with transcript once available
          voiceBlobUrl: blobUrl,
          voiceDuration: duration,
          voiceTranscript: undefined, // pending
          voiceWaveform: capturedSamples,
          attachedDocIds: snapshotDocIds.length > 0 ? snapshotDocIds : undefined,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, voiceMsg]);

        try {
          const formData = new FormData();
          formData.append("audio", blob, "recording.webm");
          formData.append("mime_type", mimeType);

          transcribeAbortRef.current = new AbortController();
          const res = await fetch(`${getApiBase()}/transcribe`, {
            method: "POST",
            body: formData,
            signal: transcribeAbortRef.current.signal,
          });
          transcribeAbortRef.current = null;

          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error((err as Record<string, string>).detail ?? `HTTP ${res.status}`);
          }

          const data = await res.json() as { transcript: string };
          const transcript = data.transcript?.trim();

          if (transcript) {
            // Use messagesRef to avoid stale closure — always reflects active conversation
            const updatedVoiceMsg: Message = { ...voiceMsg, content: transcript, voiceTranscript: transcript };
            const currentMsgs = messagesRef.current;
            const voiceMessages = currentMsgs.some(m => m.id === voiceMsgId)
              ? currentMsgs.map(m => m.id === voiceMsgId ? updatedVoiceMsg : m)
              : [...currentMsgs, updatedVoiceMsg];
            // Transcript leads so the language judge reads the user's language first.
            // The trailing tag signals voice input without polluting language detection.
            const voiceQuery = `${transcript}\n\n[voice_input=true]`;
            const convId = ensureActiveConversationId();
            const transcriptTitle = transcript.length > 50 ? transcript.slice(0, 50) + "…" : transcript;
            const title = activeConversation?.title ?? transcriptTitle;
            // Batch all synchronous state updates together before the first await so
            // React 18 can flush them in a single render — avoids a visible flicker.
            // Note: do NOT call setIsLoading(true) here; runStreamingQuery owns that
            // state and sets it itself, so a redundant call here would cause an extra
            // render cycle right before runStreamingQuery's own synchronous updates.
            updateConversationMessages(convId, voiceMessages, "", title);
            setPendingDocIds([]);
            markPendingConversation(convId);
            setTypingConvId(convId);
            if (!notifyDismissed) setShowNotifyBanner(true);
            try {
              const assistantMsg = await runStreamingQuery(voiceQuery, undefined, snapshotDocIds);
              if (!assistantMsg) return;
              const updated = [...voiceMessages, assistantMsg];
              const preview = assistantMsg.content.replace(/\[.*?\]/g, "").replace(/#{1,3}\s/g, "").slice(0, 80) + "…";
              updateConversationMessages(convId, updated, preview, title);
              // Persist voice reply to DB
              saveConversationToDB(convId, sessionIdRef.current ?? "", title, preview, updated);
              setTypingMessageId(assistantMsg.id);
              fetchSuggestions(transcript, assistantMsg.content);
              fireNotification();
            } catch (error) {
              const msg = serializeError(error);
              const errorMsg: Message = {
                id: createUniqueId("msg-"),
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
          // Silently drop AbortError — user deleted the chat or cancelled
          if (err instanceof Error && err.name === "AbortError") return;
          console.error("Transcription error:", err);
          setMessages(prev => prev.map(m =>
            m.id === voiceMsgId
              ? { ...m, voiceTranscript: "(transcription failed)" }
              : m
          ));
          toast({ title: "Transcription failed", description: err instanceof Error ? err.message : "Could not reach the transcription service.", variant: "destructive" });
        } finally {
          setVoiceState("idle");
          transcribeAbortRef.current = null;
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
  }, [voiceState, toast, pendingDocIds]);

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
  // AbortController for any in-flight /autocomplete request — cancelled on send
  const ghostAbortRef = useRef<AbortController | null>(null);

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
    // Only generate a title after the very first exchange (1 user msg + 1 assistant reply).
    // If there are already 2+ user messages, the conversation was already titled — skip.
    if (userMsgs.length > 1) return;

    try {
      const res = await fetch(`${getApiBase()}/title`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
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
      setConversations(convs => {
        const _conv = convs.find(c => c.id === convId);
        if (_conv && !_conv.titleLocked) {
          const messagesToSave = (_conv.messages?.length ?? 0) > 0 ? _conv.messages : messages;
          saveConversationToDB(
            convId,
            (_conv as any).sessionId ?? sessionIdRef.current ?? "",
            smartTitle,
            _conv.preview,
            messagesToSave
          );
        }
        return convs.map(c =>
          // Also update initialTitle so the topbar TypewriterText re-renders immediately
          c.id === convId ? { ...c, title: smartTitle, initialTitle: smartTitle, titleLocked: true } : c
        );
      });
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
    const sessionQuery = sessionIdRef.current ? `?session_id=${encodeURIComponent(sessionIdRef.current)}` : "";

    let blob: Blob;
    if (type === "image") {
      // Images are served from the dedicated /image endpoint (returns raw bytes, not text)
      blob = await (api as any).getImageDocument(doc.document_id, sessionIdRef.current ?? "");
    } else {
      const res = await fetch(`${base}/documents/${doc.document_id}/download${sessionQuery}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      blob = await res.blob();
    }
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
        console.log(`[openBubbleDoc] fetching content for doc_id=${doc.document_id} session_id=${sessionIdRef.current}`);
        const res = await (api as any).getDocumentContent(doc.document_id, sessionIdRef.current ?? undefined);
        console.log(`[openBubbleDoc] success, content length=${res.content?.length}`);
        setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, content: res.content, loading: false } : p);
      } catch (err) {
        console.error(`[openBubbleDoc] FAILED doc_id=${doc.document_id}`, err);
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
          const res = await (api as any).getDocumentContent(doc.document_id, sessionIdRef.current ?? undefined);
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
      const res = await (api as any).getDocumentContent(doc.document_id, sessionIdRef.current ?? undefined);
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
      const effectiveSid = sid ?? sessionIdRef.current;
      if (!effectiveSid) {
        setDocuments([]);
        return;
      }
      // Pass session_id so the backend returns only this session's documents.
      const response = await api.listDocuments(effectiveSid);
      const uniqueDocs = response.documents.filter((doc, index, arr) =>
        arr.findIndex(d => d.document_id === doc.document_id) === index
      );
      setDocuments(uniqueDocs);
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
    const token = currentUser?.token ?? "";
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const loadConversationsFromDB = async () => {
    try {
      const base = getApiBase();
      const res = await fetch(`${base}/auth/conversations`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
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
    if (deletedConvIdsRef.current.has(convId)) {
      console.warn("saveConversationToDB skipped: conversation deleted", convId);
      return;
    }
    try {
      const base = getApiBase();
      const authHeader = getAuthHeader();
      console.log("💾 saveConversationToDB", { convId, sessionId: sid, title, preview, messageCount: msgs.length, currentUser: !!currentUser });
      const res = await fetch(`${base}/auth/conversations/save`, {
        method: "POST",
        credentials: "include",
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
            voiceDuration:   m.voiceDuration   ?? undefined,
            voiceWaveform:   m.voiceWaveform   ?? undefined,
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

  const loadReports = async (sid?: string | null) => {
    try {
      const base = getApiBase();
      const resolvedSid = sid ?? sessionIdRef.current ?? sessionId;
      if (!resolvedSid) { setReports([]); return; }
      const res = await fetch(`${base}/reports?session_id=${encodeURIComponent(resolvedSid)}`);
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
      // Report generated successfully, no toast needed
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
      // Report updated silently - v${updated.version}` });
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
      // Version restored successfully, no toast needed
    } catch (err) {
      toast({ title: "Restore failed", description: serializeError(err), variant: "destructive" });
    } finally {
      setRestoringVersion(null);
    }
  };

  const removeDocument = async (documentId: string) => {
    setConfirmingDeleteId(null);

    try {
      await api.deleteDocument(documentId, sessionIdRef.current ?? undefined);
      revokePreviewUrl(documentId);
      // Document deleted successfully, no toast needed

      await loadDocuments(sessionIdRef.current);
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
    activeConvIdRef.current = "";
    setSessionId(null);
    sessionIdRef.current = null;
    setMessages([]);
    setDocuments([]);
    setReports([]);
  };

  const ensureActiveConversationId = () => {
    if (activeConvIdRef.current) return activeConvIdRef.current;
    const newId = createUniqueId("conv-");
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
      activeConvIdRef.current = conv.id;
      setSessionId(null);
      sessionIdRef.current = null;
      setMessages(conv.messages.map(m => {
        const t = (m.analytics as any)?.type;
        if (t === "chart_clarification" || t === "report_chart_clarification") {
          return { ...m, analytics: null };
        }
        return m;
      }));
      setDocuments([]);
      setConvLoading(false);
      if (sid) { setSessionId(sid); sessionIdRef.current = sid; loadDocuments(sid); loadReports(sid); }
      setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
      return;
    }

    // 2. Fetch from DB — auth conversations endpoint (has full payload)
    const base = getApiBase();
    const rawToken = currentUser?.token ?? "";

    setActiveConvId(conv.id);
    activeConvIdRef.current = conv.id;
    setSessionId(null);
    sessionIdRef.current = null;
    setConvLoading(true);
    setMessages([]);
    setDocuments([]);
    setReports([]);
    const abortLoad = () => {
      setConvLoading(false);
      setActiveConvId("");
      activeConvIdRef.current = "";
      setMessages([]);
    };

    try {
      const authHeaders: Record<string, string> = {
        "Content-Type": "application/json",
        ...(rawToken ? { Authorization: `Bearer ${rawToken}` } : {}),
      };
      const res = await fetch(`${base}/auth/conversations/${conv.id}`, { headers: authHeaders, credentials: "include" });
      if (res.ok) {
        const data = await res.json();
        console.log("✅ loadConversation DB response", { conversationId: conv.id, messageCount: (data.messages ?? []).length, sessionId: data.session_id });
        const restored: Message[] = (data.messages ?? []).map((m: any) => {
          const merged = {
            // Spread the raw message first (captures any flat rich fields the backend returns)
            ...m,
            // Then spread payload if the backend nests rich fields there
            ...(m.payload ?? {}),
            // Always ensure these core fields are correct types and can't be overwritten
            id:        m.id ?? createUniqueId("msg-"),
            role:      m.role as "user" | "assistant",
            content:   m.content ?? "",
            timestamp: new Date(m.timestamp ?? (m.payload?.timestamp) ?? Date.now()),
          };
          // Strip transient clarification analytics on restore — the user already
          // acted on these and they must not re-appear as clickable options after refresh.
          const analyticsType = merged.analytics?.type;
          if (
            analyticsType === "chart_clarification" ||
            analyticsType === "report_chart_clarification"
          ) {
            merged.analytics = null;
          }
          return merged;
        });
        setMessages(restored);
        setConvLoading(false);
        setConversations(prev => prev.map(c => c.id === conv.id ? { ...c, messages: restored } : c));
        if (data.session_id) { setSessionId(data.session_id); sessionIdRef.current = data.session_id; loadDocuments(data.session_id); loadReports(data.session_id); }
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
            setSessionId(sid); sessionIdRef.current = sid; loadDocuments(sid); loadReports(sid);
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
    // Mark as deleted first so any in-flight response won't re-insert it
    deletedConvIdsRef.current.add(convId);
    setConversations(prev => prev.filter(c => c.id !== convId));
    if (convId === activeConvId) {
      // Tell the backend to cancel any in-flight generation to avoid wasting tokens
      const sid = sessionIdRef.current || sessionId;
      if (sid) {
        const base =
          (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
          "http://localhost:8000";
        fetch(`${base}/query-stream/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sid }),
        }).catch(() => {/* non-critical */});
      }
      // Abort any in-flight transcription or streaming for this conversation
      transcribeAbortRef.current?.abort();
      transcribeAbortRef.current = null;
      mediaRecorderRef.current?.stop();
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      setVoiceState("idle");
      setIsLoading(false);
      startNewConversation();
    }
    if (currentUser) {
      try {
        const base = getApiBase();
        await fetch(`${base}/auth/conversations/${convId}`, {
          method: "DELETE",
          credentials: "include",
          headers: getAuthHeader() as any,
        });
      } catch (e) { console.error("deleteConversation from DB failed:", e); }
    }
  };

  // ── Shared streaming query runner ────────────────────────────────────────
  // Used by handleSend, handleRedo, and handleEditSubmit so they all get
  // thinking steps and the SSE stream.
  const runStreamingQuery = useCallback(async (query: string, forceRoute?: string, pendingDocIdsToCommit: string[] = []) => {
    console.log("[Chat] runStreamingQuery", { query, forceRoute, selectedModel, pendingDocIdsToCommit });
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);
    // Always set typingConvId here so the thinking indicator appears regardless
    // of whether the caller is handleSend or an onSelect clarification handler.
    if (activeConvIdRef.current) setTypingConvId(activeConvIdRef.current);

    try {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";

      abortControllerRef.current = new AbortController();

      // Build auth header so the backend links auto-saved messages to the user account
      const rawToken = currentUser?.token ?? "";
      const streamHeaders: Record<string, string> = { "Content-Type": "application/json" };
      if (rawToken) streamHeaders["Authorization"] = `Bearer ${rawToken}`;

      // Resolve the filenames of docs attached to THIS message so the backend
      // can use them as the highest-priority scope signal (no guessing needed).
      const attachedDocFilenames = pendingDocIdsToCommit
        .map(id => documents.find(d => d.document_id === id)?.filename)
        .filter((f): f is string => Boolean(f));

      const res = await fetch(`${base}/query-stream`, {
        method: "POST",
        credentials: "include",
        headers: streamHeaders,
        body: JSON.stringify({
          query,
          top_k: 5,
          ...(sessionId ? { session_id: sessionId } : {}),
          ...(forceRoute ? { force_route: forceRoute } : {}),
          // Tell the backend to save messages under the frontend's conversation ID
          // so loadConversation finds them without a mismatch.
          ...(activeConvIdRef.current ? { conversation_id: activeConvIdRef.current } : {}),
          // Commit any files uploaded before this message so the backend indexes them
          // into the user/session collection before running the vector search.
          pending_doc_ids: pendingDocIdsToCommit,
          // Pass resolved filenames so the backend knows EXACTLY which docs are attached
          // to THIS message — used as the highest-priority scope signal in the router.
          attached_doc_filenames: attachedDocFilenames,
          preferred_provider: selectedModelRef.current,
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
                      // Report generated successfully, no toast needed
                    })
                    .catch(() => { if (attempt < 5) setTimeout(() => _fetchReport(attempt + 1), 3000 * (attempt + 1)); });
                };
                // Small initial delay — report save happens synchronously before SSE fires,
                // but disk write may lag slightly. Start at 500ms.
                setTimeout(() => _fetchReport(), 500);
              }
              resultMsg = {
                id: createUniqueId("msg-"),
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
      } else {
        // Stream ended without a result event — the backend threw an unhandled
        // exception AFTER sending 200 headers (common in Docker when a LangGraph
        // node crashes mid-flight). Surface a real error so the UI doesn't just
        // silently remove the bouncing dots.
        throw new Error("The server stopped responding before sending a result. Please try again.");
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
                      // Report generated successfully, no toast needed
                    })
                    .catch(() => { if (attempt < 5) setTimeout(() => _fetchReport2(attempt + 1), 3000 * (attempt + 1)); });
                };
                setTimeout(() => _fetchReport2(), 500);
              }
              resultMsg = {
                id:          createUniqueId("msg-"),
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
                id:        createUniqueId("msg-"),
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
    // ── Cancel any pending ghost-text activity immediately ───────────────
    if (ghostDebounceRef.current) { clearTimeout(ghostDebounceRef.current); ghostDebounceRef.current = null; }
    if (ghostAbortRef.current) { ghostAbortRef.current.abort(); ghostAbortRef.current = null; }
    setWordSuffix("");
    // ─────────────────────────────────────────────────────────────────────

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
            // Natural call-intent phrasing
            "can we call","can i call","wanna call","want to call","let's call","lets call",
            "hop on a call","jump on a call","get on a call","have a call","do a call",
            "call me","audio call","voice chat","talk to you","speak to you","chat verbally",
            "on appelle","on se call","appel vocal","appelez","parler de vive voix",
            "يمكننا الاتصال","نتصل","مكالمة صوتية","اتصال",
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
      const userId   = createUniqueId("msg-");
      const assistId = createUniqueId("msg-");
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
    // Snapshot pendingDocIds BEFORE clearing so we can send them with the query request
    const snapshotDocIds = [...pendingDocIds];
    const userMsg: Message = {
      id: createUniqueId("msg-"),
      role: "user",
      content: trimmedInput,
      timestamp: new Date(),
      attachedDocIds: snapshotDocIds.length > 0 ? snapshotDocIds : undefined,
    };
    const userRawTitle = trimmedInput.length > 50 ? trimmedInput.slice(0, 50) + "…" : trimmedInput;
    const conversationTitle = userRawTitle;
    const updatedUserMessages = [...originMessages, userMsg];
    setMessages(prev => [...prev, userMsg]);
    const titleToSet = activeConversation?.initialTitle ?? activeConversation?.title ?? conversationTitle;
    // Register the conversation in the sidebar IMMEDIATELY when user sends,
    // so the chat pill shows it right away (with the green pulsating dot while
    // the agent is still responding). We use the user message as the title.
    updateConversationMessages(convId, updatedUserMessages, trimmedInput.slice(0, 80), activeConvId ? titleToSet : conversationTitle);
    setInput("");
    setPendingDocIds([]);
    if (!notifyDismissed) setShowNotifyBanner(true);
    markPendingConversation(convId);
    setTypingConvId(convId);

    try {
      // All routes (including CAD/IFC) go through the backend intent classifier.
      const assistantMsg = await runStreamingQuery(trimmedInput, undefined, snapshotDocIds);
      if (!assistantMsg) return;
      const updatedMessages: Message[] = [...originMessages, userMsg, assistantMsg];
      const rawTitle = trimmedInput.length > 50 ? trimmedInput.slice(0, 50) + "…" : trimmedInput;
      const preview  = assistantMsg.content.replace(/\[.*?\]/g, "").replace(/#{1,3}\s/g, "").slice(0, 80) + "…";
      const finalTitle = activeConversation?.titleLocked
        ? activeConversation.title
        : (activeConversation?.initialTitle ?? activeConversation?.title ?? conversationTitle ?? rawTitle);
      // Always directly update messages state so the answer is guaranteed to render,
      // regardless of whether activeConvIdRef matches (fixes the "dots disappear, no answer" bug).
      setMessages(updatedMessages);
      updateConversationMessages(convId, updatedMessages, preview, finalTitle);
      // Persist to DB (non-blocking); keep the current conversation title if it was already set
      const currentTitle = activeConversation?.title ?? finalTitle ?? rawTitle;
      saveConversationToDB(convId, sessionIdRef.current ?? "", currentTitle, preview, updatedMessages);
      setTypingMessageId(assistantMsg.id);
      fetchSuggestions(trimmedInput, assistantMsg.content);
      fireNotification();
      if (activeConvIdRef.current !== convId) {
        // Response arrived from another conversation, description: "A response finished in another conversation.", });
      }
      // Smart conversation title:
      // • Report route → call /title with the prompt so the sidebar shows
      //   "Telecom Site Survey Field Results" instead of the raw user message.
      // • All other routes → call generateSmartTitle with the first few messages.
      if (assistantMsg.reportId) {
        try {
          const titleRes = await fetch(`${(typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000"}/title`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json", ...getAuthHeader() },
            body: JSON.stringify({ type: "report", text: trimmedInput }),
            signal: AbortSignal.timeout(8000),
          });
          if (titleRes.ok) {
            const { title: reportConvTitle } = await titleRes.json();
            if (reportConvTitle?.trim()) {
              setConversations(convs => convs.map(c =>
                c.id === convId && !c.titleLocked
                  ? { ...c, title: reportConvTitle.trim(), titleLocked: true }
                  : c
              ));
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
        id: createUniqueId("msg-"),
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
    // Tell the backend to abort the in-flight generation to avoid wasting tokens
    const sid = sessionIdRef.current || sessionId;
    if (sid) {
      const base =
        (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) ||
        "http://localhost:8000";
      fetch(`${base}/query-stream/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      }).catch(() => {/* non-critical */});
    }
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
        id: createUniqueId("msg-"),
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
        // Create a fresh AbortController for this fetch
        const ac = new AbortController();
        ghostAbortRef.current = ac;
        const base = (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000";
        const docNames = documents.map((d: { filename: string }) => d.filename);
        const suffix = await fetchAICompletion(before, messages, base, docNames, ac.signal);
        // Only apply if this request wasn't aborted
        if (!ac.signal.aborted) {
          setWordSuffix(suffix);
          ghostAbortRef.current = null;
        }
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
      const errorMsg: Message = { id: createUniqueId("msg-"), role: "assistant", content: `Sorry: ${msg}`, timestamp: new Date() };
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

  return {
    activeConversation, activeConvId, activeConvIdRef, activeReport,
    analyserRef, animFrameRef, applyAutocomplete, audioCtxRef,
    autocomplete, autocompletePos, autocompleteRef,
    blobUrlMapRef, bubbleDoc, bubbleHighlightRef, bubbleScrollRef, bubbleViewer,
    chatDragCounterRef, confirmingDeleteId, convLoading, conversations,
    convsPanelOpen, copiedMsgId, currentUser,
    deleteConversation, deletingReportId, docFileInputRef, docsPanelOpen,
    docsPanelOpenRef, documents, downloadingReportId, dragCounterRef,
    duplicateBanner, duplicateBannerTimeoutRef,
    editDraft, editingMsgId, editTextareaRef, elapsedTimerRef,
    ensureAudio, expandSuggestion, expandedTranscripts, expandingSuggestion,
    feedback, fetchSuggestions, fileInputRef, fireNotification, formatSize,
    getApiBase, getAuthHeader,
    handleDeleteReport, handleDownloadReport, handleEditCancel,
    handleEditSubmit, handleFiles, handleGenerateReport, handleInputChange,
    handleInputKeyDown, handlePatchReport,
    handleRedo, handleRestoreVersion, handleSend, handleSourceClick,
    handleStop, handleVoiceClick, historySearch, historySort,
    input, inputAreaRef, isChatDragOver, isDark, isDragOver,
    isGeneratingReport, isHoveringVoice, isLoading, isLoading_ref,
    isPatchingReport, isUploading,
    loadConversation, loadConversationsFromDB, loadDocuments, loadReports,
    logout, markPendingConversation, clearPendingConversation,
    maxDurationTimerRef, mediaRecorderRef, messages, messagesEndRef,
    messagesRef, MODEL_OPTIONS, modelDropdownOpen, modelDropdownRef,
    navigate, notifyBottomOffset, notifyDismissed, notifyEnabled,
    openBubbleDoc, openDocumentAtExcerpt, openDocumentAtNumber, openSourceKey,
    pageDragCounterRef, pendingConvIds, pendingDocIds, playBeep,
    previewedVersion, recordingElapsed, removeDocument, removePendingAttachment,
    reportEditInputRef, reportEditInstruction, reportEditMode, reports,
    reportsPanelOpen, reportsPanelOpenRef, reportsPanelRef, restoringVersion,
    revokePreviewUrl, runStreamingQuery, saveConversationToDB, selectedModel,
    selectedModelRef, serializeError, sessionId, sessionIdRef,
    setActiveConvId, setActiveReport, setAutocomplete, setAutocompletePos,
    setBubbleDoc, setBubbleViewer, setConfirmingDeleteId, setConvsPanelOpen,
    setCopiedMsgId, setDocsPanelOpen, setDuplicateBanner, setEditDraft,
    setEditingMsgId, setExpandedTranscripts, setFeedback, setHistorySearch,
    setHistorySort, setIsChatDragOver, setIsDragOver, setIsGeneratingReport,
    setIsHoveringVoice, setIsLoading, setIsPatchingReport, setIsUploading,
    setMessages, setModelDropdownOpen, setNotifyDismissed, setNotifyEnabled,
    setOpenSourceKey, setPendingDocIds, setPreviewedVersion,
    setRecordingElapsed, setReportEditInstruction, setReportEditMode,
    setReports, setReportsPanelOpen, setSelectedModel, setSessionId,
    setShowNotifyBanner, setShowSilenceWarning, setShowVersionHistory,
    setSidebarOpen, setSuggestions, setThinkingExpanded, setThinkingSteps,
    setTypingMessageId, setViewer, setVoiceState, setWaveformBars,
    setWordSuffix, showAuthModal, showNotifyBanner, showSilenceWarning,
    showVersionHistory, sidebarOpen, startNewConversation,
    suggestions, suggestionsLoading, textareaRef,
    thinkingExpanded, thinkingSteps, thinkingVisible, toast,
    typingConvId, typingConvIdRef, typingMessageId,
    updateConversationMessages, viewer, voiceState,
    waveformBars, waveformSamplesRef, wordSuffix,
  };
}
