import { useState, useRef, useEffect, useCallback } from "react";
import {
  ThinkingStep,
  Message,
  Conversation,
  createUniqueId,
  ModelProvider,
} from "@/pages/chatTypes";
import {
  serializeError,
  TrieNode,
  makeTrie,
  trieInsert,
  tokenise,
} from "@/utils/chatUtils";
import { fetchAICompletion } from "@/components/chat/queryHelpers";
import api from "@/services/api";

export interface StreamResult {
  msg: Message | null;
  reportId?: string;
}

export function useChatStream(options: {
  toast: (props: { title: string; description?: string; variant?: "default" | "destructive" }) => void;
  currentUser: { token?: string; user_id?: string } | null;
  showAuthModal: (callback?: () => void) => void;
  sessionIdRef: React.MutableRefObject<string | null>;
  sessionId: string | null;
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>;
  activeConvIdRef: React.MutableRefObject<string>;
  activeConvId: string;
  setActiveConvId: React.Dispatch<React.SetStateAction<string>>;
  activeConversation: Conversation | undefined;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  documents: any[];
  conversations: Conversation[];
  setConversations: React.Dispatch<React.SetStateAction<Conversation[]>>;
  ensureActiveConversationId: () => string;
  updateConversationMessages: (convId: string, updatedMessages: Message[], preview: string, title?: string) => void;
  saveConversationToDB: (convId: string, sid: string, title: string, preview: string, msgs: Message[]) => Promise<void>;
  generateSmartTitle: (convId: string, messages: Message[], currentTitle: string) => Promise<void>;
  fetchSuggestions: (userQuery: string, assistantReply: string, setterSuggestions: React.Dispatch<React.SetStateAction<string[]>>, setterLoading: React.Dispatch<React.SetStateAction<boolean>>) => Promise<void>;
  expandSuggestion: (label: string, isGeneral: boolean, setExpanding: React.Dispatch<React.SetStateAction<string | null>>) => Promise<string>;
  getAuthHeader: () => Record<string, string>;
  getApiBase: () => string;
  loadDocuments: (sid?: string | null) => Promise<void>;
  setReports: React.Dispatch<React.SetStateAction<any[]>>;
  setActiveReport: React.Dispatch<React.SetStateAction<any>>;
  reportsPanelOpenRef: React.MutableRefObject<boolean>;
  setReportsPanelOpen: React.Dispatch<React.SetStateAction<boolean>>;
  markPendingConversation: (convId: string) => void;
  clearPendingConversation: (convId: string) => void;
  notifyDismissed: boolean;
  setNotifyDismissed: React.Dispatch<React.SetStateAction<boolean>>;
  setShowNotifyBanner: React.Dispatch<React.SetStateAction<boolean>>;
  showNotifyBanner: boolean;
  setNotifyEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  notifyEnabled: boolean;
  fireNotification: () => void;
  playBeep: () => void;
  ensureAudio: () => HTMLAudioElement;
  pendingDocIds: string[];
  setPendingDocIds: React.Dispatch<React.SetStateAction<string[]>>;
}) {
  const {
    toast, currentUser, showAuthModal, sessionIdRef, sessionId, setSessionId,
    activeConvIdRef, activeConvId, setActiveConvId, activeConversation,
    messages, setMessages, documents, conversations, setConversations,
    ensureActiveConversationId, updateConversationMessages, saveConversationToDB,
    generateSmartTitle, fetchSuggestions, expandSuggestion, getAuthHeader, getApiBase,
    loadDocuments, setReports, setActiveReport, reportsPanelOpenRef, setReportsPanelOpen,
    markPendingConversation, clearPendingConversation,
    notifyDismissed, setNotifyDismissed, setShowNotifyBanner, showNotifyBanner,
    setNotifyEnabled, notifyEnabled, fireNotification, playBeep, ensureAudio,
    pendingDocIds, setPendingDocIds,
  } = options;

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const isLoading_ref = useRef(false);
  const [thinkingVisible, setThinkingVisible] = useState(false);
  const thinkingLingerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [thinkingExpanded, setThinkingExpanded] = useState(true);
  const [convLoading, setConvLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [feedback, setFeedback] = useState<Record<string, "like" | "dislike" | null>>({});
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [typingConvId, setTypingConvId] = useState<string | null>(null);
  const typingConvIdRef = useRef<string | null>(null);
  useEffect(() => { typingConvIdRef.current = typingConvId; }, [typingConvId]);
  const [openSourceKey, setOpenSourceKey] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<Message[]>([]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  const inputAreaRef = useRef<HTMLDivElement>(null);

  type ModelProvider_type = "cf_primary" | "cf_backup" | "groq" | "nvidia";
  const [selectedModel, setSelectedModel] = useState<ModelProvider_type>("cf_primary");
  const selectedModelRef = useRef<ModelProvider_type>("cf_primary");
  useEffect(() => { selectedModelRef.current = selectedModel; }, [selectedModel]);

  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const MODEL_OPTIONS: { id: ModelProvider_type; label: string; shortLabel: string; desc: string; color: string }[] = [
    { id: "cf_primary",  label: "Cloudflare Primary",  shortLabel: "CF Primary",  desc: "Primary Cloudflare Workers AI",    color: "#f97316" },
    { id: "cf_backup",   label: "Cloudflare Backup",   shortLabel: "CF Backup",   desc: "Backup Cloudflare Workers AI",     color: "#3b82f6" },
    { id: "groq",        label: "Groq",                shortLabel: "Groq",        desc: "Groq (llama-3.3-70b-versatile)",   color: "#10b981" },
    { id: "nvidia",      label: "MiniMax M2.7",         shortLabel: "MiniMax",     desc: "MiniMax M2.7 via NVIDIA NIM",      color: "#76b900" },
  ];

  useEffect(() => {
    if (isLoading) {
      if (thinkingLingerRef.current) clearTimeout(thinkingLingerRef.current);
      setThinkingVisible(true);
    } else {
      thinkingLingerRef.current = setTimeout(() => setThinkingVisible(false), 350);
    }
    return () => { if (thinkingLingerRef.current) clearTimeout(thinkingLingerRef.current); };
  }, [isLoading]);

  useEffect(() => {
    const ta = editTextareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 300)}px`;
  }, [editDraft]);

  useEffect(() => {
    if (editingMsgId && editTextareaRef.current) {
      const ta = editTextareaRef.current;
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
    }
  }, [editingMsgId]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [expandingSuggestion, setExpandingSuggestion] = useState<string | null>(null);

  useEffect(() => {
    const el = inputAreaRef.current;
    if (!el) return;
    const update = () => setNotifyBottomOffset(el.offsetHeight + 8);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [suggestions, suggestionsLoading]);

  useEffect(() => {
    if (messages.length <= 1) return;
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  type VoiceState = "idle" | "recording" | "transcribing";
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [showSilenceWarning, setShowSilenceWarning] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const transcribeAbortRef = useRef<AbortController | null>(null);
  const recordingStartRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>(0);
  const maxDurationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [waveformBars, setWaveformBars] = useState<number[]>(Array(48).fill(0));
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const waveformSamplesRef = useRef<number[]>([]);
  const [expandedTranscripts, setExpandedTranscripts] = useState<Set<string>>(new Set());
  const [isHoveringVoice, setIsHoveringVoice] = useState(false);
  const [notifyBottomOffset, setNotifyBottomOffset] = useState(125);

  const [autocomplete, setAutocomplete] = useState<{
    query: string;
    triggerPos: number;
    results: any[];
    activeIdx: number;
  } | null>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);
  const [autocompletePos, setAutocompletePos] = useState<{ top: number; left: number; width: number } | null>(null);

  const trieRef = useRef<TrieNode>(makeTrie());
  const [wordSuffix, setWordSuffix] = useState<string>("");
  const ghostDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ghostAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const root = makeTrie();
    for (const msg of messages) {
      const weight = msg.role === "user" ? 3 : 1;
      for (const word of tokenise(msg.content)) trieInsert(root, word, weight);
    }
    for (const doc of documents) {
      for (const word of tokenise(doc.filename)) trieInsert(root, word, 5);
      const slug = doc.filename.replace(/\.[^.]+$/, "").toLowerCase().replace(/[^a-z0-9]/g, "");
      if (slug.length >= 3) trieInsert(root, slug, 5);
    }
    trieRef.current = root;
  }, [messages, documents]);

  const inputOverrideRef = useRef<string | null>(null);

  const runStreamingQuery = useCallback(async (query: string, forceRoute?: string, pendingDocIdsToCommit: string[] = []): Promise<Message | null> => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);
    if (activeConvIdRef.current) setTypingConvId(activeConvIdRef.current);

    try {
      const base = getApiBase();
      abortControllerRef.current = new AbortController();
      const rawToken = currentUser?.token ?? "";
      const streamHeaders: Record<string, string> = { "Content-Type": "application/json" };
      if (rawToken) streamHeaders["Authorization"] = `Bearer ${rawToken}`;

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
          ...(activeConvIdRef.current ? { conversation_id: activeConvIdRef.current } : {}),
          pending_doc_ids: pendingDocIdsToCommit,
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
              if (reportId) {
                const _fetchReport = (attempt = 0) => {
                  fetch(`${base}/reports/${reportId}`)
                    .then(r => r.ok ? r.json() : Promise.reject(r.status))
                    .then(report => {
                      setReports(prev => [report, ...prev.filter((r: any) => r.report_id !== reportId)]);
                      setActiveReport(report);
                      reportsPanelOpenRef.current = true;
                      setReportsPanelOpen(true);
                    })
                    .catch(() => { if (attempt < 5) setTimeout(() => _fetchReport(attempt + 1), 3000 * (attempt + 1)); });
                };
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
          } catch { }
        }
      }

      if (resultMsg) {
        resultMsg = { ...resultMsg, thinkingSteps: accumulatedSteps };
      } else {
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

  const runReportStream = useCallback(async (
    query: string,
    explicitDocs: string[],
  ): Promise<Message | null> => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);

    try {
      const base = getApiBase();
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
                      setReports(prev => [report, ...prev.filter((r: any) => r.report_id !== reportId)]);
                      setActiveReport(report);
                      reportsPanelOpenRef.current = true;
                      setReportsPanelOpen(true);
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
          } catch { }
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

  const runCadQuery = useCallback(async (query: string, fileId: string): Promise<Message | null> => {
    setThinkingSteps([]);
    setThinkingExpanded(true);
    setIsLoading(true);

    try {
      const base = getApiBase();
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
          } catch { }
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

  const handleSendWithText = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;
    inputOverrideRef.current = trimmed;
    setInput(trimmed);
    setTimeout(() => handleSendDirectRef.current?.(), 0);
  }, [isLoading]);

  const handleSendDirectRef = useRef<(() => Promise<void>) | null>(null);

  const handleSend = async () => {
    if (ghostDebounceRef.current) { clearTimeout(ghostDebounceRef.current); ghostDebounceRef.current = null; }
    if (ghostAbortRef.current) { ghostAbortRef.current.abort(); ghostAbortRef.current = null; }
    setWordSuffix("");

    const rawInput = inputOverrideRef.current ?? input;
    inputOverrideRef.current = null;
    const trimmedInput = rawInput.trim();
    if (!trimmedInput || isLoading) return;

    if (!currentUser) {
      showAuthModal(() => handleSend());
      return;
    }

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
      setTypingMessageId(assistId);
      return;
    }

    ensureAudio();
    setOpenSourceKey(null);
    setSuggestions([]);
    setShowSilenceWarning(false);

    const convId = ensureActiveConversationId();
    const originMessages = messages;
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
      setMessages(updatedMessages);
      updateConversationMessages(convId, updatedMessages, preview, finalTitle);
      const currentTitle = activeConversation?.title ?? finalTitle ?? rawTitle;
      saveConversationToDB(convId, sessionIdRef.current ?? "", currentTitle, preview, updatedMessages);
      setTypingMessageId(assistantMsg.id);
      fetchSuggestions(trimmedInput, assistantMsg.content, setSuggestions, setSuggestionsLoading);
      fireNotification();
      if (assistantMsg.reportId) {
        try {
          const base = getApiBase();
          const titleRes = await fetch(`${base}/title`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json", ...getAuthHeader() },
            body: JSON.stringify({ type: "report", text: trimmedInput }),
            signal: AbortSignal.timeout(8000),
          });
          if (titleRes.ok) {
            const { title: reportConvTitle } = await titleRes.json();
            if (reportConvTitle?.trim()) {
              setConversations((convs: any) => convs.map((c: any) =>
                c.id === convId && !c.titleLocked
                  ? { ...c, title: reportConvTitle.trim(), titleLocked: true }
                  : c
              ));
            }
          }
        } catch { }
      } else {
        const curTitle = conversations.find((c: any) => c.id === convId)?.title ?? rawTitle;
        generateSmartTitle(convId, updatedMessages, curTitle);
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

  handleSendDirectRef.current = handleSend;

  const handleStop = () => {
    const sid = sessionIdRef.current || sessionId;
    if (sid) {
      const base = getApiBase();
      fetch(`${base}/query-stream/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sid }),
      }).catch(() => {});
    }
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
    setTypingMessageId(null);
    setTypingConvId(null);
    setThinkingSteps([]);
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

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setInput(val);
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
        const results = documents.filter((d: any) =>
          d.filename.toLowerCase().includes(fragment)
        ).slice(0, 6);
        const rect = e.target.getBoundingClientRect();
        setAutocompletePos({
          top: rect.top,
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

    const trimmed = before.trim();
    if (trimmed.length >= 4 && !isLoading) {
      ghostDebounceRef.current = setTimeout(async () => {
        const ac = new AbortController();
        ghostAbortRef.current = ac;
        const base = getApiBase();
        const docNames = documents.map((d: any) => d.filename);
        const suffix = await fetchAICompletion(before, messages, base, docNames, ac.signal);
        if (!ac.signal.aborted) {
          setWordSuffix(suffix);
          ghostAbortRef.current = null;
        }
      }, 280);
    }
  };

  const applyAutocomplete = (doc: any) => {
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
      const existingConv = conversations.find((c: any) => c.id === originConvId);
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
      const existingConv = conversations.find((c: any) => c.id === originConvId);
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
    setTimeout(() => {
      const sourceCard = document.getElementById(`source-${msgId}-${sourceNum}`);
      sourceCard?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      if (sourceCard) {
        sourceCard.style.transition = "box-shadow 0.25s ease";
        sourceCard.style.boxShadow = "0 0 0 2px hsl(var(--primary) / 0.7)";
        setTimeout(() => { sourceCard.style.boxShadow = ""; }, 1800);
      }
    }, 150);

    const msg = messages.find(m => m.id === msgId);
    const source = msg?.sources?.find(s => s.source_number === sourceNum);
  };

  const handleNumberClick = (sourceNum: number, msgId: string, numValue: string, bubbleViewerRef?: React.MutableRefObject<any>, openDocumentAtNumber?: (filename: string, numValue: string, fallbackExcerpt: string) => void) => {
    setOpenSourceKey(`${msgId}-${sourceNum}`);

    const msg = messages.find(m => m.id === msgId);
    const source = msg?.sources?.find(s => s.source_number === sourceNum);

    let highlightSentence: string | null = null;

    const findLineWithNumber = (text: string, numVal: string): string | null => {
      const norm = (s: string) => s.replace(/[\s,]/g, '').toLowerCase();
      const needle = norm(numVal);
      const lines = text.split(/\n/).map(l => l.trim()).filter(l => l.length > 0);
      return lines.find(l => norm(l).includes(needle)) ?? null;
    };

    if (bubbleViewerRef?.current && bubbleViewerRef.current.doc.filename === source?.filename && bubbleViewerRef.current.content) {
      highlightSentence = findLineWithNumber(bubbleViewerRef.current.content, numValue);
    }
    if (!highlightSentence && source?.excerpt) {
      highlightSentence = findLineWithNumber(source.excerpt, numValue);
    }
    if (!highlightSentence) {
      highlightSentence = source?.excerpt ?? null;
    }

    setTimeout(() => {
      const sourceCard = document.getElementById(`source-${msgId}-${sourceNum}`);
      if (!sourceCard) return;
      sourceCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
      sourceCard.style.transition = "box-shadow 0.2s ease";
      sourceCard.style.boxShadow = "0 0 0 2px hsl(var(--primary) / 0.6)";
      setTimeout(() => { sourceCard.style.boxShadow = ""; }, 1800);
    }, 150);

    if (source && openDocumentAtNumber) {
      openDocumentAtNumber(source.filename, numValue, highlightSentence ?? source.excerpt ?? "");
    }
  };

  const handleVoiceClick = useCallback(async () => {
    if (voiceState === "recording") {
      mediaRecorderRef.current?.stop();
      return;
    }

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
        stream.getTracks().forEach((t) => t.stop());

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

        const blobUrl = URL.createObjectURL(blob);
        const capturedSamples = [...waveformSamplesRef.current];
        waveformSamplesRef.current = [];

        const snapshotDocIds = [...pendingDocIds];
        const voiceMsgId = createUniqueId("msg-");
        const voiceMsg: Message = {
          id: voiceMsgId,
          role: "user",
          content: "",
          voiceBlobUrl: blobUrl,
          voiceDuration: duration,
          voiceTranscript: undefined,
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
          const base = getApiBase();
          const res = await fetch(`${base}/transcribe`, {
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
            const updatedVoiceMsg: Message = { ...voiceMsg, content: transcript, voiceTranscript: transcript };
            const currentMsgs = messagesRef.current;
            const voiceMessages = currentMsgs.some(m => m.id === voiceMsgId)
              ? currentMsgs.map(m => m.id === voiceMsgId ? updatedVoiceMsg : m)
              : [...currentMsgs, updatedVoiceMsg];
            const voiceQuery = `${transcript}\n\n[voice_input=true]`;
            const convId = ensureActiveConversationId();
            const transcriptTitle = transcript.length > 50 ? transcript.slice(0, 50) + "…" : transcript;
            const title = activeConversation?.title ?? transcriptTitle;
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
              saveConversationToDB(convId, sessionIdRef.current ?? "", title, preview, updated);
              setTypingMessageId(assistantMsg.id);
              fetchSuggestions(transcript, assistantMsg.content, setSuggestions, setSuggestionsLoading);
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
            setMessages(prev => prev.map(m =>
              m.id === voiceMsgId
                ? { ...m, voiceTranscript: "(no speech detected)" }
                : m
            ));
            toast({ title: "Nothing detected", description: "No speech was recognised — please try again.", variant: "destructive" });
          }
        } catch (err) {
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
          const bars = Array.from({ length: BAR_COUNT }, (_, i) => {
            const binIdx = Math.floor((i / BAR_COUNT) * (freqData.length * 0.6));
            const raw = freqData[binIdx] / 255;
            const centreFactor = 1 - Math.abs((i / (BAR_COUNT - 1)) - 0.5) * 0.35;
            return Math.min(1, raw * centreFactor * 1.4);
          });
          setWaveformBars(bars);

          if (timestamp - lastSampleTime >= 50) {
            lastSampleTime = timestamp;
            const avg = bars.reduce((s, v) => s + v, 0) / bars.length;
            waveformSamplesRef.current.push(avg);

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
      }

      elapsedTimerRef.current = setInterval(() => {
        setRecordingElapsed(s => s + 1);
      }, 1000);

      maxDurationTimerRef.current = setTimeout(() => {
        mediaRecorderRef.current?.stop();
      }, 60_000);

    } catch (err) {
      console.error("Microphone error:", err);
      toast({ title: "Microphone access denied", description: "Allow microphone access in your browser settings.", variant: "destructive" });
      setVoiceState("idle");
    }
  }, [voiceState, toast, pendingDocIds, notifyDismissed]);

  return {
    messages, setMessages,
    messagesRef,
    input, setInput,
    isLoading, setIsLoading,
    convLoading, setConvLoading,
    thinkingSteps, setThinkingSteps,
    thinkingExpanded, setThinkingExpanded,
    thinkingVisible,
    thinkingLingerRef,
    abortControllerRef,
    feedback, setFeedback,
    editingMsgId, setEditingMsgId,
    editDraft, setEditDraft,
    copiedMsgId, setCopiedMsgId,
    editTextareaRef,
    typingMessageId, setTypingMessageId,
    typingConvId, setTypingConvId,
    typingConvIdRef,
    openSourceKey, setOpenSourceKey,
    textareaRef,
    inputAreaRef,
    messagesEndRef,
    selectedModel, setSelectedModel,
    selectedModelRef,
    MODEL_OPTIONS,
    modelDropdownOpen, setModelDropdownOpen,
    modelDropdownRef,
    notifyBottomOffset, setNotifyBottomOffset,
    runStreamingQuery,
    runReportStream,
    runCadQuery,
    handleSend,
    handleSendWithText,
    handleSendDirectRef,
    inputOverrideRef,
    handleStop,
    handleRedo,
    handleEditSubmit,
    handleEditCancel,
    handleInputChange,
    handleInputKeyDown,
    applyAutocomplete,
    autocomplete, setAutocomplete,
    autocompleteRef,
    autocompletePos, setAutocompletePos,
    wordSuffix, setWordSuffix,
    trieRef,
    ghostDebounceRef, ghostAbortRef,
    voiceState, setVoiceState,
    showSilenceWarning, setShowSilenceWarning,
    mediaRecorderRef,
    audioChunksRef,
    transcribeAbortRef,
    recordingStartRef,
    audioCtxRef, analyserRef, animFrameRef,
    maxDurationTimerRef, elapsedTimerRef,
    waveformSamplesRef,
    waveformBars, setWaveformBars,
    recordingElapsed, setRecordingElapsed,
    expandedTranscripts, setExpandedTranscripts,
    isHoveringVoice, setIsHoveringVoice,
    handleVoiceClick,
    handleSourceClick,
    handleNumberClick,
    suggestions, setSuggestions,
    suggestionsLoading, setSuggestionsLoading,
    expandingSuggestion, setExpandingSuggestion,
  };
}
