import { useState, useRef, useEffect, useCallback } from "react";
import { Conversation, Message, createUniqueId } from "@/pages/chatTypes";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys, useConversationsQuery, useSaveConversationMutation, useDeleteConversationMutation } from "@/services/queries";

export function useSessions(options: {
  toast: (props: { title: string; description?: string; variant?: "default" | "destructive" }) => void;
  currentUser: { token?: string; user_id?: string } | null;
  loadDocuments: (sid?: string | null) => Promise<void>;
  loadReports: (sid?: string | null) => Promise<void>;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  transcribeAbortRef: React.MutableRefObject<AbortController | null>;
  mediaRecorderRef: React.MutableRefObject<MediaRecorder | null>;
  abortControllerRef: React.MutableRefObject<AbortController | null>;
  setVoiceState: React.Dispatch<React.SetStateAction<"idle" | "recording" | "transcribing">>;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setDocuments: React.Dispatch<React.SetStateAction<any[]>>;
  setReports: React.Dispatch<React.SetStateAction<any[]>>;
}) {
  const { toast, currentUser, loadDocuments, loadReports, messagesEndRef, transcribeAbortRef, mediaRecorderRef, abortControllerRef, setVoiceState, setIsLoading, messages, setMessages, setDocuments, setReports } = options;

  const queryClient = useQueryClient();
  const convsQuery = useConversationsQuery();
  const saveConvMutation = useSaveConversationMutation();
  const deleteConvMutation = useDeleteConversationMutation();

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>(() => convsQuery.data ?? []);
  const [activeConvId, setActiveConvId] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  const activeConvIdRef = useRef<string>("");
  useEffect(() => { activeConvIdRef.current = activeConvId; }, [activeConvId]);
  const activeConversation = conversations.find(c => c.id === activeConvId);
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
  const [historySearch, setHistorySearch] = useState("");
  const [historySort, setHistorySort] = useState<"newest" | "oldest">("newest");
  const [docsPanelOpen, setDocsPanelOpen] = useState(false);
  const docsPanelOpenRef = useRef(false);
  docsPanelOpenRef.current = docsPanelOpen;
  const [convsPanelOpen, setConvsPanelOpen] = useState(false);
  const convsPanelOpenRef = useRef(false);
  convsPanelOpenRef.current = convsPanelOpen;
  const [convLoading, setConvLoading] = useState(false);

  useEffect(() => {
    if (!convsQuery.data) return;
    setConversations(prev => {
      const existingById = new Map(prev.map(c => [c.id, c]));
      const refreshed = convsQuery.data.map((r: any) => {
        const existing = existingById.get(r.id);
        return {
          ...r,
          messages: existing?.messages?.length ? existing.messages : [],
        } as any;
      });
      const localOnly = prev.filter(c => !convsQuery.data.some((r: any) => r.id === c.id));
      return [...localOnly, ...refreshed];
    });
  }, [convsQuery.data]);

  const getApiBase = () =>
    ((typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000");

  const getAuthHeader = (): Record<string, string> => {
    const token = currentUser?.token ?? "";
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const updateConversationMessages = useCallback(
    (convId: string, updatedMessages: Message[], preview: string, title?: string) => {
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
        if (deletedConvIdsRef.current.has(convId)) return prev;
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
        // setMessages is external; caller handles this
      }
    },
    []
  );

  const generateSmartTitle = useCallback(async (
    convId: string,
    msgs: Message[],
    currentTitle: string,
  ) => {
    const userMsgs = msgs.filter(m => m.role === "user");
    if (userMsgs.length < 1) return;
    if (userMsgs.length > 1) return;

    try {
      const base = getApiBase();
      const res = await fetch(`${base}/title`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", ...getAuthHeader() },
        body: JSON.stringify({
          type: "chat",
          messages: msgs.slice(0, 6).map(m => ({ role: m.role, content: m.content.slice(0, 150) })),
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
          const messagesToSave = (_conv.messages?.length ?? 0) > 0 ? _conv.messages : msgs;
          saveConversationToDB(
            convId,
            (_conv as any).sessionId ?? sessionIdRef.current ?? "",
            smartTitle,
            _conv.preview,
            messagesToSave
          );
        }
        return convs.map(c =>
          c.id === convId ? { ...c, title: smartTitle, initialTitle: smartTitle, titleLocked: true } : c
        );
      });
    } catch {
    }
  }, []);

  const loadConversationsFromDB = async () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.conversations });
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
    saveConvMutation.mutate({
      conversationId: convId,
      sessionId: sid,
      title,
      preview,
      messages: msgs.map(m => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : m.timestamp,
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
    });
  };

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
    activeConvIdRef.current = newId;
    return newId;
  };

  const loadConversation = async (conv: Conversation) => {
    const currentActiveConvId = activeConvIdRef.current;
    const currentMessages = messages.length;
    if (currentMessages > 0 && conv.id === currentActiveConvId) {
      return;
    }

    const sid = (conv as any).sessionId ?? null;

    if (conv.messages && conv.messages.length > 0) {
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

    const base = getApiBase();
    const rawToken = currentUser?.token ?? "";

    setActiveConvId(conv.id);
    activeConvIdRef.current = conv.id;
    setSessionId(null);
    sessionIdRef.current = null;
    setConvLoading(true);

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
        const restored: Message[] = (data.messages ?? []).map((m: any) => {
          const merged = {
            ...m,
            ...(m.payload ?? {}),
            id:        m.id ?? createUniqueId("msg-"),
            role:      m.role as "user" | "assistant",
            content:   m.content ?? "",
            timestamp: new Date(m.timestamp ?? (m.payload?.timestamp) ?? Date.now()),
          };
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
      if (!sid) {
        abortLoad();
        return;
      }
    } catch (e) {
      console.error("loadConversation from DB failed:", e);
      abortLoad();
      return;
    }

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

    abortLoad();
  };

  const deleteConversation = async (convId: string) => {
    deletedConvIdsRef.current.add(convId);
    setConversations(prev => prev.filter(c => c.id !== convId));
    if (convId === activeConvId) {
      const sid = sessionIdRef.current || sessionId;
      if (sid) {
        const base = getApiBase();
        fetch(`${base}/query-stream/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sid }),
        }).catch(() => {});
      }
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
      deleteConvMutation.mutate(convId);
    }
  };

  return {
    sidebarOpen, setSidebarOpen,
    conversations, setConversations,
    activeConvId, setActiveConvId,
    sessionId, setSessionId,
    sessionIdRef, activeConvIdRef,
    activeConversation,
    deletedConvIdsRef,
    pendingConvIds, markPendingConversation, clearPendingConversation,
    updateConversationMessages,
    historySearch, setHistorySearch,
    historySort, setHistorySort,
    docsPanelOpen, setDocsPanelOpen,
    docsPanelOpenRef,
    convsPanelOpen, setConvsPanelOpen,
    convsPanelOpenRef,
    convLoading, setConvLoading,
    getApiBase,
    getAuthHeader,
    loadConversationsFromDB,
    saveConversationToDB,
    loadConversation,
    deleteConversation,
    startNewConversation,
    ensureActiveConversationId,
    generateSmartTitle,
  };
}
