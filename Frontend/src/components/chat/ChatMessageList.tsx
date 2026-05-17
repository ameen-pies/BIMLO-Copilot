import React from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { WelcomeSplash } from "@/components/chat/WelcomeSplash";
import { ReportCard } from "@/components/chat/ReportCard";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ThinkingTrace } from "@/components/chat/ThinkingTrace";
import { VoiceMessageBubble } from "@/components/chat/VoiceMessageBubble";
import { ChartClarification } from "@/components/chat/ChartMessage";
import { ChartMessage } from "@/components/chat/ChartMessage";
import { renderContent } from "@/components/chat/chatRenderers";
import { MarkdownRenderer } from "@/components/chat/MarkdownRenderer";
import TypewriterText from "@/components/TypewriterText";
import { Message, ThinkingStep, Conversation, ReportRecord, createUniqueId } from "@/pages/chatTypes";
import { Source } from "@/services/api";
// Icons
import { Plus, Loader2, ChevronDown, FileText, Search, ScrollText, Pencil, Check, Sparkles, Phone, Copy, Square, X, ThumbsUp, ThumbsDown, RotateCcw, ChevronRight, Bell, Trash2, ImageIcon } from "lucide-react";

interface ChatMessageListProps {
  messages: Message[];
  isChatDragOver: boolean;
  setIsChatDragOver: (v: boolean) => void;
  chatDragCounterRef: React.MutableRefObject<number>;
  handleFiles: (files: FileList | File[]) => Promise<void>;
  convLoading: boolean;
  activeConvId: string;
  openSourceKey: string | null;
  setOpenSourceKey: (v: string | null) => void;
  expandedTranscripts: Set<string>;
  setExpandedTranscripts: (v: Set<string> | ((prev: Set<string>) => Set<string>)) => void;
  copiedMsgId: string | null;
  setCopiedMsgId: (v: string | null) => void;
  typingMessageId: string | null;
  setTypingMessageId: (v: string | null) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
  notifyDismissed: boolean;
  editingMsgId: string | null;
  setEditingMsgId: (v: string | null) => void;
  editDraft: string;
  setEditDraft: (v: string) => void;
  feedback: Record<string, "like" | "dislike" | null>;
  setFeedback: (v: Record<string, "like" | "dislike" | null> | ((prev: Record<string, "like" | "dislike" | null>) => Record<string, "like" | "dislike" | null>)) => void;
  documents: any[];
  blobUrlMapRef: React.MutableRefObject<Map<string, any>>;
  openBubbleDoc: (doc: any) => void;
  thinkingVisible: boolean;
  typingConvId: string | null;
  thinkingSteps: ThinkingStep[];
  thinkingExpanded: boolean;
  setThinkingExpanded: (v: boolean) => void;
  showNotifyBanner: boolean;
  conversations: Conversation[];
  reports: ReportRecord[];
  activeReport: ReportRecord | null;
  // Refs
  activeConvIdRef: React.MutableRefObject<string>;
  messagesRef: React.MutableRefObject<Message[]>;
  sessionIdRef: React.MutableRefObject<string | null>;
  editTextareaRef: React.RefObject<HTMLTextAreaElement>;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  // Callbacks
  handleSourceClick: (sourceNum: number, msgId: string) => void;
  runStreamingQuery: (query: string, forceRoute?: string, pendingDocIdsToCommit?: string[]) => Promise<Message | null>;
  updateConversationMessages: (convId: string, messages: Message[], preview: string, title?: string) => void;
  saveConversationToDB: (convId: string, sid: string, title: string, preview: string, msgs: Message[]) => Promise<void>;
  fetchSuggestions: (userQuery: string, assistantReply: string) => Promise<void>;
  fireNotification: () => void;
  navigate: (path: string) => void;
  handleEditSubmit: (msgId: string) => Promise<void>;
  handleEditCancel: () => void;
  handleRedo: (msgId: string) => Promise<void>;
  openDocumentAtExcerpt: (filename: string, excerpt: string) => void;
  setMessages: (v: Message[] | ((prev: Message[]) => Message[])) => void;
  setThinkingSteps: (v: ThinkingStep[]) => void;
  setSuggestions: (v: string[]) => void;
  setShowNotifyBanner: (v: boolean) => void;
  serializeError: (err: unknown) => string;
}

const ChatMessageList: React.FC<ChatMessageListProps> = ({
  messages,
  isChatDragOver,
  setIsChatDragOver,
  chatDragCounterRef,
  handleFiles,
  convLoading,
  activeConvId,
  openSourceKey,
  setOpenSourceKey,
  expandedTranscripts,
  setExpandedTranscripts,
  copiedMsgId,
  setCopiedMsgId,
  typingMessageId,
  setTypingMessageId,
  isLoading,
  setIsLoading,
  notifyDismissed,
  editingMsgId,
  setEditingMsgId,
  editDraft,
  setEditDraft,
  feedback,
  setFeedback,
  documents,
  blobUrlMapRef,
  openBubbleDoc,
  thinkingVisible,
  typingConvId,
  thinkingSteps,
  thinkingExpanded,
  setThinkingExpanded,
  showNotifyBanner,
  conversations,
  reports,
  activeReport,
  activeConvIdRef,
  messagesRef,
  sessionIdRef,
  editTextareaRef,
  messagesEndRef,
  handleSourceClick,
  runStreamingQuery,
  updateConversationMessages,
  saveConversationToDB,
  fetchSuggestions,
  fireNotification,
  navigate,
  handleEditSubmit,
  handleEditCancel,
  handleRedo,
  openDocumentAtExcerpt,
  setMessages,
  setThinkingSteps,
  setSuggestions,
  setShowNotifyBanner,
  serializeError,
}) => {
  const { t } = useTranslation();
  return (
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
                        <span className="text-[13px] font-medium text-emerald-300 leading-none">{t("chat.voice_call")}</span>
                        <span className="text-[11px] text-emerald-400/60 tabular-nums">
                          {Math.floor(msg.callCard.duration / 60)}:{String(msg.callCard.duration % 60).padStart(2, "0")}
                          {" · "}
                          {msg.callCard.startedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                <>
                {/* Attached docs — shown above the bubble for user/voice messages */}
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
                {/* Voice bubble sits below attached docs, text bubble for all other messages */}
                {(msg.voiceBlobUrl || msg.voiceTranscript) ? (
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
                ) : <div
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
                    title={t("chat.copy_response")}
                  >
                    {copiedMsgId === msg.id
                      ? <><Check className="h-3 w-3" /><span>{t("chat.copied")}</span></>
                      : <><Copy className="h-3 w-3" /><span>{t("chat.copy")}</span></>}
                  </button>
                )}

                  {msg.role === "assistant" && msg.id === typingMessageId && !(msg as any).isStreaming && !msg.content && !(msg.analytics?.type === "chart_config" || msg.analytics?.type === "chart_error" || msg.analytics?.type === "chart_clarification" || msg.analytics?.type === "report_chart_clarification") ? (
                    /* ── Empty placeholder: show loading dots inline ── */
                    <span className="inline-flex items-center gap-1 text-muted-foreground/60 text-sm py-2">
                      <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                      <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                      <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
                    </span>
                  ) : msg.role === "assistant" && msg.id === typingMessageId && !(msg.analytics?.type === "chart_config" || msg.analytics?.type === "chart_error" || msg.analytics?.type === "chart_clarification" || msg.analytics?.type === "report_chart_clarification") ? (
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
                      {t("chat.response_stopped")}
                    </span>
                  ) : msg.role === "assistant" && msg.analytics?.type === "chart_clarification" ? (
                    <ChartClarification
                      analytics={msg.analytics as any}
                      onSelect={async (hint) => {
                        if (isLoading) return;
                        // Show loading indicator immediately — before the async query starts
                        setIsLoading(true);
                        setThinkingSteps([]);
                        setThinkingExpanded(true);
                        const convId = activeConvIdRef.current;
                        const userMsg: Message = {
                          id: createUniqueId("msg-"),
                          role: "user",
                          content: hint,
                          timestamp: new Date(),
                        };
                        setMessages(prev => [...prev, userMsg]);
                        setOpenSourceKey(null);
                        setSuggestions([]);
                        if (!notifyDismissed) setShowNotifyBanner(true);
                        // Add placeholder assistant message with loading indicator
                        const placeholderId = createUniqueId("msg-");
                        const placeholder: Message = {
                          id: placeholderId,
                          role: "assistant",
                          content: "",
                          timestamp: new Date(),
                          isStreaming: true,
                        };
                        setMessages(prev => [...prev, placeholder]);
                        setTypingMessageId(placeholderId);
                        try {
                          const assistantMsg = await runStreamingQuery(hint, "graph");
                          if (!assistantMsg) {
                            setMessages(prev => prev.filter(m => m.id !== placeholderId));
                            setTypingMessageId(null);
                            setIsLoading(false);
                            return;
                          }
                          // Clear typing indicator BEFORE replacing message to prevent empty shell flash
                          setTypingMessageId(null);
                          // Replace placeholder with actual response (strip isStreaming so it renders normally)
                          // Also capture the final messages array directly from the updater so we
                          // pass the CORRECT snapshot to persistence — messagesRef.current is stale
                          // at this point because useEffect hasn't fired yet.
                          let finalMsgs: Message[] = [];
                          setMessages(prev => {
                            finalMsgs = prev.map(m => m.id === placeholderId ? { ...assistantMsg, isStreaming: undefined } : m);
                            return finalMsgs;
                          });
                          // Persist user + assistant messages so they survive a refresh
                          const preview = assistantMsg.content.replace(/\[.*?\]/g, "").slice(0, 80) + "…";
                          const currentTitle = conversations.find(c => c.id === convId)?.title ?? hint.slice(0, 50);
                          updateConversationMessages(convId, finalMsgs, preview, currentTitle);
                          saveConversationToDB(convId, sessionIdRef.current ?? "", currentTitle, preview, finalMsgs);
                          fetchSuggestions(hint, assistantMsg.content);
                          fireNotification();
                        } catch (error) {
                          setIsLoading(false);
                          const errMsg: Message = {
                            id: createUniqueId("msg-"),
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
                        // Show loading indicator immediately — before the async query starts
                        setIsLoading(true);
                        setThinkingSteps([]);
                        setThinkingExpanded(true);
                        const convId = activeConvIdRef.current;
                        const userMsg: Message = {
                          id: createUniqueId("msg-"),
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
                          // Capture final messages from updater — messagesRef.current is stale
                          let finalMsgs: Message[] = [];
                          setMessages(prev => {
                            finalMsgs = [...prev, assistantMsg];
                            return finalMsgs;
                          });
                          setTypingMessageId(assistantMsg.id);
                          // Persist user + assistant messages so they survive a refresh
                          const preview = assistantMsg.content.replace(/\[.*?\]/g, "").slice(0, 80) + "…";
                          const currentTitle = conversations.find(c => c.id === convId)?.title ?? hint.slice(0, 50);
                          updateConversationMessages(convId, finalMsgs, preview, currentTitle);
                          saveConversationToDB(convId, sessionIdRef.current ?? "", currentTitle, preview, finalMsgs);
                          fetchSuggestions(hint, assistantMsg.content);
                          fireNotification();
                        } catch (error) {
                          setIsLoading(false);
                          const errMsg: Message = {
                            id: createUniqueId("msg-"),
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
                      {renderContent(msg.rawAnswer ?? msg.content, msg.id, msg.sources, handleSourceClick)}
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
                              const replyId = createUniqueId("msg-");
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
                </div>}
                </> )}

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
                        title={t("chat.copy_message")}
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
                        title={t("chat.edit_message")}
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
                        title={t("chat.good_response")}
                      >
                        <ThumbsUp className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => setFeedback(prev => ({ ...prev, [msg.id]: prev[msg.id] === "dislike" ? null : "dislike" }))}
                        className={`p-1 rounded-md transition-colors ${feedback[msg.id] === "dislike" ? "text-destructive" : "text-muted-foreground/40 hover:text-destructive"}`}
                        title={t("chat.bad_response")}
                      >
                        <ThumbsDown className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => handleRedo(msg.id)}
                        disabled={isLoading}
                        className="p-1 rounded-md text-muted-foreground/40 hover:text-foreground transition-colors disabled:opacity-30"
                        title={t("chat.regenerate_response")}
                      >
                        <RotateCcw className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
                )}

                <SourcesPanel msg={msg} onOpenDocument={openDocumentAtExcerpt} />

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

          {thinkingVisible && typingConvId === activeConvId && (
            <ThinkingTrace
              steps={thinkingSteps}
              isExpanded={thinkingExpanded}
              onToggle={() => setThinkingExpanded(v => !v)}
              showDots
              isLoading={isLoading}
            />
          )}

          <div ref={messagesEndRef} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
  );
};

export { ChatMessageList };
