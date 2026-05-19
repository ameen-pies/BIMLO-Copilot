import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, FileText, X, Plus, Loader2, ChevronDown, Square, Trash2, Sparkles, Bell, BellOff, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import BorderGlow from "@/components/BorderGlow";
import { useTranslation } from "react-i18next";

interface ChatInputProps {
  inputAreaRef: React.RefObject<HTMLDivElement>;
  messagesLength: number;
  isChatDragOver: boolean;
  setIsChatDragOver: React.Dispatch<React.SetStateAction<boolean>>;
  chatDragCounterRef: React.MutableRefObject<number>;
  handleFiles: (files: FileList | File[]) => Promise<void>;
  suggestions: string[];
  suggestionsLoading: boolean;
  expandingSuggestion: string | null;
  expandSuggestion: (label: string, isGeneral: boolean) => Promise<void>;
  isLoading: boolean;
  pendingDocIds: string[];
  isUploading: boolean;
  documents: any[];
  blobUrlMapRef: React.MutableRefObject<Map<string, any>>;
  openBubbleDoc: (doc: any) => void;
  confirmingDeleteId: string | null;
  setConfirmingDeleteId: React.Dispatch<React.SetStateAction<string | null>>;
  removePendingAttachment: (documentId: string) => void;
  voiceState: string;
  setVoiceState: React.Dispatch<React.SetStateAction<string>>;
  maxDurationTimerRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>;
  elapsedTimerRef: React.MutableRefObject<ReturnType<typeof setInterval> | null>;
  animFrameRef: React.MutableRefObject<number>;
  analyserRef: React.MutableRefObject<AnalyserNode | null>;
  audioCtxRef: React.MutableRefObject<AudioContext | null>;
  waveformSamplesRef: React.MutableRefObject<number[]>;
  waveformBars: number[];
  setWaveformBars: React.Dispatch<React.SetStateAction<number[]>>;
  recordingElapsed: number;
  setRecordingElapsed: React.Dispatch<React.SetStateAction<number>>;
  mediaRecorderRef: React.MutableRefObject<MediaRecorder | null>;
  handleVoiceClick: () => void;
  isHoveringVoice: boolean;
  setIsHoveringVoice: React.Dispatch<React.SetStateAction<boolean>>;
  docFileInputRef: React.RefObject<HTMLInputElement>;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleInputKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  input: string;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
  wordSuffix: string;
  setWordSuffix: React.Dispatch<React.SetStateAction<string>>;
  autocomplete: any;
  setAutocomplete: React.Dispatch<React.SetStateAction<any>>;
  autocompletePos: any;
  setAutocompletePos: React.Dispatch<React.SetStateAction<any>>;
  modelDropdownRef: React.RefObject<HTMLDivElement>;
  MODEL_OPTIONS: any[];
  selectedModel: string;
  setSelectedModel: React.Dispatch<React.SetStateAction<string>>;
  modelDropdownOpen: boolean;
  setModelDropdownOpen: React.Dispatch<React.SetStateAction<boolean>>;
  notifyEnabled: boolean;
  setNotifyEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  notifyDismissed: boolean;
  handleSend: () => void;
  handleStop: () => void;
  autocompleteRef: React.RefObject<HTMLDivElement>;
  applyAutocomplete: (doc: any) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  inputAreaRef,
  messagesLength,
  isChatDragOver,
  setIsChatDragOver,
  chatDragCounterRef,
  handleFiles,
  suggestions,
  suggestionsLoading,
  expandingSuggestion,
  expandSuggestion,
  isLoading,
  pendingDocIds,
  isUploading,
  documents,
  blobUrlMapRef,
  openBubbleDoc,
  confirmingDeleteId,
  setConfirmingDeleteId,
  removePendingAttachment,
  voiceState,
  setVoiceState,
  maxDurationTimerRef,
  elapsedTimerRef,
  animFrameRef,
  analyserRef,
  audioCtxRef,
  waveformSamplesRef,
  waveformBars,
  setWaveformBars,
  recordingElapsed,
  setRecordingElapsed,
  mediaRecorderRef,
  handleVoiceClick,
  isHoveringVoice,
  setIsHoveringVoice,
  docFileInputRef,
  handleInputChange,
  handleInputKeyDown,
  input,
  textareaRef,
  wordSuffix,
  setWordSuffix,
  autocomplete,
  setAutocomplete,
  autocompletePos,
  setAutocompletePos,
  modelDropdownRef,
  MODEL_OPTIONS,
  selectedModel,
  setSelectedModel,
  modelDropdownOpen,
  setModelDropdownOpen,
  notifyEnabled,
  setNotifyEnabled,
  notifyDismissed,
  handleSend,
  handleStop,
  autocompleteRef,
  applyAutocomplete,
}) => {
  const { t } = useTranslation();
  return (
    <div
      ref={inputAreaRef}
      className={`transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] ${messagesLength > 0 ? "relative z-[60] border-t border-border pt-3 pb-4 px-4 shadow-[0_-4px_24px_0_rgba(0,0,0,0.06)] bg-background/80 backdrop-blur-sm overflow-visible" : "absolute left-0 right-0 px-4 pt-3 pb-4 z-[60] bg-transparent overflow-visible"}`}
      style={messagesLength === 0 ? { top: "50%", transform: "translateY(calc(-50% + 60px))" } : {}}
    >
      <div className="max-w-3xl mx-auto">

        {/* ── Contextual suggestion chips ── */}
        <AnimatePresence>
          {(suggestions.length > 0 || suggestionsLoading || expandingSuggestion !== null) && !isLoading && messagesLength > 0 && (
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
                  <div key={doc.document_id} className="relative">
                    <button
                      type="button"
                      onClick={() => openBubbleDoc(doc)}
                      className="group flex items-center gap-1.5 ps-1 pe-2.5 py-1 rounded-lg bg-card border border-border hover:border-primary/40 hover:bg-primary/5 transition-all text-start shadow-sm"
                      title={doc.filename}
                    >
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
                    <button
                      type="button"
                      onClick={e => {
                        e.stopPropagation();
                        if (confirmingDeleteId === doc.document_id) {
                          removePendingAttachment(doc.document_id);
                        } else {
                          setConfirmingDeleteId(doc.document_id);
                        }
                      }}
                      onMouseLeave={() => {
                        if (confirmingDeleteId === doc.document_id) {
                          setConfirmingDeleteId(null);
                        }
                      }}
                      className={`absolute -top-1 -end-1 h-5 w-5 rounded-full flex items-center justify-center transition-all ${
                        confirmingDeleteId === doc.document_id
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted border border-border text-muted-foreground hover:text-foreground hover:bg-muted/80"
                      }`}
                      title={confirmingDeleteId === doc.document_id ? t("chat.confirm_remove") : t("chat.remove_attachment")}
                    >
                      {confirmingDeleteId === doc.document_id ? (
                        <Check className="h-3 w-3" />
                      ) : (
                        <X className="h-3 w-3" />
                      )}
                    </button>
                  </div>
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
            {/* Actual input row */}
            <div className="flex items-center gap-2 px-3 py-2.5">

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
                    title={t("chat.discard_recording")}
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
                  <span className="text-[11px] text-primary/70 font-mono tabular-nums shrink-0 w-8 text-end">
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
                  title={t("chat.upload_file")}
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
                  placeholder={t("chat.input_placeholder")}
                  rows={1}
                  disabled={isLoading}
                  style={{ maxHeight: "160px" }}
                  className={`flex-1 resize-none bg-transparent text-sm text-foreground focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed leading-5 py-1.5 overflow-y-auto scrollbar-thin self-center transition-[opacity] duration-200 ${
                    isChatDragOver
                      ? "placeholder:opacity-0"
                      : "placeholder:opacity-100 placeholder:text-muted-foreground/60"
                  } placeholder:align-middle`}
                />
                {/* Ghost-text suffix — overlaid on top of the textarea, never shown on empty input */}
                {wordSuffix && input.length > 0 && !autocomplete && (
                  <span
                    aria-hidden="true"
                    className="pointer-events-none absolute inset-0 flex items-center text-sm leading-5 select-none py-1.5 overflow-hidden"
                    style={{ insetInlineStart: "calc(1.75rem + 6px)", color: "transparent" }}
                  >
                    <span className="whitespace-pre">{input}</span>
                    <span className="text-muted-foreground/40 whitespace-pre">{wordSuffix}</span>
                  </span>
                )}
              </div>
              )}

              {/* ── Model Picker ── */}
              <div ref={modelDropdownRef} className="relative shrink-0">
                <button
                  type="button"
                  onClick={() => setModelDropdownOpen(o => !o)}
                  className="group flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] font-medium text-muted-foreground/60 hover:text-foreground hover:bg-muted/50 transition-all border border-transparent hover:border-border/40"
                  title={t("chat.select_model")}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full shrink-0"
                    style={{ backgroundColor: MODEL_OPTIONS.find(m => m.id === selectedModel)?.color ?? "#60a5fa" }}
                  />
                  <span className="tabular-nums">{MODEL_OPTIONS.find(m => m.id === selectedModel)?.shortLabel}</span>
                  <ChevronDown className={`h-3 w-3 transition-transform duration-150 ${modelDropdownOpen ? "rotate-180" : ""}`} />
                </button>

                <AnimatePresence>
                  {modelDropdownOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 6, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 6, scale: 0.97 }}
                      transition={{ duration: 0.13 }}
                      className="absolute bottom-full mb-2 start-0 z-[9999] bg-card border border-border rounded-xl shadow-xl overflow-hidden min-w-[220px]"
                    >
                      <div className="px-3 py-2 border-b border-border/60">
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{t("chat.select_model")}</p>
                      </div>
                      {MODEL_OPTIONS.map(opt => (
                        <button
                          key={opt.id}
                          type="button"
                          onClick={() => {
                            setSelectedModel(opt.id);
                            setModelDropdownOpen(false);
                          }}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 text-start transition-colors ${
                            selectedModel === opt.id
                              ? "bg-primary/8 text-foreground"
                              : "hover:bg-muted text-foreground/70 hover:text-foreground"
                          }`}
                        >
                          <span
                            className="h-2 w-2 rounded-full shrink-0"
                            style={{ backgroundColor: opt.color }}
                          />
                          <div className="flex-1 min-w-0">
                            <p className="text-[12px] font-medium leading-none mb-0.5">{opt.label}</p>
                            <p className="text-[10px] text-muted-foreground/60 leading-none">{opt.desc}</p>
                          </div>
                          {selectedModel === opt.id && (
                            <Check className="h-3 w-3 text-primary shrink-0" />
                          )}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

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
                  title={notifyEnabled ? t("chat.notifications_on") : t("chat.enable_notifications")}
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
                    ? t("chat.stop_recording")
                    : voiceState === "transcribing"
                    ? t("chat.transcribing")
                    : t("chat.voice_message")
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
                  title={t("chat.stop_generating")}
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
        {t("chat.disclaimer")}
      </p>
    </div>
  );
};
