import React from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, X, Plus } from "lucide-react";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import { ChatInput } from "@/components/chat/ChatInput";
import { DocumentViewer } from "@/components/chat/DocumentViewer";
import { AutocompletePanel } from "@/components/chat/AutocompletePanel";
import { useChatPage } from "@/hooks/useChatPage.tsx";

const Chat = () => {
  const { t } = useTranslation();
  const {
    activeConversation, activeConvId, activeConvIdRef, activeReport,
    analyserRef, animFrameRef, applyAutocomplete, audioCtxRef,
    autocomplete, autocompletePos, autocompleteRef,
    blobUrlMapRef, bubbleDoc, bubbleHighlightRef, bubbleScrollRef, bubbleViewer,
    chatDragCounterRef, confirmingDeleteId, convLoading, conversations,
    convsPanelOpen, copiedMsgId, currentUser,
    deleteConversation, renameConversation, deletingReportId, docFileInputRef, docsPanelOpen,
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
  } = useChatPage();

  return (
    <div 
      className="h-screen flex bg-background relative overflow-hidden"
      style={{ ...(isDark && { background: "#07080f" }), transition: "background 0.15s ease" }}
      onDragEnterCapture={e => { e.preventDefault(); pageDragCounterRef.current++; if (pageDragCounterRef.current === 1) setIsDragOver(true); }}
      onDragLeaveCapture={e => { e.preventDefault(); pageDragCounterRef.current--; if (pageDragCounterRef.current <= 0) { pageDragCounterRef.current = 0; setIsDragOver(false); } }}
      onDragOverCapture={e => e.preventDefault()}
      onDropCapture={e => { e.preventDefault(); e.stopPropagation(); pageDragCounterRef.current = 0; dragCounterRef.current = 0; setIsDragOver(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
      onDragEnter={e => { e.preventDefault(); dragCounterRef.current++; if (dragCounterRef.current === 1) setIsDragOver(true); }}
      onDragLeave={e => { e.preventDefault(); dragCounterRef.current--; if (dragCounterRef.current <= 0) { dragCounterRef.current = 0; setIsDragOver(false); } }}
      onDragOver={e => e.preventDefault()}
      onDrop={e => { e.preventDefault(); e.stopPropagation(); dragCounterRef.current = 0; pageDragCounterRef.current = 0; setIsDragOver(false); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); }}
    >

      {/* Drag-and-drop overlay (page-level) */}
      <AnimatePresence>
        {isDragOver && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-[9999] bg-background/50 backdrop-blur-[3px] flex items-center justify-center pointer-events-none"
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
              <p className="text-sm font-medium text-foreground">{t("chat.drop_to_upload_chat")}</p>
              <p className="text-[11px] text-muted-foreground/60">{t("chat.supported_formats")}</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <AnimatePresence>
          {duplicateBanner && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="fixed top-4 left-0 right-0 flex justify-center z-50 pointer-events-none"
            >
              <div className="pointer-events-auto inline-flex items-center gap-2 px-4 py-2 rounded-full border border-emerald-500/50 bg-emerald-950 text-emerald-300 text-xs font-medium shadow-sm">
                <span>{duplicateBanner}</span>
                <button
                  onClick={() => {
                    setDuplicateBanner(null);
                    if (duplicateBannerTimeoutRef.current) {
                      window.clearTimeout(duplicateBannerTimeoutRef.current);
                      duplicateBannerTimeoutRef.current = null;
                    }
                  }}
                  className="ms-1 text-emerald-300 hover:text-emerald-100 transition-colors"
                  aria-label={t("chat.dismiss")}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <ChatHeader
          activeConversation={activeConversation}
          convsPanelOpen={convsPanelOpen}
          setConvsPanelOpen={setConvsPanelOpen}
          loadConversationsFromDB={loadConversationsFromDB}
          conversations={conversations}
          startNewConversation={startNewConversation}
          historySearch={historySearch}
          setHistorySearch={setHistorySearch}
          historySort={historySort}
          setHistorySort={setHistorySort}
          activeConvId={activeConvId}
          loadConversation={loadConversation}
          pendingConvIds={pendingConvIds}
          deleteConversation={deleteConversation}
          reports={reports}
          reportsPanelOpenRef={reportsPanelOpenRef}
          setReportsPanelOpen={setReportsPanelOpen}
          setActiveReport={setActiveReport}
          setReportEditMode={setReportEditMode}
          previewedVersion={previewedVersion}
          setPreviewedVersion={setPreviewedVersion}
          reportsPanelOpen={reportsPanelOpen}
          isGeneratingReport={isGeneratingReport}
          handleDownloadReport={handleDownloadReport}
          downloadingReportId={downloadingReportId}
          handleDeleteReport={handleDeleteReport}
          deletingReportId={deletingReportId}
          activeReport={activeReport}
          showVersionHistory={showVersionHistory}
          setShowVersionHistory={setShowVersionHistory}
          isPatchingReport={isPatchingReport}
          reportEditMode={reportEditMode}
          reportEditInstruction={reportEditInstruction}
          setReportEditInstruction={setReportEditInstruction}
          reportEditInputRef={reportEditInputRef}
          handlePatchReport={handlePatchReport}
          restoringVersion={restoringVersion}
          handleRestoreVersion={handleRestoreVersion}
          docsPanelOpenRef={docsPanelOpenRef}
          setDocsPanelOpen={setDocsPanelOpen}
          docsPanelOpen={docsPanelOpen}
          documents={documents}
          bubbleDoc={bubbleDoc}
          setBubbleDoc={setBubbleDoc}
          setBubbleViewer={setBubbleViewer}
          bubbleViewer={bubbleViewer}
          openBubbleDoc={openBubbleDoc}
          confirmingDeleteId={confirmingDeleteId}
          setConfirmingDeleteId={setConfirmingDeleteId}
          removeDocument={removeDocument}
          isUploading={isUploading}
          bubbleScrollRef={bubbleScrollRef}
          bubbleHighlightRef={bubbleHighlightRef}
          sessionIdRef={sessionIdRef}
          currentUser={currentUser}
          showAuthModal={showAuthModal}
          logout={logout}
          onRenameConversation={renameConversation}
        />

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
                <span>                  {t("chat.mic_issue")}</span>
                <button onClick={() => setShowSilenceWarning(false)} className="ms-1 hover:text-red-300 transition-colors">
                  <X className="h-3 w-3" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <ChatMessageList
          messages={messages}
          convLoading={convLoading}
          activeConvId={activeConvId}
          openSourceKey={openSourceKey}
          setOpenSourceKey={setOpenSourceKey}
          expandedTranscripts={expandedTranscripts}
          setExpandedTranscripts={setExpandedTranscripts}
          copiedMsgId={copiedMsgId}
          setCopiedMsgId={setCopiedMsgId}
          typingMessageId={typingMessageId}
          setTypingMessageId={setTypingMessageId}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
          notifyDismissed={notifyDismissed}
          editingMsgId={editingMsgId}
          setEditingMsgId={setEditingMsgId}
          editDraft={editDraft}
          setEditDraft={setEditDraft}
          feedback={feedback}
          setFeedback={setFeedback}
          documents={documents}
          blobUrlMapRef={blobUrlMapRef}
          openBubbleDoc={openBubbleDoc}
          thinkingVisible={thinkingVisible}
          typingConvId={typingConvId}
          thinkingSteps={thinkingSteps}
          thinkingExpanded={thinkingExpanded}
          setThinkingExpanded={setThinkingExpanded}
          setModelDropdownOpen={setModelDropdownOpen}
          showNotifyBanner={showNotifyBanner}
          conversations={conversations}
          reports={reports}
          activeReport={activeReport}
          activeConvIdRef={activeConvIdRef}
          messagesRef={messagesRef}
          sessionIdRef={sessionIdRef}
          editTextareaRef={editTextareaRef}
          messagesEndRef={messagesEndRef}
          handleSourceClick={handleSourceClick}
          runStreamingQuery={runStreamingQuery}
          updateConversationMessages={updateConversationMessages}
          saveConversationToDB={saveConversationToDB}
          fetchSuggestions={fetchSuggestions}
          fireNotification={fireNotification}
          navigate={navigate}
          handleEditSubmit={handleEditSubmit}
          handleEditCancel={handleEditCancel}
          handleRedo={handleRedo}
          openDocumentAtExcerpt={openDocumentAtExcerpt}
          handleDownloadReport={handleDownloadReport}
          downloadingReportId={downloadingReportId}
          setMessages={setMessages}
          setThinkingSteps={setThinkingSteps}
          setSuggestions={setSuggestions}
          setShowNotifyBanner={setShowNotifyBanner}
          serializeError={serializeError}
        />

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
                <span>{t("chat.notify_when_done")}</span>
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
                  {t("chat.allow")}
                </button>
                <span className="text-border">·</span>
                <button
                  onClick={() => {
                    setShowNotifyBanner(false);
                    setNotifyDismissed(true);
                  }}
                  className="hover:text-foreground transition-colors"
                >
                  {t("chat.no_thanks")}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <ChatInput
          inputAreaRef={inputAreaRef}
          messagesLength={messages.length}
          isChatDragOver={isChatDragOver}
          setIsChatDragOver={setIsChatDragOver}
          chatDragCounterRef={chatDragCounterRef}
          handleFiles={handleFiles}
          suggestions={suggestions}
          suggestionsLoading={suggestionsLoading}
          expandingSuggestion={expandingSuggestion}
          expandSuggestion={expandSuggestion}
          isLoading={isLoading}
          pendingDocIds={pendingDocIds}
          isUploading={isUploading}
          documents={documents}
          blobUrlMapRef={blobUrlMapRef}
          openBubbleDoc={openBubbleDoc}
          confirmingDeleteId={confirmingDeleteId}
          setConfirmingDeleteId={setConfirmingDeleteId}
          removePendingAttachment={removePendingAttachment}
          voiceState={voiceState}
          setVoiceState={setVoiceState}
          maxDurationTimerRef={maxDurationTimerRef}
          elapsedTimerRef={elapsedTimerRef}
          animFrameRef={animFrameRef}
          analyserRef={analyserRef}
          audioCtxRef={audioCtxRef}
          waveformSamplesRef={waveformSamplesRef}
          waveformBars={waveformBars}
          setWaveformBars={setWaveformBars}
          recordingElapsed={recordingElapsed}
          setRecordingElapsed={setRecordingElapsed}
          mediaRecorderRef={mediaRecorderRef}
          handleVoiceClick={handleVoiceClick}
          isHoveringVoice={isHoveringVoice}
          setIsHoveringVoice={setIsHoveringVoice}
          docFileInputRef={docFileInputRef}
          handleInputChange={handleInputChange}
          handleInputKeyDown={handleInputKeyDown}
          input={input}
          textareaRef={textareaRef}
          wordSuffix={wordSuffix}
          setWordSuffix={setWordSuffix}
          autocomplete={autocomplete}
          setAutocomplete={setAutocomplete}
          autocompletePos={autocompletePos}
          setAutocompletePos={setAutocompletePos}
          modelDropdownRef={modelDropdownRef}
          MODEL_OPTIONS={MODEL_OPTIONS}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          modelDropdownOpen={modelDropdownOpen}
          setModelDropdownOpen={setModelDropdownOpen}
          notifyEnabled={notifyEnabled}
          setNotifyEnabled={setNotifyEnabled}
          notifyDismissed={notifyDismissed}
          handleSend={handleSend}
          handleStop={handleStop}
          autocompleteRef={autocompleteRef}
          applyAutocomplete={applyAutocomplete}
        />

        <AnimatePresence>
          {viewer && (
            <DocumentViewer
              state={viewer}
              onClose={() => setViewer(null)}
            />
          )}
        </AnimatePresence>

        <AutocompletePanel
          autocomplete={autocomplete}
          autocompletePos={autocompletePos}
          autocompleteRef={autocompleteRef}
          applyAutocomplete={applyAutocomplete}
          setAutocomplete={setAutocomplete}
        />
      </div>
    </div>
  );
};

export default Chat;
