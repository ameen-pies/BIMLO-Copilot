# BIMLO — Implementation Plan

> **Scope:** All crashes from CRASH_REPORT.md + all UI/UX fixes from stress test session
> **Priority:** P0 = crash/security, P1 = functional bug, P2 = UX polish, P3 = new feature
> **Status:** `[ ]` = todo, `[x]` = done, `[-]` = skipped/deferred

---

## Phase 1 — Critical Fixes (P0)

| # | Issue | Priority | Files | Fix | Status |
|---|-------|----------|-------|-----|--------|
| 1.1 | **Document viewer CSP blocks blob URLs + esm.sh** — PDF/image/IFC preview broken in DocumentViewer | P0 | `Frontend/index.html` (CSP meta), `Frontend/nginx.conf:14`, `Frontend/src/components/chat/DocumentViewer.tsx:250-258`, `Frontend/src/components/chat/PdfViewer.tsx:185`, `Frontend/src/components/chat/XeokitViewer.tsx:24` | Add `blob:` to `connect-src` and `img-src`. Add `https://esm.sh` to `script-src` and `connect-src` for xeokit dynamic import. | `[x]` |
| 1.2 | **Empty/corrupted file crash** — `ValueError: Expected IDs to be a non-empty list, got []` in ChromaDB when uploading 0-byte, encrypted, or corrupted files | P0 | `Backend/services/vector_store.py:add_document`, `Backend/core/session_state.py:228`, `Frontend/src/hooks/useDocuments.ts:120` | Guard: early return in `add_document` if empty chunks. Skip in `session_state.py` fallback path. Frontend rejects 0-byte files with toast. i18n keys added. | `[x]` |
| 1.3 | **IFC hijacks routing** — IFC file in session forces all queries to CAD path, ignoring other attached docs | P0 | `Frontend/src/hooks/useChatPage.tsx`, `Frontend/src/hooks/useChatStream.ts`, `Backend/services/rag_engine.py`, `Backend/services/intent_classifier.py` | Root cause: frontend pre-routed to `/api/cad/query` directly, bypassing intent classifier. Fix: removed frontend CAD pre-routing (81 lines deleted), all queries now go through `/query-stream` → intent_classifier → `cad_node`. Added `"cad"` to AgentState.route type + status message. | `[x]` |
| 1.4 | **Multi-doc RAG ignores second doc** — attached_doc_filenames processed but retrieval only returns chunks from one document | P0 | `Backend/services/rag_engine.py` retrieve_vector, `Backend/services/intent_classifier.py` post-process step 6 | Root cause: intent_classifier collapsed attached list to last file; frontend sends empty attached_doc_filenames; retrieve_vector only searched one file. Fix: retrieve_vector now falls back to all session docs when attached list empty. Intent classifier step 6 only sets doc_scope for single-file attachments. Also: OpenRouter added to fallback chain (before Groq). Autocomplete debounce increased 280→500ms with proper abort cancellation. | `[x]` |
| 1.5 | **Cancel stops working after route picked** — backend keeps burning API credits after user cancels | P0 | `Backend/routers/chat.py:181-188` (cancel checks), `Backend/services/rag_engine.py` (LLM calls inside nodes) | Add `cancel_event.is_set()` checks inside each pipeline node before LLM calls. Currently cancel only checked between nodes, not inside them. Add guard before every `self.llm.chat()` call in rag_engine.py. | `[ ]` |

---

## Phase 2 — Functional Bugs (P1)

| # | Issue | Priority | Files | Fix | Status |
|---|-------|----------|-------|-----|--------|
| 2.1 | **Rate limit shows "Something went wrong"** — 429 response not handled, generic error displayed | P1 | `Frontend/src/components/AuthModal.tsx:26-41` (`apiPost`), `Backend/neo4j_auth.py:418-419` (limiter) | Add 429 status check in `apiPost`: if `res.status === 429`, show "Too many attempts. Try again in 1 minute." with countdown. Backend: ensure `detail` field includes retry-after hint. | `[ ]` |
| 2.2 | **50MB+ upload shows "unsupported file type"** — wrong error message for size limit | P1 | `Frontend/src/hooks/useDocuments.ts:178-180` (error handler), `Backend/main.py:94-100` (upload middleware) | Frontend: check for 413 status specifically, show "File too large. Maximum size is 50 MB." Backend: ensure middleware returns `{"detail": "File exceeds 50 MB limit"}` with 413 status. | `[ ]` |
| 2.3 | **Chart selection broken** — choosing chart option replaces message with "Got it - building X chart" but no chart renders, refresh shows original text with no options | P1 | `Frontend/src/hooks/useChatPage.tsx:1652-1654` (strips analytics on restore), `Frontend/src/hooks/useChatPage.tsx:1915` (SSE analytics), `Backend/services/rag_engine.py` analytics node | Two bugs: (a) analytics node in rag_engine may not actually generate chart after selection — check analytics node flow. (b) On conversation restore, `chart_clarification` analytics stripped at line 1652 — options disappear on refresh. Fix: preserve chart_clarification state, re-render options on restore. | `[ ]` |
| 2.4 | **Transform route returns wrong language** — user asks to translate page to English, system returns French | P1 | `Backend/services/rag_engine.py` transform node prompt | Check transform node system prompt. Likely missing instruction to respect user's language request. Add: "If user requests translation or language change, output in the requested language regardless of source document language." | `[ ]` |
| 2.5 | **News chat ignores language instruction** — "talk in English" not followed, responds in article language | P1 | `Backend/services/news_chat_agent.py:537-557` (`_SYSTEM_TEMPLATE`) | Add to system prompt: "ALWAYS respond in the same language as the user's query. If user requests a specific language, use that language regardless of article source language." | `[ ]` |
| 2.6 | **TTS voice doesn't change until refresh** — selecting new voice doesn't apply until page reload | P1 | `Frontend/src/pages/CallPage.tsx:171` (`selectedVoice` state), voice picker at lines 1322-1343 | Check if `selectedVoice` is passed to TTS API call. Likely state update not triggering re-render of TTS function. Ensure TTS request uses current `selectedVoice` state, not stale closure. | `[ ]` |
| 2.7 | **Smart vague-prompt detection** — system should ask for clarification when prompt is vague, not proceed blindly | P1 | `Backend/services/intent_classifier.py`, `Backend/services/rag_engine.py` (judge_plan node) | Add vagueness detection in intent classifier or judge node. If query is generic ("summarize", "analyze this"), ask user: "I can do X, Y, or Z — what would you prefer?" Don't hardcode; use LLM to detect vagueness and generate clarification options. | `[ ]` |
| 2.8 | **Full document visibility** — system only sees top_k chunks (8), misses content spread across large documents. Important info gets skipped. | P1 | `Backend/services/rag_engine.py` (iterative_rag node, synthesis logic) | Implement map-reduce approach for document-wide queries: (1) retrieve ALL chunks for attached docs, (2) synthesize in batches of 8, (3) merge batch summaries into final answer. Use existing `iterative_rag` route or extend it. For "summarize" / "analyze all" queries, bypass top_k limit and process full document. Keep current top_k for specific questions. | `[ ]` |

---

## Phase 3 — UI/UX Polish (P2)

| # | Issue | Priority | Files | Fix | Status |
|---|-------|----------|-------|-----|--------|
| 3.1 | **Long email overflows profile dropdown + admin table** | P2 | `Frontend/src/components/Navbar.tsx:119-126` (dropdown), `Frontend/src/pages/AdminPage.tsx:1295` (table cell) | Add `truncate max-w-[200px]` (or similar) to email display in both locations. CSS `overflow: hidden; text-overflow: ellipsis; white-space: nowrap`. | `[ ]` |
| 3.2 | **Emoji unicode shows "?" in avatar** | P2 | `Frontend/src/components/Navbar.tsx:38-40` (`displayName`), avatar initial logic | Check: avatar shows first char of `displayName`. Emoji is multi-byte — `charAt(0)` may split it. Use `[...displayName][0]` or check if first char is emoji, render emoji directly instead of text. | `[ ]` |
| 3.3 | **Chat title overlaps navbar buttons at small widths** | P2 | `Frontend/src/components/chat/ChatHeader.tsx` (title rendering) | Add `truncate` to title container, set `max-width` or `flex-shrink` to prevent overlap with action buttons. | `[ ]` |
| 3.4 | **Route badge not shown in thought process** | P2 | `Frontend/src/components/chat/ChatMessageList.tsx` (thinking trace rendering) | Check if route info from SSE status events is displayed. Add route badge/icon in the thinking/processing status section. | `[ ]` |
| 3.5 | **Edit prompt too tiny + blue** | P2 | `Frontend/src/components/chat/ChatMessageList.tsx:524-539` (edit textarea) | Change: `text-sm` → `text-base`, increase padding (`px-3 py-2`), change `text-primary-foreground` → `text-foreground` (black/white), match original message input styling. | `[ ]` |
| 3.6 | **Username smooshed in Google login** — no space between name and surname | P2 | `Backend/neo4j_auth.py:573` (username sanitization), line 585 (`display_name`) | `display_name = req.name` already stores full name with spaces. Issue is `username` at line 573 strips spaces: `c.isalnum() or c in "_-"`. Keep as-is (usernames shouldn't have spaces). Check if `display_name` is what's shown in UI — if UI shows `username`, switch to `display_name`. | `[ ]` |
| 3.7 | **Pending reroute after login** — news button → /news, chat button → /chat, navbar login → stay on index | P2 | `Frontend/src/context/AuthContext.tsx:111-114` (`showAuthModal`), `Frontend/src/components\AuthModal.tsx:186-192`, all components calling `showAuthModal` | Extend `showAuthModal` to accept a `redirectPath` parameter. Store in ref. In `handleAuthSuccess`, if `redirectPath` set, `navigate(redirectPath)` after login. Update all `showAuthModal` call sites with appropriate path. | `[ ]` |
| 3.8 | **Chat input expand direction** — splash (new chat): expand downward. In conversation: expand upward (current) | P2 | `Frontend/src/components/chat/ChatInput.tsx:412-429` (textarea), line 135 (splash positioning) | When `messagesLength === 0` (splash): set textarea `align-self: flex-end` or use `flex-direction: column` so it grows down. When in conversation: keep current upward behavior. | `[ ]` |
| 3.9 | **Delete doc from chat header → remove everywhere** — currently only removes from current view, persists in navbar/system | P2 | `Frontend/src/hooks/useDocuments.ts` (delete mutation), `Frontend/src/components/chat/ChatHeader.tsx` (doc list in header) | Ensure delete mutation invalidates all document queries (TanStack Query). Check: `useDeleteDocumentMutation` should `queryClient.invalidateQueries(['documents'])` to update navbar. Also clear from local state. | `[ ]` |
| 3.10 | **Chart selection loading indicator** — show spinner/dots when chart is being built after user picks dataset | P2 | `Frontend/src/components/chat/ChatMessageList.tsx:391-393` (ChartClarification handler) | After user clicks chart option, show loading state in message bubble (typing dots or spinner) until `chart_config` or `chart_error` analytics arrives. | `[ ]` |
| 3.11 | **"Can't hear you" popup X doesn't close** | P2 | `Frontend/src/pages/CallPage.tsx:995-1013` (silence warning banner) | Check dismiss button handler. Likely `setShowSilenceWarning(false)` not wired to X button click. Fix onclick handler. | `[ ]` |
| 3.12 | **"Haute parleur active" too long** — shorten translation | P2 | `Frontend/src/locales/fr.json` (call-related keys) | Find key for speaker active message. Change to "Haut-parleur ON" or "HP actif" — shorter. | `[ ]` |
| 3.13 | **Duplicate drag-and-drop overlays** — need one full-screen overlay | P2 | `Frontend/src/components/chat/ChatInput.tsx:306-322` (chat-level overlay) | Add global drag listener on `document` for full-screen overlay. Keep chat-level overlay for chat area. When file dragged over full screen → show full overlay. Over chat area → show chat overlay (more specific). | `[ ]` |

---

## Phase 4 — Judge & Quality (Future)

| # | Issue | Priority | Files | Fix | Status |
|---|-------|----------|-------|-----|--------|
| 4.1 | **Judge too aggressive / false rejections** — judge rejects correct answers that contain real info from uploaded docs, flags citations as "hallucinated" | P1 | `Backend/services/cad_ifc_agent.py` (_judge_answer), `Backend/services/rag_engine.py` (judge_evaluate node) | Current judge uses LLM-only verification (no grounding check). Implement source-verified judge: (1) extract claims from answer, (2) search source chunks for each claim, (3) score based on actual evidence presence. Reject only when claim has no supporting chunk. Brainstorm approaches before implementing. | `[ ]` |
| 3.14 | **Offline indicator** — no network status detection | P2 | `Frontend/src/App.tsx` or new component | Add `navigator.onLine` + `online`/`offline` event listeners. Show banner "Connection lost" when offline, dismiss when back online. | `[ ]` |
| 3.16 | **Reset zoom button too big in image preview** — make smaller and more subtle | P2 | `Frontend/src/components/chat/ZoomableImage.tsx:152-154` (reset button) | Reduce button size, use icon-only (no text), lower opacity, smaller font. Match zoom-in/zoom-out button style. | `[ ]` |
| 3.15 | **Admin dashboard always shows 0 online** | P2 | `Backend/neo4j_auth.py:1389-1424` (admin_stats) | Logic based on `last_seen` timestamp with 5-min threshold. In dev with single user, heartbeat every 60s should keep user "online". Check: is `last_seen` actually updated by heartbeat? Verify at line 646-653. If working, issue is timing — admin page checks may lag. Add auto-refresh to admin stats. | `[ ]` |

---

## Phase 4 — New Features (P3)

| # | Issue | Priority | Files | Fix | Status |
|---|-------|----------|-------|-----|--------|
| 4.1 | **Arabic i18n** | P3 | `Frontend/src/locales/` (new `ar.json`), `Frontend/src/i18n.ts` (add ar resource), RTL support | Create `ar.json` with all translation keys. Add `ar` to i18n resources. Add RTL direction handling: `dir="rtl"` on `<html>` when lang=ar. Test all UI components for RTL layout. | `[ ]` |
| 4.2 | **Multi-user terminal view** | P3 | New admin feature | Design: admin terminal showing active sessions, concurrent queries, real-time pipeline status. Use existing `/auth/admin/logs/stream` SSE. Add session-level monitoring. | `[ ]` |

---

## Execution Order

```
Phase 1 (P0): 1.1 → 1.2 → 1.3 → 1.4 → 1.5
Phase 2 (P1): 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8
Phase 3 (P2): 3.1 → 3.2 → 3.3 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 → 3.4 → 3.10 → 3.11 → 3.12 → 3.13 → 3.14 → 3.15
Phase 4 (P3): 4.1 → 4.2
```

**Total: 31 fixes**
- P0: 5 (crash/security)
- P1: 8 (functional bugs)
- P2: 16 (UX polish)
- P3: 2 (new features)

---

## Cleanup — Post-Stress-Test

| # | Task | Files | Status |
|---|------|-------|--------|
| C.1 | **Remove OpenRouter temp provider** — delete `_OPENROUTER_API_URL`, `_OPENROUTER_MODEL`, `_call_openrouter()`, OpenRouter block in `call_llm()`, `openrouter_api_key` from config.py, `OPENROUTER_API_KEY` from .env | `Backend/services/llm_client.py`, `Backend/core/config.py`, `Backend/.env` | `[ ]` |

---

*Generated: 2026-05-17*
*BIMLO Copilot v3.0.0*
