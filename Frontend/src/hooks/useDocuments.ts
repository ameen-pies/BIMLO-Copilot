import { useState, useRef, useEffect, useCallback } from "react";
import api, { Document } from "@/services/api";
import {
  IMAGE_EXTS,
  CAD_EXTS,
  Message,
  createUniqueId,
} from "@/pages/chatTypes";
import { ViewerState } from "@/components/chat/chatRenderers";
import { serializeError } from "@/utils/chatUtils";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys, useDocumentsQuery } from "@/services/queries";

export function useDocuments(options: {
  toast: (props: { title: string; description?: string; variant?: "default" | "destructive" }) => void;
  sessionIdRef: React.MutableRefObject<string | null>;
  sessionId: string | null;
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}) {
  const { toast, sessionIdRef, sessionId, setSessionId } = options;

  const queryClient = useQueryClient();
  const docQuery = useDocumentsQuery(sessionId);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [pendingDocIds, setPendingDocIds] = useState<string[]>([]);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [duplicateBanner, setDuplicateBanner] = useState<string | null>(null);
  const duplicateBannerTimeoutRef = useRef<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [bubbleDoc, setBubbleDoc] = useState<Document | null>(null);
  const [bubbleViewer, setBubbleViewer] = useState<ViewerState | null>(null);
  const bubbleViewerRef = useRef<ViewerState | null>(null);
  bubbleViewerRef.current = bubbleViewer;
  const [viewer, setViewer] = useState<ViewerState | null>(null);
  const blobUrlMapRef = useRef<Map<string, { url: string; type: "pdf" | "image" | "txt" | "cad"; cadSummary?: Record<string, unknown>; ifcBlobUrl?: string }>>(new Map());
  const bubbleHighlightRef = useRef<HTMLElement>(null);
  const bubbleScrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docFileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const dragCounterRef = useRef(0);
  const pageDragCounterRef = useRef(0);
  const [isChatDragOver, setIsChatDragOver] = useState(false);
  const chatDragCounterRef = useRef(0);

  useEffect(() => {
    if (docQuery.data && sessionIdRef.current) {
      const uniqueDocs = docQuery.data.documents.filter((doc, index, arr) =>
        arr.findIndex(d => d.document_id === doc.document_id) === index
      );
      setDocuments(uniqueDocs);
    }
  }, [docQuery.data]);

  useEffect(() => {
    return () => {
      if (duplicateBannerTimeoutRef.current) {
        window.clearTimeout(duplicateBannerTimeoutRef.current);
      }
    };
  }, []);

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

  const createPreviewUrlForFile = (file: File) => {
    const ext = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`;
    const isImage = IMAGE_EXTS.includes(ext);
    const isPdf = ext === '.pdf';
    const isCad = CAD_EXTS.includes(ext);
    if (!isImage && !isPdf && !isCad) return null;
    const url = URL.createObjectURL(file);
    const resultType: "pdf" | "image" | "cad" = isPdf ? 'pdf' : isImage ? 'image' : 'cad';
    return {
      url,
      type: resultType,
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

  const handleFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    let sid = sessionIdRef.current;
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
      if (file.size === 0) {
        toast({ title: t("chat.upload_empty_title") || "Empty file", description: t("chat.upload_empty_desc") || `"${file.name}" is empty or unreadable.`, variant: "destructive" });
        continue;
      }
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
        if (res.is_image && !blobUrlMapRef.current.has(res.document_id)) {
          const objectUrl = URL.createObjectURL(file);
          blobUrlMapRef.current.set(res.document_id, { url: objectUrl, type: 'image' });
        }
        setPendingDocIds(prev => [...prev, res.document_id]);
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
      } catch (err) {
        setDocuments(prev => prev.filter(d => d.document_id !== placeholderId));
        toast({ title: "Upload failed", description: String(err).replace(/^Error:?\s*/i, "") });
      }
    }
    setIsUploading(false);
  };

  const fetchBlobUrl = async (doc: Document): Promise<{ url: string; type: "pdf" | "image" | "txt" }> => {
    const cached = blobUrlMapRef.current.get(doc.document_id);
    if (cached) return cached;

    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
    const type: "pdf" | "image" | "txt" = IMAGE_EXTS.includes(`.${ext}`) ? "image" : ext === "pdf" ? "pdf" : "txt";

    const base =
      (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_URL) || "http://localhost:8000";
    const sessionQuery = sessionIdRef.current ? `?session_id=${encodeURIComponent(sessionIdRef.current)}` : "";

    let blob: Blob;
    if (type === "image") {
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
    const current = bubbleViewerRef.current;

    if (current && current.doc.document_id === doc.document_id && !current.loading && !current.error) {
      setBubbleViewer(prev => prev ? { ...prev, highlightText, highlightLines, highlightKey: (prev.highlightKey ?? 0) + 1 } : prev);
      return;
    }

    const ext = doc.filename.split('.').pop()?.toLowerCase() ?? '';
    const isPdfOrImage = ext === 'pdf' || IMAGE_EXTS.includes(`.${ext}`);
    const isCadExt = ['.ifc', '.ifczip', '.dxf', '.dwg', '.step', '.stp'].includes(`.${ext}`);
    setBubbleDoc(doc);

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
        const res = await (api as any).getDocumentContent(doc.document_id, sessionIdRef.current ?? undefined);
        setBubbleViewer(p => p && p.doc.document_id === doc.document_id ? { ...p, content: res.content, loading: false } : p);
      } catch (err) {
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

  const openDocumentAtNumber = async (filename: string, numValue: string, fallbackExcerpt: string) => {
    const doc = documents.find(d => d.filename === filename);
    if (!doc) return;
    const current = bubbleViewerRef.current;
    if (current && current.doc.document_id === doc.document_id && current.content) {
      const line = findLineForNumber(current.content, numValue) ?? fallbackExcerpt;
      setBubbleViewer(prev => prev ? { ...prev, highlightText: line, highlightLines: null, highlightKey: (prev.highlightKey ?? 0) + 1 } : prev);
      return;
    }
    setBubbleDoc(doc);
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

  const loadDocuments = async (sid?: string | null) => {
    const effectiveSid = sid ?? sessionIdRef.current;
    if (!effectiveSid) {
      setDocuments([]);
      return;
    }
    try {
      const data = await queryClient.fetchQuery({
        queryKey: queryKeys.documents(effectiveSid),
        queryFn: () => api.listDocuments(effectiveSid),
        staleTime: 30_000,
      });
      const uniqueDocs = data.documents.filter((doc, index, arr) =>
        arr.findIndex(d => d.document_id === doc.document_id) === index
      );
      setDocuments(uniqueDocs);
    } catch (error) {
      console.error("Failed to load documents:", error);
    }
  };

  const removeDocument = async (documentId: string) => {
    setConfirmingDeleteId(null);
    revokePreviewUrl(documentId);
    // Optimistic: remove from local state immediately
    setDocuments(prev => prev.filter(d => d.document_id !== documentId));
    setPendingDocIds(prev => prev.filter(id => id !== documentId));
    try {
      await api.deleteDocument(documentId, sessionIdRef.current ?? undefined);
      queryClient.invalidateQueries({ queryKey: queryKeys.documents(sessionIdRef.current ?? "") });
    } catch (error) {
      // Revert on failure by reloading
      await loadDocuments(sessionIdRef.current);
      toast({
        title: "Delete failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return {
    documents, setDocuments,
    pendingDocIds, setPendingDocIds,
    confirmingDeleteId, setConfirmingDeleteId,
    duplicateBanner, setDuplicateBanner,
    duplicateBannerTimeoutRef,
    isUploading, setIsUploading,
    isDragOver, setIsDragOver,
    isChatDragOver, setIsChatDragOver,
    dragCounterRef, pageDragCounterRef, chatDragCounterRef,
    blobUrlMapRef,
    bubbleDoc, setBubbleDoc,
    bubbleViewer, setBubbleViewer,
    bubbleViewerRef,
    bubbleHighlightRef, bubbleScrollRef,
    viewer, setViewer,
    fileInputRef, docFileInputRef,
    loadDocuments,
    handleFiles,
    removeDocument,
    createPreviewUrlForFile,
    revokePreviewUrl,
    formatSize,
    fetchBlobUrl,
    openBubbleDoc,
    openDocumentViewer,
    openDocumentAtExcerpt,
    openDocumentAtExcerpts,
    openDocumentAtNumber,
    findLineForNumber,
    removePendingAttachment,
  };
}
