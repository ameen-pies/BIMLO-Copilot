import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "./api";

export const queryKeys = {
  documents: (sessionId: string) => ["documents", sessionId] as const,
  reports: (sessionId: string) => ["reports", sessionId] as const,
  conversations: ["conversations"] as const,
  conversation: (id: string) => ["conversations", id] as const,
};

export function useDocumentsQuery(sessionId: string | null) {
  return useQuery({
    queryKey: queryKeys.documents(sessionId ?? ""),
    queryFn: () => api.listDocuments(sessionId ?? undefined),
    enabled: !!sessionId,
    staleTime: 30_000,
  });
}

export function useDeleteDocumentMutation(sessionId: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: string) => api.deleteDocument(documentId, sessionId ?? undefined),
    onSuccess: () => {
      if (sessionId) {
        queryClient.invalidateQueries({ queryKey: queryKeys.documents(sessionId) });
      }
    },
  });
}

export function useReportsQuery(sessionId: string | null) {
  return useQuery({
    queryKey: queryKeys.reports(sessionId ?? ""),
    queryFn: async () => {
      if (!sessionId) return [];
      const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/reports?session_id=${encodeURIComponent(sessionId)}`);
      if (!res.ok) return [];
      return res.json();
    },
    enabled: !!sessionId,
    staleTime: 30_000,
  });
}

export function useConversationsQuery() {
  return useQuery({
    queryKey: queryKeys.conversations,
    queryFn: async () => {
      const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/auth/conversations`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) return [];
      const rows = await res.json();
      return rows.map((r: any) => ({
        id: r.id,
        title: r.title || "Untitled",
        preview: r.preview || "",
        timestamp: new Date(r.updated_at || Date.now()),
        messages: [],
        sessionId: r.session_id,
      }));
    },
    staleTime: 60_000,
  });
}

export function useConversationQuery(conversationId: string | null) {
  return useQuery({
    queryKey: queryKeys.conversation(conversationId ?? ""),
    queryFn: async () => {
      if (!conversationId) return null;
      const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/auth/conversations/${conversationId}`, {
        credentials: "include",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) return null;
      return res.json();
    },
    enabled: !!conversationId,
    staleTime: 30_000,
  });
}

export function useSaveConversationMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: {
      conversationId: string;
      sessionId: string;
      title: string;
      preview: string;
      messages: any[];
      chatType?: string;
    }) => {
      const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/auth/conversations/save`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: params.conversationId,
          session_id: params.sessionId,
          title: params.title,
          preview: params.preview,
          chat_type: params.chatType || "rag",
          messages: params.messages.map((m) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp:
              m.timestamp instanceof Date
                ? m.timestamp.toISOString()
                : m.timestamp,
            sources: m.sources ?? undefined,
            confidence: m.confidence ?? undefined,
            analytics: m.analytics ?? undefined,
          })),
        }),
      });
      if (!res.ok) throw new Error(`Save conversation failed: ${res.status}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.conversations });
    },
  });
}

export function useDeleteConversationMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (conversationId: string) => {
      const base = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/auth/conversations/${conversationId}`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.conversations });
    },
  });
}
