import { Source } from "@/services/api";
import { serializeError } from "@/utils/chatUtils";

export async function queryBackend(
  question: string,
  sessionId: string | null,
  signal?: AbortSignal,
): Promise<{ answer: string; raw_answer?: string; sources: Source[]; confidence: number; session_id: string }> {
  const base =
    (typeof import.meta !== "undefined" && (import.meta as Record<string, unknown>).env
      ? ((import.meta as Record<string, { VITE_API_URL?: string }>).env.VITE_API_URL ?? "")
      : "") || "http://localhost:8000";

  const res = await fetch(`${base}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: question,
      top_k: 5,
      ...(sessionId ? { session_id: sessionId } : {}),
    }),
    signal,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = serializeError(body);
    } catch { /* ignore */ }
    throw new Error(detail);
  }

  return res.json();
}

export async function fetchAICompletion(
  typed: string,
  recentMessages: { role: string; content: string }[],
  apiBase: string,
  availableDocs: string[] = [],
  signal?: AbortSignal,
): Promise<string> {
  if (typed.trim().length < 4) return "";

  const lastAssistant = [...recentMessages].reverse().find(m => m.role === "assistant");
  const sessionContext = lastAssistant ? lastAssistant.content.slice(0, 300) : "";

  try {
    const res = await fetch(`${apiBase}/autocomplete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        partial:         typed,
        session_context: sessionContext,
        available_docs:  availableDocs.slice(0, 8),
        max_tokens:      28,
      }),
      signal: signal
        ? (AbortSignal as any).any
          ? (AbortSignal as any).any([signal, AbortSignal.timeout(5000)])
          : signal
        : AbortSignal.timeout(5000),
    });
    if (!res.ok) return "";
    const data = await res.json();
    let completion = data.completion ?? "";
    if (!completion || completion.trim().length === 0 || completion.length > 80) return "";
    if (!typed.match(/\s$/)) {
      completion = completion.replace(/^\s+/, "");
    }
    return completion;
  } catch {
    return "";
  }
}
