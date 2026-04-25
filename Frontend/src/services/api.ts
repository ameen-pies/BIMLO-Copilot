/**
 * API Service for BIMLO Copilot v3
 * 
 * Per-user/session data isolation:
 *   - All document uploads include session_id in query params
 *   - Backend scopes documents to (user_id, session_id) collection
 *   - /query endpoint receives pending_doc_ids to commit files on send
 *   - No cross-user or cross-session data leakage
 * 
 * Global news feed (separate from user data):
 *   - /api/news/* endpoints serve a global, auto-refreshed feed
 *   - ALL users access the same cached news
 *   - Completely isolated from the RAG engine's document storage
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// TYPES
// ============================================================================

export interface Document {
  document_id: string;
  filename: string;
  doc_type: string;
  timestamp: string;
  image_chunks?: number;
  table_chunks?: number;
  chunk_count?: number;
}

export interface Source {
  source_number: number;
  filename: string;
  doc_type: string;
  excerpt: string;
  has_images?: boolean;
  has_tables?: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  route?: string;
  analytics?: Record<string, unknown>;
  session_id?: string;
}

export interface UploadResponse {
  status: string;
  filename: string;
  document_id: string;
  session_id?: string;
  chunks_processed: number;
  cad_summary?: Record<string, unknown>;
  message: string;
}

export interface DocumentListResponse {
  status: string;
  documents: Document[];
  total: number;
  scoped?: boolean;
}

export interface DeleteResponse {
  status: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  vector_store: string;
  services: string;
  statistics?: {
    total_chunks?: number;
    total_documents?: number;
    active_sessions?: number;
    cached_documents?: number;
  };
}

// ============================================================================
// API CLIENT
// ============================================================================

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Generic request handler with error handling and auth token injection
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    // Attach auth token if available (stored by AuthContext)
    let token: string | null = null;
    if (typeof localStorage !== 'undefined') {
      try {
        const stored = localStorage.getItem('bimlo_auth');
        if (stored) token = JSON.parse(stored).token ?? null;
      } catch { /* ignore parse errors */ }
    }
    const authHeaders: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...authHeaders,
          ...options.headers,
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new Error(error.detail || 'Request failed');
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Unknown error occurred');
    }
  }

  // ==========================================================================
  // DOCUMENT OPERATIONS (per-user/session scoped with per-user isolation)
  // ==========================================================================

  /**
   * Upload a document for processing.
   * 
   * Per-user/session isolation:
   *   - Files are temporarily cached on backend (30min default)
   *   - File is ONLY committed to vector_store when message is sent
   *   - Backend uses (user_id, session_id) to create isolated Chroma collection
   *   - Other users cannot see this file
   * 
   * Pass sessionId so the backend scopes the doc to the current chat session.
   * User ID is extracted from auth token automatically.
   */
  async uploadDocument(file: File, sessionId?: string): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const url = sessionId ? `/upload?session_id=${encodeURIComponent(sessionId)}` : '/upload';
    return this.request<UploadResponse>(url, {
      method: 'POST',
      body: formData,
    });
  }

  /**
   * List documents — filtered by session_id (and user from auth token).
   * 
   * Per-user isolation:
   *   - Only returns documents uploaded by the current user in this session
   *   - Different users see different documents
   *   - Backend queries: MATCH (u:User)-[:UPLOADED]->(d:Document)
   *                      WHERE d.session_id = $session_id
   */
  async listDocuments(sessionId?: string): Promise<DocumentListResponse> {
    const url = sessionId ? `/documents?session_id=${encodeURIComponent(sessionId)}` : '/documents';
    return this.request<DocumentListResponse>(url);
  }

  /**
   * Delete a document by ID.
   * 
   * Per-user isolation:
   *   - Removes document from user_{user_id}_session_{session_id} collection in Chroma
   *   - User can only delete their own documents in their own sessions
   *   - Scoped to sessionId to prevent cross-session deletion
   */
  async deleteDocument(documentId: string, sessionId?: string): Promise<DeleteResponse> {
    const url = sessionId
      ? `/documents/${documentId}?session_id=${encodeURIComponent(sessionId)}`
      : `/documents/${documentId}`;
    return this.request<DeleteResponse>(url, {
      method: 'DELETE',
    });
  }

  /**
   * Fetch raw text content of a document for in-app viewing.
   * 
   * Per-user isolation:
   *   - Searches vector_store within user_{user_id}_session_{session_id} collection
   *   - Returns content reassembled from chunks
   *   - Scoped to sessionId for isolation
   */
  async getDocumentContent(documentId: string, sessionId?: string): Promise<{ document_id: string; filename: string; content: string }> {
    const url = sessionId
      ? `/documents/${documentId}/content?session_id=${encodeURIComponent(sessionId)}`
      : `/documents/${documentId}/content`;
    return this.request(url);
  }

  /**
   * Download the original uploaded file.
   * 
   * Per-user isolation:
   *   - Can only download files from your own session
   *   - Scoped to sessionId
   */
  async downloadDocument(documentId: string, sessionId?: string): Promise<Blob> {
    const url = sessionId
      ? `/documents/${documentId}/download?session_id=${encodeURIComponent(sessionId)}`
      : `/documents/${documentId}/download`;
    return this.request(url).then(data => new Blob([JSON.stringify(data)], { type: 'application/octet-stream' }));
  }

  // ==========================================================================
  // QUERY OPERATIONS (per-session RAG with per-user isolation)
  // ==========================================================================

  /**
   * Query the RAG system with per-user/session isolation.
   * 
   * Per-user/session isolation:
   *   - Searches ONLY within user_{user_id}_session_{session_id} Chroma collection
   *   - Other users' documents are completely invisible
   *   - Backend extracts user_id from auth token automatically
   * 
   * File commitment:
   *   - pending_doc_ids: documents uploaded but not yet indexed
   *   - Backend receives this list and commits them to vector_store
   *   - Future queries include these documents in search results
   * 
   * When you call this with sessionId and pendingDocIds:
   *   1. Backend commits pending files to vector_store (with user_id + session_id)
   *   2. Searches ONLY within that user/session collection
   *   3. Generates sources only from accessible documents
   *   4. Stores query+response in session history (Neo4j)
   */
  async query(
    query: string,
    topK: number = 5,
    sessionId?: string,
    pendingDocIds?: string[]
  ): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        top_k: topK,
        session_id: sessionId,
        pending_doc_ids: pendingDocIds || [],  // ← Files to commit
      }),
    });
  }

  /**
   * Query the RAG system with streaming response (Server-Sent Events).
   * 
   * Same per-user/session isolation as /query
   * Returns streamed text events for real-time response display
   */
  async queryStream(
    query: string,
    topK: number = 5,
    sessionId?: string,
    pendingDocIds?: string[]
  ): Promise<ReadableStream<any>> {
    const url = `${this.baseURL}/query-stream`;
    let token: string | null = null;
    if (typeof localStorage !== 'undefined') {
      try {
        const stored = localStorage.getItem('bimlo_auth');
        if (stored) token = JSON.parse(stored).token ?? null;
      } catch { /* ignore */ }
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        query,
        top_k: topK,
        session_id: sessionId,
        pending_doc_ids: pendingDocIds || [],
      }),
    });

    if (!response.ok) {
      throw new Error(`Stream error: ${response.statusText}`);
    }

    return response.body as ReadableStream<any>;
  }

  /**
   * Generate a detailed report.
   * 
   * Per-user isolation:
   *   - Works within the current session's document scope
   *   - Report is saved to Neo4j and tied to the session
   *   - Uses documents from user_{user_id}_session_{session_id} collection
   */
  async generateReport(query: string, sessionId?: string): Promise<any> {
    return this.request('/generate-report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
      }),
    });
  }

  // ==========================================================================
  // GLOBAL NEWS FEED (NOT per-user, shared across all users)
  // ==========================================================================

  /**
   * Fetch the global news feed.
   * 
   * GLOBAL (NOT per-user):
   *   - This feed is GLOBAL — all users see the same articles
   *   - Refreshed automatically every 4 days by news_pipeline.py
   *   - Completely separate from user documents and the RAG engine
   *   - Does NOT require sessionId — it's not scoped
   * 
   * Returns:
   *   - Titles, summaries, categories
   *   - AI impact analysis
   *   - Article URLs for full reading
   *   - Published dates and sources
   */
  async getNewsFeed(page?: number, limit?: number): Promise<any> {
    const params = new URLSearchParams();
    if (page !== undefined) params.append('page', String(page));
    if (limit !== undefined) params.append('limit', String(limit));
    const url = params.toString() ? `/api/news?${params}` : '/api/news';
    return this.request(url);
  }

  /**
   * Get news feed metadata (total count, last refresh, etc.)
   * 
   * GLOBAL — same metadata for all users
   */
  async getNewsMeta(): Promise<any> {
    return this.request('/api/news/meta');
  }

  /**
   * Get a specific page of news articles.
   * 
   * GLOBAL — same articles for all users on this page
   */
  async getNewsPage(pageNum: number): Promise<any> {
    return this.request(`/api/news/page/${pageNum}`);
  }

  /**
   * Chat about news articles.
   * 
   * COMPLETELY ISOLATED from the RAG engine:
   *   - Does NOT touch user documents
   *   - Accepts pinned articles and generates insights about them
   *   - Maintains its own session history (separate from /query)
   *   - Uses global news articles, not user documents
   * 
   * Different from /query:
   *   - /query searches your documents in a per-user collection
   *   - /api/news/chat searches global news feed
   *   - They never mix or interact
   */
  async newsChatQuery(
    query: string,
    pinnedArticles?: any[],
    sessionId?: string
  ): Promise<any> {
    return this.request('/api/news/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        pinned_articles: pinnedArticles || [],
        session_id: sessionId,
      }),
    });
  }

  // ==========================================================================
  // HEALTH CHECK
  // ==========================================================================

  /**
   * Check API health status.
   * 
   * Returns:
   *   - LLM provider status (CF Workers, Groq)
   *   - Vector store status
   *   - Global statistics (not scoped to session)
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const api = new APIClient();

export default api;

// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================

export const {
  uploadDocument,
  listDocuments,
  deleteDocument,
  getDocumentContent,
  downloadDocument,
  query,
  queryStream,
  generateReport,
  healthCheck,
  getNewsFeed,
  getNewsMeta,
  getNewsPage,
  newsChatQuery,
} = api;

// ============================================================================
// USAGE EXAMPLES
// ============================================================================

/*

// Example 1: Upload a file (cached, not indexed yet)
const sessionId = "session_12345";
const file = new File(["content"], "report.pdf");
const uploadResponse = await api.uploadDocument(file, sessionId);
// File is now cached but NOT in vector store yet

// Example 2: Send a message with pending files (files get indexed)
const queryResponse = await api.query(
  "Summarize the report",
  5,
  sessionId,
  [uploadResponse.document_id]  // ← Commit this file
);
// File is now indexed to user_{user_id}_session_{sessionId} collection

// Example 3: List documents for this session
const docs = await api.listDocuments(sessionId);
// Only returns documents from this user's session

// Example 4: Delete a document
await api.deleteDocument(uploadResponse.document_id, sessionId);
// Removed from vector_store collection

// Example 5: Access global news (same for all users)
const newsFeed = await api.getNewsFeed(page = 1);
// All users see the same articles

// Example 6: Chat about news (separate from documents)
const newsResponse = await api.newsChatQuery(
  "What's the impact of 5G on BIM?",
  [article1, article2],  // ← Pinned news articles
  newsSessionId
);
// This searches articles, NOT user documents

*/