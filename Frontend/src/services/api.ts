/**
 * API Service for BIMLO Copilot
 * 
 * Handles all communication with the FastAPI backend
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
}

export interface Source {
  source_number: number;
  filename: string;
  doc_type: string;
  excerpt: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
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
    total_chunks: number;
    total_documents: number;
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
   * Generic request handler with error handling
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    // Attach auth token if available (stored by AuthContext as JSON under 'bimlo_auth')
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
  // DOCUMENT OPERATIONS
  // ==========================================================================

  /**
   * Upload a document for processing.
   * Pass sessionId so the backend scopes the doc to the current chat session.
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
   * List documents — filtered by session_id when provided.
   * Each chat session only sees files uploaded in that session.
   */
  async listDocuments(sessionId?: string): Promise<DocumentListResponse> {
    const url = sessionId ? `/documents?session_id=${encodeURIComponent(sessionId)}` : '/documents';
    return this.request<DocumentListResponse>(url);
  }

  /**
   * Delete a document by ID
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
   * Fetch raw text content of a document for in-app viewing
   */
  async getDocumentContent(documentId: string, sessionId?: string): Promise<{ document_id: string; filename: string; content: string }> {
    const url = sessionId
      ? `/documents/${documentId}/content?session_id=${encodeURIComponent(sessionId)}`
      : `/documents/${documentId}/content`;
    return this.request(url);
  }

  // ==========================================================================
  // QUERY OPERATIONS
  // ==========================================================================

  /**
   * Query the RAG system
   */
  async query(query: string, topK: number = 5): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        top_k: topK,
      }),
    });
  }

  /**
   * Generate a detailed report
   */
  async generateReport(query: string): Promise<any> {
    return this.request('/generate-report', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
  }

  // ==========================================================================
  // HEALTH CHECK
  // ==========================================================================

  /**
   * Check API health status
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
  query,
  generateReport,
  healthCheck,
} = api;