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
  chunks_processed: number;
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

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
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
   * Upload a document for processing
   */
  async uploadDocument(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<UploadResponse>('/upload', {
      method: 'POST',
      body: formData,
    });
  }

  /**
   * List all indexed documents
   */
  async listDocuments(): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>('/documents');
  }

  /**
   * Delete a document by ID
   */
  async deleteDocument(documentId: string): Promise<DeleteResponse> {
    return this.request<DeleteResponse>(`/documents/${documentId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Fetch raw text content of a document for in-app viewing
   */
  async getDocumentContent(documentId: string): Promise<{ document_id: string; filename: string; content: string }> {
    return this.request(`/documents/${documentId}/content`);
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