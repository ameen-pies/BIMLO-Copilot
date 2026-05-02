import os
import re
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime


# Defaults — all overridable via environment variables or constructor args
_DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")


class VectorStoreManager:
    """
    Per-user/session ChromaDB vector store with automatic collection isolation.
    
    Key changes (v3):
      - NO global collection. Instead, each user/session pair gets its own collection.
      - Collection naming: user_{user_id}_session_{session_id}
      - OR if only session_id is known: session_{session_id}
      - search() and add_document() accept user_id and session_id parameters.
      - Complete data isolation: User A cannot see User B's vectors.
      - Cleanup: delete a user/session collection when the session ends.

    Env vars:
      CHROMA_PERSIST_DIR  — where ChromaDB stores data  (default: ./data/chroma)
      EMBEDDING_MODEL     — SentenceTransformer model   (default: all-MiniLM-L6-v2)

    Image/table detection (unchanged from v2):
      - Chunks with image descriptions are flagged has_images=True
      - search() returns has_images/has_tables in metadata
    """

    def __init__(
        self,
        persist_directory: str = _DEFAULT_PERSIST_DIR,
        embedding_model:   str = _DEFAULT_EMBED_MODEL,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self._collection_cache: Dict[str, any] = {}
        print(
            f"✅ Vector store ready (per-user/session isolation) "
            f"(model: {embedding_model}, persist_dir: {persist_directory})"
        )

    # ── Collection name utilities ──────────────────────────────────────────

    @staticmethod
    def _get_collection_name(user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Generate a unique, safe collection name for a user/session pair.
        Uses hashing to keep names under ChromaDB's 63 character limit.
        
        Rules:
          - If both user_id and session_id: u_{hash(user_id)}_s_{hash(session_id)}
          - If only user_id: u_{hash(user_id)}
          - If only session_id: s_{hash(session_id)}
          - Fallback: "global"
        
        Names are always under 63 chars and contain only alphanumeric/underscore/dash.
        """
        import hashlib
        
        def _hash(s: str) -> str:
            """Return first 12 chars of SHA256 hash (32 bits of entropy)"""
            if not s:
                return ""
            return hashlib.sha256(s.encode()).hexdigest()[:12]
        
        if user_id and session_id:
            u_hash = _hash(user_id)
            s_hash = _hash(session_id)
            name = f"u{u_hash}s{s_hash}"  # u<12>s<12> = 26 chars max
        elif user_id:
            u_hash = _hash(user_id)
            name = f"u{u_hash}"  # u<12> = 13 chars max
        elif session_id:
            s_hash = _hash(session_id)
            name = f"s{s_hash}"  # s<12> = 13 chars max
        else:
            name = "global"
        
        return name.lower()

    def _get_collection(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Get or create a collection for this user/session pair.
        Results are cached in memory to avoid repeated .get_or_create_collection calls.
        """
        collection_name = self._get_collection_name(user_id, session_id)
        
        if collection_name not in self._collection_cache:
            self._collection_cache[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        
        return self._collection_cache[collection_name]

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        chunks: List[Dict],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add document chunks to the user/session's isolated collection.
        
        Args:
            filename: Original filename
            chunks: List of chunk dicts with 'text' and optional 'metadata'
            user_id: User ID (optional but recommended)
            session_id: Chat session ID (optional but recommended)
            doc_id: Document ID to use (generated if not provided)
        
        Returns:
            The document ID
        """
        doc_id = doc_id or str(uuid.uuid4())
        collection = self._get_collection(user_id, session_id)
        
        texts, embeddings, metadatas, ids = [], [], [], []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            meta = chunk.get("metadata", {}).copy()

            # Detect whether this chunk contains any image/table descriptions
            has_images = bool(
                re.search(r"\[IMAGE (?:on page|in document)", text, re.IGNORECASE)
            )
            has_tables = bool(
                re.search(r"\[TABLE on page", text, re.IGNORECASE)
            )

            meta.update({
                "document_id": doc_id,
                "chunk_index": i,
                "timestamp":   datetime.now().isoformat(),
                "filename":    filename,
                "user_id":     user_id or "",
                "session_id":  session_id or "",
                "has_images":  has_images,
                "has_tables":  has_tables,
            })
            texts.append(text)
            embeddings.append(self.embedding_model.encode(text).tolist())
            metadatas.append(meta)
            ids.append(f"{doc_id}_{i}")

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        image_chunks = sum(1 for m in metadatas if m.get("has_images"))
        table_chunks = sum(1 for m in metadatas if m.get("has_tables"))
        coll_name = self._get_collection_name(user_id, session_id)
        print(
            f"Added {len(chunks)} chunks for '{filename}' to {coll_name} "
            f"({image_chunks} with images, {table_chunks} with tables)"
        )
        return doc_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search within a user/session's collection.
        
        Only returns chunks from that specific user/session — no cross-user leakage.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            user_id: User ID (optional)
            session_id: Session ID (optional)
            filter_dict: Additional ChromaDB where filter (AND'd with user/session scope)
        
        Returns:
            List of dicts with text, metadata, distance, id, has_images, has_tables
        """
        collection = self._get_collection(user_id, session_id)
        query_embedding = self.embedding_model.encode(query).tolist()
        
        where = {**(filter_dict or {})}
        # The where filter is already scoped to this collection implicitly,
        # but we can add extra filters if needed
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )
        
        formatted: List[Dict] = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                meta = results["metadatas"][0][i]
                formatted.append({
                    "text":       results["documents"][0][i],
                    "metadata":   meta,
                    "distance":   results["distances"][0][i] if "distances" in results else None,
                    "id":         results["ids"][0][i],
                    "has_images": meta.get("has_images", False),
                    "has_tables": meta.get("has_tables", False),
                })
        return formatted

    def delete_document(self, doc_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Delete all chunks of a document from the user/session's collection.
        """
        collection = self._get_collection(user_id, session_id)
        results = collection.get(where={"document_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"Deleted document {doc_id} ({len(results['ids'])} chunks)")
        else:
            print(f"No document found with ID {doc_id}")

    def list_documents(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> List[Dict]:
        """
        List unique documents in the user/session's collection.
        """
        collection = self._get_collection(user_id, session_id)
        all_results = collection.get()
        docs: Dict[str, Dict] = {}
        for meta in all_results.get("metadatas", []):
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id":    doc_id,
                    "filename":       meta.get("filename", "unknown"),
                    "doc_type":       meta.get("doc_type", "unknown"),
                    "timestamp":      meta.get("timestamp", "unknown"),
                    "image_chunks":   0,
                    "table_chunks":   0,
                }
            if meta.get("has_images"):
                docs[doc_id]["image_chunks"] += 1
            if meta.get("has_tables"):
                docs[doc_id]["table_chunks"] += 1
        return list(docs.values())

    def has_documents(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """
        Return True if the user/session's collection contains any documents.
        """
        collection = self._get_collection(user_id, session_id)
        return collection.count() > 0

    def get_collection_stats(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """
        Get stats for the user/session's collection.
        """
        collection = self._get_collection(user_id, session_id)
        all_results = collection.get()
        image_chunks = sum(
            1 for m in all_results.get("metadatas", []) if m.get("has_images")
        )
        table_chunks = sum(
            1 for m in all_results.get("metadatas", []) if m.get("has_tables")
        )
        return {
            "total_chunks":    collection.count(),
            "total_documents": len(self.list_documents(user_id, session_id)),
            "image_chunks":    image_chunks,
            "table_chunks":    table_chunks,
        }

    def get_global_stats(self) -> Dict:
        """
        Aggregate stats across ALL collections (for admin dashboard).
        Replaces get_collection_stats() in health checks so the admin panel
        shows the real total instead of the empty 'global' fallback collection.
        """
        total_chunks    = 0
        total_documents = 0
        image_chunks    = 0
        table_chunks    = 0

        try:
            all_collections = self.client.list_collections()
            for col in all_collections:
                col_obj   = self.client.get_collection(col.name)
                results   = col_obj.get()
                metadatas = results.get("metadatas", [])

                doc_ids = set(
                    m.get("document_id") for m in metadatas if m.get("document_id")
                )
                total_chunks    += col_obj.count()
                total_documents += len(doc_ids)
                image_chunks    += sum(1 for m in metadatas if m.get("has_images"))
                table_chunks    += sum(1 for m in metadatas if m.get("has_tables"))
        except Exception as e:
            print(f"Warning: could not aggregate global stats: {e}")

        return {
            "total_chunks":    total_chunks,
            "total_documents": total_documents,
            "image_chunks":    image_chunks,
            "table_chunks":    table_chunks,
        }

    def delete_collection(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Completely delete a user/session's collection.
        Called when a session ends or user wants to clear all data.
        """
        collection_name = self._get_collection_name(user_id, session_id)
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self._collection_cache:
                del self._collection_cache[collection_name]
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")