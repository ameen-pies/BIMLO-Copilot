import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime


# Defaults — all overridable via environment variables or constructor args
_DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_DEFAULT_COLLECTION  = os.getenv("CHROMA_COLLECTION",  "documents")
_DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")


class VectorStoreManager:
    """
    Generic vector store — domain-agnostic, fully configurable via env vars
    or constructor arguments.

    Env vars (all optional, sensible defaults provided):
      CHROMA_PERSIST_DIR  — where ChromaDB stores data  (default: ./data/chroma)
      CHROMA_COLLECTION   — collection name              (default: documents)
      EMBEDDING_MODEL     — SentenceTransformer model   (default: all-MiniLM-L6-v2)

    v2 additions:
      - Chunks that contain image descriptions (injected by DocumentProcessor)
        are automatically flagged with has_images=True in their metadata.
        This lets the UI and RAG engine know a chunk has visual context without
        any extra storage or separate collection.
      - search() returns has_images in every result dict so callers can render
        a visual indicator in source cards.
      - get_document_stats() returns per-document image description count so
        the upload confirmation can report "found 4 diagram descriptions".
    """

    def __init__(
        self,
        persist_directory: str = _DEFAULT_PERSIST_DIR,
        collection_name:   str = _DEFAULT_COLLECTION,
        embedding_model:   str = _DEFAULT_EMBED_MODEL,
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        print(
            f"✅ Vector store ready — collection '{collection_name}' "
            f"({self.collection.count()} chunks, model: {embedding_model})"
        )

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_document(self, filename: str, chunks: List[Dict]) -> str:
        """
        Add document chunks to the vector store. Returns the new doc ID.

        Automatically detects chunks that contain image descriptions
        (produced by DocumentProcessor._describe_page_images) and marks
        them with has_images=True so they can be surfaced distinctly in
        source cards.
        """
        doc_id = str(uuid.uuid4())
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
                "has_images":  has_images,
                "has_tables":  has_tables,
            })
            texts.append(text)
            embeddings.append(self.embedding_model.encode(text).tolist())
            metadatas.append(meta)
            ids.append(f"{doc_id}_{i}")

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        image_chunks = sum(1 for m in metadatas if m.get("has_images"))
        table_chunks = sum(1 for m in metadatas if m.get("has_tables"))
        print(
            f"Added {len(chunks)} chunks for '{filename}' "
            f"({image_chunks} with image descriptions, {table_chunks} with tables)"
        )
        return doc_id

    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search — returns top_k most relevant chunks.

        Each result includes has_images and has_tables flags so the RAG
        engine and frontend can handle visually-enriched chunks distinctly
        (e.g. badge on source card, adjusted prompt phrasing).
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
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

    def delete_document(self, doc_id: str):
        """Delete all chunks belonging to a document ID."""
        results = self.collection.get(where={"document_id": doc_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"Deleted document {doc_id} ({len(results['ids'])} chunks)")
        else:
            print(f"No document found with ID {doc_id}")

    def list_documents(self) -> List[Dict]:
        """Return one entry per unique document in the store."""
        all_results = self.collection.get()
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

    def get_collection_stats(self) -> Dict:
        all_results = self.collection.get()
        image_chunks = sum(
            1 for m in all_results.get("metadatas", []) if m.get("has_images")
        )
        table_chunks = sum(
            1 for m in all_results.get("metadatas", []) if m.get("has_tables")
        )
        return {
            "total_chunks":    self.collection.count(),
            "total_documents": len(self.list_documents()),
            "image_chunks":    image_chunks,
            "table_chunks":    table_chunks,
        }


# ── tiny helper used by add_document ──────────────────────────────────────
import re