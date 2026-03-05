import os
import uuid
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from datetime import datetime


# Defaults — all overridable via environment variables or constructor args
_DEFAULT_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_DEFAULT_COLLECTION   = os.getenv("CHROMA_COLLECTION",  "documents")
_DEFAULT_EMBED_MODEL  = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")


class VectorStoreManager:
    """
    Generic vector store — domain-agnostic, fully configurable via env vars
    or constructor arguments.

    Env vars (all optional, sensible defaults provided):
      CHROMA_PERSIST_DIR  — where ChromaDB stores data  (default: ./data/chroma)
      CHROMA_COLLECTION   — collection name              (default: documents)
      EMBEDDING_MODEL     — SentenceTransformer model   (default: all-MiniLM-L6-v2)
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
        print(f"✅ Vector store ready — collection '{collection_name}' "
              f"({self.collection.count()} chunks, model: {embedding_model})")

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_document(self, filename: str, chunks: List[Dict]) -> str:
        """Add document chunks to the vector store. Returns the new doc ID."""
        doc_id = str(uuid.uuid4())
        texts, embeddings, metadatas, ids = [], [], [], []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            meta = chunk.get("metadata", {}).copy()
            meta.update({
                "document_id": doc_id,
                "chunk_index": i,
                "timestamp":   datetime.now().isoformat(),
                "filename":    filename,
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
        print(f"Added {len(chunks)} chunks for '{filename}'")
        return doc_id

    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Semantic search — returns top_k most relevant chunks."""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
        )
        formatted = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text":     results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "id":       results["ids"][0][i],
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
        docs = {}
        for meta in all_results.get("metadatas", []):
            doc_id = meta.get("document_id")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "filename":    meta.get("filename", "unknown"),
                    "doc_type":    meta.get("doc_type", "unknown"),
                    "timestamp":   meta.get("timestamp", "unknown"),
                }
        return list(docs.values())

    def get_collection_stats(self) -> Dict:
        return {
            "total_chunks":    self.collection.count(),
            "total_documents": len(self.list_documents()),
        }