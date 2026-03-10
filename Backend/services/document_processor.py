import os
import re
import requests
from typing import List, Dict


class DocumentProcessor:
    """
    Generic document processor — works with any domain, any language.
    Doc-type classification is LLM-driven (no hardcoded keyword lists).
    Falls back to file-extension heuristics when LLM is unavailable.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._api_key = os.getenv("CF_API_KEY", "")
        self._api_url = os.getenv("CF_API_URL", "https://bimloapi.medhelaliamin125.workers.dev")

    # ── Public entry point ────────────────────────────────────────────────

    def process_document(self, file_path: str) -> List[Dict]:
        """Process a document and return text chunks with metadata."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            text = self._extract_pdf(file_path)
        elif ext in ('.docx', '.doc'):
            text = self._extract_docx(file_path)
        elif ext == '.txt':
            text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        metadata = self._extract_metadata(text, file_path)
        return self._create_chunks(text, metadata)

    # ── Text extraction ───────────────────────────────────────────────────

    def _extract_pdf(self, file_path: str) -> str:
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""

    def _extract_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return ""

    def _extract_txt(self, file_path: str) -> str:
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error extracting TXT: {e}")
                return ""
        return ""

    # ── Metadata extraction ───────────────────────────────────────────────

    def _extract_metadata(self, text: str, file_path: str) -> Dict:
        metadata = {
            "filename": os.path.basename(file_path),
            "doc_type": self._classify_doc_type(text, file_path),
        }
        ref = self._extract_reference(text)
        if ref:
            metadata["project_ref"] = ref
        return metadata

    def _classify_doc_type(self, text: str, file_path: str) -> str:
        """
        Ask the LLM to classify the document type from a short excerpt.
        Returns a short snake_case label (e.g. 'specification', 'report',
        'invoice', 'manual', 'contract').  Falls back to 'document' if
        the LLM is unavailable or the call fails.
        """
        if not self._api_key:
            return "document"

        snippet = text[:800].strip()
        filename = os.path.basename(file_path)

        prompt = f"""Classify this document into a short doc_type label (snake_case, max 3 words).
Examples: specification, technical_report, invoice, contract, user_manual, meeting_notes, datasheet, plan, calculation_note, legal_document

Filename: {filename}
Content preview:
{snippet}

Reply with ONLY the doc_type label. Nothing else."""

        try:
            resp = requests.post(
                self._api_url,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={
                    "prompt":     prompt,
                    "max_tokens": 20,
                    "task":       "classify",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                label = (resp.json().get("response") or "").strip().lower()
                # Sanitise: keep only word chars and underscores
                label = re.sub(r'[^\w]', '_', label).strip('_')
                return label or "document"
        except Exception as e:
            print(f"⚠️  Doc-type classification failed: {e}")

        return "document"

    def _extract_reference(self, text: str) -> str:
        """
        Extract a project/document reference number from the text using
        common patterns (ref:, reference:, #, document no., etc.).
        Language-agnostic — matches the structure, not the language.
        """
        patterns = [
            r'(?:ref(?:erence)?|réf(?:érence)?|dokument(?:nummer)?|n[°o]\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_/\.]{2,})',
            r'#\s*([A-Z0-9][A-Z0-9\-_/\.]{2,})',
            r'\b([A-Z]{2,6}[-_]\d{3,}(?:[-_][A-Z0-9]+)*)\b',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return ""

    # ── Chunking ──────────────────────────────────────────────────────────

    def _create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = []
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return chunks

        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Prefer breaking at sentence boundary
                boundary = text.rfind('.', end, min(end + 100, len(text)))
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "metadata": metadata.copy(),
                })
                chunk_id += 1

            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks