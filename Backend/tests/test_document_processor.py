import pytest
from services.document_processor import DocumentProcessor


def test_txt_processing(tmp_path):
    doc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    f = tmp_path / "test.txt"
    f.write_text("This is a test document for telecom network documentation. " * 10)
    chunks = doc.process_document(str(f))
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)


def test_empty_text_returns_empty():
    doc = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    chunks = doc._create_chunks("   ", {"source": "test"})
    assert chunks == []
