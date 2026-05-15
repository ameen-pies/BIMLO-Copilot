def test_report_structure():
    """Verify report data structure."""
    report = {
        "id": "rpt-123",
        "title": "Network Analysis Report",
        "content": "# Section 1\n\nContent here.\n\n## Section 2\n\nMore content.",
        "source_docs": ["doc1.pdf", "doc2.pdf"],
    }
    assert "id" in report
    assert "title" in report
    assert "content" in report
    assert len(report["source_docs"]) == 2
    word_count = len(report["content"].split())
    assert word_count == 11
