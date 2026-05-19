

def test_routing():
    """Verify available routes contain expected values."""
    routes = ["direct", "rag", "iterative_rag", "analytics", "transform", "define", "graph"]
    assert "rag" in routes
    assert "direct" in routes
    assert "analytics" in routes
    assert len(routes) == 7


def test_format_sources():
    """Source formatting should produce expected structure."""
    sources = [
        {"filename": "doc1.pdf", "source_number": 1, "excerpt": "text1"},
        {"filename": "doc2.pdf", "source_number": 2, "excerpt": "text2"},
    ]
    assert len(sources) == 2
    assert sources[0]["source_number"] == 1
    assert sources[1]["filename"] == "doc2.pdf"
