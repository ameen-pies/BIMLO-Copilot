from services.vector_store import VectorStoreManager


def test_collection_isolation():
    """
    User A should not see documents from User B.
    Uses separate in-memory ChromaDB instances to verify isolation.
    """
    vs = VectorStoreManager(persist_directory=":memory:")

    chunks_a = [{"text": "secret A data", "metadata": {}}]
    chunks_b = [{"text": "secret B data", "metadata": {}}]

    vs.add_document("a.txt", chunks_a, user_id="userA", session_id="s1")
    vs.add_document("b.txt", chunks_b, user_id="userB", session_id="s2")

    results = vs.search("secret", user_id="userA", session_id="s1")
    assert len(results) == 1
    assert results[0]["text"] == "secret A data"

    results_b = vs.search("secret", user_id="userB", session_id="s2")
    assert len(results_b) == 1
    assert results_b[0]["text"] == "secret B data"


def test_collection_name_hashed():
    """Collection names should be within ChromaDB's 63-char limit."""
    vs = VectorStoreManager(persist_directory=":memory:")
    name = vs._get_collection_name("userlong" * 10, "session" * 5)
    assert len(name) <= 63
