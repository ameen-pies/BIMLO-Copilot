import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.add_document.return_value = None
    vs.search.return_value = []
    vs.list_documents.return_value = []
    vs.delete_document.return_value = None
    vs._get_collection.return_value = MagicMock()
    return vs


@pytest.fixture
def sample_chunks():
    return [
        {"text": "Network topology design for 5G small cells", "metadata": {}},
        {"text": "RAN optimization techniques include carrier aggregation", "metadata": {}},
        {"text": "Transport network requires at least 10 Gbps backhaul", "metadata": {}},
    ]
