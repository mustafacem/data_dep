from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from chromadb import EphemeralClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from insight_engine.vectordb import ChromaDBConnector, Embedding, get_vectorstore

TEST_COLLETION_NAME = "dummy_collection"


@pytest.fixture
def mock_embedding() -> Generator[None, Any, None]:
    """Fixture to mock the embedding function."""
    with patch.object(
        target=Embedding, attribute="function", new=FakeEmbeddings(size=69)
    ):
        yield


@pytest.fixture
def mock_chroma_client() -> Generator[MagicMock | AsyncMock, Any, None]:
    """Fixture to mock the ChromaDB client."""
    with patch.object(ChromaDBConnector, "_client", new=None):
        with patch(
            "chromadb.HttpClient", return_value=EphemeralClient()
        ) as mock_client:
            yield mock_client


def test_embedding_initialization() -> None:
    """Test that the Embedding class initializes with the correct function."""
    assert isinstance(
        Embedding.function, Embeddings
    ), "Embedding function is not initialized correctly."


def test_chromadb_connector_singleton(
    mock_chroma_client: MagicMock | AsyncMock,
) -> None:
    """Test that ChromaDBConnector returns a singleton instance."""
    client1 = ChromaDBConnector.get_client()
    client2 = ChromaDBConnector.get_client()

    assert client1 is client2, "ChromaDBConnector did not return a singleton instance."
    mock_chroma_client.assert_called_once()


def test_get_vectorstore(
    mock_embedding: Any, mock_chroma_client: MagicMock | AsyncMock
) -> None:
    """Test that get_vectorstore returns a VectorStore with correct parameters."""
    get_vectorstore(TEST_COLLETION_NAME)
    mock_chroma_client.assert_called_once()


def test_add_documents_to_vectorstore(
    mock_embedding: Any, mock_chroma_client: MagicMock | AsyncMock
) -> None:
    """Test adding documents to the vector store."""
    vector_store = get_vectorstore(TEST_COLLETION_NAME)
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast "
        "this morning.",
        metadata={"source": "tweet"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with "
        "a high of 62 degrees.",
        metadata={"source": "news"},
        id=2,
    )

    documents = [document_1, document_2]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    all_documents = (
        ChromaDBConnector.get_client()
        .get_collection(TEST_COLLETION_NAME)
        .get()["documents"]
    )
    assert all_documents
    assert (
        len(all_documents) == 2
    ), "Documents were not added correctly to the vector store."


def test_similarity_search(
    mock_embedding: Any, mock_chroma_client: MagicMock | AsyncMock
) -> None:
    """Test similarity search in the vector store."""
    vector_store = get_vectorstore(TEST_COLLETION_NAME)
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast "
        "this morning.",
        metadata={"source": "tweet"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a "
        "high of 62 degrees.",
        metadata={"source": "news"},
        id=2,
    )

    documents = [document_1, document_2]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    results = vector_store.similarity_search(
        query="cereals",
        k=4,
        filter={"source": "tweet"},
    )

    assert len(results) == 2


def test_contain_embbeds(
    mock_embedding: Any, mock_chroma_client: MagicMock | AsyncMock
) -> None:
    """Test similarity search in the vector store."""
    vector_store = get_vectorstore(TEST_COLLETION_NAME)

    documents = []
    n = 1000
    for i in range(n):
        doc = Document(
            page_content=f"I am {i} years old.",
            metadata={"source": str(i)},
        )
        documents.append(doc)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    for i in range(n):
        filter_q = {"source": str(i)}
        docs = vector_store.similarity_search(query="", k=4, filter=filter_q)
        assert len(docs) > 0
