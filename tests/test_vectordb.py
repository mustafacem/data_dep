from unittest.mock import patch
from uuid import uuid4

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from insight_engine.vectordb import Embedding, get_vectorstore

TEST_COLLETION_NAME = "dummy_collection"


def mocked_get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        client_settings=Settings(persist_directory=""),
        collection_name=collection_name,
        embedding_function=FakeEmbeddings(size=69),
    )


Chroma.from_documents


def test_embedding_initialization() -> None:
    """Test that the Embedding class initializes with the correct function."""
    assert isinstance(
        Embedding.function, Embeddings
    ), "Embedding function is not initialized correctly."


@patch("tests.test_vectordb.get_vectorstore", side_effect=mocked_get_vectorstore)
def test_get_vectorstore(mock_get_vectorstore) -> None:
    """Test that get_vectorstore works."""
    get_vectorstore(TEST_COLLETION_NAME)


@patch("tests.test_vectordb.get_vectorstore", side_effect=mocked_get_vectorstore)
def test_add_documents_to_vectorstore(mock_get_vectorstore) -> None:
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

    all_documents = vector_store.get()["documents"]
    assert all_documents
    assert (
        len(all_documents) == 2
    ), "Documents were not added correctly to the vector store."


@patch("tests.test_vectordb.get_vectorstore", side_effect=mocked_get_vectorstore)
def test_similarity_search(mock_get_vectorstore) -> None:
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


@patch("tests.test_vectordb.get_vectorstore", side_effect=mocked_get_vectorstore)
def test_contain_embbeds(mock_get_vectorstore) -> None:
    """Test similarity search in the vector store."""
    vector_store = get_vectorstore(TEST_COLLETION_NAME)

    documents = []
    n = 100
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
