import os

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class Embedding:
    """Class that initializes the embedding function using OpenAI embeddings.

    Attributes:
        function (Embeddings): The embedding function using the specified OpenAI model.
    """

    function: Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )


class ChromaDBConnector:
    """Singleton Chroma DB connector.

    This class represents a singleton Chroma DB connector. It provides a method
    to get the singleton instance of the connector.

    Attributes:
        _client: The singleton instance of the Chroma DB connector.

    Methods:
        get_client: Get the singleton instance of the connector.

    """

    _client: ClientAPI | None = None

    @staticmethod
    def get_client() -> ClientAPI:
        """Gets the singleton instance of the Chroma DB client.

        If the client does not exist, it initializes the client.

        Returns:
            ClientAPI: The singleton instance of the Chroma DB client.

        """
        if ChromaDBConnector._client is None:
            ChromaDBConnector._client = chromadb.HttpClient(
                host=os.getenv("CHROMADB_HOST", "chromadb"),
                settings=Settings(allow_reset=True),
            )
        return ChromaDBConnector._client


def get_vectorstore(collection_name: str) -> VectorStore:
    """Gets a vector store for the specified collection in Chroma DB. Note that
    it uses embedding function as specified in `insight_engine.vectordb.Embedding`

    Args:
        collection_name (str): The name of the collection for which the vector store is
            to be retrieved.

    Returns:
        VectorStore: The vector store associated with the specified collection.

    """
    return Chroma(
        client=ChromaDBConnector.get_client(),
        collection_name=collection_name,
        embedding_function=Embedding.function,
    )
