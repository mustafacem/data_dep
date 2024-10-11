from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from insight_engine import DB_PATH


class Embedding:
    """Class that initializes the embedding function using OpenAI embeddings.

    Attributes:
        function (Embeddings): The embedding function using the specified OpenAI model.
    """

    function: Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )


def get_vectorstore(collection_name: str) -> Chroma:
    """Gets a vector store for the specified collection in Chroma DB. Note that
    it uses embedding function as specified in `insight_engine.vectordb.Embedding`

    Args:
        collection_name (str): The name of the collection for which the vector store is
            to be retrieved.

    Returns:
        Chroma: The Chroma vector store associated with the specified collection.

    """
    return Chroma(
        persist_directory=DB_PATH,
        collection_name=collection_name,
        embedding_function=Embedding.function,
    )
