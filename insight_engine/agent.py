from dataclasses import dataclass, field
from typing import AsyncGenerator, NotRequired, TypedDict

from langchain_core.documents.base import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import CustomStreamEvent, StandardStreamEvent
from langchain_core.vectorstores import VectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic.v1 import BaseModel  # langchain will start using v2 in 0.3

from insight_engine.prompt import CHAT_TEMPLATE, AudienceType, UserType


class AnswerEvent(BaseModel):
    """Event representing an AI-generated message chunk of a final
    answer, that is to be visualized in the FE.

    Attributes:
        chunk (str): The content of the message chunk generated by the AI.
    """

    chunk: str


class RetrieveEvent(BaseModel):
    """Event representing the retrieval of document sources.

    Attributes:
        doc_sources (list[str]): A list of document sources retrieved by the agent.
    """

    doc_sources: list[str]


StreamEvent = AnswerEvent | RetrieveEvent


class AgentState(TypedDict):
    """Represents the state of the agent graph during its execution.

    Attributes:
        audience (AudienceType): The type of audience the output is for.
        chat_history (list[BaseMessage]): A list of chat messages that constitute
            the conversation history.
        user_type (UserType, optional): The type of user interacting with the agent.
            The agent should behave personalized for this type. TODO inference of this
        prompt (str): The last prompt provided by the user.
        retrieved_docs (list[Document], optional): A list of documents retrieved from
            the vector store based on the user's prompt.
        output (BaseMessage, optional): The response generated by the agent.
    """

    audience: AudienceType
    chat_history: list[BaseMessage]
    user_type: NotRequired[UserType]
    prompt: str
    retrieved_docs: NotRequired[list[Document]]
    output: NotRequired[BaseMessage]
    # db_filter: Any  # TODO to be determined how
    # output_structure: Any  # TODO


@dataclass
class Agent:
    """WIP - A conversational agent.

    Attributes:
        llm (BaseChatModel): The language model used by the agent to generate responses.
        vectorstore (VectorStore): The vector store used to retrieve relevant documents
            based on the user's prompt.
        k (int): The number of documents to retrieve from the vector store.
            Defaults to 4.
        graph (CompiledStateGraph): The compiled state graph that defines the
            agent's workflow.
    """

    llm: BaseChatModel
    vectorstore: VectorStore
    k: int = 4
    graph: CompiledStateGraph = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the state graph for the agent by adding nodes and edges."""
        builder = StateGraph(state_schema=AgentState)

        builder.add_node("retrieve", self.retrieve)
        builder.add_node("answer", self.answer)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "answer")

        self.graph: CompiledStateGraph = builder.compile()

    def retrieve(self, state: AgentState) -> dict:
        """Retrieves documents from the vector store based on the user's prompt.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: A dictionary containing the retrieved documents.
        """
        docs = self.vectorstore.similarity_search(query=state.get("prompt", ""), k=4)
        return {"retrieved_docs": docs}

    def answer(self, state: AgentState) -> dict:
        """Generates an answer based on the retrieved documents and the user's context.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            dict: A dictionary containing the generated output message.
        """
        context = "\n\n".join(
            doc.page_content for doc in state.get("retrieved_docs", [])
        )
        chain = CHAT_TEMPLATE | self.llm

        usertype = state.get("user_type")
        if usertype is None:
            usertype = UserType.BD
        audience: AudienceType = state["audience"]

        output = chain.invoke(
            input={
                "user_type_name": usertype.value.name,
                "user_type_description": usertype.value.description,
                "audience_name": audience.value.name,
                "audience_description": audience.value.description,
                "chat_history": state.get("chat_history", []),
                "prompt": state.get("prompt", ""),
                "context": context,
            }
        )

        return {"output": output}

    async def _run(
        self, input: AgentState, config: RunnableConfig | None = None
    ) -> AsyncGenerator[StandardStreamEvent | CustomStreamEvent, None]:
        """Runs the agent, yielding Langgraph events from the state graph.

        Args:
            inp (AgentState): The initial state input for the agent.
            config (RunnableConfig, optional): Optional configuration for the run.

        Yields:
            AsyncGenerator[StandardStreamEvent | CustomStreamEvent, Any]: Events
                generated during the agent's execution.
        """
        async for event in self.graph.astream_events(
            input=input, config=config, version="v2"
        ):
            yield event

    async def run(
        self, input: AgentState, config: RunnableConfig | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Runs the agent, processing Langgraph events from the state graph to
        our own events.

        Args:
            input (AgentState): The initial state input for the agent.
            config (RunnableConfig, optional): Optional configuration for the run.

        Yields:
            AsyncGenerator[StreamEvent, None]: Stream events representing events
                relevant to the FE.
        """
        async for event in self._run(input=input, config=config):
            if event["event"] == "on_chat_model_stream":
                if "chunk" in event["data"] and isinstance(
                    event["data"]["chunk"], AIMessageChunk
                ):
                    yield AnswerEvent(chunk=event["data"]["chunk"].content)

            elif event["event"] == "on_chain_stream" and event["name"] == "retrieve":
                if (
                    "chunk" in event["data"]
                    and "retrieved_docs" in event["data"]["chunk"]
                ):
                    retrieved_docs = event["data"]["chunk"]["retrieved_docs"]
                    doc_sources = [doc.metadata["source"] for doc in retrieved_docs]
                    yield RetrieveEvent(doc_sources=doc_sources)