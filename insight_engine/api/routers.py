import logging
import os
import uuid
from typing import Any, AsyncIterator, Literal, TypedDict

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from kd_logging import setup_logger
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, HttpUrl

from insight_engine.agent import Agent, AgentState
from insight_engine.prompt import AUDIENCE_STORE
from insight_engine.vectordb import get_vectorstore


class Task(TypedDict):
    input: AgentState
    config: Any


load_dotenv()
root = APIRouter()
prompt_tasks_store: dict[str, Task] = {}  # TODO: make serverless
agent = Agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=get_vectorstore(os.environ["DB_COLLECTION"]),
)
logger = setup_logger(__name__, level=logging.DEBUG)


class AudienceContext(BaseModel):
    msq_type: Literal["audience"]
    audience: str


class MessageContext(BaseModel):
    """One message entry. AI or human."""

    msq_type: Literal["message"]
    message: str


class AttachmentContext(BaseModel):
    msq_type: Literal["attachment"]
    attachment: HttpUrl


ContextItem = AudienceContext | MessageContext | AttachmentContext


class PromptRequest(BaseModel):
    """Model representing a API request for llm generation.

    Attributes:
        agent (str): The agent identifier.
            #TODO in future from this identifier will be possible to infer what
            # KBs to use.
        current_context (list[ContextItem]): The current context items in the prompt.
            It is a list of messages, changes of audience. The first message is assumed
            to be the welcome AI message, and afterwads it is alternation of human and
            AI messages. Every time the user changes history the new value is appended
            into the context.
            #TODO in future also context upload attachments urls will be appended
    """

    agent: str
    current_context: list[ContextItem]


class PromptAnswer(BaseModel):
    run_id: str


def process_context_request(req: PromptRequest) -> AgentState:
    """Processes a context from the request and converts it into an AgentState.

    Args:
        req (PromptRequest): The request containing the context.

    Returns:
        AgentState: The starting state of the agent based on the processed context.

    Raises:
        ValueError: If no audience is specified or if an invalid audience type is
            provided.
    """
    audiences = [m for m in req.current_context if m.msq_type == "audience"]
    if not audiences:
        raise ValueError("There is missing audience specification in the context")

    try:
        audience = AUDIENCE_STORE[audiences[-1].audience]
    except KeyError:
        raise ValueError(
            f"Invalid audience type specified in the context: {audiences[-1].audience}"
        )

    # TODO: attachments

    history = InMemoryChatMessageHistory()
    messages = [m for m in req.current_context if m.msq_type == "message"]
    for i, message in enumerate(messages):
        if i % 2 == 1:
            history.add_user_message(message.message)
        else:
            history.add_ai_message(message.message)

    return {
        "prompt": messages[-1].message,
        "audience": audience,
        "chat_history": history.messages,
    }


@root.post("/prompt/")
async def answer_prompt(req: PromptRequest) -> PromptAnswer:
    """Handles the HTTP POST request to answer a prompt.

    Args:
        req (PromptRequest): The request containing the prompt details.

    Returns:
        PromptAnswer: The answer to the prompt containing the run ID.
    """
    run_id = str(uuid.uuid4())
    input = process_context_request(req=req)
    logger.debug(f"Processed context from PromptRequest:\n{input}")

    config = {"run_id": run_id}
    prompt_tasks_store[run_id] = {
        "input": input,
        "config": config,
    }

    return PromptAnswer(run_id=run_id)


async def stream_events(
    input: AgentState, config: RunnableConfig | None = None
) -> AsyncIterator[str]:
    """Streams events generated by the agent.

    Args:
        input (AgentState): The state of the agent.
        config (RunnableConfig | None, optional): Configuration for the Runnable.

    Yields:
        AsyncIterator[str]: A string stream of JSON-encoded events. One line is
            one event. The events are delimited by newlines.

    #TODO Maybe problem when the newlines are inside a JSON string value. This will
    # be harder to parse for the API user.
    """
    async for event in agent.run(input=input, config=config):
        yield event.json() + "\n"  # TODO server sent events


@root.get("/stream/{run_id}")
async def stream_response(run_id: str) -> StreamingResponse:
    """Handles the HTTP GET request to stream the response for a specific run ID.

    Args:
        run_id (str): The unique identifier for the run.

    Returns:
        StreamingResponse: A streaming response containing the events.
    """
    input = prompt_tasks_store[run_id]["input"]
    config = prompt_tasks_store[run_id]["config"]
    return StreamingResponse(
        content=stream_events(input=input, config=config),
        media_type="text/event-stream",
    )
