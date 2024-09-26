from typing import Any, Self
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from insight_engine.agent import AnswerEvent, FinalEvent, RetrieveEvent
from insight_engine.api import root
from insight_engine.api.routers import (
    AudienceContext,
    MessageContext,
    PromptRequest,
    process_context_request,
)
from insight_engine.prompt import AUDIENCE_STORE

client = TestClient(root)
DUMMY_REQUEST = {
    "agent": "test-agent",
    "current_context": [
        {"msq_type": "message", "message": "Hi, how can I help you?"},
        {"msq_type": "audience", "audience": "CEO"},
        {"msq_type": "message", "message": "Hello! I want to know who is packing?"},
        {"msq_type": "audience", "audience": "CFO"},
        {"msq_type": "message", "message": "Of course, that must be Mr. N."},
        {"msq_type": "message", "message": "How do you know it?"},
    ],
}
EVENTS: list[Any] = [
    RetrieveEvent(doc_sources=["doc1", "doc2", "doc3", "doc4"]),
    AnswerEvent(chunk="Ola"),
    AnswerEvent(chunk=" konichiwa"),
    AnswerEvent(chunk=" tatakae!"),
    FinalEvent(
        doc_sources=["doc1", "doc2", "doc3", "doc4"], output="Ola konichiwa tatakae!"
    ),
]


class AsyncIterator:
    def __init__(self, items: Any) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def test_process_context_request() -> None:
    """Test the process_context_request function directly."""
    request = PromptRequest(
        agent="test-agent",
        current_context=[
            MessageContext(msq_type="message", message="Hi, how can I help you?"),
            AudienceContext(msq_type="audience", audience="CEO"),
            MessageContext(
                msq_type="message", message="Hello! I want to know who is packing?"
            ),
            AudienceContext(msq_type="audience", audience="CFO"),
            MessageContext(msq_type="message", message="Ofcourse that must be mr. N."),
            MessageContext(msq_type="message", message="How do you know it?"),
        ],
    )

    response = process_context_request(request)

    assert response["prompt"] == "How do you know it?"
    assert response["audience"] == AUDIENCE_STORE["CFO"]
    assert len(response["chat_history"]) == 4


def test_answer_prompt() -> None:
    """Test the /prompt/ endpoint."""
    request_data = DUMMY_REQUEST

    response = client.post("/prompt/", json=request_data)
    assert response.status_code == 200
    assert "run_id" in response.json()


def mock_run(self: Any, input: Any, config: Any) -> AsyncIterator:
    return AsyncIterator(EVENTS)


@pytest.mark.asyncio
async def test_stream_response() -> None:
    """Test the /stream/{run_id} endpoint."""
    request_data = DUMMY_REQUEST

    prompt_response = client.post("/prompt/", json=request_data)
    run_id = prompt_response.json()["run_id"]

    with patch("insight_engine.agent.Agent.run", mock_run):
        stream_response = client.get(f"/stream/{run_id}")

    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"].startswith("text/event-stream")

    streamed_lines = [line for line in stream_response.iter_lines()]

    for i in range(1, 4):
        assert EVENTS[i].chunk in streamed_lines[i * 3 + 1]

    final_idx = 4
    assert EVENTS[final_idx].output in streamed_lines[final_idx * 3 + 1]
