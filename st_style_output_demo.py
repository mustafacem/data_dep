import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI

from insight_engine.prompt.knowledge import COCA_COLA
from insight_engine.prompt.system_prompts import (
    PRACTICE_GROUPS,
    REPORT_STRUCTURES,
    USER_QUERY,
)

load_dotenv()
openai_api = os.getenv("OPENAI_API_KEY")

practice_groups = [k for k in PRACTICE_GROUPS.keys()]
audiences = ["CFO", "CEO", "Chief Legal Counsel", "Chief of Tax", "Head of M&A"]
output_styles = [k for k in REPORT_STRUCTURES.keys()]

st.title("Insight Report Generator")
practice_group = st.selectbox("Select Practice Group:", practice_groups)
audience = st.selectbox("Select Audience:", audiences)
output_style = st.selectbox("Select Output Style:", output_styles)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


if st.button("Generate Report"):
    sys_prompt = PRACTICE_GROUPS[practice_group]
    report_structure = REPORT_STRUCTURES[output_style]

    st.session_state["messages"] = [
        ChatMessage(
            role="system",
            content=sys_prompt.format(
                audience=audience, report_structure=report_structure
            ),
        ),
        ChatMessage(role="user", content=USER_QUERY.format(data=COCA_COLA)),
    ]
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            api_key=openai_api,
            streaming=True,
            callbacks=[stream_handler],
        )
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )
