import base64
from io import BytesIO
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from streamlit.delta_generator import DeltaGenerator
from PIL import Image
from insight_engine.pdf_extraction.pdf_extraction import topic_extraction
from insight_engine.prompt.knowledge import COCA_COLA
from insight_engine.prompt.system_prompts import (
    ALG,
    PRACTICE_GROUPS,
    REPORT_STRUCTURES,
    USER_QUERY,
)
from insight_engine.rag_creation.rag_creation import call_for_answer
from insight_engine.vector_db_building.vector_db_building import (
    build_the_db,
    build_the_db_multi,
    truncate_to_token_limit,
)
from insight_engine.word_cloud.word_cloud import (
    create_keyword_network,
    extract_words,
    generate_word_cloud_2,
    visualize_network,
)

import nltk


# Call this function when loading


# nltk downloads if needed
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('popular')  # Downloads a collection of commonly used resources.

load_dotenv()

openai_api = os.getenv("OPENAI_API_KEY")
if openai_api is None:
    raise ValueError("OpenAI key not specified!")

# Streamlit UI elements
st.title("Insight Report Generator")
practice_groups = [k for k in PRACTICE_GROUPS.keys()]
audiences = ["CFO", "CEO", "Chief Legal Counsel", "Chief of Tax", "Head of M&A"]
output_styles = [k for k in REPORT_STRUCTURES.keys()]

practice_group = st.selectbox("Select Practice Group:", practice_groups)
audience = st.selectbox("Select Audience:", audiences)
output_style = st.selectbox("Select Output Style:", output_styles)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container:DeltaGenerator, initial_text: str = "") -> None:
        
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Generate Report button functionality
if st.button("Generate Report") and practice_group and output_style:
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
            streaming=True,
            callbacks=[stream_handler],
        )
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )

# Fixed path for the single PDF file
single_pdf_path = os.path.join(os.getcwd(), "path_of_single_pdf")  # Set the path for single PDF
output_path = os.path.join(os.getcwd(), "output")

# Process a single PDF with fixed path
st.header("Process Single PDF Report")
if st.button("Process Single PDF"):
    retriever_multi_vector_img, chain_multimodal_rag, whole_str = build_the_db(
        single_pdf_path, os.getcwd(), output_path
    )

    spesefic_topic = st.text_input("Enter specific topic if desired:")
    max_attempts = 1
    attempts = 0
    achieved = False

    with st.expander("Show Algorithm", expanded=False):
        st.write(ALG)

    while attempts < max_attempts and not achieved:
        try:
            if spesefic_topic == "":
                sized_down_whole_text = truncate_to_token_limit(whole_str)
                keywords = topic_extraction(
                    sized_down_whole_text, "gpt-4-turbo"
                ).split(",")
                top_10 = topic_extraction(
                    whole_str,
                    "gpt-4-turbo",
                    topic_extraction(whole_str, "gpt-4-turbo").split(","),
                )
            else:
                sized_down_whole_text = truncate_to_token_limit(whole_str)
                keywords = topic_extraction(
                    sized_down_whole_text, "gpt-4-turbo"
                ).split(",")
                top_10 = topic_extraction(
                    whole_str,
                    "gpt-4-turbo",
                    topic_extraction(
                        whole_str, "gpt-4-turbo", spesefic_topic
                    ).split(","),
                    spesefic_topic,
                )

            ani = topic_extraction(whole_str, "gpt-3.5-turbo-instruct", keywords)
            ani = extract_words(
                topic_extraction(ani, "gpt-3.5-turbo-instruct", [], "purefaction")
            )

            # Word Cloud and Network Graph in collapsible sections
            with st.expander("Show Word Cloud", expanded=False):
                word_cloud = generate_word_cloud_2(
                    create_keyword_network(sized_down_whole_text, keywords), ani
                )
                st.pyplot(word_cloud)

            with st.expander("Show Network Graph", expanded=False):
                network = create_keyword_network(sized_down_whole_text, keywords)
                plot_of_network = visualize_network(network)
                st.pyplot(plot_of_network)

            st.write(top_10)
            achieved = True
            break  # If successful, break out of the loop
        except Exception as e:
            attempts += 1
            if attempts == max_attempts:
                st.error(f"Failed to generate word cloud and network graph {e}")
            else:
                continue  # Try again if not yet reached max_attempts

    st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
    st.session_state["chain_multimodal_rag"] = chain_multimodal_rag

# Fixed paths for multiple PDFs
pdfs_directory = os.path.join(os.getcwd(), "directory_of_multi_pdfs")  # Specify directory with multiple PDFs
multiple_pdf_files = [os.path.join(pdfs_directory, pdf) for pdf in os.listdir(pdfs_directory) if pdf.endswith('.pdf')]

emp_directory = os.path.join(os.getcwd(), "empty")  # Specify directory with multiple PDFs
emp_files = [os.path.join(pdfs_directory, pdf) for pdf in os.listdir(pdfs_directory) if pdf.endswith('.pdf')]

# Process multiple PDFs with fixed paths
st.header("Process Multiple PDF Reports")
if st.button("Process Multiple PDFs"):
    retriever_multi_vector_img, chain_multimodal_rag, whole_str = build_the_db_multi(
        multiple_pdf_files, output_path
    )

    spesefic_topic = st.text_input("Enter specific topic if desired:")
    max_attempts = 1
    attempts = 0
    achieved = False

    with st.expander("Show Algorithm", expanded=False):
        st.write(ALG)

    while attempts < max_attempts and not achieved:
        try:
            if spesefic_topic == "":
                sized_down_whole_text = truncate_to_token_limit(whole_str)
                keywords = topic_extraction(
                    sized_down_whole_text, "gpt-4-turbo"
                ).split(",")
                top_10 = topic_extraction(whole_str, "gpt-4-turbo", keywords)
            else:
                sized_down_whole_text = truncate_to_token_limit(whole_str)
                keywords = topic_extraction(
                    sized_down_whole_text, "gpt-4-turbo"
                ).split(",")
                top_10 = topic_extraction(
                    whole_str,
                    "gpt-4-turbo",
                    topic_extraction(
                        whole_str, "gpt-4-turbo", spesefic_topic
                    ).split(","),
                    spesefic_topic,
                )

            ani = topic_extraction(whole_str, "gpt-3.5-turbo-instruct", keywords)
            ani = extract_words(
                topic_extraction(ani, "gpt-3.5-turbo-instruct", [], "purefaction")
            )

            # Word Cloud and Network Graph in collapsible sections
            with st.expander("Show Word Cloud", expanded=False):
                word_cloud = generate_word_cloud_2(
                    create_keyword_network(sized_down_whole_text, keywords), ani
                )
                st.pyplot(word_cloud)

            with st.expander("Show Network Graph", expanded=False):
                network = create_keyword_network(sized_down_whole_text, keywords)
                plot_of_network = visualize_network(network)
                st.pyplot(plot_of_network)

            st.write(top_10)
            achieved = True
            break  # If successful, break out of the loop
        except Exception as e:
            attempts += 1
            if attempts == max_attempts:
                st.error(f"Failed to generate word cloud and network graph {e}")
            else:
                continue  # Try again if not yet reached max_attempts

    st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
    st.session_state["chain_multimodal_rag"] = chain_multimodal_rag


st.header("Chat with your PDFs")

# Initialize or retrieve the session state for storing chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.chat_input("Enter your query:")

# If user input is provided, add it to chat history
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    retriever_multi_vector_img = st.session_state.get("retriever_multi_vector_img")
    chain_multimodal_rag = st.session_state.get("chain_multimodal_rag")

    if retriever_multi_vector_img and chain_multimodal_rag:
        # Call the function to get the content and image based on the query
        content, image = call_for_answer(
            user_input, retriever_multi_vector_img, chain_multimodal_rag
        )
        
        # Store the response in the session state as the assistant's message
        st.session_state["messages"].append({"role": "assistant", "content": content})

        # Display the chat history including the latest user and assistant messages
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
                
        # Display the image if available
        if image:
            base64_image = image
            decoded_image = base64.b64decode(base64_image)
            image = Image.open(BytesIO(decoded_image))
            st.image(image, caption="Base64 Decoded Image")
    else:
        retriever_multi_vector_img, chain_multimodal_rag, whole_str = build_the_db_multi(emp_files
        , output_path)
        st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
        st.session_state["chain_multimodal_rag"] = chain_multimodal_rag
        
        if retriever_multi_vector_img and chain_multimodal_rag:
            # Call the function to get the content and image based on the query
            content, image = call_for_answer(
                user_input, retriever_multi_vector_img, chain_multimodal_rag
            )
            
            # Store the response in the session state as the assistant's message
            st.session_state["messages"].append({"role": "assistant", "content": content})

            # Display the chat history including the latest user and assistant messages
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])
                    
            # Display the image if available
            if image:
                base64_image = image
                decoded_image = base64.b64decode(base64_image)
                image = Image.open(BytesIO(decoded_image))
                st.image(image, caption="Base64 Decoded Image")
else:
    # Display the chat history if no new input was given
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])  