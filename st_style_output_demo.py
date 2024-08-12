import os
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from streamlit.delta_generator import DeltaGenerator
from insight_engine.prompt.knowledge import COCA_COLA
from insight_engine.prompt.system_prompts import (
    PRACTICE_GROUPS,
    REPORT_STRUCTURES,
    USER_QUERY,
    ALG
)
import tempfile
import matplotlib.pyplot as plt
import base64
import pdfminer.utils
from io import BytesIO
from PIL import Image
from insight_engine.pdf_extraction.pdf_extraction import (
    generate_img_summaries, 
    extract_pdf_elements, 
    categorize_elements,
    generate_text_summaries,
    topic_extraction, 
)
from insight_engine.rag_creation.rag_creation import (call_for_answer, create_multi_vector_retriever, multi_modal_rag_chain) 
from insight_engine.word_cloud.word_cloud import (   generate_word_cloud_2,     visualize_network,     create_keyword_network, extract_words ) 

from transformers import GPT2Tokenizer
# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

poppler_path= r"C:\Users\cem\Desktop\pooppker\poppler-24.02.0\Library\bin"

def truncate_to_token_limit(text, token_limit=3998 ):
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Truncate the tokens if they exceed the limit
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]
    
    # Decode the tokens back to text
    truncated_text = tokenizer.decode(tokens)
    
    return truncated_text


# Load environment variables
load_dotenv()
dotenv_path = 'op.env'
load_dotenv(dotenv_path)
openai_api = os.getenv("OPENAI_API_KEY")
if openai_api is None:
    raise ValueError("OpenAI key not specified!")
openai_api = 'sk-proj-rHddHmt0SxNlqOiiSL4VT3BlbkFJMiBrmdx7qmPvk6qqPhxN'                  

# Streamlit UI elements
st.title("Insight Report Generator")
practice_groups = [k for k in PRACTICE_GROUPS.keys()]
audiences = ["CFO", "CEO", "Chief Legal Counsel", "Chief of Tax", "Head of M&A"]
output_styles = [k for k in REPORT_STRUCTURES.keys()]

practice_group = st.selectbox("Select Practice Group:", practice_groups)
audience = st.selectbox("Select Audience:", audiences)
output_style = st.selectbox("Select Output Style:", output_styles)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator, initial_text: str = "") -> None:
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

# PDF File Uploader functionality
st.header("Upload a single PDF report")
uploaded_file = st.file_uploader("Upload PDF report here:", type=['pdf'], key='single_pdf')
if uploaded_file and not st.session_state.get("single_pdf_processed", False):
    st.success("PDF file uploaded successfully!")

    spesefic_topic = st.text_input("Enter spesefic topic if desired:")
    if st.button("Proceed with text processing"):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.markdown(f"Uploaded PDF file path: `{temp_file.name}`")

        input_path = os.getcwd()
        output_path = os.path.join(os.getcwd(), "output")
        raw_pdf_elements = extract_pdf_elements(temp_file.name, output_path, 4000, 500, 300)
        texts, tables = categorize_elements(raw_pdf_elements)

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
        joined_texts = " ".join(texts)
        texts_4k_token = text_splitter.split_text(joined_texts)

        text_summaries, table_summaries = generate_text_summaries(texts_4k_token, tables, 1, summarize_texts=True)
        img_base64_list, image_summaries = generate_img_summaries("figures")

        vectorstore = Chroma(
            collection_name="mm_rag_cj_blog", embedding_function=OpenAIEmbeddings()
        )

        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            text_summaries,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list,
        )
        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

        query = "Line graph displaying a fluctuating trend over fiscal years from 2004 to 2022"
        docs = retriever_multi_vector_img.get_relevant_documents(query, limit=6)

        whole_str = ' '.join(text_summaries)
        #network = network = create_keyword_network(truncate_to_token_limit(whole_str), topic_extraction(truncate_to_token_limit(whole_str)).split(","))
        #sorted_nodes = print_node_values_sorted(network)
        max_attempts = 25
        attempts = 0
        achieved = False
        with st.expander("Show Algorithm", expanded=False):
            st.write(ALG)
        while attempts < max_attempts and not achieved:
            try:
                if spesefic_topic == "" :
                    sized_down_whole_text =  truncate_to_token_limit(whole_str)
                    keywords= topic_extraction(sized_down_whole_text).split(",")
                    top_10 = topic_extraction(whole_str,"gpt-4-turbo",  topic_extraction(whole_str).split(","))
                else:
                    sized_down_whole_text =  truncate_to_token_limit(whole_str)
                    keywords= topic_extraction(sized_down_whole_text).split(",")
                    top_10 = topic_extraction(whole_str, "gpt-4-turbo", topic_extraction(whole_str, "gpt-4-turbo", spesefic_topic).split(","), spesefic_topic)                
                ani = extract_words(top_10)
                network = create_keyword_network( sized_down_whole_text, topic_extraction(whole_str).split("," ))
                plot_of_network = visualize_network(network)
                st.pyplot(plot_of_network)
                word_cloud = generate_word_cloud_2(network,ani)
                achieved = True
                
                st.pyplot(word_cloud)
                st.write(top_10)
                break  # If successful, break out of the loop
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    st.error(f"Failed to generate word cloud after {e} attempts.")
                else:
                    continue  # Try again if not yet reached max_attempts

        st.session_state["multiple_pdfs_processed"] = True
        st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
        st.session_state["chain_multimodal_rag"] = chain_multimodal_rag


# Handle multiple PDF uploads
st.header("Upload multiple PDF reports")
uploaded_files = st.file_uploader("Upload multiple PDF reports here:", type=['pdf'], accept_multiple_files=True, key='multiple_pdfs')
if uploaded_files and not st.session_state.get("multiple_pdfs_processed", False):
    st.success("PDF files uploaded successfully!")
    
    texts, tables = [], []
    pdf_names = []
    text_summaries_with_names = []
    spesefic_topic = st.text_input("Enter spesefic topic if desired:")
    if st.button("Proceed with text processing"):

        for uploaded_file in uploaded_files:
            print(uploaded_file)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            pdf_name = uploaded_file.name
            pdf_names.append(pdf_name)
            st.markdown(f"Uploaded PDF file path: `{temp_file.name}`, Name: `{pdf_name}`")

            input_path = os.getcwd()
            output_path = os.path.join(os.getcwd(), "output")
            raw_pdf_elements = extract_pdf_elements(temp_file.name, output_path, 4000, 500, 300)
            file_texts, file_tables = categorize_elements(raw_pdf_elements)
            texts.extend(file_texts)
            tables.extend(file_tables)

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
            joined_texts = " ".join(file_texts)
            texts_4k_token = text_splitter.split_text(joined_texts)

            text_summaries, table_summaries = generate_text_summaries(texts_4k_token, file_tables, 1, summarize_texts=True)
            
            # Prefix each text summary with the PDF name
            text_summaries_with_names.extend([f"{pdf_name}: {summary}" for summary in text_summaries])

        img_base64_list, image_summaries = generate_img_summaries("figures")

        vectorstore = Chroma(
            collection_name="mm_rag_cj_blog", embedding_function=OpenAIEmbeddings()
        )

        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            text_summaries_with_names,
            texts,
            table_summaries,
            tables,
            image_summaries,
            img_base64_list,
        )
        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

        whole_str = ' '.join(text_summaries_with_names)
        #network = create_keyword_network(truncate_to_token_limit(whole_str), topic_extraction(truncate_to_token_limit(whole_str)).split(","))
        #sorted_nodes = print_node_values_sorted(network)

        max_attempts = 25
        attempts = 0
        achieved = False
        with st.expander("Show Algorithm", expanded=False):
            st.write(ALG)
        while attempts < max_attempts and not achieved:
            try:
                if spesefic_topic == "" :
                    sized_down_whole_text =  truncate_to_token_limit(whole_str)
                    keywords= topic_extraction(sized_down_whole_text).split(",")
                    top_10 =    (whole_str,"gpt-4-turbo", topic_extraction(whole_str).split(","))
                else:
                    sized_down_whole_text =  truncate_to_token_limit(whole_str)
                    keywords= topic_extraction(sized_down_whole_text).split(",")
                    top_10 = topic_extraction(whole_str, "gpt-4-turbo", topic_extraction(whole_str,"gpt-4-turbo", spesefic_topic).split(","), spesefic_topic)
                print(2)
                
                ani = extract_words(top_10)
                network = create_keyword_network( sized_down_whole_text, topic_extraction(whole_str,"gpt-3.5-turbo-instruct").split("," ))
                plot_of_network = visualize_network(network)
                st.pyplot(plot_of_network)
                word_cloud = generate_word_cloud_2(network,ani)
                achieved = True
                
                st.pyplot(word_cloud)
                st.write(top_10)
                break  # If successful, break out of the loop
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    st.error(f"Failed to generate word cloud after {e} attempts.")
                else:
                    continue  # Try again if not yet reached max_attempts

        st.session_state["multiple_pdfs_processed"] = True
        st.session_state["retriever_multi_vector_img"] = retriever_multi_vector_img
        st.session_state["chain_multimodal_rag"] = chain_multimodal_rag



st.header("Enter a query to search the PDFs")
user_input = st.text_input("Enter some text:")
if user_input:
    print("user called for answer9ab")
    retriever_multi_vector_img = st.session_state.get("retriever_multi_vector_img")
    chain_multimodal_rag = st.session_state.get("chain_multimodal_rag")
    
    if retriever_multi_vector_img and chain_multimodal_rag:
        content, image = call_for_answer(user_input, retriever_multi_vector_img, chain_multimodal_rag)
        st.write("output:", content)
        if image:
            base64_image = image
            decoded_image = base64.b64decode(base64_image)
            image = Image.open(BytesIO(decoded_image))
            st.image(image, caption='Base64 Decoded Image')
    else:
        st.error("Please upload and process a PDF file first.")
