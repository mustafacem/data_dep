import glob
import os
import tempfile

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import GPT2Tokenizer

from insight_engine.pdf_extraction.pdf_extraction import (
    categorize_elements,
    extract_pdf_elements,
    generate_img_summaries,
    generate_text_summaries,
)
from insight_engine.rag_creation.rag_creation import (
    create_multi_vector_retriever,
    multi_modal_rag_chain,
)
from insight_engine.vectordb import get_vectorstore
from insight_engine.agent import Agent
# Remove the promptstore import for now
# from insight_engine.prompt.store import promptstore  # This caused the issue

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def truncate_to_token_limit(text, token_limit=3998):
    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Truncate the tokens if they exceed the limit
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]

    # Decode the tokens back to text
    truncated_text = tokenizer.decode(tokens)

    return truncated_text


def delete_images_in_folder(folder_path):
    # List of image file extensions to consider
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]

    # Iterate over the list of image extensions
    for extension in image_extensions:
        # Construct the search pattern
        search_pattern = os.path.join(folder_path, extension)

        # Get a list of files matching the search pattern
        images = glob.glob(search_pattern)

        # Iterate over the list of images and delete them
        for image in images:
            try:
                os.remove(image)
            except Exception as e:
                print(f"ERROR {image}: {e}")


def return_content(path_of_pdf, output_path):
    raw_pdf_elements = extract_pdf_elements(path_of_pdf, output_path, 4000, 500, 300)
    texts, tables = categorize_elements(raw_pdf_elements)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)

    text_summaries, table_summaries = generate_text_summaries(
        texts_4k_token, tables, 1, summarize_texts=True
    )
    img_base64_list, image_summaries = generate_img_summaries("figures")
    delete_images_in_folder("figures")
    return (
        text_summaries,
        table_summaries,
        img_base64_list,
        image_summaries,
        texts,
        tables,
    )


def build_the_db(path_of_pdf, input_path, output_path):
    text_summaries, table_summaries, img_base64_list, image_summaries, texts, tables = (
        return_content(path_of_pdf, output_path)
    )
    delete_images_in_folder("figures")

    # Use vectorstore from insight_engine.vectordb
    vectorstore = get_vectorstore(collection_name="mm_rag_cj_blog")

    # Create the multi-vector retriever for multimodal RAG
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )
    
    # Use modified agent for retrieval
    agent = Agent(llm=None, vectorstore=retriever_multi_vector_img)  # LLM can be passed as needed

    # Mocking the prompt response
    agent.retrieve = lambda state: {"retrieved_docs": retriever_multi_vector_img.search(state['prompt'], k=4)}
    agent.answer = lambda state: {
        "output": f"Generated response for prompt: {state['prompt']}, context: {', '.join([doc.page_content for doc in state['retrieved_docs']])}"
    }

    # Chain creation for multimodal RAG
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    whole_str = " ".join(text_summaries)

    # Ensure all three values are returned
    return retriever_multi_vector_img, chain_multimodal_rag, whole_str



def build_the_db_multi(file_paths, output_path):
    texts, tables = [], []
    pdf_names = []
    text_summaries_with_names = []

    for file_path in file_paths:
        print(file_path)

        # Open the file from the file path
        with open(file_path, 'rb') as temp_file:
            pdf_name = os.path.basename(file_path)  # Extract the file name
            pdf_names.append(pdf_name)

            # input_path = os.getcwd()
            output_path = os.path.join(os.getcwd(), "output")
            raw_pdf_elements = extract_pdf_elements(
                file_path, output_path, 4000, 500, 300
            )
            file_texts, file_tables = categorize_elements(raw_pdf_elements)
            texts.extend(file_texts)
            tables.extend(file_tables)

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=4000, chunk_overlap=0
            )
            joined_texts = " ".join(file_texts)
            texts_4k_token = text_splitter.split_text(joined_texts)

            text_summaries, table_summaries = generate_text_summaries(
                texts_4k_token, file_tables, 1, summarize_texts=True
            )

            # Prefix each text summary with the PDF name
            text_summaries_with_names.extend(
                [f"{pdf_name}: {summary}" for summary in text_summaries]
            )

    img_base64_list, image_summaries = generate_img_summaries("figures")
    delete_images_in_folder("figures")

    # Use vectorstore from insight_engine.vectordb
    vectorstore = get_vectorstore(collection_name="mm_rag_cj_blog")

    # Create the multi-vector retriever for multimodal RAG
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries_with_names,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )
    
    # Use modified agent for retrieval
    agent = Agent(llm=None, vectorstore=retriever_multi_vector_img)  # LLM can be passed as needed

    # Mocking the prompt response
    agent.retrieve = lambda state: {"retrieved_docs": retriever_multi_vector_img.search(state['prompt'], k=4)}
    agent.answer = lambda state: {
        "output": f"Generated response for prompt: {state['prompt']}, context: {', '.join([doc.page_content for doc in state['retrieved_docs']])}"
    }

    # Chain creation for multimodal RAG
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    whole_str = " ".join(text_summaries_with_names)

    # Ensure all three values are returned
    return retriever_multi_vector_img, chain_multimodal_rag, whole_str
