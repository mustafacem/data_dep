import glob
import os
import tempfile

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import GPT2Tokenizer
from dotenv import load_dotenv

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
    vectorstore = get_vectorstore(collection_name="insight_engine_rag_Store")

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
    
 

    # Chain creation for multimodal RAG
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    whole_str = " ".join(text_summaries)

    # Ensure all three values are returned
    return retriever_multi_vector_img, chain_multimodal_rag, whole_str



import json
from langchain_openai import OpenAI

PROCESSED_PDFS_FILE = "processed_pdfs.json"

def load_processed_pdfs():
    """Load the list of processed PDFs from a JSON file."""
    if os.path.exists(PROCESSED_PDFS_FILE):
        with open(PROCESSED_PDFS_FILE, "r") as f:
            return json.load(f)
    return []

def save_processed_pdfs(processed_pdfs):
    """Save the list of processed PDFs to a JSON file."""
    with open(PROCESSED_PDFS_FILE, "w") as f:
        json.dump(processed_pdfs, f)

def build_the_db_multi(file_paths, output_path):
    # Initialize lists
    texts, tables = [], []
    pdf_names = []
    text_summaries_with_names = []
    table_summaries = []  # Initialize table_summaries to avoid UnboundLocalError

    # Load previously processed PDFs
    processed_pdfs = load_processed_pdfs()

    if not file_paths:
        print("No new PDFs to process. Accessing previously processed data...")
        if not processed_pdfs:
            raise ValueError("No previously processed PDFs available.")
        
        # Load previous data from the vectorstore
        vectorstore = get_vectorstore(collection_name="insight_engine_rag_Store")
        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            [], [], [], [], [], []
        )
        

        
        # Chain creation for multimodal RAG
        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
        whole_str = "Previously processed data accessed."
        
        # Return the retriever, chain, and a message indicating previous data is used
        return retriever_multi_vector_img, chain_multimodal_rag, whole_str

    for file_path in file_paths:
        pdf_name = os.path.basename(file_path)  # Extract the file name
        
        # Skip processing if the PDF was already processed
        if pdf_name in processed_pdfs:
            print(f"{pdf_name} has already been processed. Skipping.")
            continue

        print(file_path)

        # Open the file from the file path
        with open(file_path, 'rb') as temp_file:
            pdf_names.append(pdf_name)

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

            text_summaries, new_table_summaries = generate_text_summaries(
                texts_4k_token, file_tables, 1, summarize_texts=True
            )

            table_summaries.extend(new_table_summaries)

            # Prefix each text summary with the PDF name
            text_summaries_with_names.extend(
                [f"{pdf_name}: {summary}" for summary in text_summaries]
            )

    img_base64_list, image_summaries = generate_img_summaries("figures")
    delete_images_in_folder("figures")

    # Use vectorstore from insight_engine.vectordb
    vectorstore = get_vectorstore(collection_name="insight_engine_rag_Store")

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
    





    # Chain creation for multimodal RAG
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    whole_str = " ".join(text_summaries_with_names)

    # Add processed PDFs to the list and save
    processed_pdfs.extend(pdf_names)
    save_processed_pdfs(processed_pdfs)

    # Ensure all three values are returned
    return retriever_multi_vector_img, chain_multimodal_rag, whole_str