import base64
import os
import re

import fitz
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image
import numpy as np

# from pdf2image import *
from unstructured.partition.pdf import partition_pdf

from insight_engine.prompt.system_prompts import PROMPT_TEXT

load_dotenv()

openai_api = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api)


def extract_pdf_elements(
    path_of_pdf,
    output_path,
    max_characters,
    new_after_n_chars,
    combine_text_under_n_chars,
):
    """
    Extract images, tables, and chunk text from a PDF file.

    path_of_pdf: File path to the PDF file.
    output_path: Directory path where images and other outputs will be saved.
    max_characters: Maximum number of characters in a chunk of text.
    new_after_n_chars: Number of characters after which a new chunk of text should be created.
    combine_text_under_n_chars: Combine text chunks smaller than this number into one chunk.

    Returns: List of Document objects extracted from the PDF.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    documents = partition_pdf(
        filename=path_of_pdf,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
        image_output_dir_path=output_path,
    )

    return documents


def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


def generate_text_summaries(texts, tables, max_concurrency, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEXT)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # summarize_chain = {"element": lambda x: x} | prompt | ChatOpenAI() | StrOutputParser() #

    text_summaries = []
    table_summaries = []
    # Be aware increasing of max concurrency can resault in failure also amount of credits spent decreases that probbility
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(
            texts, {"max_concurrency": max_concurrency}
        )
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(
            tables, {"max_concurrency": max_concurrency}
        )

    return text_summaries, table_summaries


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def is_single_color(image_path):
    """
    Check if the image is a single-color image.
    
    image_path: Path to the image file.
    
    Returns: True if the image is a single-color image, False otherwise.
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Check if all pixel values in the image are the same
    return np.all(img_array == img_array[0, 0])


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images, only if the image is not a single-color image.
    path: Path to the list of .jpg files extracted by Unstructured.
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            
            # Check if the image is not a single-color image
            if not is_single_color(img_path):
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries

def find_fig_text(text):
    pattern = r"Fig\..*?\."
    fig_sentences = re.findall(pattern, text, flags=re.DOTALL)
    return " ".join(fig_sentences) if fig_sentences else ""


def image_def(img_base64, prompt):
    return prompt


def generate_img_summaries_rule(pdf_path, images_dir="extracted_images_main_0"):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    doc = fitz.open(pdf_path)

    img_base64_list = []
    image_summaries = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            fig_text = find_fig_text(text)
            fig_text = fig_text.replace("/", " or ")
            if fig_text:
                print(
                    f"Text related to images on Page {page_num + 1}, Image {image_index}:\n{fig_text}\n"
                )

            image_filename = f"{fig_text}.png"
            image_path = os.path.join(images_dir, image_filename)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            base64_image = encode_image(image_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_def(base64_image, fig_text))
            print(img_base64_list, image_summaries)

    return img_base64_list, image_summaries


def topic_extraction(text_to_analyze, model, word_list=[], topic=""):
    if model == "gpt-3.5-turbo-instruct":
        if topic == "":
            if word_list == []:
                completion = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=f"Extract distinct keywords  from: {text_to_analyze}  \n\nTopics:",
                    max_tokens=600,
                    temperature=0,
                )
                return completion.choices[0].text.strip()
            else:
                completion = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=f"choose top 10  keywords from {word_list} most relevant to text :  {text_to_analyze}   \n\n Then itemeze and explain why they are important :",
                    max_tokens=600,
                    temperature=0,
                )
                return completion.choices[0].text.strip()
        elif topic == "purefaction":
            completion = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=f"you are given list of keywords from  given text : {text_to_analyze} return just keywords   \n\n keywords:",
                max_tokens=600,
                temperature=0,
            )
            return completion.choices[0].text.strip()
    elif model == "gpt-4-turbo":
        if topic == "":
            if word_list == []:
                # gpt4o doesnt work as good as this
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that helps with extracting keywords.",
                        },
                        {
                            "role": "user",
                            "content": f"Extract distinct keywords  from: {text_to_analyze}  \n\nTopics:",
                        },
                    ],
                )
                return completion.choices[0].message.content
            else:
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that helps with extracting keywords.",
                        },
                        {
                            "role": "user",
                            "content": f"choose top 10 keywords from {word_list} most relevant to text: {text_to_analyze}\n\n Then itemize and explain why they are important to text:",
                        },
                    ],
                )
                return completion.choices[0].message.content
        else:
            if word_list == []:
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that helps with extracting keywords.",
                        },
                        {
                            "role": "user",
                            "content": f"Extract distinct keywords with focus on {topic} realted ones  from: {text_to_analyze}  \n\nTopics:",
                        },
                    ],
                )
                return completion.choices[0].message.content
            else:
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant that helps with extracting keywords.",
                        },
                        {
                            "role": "user",
                            "content": f"choose top 10 keywords with focus on {topic} realted ones  from {word_list} most relevant to text: {text_to_analyze}\n\n Then itemize and explain why they are important to text:",
                        },
                    ],
                )
                return completion.choices[0].message.content
