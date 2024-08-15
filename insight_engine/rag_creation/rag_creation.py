import base64
import io
import re
import uuid
from PIL import Image
from collections import defaultdict
import openai
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

def looks_like_base64(sb):
    if isinstance(sb, bytes):
        sb = sb.decode('utf-8')  # Decode bytes to string
    return re.match(r"^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    resized_img = img.resize(size, Image.LANCZOS)

    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    #formatted_texts = "\n".join(data_dict["context"]["texts"])
    formatted_texts = "\n".join(
    text.decode('utf-8') if isinstance(text, bytes) else str(text)
    for text in data_dict["context"]["texts"])
    messages = []

    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    text_message = {
        "type": "text",
        "text": (
            "You are an analyst tasked with answering questions.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers related to the user question.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and/or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    store =  LocalFileStore("insight_engine/rag_creation/data_depo")#("insight_engine\rag_creation\data_depo")

    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        
        # Convert content to bytes if it's a string before saving to the store
        byte_content = [
            content.encode('utf-8') if isinstance(content, str) else content
            for content in doc_contents
        ]
        retriever.docstore.mset(list(zip(doc_ids, byte_content)))

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

def call_for_answer(query, retriever_multi_vector_img, chain_multimodal_rag):
    docs = retriever_multi_vector_img.get_relevant_documents(query, limit=7)
    img  = ""
    con =  ""
    for x in docs:
        if " " in str(x):
            pass # this section is for llms that run locally 
        else:
            if img ==  "":
                img = x
    
    con = chain_multimodal_rag.invoke(query )
    if img == "":
        print("No image found")
    if con == "":
        print("No content found")

    return con, img
