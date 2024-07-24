import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import networkx as nx
from collections import defaultdict
import re
from openai import OpenAI

import os 
from pdfminer.utils import open_filename
from unstructured.partition.pdf import partition_pdf
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


import uuid
import fitz
import pymupdf
import pytesseract
from pdf2image import*
from unstructured.partition.utils.ocr_models.tesseract_ocr import OCRAgentTesseract
import nltk
import io
import re
import IPython
from IPython.display import HTML, display
from io import BytesIO
from PIL import Image

#os.environ["OCR_AGENT"] = "TESSERACT"
#os.environ['TESSERACT_CMD'] = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'insight_engine\ocrTesseract-OCR\tesseract.exe'

#r'insight_engine\ocrTesseract-OCR\tesseract.exe'#this might 

def extract_pdf_elements(path,output_path,max_characters,new_after_n_chars,combine_text_under_n_chars):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return partition_pdf(
        filename=path ,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters= max_characters,
        new_after_n_chars= new_after_n_chars,
        combine_text_under_n_chars= combine_text_under_n_chars,
        image_output_dir_path= output_path,
    )

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

#streamlit run .\st_style_output_demo.py  
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a  summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")#model = ChatOpenAI(temperature=0, model="gpt-4")

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    elif texts:
        text_summaries = texts

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries



def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

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


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
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
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))


    return img_base64_list, image_summaries


def find_fig_text(text):
    pattern = r'Fig\..*?\.'
    fig_sentences = re.findall(pattern, text, flags=re.DOTALL)
    return ' '.join(fig_sentences) if fig_sentences else ''

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_def(img_base64, prompt):
    return prompt




def generate_img_summaries_rule(pdf_path, images_dir='extracted_images_main_0'):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    doc = fitz.open(pdf_path)
    text_file_path = pdf_path.replace('.pdf', '_extracted_text.txt')

    img_base64_list = []

    image_summaries = []

    with open(text_file_path, 'w') as text_file:
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
                    print(f"Text related to images on Page {page_num + 1}, Image {image_index}:\n{fig_text}\n")

                image_filename = f"{fig_text}.png"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_bytes)


                base64_image = encode_image(image_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_def(base64_image, fig_text))


    return img_base64_list, image_summaries

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    store = InMemoryStore()
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
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever




def plt_img_base64(img_base64):
    """Display base64 encoded string as image in Streamlit"""
    img_data = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_data))
    st.image(img)


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


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
    formatted_texts = "\n".join(data_dict["context"]["texts"])
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
            "You are analyst tasking with answering questions .\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answer related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


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
            if con ==  "":
                con = x
        else:
            if img ==  "":
                img = x
    
    con = chain_multimodal_rag.invoke(query )
    if img == "":
        print("No image found")
    if con == "":
        print("No content found")

    return con, img

def create_keyword_network(text, keywords):
    # Initialize a graph
    G = nx.Graph()
    
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Create a defaultdict to store connections (edges) between keywords
    connections = defaultdict(int)
    
    # Iterate through each sentence to find connections between keywords
    for sentence in sentences:
        for keyword1 in keywords:
            if keyword1.lower() in sentence.lower():
                for keyword2 in keywords:
                    if keyword2.lower() in sentence.lower() and keyword1 != keyword2:
                        connections[(keyword1, keyword2)] += 1
    
    # Add nodes (keywords) to the graph
    G.add_nodes_from(keywords)
    
    # Add edges with weights (connection strengths)
    for (k1, k2), weight in connections.items():
        G.add_edge(k1, k2, weight=weight)
    
    return G


from dotenv import load_dotenv
dotenv_path = 'op.env'
load_dotenv(dotenv_path)
client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

def topic_extraction(text):
  completion = client.completions.create(
      model = "gpt-3.5-turbo-instruct",
      prompt = f"Extract distinct keywords  from: {text}  \n\nTopics:",
      max_tokens = 100,
      temperature = 0
  )
  return completion.choices[0].text.strip()


def print_node_values_sorted(network):
    # Get nodes with their degree (total weight of edges connected to the node)
    node_values = {node: sum(data['weight'] for _, data in network[node].items()) for node in network.nodes()}
    
    # Remove any empty node or node with empty string as name
    node_values = {node: value for node, value in node_values.items() if node.strip() != ''}
    
    # Set minimum value for nodes with value 0 to 1
    node_values = {node: max(value, 1) for node, value in node_values.items()}
    
    # Sort nodes by their values in descending order
    sorted_nodes = dict(sorted(node_values.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_nodes



# Define function to generate word cloud
def generate_word_cloud(network):
    # Example network processing (replace with your actual network processing)
    node_values = {node: sum(data['weight'] for _, data in network[node].items()) for node in network.nodes()}
    node_values = {node: value for node, value in node_values.items() if node.strip() != ''}
    node_values = {node: max(value, 8) for node, value in node_values.items()}
    sorted_nodes = dict(sorted(node_values.items(), key=lambda x: x[1], reverse=True))
    
    # Define custom colormap with specified colors
    colors = [
        ['green', 'blue', 'purple', 'red'],          # Green, Blue, Purple, Red
        ['teal', 'royalblue', 'crimson', 'limegreen'],   # Teal, Royal Blue, Crimson, Lime Green
        ['darkorchid', 'orange', 'pink', 'yellow'],  # Dark Orchid, Orange, Pink, Yellow
        ['turquoise', 'gold', 'magenta', 'chartreuse'],  # Turquoise, Gold, Magenta, Chartreuse
        ['green', 'indigo', 'orchid', 'black'],      # Green, Indigo, Orchid, Black
        ['cyan', 'indigo', 'orchid', 'slategray'],   # Cyan, Indigo, Orchid, Slate Gray
    ]
    cmap_index = 4  # Choose the index of the colormap from your `colors` list
    cmap = ListedColormap(colors[cmap_index])
    
    # Generate word cloud with advanced customization
    wordcloud = WordCloud(width=1920, height=1080, background_color='white', colormap=cmap, 
                          max_words=40, contour_width=0, 
                          prefer_horizontal=0.85).generate_from_frequencies(sorted_nodes)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Keyword Importance Word Cloud', fontsize=16)
    plt.axis('off')
    plt.tight_layout()  # Ensures tight layout to avoid overlapping with title

    return plt

def print_node_values_sorted_2(network):
    # Get nodes with their degree (total weight of edges connected to the node)
    node_values = {node: sum(data['weight'] for _, data in network[node].items()) for node in network.nodes()}

    # Sort nodes by their values in descending order
    sorted_nodes = sorted(node_values.items(), key=lambda x: x[1], reverse=True)

    # Print nodes and their values
    for node, value in sorted_nodes:
        print(f"Node: {node}, Value: {value}")




def extract_words(text):
    # Define a regular expression pattern to match words between "number." and ":"
    pattern = r'\d+\.\s*([^\:]+):'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Strip any extra whitespace and return the list of words
    return [match.strip() for match in matches]
    # Generate word cloud


def generate_word_cloud_2(network, boost_nodes=[]):
    node_values = {node: sum(data['weight'] for _, data in network[node].items()) for node in network.nodes()}
    node_values = {node: value for node, value in node_values.items() if node.strip() != ''}
    node_values = {node: max(value, 8) for node, value in node_values.items()}
    
    # Boost the value of nodes in boost_nodes list
    for node in boost_nodes:
      for value in node_values:
        x = node
        y = value
        if x.lower().strip() == y.lower().strip() :
            node_values[value] *= 1.21
            print("!!!!!!!")
    
    sorted_nodes = dict(sorted(node_values.items(), key=lambda x: x[1], reverse=True))
    
    # Define custom colormap with specified colors
    colors = [
        ['green', 'blue', 'purple', 'red'],          # Green, Blue, Purple, Red
        ['teal', 'royalblue', 'crimson', 'limegreen'],   # Teal, Royal Blue, Crimson, Lime Green
        ['darkorchid', 'orange', 'pink', 'yellow'],  # Dark Orchid, Orange, Pink, Yellow
        ['turquoise', 'gold', 'magenta', 'chartreuse'],  # Turquoise, Gold, Magenta, Chartreuse
        ['green', 'indigo', 'orchid', 'black'],      # Green, Indigo, Orchid, Black
        ['cyan', 'indigo', 'orchid', 'slategray'],   # Cyan, Indigo, Orchid, Slate Gray
    ]
    cmap_index = 4  # Choose the index of the colormap from your `colors` list
    cmap = ListedColormap(colors[cmap_index])
    
    # Generate word cloud with advanced customization
    wordcloud = WordCloud(width=1920, height=1080, background_color='white', colormap=cmap, 
                          max_words=40, contour_width=0, 
                          prefer_horizontal=0.85).generate_from_frequencies(sorted_nodes)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Keyword Importance Word Cloud', fontsize=16)
    plt.axis('off')
    plt.tight_layout()  # Ensures tight layout to avoid overlapping with title

    return plt

def topic_extraction_top10(text, word_list ):
  completion = client.completions.create(
      model = "gpt-3.5-turbo-instruct",
      prompt = f"choose top 10  keywords from {word_list} most relevant to text :  {text}   \n\n Then itemeze and explain why they are important :",
      max_tokens = 400,
      temperature = 0
  )
  return completion.choices[0].text.strip()

def topic_extraction_top10_gpt4(text_to_analyze, word_list):
    # gpt4o doesnt work as good as this 
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps with extracting keywords."},
            {"role": "user", "content": f"choose top 10 keywords from {word_list} most relevant to text: {text_to_analyze}\n\n Then itemize and explain why they are important to text:"}
        ]
    )
    
    # Return the response content
    return completion.choices[0].message.content

def topic_extraction_gpt4(text_to_analyze):
    # gpt4o doesnt work as good as this 
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps with extracting keywords."},
            {"role": "user", "content": f"Extract distinct keywords  from: {text_to_analyze}  \n\nTopics:",}
        ]
    )
    
    # Return the response content
    return completion.choices[0].message.content

def visualize_network(G):
    plt.figure(figsize=(14, 10))
    
    # Use spring_layout with adjusted k parameter
    pos = nx.spring_layout(G, k=0.5, iterations=5)  # Increase iterations for better layout
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, edge_color='gray', font_size=12, font_weight='bold', alpha=0.9)
    
    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title('Keyword Network')
    return plt 

def topic_extraction_gpt4_topic(text_to_analyze, topic):
    # gpt4o doesnt work as good as this 
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps with extracting keywords."},
            {"role": "user", "content": f"Extract distinct keywords with focus on {topic} realted ones  from: {text_to_analyze}  \n\nTopics:",}
        ]
    )
    
    # Return the response content
    return completion.choices[0].message.content

def topic_extraction_top10_gpt4_topic(text_to_analyze, word_list, topic):
    # gpt4o doesnt work as good as this 
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps with extracting keywords."},
            {"role": "user", "content": f"choose top 10 keywords with focus on {topic} realted ones  from {word_list} most relevant to text: {text_to_analyze}\n\n Then itemize and explain why they are important to text:"}
        ]
    )
    
    # Return the response content
    return completion.choices[0].message.content