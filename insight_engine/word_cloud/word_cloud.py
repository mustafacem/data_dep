import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud
import networkx as nx
import re
from collections import defaultdict


def visualize_network(G):
    plt.figure(figsize=(14, 10))

    # Use spring_layout with adjusted k parameter
    pos = nx.spring_layout(
        G, k=0.5, iterations=5
    )  # Increase iterations for better layout

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        edge_color="gray",
        font_size=12,
        font_weight="bold",
        alpha=0.9,
    )

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Keyword Network")
    return plt


def clean_and_split(input_text):
    # Join the list into a single string if it's a list
    if isinstance(input_text, list):
        input_text = "\n".join(input_text)

    # Remove numbers and periods followed by whitespace or newlines
    cleaned_text = re.sub(r"\d+\.\s*", "", input_text)

    # Split the text by newlines while preserving groups of words
    split_items = cleaned_text.splitlines()

    # Filter out any empty strings that may result from splitting
    cleaned_list = [item.strip() for item in split_items if item.strip()]

    return cleaned_list


def create_keyword_network(text, keywords):
    # Initialize a graph
    print("key : ")
    print(keywords)
    G = nx.Graph()
    keywords = clean_and_split(keywords)
    if keywords[0] == "1":
        keywords.remove("1")
    print("key : ")
    print(keywords)
    print("text : " + text)
    # Split text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

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

    print("////////////")
    print(connections)
    G.add_nodes_from(keywords)

    # Add edges with weights (connection strengths)
    for (k1, k2), weight in connections.items():
        G.add_edge(k1, k2, weight=weight)

    return G


def generate_word_cloud_2(network, boost_nodes=[]):
    node_values = {
        node: sum(data["weight"] for _, data in network[node].items())
        for node in network.nodes()
    }
    node_values = {
        node: value for node, value in node_values.items() if node.strip() != ""
    }
    node_values = {node: max(value, 8) for node, value in node_values.items()}

    # Boost the value of nodes in boost_nodes list
    for node in boost_nodes:
        for value in node_values:
            x = node
            y = value
            if x.lower().strip() == y.lower().strip():
                node_values[value] *= 1.21

    sorted_nodes = dict(sorted(node_values.items(), key=lambda x: x[1], reverse=True))

    # Define custom colormap with specified colors
    colors = [
        ["green", "blue", "purple", "red"],  # Green, Blue, Purple, Red
        [
            "teal",
            "royalblue",
            "crimson",
            "limegreen",
        ],  # Teal, Royal Blue, Crimson, Lime Green
        ["darkorchid", "orange", "pink", "yellow"],  # Dark Orchid, Orange, Pink, Yellow
        [
            "turquoise",
            "gold",
            "magenta",
            "chartreuse",
        ],  # Turquoise, Gold, Magenta, Chartreuse
        ["green", "indigo", "orchid", "black"],  # Green, Indigo, Orchid, Black
        ["cyan", "indigo", "orchid", "slategray"],  # Cyan, Indigo, Orchid, Slate Gray
    ]
    cmap_index = 4  # Choose the index of the colormap from your `colors` list
    cmap = ListedColormap(colors[cmap_index])

    # Generate word cloud with advanced customization
    wordcloud = WordCloud(
        width=1920,
        height=1080,
        background_color="white",
        colormap=cmap,
        max_words=40,
        contour_width=0,
        prefer_horizontal=0.85,
    ).generate_from_frequencies(sorted_nodes)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Keyword Importance Word Cloud", fontsize=16)
    plt.axis("off")
    plt.tight_layout()  # Ensures tight layout to avoid overlapping with title

    return plt


def extract_words(text):
    # Define a regular expression pattern to match words between "number." and ":"
    pattern = r"\d+\.\s*([^\:]+):"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Strip any extra whitespace and return the list of words
    return [match.strip() for match in matches]
    # Generate word cloud


def print_node_values_sorted(network):
    # Get nodes with their degree (total weight of edges connected to the node)
    node_values = {
        node: sum(data["weight"] for _, data in network[node].items())
        for node in network.nodes()
    }

    # Remove any empty node or node with empty string as name
    node_values = {
        node: value for node, value in node_values.items() if node.strip() != ""
    }

    # Set minimum value for nodes with value 0 to 1
    node_values = {node: max(value, 1) for node, value in node_values.items()}

    # Sort nodes by their values in descending order
    sorted_nodes = dict(sorted(node_values.items(), 
                               key=lambda x: x[1], reverse=True))

    return sorted_nodes


def print_node_values_sorted_2(network):
    # Get nodes with their degree (total weight of edges connected to the node)
    node_values = {
        node: sum(data["weight"] for _, data in network[node].items())
        for node in network.nodes()
    }

    # Sort nodes by their values in descending order
    sorted_nodes = sorted(node_values.items(), 
                          key=lambda x: x[1], reverse=True)

    # Print nodes and their values
    for node, value in sorted_nodes:
        print(f"Node: {node}, Value: {value}")
