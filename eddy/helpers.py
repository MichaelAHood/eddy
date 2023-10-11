""" Test assets. """
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime


def load_rng(seed):
    return np.random.default_rng(seed)


def load_index(dimension):
    return faiss.IndexFlatIP(dimension)


def load_vectors(n_vectors, dimension):
    rng = load_rng(1234)
    return rng.normal(size=(n_vectors, dimension)).astype("float32")


def load_normalized_vectors(n_vectors, dimension):
    vectors = load_vectors(n_vectors, dimension)
    faiss.normalize_L2(vectors)
    return vectors


def load_vectors_and_index(n_vectors, dimension):
    vectors = load_normalized_vectors(n_vectors, dimension)
    index = load_index(dimension)
    index.add(vectors)
    return vectors, index


def plot_graph(G, node_size=40, font_size=5):
    """Plot the network."""
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=node_size,
        font_size=font_size,
    )
    plt.show()


def save_image(G, filename, node_size=40, font_size=5):
    """Save a plot of the graph."""
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=node_size,
        font_size=font_size,
    )
    # Save the plot to a file
    plt.savefig(filename, bbox_inches="tight")


def ts_from_iso(ts_string: str) -> int:
    """Timestamp as integer."""
    return int(datetime.fromisoformat(ts_string).timestamp())
