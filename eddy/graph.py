import networkx as nx
import numpy as np
from faiss.swigfaiss_avx2 import IndexFlatIP


def build_threshold_graph(
    vectors: np.array, index: IndexFlatIP, k: int, threshold: float
) -> nx.Graph:
    """Construct a threshold graph."""
    distances, indices = index.search(vectors, k)
    print(distances, indices)


def build_adjacency_matrix(
    distances: np.array, indices: np.array, threshold: float
) -> np.array:
    """Construct an undirected adjacency matrix with weighted edges."""
    n_vectors = len(distances)
    shape = (n_vectors, n_vectors)
    adj_matrix = np.zeros(shape, dtype=np.float32)

    mask = distances > threshold
    for row, (distance_row, index_row) in enumerate(zip(mask, indices)):
        for dist, (i, col) in zip(distance_row, enumerate(index_row)):
            if dist:
                adj_matrix[row, col] = distances[row, i]

    return adj_matrix
