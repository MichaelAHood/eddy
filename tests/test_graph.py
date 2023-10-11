import numpy as np
import faiss
from eddy.graph import build_adjacency_matrix
from eddy.helpers import load_vectors_and_index


def test_build_graph():
    pass


def test_build_adjacency_matrix() -> None:
    dimension = 2
    k = 3
    threshold = 0.8
    n_vectors = 6

    vectors, index = load_vectors_and_index(n_vectors, dimension)
    distances, indices = index.search(vectors, k)
    A = build_adjacency_matrix(distances, indices, threshold)
    got = np.array(
        [
            [1.0, 0.0, 0.0, 0.9, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 1.0, 0.9, 0.0],
            [1.0, 0.0, 0.0, 0.9, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert A is not None
    assert A.shape == (n_vectors, n_vectors)
    assert all(np.isclose(A.round(1), got).flatten())
