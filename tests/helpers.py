""" Test assets. """
import faiss
import numpy as np


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
