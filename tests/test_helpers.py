from .helpers import (
    load_rng,
    load_index,
    load_vectors,
    load_normalized_vectors,
    load_vectors_and_index,
)
import numpy as np
import faiss


def test_load_rng() -> None:
    """Verify RNG loads and performs as expected."""
    rng = load_rng(1932)
    assert isinstance(rng, np.random.Generator)
    assert np.isclose(rng.random(), 0.33750949668570174)
    assert np.isclose(rng.random(), 0.21245364971526193)


def test_load_index() -> None:
    """Verify the index loads."""
    index = load_index(2)
    assert isinstance(index, faiss.swigfaiss_avx2.IndexFlatIP)
    assert index.is_trained


def test_load_vectors() -> None:
    """Verify that we get the expected vectors."""
    want = np.array(
        [
            [-1.6, 0.1],
            [0.7, 0.2],
            [0.9, 2.9],
            [-1.5, 0.9],
        ]
    )
    got = load_vectors(4, 2).round(1)
    assert all(np.isclose(got, want).flatten())


def test_load_normalized_vectors() -> None:
    """Verify that we get the expected vectors."""
    want = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.2],
            [0.3, 1.0],
            [-0.8, 0.5],
        ]
    )
    got = load_normalized_vectors(4, 2).round(1)
    print(got)
    assert all(np.isclose(got, want).flatten())


def test_load_vectors_and_index() -> None:
    vectors, index = load_vectors_and_index(4, 2)
    print(vectors)
    got = vectors.round(1)
    want = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.2],
            [0.3, 1.0],
            [-0.8, 0.5],
        ]
    )
    assert all(np.isclose(got, want).flatten())
    assert index is not None
