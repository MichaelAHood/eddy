import numpy as np
import faiss


def test_add_and_query():
    dimension = 3
    k = 3
    n_vectors = 100
    index = faiss.IndexFlatIP(dimension)
    rng = np.random.default_rng(seed=1234)

    vectors = rng.normal(size=(n_vectors, dimension)).astype("float32")
    faiss.normalize_L2(vectors)

    index.add(vectors)

    new_vector = rng.normal(size=(1, dimension)).astype("float32")
    faiss.normalize_L2(new_vector)
    distances, indices = index.search(new_vector, k)

    for index, distance in zip(indices.flatten(), distances.flatten()):
        assert np.isclose(np.inner(new_vector, vectors[index]), distance)
