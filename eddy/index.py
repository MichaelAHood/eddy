import faiss
from faiss.swigfaiss_avx2 import IndexFlatIP


def load_inner_product_index(dimension: int) -> IndexFlatIP:
    return faiss.IndexFlatIP(dimension)
