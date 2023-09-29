from faiss.swigfaiss_avx2 import IndexFlatIP
from eddy.index import load_inner_product_index


def test_load_inner_product_index() -> None:
    assert isinstance(load_inner_product_index(3), IndexFlatIP)
