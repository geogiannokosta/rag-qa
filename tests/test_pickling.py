import os
import pickle
from pathlib import Path
from rag.retriever import Retriever

def test_pickle_caching(simple_docs, embedder, pdf_reader, tmp_path):
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    
    # First run: should create pickle
    retriever1 = Retriever(
        embedder=embedder,
        pdf_reader=pdf_reader,
        docs_paths=simple_docs,
        chunk_size=50
    )
    
    pickle_file = Path("storage") / f"{Path(simple_docs[0]).stem}.pkl"
    assert pickle_file.exists()  # Pickle file should be created

    # Read pickle content
    with open(pickle_file, "rb") as f:
        first_docs = pickle.load(f)
    
    # Second run: should load from pickle
    retriever2 = Retriever(
        embedder=embedder,
        pdf_reader=pdf_reader,
        docs_paths=simple_docs,
        chunk_size=50
    )

    # Compare only text and metadata fields, ignore embedding identity
    for d1, d2 in zip(first_docs, retriever2.documents):
        assert d1["text"] == d2["text"]
        assert d1["metadata"]["file"] == d2["metadata"]["file"]
        assert d1["metadata"]["page"] == d2["metadata"]["page"]
        assert d1["metadata"]["chunk_id"] == d2["metadata"]["chunk_id"]
        assert len(d1["embedding"]) == len(d2["embedding"])
        assert all(abs(a - b) < 1e-6 for a, b in zip(d1["embedding"], d2["embedding"]))

    os.remove("storage/doc0.pkl")