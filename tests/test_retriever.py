import pytest
from rag.retriever import Retriever

@pytest.fixture
def retriever(embedder, pdf_reader, simple_docs, tmp_path):
    # override storage dir to temp folder
    storage = tmp_path / "storage"
    storage.mkdir()
    r = Retriever(
        embedder=embedder,
        pdf_reader=pdf_reader,
        docs_paths=simple_docs,
        chunk_size=20,
        overlap_ratio=0.2,
        chunking_strategy="basic",
        save=False
    )
    return r

@pytest.fixture
def sample_text():
    return (
        "Paragraph one.\n\n"
        "Paragraph two with more content.\n\n"
        "Paragraph three."
    )

def test_documents_loaded(retriever):
    assert len(retriever.documents) > 0
    for doc in retriever.documents:
        assert "text" in doc
        assert "embedding" in doc
        assert "metadata" in doc
        assert "file" in doc["metadata"]
        assert "page" in doc["metadata"]
        assert "chunk_id" in doc["metadata"]

def test_retrieve_scores(retriever):
    results = retriever.retrieve("First page content")
    assert len(results) > 0
    for doc, score in results:
        assert isinstance(score, float)
        assert score >= 0

def test_basic_chunking_produces_chunks(retriever, sample_text):
    chunks = list(retriever._chunk_text(sample_text))

    assert len(chunks) > 1
    assert all(len(c) <= 20 for c in chunks)

def test_chunk_overlap(retriever):
    texts = [doc["text"] for doc in retriever.documents]
    print(texts)
    overlap_len = int(20 * 0.2)

    for i in range(1, len(texts)):
        assert texts[i-1][-overlap_len:] in texts[i]
