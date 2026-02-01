from rag.retriever import Retriever

def test_tfidf_fallback(pdf_reader, simple_docs):
    # Pass an embedder that will fail
    retriever = Retriever(
        embedder=None, 
        pdf_reader=pdf_reader,
        docs_paths=simple_docs,
        chunk_size=50,
        save=False
    )

    # Ensure TF-IDF matrix was created
    assert retriever.tfidf_matrix is not None

    # Test that retrieval works without embeddings
    results = retriever.retrieve("page content")
    assert len(results) > 0
    # The score should be > 0 because 'page content' exists in texts
    assert all(score > 0 for _, score in results)
