from rag.retriever import Retriever
import os
    
# class FakePdfReader:
#     def __init__(self, path):
#         # Each file has one "page" with the file content
#         self.pages = [type('Page', (), {'extract_text': lambda self=None: open(path).read()})()]

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
    print(results)
    # The score should be > 0 because 'document' exists in texts
    assert all(score > 0 for _, score in results)

    # os.remove("storage/doc0.pkl")
    # os.remove("storage/doc1.pkl")
