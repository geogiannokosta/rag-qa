import logging
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(
            self,
            embedder,
            pdf_reader,
            docs_paths: list[str],
            chunk_size: int = 2000,
            overlap_ratio: float = 0.15,  # 10-20% recommended
            chunking_strategy: str = "basic",
            save: bool = True

    ):
        self.documents = []
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        self.step = chunk_size - self.overlap
        self.chunking_strategy = chunking_strategy
        self.pdf_reader = pdf_reader
        self.save = save

        if embedder:
            self.embedder = embedder
            self.use_embbeder = True
        else:
            logging.warning("Embeddinggemma unavailable, falling back to TF-IDF: ")
            self.use_embbeder = False
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

        self._load_and_embed_docs(docs_paths)

    def _chunk_text(self, text: str):
        """
        Yield overlapping chunks of text
        """
        for start in range(0, len(text), self.step):
            chunk = text[start:start + self.chunk_size]
            if chunk.strip():
                yield chunk

    def _semantic_chunk_text(self, text: str):
        pass

    def _load_and_embed_docs(self, docs_paths):
        chunk_id = 0
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)

        for path in docs_paths:
            pdf_name = Path(path).stem
            pickle_file = storage_dir / f"{pdf_name}.pkl"

            # Load from storage
            if pickle_file.exists():
                logging.info(f"Loading cached embeddings for {path}")
                with open(pickle_file, "rb") as f:
                    saved_docs = pickle.load(f)

                # Ensure chunk_id continuity
                for doc in saved_docs:
                    doc["metadata"]["chunk_id"] = chunk_id
                    chunk_id += 1

                self.documents.extend(saved_docs)
                continue

            # Process Pdfs
            logging.info(f"Processing and embedding {path}")
            data_to_store = []
            pdf_reader = self.pdf_reader(path)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                if self.chunking_strategy == "semantic":
                    chunks = self._semantic_chunk_text(text)
                else:
                    chunks = self._chunk_text(text)
                
                for chunk in chunks:
                    embedding = None

                    if self.use_embbeder:
                        embedding = self.embedder.embed(chunk)

                    chunk_data = {
                        "text": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "file": path,
                            "page": page_num,
                            "chunk_id": chunk_id
                        }
                    }

                    data_to_store.append(chunk_data)
                    self.documents.append(chunk_data)
                    chunk_id += 1
            
            if self.save:
                # Save to storage 
                with open(pickle_file, "wb") as f:
                    pickle.dump(data_to_store, f)

                logging.info(f"Saved embeddings to {pickle_file}")

        # TF-IDF fallback
        if not self.use_embbeder:
            self.tfidf_matrix = self.vectorizer.fit_transform(
                [d["text"] for d in self.documents]
            )

    def retrieve(self, query: str, top_k: int = 3):
        """
        Returns top_k (document, score) pairs
        """

        if self.use_embbeder:
            query_vector = self.embedder.embed(query)
            scores = [
                cosine_similarity([query_vector], [d["embedding"]])[0][0] for d in self.documents
            ]
        else:
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]
    