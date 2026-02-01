import logging
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROMAN_HEADER = re.compile(r"^[IVXLCDM]+\.\s+.+")
LETTER_HEADER = re.compile(r"^[A-Z]\.\s+.+")
NUMBER_HEADER = re.compile(r"^\d+(\.\d+)*\s+.+")
QUESTION_HEADER = re.compile(r"^(Q\d+[:.]|\d+\.)\s+.+")
ANSWER_HEADER = re.compile(r"^A\d+[:.]")

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
            logging.warning("Embeddinggema model unavailable, falling back to TF-IDF: ")
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
        """
        Chunk text using document structure such as section headers and Q&A blocks.
        Preserves hierarchical context for better retrieval accuracy.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        chunks = []

        section = {
            "roman": None,
            "letter": None,
            "number": None,
            "question": None,
        }

        buffer = []

        def flush():
            if not buffer:
                return

            header_path = [
                h for h in section.values() if h is not None
            ]

            chunk_text = "\n".join(header_path + [""] + buffer)

            chunks.append({
                "text": chunk_text,
                "section_path": header_path.copy()
            })

            buffer.clear()

        for line in lines:
            if ROMAN_HEADER.match(line):
                flush()
                section["roman"] = line
                section["letter"] = None
                section["number"] = None
                section["question"] = None

            elif LETTER_HEADER.match(line):
                flush()
                section["letter"] = line
                section["number"] = None
                section["question"] = None

            elif QUESTION_HEADER.match(line):
                flush()
                section["question"] = line

            elif ANSWER_HEADER.match(line):
                buffer.append(line)

            else:
                buffer.append(line)

        flush()
        return chunks

    def _load_and_embed_docs(self, docs_paths):
        chunk_id = 0
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)

        for path in docs_paths:
            pdf_name = Path(path).stem
            pickle_file = storage_dir / f"{pdf_name}_{self.chunking_strategy}.pkl"

            # Load from storage if default chunk size
            if pickle_file.exists() and self.chunk_size==2000:
                logging.info("Loading cached embeddings for %s", path)
                with open(pickle_file, "rb") as f:
                    saved_docs = pickle.load(f)

                # Ensure chunk_id continuity
                for doc in saved_docs:
                    doc["metadata"]["chunk_id"] = chunk_id
                    chunk_id += 1

                self.documents.extend(saved_docs)
                continue

            # Process Pdfs
            logging.info("Processing and embedding: %s", path)
            data_to_store = []
            pdf_reader = self.pdf_reader(path)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue

                if self.chunking_strategy == "semantic":
                    chunks = self._semantic_chunk_text(text)
                else:
                    chunks = [{"text": c, "section_path": []} for c in self._chunk_text(text)]

                for chunk in chunks:
                    embedding = None
                    if self.use_embbeder:
                        embedding = self.embedder.embed(chunk["text"])

                    chunk_data = {
                        "text": chunk["text"],
                        "embedding": embedding,
                        "metadata": {
                            "file": path,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "section_path": chunk.get("section_path", [])
                        }
                    }

                    data_to_store.append(chunk_data)
                    self.documents.append(chunk_data)
                    chunk_id += 1

            if self.save and self.chunk_size==2000:
                # Save to storage
                with open(pickle_file, "wb") as f:
                    pickle.dump(data_to_store, f)

                logging.info("Saved embeddings to %s", pickle_file)

        # TF-IDF fallback
        if not self.use_embbeder:
            self.tfidf_matrix = self.vectorizer.fit_transform(
                [d["text"] for d in self.documents]
            )

    def retrieve(self, question: str, top_k: int = 3):
        """
        Returns top_k (document, score) pairs
        """

        if self.use_embbeder:
            question_vector = self.embedder.embed(question)
            scores = [
                cosine_similarity([question_vector], [d["embedding"]])[0][0] for d in self.documents
            ]
        else:
            question_vector = self.vectorizer.transform([question])
            scores = cosine_similarity(question_vector, self.tfidf_matrix)[0]

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]
    