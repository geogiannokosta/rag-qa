import ollama

class Embedder:

    def embed(self, prompt: str):
        return ollama.embed(
            # model='nomic-embed-text',
            model='embeddinggemma',
            input=prompt,
        ).embeddings[0]
