import ollama

class Embedder:
    """
    Wrapper class responsible for generating vector embeddings for text prompts
    using an Ollama embedding model.
    """

    def embed(self, prompt: str):
        """
        Generate an embedding vector for the given text prompt.

        This method sends the input text to an Ollama embedding model and returns
        the resulting numerical embedding.

        Args:
            prompt (str): The input text to be embedded.

        Returns:
            list[float]: The embedding vector representing the semantic meaning
            of the input prompt.
        """
        return ollama.embed(
            # model='nomic-embed-text',
            model='embeddinggemma',
            input=prompt,
        ).embeddings[0]
