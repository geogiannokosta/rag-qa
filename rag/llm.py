from ollama import chat

def run_llm(prompt: str) -> str:
    """
    Execute a chat-based large language model (LLM) request with a user prompt.

    This function sends the provided prompt to the configured Ollama chat model
    and returns the model's generated response text.

    Args:
        prompt (str): The user input prompt to send to the LLM.

    Returns:
        str: The generated response content from the LLM.
    """
    return chat(
        model='phi3',
        messages=[{'role': 'user', 'content': prompt}]
    ).message.content
