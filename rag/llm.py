from ollama import chat

def run_llm(prompt: str) -> str:
    return chat(
        model='phi3',
        messages=[{'role': 'user', 'content': prompt}]
    ).message.content
