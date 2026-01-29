from rag.retriever import Retriever
from rag.embeddings import Embedder
from PyPDF2 import PdfReader
from rag.llm import run_llm
from rag.agent import Agent
import logging 
import argparse
import ollama

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def ensure_models():
    REQUIRED_MODELS = [
        "phi3:latest",
        "embeddinggemma:latest"
    ]

    existing = {m["model"] for m in ollama.list()["models"]}

    for model in REQUIRED_MODELS:
        if model not in existing:
            logging.info(f"Pulling model: {model}")
            ollama.pull(model)
        else:
            logging.info(f"Model already present: {model}")

def main():
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument(
        "--chunking",
        choices=["basic", "semantic"],
        default="basic",
        help="Chunking strategy to use"
    )

    args = parser.parse_args()

    print_banner()

    ensure_models()

    # Try creating embedder, otherwise return None so as to fall to TF-IDF
    try:
        embedder = Embedder()
    except Exception as e:
        embedder = None

    retriever = Retriever(
        embedder=embedder,
        pdf_reader=PdfReader,
        docs_paths=[
        "docs/E3 Structure - Document 2.pdf",
        "docs/E10 - Document 3.pdf",
        "docs/ICD - Document 1.pdf"
        ],
        chunking_strategy=args.chunking
    )

    agent = Agent(retriever, run_llm)

    while True:
        question = input("Ask a question: ")
        answer, log = agent.run(question)

        print_answer(answer)
        print_log(log)

def print_banner():
    banner = """
══════════════════════════════════════════════════════
        ██████╗  █████╗  ██████╗ 
        ██╔══██╗██╔══██╗██╔════╝ 
        ██████╔╝███████║██║  ███╗
        ██╔══██╗██╔══██║██║   ██║
        ██║  ██║██║  ██║╚██████╔╝
        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 

        Retrieval-Augmented Q&A CLI
        LLM: phi-3  |  Embeddings: embeddinggemma
══════════════════════════════════════════════════════
"""
    print(banner)

def print_answer(answer: str):
    print("\n" + "-" * 80)
    print(" ANSWER ".center(80, "-"))
    print("-" * 80)
    print(answer.strip())
    # print("-" * 80 + "\n")

def print_log(log: dict):
    print("\n" + "-" * 80)
    print(" EXECUTION LOG ".center(80, "-"))
    print("-" * 80)

    print(f"Trace ID : {log['trace_id']}")
    print(f"Question : {log['question']}")
    print(f"Plan     : {' → '.join(log['plan'])}")

    print("\nRetrieval:")
    for i, r in enumerate(log["retrieval"], 1):
        print(
            f"  {i}. {r['file']} "
            f"(chunk {r['chunk_id']}, score={r['score']:.3f})\n"
            # f"{r['text']}"
        )

    print("\nLatency (ms):")
    for k, v in log["latency_ms"].items():
        print(f"  {k:>8}: {v}")

    if log["errors"]:
        print("\nErrors:")
        for e in log["errors"]:
            print(f"  - {e}")
    else:
        print("\nErrors: None ✅")

    print("-" * 80 + "\n")

if __name__ == "__main__":
    main()