"""
Command-line interface for running a Retrieval-Augmented Generation (RAG) pipeline.

This CLI:
- Validates user questions
- Ensures required Ollama models are available
- Initializes retrieval and embedding components
- Executes a RAG agent
- Displays the generated answer and a structured execution log

It is intended as a lightweight, user-facing entry point and is excluded from
coverage and strict linting rules where appropriate.
"""

import json
from rag.retriever import Retriever
from rag.embeddings import Embedder
from PyPDF2 import PdfReader
from rag.llm import run_llm
from rag.agent import Agent
from rag.utils.validator import QValidator
import logging 
import argparse
import ollama
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence Ollama logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def ensure_models():
    """
    Ensure that all required Ollama models are available.

    If a required model is missing, it will be pulled automatically.
    This function blocks until all models are present.
    """
    REQUIRED_MODELS = [
        "phi3:latest",
        "embeddinggemma:latest"
    ]

    existing = {m["model"] for m in ollama.list()["models"]}

    for model in REQUIRED_MODELS:
        if model not in existing:
            logger.info(f"Pulling model: {model}")
            ollama.pull(model)
        else:
            logger.info(f"Model already present: {model}")

def main():
    """
    Entry point for the RAG CLI.

    This function:
    - Parses CLI arguments
    - Displays a banner
    - Ensures required LLM and embedding models are available
    - Initializes retrieval and agent components
    - Validates the user question
    - Executes the RAG workflow and prints results
    """
    parser = argparse.ArgumentParser(description="RAG CLI")
    parser.add_argument(
        "question",
        help="Question to ask",
    )
    parser.add_argument(
        "--top_k",
        default=3,
        type=int,
        help="Number of top document chunks to retrieve"
    )
    parser.add_argument(
        "--chunking",
        choices=["basic", "semantic"],
        default="basic",
        help="Chunking strategy to use"
    )
    parser.add_argument(
        "--chunk_size",
        default=2000,
        help="Chunk size to use"
    )
    parser.add_argument(
        "--overlap_ratio",
        default=0.15,
        help="Chunk overlap ratio to use"
    )

    args = parser.parse_args()

    log_banner()

    # Wait for Ollama server to be available
    while True:
        try:
            ensure_models()
            break
        except Exception as e:
            logger.exception("Cannot set up the required Ollama models. %s", e)
            logger.info("Will try again in 1 minute.")
            time.sleep(60)

    # Try creating embedder, otherwise return None so as to fall to TF-IDF
    try:
        embedder = Embedder()
    except Exception as e:
        embedder = None

    docs_path = "docs/"
    retriever = Retriever(
        embedder=embedder,
        pdf_reader=PdfReader,
        docs_paths = [docs_path + doc_path for doc_path in os.listdir(docs_path) if doc_path.endswith(".pdf")],
        chunking_strategy=args.chunking,
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap_ratio
    )

    agent = Agent(retriever, run_llm)
    qvalidator = QValidator()

    question = args.question
    valid, error_code = qvalidator.validate_question(question)
    if not valid:
        log_answer(qvalidator.human_readable_message(error_code))
        return
    
    answer, log = agent.run(question, args.top_k)

    log_answer(answer)
    log_logs_json(log)

def log_banner():
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
    logger.info(banner)

def log_answer(answer: str):
    message = f"""
{"-" * 80}
{"ANSWER".center(80)}
{"-" * 80}
{answer.strip()}
"""
    logger.info(message)

def log_logs_json(log: dict):
    """
    Log the agent execution log as structured JSON.

    The JSON format matches:

    {
        "trace_id": "...",
        "question": "...",
        "plan": [...],
        "retrieval": [{"file":"a.txt","chunk_id":42,"score":0.78}],
        "draft_tokens": ...,
        "latency_ms": {...},
        "errors": [...]
    }

    :param log: Execution log dictionary from Agent.run()
    """
    # Build a minimal structured log
    structured_log = {
        "trace_id": log.get("trace_id"),
        "question": log.get("question"),
        "plan": log.get("plan"),
        "retrieval": [
            {
                "file": r["file"],
                "chunk_id": r["chunk_id"],
                "score": r["score"],
                "section_path": r.get("section_path", [])
            }
            for r in log.get("retrieval", [])
        ],
        "draft_tokens": log.get("draft_tokens"),
        "latency_ms": log.get("latency_ms"),
        "errors": log.get("errors", []),
    }

    logger.info("\n" + json.dumps(structured_log, indent=2))

if __name__ == "__main__":
    main()
