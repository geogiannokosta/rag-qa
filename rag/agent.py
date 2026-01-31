"""
The agent will:
    orchestrate calls,
    format prompts,
    assemble citations,
    produce the required log object
"""

import uuid
import time

class Agent:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def run(self, question: str, top_k: int = 3):
        trace_id = str(uuid.uuid4())

        plan = ["retrieve", "draft", "cite"]
        errors = []

        # Retrieve
        start_time = time.time()
        retrieved = self.retriever.retrieve(question, top_k)
        retrieve_latency = int((time.time() - start_time) * 1000)

        # Draft
        draft_start_time = time.time()
        prompt, sources = self._create_prompt(question, retrieved)
        answer = self.llm(prompt)
        draft_latency = int((time.time() - draft_start_time) * 1000)

        # Cite
        response = self._add_citations(answer, sources)

        total_latency = int((time.time() - start_time) * 1000)

        log = {
            "trace_id": trace_id,
            "question": question,
            "plan": plan,
            "retrieval": [
                {
                    "file": r[0]["metadata"]["file"],
                    "chunk_id": r[0]["metadata"]["chunk_id"],
                    "text": r[0]["text"],
                    "score": round(float(r[1]), 4)
                }
                for r in retrieved
            ],
            "draft_tokens": len(prompt.split()),
            "latency_ms": {
                "retrieve": retrieve_latency,
                "draft": draft_latency,
                "total": total_latency
            },
            "errors": errors
        }

        return response, log

    def _create_prompt(self, question, retrieved):
        context_blocks = []
        sources = []

        for i, (document, score) in enumerate(retrieved):
            metadata = document["metadata"]
            source_id = f"[{i + 1}]"

            context_blocks.append(
                f"{source_id} File: {metadata['file']}, Page: {metadata['page']}\n{document['text']}"
            )

            sources.append({
                "id": source_id,
                "file": metadata["file"],
                "page": metadata["page"]
            })

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a QA assistant.
Answer the question **only** using the provided context.
If the answer is not contained, say "I don't know".

Context:
{context}

Question:
{question}
"""
        return prompt.strip(), sources

    def _add_citations(self, answer: str, sources):
        citation_lines = "\n\nSources:\n"
        for s in sources:
            citation_lines += f"{s['id']} {s['file']} (page {s['page']})\n"

        return answer.strip() + citation_lines
