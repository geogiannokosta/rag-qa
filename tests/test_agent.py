import pytest
from rag.agent import Agent

class FakeRetriever:
    def retrieve(self, question, top_k=3):
        return [
            ({"text": f"text {i}", "metadata": {"file": f"doc{i}.pdf", "page": i, "chunk_id": i}}, 0.5)
            for i in range(top_k)
        ]

def fake_llm(prompt):
    return "ANSWER"

@pytest.fixture
def agent():
    return Agent(FakeRetriever(), fake_llm)

def test_agent_run(agent):
    answer, log = agent.run("Question?")
    assert answer.startswith("ANSWER")
    assert "trace_id" in log
    assert "retrieval" in log
    assert len(log["retrieval"]) == 3
    assert log["draft_tokens"] > 0
    assert "latency_ms" in log
    assert log["plan"] == ["retrieve", "draft", "cite"]

def test_create_prompt_and_citations(agent):
    retrieved = [
        ({"text": "text1", "metadata": {"file": "a.pdf", "page": 1, "chunk_id": 0}}, 0.9),
        ({"text": "text2", "metadata": {"file": "b.pdf", "page": 2, "chunk_id": 1}}, 0.8)
    ]
    prompt, sources = agent._create_prompt("Q?", retrieved)
    assert "text1" in prompt
    assert "text2" in prompt
    assert sources[0]["id"] == "[1]"
    answer_with_cites = agent._add_citations("Some answer", sources)
    assert "Sources:" in answer_with_cites
    assert "[1] a.pdf" in answer_with_cites