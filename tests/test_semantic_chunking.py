import pytest
from rag.retriever import Retriever


@pytest.fixture
def semantic_retriever(embedder, pdf_reader, simple_docs, tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()

    return Retriever(
        embedder=embedder,
        pdf_reader=pdf_reader,
        docs_paths=simple_docs,
        chunk_size=2000,
        overlap_ratio=0.15,
        chunking_strategy="semantic",
        save=False,
    )


@pytest.fixture
def semantic_text():
    return (
        "I. INTRODUCTION\n"
        "This is the introduction text.\n"
        "\n"
        "A. Background\n"
        "Some background information.\n"
        "\n"
        "Q1: What is the purpose?\n"
        "The purpose is to test semantic chunking.\n"
        "\n"
        "Q2: Another question\n"
        "Another answer follows here.\n"
    )


def test_semantic_chunking_produces_chunks(semantic_retriever, semantic_text):
    chunks = semantic_retriever._semantic_chunk_text(semantic_text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "section_path" in chunk
        assert chunk["text"].strip() != ""


def test_semantic_chunk_contains_headers_in_text(semantic_retriever, semantic_text):
    chunks = semantic_retriever._semantic_chunk_text(semantic_text)

    first_chunk = chunks[0]["text"]

    # Roman header should be included
    assert "I. INTRODUCTION" in first_chunk


def test_section_path_is_preserved(semantic_retriever, semantic_text):
    chunks = semantic_retriever._semantic_chunk_text(semantic_text)

    # Find chunk related to first question
    q1_chunk = next(
        c for c in chunks if "Q1:" in c["text"]
    )

    assert q1_chunk["section_path"] == [
        "I. INTRODUCTION",
        "A. Background",
        "Q1: What is the purpose?",
    ]


def test_section_path_resets_on_new_roman_section(semantic_retriever):
    text = (
        "I. INTRODUCTION\n"
        "Intro text.\n"
        "\n"
        "II. METHODS\n"
        "Method description.\n"
    )

    chunks = semantic_retriever._semantic_chunk_text(text)

    intro_chunk = chunks[0]
    methods_chunk = chunks[1]

    assert intro_chunk["section_path"] == ["I. INTRODUCTION"]
    assert methods_chunk["section_path"] == ["II. METHODS"]


def test_no_empty_section_paths_for_content_chunks(semantic_retriever, semantic_text):
    chunks = semantic_retriever._semantic_chunk_text(semantic_text)

    for chunk in chunks:
        # Content chunks should always have at least one header
        assert len(chunk["section_path"]) >= 1


def test_question_creates_new_chunk(semantic_retriever, semantic_text):
    chunks = semantic_retriever._semantic_chunk_text(semantic_text)

    question_chunks = [
        c for c in chunks if any(h.startswith("Q") for h in c["section_path"])
    ]

    assert len(question_chunks) == 2