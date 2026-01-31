import pytest
from rag.utils.validator import QValidator  

@pytest.fixture
def validator():
    return QValidator()


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_empty_query(validator, query):
    valid, error = validator.validate_question(query)

    assert not valid
    assert error == "EMPTY_QUERY"
    assert validator.human_readable_message(error) == "Please provide a question."


@pytest.mark.parametrize(
    "query",
    [
        ".",
        "!!!",
        "???",
        "---",
        "@#$%^&*()",
    ],
)
def test_semantically_empty_query(validator, query):
    valid, error = validator.validate_question(query)

    assert not valid
    assert error == "NO_SEMANTIC_CONTENT"
    assert (
        validator.human_readable_message(error)
        == "Your input doesn’t contain a meaningful question."
    )


@pytest.mark.parametrize(
    "query",
    [
        "a",
        "ab",
        "1",
        "a!",
        "x?",
    ],
)
def test_too_short_query(validator, query):
    valid, error = validator.validate_question(query)

    assert not valid
    assert error == "QUERY_TOO_SHORT"
    assert (
        validator.human_readable_message(error)
        == "Could you be a bit more specific?"
    )


@pytest.mark.parametrize(
    "query",
    [
        "What is RAG?",
        "Explain TF-IDF",
        "How does retrieval work?",
        "Why use embeddings?",
    ],
)
def test_valid_queries_pass_validation(validator, query):
    valid, error = validator.validate_question(query)

    assert valid
    assert error is None


def test_unknown_error_code_fallback_message(validator):
    msg = validator.human_readable_message("SOME_UNKNOWN_CODE")

    assert msg == "Your request could not be processed."