import re

MIN_CHARS = 3
MIN_ALPHA_CHARS = 2
ERROR_MESSAGES = {
    "EMPTY_QUERY": "Please provide a question.",
    "NO_SEMANTIC_CONTENT": "Your input doesn’t contain a meaningful question.",
    "QUERY_TOO_SHORT": "Could you be a bit more specific?",
}

class QValidator: 
    """
    Validates user-provided questions before they are processed by the system.

    This validator performs lightweight, deterministic checks to ensure that
    a query contains meaningful content and meets minimum length requirements.
    It is intended to act as a guardrail before invoking heavier processing.
    """

    def _normalize_question(self, q: str) -> str:
        """
        Normalize the input question.

        Currently trims leading and trailing whitespace.

        :param q: Raw user input.
        :return: Normalized question string.
        """
        return q.strip()

    def _is_semantically_empty(self, q: str) -> bool:
        """
        Determine whether the question lacks semantic content.

        A question is considered semantically empty if it contains only
        punctuation, symbols, or whitespace, with no alphanumeric characters.

        :param q: Normalized question string.
        :return: True if the question has no semantic content, False otherwise.
        """
        return not re.search(r"[a-zA-Z0-9]", q)

    def _is_too_short(self, q: str) -> bool:
        """
        Check whether the question is too short to be meaningful.

        This validation considers both:
        - The total length of the string.
        - The number of alphanumeric characters.

        :param q: Normalized question string.
        :return: True if the question is too short, False otherwise.
        """
        alpha_count = sum(c.isalnum() for c in q)
        return alpha_count < MIN_ALPHA_CHARS or len(q) < MIN_CHARS

    def validate_question(self, q: str) -> tuple[bool, str | None]:
        """
        Validate a user question and return a structured result.

        The validation is performed in the following order:
        1. Empty input
        2. Semantically empty input
        3. Input that is too short

        :param q: Raw user input.
        :return: A tuple of (is_valid, error_code).
                 If valid, error_code is None.
                 If invalid, error_code is a string key describing the failure.
        """
        q = self._normalize_question(q)

        if not q:
            return False, "EMPTY_QUERY"

        if self._is_semantically_empty(q):
            return False, "NO_SEMANTIC_CONTENT"

        if self._is_too_short(q):
            return False, "QUERY_TOO_SHORT"

        return True, None
    
    def human_readable_message(self, error_code: str) -> str:
        """
        Convert a validation error code into a user-facing message.

        :param error_code: Validation error code returned by validate_question.
        :return: Human-readable error message.
        """
        return ERROR_MESSAGES.get(
            error_code,
            "Your request could not be processed."
        )