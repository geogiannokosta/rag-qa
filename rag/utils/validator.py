import re

MIN_CHARS = 3
MIN_ALPHA_CHARS = 2
ERROR_MESSAGES = {
    "EMPTY_QUERY": "Please provide a question.",
    "NO_SEMANTIC_CONTENT": "Your input doesn’t contain a meaningful question.",
    "QUERY_TOO_SHORT": "Could you be a bit more specific?",
}

class QValidator:

    def _normalize_question(self, q: str) -> str:
        return q.strip()

    def _is_semantically_empty(self, q: str) -> bool:
        # only punctuation or symbols
        return not re.search(r"[a-zA-Z0-9]", q)

    def _is_too_short(self, q: str) -> bool:
        alpha_count = sum(c.isalnum() for c in q)
        return alpha_count < MIN_ALPHA_CHARS or len(q) < MIN_CHARS

    def validate_question(self, q: str) -> tuple[bool, str | None]:
        q = self._normalize_question(q)

        if not q:
            return False, "EMPTY_QUERY"

        if self._is_semantically_empty(q):
            return False, "NO_SEMANTIC_CONTENT"

        if self._is_too_short(q):
            return False, "QUERY_TOO_SHORT"

        return True, None
    
    def human_readable_message(self, error_code: str) -> str:
        return ERROR_MESSAGES.get(
            error_code,
            "Your request could not be processed."
        )