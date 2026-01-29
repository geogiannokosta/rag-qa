import pytest

# ------------------ Fakes ------------------

class FakeEmbedder:
    def embed(self, text: str):
        return [0.1] * 8  


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakePdf:
    def __init__(self, pages):
        self.pages = [FakePage(p) for p in pages]


def fake_pdf_reader(path):
    """Simulate PdfReader(path)"""
    return FakePdf([
        "First page content. First page content.",
        "Second page content. Second page content."
    ])


# ------------------ Fixtures ------------------

@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def pdf_reader():
    return fake_pdf_reader

@pytest.fixture
def simple_docs(tmp_path):
    """Create temporary PDF-like files for testing."""
    docs = []
    for i, content in enumerate([
        FakePdf([
        "First page content. First page content.",
        "Second page content. Second page content."
        ])
    ]):
        file_path = tmp_path / f"doc{i}.pdf"
        file_path.write_text(content.pages[0].extract_text() + "\n" + content.pages[1].extract_text())
        docs.append(str(file_path))
    return docs