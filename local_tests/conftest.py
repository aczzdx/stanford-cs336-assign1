import pytest as pt
from pathlib import Path

@pt.fixture
def small_corpus_path(request: pt.FixtureRequest):
    print(f"resolving cur path for small_corpus: {request.path}")
    test_parent = Path(request.path).parent
    yield test_parent / "data" / "small.txt"
