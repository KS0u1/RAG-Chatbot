import sys
import uuid
from pathlib import Path

import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infra.chroma_db import build_vector_store  # noqa: E402
from infra.settings import settings  # noqa: E402


@pytest.fixture
def store(monkeypatch):
    """Echte (In-Memory-)Chroma mit Fake-Embeddings, damit weder Ollama noch ein Server nötig sind."""
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", f"test-{uuid.uuid4().hex}")
    monkeypatch.setattr(settings, "PAPERLESS_PUBLIC_URL", "http://paperless.test")
    return build_vector_store(client=chromadb.EphemeralClient(), embedding=DeterministicFakeEmbedding(size=32))
