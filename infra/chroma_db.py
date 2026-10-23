"""Anbindung an ChromaDB (Vektordatenbank) und die Ollama-Embeddings."""

import logging
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from .settings import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client():
    """HTTP-Client zum Chroma-Server. Wirft, wenn der Server nicht erreichbar ist
    (Fehler werden von lru_cache nicht zwischengespeichert, der nächste Aufruf versucht es neu)."""
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        ssl=settings.CHROMA_SSL,
    )
    client.heartbeat()  # prüft wirklich, ob der Server antwortet
    log.info("Verbunden mit ChromaDB %s:%s", settings.CHROMA_HOST, settings.CHROMA_PORT)
    return client


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_EMBED_MODEL,
    )


def build_vector_store(client=None, embedding=None) -> Chroma:
    """Baut den Vektorspeicher. `client` und `embedding` lassen sich für Tests austauschen."""
    return Chroma(
        client=client or get_client(),
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=embedding or get_embeddings(),
        # Kosinus-Abstand passt besser zu Text-Embeddings als der Chroma-Standard (L2)
        collection_metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return build_vector_store()
