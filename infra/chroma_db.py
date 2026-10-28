"""Anbindung an ChromaDB (Vektordatenbank) und die Ollama-Embeddings."""

import logging
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
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


class PrefixedEmbeddings(Embeddings):
    """Setzt vor Fragen und Dokumente das Präfix, das das Embedding-Modell erwartet.

    nomic-embed-text ist darauf trainiert, "search_query: " vor Suchanfragen und
    "search_document: " vor Dokumenten zu sehen. Ohne Präfix liegen Frage und passende
    Abschnitte im Vektorraum weiter auseinander. In Chroma landet nur der Text ohne Präfix.
    """

    def __init__(self, inner: Embeddings, query_prefix: str = "", doc_prefix: str = ""):
        self.inner = inner
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def embed_prefixes(model: str) -> tuple[str, str]:
    """(Frage-Präfix, Dokument-Präfix). Gesetzte Umgebungsvariablen haben Vorrang, auch leere."""
    if "nomic-embed" in model.lower():
        query, doc = "search_query: ", "search_document: "
    else:
        query, doc = "", ""
    if settings.EMBED_QUERY_PREFIX is not None:
        query = settings.EMBED_QUERY_PREFIX
    if settings.EMBED_DOC_PREFIX is not None:
        doc = settings.EMBED_DOC_PREFIX
    return query, doc


def get_embeddings() -> Embeddings:
    query_prefix, doc_prefix = embed_prefixes(settings.OLLAMA_EMBED_MODEL)
    return PrefixedEmbeddings(
        OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBED_MODEL),
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
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
