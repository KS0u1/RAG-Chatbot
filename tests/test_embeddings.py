import chromadb
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from infra.chroma_db import PrefixedEmbeddings, build_vector_store, embed_prefixes
from infra.settings import settings


class Recording(Embeddings):
    def __init__(self):
        self.seen: list[str] = []
        self.fake = DeterministicFakeEmbedding(size=8)

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return self.fake.embed_documents(texts)

    def embed_query(self, text):
        self.seen.append(text)
        return self.fake.embed_query(text)


def test_praefixe_fuer_frage_und_dokument():
    inner = Recording()
    emb = PrefixedEmbeddings(inner, query_prefix="search_query: ", doc_prefix="search_document: ")

    emb.embed_documents(["Rechnung"])
    emb.embed_query("Was kostet Strom?")

    assert inner.seen == ["search_document: Rechnung", "search_query: Was kostet Strom?"]


def test_nomic_bekommt_seine_praefixe_andere_modelle_keine(monkeypatch):
    monkeypatch.setattr(settings, "EMBED_QUERY_PREFIX", None)
    monkeypatch.setattr(settings, "EMBED_DOC_PREFIX", None)

    assert embed_prefixes("nomic-embed-text-v2-moe") == ("search_query: ", "search_document: ")
    assert embed_prefixes("mxbai-embed-large") == ("", "")


def test_gesetzte_umgebungsvariablen_haben_vorrang_auch_leer(monkeypatch):
    monkeypatch.setattr(settings, "EMBED_QUERY_PREFIX", "")
    monkeypatch.setattr(settings, "EMBED_DOC_PREFIX", "passage: ")

    assert embed_prefixes("nomic-embed-text-v2-moe") == ("", "passage: ")


def test_in_chroma_steht_der_text_ohne_praefix(monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "test-praefix")
    inner = Recording()
    store = build_vector_store(
        client=chromadb.EphemeralClient(),
        embedding=PrefixedEmbeddings(inner, "search_query: ", "search_document: "),
    )

    store.add_texts(["Stromrechnung Januar"], ids=["x"])
    hits = store.similarity_search("Strom", k=1)

    assert store.get(ids=["x"])["documents"] == ["Stromrechnung Januar"]
    assert hits[0].page_content == "Stromrechnung Januar"
    assert "search_document: Stromrechnung Januar" in inner.seen
    assert "search_query: Strom" in inner.seen
