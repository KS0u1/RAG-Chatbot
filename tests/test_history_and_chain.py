from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine

from core.chains import _format_docs, _format_sources, _parse_score, qwen_rerank
from infra.history import SQLiteChatHistory
from infra.settings import settings


@pytest.fixture
def engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'chat.db'}")


def test_verlauf_speichert_nachrichten_mit_zeitstempel(engine):
    history = SQLiteChatHistory("a", engine)
    history.add_messages([HumanMessage("Hallo"), AIMessage("Hi, wie kann ich helfen?")])

    assert [(m.type, m.content) for m in history.messages] == [("human", "Hallo"), ("ai", "Hi, wie kann ich helfen?")]
    stamped = history.timestamped_messages()
    assert all(isinstance(ts, datetime) for _, ts in stamped)


def test_verlauf_trennt_sessions_und_loescht_nur_die_eigene(engine):
    a, b = SQLiteChatHistory("a", engine), SQLiteChatHistory("b", engine)
    a.add_messages([HumanMessage("A")])
    b.add_messages([HumanMessage("B")])

    a.clear()

    assert a.messages == []
    assert [m.content for m in b.messages] == ["B"]


def test_verlauf_legt_tabelle_beim_ersten_start_an(engine):
    # Frische Datenbank, noch kein Schreibzugriff: Lesen darf nicht mit "no such table" scheitern
    assert SQLiteChatHistory("neu", engine).messages == []


def test_verlauf_ueberlebt_neustart(tmp_path):
    url = f"sqlite:///{tmp_path / 'persist.db'}"
    SQLiteChatHistory("x", create_engine(url)).add_messages([HumanMessage("bleibt")])

    assert SQLiteChatHistory("x", create_engine(url)).messages[0].content == "bleibt"


@pytest.mark.parametrize("text,expected", [
    ("8", 0.8), ("Score: 7.5", 0.75), ("7,5", 0.75), ("kein Score", 0.0), ("15", 1.0), ("", 0.0),
])
def test_reranker_score_wird_robust_gelesen(text, expected):
    assert _parse_score(text) == pytest.approx(expected)


def test_quellen_pro_dokument_nur_einmal_und_als_link():
    docs = [
        Document("a", metadata={"doc_id": 1, "title": "Stromrechnung", "url": "http://p/documents/1/details"}),
        Document("b", metadata={"doc_id": 1, "title": "Stromrechnung", "url": "http://p/documents/1/details"}),
        Document("c", metadata={"doc_id": 2, "title": "Mietvertrag"}),
    ]

    assert _format_sources(docs) == ["[Stromrechnung](http://p/documents/1/details)", "Mietvertrag"]


def test_kontext_enthaelt_titel_datum_und_abschnitt():
    doc = Document("Inhalt", metadata={"title": "Brief", "created": "2026-01-02", "chunk_index": 0, "chunk_count": 3})

    text = _format_docs([doc])

    assert "Dokument: Brief" in text and "Datum: 2026-01-02" in text and "Abschnitt: 1/3" in text
    assert _format_docs([]) == "Keine relevanten Dokumente gefunden."


def test_rerank_aus_gibt_suchreihenfolge_zurueck(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    docs = [Document(str(i)) for i in range(6)]

    assert qwen_rerank("frage", docs, top_n=3) == docs[:3]


def test_rerank_fehler_faellt_auf_suchreihenfolge_zurueck(monkeypatch):
    class Kaputt:
        def invoke(self, prompt):
            raise ConnectionError("Ollama weg")

    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr("core.chains._get_reranker", lambda: Kaputt())
    docs = [Document(str(i)) for i in range(6)]

    assert qwen_rerank("frage", docs, top_n=2) == docs[:2]


def test_rerank_sortiert_nach_score(monkeypatch):
    scores = iter(["2", "9", "5"])

    class Fake:
        def invoke(self, prompt):
            return next(scores)

    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr("core.chains._get_reranker", lambda: Fake())
    docs = [Document("a"), Document("b"), Document("c")]

    assert [d.page_content for d in qwen_rerank("frage", docs, top_n=2)] == ["b", "c"]
