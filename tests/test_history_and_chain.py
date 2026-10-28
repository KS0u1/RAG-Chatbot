from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine

from core.chains import _format_docs, _format_sources
from infra.history import SQLiteChatHistory


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
