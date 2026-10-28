import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine

from core.chains import ChatEngine, strip_sources, trim_history
from infra.history import SQLiteChatHistory
from infra.settings import settings

DOC = Document(
    "Titel: Versicherungsschein - HUK\n\nBeitrag 25,54 EUR",
    metadata={"doc_id": 24, "title": "Versicherungsschein - HUK", "url": "http://p/documents/24/details"},
)


class RecordingModel(BaseChatModel):
    """Antwortet der Reihe nach mit `answers` und merkt sich, welche Nachrichten es bekam."""

    answers: list[str]
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        answer = self.answers[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(answer))])


@pytest.fixture
def engine_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    db = create_engine(f"sqlite:///{tmp_path / 'chat.db'}")
    model = RecordingModel(answers=["Antwort eins.", "Antwort zwei.", "Antwort drei."], calls=[])
    engine = ChatEngine(RunnableLambda(lambda q: [DOC]), model, lambda sid: SQLiteChatHistory(sid, db))
    return engine, model, db


def ask(engine, question, session="s"):
    return "".join(engine.stream(question, session))


def test_antwort_endet_mit_quellen_als_link(engine_and_model):
    engine, _, _ = engine_and_model

    text = ask(engine, "Was kostet die Haftpflicht?")

    assert text == "Antwort eins.\n\n**Quellen:** [Versicherungsschein - HUK](http://p/documents/24/details)"


def test_verlauf_speichert_antwort_ohne_quellen_und_quellen_getrennt(engine_and_model):
    engine, _, db = engine_and_model

    ask(engine, "Frage 1")

    human, ai = SQLiteChatHistory("s", db).messages
    assert human.content == "Frage 1"
    assert ai.content == "Antwort eins."
    assert ai.additional_kwargs["sources"] == ["[Versicherungsschein - HUK](http://p/documents/24/details)"]


def test_modell_sieht_im_verlauf_keine_quellenlinks(engine_and_model):
    engine, model, _ = engine_and_model

    ask(engine, "Frage 1")
    ask(engine, "Frage 2")

    second = model.calls[1]
    assert isinstance(second[0], SystemMessage)
    assert [m.content for m in second[1:3]] == ["Frage 1", "Antwort eins."]
    assert not any("Quellen" in m.content for m in second[1:-1])
    assert "NUTZER-FRAGE: Frage 2" in second[-1].content


def test_modell_bekommt_nur_die_letzten_nachrichten(engine_and_model, monkeypatch):
    engine, model, _ = engine_and_model
    monkeypatch.setattr(settings, "HISTORY_MAX_MESSAGES", 2)

    ask(engine, "Frage 1")
    ask(engine, "Frage 2")
    ask(engine, "Frage 3")

    history_part = [m.content for m in model.calls[2][1:-1]]
    assert history_part == ["Frage 2", "Antwort zwei."]


def test_sessions_bleiben_getrennt(engine_and_model):
    engine, model, _ = engine_and_model

    ask(engine, "Frage A", session="a")
    ask(engine, "Frage B", session="b")

    assert len(model.calls[1]) == 2  # nur System und neue Frage, kein Verlauf aus Session a


def test_alte_eintraege_mit_quellen_im_text_werden_bereinigt():
    old = AIMessage("Sie haben eine Haftpflicht.\n\n**Quellen:** [HUK](http://p/1); [Beitrag](http://p/2)")
    assert strip_sources(old.content) == "Sie haben eine Haftpflicht."
    assert strip_sources("Ohne Quellen.") == "Ohne Quellen."


def test_trim_history_beginnt_mit_einer_frage_und_wirft_zusatzdaten_weg():
    messages = [
        HumanMessage("F1"), AIMessage("A1"),
        HumanMessage("F2"), AIMessage("A2\n\n**Quellen:** [x](y)", additional_kwargs={"sources": ["[x](y)"]}),
    ]

    trimmed = trim_history(messages, 3)  # würde mit A1 beginnen

    assert [m.content for m in trimmed] == ["F2", "A2"]
    assert trimmed[1].additional_kwargs == {}
    assert trim_history(messages, 0) == []
