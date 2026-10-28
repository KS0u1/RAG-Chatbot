import logging
import re
from datetime import datetime
from typing import Callable, Iterator

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from infra.chroma_db import get_vector_store
from infra.db import get_engine
from infra.history import SQLiteChatHistory
from infra.settings import settings

from .llm import get_llm
from .rerank import rerank

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist ein persönlicher Dokumenten-Assistent. Du beantwortest Fragen zu den Dokumenten \
des Nutzers aus seinem Paperless-System, zum Beispiel Rechnungen, Verträge, Bescheide, Briefe und Handbücher.
AKTUELLES DATUM: {datum} (Nutze es für zeitbezogene Antworten wie "heute", "dieses Jahr" oder Fristen. \
Nenne das Datum nur, wenn danach gefragt wird oder es für die Antwort nötig ist.)

DEINE OBERSTE PRIORITÄT: KORREKTHEIT VOR SCHNELLIGKEIT.

REGELN:
1. Nutze für Fragen zu Dokumenten NUR den mitgelieferten Kontext. Erfinde keine Fakten und keine Quellen.
2. Die Texte stammen aus Texterkennung (OCR). Ignoriere Erkennungsfehler, etwa "8 4" statt "§ 4" oder \
"Diestwagen" statt "Dienstwagen", und versuche den Sinn zu verstehen.
3. Nenne Beträge, Daten, Fristen und Nummern genau so, wie sie im Dokument stehen, mit Währung und Einheit.
4. Passen mehrere Dokumente zur Frage (z. B. Rechnungen aus verschiedenen Monaten), nenne sie getrennt \
mit Datum und Absender, statt sie zu vermischen.
5. Nennt das Dokument eine andere Einheit oder einen anderen Zeitraum als die Frage (z. B. "pro Jahr" statt \
"im Monat"), antworte mit dem, was im Dokument steht. Sage nicht "dazu steht nichts", nur weil die Einheit abweicht.
6. Reicht der Kontext nicht aus, sage klar, dass du es in den Dokumenten nicht findest. Nenne, was du gefunden \
hast, und stelle höchstens 1-2 Rückfragen.
7. Antworte auf Deutsch, knapp und zuerst mit der konkreten Antwort (Zahl, Datum, Regel).
8. Schreibe KEINEN Quellen-Abschnitt am Ende. Das System fügt die Quellen automatisch hinzu."""


def _history(session_id: str) -> SQLiteChatHistory:
    return SQLiteChatHistory(session_id, get_engine())


def _format_docs(docs: list[Document]) -> str:
    """Formatiert die gefundenen Textabschnitte für den Prompt."""
    if not docs:
        return "Keine relevanten Dokumente gefunden."

    formatted = []
    for d in docs:
        meta = d.metadata or {}
        parts = [f"Dokument: {meta.get('title') or meta.get('source', 'unbekannt')}"]
        if meta.get("created"):
            parts.append(f"Datum: {meta['created']}")
        if meta.get("chunk_index") is not None:
            parts.append(f"Abschnitt: {meta['chunk_index'] + 1}/{meta.get('chunk_count', '?')}")
        formatted.append(" | ".join(parts) + f"\n{d.page_content}")

    return "\n\n---\n\n".join(formatted)


def _format_sources(docs: list[Document]) -> list[str]:
    """Eine Quelle pro Dokument (nicht pro Abschnitt), als Markdown-Link, falls eine URL bekannt ist."""
    seen: set = set()
    sources = []
    for d in docs:
        meta = d.metadata or {}
        key = meta.get("doc_id", meta.get("source"))
        title = meta.get("title") or meta.get("source")
        if not title or key in seen:
            continue
        seen.add(key)
        url = meta.get("url")
        sources.append(f"[{title}]({url})" if url else title)
    return sources


# Ältere Versionen haben die Quellenzeile mit in die gespeicherte Antwort geschrieben.
_SOURCES_RE = re.compile(r"\n*\*\*Quellen:\*\*.*\Z", re.DOTALL)


def strip_sources(text: str) -> str:
    return _SOURCES_RE.sub("", text).rstrip()


def trim_history(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Die letzten `max_messages` Nachrichten für den Prompt, ohne Quellenzeilen und Zusatzdaten.

    Das Modell bekommt so nur das Gespräch selbst. Sieht es frühere Quellenlinks, schreibt es sie
    nach, und ein unbegrenzter Verlauf macht jede Antwort langsamer und sprengt irgendwann das
    Kontextfenster.
    """
    if max_messages <= 0:
        return []
    recent = messages[-max_messages:]
    while recent and recent[0].type != "human":  # nicht mitten in einem Frage-Antwort-Paar beginnen
        recent = recent[1:]
    cleaned: list[BaseMessage] = []
    for m in recent:
        if m.type == "human":
            cleaned.append(HumanMessage(m.content))
        elif m.type == "ai":
            cleaned.append(AIMessage(strip_sources(m.content)))
    return cleaned


def sources_line(sources: list[str]) -> str:
    return "\n\n**Quellen:** " + "; ".join(sources) if sources else ""


PROMPT = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    HumanMessagePromptTemplate.from_template("KONTEXT AUS DOKUMENTEN:\n{context}\n\nNUTZER-FRAGE: {input}"),
])


class ChatEngine:
    """Suche, Reranking, Antwort und Verlauf.

    Die Antwort wird gestreamt, danach folgt die Quellenzeile für die Anzeige. Im Verlauf
    landet nur die Antwort, die Quellen stehen getrennt in `additional_kwargs["sources"]`.
    """

    def __init__(self, retriever, llm, history_factory: Callable[[str], BaseChatMessageHistory]):
        self.retriever = retriever
        self.generation_chain = PROMPT | llm | StrOutputParser()
        self.history_factory = history_factory

    def retrieve(self, question: str) -> list[Document]:
        docs = self.retriever.invoke(question)
        top = rerank(question, docs, top_n=settings.RERANK_TOP_N)
        log.info("Frage: %r, %d Treffer, %d nach Reranking", question, len(docs), len(top))
        return top

    def stream(self, question: str, session_id: str) -> Iterator[str]:
        history = self.history_factory(session_id)
        past = trim_history(history.messages, settings.HISTORY_MAX_MESSAGES)
        docs = self.retrieve(question)
        sources = _format_sources(docs)

        inputs = {
            "input": question,
            "history": past,
            "context": _format_docs(docs),
            # Zur Laufzeit statt beim Start: Die App läuft als Dienst tagelang durch.
            "datum": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        parts: list[str] = []
        for chunk in self.generation_chain.stream(inputs):
            parts.append(chunk)
            yield chunk

        answer = "".join(parts).strip()
        history.add_messages([
            HumanMessage(question),
            AIMessage(answer, additional_kwargs={"sources": sources}),
        ])
        yield sources_line(sources)


def build_chain() -> ChatEngine:
    retriever = get_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_K,
            "fetch_k": max(settings.RETRIEVER_FETCH_K, settings.RETRIEVER_K),
            "lambda_mult": 0.9,
        },
    )
    return ChatEngine(retriever, get_llm(), _history)
