import logging
import re
from datetime import datetime
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

from infra.chroma_db import get_vector_store
from infra.db import get_engine
from infra.history import SQLiteChatHistory
from infra.settings import settings

from .llm import get_llm

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


@lru_cache(maxsize=1)
def _get_reranker() -> OllamaLLM:
    return OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_RERANK_MODEL,
        temperature=0.0,
    )


def _parse_score(text: str) -> float:
    """Liest die erste Zahl aus der Modellantwort, auch wenn Text drumherum steht. Ergebnis 0.0 bis 1.0."""
    match = re.search(r"\d+(?:[.,]\d+)?", text)
    if not match:
        return 0.0
    return min(max(float(match.group(0).replace(",", ".")), 0.0), 10.0) / 10.0


def qwen_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Sortiert die Treffer mit dem Reranker-Modell neu. Fällt bei Fehlern auf die Suchreihenfolge zurück."""
    if not docs:
        return []
    if not settings.RERANK_ENABLED:
        return docs[:top_n]

    try:
        reranker = _get_reranker()
        scored = []
        for doc in docs:
            prompt = (
                "Bewerte die Relevanz dieses Dokuments zur Query (nur Score 0-10 als Zahl).\n\n"
                f"Query: {query}\n"
                f"Document: {doc.page_content[:1500]}\n\n"
                "Relevanz-Score (0-10):"
            )
            scored.append((doc, _parse_score(reranker.invoke(prompt))))
    except Exception as e:
        log.warning("Reranking fehlgeschlagen, nutze Suchreihenfolge: %s", e)
        return docs[:top_n]

    scored.sort(key=lambda x: x[1], reverse=True)  # stabil: bei Gleichstand bleibt die Suchreihenfolge
    return [doc for doc, _ in scored[:top_n]]


def build_chain():
    retriever = get_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.RETRIEVER_K,
            "fetch_k": max(settings.RETRIEVER_FETCH_K, settings.RETRIEVER_K),
            "lambda_mult": 0.9,
        },
    )

    prompt = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        HumanMessagePromptTemplate.from_template(
            "KONTEXT AUS DOKUMENTEN:\n{context}\n\nNUTZER-FRAGE: {input}"
        ),
    ])
    generation_chain = prompt | get_llm() | StrOutputParser()

    def add_context(inputs: dict) -> dict:
        """Holt passende Textabschnitte aus Chroma und ergänzt Kontext, Quellen und Datum."""
        query = inputs["input"]
        docs = retriever.invoke(query)
        top = qwen_rerank(query, docs, top_n=settings.RERANK_TOP_N)
        log.info("Frage: %r, %d Treffer, %d nach Reranking", query, len(docs), len(top))
        return {
            **inputs,
            "context": _format_docs(top),
            "sources": _format_sources(top),
            # Zur Laufzeit statt beim Start: Die App läuft als Dienst tagelang durch.
            "datum": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def stream_with_sources(inputs: dict):
        """Streamt die Antwort und hängt danach die Quellen an."""
        yield from generation_chain.stream(inputs)
        if inputs.get("sources"):
            yield "\n\n**Quellen:** " + "; ".join(inputs["sources"])

    return RunnableWithMessageHistory(
        RunnableLambda(add_context) | RunnableLambda(stream_with_sources),
        _history,
        input_messages_key="input",
        history_messages_key="history",
    )
