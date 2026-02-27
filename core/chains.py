import re

from datetime import datetime
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import OllamaLLM

from .llm import get_llm
from infra.db import get_engine
from infra.chroma_db import get_vector_store
from infra.settings import settings

engine = get_engine()


def _history(sid):
    return SQLChatMessageHistory(session_id=sid, connection=engine)


def _format_docs(docs: list[Document]) -> str:
    """Formatiert die gefundenen Dokumente für den Prompt."""
    if not docs:
        return "Keine relevanten Dokumente gefunden."

    formatted = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "unbekannt")
        dateityp = meta.get("Dateityp", None)
        chunk_index = meta.get("chunk_index", None)

        meta_parts = [f"Quelle: {source}"]
        if dateityp:
            meta_parts.append(f"Dateityp: {dateityp}")
        if chunk_index is not None:
            meta_parts.append(f"Chunk: {chunk_index}")

        meta_str = " | ".join(meta_parts)
        formatted.append(f"{meta_str}\n{d.page_content}")

    return "\n\n---\n\n".join(formatted)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _append_sources(answer: str, sources: list[str]) -> str:
    if not sources:
        return answer

    # Falls das Modell selbst schon einen Quellenblock generiert: entferne ihn
    cleaned = re.sub(r"\n?Quellen:\s.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).rstrip()
    return cleaned + "\n\nQuellen: " + "; ".join(sources)

def qwen_rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """Rerankt Docs mit Qwen3-Reranker über Ollama."""
    reranker = OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_RERANK_MODEL,
        temperature=0.0
    )
    scored_docs = []

    for doc in docs:
        prompt = f"""Bewerte die Relevanz dieses Dokuments zur Query (nur Score 0-10 als Zahl).

    Query: {query}
    Document: {doc.page_content[:1500]}

    Relevanz-Score (0-10):"""

        score_str = reranker.invoke(prompt).strip()
        # Extrahiere die erste Zahl aus dem String, falls das Modell Text mitliefert
        match = re.search(r"(\d+(\.\d+)?)", score_str)
        try:
            score = float(match.group(1)) / 10.0 if match else 0.0
        except (ValueError, AttributeError):
            score = 0.0

        scored_docs.append((doc, score))

    # Sortiere nach Score und nimm top_n
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]

def build_chain():
    # Vektorstore initialisieren und Retriever erstellen
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.9}
    )

    aktuelles_datum = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    system_instruction = f"""Du bist ein KI-Assistent für Mitarbeiter, dabei hast du Zugriff auf Dokumente aus dem SharePoint.
    AKTUELLES DATUM: {aktuelles_datum} (Nutze dieses Datum für alle zeitbezogenen Antworten wie "heute" oder "dieses Jahr").
    Datum nur angeben, wenn gefragt wird.
    
    DEINE OBERSTE PRIORITÄT: KORREKTHEIT VOR SCHNELLIGKEIT.

    REGELN:
    1. Nutze für fachliche Fragen NUR die vorhandenen Quellen.
    2. ZEIT-EINHEITEN: Wenn der Nutzer nach einer Einheit fragt (z.B. "im Monat"), das Dokument aber eine andere nennt (z.B. "pro Woche" oder "pro Jahr"), dann antworte mit der Einheit aus dem Dokument. Sag nicht "dazu steht nichts", nur weil die Einheit abweicht.
    3. FEHLERTOLERANZ: Ignoriere Scanner-Fehler in den Texten (z.B. wenn "8 4" statt "§ 4" steht oder "Diestwagen" statt "Dienstwagen"). Versuche den Sinn zu verstehen.
    4. AUSNAHME: Wenn nach dem aktuellen Datum oder der Uhrzeit gefragt wird, nutze die Info "AKTUELLES DATUM" aus diesem Prompt, auch wenn es nicht in den Dokumenten steht.
    5. Erfinde keine Quellen und erfinde keine Fakten.
    6. Antworte mit dem, was sicher aus dem Kontext ableitbar ist, und stelle dann 1-2 Rückfragen für die fehlenden Infos.
    7. Wenn du etwas nicht beantworten kannst, dann antworte mit ich "weiß es nicht".
    
    REGELN ZUR QUELLEN-PRIORISIERUNG:
    1. THEMEN-CHECK: Prüfe zuerst, worum es geht.
        - Geht es um **Personal/HR** (Gehalt, Urlaub, Studenten, Abschlussarbeit)? -> Nutze NUR Quellen wie "Information", "Betriebsvereinbarung" oder "Leitfaden".
        - Geht es um **Prozesse/Einkauf** (Vergabe, Beschaffung)? -> Nutze "Direktiven" oder "Prozessanweisungen".

    2. KONFLIKT-LÖSUNG: 
        - Wenn ein Dokument "Vergabe" oder "Einkauf" heißt, gelten die Zahlen darin NICHT für Gehälter oder Studentenvergütungen.
        - Beispiel: "Vergütung" in einer Vergabe-Richtlinie betrifft Firmen, nicht Studenten. Ignoriere diese Zahlen für Personalfragen.

    3. ANTWORT-STRUKTUR:
        - Nenne zuerst die konkrete Antwort (Zahl, Frist, Regel).
        - Schreibe KEINEN Quellen-Abschnitt am Ende. Das System fügt die Quellen automatisch hinzu.
    """

    # Prompt mit Kontext + History
    msgs = [
        SystemMessagePromptTemplate.from_template(
            system_instruction
        ),
        MessagesPlaceholder("history"),
        HumanMessagePromptTemplate.from_template(
            "KONTEXT AUS DOKUMENTEN:\n{context}\n\n"
            "NUTZER-FRAGE: {input}"
        ),
    ]

    prompt = ChatPromptTemplate(messages=msgs)

    # Kontext aus Chroma holen und in das Input-Dict einfügen
    def add_context(inputs: dict) -> dict:
        """
        inputs kommt von RunnableWithMessageHistory und enthält:
        - 'input': die Nutzerfrage (String)
        - 'history': die bisherigen Chatnachrichten
        Wir fügen 'context' hinzu, der aus dem Retriever kommt.
        """
        query = inputs["input"]

        # Initiale Docs holen
        docs = retriever.invoke(query)

        # Rerank anwenden
        reranked_docs = qwen_rerank(query, docs, top_n=4)

        # Context formatieren
        context = _format_docs(reranked_docs)

        # Sources extrahieren
        sources = []
        for d in reranked_docs:
            meta = d.metadata or {}
            src = meta.get("source")
            if src:
                sources.append(src)
        sources = _dedupe_keep_order(sources)

        print(context)
        print(inputs)
        print(f"Reranked {len(reranked_docs)} Docs (von {len(docs)} initial)")
        return {**inputs, "context": context, "sources": sources}

    llm_generation_chain = (
            prompt
            | get_llm()
            | StrOutputParser()
    )

    def stream_with_sources(input_dict: dict):
        """
        Diese Funktion streamt erst die LLM-Antwort und
        hängt danach die Quellen an.
        """
        # LLM Antwort streamen
        for chunk in llm_generation_chain.stream(input_dict):
            yield chunk

        # Quellen anhängen (als letzter Chunk)
        sources = input_dict.get("sources", [])
        if sources:
            source_text = "\n\n**Quellen:** " + "; ".join(sources)
            yield source_text

    rag_chain = (
            RunnableLambda(add_context)
            | RunnableLambda(stream_with_sources)
    )

    return RunnableWithMessageHistory(
        rag_chain,
        _history,
        input_messages_key="input",
        history_messages_key="history",
    )
