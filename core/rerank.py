"""Reranking der Suchtreffer mit Qwen3-Reranker über Ollama.

Qwen3-Reranker ist darauf trainiert, für ein Paar aus Frage und Dokument genau ein Token zu
antworten: "yes" oder "no". Der Score ist die Wahrscheinlichkeit für "yes" (siehe Modellkarte
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Deshalb:

- das Prompt-Format der Modellkarte, als `raw` an Ollama, damit keine fremde Vorlage dazukommt
- `num_predict: 1`, das Modell rechnet nur den Prompt und erzeugt ein einziges Token
- liefert Ollama Logprobs (neuere Versionen), wird daraus P(yes) berechnet, sonst zählt das
  Antwort-Token selbst (yes = 1, no = 0)
"""

import logging
import math

import requests
from langchain_core.documents import Document

from infra.settings import settings

log = logging.getLogger(__name__)

INSTRUCTION = (
    "Given a question about the user's personal documents (invoices, contracts, payslips, "
    "official notices, certificates), judge whether the document passage contains information "
    "that helps answer the question"
)

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

MAX_DOC_CHARS = 1500  # Abschnitte sind ~1000 Zeichen plus Kopfzeile, das ist nur eine Sicherung
NEUTRAL = 0.5  # unklare Antwort: weder nach oben noch nach unten schieben


def build_prompt(query: str, document: str) -> str:
    return (
        f"{_PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n"
        f"<Document>: {document[:MAX_DOC_CHARS]}{_SUFFIX}"
    )


def score_from_response(data: dict) -> float:
    """Score 0.0 bis 1.0 aus der Antwort von /api/generate."""
    logprobs = data.get("logprobs") or []
    if logprobs:
        first = logprobs[0]
        candidates = first.get("top_logprobs") or [first]
        p_yes = sum(math.exp(c["logprob"]) for c in candidates if c.get("token", "").strip().lower() == "yes")
        p_no = sum(math.exp(c["logprob"]) for c in candidates if c.get("token", "").strip().lower() == "no")
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)

    text = (data.get("response") or "").strip().lower()
    if text.startswith("yes"):
        return 1.0
    if text.startswith("no"):
        return 0.0
    return NEUTRAL


def score(query: str, document: str, session: requests.Session | None = None) -> float:
    """Fragt das Reranker-Modell nach der Relevanz eines Abschnitts. Wirft bei Verbindungsfehlern."""
    http = session or requests
    response = http.post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        json={
            "model": settings.OLLAMA_RERANK_MODEL,
            "prompt": build_prompt(query, document),
            "raw": True,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": 2048},
            # Ältere Ollama-Versionen ignorieren diese Felder, dann zählt das Antwort-Token
            "logprobs": True,
            "top_logprobs": 5,
        },
        timeout=settings.RERANK_TIMEOUT,
    )
    response.raise_for_status()
    return score_from_response(response.json())


def rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Sortiert die Treffer nach Relevanz neu. Fällt bei Fehlern auf die Suchreihenfolge zurück."""
    if not docs:
        return []
    if not settings.RERANK_ENABLED:
        return docs[:top_n]

    try:
        with requests.Session() as session:
            scored = [(doc, score(query, doc.page_content, session)) for doc in docs]
    except Exception as e:
        log.warning("Reranking fehlgeschlagen, nutze Suchreihenfolge: %s", e)
        return docs[:top_n]

    log.info("Rerank-Scores: %s", ", ".join(f"{s:.2f}" for _, s in scored))
    scored.sort(key=lambda x: x[1], reverse=True)  # stabil: bei Gleichstand bleibt die Suchreihenfolge
    return [doc for doc, _ in scored[:top_n]]
