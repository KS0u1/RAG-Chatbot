"""Zentrale Konfiguration. Alle Werte kommen aus Umgebungsvariablen (oder einer .env)."""

import os

from dotenv import load_dotenv

load_dotenv()  # lädt Werte aus .env in os.environ, falls vorhanden


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, str(default)).lower() in {"1", "true", "yes", "ja", "on"}


class Settings:
    # --- Ollama ---
    OLLAMA_BASE_URL = _str("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = _str("OLLAMA_MODEL", "llama3.2:1b")
    OLLAMA_TEMPERATURE = _float("OLLAMA_TEMPERATURE", 0.0)
    OLLAMA_EMBED_MODEL = _str("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
    OLLAMA_RERANK_MODEL = _str("OLLAMA_RERANK_MODEL", "dengcao/Qwen3-Reranker-0.6B:Q8_0")
    # Kontextfenster des Sprachmodells in Tokens. Ollamas Standard (je nach Version 2048-4096) ist für
    # Systemprompt + Verlauf + Dokumentabschnitte zu knapp, dann schneidet Ollama still Text ab.
    # Größer kostet RAM (bei 8B-Modellen grob 0,13 MB pro Token).
    OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 6144)
    # Denkmodus für Modelle wie qwen3: "false" schaltet ihn ab (kein <think>-Text in der Antwort),
    # "true" an, leer = Standard des Modells. Bei Modellen ohne Denkmodus leer lassen.
    OLLAMA_REASONING = _str("OLLAMA_REASONING", "")

    # Embedding-Präfixe. Leer gelassen wählt der Code sie passend zum Modell (nomic-embed braucht
    # "search_query: " und "search_document: "). Ändert sich das Präfix, liest der Sync alles neu ein.
    EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX")
    EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX")

    # --- ChromaDB ---
    CHROMA_HOST = _str("CHROMA_HOST", "localhost")
    CHROMA_PORT = _int("CHROMA_PORT", 8000)
    CHROMA_SSL = _bool("CHROMA_SSL", False)
    CHROMA_COLLECTION = _str("CHROMA_COLLECTION", "paperless")

    # --- Chatverlauf ---
    SQLITE_URL = _str("SQLITE_URL", "sqlite:///chat_history.db")
    # So viele der letzten Nachrichten (Fragen und Antworten) bekommt das Modell als Gesprächskontext
    HISTORY_MAX_MESSAGES = _int("HISTORY_MAX_MESSAGES", 6)

    # --- Paperless-ngx ---
    PAPERLESS_URL = _str("PAPERLESS_URL").rstrip("/")  # z. B. http://192.168.178.64:8010
    PAPERLESS_TOKEN = _str("PAPERLESS_TOKEN")
    # URL, unter der du Paperless im Browser öffnest (für Links in den Quellen)
    PAPERLESS_PUBLIC_URL = _str("PAPERLESS_PUBLIC_URL", _str("PAPERLESS_URL")).rstrip("/")
    SYNC_INTERVAL_MIN = _int("SYNC_INTERVAL_MIN", 30)

    # --- Retrieval ---
    RETRIEVER_K = _int("RETRIEVER_K", 8)
    RETRIEVER_FETCH_K = _int("RETRIEVER_FETCH_K", 20)
    RERANK_ENABLED = _bool("RERANK_ENABLED", True)
    RERANK_TOP_N = _int("RERANK_TOP_N", 4)
    RERANK_TIMEOUT = _int("RERANK_TIMEOUT", 60)  # Sekunden pro Abschnitt

    def reasoning(self) -> bool | None:
        value = self.OLLAMA_REASONING.lower()
        if value in {"1", "true", "yes", "ja", "on"}:
            return True
        if value in {"0", "false", "no", "nein", "off"}:
            return False
        return None

    def paperless_configured(self) -> bool:
        return bool(self.PAPERLESS_URL and self.PAPERLESS_TOKEN)


settings = Settings()
