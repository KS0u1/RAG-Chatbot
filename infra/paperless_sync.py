"""Gleicht Paperless-ngx mit der Chroma-Collection ab.

Nur neue oder geänderte Dokumente werden neu eingelesen, gelöschte werden entfernt.
Aufruf als Dienst:   python -m infra.paperless_sync --loop
Einmalig:            python -m infra.paperless_sync
Alles neu aufbauen:  python -m infra.paperless_sync --full
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .paperless import PaperlessClient, PaperlessError
from .settings import settings

log = logging.getLogger(__name__)

SOURCE_TYPE = "paperless"
BATCH_SIZE = 64
MAX_CONSECUTIVE_FAILURES = 3

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

ProgressCallback = Callable[[int, int, str], None]


class SyncError(RuntimeError):
    """Der Abgleich musste abgebrochen werden (z. B. Ollama oder Chroma nicht erreichbar)."""


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped_empty: int = 0
    failed: int = 0
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{self.added} neu, {self.updated} aktualisiert, {self.deleted} gelöscht, "
            f"{self.unchanged} unverändert, {self.skipped_empty} ohne Text, {self.failed} fehlgeschlagen "
            f"({self.chunks} Chunks geschrieben)"
        )


def _sync_key(modified: str, embed_model: str) -> str:
    # Wechselt das Embedding-Modell, passen alte Vektoren nicht mehr zu neuen Anfragen.
    # Mit dem Modell im Schlüssel werden dann automatisch alle Dokumente neu eingelesen.
    return f"{modified}|{embed_model}"


def build_chunks(doc: dict, names: dict[str, dict[int, str]]) -> tuple[list[str], list[str], list[dict]]:
    """Zerlegt ein Paperless-Dokument in Chunks (ids, texte, metadaten)."""
    content = (doc.get("content") or "").strip()
    if not content:
        return [], [], []

    doc_id = int(doc["id"])
    title = doc.get("title") or f"Dokument {doc_id}"
    correspondent = names["correspondents"].get(doc.get("correspondent"), "")
    doc_type = names["document_types"].get(doc.get("document_type"), "")
    tags = [names["tags"][t] for t in doc.get("tags", []) if t in names["tags"]]
    created = str(doc.get("created") or doc.get("created_date") or "")[:10]

    # Dieser Kopf steht in jedem Chunk. So findet die Suche auch Fragen wie
    # "Stromrechnung 2025", obwohl das Wort im jeweiligen Textabschnitt gar nicht vorkommt.
    header_lines = [f"Titel: {title}"]
    if correspondent:
        header_lines.append(f"Korrespondent: {correspondent}")
    if doc_type:
        header_lines.append(f"Dokumenttyp: {doc_type}")
    if created:
        header_lines.append(f"Datum: {created}")
    if tags:
        header_lines.append(f"Tags: {', '.join(tags)}")
    header = "\n".join(header_lines)

    parts = text_splitter.split_text(content)
    url = f"{settings.PAPERLESS_PUBLIC_URL}/documents/{doc_id}/details" if settings.PAPERLESS_PUBLIC_URL else ""
    key = _sync_key(str(doc.get("modified", "")), settings.OLLAMA_EMBED_MODEL)

    ids, texts, metadatas = [], [], []
    for i, part in enumerate(parts):
        meta = {
            "type": SOURCE_TYPE,
            "doc_id": doc_id,
            "source": title,
            "title": title,
            "url": url,
            "created": created,
            "correspondent": correspondent,
            "document_type": doc_type,
            "tags": ", ".join(tags),
            "sync_key": key,
            "chunk_index": i,
            "chunk_count": len(parts),
        }
        ids.append(f"paperless-{doc_id}-{i}")
        texts.append(f"{header}\n\n{part}")
        metadatas.append({k: v for k, v in meta.items() if v not in (None, "")})
    return ids, texts, metadatas


def _load_state(vector_store) -> dict[int, dict]:
    """Was steht aktuell in Chroma? doc_id -> {ids, keys, counts}"""
    result = vector_store.get(where={"type": SOURCE_TYPE}, include=["metadatas"])
    state: dict[int, dict] = {}
    for chunk_id, meta in zip(result["ids"], result["metadatas"]):
        entry = state.setdefault(int(meta["doc_id"]), {"ids": [], "keys": set(), "counts": set()})
        entry["ids"].append(chunk_id)
        entry["keys"].add(meta.get("sync_key"))
        entry["counts"].add(meta.get("chunk_count"))
    return state


def _is_current(entry: dict | None, key: str) -> bool:
    """Aktuell heißt: alle Chunks tragen den erwarteten Schlüssel und sind vollständig.
    Ein halb geschriebenes Dokument (z. B. nach einem Abbruch) wird so erkannt und repariert."""
    return bool(
        entry
        and entry["keys"] == {key}
        and entry["counts"] == {len(entry["ids"])}
    )


def _write_batches(vector_store, ids, texts, metadatas):
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        vector_store.add_texts(texts=texts[start:end], metadatas=metadatas[start:end], ids=ids[start:end])


def sync_paperless(
    vector_store,
    client: PaperlessClient | None = None,
    full: bool = False,
    progress: ProgressCallback | None = None,
) -> SyncResult:
    client = client or PaperlessClient(settings.PAPERLESS_URL, settings.PAPERLESS_TOKEN)
    result = SyncResult()

    remote = {stub.id: stub.modified for stub in client.list_document_stubs()}
    state = _load_state(vector_store)
    log.info("Paperless: %d Dokumente, Chroma: %d bereits indexiert", len(remote), len(state))

    # --- Gelöschte Dokumente entfernen ---
    gone = [doc_id for doc_id in state if doc_id not in remote]
    if gone and not remote:
        # Paperless meldet gar nichts, Chroma kennt aber Dokumente. Das ist fast sicher ein
        # Konfigurationsfehler (falscher Benutzer, fehlende Rechte) und kein leeres Archiv.
        log.warning("Paperless lieferte 0 Dokumente, Chroma enthält %d. Löschen übersprungen.", len(state))
    else:
        for doc_id in gone:
            vector_store.delete(ids=state[doc_id]["ids"])
            result.deleted += 1

    # --- Neue und geänderte Dokumente ---
    todo = []
    for doc_id, modified in remote.items():
        key = _sync_key(modified, settings.OLLAMA_EMBED_MODEL)
        if full or not _is_current(state.get(doc_id), key):
            todo.append(doc_id)
        else:
            result.unchanged += 1

    if todo:
        names = {
            "tags": client.name_map("tags"),
            "correspondents": client.name_map("correspondents"),
            "document_types": client.name_map("document_types"),
        }

    consecutive_failures = 0
    for n, doc_id in enumerate(todo, start=1):
        try:
            doc = client.get_document(doc_id)
            ids, texts, metadatas = build_chunks(doc, names)
            existing = state.get(doc_id)

            if not ids:
                # Kein Text (z. B. OCR noch nicht fertig): alte Chunks weg, nichts Neues
                if existing:
                    vector_store.delete(ids=existing["ids"])
                result.skipped_empty += 1
            else:
                # Erst schreiben, dann Altes aufräumen: Schlägt das Embedding fehl,
                # bleibt der bisherige Stand des Dokuments erhalten.
                _write_batches(vector_store, ids, texts, metadatas)
                stale = [i for i in (existing["ids"] if existing else []) if i not in set(ids)]
                if stale:
                    vector_store.delete(ids=stale)
                result.chunks += len(ids)
                if existing:
                    result.updated += 1
                else:
                    result.added += 1
            consecutive_failures = 0
            title = doc.get("title", doc_id)
        except PaperlessError:
            raise
        except Exception as e:  # ein defektes Dokument soll den Rest nicht aufhalten
            result.failed += 1
            consecutive_failures += 1
            title = f"#{doc_id}"
            log.exception("Dokument %s konnte nicht verarbeitet werden: %s", doc_id, e)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SyncError(
                    f"{MAX_CONSECUTIVE_FAILURES} Fehler in Folge, Abbruch. "
                    "Sind Ollama (Embedding-Modell geladen?) und ChromaDB erreichbar?"
                ) from e

        if progress:
            progress(n, len(todo), str(title))

    log.info("Sync fertig: %s", result.summary())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paperless-ngx nach ChromaDB synchronisieren")
    parser.add_argument("--loop", action="store_true", help="dauerhaft laufen und regelmäßig abgleichen")
    parser.add_argument("--full", action="store_true", help="alle Dokumente neu einlesen")
    parser.add_argument("--interval", type=int, default=settings.SYNC_INTERVAL_MIN, help="Minuten zwischen Läufen")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # sonst eine Zeile pro HTTP-Aufruf

    if not settings.paperless_configured():
        raise SystemExit("PAPERLESS_URL und PAPERLESS_TOKEN müssen gesetzt sein.")

    from .chroma_db import get_vector_store  # erst hier, damit Tests ohne Chroma-Server laufen

    full = args.full
    while True:
        try:
            sync_paperless(get_vector_store(), full=full)
            full = False  # --full gilt nur für den ersten Lauf
        except Exception as e:
            log.error("Sync fehlgeschlagen: %s", e)
            if not args.loop:
                raise SystemExit(1)
        if not args.loop:
            return
        time.sleep(max(args.interval, 1) * 60)


if __name__ == "__main__":
    main()
