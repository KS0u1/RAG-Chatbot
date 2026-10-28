# 💬 Paperless-Chat

Ein lokaler KI-Assistent, mit dem du Fragen zu deinen **Paperless-ngx**-Dokumenten stellst (Rechnungen, Verträge, Bescheide, Briefe …). Alles läuft im eigenen Netz: Ollama für das Sprachmodell, ChromaDB als Vektordatenbank, Streamlit als Oberfläche. Es verlassen keine Daten das Homelab.

## So funktioniert es

```
Paperless-ngx ──(REST-API, nur lesend)──► sync ──► ChromaDB ◄── app (Streamlit) ──► Ollama
   OCR-Text                          Chunks + Embeddings      Suche, Reranking, Antwort
```

* **sync** gleicht Paperless alle `SYNC_INTERVAL_MIN` Minuten mit Chroma ab. Nur neue und geänderte Dokumente werden neu eingelesen, gelöschte werden entfernt. Verwendet wird der OCR-Text, den Paperless selbst liefert.
* **app** sucht passende Textabschnitte (MMR), sortiert sie optional mit einem Reranker neu, übergibt sie an das Sprachmodell und hängt die verwendeten Dokumente als Links nach Paperless an die Antwort.
* Jeder Textabschnitt trägt Titel, Absender, Dokumenttyp, Datum und Tags im Kopf. Dadurch findet die Suche auch Fragen wie „Stromrechnung 2025“.
* Der Chatverlauf liegt in SQLite (Volume `app_data`). Das Modell bekommt davon nur die letzten `HISTORY_MAX_MESSAGES` Nachrichten, ohne die Quellenzeilen.
* Das Embedding-Modell bekommt die Präfixe, auf die es trainiert ist (`search_query:` für Fragen, `search_document:` für Abschnitte). Die Abschnitte sind höchstens 1000 Zeichen lang, damit sie in die 512 Tokens von `nomic-embed-text` passen.
* Der Reranker bewertet jeden Treffer im Format seiner Modellkarte mit „yes“ oder „no“ und erzeugt dafür nur ein Token. Liefert Ollama Logprobs, wird daraus die Wahrscheinlichkeit für „yes“.

## Voraussetzungen

1. **Docker** mit Compose (z. B. über Portainer).
2. **Ollama** ist Teil des Stacks. Beim ersten Start lädt der Dienst `ollama-init` die Modelle `llama3.2:1b`, `nomic-embed-text-v2-moe` und `dengcao/Qwen3-Reranker-0.6B:Q8_0` herunter (einige GB, das dauert). Fortschritt: `docker compose logs -f ollama-init`. Den Antworten-Chat kannst du erst nutzen, wenn dort „Alle Modelle geladen.“ steht.
   Du hast schon einen Ollama? Dann setze `OLLAMA_BASE_URL` und lasse die Dienste `ollama` und `ollama-init` weg (siehe Kommentar in `compose.yaml`).
3. **Paperless-ngx** mit API-Token. Lege am besten einen eigenen Benutzer an, der Dokumente nur *ansehen* darf (Paperless → Mein Profil → API-Token).

## Installation

```bash
cp .env.example .env     # PAPERLESS_URL und PAPERLESS_TOKEN eintragen
docker compose up -d --build
```

Die Oberfläche läuft danach auf Port **8501**. Der erste Abgleich startet automatisch mit dem Sync-Dienst (Fortschritt: `docker compose logs -f sync`) oder per Knopf in der Seitenleiste.

**Portainer:** Stack aus dem Git-Repository anlegen (Compose-Pfad `compose.yaml`) und die Variablen aus `.env.example` unter *Environment variables* eintragen.

**Nginx Proxy Manager:** Streamlit nutzt WebSockets. Aktiviere im Proxy-Host *Websockets Support*, sonst bleibt die Seite leer oder lädt ständig neu.

## Konfiguration

| Variable | Standard | Bedeutung |
| :--- | :--- | :--- |
| `PAPERLESS_URL` | – | Adresse von Paperless (Pflicht) |
| `PAPERLESS_TOKEN` | – | API-Token (Pflicht) |
| `PAPERLESS_PUBLIC_URL` | `PAPERLESS_URL` | Adresse für die Links in den Quellen, falls du Paperless unter einem anderen Namen öffnest |
| `SYNC_INTERVAL_MIN` | `30` | Minuten zwischen den Abgleichen |
| `OLLAMA_BASE_URL` | `http://ollama:11434` (Stack) bzw. `http://localhost:11434` (ohne Docker) | Ollama. Bei externem Ollama aus Containern die Host-IP verwenden |
| `OLLAMA_MODEL` | `llama3.2:1b` | Antwortmodell |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Embedding-Modell |
| `OLLAMA_RERANK_MODEL` | `dengcao/Qwen3-Reranker-0.6B:Q8_0` | Reranker |
| `OLLAMA_NUM_CTX` | `6144` | Kontextfenster des Antwortmodells in Tokens. Mehr braucht mehr RAM |
| `OLLAMA_REASONING` | leer | `false` schaltet den Denkmodus von Modellen wie `qwen3` ab. Bei `llama3.x` leer lassen |
| `OLLAMA_KEEP_ALIVE` | `24h` | Wie lange Ollama die Modelle nach der letzten Anfrage im RAM hält (Ollama-Standard: 5 Minuten) |
| `OLLAMA_NUM_PARALLEL` | `1` | Gleichzeitige Anfragen an Ollama. Jede braucht ein eigenes Kontextfenster im RAM |
| `HISTORY_MAX_MESSAGES` | `6` | So viele letzte Nachrichten bekommt das Modell als Gesprächskontext |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | passend zum Modell | Präfixe für das Embedding-Modell. Nur setzen, wenn du ein anderes Modell nimmst |
| `RERANK_ENABLED` | `false` | Reranking der Treffer, siehe Hinweis unten |
| `RERANK_TOP_N` / `RETRIEVER_K` / `RETRIEVER_FETCH_K` | `5` / `8` / `20` | Wie viele Abschnitte gesucht und verwendet werden |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_COLLECTION` | `localhost` / `8000` / `paperless` | ChromaDB (im Compose-Stack automatisch gesetzt) |
| `SQLITE_URL` | `sqlite:///chat_history.db` | Chatverlauf (im Compose-Stack automatisch gesetzt) |

## Betrieb

* **Embedding-Modell wechseln:** Der Sync erkennt das und liest beim nächsten Lauf automatisch alle Dokumente neu ein. Das gilt auch für neue Präfixe und für Änderungen an der Art, wie Abschnitte gebaut werden (`INDEX_VERSION` in `infra/paperless_sync.py`).
* **Alles neu einlesen:** Kästchen in der Seitenleiste oder `docker compose run --rm sync python -m infra.paperless_sync --full`.
* **Einmaliger Abgleich per Kommando:** `python -m infra.paperless_sync`

## Entwicklung

```bash
pip install -r requirements.txt pytest
python -m pytest tests
```

Die Tests laufen ohne Ollama und ohne Chroma-Server (In-Memory-Chroma, nachgestellte Paperless-API).

## Hinweise

* **Platzbedarf:** Die Python-Pakete (vor allem ChromaDB und Streamlit) belegen rund 0,9 GB, das App-Image ist also etwa 1 GB groß (geschätzt, nicht gebaut). Dazu kommen das Chroma-Image und die Ollama-Modelle.
* **Keine Anmeldung:** Die Oberfläche ist offen für jeden im Netz. Sie gibt Auskunft über *alle* Dokumente, die der Paperless-Benutzer sehen darf. Stelle sie nur im Heimnetz bereit oder setze eine Authentifizierung davor. ChromaDB ist bewusst nicht nach außen veröffentlicht.
* **Reranker:** Mit Ollama 0.40.9 gemessen (2026-10-28): `dengcao/Qwen3-Reranker-0.6B:Q8_0` antwortet für einen passenden und einen unpassenden Abschnitt gleich, alle Token haben dieselbe Wahrscheinlichkeit (Logprob −11,93 = ln der Vokabelgröße). Das GGUF erzeugt unter Ollama also keine verwertbare Ausgabe, deshalb ist Reranking standardmäßig aus. Der Code folgt dem Format der Modellkarte und lässt sich mit einem funktionierenden Modell per `RERANK_ENABLED=true` einschalten. Im App-Log stehen dann die Scores (`Rerank-Scores:`).
* **Quellen:** Angezeigt werden die Dokumente, aus denen der Kontext stammt, nicht zwingend nur die, die das Modell wörtlich verwendet hat.
* **Qualität:** Mit `llama3.2:1b` sind Antworten schnell, aber bei komplexen Fragen ungenau. Ein größeres Modell verbessert das deutlich, braucht ohne GPU aber Geduld.
