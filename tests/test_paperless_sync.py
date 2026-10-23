import pytest

from infra.paperless import DocumentStub, PaperlessError
from infra.paperless_sync import SOURCE_TYPE, SyncError, build_chunks, sync_paperless
from infra.settings import settings

LONG_TEXT = "\n\n".join(f"Absatz {i}: " + "Stromverbrauch und Abschlag. " * 40 for i in range(8))


class FakePaperless:
    """Stellt Paperless nach. Dokumente lassen sich zwischen den Sync-Läufen ändern."""

    def __init__(self):
        self.docs: dict[int, dict] = {}
        self.fetched: list[int] = []

    def put(self, doc_id, content, modified="2026-01-01T10:00:00Z", **extra):
        self.docs[doc_id] = {
            "id": doc_id,
            "title": f"Dokument {doc_id}",
            "content": content,
            "modified": modified,
            "created": "2026-01-01T00:00:00+01:00",
            "correspondent": 1,
            "document_type": 1,
            "tags": [1],
            **extra,
        }

    def list_document_stubs(self):
        return [DocumentStub(d["id"], d["modified"]) for d in self.docs.values()]

    def get_document(self, doc_id):
        self.fetched.append(doc_id)
        return self.docs[doc_id]

    def name_map(self, endpoint):
        return {
            "tags": {1: "Strom"},
            "correspondents": {1: "Stadtwerke"},
            "document_types": {1: "Rechnung"},
        }[endpoint]


def stored(store):
    return store.get(where={"type": SOURCE_TYPE}, include=["metadatas"])


def test_erster_lauf_legt_chunks_mit_metadaten_an(store):
    paperless = FakePaperless()
    paperless.put(1, LONG_TEXT)
    paperless.put(2, "Kurzer Text")

    result = sync_paperless(store, client=paperless)

    assert (result.added, result.updated, result.deleted, result.failed) == (2, 0, 0, 0)
    data = stored(store)
    metas = data["metadatas"]
    assert {m["doc_id"] for m in metas} == {1, 2}
    doc1 = [m for m in metas if m["doc_id"] == 1]
    assert len(doc1) > 1
    assert all(m["chunk_count"] == len(doc1) for m in doc1)
    assert doc1[0]["url"] == "http://paperless.test/documents/1/details"
    assert doc1[0]["correspondent"] == "Stadtwerke"
    assert doc1[0]["tags"] == "Strom"
    assert "paperless-1-0" in data["ids"]


def test_chunks_tragen_titel_und_korrespondent_im_text(store):
    paperless = FakePaperless()
    paperless.put(1, "Nur ein Satz.")
    sync_paperless(store, client=paperless)

    text = store.get(ids=["paperless-1-0"])["documents"][0]
    assert text.startswith("Titel: Dokument 1")
    assert "Korrespondent: Stadtwerke" in text
    assert "Dokumenttyp: Rechnung" in text
    assert "Datum: 2026-01-01" in text
    assert text.endswith("Nur ein Satz.")


def test_zweiter_lauf_ohne_aenderung_laedt_nichts_neu(store):
    paperless = FakePaperless()
    paperless.put(1, LONG_TEXT)
    paperless.put(2, "Kurzer Text")
    sync_paperless(store, client=paperless)
    paperless.fetched.clear()

    result = sync_paperless(store, client=paperless)

    assert (result.added, result.updated, result.unchanged) == (0, 0, 2)
    assert paperless.fetched == []  # nicht einmal der Volltext wurde angefragt


def test_geaendertes_dokument_ersetzt_chunks_und_raeumt_ueberzaehlige_auf(store):
    paperless = FakePaperless()
    paperless.put(1, LONG_TEXT)
    sync_paperless(store, client=paperless)
    before = len(stored(store)["ids"])
    assert before > 1

    paperless.put(1, "Jetzt nur noch kurz.", modified="2026-02-01T10:00:00Z")
    result = sync_paperless(store, client=paperless)

    assert (result.updated, result.added) == (1, 0)
    data = stored(store)
    assert data["ids"] == ["paperless-1-0"]  # die alten Chunks 1..n sind weg
    assert data["metadatas"][0]["chunk_count"] == 1


def test_geloeschtes_dokument_wird_entfernt(store):
    paperless = FakePaperless()
    paperless.put(1, "Eins")
    paperless.put(2, "Zwei")
    sync_paperless(store, client=paperless)

    del paperless.docs[2]
    result = sync_paperless(store, client=paperless)

    assert result.deleted == 1
    assert {m["doc_id"] for m in stored(store)["metadatas"]} == {1}


def test_leere_paperless_antwort_loescht_nicht_alles(store):
    paperless = FakePaperless()
    paperless.put(1, "Eins")
    sync_paperless(store, client=paperless)

    paperless.docs.clear()  # z. B. falscher Benutzer ohne Rechte
    result = sync_paperless(store, client=paperless)

    assert result.deleted == 0
    assert len(stored(store)["ids"]) == 1


def test_dokument_ohne_text_wird_uebersprungen(store):
    paperless = FakePaperless()
    paperless.put(1, "   ")
    result = sync_paperless(store, client=paperless)

    assert result.skipped_empty == 1
    assert stored(store)["ids"] == []


def test_dokument_verliert_seinen_text_alte_chunks_verschwinden(store):
    paperless = FakePaperless()
    paperless.put(1, "Text vorhanden")
    sync_paperless(store, client=paperless)

    paperless.put(1, "", modified="2026-03-01T10:00:00Z")
    sync_paperless(store, client=paperless)

    assert stored(store)["ids"] == []


def test_wechsel_des_embedding_modells_liest_alles_neu_ein(store, monkeypatch):
    paperless = FakePaperless()
    paperless.put(1, "Eins")
    paperless.put(2, "Zwei")
    sync_paperless(store, client=paperless)

    monkeypatch.setattr(settings, "OLLAMA_EMBED_MODEL", "anderes-modell")
    result = sync_paperless(store, client=paperless)

    assert result.updated == 2


def test_halb_geschriebenes_dokument_wird_repariert(store):
    paperless = FakePaperless()
    paperless.put(1, LONG_TEXT)
    sync_paperless(store, client=paperless)
    store.delete(ids=["paperless-1-1"])  # Abbruch mitten im Schreiben simulieren

    result = sync_paperless(store, client=paperless)

    assert result.updated == 1
    metas = stored(store)["metadatas"]
    assert len(metas) == metas[0]["chunk_count"]


def test_full_liest_unveraenderte_dokumente_erneut(store):
    paperless = FakePaperless()
    paperless.put(1, "Eins")
    sync_paperless(store, client=paperless)

    result = sync_paperless(store, client=paperless, full=True)

    assert result.updated == 1


class FlakyStore:
    """Wrapper, der das Schreiben für bestimmte Dokumente scheitern lässt (z. B. Ollama hängt)."""

    def __init__(self, inner, fail_for):
        self.inner = inner
        self.fail_for = fail_for

    def add_texts(self, texts, metadatas, ids):
        if any(m["doc_id"] in self.fail_for for m in metadatas):
            raise RuntimeError("Embedding fehlgeschlagen")
        return self.inner.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_einzelner_fehler_stoppt_den_rest_nicht(store):
    paperless = FakePaperless()
    for i in (1, 2, 3):
        paperless.put(i, f"Text {i}")

    result = sync_paperless(FlakyStore(store, fail_for={2}), client=paperless)

    assert (result.added, result.failed) == (2, 1)
    assert {m["doc_id"] for m in stored(store)["metadatas"]} == {1, 3}


def test_fehler_in_folge_brechen_ab(store):
    paperless = FakePaperless()
    for i in range(1, 6):
        paperless.put(i, f"Text {i}")

    with pytest.raises(SyncError):
        sync_paperless(FlakyStore(store, fail_for={1, 2, 3, 4, 5}), client=paperless)


def test_fehlgeschlagenes_update_behaelt_den_alten_stand(store):
    paperless = FakePaperless()
    paperless.put(1, "Alter Text")
    sync_paperless(store, client=paperless)

    paperless.put(1, "Neuer Text", modified="2026-05-01T00:00:00Z")
    sync_paperless(FlakyStore(store, fail_for={1}), client=paperless)

    assert "Alter Text" in store.get(ids=["paperless-1-0"])["documents"][0]


def test_paperless_fehler_werden_nicht_verschluckt(store):
    class Down(FakePaperless):
        def list_document_stubs(self):
            raise PaperlessError("nicht erreichbar")

    with pytest.raises(PaperlessError):
        sync_paperless(store, client=Down())


def test_build_chunks_laesst_leere_metadaten_weg():
    doc = {"id": 7, "title": "Brief", "content": "Hallo", "modified": "x", "tags": [], "correspondent": None}
    names = {"tags": {}, "correspondents": {}, "document_types": {}}

    _, _, metas = build_chunks(doc, names)

    assert "correspondent" not in metas[0]  # Chroma erlaubt keine None-Werte
    assert "tags" not in metas[0]
    assert all(v is not None for v in metas[0].values())
