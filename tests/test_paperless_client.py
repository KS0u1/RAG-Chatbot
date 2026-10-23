import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from infra.paperless import PaperlessClient, PaperlessError

TOKEN = "geheim"
DOCS = [{"id": i, "modified": f"2026-01-{i:02d}T00:00:00Z", "title": f"Doc {i}", "content": f"Text {i}"} for i in range(1, 8)]
SEEN: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        SEEN.append({"path": url.path, "query": query, "auth": self.headers.get("Authorization")})

        if self.headers.get("Authorization") != f"Token {TOKEN}":
            return self._send(401, {"detail": "Invalid token."})

        if url.path == "/api/documents/":
            size, page = int(query["page_size"]), int(query["page"])
            chunk = DOCS[(page - 1) * size: page * size]
            has_next = page * size < len(DOCS)
            # Bewusst eine "falsche" next-URL wie hinter einem Reverse Proxy: Der Client darf ihr nicht folgen.
            return self._send(200, {
                "count": len(DOCS),
                "next": "http://unerreichbar.invalid/api/documents/?page=%d" % (page + 1) if has_next else None,
                "results": chunk,
            })
        if url.path.startswith("/api/documents/"):
            doc_id = int(url.path.strip("/").split("/")[-1])
            doc = next((d for d in DOCS if d["id"] == doc_id), None)
            return self._send(200, doc) if doc else self._send(404, {"detail": "Not found."})
        if url.path == "/api/tags/":
            return self._send(200, {"count": 1, "next": None, "results": [{"id": 3, "name": "Strom"}]})
        return self._send(404, {})


@pytest.fixture
def server():
    SEEN.clear()
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_stubs_ueber_mehrere_seiten_und_ohne_next_url_zu_folgen(server):
    client = PaperlessClient(server, TOKEN, page_size=3)

    stubs = client.list_document_stubs()

    assert [s.id for s in stubs] == list(range(1, 8))
    assert stubs[0].modified == "2026-01-01T00:00:00Z"
    pages = [r["query"]["page"] for r in SEEN if r["path"] == "/api/documents/"]
    assert pages == ["1", "2", "3"]
    assert SEEN[0]["query"]["fields"] == "id,modified"
    assert SEEN[0]["auth"] == f"Token {TOKEN}"


def test_einzelnes_dokument_und_namenslisten(server):
    client = PaperlessClient(server, TOKEN)

    assert client.get_document(4)["content"] == "Text 4"
    assert client.name_map("tags") == {3: "Strom"}


def test_falscher_token_gibt_verstaendliche_fehlermeldung(server):
    client = PaperlessClient(server, "falsch")

    with pytest.raises(PaperlessError, match="Zugriff verweigert"):
        client.list_document_stubs()


def test_nicht_erreichbarer_server_gibt_paperless_error():
    client = PaperlessClient("http://127.0.0.1:1", TOKEN, timeout=1)
    client.session.mount("http://", __import__("requests").adapters.HTTPAdapter(max_retries=0))

    with pytest.raises(PaperlessError, match="nicht erreichbar"):
        client.list_document_stubs()


def test_fehlende_konfiguration_wird_abgelehnt():
    with pytest.raises(PaperlessError):
        PaperlessClient("", "")
