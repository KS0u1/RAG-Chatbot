"""Schlanker Client für die Paperless-ngx REST API (nur lesend)."""

import logging
from dataclasses import dataclass
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class PaperlessError(RuntimeError):
    """Paperless ist nicht erreichbar oder lehnt die Anfrage ab."""


@dataclass(frozen=True)
class DocumentStub:
    id: int
    modified: str


class PaperlessClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: int = 30,
        page_size: int = 100,
        session: requests.Session | None = None,
    ):
        if not base_url or not token:
            raise PaperlessError("PAPERLESS_URL und PAPERLESS_TOKEN müssen gesetzt sein.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.page_size = page_size

        self.session = session or requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Accept": "application/json",
        })
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise PaperlessError(f"Paperless nicht erreichbar ({url}): {e}") from e

        if response.status_code in (401, 403):
            raise PaperlessError(
                f"Paperless hat den Zugriff verweigert (HTTP {response.status_code}). "
                "Token und Berechtigungen des Benutzers prüfen."
            )
        try:
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.HTTPError, ValueError) as e:
            raise PaperlessError(f"Unerwartete Antwort von {url}: {e}") from e

    def _paginate(self, path: str, params: dict | None = None) -> Iterator[dict]:
        """Läuft über alle Seiten. Die Seitennummer wird selbst gezählt, statt der `next`-URL
        zu folgen, denn hinter einem Reverse Proxy enthält `next` oft Host oder Schema,
        die von hier aus nicht erreichbar sind."""
        page = 1
        while True:
            data = self._get(path, {**(params or {}), "page": page, "page_size": self.page_size})
            yield from data.get("results", [])
            if not data.get("next"):
                return
            page += 1

    def list_document_stubs(self) -> list[DocumentStub]:
        """Nur ID und Änderungsdatum aller Dokumente. Das ist billig und reicht für den Abgleich."""
        stubs = []
        for item in self._paginate("/api/documents/", {"fields": "id,modified", "ordering": "id"}):
            stubs.append(DocumentStub(id=int(item["id"]), modified=str(item.get("modified", ""))))
        return stubs

    def get_document(self, doc_id: int) -> dict:
        return self._get(f"/api/documents/{doc_id}/")

    def name_map(self, endpoint: str) -> dict[int, str]:
        """ID -> Name für `tags`, `correspondents` oder `document_types`."""
        return {int(item["id"]): item.get("name", "") for item in self._paginate(f"/api/{endpoint}/")}
