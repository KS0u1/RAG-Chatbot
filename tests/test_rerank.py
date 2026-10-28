import math

import pytest
from langchain_core.documents import Document

import core.rerank as rr
from infra.settings import settings


def test_prompt_folgt_dem_format_der_modellkarte():
    prompt = rr.build_prompt("Was kostet die Haftpflicht?", "Beitrag 25,54 EUR")

    assert prompt.startswith("<|im_start|>system\nJudge whether the Document meets the requirements")
    assert '"yes" or "no"' in prompt
    assert "<Query>: Was kostet die Haftpflicht?" in prompt
    assert "<Document>: Beitrag 25,54 EUR" in prompt
    # leerer Denkblock: das Modell soll sofort mit yes/no antworten
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_prompt_kuerzt_sehr_lange_abschnitte():
    prompt = rr.build_prompt("q", "x" * 10_000)
    assert prompt.count("x") == rr.MAX_DOC_CHARS


def test_score_aus_logprobs_ist_wahrscheinlichkeit_fuer_yes():
    data = {
        "response": "yes",
        "logprobs": [{
            "token": "yes",
            "logprob": math.log(0.9),
            "top_logprobs": [
                {"token": "yes", "logprob": math.log(0.9)},
                {"token": "no", "logprob": math.log(0.05)},
                {"token": "Yes", "logprob": math.log(0.03)},
            ],
        }],
    }
    assert rr.score_from_response(data) == pytest.approx(0.93 / 0.98)


@pytest.mark.parametrize("response,expected", [
    ("yes", 1.0), (" Yes", 1.0), ("no", 0.0), ("No.", 0.0), ("vielleicht", rr.NEUTRAL), ("", rr.NEUTRAL),
])
def test_ohne_logprobs_zaehlt_das_antwort_token(response, expected):
    assert rr.score_from_response({"response": response}) == expected


def test_logprobs_ohne_yes_oder_no_fallen_auf_das_antwort_token_zurueck():
    data = {"response": "no", "logprobs": [{"token": "\n", "logprob": -0.1, "top_logprobs": []}]}
    assert rr.score_from_response(data) == 0.0


def test_score_fragt_ollama_raw_mit_einem_token(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "yes"}

    class Session:
        def post(self, url, json, timeout):
            sent.update(url=url, body=json, timeout=timeout)
            return Response()

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://ollama:11434/")
    monkeypatch.setattr(settings, "OLLAMA_RERANK_MODEL", "reranker")

    assert rr.score("frage", "text", session=Session()) == 1.0
    assert sent["url"] == "http://ollama:11434/api/generate"
    body = sent["body"]
    assert body["model"] == "reranker"
    assert body["raw"] is True and body["stream"] is False
    assert body["options"]["num_predict"] == 1
    assert body["logprobs"] is True


def test_rerank_aus_gibt_suchreihenfolge_zurueck(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    docs = [Document(str(i)) for i in range(6)]

    assert rr.rerank("frage", docs, top_n=3) == docs[:3]


def test_rerank_fehler_faellt_auf_suchreihenfolge_zurueck(monkeypatch):
    def kaputt(*args, **kwargs):
        raise ConnectionError("Ollama weg")

    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(rr, "score", kaputt)
    docs = [Document(str(i)) for i in range(6)]

    assert rr.rerank("frage", docs, top_n=2) == docs[:2]


def test_rerank_sortiert_nach_score_und_bleibt_bei_gleichstand_stabil(monkeypatch):
    scores = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.9}
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(rr, "score", lambda query, text, session=None: scores[text])
    docs = [Document(t) for t in "abcd"]

    assert [d.page_content for d in rr.rerank("frage", docs, top_n=3)] == ["b", "d", "c"]


def test_rerank_ohne_treffer():
    assert rr.rerank("frage", [], top_n=3) == []
