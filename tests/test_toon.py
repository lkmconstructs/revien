"""
LEG P2: TOON recall wire format — round-trip tests.

The lossless claim is TESTED here (serialize -> parse -> IDENTICAL dict),
not inherited from the upstream TOON README. Covered: real recall output
from a seeded temp graph (through POST /v1/recall), unicode content,
delimiter/quote/newline-containing content, empty results, float type
preservation, and the API surface (format=toon media type, json path
unchanged, invalid format rejected).
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from revien.daemon.server import create_app
from revien.toon import ToonError, parse_recall, serialize_recall


SAMPLE_CONVERSATION = """User: We need to decide on the pricing for the enterprise tier.
Assistant: Based on our analysis, I recommend $499/month with a 20% annual discount.
User: That works. Let's go with that. Also, make sure the deployment uses PostgreSQL, not MySQL. We decided that last week.
Assistant: Confirmed. Enterprise tier at $499/month, 20% annual discount, PostgreSQL for the database layer. I'll update the architecture doc."""

# Delimiter-heavy + unicode content, ingested so weird strings flow through
# REAL recall payloads, not just hand-built dicts.
TRICKY_CONVERSATION = """User: The launch checklist is: staging, canary, prod — in that order, no exceptions.
Assistant: Noted. Checklist order confirmed: staging, canary, prod.
User: Also the café rebrand ships with the 中文 locale and the "grand opening" banner 😀.
Assistant: Understood. Café rebrand includes the 中文 locale and the "grand opening" banner."""


@pytest.fixture
def seeded_client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        c.post("/v1/ingest", json={
            "source_id": "toon-test-1",
            "content": SAMPLE_CONVERSATION,
        })
        c.post("/v1/ingest", json={
            "source_id": "toon-test-2",
            "content": TRICKY_CONVERSATION,
        })
        yield c
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


def _assert_types_identical(a, b, path="$"):
    """dict == is type-loose in Python (1 == 1.0, True == 1) — IDENTICAL
    means values AND types survive the round-trip."""
    assert type(a) is type(b), f"type drift at {path}: {type(a)} -> {type(b)}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"key drift at {path}"
        for k in a:
            _assert_types_identical(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"length drift at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_types_identical(x, y, f"{path}[{i}]")


def _round_trip(payload):
    toon = serialize_recall(payload)
    back = parse_recall(toon)
    assert back == payload
    _assert_types_identical(payload, back)
    return toon


# ── Unit round-trips (hand-built recall-shaped payloads) ──


class TestRoundTripUnits:
    def test_empty_results(self):
        toon = _round_trip({
            "query": "nothing matches this",
            "results": [],
            "nodes_examined": 0,
            "retrieval_time_ms": 0.42,
            "semantic_active": False,
            "semantic_note": "semantic layer disabled (REVIEN_SEMANTIC=0)",
        })
        assert "results: []" in toon

    def test_unicode_content(self):
        _round_trip({
            "query": "café 中文 😀 — naïve façade Ω",
            "results": [{
                "node_id": "n-1",
                "node_type": "FACT",
                "label": "unicode label: 中文",
                "content": "café ☕ + 日本語テキスト + emoji 😀🎉 + accents éèêë",
                "score": 0.75,
                "score_breakdown": {"recency": 0.9, "frequency": 0.6},
                "path": ["ctx-α", "n-1"],
            }],
            "nodes_examined": 3,
            "retrieval_time_ms": 1.5,
            "semantic_active": True,
            "semantic_note": None,
        })

    def test_delimiter_and_structural_chars_in_content(self):
        _round_trip({
            "query": "a, b, c: the works",
            "results": [{
                "node_id": "n-1",
                "node_type": "FACT",
                "label": "commas, colons: and \"quotes\"",
                "content": 'line one\nline two\twith tab, commas, "quotes", '
                           "back\\slash, [brackets] and {braces}: all of it",
                "score": 0.5,
                "score_breakdown": {"recency": 0.1},
                "path": ["a,b", "c:d", "-leading-hyphen", " padded "],
            }],
            "nodes_examined": 1,
            "retrieval_time_ms": 0.1,
            "semantic_active": False,
            "semantic_note": "graph-only",
        })

    def test_ambiguous_strings_stay_strings(self):
        """Strings that look like TOON keywords/numbers must round-trip as
        strings, not get re-typed on parse."""
        payload = {
            "query": "true",
            "results": [{
                "node_id": "123",
                "node_type": "null",
                "label": "-0.5",
                "content": "false",
                "score": 0.0,
                "score_breakdown": {},
                "path": ["1e5", "[]"],
            }],
            "nodes_examined": 1,
            "retrieval_time_ms": 1.0,
            "semantic_active": True,
            "semantic_note": None,
        }
        _round_trip(payload)

    def test_float_type_and_precision_preserved(self):
        """repr()-based floats: shortest exact IEEE-754 round-trip, and
        integral floats keep .0 so they never come back as int."""
        payload = {
            "query": "floats",
            "results": [{
                "node_id": "n",
                "node_type": "FACT",
                "label": "l",
                "content": "c",
                "score": 0.1 + 0.2,  # 0.30000000000000004
                "score_breakdown": {
                    "recency": 1.0,
                    "frequency": 1e-07,
                    "centrality": 12345678901234567890.0,
                    "path_strength": 0.6180339887498949,
                },
                "path": [],
            }],
            "nodes_examined": 7,
            "retrieval_time_ms": 2847.991,
            "semantic_active": True,
            "semantic_note": None,
        }
        _round_trip(payload)

    def test_tabular_form_round_trips(self):
        """Uniform primitive-valued object arrays take the tabular form —
        exercised even though recall results (nested) use list form."""
        payload = {
            "query": "q",
            "rows": [
                {"id": 1, "name": "Alice", "ok": True},
                {"id": 2, "name": "Bob, the second", "ok": False},
            ],
        }
        toon = serialize_recall(payload)
        assert "rows[2]{id,name,ok}:" in toon
        back = parse_recall(toon)
        assert back == payload
        _assert_types_identical(payload, back)

    def test_non_finite_float_rejected(self):
        with pytest.raises(ToonError):
            serialize_recall({"x": float("nan")})
        with pytest.raises(ToonError):
            serialize_recall({"x": float("inf")})


# ── Real recall payloads from a seeded temp graph ─────────


class TestRoundTripRealRecall:
    def test_seeded_graph_recall_round_trips(self, seeded_client):
        got_results = False
        for query in ("enterprise pricing", "which database was chosen",
                      "launch checklist order", "café rebrand locale"):
            payload = seeded_client.post(
                "/v1/recall", json={"query": query, "top_n": 5}
            ).json()
            got_results = got_results or bool(payload["results"])
            _round_trip(payload)
        assert got_results, "seeded graph produced no results for any query"

    def test_empty_results_through_api(self, seeded_client):
        payload = seeded_client.post("/v1/recall", json={
            "query": "zzz qqq xyzzy plugh", "min_score": 99.0,
        }).json()
        assert payload["results"] == []
        _round_trip(payload)

    def test_include_tensions_shape_round_trips(self, seeded_client):
        payload = seeded_client.post("/v1/recall", json={
            "query": "enterprise pricing", "include_tensions": True,
        }).json()
        _round_trip(payload)


# ── API surface ───────────────────────────────────────────


class TestRecallFormatParam:
    def test_toon_format_returns_text_toon(self, seeded_client):
        resp = seeded_client.post("/v1/recall", json={
            "query": "enterprise pricing", "format": "toon",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/toon")

    def test_toon_body_parses_to_json_payload(self, seeded_client):
        """The TOON body IS the json payload — same graph, same query;
        only retrieval_time_ms (wall clock) may differ between calls."""
        json_payload = seeded_client.post(
            "/v1/recall", json={"query": "enterprise pricing"}
        ).json()
        toon_resp = seeded_client.post("/v1/recall", json={
            "query": "enterprise pricing", "format": "toon",
        })
        toon_payload = parse_recall(toon_resp.text)
        json_payload.pop("retrieval_time_ms")
        toon_payload.pop("retrieval_time_ms")
        assert toon_payload == json_payload

    def test_default_and_explicit_json_stay_json(self, seeded_client):
        default = seeded_client.post("/v1/recall", json={"query": "pricing"})
        explicit = seeded_client.post("/v1/recall", json={
            "query": "pricing", "format": "json",
        })
        for resp in (default, explicit):
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/json")
            body = resp.json()
            assert set(body.keys()) == {
                "query", "results", "nodes_examined", "retrieval_time_ms",
                "semantic_active", "semantic_note",
            }

    def test_invalid_format_rejected(self, seeded_client):
        resp = seeded_client.post("/v1/recall", json={
            "query": "pricing", "format": "yaml",
        })
        assert resp.status_code == 400


class TestReservedPathsKey:
    def test_reserved_paths_key_refused_loudly(self):
        """A generic payload carrying a top-level 'paths' key cannot
        round-trip (parse_recall would mistake it for a flattened doc and
        silently return a DIFFERENT dict) — serialize_recall must refuse."""
        from revien.toon import ToonError, serialize_recall

        payload = {"query": "q", "results": [], "paths": [["p1"]]}
        with pytest.raises(ToonError, match="reserved"):
            serialize_recall(payload)
