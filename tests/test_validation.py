"""
Protocol-surface validation (R4): one shared rule core (revien/validation.py),
three faces, none allowed to drift.

Contract under test:
  - REST: /v1/ingest and /v1/recall return 400 with a clear message for blank
    content/source_id/query, unknown content_type, unparseable timestamp
    (previously a SILENT drop), out-of-range top_n (reject, not clamp), and
    negative min_score. Valid edge values (top_n=1, top_n=20, min_score=0)
    pass.
  - MCP: the same violations surface as a clean ToolError (the bad-as_of
    precedent pinned by tests/test_mcp.py), never a stack trace.
  - Hermes: handle_tool_call returns the face's existing error payload shape —
    a json.dumps'd {"error": ...} string — never an exception into Hermes.
  - Version: /v1/health and the FastAPI app metadata report
    revien.__version__ (0.3.0), not a hardcoded drifted string.

The engine/pipeline layer itself stays permissive on purpose: existing tests
pin pipeline.ingest(content="") creating a bare context node and
engine.recall(top_n=100) clamping to 20, and revien_bench drives the engine
directly — so the rules live at the protocol edges only.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import revien
from revien.daemon.server import create_app
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex
from revien.validation import (
    TOP_N_MAX,
    TOP_N_MIN,
    VALID_CONTENT_TYPES,
    ValidationError,
    validate_ingest,
    validate_recall,
)

try:
    from mcp.server.fastmcp.exceptions import ToolError

    from revien.mcp_server import build_mcp_server

    MCP_INSTALLED = True
except ImportError:  # pragma: no cover - core install without the extra
    MCP_INSTALLED = False


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


VALID_INGEST = {
    "source_id": "test",
    "content": "Decision: validation now happens at every protocol edge.",
}


def _ingest(client, **overrides):
    return client.post("/v1/ingest", json={**VALID_INGEST, **overrides})


def _recall(client, **overrides):
    return client.post("/v1/recall", json={"query": "validation", **overrides})


# ── Shared core (unit) ────────────────────────────────────


class TestValidationCore:
    def test_content_types_match_pipeline_doc(self):
        assert VALID_CONTENT_TYPES == {"conversation", "document", "note", "code"}

    def test_validate_ingest_returns_parsed_timestamp(self):
        ts = validate_ingest(
            content="x", source_id="s", content_type="note",
            timestamp="2026-01-02T03:04:05+00:00",
        )
        assert ts is not None and ts.year == 2026

    def test_validate_ingest_none_timestamp_ok(self):
        assert validate_ingest(
            content="x", source_id="s", content_type="note"
        ) is None

    @pytest.mark.parametrize("ts", [
        "2026-07-13T00:00:00.123Z",  # exact JS Date.toISOString() wire format
        "2026-07-13T00:00:00Z",
        "2026-07-13T00:00:00z",
    ])
    def test_z_suffix_timestamp_parses_on_all_pythons(self, ts):
        """datetime.fromisoformat only accepts a trailing 'Z' from 3.11+ —
        the core normalizes it so a 3.10 daemon accepts the JS wire format."""
        parsed = validate_ingest(
            content="x", source_id="s", content_type="note", timestamp=ts
        )
        assert parsed is not None
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0

    def test_z_suffix_on_garbage_still_rejected(self):
        with pytest.raises(ValidationError, match="Invalid timestamp"):
            validate_ingest(
                content="x", source_id="s", content_type="note",
                timestamp="not-a-timeZ",
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_min_score_rejected(self, bad):
        # NaN < 0 is False — a plain comparison would let NaN silently
        # disable the score filter.
        with pytest.raises(ValidationError, match="Invalid min_score"):
            validate_recall(query="q", top_n=5, min_score=bad)

    def test_validation_error_is_valueerror(self):
        # FastMCP wraps ValueError in ToolError — the subclassing is load-bearing.
        assert issubclass(ValidationError, ValueError)

    def test_top_n_bool_rejected(self):
        with pytest.raises(ValidationError, match="Invalid top_n"):
            validate_recall(query="q", top_n=True)


# ── REST 400 matrix ───────────────────────────────────────


class TestRestIngestValidation:
    def test_blank_content_400(self, client):
        r = _ingest(client, content="")
        assert r.status_code == 400
        assert "Invalid content" in r.json()["detail"]

    def test_whitespace_content_400(self, client):
        r = _ingest(client, content="   \n\t ")
        assert r.status_code == 400
        assert "Invalid content" in r.json()["detail"]

    def test_blank_source_id_400(self, client):
        r = _ingest(client, source_id="  ")
        assert r.status_code == 400
        assert "Invalid source_id" in r.json()["detail"]

    def test_bad_content_type_400(self, client):
        r = _ingest(client, content_type="memo")
        assert r.status_code == 400
        assert "Invalid content_type" in r.json()["detail"]
        assert "memo" in r.json()["detail"]

    def test_unparseable_timestamp_400_not_silent_drop(self, client):
        """The old server behavior swallowed a bad timestamp (try/except pass)
        and ingested anyway — now it's a refusal, and nothing is written."""
        before = client.get("/v1/health").json()["node_count"]
        r = _ingest(client, timestamp="not-a-time")
        assert r.status_code == 400
        assert "Invalid timestamp" in r.json()["detail"]
        assert "ISO-8601" in r.json()["detail"]
        assert client.get("/v1/health").json()["node_count"] == before

    def test_valid_timestamp_accepted(self, client):
        r = _ingest(client, timestamp="2026-01-02T03:04:05+00:00")
        assert r.status_code == 200
        assert r.json()["context_node_id"]

    def test_js_toisostring_timestamp_accepted(self, client):
        """The exact Date.toISOString() wire format — must pass on every CI
        python leg (3.10 included), not just 3.11+."""
        r = _ingest(client, timestamp="2026-07-13T00:00:00.123Z")
        assert r.status_code == 200
        assert r.json()["context_node_id"]

    @pytest.mark.parametrize("ct", sorted(VALID_CONTENT_TYPES))
    def test_all_documented_content_types_accepted(self, client, ct):
        assert _ingest(client, content_type=ct).status_code == 200


class TestRestRecallValidation:
    def test_blank_query_400(self, client):
        r = _recall(client, query="")
        assert r.status_code == 400
        assert "Invalid query" in r.json()["detail"]

    def test_whitespace_query_400(self, client):
        r = _recall(client, query=" \t ")
        assert r.status_code == 400

    @pytest.mark.parametrize("bad_top_n", [0, -1, -5, 21, 100])
    def test_out_of_range_top_n_400(self, client, bad_top_n):
        r = _recall(client, top_n=bad_top_n)
        assert r.status_code == 400
        assert "Invalid top_n" in r.json()["detail"]
        assert str(bad_top_n) in r.json()["detail"]

    def test_negative_min_score_400(self, client):
        r = _recall(client, min_score=-0.1)
        assert r.status_code == 400
        assert "Invalid min_score" in r.json()["detail"]

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_min_score_400(self, client, token):
        """Raw-JSON NaN/Infinity through the wire — python's json.loads (and
        many JS serializers) emit/accept the bare tokens even though strict
        JSON doesn't, and pydantic's float accepts them (allow_inf_nan
        default). Must 400, not silently disable the score filter
        (NaN < 0 is False). httpx's own json= kwarg refuses to serialize
        NaN, so the body is sent raw — exactly what a lenient client does."""
        r = client.post(
            "/v1/recall",
            content=f'{{"query": "validation", "min_score": {token}}}',
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        assert "Invalid min_score" in r.json()["detail"]

    @pytest.mark.parametrize("top_n", [TOP_N_MIN, 5, TOP_N_MAX])
    def test_boundary_top_n_pass(self, client, top_n):
        r = _recall(client, top_n=top_n)
        assert r.status_code == 200
        assert "results" in r.json()

    def test_min_score_zero_passes(self, client):
        assert _recall(client, min_score=0).status_code == 200

    def test_format_validation_unchanged(self, client):
        # The pre-existing check whose style the new rules follow.
        r = _recall(client, format="xml")
        assert r.status_code == 400
        assert "Invalid format" in r.json()["detail"]


# ── MCP ToolError matrix ──────────────────────────────────


@pytest.mark.skipif(not MCP_INSTALLED, reason="mcp extra not installed")
class TestMcpValidation:
    @pytest.fixture
    def server(self, db_path):
        return build_mcp_server(db_path=db_path)

    @staticmethod
    def _call(server, name, arguments):
        import asyncio
        import json as _json

        res = asyncio.run(server.call_tool(name, arguments))
        content = res[0] if isinstance(res, tuple) else res
        return _json.loads(content[0].text)

    def test_recall_blank_query_toolerror(self, server):
        with pytest.raises(ToolError) as e:
            self._call(server, "revien_recall", {"query": "   "})
        assert "Invalid query" in str(e.value)

    @pytest.mark.parametrize("bad_top_n", [0, -1, 21])
    def test_recall_out_of_range_top_n_toolerror(self, server, bad_top_n):
        with pytest.raises(ToolError) as e:
            self._call(server, "revien_recall", {"query": "x", "top_n": bad_top_n})
        assert "Invalid top_n" in str(e.value)

    def test_store_blank_content_toolerror(self, server):
        with pytest.raises(ToolError) as e:
            self._call(server, "revien_store", {"content": ""})
        assert "Invalid content" in str(e.value)

    def test_store_blank_source_id_toolerror(self, server):
        with pytest.raises(ToolError) as e:
            self._call(server, "revien_store", {"content": "x", "source_id": " "})
        assert "Invalid source_id" in str(e.value)

    def test_store_bad_content_type_toolerror(self, server):
        with pytest.raises(ToolError) as e:
            self._call(
                server, "revien_store", {"content": "x", "content_type": "memo"}
            )
        assert "Invalid content_type" in str(e.value)

    @pytest.mark.parametrize("top_n", [TOP_N_MIN, TOP_N_MAX])
    def test_recall_boundary_top_n_pass(self, server, top_n):
        out = self._call(server, "revien_recall", {"query": "x", "top_n": top_n})
        assert "results" in out

    def test_store_valid_passes(self, server):
        out = self._call(
            server, "revien_store", {"content": "The prod db is postgres."}
        )
        assert out["context_node_id"]


# ── Hermes error-payload matrix ───────────────────────────


class TestHermesValidation:
    @pytest.fixture
    def provider(self, db_path):
        """A genuine RevienMemoryProvider with its stack prewired, built
        WITHOUT invoking __init__ (SDK-gated) — the test_hermes_provider.py
        pattern. handle_tool_call is real code either way."""
        import queue
        import threading

        from revien.hermes_provider import RevienMemoryProvider

        store = GraphStore(db_path=db_path)
        semantic = SemanticIndex(store)
        prov = object.__new__(RevienMemoryProvider)
        prov._db_path = db_path
        prov._session_id = ""
        prov._store = store
        prov._pipeline = IngestionPipeline(store, semantic=semantic)
        prov._engine = RetrievalEngine(store, semantic=semantic)
        prov._sync_queue = queue.Queue()
        prov._sync_worker = None
        prov._worker_lock = threading.Lock()
        yield prov
        store.close()

    @staticmethod
    def _call(provider, name, args):
        import json as _json

        raw = provider.handle_tool_call(name, args)
        assert isinstance(raw, str), "Hermes ABC requires a string result"
        return _json.loads(raw)

    def test_recall_blank_query_error_payload(self, provider):
        out = self._call(provider, "revien_recall", {"query": ""})
        assert "Invalid query" in out["error"]

    @pytest.mark.parametrize("bad_top_n", [0, -1, 21])
    def test_recall_out_of_range_top_n_error_payload(self, provider, bad_top_n):
        out = self._call(provider, "revien_recall", {"query": "x", "top_n": bad_top_n})
        assert "Invalid top_n" in out["error"]

    def test_recall_non_integer_top_n_error_payload(self, provider):
        out = self._call(
            provider, "revien_recall", {"query": "x", "top_n": "lots"}
        )
        assert "Invalid top_n" in out["error"]

    def test_store_blank_content_error_payload(self, provider):
        out = self._call(provider, "revien_store", {"content": "  "})
        assert "Invalid content" in out["error"]

    # ── args honesty: never raise into Hermes ──

    @pytest.mark.parametrize("bad_args", [["query"], "query", 42])
    def test_non_dict_args_error_payload(self, provider, bad_args):
        """`args or {}` used to keep truthy non-dicts and AttributeError'd
        straight into Hermes — must be an error payload instead."""
        out = self._call(provider, "revien_recall", bad_args)
        assert "Invalid args" in out["error"]

    def test_none_args_treated_as_empty(self, provider):
        # None args -> {} -> blank query -> validation error, not a crash.
        out = self._call(provider, "revien_recall", None)
        assert "Invalid query" in out["error"]

    def test_engine_exception_becomes_error_payload(self, provider, monkeypatch):
        """An engine/pipeline exception must not propagate into Hermes —
        the never-raises contract covers the dispatch, not just validation."""
        def _boom(*a, **kw):
            raise RuntimeError("vec0 table corrupt")

        monkeypatch.setattr(provider._engine, "recall", _boom)
        out = self._call(provider, "revien_recall", {"query": "anything"})
        assert "internal error" in out["error"]
        assert "RuntimeError" in out["error"]

    # ── top_n coercion matrix (matches the pydantic/FastMCP faces) ──

    def test_top_n_bool_error_payload(self, provider):
        out = self._call(provider, "revien_recall", {"query": "x", "top_n": True})
        assert "Invalid top_n" in out["error"]

    def test_top_n_fractional_float_error_payload(self, provider):
        # 5.9 must be refused, not silently truncated to 5.
        out = self._call(provider, "revien_recall", {"query": "x", "top_n": 5.9})
        assert "Invalid top_n" in out["error"]

    @pytest.mark.parametrize("coercible", [5.0, "5", " 5 "])
    def test_top_n_int_valued_coerces(self, provider, coercible):
        out = self._call(
            provider, "revien_recall", {"query": "x", "top_n": coercible}
        )
        assert "error" not in out
        assert "results" in out

    @pytest.mark.parametrize("top_n", [TOP_N_MIN, TOP_N_MAX])
    def test_recall_boundary_top_n_pass(self, provider, top_n):
        out = self._call(provider, "revien_recall", {"query": "x", "top_n": top_n})
        assert "error" not in out
        assert "results" in out

    def test_store_valid_passes(self, provider):
        out = self._call(
            provider, "revien_store", {"content": "The staging box is blue."}
        )
        assert out["context_node_id"]


# ── Version unification ───────────────────────────────────


class TestVersionUnification:
    def test_package_version_is_0_3_0(self):
        assert revien.__version__ == "0.3.0"

    def test_health_reports_package_version(self, client):
        payload = client.get("/v1/health").json()
        assert payload["version"] == revien.__version__

    def test_app_metadata_reports_package_version(self, db_path):
        app = create_app(db_path=db_path)
        try:
            assert app.version == revien.__version__
        finally:
            # bare create_app (no TestClient lifespan) — release the store
            # so Windows can unlink the temp db.
            app.state.store.close()
