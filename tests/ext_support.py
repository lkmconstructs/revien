"""Shared helpers for the extension-registry tests.

``clean_registry`` is an autouse fixture the extension test modules import: it
gives every test an empty registry, no REVIEN_EXTENSIONS and no installed entry
points, so a test never sees another test's types, the developer's environment
or a package installed on the machine. The test that needs entry points puts a
fake distribution on a temp path and scans only that.
"""
import importlib
import importlib.metadata
import sqlite3
import sys
import textwrap
import uuid
from typing import Dict, List

import pytest

from revien import extensions

#: The real scanner, before the fixture below replaces it with an empty one.
REAL_ENTRY_POINTS = extensions._entry_points

#: A whole extension package in miniature: a callable that registers a node
#: type and an edge type, and a counter of how many times it ran.
TOY = '''
from revien.extensions import register_edge_type, register_node_type

CALLS = []


def register():
    CALLS.append(1)
    register_node_type("recipe", description="a dish and how to make it")
    register_edge_type("pairs_with", description="two dishes that suit each other")
'''


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.delenv(extensions.ENV_VAR, raising=False)
    extensions._reset_for_tests()
    monkeypatch.setattr(extensions, "_entry_points", lambda: [])
    yield
    extensions._reset_for_tests()


@pytest.fixture
def make_ext(tmp_path, monkeypatch, request):
    """make_ext(source) -> module name: write a module to a temp directory on
    sys.path (removed again at teardown)."""
    made: List[str] = []

    def make(source: str = TOY, name: str = "") -> str:
        name = name or f"revien_testext_{uuid.uuid4().hex[:8]}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
        monkeypatch.syspath_prepend(str(tmp_path))
        made.append(name)
        return name

    def forget_modules():
        for name in made:
            sys.modules.pop(name, None)

    request.addfinalizer(forget_modules)
    return make


def write_dist(root, dist_name: str, entry_points_txt: str) -> None:
    """A distribution's metadata directory, as pip would leave it."""
    info = root / f"{dist_name.replace('-', '_')}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(textwrap.dedent(entry_points_txt))


def scan_only(monkeypatch, root) -> None:
    """Make entry-point discovery use the real code on the distributions found
    in ``root`` only. The double answers like importlib.metadata.entry_points:
    every entry point of every group, unless a group is asked for."""
    def entry_points(**params):
        found = []
        for dist in importlib.metadata.distributions(path=[str(root)]):
            found.extend(dist.entry_points)
        group = params.get("group")
        return [ep for ep in found if group is None or ep.group == group]

    monkeypatch.setattr(extensions._metadata, "entry_points", entry_points)
    monkeypatch.setattr(extensions, "_entry_points", REAL_ENTRY_POINTS)


def register_toy() -> None:
    """The toy extension's registrations, in this process."""
    extensions.register_node_type("recipe", description="a dish and how to make it")
    extensions.register_edge_type("pairs_with", description="two dishes that suit each other")


def raw_rows(db: str, table: str) -> List[tuple]:
    """Every row of a table, read with plain sqlite (no Revien code)."""
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    finally:
        conn.close()


def raw_types(db: str) -> Dict[str, List[str]]:
    conn = sqlite3.connect(db)
    try:
        return {
            "nodes": sorted(r[0] for r in conn.execute("SELECT node_type FROM nodes")),
            "edges": sorted(r[0] for r in conn.execute("SELECT edge_type FROM edges")),
        }
    finally:
        conn.close()
