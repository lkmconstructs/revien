"""docs/EXTENSIONS.md is run, not just read: its example package and script are
extracted and executed, its entry point is the one Revien scans for, and the
error it quotes is the one Revien raises."""
import re
from pathlib import Path

from click.testing import CliRunner

from revien import extensions
from revien.cli import main
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from tests.ext_support import (  # noqa: F401  (clean_registry is autouse)
    clean_registry, make_ext, raw_types, scan_only, write_dist,
)

DOC = Path(__file__).resolve().parents[1] / "docs" / "EXTENSIONS.md"


def doc_text():
    return DOC.read_text(encoding="utf-8")


def python_files():
    """{file name: source} for the doc's ```python blocks that begin with a
    '# file: NAME' line."""
    found = re.findall(r"```python\n# file: (\S+)\n(.*?)```", doc_text(), re.S)
    return dict(found)


def test_the_doc_has_the_example_files():
    assert set(python_files()) == {"revien_recipes.py", "use_recipes.py"}


def test_the_example_package_and_script_run_as_written(tmp_path, monkeypatch):
    files = python_files()
    (tmp_path / "revien_recipes.py").write_text(files["revien_recipes.py"], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(extensions.ENV_VAR, "revien_recipes:register")
    try:
        exec(compile(files["use_recipes.py"], "use_recipes.py", "exec"), {"__name__": "__main__"})
    finally:
        import sys
        sys.modules.pop("revien_recipes", None)
    assert raw_types(str(tmp_path / "recipes.db")) == {
        "nodes": ["fact", "recipe"], "edges": ["pairs_with"]}
    assert extensions.registered_node_types()["meal_plan"].dedup is False
    assert extensions.registered_node_types()["recipe"].description == "a dish and how to make it"


def test_the_entry_point_the_doc_shows_is_found(tmp_path, monkeypatch):
    block = re.search(
        r'```toml\n[ \t]*\[project\.entry-points\."([^"]+)"\]\n[ \t]*(\w+) = "([^"]+)"\n[ \t]*```',
        doc_text())
    assert block, "the doc has no entry point snippet"
    group, name, target = block.groups()
    assert group == extensions.ENTRY_POINT_GROUP
    files = python_files()
    (tmp_path / "revien_recipes.py").write_text(files["revien_recipes.py"], encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    write_dist(site, "revien-recipes", f"[{group}]\n{name} = {target}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    scan_only(monkeypatch, site)
    try:
        assert list(extensions.load_extensions()) == [target]
    finally:
        import sys
        sys.modules.pop("revien_recipes", None)
    assert NodeType("recipe").value == "recipe" and EdgeType("pairs_with").value == "pairs_with"


def test_the_refusal_the_doc_quotes_is_the_one_raised(tmp_path, monkeypatch):
    quoted = re.search(r"```\n(Error: Cannot open the database .*?)```", doc_text(), re.S)
    assert quoted, "the doc does not quote the refusal"
    extensions.register_node_type("recipe")
    extensions.register_edge_type("pairs_with")
    db = str(tmp_path / "memory.db")
    s = GraphStore(db)
    nodes = [s.add_node(Node(node_type=NodeType("recipe"), label=f"r{i}", content=f"c{i}"))
             for i in range(12)]
    for i in range(4):
        s.add_edge(Edge(edge_type=EdgeType("pairs_with"), source_node_id=nodes[i].node_id,
                        target_node_id=nodes[i + 1].node_id))
    s.close()
    extensions._reset_for_tests()
    res = CliRunner().invoke(main, ["status", "--db", db])
    assert res.exit_code == 1
    actual = " ".join(res.output.replace(db, "/path/memory.db").split())
    assert actual == " ".join(quoted.group(1).split())


def test_the_doc_says_not_to_use_in_on_the_enum():
    text = " ".join(doc_text().split())
    assert "`x in NodeType` is not a reliable test for an extension value" in text
    assert "differs across Python versions" in text
    assert "`x in all_node_type_values()`" in text
