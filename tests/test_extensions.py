"""The extension registry (revien/extensions.py): registration rules, how the
enums accept registered values, discovery and loading, failing extensions,
databases holding types nobody registered, the dedup declaration, and that
built-in behaviour does not move.

The surfaces (store, REST, MCP, CLI, export/import, TOON, recall, forget) are
in tests/test_extensions_surfaces.py.
"""
import copy
import hashlib
import json
import pickle
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from revien import extensions
from revien.extensions import (
    ExtensionError, ExtensionLoadError, ExtensionRegistrationError,
    UnknownTypeError, load_extensions, register_edge_type, register_node_type,
)
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.dedup import Deduplicator
from revien.ingestion.extractor import ExtractionResult
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from tests.ext_support import (  # noqa: F401  (clean_registry is autouse)
    REAL_ENTRY_POINTS, TOY, clean_registry, make_ext, raw_rows, raw_types,
    register_toy, scan_only, write_dist,
)

BUILTIN_NODE_TYPES = [
    "entity", "topic", "decision", "fact", "preference", "event", "context",
    "action", "skill",
]
BUILTIN_EDGE_TYPES = [
    "related_to", "decided_in", "mentioned_by", "depends_on", "followed_by",
    "contradicts", "conflicts_with", "corrects", "derived_from", "alias_of",
]


# ── registration rules ────────────────────────────────────────────────────


class TestRegistration:
    def test_a_registered_node_type_is_accepted_by_the_enum(self):
        member = register_node_type("recipe", description="a dish")
        assert NodeType("recipe") is member
        assert member.value == "recipe"
        assert isinstance(member, NodeType)
        assert member == "recipe"

    def test_a_registered_edge_type_is_accepted_by_the_enum(self):
        member = register_edge_type("pairs_with")
        assert EdgeType("pairs_with") is member
        assert member.value == "pairs_with"
        assert isinstance(member, EdgeType)

    def test_before_registration_the_usual_error_stands(self):
        with pytest.raises(ValueError, match="'recipe' is not a valid NodeType"):
            NodeType("recipe")
        with pytest.raises(ValueError, match="'pairs_with' is not a valid EdgeType"):
            EdgeType("pairs_with")

    @pytest.mark.parametrize("bad", [
        "", "Recipe", "RECIPE", "meal-plan", "meal plan", "meal.plan", "_recipe",
        "recipe_", "meal__plan", "1recipe", "recipe\n", "\nrecipe", "récipe",
        "a" * 65,
    ])
    def test_the_value_must_be_lowercase_snake_case(self, bad):
        with pytest.raises(ExtensionRegistrationError, match="snake_case"):
            register_node_type(bad)
        with pytest.raises(ExtensionRegistrationError, match="snake_case"):
            register_edge_type(bad)
        assert extensions.registered_node_types() == {}
        assert extensions.registered_edge_types() == {}

    @pytest.mark.parametrize("good", ["recipe", "meal_plan", "a1", "x_2_y", "a" * 64])
    def test_lowercase_snake_case_values_are_accepted(self, good):
        assert register_node_type(good).value == good

    @pytest.mark.parametrize("bad", [None, 5, b"recipe", ["recipe"], NodeType.FACT])
    def test_the_value_must_be_a_plain_string(self, bad):
        with pytest.raises(ExtensionRegistrationError, match="plain str"):
            register_node_type(bad)

    @pytest.mark.parametrize("value", BUILTIN_NODE_TYPES)
    def test_a_builtin_node_type_cannot_be_registered(self, value):
        with pytest.raises(ExtensionRegistrationError, match="built-in"):
            register_node_type(value)
        assert extensions.registered_node_types() == {}
        assert NodeType(value).value == value

    @pytest.mark.parametrize("value", BUILTIN_EDGE_TYPES)
    def test_a_builtin_edge_type_cannot_be_registered(self, value):
        with pytest.raises(ExtensionRegistrationError, match="built-in"):
            register_edge_type(value)
        assert extensions.registered_edge_types() == {}

    def test_registering_the_same_thing_again_is_a_noop(self):
        first = register_node_type("recipe", description="a dish", dedup=False)
        again = register_node_type("recipe", description="a dish", dedup=False)
        assert again is first
        assert list(extensions.registered_node_types()) == ["recipe"]
        e1 = register_edge_type("pairs_with", description="x")
        assert register_edge_type("pairs_with", description="x") is e1
        assert list(extensions.registered_edge_types()) == ["pairs_with"]

    def test_registering_different_metadata_fails_and_keeps_the_first(self):
        register_node_type("recipe", description="a dish")
        with pytest.raises(ExtensionRegistrationError, match="different metadata"):
            register_node_type("recipe", description="a meal")
        with pytest.raises(ExtensionRegistrationError, match="different metadata"):
            register_node_type("recipe", description="a dish", dedup=False)
        spec = extensions.registered_node_types()["recipe"]
        assert (spec.description, spec.dedup) == ("a dish", True)
        register_edge_type("pairs_with", description="x")
        with pytest.raises(ExtensionRegistrationError, match="different metadata"):
            register_edge_type("pairs_with", description="y")
        assert extensions.registered_edge_types()["pairs_with"].description == "x"

    def test_metadata_is_type_checked(self):
        with pytest.raises(ExtensionRegistrationError, match="description"):
            register_node_type("recipe", description=5)
        with pytest.raises(ExtensionRegistrationError, match="description"):
            register_node_type("recipe", description="x" * 501)
        with pytest.raises(ExtensionRegistrationError, match="dedup"):
            register_node_type("recipe", dedup="no")
        with pytest.raises(ExtensionRegistrationError, match="description"):
            register_edge_type("pairs_with", description=None)
        assert extensions.registered_node_types() == {}

    def test_every_error_is_an_extension_error_and_a_value_error(self):
        with pytest.raises(ExtensionError):
            register_node_type("Bad")
        with pytest.raises(ValueError):
            register_node_type("fact")

    def test_node_and_edge_types_are_separate_namespaces(self):
        register_node_type("pairs_with")
        with pytest.raises(ValueError):
            EdgeType("pairs_with")
        register_edge_type("pairs_with")
        assert NodeType("pairs_with") is not EdgeType("pairs_with")

    def test_a_value_registered_only_as_an_edge_type_is_not_a_node_type(self):
        register_edge_type("pairs_with")
        assert EdgeType("pairs_with").value == "pairs_with"
        with pytest.raises(ValueError, match="'pairs_with' is not a valid NodeType"):
            NodeType("pairs_with")
        with pytest.raises(ValidationError):
            Node(node_type="pairs_with", label="l", content="c")
        assert extensions.all_node_type_values() == BUILTIN_NODE_TYPES

    def test_a_value_registered_only_as_a_node_type_is_not_an_edge_type(self):
        register_node_type("recipe")
        assert NodeType("recipe").value == "recipe"
        with pytest.raises(ValueError, match="'recipe' is not a valid EdgeType"):
            EdgeType("recipe")
        with pytest.raises(ValidationError):
            Edge(edge_type="recipe", source_node_id="a", target_node_id="b")
        assert extensions.all_edge_type_values() == BUILTIN_EDGE_TYPES

    def test_containment_in_the_enum_is_documented_as_unreliable(self):
        """docs/EXTENSIONS.md says what `x in NodeType` does for an extension
        value on each Python version; this is that sentence."""
        member = register_node_type("recipe")
        if sys.version_info >= (3, 12):
            assert (member in NodeType) is True
            assert ("recipe" in NodeType) is False
            assert ("fact" in NodeType) is True
        else:
            assert (member in NodeType) is False
            with pytest.raises(TypeError):
                "recipe" in NodeType
            with pytest.raises(TypeError):
                "fact" in NodeType
        assert "recipe" in extensions.all_node_type_values()

    def test_introspection(self):
        assert extensions.all_node_type_values() == BUILTIN_NODE_TYPES
        assert extensions.all_edge_type_values() == BUILTIN_EDGE_TYPES
        register_node_type("recipe", description="a dish")
        register_node_type("meal_plan", dedup=False)
        register_edge_type("pairs_with", description="suits")
        nodes = extensions.registered_node_types()
        assert list(nodes) == ["recipe", "meal_plan"]
        assert nodes["recipe"] == extensions.NodeTypeSpec("recipe", "a dish", True)
        assert nodes["meal_plan"].dedup is False
        assert extensions.registered_edge_types() == {
            "pairs_with": extensions.EdgeTypeSpec("pairs_with", "suits")}
        assert extensions.all_node_type_values() == BUILTIN_NODE_TYPES + ["recipe", "meal_plan"]
        assert extensions.all_edge_type_values() == BUILTIN_EDGE_TYPES + ["pairs_with"]

    def test_introspection_returns_copies(self):
        register_node_type("recipe")
        extensions.registered_node_types().clear()
        extensions.registered_node_types()["x"] = None
        assert list(extensions.registered_node_types()) == ["recipe"]
        assert NodeType("recipe").value == "recipe"

    def test_concurrent_conflicting_registrations_let_exactly_one_win(self, monkeypatch):
        """The check and the insert are one step. A pause between them (here
        a sleep while the member is made) lets every thread pass the check
        unless the registry serialises them."""
        real = extensions._make_member

        def slow(enum_cls, value):
            time.sleep(0.05)
            return real(enum_cls, value)

        monkeypatch.setattr(extensions, "_make_member", slow)
        n = 8
        barrier = threading.Barrier(n)
        outcomes = []

        def attempt(i):
            barrier.wait()
            try:
                outcomes.append(("ok", register_node_type("recipe", description=f"d{i}")))
            except ExtensionRegistrationError:
                outcomes.append(("conflict", None))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winners = [m for kind, m in outcomes if kind == "ok"]
        assert len(winners) == 1 and len(outcomes) == n
        assert NodeType("recipe") is winners[0]
        won = extensions.registered_node_types()["recipe"].description
        assert won in {f"d{i}" for i in range(n)}

    def test_concurrent_identical_registrations_share_one_member(self, monkeypatch):
        real = extensions._make_member

        def slow(enum_cls, value):
            time.sleep(0.05)
            return real(enum_cls, value)

        monkeypatch.setattr(extensions, "_make_member", slow)
        n = 8
        barrier = threading.Barrier(n)
        got = []

        def attempt():
            barrier.wait()
            got.append(register_node_type("recipe", description="same"))

        threads = [threading.Thread(target=attempt) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(got) == n and len({id(m) for m in got}) == 1
        assert NodeType("recipe") is got[0]


# ── how the enums accept registered values ────────────────────────────────


class TestEnumMechanism:
    def test_the_member_behaves_like_a_builtin_str_enum_member(self):
        m = register_node_type("recipe")
        assert (m.name, m.value) == ("RECIPE", "recipe")
        assert repr(m) == "<NodeType.RECIPE: 'recipe'>"
        assert m == "recipe" and "recipe" == m
        assert hash(m) == hash("recipe")
        assert {m: 1}["recipe"] == 1 and {"recipe": 1}[m] == 1
        assert NodeType(m) is m
        assert json.dumps({"t": m}) == '{"t": "recipe"}'
        assert m.upper() == "RECIPE"                   # it is a str

    def test_one_object_per_value(self):
        m = register_node_type("recipe")
        assert NodeType("recipe") is m and NodeType("recipe") is NodeType("recipe")

    def test_pickle_and_copy_return_the_same_object(self):
        m = register_node_type("recipe")
        assert pickle.loads(pickle.dumps(m)) is m
        assert copy.copy(m) is m and copy.deepcopy(m) is m

    def test_iteration_and_members_still_list_only_the_builtins(self):
        register_node_type("recipe")
        register_edge_type("pairs_with")
        assert [m.value for m in NodeType] == BUILTIN_NODE_TYPES
        assert [m.value for m in EdgeType] == BUILTIN_EDGE_TYPES
        assert "RECIPE" not in NodeType.__members__
        assert len(NodeType) == len(BUILTIN_NODE_TYPES)

    @pytest.mark.parametrize("junk", [5, None, 1.5, b"recipe", [], {}, ("recipe",)])
    def test_a_non_string_is_still_a_value_error(self, junk):
        register_node_type("recipe")
        with pytest.raises(ValueError):
            NodeType(junk)

    def test_pydantic_models_accept_registered_values(self):
        register_node_type("recipe")
        register_edge_type("pairs_with")
        node = Node(node_type="recipe", label="Soup", content="tomato soup")
        assert node.node_type is NodeType("recipe")
        assert node.model_dump(mode="json")["node_type"] == "recipe"
        assert json.loads(node.model_dump_json())["node_type"] == "recipe"
        again = Node.model_validate_json(node.model_dump_json())
        assert again.node_type is node.node_type
        edge = Edge(edge_type="pairs_with", source_node_id="a", target_node_id="b")
        assert edge.edge_type is EdgeType("pairs_with")
        assert json.loads(edge.model_dump_json())["edge_type"] == "pairs_with"
        graph = Graph.model_validate({"nodes": [node.model_dump(mode="json")],
                                      "edges": [edge.model_dump(mode="json")]})
        assert graph.nodes[0].node_type.value == "recipe"
        assert graph.edges[0].edge_type.value == "pairs_with"

    def test_pydantic_rejects_an_unregistered_value(self):
        with pytest.raises(ValidationError):
            Node(node_type="recipe", label="Soup", content="x")
        with pytest.raises(ValidationError):
            Edge(edge_type="pairs_with", source_node_id="a", target_node_id="b")
        register_node_type("recipe")
        with pytest.raises(ValidationError):
            Node(node_type="meal_plan", label="Soup", content="x")

    def test_the_builtins_are_exactly_what_they_were(self):
        """The enum's own surface does not move when types are registered."""
        before = (
            [(m.name, m.value) for m in NodeType], [(m.name, m.value) for m in EdgeType],
            Node.model_json_schema()["$defs"]["NodeType"],
            Edge.model_json_schema()["$defs"]["EdgeType"],
        )
        register_node_type("recipe")
        register_edge_type("pairs_with")
        after = (
            [(m.name, m.value) for m in NodeType], [(m.name, m.value) for m in EdgeType],
            Node.model_json_schema()["$defs"]["NodeType"],
            Edge.model_json_schema()["$defs"]["EdgeType"],
        )
        assert before == after
        assert before[2]["enum"] == BUILTIN_NODE_TYPES
        assert before[3]["enum"] == BUILTIN_EDGE_TYPES

    def test_invalid_builtin_values_fail_with_the_message_they_always_had(self):
        with pytest.raises(ValueError) as e:
            NodeType("bogus")
        assert str(e.value) == "'bogus' is not a valid NodeType"
        with pytest.raises(ValueError) as e:
            EdgeType("bogus")
        assert str(e.value) == "'bogus' is not a valid EdgeType"

    def test_the_specs_default_to_the_documented_values(self):
        assert extensions.NodeTypeSpec("x") == extensions.NodeTypeSpec("x", "", True)
        assert extensions.EdgeTypeSpec("x") == extensions.EdgeTypeSpec("x", "")
        register_node_type("recipe")
        register_edge_type("pairs_with")
        assert extensions.registered_node_types()["recipe"] == extensions.NodeTypeSpec("recipe")
        assert extensions.registered_edge_types()["pairs_with"] == extensions.EdgeTypeSpec("pairs_with")

    def test_registering_leaves_builtin_dedup_alone(self):
        register_node_type("recipe", dedup=False)
        for member in NodeType:
            assert extensions.node_type_dedups(member) is True


# ── discovery and loading ─────────────────────────────────────────────────


class TestEnvDiscovery:
    def test_the_env_var_names_the_function_that_registers(self, monkeypatch, make_ext):
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        loaded = load_extensions()
        assert loaded == {f"{mod}:register": extensions.ENV_VAR}
        assert NodeType("recipe").value == "recipe"
        assert EdgeType("pairs_with").value == "pairs_with"
        assert extensions.loaded_extensions() == loaded

    def test_the_env_list_loads_in_the_order_given(self, monkeypatch, make_ext):
        a = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("first_kind")
        ''')
        b = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("second_kind")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{b}:go,{a}:go")
        assert list(load_extensions()) == [f"{b}:go", f"{a}:go"]
        assert list(extensions.registered_node_types()) == ["second_kind", "first_kind"]

    def test_a_list_with_spaces_and_empty_items(self, monkeypatch, make_ext):
        a = make_ext(TOY)
        b = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f" {a}:register , ,, {b}:go ,")
        assert set(load_extensions()) == {f"{a}:register", f"{b}:go"}
        assert NodeType("meal_plan").value == "meal_plan"

    def test_a_dotted_callable_path_is_followed(self, monkeypatch, make_ext):
        mod = make_ext('''
            from revien.extensions import register_node_type
            class Plugin:
                @staticmethod
                def setup():
                    register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:Plugin.setup")
        load_extensions()
        assert NodeType("meal_plan").value == "meal_plan"

    def test_opening_a_store_loads_them(self, monkeypatch, make_ext, tmp_path):
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        assert extensions.registered_node_types() == {}
        GraphStore(str(tmp_path / "s.db")).close()
        assert list(extensions.registered_node_types()) == ["recipe"]

    def test_each_extension_runs_once_however_many_stores_open(self, monkeypatch, make_ext, tmp_path):
        import sys
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        for i in range(3):
            GraphStore(str(tmp_path / f"s{i}.db")).close()
        load_extensions()
        load_extensions()
        assert len(sys.modules[mod].CALLS) == 1

    def test_a_newly_configured_extension_is_picked_up_without_rerunning_the_old(
            self, monkeypatch, make_ext):
        import sys
        a = make_ext(TOY)
        b = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register")
        load_extensions()
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register,{b}:go")
        load_extensions()
        assert len(sys.modules[a].CALLS) == 1
        assert NodeType("meal_plan").value == "meal_plan"

    def test_a_failing_item_added_after_a_load_is_an_error_at_the_next_call(
            self, monkeypatch, make_ext, tmp_path):
        good = make_ext(TOY)
        monkeypatch.setenv(extensions.ENV_VAR, f"{good}:register")
        load_extensions()
        monkeypatch.setenv(extensions.ENV_VAR, f"{good}:register,no_such_module_xyz:go")
        with pytest.raises(ExtensionLoadError, match="no_such_module_xyz:go"):
            load_extensions()
        with pytest.raises(ExtensionLoadError):
            GraphStore(str(tmp_path / "s.db"))
        assert NodeType("recipe").value == "recipe"            # the good one stays

    def test_an_item_removed_after_a_load_stays_loaded_and_is_not_run_again(
            self, monkeypatch, make_ext):
        a = make_ext(TOY)
        b = make_ext('''
            from revien.extensions import register_node_type
            CALLS = []
            def go():
                CALLS.append(1)
                register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register,{b}:go")
        load_extensions()
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register")
        assert set(load_extensions()) == {f"{a}:register", f"{b}:go"}
        monkeypatch.delenv(extensions.ENV_VAR)
        assert set(load_extensions()) == {f"{a}:register", f"{b}:go"}
        assert NodeType("meal_plan").value == "meal_plan"
        assert len(sys.modules[b].CALLS) == 1

    def test_a_refused_database_opens_once_the_env_names_the_extension(
            self, monkeypatch, make_ext, tmp_path):
        """The same process: refused, then configured, then open."""
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        with pytest.raises(UnknownTypeError):
            GraphStore(db)
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        s = GraphStore(db)
        try:
            assert sorted(n.node_type.value for n in s.list_nodes()) == [
                "fact", "recipe", "recipe"]
        finally:
            s.close()

    def test_a_second_store_picks_up_an_item_added_since_the_first(
            self, monkeypatch, make_ext, tmp_path):
        a = make_ext(TOY)
        b = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register")
        GraphStore(str(tmp_path / "one.db")).close()
        assert "meal_plan" not in extensions.registered_node_types()
        monkeypatch.setenv(extensions.ENV_VAR, f"{a}:register,{b}:go")
        GraphStore(str(tmp_path / "two.db")).close()
        assert "meal_plan" in extensions.registered_node_types()

    def test_concurrent_loads_run_the_extension_once(self, monkeypatch, make_ext):
        import sys
        mod = make_ext('''
            import time
            from revien.extensions import register_node_type
            CALLS = []
            def register():
                CALLS.append(1)
                time.sleep(0.1)
                register_node_type("recipe")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        n = 8
        barrier = threading.Barrier(n)

        def go():
            barrier.wait()
            load_extensions()

        threads = [threading.Thread(target=go) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(sys.modules[mod].CALLS) == 1
        assert NodeType("recipe").value == "recipe"

    def test_an_extension_that_opens_a_store_while_loading_does_not_loop(
            self, monkeypatch, make_ext, tmp_path):
        import sys
        mod = make_ext(f'''
            from revien.extensions import register_node_type
            from revien.graph.store import GraphStore
            CALLS = []
            def register():
                CALLS.append(1)
                register_node_type("recipe")
                GraphStore({str(tmp_path / "inner.db")!r}).close()
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        load_extensions()
        assert len(sys.modules[mod].CALLS) == 1


@pytest.fixture
def real_scan(monkeypatch, tmp_path):
    """The real entry-point scanner over an empty site directory. The autouse
    fixture replaces the scanner with one that always answers "nothing", which
    leaves the scan cache cold, so the loader's warm path (the one every store
    open after the first takes in a real process) never runs under it."""
    site = tmp_path / "empty_site"
    site.mkdir()
    scan_only(monkeypatch, site)
    return site


class TestAWarmLoader:
    """After the first load the scan is cached and a call with an unchanged
    REVIEN_EXTENSIONS returns at once. Every change to the variable after
    that must still be seen."""

    def warm(self, monkeypatch, make_ext, spec_body=TOY):
        mod = make_ext(spec_body)
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        load_extensions()
        assert extensions._entry_points_cache is not None, "the cache is warm"
        return mod

    def test_later_calls_scan_nothing_and_run_nothing_again(self, real_scan, monkeypatch, make_ext, tmp_path):
        mod = self.warm(monkeypatch, make_ext)
        scans = []
        real = extensions._metadata.entry_points

        def counting(**params):
            scans.append(params)
            return real(**params)

        monkeypatch.setattr(extensions._metadata, "entry_points", counting)
        for i in range(3):
            load_extensions()
            GraphStore(str(tmp_path / f"s{i}.db")).close()
        assert scans == []
        assert len(sys.modules[mod].CALLS) == 1

    def test_an_item_added_after_the_load_is_loaded(self, real_scan, monkeypatch, make_ext):
        first = self.warm(monkeypatch, make_ext)
        second = make_ext('''
            from revien.extensions import register_node_type
            def go():
                register_node_type("meal_plan")
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{first}:register,{second}:go")
        assert set(load_extensions()) == {f"{first}:register", f"{second}:go"}
        assert NodeType("meal_plan").value == "meal_plan"
        assert len(sys.modules[first].CALLS) == 1

    def test_a_failing_item_added_after_the_load_is_an_error(self, real_scan, monkeypatch, make_ext, tmp_path):
        first = self.warm(monkeypatch, make_ext)
        monkeypatch.setenv(extensions.ENV_VAR, f"{first}:register,no_such_module_xyz:go")
        with pytest.raises(ExtensionLoadError, match="no_such_module_xyz:go"):
            load_extensions()
        with pytest.raises(ExtensionLoadError):
            GraphStore(str(tmp_path / "s.db"))
        monkeypatch.setenv(extensions.ENV_VAR, f"{first}:register")      # fixed again
        load_extensions()
        GraphStore(str(tmp_path / "s.db")).close()

    def test_an_item_removed_after_the_load_stays_loaded(self, real_scan, monkeypatch, make_ext):
        first = self.warm(monkeypatch, make_ext)
        monkeypatch.delenv(extensions.ENV_VAR)
        assert list(load_extensions()) == [f"{first}:register"]
        assert NodeType("recipe").value == "recipe"
        assert len(sys.modules[first].CALLS) == 1

    def test_a_refused_database_opens_once_the_env_names_the_extension(
            self, real_scan, monkeypatch, make_ext, tmp_path):
        load_extensions()                                    # warm, nothing configured
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        load_extensions()
        with pytest.raises(UnknownTypeError):
            GraphStore(db)
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        GraphStore(db).close()


class TestEntryPointDiscovery:
    def test_an_installed_package_is_found_by_its_entry_point(
            self, monkeypatch, make_ext, tmp_path):
        mod = make_ext()
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-recipes", f"""
            [revien.extensions]
            recipes = {mod}:register
        """)
        scan_only(monkeypatch, dists)
        loaded = load_extensions()
        assert loaded == {f"{mod}:register":
                          "entry point 'recipes' of package 'fake-recipes'"}
        assert NodeType("recipe").value == "recipe"
        assert EdgeType("pairs_with").value == "pairs_with"

    def test_an_entry_point_loads_when_a_store_opens(self, monkeypatch, make_ext, tmp_path):
        mod = make_ext()
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-recipes", f"[revien.extensions]\nrecipes = {mod}:register\n")
        scan_only(monkeypatch, dists)
        GraphStore(str(tmp_path / "s.db")).close()
        assert list(extensions.registered_node_types()) == ["recipe"]

    def test_only_the_revien_group_is_read(self, monkeypatch, make_ext, tmp_path):
        import sys
        mod = make_ext()
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-other", f"""
            [some.other.group]
            recipes = {mod}:register
            [console_scripts]
            recipes-cli = {mod}:register
        """)
        scan_only(monkeypatch, dists)
        assert load_extensions() == {}
        assert mod not in sys.modules
        assert extensions.registered_node_types() == {}

    def test_entry_points_load_sorted_by_name_not_in_file_order(self, monkeypatch, make_ext, tmp_path):
        mods = {name: make_ext(f'''
            from revien.extensions import register_node_type
            def register():
                register_node_type("{name}_kind")
        ''') for name in ("mid", "zeta", "alpha")}
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-three", "[revien.extensions]\n" + "".join(
            f"{name} = {mods[name]}:register\n" for name in ("mid", "zeta", "alpha")))
        scan_only(monkeypatch, dists)
        load_extensions()
        assert list(extensions.registered_node_types()) == ["alpha_kind", "mid_kind", "zeta_kind"]

    def test_entry_points_load_in_name_order(self, monkeypatch, make_ext, tmp_path):
        """Two packages that register the same value with different metadata:
        the first loaded wins and the other is reported, the same way every
        time."""
        def body(desc):
            return f'''
                from revien.extensions import register_node_type
                def register():
                    register_node_type("recipe", description="{desc}")
            '''
        first, second = make_ext(body("from alpha")), make_ext(body("from zeta"))
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-two", f"""
            [revien.extensions]
            zeta = {second}:register
            alpha = {first}:register
        """)
        scan_only(monkeypatch, dists)
        with pytest.raises(ExtensionLoadError) as e:
            load_extensions()
        assert [spec for spec, _, _ in e.value.failures] == [f"{second}:register"]
        assert extensions.registered_node_types()["recipe"].description == "from alpha"

    def test_the_same_target_from_entry_point_and_env_runs_once(
            self, monkeypatch, make_ext, tmp_path):
        import sys
        mod = make_ext()
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-recipes", f"[revien.extensions]\nrecipes = {mod}:register\n")
        scan_only(monkeypatch, dists)
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        load_extensions()
        assert len(sys.modules[mod].CALLS) == 1

    def test_a_broken_entry_point_is_an_error_naming_the_package(
            self, monkeypatch, tmp_path):
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-broken", """
            [revien.extensions]
            gone = no_such_module_xyz:register
        """)
        scan_only(monkeypatch, dists)
        with pytest.raises(ExtensionLoadError) as e:
            load_extensions()
        text = str(e.value)
        assert "no_such_module_xyz:register" in text
        assert "entry point 'gone' of package 'fake-broken'" in text
        assert "ModuleNotFoundError" in text

    def test_unreadable_package_metadata_is_an_error_not_a_skip(self, monkeypatch):
        def boom(*, group):
            raise OSError("metadata unreadable")

        monkeypatch.setattr(extensions._metadata, "entry_points", boom)
        monkeypatch.setattr(extensions, "_entry_points", REAL_ENTRY_POINTS)
        with pytest.raises(ExtensionLoadError, match="metadata unreadable"):
            load_extensions()


class TestOnlyConfiguredModulesAreImported:
    # named the way a convention-based finder would look for it
    SNEAKY = '''
        import pathlib
        pathlib.Path(__file__).with_name("sneaky.flag").write_text("imported")
        def register():
            raise AssertionError("never configured")
    '''

    def test_a_module_that_is_not_configured_is_never_imported(
            self, monkeypatch, make_ext, tmp_path):
        import sys
        mod = make_ext(self.SNEAKY, name="revien_ext_sneaky")
        GraphStore(str(tmp_path / "s.db")).close()
        load_extensions()
        assert not (tmp_path / "sneaky.flag").exists()
        assert mod not in sys.modules

    def test_control_the_same_module_is_imported_once_configured(
            self, monkeypatch, make_ext, tmp_path):
        mod = make_ext(self.SNEAKY, name="revien_ext_sneaky")
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        with pytest.raises(ExtensionLoadError):
            load_extensions()
        assert (tmp_path / "sneaky.flag").exists()

    def test_nothing_configured_means_nothing_loaded(self, tmp_path):
        assert load_extensions() == {}
        GraphStore(str(tmp_path / "s.db")).close()
        assert extensions.loaded_extensions() == {}


class TestFailingExtensions:
    def fail(self, monkeypatch, spec):
        monkeypatch.setenv(extensions.ENV_VAR, spec)
        with pytest.raises(ExtensionLoadError) as e:
            load_extensions()
        return e.value

    def test_a_missing_module(self, monkeypatch):
        err = self.fail(monkeypatch, "no_such_module_xyz:register")
        assert "no_such_module_xyz:register" in str(err)
        assert "REVIEN_EXTENSIONS" in str(err)
        assert "ModuleNotFoundError" in str(err)
        assert isinstance(err.__cause__, ModuleNotFoundError)

    def test_a_missing_function(self, monkeypatch, make_ext):
        mod = make_ext()
        err = self.fail(monkeypatch, f"{mod}:nope")
        assert f"{mod}:nope" in str(err) and "AttributeError" in str(err)

    def test_something_that_is_not_callable(self, monkeypatch, make_ext):
        mod = make_ext("VALUE = 5\n")
        err = self.fail(monkeypatch, f"{mod}:VALUE")
        assert f"{mod}:VALUE" in str(err) and "TypeError" in str(err)
        assert "point it at the function that registers the types" in str(err)

    def test_a_function_that_raises(self, monkeypatch, make_ext):
        mod = make_ext('''
            def register():
                raise RuntimeError("the cupboard is bare")
        ''')
        err = self.fail(monkeypatch, f"{mod}:register")
        assert "RuntimeError: the cupboard is bare" in str(err)
        assert err.failures[0][0] == f"{mod}:register"

    def test_a_module_that_fails_while_importing(self, monkeypatch, make_ext):
        mod = make_ext("raise ImportError('missing dependency of the extension')\n")
        err = self.fail(monkeypatch, f"{mod}:register")
        assert "missing dependency of the extension" in str(err)

    def test_an_extension_whose_registration_is_refused(self, monkeypatch, make_ext):
        mod = make_ext('''
            from revien.extensions import register_node_type
            def register():
                register_node_type("fact")
        ''')
        err = self.fail(monkeypatch, f"{mod}:register")
        assert "built-in" in str(err) and "ExtensionRegistrationError" in str(err)

    @pytest.mark.parametrize("spec", [
        "register", "pkg.mod:", ":register", "pkg mod:register", "pkg-mod:register",
        "pkg.mod:register extra", "pkg..mod:register", "1pkg:register",
        "pkg.mod:fn[extra]", "pkg/mod:register",
    ])
    def test_a_badly_written_spec_is_an_error(self, monkeypatch, spec):
        err = self.fail(monkeypatch, spec)
        assert "package.module:callable" in str(err)

    def test_a_spec_named_twice_is_reported_once(self, monkeypatch):
        err = self.fail(monkeypatch, "no_such_module_xyz:go, no_such_module_xyz:go")
        assert [spec for spec, _, _ in err.failures] == ["no_such_module_xyz:go"]

    def test_a_spec_named_by_an_entry_point_and_the_env_is_reported_once(
            self, monkeypatch, tmp_path):
        dists = tmp_path / "site"
        dists.mkdir()
        write_dist(dists, "fake-broken", """
            [revien.extensions]
            gone = no_such_module_xyz:go
        """)
        scan_only(monkeypatch, dists)
        err = self.fail(monkeypatch, "no_such_module_xyz:go")
        assert [(spec, where) for spec, where, _ in err.failures] == [
            ("no_such_module_xyz:go", "entry point 'gone' of package 'fake-broken'")]

    def test_one_failure_does_not_stop_the_others_but_is_still_reported(
            self, monkeypatch, make_ext):
        good = make_ext(TOY)
        err = self.fail(monkeypatch, f"no_such_module_xyz:go,{good}:register,also_missing_abc:go")
        assert len(err.failures) == 2
        assert "no_such_module_xyz:go" in str(err) and "also_missing_abc:go" in str(err)
        assert NodeType("recipe").value == "recipe"           # the good one ran

    def test_an_extension_that_calls_sys_exit_is_a_load_error_not_a_quiet_exit(
            self, monkeypatch, make_ext):
        mod = make_ext('''
            import sys
            def register():
                sys.exit(0)
        ''')
        err = self.fail(monkeypatch, f"{mod}:register")
        assert f"{mod}:register" in str(err) and "called sys.exit(0)" in str(err)
        assert isinstance(err.failures[0][2], SystemExit)
        assert isinstance(err.__cause__, SystemExit)

    def test_sys_exit_while_importing_and_with_other_codes_and_none(self, monkeypatch, make_ext):
        importing = make_ext("import sys\nsys.exit('giving up')\n")
        silent = make_ext("import sys\ndef register():\n    sys.exit()\n")
        err = self.fail(monkeypatch, f"{importing}:register,{silent}:register")
        text = str(err)
        assert "called sys.exit('giving up')" in text and "called sys.exit(None)" in text
        assert len(err.failures) == 2

    def test_any_other_base_exception_is_a_load_error(self, monkeypatch, make_ext):
        mod = make_ext('''
            class Stop(BaseException):
                pass
            def register():
                raise Stop("not an Exception")
        ''')
        err = self.fail(monkeypatch, f"{mod}:register")
        assert "Stop: not an Exception" in str(err)

    def test_the_others_still_load_when_one_exits(self, monkeypatch, make_ext):
        quitter = make_ext("import sys\ndef register():\n    sys.exit(2)\n")
        good = make_ext(TOY)
        err = self.fail(monkeypatch, f"{quitter}:register,{good}:register")
        assert [spec for spec, _, _ in err.failures] == [f"{quitter}:register"]
        assert NodeType("recipe").value == "recipe"

    def test_a_store_does_not_open_when_an_extension_exits(self, monkeypatch, make_ext, tmp_path):
        mod = make_ext("import sys\ndef register():\n    sys.exit(0)\n")
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        db = tmp_path / "s.db"
        with pytest.raises(ExtensionLoadError):
            GraphStore(str(db))
        assert not db.exists()

    def test_the_cli_exits_1_with_the_error_when_an_extension_exits_0(
            self, monkeypatch, make_ext, tmp_path):
        from click.testing import CliRunner
        from revien.cli import main
        mod = make_ext("import sys\ndef register():\n    sys.exit(0)\n")
        db = str(tmp_path / "s.db")
        GraphStore(db).close()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        res = CliRunner().invoke(main, ["status", "--db", db])
        assert res.exit_code == 1
        assert res.output.startswith("Error: ") and "called sys.exit(0)" in res.output
        assert f"{mod}:register" in res.output

    def test_a_keyboard_interrupt_is_not_swallowed(self, monkeypatch, make_ext):
        mod = make_ext('''
            def register():
                raise KeyboardInterrupt()
        ''')
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        with pytest.raises(KeyboardInterrupt):
            load_extensions()
        # nothing is stuck: it was not marked loaded, and a fixed config loads
        assert extensions.loaded_extensions() == {}
        good = make_ext(TOY)
        monkeypatch.setenv(extensions.ENV_VAR, f"{good}:register")
        assert f"{good}:register" in load_extensions()
        assert NodeType("recipe").value == "recipe"

    def test_types_registered_before_the_failure_stay_registered(self, monkeypatch, make_ext):
        mod = make_ext('''
            from revien.extensions import register_node_type
            def register():
                register_node_type("recipe")
                raise RuntimeError("half done")
        ''')
        self.fail(monkeypatch, f"{mod}:register")
        assert NodeType("recipe").value == "recipe"

    def test_a_failure_is_raised_again_on_every_call_not_skipped_once_seen(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv(extensions.ENV_VAR, "no_such_module_xyz:register")
        for _ in range(3):
            with pytest.raises(ExtensionLoadError):
                load_extensions()
        for i in range(2):
            with pytest.raises(ExtensionLoadError):
                GraphStore(str(tmp_path / f"s{i}.db"))

    def test_a_store_does_not_open_when_an_extension_fails(self, monkeypatch, tmp_path):
        monkeypatch.setenv(extensions.ENV_VAR, "no_such_module_xyz:register")
        db = tmp_path / "s.db"
        with pytest.raises(ExtensionLoadError):
            GraphStore(str(db))
        assert not db.exists()

    def test_once_fixed_the_next_call_loads_it(self, monkeypatch, make_ext):
        monkeypatch.setenv(extensions.ENV_VAR, "no_such_module_xyz:register")
        with pytest.raises(ExtensionLoadError):
            load_extensions()
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        assert f"{mod}:register" in load_extensions()
        assert NodeType("recipe").value == "recipe"

    def test_a_failure_in_the_cli_is_a_one_line_error(self, monkeypatch, tmp_path):
        from click.testing import CliRunner
        from revien.cli import main
        db = str(tmp_path / "s.db")
        GraphStore(db).close()
        monkeypatch.setenv(extensions.ENV_VAR, "no_such_module_xyz:register")
        res = CliRunner().invoke(main, ["status", "--db", db])
        assert res.exit_code == 1
        assert res.output.startswith("Error: ") and "no_such_module_xyz:register" in res.output
        assert "Traceback" not in res.output


class TestErrorsCrossProcessBoundaries:
    """A process pool pickles the exception a worker raised."""

    def test_load_error(self, monkeypatch):
        monkeypatch.setenv(extensions.ENV_VAR, "no_such_module_xyz:register")
        with pytest.raises(ExtensionLoadError) as e:
            load_extensions()
        again = pickle.loads(pickle.dumps(e.value))
        assert isinstance(again, ExtensionLoadError)
        assert str(again) == str(e.value)
        assert [(spec, where) for spec, where, _ in again.failures] == [
            (spec, where) for spec, where, _ in e.value.failures]

    def test_unknown_type_error(self, tmp_path):
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        with pytest.raises(UnknownTypeError) as e:
            GraphStore(db)
        again = pickle.loads(pickle.dumps(e.value))
        assert str(again) == str(e.value)
        assert (again.node_types, again.edge_types) == (e.value.node_types, e.value.edge_types)

    def test_registration_error(self):
        with pytest.raises(ExtensionRegistrationError) as e:
            register_node_type("Bad")
        assert str(pickle.loads(pickle.dumps(e.value))) == str(e.value)


# ── a database holding types nobody registered ───────────────────────────


def seed_with_extension_types(db):
    """A database holding a recipe, a fact and a pairs_with edge, written
    while the toy extension was registered, then handed back to a process
    that has not registered it."""
    register_toy()
    s = GraphStore(db)
    r1 = s.add_node(Node(node_type=NodeType("recipe"), label="Soup", content="tomato soup"))
    r2 = s.add_node(Node(node_type=NodeType("recipe"), label="Stew", content="bean stew"))
    f = s.add_node(Node(node_type=NodeType.FACT, label="Bread", content="sourdough"))
    s.add_edge(Edge(edge_type=EdgeType("pairs_with"), source_node_id=r1.node_id,
                    target_node_id=f.node_id))
    s.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=r2.node_id,
                    target_node_id=f.node_id))
    s.close()
    extensions._reset_for_tests()
    return r1, r2, f


class TestUnknownTypesAtOpen:
    def test_the_open_fails_naming_the_types_the_counts_and_the_fix(self, tmp_path):
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        with pytest.raises(UnknownTypeError) as e:
            GraphStore(db)
        text = str(e.value)
        assert "node types 'recipe' (2 nodes)" in text
        assert "edge types 'pairs_with' (1 edge)" in text
        assert "REVIEN_EXTENSIONS" in text and "revien.extensions" in text
        assert "docs/EXTENSIONS.md" in text
        # both causes are named, and neither is picked
        assert "not installed or not loaded" in text and "newer Revien" in text
        assert e.value.node_types == {"recipe": 2}
        assert e.value.edge_types == {"pairs_with": 1}
        assert isinstance(e.value, ValueError)                    # old handlers still catch it

    def test_no_row_is_read_as_another_type_changed_or_dropped(self, tmp_path):
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        before = (raw_rows(db, "nodes"), raw_rows(db, "edges"))
        for _ in range(2):
            with pytest.raises(UnknownTypeError):
                GraphStore(db)
        assert (raw_rows(db, "nodes"), raw_rows(db, "edges")) == before
        assert raw_types(db) == {"nodes": ["fact", "recipe", "recipe"],
                                 "edges": ["pairs_with", "related_to"]}

    def test_only_what_is_missing_is_named(self, tmp_path):
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        register_node_type("recipe", description="a dish and how to make it")
        with pytest.raises(UnknownTypeError) as e:
            GraphStore(db)
        assert e.value.node_types == {} and e.value.edge_types == {"pairs_with": 1}
        assert "node types" not in str(e.value)

    def test_registered_the_same_database_opens_and_keeps_every_row(self, tmp_path, monkeypatch, make_ext):
        db = str(tmp_path / "s.db")
        r1, r2, f = seed_with_extension_types(db)
        before = raw_rows(db, "nodes")
        mod = make_ext()
        monkeypatch.setenv(extensions.ENV_VAR, f"{mod}:register")
        s = GraphStore(db)
        try:
            assert s.get_node(r1.node_id).node_type.value == "recipe"
            assert sorted(n.node_type.value for n in s.list_nodes()) == ["fact", "recipe", "recipe"]
        finally:
            s.close()
        assert raw_rows(db, "nodes") == before

    def test_the_refused_open_leaves_no_connection_open(self, tmp_path, monkeypatch):
        db = str(tmp_path / "s.db")
        seed_with_extension_types(db)
        opened = []
        real = sqlite3.connect

        def spy(*args, **kwargs):
            conn = real(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(sqlite3, "connect", spy)
        with pytest.raises(UnknownTypeError):
            GraphStore(db)
        assert opened, "the check looked at the database"
        for conn in opened:
            with pytest.raises(sqlite3.ProgrammingError):         # closed
                conn.execute("SELECT 1")

    @pytest.mark.parametrize("fixture", [
        "db_v02x_schema.sql",            # hand-built, before the confidence layer
        "db_v030_7757085.sql",           # v0.3.0: migrations 002 to 006 all still to run
        "db_main_03217f4.sql",           # main at user_version 4: 005 and 006 still to run
    ])
    def test_the_refusal_comes_before_any_ddl_migration_or_wal_switch(
            self, fixture, tmp_path):
        """A refused open leaves the file byte for byte as it was: no table or
        column added, user_version unchanged, and a database that is not in WAL
        yet is not switched to it (that rewrites its header)."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript((Path(__file__).parent / "fixtures" / fixture).read_text())
        conn.execute("UPDATE nodes SET node_type = 'recipe' "
                     "WHERE rowid = (SELECT MIN(rowid) FROM nodes)")
        conn.commit()
        conn.close()
        data = db.read_bytes()
        assert data[18:20] == b"\x01\x01"                           # format 1: not WAL
        before = hashlib.sha256(data).hexdigest()
        files = sorted(p.name for p in tmp_path.iterdir())

        for _ in range(2):
            with pytest.raises(UnknownTypeError) as e:
                GraphStore(str(db))
            assert e.value.node_types == {"recipe": 1}
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == files    # no side files

        # and it was only the refusal that kept it so: registered, the same
        # file is migrated and switched to WAL
        register_node_type("recipe")
        GraphStore(str(db)).close()
        assert hashlib.sha256(db.read_bytes()).hexdigest() != before
        assert db.read_bytes()[18:20] == b"\x02\x02"

    def test_a_hostile_type_string_cannot_flood_or_break_the_message(self, tmp_path):
        db = str(tmp_path / "s.db")
        GraphStore(db).close()
        conn = sqlite3.connect(db)
        hostile = "xé\n\x1b[31m" + "y" * 500
        conn.execute(
            "INSERT INTO nodes (node_id, node_type, label, content, created_at, "
            "last_accessed) VALUES ('n1', ?, 'l', 'c', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00')", (hostile,))
        conn.commit()
        conn.close()
        with pytest.raises(UnknownTypeError) as e:
            GraphStore(db)
        text = str(e.value)
        assert text.isascii() and "\n" not in text.split("which no loaded")[0]
        assert len(text) < 1200

    def test_every_distinct_unknown_type_is_found_however_many_and_wherever_they_sort(
            self, tmp_path):
        db = str(tmp_path / "s.db")
        GraphStore(db).close()
        conn = sqlite3.connect(db)
        names = [f"kind_{i:02d}" for i in range(40)] + ["aaa_first", "zzz_last"]
        rows = ([(f"n{i}", names[i % len(names)]) for i in range(420)]
                + [(f"f{i}", "fact") for i in range(50)])
        conn.executemany(
            "INSERT INTO nodes (node_id, node_type, label, content, created_at, "
            "last_accessed) VALUES (?, ?, 'l', 'c', '2026-01-01T00:00:00+00:00', "
            "'2026-01-01T00:00:00+00:00')", rows)
        conn.executemany(
            "INSERT INTO edges (edge_id, edge_type, source_node_id, target_node_id, "
            "created_at) VALUES (?, ?, 'n1', 'n2', '2026-01-01T00:00:00+00:00')",
            [(f"e{i}", ["related_to", "aaa_edge", "zzz_edge"][i % 3]) for i in range(30)])
        conn.commit()
        conn.close()
        with pytest.raises(UnknownTypeError) as e:
            GraphStore(db)
        assert e.value.node_types == {name: 10 for name in names}
        assert e.value.edge_types == {"aaa_edge": 10, "zzz_edge": 10}

    def test_a_new_database_and_a_builtin_only_database_open_as_before(self, tmp_path):
        db = str(tmp_path / "new.db")
        s = GraphStore(db)
        s.add_node(Node(node_type=NodeType.FACT, label="a", content="b"))
        s.close()
        s = GraphStore(db)
        assert s.count_nodes() == 1
        s.close()

    def test_check_known_types_is_a_pure_check_on_counts(self):
        extensions.check_known_types({"fact": 3, "entity": 1}, {"related_to": 9})
        with pytest.raises(UnknownTypeError, match="'zzz' \\(4 nodes\\)"):
            extensions.check_known_types({"fact": 3, "zzz": 4}, {})
        register_node_type("zzz")
        extensions.check_known_types({"fact": 3, "zzz": 4}, {})


class TestUnknownTypeInARowReadLater:
    """A row another process wrote after this store opened: the open-time scan
    cannot see it, so the read says what is wrong instead of 'is not a valid'."""

    def test_a_node_row(self, tmp_path):
        db = str(tmp_path / "s.db")
        s = GraphStore(db)
        good = s.add_node(Node(node_type=NodeType.FACT, label="ok", content="fine"))
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO nodes (node_id, node_type, label, content, created_at, "
            "last_accessed) VALUES ('late-node', 'recipe', 'l', 'c', "
            "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')")
        conn.commit()
        conn.close()
        try:
            with pytest.raises(UnknownTypeError) as e:
                s.get_node("late-node")
            assert "'late-node'" in str(e.value) and "'recipe'" in str(e.value)
            assert "REVIEN_EXTENSIONS" in str(e.value)
            assert "not installed or not loaded" in str(e.value) and "newer Revien" in str(e.value)
            assert e.value.node_types == {"recipe": 1}
            with pytest.raises(ValueError):
                s.list_nodes()
            assert s.get_node(good.node_id).label == "ok"        # the rest still reads
        finally:
            s.close()
        assert ("late-node",) in [(r[0],) for r in raw_rows(db, "nodes")]
        assert raw_types(db)["nodes"] == ["fact", "recipe"]       # not rewritten

    def test_an_edge_row(self, tmp_path):
        db = str(tmp_path / "s.db")
        s = GraphStore(db)
        a = s.add_node(Node(node_type=NodeType.FACT, label="a", content="a"))
        b = s.add_node(Node(node_type=NodeType.FACT, label="b", content="b"))
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO edges (edge_id, edge_type, source_node_id, target_node_id, "
            "created_at) VALUES ('late-edge', 'pairs_with', ?, ?, '2026-01-01T00:00:00+00:00')",
            (a.node_id, b.node_id))
        conn.commit()
        conn.close()
        try:
            with pytest.raises(UnknownTypeError) as e:
                s.get_edges_for_node(a.node_id)
            assert "'late-edge'" in str(e.value) and "'pairs_with'" in str(e.value)
            assert e.value.edge_types == {"pairs_with": 1}
        finally:
            s.close()
        assert raw_types(db)["edges"] == ["pairs_with"]


class TestDescribeUnregistered:
    def payload(self):
        return {"nodes": [{"node_type": "recipe"}, {"node_type": "recipe"},
                          {"node_type": "fact"}],
                "edges": [{"edge_type": "pairs_with"}, {"edge_type": "related_to"}]}

    def test_names_what_is_unregistered_and_how_to_fix_it(self):
        text = extensions.describe_unregistered(self.payload())
        assert "node types 'recipe' (2 nodes)" in text
        assert "edge types 'pairs_with' (1 edge)" in text
        assert "REVIEN_EXTENSIONS" in text
        assert "not installed or not loaded" in text and "newer Revien" in text

    def test_empty_when_everything_is_known(self):
        register_toy()
        assert extensions.describe_unregistered(self.payload()) == ""
        assert extensions.describe_unregistered({"nodes": [], "edges": []}) == ""

    @pytest.mark.parametrize("junk", [
        None, 5, "text", [], {}, {"nodes": 5}, {"nodes": [None, 5, "x", {"node_type": 5}]},
        {"edges": {"edge_type": "pairs_with"}},
    ])
    def test_malformed_payloads_are_not_a_second_error(self, junk):
        assert extensions.describe_unregistered(junk) == ""


# ── the dedup declaration ─────────────────────────────────────────────────


def recipe_node(label, content="stir well", ntype="recipe"):
    return Node(node_type=NodeType(ntype), label=label, content=content)


class TestDedupDeclaration:
    @pytest.fixture
    def parts(self, tmp_path):
        store = GraphStore(str(tmp_path / "d.db"))
        ops = GraphOperations(store)
        yield store, Deduplicator(store, ops)
        store.close()

    def test_by_default_an_extension_type_is_deduplicated_like_a_claim(self, parts):
        register_node_type("recipe")
        store, dedup = parts
        first, new1 = dedup.deduplicate_node(recipe_node("Tomato soup"))
        second, new2 = dedup.deduplicate_node(recipe_node("tomato soup"))
        assert new1 is True and new2 is False and second.node_id == first.node_id
        near, new3 = dedup.deduplicate_node(recipe_node("Tomato soups"))
        assert new3 is False and near.node_id == first.node_id      # fuzzy layer too
        assert store.count_nodes() == 1

    def test_dedup_false_keeps_every_candidate_as_its_own_node(self, parts):
        register_node_type("recipe", dedup=False)
        store, dedup = parts
        a, new_a = dedup.deduplicate_node(recipe_node("Tomato soup"))
        b, new_b = dedup.deduplicate_node(recipe_node("Tomato soup"))
        c, new_c = dedup.deduplicate_node(recipe_node("Tomato soups"))
        assert (new_a, new_b, new_c) == (True, True, True)
        assert len({a.node_id, b.node_id, c.node_id}) == 3
        assert store.count_nodes() == 3

    def test_dedup_is_per_type_and_never_crosses_types(self, parts):
        register_node_type("recipe")
        register_node_type("meal_plan", dedup=False)
        store, dedup = parts
        dedup.deduplicate_node(recipe_node("Tomato soup"))
        _, fact_new = dedup.deduplicate_node(
            Node(node_type=NodeType.FACT, label="Tomato soup", content="x"))
        _, plan_new = dedup.deduplicate_node(recipe_node("Tomato soup", ntype="meal_plan"))
        assert fact_new is True and plan_new is True
        assert store.count_nodes() == 3

    def test_a_builtin_type_is_still_deduplicated(self, parts):
        register_node_type("recipe", dedup=False)
        store, dedup = parts
        _, n1 = dedup.deduplicate_node(Node(node_type=NodeType.FACT, label="Same", content="x"))
        _, n2 = dedup.deduplicate_node(Node(node_type=NodeType.FACT, label="Same", content="x"))
        assert (n1, n2) == (True, False)

    def test_through_the_pipeline_with_an_extractor_that_produces_the_type(self, tmp_path):
        """An extension's own extractor feeds the same ingestion path."""
        register_node_type("recipe")
        register_node_type("meal_plan", dedup=False)

        class Extractor:
            def extract(self, content, source_id=""):
                ctx = Node(node_type=NodeType.CONTEXT, label="ctx", content=content,
                           source_id=source_id)
                nodes = [ctx,
                         recipe_node("Tomato soup"), recipe_node("Tomato soup"),
                         recipe_node("Week one", ntype="meal_plan"),
                         recipe_node("Week one", ntype="meal_plan")]
                return ExtractionResult(nodes=nodes, edges=[], context_node=ctx)

        store = GraphStore(str(tmp_path / "p.db"))
        try:
            out = IngestionPipeline(store, extractor=Extractor()).ingest(
                IngestionInput(source_id="note:1", content="two dishes, two plans"))
            # context + 1 soup (second merged) + 2 plans (both kept)
            assert out.nodes_created == 4 and out.nodes_deduplicated == 1
            kinds = sorted(n.node_type.value for n in store.list_nodes())
            assert kinds == ["context", "meal_plan", "meal_plan", "recipe"]
        finally:
            store.close()


# ── only configured code is ever imported ────────────────────────────────


def test_the_registry_is_the_only_module_that_finds_or_loads_extensions():
    """Revien imports what is configured, through one reviewed place. If
    another module started reading entry points, this fails."""
    import pathlib
    root = pathlib.Path(extensions.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "extensions.py" and path.parent == root:
            continue
        text = path.read_text(encoding="utf-8")
        if "entry_points" in text or "EntryPoint" in text:
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


# ── built-in behaviour does not move ──────────────────────────────────────


class TestBuiltinsUnaffected:
    CONVERSATION = (
        "User: We decided to use PostgreSQL for the billing database.\n"
        "Assistant: Noted. I prefer small migrations. The server runs on port 8443.\n"
        "User: Remember to rotate the keys next week."
    )

    def run_scenario(self, tmp_path, name):
        from revien.retrieval.engine import RetrievalEngine
        store = GraphStore(str(tmp_path / f"{name}.db"))
        try:
            IngestionPipeline(store).ingest(IngestionInput(
                source_id="note:1", content=self.CONVERSATION))
            IngestionPipeline(store).ingest(IngestionInput(
                source_id="note:2", content=self.CONVERSATION))
            graph = json.loads(store.export_graph().model_dump_json())
            ids = {}
            nodes = []
            for n in sorted(graph["nodes"], key=lambda n: (n["node_type"], n["label"], n["content"])):
                ids[n["node_id"]] = len(ids)
                nodes.append((n["node_type"], n["label"], n["content"], n["confidence"],
                              n["source_type"], n["access_count"]))
            edges = sorted((e["edge_type"], ids[e["source_node_id"]], ids[e["target_node_id"]],
                            e["weight"]) for e in graph["edges"])
            resp = RetrievalEngine(store).recall("PostgreSQL billing database", top_n=10)
            ranked = [(r.node_type, r.label, r.content, r.path) for r in resp.results]
            return nodes, edges, ranked
        finally:
            store.close()

    def test_registered_types_that_nothing_uses_change_no_builtin_output(self, tmp_path):
        plain = self.run_scenario(tmp_path, "plain")
        register_toy()
        register_node_type("meal_plan", dedup=False)
        with_ext = self.run_scenario(tmp_path, "withext")
        assert plain == with_ext
        assert plain[0] and plain[1] and plain[2]               # it did something

    def test_openapi_document_is_the_same(self, tmp_path):
        from revien.daemon.server import create_app
        a = create_app(db_path=str(tmp_path / "a.db"))
        before = json.dumps(a.openapi(), sort_keys=True)
        a.state.store.close()
        register_toy()
        b = create_app(db_path=str(tmp_path / "b.db"))
        after = json.dumps(b.openapi(), sort_keys=True)
        b.state.store.close()
        assert before == after
        assert json.loads(before)["components"]["schemas"]["NodeResponse"][
            "properties"]["node_type"]["type"] == "string"
