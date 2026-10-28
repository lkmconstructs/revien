"""Extension registry: how an outside package adds node types and edge types.

Revien ships a fixed set of node types and edge types (``NodeType`` and
``EdgeType`` in ``revien/graph/schema.py``). A package that keeps its own kinds
of memory registers them here, in its own code, without patching Revien:

    from revien.extensions import register_edge_type, register_node_type

    def register():                      # the callable Revien is pointed at
        register_node_type("recipe", description="a dish and how to make it")
        register_edge_type("pairs_with", description="two dishes that suit each other")

After that, ``NodeType("recipe")`` and ``EdgeType("pairs_with")`` work everywhere
a type value is parsed: store reads, ``import_graph``, the REST API, the CLI's
``import``. ``node.node_type.value`` is ``"recipe"``. See docs/EXTENSIONS.md.

How a value becomes an enum member (the mechanism)
--------------------------------------------------
The two enums define ``_missing_``. Python calls it when a value is not a
declared member; it returns the registered type's *pseudo-member*: a real
instance of the enum class (``isinstance(m, NodeType)``), a ``str`` equal to
its value, hashed like it, one object per value (``NodeType("recipe") is
NodeType("recipe")``), pickled by value. This is the hook the standard
library's own ``enum.Flag`` uses for composite values. Pseudo-members are NOT
added to ``__members__``, so iterating ``NodeType`` still yields exactly the
built-in types, and everything that depends on that (the JSON schema, the
OpenAPI document, error messages) is unchanged. Use
``all_node_type_values()`` / ``all_edge_type_values()`` for built-ins plus
registered.

Why not the alternatives: rebuilding the enum on registration breaks every
reference imported before it, every ``is`` comparison and the schema pydantic
compiled at class creation; widening the model fields to ``str`` loses
validation (a typo would be stored) and every ``node_type.value`` call; a
pass-through member for any unknown string would accept bad input at the REST
boundary.

How extensions are found
------------------------
Only what is configured, and only by name:

* the entry-point group ``revien.extensions`` of installed packages, each
  pointing at a callable that does the registering, and
* the environment variable ``REVIEN_EXTENSIONS=pkg.module:func,other.mod:fn``.

Nothing else is imported: no scan of ``sys.path``, no guessing from names. They
are loaded once per process by ``load_extensions()``, which ``GraphStore`` calls
when it opens (so every surface gets them) and ``revien import`` calls before it
reads its file. An extension that cannot be imported, is not callable, raises,
or calls ``sys.exit()`` (any ``BaseException`` but ``KeyboardInterrupt``) is an
``ExtensionLoadError`` naming it and the cause. It is never skipped, and it is
raised again on the next call until it works. Registrations are kept in memory
only: nothing here writes to the database, the network or the disk.

A database that holds a type nobody registered is refused at open with
``UnknownTypeError`` (see ``check_known_types``). The check reads only the two
type columns, before the store opens its own connection, switches it to WAL,
creates a table or runs a migration; nothing is read as another type, dropped
or rewritten.
"""
from __future__ import annotations

import importlib.metadata as _metadata
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: The entry-point group installed packages declare to be loaded.
ENTRY_POINT_GROUP = "revien.extensions"
#: Comma-separated ``module:callable`` list, loaded in addition to the group.
ENV_VAR = "REVIEN_EXTENSIONS"
#: Longest accepted type value.
MAX_VALUE_LENGTH = 64
#: Longest accepted description.
MAX_DESCRIPTION_LENGTH = 500

# lowercase snake_case: a-z and 0-9, single underscores between words, starts
# with a-z. fullmatch only: "$" would let a trailing newline in.
_VALUE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
# module.path:callable.path, plain identifiers only (no extras, no spaces).
_SPEC_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)

_HOW_TO_REGISTER = (
    "Either an extension that registers such types is not installed or not "
    "loaded, or the data came from a newer Revien than this one. For an "
    "extension, install the package that defines it (it declares the "
    f"'{ENTRY_POINT_GROUP}' entry point group) or set "
    f"{ENV_VAR}=package.module:function; for a newer Revien, upgrade it. Then "
    "retry. See docs/EXTENSIONS.md."
)


# ── errors ────────────────────────────────────────────────────────────────


class ExtensionError(Exception):
    """Base class of every error raised by the extension registry."""


class ExtensionRegistrationError(ExtensionError, ValueError):
    """A type could not be registered: bad value, a built-in, or a
    conflicting second registration. Nothing was registered."""


class ExtensionLoadError(ExtensionError, RuntimeError):
    """A configured extension failed to load. ``failures`` is a list of
    ``(spec, where_configured, exception)``; the first one is the cause."""

    def __init__(self, failures: List[Tuple[str, str, BaseException]]):
        self.failures = list(failures)
        lines = [
            f"  - {spec} ({where}): "
            + (f"called sys.exit({exc.code!r})" if isinstance(exc, SystemExit)
               else f"{type(exc).__name__}: {exc}")
            for spec, where, exc in self.failures
        ]
        noun = "extension" if len(lines) == 1 else "extensions"
        super().__init__(
            f"{len(lines)} configured {noun} failed to load; nothing was "
            "skipped:\n" + "\n".join(lines) + "\nFix or remove it (an installed "
            f"package's '{ENTRY_POINT_GROUP}' entry point, or the {ENV_VAR} "
            "variable) and retry."
        )

    def __reduce__(self):
        # The default would rebuild it from the message alone.
        return (self.__class__, (self.failures,))


class UnknownTypeError(ExtensionError, ValueError):
    """A node type or edge type is not built in and not registered.

    A ``ValueError``, like the ``'x' is not a valid NodeType`` it replaces, so
    existing handlers keep working. ``node_types`` / ``edge_types`` map each
    unknown value to how many rows hold it (empty for a single-row error).
    """

    def __init__(self, message: str,
                 node_types: Optional[Mapping[str, int]] = None,
                 edge_types: Optional[Mapping[str, int]] = None):
        super().__init__(message)
        self.node_types: Dict[str, int] = dict(node_types or {})
        self.edge_types: Dict[str, int] = dict(edge_types or {})

    def __reduce__(self):
        return (self.__class__, (str(self), self.node_types, self.edge_types))


# ── registry ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NodeTypeSpec:
    """What an extension declared about a node type.

    ``dedup=False`` keeps the type out of extraction dedup: every candidate of
    this type becomes its own node, as a context node does, instead of being
    merged into a same-label (or near-same-label) node of the same type.
    """
    value: str
    description: str = ""
    dedup: bool = True


@dataclass(frozen=True)
class EdgeTypeSpec:
    """What an extension declared about an edge type."""
    value: str
    description: str = ""


# Registrations are serialised by _LOCK (short sections: never held while an
# extension's code runs). Reads of the dicts below take no lock: a dict get is
# atomic, and a member is inserted last, after its spec.
_LOCK = threading.RLock()
_NODE_SPECS: Dict[str, NodeTypeSpec] = {}
_EDGE_SPECS: Dict[str, EdgeTypeSpec] = {}
_NODE_MEMBERS: Dict[str, Any] = {}
_EDGE_MEMBERS: Dict[str, Any] = {}


def _check_value(kind: str, value: Any) -> str:
    if type(value) is not str:
        raise ExtensionRegistrationError(
            f"{kind} value must be a plain str, got {type(value).__name__}")
    if len(value) > MAX_VALUE_LENGTH or not _VALUE_RE.fullmatch(value):
        raise ExtensionRegistrationError(
            f"{kind} value {ascii(value)} is not lowercase snake_case: use "
            f"a-z and 0-9, starting with a-z, words joined by single "
            f"underscores, at most {MAX_VALUE_LENGTH} characters "
            "(for example 'recipe' or 'meal_plan')")
    return value


def _check_description(kind: str, description: Any) -> str:
    if type(description) is not str:
        raise ExtensionRegistrationError(
            f"{kind} description must be a str, got {type(description).__name__}")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ExtensionRegistrationError(
            f"{kind} description is longer than {MAX_DESCRIPTION_LENGTH} characters")
    return description


def _make_member(enum_cls: Any, value: str) -> Any:
    """The pseudo-member of ``enum_cls`` for ``value``: a genuine instance of
    the enum class, built the way ``enum`` builds a ``str`` mixin member."""
    member = str.__new__(enum_cls, value)
    member._name_ = value.upper()
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)
    return member


def _builtin_values(enum_cls: Any) -> List[str]:
    # Iterating an Enum yields its declared members only, never pseudo-members.
    return [m.value for m in enum_cls]


def _register(kind: str, enum_cls: Any, spec: Any, specs: Dict[str, Any],
              members: Dict[str, Any]) -> Any:
    with _LOCK:
        if spec.value in _builtin_values(enum_cls):
            raise ExtensionRegistrationError(
                f"{kind} {spec.value!r} is a built-in Revien type and cannot "
                "be registered")
        existing = specs.get(spec.value)
        if existing is not None:
            if existing == spec:
                return members[spec.value]            # identical: a no-op
            raise ExtensionRegistrationError(
                f"{kind} {spec.value!r} is already registered with different "
                f"metadata: {existing!r} (this call: {spec!r})")
        member = _make_member(enum_cls, spec.value)
        specs[spec.value] = spec
        members[spec.value] = member                  # last: readers see it whole
        return member


def register_node_type(value: str, *, description: str = "",
                       dedup: bool = True) -> Any:
    """Register a node type and return its ``NodeType`` member.

    ``value`` must be lowercase snake_case and not a built-in value. The same
    value registered again with the same ``description`` and ``dedup`` is a
    no-op; with different ones it raises ``ExtensionRegistrationError``.
    ``dedup=False`` keeps the type out of extraction dedup (see
    ``NodeTypeSpec``). Process-global and thread-safe.
    """
    from revien.graph.schema import NodeType

    kind = "node type"
    if type(dedup) is not bool:
        raise ExtensionRegistrationError("node type dedup must be True or False")
    spec = NodeTypeSpec(_check_value(kind, value),
                        _check_description(kind, description), dedup)
    return _register(kind, NodeType, spec, _NODE_SPECS, _NODE_MEMBERS)


def register_edge_type(value: str, *, description: str = "") -> Any:
    """Register an edge type and return its ``EdgeType`` member. Same rules as
    ``register_node_type``."""
    from revien.graph.schema import EdgeType

    kind = "edge type"
    spec = EdgeTypeSpec(_check_value(kind, value),
                        _check_description(kind, description))
    return _register(kind, EdgeType, spec, _EDGE_SPECS, _EDGE_MEMBERS)


def registered_node_types() -> Dict[str, NodeTypeSpec]:
    """A copy of the registered node types, ``{value: NodeTypeSpec}``, in the
    order they were registered. Built-in types are not listed."""
    with _LOCK:
        return dict(_NODE_SPECS)


def registered_edge_types() -> Dict[str, EdgeTypeSpec]:
    """A copy of the registered edge types. Built-in types are not listed."""
    with _LOCK:
        return dict(_EDGE_SPECS)


def all_node_type_values() -> List[str]:
    """Every valid node type value: built-ins in declaration order, then the
    registered ones in registration order."""
    from revien.graph.schema import NodeType

    with _LOCK:
        return _builtin_values(NodeType) + list(_NODE_SPECS)


def all_edge_type_values() -> List[str]:
    """Every valid edge type value: built-ins, then registered."""
    from revien.graph.schema import EdgeType

    with _LOCK:
        return _builtin_values(EdgeType) + list(_EDGE_SPECS)


def node_type_member(value: Any) -> Optional[Any]:
    """The pseudo-member for a registered node type value, else None. This is
    what ``NodeType._missing_`` calls."""
    try:
        return _NODE_MEMBERS.get(value)
    except TypeError:                                  # unhashable input
        return None


def edge_type_member(value: Any) -> Optional[Any]:
    """The pseudo-member for a registered edge type value, else None. This is
    what ``EdgeType._missing_`` calls."""
    try:
        return _EDGE_MEMBERS.get(value)
    except TypeError:
        return None


def node_type_dedups(node_type: Any) -> bool:
    """Does extraction dedup apply to this node type? True for every built-in
    type and for a registered type unless it declared ``dedup=False``."""
    spec = _NODE_SPECS.get(getattr(node_type, "value", node_type))
    return True if spec is None else spec.dedup


# ── behaviour sets ────────────────────────────────────────────────────────
#
# Three sets of type *values* (plain strings; compare with ``.value``, never
# the enum member: an Enum hashes by name). All empty by default, so stock
# behaviour is unchanged. They only grow, in place, so a module that imported
# one keeps seeing it. Fill them from an extension (``register_leaf_node_type``
# and friends) or the REVIEN_*_TYPES variables.

#: Node types a walk may reach but never expands through.
LEAF_NODE_TYPES: set = set()
#: Edge types followed source -> target only.
ASYMMETRIC_EDGE_TYPES: set = set()
#: Node types whose ``metadata`` rides in recall results.
RECALL_METADATA_NODE_TYPES: set = set()

ENV_LEAF_NODE_TYPES = "REVIEN_LEAF_NODE_TYPES"
ENV_ASYMMETRIC_EDGE_TYPES = "REVIEN_ASYMMETRIC_EDGE_TYPES"
ENV_RECALL_METADATA_NODE_TYPES = "REVIEN_RECALL_METADATA_NODE_TYPES"


def _add_to_set(kind: str, target: set, known: List[str], value: Any) -> str:
    _check_value(kind, value)
    if value not in known:
        raise ExtensionRegistrationError(
            f"{kind} {value!r} is not a built-in or registered type; register "
            "it first")
    with _LOCK:
        target.add(value)
    return value


def register_leaf_node_type(value: str) -> str:
    """Mark a node type as a leaf: recall may return it, but the graph walk
    never expands through it (its neighbours are not visited from it). The
    type must be built in or already registered. Idempotent."""
    return _add_to_set("leaf node type", LEAF_NODE_TYPES,
                       all_node_type_values(), value)


def register_asymmetric_edge_type(value: str) -> str:
    """Mark an edge type as one-way: neighbour lookups follow it from source
    to target only, never back. The type must be built in or already
    registered. Idempotent."""
    return _add_to_set("asymmetric edge type", ASYMMETRIC_EDGE_TYPES,
                       all_edge_type_values(), value)


def register_recall_metadata_node_type(value: str) -> str:
    """Include a node type's ``metadata`` in recall results (REST ``/v1/recall``).
    Other types keep the stock shape. The type must be built in or already
    registered. Idempotent."""
    return _add_to_set("recall metadata node type", RECALL_METADATA_NODE_TYPES,
                       all_node_type_values(), value)


def _apply_env_sets() -> None:
    """Read the REVIEN_*_TYPES variables (comma lists) into the sets. Called by
    ``load_extensions`` after the extensions ran, so a listed type may be one
    an extension registers. A name that is not a known type is an error."""
    for env, register in ((ENV_LEAF_NODE_TYPES, register_leaf_node_type),
                          (ENV_ASYMMETRIC_EDGE_TYPES, register_asymmetric_edge_type),
                          (ENV_RECALL_METADATA_NODE_TYPES,
                           register_recall_metadata_node_type)):
        for item in os.environ.get(env, "").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                register(item)
            except ExtensionRegistrationError as exc:
                raise ExtensionRegistrationError(f"{env}: {exc}") from exc


# ── unknown types ─────────────────────────────────────────────────────────


def _quoted(values: Mapping[str, int], unit: str) -> str:
    parts = []
    for value, count in sorted(values.items(), key=lambda kv: str(kv[0])):
        shown = ascii(value if isinstance(value, str) else str(value))
        if len(shown) > 60:
            shown = shown[:57] + "..."
        parts.append(f"{shown} ({count} {unit if count == 1 else unit + 's'})")
    return ", ".join(parts)


def _unknown_summary(nodes: Mapping[str, int], edges: Mapping[str, int]) -> str:
    out = []
    if nodes:
        out.append("node types " + _quoted(nodes, "node"))
    if edges:
        out.append("edge types " + _quoted(edges, "edge"))
    return "; ".join(out)


def check_known_types(node_counts: Mapping[str, int],
                      edge_counts: Mapping[str, int], *,
                      where: str = "this database") -> None:
    """Raise ``UnknownTypeError`` if any stored type is neither built in nor
    registered. ``node_counts`` / ``edge_counts`` map each distinct stored type
    value to its row count. This is the store-open check: a graph is never
    opened half-understood, and no row is converted or dropped."""
    known_nodes = set(all_node_type_values())
    known_edges = set(all_edge_type_values())
    bad_nodes = {v: c for v, c in node_counts.items() if v not in known_nodes}
    bad_edges = {v: c for v, c in edge_counts.items() if v not in known_edges}
    if not bad_nodes and not bad_edges:
        return
    raise UnknownTypeError(
        f"Cannot open {where}: it holds "
        f"{_unknown_summary(bad_nodes, bad_edges)}, which no loaded extension "
        "registers. No row was changed or dropped. " + _HOW_TO_REGISTER,
        bad_nodes, bad_edges)


def unknown_row_error(kind: str, value: Any, row_id: str) -> UnknownTypeError:
    """The error for ONE stored row whose type is not known (a row another
    process wrote after this store opened). ``kind`` is 'node' or 'edge'."""
    shown = ascii(value if isinstance(value, str) else str(value))[:60]
    nodes = {str(value): 1} if kind == "node" else {}
    edges = {str(value): 1} if kind == "edge" else {}
    return UnknownTypeError(
        f"The {kind} {ascii(str(row_id))[:60]} has the {kind} type {shown}, "
        "which no loaded extension registers. No row was changed or dropped. "
        + _HOW_TO_REGISTER, nodes, edges)


def describe_unregistered(graph_data: Any) -> str:
    """For a graph payload (the dict form of ``Graph``, as exported to JSON):
    a sentence naming the node and edge types in it that are neither built in
    nor registered, with how to register them; "" when there are none. Used to
    make an import's validation error say what is actually wrong."""
    if not isinstance(graph_data, dict):
        return ""
    known = {"node_type": set(all_node_type_values()),
             "edge_type": set(all_edge_type_values())}
    found: Dict[str, Dict[str, int]] = {"node_type": {}, "edge_type": {}}
    for field, key in (("node_type", "nodes"), ("edge_type", "edges")):
        rows = graph_data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            value = row.get(field) if isinstance(row, dict) else None
            if isinstance(value, str) and value not in known[field]:
                found[field][value] = found[field].get(value, 0) + 1
    if not found["node_type"] and not found["edge_type"]:
        return ""
    return (f"The graph uses {_unknown_summary(found['node_type'], found['edge_type'])}"
            ", which no loaded extension registers. " + _HOW_TO_REGISTER)


# ── discovery and loading ─────────────────────────────────────────────────

_LOAD_LOCK = threading.RLock()
_loading = threading.local()
_entry_points_cache: Optional[List[Any]] = None
# spec -> where it was configured, for every extension that loaded.
_LOADED: Dict[str, str] = {}
_last_env: Optional[str] = None


def _entry_points() -> List[Any]:
    """Installed packages' entry points in the group, scanned once per
    process, in a stable order, one per distinct target."""
    global _entry_points_cache
    if _entry_points_cache is None:
        seen = set()
        found = []
        for ep in _metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.value not in seen:
                seen.add(ep.value)
                found.append(ep)
        _entry_points_cache = sorted(found, key=lambda e: (e.name, e.value))
    return _entry_points_cache


def _configured() -> Tuple[List[Tuple[str, str, Any]],
                           List[Tuple[str, str, BaseException]]]:
    """(what is configured now, what is configured wrongly). What is
    configured is ``[(spec, where, EntryPoint)]``: the installed entry points,
    then the REVIEN_EXTENSIONS list. A badly written REVIEN_EXTENSIONS item
    is reported as a failure, never ignored."""
    configured: List[Tuple[str, str, Any]] = []
    wrong: List[Tuple[str, str, BaseException]] = []
    seen = set()
    try:
        entry_points = _entry_points()
    except Exception as exc:                   # noqa: BLE001 - broken metadata
        wrong.append(("(scanning installed packages)",
                      f"entry point group {ENTRY_POINT_GROUP!r}", exc))
        entry_points = []
    for ep in entry_points:
        dist = getattr(getattr(ep, "dist", None), "name", None)
        where = f"entry point {ep.name!r}" + (f" of package {dist!r}" if dist else "")
        if ep.value not in seen:
            seen.add(ep.value)
            configured.append((ep.value, where, ep))
    for item in os.environ.get(ENV_VAR, "").split(","):
        item = item.strip()
        if not item:
            continue
        if not _SPEC_RE.fullmatch(item):
            wrong.append((item, ENV_VAR, ExtensionError(
                "expected 'package.module:callable'")))
        elif item not in seen:
            seen.add(item)
            configured.append((item, ENV_VAR, _metadata.EntryPoint(
                name=ENV_VAR, value=item, group=ENTRY_POINT_GROUP)))
    return configured, wrong


def _run(spec: str, entry_point: Any) -> None:
    func = entry_point.load()                  # imports ONLY the configured module
    if not callable(func):
        raise TypeError(f"{spec!r} is not callable; point it at the function "
                        "that registers the types")
    func()


def load_extensions() -> Dict[str, str]:
    """Load every configured extension that is not loaded yet; return
    ``{spec: where configured}`` for all that are.

    Idempotent and thread-safe: each extension runs once per process, however
    many stores open. REVIEN_EXTENSIONS is read on every call: an item added
    after an earlier load is loaded now, one that was already loaded is not run
    again, and one removed stays loaded (the registry only grows). If any fail,
    the rest still run, then one ``ExtensionLoadError`` names every failure; a
    failed extension is not marked loaded, so the next call raises again
    instead of carrying on without it. A ``SystemExit`` (``sys.exit()``) or any
    other ``BaseException`` an extension raises counts as a failure, so it
    cannot end the process quietly; ``KeyboardInterrupt`` propagates.
    Registrations an extension made before it failed stay. A call made from
    inside an extension while it loads returns at once.
    """
    global _last_env
    env = os.environ.get(ENV_VAR, "")
    if _entry_points_cache is not None and env == _last_env:
        _apply_env_sets()
        return dict(_LOADED)                   # nothing new to look at
    if getattr(_loading, "active", False):
        return dict(_LOADED)
    with _LOAD_LOCK:
        _loading.active = True
        try:
            configured, failures = _configured()
            for spec, where, entry_point in configured:
                if spec in _LOADED:
                    continue
                try:
                    _run(spec, entry_point)
                except KeyboardInterrupt:
                    raise
                except BaseException as exc:   # noqa: BLE001 - incl. sys.exit(); reported, never skipped
                    failures.append((spec, where, exc))
                else:
                    _LOADED[spec] = where
            if failures:
                raise ExtensionLoadError(failures) from failures[0][2]
            _apply_env_sets()
            _last_env = env
            return dict(_LOADED)
        finally:
            _loading.active = False


def loaded_extensions() -> Dict[str, str]:
    """``{spec: where configured}`` for the extensions loaded so far."""
    with _LOAD_LOCK:
        return dict(_LOADED)


def _reset_for_tests() -> None:
    """Forget every registration, loaded extension and cached scan. Tests only:
    a real process never unregisters a type (rows may already use it)."""
    global _entry_points_cache, _last_env
    with _LOAD_LOCK, _LOCK:
        _NODE_SPECS.clear()
        _EDGE_SPECS.clear()
        _NODE_MEMBERS.clear()
        _EDGE_MEMBERS.clear()
        LEAF_NODE_TYPES.clear()
        ASYMMETRIC_EDGE_TYPES.clear()
        RECALL_METADATA_NODE_TYPES.clear()
        _LOADED.clear()
        _entry_points_cache = None
        _last_env = None


__all__ = [
    "ASYMMETRIC_EDGE_TYPES", "ENTRY_POINT_GROUP", "ENV_ASYMMETRIC_EDGE_TYPES",
    "ENV_LEAF_NODE_TYPES", "ENV_RECALL_METADATA_NODE_TYPES", "ENV_VAR",
    "EdgeTypeSpec", "LEAF_NODE_TYPES", "RECALL_METADATA_NODE_TYPES", "ExtensionError",
    "ExtensionLoadError", "ExtensionRegistrationError", "NodeTypeSpec",
    "UnknownTypeError", "all_edge_type_values", "all_node_type_values",
    "check_known_types", "describe_unregistered", "load_extensions",
    "loaded_extensions", "register_asymmetric_edge_type", "register_edge_type",
    "register_leaf_node_type", "register_node_type",
    "register_recall_metadata_node_type", "registered_edge_types", "registered_node_types",
]
