# Extensions: your own node types and edge types

Revien ships a fixed set of node types (`entity`, `topic`, `decision`, `fact`,
`preference`, `event`, `context`, `action`, `skill`) and edge types
(`related_to`, `decided_in`, `mentioned_by`, `depends_on`, `followed_by`,
`contradicts`, `conflicts_with`, `corrects`, `derived_from`, `alias_of`). A
package that keeps its own kinds of memory can register more of both, in its
own code, without patching Revien. This page covers how to write such a package,
what Revien does with the types, and what is not supported.

## A complete example

A toy package that stores recipes. It adds two node types and one edge type.

`revien_recipes.py`:

```python
# file: revien_recipes.py
from revien.extensions import register_edge_type, register_node_type


def register():
    register_node_type("recipe", description="a dish and how to make it")
    register_node_type("meal_plan", description="what is cooked on which day",
                       dedup=False)
    register_edge_type("pairs_with", description="two dishes that suit each other")
```

Tell Revien about it in one of two ways:

- Declare an entry point in the package's `pyproject.toml`. Revien loads every
  installed package that declares one:

  ```toml
  [project.entry-points."revien.extensions"]
  recipes = "revien_recipes:register"
  ```

- Or set the environment variable to `module:function`, comma-separated for
  several. The module has to be importable (on `sys.path`):

  ```
  export REVIEN_EXTENSIONS=revien_recipes:register
  ```

Then the types work like the built-in ones:

```python
# file: use_recipes.py
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore

store = GraphStore("recipes.db")          # loads the extension

soup = store.add_node(Node(node_type=NodeType("recipe"), label="Tomato soup",
                           content="Simmer ripe tomatoes with basil and cream."))
bread = store.add_node(Node(node_type=NodeType.FACT, label="Sourdough bread",
                            content="Sourdough bread keeps for three days."))
store.add_edge(Edge(edge_type=EdgeType("pairs_with"),
                    source_node_id=soup.node_id, target_node_id=bread.node_id))

assert store.get_node(soup.node_id).node_type.value == "recipe"
store.close()
```

## What you can declare

| Call | Declares |
|---|---|
| `register_node_type(value, *, description="", dedup=True)` | a node type |
| `register_edge_type(value, *, description="")` | an edge type |

Both return the enum member (`NodeType("recipe")`). The rules:

- `value` is lowercase snake_case: `a-z` and `0-9`, starting with `a-z`, words
  joined by single underscores, at most 64 characters (`recipe`, `meal_plan`).
- A value that is a built-in type of the same kind is refused
  (`ExtensionRegistrationError`).
- Registering the same value again with the same arguments does nothing.
  Registering it with different arguments raises `ExtensionRegistrationError`
  and keeps the first.
- `description` is for people and for `registered_node_types()`; Revien does
  not store it. At most 500 characters.
- `dedup=False` (node types only) keeps the type out of extraction dedup; see
  below. It is the only behaviour a type can declare.
- The registry is process-wide and thread-safe. It lives in memory: nothing is
  written to the database, the disk or the network when a type is registered.
- A value that a later Revien release adds as a built-in type makes your
  registration fail with the "built-in" error at the next start. Choose
  distinctive values.

Reading the registry:

| Call | Returns |
|---|---|
| `registered_node_types()`, `registered_edge_types()` | `{value: spec}`, a copy, in registration order, built-ins not included |
| `all_node_type_values()`, `all_edge_type_values()` | every valid value: built-ins, then registered |
| `loaded_extensions()` | `{spec: where it was configured}` for what has loaded |

## How extensions are found and loaded

Revien imports only the modules that two sources name, and reads nothing else:

1. the entry-point group `revien.extensions` of installed packages, each
   entry point naming a function that registers, and
2. the environment variable `REVIEN_EXTENSIONS=package.module:function,other.module:function`
   (`module:Class.method` also works; an item is plain dotted names with one
   colon, no spaces and no extras; blank items are ignored).

An entry point and an environment item that name the same function run it once.
Loading an entry point imports a module by name, so the egress gate
(`scripts/gates/check_egress.py`) reports a `.load()` call like a dynamic import
and allows exactly one: the statement in `_run` in `revien/extensions.py`.
Revien scans installed packages' metadata files for the group and reads the
environment; it makes no network call to find or load an extension. Loading
runs the extension's code with your privileges, so install only packages you
trust: every installed package that declares an entry point in the group is
loaded, and there is no switch to turn that off short of uninstalling it.

**When.** Each configured extension runs once per process, the first time a
`GraphStore` opens: the store calls `load_extensions()`. The CLI commands,
`revien start`, `revien mcp`, the REST API, the Hermes provider and the
adapters all open a store, so all of them load extensions. `revien import`
loads them before it reads the file, because it parses the file before it
opens a store. Library code that builds a `Node` or an `Edge` of an extension
type before any store is open calls `revien.extensions.load_extensions()`
first, or registers directly. A call made from inside an extension while it is
loading returns at once. `REVIEN_EXTENSIONS` is read on every call: an item
added after an earlier load is loaded at the next call (a store opening, or
`load_extensions()`), an item already loaded is not run again, and an item
removed stays loaded, because a registered type is never unregistered.

**When one fails.** An extension that cannot be imported, is not callable,
raises, calls `sys.exit()` (or raises any other `BaseException` but
`KeyboardInterrupt`, which propagates), or has a malformed `REVIEN_EXTENSIONS`
item is an `ExtensionLoadError`.
It names every failing extension, where it was configured, and the cause. It is
not skipped: the other extensions still load, then the error is raised, and it
is raised again on every later attempt until the cause is fixed. The store does
not open. The CLI prints `Error: ...` and exits 1. Types a failing function
registered before it failed stay registered.

## What `NodeType("recipe")` is

After registration, `NodeType("recipe")` and `EdgeType("pairs_with")` return
an object that is an instance of the enum class, equal to the string, with
`.value == "recipe"`, the same object every time and pickled by value. It works
wherever a value is parsed: store reads, `Node`, `Edge` and `Graph` (so
`import_graph`, `POST /v1/graph/import` and `revien import`),
`GET /v1/nodes?node_type=`, `POST /v1/edges`. Recall results, the REST
responses, the MCP tools, TOON and exports carry the plain string.

How: `NodeType` and `EdgeType` define Python's `_missing_` hook, which Python
calls for a value that is not a declared member and which here returns the
registered type's member. The standard library's `enum.Flag` uses the same
hook for combined values. It was chosen over rebuilding the enum on
registration (that breaks every reference imported earlier and the schema
pydantic compiled), over widening the model fields to `str` (a misspelt type
would be stored, and every `.value` call would break), and over accepting any
unknown string (the REST API would stop rejecting bad input).

What it means in practice:

- Registered types are not members: `list(NodeType)`, `len(NodeType)` and
  `NodeType.__members__` still list only the built-ins, and so do the JSON
  schema pydantic generates and the OpenAPI document. Use
  `all_node_type_values()`. `NodeType["RECIPE"]` fails; use `NodeType("recipe")`.
- `x in NodeType` is not a reliable test for an extension value, and it differs
  across Python versions. For the registered member it is `False` on 3.10 and
  3.11 and `True` on 3.12 and later. For the plain string `"recipe"` it raises
  `TypeError` on 3.10 and 3.11 and is `False` on 3.12 and later (for a built-in
  string such as `"fact"`: `TypeError`, then `True`). Test membership with
  `x in all_node_type_values()` instead.
- The mechanism was exercised on Python 3.10, 3.11, 3.12 and 3.13, and on
  pydantic 2.0.3 through 2.14.1.

## What Revien does with an extension type

An extension type is a value in the existing `node_type` or `edge_type` column.
No schema change and no migration is involved, and the forget inventory's list
of columns (`revien/forget/inventory.py`) is unchanged.

| Where | What happens |
|---|---|
| Store, `list_nodes(node_type=...)`, keyword search, neighbours | as for any type |
| Recall | takes part like a fact: found by keyword search and the graph walk, and across extension edges, with the same score. `REVIEN_TYPE_WEIGHTS`, `REVIEN_TYPE_HALF_LIFE_DAYS` and `prefer_types` accept the value like any other. The result's `node_type` is the value. |
| Ingestion dedup | for nodes an extractor produces: merged into an existing node of the same type with the same or a near-identical label (and, if `REVIEN_SEMANTIC_DEDUP` is on, a close paraphrase), like a fact. `dedup=False` stores every candidate as its own node. |
| Forget (`revien forget NODE_ID`, `--wording`) | like a claim of a built-in type: the node is removed and its wording cut from the source it came from; `--wording` cuts the string out of its label and content. |
| Delete (`revien delete`, `DELETE /v1/nodes/{id}`) | removes the node and its edges and leaves one content-free audit entry, as for any node |
| Export, import | the value is written to the file. Importing a file that has the type requires the extension to be loaded (see the next section for what happens otherwise). |
| Consolidation (`revien dream`) | orphan reports carry the value. The alias pass pairs entities only, so extension nodes are not paired. |
| Vault notes (`revien distill-vault`) | an entity's note has sections for the built-in claim types only. An extension node linked to the entity is not rendered there, and is not removed from the graph. |

Revien's built-in extractors produce only built-in types. A node of your type
gets into the graph through `store.add_node(...)`, a graph import, or your own
extractor passed to `IngestionPipeline(store, extractor=...)`. The REST API has
no endpoint that creates a node of a chosen type, and `revien ingest`,
`POST /v1/ingest` and the MCP `revien_store` tool use the configured built-in
extractor (`REVIEN_EXTRACTOR`).

## A database that holds a type nobody registered

Opening a database whose node or edge types are not all built in or registered
fails with `UnknownTypeError` (a `ValueError`). The check reads only the two
type columns, on a connection of its own, before the store opens its connection,
switches the database to WAL, creates a table or runs a migration, so the file is
left exactly as it was:

```
Error: Cannot open the database /path/memory.db: it holds node types 'recipe'
(12 nodes); edge types 'pairs_with' (4 edges), which no loaded extension
registers. No row was changed or dropped. Either an extension that registers
such types is not installed or not loaded, or the data came from a newer Revien
than this one. For an extension, install the package that defines it (it
declares the 'revien.extensions' entry point group) or set
REVIEN_EXTENSIONS=package.module:function; for a newer Revien, upgrade it. Then
retry. See docs/EXTENSIONS.md.
```

The message names both causes and does not guess which one applies.

This is deliberate. Reading such a row as some other type would change what it
means. Passing it through as an "unknown" type would let recall, dedup,
supersession and forget act on a row nobody can classify. Skipping it would
make memory vanish without a word. An error that names the type, how many rows
hold it and how to register it cannot lose or corrupt memory. No row is read
as another type, rewritten, or dropped. Every command that opens the
store is refused alike, including `revien forget` and `revien delete`; to
inspect such a database without the extension, use `sqlite3` directly.

The check is made once, when the store opens, with one index probe per distinct
type. A row of an unregistered type that another process writes later is not
seen by that check; a read that returns it raises `UnknownTypeError` naming
the row's id and type, and a read that does not reach it is unaffected.

An import is the same: a file with an unregistered type fails to validate, and
`revien import` and `POST /v1/graph/import` add a line naming the unregistered
types and how to register them.

## Not supported

- **Changing how recall works.** There is no plug-in point for the retrieval
  engine, the scorer, ranking or the semantic index, and an extension cannot
  subclass them. A type takes the generic path above; the only declaration is
  `dedup`.
- **Plugging an extractor into `revien start`, `revien ingest` or MCP.** Those
  build the configured built-in extractor. Yours works through the Python API.
- **Unregistering or renaming a type**, and migrating existing rows from one
  type to another. Nothing in Revien rewrites a stored type.
- **Per-type behaviour in forget, vault notes, supersession or skills** beyond
  what the table above says.
- **A read-only mode for a database with unregistered types.**

## Writing a good extension

- Register from code, with constant values. `forget --wording` does not touch
  the `node_type` and `edge_type` columns (the forget inventory excludes them
  because their values come from code), so never build a type value from
  content.
- Keep the function to registering. It runs once per process, before the first
  store opens.
- Test it in a fresh process, or call `revien.extensions._reset_for_tests()`
  between tests. That function is private and is for tests only: a real
  process never unregisters a type, because rows may already use it.

## Errors

| Class | Base | Raised when |
|---|---|---|
| `ExtensionError` | `Exception` | base class of the rest |
| `ExtensionRegistrationError` | `ExtensionError`, `ValueError` | a value is malformed, built in, or registered twice with different arguments |
| `ExtensionLoadError` | `ExtensionError`, `RuntimeError` | a configured extension failed; `.failures` lists `(spec, where, exception)` |
| `UnknownTypeError` | `ExtensionError`, `ValueError` | a stored type is not built in or registered; `.node_types` and `.edge_types` map each to its row count |
