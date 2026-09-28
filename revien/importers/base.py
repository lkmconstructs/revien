"""
Revien Importers — shared plumbing: export unwrapping, the ImportUnit
envelope, and the loop that hands units to the ingestion pipeline.

Every per-source importer (chatgpt.py, claude.py, readwise.py) parses its
export into a stream of ImportUnit objects and lets run_import() do the
actual ingesting — one IngestionInput per unit, one pipeline.ingest() call
each. That keeps the deny-list check, context fence, and idempotent
ingest_key refresh (revien/ingestion/pipeline.py) in the single place that
already owns them; an importer never writes to the store directly.
"""

import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from revien.ingestion.pipeline import IngestionInput, IngestionPipeline, _ingest_deny_set

# Case-insensitive match for the member we're looking for inside a zip,
# at any depth (ChatGPT/Claude exports sometimes nest it one folder in).
_CONVERSATIONS_JSON = "conversations.json"


def open_export(path: str) -> bytes:
    """Read an export file's raw bytes, unwrapping a zip if given one.

    Accepts either a .zip archive (the file OpenAI/Claude.ai actually hand
    you when you export) or a direct .json/.csv path (already unzipped, or
    a Readwise CSV, which is never zipped). A zip is searched for every
    member literally named "conversations.json" (case-insensitive), at any
    depth — never extracted to disk, read straight out of the archive. If
    more than one matches (F10: e.g. a re-zipped export nesting a prior
    export inside it), the SHALLOWEST path wins and the others are named in
    a one-line stderr warning rather than silently picked/ignored.

    Raises FileNotFoundError if path doesn't exist, and ValueError if a
    zip is given but contains no conversations.json member.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Export not found: {path}")

    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            matches = [
                name for name in zf.namelist()
                if Path(name).name.lower() == _CONVERSATIONS_JSON
            ]
            if not matches:
                raise ValueError(
                    f"{path}: zip has no conversations.json member (looked at any depth)."
                )
            matches.sort(key=lambda n: len(Path(n).parts))
            chosen, others = matches[0], matches[1:]
            if others:
                sys.stderr.write(
                    f"WARNING: {path}: multiple conversations.json members found; "
                    f"using the shallowest {chosen!r}, ignoring {others!r}.\n"
                )
            return zf.read(chosen)

    return p.read_bytes()


def slugify(text: str) -> str:
    """S2: delegates to the obsidian adapter's public `slug` (one
    implementation, not two copies) — 'untitled' fallback for
    empty/all-punctuation input (obsidian's own default is 'section'; nothing
    here is ever a section). Name kept for the importers that already import
    it (readwise.py)."""
    from revien.adapters.obsidian import slug as _obsidian_slug
    return _obsidian_slug(text, empty="untitled")


def parse_iso_timestamp(value) -> Optional[datetime]:
    """ISO 8601, optionally with a trailing 'Z' (swapped for '+00:00' since
    datetime.fromisoformat only accepts a bare 'Z' from Python 3.11, and this
    project targets >=3.10). A naive result is assumed UTC — every export
    this package reads stamps UTC on the wire. None on missing/unparseable
    input, never guessed. Shared by claude.py and readwise.py (S4) —
    readwise.py layers its own older-format strptime fallback on top."""
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def render_turns(messages: List[Tuple[str, str]]) -> str:
    """Render (role, text) pairs — role is "user" or "assistant" — as the
    same transcript shape the rule extractor already knows how to read:
    hermes_provider.py's "User: ...\\nAssistant: ..." per turn
    (hermes_provider.py:624). A new turn starts at every "user" message
    that follows prior content; consecutive same-role messages (a user
    edit followed by another user message with no assistant reply
    between them, or multi-part assistant output) stay in the current
    turn. Turns are joined by a blank line."""
    blocks: List[List[str]] = []
    current: List[str] = []
    for role, text in messages:
        line = f"{'User' if role == 'user' else 'Assistant'}: {text}"
        if role == "user" and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks)


@dataclass
class ImportUnit:
    """One importable thing — a whole ChatGPT/Claude conversation, or one
    Readwise highlight — parsed and ready to become an IngestionInput.
    Mirrors IngestionInput's own field names so run_import() can pass
    almost everything straight through.

    `error` (F3): set by an importer's iter_units when ONE item (e.g. one
    malformed conversation in an otherwise-good export) fails to parse. A
    generator that raises is dead after the exception — it can't just skip
    ahead to the next conversation — so a per-item parse failure is instead
    turned into an error-marker unit (content/timestamp left at their
    defaults) that run_import logs and skips, rather than killing the whole
    batch. A normal unit always has error=None."""
    source_id: str
    ingest_key: str
    content: str
    content_type: str
    timestamp: Optional[datetime]
    origin_runtime: str
    origin_source: str = "import"
    session_key: Optional[str] = None
    links: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ImportReport:
    """Summary counts for one run_import() call. `errors` keeps only the
    first 20 (source_id, message) pairs — enough to diagnose a bad export
    without an unbounded list for a bulk import gone wrong."""
    units_seen: int = 0
    units_ingested: int = 0
    units_unchanged: int = 0
    units_denied: int = 0
    units_skipped_empty: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    errors: List[Tuple[str, str]] = field(default_factory=list)


_MAX_IMPORT_ERRORS = 20


def run_import(
    units: Iterable[ImportUnit],
    pipeline: Optional[IngestionPipeline],
    dry_run: bool = False,
    progress: Optional[Callable[[ImportUnit], None]] = None,
) -> ImportReport:
    """Feed every unit through the pipeline, one IngestionInput per unit.

    dry_run=True parses and counts everything (including which units the
    deny list would catch) but never calls pipeline.ingest() — nothing is
    written (pipeline may be None). units_ingested/units_unchanged are only
    meaningful for a real run; a dry run reports units that WOULD be
    ingested as units_ingested with nodes_created/edges_created left at 0
    (unknown without writing).

    One unit's failure — whether an importer-caught per-item parse error
    (ImportUnit.error) or a pipeline.ingest() exception — is caught, logged
    into report.errors, and counted; it never aborts the rest of the batch.
    The `units` ITERATOR itself is also guarded (F3): if the generator
    raises directly instead of yielding an error-marker unit, that's logged
    too and iteration stops there (the generator is dead after raising —
    nothing more can be pulled from it), rather than propagating and losing
    every count collected so far.
    """
    report = ImportReport()
    denied = _ingest_deny_set()
    it = iter(units)

    while True:
        try:
            unit = next(it)
        except StopIteration:
            break
        except Exception as exc:  # noqa: BLE001 - the iterator itself failed
            if len(report.errors) < _MAX_IMPORT_ERRORS:
                report.errors.append(("<import>", str(exc)))
            break

        report.units_seen += 1

        if unit.error:
            if len(report.errors) < _MAX_IMPORT_ERRORS:
                report.errors.append((unit.source_id, unit.error))
        elif not unit.content or not unit.content.strip():
            report.units_skipped_empty += 1
        elif unit.source_id in denied:
            report.units_denied += 1
        elif dry_run:
            report.units_ingested += 1
        else:
            try:
                output = pipeline.ingest(IngestionInput(
                    source_id=unit.source_id,
                    content=unit.content,
                    content_type=unit.content_type,
                    timestamp=unit.timestamp,
                    metadata=unit.metadata,
                    links=unit.links,
                    ingest_key=unit.ingest_key,
                    origin_runtime=unit.origin_runtime,
                    origin_source=unit.origin_source,
                    session_key=unit.session_key,
                ))
            except Exception as exc:  # noqa: BLE001 - one bad unit must not sink the batch
                if len(report.errors) < _MAX_IMPORT_ERRORS:
                    report.errors.append((unit.source_id, str(exc)))
            else:
                # F2: unchanged vs ingested. A keyed re-ingest of
                # byte-identical content is pipeline.ingest()'s true no-op
                # path: 0 new nodes, 0 new edges, refreshed=False, the
                # EXISTING context node's id comes back. The FIRST-ever
                # ingest of a unit ALSO has refreshed=False (it isn't a
                # refresh either) but creates real nodes, so refreshed alone
                # can't distinguish "nothing happened" from "first ingest" —
                # both node/edge counts AND refreshed are needed. A keyed
                # REFRESH (content changed) counts as ingested even when
                # re-extraction happened to add 0 new nodes/edges — the
                # plan's "nodes created or refreshed" — because refreshed is
                # True there. A denied unit never reaches here (caught
                # above); the only other empty-id return is "content was
                # entirely recall re-entry" (fence), which a real export's
                # own words never trigger.
                if (
                    output.context_node_id
                    and not output.refreshed
                    and output.nodes_created == 0
                    and output.edges_created == 0
                ):
                    report.units_unchanged += 1
                else:
                    report.units_ingested += 1
                    report.nodes_created += output.nodes_created
                    report.edges_created += output.edges_created

        if progress is not None:
            progress(unit)

    return report
