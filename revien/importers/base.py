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

import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

# Case-insensitive match for the member we're looking for inside a zip,
# at any depth (ChatGPT/Claude exports sometimes nest it one folder in).
_CONVERSATIONS_JSON = "conversations.json"


def open_export(path: str) -> bytes:
    """Read an export file's raw bytes, unwrapping a zip if given one.

    Accepts either a .zip archive (the file OpenAI/Claude.ai actually hand
    you when you export) or a direct .json/.csv path (already unzipped, or
    a Readwise CSV, which is never zipped). A zip is searched for the first
    member literally named "conversations.json" (case-insensitive), at any
    depth — never extracted to disk, read straight out of the archive.

    Raises FileNotFoundError if path doesn't exist, and ValueError if a
    zip is given but contains no conversations.json member.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Export not found: {path}")

    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                if Path(name).name.lower() == _CONVERSATIONS_JSON:
                    return zf.read(name)
        raise ValueError(
            f"{path}: zip has no conversations.json member (looked at any depth)."
        )

    return p.read_bytes()


def slugify(text: str) -> str:
    """Lowercase, non-alphanumeric runs collapsed to '-'. Same shape as
    adapters/obsidian.py's _slug — kept local rather than importing a
    private name across modules. Empty/all-punctuation input -> 'untitled'
    (obsidian's fallback is 'section'; nothing here is ever a section)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


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
    almost everything straight through."""
    source_id: str
    ingest_key: str
    content: str
    content_type: str
    timestamp: Optional[datetime]
    origin_runtime: str
    origin_source: str = "import"
    project_key: Optional[str] = None
    session_key: Optional[str] = None
    links: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


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

    _MAX_ERRORS = 20

    def add_error(self, source_id: str, message: str) -> None:
        if len(self.errors) < self._MAX_ERRORS:
            self.errors.append((source_id, message))


def _denied_source_ids() -> set:
    """Mirrors ingestion.pipeline._ingest_deny_set() exactly (same env var,
    same parsing) so an importer can count denials up front instead of
    inferring them from a zero-count IngestionOutput — the pipeline
    enforces the deny list either way; this just makes the count honest
    without a second, divergent copy of the policy."""
    import os
    raw = os.environ.get("REVIEN_INGEST_DENY", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def run_import(
    units: Iterable[ImportUnit],
    store: GraphStore,
    pipeline: IngestionPipeline,
    dry_run: bool = False,
    progress: Optional[Callable[[ImportUnit], None]] = None,
) -> ImportReport:
    """Feed every unit through the pipeline, one IngestionInput per unit.

    dry_run=True parses and counts everything (including which units the
    deny list would catch) but never calls pipeline.ingest() — nothing is
    written. units_ingested/units_unchanged are only meaningful for a real
    run; a dry run reports units that WOULD be ingested as units_ingested
    with nodes_created/edges_created left at 0 (unknown without writing).

    One unit's failure is caught, logged into report.errors, and counted —
    it never aborts the rest of the batch.
    """
    report = ImportReport()
    denied = _denied_source_ids()

    for unit in units:
        report.units_seen += 1

        if not unit.content or not unit.content.strip():
            report.units_skipped_empty += 1
            if progress is not None:
                progress(unit)
            continue

        if unit.source_id in denied:
            report.units_denied += 1
            if progress is not None:
                progress(unit)
            continue

        if dry_run:
            report.units_ingested += 1
            if progress is not None:
                progress(unit)
            continue

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
                project_key=unit.project_key,
                session_key=unit.session_key,
            ))
        except Exception as exc:  # noqa: BLE001 - one bad unit must not sink the batch
            report.add_error(unit.source_id, str(exc))
            if progress is not None:
                progress(unit)
            continue

        # A keyed re-ingest of unchanged content is pipeline.ingest()'s
        # no-op path (pipeline.py ~349-397): zero nodes, zero edges, but
        # the EXISTING context node's id comes back (non-empty). A denied
        # unit never reaches here (caught above); the only other empty-id
        # return is "content was entirely recall re-entry" (fence), which
        # a real export's own words never trigger.
        if (
            output.nodes_created == 0
            and output.edges_created == 0
            and output.context_node_id
        ):
            report.units_unchanged += 1
        else:
            report.units_ingested += 1
            report.nodes_created += output.nodes_created
            report.edges_created += output.edges_created

        if progress is not None:
            progress(unit)

    return report
