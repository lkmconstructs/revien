"""
Revien Importers — one-shot exports (ChatGPT, Claude.ai, Readwise) through
the ingestion pipeline.

Unlike the adapters in revien/adapters/ (which poll a live source and write
nodes straight to the store), an importer is a batch job: it parses a
downloaded export into ImportUnit objects (base.ImportUnit) and hands each
one to IngestionPipeline.ingest() via run_import(), so imported content gets
the same deny-list check, context fence, dedup, and idempotent-refresh
handling as everything else — see revien/ingestion/pipeline.py.

Submodules:
    base.py     open_export(), ImportUnit, run_import(), ImportReport.
    chatgpt.py  iter_units() for a ChatGPT conversations.json export.
    claude.py   iter_units() for a Claude.ai conversations.json export.
    readwise.py iter_units() for a Readwise highlights CSV export.

Zero egress: stdlib zipfile/csv/json only, no dependency added.
"""
