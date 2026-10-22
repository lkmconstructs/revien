Database fixtures for the migration tests (tests/test_migration_005.py).

Each `.sql` file is `sqlite3.Connection.iterdump()` output from a real
database created by running the named Revien code on a tiny graph (3 nodes,
1 edge, 1 audit row; a sync cursor where that version has the table):

- `db_main_03217f4.sql`: origin/main at 03217f4 (user_version 4, no usage layer).
- `db_v030_tag_7bf8738.sql`: tag v0.3.0 (7bf8738), user_version 0.
- `db_v030_7757085.sql`: commit 7757085 "Revien 0.3.0", user_version 0.
- `db_main_used_03217f4.sql`: origin/main at 03217f4 after real use: ChatGPT,
  Claude.ai and Readwise imports, a Claude Code adapter sync via the
  scheduler, an mcp-source ingest and a recall (user_version 4).
- `db_main_b905109.sql`: origin/main at b905109 (the first PR: usage signal,
  receipts, sync reset with fencing, migration 005; user_version 5) after real
  use: ChatGPT and Readwise importers, a Claude Code adapter sync through the
  scheduler, an mcp-source ingest, a usage judgment, a recall receipt, a
  supersession candidate and a sync reset. Used by tests/test_forget_upgrade.py.
- `db_v02x_schema.sql`: HAND-BUILT in the v0.2.x shape (nodes and edges before
  the confidence layer, no audit_log, user_version 0). The repository history
  starts at 0.3.0, so there is no 0.2.x code to generate one from; the shape is
  what migration 001 documents as "pre-confidence".
