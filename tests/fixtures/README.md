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
