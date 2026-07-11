---
tags: [reading, database]
date: 2026-01-12
---
Comparison for Halcyon's server datastore.

## Conclusion
SQLite wins on-device; the server side needs concurrent writers and logical
replication, so the server runs the [[Postgres Cluster]]. Revisit if sync load
ever drops to single-writer.
