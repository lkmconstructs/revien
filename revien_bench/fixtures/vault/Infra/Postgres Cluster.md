---
tags: [infra, database]
date: 2026-01-15
---
Primary datastore for [[Halcyon]]. Runs on the [[Atlas Server]] under systemd.

## Configuration
PostgreSQL 16, one primary plus a warm standby. Tuned by [[Jonas Petrov]] in February —
shared_buffers 16GB, work_mem 64MB.

## Backups
Nightly dumps shipped per the [[Backup Policy]].
