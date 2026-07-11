---
tags: [infra, runbook]
date: 2026-03-12
---
Operational procedures for the [[Atlas Server]].

## Service restart order
Stop [[Halcyon API]] first, then workers, then the [[Postgres Cluster]] standby —
never the primary during business hours.

## After the March OOM
Updated after [[Incident 2026-03 OOM]]: memory alerts now page at 80 percent
and the API runs under a 12GB cgroup limit.
