---
tags: [incident]
date: 2026-03-09
---
The [[Atlas Server]] hit OOM at 02:14; the kernel killed the [[Halcyon API]] worker pool.

## Root cause
A sync client retry storm ballooned worker memory. No data loss — the
[[Postgres Cluster]] was untouched.

## Follow-up
Cgroup limits and earlier paging, folded into the [[Ops Runbook]].
