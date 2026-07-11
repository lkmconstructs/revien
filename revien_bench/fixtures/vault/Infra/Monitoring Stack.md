---
tags: [infra, observability]
date: 2026-02-20
---
Prometheus plus Grafana on the [[Atlas Server]]. Alertmanager pages via ntfy.

## Alert policy
Disk over 85 percent, OOM events, and certificate expiry under 14 days all page
immediately. Response steps live in the [[Ops Runbook]].
