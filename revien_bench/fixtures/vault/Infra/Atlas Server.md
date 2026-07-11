---
tags: [infra]
date: 2026-01-08
---
Dedicated server for Fernweh Labs. Hosts [[Halcyon API]], the [[Postgres Cluster]],
and the [[Monitoring Stack]]. Reachable only through the [[Mesh VPN]].

## Hardware
AMD EPYC 8 cores, 64GB ECC RAM, 2x2TB NVMe in RAID 1.

## Access
SSH keys only, no password auth. Fail2ban active. See [[Ops Runbook]] for restart procedure.
