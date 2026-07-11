---
tags: [product, backend]
date: 2026-01-20
---
FastAPI service backing [[Halcyon]]. Deployed on the [[Atlas Server]] behind Caddy.

## Auth
Passkeys only — see [[Auth Decision]]. Session tokens are 24h, refresh via
device-bound credentials. Shipped in [[Passkey Rollout]].

## Data
Reads and writes the [[Postgres Cluster]]. Sync protocol is CRDT-based per-list.
