---
tags: [infra, security]
date: 2026-01-10
---
WireGuard mesh connecting the laptop, the studio workstation, and the [[Atlas Server]].
The mesh is the only ingress to production — nothing else listens publicly.

## Peers
Three peers, keys rotated quarterly. DNS via the mesh resolver.
