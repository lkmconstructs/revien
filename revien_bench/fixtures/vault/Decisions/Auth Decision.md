---
tags: [decision, security]
date: 2026-02-14
---
Decision: [[Halcyon]] uses passkeys instead of passwords. No password column exists.

## Rationale
Password resets were 30 percent of projected support load. Passkeys kill credential
stuffing outright and fit the local-first story: the credential stays on the device.

## Consequences
Recovery flow needs a printed recovery code. Implementation notes in [[Passkey Rollout]].
