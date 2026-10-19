"""Forget, delete and cleanup (docs/FORGET_SPEC.md).

Three verbs, never folded together: forget (privacy: gone, and not coming
back), delete (operational: remove this node now) and automatic cleanup
(retention). The engine for forget is revien.forget.engine; the gate every
capture path asks is revien.forget.gate.
"""
