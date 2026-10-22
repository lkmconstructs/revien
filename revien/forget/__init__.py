"""Forget, delete and cleanup (docs/FORGET_SPEC.md).

Three verbs, never folded together: forget (privacy, for one specific thing:
gone, and not coming back), delete (operational: remove this node now, a
whole conversation included) and automatic cleanup (retention). The engine
for forget is revien.forget.engine; the gate every capture path asks is
revien.forget.gate.
"""
