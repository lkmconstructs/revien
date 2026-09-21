"""
Revien Skills — thin WS3, leg D1: skill ingest.

A skill is a named, reusable procedure (SKILL.md-style: a small frontmatter
block plus a markdown body) that Revien ingests as a first-class SKILL node
so it can be recalled, listed, and shown like any other memory. This package
is deliberately thin: `frontmatter.py` reads the four keys skills use, and
`ingest.py` turns a skill folder into a SKILL node, idempotently.

Leg D2 (proposals — engine-authored skill drafts from repeated ACTION
sequences) lands in this same package later; nothing here assumes it exists.
"""

from revien.skills.ingest import (
    ingest_roots,
    list_skills,
    show_skill,
    skill_ingest_key,
    sort_user_before_engine,
)

__all__ = [
    "ingest_roots",
    "list_skills",
    "show_skill",
    "skill_ingest_key",
    "sort_user_before_engine",
]
