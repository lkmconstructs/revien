"""
Revien label normalization — ONE definition of "same name, different surface."

"Fernweh-Core", "fernweh core", and "Fernweh Core" are the same
entity wearing three outfits. Every place labels meet (dedup, lookups, anchor
matching, link resolution, mention linking) normalizes through this function,
so surface-form equivalence has exactly one definition to test, tune, or blame.

Deliberately NOT here (each is a false-merge machine in this form):
  * stemming / plural folding ("Redis" is not "Redi")
  * abbreviation or alias mapping ("PostgreSQL" vs "postgres" is a synonym
    problem — that's vocabulary, not surface form)
  * unicode confusable folding (future work if it ever bites)
"""

import re
from functools import lru_cache

_SEPARATORS = re.compile(r"[-_./\\]+")
_POSSESSIVE = re.compile(r"'s\b")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an) ")


@lru_cache(maxsize=65536)
def normalize_label(label: str) -> str:
    """Canonical surface form: lowercase, separators to spaces, possessives
    and stray punctuation dropped, leading article stripped, whitespace
    collapsed. Article stripping is load-bearing: the extractor captures
    'The Atlas Server' while a vault note is titled 'Atlas Server' — same
    entity, and this is where they meet. ('The Who'-style names lose their
    article; accepted trade for v1.)"""
    s = (label or "").lower()
    s = _SEPARATORS.sub(" ", s)
    s = _POSSESSIVE.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return _LEADING_ARTICLE.sub("", s)


def normalize_text(text: str) -> str:
    """Same canonicalization applied to running text, for mention scanning —
    WITHOUT the leading-article strip (that's a label-boundary rule; inside
    running text 'the' is just a word and the space-padded search handles it).
    Pad-and-search with single spaces gives word-boundary matching:
    ``f" {normalize_label(entity)} " in f" {normalize_text(content)} "``."""
    s = (text or "").lower()
    s = _SEPARATORS.sub(" ", s)
    s = _POSSESSIVE.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()
