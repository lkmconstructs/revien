"""Ordinary fact-change detection — CSL Leg B fix.

Measured baseline: 0/68 ordinary fact-changes left the memory holding the current
truth. Root cause: the rule classifier recognizes personal CLAIMS (identity,
preference, emotion, health, ...) but NOT ordinary biographical FACTS (where you
live, where you work, what you drive). Those came back `unclassified`, so the
sensitive floor blocked every update; and contradiction detection only handled
favourite/status/place/polarity templates.

This module decides ONE thing: does the NEW claim CHANGE an EXISTING claim about
the same single-valued topic? It keys on (a) per-dimension value substitution and
(b) explicit retraction / transition markers over a shared topic. It is
deliberately about CONTRADICTION ONLY — sensitivity is decided DOWNSTREAM by the
gate (recognizer + tripwire + protected check), so a fact-shaped sensitive
disclosure ("I moved into a sober-living house") still routes to review, never a
silent erase. Restatements with the SAME value (and no retraction) are NOT changes.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# ── transition / retraction markers on the NEW claim ──────────────────────────
_RETRACT = re.compile(
    r"\b(?:no longer|not\s+\w+\s+anymore|anymore|don'?t|doesn'?t|isn'?t|aren'?t|"
    r"stopped|quit|gave up|gave (?:it |them )?away|sold|cancell?ed|broke (?:up|the lease)|"
    r"laid off|got let go|let go|fired|ended|moved out|drop(?:ped)? out|dropped|ditched|"
    r"got rid of|left\b|passed away|died|switched off|went off|off it)\b", re.I)
_CHANGE = re.compile(
    r"\b(?:now|these days|instead|switched (?:to|schools|jobs|over)?|moved (?:to|in|into|over)|"
    r"relocat\w+|got a new|new\b|upgraded|traded|replaced|finally|took (?:a|the|my)|"
    r"got (?:hired|promoted|my own|certified)|promoted|closed (?:on|my|the|out)?|opened|"
    r"signed (?:a|my|the|up)|taken over|has become|become my|started|again|"
    r"back (?:in|at|to|on)|joined|rescheduled|picked up|used to|cut (?:it|my|off))\b", re.I)

_STOP = {
    "the", "and", "but", "for", "with", "that", "this", "have", "has", "had", "now",
    "you", "your", "are", "was", "were", "been", "being", "from", "into", "out",
    "got", "get", "just", "still", "back", "over", "off", "any", "more", "anymore",
    "not", "dont", "don", "really", "actually", "honestly", "these", "days", "right",
    "what", "when", "where", "they", "them", "his", "her", "she", "him", "our", "their",
    "year", "years", "month", "months", "week", "weeks", "last", "next", "ago", "old",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


_PLACE = r"([A-Z][\w'.\-]+(?:\s+[A-Z][\w'.\-]+){0,2})"

# dimension -> (presence regex, value-capture regex or None)
_DIMS = {
    "residence": (
        re.compile(r"\b(?:live|living|lived|reside|residing|moved|relocat\w+|renting|"
                   r"rents?|sublet|crash(?:ing)?|staying|condo|apartment|walk-?up|"
                   r"homeowner|own (?:a |my )?(?:place|home|house|condo))\b", re.I),
        re.compile(r"\b(?:in|to|near|back to|home to|into)\s+" + _PLACE)),
    "employer": (
        re.compile(r"\b(?:work(?:s|ing)?\s+(?:at|for|as)|employed|job|barista|teacher|"
                   r"teach(?:es)?|nurse|analyst|manager|engineer|designer|coordinator|"
                   r"hired|laid off|promoted|reorg|shift at|the warehouse|freelance)\b", re.I),
        re.compile(r"\b(?:at|for)\s+" + _PLACE)),
    "vehicle": (
        re.compile(r"\b(?:drive|driving|car|truck|suv|sedan|lease|leasing|jeep|vehicle|"
                   r"pickup|carless|civic|tacoma|outback|tesla|mazda|focus|golf|elantra|"
                   r"f-?150|accord|wrangler|subaru|honda|toyota|hyundai|ford|volkswagen)\b", re.I),
        re.compile(r"\b(?:drive|driving|lease|leasing|bought|own|upgraded to|for a|to an?)\s+"
                   r"(?:an?\s+|my\s+|the\s+|new\s+)*(?:\d{4}\s+)?([A-Z][\w]+(?:\s+[\w\-]+)?)")),
    "contact": (
        re.compile(r"\b(?:phone number|cell number|number is|email(?:\s+address)?|reach me at)\b", re.I),
        None),
}

# Favourite is single-valued PER CATEGORY (favourite show vs favourite food differ).
_FAV = re.compile(r"\b(?:favou?rite|go-?to|my team|root for|i'?m an?\s+[\w\s]+\bperson)\b", re.I)
_FAV_CAT = re.compile(
    r"\bfavou?rite\s+(\w+)|(\w+)\s+is\s+(?:my|always been my)\s+favou?rite|"
    r"\b(coffee|tea|show|colou?r|team|meal|food|snack|beer|wine|movie|book|song|drink)\b", re.I)


def _dim(text: str) -> Optional[str]:
    for name, (rx, _v) in _DIMS.items():
        if rx.search(text):
            return name
    if _FAV.search(text):
        return "favourite"
    return None


def _value(text: str, dim: str) -> Optional[str]:
    spec = _DIMS.get(dim)
    if not spec or spec[1] is None:
        return None
    m = spec[1].search(text)
    return m.group(1).strip().strip(".,;:!?'\"").strip() if m else None


# The favourite CATEGORY is the SLOT noun after "favourite" ("favourite drink",
# "favourite meal", "favourite comfort show") — NOT the value (tea, ramen). So two
# "favourite drink" claims compare like-for-like even when the drink differs.
_FAV_FILLER = {
    "kind", "type", "of", "comfort", "way", "part", "go", "thing", "my", "a", "an",
    "the", "new", "absolute", "all", "time", "right", "now", "these", "days",
}


def _fav_category(text: str) -> Optional[str]:
    m = re.search(r"\bfavou?rite\s+((?:[a-z']+\s+){0,3}[a-z']+)", text.lower())
    if m:
        for w in m.group(1).split():
            if w not in _FAV_FILLER and len(w) > 2:
                return "colour" if w == "color" else w
    return None


def _same_value(a: str, b: str) -> bool:
    """Same entity if equal or one is contained in the other (Lincoln ⊂ Lincoln
    Elementary; Honda Civic ⊃ Civic) — a shorter restatement is NOT a change.
    Compared on alphanumeric words so trailing punctuation can't break the match."""
    na = " ".join(re.findall(r"[a-z0-9]+", a.lower()))
    nb = " ".join(re.findall(r"[a-z0-9]+", b.lower()))
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def detect_change(old: str, new: str) -> Tuple[Optional[str], bool]:
    """Return (dimension_label, is_change). is_change True ONLY when the NEW claim
    genuinely changes/retracts the OLD about the same single-valued topic — a real
    transition marker OR a genuinely different value. A bare re-mention of the same
    topic (restatement) is NOT a change."""
    old = (old or "").strip()
    new = (new or "").strip()
    if not old or not new:
        return (None, False)

    has_marker = bool(_RETRACT.search(new) or _CHANGE.search(new))
    do = _dim(old)

    # ── Favourite: single-valued per CATEGORY; a change needs a transition marker ──
    if do == "favourite":
        co, cn = _fav_category(old), _fav_category(new)
        if co is not None and cn is not None and co != cn:
            return ("favourite", False)               # different categories — independent
        return ("favourite", has_marker)              # same/unknown category -> needs a marker

    if do is not None:
        dn = _dim(new)
        vo, vn = _value(old, do), _value(new, do)
        linked = (dn == do) or (vn is not None) or bool(_tokens(old) & _tokens(new))
        if not linked:
            return (do, False)
        if vo and vn and not _same_value(vo, vn):
            return (do, True)                         # genuinely different value
        if vo and vn and _same_value(vo, vn) and not _RETRACT.search(new):
            return (do, False)                        # restatement of the same value
        return (do, has_marker)                       # otherwise: change only with a marker

    # ── Generic (OLD isn't a known dimension): a retraction OR a substitution
    #    marker, over a shared topic token. Catches plans cancelled, habits
    #    stopped, subscriptions dropped, etc. ──
    if has_marker and (_tokens(old) & _tokens(new)):
        return ("generic_change", True)
    return (None, False)
