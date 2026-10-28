"""
Exact lane (REVIEN_EXACT) - EXPERIMENTAL, default off.

A recall lane that switches on only when the query contains something exact,
and otherwise does nothing at all. It is not a text-scoring lane: it never
looks at node content. It matches three structured things:

  * quoted phrases, names/titles and code-like tokens against node LABELS
    (exact, case-insensitive; whitespace is collapsed in the query only, so a
    stored label with a tab or repeated spaces between words does not match),
    and, when a query names an entity but asks about something else, the
    nodes CONNECTED to that entity by an edge (one hop);
  * dates against the structured event-time fields (event_time_start /
    event_time_end / event_time_granularity) at full weight, then against
    recorded_at (when the content was SAID, local date in its own offset) at
    half weight, and only where recorded_at_source is content, capture or
    import. Never against text, and never against created_at (ingest time);
  * node ids typed into the query.

A match yields a small, capped, additive boost on the node's normal score and
may admit a few matched nodes into the candidate set. It never promotes: see
ExactLane.build and the constants below.

DETECTION RULES (deterministic; no model; every rule is here)

 1. Quoted phrases. "double quotes", curly double quotes, and 'single quotes'
    that are not apostrophes (not touching a letter/digit on the outside), 2 to
    80 characters. A quoted phrase is matched against labels as a whole. A
    quoted phrase is an explicit title: the common-word guard (below) does not
    apply to it, the df guard does.
 2. Dates (resolved to a day or a month range):
      2026-09-21  2026/09/21            ISO, year first, "-" or "/" only
      09/21/2026  9/21/26               month/day/year (if the first number is
                                        over 12 and the second is not, it is
                                        read day/month/year: a year is there
                                        to settle it)
      9/21                              month/day, no year. Day-first needs a
                                        year or a month name ("21/09/2026",
                                        "21 September"); "21/9" is not a date.
      September 21   Sept 21   Sep. 21st   month name (full, 3-letter or
                                        "sept") + day, optional ordinal
      September 21, 2026   Sep 21st 2026
      21 September 2026   21st of September   day + month name
      September 2026                    a whole month
      May 5-10   Sept 28 - Oct 3   9/21-9/23   12/24 to 12/26
      21-23 September 2026   9/21/2026-9/23/2026   2026-09-21 to 2026-09-23
                                        a RANGE: two dates joined by a hyphen,
                                        an en or em dash, or "to", spaced or
                                        not, read as ONE signal for the whole
                                        span, first day to last. Both ends must
                                        be real dates, in order, at most
                                        MAX_RANGE_DAYS (31) days first to last
                                        ("Dec 28 - Jan 3" crosses the year; a
                                        year written once belongs to the end it
                                        is written at). The ratio rule applies
                                        to the start ("1/2-3/4 cup" is two
                                        fractions). A range that breaks a rule
                                        is not a range; its parts are read only
                                        as single dates can be, edges included:
                                        "May 10-5" and "Sept 21-45" give no
                                        date, "May 1 - June 15" gives May 1 and
                                        June 15, "May 5 to 3 pm" gives May 5.
                                        Ranges are read first. A month word
                                        is never taken into a range when it
                                        is contested (below) or when its day
                                        belongs to another month ("May 5 - 10
                                        June" is not May 5-10).
    A date with no year ("September 21", "9/21") matches that day in ANY year
    the graph has a dated node for. Relative dates ("next Friday", "last
    week") are out of scope. A month alone ("in September") is not a date
    signal. THINGS THAT LOOK LIKE DATES AND ARE NOT (each is a fixed rule):
      * ratios and scores. A bare a/b (no year, no month name) with b in
        RATIO_DENOMS (2 3 4 5 6 8 10 20) and a <= b is a ratio or a score:
        "1/2 cup", "3/4", "8/10", "rated 9/10", "10/20", "5/5", UNLESS it is
        marked as a date. A wrong date is worse than none, so only three
        things mark one: it is zero-padded ("03/10", "05/05"); it has a year
        ("3/10/2026"; any M/D/Y is a date); or it follows ON, BY, BEFORE,
        AFTER, SINCE, UNTIL, TILL or THROUGH (SLASH_DATE_PREPOSITIONS,
        optionally with "the" between: "on 3/10", "by the 3/10"), is not
        followed by "of" or a hyphenated unit ("through 3/4 of the book",
        "on 3/4-inch pipe"), and, after BY, does not
        follow a verb of quantity ("reduced by 1/2", FRACTION_VERBS). No other
        word marks one: "of", "from", "between", "during", "in", "out of",
        "score of" do not ("a score of 8/10", "the ratio of 3/4", "between
        1/2 and 3/4 cup", "in 8/10 cases"), and "from 3/4 to 3/8" is two
        ratios unless padded or given a year. A bare a/b with a > 12 has no
        month ("24/7", "16/9") and is not a date even after a preposition. Of
        the 366 bare M/D forms, 316 parse with no preposition and all 366
        after "on".
      * a date's characters are blanked once it is read, and the text on
        either side of a blanked date never joins into another date; a date
        preposition marks only the date right after it ("on Sept 21 the 3/4
        mark" reads no 3/4).
      * edges. No date reads across a slash, a hyphen or a dot into another
        number: a number at either end of a match touches no digit, "/" or
        "-", nor a "." or "-" with a digit beyond it, and a month word at
        either end touches no "/" and no "-" joined to a digit (_NUM_L,
        _NUM_R, _MON_L, _MON_R). "21 September 3/4" is Sept 21, not Sept 3;
        "the 3/4 May meeting" and "on 9/21 Mar" hold no May 4 and no Mar 21;
        "1-3/4 cups", "v1.5 May", "Apr/May 2026" and "September 2026/27"
        hold no date. A range is read only whole, as a range (above): "May
        5-10" is never "May 5". "mid-May 2026" is still May 2026.
      * a CONTESTED month word, one with a day claim on each side, gives
        no date at all, and neither claim is read with another month: "on 5
        March 3 people came", "page 3 March 5", "room 21 Jun 19-23", "top 10
        May 4-7", "ages 5-10 May 3", "10-11 Sept 21", "3 May 5-10", "15-16
        June 3" and "28 Sept 10 June" give none. A claim BEFORE the month is
        a standalone day ("5 ", "5th of ") or the end of an ORDERED number
        range ("5-10 ", "3 to 21 "); not a time ("10:30", "3:00"), money
        ("$20"), a decimal, a code ("PR-12", "Q3"), a slash number, an
        unordered pair ("won 3-1 May 5", "hours 9-5 May 1st" keep May 5 and
        May 1), nor the possible day of a month right before it ("May 5
        June 3" is May 5 and June 3; that month contested, the number counts
        for both). A claim AFTER the month is a day or a range start ("May
        3", "May 5-10", "Sept. 21"), never after a comma ("5 May, 3 people"
        is 5 May), and not a year, a time or a unit ("May 2 pm", "17-20 Oct
        3:00", "15-16 June 5 km" and "4 of us 6-7 May 2 pm" keep their
        range). An ordinal and "of" decide ("the 5th of May 2 tickets" is 5
        May). Contested months are found (_RE_CLAIM_BEFORE, _RE_CLAIM_AFTER)
        and blanked, with their claims, before any month-name form is read.
        A month after a number it cannot be the day of ("31 Sept 2026") is
        not read as the month either.
      * versions. Dotted numbers are never dates ("release 2024.1.5",
        "v2024.01.05"): the year-first form accepts "-" and "/" only.
      * words. "march", "mar", "may" and "jan" are ordinary words or names as
        well as months ("we march 3 miles", "it may 5 times", "jan 5 said").
        They are read as a month only when written capitalized ("March 3",
        "May 5", "MAR 3"), or after one of ON, BY, SINCE, UNTIL, TILL, FROM,
        BEFORE, AFTER, DURING, THROUGH, BETWEEN, OF or IN (optionally with
        "the" between: "on march 3", "from may 5", "in march 2026"), or with
        an ordinal day or "of" ("the 5th of may", "3rd march"). Every other
        month name and abbreviation is read in any case.
 3. Names and titles. Runs of capitalized tokens (joiners "of the and de la
    van von du" allowed inside a run), every contiguous sub-span up to 6
    tokens. A possessive ending ('s or \u2019s) is dropped from the word, so
    "Dana's" is the name "Dana", and ends the run. A span that contains the
    first token of a sentence is eligible only if it is an existing label; a
    span elsewhere is a name signal even if no label matches (it is reported
    as detected, unmatched). A sentence-initial single word never counts as a
    name when it is a COMMON WORD: a question/function word ("What", "Did"),
    a month ("March is coming"), a weekday ("Monday"), a short common acronym
    ("OK", "AI") or a generic title word ("Notes"); the pronoun "I" never
    counts either. Quote it to mean the title. Matching is label equality,
    longest span first, no token used twice. Lowercase text is a name signal
    only when the WHOLE query (trimmed, punctuation-trimmed, possessive
    dropped) equals a label ("harbor"), at most 6 tokens.
    WHAT A NAME DOES. A query is ABOUT the entity when, once the name and its
    framing words are set aside, nothing is left: the bare name, "who is X",
    "tell me about X", "what do you know about X" (FRAMING_WORDS; extend with
    REVIEN_EXACT_FRAMING_WORDS). Then the node labelled X is boosted. When the
    query names an entity AND carries other content words, a date, a quote or
    a code ("What did X write about Y?"), the entity node is NOT boosted (it
    would outrank the node that answers); the nodes connected to it by an
    edge, one hop, either direction, any edge type but ALIAS_OF, are boosted
    instead, at NEIGHBOR_STRENGTH of a name boost, and only where they are
    already candidates (they are never admitted to the candidate set). An
    entity with more than NEIGHBOR_MAX neighbours is a hub: its neighbours'
    boost is scaled by NEIGHBOR_MAX / count, and 0 beyond 8 * NEIGHBOR_MAX
    (REVIEN_EXACT_NEIGHBOR_MAX). Nodes whose CONTENT mentions the label are
    not searched for: content is not indexed, so that would be a scan of the
    whole content column on every question.
 4. IDs and codes. A token (trailing punctuation trimmed) that contains a
    letter and a digit ("PR-1234", "v0.3.0", "abc123"), or an underscore
    between alphanumerics ("user_id_map"), or a hyphen join where one side is
    ALL CAPS or has a digit ("REV-FIX"), or a run of 5+ digits. Ordinals and
    clock/size shorthand ("21st", "3pm", "10:30", "5k") and plain years are
    not IDs, and neither is a token made only of digits and "/", "-", en or
    em dashes ("5-10", "21-23", "9/21-9/23", "555-1234"), alone or with a
    time or unit after it ("5-6pm", "10:30-11:30", "10-20%"): a range, a
    score or a fraction, which would otherwise lift a label such as "Ages
    5-10". Tokens consumed by a date or quote are not IDs. Matching is a
    label equal to the token, a label containing the token as a whole token,
    or a node id equal to the token. Never node content. The containment
    lookup reads the label column once per code, so it is made for at most
    MAX_CONTAINED_CODES (16, REVIEN_EXACT_MAX_CONTAINED_CODES; 0 turns it
    off) codes of 4+ characters per query, the most distinctive first:
    longest first, ties in query order. Equality and node-id matching still
    run for every code. A code left out is reported, never silently
    unmatched: its ids entry says "contained_search": "skipped_by_cap", and
    detected["ids_capped"] gives {"capped": true, "searched", "skipped",
    "skipped_codes", "max_contained_codes"}.
 5. How much of a query is read. The lane reads at most SCAN_CHARS (4,096,
    REVIEN_EXACT_SCAN_CHARS) characters of a query. A longer query is read
    at both ends, since the question often comes last: the first half of
    the budget (2,048) and the last half (2,048), each cut at whitespace so
    no token is split, and never inside a quoted phrase: the quote rule is
    run over the whole query and a phrase a cut would land in is left out of
    that part, so the text between two quotes is never read as one. The middle is not searched for signals; the two ends
    are read apart (nothing joins across the gap, and the tail starts a
    sentence), and a query that was cut has no whole-query label (rule 3).
    A cut is always reported, signal or not: detected["truncated"] is true,
    with "scanned_chars" and "query_chars" (build returns a result with no
    boost instead of None, so debug diagnostics show it). Detection is a
    single left-to-right pass per rule (linear in what it reads). The bound
    is the lane's own: the rest of recall sees the whole query, and with the
    lane off none of this runs.

THE BOOST (bounded, not a promotion)

    boost(node) = min( BOOST_MAX, sum_over_signals( BOOST_MAX * strength * weight ) )

    strength  quoted / multiword name / id equal / node id  1.0
              single-token name / whole-query label          0.8
              id contained in a label                        0.7
              neighbour of a named entity: the name's strength * NEIGHBOR_STRENGTH (0.5)
              date on event_time: min(node granularity, query granularity)
                    with day 1.0, week 0.6, month 0.4, year 0.15, fuzzy 0.1
                    A RANGE is one signal at day granularity: a node whose
                    event_time falls in the span matches as it would match
                    one exact day (a day node 1.0, a month node 0.4), once,
                    never once per day, so a span can lift a node no more
                    than one exact day would. A wide span matches more nodes,
                    and the DATE_MAX_MATCHES guard scales it down as one
                    signal over all of them.
              date on recorded_at (only for a node event_time did not
                    match): 0.5 * the query granularity weight
    weight    generic guard, below. 1.0 for an unshared title.

    It is added to the node's final score (after confidence, type prior and
    prefer_types), so the node can rise by at most BOOST_MAX (default 0.10,
    REVIEN_EXACT_BOOST_MAX) and a node more than that behind the leader cannot
    reach #1 however exactly it matches.

    WITH THE RERANKER ON (it is by default) the cross-encoder re-sorts its
    head by its own score, on a scale that is not the engine's. The boost is
    therefore applied a second time, AFTER reranking, on the engine's scale:
    the head keeps the reranker's ORDER, the head's own (pre-boost) scores are
    dealt back out down that order, the boost is added, and the head is
    re-sorted (lift_after_rerank). The cap means the same thing as with the
    reranker off: a node whose slot is more than BOOST_MAX of score below the
    leader's cannot reach #1. The first application still decides which nodes
    enter the head. With no boost, or no reranker, none of this runs.

    WITH BM25 ON (REVIEN_LEXICAL=bm25) the BM25 score already counted the
    label's words (it indexes label and content), so every boost that comes
    from label text (name, quote, id, whole-query) is scaled by
    BM25_LABEL_SCALE (default 0.5, REVIEN_EXACT_BM25_LABEL_SCALE). Dates,
    typed node ids and neighbours are not in the BM25 score and are not scaled.

GENERIC GUARD (document frequency over labels)

    A matched label shared by df live nodes: weight = 1/df when
    df <= MAX_DF (default 3, REVIEN_EXACT_MAX_DF), else 0. A label made only
    of common words (function words, generic titles such as notes, plan,
    meeting, month names, weekday names and short common acronyms; the whole
    list is COMMON_WORDS, extend it with REVIEN_EXACT_COMMON_WORDS=a,b,c)
    gets weight 0, except a quoted phrase. A code contained in many labels is
    guarded the same way by its match count. For dates, a signal matching more
    than DATE_MAX_MATCHES nodes (default 25, REVIEN_EXACT_DATE_MAX_MATCHES) is
    scaled by DATE_MAX_MATCHES / matches.

CANDIDATE ENTRY (bounded)

    A matched node with boost > 0 that is not already an anchor is added to
    the anchor set, at most MAX_INJECT (default 5, REVIEN_EXACT_MAX_INJECT)
    per recall, best boost first (ties by node id). A node that is boosted
    only as a neighbour of a named entity is not admitted. Nor is a node
    matched only by a CROWDED date, one with more matches than
    DATE_MAX_MATCHES ("Sept 1-30" on a dense graph): it cannot say which of
    its nodes is meant, so it lifts the ones already candidates (scaled by
    DATE_MAX_MATCHES / matches) and admits none. Its entry in
    detected["dates"] says "crowded": true, with "matches_total" (each date
    lookup reads at most 400 rows, the store's bound).
"""

from __future__ import annotations

import bisect
import calendar
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from revien.graph.schema import TemporalGranularity

# ── defaults (class-level constants of the lane; env overrides in ExactConfig) ─
BOOST_MAX = 0.10
MAX_DF = 3
DATE_MAX_MATCHES = 25
MAX_INJECT = 5
# A neighbour of a named entity counts half as much as the entity itself would.
NEIGHBOR_STRENGTH = 0.5
# An entity with more neighbours than this is a hub: scaled down by
# NEIGHBOR_MAX / count, and ignored past HUB_FACTOR * NEIGHBOR_MAX.
NEIGHBOR_MAX = 25
HUB_FACTOR = 8
# Label-text boosts are scaled by this when the BM25 lane (which already
# counted the label's words) is on.
BM25_LABEL_SCALE = 0.5

GRAN_WEIGHT = {
    "day": 1.0, "week": 0.6, "month": 0.4, "year": 0.15, "fuzzy": 0.1,
}
SINGLE_NAME_STRENGTH = 0.8
# A date matched on recorded_at (when the content was SAID) counts half as
# much as one matched on a structured event_time.
RECORDED_AT_STRENGTH = 0.5
ID_CONTAINED_STRENGTH = 0.7
MAX_SPAN_TOKENS = 6
MAX_YEARS = 120
# The lane reads at most this many characters of a query (rule 5).
SCAN_CHARS = 4096
# Codes looked for INSIDE labels per query (rule 4): each one costs a pass
# over the label column.
MAX_CONTAINED_CODES = 16

_QUESTION_START = {
    "what", "whats", "who", "whom", "whose", "when", "where", "why", "how",
    "which", "did", "do", "does", "is", "are", "was", "were", "am", "can",
    "could", "should", "would", "will", "shall", "may", "might", "have",
    "has", "had", "tell", "show", "give", "find", "list", "remind", "recall",
    "remember", "any", "anything", "i", "im", "ive", "id", "ill", "my", "me",
    "we", "you", "he", "she", "it", "they", "the", "a", "an", "and", "or",
    "but", "if", "so", "to", "of", "in", "on", "at", "for", "with", "about",
    "this", "that", "these", "those", "there", "here", "please", "also",
}
_JOINERS = {"of", "the", "and", "de", "la", "van", "von", "du", "del", "da"}

# Words that are only framing around a name: "who is X", "tell me about X",
# "what do you know about X". A query whose words, once the name is set aside,
# are all in this set is ABOUT the entity. Extend with REVIEN_EXACT_FRAMING_WORDS.
FRAMING_WORDS = _QUESTION_START | {
    "know", "knew", "known", "info", "information", "details", "detail",
    "describe", "explain", "everything", "look", "lookup", "up", "overview",
    "background", "profile", "bio", "us", "mean", "means", "again", "yet",
}

MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
WEEKDAY_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun",
}
ACRONYM_WORDS = {
    "ok", "okay", "ai", "tv", "pc", "us", "uk", "eu", "un", "usa", "faq",
    "fyi", "asap", "eta", "tbd", "tba", "rsvp", "ceo", "cto", "cfo", "hr",
    "qa", "ui", "ux", "api", "url", "pdf", "gps", "diy", "pm",
}

# Words that make a label generic when a label is made of nothing else; a
# sentence-initial single word in this set is never read as a name.
COMMON_WORDS = _QUESTION_START | MONTH_WORDS | WEEKDAY_WORDS | ACRONYM_WORDS | {
    "notes", "note", "plan", "plans", "meeting", "meetings", "todo", "todos",
    "task", "tasks", "summary", "idea", "ideas", "thought", "thoughts",
    "misc", "general", "draft", "drafts", "list", "lists", "update",
    "updates", "info", "information", "stuff", "things", "thing", "project",
    "projects", "work", "personal", "home", "today", "tomorrow", "yesterday",
    "reminder", "reminders", "question", "questions", "topic", "topics",
    "context", "memory", "memories", "conversation", "chat", "log", "logs",
    "journal", "review", "reviews", "day", "week", "month", "year", "time",
    "event", "events", "people", "person", "preference", "preferences",
    "decision", "decisions", "fact", "facts", "entity", "default", "other",
    "new", "old", "untitled", "test", "tests", "data", "document", "file",
}

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11,
    "dec": 12,
}
_MON = "(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + ")"
_ORD = r"(?:st|nd|rd|th)?"

_RE_QUOTE = re.compile(
    r"\"([^\"]{2,80})\"|“([^”]{2,80})”|"
    r"(?<![A-Za-z0-9])'([^']{2,80})'(?![A-Za-z0-9])")
# The edges of a date (rule 2): a date never reads across a slash, a hyphen
# or a dot into another number ("21 September 3/4" is not Sept 3, "the 3/4
# May meeting" is not May 4, "on 9/21 Mar" is not Mar 21, "May 5-10" and
# "1-3/4" are not dates). A number at either end of a match touches no digit,
# no "/" and no "-"; nor a "." or "-" with a digit beyond it. A month word at
# either end touches no "/" and no "-" joined to a digit ("mid-May 2026" is
# still May 2026).
_NUM_L = r"(?<![\d/\-])(?<!\d\.)"
_NUM_R = r"(?!\d)(?![/\-.:]\d)"
_MON_L = r"(?<!/)(?<!\d-)"
_MON_R = r"(?!/)(?!-\d)"
# Year-first accepts "-" and "/" only: dotted numbers are versions ("2024.1.5").
_RE_ISO = re.compile(r"(?<![\d/.-])(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d|[/.\-]\d)")
_RE_SLASH_Y = re.compile(
    r"(?<![\d/.\-])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])(?![\-.]\d)")
_RE_DMY = re.compile(
    rf"{_NUM_L}\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b\.?,?\s+(\d{{4}})\b{_NUM_R}",
    re.I)
_RE_MDY = re.compile(
    rf"{_MON_L}\b{_MON}\b\.?\s+(\d{{1,2}}){_ORD}\b,?\s+(\d{{4}})\b{_NUM_R}", re.I)
_RE_MD = re.compile(rf"{_MON_L}\b{_MON}\b\.?\s+(\d{{1,2}}){_ORD}\b{_NUM_R}", re.I)
_RE_DM = re.compile(
    rf"{_NUM_L}\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b{_MON_R}", re.I)
_RE_SLASH = re.compile(r"(?<![\d/.\-])(\d{1,2})/(\d{1,2})(?![\d/])(?![\-.]\d)")
_RE_MY = re.compile(rf"{_MON_L}\b{_MON}\b\.?,?\s+(\d{{4}})\b{_NUM_R}", re.I)

# Ranges (rule 2): two dates joined by a hyphen, an en dash, an em dash or
# "to", spaced or not. Read before the single dates, and whole or not at all.
_SEP = r"\s*(?:[-\u2013\u2014]|to)\s*"


def _mon(name: str) -> str:
    return (rf"(?P<{name}>"
            + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + ")")


_RE_ISO_RANGE = re.compile(
    r"(?<![\d/.\-])(?P<y1>\d{4})[-/](?P<m1>\d{1,2})[-/](?P<d1>\d{1,2})" + _SEP
    + r"(?P<y2>\d{4})[-/](?P<m2>\d{1,2})[-/](?P<d2>\d{1,2})(?!\d|[/.\-]\d)")
_RE_SLASH_RANGE = re.compile(
    r"(?<![\d/.\-])(?P<m1>\d{1,2})/(?P<d1>\d{1,2})(?:/(?P<y1>\d{4}|\d{2}))?" + _SEP
    + r"(?P<m2>\d{1,2})/(?P<d2>\d{1,2})(?:/(?P<y2>\d{4}|\d{2}))?"
    + r"(?![\d/])(?![\-.]\d)")
# month day - [month] day [, year]   ("May 5-10", "Sept 28 - Oct 3, 2026")
_RE_MD_RANGE = re.compile(
    rf"{_MON_L}\b{_mon('m1')}\b\.?\s+(?P<d1>\d{{1,2}}){_ORD}\b{_SEP}"
    rf"(?:{_mon('m2')}\b\.?\s+)?(?P<d2>\d{{1,2}}){_ORD}\b"
    rf"(?:,?\s+(?P<y2>\d{{4}})\b)?{_NUM_R}", re.I)
# month day, year - month day [, year]: after a year the end names its month
# ("Sept 21, 2026-27" is not Sept 21-27)
_RE_MDY_RANGE = re.compile(
    rf"{_MON_L}\b{_mon('m1')}\b\.?\s+(?P<d1>\d{{1,2}}){_ORD}\b,?\s+"
    rf"(?P<y1>\d{{4}})\b{_SEP}{_mon('m2')}\b\.?\s+(?P<d2>\d{{1,2}}){_ORD}\b"
    rf"(?:,?\s+(?P<y2>\d{{4}})\b)?{_NUM_R}", re.I)
# day [month [year]] - day month [year]   ("21-23 September 2026", "28 Dec - 3 Jan")
_RE_DM_RANGE = re.compile(
    rf"{_NUM_L}\b(?P<d1>\d{{1,2}}){_ORD}"
    rf"(?:\s+(?:of\s+)?{_mon('m1')}\b\.?(?:,?\s+(?P<y1>\d{{4}})\b)?)?{_SEP}"
    rf"(?P<d2>\d{{1,2}}){_ORD}\s+(?:of\s+)?{_mon('m2')}\b"
    rf"(?:\.?,?\s+(?P<y2>\d{{4}})\b{_NUM_R}|{_MON_R})", re.I)
# The longest range read, first day to last inclusive.
MAX_RANGE_DAYS = 31
# Which day a month word has (rule 2, "a month between two days"). A month
# word with a DAY CLAIM on each side is contested and gives no date at all
# ("on 5 March 3 people", "room 21 Jun 19-23", "15-16 June 5 km"); with a
# claim on one side only, that side is its day. Claims:
#   before: a standalone day ("5 ", "5th of ", "21 ") or the end of a number
#     range ("5-10 ", "3 to 21 "; only an ordered one: "won 3-1 May 5" and
#     "hours 9-5 May 1st" are a score and hours); not a time ("10:30"), money ("$20"), a
#     decimal ("1.5"), part of a code ("PR-12", "Q3"), a slash number, nor a
#     number that is itself the day of a month right before it ("May 5 June").
#   after: a day or the start of a range ("May 3", "May 5-10", "Sept. 21");
#     not a year, a time or a unit ("May 2 pm", "Oct 3:00", "June 5 km"),
#     a slash number, a percent; and never after a comma ("5 May, 3 people":
#     a comma after the month closes a day-first date).
# A left claim with an ordinal and "of" ("the 5th of May") is decisive:
# nothing after it is a claim ("10 of June 1" is still contested).
_RE_CLAIM_BEFORE = re.compile(
    r"(?:(?<![\w/.:$])(?P<a>\d{1,2})(?:st|nd|rd|th)?\s*(?:[-\u2013\u2014]|to)\s*"
    r"|(?<![\w/.:$\-\u2013\u2014]))"
    r"(?P<d>\d{1,2})(?P<ord>st|nd|rd|th)?\s+(?P<of>of\s+)?\Z", re.I)
_TIME_OR_UNIT = (r"(?:am|pm|a\.m\.|p\.m\.|o'?clock|h|hrs?|hours?|mins?|minutes?"
                 r"|km|mi|kg|lbs?)\b")
_RE_CLAIM_AFTER = re.compile(
    rf"\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?![\w/:%]|\.\d|\s*{_TIME_OR_UNIT})", re.I)
_RE_MONTH_WORD = re.compile(rf"\b{_MON}\b", re.I)
_RE_MONTH_BEFORE = re.compile(rf"\b{_MON}\b\.?\s+\Z", re.I)
_RE_MONTH_AFTER = re.compile(rf"\.?,?\s+(?:of\s+)?{_MON}\b", re.I)

_RE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'’.\-/:]*[A-Za-z0-9]|[A-Za-z0-9]")
_RE_ORDINAL_OR_UNIT = re.compile(
    r"^\d+(?:st|nd|rd|th|am|pm|k|m|s|x|ms|kb|mb|gb|hz|%)$|^\d{1,2}:\d{2}(?::\d{2})?(?:am|pm)?$",
    re.I)
# A bare a/b with b in this set and a <= b is a ratio or a score, not a date.
RATIO_DENOMS = (2, 3, 4, 5, 6, 8, 10, 20)
# Month names that are also ordinary words or first names. Read as a month only
# when capitalized, after a date preposition, or with an ordinal day / "of".
AMBIGUOUS_MONTHS = frozenset({"march", "mar", "may", "jan"})
DATE_PREPOSITIONS = frozenset({
    "on", "by", "since", "until", "till", "from", "before", "after", "during",
    "through", "between", "of", "in",
})
# The words that mark a ratio-shaped a/b as a date. "of", "from", "between",
# "during" and "in" mark a month word ("from may 5", "in march 2026") but not
# a slash pair: "a score of 8/10", "between 1/2 and 3/4 cup", "in 8/10 cases".
SLASH_DATE_PREPOSITIONS = frozenset({
    "on", "by", "before", "after", "since", "until", "till", "through",
})
# "by" after one of these is an amount, not a deadline: "reduced by 1/2".
FRACTION_VERBS = frozenset({
    "reduce", "reduced", "reduces", "increase", "increased", "increases",
    "decrease", "decreased", "decreases", "cut", "divide", "divided",
    "divides", "multiply", "multiplied", "multiplies", "scale", "scaled",
    "scales", "grow", "grows", "grew", "shrink", "shrinks", "shrank", "rose",
    "fell", "lowered", "raised",
})
# A taken date is blanked with this in the text the date rules read, so the
# text either side of it can never be joined into another date by "\s+".
_GONE = "\x00"
# After a ratio-shaped a/b: "of" ("3/4 of the book") or a hyphenated unit
# ("3/4-inch") makes it a fraction.
_RE_OF_NEXT = re.compile(r"\s+of\b|-[A-Za-z]", re.I)
# A contested month phrase is blanked with _CONTESTED. A blanked date or
# contested phrase counts as a word for "the word before": a date
# preposition marks the date right after it and nothing past it ("on Sept 21
# the 3/4 mark", "due on 9/21 the 1/2 payment", "on 5 may 5 4/5" read no 3/4,
# 1/2 or 4/5).
_CONTESTED = "\x01"
_RE_ALPHA_RUN = re.compile(r"[A-Za-z]+|[\x00\x01]+")


def _norm_ws(text: str) -> str:
    """Lowercase, collapse all whitespace runs to one space, trim."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def ascii_lower(text: str) -> str:
    """Lowercase ASCII letters only: what SQLite's lower() does, so a query
    candidate is folded exactly the way the indexed label expression is."""
    return "".join(c.lower() if c.isascii() else c for c in text)


def _sql_key(text: str) -> str:
    return ascii_lower(re.sub(r"\s+", " ", (text or "").strip()))


# ── configuration ─────────────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v == v and abs(v) != float("inf") else default


def exact_enabled_by_env() -> bool:
    return os.environ.get("REVIEN_EXACT", "").strip().lower() in (
        "on", "1", "true", "yes")


@dataclass
class ExactConfig:
    boost_max: float = BOOST_MAX
    max_df: int = MAX_DF
    date_max_matches: int = DATE_MAX_MATCHES
    max_inject: int = MAX_INJECT
    neighbor_max: int = NEIGHBOR_MAX
    bm25_label_scale: float = BM25_LABEL_SCALE
    scan_chars: int = SCAN_CHARS
    max_contained_codes: int = MAX_CONTAINED_CODES
    common_words: Set[str] = field(default_factory=lambda: set(COMMON_WORDS))
    framing_words: Set[str] = field(default_factory=lambda: set(FRAMING_WORDS))

    @staticmethod
    def _words_from_env(name: str, base: Set[str]) -> Set[str]:
        extra = os.environ.get(name, "")
        return set(base) | {
            w.strip().lower() for w in extra.split(",") if w.strip()}

    @classmethod
    def from_env(cls) -> "ExactConfig":
        return cls(
            boost_max=max(0.0, _env_float("REVIEN_EXACT_BOOST_MAX", BOOST_MAX)),
            max_df=max(0, int(_env_float("REVIEN_EXACT_MAX_DF", MAX_DF))),
            date_max_matches=max(1, int(_env_float(
                "REVIEN_EXACT_DATE_MAX_MATCHES", DATE_MAX_MATCHES))),
            max_inject=max(0, int(_env_float(
                "REVIEN_EXACT_MAX_INJECT", MAX_INJECT))),
            neighbor_max=max(1, int(_env_float(
                "REVIEN_EXACT_NEIGHBOR_MAX", NEIGHBOR_MAX))),
            bm25_label_scale=min(1.0, max(0.0, _env_float(
                "REVIEN_EXACT_BM25_LABEL_SCALE", BM25_LABEL_SCALE))),
            scan_chars=max(1, int(_env_float(
                "REVIEN_EXACT_SCAN_CHARS", SCAN_CHARS))),
            max_contained_codes=max(0, int(_env_float(
                "REVIEN_EXACT_MAX_CONTAINED_CODES", MAX_CONTAINED_CODES))),
            common_words=cls._words_from_env(
                "REVIEN_EXACT_COMMON_WORDS", COMMON_WORDS),
            framing_words=cls._words_from_env(
                "REVIEN_EXACT_FRAMING_WORDS", FRAMING_WORDS),
        )


# ── signals ───────────────────────────────────────────────────────────────────
@dataclass
class DateSignal:
    text: str
    granularity: str            # "day" | "month" (a range is "day")
    month: int
    day: Optional[int] = None   # None for a whole month
    year: Optional[int] = None  # None: matches any year
    # A range ("May 5-10", "9/21-9/23", "Dec 28 - Jan 3"): its last day, in
    # the start's year (end_year_offset 0) or the next one (1).
    end_month: Optional[int] = None
    end_day: Optional[int] = None
    end_year_offset: int = 0

    @property
    def is_range(self) -> bool:
        return self.end_day is not None

    def ranges_for_year(self, year: int) -> Optional[Tuple[str, str]]:
        """Inclusive ('YYYY-MM-DD', 'YYYY-MM-DD') for ``year`` (a range: its
        whole span, first day to last), or None when the date does not exist
        in that year (Feb 29)."""
        try:
            if self.is_range:
                return (date(year, self.month, self.day).isoformat(),
                        date(year + self.end_year_offset, self.end_month,
                             self.end_day).isoformat())
            if self.day is None:
                last = calendar.monthrange(year, self.month)[1]
                return (date(year, self.month, 1).isoformat(),
                        date(year, self.month, last).isoformat())
            d = date(year, self.month, self.day).isoformat()
            return (d, d)
        except ValueError:
            return None

    def describe(self) -> Dict[str, Any]:
        out = {"text": self.text, "granularity": self.granularity,
               "year": self.year, "month": self.month, "day": self.day,
               "any_year": self.year is None}
        if self.is_range:
            out["range_end"] = {
                "year": (None if self.year is None
                         else self.year + self.end_year_offset),
                "month": self.end_month, "day": self.end_day}
        return out


@dataclass
class NameSpan:
    text: str
    start: int                  # token index (inclusive)
    end: int                    # token index (exclusive)
    sentence_start: bool        # contains a sentence-initial token

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass
class ExactSignals:
    quoted: List[str] = field(default_factory=list)
    dates: List[DateSignal] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    spans: List[NameSpan] = field(default_factory=list)       # name candidates
    whole_query: Optional[str] = None
    # Every word left once dates and quotes are blanked, lowercased, trimmed,
    # possessive dropped; indexed like NameSpan.start/end.
    words: List[str] = field(default_factory=list)
    # Characters read when the query was longer than the scan bound (rule 5),
    # and the query's length; both None when the whole query was read.
    scanned_chars: Optional[int] = None
    query_chars: Optional[int] = None

    @property
    def truncated(self) -> bool:
        return self.scanned_chars is not None

    @property
    def needs_lookup(self) -> bool:
        return bool(self.quoted or self.ids or self.spans
                    or self.whole_query or self.dates)


@dataclass
class DateMatch:
    """What one date signal matched (ExactLane.match_dates)."""
    signal: DateSignal
    # (node_id, node's event_time granularity, strength), store order
    event: List[Tuple[str, str, float]]
    # (node_id, strength): trusted recorded_at, nodes event_time did not match
    recorded: List[Tuple[str, float]]
    # the DATE_MAX_MATCHES guard: min(1, date_max_matches / total)
    scale: float
    # more matches than the guard allows: boosts nodes that are already
    # candidates (scaled), admits none to the anchor set
    crowded: bool = False

    @property
    def total(self) -> int:
        return len(self.event) + len(self.recorded)


def _valid_md(m: int, d: int, year: Optional[int] = None) -> bool:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return False
    try:
        date(year if year is not None else 2000, m, d)  # 2000: leap year
        return True
    except ValueError:
        return False


def _blank(text: str, spans: List[Tuple[int, int]], ch: str) -> str:
    """``text`` with every (sorted, non-overlapping) span replaced by ``ch``."""
    parts, last = [], 0
    for s, e in spans:
        parts.append(text[last:s])
        parts.append(ch * (e - s))
        last = e
    parts.append(text[last:])
    return "".join(parts)


def _preceding(words: List[str]) -> Tuple[str, str]:
    """(the word before, the word before that), lowercase, skipping one
    "the" right before: "on the 3/10" -> ("on", ...)."""
    w = [x.lower() for x in words[-3:]]
    if w and w[-1] == "the":
        w = w[:-1]
    return (w[-1] if w else "", w[-2] if len(w) > 1 else "")


def _slash_marked_as_date(m, prev: str, prev2: str, after: str) -> bool:
    """A bare a/b the ratio rule would drop is still a date when it is
    zero-padded ("03/10"), or when it follows a SLASH_DATE_PREPOSITIONS word
    ("on 3/10"), is not a fraction of something or of a unit ("through 3/4
    of the book", "on 3/4-inch pipe"), and is not an amount after a verb of
    quantity ("reduced by 1/2")."""
    if any(len(g) == 2 and g[0] == "0" for g in (m[1], m[2])):
        return True
    if prev not in SLASH_DATE_PREPOSITIONS:
        return False
    if _RE_OF_NEXT.match(after):
        return False
    return not (prev == "by" and prev2 in FRACTION_VERBS)


def _month_ok(word: str, prev, strong: bool = False) -> bool:
    """May this month word be read as a month? Unambiguous names always;
    AMBIGUOUS_MONTHS (march, mar, may, jan) only when capitalized, after a
    date preposition, or when ``strong`` (an ordinal day or "of" in the
    match). ``prev`` returns the word before (only called when needed)."""
    if word.lower() not in AMBIGUOUS_MONTHS:
        return True
    if word[:1].isupper() or strong:
        return True
    return prev() in DATE_PREPOSITIONS


def _two_digit_year(y: int) -> int:
    return 2000 + y if y < 100 else y


def _range_signal(text: str, mo1: int, d1: int, y1: Optional[int],
                  mo2: int, d2: int, y2: Optional[int]) -> Optional[DateSignal]:
    """A range signal, or None when the range breaks the rules: both ends
    must be real dates, the end after the start, at most MAX_RANGE_DAYS days
    first to last. A year written once belongs to the end it is written at
    ("Sept 21-23, 2026"; "Dec 28 - Jan 3, 2027" starts in 2026); with no year
    the range is checked in a leap year and matches every year it exists in."""
    if y1 is not None and y2 is not None:
        cands = [(y1, y2)]
    elif y2 is not None:
        cands = [(y2, y2), (y2 - 1, y2)]
    elif y1 is not None:
        cands = [(y1, y1), (y1, y1 + 1)]
    else:
        cands = [(2000, 2000), (2000, 2001)]
    for ys, ye in cands:
        if ye - ys not in (0, 1):
            continue
        try:
            s, e = date(ys, mo1, d1), date(ye, mo2, d2)
        except ValueError:
            continue
        if s < e and (e - s).days < MAX_RANGE_DAYS:
            year = None if y1 is None and y2 is None else ys
            return DateSignal(text, "day", mo1, d1, year, end_month=mo2,
                              end_day=d2, end_year_offset=ye - ys)
    return None


def parse_dates(query: str) -> Tuple[List[DateSignal], str]:
    """Return (date signals, query with each date's characters blanked).

    One left-to-right pass per pattern, in a fixed order (ISO, M/D/Y, day
    month year, month day year, month day, day month, M/D, month year): each
    pass reads the text left by the passes before it, and every date it takes
    is blanked once, at the end of the pass. Linear in the length of
    ``query``."""
    work = query      # what the patterns read; a taken date becomes _GONE
    shown = query     # what is returned; a taken date becomes spaces
    out: List[DateSignal] = []

    def each(rx, fn):
        nonlocal work, shown
        taken: List[Tuple[int, int]] = []

        def before(pos: int) -> Tuple[str, str]:
            # The words before ``pos`` as the text reads with this pass's
            # earlier dates blanked (bounded window: at most 40 characters).
            lo = max(0, pos - 40)
            window = work[lo:pos]
            k = len(taken) - 1
            if k >= 0 and taken[k][1] > lo:
                chars = list(window)
                while k >= 0 and taken[k][1] > lo:
                    s, e = taken[k]
                    for i in range(max(s, lo), min(e, pos)):
                        chars[i - lo] = _GONE
                    k -= 1
                window = "".join(chars)
            return _preceding(_RE_ALPHA_RUN.findall(window))

        def window(pos: int, n: int = 16) -> str:
            # the n characters before ``pos``, this pass's earlier dates blanked
            lo = max(0, pos - n)
            w = work[lo:pos]
            k = len(taken) - 1
            if k >= 0 and taken[k][1] > lo:
                chars = list(w)
                while k >= 0 and taken[k][1] > lo:
                    s, e = taken[k]
                    for i in range(max(s, lo), min(e, pos)):
                        chars[i - lo] = _GONE
                    k -= 1
                w = "".join(chars)
            return w

        before.window = window
        for m in rx.finditer(work):
            sig = fn(m, before)
            if sig is not None:
                out.append(sig)
                taken.append(m.span())
        if taken:
            work = _blank(work, taken, _GONE)
            shown = _blank(shown, taken, " ")

    def iso(m, before):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def slash_y(m, before):
        a, b, y = int(m[1]), int(m[2]), _two_digit_year(int(m[3]))
        mo, d = (b, a) if a > 12 and b <= 12 else (a, b)
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def strong_dm(m) -> bool:
        return bool(re.search(r"^\d{1,2}(?:st|nd|rd|th)|\bof\b", m[0], re.I))

    # A month word with a day on each side ("on 5 March 3 people", "page 3
    # March 5") is ambiguous: neither pairing is read (rule 2, edges).
    def dmy(m, before):
        if not _month_ok(m[2], lambda: before(m.start())[0], strong_dm(m)):
            return None
        if month_before(m, before):
            return None
        d, mo, y = int(m[1]), _MONTH_NAMES[m[2].lower()], int(m[3])
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def mdy(m, before):
        if not _month_ok(m[1], lambda: before(m.start())[0]):
            return None
        if day_before(m, before):
            return None
        mo, d, y = _MONTH_NAMES[m[1].lower()], int(m[2]), int(m[3])
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def md(m, before):
        if not _month_ok(m[1], lambda: before(m.start())[0]):
            return None
        if day_before(m, before) or month_word_after(m):
            return None
        mo, d = _MONTH_NAMES[m[1].lower()], int(m[2])
        return DateSignal(m[0], "day", mo, d) if _valid_md(mo, d) else None

    def dm(m, before):
        if not _month_ok(m[2], lambda: before(m.start())[0], strong_dm(m)):
            return None
        if month_before(m, before) or (own_day_after(m) and not _of(m)):
            return None
        d, mo = int(m[1]), _MONTH_NAMES[m[2].lower()]
        return DateSignal(m[0], "day", mo, d) if _valid_md(mo, d) else None

    def slash(m, before):
        a, b = int(m[1]), int(m[2])
        if a <= b and b in RATIO_DENOMS:
            prev, prev2 = before(m.start())
            if not _slash_marked_as_date(
                    m, prev, prev2, work[m.end():m.end() + 40]):
                return None      # a ratio or a score ("3/4", "8/10"), not a date
        # month/day only: with no year, "24/7" and "16/9" have no month
        return DateSignal(m[0], "day", a, b) if _valid_md(a, b) else None

    def my(m, before):
        if not _month_ok(m[1], lambda: before(m.start())[0]):
            return None
        if number_before(m, before):  # "31 Sept 2026": a day that is no date
            return None
        return DateSignal(m[0], "month", _MONTH_NAMES[m[1].lower()], None,
                          int(m[2]))

    def range_of(m, mo1, d1, y1, mo2, d2, y2):
        # A range that breaks the rules is not taken: its parts are read
        # only as the single-date rules (and their edges) allow.
        return _range_signal(m[0], mo1, d1, y1, mo2, d2, y2)

    def claim_before(text: str, free_month_end: int = -1,
                     any_month: bool = False) -> Tuple[bool, bool]:
        """(``text`` ends in a day claim for a month word after it, the claim
        is decisive: an ordinal and "of"). A number that is the (possible)
        day of a month right before it ("May 5 June") claims nothing for the
        next one, unless that month is contested itself (its month word
        ends at ``free_month_end`` in ``text``): then the number is the next
        month's claim too."""
        c = _RE_CLAIM_BEFORE.search(text)
        if not c:
            return False, False
        if c["a"] is not None and not (
                1 <= int(c["a"]) < int(c["d"]) <= 31
                and int(c["d"]) - int(c["a"]) < MAX_RANGE_DAYS):
            return False, False          # "3-1", "9-5": not a range of days
        b = _RE_MONTH_BEFORE.search(text, 0, c.start())
        if (not any_month and b and is_month_word(b[1], text[:b.start(1)])
                and b.end(1) != free_month_end
                and _valid_md(_MONTH_NAMES[b[1].lower()], int(c["a"] or c["d"]))):
            return False, False
        return True, bool(c["of"] and c["ord"])

    def claim_after(pos: int) -> bool:
        """A day claim right after a month word that ends at ``pos``."""
        return bool(_RE_CLAIM_AFTER.match(work, pos))

    def contested() -> None:
        """Blank, in the text the date patterns read, every month word with
        a day claim on each side, and both claims: no reading may use the
        month or either number ("28 Sept 10 June" reads neither Sept 10
        nor 10 June)."""
        nonlocal work
        spans = []
        last_contested_end = -1          # where the last contested month word ends
        for mm in _RE_MONTH_WORD.finditer(work):
            s, e = mm.span()
            lo = max(0, s - 40)
            text_before = work[lo:s]
            if not is_month_word(mm[1], text_before):
                continue
            left, strong = claim_before(text_before, last_contested_end - lo)
            r = _RE_CLAIM_AFTER.match(work, e) if left and not strong else None
            if r:
                c = _RE_CLAIM_BEFORE.search(text_before)
                spans.append((lo + c.start(), r.end()))
                last_contested_end = e
        if spans:
            merged: List[Tuple[int, int]] = []
            for a, b in sorted(spans):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            work = _blank(work, merged, _CONTESTED)

    def _of(m) -> bool:
        """An ordinal day and "of" decide ("the 5th of May 2 tickets")."""
        return bool(re.search(r"\d(?:st|nd|rd|th)\s+of\b", m[0], re.I))

    def day_before(m, before) -> bool:
        """A day claim right before this match's month word."""
        return claim_before(before.window(m.start(), 40))[0]

    def number_before(m, before) -> bool:
        """Any day-like number right before this match's month word, even one
        that a month before it may own ("Sept 31 May 2026")."""
        return claim_before(before.window(m.start(), 40), any_month=True)[0]

    def is_month_word(word: str, text_before: str, strong: bool = True) -> bool:
        """The _month_ok rule for a month word next to a match: "may" /
        "march" in lower case are words unless a date preposition comes
        right before them ("of march 3") or (``strong``) an ordinal day or
        "of" does ("the 5th of may", "3rd march")."""
        if word.lower() not in AMBIGUOUS_MONTHS or word[:1].isupper():
            return True
        if strong and re.search(r"\d(?:st|nd|rd|th)\s+(?:of\s+)?\Z|\bof\s+\Z",
                                text_before, re.I):
            return True
        return _preceding(_RE_ALPHA_RUN.findall(text_before))[0] in DATE_PREPOSITIONS

    def month_before(m, before) -> bool:
        """A month word right before this match's first day."""
        w = before.window(m.start(), 40)
        b = _RE_MONTH_BEFORE.search(w)
        return bool(b) and is_month_word(b[1], w[:b.start(1)])

    def own_day_after(m) -> bool:
        """This match ends in a month word that has a day claim after it."""
        return claim_after(m.end())

    def month_word_after(m) -> bool:
        """Is the range's end day followed by a month that would claim it
        ("May 5 - 10 June")? Not when that month has a day of its own
        ("Sept 21-23 May 5"), and not for "may" / "march" as words."""
        a = _RE_MONTH_AFTER.match(work, m.end())
        if not a:
            return False
        # the ordinal before it is this match's own ("June 1st may be")
        if not is_month_word(a[1], work[max(0, a.start(1) - 40):a.start(1)],
                             strong=False):
            return False
        return not claim_after(a.end())

    def iso_range(m, before):
        return range_of(m, int(m["m1"]), int(m["d1"]), int(m["y1"]),
                        int(m["m2"]), int(m["d2"]), int(m["y2"]))

    def slash_range(m, before):
        a1, b1, a2, b2 = (int(m[k]) for k in ("m1", "d1", "m2", "d2"))
        y1 = _two_digit_year(int(m["y1"])) if m["y1"] else None
        y2 = _two_digit_year(int(m["y2"])) if m["y2"] else None
        if y1 is None and a1 <= b1 and b1 in RATIO_DENOMS:
            # the ratio rule, on the start: "1/2-3/4 cup" is two fractions
            prev, prev2 = before(m.start())
            if not _slash_marked_as_date(
                    m, prev, prev2, work[m.end():m.end() + 40]):
                return None
        # day-first only with a year, as for one date
        mo1, d1 = (b1, a1) if y1 is not None and a1 > 12 and b1 <= 12 else (a1, b1)
        mo2, d2 = (b2, a2) if y2 is not None and a2 > 12 and b2 <= 12 else (a2, b2)
        return range_of(m, mo1, d1, y1, mo2, d2, y2)

    def md_range(m, before):
        if not _month_ok(m["m1"], lambda: before(m.start())[0]):
            return None
        # the month belongs to a day before it ("3 May 5-10"), or the end
        # day to a month after it ("May 5 - 10 June")
        if day_before(m, before):
            return None
        if not m["y2"] and month_word_after(m):
            return None
        mo1 = _MONTH_NAMES[m["m1"].lower()]
        mo2 = _MONTH_NAMES[m["m2"].lower()] if m["m2"] else mo1
        y1 = int(m["y1"]) if "y1" in m.re.groupindex and m["y1"] else None
        y2 = int(m["y2"]) if m["y2"] else None
        return range_of(m, mo1, int(m["d1"]), y1, mo2, int(m["d2"]), y2)

    def dm_range(m, before):
        strong = bool(re.search(r"\d(?:st|nd|rd|th)\b|\bof\b", m[0], re.I))
        if not _month_ok(m["m2"], lambda: before(m.start())[0], strong):
            return None
        # the first day belongs to a month before it ("May 10-11 June");
        # "may" / "march" in lower case are words; the month has a day of
        # its own after it ("10-11 Sept 21")
        if month_before(m, before) or (
                not m["y2"] and own_day_after(m) and not _of(m)):
            return None
        mo2 = _MONTH_NAMES[m["m2"].lower()]
        mo1 = _MONTH_NAMES[m["m1"].lower()] if m["m1"] else mo2
        y1 = int(m["y1"]) if m["y1"] else None
        y2 = int(m["y2"]) if m["y2"] else None
        return range_of(m, mo1, int(m["d1"]), y1, mo2, int(m["d2"]), y2)

    # Ranges first: a range is read whole, or not as a range (rule 2). The
    # forms with no month word go first; then contested month words are
    # blanked; then the month-name forms.
    each(_RE_ISO_RANGE, iso_range)
    each(_RE_SLASH_RANGE, slash_range)
    each(_RE_ISO, iso)
    each(_RE_SLASH_Y, slash_y)
    contested()
    each(_RE_MDY_RANGE, md_range)
    each(_RE_MD_RANGE, md_range)
    each(_RE_DM_RANGE, dm_range)
    each(_RE_DMY, dmy)
    each(_RE_MDY, mdy)
    each(_RE_MD, md)
    each(_RE_DM, dm)
    each(_RE_SLASH, slash)
    each(_RE_MY, my)
    return out, shown


def _is_capitalized(tok: str) -> bool:
    return bool(tok) and tok[0].isalpha() and tok[0].isupper()


_TRIM = ".,;:!?)('\"’-/"
_RE_POSSESSIVE = re.compile(r"['’][sS]$")


def _strip_token(tok: str) -> str:
    """Trim punctuation and drop a possessive: "Dana's" -> "Dana"."""
    t = tok.strip(_TRIM)
    t = _RE_POSSESSIVE.sub("", t)
    return t.strip(_TRIM)


def _is_possessive(tok: str) -> bool:
    return bool(_RE_POSSESSIVE.search(tok.strip(_TRIM)))


# Separators that, with digits alone, make a range, a score or a fraction
# ("5-10", "21-23", "9/21-9/23", "1-3/4"), never a code.
_NUMERIC_SEPARATORS = frozenset("/-\u2013\u2014")
# ... also with a time or unit after it ("5-6pm", "10:30-11:30", "10-20%").
_RE_NUMERIC_RANGE_TOKEN = re.compile(
    r"[\d/:\-\u2013\u2014]+(?:st|nd|rd|th|am|pm|k|m|s|x|ms|kb|mb|gb|hz|%)?", re.I)


def _is_id_token(tok: str) -> bool:
    t = _strip_token(tok)
    if len(t) < 3 or _RE_ORDINAL_OR_UNIT.match(t):
        return False
    if not t.isdigit() and (
            all(c.isdigit() or c in _NUMERIC_SEPARATORS for c in t)
            or (_RE_NUMERIC_RANGE_TOKEN.fullmatch(t)
                and any(c in _NUMERIC_SEPARATORS or c == ":" for c in t))):
        return False
    has_digit = any(c.isdigit() for c in t)
    has_alpha = any(c.isalpha() for c in t)
    if has_digit and has_alpha:
        return True
    if has_digit and t.isdigit():
        return len(t) >= 5
    if "_" in t and re.search(r"[A-Za-z0-9]_[A-Za-z0-9]", t):
        return True
    if "-" in t:
        parts = [p for p in t.split("-") if p]
        if len(parts) >= 2 and any(
                (p.isalpha() and len(p) >= 2 and p.isupper())
                or any(c.isdigit() for c in p) for p in parts):
            return True
    return False


_WS = " \t\n\r\f\v"


def _scan_ends(text: str, max_chars: int) -> Tuple[str, str]:
    """(head, tail) of a query longer than ``max_chars`` (rule 5): the first
    half of the budget from the start, the rest from the end. Neither cut
    splits a token (it moves to whitespace; a part that is one split token
    is dropped) or a quoted phrase: the lane's own quote rule (_RE_QUOTE) is
    run over the whole query, and a phrase the cut would land inside is left
    out of the part, so the text between two quotes can never be read as
    one. That is the lane's pairing, not a count of quote marks (a stray
    inch mark would flip a count for the rest of the query). Linear."""
    h = max_chars // 2
    quotes = [m.span() for m in _RE_QUOTE.finditer(text)]
    starts = [s for s, _e in quotes]

    def inside(pos: int) -> Optional[Tuple[int, int]]:
        i = bisect.bisect_left(starts, pos) - 1
        if i >= 0 and quotes[i][0] < pos < quotes[i][1]:
            return quotes[i]
        return None

    def splits(pos: int) -> bool:
        return (0 < pos < len(text) and not text[pos].isspace()
                and not text[pos - 1].isspace())

    end = h                         # head = text[:end]; only moves left
    while True:
        if splits(end):
            end = max(max(text.rfind(c, 0, end) for c in _WS), 0)
        elif inside(end):
            end = inside(end)[0]
        else:
            break
    start = len(text) - (max_chars - h)   # tail = text[start:]; only moves right
    while True:
        if splits(start):
            ks = [k for k in (text.find(c, start) for c in _WS) if k >= 0]
            start = min(ks) + 1 if ks else len(text)
        elif inside(start):
            start = inside(start)[1]
        else:
            break
    return text[:end], text[start:]


_RE_EDGE_JUNK = re.compile(r"[\W_]*")


def _trim_edges(text: str) -> str:
    """``text`` without its leading and trailing non-alphanumeric runs: what
    ``re.sub(r"^[\\W_]+|[\\W_]+$", "", text)`` returns, in linear time (that
    pattern retries a long inner run of punctuation from every position)."""
    lead = _RE_EDGE_JUNK.match(text).end()
    if lead == len(text):
        return ""
    trail = _RE_EDGE_JUNK.match(text[::-1]).end()
    return text[lead:len(text) - trail]


def parse_signals(query: str, common_words: Optional[Set[str]] = None,
                  max_chars: int = SCAN_CHARS) -> ExactSignals:
    """Pure, deterministic detection (rules in the module docstring).
    ``common_words`` is the set a sentence-initial single word is dropped on
    (default COMMON_WORDS; the lane passes its configured set). At most
    ``max_chars`` characters are read: a longer query is read at its start
    and its end, half the budget each (rule 5; the lane passes its
    configured bound)."""
    common = COMMON_WORDS if common_words is None else common_words
    text = query or ""
    if len(text) <= max_chars:
        sig = _parse_text(text, common)
        # Whole-query label (lowercase single tokens like "harbor"); only
        # when the whole query was read.
        whole = re.sub(r"\s+", " ", _trim_edges(text)).strip()
        whole = _RE_POSSESSIVE.sub("", whole)
        if whole and len(whole.split()) <= MAX_SPAN_TOKENS:
            sig.whole_query = whole
        return sig
    # Head and tail are read apart, so nothing in one can join anything in
    # the other (a quote, a date, a name); the tail's first token counts as
    # the start of a sentence.
    head, tail = _scan_ends(text, max_chars)
    sig = _parse_text(head, common)
    rest = _parse_text(tail, common)
    off = len(sig.words)
    sig.quoted += rest.quoted
    sig.dates += rest.dates
    seen = {x.lower() for x in sig.ids}
    sig.ids += [x for x in rest.ids if x.lower() not in seen]
    sig.spans += [NameSpan(s.text, s.start + off, s.end + off, s.sentence_start)
                  for s in rest.spans]
    sig.words += rest.words
    sig.scanned_chars = len(head) + len(tail)
    sig.query_chars = len(text)
    return sig


def _parse_text(work: str, common: Set[str]) -> ExactSignals:
    """Quotes, dates, codes and names in ``work`` (no bound, no whole-query
    label: parse_signals adds those)."""
    sig = ExactSignals()
    for m in _RE_QUOTE.finditer(work):
        phrase = next(g for g in m.groups() if g is not None)
        if phrase.strip():
            sig.quoted.append(re.sub(r"\s+", " ", phrase.strip()))
    work = _RE_QUOTE.sub(lambda m: " " * len(m.group(0)), work)

    sig.dates, work = parse_dates(work)

    # Tokenize what is left (dates and quotes blanked).
    tokens = [(m.group(0), m.start(), m.end()) for m in _RE_TOKEN.finditer(work)]
    n = len(tokens)
    sig.words = [_strip_token(t[0]).lower() for t in tokens]
    sentence_start = [False] * n
    for i, (_, s, _e) in enumerate(tokens):
        if i == 0:
            sentence_start[i] = True
        else:
            gap = work[tokens[i - 1][2]:s]
            sentence_start[i] = bool(re.search(r"[.!?:]\s*$|\n", gap)) or (
                gap.strip() != "" and gap.strip()[-1:] in ".!?:")

    # IDs / codes.
    seen_ids: Set[str] = set()
    for tok, _s, _e in tokens:
        if _is_id_token(tok):
            t = _strip_token(tok)
            if t.lower() not in seen_ids:
                seen_ids.add(t.lower())
                sig.ids.append(t)

    # Names: runs of capitalized tokens.
    def cap_ok(i: int) -> bool:
        tok = _strip_token(tokens[i][0])
        if not _is_capitalized(tok) or len(tok) < 2 and tok != "I":
            return False
        low = tok.lower()
        if low in ("i", "im", "ive", "id", "ill") or low.startswith(("i'", "i’")):
            return False
        return True

    i = 0
    while i < n:
        if not cap_ok(i) or _is_id_token(tokens[i][0]):
            i += 1
            continue
        j = i + 1
        # A possessive ends a name: "Dana's Project Kestrel" is two names.
        while j < n and not _is_possessive(tokens[j - 1][0]):
            if cap_ok(j) and not _is_id_token(tokens[j][0]):
                j += 1
            elif (tokens[j][0].lower() in _JOINERS and j + 1 < n
                  and cap_ok(j + 1) and not _is_id_token(tokens[j + 1][0])):
                j += 2
            else:
                break
        # Spans over tokens[i:j], capitalized ends only, longest first later.
        for a in range(i, j):
            if not cap_ok(a):
                continue
            for b in range(a + 1, min(j, a + MAX_SPAN_TOKENS) + 1):
                if not cap_ok(b - 1):
                    continue
                words = [_strip_token(t[0]) for t in tokens[a:b]]
                text = " ".join(words)
                starts = any(sentence_start[k] for k in range(a, b))
                if starts and b - a == 1 and words[0].lower() in common:
                    continue
                sig.spans.append(NameSpan(text, a, b, starts))
        i = j
    return sig


# ── lane ──────────────────────────────────────────────────────────────────────
@dataclass
class ExactResult:
    detected: Dict[str, Any]
    matches: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    boost: Dict[str, float] = field(default_factory=dict)
    inject_ids: List[str] = field(default_factory=list)
    # Filled by the engine when the boost is re-applied after the reranker:
    # {node_id: (position before, position after)} for nodes that moved.
    rerank_moves: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "detected": self.detected,
            "matches": {nid: list(ms) for nid, ms in self.matches.items()},
            "boost": dict(self.boost),
            "injected": list(self.inject_ids),
            "after_rerank": {nid: list(mv) for nid, mv in self.rerank_moves.items()},
        }


def lift_after_rerank(
    results: list, head_n: int, boosts: Dict[str, float],
) -> Tuple[list, Dict[str, Tuple[int, int]]]:
    """Re-apply the exact boost to a reranked head, on the engine's scale.

    The cross-encoder re-sorts its head by a score on its own scale, so the
    boost added to ``.score`` before reranking cannot act inside the head. Here
    the head keeps the reranker's ORDER; the head's own pre-boost scores
    (``score`` minus the boost) are dealt back out down that order, in
    descending value; each node's boost is added to its slot; the head is
    re-sorted by that (ties keep the reranker's order). A node therefore rises
    past a neighbour only if the neighbour's slot is less than ITS boost
    (never more than the cap) above its own, and unboosted nodes never change
    order among themselves. ``.score`` itself is not touched.

    Returns (results, {node_id: (position before, position after)}). A head
    nothing in it was reranked, or nothing in it is boosted, comes back as is.
    """
    head = results[:head_n]
    if not head or not any(r.node_id in boosts for r in head):
        return results, {}
    if not any("rerank_score" in r.score_breakdown for r in head):
        return results, {}
    slots = sorted((r.score - boosts.get(r.node_id, 0.0) for r in head),
                   reverse=True)
    keyed = sorted(
        ((slots[i] + boosts.get(r.node_id, 0.0), i, r) for i, r in enumerate(head)),
        key=lambda t: (-t[0], t[1]))
    moves = {r.node_id: (i, new) for new, (_k, i, r) in enumerate(keyed)
             if r.node_id in boosts and i != new}
    return [r for _k, _i, r in keyed] + results[head_n:], moves


class ExactLane:
    """Parses a query, looks the exact things up in the store (indexed SQL
    only), and returns per-node boosts. Holds no state between calls."""

    def __init__(self, store, config: Optional[ExactConfig] = None):
        self.store = store
        self.config = config or ExactConfig.from_env()

    # -- guard ---------------------------------------------------------------
    def _label_weight(self, label_norm: str, df: int,
                      quoted: bool = False) -> Tuple[float, Optional[str]]:
        words = label_norm.split()
        if not quoted and words and all(
                w in self.config.common_words for w in words):
            return 0.0, "common_words"
        if df > self.config.max_df:
            return 0.0, "shared_by_%d_nodes" % df
        return 1.0 / max(df, 1), None

    def _neighbor_scale(self, degree: int) -> Tuple[float, Optional[str]]:
        cap = self.config.neighbor_max
        if degree > HUB_FACTOR * cap:
            return 0.0, "hub_%d_neighbors" % degree
        return min(1.0, cap / max(degree, 1)), None

    # -- detection, and the date rule on its own ---------------------------
    def parse(self, query: str) -> ExactSignals:
        """``parse_signals`` with this lane's configured word list and scan
        bound: exactly what ``build`` reads from a query."""
        return parse_signals(query, self.config.common_words,
                             self.config.scan_chars)

    def match_dates(
        self,
        dates: List[DateSignal],
        origin_runtime: Optional[List[str]] = None,
        include_context: bool = False,
        live_only: bool = True,
    ) -> List[DateMatch]:
        """The lane's whole date rule, one DateMatch per date signal, in order.

        A date with a year is that day (or month); one without is that day in
        every year the graph has a dated node for (either field). Nodes are
        matched on event_time first (overlap with the date, strength
        min(node granularity, query granularity) from GRAN_WEIGHT), then on
        recorded_at (trusted sources only, RECORDED_AT_STRENGTH times the
        query granularity) for nodes event_time did not match; never on
        created_at or text. ``build`` boosts from this, and anything that
        reports what the lane can see by date (revien_bench.ab_exact) calls
        it too, so the two cannot disagree."""
        if not dates:
            return []
        scope = dict(origin_runtime=origin_runtime,
                     include_context=include_context, live_only=live_only)
        years: Set[int] = set(self.store.recorded_year_list())
        for a, b in self.store.event_time_year_spans():
            if b >= a and b - a <= MAX_YEARS:
                years.update(range(a, b + 1))
        out: List[DateMatch] = []
        for d in dates:
            if d.year is not None:
                ranges = [r for r in [d.ranges_for_year(d.year)] if r]
            else:
                ranges = [r for y in sorted(years)
                          for r in [d.ranges_for_year(y)] if r]
            ev_rows = (self.store.exact_event_matches(ranges, **scope)
                       if ranges else [])
            rec_rows = (self.store.exact_recorded_matches(ranges, **scope)
                        if ranges else [])
            # event_time is the strongest source; recorded_at (when it was
            # SAID) counts for a node only if event_time did not.
            ev_ids = {r[0] for r in ev_rows}
            rec_rows = [r for r in rec_rows if r[0] not in ev_ids]
            total = len(ev_rows) + len(rec_rows)
            event = []
            for nid, _ntype, _s, _e, gran in ev_rows:
                g = (gran or "fuzzy")
                event.append((nid, g, min(GRAN_WEIGHT.get(g, GRAN_WEIGHT["fuzzy"]),
                                          GRAN_WEIGHT[d.granularity])))
            recorded = [(nid, RECORDED_AT_STRENGTH * GRAN_WEIGHT[d.granularity])
                        for nid, _ntype, _rec in rec_rows]
            out.append(DateMatch(
                signal=d, event=event, recorded=recorded,
                scale=(min(1.0, self.config.date_max_matches / total)
                       if total else 0.0),
                crowded=total > self.config.date_max_matches))
        return out

    # -- main ----------------------------------------------------------------
    def build(
        self,
        query: str,
        anchor_ids: List[str],
        origin_runtime: Optional[List[str]] = None,
        include_context: bool = False,
        live_only: bool = True,
        entity_anchor_ids: Optional[List[str]] = None,
        label_scale: float = 1.0,
    ) -> Optional[ExactResult]:
        """None when the query carries no exact signal (the caller then does
        nothing and recall is byte-identical to the lane being off), except
        for a query longer than the scan bound: that returns a result with
        no boost and no injection, whose diagnostics report the cut (results
        and scores are then the same as with the lane off; only
        diagnostics["exact"] is added).

        ``label_scale`` multiplies every boost that comes from label text
        (name, quote, id, whole-query); the engine passes
        ``config.bm25_label_scale`` when the BM25 lane, which indexes labels,
        is on, and 1.0 otherwise."""
        cfg = self.config
        sig = self.parse(query)
        if not sig.needs_lookup and not sig.truncated:
            return None
        ls = label_scale
        scope = dict(origin_runtime=origin_runtime,
                     include_context=include_context, live_only=live_only)

        # ---- candidate label keys -------------------------------------------
        # key (sql-folded, ws-normalized) -> list of (kind, display, strength)
        wanted: Dict[str, List[Tuple[str, str, float]]] = {}

        def want(text: str, kind: str, strength: float):
            k = _sql_key(text)
            if k:
                wanted.setdefault(k, []).append((kind, text, strength))

        for q in sig.quoted:
            want(q, "quoted", 1.0)
        for s in sig.spans:
            want(s.text, "name", 1.0 if s.n_tokens > 1 else SINGLE_NAME_STRENGTH)
        for t in sig.ids:
            want(t, "id", 1.0)
        if sig.whole_query:
            want(sig.whole_query, "whole_query",
                 1.0 if len(sig.whole_query.split()) > 1 else SINGLE_NAME_STRENGTH)

        label_rows = self.store.exact_label_matches(list(wanted), **scope) if wanted else []
        by_key: Dict[str, List[Tuple[str, str]]] = {}
        for nid, low, ntype in label_rows:
            by_key.setdefault(low, []).append((nid, ntype))

        matches: Dict[str, List[Dict[str, Any]]] = {}
        boost_parts: Dict[str, float] = {}
        admit: Set[str] = set()      # nodes with a signal that may be injected

        def add(nid: str, entry: Dict[str, Any], contrib: float,
                admissible: bool = True):
            matches.setdefault(nid, []).append(entry)
            boost_parts[nid] = boost_parts.get(nid, 0.0) + contrib
            if admissible and contrib > 0.0:
                admit.add(nid)

        # ---- name spans: longest first, no token reused ------------------------
        used_tokens: Set[int] = set()
        accepted_names: List[Dict[str, Any]] = []
        unmatched_names: List[str] = []
        accepted: List[Tuple[NameSpan, List[Tuple[str, str]], float, Optional[str], float]] = []
        for s in sorted(sig.spans, key=lambda x: (-x.n_tokens, x.start)):
            key = _sql_key(s.text)
            hits = by_key.get(key, [])
            toks = set(range(s.start, s.end))
            if hits and not (toks & used_tokens):
                used_tokens |= toks
                strength = 1.0 if s.n_tokens > 1 else SINGLE_NAME_STRENGTH
                weight, why = self._label_weight(_norm_ws(s.text), len(hits))
                accepted.append((s, hits, strength, why, weight))

        # Is the query ABOUT the entity, or does it ask something of it? About:
        # nothing is left once the names are set aside but framing words
        # ("who is X", "tell me about X"). Anything else (a topic word, an
        # unmatched capitalized name, a date, a quote, a code) is a question
        # that merely names the entity.
        about_entity = True
        if accepted:
            rest = [w for i, w in enumerate(sig.words) if i not in used_tokens]
            about_entity = not (
                sig.dates or sig.quoted or sig.ids
                or any(w and w not in cfg.framing_words for w in rest))

        topic_entities: Dict[str, Tuple[float, float, str]] = {}
        for s, hits, strength, why, weight in accepted:
            rec = {"text": s.text, "matched": [h[0] for h in hits],
                   "weight": weight, "guard": why,
                   "mode": "entity" if about_entity else "topic"}
            accepted_names.append(rec)
            for nid, _t in hits:
                ent = {"signal": "name", "text": s.text, "strength": strength,
                       "weight": weight, "mode": rec["mode"]}
                if why:
                    ent["guard"] = why
                if about_entity:
                    if ls != 1.0:
                        ent["label_scale"] = ls
                    add(nid, ent, cfg.boost_max * strength * weight * ls)
                else:
                    # The entity node itself is not lifted: it would outrank
                    # the node that answers. Its neighbours are, below.
                    add(nid, ent, 0.0)
                    if weight > 0.0:
                        old = topic_entities.get(nid)
                        if old is None or strength * weight > old[0] * old[1]:
                            topic_entities[nid] = (strength, weight, s.text)
        # Unmatched name spans that were eligible without a label: maximal only.
        covered: Set[int] = set(used_tokens)
        for s in sorted(sig.spans, key=lambda x: (-x.n_tokens, x.start)):
            if s.sentence_start:
                continue
            toks = set(range(s.start, s.end))
            if not (toks & covered):
                covered |= toks
                unmatched_names.append(s.text)

        # ---- neighbours of a named entity (topic questions) --------------------
        if topic_entities:
            neigh = self.store.exact_neighbors(sorted(topic_entities), **scope)
            for ent_id in sorted(topic_entities):
                strength, weight, text = topic_entities[ent_id]
                ns = neigh.get(ent_id, [])
                scale, why = self._neighbor_scale(len(ns))
                for rec in accepted_names:
                    if ent_id in rec["matched"]:
                        rec["neighbors"] = len(ns)
                        rec["neighbor_scale"] = round(scale, 4)
                        if why:
                            rec["neighbor_guard"] = why
                if scale <= 0.0:
                    continue
                for nid in ns:
                    ent = {"signal": "name_neighbor", "text": text,
                           "entity": ent_id,
                           "strength": strength * NEIGHBOR_STRENGTH,
                           "weight": weight * scale}
                    add(nid, ent,
                        cfg.boost_max * strength * NEIGHBOR_STRENGTH * weight * scale,
                        admissible=False)

        # ---- quoted phrases ------------------------------------------------------
        quoted_report = []
        for q in sig.quoted:
            hits = by_key.get(_sql_key(q), [])
            weight, why = (self._label_weight(_norm_ws(q), len(hits), quoted=True)
                           if hits else (0.0, None))
            quoted_report.append({"text": q, "matched": [h[0] for h in hits],
                                  "weight": weight, "guard": why})
            for nid, _t in hits:
                ent = {"signal": "quoted", "text": q, "strength": 1.0,
                       "weight": weight}
                if why:
                    ent["guard"] = why
                if ls != 1.0:
                    ent["label_scale"] = ls
                add(nid, ent, cfg.boost_max * weight * ls)

        # ---- ids and codes ----------------------------------------------------------
        ids_report = []
        if sig.ids:
            id_rows = self.store.exact_node_ids(sig.ids, **scope)
            id_hit = {r[0] for r in id_rows}
            # Codes of 4+ characters are also looked for INSIDE labels, at
            # most max_contained_codes of them, the most distinctive first:
            # longest first (a longer code is contained in fewer labels),
            # ties in query order. The rest are reported as skipped.
            long_ids = [t for t in sig.ids if len(t) >= 4]
            ranked = sorted(range(len(long_ids)),
                            key=lambda i: (-len(long_ids[i]), i))
            searched = [long_ids[i] for i in ranked[:cfg.max_contained_codes]]
            skipped = [long_ids[i] for i in ranked[cfg.max_contained_codes:]]
            contain_set, skipped_set = set(searched), set(skipped)
            # ONE pass over the label column for every searched code.
            contain_rows = (
                self.store.exact_label_contains(
                    [ascii_lower(t) for t in searched], **scope)
                if searched else [])
            for t in sig.ids:
                label_equal: Set[str] = {nid for nid, _t in by_key.get(_sql_key(t), [])}
                node_id_equal: Set[str] = {nid for nid in id_hit if nid == t}
                equal = label_equal | node_id_equal
                contained: Set[str] = set()
                if t in contain_set:
                    rx = re.compile(
                        r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])",
                        re.I)
                    for nid, label, _ty in contain_rows:
                        if nid not in equal and rx.search(label):
                            contained.add(nid)
                # Document frequency is taken per class: an exact label hit
                # is not diluted by the labels that merely contain the code.
                rep_matched = []
                for cls, found, strength in (
                        ("equal", equal, 1.0),
                        ("contained", contained, ID_CONTAINED_STRENGTH)):
                    if not found:
                        continue
                    weight, why = self._label_weight(_norm_ws(t), len(found))
                    rep_matched.append({"class": cls, "nodes": sorted(found),
                                        "weight": weight, "guard": why})
                    for nid in found:
                        # a node id typed in is not label text: BM25 never saw it
                        scale = 1.0 if nid in node_id_equal else ls
                        ent = {"signal": "id", "text": t, "match": cls,
                               "strength": strength, "weight": weight}
                        if why:
                            ent["guard"] = why
                        if scale != 1.0:
                            ent["label_scale"] = scale
                        add(nid, ent, cfg.boost_max * strength * weight * scale)
                entry = {"text": t, "matched": rep_matched}
                if t in skipped_set:
                    entry["contained_search"] = "skipped_by_cap"
                ids_report.append(entry)

        # ---- whole-query label (lowercase titles such as "harbor") -----------------
        whole_report = None
        if sig.whole_query:
            key = _sql_key(sig.whole_query)
            hits = by_key.get(key, [])
            already = {nid for nid, ms in matches.items()
                       if any(m["signal"] in ("name", "quoted", "id") for m in ms)}
            hits = [h for h in hits if h[0] not in already]
            if hits:
                strength = 1.0 if len(sig.whole_query.split()) > 1 else SINGLE_NAME_STRENGTH
                weight, why = self._label_weight(
                    _norm_ws(sig.whole_query), len(by_key.get(key, [])))
                whole_report = {"text": sig.whole_query,
                                "matched": [h[0] for h in hits],
                                "weight": weight, "guard": why}
                for nid, _t in hits:
                    ent = {"signal": "whole_query", "text": sig.whole_query,
                           "strength": strength, "weight": weight}
                    if why:
                        ent["guard"] = why
                    if ls != 1.0:
                        ent["label_scale"] = ls
                    add(nid, ent, cfg.boost_max * strength * weight * ls)

        # ---- dates -----------------------------------------------------------------------
        dates_report = []
        for dm in self.match_dates(sig.dates, **scope):
            d, scale = dm.signal, dm.scale
            # A crowded date (more matches than DATE_MAX_MATCHES) cannot say
            # which of its nodes is meant: it still lifts the ones that are
            # already candidates (scaled), and admits none of them.
            admissible = not dm.crowded
            hit_ids = []
            for nid, g, strength in dm.event:
                hit_ids.append(nid)
                add(nid, {"signal": "date", "text": d.text,
                          "field": "event_time", "node_granularity": g,
                          "strength": strength, "weight": scale},
                    cfg.boost_max * strength * scale, admissible)
            for nid, strength in dm.recorded:
                hit_ids.append(nid)
                add(nid, {"signal": "date", "text": d.text,
                          "field": "recorded_at", "strength": strength,
                          "weight": scale},
                    cfg.boost_max * strength * scale, admissible)
            entry = {**d.describe(), "matched": hit_ids,
                     "matched_by_event_time": sorted(
                         nid for nid, _g, _s in dm.event),
                     "matched_by_recorded_at": sorted(
                         nid for nid, _s in dm.recorded),
                     "matches_total": dm.total,
                     "scale": round(scale, 4)}
            if dm.crowded:
                entry["crowded"] = True
            dates_report.append(entry)

        # ---- boosts, capped; injection, bounded -------------------------------------
        boost = {nid: min(cfg.boost_max, parts)
                 for nid, parts in boost_parts.items() if parts > 0.0}
        anchors = set(anchor_ids)
        inject = sorted((nid for nid in boost
                         if nid in admit and nid not in anchors),
                        key=lambda n: (-boost[n], n))[:cfg.max_inject]

        detected = {
            "quoted": quoted_report,
            "dates": dates_report,
            "names": accepted_names + [
                {"text": t, "matched": [], "weight": 0.0, "guard": None}
                for t in unmatched_names],
            "ids": ids_report,
            "whole_query": whole_report,
            "entity_anchors": list(entity_anchor_ids or []),
        }
        if sig.truncated:
            # Reported whether or not a signal was found (build then returns
            # a result with no boost rather than None).
            detected["truncated"] = True
            detected["scanned_chars"] = sig.scanned_chars
            detected["query_chars"] = sig.query_chars
        if sig.ids and skipped:
            detected["ids_capped"] = {
                "capped": True, "max_contained_codes": cfg.max_contained_codes,
                "searched": len(searched), "skipped": len(skipped),
                "skipped_codes": skipped}
        has_signal = bool(
            quoted_report or dates_report or ids_report or accepted_names
            or unmatched_names or whole_report)
        if not has_signal and not sig.truncated:
            return None
        return ExactResult(detected=detected, matches=matches, boost=boost,
                           inject_ids=inject)
