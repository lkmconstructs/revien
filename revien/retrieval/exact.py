"""
Exact lane (REVIEN_EXACT) - EXPERIMENTAL, default off.

A recall lane that switches on only when the query contains something exact,
and otherwise does nothing at all. It is not a text-scoring lane: it never
looks at node content. It matches three structured things:

  * quoted phrases, names/titles and code-like tokens against node LABELS
    (exact, case-insensitive, whitespace-normalized), and, when a query names
    an entity but asks about something else, the nodes CONNECTED to that
    entity by an edge (one hop);
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
        followed by "of" ("through 3/4 of the book"), and, after BY, does not
        follow a verb of quantity ("reduced by 1/2", FRACTION_VERBS). No other
        word marks one: "of", "from", "between", "during", "in", "out of",
        "score of" do not ("a score of 8/10", "the ratio of 3/4", "between
        1/2 and 3/4 cup", "in 8/10 cases"), and "from 3/4 to 3/8" is two
        ratios unless padded or given a year. A bare a/b with a > 12 has no
        month ("24/7", "16/9") and is not a date even after a preposition. Of
        the 366 bare M/D forms, 316 parse with no preposition and all 366
        after "on".
      * a date's characters are blanked once it is read, and the text on
        either side of a blanked date never joins into another date.
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
    not IDs. Tokens consumed by a date or quote are not IDs. Matching is a
    label equal to the token, a label containing the token as a whole token
    (the first MAX_CONTAINED_CODES (16) codes of 4+ characters only: that
    lookup reads the label column once per code), or a node id equal to the
    token. Never node content.
 5. How much of a query is read. The lane reads at most the first
    SCAN_CHARS (4,096, REVIEN_EXACT_SCAN_CHARS) characters of a query, cut
    back to the last whitespace so no token is split; the rest is not
    searched for signals, and a query that was cut has no whole-query label
    (rule 3). Diagnostics say "scanned_chars" when a query was cut. Detection
    is a single left-to-right pass per rule (linear in what it reads). The
    bound is the lane's own: the rest of recall sees the whole query, and
    with the lane off none of this runs.

THE BOOST (bounded, not a promotion)

    boost(node) = min( BOOST_MAX, sum_over_signals( BOOST_MAX * strength * weight ) )

    strength  quoted / multiword name / id equal / node id  1.0
              single-token name / whole-query label          0.8
              id contained in a label                        0.7
              neighbour of a named entity: the name's strength * NEIGHBOR_STRENGTH (0.5)
              date on event_time: min(node granularity, query granularity)
                    with day 1.0, week 0.6, month 0.4, year 0.15, fuzzy 0.1
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
    only as a neighbour of a named entity is not admitted.
"""

from __future__ import annotations

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
# Year-first accepts "-" and "/" only: dotted numbers are versions ("2024.1.5").
_RE_ISO = re.compile(r"(?<![\d/.-])(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d|[/.]\d)")
_RE_SLASH_Y = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])")
_RE_DMY = re.compile(
    rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b\.?,?\s+(\d{{4}})\b", re.I)
_RE_MDY = re.compile(
    rf"\b{_MON}\b\.?\s+(\d{{1,2}}){_ORD}\b,?\s+(\d{{4}})\b", re.I)
_RE_MD = re.compile(rf"\b{_MON}\b\.?\s+(\d{{1,2}}){_ORD}\b", re.I)
_RE_DM = re.compile(rf"\b(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MON}\b", re.I)
_RE_SLASH = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")
_RE_MY = re.compile(rf"\b{_MON}\b\.?,?\s+(\d{{4}})\b", re.I)

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
_RE_OF_NEXT = re.compile(r"\s+of\b", re.I)
_RE_ALPHA_RUN = re.compile(r"[A-Za-z]+")


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
            common_words=cls._words_from_env(
                "REVIEN_EXACT_COMMON_WORDS", COMMON_WORDS),
            framing_words=cls._words_from_env(
                "REVIEN_EXACT_FRAMING_WORDS", FRAMING_WORDS),
        )


# ── signals ───────────────────────────────────────────────────────────────────
@dataclass
class DateSignal:
    text: str
    granularity: str            # "day" | "month"
    month: int
    day: Optional[int] = None   # None for a whole month
    year: Optional[int] = None  # None: matches any year

    def ranges_for_year(self, year: int) -> Optional[Tuple[str, str]]:
        """Inclusive ('YYYY-MM-DD', 'YYYY-MM-DD') for ``year``, or None when
        the date does not exist in that year (Feb 29)."""
        try:
            if self.day is None:
                last = calendar.monthrange(year, self.month)[1]
                return (date(year, self.month, 1).isoformat(),
                        date(year, self.month, last).isoformat())
            d = date(year, self.month, self.day).isoformat()
            return (d, d)
        except ValueError:
            return None

    def describe(self) -> Dict[str, Any]:
        return {"text": self.text, "granularity": self.granularity,
                "year": self.year, "month": self.month, "day": self.day,
                "any_year": self.year is None}


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
    # Characters read when the query was longer than the scan bound (rule 5);
    # None when the whole query was read.
    scanned_chars: Optional[int] = None

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
    ("on 3/10"), is not a fraction OF something ("through 3/4 of the book"),
    and is not an amount after a verb of quantity ("reduced by 1/2")."""
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

    def dmy(m, before):
        if not _month_ok(m[2], lambda: before(m.start())[0], strong_dm(m)):
            return None
        d, mo, y = int(m[1]), _MONTH_NAMES[m[2].lower()], int(m[3])
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def mdy(m, before):
        if not _month_ok(m[1], lambda: before(m.start())[0]):
            return None
        mo, d, y = _MONTH_NAMES[m[1].lower()], int(m[2]), int(m[3])
        return DateSignal(m[0], "day", mo, d, y) if _valid_md(mo, d, y) else None

    def md(m, before):
        if not _month_ok(m[1], lambda: before(m.start())[0]):
            return None
        mo, d = _MONTH_NAMES[m[1].lower()], int(m[2])
        return DateSignal(m[0], "day", mo, d) if _valid_md(mo, d) else None

    def dm(m, before):
        if not _month_ok(m[2], lambda: before(m.start())[0], strong_dm(m)):
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
        return DateSignal(m[0], "month", _MONTH_NAMES[m[1].lower()], None,
                          int(m[2]))

    each(_RE_ISO, iso)
    each(_RE_SLASH_Y, slash_y)
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


def _is_id_token(tok: str) -> bool:
    t = _strip_token(tok)
    if len(t) < 3 or _RE_ORDINAL_OR_UNIT.match(t):
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


def _scan_prefix(text: str, max_chars: int) -> str:
    """The first ``max_chars`` characters of ``text``, cut back to the last
    whitespace when the cut would split a token (rule 5)."""
    head = text[:max_chars]
    if len(text) > max_chars and not text[max_chars].isspace():
        k = max(head.rfind(c) for c in " \t\n\r\f\v")
        if k > 0:
            head = head[:k]
    return head


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
    (default COMMON_WORDS; the lane passes its configured set). Only the
    first ``max_chars`` characters are read (rule 5; the lane passes its
    configured bound)."""
    common = COMMON_WORDS if common_words is None else common_words
    sig = ExactSignals()
    work = query or ""
    if len(work) > max_chars:
        work = _scan_prefix(work, max_chars)
        sig.scanned_chars = len(work)

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

    # Whole-query label (lowercase single tokens like "harbor"); only when the
    # whole query was read.
    if sig.scanned_chars is None:
        whole = re.sub(r"\s+", " ", _trim_edges(query or "")).strip()
        whole = _RE_POSSESSIVE.sub("", whole)
        if whole and len(whole.split()) <= MAX_SPAN_TOKENS:
            sig.whole_query = whole
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
                       if total else 0.0)))
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
        nothing and recall is byte-identical to the lane being off).

        ``label_scale`` multiplies every boost that comes from label text
        (name, quote, id, whole-query); the engine passes
        ``config.bm25_label_scale`` when the BM25 lane, which indexes labels,
        is on, and 1.0 otherwise."""
        cfg = self.config
        sig = self.parse(query)
        if not sig.needs_lookup:
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
            long_ids = [t for t in sig.ids
                        if len(t) >= 4][:MAX_CONTAINED_CODES]
            contain_set = set(long_ids)
            # ONE pass over the label column for every code in the query.
            contain_rows = (
                self.store.exact_label_contains(
                    [ascii_lower(t) for t in long_ids], **scope)
                if long_ids else [])
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
                ids_report.append({"text": t, "matched": rep_matched})

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
            hit_ids = []
            for nid, g, strength in dm.event:
                hit_ids.append(nid)
                add(nid, {"signal": "date", "text": d.text,
                          "field": "event_time", "node_granularity": g,
                          "strength": strength, "weight": scale},
                    cfg.boost_max * strength * scale)
            for nid, strength in dm.recorded:
                hit_ids.append(nid)
                add(nid, {"signal": "date", "text": d.text,
                          "field": "recorded_at", "strength": strength,
                          "weight": scale},
                    cfg.boost_max * strength * scale)
            dates_report.append({**d.describe(), "matched": hit_ids,
                                 "matched_by_event_time": sorted(
                                     nid for nid, _g, _s in dm.event),
                                 "matched_by_recorded_at": sorted(
                                     nid for nid, _s in dm.recorded),
                                 "matches_total": dm.total,
                                 "scale": round(scale, 4)})

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
        if sig.scanned_chars is not None:
            detected["scanned_chars"] = sig.scanned_chars
        has_signal = bool(
            quoted_report or dates_report or ids_report or accepted_names
            or unmatched_names or whole_report)
        if not has_signal:
            return None
        return ExactResult(detected=detected, matches=matches, boost=boost,
                           inject_ids=inject)
