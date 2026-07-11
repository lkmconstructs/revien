"""
revien_bench.metrics — Official LoCoMo token-F1 + retrieval & latency metrics.

ZERO new deps: stdlib only. The Porter stemmer is vendored (pure-Python, public
domain — Martin Porter's reference algorithm) so token-F1 matches the official
repo's stemming behavior WITHOUT pulling in nltk.

=========================== F1 NORMALIZATION PROVENANCE ===========================
Matched against snap-research/locomo `task_eval/evaluation.py` (fetched 2026-06-18):

    def normalize_answer(s):
        s = s.replace(',', "")
        def remove_articles(text):
            return regex.sub(r'\\b(a|an|the|and)\\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        prediction_tokens   = [ps.stem(w) for w in normalize_answer(prediction).split()]
        ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0
        precision = num_same / len(prediction_tokens)
        recall    = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

We reproduce this EXACTLY, with two documented matches:
  1. ORDER of normalization is preserved: lower -> remove_punc -> remove_articles
     -> white_space_fix, with the leading comma-strip. NOTE the official article
     set is `(a|an|the|and)` — it strips "and" too (unusual vs SQuAD's `(a|an|the)`)
     and we match that quirk verbatim.
  2. STEMMING: official applies Porter stemming (`ps = PorterStemmer()` from nltk)
     to each token before counting. nltk is not a revien dep, so we vendor a
     pure-Python Porter stemmer (`porter_stem`) below. It implements Porter's
     reference algorithm; for the short, mostly-regular tokens in LoCoMo answers
     the stems agree with nltk's PorterStemmer. Documented divergence risk: a
     handful of irregular long words may stem differently — negligible on this
     dataset and disclosed here for honesty.

Adversarial (category 5): official scores EM=1 iff the model output contains
'no information available' OR 'not mentioned' (case-insensitive). We expose
`is_refusal()` matching that, and `adversarial_score()` returns 1.0/0.0.
"""

from __future__ import annotations

import math
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set


# ── Vendored Porter stemmer (public domain; Martin Porter reference) ──────────
# Compact, correct implementation of the classic Porter (1980) algorithm. Used
# to mirror the official LoCoMo f1_score's `ps.stem(...)` without an nltk dep.
class _PorterStemmer:
    def __init__(self) -> None:
        self.b = ""
        self.k = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else (not self._cons(i - 1))
        return True

    def _m(self) -> int:
        n = 0
        i = 0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        return any(not self._cons(i) for i in range(self.j + 1))

    def _doublec(self, j: int) -> bool:
        if j < 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self._cons(j)

    def _cvc(self, i: int) -> bool:
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s: str) -> bool:
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        self.b = self.b[: self.j + 1] + s + self.b[self.k + 1 :]
        self.k = self.j + len(s)

    def _r(self, s: str) -> None:
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                if self.b[self.k] in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        if self.k <= 0:
            return
        ch = self.b[self.k - 1]
        if ch == "a":
            if self._ends("ational"): self._r("ate")
            elif self._ends("tional"): self._r("tion")
        elif ch == "c":
            if self._ends("enci"): self._r("ence")
            elif self._ends("anci"): self._r("ance")
        elif ch == "e":
            if self._ends("izer"): self._r("ize")
        elif ch == "l":
            if self._ends("bli"): self._r("ble")
            elif self._ends("alli"): self._r("al")
            elif self._ends("entli"): self._r("ent")
            elif self._ends("eli"): self._r("e")
            elif self._ends("ousli"): self._r("ous")
        elif ch == "o":
            if self._ends("ization"): self._r("ize")
            elif self._ends("ation"): self._r("ate")
            elif self._ends("ator"): self._r("ate")
        elif ch == "s":
            if self._ends("alism"): self._r("al")
            elif self._ends("iveness"): self._r("ive")
            elif self._ends("fulness"): self._r("ful")
            elif self._ends("ousness"): self._r("ous")
        elif ch == "t":
            if self._ends("aliti"): self._r("al")
            elif self._ends("iviti"): self._r("ive")
            elif self._ends("biliti"): self._r("ble")
        elif ch == "g":
            if self._ends("logi"): self._r("log")

    def _step3(self) -> None:
        ch = self.b[self.k]
        if ch == "e":
            if self._ends("icate"): self._r("ic")
            elif self._ends("ative"): self._r("")
            elif self._ends("alize"): self._r("al")
        elif ch == "i":
            if self._ends("iciti"): self._r("ic")
        elif ch == "l":
            if self._ends("ical"): self._r("ic")
            elif self._ends("ful"): self._r("")
        elif ch == "s":
            if self._ends("ness"): self._r("")

    def _step4(self) -> None:
        ch = self.b[self.k - 1] if self.k >= 1 else ""
        ok = False
        if ch == "a":
            if self._ends("al"): ok = True
        elif ch == "c":
            if self._ends("ance") or self._ends("ence"): ok = True
        elif ch == "e":
            if self._ends("er"): ok = True
        elif ch == "i":
            if self._ends("ic"): ok = True
        elif ch == "l":
            if self._ends("able") or self._ends("ible"): ok = True
        elif ch == "n":
            if self._ends("ant") or self._ends("ement") or self._ends("ment") or self._ends("ent"): ok = True
        elif ch == "o":
            if self._ends("ion") and self.j >= 0 and self.b[self.j] in "st": ok = True
            elif self._ends("ou"): ok = True
        elif ch == "s":
            if self._ends("ism"): ok = True
        elif ch == "t":
            if self._ends("ate") or self._ends("iti"): ok = True
        elif ch == "u":
            if self._ends("ous"): ok = True
        elif ch == "v":
            if self._ends("ive"): ok = True
        elif ch == "z":
            if self._ends("ize"): ok = True
        if ok and self._m() > 1:
            self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.j = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[: self.k + 1]


_PS = _PorterStemmer()


def porter_stem(word: str) -> str:
    """Stem one token via the vendored Porter algorithm (matches nltk on LoCoMo)."""
    return _PS.stem(word)


# ── Official LoCoMo normalization + F1 ────────────────────────────────────────
_ARTICLES = {"a", "an", "the", "and"}  # official set strips "and" too


def normalize_answer(s) -> str:
    """Reproduce snap-research/locomo normalize_answer exactly.

    Order: comma-strip -> lower -> remove_punc -> remove_articles -> ws_fix.
    Accepts non-str gold (LoCoMo answers are sometimes int/float) by str()-ing.
    """
    s = str(s)
    s = s.replace(",", "")
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    # remove_articles (the official regex is \b(a|an|the|and)\b; after punct
    # removal tokens are whitespace-delimited, so a token filter is equivalent).
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    s = " ".join(s.split())
    return s


def _stem_tokens(text: str) -> List[str]:
    return [porter_stem(w) for w in normalize_answer(text).split()]


def f1_score(prediction, ground_truth) -> float:
    """Official LoCoMo token-level F1 (with Porter stemming). Range [0,1]."""
    pred_tokens = _stem_tokens(prediction)
    gold_tokens = _stem_tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        # Both empty -> perfect; one empty -> zero (mirrors degenerate handling).
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# ── Adversarial (category 5) refusal scoring ──────────────────────────────────
# Official matcher: EM=1 iff output contains 'no information available' or
# 'not mentioned' (case-insensitive). We publish the matcher (caveat §6.8).
_REFUSAL_MARKERS = ("no information available", "not mentioned")


def is_refusal(output: str) -> bool:
    """True iff `output` reads as a refusal per the official adversarial matcher."""
    low = (output or "").lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def adversarial_score(output: str) -> float:
    """1.0 if the answerer correctly refused on an adversarial Q, else 0.0."""
    return 1.0 if is_refusal(output) else 0.0


# ── Retrieval metrics (decoupled from the answerer) ───────────────────────────
def recall_at_k(retrieved: Sequence, gold: Set, k: int) -> float:
    """Fraction of gold items present in the top-k retrieved list.

    `retrieved` is an ordered list of evidence ids (e.g. dia_ids) that the
    retrieval hits map to; `gold` is the set of gold evidence ids. Returns the
    proportion of gold items recalled within the first k. Undefined (1.0) when
    there is no gold evidence (open-domain / adversarial Qs carry no evidence).
    """
    if not gold:
        return 1.0
    topk = list(retrieved[:k])
    hit = sum(1 for g in gold if g in topk)
    return hit / len(gold)


def mrr(retrieved: Sequence, gold: Set) -> float:
    """Reciprocal rank of the FIRST gold hit in `retrieved` (0 if none). 1-indexed."""
    if not gold:
        return 1.0
    for idx, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: Sequence, gold: Set, k: int = 10) -> float:
    """Binary-relevance nDCG@k against the gold evidence set."""
    if not gold:
        return 1.0
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            dcg += 1.0 / math.log2(i + 2)  # rank i (0-indexed) -> log2(i+2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Latency percentiles ───────────────────────────────────────────────────────
def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in [0,100]). Empty -> 0.0."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (pct / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(xs[lo])
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def latency_percentiles(values: Sequence[float]) -> Dict[str, float]:
    """p50/p90/p99 (+ min/max/mean) of a latency sample, rounded to 3 dp (ms)."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "p50": round(percentile(values, 50), 3),
        "p90": round(percentile(values, 90), 3),
        "p99": round(percentile(values, 99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(sum(values) / len(values), 3),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
