"""
The judgment rule (N1): did the answer actually draw on this memory?

Policy: BOTH signals must agree. A false "used" is the expensive mistake —
it strengthens the memory, the memory comes back more often, and the next
false "used" gets easier. That is the same compounding loop the context
fence exists to stop, so when the evidence is thin the answer is "not used",
and when a signal is missing the rule abstains.

Three outcomes:
  used=True                — both signals clear the bar; counted as a use.
  used=False               — judged, and the answer did not draw on it.
  abstain=True             — not enough to judge (e.g. similarity is None
                             because the semantic layer is off). Nothing is
                             written; the memory is judged again the next time
                             its session file changes and is rescanned (or
                             on ``revien usage scan --full``).
"""

from __future__ import annotations

from pydantic import BaseModel

from .evidence import UseEvidence


class UseVerdict(BaseModel):
    """The rule's answer for one (answer, memory) pair."""

    used: bool
    abstain: bool = False
    reason: str = ""  # one line, stored in the ledger next to the evidence


def judge_use(evidence: UseEvidence) -> UseVerdict:
    """Decide whether one answer drew on one recalled memory.

    ``evidence.shared_terms``: distinctive memory words that reappear in the
        answer, with the prompt's and query's words already removed. One
        shared word can be coincidence ("config"); several are harder to fake.
    ``evidence.memory_terms``: how many distinctive words the memory had
        after that removal — useful if you would rather require a share of
        them than a fixed count (a two-word memory can only ever share two).
    ``evidence.similarity``: best cosine between the memory and any answer
        paragraph, or None when the semantic layer is off. With the default
        local model (bge-small-en-v1.5), unrelated text tends to land around
        0.4-0.6 and close paraphrase around 0.8+; treat those as starting
        points, not facts. Shadow mode exists so you can check the bar
        against real sessions with ``revien usage list`` before turning on
        apply.
    """
    # TODO(Lissa): write the rule, 5-10 lines. It has to:
    #   - abstain when a signal is missing (similarity is None),
    #   - return used=True only when the shared terms AND the similarity
    #     both clear your bars,
    #   - put the deciding numbers in `reason` (it is what `revien usage
    #     list` shows when you audit a judgment later).
    # tests/test_usage_judge.py is the contract; it passes once this does.
    raise NotImplementedError(
        "judge_use() is not written yet: see the TODO in revien/usage/judge.py"
    )
