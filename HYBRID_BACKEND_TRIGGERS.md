# Hybrid Classifier Backend — Trigger List

The Claim Sovereignty Layer ships **rule-based**. The classifier (Leg 2.5) and the
supersession gate (Leg 3) both clear their hard safety bars on rules alone:

- Classifier: **0.0% confident error**, 100% accuracy-when-committed, on the
  paraphrase-stressed fixtures.
- Gate: **0 unsafe auto-fires**, 100% gate-logic accuracy on engaged pairs.

What rules *don't* give is coverage: ~49% classifier coverage, ~54% gate
engagement, 33% on paraphrases. A hybrid/LLM classifier backend (never first)
would lift coverage — but the decision to build it is gated on **two triggers**,
and they are deliberately separate. A "queue's fine, skip the backend" decision
must never accidentally skip the safety close.

---

## Trigger 1 — Queue depth (WORKLOAD). May or may not fire.

**Kind:** workload. **Measured:** in production, by real volume — not a benchmark.
**Status:** instrumented, watching.

End-to-end auto-fire is ~30% (50% of what the gate engages); the rest routes to
the candidate-review queue. That is **safe** — everything below the line is
human-adjudicated, never wrongly auto-superseded — but it is a workload cost.
Whether that cost is *painful* depends on volume only production can show.

- Instrument with `revien.supersession.SupersessionMetrics`: feed every gate
  decision, watch `candidate_queue_depth` and `auto_fire_rate` over time.
- If the queue is manageable at real volume, this trigger **does not fire** and
  the backend may be deferred indefinitely on workload grounds.
- Do **not** pre-commit to the backend on this trigger. Let volume justify it.

## Trigger 2 — Sensitive-claim recognition (SAFETY). Fires regardless.

**Kind:** safety. **Measured:** by capability, not volume. **Status:** open —
**non-negotiable before launch.**

The rule classifier cannot recognize every sensitive claim. The recognition gap
has **two manifestations**, and the interim floor only covers the first:

1. **Unclassified** — sensitive content the rules classify as *nothing*: "I'm
   sober" → unclassified. **Covered by the interim floor** (never auto-superseded).
2. **Confidently misnamed** — sensitive content the rules classify, with
   confidence, into a *non-protected* type: "I love being sober", "I enjoy being
   single", "I like my transition" → `preference_habit` (classified,
   non-protected, changeable). **NOT covered by any current guard.** The interim
   floor keys on classification *status* (this is classified) and the protected
   guard keys on claim *type* (`preference_habit` is correctly non-protected), so
   the claim threads between both. A later same-type contradictor ("I don't enjoy
   being sober") yields a scoped contradiction and the gate returns
   `AUTO_SUPERSEDE` — the sensitive disclosure is silently erased, under the
   DEFAULT config, no back door. Reproduced live across sobriety/recovery,
   transition/identity, and relationship status (see
   `tests/test_sensitive_gap_regression.py`).

The hybrid backend must close **both** manifestations so sensitive claims are
confidently classified AND protected by the gate's own mechanisms (protected-set,
iron-grip), not only by the interim floor.

- This trigger fires **regardless of queue depth.** A comfortable queue does not
  retire it. (And note: floor-caught sensitive volume is counted separately in
  `SupersessionMetrics.sensitive_floor_caught`, NOT in `candidate_queue_depth` —
  a quiet queue during a sensitive-heavy stream must not be read as "safe".)
- Until it lands, the **interim sensitive floor** (below) holds the line **for
  manifestation 1 only.** Manifestation 2 is a known, pinned, live gap during the
  interim — it is the strongest single argument that Trigger 2 is non-negotiable.

> The triggers are different kinds. Do not collapse them. Trigger 1 is a
> workload question that volume may answer "no" to. Trigger 2 is a safety close
> that must happen before launch no matter what Trigger 1 says.

---

## Interim sensitive floor (in force NOW, until Trigger 2 lands)

A bring-forward of Leg 6's `irongrip_floor_minimum`, scoped to the
sensitive-recognition gap:

- The generic candidate-only default for **not-confidently-classified**
  (`unclassified` / `low_confidence`) claims is pinned at the named-sensitive
  protection level — **candidate_only / never auto-superseded** — and is
  **non-configurable.**
- Enforced in `SupersessionGate.evaluate` as the FIRST check, ahead of all
  configurable logic, reading no config. It holds even with
  `protected_set=frozenset()`. (`revien/claims.py::SENSITIVE_FLOOR` names the
  reference level; `tests/test_sensitive_floor.py` proves it config-proof.)
- This closes the config back door: an operator reconfiguring the protected set
  (a Leg 6 capability) cannot lower the unclassified default below sensitive, so
  an unclassified-sensitive claim cannot lose protection before the backend lands.

The configurable protected set (taxonomy contract §7.3) still governs *classified*
claims and remains operator-configurable under governance; the interim floor
governs only the *unclassified* default, and is not configurable at all.

**Scope note (avoid over-trusting the floor):** "holds even with
`protected_set=frozenset()`" is a statement about the *unclassified* floor only.
Emptying the protected set is a governance action that **does** de-protect
*classified* named-sensitive claims by design — a classified relationship/health/
identity/belief claim becomes auto-eligible when its type is removed from the set.
The floor is a minimum for the unclassified default, **not** a blanket minimum for
all sensitive content.
