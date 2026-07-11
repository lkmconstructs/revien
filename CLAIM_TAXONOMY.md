# Claim Taxonomy — Claim Sovereignty Layer, Leg 2.5 (CONTRACT)

> This is a **contract, not a table.** The enums below are vocabulary; the
> invariants are binding. The danger was never the list of types — it is a future
> implementation treating the list as a lookup and turning "type" into "truth
> behavior." The invariants in §7 exist to prevent exactly that, and they are
> committed verbatim. Nothing in Leg 3 (gated supersession) may consume this
> taxonomy in a way that violates §7.

Status: **blessed at the gate (2026-06-21)**, with three conditions folded in
(durability independence §7.1, boundary rules §8, configurable hard floor §7.3).
Build order is staged: this document (L2.5a) is committed **before** any
classifier exists, and the classifier's error rate is measured in isolation
(L2.5b–d) **before** Leg 3 supersession is allowed to consume `claim_type`
(L2.5e). Leg 2.5's classification error must never become Leg 3's silent bug.

---

## 1. Purpose

Give every stored claim two orthogonal classifications — **what kind of truth it
is** (`claim_type`) and **how changeable it is by nature** (`durability`) — so
that the supersession layer (Leg 3) can resolve contradictions *conservatively*:
aggressively detected, cautiously applied. Classification produces a prior and a
confidence; it never produces permission to mutate.

## 2. Non-goals

- **Not** a decision to supersede. Classification informs Leg 3; it never acts.
- **Not** a lock. `claim_type` is not authority over a claim (see §7). Lock
  authority (normal / protected / irongrip) is a separate Leg 3 axis.
- **Not** a forced label. The classifier is never pressured into a guess because
  the schema demands a type — "unknown" is a first-class, allowed outcome (§6).
- **Not** durability. Type provides only a *prior* for durability (§7.1).

## 3. Claim type enum

Twelve types. **Final enum names. No merges.** Specifically, `aspiration_goal`
does NOT fold into `project_status_plan`: directional identity pressure ("I want
to become someone") and operational state ("I am currently doing a thing") are
different classes of truth, and merging them makes supersession dumb.

| enum | what it is |
|---|---|
| `identity` | who someone fundamentally is (gender, origin, name, biographical fixtures) |
| `relationship` | a tie to another person (married, single, friend, family) |
| `preference_habit` | likes, routines, recurring activities |
| `current_state` | a presently-true, inherently temporary condition |
| `health_state` | physical/medical condition |
| `emotion_state` | a feeling |
| `historical_event` | something that happened at a time |
| `project_status_plan` | an active, bounded undertaking with steps/deadlines/completion state |
| `schedule` | a future-dated commitment |
| `belief_value` | what the person believes, values, prioritizes, morally endorses |
| `aspiration_goal` | a desired identity, future direction, long-term ambition |
| `semantic_fact` | what an external object/artifact/group/event/entity means, contains, represents, or is |

## 4. Durability enum

How changeable the claim is **by nature** — classified per claim (§7.1), not read
off the type.

| enum | meaning |
|---|---|
| `stable` | changes rarely or never (identity, most semantic facts) |
| `slow_change` | changes over long timescales (preferences, beliefs, aspirations) |
| `fast_change` | changes quickly / inherently transient (emotion, current state, plans, schedule) |
| `one_time` | a fixed point that does not change once recorded (historical events) |
| `unknown` | durability could not be confidently classified — treated as non-changeable for safety |

## 5. Status enum (`classification_status`)

The outcome of the *type* classification attempt. Not a 13th type.

| enum | meaning |
|---|---|
| `classified` | confidently typed; `claim_type` set |
| `low_confidence` | a tentative type may be retained as a hint, but treated as not confidently classified |
| `unclassified` | no confident type; `claim_type` is `null` |

The confidence bands that separate these three are a **mechanism-phase parameter
(L2.5b)**, deliberately not fixed in this contract. Both `low_confidence` and
`unclassified` share the same safety behavior: stored, retrievable, and barred
from automatic mutation (§7, §10).

## 6. Classification outcome enum

The mutation **disposition** classification yields — the bridge to Leg 3.

| enum | meaning |
|---|---|
| `candidate_only` | the claim may be stored and retrieved but **never auto-mutated**; any supersession goes to candidate review |
| `auto_eligible` | the claim *may* be considered for automatic supersession — **only** when `classified` AND non-protected (§7.3) AND a changeable durability AND Leg 3's full gate (explicit + high-confidence + same-scope + non-irongrip) also passes |

**Leg 2.5 emits `auto_supersession_allowed = false` unconditionally.** Only Leg 3,
applying its full gate, may upgrade a claim to `auto_eligible`. This enforces
"type is not lock authority" (§7): classification never grants permission to
mutate. Anything `low_confidence`, `unclassified`, `unknown`-durability, or
protected (§7.3) is `candidate_only` and stays there.

## 7. Invariants (the contract spine — binding)

```
Type is not durability.
Type is not status.
Type is not lock authority.
Unknown is allowed.
Low confidence never auto-supersedes.
Irongrip is never auto-superseded.
Historical claims are not overwritten; they are versioned or bounded.
Semantic similarity alone is not contradiction.
Negative sentiment alone is not contradiction.
Ambiguous claims that could plausibly touch a protected category
  inherit that protection until resolved.
Protected-set membership is operator-configurable but never
  modifiable by the learning loop.
```

### 7.1 Durability is an independent axis

```
Claim type provides a prior.
Durability is classified per claim.
Durability may override the default prior.
Supersession may never rely on claim_type alone.
```

This is the seam most likely to be implemented lazily as
`durability = DEFAULTS[claim_type]`. **That implementation is the rot.** "Single"
is `relationship` by type but inherits `fast_change` so "I'm dating someone" can
supersede it. "Born in Germany" is `identity` / `stable` and must never
auto-supersede. Durability is classified per-claim, with type as prior only.

### 7.2 What contradiction is NOT

Two failure modes, barred as invariants above and restated for the classifier and
Leg 3:

- **Semantic similarity alone is not contradiction.** Two claims about the same
  subject are not in conflict merely because they embed near each other.
- **Negative sentiment alone is not contradiction.** "I'm frustrated with X
  today" (emotion, fast_change) does not contradict — and may never supersede — a
  standing claim about X.

### 7.3 High-impact categories: hard-floor protected, configurable, learning-locked

High-impact categories default to **protected** — barred from automatic mutation
regardless of classification confidence. Not because they cannot change, but
because false updates are costly and recovery is painful.

**Default hard-floor-protected set:**

```
health_state
identity
belief_value
relationship
```

Membership of the protected set is **operator-configurable** — what is sensitive
depends on the deployment (a B2B user storing no health data may remove
`health_state`; a personal user keeps the defaults). The config is what varies
with use; the protection is what is safe by default. Three constraints keep
configurability from becoming a vulnerability:

1. **Governance-level change only.** Removing a category from the protected set is
   a deliberate, surfaced, affirmative, audit-logged action — at setup or in a
   clearly-marked governance setting that announces what it does ("health claims
   will now be subject to normal supersession"). NOT a casual runtime toggle.
2. **Learning-loop-locked.** The learning loop (Leg 5) can NEVER modify
   protected-set membership — the same hard rule as it can never lower the
   irongrip floor. This closes the back door where a script, migration, or the
   loop itself silently strips protection.
3. **Ambiguity inheritance tracks the set.** Ambiguous claims inherit protection
   only toward categories *currently* in the protected set. Remove `health_state`
   and "tired and achy lately" no longer inherits health protection — the
   hard-floor and the ambiguity-inheritance behaviors move together, so the
   system never reaches the incoherent state where health is unprotected but
   health-adjacent uncertainty still is.

## 8. Boundary rules (committed before the classifier is built)

The two seams most likely to be misclassified, with binding definitions.

```
semantic_fact:  what an external object, artifact, group, event, or entity
                means, contains, represents, or is.
belief_value:   what the person believes, values, prioritizes, morally
                endorses, or interprets as important.

aspiration_goal:      a desired identity, future direction, long-term ambition.
project_status_plan:  an active, bounded undertaking with steps, deadlines,
                      completion state, or near-term operational movement.
```

## 9. Examples

**Boundary cases (§8):**

| utterance | type | note |
|---|---|---|
| "The necklace symbolizes love, faith, strength" | `semantic_fact` | meaning of an external object |
| "I value love, faith, and strength" | `belief_value` | the person's own values |
| "I want to become a counselor" | `aspiration_goal` | desired future identity |
| "I'm researching counseling programs this month" | `project_status_plan` | active bounded undertaking |
| "I'm applying next week" | `schedule` *or* `project_status_plan` | depends on whether the date-commitment is the main point |

**Protected / durability cases (§7.1, §7.3):**

| utterance | behavior |
|---|---|
| "Theo has a cough today" | expires — ephemeral `health_state`-adjacent, not a protected standing claim |
| "Mara has celiac disease" | not casually overwritten — `health_state`, protected |
| "Mara is politically liberal" | not flipped by one sarcastic comment — `belief_value`, protected |
| "Mara is married" | not superseded by "annoyed at Sam today" — `relationship` protected, and `emotion_state` cannot supersede it anyway |

**Classification-outcome records:**

```json
// confident
{ "claim_type": "project_status_plan", "claim_type_confidence": 0.82,
  "classification_status": "classified", "durability": "fast_change",
  "auto_supersession_allowed": false }

// uncertain — the safety floor
{ "claim_type": null, "claim_type_confidence": 0.41,
  "classification_status": "unclassified", "durability": "unknown",
  "auto_supersession_allowed": false, "route": "candidate_only" }
```

If the system cannot confidently classify, it **stores the claim, keeps it
retrievable, and bars it from automatic mutation.** The classifier is never
pressured into a guess because the schema demands a type. Guessing is how you get
elegant-looking garbage.

## 10. Routing consequences

How a classification outcome dispositions a claim toward Leg 3 / Leg 4:

| condition | disposition |
|---|---|
| `unclassified` or `low_confidence` | `candidate_only` — stored, retrievable, never auto-mutated |
| `durability = unknown` | `candidate_only` — treated as non-changeable for safety |
| `claim_type ∈ protected set` (§7.3) | `candidate_only` — auto-mutation barred regardless of confidence |
| ambiguous toward a protected category | inherits protection → `candidate_only` until resolved |
| `classified` + non-protected + changeable durability | eligible for Leg 3's gate (which may still route to candidate); Leg 2.5 itself still emits `auto_supersession_allowed = false` |

The canonical classifier output record (shape fixed here; thresholds and features
are mechanism-phase, L2.5b):

```json
{
  "claim_type": "project_status_plan",
  "claim_type_confidence": 0.82,
  "classification_status": "classified",
  "durability": "fast_change",
  "durability_confidence": 0.78,
  "boundary_notes": [],
  "auto_supersession_allowed": false
}
```
