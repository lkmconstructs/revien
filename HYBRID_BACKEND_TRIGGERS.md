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

### Trigger 2 — Attempt 1 (embedding recognizer): adversarially proven INSUFFICIENT

`revien/sensitivity.py::SemanticSensitivityRecognizer` (local bge-small embeddings
vs sensitive/neutral prototype banks, two-sided routing with required abstention)
passed the seeded acceptance fixtures (100% sensitive→candidate incl. 5 unseeded,
0 unsafe leaks, measure_sensitivity.py) — but the **required adversarial pass
(`wf_194fc6c9`, 6 agents) FAILED it.** 31 real disclosures routed
`confident_neutral` (recovery chip/sponsor/AA, psych meds, cancer imaging,
immigration, coming-out, domestic-violence orders, **bankruptcy via procedural
rephrasing — a *seeded* domain**, parole, IVF). Proof it keys on **surface
construction, not cost-of-erasure**: a contraction flips "I am out at work now"
(sensitive 0.669) vs "I'm out at work now" (neutral 0.644) across the line, and
the classes overlap in embedding space ("refill at the pharmacy" leak sens 0.637 >
"I love painting" neutral sens 0.616) so no threshold separates them. This is the
"fuzzier lexeme matcher with a bigger wordlist" the spec warned against.

**Status: embedding recognizer is an ADDITIVE layer (more protective than
tripwire-only, opt-in `recognizer=` on the gate), but it does NOT close Trigger 2.
Trigger 2 stays RED.** Genuine cost-of-erasure recognition needs a mechanism that
reasons about MEANING (an LLM that can answer "would erasing this betray the
user?"), not embedding-cosine. The mechanism is an open architecture decision
(local Ollama vs cloud vs hybrid embedding-prescreen + LLM). The 31 leaks are the
measurable success criterion: the real recognizer must drive them to 0
confident_neutral.

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

---

## Interim distrust tripwire (in force NOW — covers manifestation 2, LEXED only)

The floor covers manifestation 1 (unclassified). The **distrust tripwire**
(`revien/tripwire.py`, default-on in `SupersessionGate`) covers the *lexed* slice
of manifestation 2: when a claim's raw/normalized **content** names a core
sensitive domain, the gate distrusts its own classifier's type and routes the
claim to **candidate** with a labeled reason (`tripwire_distrust:<domain>`), even
though the type came out non-protected. So "I love being sober" no longer
auto-erases — it goes to human review.

It is a deliberately blunt, interim instrument, governed by **six invariants**
(committed verbatim in `revien/tripwire.py`):

1. **Strictly additive** — only routes to candidate; never grants auto, never
   removes protection, never alters type/durability/content. Only ever more
   conservative.
2. **Candidate-only** — a match is human review; it never auto-acts.
3. **Type-independent** — inspects raw + normalized content, not `claim_type`;
   the distrust of the classifier's type is the point.
4. **Core domain is a config-floor** — operators may ADD domains/lexemes; they may
   NEVER remove the reproduced-harm core set; the learning loop never touches it;
   no config/attribute/subclass can shrink it.
5. **Does not close Trigger 2** — lexemes, not meaning. It WILL miss sensitive
   content phrased without a core lexeme ("I love being off the bottle" still
   auto-supersedes today — see `tests/test_sensitive_gap_regression.py`). It
   structurally cannot satisfy the sensitive-recognition test and never retires it.
6. **False positives are humility, not failure** — deliberately over-broad
   ("recovery from the flu", "single-minded" trip it). Over-catch = an extra
   review = safe. Narrowing to reduce false positives is the defect, not the fix:
   betrayal with a stack trace costs more than an extra review.

**This does NOT move Trigger 2 off red.** The tripwire closes the lexed slice; the
unlexed/semantic slice stays open and is pinned as the backend's success criterion.

### Trigger 2 — Attempt 2 (LLM recognizer): VERIFIED at 0 leaks (recognition closed)

`revien/sensitivity_llm.py::LLMSensitivityRecognizer` asks an LLM "would silently
auto-erasing this betray the user?" → SENSITIVE / NEUTRAL / UNSURE; only a clean
NEUTRAL auto-clears, UNSURE/unparseable/failure abstains. Backend-pluggable (local
ollama = production zero-cloud default; openai/anthropic = cloud, opt-in, loud
egress disclosure). Verified against **gpt-4.1**:

- **Gate-1 battery: 0 leaks / 35** known sensitive incl. all 20 disclosures the
  embedding recognizer failed (`measure_sensitivity_llm.py`).
- **Adversarial battery (64 hand-authored: euphemism / indirect / procedural /
  unseeded HIV·sexwork·gambling·military·conversion-therapy / context-dependent):
  0 leaks.**
- **Independent corpus (217 LLM-generated realistic entries, benign framing):
  0 clear leaks** — 130 routed candidate (incl. every buried oblique disclosure:
  "9 months… the chip", "six months on T", "egg retrieval scheduled", "biometrics
  at USCIS", "P.O. check-in, pee test"); the 87 cleared-NEUTRAL are all genuinely
  mundane (chores / pets / work / shopping). One borderline ("standing thing with
  the group downtown" — discloses nothing on its face) flagged, defensibly neutral.
- **Abstains** (~20% safe FP) on ambiguous/novel — required, measured, acceptable.
- **Gate end-to-end** (recognizer wired): classified-sensitive → candidate; neutral
  → auto. Tripwire + floor remain additive backup.

Contrast: embedding leaked 31 and needed 96.7% stream abstention for zero leaks;
the LLM holds 0 leaks at ~20% abstention because it reasons about meaning.

**RECOGNITION CAPABILITY is closed** — an LLM genuinely recognizes cost-of-erasure.
**Two deployment items remain before Trigger 2 is fully GREEN for launch:** (1) the
zero-cloud production path (local ollama) is built but UNVERIFIED — needs the same
batteries run against the chosen local model; a cloud recognizer egresses possibly-
sensitive claims and is the verified *reference*, not the sovereign default. (2)
Lissa's ruling on whether this clears her closing criterion.

**Demotion path (the only way the tripwire retires):** ship it → instrument
catches (`SupersessionMetrics.tripwire_caught` / `tripwire_by_domain`) + queue +
miss data in real use → build the semantic recognition backend (Trigger 2) →
**demote the tripwire only after the backend's MEASURED performance proves it
safe to.** Until then it stands as the interim promise that the trust product
will not betray a sensitive disclosure with a stack trace while the real fix is
built.
