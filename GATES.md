# Gates: revien v0.4 — origin layer, source filter, pairing token, skills

OWNS: revien/**, tests/**, CHANGELOG.md, README.md

Scope: v0.4 delivers the origin layer, --source filter, pairing token, thin skills slice, the ChatGPT/Claude/Readwise importers, and the LoCoMo LLM-judge track, with governance intact and zero egress on the default path.

- [ ] G1: full test suite passes with no failures
  CHECK: python -m pytest -q -p no:cacheprovider --basetemp=.gates-tmp -x
  EXPECT: /\d+ passed, \d+ skipped, 1 xfailed/
  EVIDENCE: pending

- [ ] G2: legacy db (no origin columns) opens, backfills every known source_id convention, and is idempotent
  CHECK: python scripts/gates/check_origin_backfill.py
  EXPECT: origin backfill verification passed
  EVIDENCE: pending

- [ ] G3: source filter never emits foreign-runtime content in results, tensions, path labels, or skill_proposals, and fails closed on empty
  CHECK: python scripts/gates/check_source_filter.py
  EXPECT: source filter verification passed
  EVIDENCE: pending

- [ ] G4: every state-changing daemon route refuses a remote caller without the pairing token and accepts one with it
  CHECK: python scripts/gates/check_mutation_auth.py
  EXPECT: mutation auth verification passed
  EVIDENCE: pending

- [ ] G5: skill proposals are engine-origin and proposed-only; single-id accept keeps engine origin; accepted body is frozen; third decline invalidates; every step audited
  CHECK: python scripts/gates/check_skill_governance.py
  EXPECT: skill governance verification passed
  EVIDENCE: pending

- [ ] G6: user-authored skill survives a same-name engine proposal untouched and sorts first
  CHECK: python scripts/gates/check_user_skill_precedence.py
  EXPECT: user skill precedence verification passed
  EVIDENCE: pending

- [ ] G7: declared origin outside the fixed vocabulary is rejected at pipeline and daemon; MCP store cannot claim the vault channel
  CHECK: python scripts/gates/check_origin_vocab.py
  EXPECT: origin vocabulary verification passed
  EVIDENCE: pending

- [ ] G8: sovereignty and egress tests pass, and no changed source file imports a network client on the default path
  CHECK: python scripts/gates/check_egress.py
  EXPECT: egress verification passed
  EVIDENCE: pending

- [ ] G9: CLI skills propose/list/accept and recall --source run under a cp1252 console without a Unicode crash
  CHECK: python scripts/gates/check_cli_ascii.py
  EXPECT: cli ascii verification passed
  EVIDENCE: pending

- [ ] G10: TOON recall round-trips origin fields, [filtered] path entries, and skill_proposals losslessly
  CHECK: python scripts/gates/check_toon_roundtrip.py
  EXPECT: toon roundtrip verification passed
  EVIDENCE: pending

- [ ] G11: version string is untouched at 0.3.0 and CHANGELOG Unreleased names every shipped feature
  CHECK: python scripts/gates/check_docs.py
  EXPECT: docs verification passed
  EVIDENCE: pending

- [ ] G13: ChatGPT, Claude, and Readwise imports go through the pipeline, honor the deny list, stamp historical recorded_at and import origin, and are idempotent on re-run
  CHECK: python scripts/gates/check_importers.py
  EXPECT: importer verification passed
  EVIDENCE: pending

- [ ] G14: ChatGPT import ingests only the displayed thread, never an edited-away branch, and dry-run writes nothing
  CHECK: python scripts/gates/check_import_branches.py
  EXPECT: import branch verification passed
  EVIDENCE: pending

- [ ] G15: LoCoMo LLM judge is separate from F1, parses CORRECT/WRONG strictly, and a cloud judge fails the egress check even at zero measured calls
  CHECK: python scripts/gates/check_locomo_judge.py
  EXPECT: locomo judge verification passed
  EVIDENCE: pending

- [ ] G12: merge to main is the repository owner's call, never an agent's
  EVIDENCE: pending
