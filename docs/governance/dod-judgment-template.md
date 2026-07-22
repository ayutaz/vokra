# GA Definition-of-Done judgment record — v1.0 GA (M5-12)

> **This file is a blank template.** Copy it to
> `quarterly-reviews/<YYYY>-Q<N>.dod.md` (or record the judgment directly in the
> GA go-nogo record) and fill in the copy. Do not record a judgment by editing
> this file.
>
> Fields marked `_(記入)_` are for the maintainer. The **judgment of the five
> DoD items is owner work** (M5-12-T11): this template only wires in the
> material a machine can honestly produce and leaves the verdicts blank. A
> mechanism cannot decide EU certification, NPU speedups, competitor changelogs,
> or committer counts against a threshold; it can only lay the evidence out
> honestly so the owner decides.

- **Judgment date**: _(記入)_
- **Decision maker**: `ayutaz`
- **Milestone under review**: v1.0 GA (M5)
- **Where the result is recorded**: the GA go-nogo review record
  `docs/governance/vokra-go-nogo-v1.0-ga.md` (created from
  `vokra-go-nogo-v0.5.md`, the M2-15-T01 naming convention). The Kill switch
  A–L verdicts live **there**, not here — this template does not duplicate them.

## Read-through notes (apply to every item below)

1. **A skip is not a pass, and an unmeasured axis is not a pass.** The item-2
   runner reports `skipped_no_weight` and `skipped_no_corpus` as *distinct*
   non-passes, and flags a generative-audio model scored without a UTMOS scorer
   as `UTMOS leg: not run`. Do not read a green coverage line as "item 2 met"
   when the tally shows skips (NFR-QL-04 / FR-EX-08).
2. **numerical-parity CI green is a CI fact, not a runtime computation.** The
   runner scores the mel/WER/roundtrip half only. "parity CI 通過" is recorded
   as the parity gate's run URL + verdict per model, by the owner.
3. **The runner never claims "item 2 satisfied".** Its strongest verdict is
   `MEASURED-GREEN` = "the measurable half is green"; the final call still needs
   the parity evidence and the owner's judgment.

## How to produce each item's material

| Item | Material producer | Command |
|---|---|---|
| 2 (zoo degradation) | `vokra_eval::dod` runner | `cargo run -p vokra-eval -- dod [--corpus-root <dir>] [--utmos-gguf <f>]` |
| 3 (release train) | X-07 cadence mechanism (SoT) | X-07-T20 `.github/workflows/release-cadence.yml` + X-07-T22 `tools/release/test_cadence.py` — **not re-implemented here** |
| 4 (Kill switch A–L) | `kill-switch-metrics.sh` | `bash scripts/kill-switch-metrics.sh > <metrics>.json` → `dod_item4_kill_switch` |
| 5 (external committers) | `kill-switch-metrics.sh` | same JSON → `contributors_excluding_owner` / `dod_item5` |
| 1 (phase acceptance) | owner (EU / NPU / Cortex-M55) | see item 1 below |
| completeness | `check-zoo-manifest-complete.sh` | `bash scripts/check-zoo-manifest-complete.sh` |

## DoD five-item judgment

`docs/milestones.md` §9 / `deliverables.md` §7 — all five must hold for a GA
declaration.

### Item 1 — every phase's acceptance criteria met (owner)

The `system-requirements.md` §6 phase criteria: NPU **2× speedup** (M5-01/02,
measured against the M5-14 CPU baseline), **Cortex-M55 Silero VAD** (M5-03),
**EU AI Act certification** (M5-10), commercial GA. All are owner-verified on
real hardware / with legal sign-off — no mechanism produces this.

- **Verdict**: _(記入: 満たす / 満たさない / 判定時期前)_
- **Evidence** (per criterion): _(記入 — NPU bakeoff numbers, Cortex-M55 run, EU cert reference)_

### Item 2 — zoo models all under the 5% gate + parity CI green

Run `cargo run -p vokra-eval -- dod --utmos-gguf <vokra.utmos.*.gguf>` with each
model's `VOKRA_*_GGUF` set and its eval corpus in place. Paste the runner's
report; the verdict line is the runner's, not item 2's.

- **Runner verdict** (`MEASURED-GREEN` / `INCOMPLETE` / `MEASURED-FAILURES`): _(記入)_
- **Coverage tally** (passed / failed / skipped_no_weight / skipped_no_corpus / parity_pointer / excluded): _(記入)_
- **UTMOS leg** (active / not run): _(記入 — if "not run", item 2's UTMOS half is UNMEASURED, not passed)_
- **Per-model numerical-parity CI** (run URL + verdict per gated model, owner-recorded):

  | Model | Parity gate | CI run URL | Verdict |
  |---|---|---|---|
  | _(記入)_ | _(from manifest `parity_gate`)_ | _(記入)_ | _(記入)_ |

- **Completeness** (`check-zoo-manifest-complete.sh` output — every ★/⚠ zoo row accounted for): _(記入)_
- **Item 2 verdict** (owner): _(記入 — requires MEASURED-GREEN **and** every parity gate green **and** the UTMOS leg run for in-distribution TTS)_

### Item 3 — distribution on a stable 4-week release train

The **cadence mechanism is X-07's** (X-07-T20 `release-cadence.yml` computes
"is the next release due at 28 days"; X-07-T22 `test_cadence.py` is the
threshold oracle). This WP does **not** re-implement the 28-day definition or
the tag/release reading — doing so would split one judgment across two places
and drift. Record item 3 from X-07's output + the go-nogo record.

- **X-07 cadence mechanism landed?**: _(記入 — as of M5-12 authoring: NOT landed)_
- **Releases published** (`git tag` / GitHub releases): _(記入 — as of M5-12 authoring: 0; cadence not-established)_
- **Release-interval history** (dates, gaps ≤ 28 days for stable operation): _(記入)_
- **Item 3 verdict** (owner): _(記入 — "stable operation" cannot be true with 0 releases)_

### Item 4 — none of Kill switch A–L has fired

Material: `kill-switch-metrics.sh` → `dod_item4_kill_switch` (C/D/K computed;
A/B/E/F/G/H/I/J/L are competitor-changelog owner judgments and are emitted as
`owner-judgment-required`, never fabricated — FR-EX-08). **The A–L verdict table
is not duplicated here** — record each switch in
`docs/governance/vokra-go-nogo-v1.0-ga.md` (X-05-T17).

- **Any switch fired?**: _(記入: いいえ / はい — which)_
- **Reference**: go-nogo record §"Kill switch A–L status"
- **Start-date basis for C/D** (no `v0.5.0` tag exists; timing undefined — X-05-T23): _(記入)_

### Item 5 — ≥ 3 committers other than Claude Code; not owner-dependent

Material: `kill-switch-metrics.sh` → `contributors_excluding_owner` /
`dod_item5.external_committers`. This is the **owner-excluded** count (external
committers only), which is what "community not dependent on the maintainer
alone" requires. The Kill switch D input (`contributors_non_bot_non_cc`) still
*includes* the owner; **whether the threshold-of-3 should itself exclude the
owner is an owner decision (X-05-T21/T23)** — the metrics JSON surfaces both and
decides neither.

- **external_committers**: _(記入 — as of authoring, real API returns 0 external / `ayutaz` only)_
- **Threshold**: 3
- **Item 5 verdict** (owner, stating which count and threshold interpretation was used): _(記入)_

## Overall GA Definition-of-Done verdict

- **All five items met?**: _(記入: はい / いいえ)_
- **GA declaration**: _(記入: 宣言する / 保留)_
- **Reasoning**: _(記入)_

## Zero-dep / provenance note

The item-2 runner, the metrics collector, and the completeness gate are all
first-party (`vokra-eval` std-only Rust; bash + python3 stdlib) — no external
crate or tool enters the runtime (NFR-DS-02). The zoo manifest's `license`
column is cross-referenced against `docs/license-audit.md` in both directions
(NFR-LC-04): `check-zoo-manifest-complete.sh` proves every ★/⚠ zoo row is
accounted for and no manifest record is a phantom or a CC-BY-NC weight.

## Publication

Like the go-nogo records, completed DoD judgments default to publication in this
public `docs/governance/` tree (develop-in-the-open; a governance output nobody
can verify from outside is not a governance output). **The maintainer decides
per record**; cite conclusions rather than unpublished planning material.
