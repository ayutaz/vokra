# SBV2 v2 Blocker 2b/2c/3/5 Verification Wave — Owner Handoff

Tracked / public. Honest summary of what landed on
`feat/sbv2-v2-blockers-2b-2c-3-2026-08-11` (23 tasks, Wave 0–4) vs what the
plan intended, plus one CI-workflow finding from this wave's own T20 that
was found and then resolved within the same branch — see the "Update
(2026-08-12, T23 close-out)" note right below for the current state
before reading the rest of this doc.

- **Date**: 2026-08-11
- **Branch**: `feat/sbv2-v2-blockers-2b-2c-3-2026-08-11`
- **HEAD**: `3d0fd18` (T21, abi-changelog entry)
- **Base**: `main` (`0937ef8` = PR #27 merge tip, "fix(voxtral+sbv2) + ci(coverage-audit): 3 blockers + 12 gaps wired")
- **Commit count**: 24 (`git rev-list --count main..HEAD`)
- **Diff stat**: 23 files changed, **+3,073 / −906** lines (`git diff --stat main..HEAD`)
- **Plan**: `docs/superpowers/plans/2026-08-11-sbv2-v2-blockers-2b-2c-3.md` (gitignore local)
- **Design spec**: `docs/superpowers/specs/2026-08-11-sbv2-v2-blockers-2b-2c-3-design.md` (gitignore local)
- **SDD ledger**: `.superpowers/sdd/2026-08-11-sbv2-v2-blockers-2b-2c-3/progress.md` (21/21 tasks complete, review-clean or approved-with-concerns on every task)
- **Prior handoffs this wave builds on**: `docs/handoff/sbv2-v2-phase1.md` (Phase 1, 43 tasks), `docs/handoff/sbv2-sdp-debug-2026-08-08.md`, `docs/handoff/sbv2-bug4-resolved-2026-08-09.md`

> **Update (2026-08-12, T23 close-out)**: the "CI workflow — what changed,
> and what needs a decision" finding below (and the matching "Owner /
> next-wave tasks" item 2 and "Non-goals" bullet) is **resolved** as of
> this update. Between T22 (this doc, HEAD `3d0fd18`) and T23 (final
> verify), **the owner ruled option (a)** from that section (SDD ledger:
> "Task 20 REGRESSION FIX (2026-08-12, owner ruled A = revert)"): commit `aaf83ed`
> reverts T20's replacement workflow (`43ca0d5`) outright, restoring
> `main`'s pre-existing 727-line pipeline with its already-open sidecar
> gate; commit `c0e6259` then re-adds the one genuinely new trigger the
> branch needs (`tools/parity/vendor/vits2/**` in the PR path filter,
> since that vendor directory postdates the workflow `main` already had).
> **No further owner decision on the CI workflow is needed** — the
> real download/convert/dump/parity pipeline is back, unchanged from
> `main` except for that one added path-filter line. The section below is
> kept as-written (T22's own investigation) because it explains *why*
> option (a) was the right call; treat every "needs a decision" /
> "not repaired by this task" phrasing in it as **historical**, not
> current. Current branch stats at T23 close-out: **HEAD `f46cd40`**,
> **28 commits** ahead of `main` (`git rev-list --count main..HEAD`),
> **24 files changed, +3,315 / −252 lines** (`git diff --stat main..HEAD`)
> — the delta from T22's snapshot above is exactly the revert + two
> small fixups (path-filter re-add, and a T23 clippy doc-comment fix in
> `parity_sbv2_real.rs`, both content-neutral). `parity_sbv2_real` was
> re-verified at `f46cd40` and is still **12/12 stages PASS**, byte-identical
> to every prior run recorded in this doc.

## Executive summary

- **All 4 named blockers (2b, 2c, 3, 5) are closed.** Blocker 3 required a
  real design decision (ADR, ratified via clean-room investigation);
  Blockers 2b/5 required real code (vendoring + verification tests);
  Blocker 2c required no code at all — the plan's premise held under test.
- **`parity_sbv2_real` is un-`#[ignore]`d and green**: 12/12 aggregated
  stages PASS against the real 3-checkpoint fixture, byte-identical across
  three independent runs (T6 baseline, T16 post-Wave-1/2/3-partial
  regeneration, T18 final verify) — zero regression introduced by this
  wave's own changes.
- **Zero new C ABI.** `scripts/gen-c-abi.sh --check` and
  `scripts/check-abi-changelog.sh` both pass; the v1.0-rc baseline (33
  exported functions + 11 typedefs) is unchanged. Every change is Rust
  surface (test visibility, test bodies, Python dumper, one CI workflow,
  two doc files).
- **zero-dep NFR-DS-02 preserved** — root `Cargo.lock` untouched throughout.
- **One finding needs owner attention before "dispatch the workflow" is a
  useful next step**: T20's replacement of `.github/workflows/parity-sbv2-real.yml`
  gates on a sidecar file (`reference_dump.manifest.json.sha256`) that T16
  — earlier in this *same* branch — explicitly decided not to create, and
  hard-blocks the actual test run behind an unconditional `exit 78` +
  `if: false`. The pre-existing workflow on `main` (which this wave
  replaced) already had its gate **open** (real per-checkpoint sidecars
  committed 2026-08-09) and was mid-flight on real HF-download-based
  parity runs. See "CI workflow — what changed, and what needs a decision"
  below; this is not a regression this task is authorized to fix, only to
  report.

## What landed, by blocker

### Blocker 2b — VITS2 flow reference (T1–T6, T15–T17)

`tools/parity/vendor/vits2/` is a new clean-room MIT vendor of
[`p0p4k/vits2_pytorch`](https://github.com/p0p4k/vits2_pytorch)
(commit pinned in the vendored `README.md`), landed across T1 (scaffold)
and T2 (content, 3 review-fix rounds for scope creep / package-import
breakage / an orphaned duplicate docstring — see ledger Task 2 entries).
6 files, all under `tools/parity/vendor/vits2/`: `LICENSE`,
`README.md`, `attentions.py` (374 lines), `commons.py` (174 lines),
`models.py` (162 lines), `modules.py` (144 lines).

T15 extended `tools/parity/sbv2_dump_reference.py` to use the vendored
VITS2 `TransformerCouplingLayer` and emit 4 new per-layer flow tensors
(`reference_dump/flow_layer_{0..3}_output.bin`) plus a `flow_layers`
sibling block in `reference_dump.manifest.json`, alongside strengthened
determinism (CUBLAS workspace config, `torch.use_deterministic_algorithms`,
numpy + `random` seeding). T16 regenerated the fixture and confirmed the
change was purely additive to the manifest (`+39/−1` lines, all 11
pre-existing `tensors[]` entries byte-identical). T17 pinned the resulting
"structurally-ready-but-inert" state with a snapshot test (new test suite
total 9, was 8) rather than fabricating a calibration for tensors nothing
reads yet.

**Known gap, explicitly scoped out of this wave**: the 4
`flow_layer_{0..3}_output.bin` files are written to disk by the Python
dumper but **`crates/vokra-models/tests/parity_sbv2_real.rs` has zero code
paths that read them** — no `find_tensor` loop over `manifest.flow_layers`,
no `tolerance_for` arm, no `StageResult` push. `grep -n "flow_layer"
crates/vokra-models/tests/parity_sbv2_real.rs` returns nothing, confirmed
independently twice (T16's own session and the resuming agent that
finished T16 after a session-limit interruption — see
`docs/handoff/sbv2-parity-baseline-2026-08-11.md` for the full trace).
`flow_layer_3_output.bin` is sha256-identical to `z_latent.bin`
(confirms the dumper's `reverse=True` execution-order convention on the
real checkpoint, not just the T15 synthetic unit test). Wiring these into
the Rust assertion set is the concrete Wave-4 follow-up item — see
"Owner / next-wave tasks" below.

### Blocker 2c — SDP (no code changes)

The plan's own premise (written before T1) was that VITS1's stochastic
duration predictor — already vendored in Phase 1 as
`tools/parity/vendor/vits/sdp.py` and consumed by the dumper — is
architecturally equivalent to VITS2's SDP module, so no new vendoring was
needed for this blocker. This wave did not independently re-derive that
equivalence from upstream source; it inherited the premise and checked
for regression. That check held: `sdp_sample`'s max\|Δ\| was `0.0` against
its `atol=0.05` bound (status `EstimatedPreFixture`, not yet `Measured` —
see the parity table below) identically at T6, T16, and T18. Rust-side
`SbV2SDP` / `DDSConv` / `ConvFlow` / `ElementwiseAffine` primitives are
untouched by this branch.

### Blocker 3 + SBV2-SPK-EMB-LINEAR-DECISION (T12–T14)

T12 investigated the three options for what `ExternalSpeakerProjection`
should do at inference time and decided **(c): inference no-op** — the
projection is computed but its output is discarded at synthesize step 5,
never contributing to `text_hidden` or `bert_bridge_out`. The decision
record is `docs/adr/sbv2-spk-emb-linear-decision.md` (gitignored local,
20 KB, citations cross-checked against the MIT-vendored sources and the
T6 12/12 baseline). T13 applied the decision to `crates/vokra-models/src/sbv2/mod.rs`
as a comment-only diff (the discard code, `let _ = projected;`, already
existed — the ADR ratifies existing behavior rather than changing it,
so this was zero regression risk by construction). T14 added
`projection_output_is_discarded_per_adr_c` to
`crates/vokra-models/tests/sbv2_speaker_external.rs`, which feeds two
different external speaker embeddings through the pipeline and asserts
`text_hidden` and `bert_bridge_out` are **identical** — locking the ADR
decision against future accidental re-wiring. Revisit trigger recorded in
the ADR: T15's per-layer flow dumps or a UTMOS delta > 0.1 once UTMOS is
actually measured (it has not been — see below).

### Blocker 5 — BERT tokenizer scheme dispatch (T7–T11)

The SentencePiece / WordPiece parsers themselves
(`crates/vokra-convert/src/spm_proto.rs` and
`crates/vokra-bert/src/wordpiece.rs`) and the
`vokra.bert.tokenizer.*` GGUF metadata were **already implemented in
Phase 1** — this wave's job was verification, not implementation, and the
ledger records it that way ("Wave 1 SEALED = Blocker 5 fully verified").
T7 added 2 synthetic SentencePiece roundtrip tests
(`crates/vokra-bert/tests/tokenizer_roundtrip.rs`). T8 added 1 real
DeBERTa v3 GGUF SentencePiece-metadata load test
(`crates/vokra-bert/tests/deberta_v3_real.rs`, 128k-piece vocab, env-gated
on the real fixture). T9 added 2 synthetic WordPiece (`bert-charsplit`)
tests plus 1 real DeBERTa v2 GGUF metadata load test
(`crates/vokra-bert/tests/deberta_v2_loader.rs`). T10 enhanced an existing
test (`deberta_v3_missing_spm_model`) with a scores-field assertion rather
than adding a new one. T11 is the capstone: 1 new integration test,
`sbv2_model_from_gguf_dispatches_both_bert_tokenizers`
(`crates/vokra-models/tests/sbv2_gguf_loader.rs`), confirming JA
(`bert-charsplit`, DeBERTa v2) and EN (`sentencepiece-unigram`, DeBERTa v3)
both dispatch correctly through the **production** `SbV2Model::from_gguf`
path, not a synthetic stand-in.

### `parity_sbv2_real` un-ignored + all-stage aggregation (T6, T18)

T6 removed the test's `#[ignore]` attribute (the 3 real GGUF checkpoints
and the real `reference_dump.manifest.json` + 15 `reference_dump/*.bin`
files were already on disk from earlier fixture work — see the
`sha256` sidecar refresh in T5) and refactored
`diff_intermediates_against_manifest` plus the waveform / mel-loss checks
from N separate early-exit `assert!`s into one `Vec<StageResult>`
aggregation asserted once at the end. First real run: **12 of 12 stages
PASS, 0 failing** — the opposite of the "0 stages passing" scenario the
task brief had flagged as an escalation trigger. T18 is the Wave-3 seal:
full `parity_sbv2_real` + all 5 sbv2 test suites (76 tests) + `cargo fmt`
+ `cargo clippy --workspace --all-targets -D warnings` + zero-dep +
abi-changelog, all green, closed with an empty marker commit (`5941186`).

## Baseline parity — 12/12 stages, byte-identical across 3 independent runs

From `docs/handoff/sbv2-parity-baseline-2026-08-11.md` (T6 first run, T16
regeneration after Wave 1+2+3-partial, and the T16 resuming-agent's
independent re-run) and the T18 verify run — every `max|Δ|` value below
reproduced byte-identical across all measured runs. Fixture:
`request.text = "テスト"`, `language = JA`, `speaker_id = 0`, all-zero
`style_vec`, `seed = 42`.

| # | stage | max\|Δ\| | atol | calibration status |
|---|-------|----------|------|---------------------|
| 1 | phoneme_embed | 0.0 | 0.01 | UnmeasuredDefault |
| 2 | text_hidden | 5.513430e-7 | 0.01 | UnmeasuredDefault |
| 3 | bert_hidden_ja | 2.472854e-2 | 0.05 | **Measured** |
| 4 | bert_bridge_out | 3.294295e-2 | 0.07 | **Measured** |
| 5 | speaker_embed | 0.0 | 0.01 | UnmeasuredDefault |
| 6 | style_projected | 0.0 | 0.01 | UnmeasuredDefault |
| 7 | sdp_sample | 0.0 | 0.05 | EstimatedPreFixture |
| 8 | mel_hidden | 3.294295e-2 | 0.07 | **Measured** |
| 9 | z_latent | 3.592241e-2 | 0.08 | **Measured** |
| 10 | waveform_length_band | 0.0 (27136==27136 samples) | 0.10 | N/A (pseudo-stage) |
| 11 | waveform | 5.220508e-2 (RMS 8.155065e-3) | 1.5 | **Measured** |
| 12 | mel_loss | 2.003741e-1 | 0.3 | UNPINNED (not a manifest tensor) |

UTMOS quality gate: **SKIPPED** at every run (`VOKRA_SBV2_UTMOS_ENABLE`
never set locally) — an explicit FR-EX-08 skip, not a fabricated pass.
`bert_hidden_en` never appears because this fixture's `request.language`
is JA-only; `to_dumper_map()` omits the inactive-language BERT bucket.

**Two honest caveats carried forward from the baseline doc**: (1) this is
one input/checkpoint/seed combination — a green run here does not by
itself prove EN/ZH text, non-zero style vectors, non-deterministic noise,
or other speakers are correct; (2) half the atols above are still
`UnmeasuredDefault` / `EstimatedPreFixture` / `UNPINNED`, not `Measured` —
a PASS at those bounds is real but weaker evidence than a PASS at a
`Measured` bound (see `crates/vokra-models/src/sbv2/parity.rs`'s module
doc, or `docs/adr/sbv2-parity-atol.md` §5–§6, for the promotion procedure).

## CI workflow — what changed, and what needs a decision

> **Resolved 2026-08-12 (T23), see the update note at the top of this
> doc.** Option (a) below was chosen: `aaf83ed` reverted T20's workflow
> replacement, `c0e6259` re-added the one needed new path-filter line.
> The rest of this section is preserved as T22's original investigation.

T20 replaced `.github/workflows/parity-sbv2-real.yml` (**+129 / −654
lines** on that one file — this was a rewrite of an existing file, not a
new addition, despite the ledger template describing it as "add"). Two
facts worth knowing before treating "dispatch the workflow" as a
meaningful next step:

1. **The workflow this wave replaced was not a stub.** It was 727 lines,
   landed in Phase 1 (PR #22) and iterated across 3 follow-up fix commits
   already on `main` before this branch started
   (`c70f18b` — add sentencepiece + protobuf to the parity venv,
   `82082a3` — 2nd-order root-cause fix, `0b1746e` — root-cause fix). Its
   gate was **already open**: `tests/fixtures/sbv2/README.md` records
   (commit `6580061`, 2026-08-09, predating this branch) that all three
   per-checkpoint `.gguf.sha256` sidecars carry real `sha256sum` output,
   and the old workflow's `setup` job gated on exactly those three files.
   Its `parity` job did a real `snapshot_download` of the SBV2 base +
   2 DeBERTa checkpoints, converted them, ran the dumper, and ran the
   real Rust parity test — the same shape as `parity-deberta-v3-large-real.yml`
   and `parity-kokoro-real.yml` (both of which do real in-runner HF
   downloads; verified by reading `parity-deberta-v3-large-real.yml`
   directly, which has no `exit 78` / `if: false` placeholder anywhere).
2. **The new workflow's gate is a different file that was deliberately
   never created.** T20's `setup` job checks
   `tests/fixtures/sbv2/reference_dump.manifest.json.sha256`. T16 — 4
   tasks earlier in this same branch — investigated exactly this file and
   documented, citing three independent sources (the fixtures README, the
   root `.gitignore`, and the old workflow's own gate logic), that no such
   sidecar exists anywhere in this repo's history and nothing consumes
   one; T16 deliberately did not create it. T20's gate therefore evaluates
   `present=false` unconditionally today and will keep doing so until
   someone creates a file T16 concluded shouldn't exist. Independently of
   that, the new workflow's "Run parity_sbv2_real" step carries
   `if: false  # unreachable until fixture provisioning step is settled`,
   and the step before it ("Provision fixtures") unconditionally
   `exit 78`s — so even if the gate opened, the parity leg still would
   not run without a further code change.

Nothing above is asserted to be a mistake — it may be a deliberate choice
to defer real-fixture CI provisioning to an explicit owner decision (the
T20 task report frames the `exit 78` step exactly that way: "LFS /
release artifact / self-hosted runner, owner responsibility"). But this
wave's own change is the reason `workflow_dispatch` today will show
`setup` green and `parity` **skipped**, not a meaningful CI run — the
previously-working real pipeline is gone, not just gated. The choice for
the owner is between (a) restoring the pre-existing 727-line pipeline's
real download/convert/dump/parity steps into the new skeleton, gated on
the three sidecars that are already real (matching every sibling
`parity-*-real.yml` workflow's actual behavior), or (b) keeping the
current deferred/placeholder state and deciding the fixture-provisioning
question from scratch, with the understanding that it is starting over
rather than finishing something already in flight.

## Owner / next-wave tasks

1. **PR review + merge** — branch is 28 commits off `main` (`0937ef8`,
   PR #27 tip); no conflicts expected with concurrent work.
2. ~~**CI workflow decision**~~ — **resolved 2026-08-12 (T23)**, see the
   update note at the top of this doc. `aaf83ed` + `c0e6259` already
   restored the pre-existing pipeline; nothing left to decide here. The
   three per-checkpoint `.gguf.sha256` sidecars remain real (committed
   2026-08-09, predating this branch), so `workflow_dispatch` is ready to
   run for real today.
3. **§3.1 license sign-off — already complete, no action needed.** Rows
   315 (`sbv2-v2-jp-extra-base`, AGPL-3.0 → Copyleft, signed 2026-07-28),
   316 (`deberta-v2-large-japanese-char-wwm`, CC-BY-SA-4.0 → Copyleft
   ShareAlike, signed 2026-08-06 — flipped from an earlier 2026-07-27
   Rejected once the owner delegated the SA-cascade decision to CC), and
   317 (`deberta-v3-large`, MIT → Permissive, signed 2026-07-27) are all
   `☑ Commercial` in `docs/license-audit.md` already. This branch touches
   none of the three checkpoints' license status.
4. **Wire `flow_layer_{0..3}_output` into the Rust assertion set** — the
   concrete, scoped-out-of-this-wave follow-up from Blocker 2b (T15–T17).
   Needs: a loop over `manifest.flow_layers` (a dict, not the fixed
   11-entry `tensors[]` list), a `tolerance_for` arm or parallel
   calibration table for `flow_layer_N_output`, and a `StageResult` push
   per layer — same pattern the existing `tensors[]` loop already uses,
   generalized to variable length. Nothing to calibrate against exists in
   the Rust harness until this wiring lands.
5. **Real end-to-end `vokra-cli run <sbv2.gguf> --text "..."` synthesis
   still hits `VokraError::NotImplemented`.** `SbV2Model::from_gguf` (the
   3-arg constructor `vokra-cli run` uses) installs `UnwiredPhonemizer`
   for both JA and EN, which loudly rejects every phonemize call by
   design (`crates/vokra-models/src/sbv2/mod.rs`). This is unchanged by
   this branch — Phase 1's follow-up added
   `SbV2Model::from_gguf_with_phonemizer` (a caller-supplied
   `Phonemizer` impl), which is what `parity_sbv2_real` itself uses via a
   fixture-loaded synthetic G2P, but no CLI flag exposes a real phonemizer
   (e.g. piper-plus G2P) through it yet. If real text-to-speech from the
   CLI is in scope for a future wave, this is the gap to close — not the
   `encode_spm`/`todo!()` framing an earlier planning note used (no
   `encode_spm` function exists anywhere in this codebase; grepped and
   confirmed absent).
6. **UTMOS gate real-GGUF provisioning** — env-gated opt-in
   (`VOKRA_SBV2_UTMOS_ENABLE=1` + `VOKRA_SBV2_UTMOS_GGUF=<path>`), pinned
   by 3 new tests in T19 (`utmos_gate_settings` module:
   disabled-when-unset / loud-panic-when-partially-set / enabled-when-both-set).
   Not required for the 12/12 parity baseline above — `mel_loss` (an
   architectural-bound spectral proxy) is in the assertion set already;
   UTMOS is a separate perceptual-quality gate that has never actually
   been measured on this checkpoint (no UTMOS GGUF present locally).
7. **Real device GPU (Metal / CUDA) SBV2 forward** — out of this branch's
   scope; falls under the M3-18-series real-hardware verification track.

## Numbers

- **Fixture size**: 3 real GGUF checkpoints total **≈2.78 GB**
  (`sbv2-v2-multilingual-base.gguf` 246.7 MB + `deberta-v2-large-japanese-char-wwm.gguf`
  1.56 GB + `deberta-v3-large.gguf` 971 MB), all gitignored. (A fourth,
  unrelated `chinese-roberta-wwm-ext-large.gguf` — 1.3 GB, ZH BERT — also
  sits in `tests/fixtures/sbv2/` from a separate WP; not touched by this
  branch.) `reference_dump/` holds 19 gitignored `.bin` files (15 from
  Phase 1 + 4 new `flow_layer_{0..3}_output.bin` from T15).
- **Test additions**: 12 new test functions verified directly against
  task reports — T7 (2, SPM roundtrip) / T8 (1, real DeBERTa v3 SPM load)
  / T9 (3, WordPiece + real DeBERTa v2 load) / T11 (1, dual-scheme
  dispatch via production loader) / T14 (1, ADR-(c) regression pin) / T17
  (1, flow_layers inert-state snapshot) / T19 (3, UTMOS gate env
  resolution). T6 and T18 add no new test functions (un-ignore +
  aggregation refactor; verify-only marker, respectively). T10 enhances
  an existing test rather than adding one.
- **New C ABI**: **0** (Rust surface only; `gen-c-abi.sh --check` and
  `check-abi-changelog.sh` both green at HEAD).
- **Vendored MIT source**: 6 files under `tools/parity/vendor/vits2/`
  (`LICENSE`, `README.md`, `attentions.py`, `commons.py`, `models.py`,
  `modules.py`).
- **zero-dep NFR-DS-02**: preserved — root `Cargo.lock` unchanged
  (`scripts/check-zero-deps.sh` clean at every task's pre-commit hook run).
- **Baseline parity**: 12/12 stages PASS, byte-identical across T6 / T16
  (+ T16's independent resuming-agent re-run) / T18 — zero regression
  introduced across Waves 1–3.
- **Full workspace verify (T18)**: `cargo test -p vokra-models --lib` →
  1653 passed / 0 failed / 1 ignored; sbv2-specific suites → 76 passed
  across 5 test binaries (0 failed, 2 intentional `#[ignore]`s pending
  real fixtures unrelated to this wave); `cargo fmt --check` clean;
  `cargo clippy --workspace --all-targets -- -D warnings` clean
  (~10 min).

## Non-goals for this wave (do not re-open without a reason)

- Re-deriving VITS1≈VITS2 SDP equivalence from upstream source — accepted
  as a plan-level premise, not independently re-verified here.
- Any change to `SbV2SDP` / `DDSConv` / `ConvFlow` / `ElementwiseAffine`,
  `SbV2TransformerCouplingLayer`, or the HiFi-GAN decoder — all untouched.
- Real GPU (Metal/CUDA) SBV2 forward — M3-18-series scope.
- CLI-level real G2P wiring (piper-plus or otherwise) for
  `vokra-cli run <sbv2.gguf>` — item 5 above, explicitly out of scope for
  this wave, not fixed here.
- ~~Fixing the T20 CI workflow gate~~ — was out of scope for Task 22
  (documentation only) but **was fixed** before T23 close-out via the
  `aaf83ed` revert + `c0e6259` fixup; see the update note at the top of
  this doc. Listed here only to avoid re-opening a question that is
  already settled.

## References

- `docs/handoff/sbv2-v2-phase1.md` — Phase 1 (43 tasks): SBV2 v2 model
  core, converter, CLI wiring, first `parity_sbv2_real` (fixture-gated
  `#[ignore]`d), original `parity-sbv2-real.yml`.
- `docs/handoff/sbv2-parity-baseline-2026-08-11.md` — full raw parity
  logs for T6, T16, and the T16 resumption run (gitignored local, but the
  numbers are reproduced in this doc's table above).
- `docs/handoff/sbv2-sdp-debug-2026-08-08.md`,
  `docs/handoff/sbv2-bug4-resolved-2026-08-09.md` — prior debugging
  history this wave's clean baseline builds on.
- `docs/adr/sbv2-spk-emb-linear-decision.md` (gitignored local) — Blocker
  3 decision record (T12).
- `docs/adr/sbv2-parity-atol.md`, `docs/adr/sbv2-cleanroom.md`,
  `docs/adr/sbv2-libm-strategy.md` (gitignored local) — atol calibration
  procedure, clean-room source policy, and the `ceil()` duration-rounding
  bound referenced in the parity table's `waveform_length_band` row.
- `docs/license-audit.md` §3.1 rows 315–317 — SBV2 v2 checkpoint sign-off.
- `.superpowers/sdd/2026-08-11-sbv2-v2-blockers-2b-2c-3/progress.md` —
  full 21-task ledger with review verdicts.
