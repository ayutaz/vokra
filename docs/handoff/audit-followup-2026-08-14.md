# Audit-followup campaign — owner handoff (2026-08-14 / 15 / 16)

**Branch**: `feat/audit-followup-cc-wave1-2026-08-14`
**Scope**: 109 implementation + audit commits ahead of `main`, plus the
documentation refresh of 2026-08-16 (§8).
**PR**: not yet opened — see §6.
**Status**: CC-actionable work is **terminal** (14 audit rounds; rounds 13 and
14 both found nothing to fix, which is this project's two-consecutive-zero
rule).

This document is for the owner. It says what landed, what the campaign found
that nobody had asked it to look for, and what is left — which is entirely
owner-gated.

---

## 1. What this campaign was

It started as "investigate whether any supportable models or features are
still missing", ran nine implementation waves, and then — because the project
discipline says to re-audit after implementing — ran fourteen audit rounds
over its own output.

The audit rounds turned out to matter more than the waves. Ten of the first
eleven found a **high-severity** defect, and most of those defects predated
this campaign or were introduced by it and invisible to every existing check.

## 2. Implementation waves (22 commits, this session)

| Wave | What landed |
|---|---|
| A | WPE dereverberation op; SI-SNR / SI-SDR / SDR / STOI metrics; utmosv2 / nisqa / squim binders |
| B | TEN-VAD / FireRed-VAD / smart-turn binders; strict arch-tag verification added to 8 legacy binders |
| C1 | 5 ASR binders (canary-1b-flash, parakeet-tdt-1.1b, gigaam ×2, whisper-medusa, firered-aed) |
| C2 | 5 SSL encoder binders (atst, eat, m2d, maest, w2v-bert-2) |
| E | **Shared ViT audio-encoder primitive** + topology stamping in 4 SSL converters |
| D | 4 brand-new categories: AudioSR, DiffSinger, ITN/TN, CT-Transformer punctuation |
| F | SSL binders wired to the ViT primitive |
| G | Last 3 convert-only arches (chattts, deepfake_detection, lang_id) + CLI reachability registry |
| — | `vokra-cli f0` subcommand: YIN / PyIN reachable from a binary for the first time |

**Coverage result**: 26 converter arches produced GGUFs that nothing in the
workspace could read back. All 26 are now consumed.

**The most useful single finding of the wave phase** was not a model: several
binders were blocked for the *same* reason — `vokra-ops` had no 2-D patch
embedding + pre-norm Transformer encoder. One primitive now carries five of
them: ATST, EAT, M2D, MAEST and Beat-This all call `vokra_ops::vit` in code.
`conformer` / `ebranchformer` / `zipformer` are conv-augmented ASR encoders
over a 1-D frame sequence and could not substitute.

Worth stating precisely, because the first version of this paragraph did not:
**W2V-BERT-2 is not one of the five.** It is Conformer-based, and its forward
is deferred for an unrelated reason (`vokra_ops::conformer::PositionEncoding`
exposes only the variants it does). Grouping it with the ViT set would send a
reader to the wrong primitive.

## 3. What the audit rounds found

Ordered by how much they would have cost if shipped.

### Silent-wrong outputs

- **CREPE returned a fabricated pitch track.** With real weights and 44.1 kHz
  audio it returned an all-zero track — indistinguishable from "this audio is
  entirely unvoiced". Its docstring claimed the opposite ("a non-16 kHz caller
  is honest-refused"), for a function with no error channel at all. FCPE had
  the same shape. Wrong pitch flows into a vocoder and produces confidently
  wrong audio.
- **RMVPE silently skipped its entire U-Net.** A fork-convention checkpoint
  passed the loader, discovered zero blocks, and fed the raw mel plane
  straight to the BiGRU. Nothing distinguished "ran the full U-Net" from "ran
  none of it".
- **`KwsMicro::detect` returned `Ok(Idle)` when unconfigured** — and `Idle`
  legitimately means "no wake word in this frame", so an unconfigured detector
  was indistinguishable from a working one hearing silence.

### Converter/binder handshake failures (shipped twice)

- **openWakeWord**: the binder required seven `vokra.openwakeword.*` keys; the
  converter stamped none. Every GGUF it produced failed to load. The
  documented owner recipe dead-ended at the first step.
- **llama_omni2**: identical shape, ten keys, found one round later.

Neither was visible to the test suite: unit tests hand-build their GGUF with
`GgufBuilder`, and the parity harnesses are env-gated and skip. The repo now
has two convert-then-bind tests that run the real converter into the real
binder — the path that had never been exercised.

### Claims that had become false

Five consecutive rounds found this class, which is why it ended in gates
rather than sweeps:

- A **falsehood pinned by a test**: `beat_this` asserted "no shared MHA
  primitive exists in `vokra-ops`" after Wave E landed one, and a test
  asserted the message *contains* that phrase — so correcting it broke CI and
  leaving the lie was the cheapest action.
- `lib.rs` said a converter "stamps NO axes at all"; it stamps 38.
- `vokra_ops::qwen2` was cited as "already wired through voxtral / kyutai_stt
  / canary_qwen". It has never existed.
- **47 of 109 cited `tools/parity/*.py` bridges did not exist**, one of them
  making `parity_dnsmos` permanently vacuous: its recipe cited a dumper that
  was never written, so the test always skipped — and the harness's own doc
  says "a skip is never a fabricated pass", which is exactly what it had
  become.
- Three vast.ai runbooks named a provisioning script at the wrong path. The
  owner pays by the minute when that fails.

### Gates that were green because they did not look

The sharpest recurrence, twice in gates this campaign had just added:

- Both arch gates discovered binders with a regex matching `ARCH` but not
  `EXPECTED_ARCH` — **29 of 89 binders unexamined**, including `charsiu`,
  which had no converter at all and was exactly what they were written to
  catch.
- Leg (d) counted a key's **declaration** as proof it gets stamped. Deleting
  six of seven openWakeWord stamps left every number byte-identical.
- Leg (d) counted **negative tests as readers**: `"yamnet"` appears only
  inside a test asserting PANNs *refuses* a YAMNet GGUF. The ledger claimed 21
  known gaps against a real 47.

### CI that was red without anyone noticing

Three separate gates were failing at `HEAD` when the campaign found them:
`check-converter-signoff` (12 converters unregistered, so the fail-closed
publish gate was off for all 12), `check-platform-support` (the published
matrix cited four CI jobs at a path they had moved from), and
`check-crate-path-citations`.

Behind the first was a worse bug: **the DTLN-AEC §3.1 row contained unescaped
pipes**, so Markdown split one cell into five and shifted the approver and
decision columns. That row was invisible to the sign-off machinery — and would
have stayed invisible *after an owner ticked its box*.

## 4. Gates added (all CI-wired, all hard-fail)

Verified: every one is in `ci.yml`'s `license` job, fires on `pull_request` as
well as `push`, and none sits under `continue-on-error`.

| Gate | Invariant |
|---|---|
| `check-arch-handshake.sh` | converter ⇄ binder arch tags (legs a/b), `convert --model` slugs (leg c), and **metadata-key groups** (leg d) — 47 self-test cases |
| `check-bound-arch-coverage.sh` | every binder arch is routed or registered in the CLI |
| `check-crate-path-citations.sh` | every `vokra_<crate>::<path>` in prose resolves to a public item |
| `check-ops-path-citations.sh` | same for `vokra_ops::` specifically |
| `check-parity-sidecar-citations.sh` | every cited `tools/parity/*.py` exists, or its line is marked not-yet-written |
| `check-runbook-path-citations.sh` | every path an operational runbook tells the owner to run exists |

Each ledger is **double-sided**: an undeclared gap fails, and a ledger entry
whose gap has closed also fails. That is not decoration — a `voila`
NO_CONVERTER entry went stale mid-wave when the converter landed, and the gate
failed until it was deleted, exactly as its own comment predicted.

## 5. Verification state

Run per-crate with `CARGO_BUILD_JOBS=1` (this machine is a 16 GB M1 and
concurrent cargo OOMs it):

```
vokra-models  2532   vokra-convert 1082   vokra-ops 1007
vokra-cli      185   vokra-core     577   vokra-eval  158   kws-micro 85
doc tests       30                        all 0 failed
```

**Confirmed by a single full-workspace run** (2026-08-16), which the per-crate
sequence above cannot fully substitute for — it never proves the crates agree
in one consistent build:

```
cargo test --workspace  →  6965 passed / 0 failed / 23 ignored / 234 suites
                           exit 0;  fmt, clippy -D warnings, 10 shell gates,
                           doc-examples all green
```

That run happened on a rented 48-core / 125 GB box, not here. Attempting it
locally exhausted memory and **rebooted the machine**; the same sweep is what
the pre-push hook performs, so pushing a Rust change from this machine can
take it down. Two tests reported failing under that local sweep are healthy:
`csm_frame_loop_allocates_zero_after_open` passes in isolation (its allocation
counter is perturbed by neighbouring test threads), and the `kyutai_stt` tests
take **155 s** legitimately, so contention pushed them past the 180 s timeout.
Neither is a regression. The box cost $0.03 and was destroyed.

- `cargo clippy --all-targets -- -D warnings` clean on every crate, and on
  `-p vokra-cli --features vokra-wfst`.
- 12 gates green; `check-zero-deps` OK (root `Cargo.lock` still `vokra-*` only,
  NFR-DS-02).
- Release build succeeds and its output is identical to debug's on the JFK
  fixture (680 frames, 120 voiced).
- **No C ABI additions.** Baseline unchanged at 33 functions + 11 typedefs.
- **No §3.1 decision column filled anywhere.** Fail-closed default preserved.

Rounds 12–14 verified the tests adversarially rather than by counting them.
Disabling strict arch verification in four binders (`emotion2vec`, `ten_vad`,
`squim`, `chattts`) fails a test in all four. Deleting a real metadata stamp
fails leg (d) by name.

## 6. What is left — all owner-gated

Nothing here is CC-actionable; each needs a decision, a credential, hardware,
or a download.

1. **Open the PR.** 109 commits, not yet proposed. CC does not open PRs
   without being asked.
2. **§3.1 sign-off.** Many rows are blank by design, including 12 newly mapped
   converters and every model this campaign added. Publishing stays blocked
   until signed — that is the fail-closed default working.
3. **Real-checkpoint parity.** The loud-partial binders need real weights.
   Note one consequence: **RMVPE's tensor-name walker is itself unverified**,
   so the first real checkpoint may hit the new loud error. That is the
   intended outcome — the error prints the artifact's own tensor names, which
   is what settles the naming convention — and the fix is to correct the
   walker, not relax the gate.
4. **ELVIS Act ADR for ChatTTS.** Its 30-d `spk_emb` is seed-derived
   officially but technically substitutable. No speaker-embedding injection
   surface was built, deliberately: reporting whether a `spk_stat` group
   exists is the input that ADR needs; providing the injection path is the
   thing the ADR is about.
5. **vast.ai jobs**, **NPU bakeoff** (M5-01/02), **v1.0 GA tag + C ABI freeze**
   (M5-13).
6. **`docs/license-audit.md:495`** still repeats the retracted beat_this MHA
   claim in a prose cell. Left untouched on purpose — that row holds owner-only
   sign-off cells and editing it would read as audit tampering.

## 7. Two things worth carrying forward

**A gate that is green because it did not look is worse than no gate**: it
certifies the thing it failed to check. This recurred three times, twice
inside gates written after learning it. Every gate added here has a
`--self-test` that plants a defect where the gate must see it, and each was
proven by breaking real code and watching it fail by name.

**Tests that pin phrasing are not tests of behaviour.** The `f0` subcommand
shipped with eight passing tests that all stopped at argument parsing —
hard-coding "every frame is voiced" left all eight green. They now check the
output, and the same mutation fails two of them.

## 8. Documentation refresh (2026-08-16)

A separate instruction — "bring all the documentation up to date" — turned up
the same classes of defect one layer out, in the pages a newcomer reads first.

- **README.md / README.ja.md** described a narrower project than the one that
  exists. The opening still scoped Vokra to speech; music generation, source
  separation and audio understanding have been in scope since 2026-07-30. The
  roster was missing the ASR distillations and TTS models that landed
  2026-07-24, plus keyword spotting, vocoders, text normalization, punctuation
  restoration and diarization. RMVPE was still described as a loud partial
  awaiting verification — untrue since `e7b6810`. The 49 architectures whose
  forward is deferred now have their own heading rather than being mixed in
  with what runs.
- **`tools/docs/check_doc_examples.py` would have rejected correct
  documentation.** Its subcommand set was a literal `(run, convert, bench)`
  tuple, so it answered "subcommand 'f0' does not exist" for a subcommand that
  does, and could not have checked one of its flags. Derived from `main.rs`'s
  dispatch now; proven by planting `--bogus-flag` on an `f0` example and
  watching it fail by name.
- **`docs/architecture.md`** said `vokra-kws-micro` was scaffold-only and that
  `detect()` returns `Idle` unconditionally. Both halves had stopped being
  true, the second because returning `Idle` while unconfigured was itself the
  defect an audit round removed.
- **CONTRIBUTING.md** advertised three gates in the `license` required check,
  which runs fifteen.
- **CHANGELOG.md** had no entry after 2026-07-23, across six merged PRs.
- Also refreshed: the CLI tutorial (en/ja), `docs/deliverables.md`,
  `docs/requirement-ids.{md,ja.md}`, and `docs/milestones.md` §9.

One claim in this handoff was corrected in the process — see §2 on which five
binders actually share the ViT primitive. It was written from the wave
narrative rather than from the code, and the code disagreed.
