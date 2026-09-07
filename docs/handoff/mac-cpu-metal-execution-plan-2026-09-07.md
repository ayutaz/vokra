# Mac CPU / Metal completion execution plan (2026-09-07)

## Objective and authoritative inputs

Finish Mac CPU and Apple Metal support for every public Vokra model row.  This
plan preserves the full 63-row scope recorded in
`mac-pre-scaleway-remaining-tasks-2026-09-05.md`; a green build, an inspection
manifest, or a device-less Metal compile does not count as completion.

The starting point is PR #79 at `43188b948daa6de7a76d93e2d003d9b2f66da19c`:

- 194 public repositories, 193 GGUF-bearing repositories and 198 GGUF files;
- Mac CPU complete for 131 repositories;
- 43 partial, 19 no-runtime-binder and one non-artifact repository remain;
- zero CPU-complete repositories are classified as Metal-unsupported;
- all 16 required PR checks pass;
- the only local worktree modification is the owner's uncommitted
  `tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json`, which is
  outside this plan and must never be staged by campaign commits.

The 2026-09-05 ledger is the source of truth for per-repository facts and
historical evidence.  This document is the execution order and completion
matrix.  When the live audit changes, update both documents in the same
management commit.

## Non-negotiable execution boundary

- Never download or execute model weights on the maintainer Mac.
- Never run workspace-wide or `vokra-models` Cargo commands on the maintainer
  Mac.  Run those commands on a disposable VAST worker.
- Use only first-party runtime crates.  No `cargo add`, ONNX/ORT/protobuf
  runtime dependency, silent CPU fallback, or invented tensor/license/parity
  fact is allowed.
- Every reference must execute the pinned independent upstream
  implementation.  A handwritten mirror is not parity evidence.
- License and operator rows remain fail-closed until the owner explicitly
  signs the exact hash-bound scope.  Source inspection is evidence, not
  approval.
- VAST instances are rent-work-recover-small-evidence-destroy.  Do not use or
  modify unrelated `ralomi-*` instances.
- Scaleway is the final compute service.  Do not provision it until every
  non-Apple source, license, dependency, converter, binder, reference and VAST
  CPU leg is complete or has an explicit owner-approved withdrawal/withholding
  disposition.
- Hugging Face uploads and repository withdrawal are separate irreversible
  permissions.  Neither this plan nor a green Scaleway run authorizes them.

## Execution state machine

Every unresolved repository moves monotonically through these states:

1. `SOURCE_BLOCKED`: exact source, topology, dependency, license, dataset,
   tokenizer or codec facts are missing.
2. `SOURCE_READY`: strict converter/binder/native/CLI and independent-reference
   paths exist, with all unsupported paths failing loudly.
3. `APPROVAL_BLOCKED`: the exact approval scope is hash-bound, but the owner or
   legal decision is absent.
4. `VAST_READY`: approvals and model-free API/dependency probes are green.
5. `CPU_PASS_METAL_NOT_RUN`: exact-head no-upload conversion and independent
   CPU parity pass on VAST, and a complete direct-transfer packet exists.
6. `APPLE_PASS`: Apple CPU/reference, Metal/reference and Metal/CPU pass on
   Scaleway with no fallback.
7. `PUBLICATION_BLOCKED` or `COMPLETE`: corrected bytes await separate upload
   permission, or the upload/withdrawal and final live audit are complete.

No state may be promoted by a skip, zero-test selection, synthetic-only
fixture, build-only result, inspection-only manifest or unreviewed numerical
bound.

## Ordered work waves

### Wave 0 — protect the current baseline

Keep PR #79 green while later changes are built as small reviewable commits.
For every implementation slice: inspect the dirty tree, preserve the owner's
manifest, run focused model-free tests plus formatting/diff hygiene, then batch
workspace/Clippy/deny/audit verification on VAST before pushing.

Exit evidence: exact commit, reviewed diff, focused local no-model checks,
green VAST full-workspace checks, green required PR checks and no Vokra VAST
instance left running.

### Wave 1 — owner-independent source and API closure

Run these three independent lanes first:

| Lane | Repositories | Required result |
|---|---|---|
| MOSS Nano source contract | `moss-audio-tokenizer-nano` | Authenticate the fixed source files, official API path and tap shapes, or add a fail-closed VAST inspector that produces those facts without claiming parity. |
| SBV2 production Japanese G2P | `sbv2-v2-jp-extra-base` | Bind the exact phone vocabulary and upstream accent/tone mapping, wire the isolated G2P integration, and reject every unsupported or unauthenticated path. |
| Model-free API smokes | four Qwen3-TTS rows plus MOSS Audio 4B/8B | Make the pinned Transformers 5.10.4 no-checkpoint probes reproducible on VAST and bind their exact dependency/source identities. |

Then close the remaining owner-independent facts in this order:

1. Kyutai STT fixed-bound measurement, Mimi/tokenizer and streaming ASR.
2. FireRedASR sidecar binding, structural markers and official beam policy.
3. Zonos exact 246-tensor/conditioner/DAC contract.
4. Dia tokenizer/generation and delayed-AR/DAC contract.
5. CLAP reference lock, HTSAT preprocessing and state-dict role manifest.
6. OWSM frontend/MVN, 1,172-tensor map, writer and independent fixtures.
7. AudioGen T5/EnCodec identities; MMS backbone/adapter/vocabulary; XY
   tokenizer topology; Yue encoder; HT-Demucs ensemble/runtime.
8. Remaining complete composites in Waves 3 and 4 below.

Exit evidence for each slice: pinned primary-source identities, strict
metadata/tensor manifests, model-free self-tests, no hidden fallback, and an
exact command for the first no-upload VAST run.

### Wave 2 — exact owner/legal disposition packet

Prepare one review packet containing the exact approval-scope hash, primary
source evidence and proposed disposition for every fail-closed row.  Do not
convert a pending row into approval automatically.

The packet must cover at least:

- Qwen3-ASR model/operator/package rows;
- Qwen3-TTS and Parler/SpeechT5 fixed-revision license 404 residuals where
  still relevant to shared closures;
- CosyVoice2 HiFT source/model/reference execution;
- BigVGAN Linux/Darwin package, native-payload and model execution scope;
- MOSS Nano, MOSS Audio 4B/8B, MOSS-TTS Local and SBV2 execution scopes;
- BiCodec and all other non-commercial/research-only rows;
- Ultravox gated Meta companion; SpeechBrain, WeSpeaker, NSNet2 and other
  corrected public-artifact replacements;
- unresolved legal facts that cannot be coded away: RMVPE missing exact-source
  license, Conv-TasNet's contradictory model/dataset terms, HT-Demucs'
  Python-3.12/torchaudio and redistribution boundary, `dynet38` exact-release
  mapping, `qwen-omni-utils` wheel/source desynchronization, `soynlp` GPL/LGPL
  conflict and Triton wheel-bundled NVIDIA payloads;
- the SeamlessM4T choice to publish a real gated artifact or withdraw the
  empty repository.

Exit evidence: every pending execution has a single immutable decision record,
and every withheld/withdrawn row has an explicit owner decision.  Rows without
that evidence remain `APPROVAL_BLOCKED`; they are not silently dropped from
the 63-row denominator.

### Wave 3 — public-artifact repair and CPU parity (27 rows)

Run small family-specific VAST jobs after Waves 1 and 2.  Each job performs
no-upload conversion, independent reference generation, native CPU parity and
the exact-head repository gates, then immediately destroys its worker.

| Batch | Repositories |
|---|---|
| Ready replacement families | `bicodec`, `canary-1b-flash`, `canary-1b-v2`, `lang-id-voxlingua107`, `moss-tts-local-transformer-v1.5`, `nsnet2`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, four Qwen3-TTS rows, `reazonspeech-nemo-v2`, `speechbrain-spkrec-ecapa-voxceleb`, `voice-gender-classifier`, `wespeaker` |
| Wave-1-dependent replacement families | `moss-audio-tokenizer-nano`, `sbv2-v2-jp-extra-base`, `moss-audio-4b-instruct`, `moss-audio-8b-instruct` |
| Deeper source/runtime repair | `audiogen-medium`, `htdemucs-multi`, `mms-1b-all-base`, `xy-tokenizer`, `yue-xcodec-mini` |
| Legal-blocked replacement | `conv-tasnet-libri1mix`, `rmvpe` |

Exit evidence per row: canonical GGUF hash, independent reference packet,
fixed bound, singleton named CPU test result, complete packet manifest and
`CPU_PASS_METAL_NOT_RUN`.  Legal-blocked rows exit only through a signed
withholding/withdrawal decision or a newly authenticated permissive contract.

### Wave 4 — finish incomplete and missing native runtimes (35 rows)

Implement complete first-party routes in bounded family commits; inspection or
component-only surfaces are not sufficient.

| Batch | Repositories |
|---|---|
| Bound runtimes (19) | `audioldm2`, `audioldm2-large`, `canary-qwen-2.5b`, three Chatterbox rows, `chattts`, `clap-htsat-fused`, `cosyvoice2-0.5b`, `dia-1.6b`, `firered-asr-aed-l`, `fun-cosyvoice3-0.5b-2512`, `irodori-tts-500m-v3`, `kyutai-stt-2.6b-en`, `owsm-v4-medium-1b`, `sortformer-diar-4spk-v1`, `vibevoice-1.5b`, `voxcpm-0.5b`, `zonos-v0.1-transformer` |
| Missing binders/runtimes (14) | `ace-step-1.5`, `baichuan-audio`, `granite-speech-4.1-2b`, `hibiki-2b`, `kimi-audio`, `kyutai-tts-1.6b-en-fr`, `qwen2-5-omni-7b`, `qwen2-audio-7b-instruct`, `sgmse-voicebank`, `step-audio2-mini`, `vibevoice-asr`, `vibevoice-realtime-0.5b`, `vieneu-tts-v3-turbo`, `xtts-v2` |
| Intentionally partial composites (2) | `csm-1b`, `ultravox-v0-5-llama-3-2-1b` |

SGMSE is already at `CPU_PASS_METAL_NOT_RUN`; preserve its evidence and do not
redo its Linux work unless the source head changes.  Every other row follows
the same converter → binder → native forward/composite → CLI → independent
reference → VAST CPU chain used in Wave 3.

The larger families are split into non-overlapping implementation commits:
frontend/tokenizer, tensor contract/converter, native compute graph, codec or
vocoder composition, independent reference, then worker/evidence contract.
Each commit must leave an honest runnable boundary or an explicit unsupported
error; no placeholder success state is allowed.

### Wave 5 — cross-cutting compute closure

Before generating Apple packets, complete these shared legs:

1. BF16: retain the green independent AVX-512 BF16 fixture; run a real BF16
   checkpoint on VAST, register the reviewed bound, and leave Arm BFMMLA for
   the Apple worker.
2. HiFTNet: rerun the model-free closure, obtain the exact owner decision, then
   run strict 328-tensor conversion and independent real-weight CPU parity.
3. BigVGAN: obtain the Linux/Darwin closure decision and execute all four
   authenticated variants against the independent release reference on VAST.
4. Rerun live catalog reality after every batch and require zero
   CPU-complete/Metal-unsupported regressions.

Exit evidence: exact-head full-workspace tests, all-target Clippy, deny, audit,
zero-dependency and all relevant real-weight CPU parity gates are green.

### Wave 6 — freeze and transfer the final Apple batch

At one reviewed clean commit:

1. repeat the read-only live audit;
2. require `partial=0`, `no-runtime-binder=0` and `not-artifact=0`, except rows
   covered by explicit owner-approved public withholding/withdrawal;
3. generate fresh VAST packets for every model lacking Apple evidence,
   including the historical GigaAM/OmniASR set and the current SGMSE,
   ReazonSpeech, BiCodec and Voice Gender workers;
4. verify every file/hash/regular-path/clean-head contract;
5. transfer directly from disposable VAST storage to Scaleway; never route
   model artifacts through the maintainer Mac;
6. destroy each VAST instance and storage immediately after transfer is
   verified.

### Wave 7 — final Scaleway Apple CPU / Metal run

Only now provision Scaleway.  Record hardware, macOS, Xcode/Rust and Metal
fingerprints, then run every named Apple worker.  Every row must record
CPU/reference, Metal/reference and Metal/CPU at its pre-registered bound plus
an explicit no-fallback verdict.  A hardware failure returns to the relevant
source/VAST wave; it does not get reclassified as success.

Exit evidence: all eligible rows are `APPLE_PASS`, Arm BFMMLA passes on real
hardware, no worker skipped its real test, and the exact final commit retains
green repository gates.

### Wave 8 — separately authorized public reconciliation

After Apple evidence is green, request repository-scoped permission for each
upload or withdrawal.  Publish only through the gated repository scripts,
never by a manual Hugging Face upload.  Repeat the live read-only audit until
CPU `partial=0`, `no-runtime-binder=0`, `not-artifact=0` and Metal
`blocked-by-cpu=0`, `cpu-only=0`, `not-artifact=0`, with any deliberately
withheld repositories documented by their owner decision.

## Commit and reporting cadence

- One logical model family or cross-cutting contract per commit.
- Commit only reviewed files; use explicit pathspecs and never `git add .`.
- Push after a coherent batch has focused evidence and, for Rust implementation
  batches, exact-head VAST workspace evidence.
- Keep PR #79 updated while it remains the campaign PR; if GitHub or review
  size requires a follow-up PR, link both directions and retain this ledger as
  the shared denominator.
- After every batch report: rows advanced, rows still blocked, exact head,
  tests/gates, cloud instance disposition and whether any user action is now
  required.

## Completion proof

The campaign is complete only when all of the following are simultaneously
true:

- all 63 original unresolved rows have either complete public CPU/Metal
  evidence or an explicit owner-approved withdrawal/withholding disposition;
- native BF16, HiFTNet and BigVGAN shared requirements are green;
- the final Scaleway batch proves Apple CPU and Metal with no fallback;
- every Vokra VAST instance and independent storage volume is destroyed;
- all requested commits and PR checks are green;
- separately authorized public changes are reconciled by a final live audit.
