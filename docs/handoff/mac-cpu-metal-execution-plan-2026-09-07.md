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

The consolidated review surface is
`docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md`. It
separates the seven families that already have an immutable approval scope
from the families whose source, dependency, native-payload or license evidence
is still incomplete. The packet is not an approval and authorizes neither
execution nor publication.

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

## 2026-09-07 Wave 1 progress ledger

The source/API closure batch through `1d14acc6` is committed as separate
family-sized changes. It binds or corrects Qwen3-TTS dependency evidence,
Zonos configuration, Dia text/generation, OWSM disposition, FireRed dispatch
tests, Kyutai STT sidecars/streaming seams, the CLAP reference contract,
AudioGen disposition and the XY-Tokenizer source topology. The owner's dirty
CosyVoice2 license manifest remains unstaged and unmodified by this batch.

On VAST instance `50138441`, exact head `5c81eff9` passed the focused FireRed,
Zonos, Dia and Kyutai STT tests, both CLAP contract self-tests, the OWSM
inspection self-test and `cargo test --workspace --no-fail-fast`. The workspace
log SHA-256 is
`c21bb4e241efdf15fc7c8aced4521a009d6934ddf061a7ed5332f547fd546f01`.
The subsequent all-target Clippy gate found only seven redundant FireRed test
casts; commit `146d75d8` removes them. AudioGen disposition commit `c6b2e62e`
and XY-Tokenizer topology commit `1d14acc6` followed, so exact-head VAST
workspace/Clippy/deny/audit verification and the PR push were the next batch
gate.

Exact VAST head `22bc4f9198ec360bfb0420745953f4e77fde7823` then passed
`cargo test --workspace --no-fail-fast`, all-target Clippy with warnings
denied, `cargo deny check` and `cargo audit`. The corresponding log SHA-256
values are `684c743a62a46a4f0793123ac2a1e196c56aba687b5d03bba2425d6010f8d4cf`,
`75a5370e2bd41a3bf965c53960100de4691a316ddfc58253209c34fdf289c8bc`,
`cae215ad3eb07523400e35aff2eb59116f4594be2ff2a417c5d397e3f26c1102`
and `49263fa54da8102c6f01ec005cd5be7e851eb6ad40bae65831a9dd28cf31514f`.
That head was pushed to PR #79. Its GitHub run completed 108 checks
successfully and skipped 13 opt-in checks; the sole failure was the
bound-architecture parser treating the substring `ARCH` inside eight new
FireRed `SEARCH` constants as model-architecture declarations. Commit
`1cc898a5` renames only those Rust constant identifiers in the converter and
binder while preserving every GGUF wire key and numeric search value; both
architecture gates pass after the correction.

The next source slice is committed as `43235c80`, `1cc898a5`, `aaace7e5`,
`d58a6d7f`, `a7010fec` and `c44d6af2`.
MOSS Audio Tokenizer Nano now binds the fixed Transformers mapping, official
API methods, nine-stage decoder layout and exact model-free tap/audio shapes.
Exact VAST head `aaace7e51cb42a2b4ea90319e7dd195d3952ae7b` authenticated
the seven non-weight files and the server-only identity of the 87,922,568-byte
weight shard without materializing it; `weights_loaded=false` and
`weights_executed=false`. The blocked evidence-manifest SHA-256 is
`33c00a78d6e9fecd350feeb760ee36bc17960427bdd7807c598c2f4c325c157b`.

SBV2 JP-Extra now has a strict model-free sidecar validator over the pinned HF
revision and official source blobs. The source authenticates language rows
`ZH=0`, `JP=1`, `EN=2`, raw-tone counts `6/2/4`, global offsets `0/6/8`,
`n_vocab=178` and `n_tones=12`. Commit `c44d6af2` corrects the runtime and
dumper to those rows, converts JP raw `0/1` to global `6/7`, preserves all four
authenticated EN raw stress/special values as global `8..11`, and rejects
out-of-band raw tones. Existing independent binary fixtures were not edited;
real sidecar regeneration and focused parity remain VAST gates before a
production JP-Extra route is enabled.

CLAP commit `d58a6d7f` pins the released HTSAT/text configuration and requires
all six state-dict roles before emitting inspection evidence. OWSM commit
`a7010fec` authenticates the fixed ESPnet frontend/STFT/LogMel/GlobalMVN source
seams and the 1,172-name source manifest while leaving target mapping, the
native writer/runtime and CPU parity explicitly blocked.

Model-free audits made no speculative edits for MMS-1B-All, Yue XCodec Mini or
HT-Demucs Multi. Their real checkpoint identities/manifests, complete native
compositions and independent CPU parity remain explicit blockers. AudioGen
still lacks authenticated T5 and 16-kHz EnCodec companions. XY-Tokenizer now
authenticates the fixed source/config/API topology, but its produced tensor
manifest still requires independent review before a native binder is allowed.
These facts do not decrement the 63-row live-public denominator.

The consolidated branch head
`67a700b9d5bedb909a18a515dc05a0ade1e7a75b` was then replayed on VAST
instance `50138441`. The preflight run passed both architecture gates and
workspace all-target Clippy with warnings denied; its log SHA-256 is
`1f51e287d0b3bffd2f797b1d8e8ffba50b3f6eaceed8c36d4e2e88a388b949a2`.
The full run passed `cargo test --workspace --no-fail-fast`, `cargo deny
check` and `cargo audit`; its combined log SHA-256 is
`8d5c9867e9ae4a649095aeb734fdd429aedb06ead6a2e1e84e35eeaa861e91d4`.
The exact-head incremental bundle SHA-256 is
`e11fc0310c3193f66674068126f7fcca62f72a822d7a6edaf36fb6d59a9f0807`.

At that same clean head, no-checkpoint VAST API smokes passed for all four
Qwen3-TTS variants and both MOSS Audio variants under the pinned Transformers
5.10.4 boundary. The Qwen evidence and summary SHA-256 values are
`8f0bec236166d0a82fa039535f924f26bc8521bdf050df49520f5c47332485a6`
and `d4c5e42e9b23e3780d030b8a510dbe381d32c7de0f18cee95139ac1ce241ceef`;
the MOSS values are
`92302f014b5b3dac39d305f90889380c6ad47754ab0e710bfaae8de5f4751c98`
and `ca40b03426d23c248dd587daf3dd4c7e593c26adb438e961d4921026901e7457`.
Both runs recorded `PASS_MODEL_FREE`, `checkpoint_load=NOT_PERFORMED` and
`publication=NO_UPLOAD`. They close only the model-free API boundary: package,
component, source, model and operator decisions remain pending, and no row is
therefore removed from the unresolved denominator.

## 2026-09-08 owner-independent identity closure

PR #79 at remote head `67a700b9d5bedb909a18a515dc05a0ade1e7a75b`
completed with 109 successful checks and 13 intentionally skipped opt-in
checks. The following local commits are not covered by that run and will not be
pushed until their own exact-head VAST workspace verification is green.

MOSS Audio commits `7fd19a91` and `54b1300a` bind the fixed source and both
model repositories without downloading checkpoint payloads. VAST recorded the
identity manifest file SHA-256
`05bdc638e1d32098c077a71563c37a0c362c394fa809ab1b1f53692bbe062ea4`;
the checked-in evidence SHA-256 is
`2998d89d2805c507bdbc3e18118bf50a9c18e06479fff1a7dce8b020dd6dae2d`.
The source tracked tree contains no LICENSE/COPYING/NOTICE filename at the
fixed revision, and neither model repository has a `LICENSE` file. The HF
cardData value `apache-2.0` is preserved only as provenance, not inferred as an
SPDX decision or approval. The 4B index maps 901 entries to three shards and
the 8B index maps 901 entries to four shards; their sizes, Git pointer blobs
and LFS OIDs are scope-bound while all shard payloads remain unacquired. The
new approval scope SHA-256 is
`08eeab246dac53187c683cfed54e07f4a64a15abd119f3c4cc186bd23b88e69e`.

CLAP commits `832bf4c8`, `85fca834`, `7df517bb` and `c143ece3` bind the
released config/preprocessor identities and inventory the frozen Linux x86_64
Python environment. Exact VAST head
`c143ece3392fba03b7eebd2724a5c9c3d825d720` recorded
`PASS_MODEL_FREE`; the audit, dependency inventory, summary and remote
identity SHA-256 values are respectively
`972dcfdbf7ae76b0ba6c37f7973e4463bdc9d1727a21d1f6fd14f46b4a1d6594`,
`e9e70f795ce9e1ccada6875384a6be036ddb8334826a7b895140da49da60ef3f`,
`3018fec62dde209f726cf85544573c31a5e885d079475595fe2b48b00a726228`
and `28e2f241e1f239279b339347f73dfc730d5a766459b5c1444b43ddff9fb54594`.
All 34 locked distributions were found. Eight packages contained native
payloads, including 22 NumPy and 13 Torch files, with no unknown native-file
inventory. Twenty license findings remain: missing SPDX `License-Expression`
metadata plus missing bundled license files for Tokenizers and tqdm. The
dependency packet therefore remains `BLOCKED` for owner review even though the
model-free boundary is green. The work directory is 1.5 MiB and contains no
checkpoint suffix, model load, forward pass or upload.

AudioGen commits `9b5c2982`, `6371a40b` and `1083851d` bind the public file
identities and the fixed AudioCraft source/config chain without acquiring its
3.7 GB language-model payload or 236 MB compression payload. The VAST manifest,
tree and acquisition-log SHA-256 values are
`6c2adb3e3948e138547aa79a2b36d8880db8bf39fc36af7b0d5b8eea9db29f4f`,
`c445caf1eb26c26ce414edb2fb9601744781a07240b0a2ff1333a1acf2f25a1d`
and `81d83040e72a2b2d4ad6eab59b823a6e1d2f2614171bdaa5fb3b6dc4c10d2960`.
The evidence stays `signable=false`, `vast_ready=false` and `NO_UPLOAD`.
Exact external T5 revision/weight identity, compression build provenance,
dependency closure, real execution and numerical parity remain blocked.

Exact VAST head `80c17e290cc163d639d88550ffaca2187f2870fb` was clean and
passed `cargo fmt --all -- --check`, metadata inspection, both architecture
gates, all-target/all-feature workspace Clippy with warnings denied,
`cargo test --workspace --no-fail-fast`, `cargo deny check` and `cargo audit`.
The preflight and full-run log SHA-256 values are respectively
`af90a3890c757533636ed594aaed18f5300dac8575e3756ebb8c4eb448c2144e` and
`a746dd9aca6fc3615735356c464e4de6e6f034b64d4ddebf7136d1d73d7130b1`.
All tests passed; the only deny diagnostic was the existing unmatched
`libfuzzer-sys` license-exception warning, while advisories, bans, licenses and
sources were all reported OK. No model payload was downloaded or executed by
this verification batch.

PR head `55a1363813e7f25061b0f4bcec6da6b6de84dda6` then completed all
122 reported checks: 109 succeeded, 13 opt-in checks were intentionally
skipped, no check failed, and GitHub reported the merge state as clean. This
head includes the hosted-runner-safe grouped-FSQ allocation proof.

The model-free XY-Tokenizer dependency collector ran on the same exact clean
VAST head without acquiring or executing a checkpoint. It covered all 57
active lock rows: 51 produced bounded license/native-payload evidence and six
remained fail-closed (`scipy`, `setuptools`, `soxr`, `sympy`, `tokenizers`,
and `tqdm`). The collection report, raw license evidence, blocked dependency
audit and blocked license-gate manifest SHA-256 values are respectively
`604e9cc74a5814f97bcd2be106e1f620f5f4d2d45052ce3c78fb485583f17210`,
`3e2471835be2b5cb767f3181050c98ff82dc12e039c9b4257af684d713306ffc`,
`428efa4d2214a21b734690b99d554dee1663fd6de2204fa29fd742e4a878c7c6`
and `deb6ebf7e6e3deded5587aff9e5a4b509a9c12f0f100877a1c9629c403efc828`.
The outcome remains `BLOCKED`, `NO_UPLOAD` and owner-sign-off-required; no
package was promoted into the tracked approval surface. After local recovery
and hash verification, the two VAST XY evidence directories and the unused
CLAP uv cache were destroyed.

The next model-free closure slice is split into commits
`0a020f9c9fc4e37f78a1f9b7115fb3cc60551283` (MMS-1B-All),
`9b87d8142673539b5d9ea67348a4630ac3f73900` (MOSS Audio Tokenizer Nano) and
`cb510404c6be556ec386d82291811393b2a1fce5` (YuE xcodec-mini). The incremental
bundle SHA-256 is
`88429420bd447ef16f4596f1ffc04e86d63d94bdc794392314f1c944d8aba5da`.
It was transferred to VAST instance `50138441`; the remote checkout was clean
at exact head `cb510404c6be556ec386d82291811393b2a1fce5` before and after all three
audits.

MMS-1B-All recorded a 34-of-34 normalized `name==version` distribution
multiset with no missing, unexpected or duplicate entry. The model-free API
and dependency report SHA-256 values are respectively
`cb1b80d2b0efc380c03a4cd49714863279088cd2bc8313c93ffd0e1071bca1f7`
and `f9153cabd44d27b2bf597dbd15f50f6d59830ae376c2881a45782c8a64878ed5`.
The API probe imported the official class only: it did not acquire or load a
checkpoint, instantiate a model, or run a forward pass. Package/native owner
review, the composed backbone/adapter/vocabulary manifest, runtime and parity
remain blocked, and publication remains `NO_UPLOAD`.

The MOSS Nano report SHA-256 is
`d1d9a05b45fbce8f20156c91c61df849af75c61ba9227a89f7e68c920e567547`.
It found the exact 51 installed distributions represented by 52 lock rows
including the virtual project, hashed 92 native files across 24 packages and
reported no `readelf` error. Missing publisher/locked-sdist license evidence
for `tokenizers==0.22.2` and `triton==3.3.1`, plus all outstanding owner review
rows, keep the result `BLOCKED` and `NO_UPLOAD`.

The YuE xcodec-mini report and matching sidecar bind SHA-256
`df32295b532ab27fdb8b4e58f98f2aa913c8f64ad329a839e564557dd92047c1`.
The audit covered 45 lock rows, 44 registry package facts and six component
reviews. Twenty-one non-target console/man RECORD traversals were recorded and
ignored; targeted path blockers and native/`readelf` errors were both zero.
All dependency/component owner reviews remain pending, so the result is
`BLOCKED_OWNER_REVIEW` and `NO_UPLOAD`. None of these three audits acquired,
imported or executed model weights, invoked Cargo, or reduced the 63-row live
public denominator.

Commit `38bb0dbd7448c9136207af115117f67d03eb1af7` records those exact evidence
files. Follow-up commit `ab1d2586b7f2b2be561ff12716e19c5f8306d84b`
separates the MMS evidence-source HEAD from the current checkout HEAD, so the
tracked evidence remains verifiable after it is committed without weakening
the runner's exact-current-HEAD and clean-worktree checks. The incremental
bundle SHA-256 from `cb510404` through that follow-up is
`f2ba8c01e132c8128fc749e03cda714b29126e7f7df5d647f6440b07a3aab5f3`.

Exact clean VAST head `ab1d2586b7f2b2be561ff12716e19c5f8306d84b`
passed formatting, metadata inspection, both architecture gates, the focused
MMS/MOSS/YuE model-free self-tests, the expected MMS pending-license refusal,
all-target/all-feature workspace Clippy with warnings denied,
`cargo test --workspace --no-fail-fast`, `cargo deny check` and `cargo audit`.
The preflight and full-run log SHA-256 values are respectively
`a742f4b2f17e9a8d49987a8bbe315d0073b1e999096c4a3a142976f1c89117fd`
and `238013c884b3ee405a3e50dd4d4c532f6b5f3a513bc2c72c07c6a63b7513551a`.
The only dependency-policy diagnostic was the existing unmatched
`libfuzzer-sys` exception warning; advisories, bans, licenses and sources were
all reported OK. This verification did not download or execute model weights.

## 2026-09-08 source-contract and dependency-evidence follow-up

The next owner-independent batch is committed through exact code head
`7cc0f2485a71a867f8c3b9b3bbf56cf56a1dc71e`.  Its eight commits register the
Kyutai tokenizer architecture, bind the MOSS Nano validation contract,
authenticate the Kyutai streaming source contract, collect locked CLAP sdist
license evidence, and correct two failures exposed by the first model-free
VAST replay.  The corrections restore authenticated resolver artifact rows to
the MOSS dependency audit and distinguish a successfully inspected empty sdist
from an unavailable sdist in the CLAP owner-review packet.  Kyutai's source
contract additionally verifies the exact upstream assignment name and the AST
source order for encode, streaming step, cache transition, token selection and
delayed output gather.  Runtime observation and numerical/PCM parity remain
explicitly `BLOCKED_NOT_EXECUTED`.

Disposable VAST instance `50212100` checked out that exact clean head from
reviewed git bundles.  No checkpoint, GGUF, safetensors, ONNX, PyTorch model or
audio inference was acquired or executed.  The exact-head workspace run
produced 322 successful `test result` groups and no failed group; all-target
workspace Clippy completed successfully.  The workspace-test and Clippy log
SHA-256 values are
`78c2d4a067e78c74a34cd6d996df4f7565a737119ed81d2069150a5b2cf93aeb`
and
`eea1a2d99c2367318879c26090e8ecbc663c4c5ffee2e7121b112d9fe2c4d3ee`.
The combined formatting, metadata, architecture, zero-dependency,
forbidden-symbol, fixture and pipefail-lint gate passed with log SHA-256
`22e529610464ab7034b4ac181a9e7e992484c45998c60ce95de174e4591ccdbc`.
`cargo-deny` 0.20.2 reported advisories, bans and licenses OK, with only the
existing unmatched `libfuzzer-sys` exception warning; `cargo-audit` 0.22.2
reported no vulnerability.  Their final log SHA-256 values are
`3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`
and
`01ed47e312cde79358be1d34373ecce034e5a4250154f6d07f8f0b7da08f4127`.

The final MOSS Nano dependency report SHA-256 is
`5cc7c9dc22331f081af6b50e80244f2805e4006590c4b2b9c828cc68e5dbc5ac`.
It recovered the exact locked resolver artifact identities, accepted the
CPU-only Torch closure and stopped only at the 37 unresolved package-review
rows; it did not promote any owner decision.  The final CLAP model-free audit,
dependency inventory and summary SHA-256 values are respectively
`6270476e34fd53b5d12cbd9cc0cb672a0633e1e72b77ba50db05132b6f17563c`,
`ada4fb32ab79a9a5ed0385c303afbb23770cc3e0314a8d5dc2e8f4935c755259`
and
`d6c449e2d933702c6a516f460b73e91138b039d70de714d423f2703de477b3f6`.
The inventory is now `PENDING_OWNER_REVIEW` with zero global findings:
Tokenizers contributes an exact locked-sdist LICENSE candidate, while tqdm's
successfully inspected empty sdist is retained as a publisher-metadata review
fact.  The pinned DSM/Moshi Kyutai source contract passed as
`AUTHENTICATED_SOURCE_CONTRACT`; its evidence SHA-256 is
`e2333319eb55dbdda12f0eef28182a84c21846defe84e6ff360a36e52237335c`.

The recovered 794-KiB evidence archive SHA-256 is
`8f40fa1ee93eac043320550f617a2c38a121e6e865de12bb46d5758242ef5994`.
It contains 56 log/JSON/metadata entries and no model suffix or file larger
than 100 MiB.  The full eight-commit recovery bundle from remote head
`37f3e2b6` has SHA-256
`1ed82bad9a501c4498b9041fce485d1adefb63ebeb12a53382f0f355cad4579d`.
After local hash and archive-content verification, instance `50212100` and its
storage were destroyed; the exact readback was `instances: null`.  The only
remaining VAST instance was the unrelated protected
`50122020`/`ralomi-m5-robustness`, which this campaign did not modify.

This batch advances authenticated source and dependency evidence only.  It
does not authorize model execution or publication and does not decrement the
63-row unresolved public denominator.  MOSS and CLAP still require immutable
owner dispositions, while Kyutai still requires actual Mimi/tokenizer/streaming
runtime completion, a reviewed numerical bound, real VAST CPU parity and the
final Apple run.

## 2026-09-08 owner-review binding and exact-head verification

Three additional owner-independent changes were reviewed and committed as
`ef24f6f8` (MOSS Nano model-free route evidence), `2f85fd9b` (FireRed
source-ready gate status) and `504858bc` (CLAP owner-review evidence
candidate).  MOSS Nano now binds the exact AutoConfig/meta-device AutoModel
route, custom-source identities, API surface and tap shapes to the earlier
inspection manifest and final dependency audit.  It remains fail-closed with
37 package-review rows, no owner approval digest, no real-weight execution and
`NO_UPLOAD`.  FireRed now distinguishes its authenticated CMVN, output
dictionary and source-implemented native seams from the still-blocked empty
config, dependency/provenance review and real parity.  CLAP now binds the
previous model-free audit, dependency inventory and summary to the existing
commercial model-license row in a non-approving candidate whose payload
SHA-256 is
`91a8a82f8c420ac5f456f12f021bd385c43b50f947e9e05e547272ae3cec85aa`.
Its dependency and runtime approvals remain pending and publication remains
`NO_UPLOAD`.

The unpushed exact head
`504858bcfe4f7809090ef6b25f5105e40b42c509` was transferred in a git bundle
with SHA-256
`5a03c13a1ba63bc180c6eaf38b9a6b32ab57786d7c77f006cffd31892eed27f6`.
The first disposable worker, `50215996`, never advanced past VAST's
`Preparing GPUs` state and was destroyed rather than left billing.  Replacement
worker `50216331` checked out the bundle at a clean exact head and passed
formatting, metadata, zero-dependency, forbidden-symbol, fixture-EOL,
pipefail, binder/converter handshake, catalog-reality and all focused
MOSS/FireRed/CLAP self-tests.  The fast-gate log SHA-256 is
`33de969c149bc09ba6af8e695827a278c2f09857fb4a19a08754c5ab4db66b44`.

The exact-head workspace run completed 322 suites with 8,008 passed, zero
failed and 100 explicitly ignored tests.  Its log SHA-256 is
`b806817aa050c1523015361ff604a9275738de892b1f9600c0ad1a3c8d78a10f`.
All-target Clippy with warnings denied also passed; its log SHA-256 is
`5d9c9a2be47364e07779796032ae9353407244fa901de68211ca1910aaec102a`.
Finally, cargo-deny 0.20.2 reported advisories, bans and licenses OK (with only
the existing unmatched `libfuzzer-sys` exception warning), cargo-audit 0.22.2
completed a 1,242-advisory database scan with exit zero, and the repeated
zero-dependency gate passed.  The combined dependency-gate log SHA-256 is
`87ae6974f7075315c4f70ea1d7e38890b4a265a2165da30acee6af6fdfdc0ec3`.

After the small text logs were recovered and their hashes rechecked, both
disposable workers and their storage returned `instances: null`.  The VAST
inventory then contained only protected unrelated instance
`50122020`/`ralomi-m5-robustness`; this campaign did not modify it.  No model
weights were downloaded or executed in this verification, and the unresolved
public denominator therefore remains 63 rows (62 model rows plus the one
non-artifact row).

## 2026-09-08 final owner-independent hardening replay

The final model-free continuation is committed through exact code head
`06eb054351681eeb92d62664f56903da68d6c47b`.  The reviewed commits cover the
remaining MOSS Audio, Irodori, SBV2, ASR replacement, Yue, speaker, FireRed,
XY-Tokenizer, Zonos, Kyutai STT, OWSM, MMS, CLAP, HT-Demucs, AudioGen and Dia
source/artifact preparation boundaries.  They preserve fail-closed execution
and publication; source inspection or evidence publication is not counted as
a complete native runtime or numerical verdict.

Disposable VAST instance `50262953` received only reviewed git bundles.  No
model checkpoint or weight was acquired or executed.  The first workspace run
found that Canary Flash/v2 checkpoint path validation had moved ahead of the
required tokenizer authentication.  Commit `ff7f8597` restored tokenizer-first
validation and added both regression tests.  The first strict Clippy pass then
found three mechanical findings; commit `06eb0543` removed two needless
borrows, used the equivalent zero-membership check and moved production OOV
helpers ahead of the test module.

At clean exact head `06eb054351681eeb92d62664f56903da68d6c47b`,
`cargo test --workspace --all-targets --locked` completed 305 result groups
with 8,000 passed, zero failed and 100 explicitly ignored tests.  Its log
SHA-256 is
`adeceb3d3a1bd9ab9fadf0a8cb59ceef4c8a68388ca652f3fd007d5134ceab61`.
All-target workspace Clippy with code warnings denied exited zero; its log
SHA-256 is
`9e0c48342d0e6239efb2bf733b7ee16739d8b403487555e204663b9b74422c3a`.
Clippy still prints the existing configuration notice that
`vokra-backend-cpu` declares MSRV 1.89 while the workspace `clippy.toml` uses
1.85; this is not a code lint and did not bypass `-D warnings`.

The repeated formatting, locked metadata, diff, zero-dependency,
forbidden-symbol, fixture-EOL, pipefail-lint, architecture-handshake,
bound-architecture and zoo-manifest gates all passed.  Their log SHA-256 is
`d6788a9e281d11eab5250cab2fb488df534f5dbe624ceee2359427a4763b8513`.
`cargo-deny` 0.20.2 reported advisories, bans, licenses and sources OK, with
only the existing unmatched `libfuzzer-sys` exception warning;
`cargo-audit` 0.22.2 loaded 1,242 advisories and exited zero.  The combined
dependency-gate log SHA-256 is
`149cb29504484717cc474ce5ce07b0740ce6cba4ebdd57b892f1cd8ea8606186`.

The read-only live Hugging Face audit was repeated at PR head
`5d2ee9fd55344f4f5d62e88c0b9fc7c1a45ceafa` after the verified batch was
pushed.  It again found 194 public repositories, 193 GGUF-bearing
repositories and 198 GGUF files.  CPU classification remained `full=131`,
`partial=43`, `no-runtime-binder=19`, `not-artifact=1`; Metal remained
`full=131`, `blocked-by-cpu=62`, `not-artifact=1`.  The audit's invariant that
no CPU-complete public repository lacks a complete Metal source route passed.
Only public metadata, README bytes and GGUF filenames were read; no model
payload was acquired or executed.

The four recovered text logs total 682,756 bytes and their local SHA-256
values match the remote values above.  Instance `50262953` and its storage
were destroyed after recovery; the individual API readback returned
`instances: null`.  The remaining VAST inventory contained only unrelated
protected instance `50243461` / `ralomi-m6-int8-net5`, which this campaign did
not modify.  The owner's pre-existing dirty CosyVoice2 license-gate manifest
was preserved and was not staged.

This replay closes the current model-free repository-verification batch.  It
does not decrement the 63-row public denominator: unresolved rows still need
their immutable owner/legal dispositions and authorized real-weight VAST CPU
conversion/reference/parity work before the final Scaleway Apple CPU/Metal
batch.

## 2026-09-08 final pre-Scaleway contract replay

At pushed PR head `1e7d0609ed34543b119ee7ea635a347f950919a2`, the
dependency-free preflight self-tests for Qwen3-ASR, Qwen3-TTS, SpeechT5 TTS,
MOSS Audio, Ultravox, WeSpeaker and Yue XCodec Mini all exited zero.  The
corresponding VAST worker `--self-test` contracts also all exited zero.  These
checks exercise the fail-closed approval path, tamper and duplicate-key
rejection, synchronization/acquisition ordering, exact Cargo-result sentinels,
no-upload boundary and generated Apple handoff contract without downloading
or executing model weights.

The tracked approval-scope SHA-256 values remain, respectively,
`da581832351b223b890814c0bf45ba036174da24dd7fd47a58236c1dde33ced1`,
`46662c9a1a1135c37a4dc00583c1a637f72aa745ca10f2172a38f77df9d1b1da`,
`99116b392c560ec40c574589305492f35d9d30e8e2f44a9c03392885c77e85ba`,
`08eeab246dac53187c683cfed54e07f4a64a15abd119f3c4cc186bd23b88e69e`,
`35e73acdfdffa729464400a11cdc2f890b216dc59476a79acebd24bfe8ae555b`,
`0133cb13d4869903f89d6bbcaee9e784a69cf158804ae76bf4a52c7a0ca3efd0`
and
`6f8378213db1ef19924c42cb76a194ed013093c048aba910808eb14e2dcef262`.
These are canonical approval-scope digests stored inside each manifest, not
whole-file hashes.  Every production manifest remains intentionally
fail-closed at owner review; this replay records readiness but grants no
approval, model-execution authority or publication authority.

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
