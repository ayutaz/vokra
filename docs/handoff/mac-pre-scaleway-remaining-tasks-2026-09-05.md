# Mac CPU / Metal pre-Scaleway remaining-task inventory (2026-09-05)

## Scope and current truth

This is the execution ledger for finishing the public Mac CPU / Apple Metal
campaign while keeping Scaleway as the final **compute and hardware-validation
service**. It does not claim that Scaleway alone can close an incomplete CPU
runtime, artifact, dependency or license contract.

The live, read-only Hugging Face audit was repeated on 2026-09-05 without
downloading or executing model weights. It found 194 public repositories, 193
GGUF-bearing repositories and 198 GGUF files:

| Dimension | Complete | Remaining |
|---|---:|---:|
| Mac CPU | 131 | 43 partial + 19 no-runtime-binder + 1 non-artifact |
| Apple Metal source route | 131 | 62 blocked by CPU + 1 non-artifact |

There are zero CPU-complete repositories classified as Metal-unsupported. The
63 unresolved public rows divide into five disjoint execution classes:

| Class | Count |
|---|---:|
| Public-artifact-specific blocker | 27 |
| Bound but incomplete runtime | 19 |
| Generic no-runtime-binder | 14 |
| Routed but intentionally partial composite | 2 |
| Non-artifact repository | 1 |

The old prepared Scaleway packets were intentionally destroyed with their VAST
storage. They must be regenerated only after all possible source, license and
VAST work is complete and immediately before the Apple run. No local model
execution or model download is permitted during this plan.

## Immediate branch and PR work

Before this ledger was added, the branch was clean with four implementation /
evidence commits ahead of the open PR #79 remote head:

- `4d5975db` removes the vulnerable unused GigaAM v3 long-form dependency
  closure that currently makes GitHub `dependency-review` fail.
- `a44450b0` adds SpeechT5 model-free dependency evidence.
- `f5ac6bb7` adds Qwen3-TTS model-free dependency evidence.
- `535d820c` accepts exact `LICENSE` / `LICENCE` evidence for the Qwen3-TTS,
  Parler-TTS and SpeechT5 audit paths.

Before any Scaleway allocation:

1. Run the three dependency-only audits at exact commit `535d820c` on a new
   disposable VAST worker, initially without acquiring any model.
2. Record the factual residuals rather than overriding them:
   - Qwen3-TTS still has five fixed model/decoder license paths returning 404
     and needs manifest plus owner review.
   - Parler-TTS still has three fixed model/DAC license paths returning 404.
   - SpeechT5 needs the locked-sdist license fallback proved in the real VAST
     environment and still needs operator review.
3. Once those dependency gates are green or honestly dispositioned, run the
   already staged authenticated Qwen3-TTS, Parler-TTS and SpeechT5 API smokes
   on VAST. Do not run them on the maintainer Mac.
4. Run the relevant exact-head full-workspace, Clippy, deny and advisory gates
   on VAST, recover only small evidence, and destroy the worker.
5. Push the four reviewed commits together with this management-ledger commit,
   and require PR #79's `dependency-review` and all other checks to pass. The
   remote PR was mergeable but still had one failing dependency-review check
   at the earlier remote head `7edae28f`; the current result is recorded below.

### 2026-09-06 exact-head progress

The later source-only continuation adds six reviewed, model-free slices at
local commits `ef2997be`, `3ae4d351`, `589f4337`, `c8d1a170`, `3ac16e00` and
`a98eb45b`:

- FireRedASR-AED-L now has a native 16-kHz/80-bin Kaldi-fbank seam, an exact
  raw-byte authenticated `cmvn.txt` constructor, and an explicit
  PCM-to-token-id composition over the existing encoder and greedy decoder.
  The ordinary transcription API remains fail-closed; tokenizer rendering,
  the official beam policy and real-weight parity are still open.
- Zonos now rejects non-floating authenticated-checkpoint payloads,
  non-finite native weights, drifted delay/sample-rate contracts, malformed
  nine-codebook packets and non-exact DAC composition before PCM decode. This
  hardening does not substitute for a real 246-tensor/DAC parity run.
- Kyutai TTS 1.6B EN/FR now records the pinned model/companion/source
  identities, exact scalar config and depformer schedule, plus a pure delayed
  channel aligner. It deliberately has no GGUF binder or runtime claim until
  the 418 tensor roles/shapes, source-level second-stream demux, conditioners,
  Mimi composition and licenses are authenticated.
- Hibiki 2B now records the pinned model/Mimi/tokenizer and Hibiki/Moshi source
  identities, exact scalar configuration, 33-channel delay vector and
  depformer schedule, plus a pure delayed-stream aligner. It deliberately
  makes no binder, translation, demux, Moshi-forward or Mimi-forward claim.
- FireRedASR-AED-L now also authenticates the exact 71,448-byte, 7,832-row
  `dict.txt` by SHA-256, records its distinct Git-blob SHA-1, validates
  contiguous ids and fixed anchors, and exposes a content-token renderer with
  the pinned SentencePiece boundary transform. Structural decoder markers,
  the official beam policy and artifact-side sidecar binding remain open.
- Dia 1.6B now requires the legacy DAC bind to match exactly nine codebooks at
  44.1 kHz; a larger codec can no longer be accepted while silently leaving
  extra codebooks outside the authenticated Dia contract.

Read-only source audits also corrected four stale ledger assumptions without
inventing implementation:

- Kyutai STT's dedicated `dep_q=0` text decoder, strict 323-BF16-tensor binder
  and official Moshi reference packet already exist. The remaining work is
  complete Mimi/tokenizer/streaming ASR, a reviewed fixed bound and real
  VAST/Apple execution.
- Qwen3-ASR's converter and runtime already require all three execution keys
  and authenticate/embed all five tokenizer/chat/generation sidecars for both
  variants. The public GGUFs, not the source contracts, must be regenerated.
- OWSM cannot yet take an honest PCM or forward slice: exact ESPnet frontend
  semantics, global-MVN application, 1,172-tensor payload mapping and an
  independent reference fixture are still missing.
- CLAP cannot yet take an honest preprocessing or fused-forward slice: its
  dedicated reference lock/license closure, exact HTSAT preprocessing and
  state-dict role/shape manifest are absent.

The next source-repair wave adds five further model-free commits:

- Qwen3-TTS conversion now rejects symlinked/non-regular sidecars and uses a
  create-new, no-clobber output boundary at `dd60e76e`; all four 12-Hz
  variant/topology/speaker/sidecar/companion contracts were already present.
- MOSS Audio Tokenizer conversion now rejects the canonical 374-tensor Nano
  manifest when a caller attempts to stamp it as Full or v2 at `24be04a8`;
  runtime identities and strict manifests remain disjoint.
- VoxLingua107 conversion now rejects duplicate prepared JSON/tensor keys,
  non-regular inputs and pre-existing/symlink outputs at `d04dde6b`. Its exact
  ECAPA/XVector/classifier axes and ordered 107-label contract already exist.
- BiCodec now requires the audited CC-BY-NC-SA-4.0 research-only/share-alike
  provenance source at runtime and writes no-clobber conversions at
  `40da236f`; permissive relabeling cannot make the artifact executable.
- Canary 1B Flash/v2 conversion now checks descriptor counts before set
  comparison at `66811766`, rejecting duplicate/partial input as well as the
  known encoder-only Flash and timestamp-auxiliary v2 artifacts.

Three further audits found no honest source edit to make: AudioGen still lacks
exact T5/16-kHz-EnCodec companion identities; MOSS TTS Local already strictly
requires its 48-kHz stereo tokenizer-v2 companion; and ReazonSpeech NeMo v2
already has its exact 965-tensor, 3,000-piece vocabulary and runtime-axis
contracts. Their remaining work is factual input closure and VAST/Apple
execution, not another speculative local shim.

The final no-model source sweep adds six more commits and closes the static
audit of all 27 public-artifact-specific rows, all 19 bound-runtime rows, all
14 generic rows and both intentionally partial composites:

- XY Tokenizer's arbitrary synthetic tensor-to-GGUF helper is test-only at
  `aea7dc10`; production remains inspection-only until an exact manifest and
  native runtime exist.
- SBV2 now stamps and requires the real JP-Extra model/repository/AGPL identity
  and rejects the retired generic multilingual label at `0ee8359c`.
- Sortformer's model-kind documentation now records its actual
  CC-BY-NC-4.0 research-only tier at `1f31deda`.
- ChatTTS now validates every represented DVAE/GFSQ/decoder/Vocos axis against
  the fixed source contract at `d2cca9f5`; no tensor/runtime claim was added.
- Ultravox public and separately licensed companion conversion outputs are
  create-new/no-clobber at `240737b8`.
- The current Rust path-component spelling is restored in RMVPE at
  `6aeb4278`; a serial `cargo check -p vokra-convert --lib` then passes at the
  exact local head.

The no-change audits are equally important: they preserve terminal gates where
exact source, dependency, license, dataset, tokenizer/codec or complete tensor
facts are absent, and identify source-ready rows whose next operation is
artifact regeneration or real-weight parity on VAST. No audit converted an
inspection disposition into execution approval. The only local dirty file
outside these commits remains the user's CosyVoice2 license-gate manifest.

These commits passed repository formatting, locked metadata and diff-hygiene
checks without model acquisition or execution. `vokra-models` compile/test is
reserved for VAST under the maintainer-Mac memory policy, so none of the
seventeen is a numerical verdict and the 63-row unresolved classification is
unchanged.

The implementation head advanced through `6aeb4278` in this wave. The PR
remote at the start of the wave was `d241305f`; all checks at that remote
commit were green and GitHub reported the PR mergeable. The local
implementation/test commits are deliberately kept unpushed until their VAST
verification completes. Do not merge the PR as the final Mac-coverage change
while the remaining inventory below is still open.

The dependency-audit hardening and exact model-free VAST reruns are now
recorded in the commits immediately before `7d0119c9`:

- Ultravox: exact-head model-free audit remains blocked only by 37 package
  reviews, four license-row reviews and three approvals. The compact evidence
  is bound at `6f987ae2`; the audit JSON SHA-256 is
  `22698a69938a657327a6ef074d4505e060ede67f4e8e3f3ece97d4085a92e6df`.
- NeuTTS Air: exact-head model-free audit passes with 36/36 active dependency
  rows and one authenticated model-license response. The audit JSON SHA-256 is
  `007c58177deb84e9323741409a5200853c179abbe581227e96660312980738d3`.
- Qwen3-ASR: exact-head model-free audit has four factual package-license
  blockers after closing the generic HF metadata failure. The audit JSON
  SHA-256 is
  `ca53d22a4c0b1c96b0eb272b0ba1be88682a3d82a93413c2e519fdaf55173fa5`.
- MOSS Audio Tokenizer v2: exact-head model-free audit has one factual Triton
  package-license blocker after accepting the exact `tqdm` `LICENCE` file.
  Review/sign-off rows remain intentionally unresolved. The audit JSON
  SHA-256 is
  `082ac1bfa899366f97cfee23387a25041ae58954c96a04b2c58e1c35364dd012`.

The live, read-only Hugging Face audit was repeated again on 2026-09-06 with
the same result: 194 public repositories, 193 GGUF-bearing repositories and
198 GGUF files; the 63 unresolved-row classification above is unchanged.

Commits `7d0119c9` and `8b7064d4` add and bind the independent PyTorch BF16
GEMM parity contract without running PyTorch on the maintainer Mac. The
fixtures and Linux-x86_64 lock were generated on VAST instance `49972360` with
Torch `2.13.0+cpu`. The forced AVX-512 BF16 Rust path passed all three cases
twice at the pre-registered `atol=1e-3`, `rtol=0`; the observed global maximum
absolute difference was `7.629394531e-6`. Compatible Arm-BF16 evidence and a
real BF16 checkpoint remain part of the final Apple/model work.

BigVGAN's model-free Linux dependency preflight also passes at exact HEAD
`1ce957df` after making the allowlisted PyTorch wheel request identifiable,
accepting only the officially specified Core Metadata multiple-use fields and
isolating the wheel-root identity metadata from setuptools' vendored metadata.
The committed lock SHA-256 is
`80ef4819e06ad5b78675da245917bf852ee7952847a1be69fbb2baf97f91b36e`;
the 10-package owner-review candidate SHA-256 is
`fd414613311cf1ca7da4504e85acbb79d43c200a4cb1dc221e2421fc67b26086`.
No model, package install/import or upload was involved. The candidate remains
fail-closed as `OWNER_REVIEW_REQUIRED`, `BLOCKED_UNREVIEWED_TRANSITIVE` and
`NO_UPLOAD`; it is evidence for owner review, not approval or real parity.

That closure was regenerated from the exact unpushed branch state
`78cfb9b7b9ba661c7404ccc287ca6101a168bf01` on disposable VAST instance
`49996341`, without installing/importing packages or acquiring a model. The
candidate reproduced byte-for-byte at SHA-256
`fd414613311cf1ca7da4504e85acbb79d43c200a4cb1dc221e2421fc67b26086`.
The separately recovered 28-file exact license-payload evidence has SHA-256
`88f0a6e98b5000243f32471c6a9a1274db5c38bbbcad0d271c11cb7176ab7f9f`;
the candidate records 10 active Linux packages and 142 native/bundled
payloads. The license payloads include the expected MIT/BSD/PSF terms plus
setuptools' vendored LGPL-3.0, Apache/BSD dual-license and MPL/GPL notice
material, and PyTorch's large bundled-license/NOTICE set. These are facts for
owner/legal review, not an inferred approval. Commit `eb4d8c00` binds the
candidate/evidence identities and counts into the future approval scope while
leaving every package row unresolved and publication `NO_UPLOAD`. Instance
`49996341` was destroyed with its storage immediately after evidence recovery;
the individual query returned `instances: null` and the complete VAST inventory
returned `[]`.

Commits `0e6f7ab2`, `e55a712a` and `011d2fba` close the corresponding
dependency-archive evidence gap for the supported Darwin arm64 target. The
audit authenticates uv's exact size-less PyTorch CPU lock row without making
that exception generic, selects the exact Darwin wheels and binds both Linux
and Darwin evidence into one fail-closed approval scope. The Darwin candidate
schema is `bigvgan-darwin-closure-candidate-v1`, its SHA-256 is
`148e44365efa92c2cd95feeef156e327975be465aad21c6b20c979433f6d25fa`,
and the linked 28-file license evidence SHA-256 is
`cd1e28d9449dc4a1e6fac1a13f1611042bcb8dddf68bc53b50026a646cbd0e42`.
It records 10 active packages and 21 native/bundled payloads. The exact
`torch==2.7.1` Darwin wheel is 68,578,858 bytes with SHA-256
`7b4f8b2b83bd08f7d399025a9a7b323bdbb53d20566f1e0d584689bb92d82f9a`;
its archive contains two license payloads and 12 native payloads. The other
native payloads are one MarkupSafe binary and eight setuptools Windows
launchers. This is evidence, not approval: package/license/native review is
still unresolved and publication remains `NO_UPLOAD`. Disposable VAST
instance `49997708` performed only the streamed wheel audit, without package
installation/import or model acquisition, and was destroyed with storage. Its
individual query returned `instances: null` and the complete inventory
returned `[]`.

Four additional reviewed commits close source-level gaps without downloading
or running a model on the maintainer Mac:

- `e21aaedd` adds a nonzero synthetic whole-chain HiFTNet CPU/Metal parity
  harness, including the one-final-readback contract and explicit off-Apple
  no-fallback result.
- `53e3e011` fixes BigVGAN's resident Metal alias-free upsample to preserve the
  upstream replicate-pad, grouped transposed-convolution and asymmetric-crop
  semantics instead of routing it through the distinct causal FIR primitive.
- `a956d5f3` exposes raw-BF16 activation x raw-BF16 weight GEMM through the
  common compute seam. CPU selects the existing Scalar, AVX-512 BF16 or Arm
  BFMMLA implementation; Metal retains both inputs as `ushort` storage and
  accumulates in FP32; CUDA/WebGPU fail explicitly without a CPU fallback.
- `316f6ab5` confines Apple-only HiFTNet parity helpers to their actual target,
  eliminating the Linux dead-code warnings found by the first VAST compile.

Disposable VAST instance `49982196` checked exact implementation HEAD
`316f6ab51ed5288ae5ab54e443e0e77128c5e914` on an AMD EPYC 9554 whose runtime
CPU flags included `avx512_bf16`. The following gates completed successfully:

- `cargo test --locked --workspace` (all executed unit, integration and
  doctest suites passed; no failures), repeated after the warning fix at the
  final implementation HEAD;
- `cargo clippy --locked --workspace --all-targets -- -D warnings`;
- the ignored AVX-512 BF16 independent PyTorch-fixture parity test;
- `cargo deny check`, `cargo audit` and `scripts/check-zero-deps.sh`.

No model was acquired, executed or published during that run. Instance
`49982196` was destroyed with its storage immediately after verification, and
the post-destroy VAST inventory returned `[]`.

The next disposable VAST run closed the non-Apple validation work for
`vokra/voice-gender-classifier` at exact HEAD
`df7f557409e5e1e785f9edfd98a7b00d0a3b3be0`. Instance `49983538` authenticated
the public historical ECAPA-misstamped artifact, the fixed upstream source
revision `49bcbecfd929ba5a043bde645fdff1a375eb79c7`, the fixed Hugging Face revision
`db1222153bd60337e900be22add7af180452adc0`, the 61,907,512-byte MIT checkpoint
and checkpoint SHA-256
`2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5`.
The official upstream generated the canned synthetic-tone reference. The
normalizer authenticated 233 input tensors, 202 floating tensors and 31
removed counters; the corrected 202-tensor classifier artifact has SHA-256
`afb03696d8a640d5d701ea0c136bb065cac648cbfe905a5dcc4eae04e0769b1a` and
`arch=voice_gender_classifier`, `license=mit`, `weight_license=permissive`.

CPU parity passed the fixed `0.01` FP32 bound with maximum absolute errors
`0.000054359` (features), `0.000044465` (embedding), `0.000018924` (logits) and
`0.000007540` (probability). The exact-head workspace tests and doctests,
all-target Clippy, forbidden-symbol, zero-dependency, cargo-deny and cargo-audit
gates completed successfully. The final status was
`CPU_PASS_METAL_NOT_RUN`; publication was not performed. Only small evidence
was recovered. Instance `49983538` and its storage were then destroyed, and
the post-destroy VAST inventory returned `[]`. This closes the row's current
pre-Scaleway CPU action but does not decrement the 63-row live-public audit:
the corrected artifact remains unpublished pending Apple CPU/Metal evidence
and separate repository-scoped upload authorization.

The remaining factual dependency-license cases were checked against primary
release sources. Commit `776baf0e` closes only the exact source-mapping gap
for `gradio-client==2.5.0`: the two locked PyPI artifact identities are bound
through their Trusted Publishing attestations to upstream commit
`43f5de68579919b0632ceb6107a99c629483ea2f`, and that commit's package,
project, license and publishing-workflow blobs are hash-pinned. The fixed
evidence file SHA-256 is
`3e582bc2dc7651e3f7196786d3b222d1cf1d00c932f136e516cdb8c3696e3c12`;
the resulting approval-scope SHA-256 is
`0491b036e0e8feffbee30c730e2b5d8a19ca034664b58cfac64a5a81a30ab2d3`.
The preflight and dependency-audit self-tests pass, while the production gate
still exits blocked at the first pending model-license row. This evidence does
not approve the package, dependencies, models or execution: every review row
and operator decision remains pending and publication remains `NO_UPLOAD`.
The other factual cases below must stay fail-closed:

- `gradio-client==2.5.0` now has authenticated exact-release source mapping,
  but its package review row remains `PENDING_REVIEW`; source mapping is a
  factual input to owner review, not an approval.
- `dynet38==2.2` has Apache-2.0 PyPI metadata but no sdist and the official
  `clab/dynet` repository has no exact 2.2 tag; its native wheel therefore has
  no authenticated exact-release source/license mapping.
- `qwen-omni-utils==0.0.9` has Apache-2.0 PyPI metadata but no license in its
  exact sdist. The official Qwen repository remains at 0.0.8 and its public
  issue tracker confirms the published wheel/source desynchronization, so no
  source revision may be inferred.
- `soynlp==0.0.493` was uploaded after source commit
  `264a05c96f0ccd1961f1a669a9df132076a67a15`, but that checkout still declares
  version 0.0.492. Its source says LGPL-3.0 while the 0.0.493 PyPI classifier
  says GPL-3.0, so the exact release license is an owner/legal blocker.
- `triton==3.3.1` maps to official tag commit
  `d654e0f2d91f07496454e0fcbec2a9b97df37d47` and a root MIT license. The wheel
  build also copies separately downloaded NVIDIA toolchain binaries and CUPTI
  material, while the exact wheel carries no publisher license/notice file;
  the root Triton license alone cannot authenticate those bundled payloads.

CosyVoice2's HiFT checkpoint structure is now authenticated without importing
the upstream Python environment or opening tensor-storage members. The first
disposable attempt, VAST instance `49986565`, stopped before downloading a
model because the official full reference closure reaches the forbidden
`librosa -> soxr` dependency and therefore has no approved `uv.lock`. Commits
`94efb1be`, `df82b085` and `de928330` add a dependency-free restricted-pickle
inspection path, keep its output contract fail-closed, and accept both exact
built-in `dict` and `OrderedDict` state-dict roots without expanding the
allowed pickle globals. An intermediate runner-order failure on instance
`49988805` produced no model execution and was fixed before the final run.

VAST instance `49989591` then authenticated the 83,390,254-byte `hift.pt` at
Hugging Face revision `eec1ae6c79877dbd9379285cf8789c9e0879293d`, checkpoint
SHA-256
`3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879`,
and upstream generator revision
`8555549e882236e6541748b1042d95693caa82ba`. The generator SHA-256 is
`f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626`,
its Git blob identity is `326a1a70ae7707662939c20493b3a8e4b0906216`,
and the source `LICENSE` SHA-256 is
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

The restricted manifest contains 328 F32 tensors and 328 storage members. Its
tensor-name/dtype/shape manifest SHA-256 is
`cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c`,
the storage manifest SHA-256 is
`44d9b17e9794cecd65b74537ce74c2c9ff7e1342b3d40a7f79b57599f86e84cb`,
and the 41,754-byte `data.pkl` SHA-256 is
`1e71121d0cd47db0eaa93d5d9a6628ac73ab0f828433c8bfc60adc0118d9312d`.
Only `data.pkl` was read; tensor storage payloads were not opened. The evidence
is explicitly `INSPECTION_ONLY`, `NOT_IMPLEMENTED_FAIL_CLOSED`, with CPU,
Metal and parity all `NOT_RUN`, and publication `NO_UPLOAD`. The final evidence
manifest and validation-log SHA-256 values are respectively
`6a134122b4b0bdc851b38ca1d41d42e185d70d513ebb3c2e8d15a42b279462ea`
and `a8bed39fba4e56a271d44b0a70491587e5b8a4d977b66fe38e9ad7f89e150d48`.
All three VAST instances and their storage were destroyed; the final instance
and independent-volume inventories both returned `[]`.

The follow-on HiFT implementation and model-free closure wave is recorded at
exact local HEAD `ea07cfc2`. Commits `6a13c005` through `3d62f5c2` add the
strict 328-tensor HiFT GGUF binder, CPU generator route, resident Apple Metal
SineGen2 phase-cache route, explicit Apache-2.0 conversion attestation,
Linux-wheel closure audit and authenticated real-parity harness. No model was
downloaded or executed on the maintainer Mac.

Disposable VAST instance `49994756` staged the exact Linux CPython 3.12 CPU
wheel closure without installing or importing it. The first two real archive
audits exposed valid setuptools vendored metadata and RFC-822 Description
body parsing cases; commits `2a56c11e` and `5fbcc487` corrected those without
relaxing the top-level package identity or dependency checks. Commit
`2a7244e8` excludes ZIP directory entries from file evidence while retaining
their path, mode, duplicate and member-count validation. The final candidate
contains 12 wheels, 13 lock package rows, 44 license files, 285 native
payloads and 1,964 owner-review suspicion markers. Its SHA-256 is
`2f5174af6cff51dc2b71121861e989de793e6121d5ed88c890a45a308daf55f9`;
the wheel aggregate SHA-256 is
`dd7f26947e07f490e858d6311ba14008db6aa3ef2359de8742ec4093a5cabd3c`.
The separately recovered, hash-linked license-text evidence SHA-256 is
`475d246a732794f3627882965155d4cc4c395a6011fc3d20ad7c408a8bd24687`.
Commit `ea07cfc2` binds every value into the pending owner-approval scope. The
state remains deliberately `OWNER_REVIEW_REQUIRED`,
`PENDING_PACKAGE_AND_NATIVE_PAYLOAD_REVIEW` and `NO_UPLOAD`; neither source,
model nor closure approval is inferred. Instance `49994756` and its storage
were destroyed after the two small evidence files were recovered. The
post-destroy individual query returned `instances: null`, and the complete
VAST inventory returned `[]`.

The authenticated topology has 80 mel channels, a 512-channel base, three
upsample stages whose weight shapes imply rates/kernels `[8, 5, 3]` /
`[16, 11, 7]`, three source residual stages with kernels `[7, 7, 11]`, nine
main residual blocks with kernels `[3, 7, 11]`, dilations `[1, 3, 5]`, eight
harmonics, an 18-channel iSTFT head, and five 512-channel F0 convolutions plus
a one-channel classifier. It contains 82 weight-normalized `g`/`v` pairs.
The HiFT converter now also authenticates the exact 7,330-byte pinned model
configuration at the same immutable release revision: SHA-256
`0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959`
and Git blob `bc19267bbfd373c9a760b7667a74349ddd487db1`. The topology constants are
therefore source-bound rather than shape-inferred. Owner closure and
independent real-weight CPU parity remain open.

The next CosyVoice2 component wave advanced the local implementation through
`b3c6afb6`. Commits `475b7ddb` through `d59820f9` add authenticated LLM/Flow
preparation, strict component conversion/binding, an independent official
Qwen2 eager reference, and exact LLM parity evidence. Commits `70a9cd52`
through `b3c6afb6` then close the dedicated reference-lock and approval-gate
integrity gaps found during model-free VAST review: the local virtual project
row is authenticated separately from registry dependencies, license reviews
bind exact name/version pairs, the PyTorch CPU index's size-less wheel row is
accepted only in its exact official form, and the approval scope binds the
project, lock, package/native/source/weight decisions, signer and fixed model
identities. The committed dedicated `uv.lock` is 27,906 bytes with SHA-256
`09cf625d693277601d7034039b346697a041a6219a2a7d13c5133e39a54c6ee6`.

Disposable VAST instance `50010765` installed that exact 34-package Linux
x86_64 CPython 3.12 closure without acquiring or executing a model. The final
model-free closure packet is 125,867 bytes with SHA-256
`8bfe4a52f3dc9b40b65474281a3c0d32a5bf1f6f189de6a747c38e3515148c57`;
its validation log SHA-256 is
`3c5064156d0a5f95ec516dceebd609421ca6fa987035ca403d0a1ac7002a4477`.
It records 63 license-file entries and 41 native payloads. The separate ELF
`NEEDED` packet has SHA-256
`7999545c471ad1e10c522071b60baaa5a301447d88c661dcdad1a766c2f4c3c5`
and contains only package-internal libraries plus glibc/libstdc++/GCC
runtime/OpenMP/zlib dependencies; no CUDA, Triton, ONNX or ONNX Runtime
payload is present. The one wheel-level license omission,
`tokenizers==0.22.2`, was resolved against its exact official upstream tag and
PyPI Apache Software License classifier rather than inferred from a sibling
release.

The technical closure is therefore review-complete for a VAST-only,
reference-only `NO_UPLOAD` run, but the execution-enabling `APPROVED` manifest
commit is deliberately still `PENDING_EXPLICIT_OWNER_AUTHORIZATION`. The
current safety review requires a direct owner statement authorizing both that
manifest commit and the subsequent official-model download/reference/parity
execution on VAST. No local model operation is permitted. Instance `50010765`
was destroyed with all of its storage while waiting for that statement; the
post-destroy instance and volume inventories both returned `[]`.

Commit `ac45012b` closes the remaining source-level text hidden-state stub
without crossing that authorization boundary. A crate-private
`CosyVoice2TextEncoder` can now be constructed only through the strict
authenticated LLM-component binder and runs token embedding, all Qwen2 blocks
and final RMSNorm to produce finite `[tokens, hidden]` rows. Empty input,
out-of-range tokens, context overflow and malformed output fail closed. The
legacy `TextEncoderStub` remains an explicitly non-runnable compatibility
surface, and the public composite loader remains `INSPECTION_ONLY`; this does
not claim Flow, HiFT or end-to-end TTS completion. The manager independently
confirmed formatting, diff hygiene and workspace metadata parsing. No Cargo
test/check, model download or model execution was performed on the maintainer
Mac.

The SGMSE source and worker path has since advanced through exact local HEAD
`bacb3bcc`. The VAST run at `dc0fc08a` authenticated the 647-tensor score
checkpoint, built the strict GGUF and completed the independent official score
reference plus one release native score forward. Its output hashes matched the
registered score packet, but the comparator stopped because the observed
`cpu_model=1` runtime variant was not yet allowlisted. Commit `a2a1da3e` admits
only that exact reviewed runtime/provenance combination; arbitrary runtime
variants remain rejected. Commits `c5b75ddb` and `bacb3bcc` add the final Apple
worker for both the single-score check and the complete 4,096-sample
enhancement: prior noise plus all 60 corrector/predictor calls, CPU/reference,
Metal/reference and Metal/CPU at the fixed `atol=0.01`, with no CPU fallback.
The worker verifies the complete independent VAST reference packet before any
Apple Cargo execution. The score comparator must still be rerun at the exact
new head, then the full independent enhancement reference and Linux CPU parity
must run on VAST before the packet is eligible for Scaleway.

Commit `9d9c006f` adds the missing CosyVoice2 HiFT VAST validation
orchestrator. Its model-free `--closure-only` phase reproduces and verifies the
12-wheel/13-package Linux closure without accepting an owner decision. The
full path verifies the exact inspection file SHA-256, extracts only the
authenticated 328 F32 tensor name/shape contract, and keeps frozen Python sync,
source/model acquisition, conversion and real CPU parity behind a separately
supplied `APPROVED` / `OWNER_SIGNED_OFF` manifest. The runner is no-upload and
records exact named-test evidence. It has passed model-free local self-tests and
the recovered inspection manifest was accepted byte-for-byte, but no model was
downloaded or executed. The full run remains blocked on direct owner approval;
the checked-in pending manifest is not approval.

The next source-only wave closed three more execution-path gaps without local
model acquisition or execution. Commits `5a1bcd93` and `9701d962` harden the
Voice Gender Classifier Apple worker around the exact corrected GGUF identity,
the complete seven-file reference packet and direct CPU/reference,
Metal/reference and Metal/CPU comparisons. The worker still needs a freshly
regenerated VAST packet and a real Apple run. Commits `1da9af87` through
`b04b5d22` add the dedicated Kyutai STT `dep_q=0` decoder seam, strict
323-tensor BF16 binder and converter, official Moshi reference dumper, exact
HEAD VAST measurement runner and no-download Apple CPU/Metal worker. Its
numerical bound intentionally remains unset: the first VAST run is
`MEASUREMENT_ONLY`, after which a reviewed fixed bound must be committed and
the VAST pass repeated before Scaleway. Commit `b7c56b52` additionally binds
the SpeechBrain VoxLingua107 reference and Apple packet to the audited upstream
revision and exact three checkpoint payload digests. These source advances do
not yet decrement the 63-row public audit because no corrected artifact has
been published and no new Apple-hardware verdict has been recorded.

The same source-only wave also hardened three public-artifact replacement
families without acquiring or executing their weights locally:

- Qwen3-ASR commits `1747dc68` and `030db7ae` require a no-clobber exact
  single-file checkpoint plus the complete authenticated sidecar set, bind the
  reference packet to the exact checkout HEAD, and make both the 0.6B and 1.7B
  workers prove CPU/official, Metal/official and Metal/CPU results rather than
  accepting a skipped or empty test. The model-free runner and dumper
  self-tests pass; actual conversion/reference generation remains VAST work
  and the hardware verdict remains a final Apple task.
- ReazonSpeech commits `3581c850` and `cefddfe3` make the 965-descriptor model,
  3,000-piece tokenizer, frontend axes, source archive/config/tensor hashes and
  provenance an exact runtime contract. The no-download Apple worker requires
  direct CPU/official, Metal/official and Metal/CPU evidence at one exact HEAD.
  Its source tests pass, while the real VAST packet and Apple execution remain
  open.
- Qwen3-TTS commits `d4c78060` and `bea7457a` harden all four 12-Hz variants
  and the shared decoder around exact revisions, single-safetensors snapshots,
  strict JSON, symlink-safe/no-clobber paths, atomic reference publication and
  exact-head VAST/Apple runners. The reference path remains deliberately
  blocked until an authenticated Transformers 5.10.4 API smoke replaces the
  current `BLOCKED_UNVERIFIED_API_SMOKE` state; fixed real parity bounds and
  the VAST/Apple executions therefore remain open.
- Canary commits `a59cafa3` and `44689ab8` harden the Flash and v2 replacement
  pipelines around a ten-entry authenticated reference packet (eight data
  files, an exact manifest and a packet digest), exact checkout and input
  identities, and separate CPU/official, Metal/official and Metal/CPU evidence.
  The reference is decoded from the same token vector compared by the runtime,
  rather than from a second forward pass. The local source-only runner and
  dumper self-tests pass; VAST conversion/official-reference generation and
  the final Apple runs remain open. VAST records the honest intermediate state
  as `CPU_PASS_METAL_NOT_RUN`.
- NSNet2 commit `eb470233` corrects the released-model provenance from the
  historical MIT/permissive mis-stamp to the pinned DNS-Challenge
  CC-BY-4.0/attribution-required identity. The strict binder now rejects the
  old public object, requires the exact source revision and ONNX digest on a
  canonical replacement, and refuses duplicate, mistyped or unexpected Vokra
  metadata and non-finite weights. Its runners bind approval to an exact clean
  checkout, require authenticated input/reference/GGUF hashes, and consume
  Rust-emitted CPU/reference, Metal/reference and Metal/CPU results. Local
  source-only checks pass; the real VAST conversion/reference/CPU run and final
  Apple execution remain open.
- BiCodec commits `ff5ba508`, `d7555882`, `65c08032` and `6d1705b6` close the
  source-side handoff around the audited CC-BY-NC-SA-4.0 research-only
  identity. The VAST worker requires an exact clean checkout and head-bound
  owner approval, produces a separate exact official-reference packet, records
  CPU/official evidence, and emits a complete no-upload Apple command. The
  Apple worker requires CPU/official, Metal/official and Metal/CPU stage
  measurements. Actual VAST and Apple execution remain open.
- WeSpeaker commits `9bdce345`, `6d1705b6` and `d3ba3f21` make the real tests
  explicit ignored gates, add strict converter input/output handling, bind the
  VAST and Apple workers to one clean expected HEAD and authenticated packet,
  and record the honest intermediate verdict `CPU_PASS_METAL_NOT_RUN`. The
  VAST workspace, Clippy, deny and advisory commands are locked/offline where
  supported. Owner approval, the real VAST conversion/reference/CPU run and
  final Apple execution remain open.
- SpeechBrain ECAPA-TDNN commits `7c7610f4` and `629b47a4` make the corrupt
  public-artifact replacement path fail closed around the exact 200-tensor
  checkpoint, fixed upstream and reference identities, external owner
  approval, one clean expected HEAD and atomic no-clobber conversion. The VAST
  worker requires the named CPU/reference result plus workspace, Clippy,
  license and advisory gates, then emits a portable direct VAST-to-Apple
  handoff. The Apple worker authenticates the complete fixed reference packet
  and records CPU/upstream and Metal/CPU passes while leaving the previously
  unmeasured Metal/upstream comparison explicitly `MEASUREMENT_ONLY`. No model
  was acquired or executed locally; the real VAST replacement run, reviewed
  Metal/upstream bound and final Apple rerun remain open.
- MOSS Audio Tokenizer Nano commits `0587b5c6`, `82e1b3ee` and `c4d9d8a1`
  reject the historical Full-misstamped public object, require the canonical
  Nano identity and complete 374-tensor F32 name/shape manifest, reject
  non-finite payloads, and use no-clobber conversion. The exact-head VAST and
  Apple workers accept only singleton Rust measurement evidence and provide a
  hash-bound direct transfer packet with CPU/upstream, Metal/upstream and
  Metal/CPU measured independently. No numerical threshold was inferred from
  an absent real run: all three comparisons remain `MEASURED_NOT_GATED`. The
  official snapshot hashes, safe Transformers API route, decoder tap shapes,
  dependency/license approval, first real measurement, reviewed bounds and
  repeat VAST/Apple runs remain explicit fail-closed gates.
- SBV2 JP-Extra commit `b2ca0449` makes the real artifact leg an explicit
  ignored gate with no silent fixture skip, rejects converter symlink/dot-path
  inputs and pre-existing outputs, and binds the four-checkpoint Japanese
  packet to the authenticated upstream revisions. The VAST worker requires an
  exact clean HEAD plus a separately hash-bound owner approval, validates the
  complete manifest-derived packet closure and runs CPU/reference before
  emitting a direct VAST-to-Apple command. The Apple worker revalidates the
  same packet and records Metal/reference and Metal/CPU only as
  `MEASURED_NOT_GATED`; production Japanese G2P remains explicitly
  `UNRESOLVED`. No model was acquired or executed locally, and the real VAST
  run, reviewed Metal bounds, production G2P closure and final Apple execution
  remain open.
- MOSS Audio 4B/8B commits `00932737`, `e14bc2cd`, `ad36dfe4` and `4312b69a`
  convert the real-weight checks into
  explicit ignored gates, rejects incomplete or ambiguous reference manifests,
  and authenticates the complete VAST-to-Apple reference packets rather than
  a manifest file alone. VAST and Apple must share one exact clean HEAD and
  emit distinct CPU/official, Metal/official and Metal/CPU sentinels. The
  historical Transformers 5.5.0 lock is retained only for its existing
  license/closure gate. The model-free API probe and actual official-reference
  execution use a separate Python-3.12 project pinned to Transformers 5.10.4;
  its exact project, lock, full package rows, source/metadata files, owner
  approval and no-checkpoint/no-upload evidence are validated before the main
  worker may synchronize dependencies or acquire weights. The dependencies
  were re-applied with `uv add --offline --no-sync`. The exact-head VAST API
  smoke, source/model license and checkpoint approvals, distinct strict
  topology/binders, VAST real-weight parity and final Apple execution remain
  open.
- MOSS-TTS Local commits `d496c08d` and `fb7ec430` bind the VAST and Apple
  workers to one clean exact HEAD, keep Cargo offline/locked/serial, and emit a
  no-clobber transfer manifest whose external SHA-256 covers both GGUFs, prompt
  and reference inputs, both approval records, and the complete VAST native
  CPU log. The Apple worker revalidates that CPU log's exact named-test and
  measurement sentinels before running anything, then rechecks the checkout
  again after CPU/Metal execution and before writing its summary. CPU/official,
  Metal/official and Metal/CPU remain measurement-only, composite PCM remains
  `NOT_RUN`, and publication remains `NO_UPLOAD`. The approved Transformers API
  smoke, actual VAST CPU run, reviewed numerical bounds and final Apple run are
  still open.
- RMVPE commit `7ab0d5ec` rejects the historical public object's inferred MIT
  provenance and every permissive converter override. The fixed
  `yxlllc/RMVPE` source has no authenticated license grant, so only explicit
  `unknown/unknown` metadata can be written or bound; non-finite weights,
  symlinked/dot-component paths and output clobbering are rejected. Both
  workers require an exact clean HEAD and stop at `BLOCKED_LICENSE` before
  model work while the source/checkpoint terms remain unresolved. The Apache
  declaration from the unrelated `Dream-High/RMVPE` repository is not reused.
  Real VAST/Apple parity and any replacement remain prohibited until an exact
  upstream license decision and checkpoint/dependency identities exist.
- XY-Tokenizer commits `04bb4a3d` and `7b091cf6` remove the stale `fnlp`
  identity, make all three VAST stages require a clean exact HEAD, and add a
  no-download Apple consumer for the authenticated inspection manifest. The
  inspection manifest itself now carries the Vokra HEAD, so its externally
  supplied SHA-256 binds the VAST producer and Apple consumer to the same
  source commit. The worker remains executable and intentionally exits
  `BLOCKED_PENDING_AUTHENTICATED_TENSOR_MANIFEST`; no CPU, Metal or publication
  success is emitted until the tensor topology, native runtime and independent
  reference exist.
- MMS-1B-All commit `b00857ed` hardens the adapter-only replacement staging
  around one explicit language, the exact backbone/adapter/vocabulary file
  set, symlink-safe paths, exclusive output publication and streaming
  reference/approval digest checks. Its ignored Rust gate no longer silently
  skips missing real inputs. The dedicated Python project, lock and reviewed
  license manifest are intentionally absent because their dependency
  identities have not yet been authenticated; both VAST and Apple therefore
  stop as `BLOCKED_PENDING_AUTHENTICATED_MANIFEST` before acquiring or running
  a model. The native backbone-plus-adapter binder and parity route remain
  open.
- HT-Demucs Multi commits `d03266ec`, `96d4197b`, `4c91f173`, `58beebf2`,
  `baef9b9c` and `e8fc2658` bind both variants to the exact ordered
  five-member ensemble, full member SHA-256 values, pinned source revision,
  clean Vokra HEAD and external approval record. Its VAST inspection packet
  includes the complete restricted-load tensor/config manifest and an external
  manifest sidecar hash; paths are absent, non-symlinked and disjoint, and all
  publication remains `NO_UPLOAD`. The former report-only worker no longer
  labels an upstream reference dump as CPU parity: it records
  `REFERENCE_ONLY_CPU_PARITY_NOT_RUN`. The Python 3.12 reference closure now
  excludes inactive `dora-search`, `lameenc`, `openunmix` and `torchaudio`
  routes, uses a strict PCM16 WAV reader and fail-closed Wiener sentinel, and
  retains the official `julius.resample_frac` path. Exact-wheel dependency
  evidence was collected on VAST without acquiring or executing a model. The
  evidence is factually complete but remains `BLOCKED_OWNER_REVIEW`: NumPy's
  exact wheel bundles `libgfortran` under `GPL-3.0-or-later WITH
  GCC-exception-3.1` and `libquadmath` under `LGPL-2.1-or-later`, which the
  current static dependency policy rejects. MUSDB18 provenance, a separate
  checkpoint redistribution grant, owner disposition, the native binder and
  real CPU parity therefore keep the row explicitly blocked and `NO_UPLOAD`.
- AudioGen Medium commit `b4179c71` makes its inspection approval external and
  SHA-bound to the exact clean HEAD, four-file Hugging Face release identity,
  fixed AudioCraft source and CC-BY-NC-4.0 research-only/no-upload scope. It
  rejects duplicate JSON, symlink/dot paths, manifest clobbering and incomplete
  reference packets, and removes Torch/model imports from local self-tests.
  The dedicated reference project still has no authenticated dependency lock,
  so both inspection and validation stop before download or environment sync;
  Apple reports no CPU/Metal verdict. The T5 conditioner identity, complete
  16-kHz EnCodec companion contract, native composite, independent CPU parity
  and transfer packet remain open.
- SpeechBrain Lang-ID commits `b8f99dfc` and `d2e07131` make the canonical
  VoxLingua107 replacement require exact Apache-2.0/permissive provenance and
  reject missing or altered provenance before forward execution. Its ignored
  real test cannot turn absent or symlinked inputs into a skip. The VAST and
  Apple workers bind an external approval digest, exact clean HEAD, complete
  reference manifest, GGUF and `NO_UPLOAD` transfer manifest, and the VAST
  worker includes the locked/offline workspace, Clippy, deny and advisory
  gates. Actual VAST CPU/reference measurement, reviewed numerical bounds and
  final Apple CPU/Metal/reference measurement remain open.
- NSNet2 commits `341f0a88`, `ee62ef5f` and `2f5fd10b` replace the former
  env-based green skip with an explicit ignored hard-fail gate and add an
  externally hash-bound approval plus a portable exact packet containing the
  corrected GGUF, input, independent reference and singleton VAST CPU evidence.
  Both workers require one clean expected HEAD, locked/offline/serial Cargo and
  `NO_UPLOAD`; Apple authenticates the packet closure before hardware execution.
  The packet must transfer directly from VAST to Apple/Scaleway rather than
  through the maintainer Mac. The real ONNX conversion/reference/CPU run and
  final Apple Metal run are still pending.
- Apple Arm BF16 commit `3d3fe227` binds the existing independent PyTorch
  fixture packet to one clean exact HEAD and one explicit ignored BFMMLA test.
  The worker rejects unsupported hardware instead of dispatching to scalar or
  ordinary NEON, requires locked/offline/serial Cargo, rechecks the checkout
  before evidence publication and records hardware identity plus `NO_UPLOAD`.
  Only the actual Apple arm64 BF16 hardware execution remains for this fixture
  leg; a real model checkpoint remains a separate cross-cutting requirement.
- Canary 1B Flash/v2 commit `c1b7c638` makes both VAST-to-Apple routes require
  caller-bound approval evidence and one clean exact HEAD. Each VAST worker
  records separate singleton ASR and AST CPU logs, an exact CPU summary and a
  portable manifest that binds the GGUF, independent reference packet,
  approval and all CPU evidence hashes. The Apple workers authenticate that
  closure before hardware execution and cannot report success from a skipped
  or zero-test run. Model-free runner, preflight, packet and prepare/dumper
  self-tests pass locally; real VAST conversion/CPU/reference and final Apple
  CPU/Metal measurements remain open.
- Yue XCodec Mini commits `c71a9562` and `4ba32a1e` bind the exact public GGUF,
  independent reference manifest, external approval and singleton VAST CPU
  log into one portable transfer manifest. Both workers require one clean
  expected HEAD, locked/offline/serial Cargo and atomic no-clobber work or
  evidence directory claims. The Apple worker authenticates all VAST evidence
  before execution. The route remains honestly `DECODE_ONLY`,
  `ENCODE_NOT_IMPLEMENTED` and `MEASURED_NOT_GATED`; real VAST and Apple decode
  measurements, reviewed bounds and the missing complete PCM encoder remain
  open.
- Conv-TasNet commits `b3e52651` and `bb491ad7` keep the current gate
  `BLOCKED_LICENSE/NO_UPLOAD` before cache, download, conversion, model or
  Cargo work. External owner approval can no longer override the unresolved
  upstream CC-BY-SA 3.0/4.0 contradiction, WHAM research-only restriction or
  Asteroid dependency closure. The runners also bind an external approval
  hash and clean expected HEAD for a future resolved contract. Source-only
  self-tests pass; real replacement and parity are prohibited until those
  factual license rows are resolved under a new reviewed gate contract.
- AudioLDM2 commit `82ac36d1` keeps both Base and Large inspection routes
  fail-closed before environment sync, cache use, download, model import or
  Cargo. Each route requires an external approval bound to the exact clean
  HEAD, an exact approval digest, and disjoint atomic work/evidence paths. The
  dedicated dependency lock and authenticated dependency/license identity do
  not yet exist, so neither route can emit a runtime or parity verdict. The
  projection model, scheduler/sidecars, complete native diffusion composite,
  independent reference and real VAST/Apple measurements remain open.
- Sortformer commit `e19d6ba1` makes its current inspection route accept only
  an exact-head, hash-bound external `BLOCKED` disposition for the pinned
  model/source identities. It rejects ambiguous or overlapping paths and then
  exits `BLOCKED_PROVENANCE/NO_UPLOAD` before checkout, model download, cache
  use or Cargo. Mutable NeMo weight-build provenance, the native diarization
  forward and independent CPU parity remain unresolved; the source-only
  runner self-test passes but no CPU or Metal result is claimed.
- Dia 1.6B commits `d2bf8d47` and `892be078` bind the fixed model/source
  identities, exact clean HEAD and external inspection disposition before any
  cache, network or work-directory activity. Approval validation is
  no-cache/offline, paths reject dot or symlink ancestry and the adapter log is
  no-clobber. The route remains blocked on the unreviewed dependency closure,
  exact DAC contract and complete native composition; no model or parity ran.
- Zonos commits `e6ed795f` and `da97156b` make the current unauthenticated
  license identity an unconditional pre-acquisition blocker and require exact
  singleton native CPU evidence for any future Apple handoff. The native log
  verifier no longer mistakes Cargo's aggregate `test result` line for a
  second test. License authentication, DAC/PCM composition and real VAST/Apple
  execution remain open.
- CLAP commits `891b6740` and `84badaca` bind exact source/model identities and
  reject placeholder approval, but unconditionally stop as
  `BLOCKED_MISSING_AUTHENTICATED_REFERENCE_LOCK_LICENSE_GATE/NO_UPLOAD` before
  host, cache, network or work activity. The dedicated reference project,
  dependency/license closure, released preprocessing/fused binder and real
  parity do not yet exist.
- VibeVoice 1.5B commits `a6c831a1` and `0945a03e` harden exact-head approval,
  atomic VAST-to-Apple packet closure, singleton native test selection and
  external sidecar binding. The current inspection disposition remains
  unconditionally `BLOCKED_PROVENANCE/NO_UPLOAD`; it cannot authorize model
  acquisition or reference execution. Dependency approval, native composite
  and real VAST/Apple execution remain open.
- ChatTTS commit `a1e6835f` binds exact project/lock, clean HEAD and a strict
  external inspection-only approval schema, uses no-cache/offline dependency
  validation and writes manifests atomically without clobbering. Even if the
  dependency audit later changes, the current scope unconditionally blocks
  acquisition, official reference execution and Apple transfer. The audit is
  still `BLOCKED_UNRESOLVED`; the native composite and CPU/Metal parity remain
  unavailable.
- Chatterbox family commit `52abc35c` applies one exact blocked scope to Base,
  Nano and Turbo, including pinned source revision, clean HEAD, approval SHA,
  no-cache/offline audit and atomic work claims. The inspection-only
  disposition cannot authorize later acquisition. Dependency/license review,
  complete generation/conditioning/watermark/PCM runtime and a real transfer
  packet remain absent, so no Apple worker or numerical verdict is claimed.
- Irodori TTS commits `34dbc969` and `ab11420b` bind the exact model/source
  identities, clean HEAD and external blocked approval before source, cache,
  checkpoint or output access. The authenticated Python 3.12 reference route
  still reaches `librosa -> soxr/soundfile -> libsndfile/cffi`; the current
  disposition remains terminal even if a hypothetical dependency probe later
  returns green. No model, reference or Apple packet was executed.
- OWSM v4 Medium 1B commits `9c0aad0c`, `b019d5ff` and `faed3b62` make its
  blocked approval and dependency/dataset/writer/native facts exact, separate
  approval and work paths, and retain strict YAML self-tests in the frozen
  parity environment. The writer plus ESPnet frontend, E-Branchformer,
  decoder/search and token contracts remain incomplete; no model or parity
  ran.
- VoxCPM 0.5B commits `1bdc09e9`, `123469e0`, `b019d5ff`, `dfd3f1d0`,
  `5f6443f9` and `d4a99a26` put the VAST, Apple and all three direct Python
  routes behind one exact blocked scope. The gate authenticates a clean HEAD,
  approval SHA, fixed model/source/public identities and `NO_UPLOAD`, then
  stops before snapshot, checkpoint, packet or output reads. AudioVAE,
  tokenizer and the complete native composite remain unresolved.
- Canary-Qwen 2.5B commits `dfd3f1d0`, `a29be1ee` and `964ca2ac` require an
  exact blocked approval, clean HEAD and fixed model/source/tokenizer
  identities before any acquisition. Strict YAML coverage remains in the
  frozen project while a separate stdlib/offline gate self-test covers the
  pre-input path. Even a future green dependency probe cannot make the current
  inspection-only approval executable. SALM, tokenizer, dataset/dependency
  closure and the native binder remain open.
- FireRedASR AED-L commits `f5ec8335` and `bb8f1494` make both the VAST runner
  and direct inspector/preparer/reference helpers terminally fail closed. A
  single-read approval byte contract binds SHA-256, clean exact HEAD and the
  unresolved dependency, training-license, binary-CMVN, tokenizer, empty
  config and native status; malformed UTF-8/JSON and path or git identity
  failures exit before input/model access. Complete AED inference and real
  VAST/Apple parity remain open.
- Fun-CosyVoice3 commits `6f85e03a` and `c5b86511` put the runner and direct
  helper routes behind one exact external blocked disposition. The schema
  rejects duplicate or malformed JSON, non-boolean `NO_UPLOAD`, stale HEADs
  and checkout-local approval evidence before any input or model access. The
  unresolved `soxr` closure and complete codec/flow/vocoder composition still
  prohibit reference, parity and Apple work.
- ACE-Step 1.5 commits `e3eaaa73` and `51503367` make its current inspection
  approval terminal before host checks, input access or heavy imports. The
  exact source/model scope, clean HEAD, single approval byte stream and strict
  boolean `no_upload` are bound, while native DiT/LM/VAE composition,
  dependency/license/dataset closure and all real parity remain open.
- Baichuan-Audio commit `8c7e4397` and Granite Speech commit `e62cd4b7`, with
  shared path hardening in `6e6702b1`, reject checkout-local approval evidence
  and stop their unresolved composite routes before inputs. They preserve the
  existing authenticated inspection bodies, but do not authorize model
  acquisition, native execution, parity or publication.
- Kyutai STT commits `76385ab5` and `4f643818` bind the Linux measurement
  packet and Apple handoff to exact clean HEAD and strict boolean approval
  evidence. The current transfer remains explicitly
  `MEASUREMENT_ONLY_NOT_APPLE_READY`; the dedicated `dep_q=0` decoder seam and
  reviewed fixed parity bound must exist before it can become `APPLE_READY`.
- Kimi-Audio commit `9707eef3`, Kyutai TTS commit `677a36de` and Hibiki commit
  `9614c504` preserve their fixed inspection identities but add terminal
  single-read approval gates before acquisition or helper inputs. Their large
  or multi-component native graphs, component/source/dependency/license and
  dataset contracts, independent references and real parity remain open.
- Qwen2.5-Omni commits `1fcfe85c` and `176d703f` bind the five-shard,
  22,366,403,936-byte model identity and recorded Apache-2.0 model status
  separately from unresolved component/source-role/dependency facts. The
  runner, inspector and checkpoint preparer all stop before model inputs or
  heavy imports. Native Thinker/Talker/audio-VAE streaming and parity remain
  unimplemented.
- Qwen2-Audio commits `ab205686` and `d77d280b` preserve the fixed five-shard
  model, official source and Transformers inspection body, but require an
  exact external blocked approval and clean HEAD before host, work-directory,
  input or download operations. The missing exact source license and complete
  native audio-language path still block VAST/Apple parity.
- Step-Audio2 Mini commit `c7892008` gives the runner and direct inspector one
  strict external approval scope, exact clean HEAD and terminal no-upload
  marker before host, work-directory, input, output, download or heavy-import
  activity. Existing authenticated inspection code is retained but remains
  unreachable until a new reviewed contract resolves the native S2S,
  component/license/dependency/dataset and tokenizer/vocoder boundaries.
- VibeVoice-ASR commit `d24f2e95` preserves its recorded Microsoft MIT model
  and source plus pinned Transformers Apache-2.0 facts, but makes the unresolved
  external Qwen2.5 dependency, training provenance, native ASR,
  diarization/timestamp semantics and independent parity disposition terminal.
  The runner and direct inspector stop before host, work, cache, network or
  tensor imports, and all evidence records are create-exclusive.
- VibeVoice Realtime commits `89d208c4`, `a10828cd` and `c60ffb58` bind its
  exact model/source/Transformers/tokenizer identities to one external blocked
  approval and preserve partial inspection failures without writing into a
  caller-owned, symlinked or replaced output directory. Streaming state,
  diffusion/CFG, acoustic decoding, tokenizer policy, dataset provenance and
  native parity remain unresolved.
- VieNeu v3 Turbo commit `a03ba6b2` records the authenticated Apache-2.0 model
  and source separately from the declared-but-unverified MOSS dependency,
  dependency/license closure, voice policy, native composite and parity
  blockers. Its runner and direct inspector now terminate before any model or
  output access.
- XTTS-v2 commit `bd13eba5` preserves the owner-signed CPML Research-only/T4
  decision while separating source, dependency, voice-consent/policy, dataset,
  native GPT, DVAE, HiFiGAN and parity blockers. The runner, inspector and
  checkpoint preparer all validate one external approval SHA and clean exact
  HEAD before touching source/model/output paths or importing tensor tooling;
  the preparer also refuses existing or symlinked outputs.
- CSM-1B commit `e742407c` preserves the owner-signed, already-published
  Apache-2.0 model and authenticated Apache-2.0 source facts while separately
  blocking the Meta tokenizer license, Mimi companion mapping, complete audio
  generation, certifi/tqdm/typing-extensions/NumPy policy, Transformers API
  smoke, native runtime and independent parity. Both VAST runners and the
  direct inspector/reference dumper terminate before dependency lock reads,
  inputs, output, imports, acquisition or execution.

These commits improve readiness only. They do not decrement the 63-row public
audit until corrected artifacts and final Apple evidence exist under their
separate approval gates.

### 2026-09-06 exact-head source-validation closure

Disposable VAST instance `50061598` validated exact clean HEAD
`77f37c8b8806e577ac695a9633b26d7a73d26483` on Linux x86_64 with an Intel
Xeon E7-8890 v4, Rust/Cargo 1.98.1, uv 0.12.5, cargo-deny 0.20.2 and
cargo-audit 0.22.2. The branch reached this head through reviewed compile,
test and Clippy repairs from `2c4c3beb` through `5bce448b`, followed by two
source-contract fixes: `f379da5f` made the Voice Gender Apple worker's
directory-mode self-test portable without relaxing its Darwin arm64 execution
gate, and `77f37c8b` removed the stale Kyutai STT `NO_STAMP` exception after
the converter began emitting the required metadata.

At that exact head the following model-free/source gates passed:

- `cargo test --locked --workspace`, including all executed unit,
  integration and doctest suites;
- `cargo clippy --locked --workspace --all-targets -- -D warnings`;
- `cargo deny --locked check licenses advisories bans`, `cargo audit` and
  `scripts/check-zero-deps.sh`;
- all 52 Apple worker syntax and `--self-test` contracts; converter/binder and
  bound-arch coverage; op/crate/parity-sidecar/runbook citation checks;
  workflow advisory/find hygiene; M5 residual ABI/blocker checks; zoo manifest
  completeness; and EnCodec exclusion.

The recovered evidence logs total 709,622 bytes. Their SHA-256 values are
`65fe7ec4d99ad6133c398db2a52a90c794ec469066fefa6484bf284071c5b894`
(workspace tests),
`7b74f3a58111e7c8faee2d9d047a98634eeace2bd0fd1a298528b0da76233922`
(Clippy),
`3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`
(cargo-deny),
`49263fa54da8102c6f01ec005cd5be7e851eb6ad40bae65831a9dd28cf31514f`
(cargo-audit),
`2427e36fecf960b1eaf88be6bb50b9bad42626db3c75eb839edac76a33af7332`
(zero-deps) and
`c8b7169653359dee761c85f8745aad6b9aa7af9532b591da6e14cf56261d496d`
(static contracts). No model was acquired, executed or published. Instance
`50061598` was destroyed with its storage after evidence recovery; its
individual query returned `instances: null` and the independent-volume
inventory returned `[]`. Other `ralomi-*` instances in the account were
outside this Vokra run and were not changed.

### 2026-09-07 source-contract continuation

Commit `ef6bf10a` normalizes the active SBV2 JP-Extra fixture and worker
contract on `sbv2-v2-jp-extra-base.gguf`. The converter retains the old
`sbv2-v2-multilingual-base` spelling only as an explicit deprecated input
alias; runtime rejection tests and historical records keep that spelling only
where it documents or exercises the retired identity. The tracked hash
sidecar was renamed without changing its digest. This closes the remaining
filename/default repair in the public-artifact row, but does not close the
production Japanese G2P, real-weight VAST parity, reviewed Metal bounds or
Apple execution gates.

Commit `85785a14` closes BigVGAN's arbitrary-safetensors conversion gap. The
four released variant configurations and their complete topology-derived
tensor name/shape contracts now have one first-party source of truth in
`vokra-ops`; both the offline converter and runtime binder consume it.
Conversion rejects missing,
extra, duplicate, wrong-shaped or unsupported-dtype tensors and non-finite
F32/F16/BF16 values before writing output. Float payloads remain byte-preserved
and no dependency or CPU fallback was added.

Disposable VAST instance `50068583` validated exact clean implementation HEAD
`85785a14ceec0a9ead6428d1196acd21641853ec` in a separate model-free checkout
while the independent SGMSE enhancement run continued in its original
checkout. Workspace all-target tests, all-target Clippy with warnings denied,
`cargo deny`, `cargo audit`, formatting, forbidden-symbol, zero-dependency,
bound-arch, fixture-pin and dynamic-load gates all passed. The recovered
132-KiB log archive has SHA-256
`a12c34f66007aefa2a4f1faa3bb530f0146667ec880000e5683dd5a1540412fd`;
the workspace and Clippy logs have SHA-256
`be60e47e1a57953548e3055a058399228fc18c4eb6294caf3eaac753201e10be`
and
`6c9a4eef60a6d83df7d9dba7642157d4d68296c0e61ffee003721f52ab497a5e`
respectively. No model was acquired or executed by this source-validation
checkout and no upload occurred. Its separate checkout, target, evidence copy
and obsolete bundles (about 20 GiB total) were removed immediately after the
small archive was recovered; the still-active SGMSE work was preserved.

That preserved SGMSE run subsequently completed at exact clean commit
`855833c65ffd4fb9827047041411160a2a4f72c2`. The authenticated source route
resolved the inspection-only distribution, EMA-selection and exact-mapping
blockers: the resolution ledger records a reviewed 647-row tensor contract and
complete official score and waveform references. Native Linux CPU score parity
passed at `atol=0.01`, with real/imaginary maximum absolute errors
`0.0001068115234375` and `0.0000896453857421875`. The complete 4,096-sample,
60-step predictor/corrector enhancement passed against the independent
reference with maximum absolute error `0.0003332793712615967` and RMSE
`0.00006269547446627377`, also at `atol=0.01`. Workspace and package tests,
all-target Clippy with warnings denied, `cargo deny`, `cargo audit`, metadata
and all static gates passed. No model ran locally and no upload occurred.

The recovered SGMSE evidence archive is
`/private/tmp/vokra-sgmse-small-evidence-855833c6.tar.gz`, with SHA-256
`01c8ab1eb3f864ad3cc8011ab6f9038eade545aeaf26920cc951a794b8f718bd`.
Its summary and resolution-ledger SHA-256 values are
`9fe03b94a85cd4cf0e5485bd88dd8b41160b2523168c2e8b9d3ebb8089066dcd`
and
`005a57deb226d1ad883684e1be35c823f476e58a065ab4a5e268c102fa971732`.
Disposable instance `50068583` was destroyed after evidence recovery and the
independent-volume inventory returned `[]`. This closes SGMSE's Linux/VAST
reference and CPU-parity leg; only the final Apple CPU/Metal worker remains for
this row. It does not by itself decrement the public audit or close unrelated
model rows.

The same source-contract wave closed HT-Demucs Multi's model-free dependency
route at exact implementation commit
`e8fc26588dd98e7189ae2d5b172f1be4f14f2f21`. Disposable VAST instance
`50138441` synchronized the frozen Python 3.12 environment and authenticated
all 16 installed package/project rows against the lock and downloaded wheel
bytes. The 829,655-byte evidence JSON is retained at
`/private/tmp/htdemucs-dependency-evidence-baef9b9c.json` with SHA-256
`2ccc87081b52d2e8fc430421d17085fce4105d2515b95ff1118478a9a438e056`;
its package-row and license-row SHA-256 values are
`4b6fc6cc81a0da62b06c4c275a4b1cdc496e40796e28228f984c962ed6cb25d3`
and
`4b3cabcae55752a24cd23cd26a935a21f0020111d19594346055ead552ee8a9b`.
The collector reported exact closure and no factual collection failures, but
correctly emitted `BLOCKED_OWNER_REVIEW` and `NO_UPLOAD` for the NumPy bundled
license finding above.

That instance also validated the exact clean implementation commit with
`cargo fmt --all -- --check`, metadata, workspace/all-target/all-feature
Clippy with warnings denied, workspace/all-target/all-feature tests, `cargo
deny check`, `cargo audit` and the HT-Demucs dependency-audit self-test. The
682,415-byte recovered validation log is
`/private/tmp/vokra-validation-e8fc2658.log` with SHA-256
`cbc893f3c0fd2a04c0fc7b246123c1783d591fd59d6a8f8287aa1b43b6de9476`.
All executed tests passed; `cargo deny` emitted only its existing unmatched
`libfuzzer-sys` exception warning. No model, checkpoint, source repository or
audio was acquired or executed, and no upload occurred. Instance `50138441`
was destroyed with its storage after both evidence files were recovered; a
follow-up query returned `not found or no longer exists`.

### 2026-09-08 model-free source-contract continuation

The last published branch evidence commit, `aa994bb3`, records an exact-head
disposable-VAST baseline: 322 suites, 8,008 passing tests, zero failures and
100 explicitly ignored tests, with workspace Clippy, cargo-deny, cargo-audit
and the zero-dependency gate also green. PR #79's checks at that remote head
are green. The commits below are newer local work and therefore require a new
exact-head VAST run before push; the older result is not reused as proof for
them.

Six separately reviewed commits advance source readiness without downloading
or executing a model on the maintainer Mac:

- `d9bcd5a6` binds FireRedASR AED-L's authenticated external CMVN and
  dictionary sidecars into the inference contract. Complete released-weight
  decoding and independent real parity remain open.
- `fd8d782f` authenticates Dia 1.6B's model-free official-source contract and
  keeps its delayed-AR/DAC composition boundary explicit. Real main-model plus
  DAC binding and same-execution parity remain open.
- `735ce172` synchronizes the MOSS Audio Tokenizer Nano VAST and Apple workers
  on the same nine authenticated contract values and rejects duplicate,
  unknown or drifted definitions. It remains `MEASURED_NOT_GATED`, with no
  download, execution or upload authorization inferred.
- `7b77fa4b` authenticates CLAP's fixed Transformers 5.10.4 Python source tree,
  source-only processor preprocessing and source-derived meta expected
  manifest. The dedicated closure stays fail-closed on dependency/license
  owner review, and no checkpoint or fused native forward was claimed.
- `ebb93ee1` adds OWSM v4 Medium 1B's strict F32 mel-matrix/GlobalMVN binding
  and native PCM STFT-to-normalized-log-mel frontend. Its dedicated Python
  3.12 project is frozen to the CPU PyTorch index and records all six official
  ESPnet import roles, but its dependency/license gate intentionally returns
  `BLOCKED_UNREVIEWED_TRANSITIVE` before source or model acquisition. The GGUF
  writer, E-Branchformer/decoder/search route and real parity remain open.
- `a6b12556` makes Qwen3-TTS model-free API evidence a strict, hash-bound
  prerequisite of the four-variant real-weight validator. The handoff binds
  exact HEAD, source, project/lock, eight runtime package versions, all four
  configurations, source facts, optional-module sentinels and the no-checkpoint
  state before approval, host, sync or model access. License/operator approval
  and the real-weight run remain separate gates.

These readiness commits do not decrement the 63 unresolved public rows: no
corrected public artifact was published and no new Apple-hardware verdict was
recorded. The next authorized operation is a new disposable VAST checkout of
the exact local head for workspace/static gates and model-free workers only.
Real-weight acquisition/execution still needs an exact no-upload owner
authorization, and publication remains a separate repository-scoped action.
VAST instance `50122020` (`ralomi-m5-robustness`) is outside this validation
wave and must not be reused, stopped or destroyed by this plan.

#### 2026-09-08 exact-head VAST execution result

Disposable VAST instance `50225437` completed the approved model-free wave.
The first allocation, `50224998`, never started because of a provider GPU
error and was destroyed before the replacement was rented. The implementation
head advanced through five separately reviewed corrective commits discovered
by the real gates: `67ddd289` documents the OWSM frontend dimensions,
`89a7ea2b` distinguishes the locked `2.7.1` distribution from the
`2.7.1+cpu` Qwen3-TTS runtime evidence, `3432d3b9` satisfies strict OWSM
Clippy without changing its calculation, `d7076a1b` uses the supported
Transformers 5.10.4 CLAP processor keyword, and `1f381cbd` authenticates the
three legitimate zero-byte Python package markers in that exact Transformers
wheel while rejecting every empty-set or digest drift.

The complete workspace/all-target/all-feature test at Rust implementation
head `d7076a1bc96f5238f272006c37ccb0e2080a0000` completed **305 suites,
8,053 passed, zero failed and 100 ignored**. Its recovered log is
`/private/tmp/vokra-vast-evidence-1f381cbd/workspace/full-test/workspace-test.log`
with SHA-256
`cf2b7c1ff870aadd9c8a193fe5bae43345942c1ca98291335a6ba9ce5f4f747a`.
The final implementation commit `1f381cbd37ec66c6bc62b632b924e76ca5eccbf6`
changes only `tools/parity/clap/clap_model_free_audit.py` after that run; the
recorded Rust/Cargo tree-equivalence proof has SHA-256
`87f6bbe592413d3804edff0352a82919e6de2103263e89d59ab2d2f58040a549`.
At the final implementation head, workspace/all-target/all-feature Clippy with
warnings denied, `cargo deny check licenses advisories bans`, `cargo audit`,
the zero-dependency gate and format check all passed. Their principal log
SHA-256 values are respectively
`b449c24b55ab19ca62970851425252bbacc40958048517f9a10637e0ebb9b336`,
`3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`,
`01ed47e312cde79358be1d34373ecce034e5a4250154f6d07f8f0b7da08f4127`
and `2427e36fecf960b1eaf88be6bb50b9bad42626db3c75eb839edac76a33af7332`.
Cargo-deny emitted only the pre-existing unmatched `libfuzzer-sys` exception
warning.

All final model-free evidence is bound to clean implementation head
`1f381cbd37ec66c6bc62b632b924e76ca5eccbf6`:

- Qwen3-TTS all four variants passed config/processor/API evidence validation
  without checkpoint files. Evidence SHA-256:
  `6dd10154b96982f6260426d07f23f1887f9c96633b3a0480cd6db0021a8e4215`.
- MOSS-Audio 4B and 8B both passed the official model-free API route without
  checkpoint loading. Evidence SHA-256:
  `64f2dcb500e63497525f4453ce65fa8d5db725125a9ead7689f60cc8b266cd72`.
- CLAP passed the locked-wheel binding, source-only processor route,
  source-derived expected manifest and full model-free audit. Its dependency
  inventory remains `PENDING_OWNER_REVIEW`; no checkpoint, model forward or
  parity was claimed. Evidence SHA-256:
  `57d9c84b1963acf9f12f8a179c37ccc9edfa6f1bd6666ae0dd7e15cc8c12386b`.
- Dia completed its authenticated official-source-only contract. Evidence
  SHA-256:
  `8e1480522fb650f679963d5805923514609feb8f8fd050a87a93d19faa60ff43`.
- MOSS Audio Tokenizer Nano authenticated the non-weight file identities,
  server-only weight identity and meta-device shape route, then honestly
  remained `BLOCKED`/`INSPECTION_ONLY` for unresolved dependency, approval and
  real-runtime parity. Evidence SHA-256:
  `50a383853b4f7ca7bf26341c742fb6b415f3c0da36888f39e6f8b1508f2abd7d`.
- OWSM stopped before source or checkpoint acquisition at the intended
  `BLOCKED_UNREVIEWED_TRANSITIVE` dependency/license gate. Evidence SHA-256:
  `797cc1565ac7d1d8a6c9045538dfedcd549cc578a0e2ba9c9482b7209d945ef2`.

No checkpoint weight was acquired, loaded or executed in this wave, no model
ran on the maintainer Mac, and no upload occurred. Instance `50225437` and its
storage were destroyed after evidence recovery; a follow-up instance query
returned null identity, status and cost fields. These results close the named
model-free actions but do not decrement the 63 unresolved public rows: the
remaining real-weight, dependency/license, native-runtime, publication and
Apple-hardware gates remain explicit below.

#### 2026-09-08 post-baseline source and metadata closure

Eleven separately reviewed commits after PR #79's remote head prepare the next
exact-head model-free VAST wave without acquiring or executing a model locally:

- `d8127d37` authenticates the OWSM reference project's locked dependency and
  source-license closure while preserving owner review as a fail-closed gate.
- `d09c1ae9` and `788edf12` isolate Irodori TextBlock's source contract and
  audit its exact Python 3.12 dependency, license and native-library closure.
- `6dda20a8` records AudioGen's exact T5 candidate and 16-kHz EnCodec companion
  metadata without claiming that the historical T5 linkage is authenticated.
- `1e238343` distinguishes the official MOSS-Audio 4B and 8B topologies instead
  of treating their configuration identities as interchangeable.
- `e4cada52` authenticates the CosyVoice3 source dependency closure and records
  the forbidden `librosa -> soxr` path; the complete composite remains blocked.
- `6f3a026e` adds MMS-1B-All's exact eight-role Hugging Face metadata audit. It
  requires an explicit adapter language and never defaults to English.
- `3d727c53` authenticates Baichuan-Audio's public source roles and keeps the
  missing Matcha source, custom Hugging Face code, HiFT payload and distinct
  license scopes unresolved rather than inferring inheritance.
- `d41b229e` audits Qwen2-Audio's complete public GitHub branch history. Only a
  completed clean-head audit may record `SOURCE_LICENSE_UNKNOWN_BLOCKER`; API,
  checkout or history failures remain a distinct incomplete-audit blocker.
- `163c58bf` authenticates VibeVoice-ASR's fixed Qwen2.5-7B 14-file metadata
  closure, four LFS shards, Apache-2.0 license bytes, source declaration and
  main-branch chronology without downloading a checkpoint.
- `943b1b12` batches nine model-free/source-only workers behind one clean exact
  HEAD, explicit MMS language and strict recovered-evidence manifest. It accepts
  only completed factual dispositions, copies at most 1 MiB per evidence/log
  file and 8 MiB total, records SHA-256 readback, and keeps `NO_UPLOAD`.

The prepared implementation head is
`943b1b126292a60827d1127e10cb0f82f7350248`. Local verification was limited to
model-free self-tests, shell syntax/diff checks and the normal five commit
gates; no local model construction, inference or broad Cargo run occurred.
This preparation does not decrement the 63 unresolved public rows. Before any
Scaleway allocation, bind a disposable VAST checkout to the final documentation
head, run the workspace/static gates and the nine-worker model-free batch,
recover only its bounded evidence, then destroy the instance and storage. MMS
execution must wait for an explicit owner-selected adapter language.

#### 2026-09-08 final model-free batch and exact-head validation

Disposable VAST instance `50237377` completed that nine-worker model-free
wave at clean implementation head
`86efd5cf0de0e201e364299bf5747a819477dc65`. Corrective commits discovered by
the real workers were kept separate and reviewed: `9c6ac30d` hardens the batch
work-root gate, `41388772` hardens Qwen2-Audio output handling, `35686c9c` and
`023aa874` separate and normalize the MOSS-Audio API contract, `bf2e60e8` and
`c98bd300` complete and canonicalize the OWSM Linux closure, `68837e4e` moves
SBV2 execution off the no-exec tmpfs, and `a327a3bb` plus `86efd5cf` bind the
VibeVoice source tree and current Hugging Face history timestamp schema.

The final batch manifest reports `PASS_MODEL_FREE`, exact expected/actual HEAD
equality, `model_payloads=NOT_ACQUIRED` and `publication=NO_UPLOAD`. All nine
rows have their expected exit and an `ACCEPTED_FACTUAL_DISPOSITION`. The
evidence SHA-256 values are:

- AudioGen: `46936deb4538c9ea25eeaddc6922964e9c5c5b263fc9d3c7dc1071735f2b5f48`.
- CosyVoice3: `9ead7ff98841183c40445e33647a9496bd9ea867b613c7154b93739e939e581d`.
- Irodori: `37ba7cb76bee99a0aa8898c09ddf11b01a3f6683fe325b557685d0e25c8944bc`.
- MOSS-Audio: `b410496010dcc64baf00e85ae34c9bbc9562c152b98241f4c74d310e0022e3ec`.
- OWSM: `e8aae7e3a16e0815c3c4c366a1718997b884e9c09e4826d7b24bd9768b8eaf34`.
- SBV2: `770e5481d0335a22b52ca511adcab2fa8a9231a74c6db49d8f60e0c6a3c4ea8f`.
- MMS (`jpn`): `1155f16d2fba80ad95a37ffad6f9b2e4776d7f3770071b9bf3f9333e9785f0cc`.
- Qwen2-Audio: `ce765ec05eb57e2c1fb3721648046090310db58a88df0920c72df6fd6d80c8de`.
- VibeVoice-ASR: `d991cab34b65d987b509bf65994ae7962c511e92776f959e10d599529ec05526`.

The 6,607-byte batch manifest has SHA-256
`5dd20296f7a39575870858e2d54a618b7bd2c1b38da1c4c16b1db8e42e92f011`.
Its 18 referenced evidence/log files were independently read back after
transfer: every recorded size and SHA-256 matched. The combined local archive
is `/private/tmp/vokra-pre-scaleway-final-evidence-86efd5cf.tar.gz`, 266,558
bytes, SHA-256
`6e240214ff1d2afe094ba1baa1ff0154d526d57146dcdaa0e764388ceaf6da0e`.

At the same exact head, `cargo fmt`, the forbidden-symbol and zero-dependency
gates, `cargo metadata`, workspace/all-target/all-feature tests, workspace/
all-target/all-feature Clippy with warnings denied, `cargo deny check` and
`cargo audit` all completed successfully on VAST. The principal log SHA-256
values are full test
`f2f5b13a5cce63e14c6e9fc6e940c88ba79eaca6bdffe527c95524789855c603`,
Clippy
`2670e36fe7991f948f762f34e210902d2319c02daf6cf27506bd9f434b551f3e`,
cargo-deny
`cae215ad3eb07523400e35aff2eb59116f4594be2ff2a417c5d397e3f26c1102`
and cargo-audit
`01ed47e312cde79358be1d34373ecce034e5a4250154f6d07f8f0b7da08f4127`.
Cargo-deny emitted only the pre-existing unmatched `libfuzzer-sys` exception
warning.

A transient unauthenticated GitHub API rate-limit response was preserved as an
incomplete audit and never accepted as factual evidence. The worker was stopped
without compute billing until the official reset, then the final batch ran from
a fresh 60-request window. The nine-worker batch acquired or executed no
external checkpoint/model weight; workspace validation used only the repository's
committed test fixtures and synthetic paths. No model ran on the maintainer Mac
and no upload occurred. After evidence recovery, instance `50237377` and its
storage were destroyed; its individual query returned `instances: null`. The
only remaining VAST inventory entry was the unrelated running instance
`50243461` (`ralomi-m6-int8-net5`), which this wave did not use, stop or destroy.

The read-only live Hugging Face audit was repeated again without weight access:
194 public repositories, 193 GGUF-bearing repositories and 198 GGUF files; CPU
is `full=131`, `partial=43`, `no-runtime-binder=19`, `not-artifact=1`, while
Metal is `full=131`, `blocked-by-cpu=62`, `not-artifact=1`. This exact-head
model-free closure therefore completes the prepared nine-worker wave but does
not convert any of those live artifact/runtime rows into a real-weight or Apple
hardware verdict.

## Cross-cutting implementation before the final Apple run

These tasks affect multiple model rows and must not be mistaken for Scaleway
work:

- **Native BF16 compute:** raw-BF16 CPU/Metal storage and GEMM seams plus the
  independent kernel fixture are landed. The Ultravox audit confirms the
  current mixed-precision contract is already explicit: audio/projector
  weights widen into F32 scratch; activations, norms, residuals, KV state and
  accumulation are F32; the Meta companion uses raw-BF16 panels with F32
  activation/output on CPU/Metal and no CPU fallback. What remains is a real
  BF16 checkpoint against AVX512-BF16 reference plus Arm-BF16/Apple evidence,
  not an unreviewed claim of end-to-end BF16 activations.
- **HiFTNet full GPU generator:** the complete resident CPU/Metal graph, strict
  328-tensor converter/binder, exact pinned config and a nonzero synthetic
  one-final-readback parity harness now exist. The no-upload VAST runner binds
  the authenticated inspection evidence to conversion and independent
  real-weight CPU parity. Run its model-free closure phase first; the reference
  and real-weight phases require explicit owner license approval. After a green
  exact-head VAST result, preserve the GGUF/reference packet for the final
  Apple CPU/Metal worker.
- **BigVGAN full GPU path:** the Linux and Darwin dependency/native archive
  graphs and runtime variant binders are evidence-complete, but owner/legal
  sign-off remains fail-closed. The converter and runtime now share one
  topology-derived four-variant tensor manifest and fail before output/load on
  descriptor drift at `85785a14`. After approval, run the authenticated fixed
  artifact against the independent reference on VAST to prove the shared
  contract against real release bytes; leave Metal hardware parity for the
  final Scaleway run.
- **Coverage invariant:** after every wave, rerun the live audit and keep the
  CPU-complete/Metal-unsupported count at zero. Unsupported learned operations
  must return an explicit error; silent CPU fallback is forbidden.

## Public-artifact-specific blockers (27)

These rows already have enough architecture-specific information to name the
public-byte failure. They need the named repair, a VAST no-upload conversion
and independent CPU parity, then an Apple worker. Publication is a separate
authorization.

| Repository | Remaining pre-Scaleway work |
|---|---|
| `vokra/audiogen-medium` | Authenticate exact T5 conditioner and official 16-kHz EnCodec companion repository/revision/config/weight topology, then replace the intentionally non-executable LM-only artifact with a complete composite. |
| `vokra/bicodec` | The converter/runtime now enforce the audited CC-BY-NC-SA-4.0 research-only/share-alike identity at `40da236f`. Regenerate and verify the public artifact on VAST, retain the non-commercial publication gate, then run Apple evidence. |
| `vokra/canary-1b-flash` | The source converter already binds the full encoder, four-layer AED decoder and tokenizer and rejects encoder-only/duplicate/partial checkpoints at `66811766`. Regenerate the public GGUF and run exact-head VAST/Apple parity. |
| `vokra/canary-1b-v2` | The source converter already binds the correct main checkpoint, eight-layer decoder and tokenizer and rejects timestamp-auxiliary/duplicate/partial checkpoints at `66811766`. Regenerate the public GGUF and run exact-head VAST/Apple parity. |
| `vokra/conv-tasnet-libri1mix` | Keep the corrected 345-tensor topology, but resolve the conflicting CC-BY-SA/WHAM declarations before any replacement. |
| `vokra/htdemucs-multi` | The five-member ensemble ordering, source roles and model-free dependency bytes are authenticated. Obtain an owner disposition for NumPy's bundled GPL-with-GCC-exception/LGPL libraries, MUSDB18 training provenance and the absent separate checkpoint redistribution grant; then complete the native binder and real-weight VAST CPU/reference parity. Keep `NO_UPLOAD` until every gate is approved. |
| `vokra/lang-id-voxlingua107` | The source converter/runtime already bind ECAPA plus the 12-tensor XVector classifier, ordered 107-label vocabulary and exact axes; conversion input/output and duplicate-key gates are hardened at `d04dde6b`. Regenerate the incomplete public artifact and run VAST/Apple parity. |
| `vokra/mms-1b-all-base` | Define a dedicated CC-BY-NC backbone-plus-language-adapter contract and vocabulary; the 8.9-MB adapter is not the 1B model. |
| `vokra/moss-audio-4b-instruct` | Authenticate its distinct topology and add a strict binder; the broad `moss_tts` tag is insufficient. |
| `vokra/moss-audio-8b-instruct` | Authenticate its distinct topology and add a strict binder; the broad `moss_tts` tag is insufficient. |
| `vokra/moss-audio-tokenizer-nano` | The source converter/runtime distinguish the canonical Nano identity and 374-tensor manifest and reject Full/v2 restamping at `24be04a8`; the VAST and Apple workers share the authenticated nine-value contract at `735ce172`. The exact-head model-free VAST inspection now authenticates the non-weight bytes, server-only weight identity and meta-device shapes at `1f381cbd`, while remaining intentionally blocked. Complete dependency/license approval, the first real-weight VAST measurement, reviewed bounds, artifact regeneration and the final Apple run. |
| `vokra/moss-tts-local-transformer-v1.5` | The source runtime already requires the exact 48-kHz stereo, 32-codebook tokenizer-v2 companion and rejects Full/Nano. Run real GGUF CPU/reference parity on VAST, then Apple evidence. |
| `vokra/nsnet2` | Resolve live MIT provenance against the audited upstream CC-BY-4.0 identity before replacement. |
| `vokra/qwen3-asr-0.6b` | The converter/runtime source already binds the three execution metadata keys and all five authenticated tokenizer/chat/generation sidecars. Regenerate the public GGUF on VAST and rerun independent CPU plus Apple evidence. |
| `vokra/qwen3-asr-1.7b` | The converter/runtime source already binds the three execution metadata keys and all five authenticated tokenizer/chat/generation sidecars. Regenerate the public GGUF on VAST and rerun independent CPU plus Apple evidence. |
| `vokra/qwen3-tts-12hz-0.6b-base` | The exact four-variant topology/sidecar/companion contract and strict API-evidence handoff are complete, and the all-variant model-free VAST smoke is green at `1f381cbd`. Real conversion/parity still requires separate license/operator authorization, then artifact regeneration and Apple evidence. |
| `vokra/qwen3-tts-12hz-0.6b-customvoice` | The exact four-variant topology/sidecar/companion contract and strict API-evidence handoff are complete, and the all-variant model-free VAST smoke is green at `1f381cbd`. Real conversion/parity still requires separate license/operator authorization, then artifact regeneration and Apple evidence. |
| `vokra/qwen3-tts-12hz-1.7b-base` | The exact four-variant topology/sidecar/companion contract and strict API-evidence handoff are complete, and the all-variant model-free VAST smoke is green at `1f381cbd`. Real conversion/parity still requires separate license/operator authorization, then artifact regeneration and Apple evidence. |
| `vokra/qwen3-tts-12hz-1.7b-customvoice` | The exact four-variant topology/sidecar/companion contract and strict API-evidence handoff are complete, and the all-variant model-free VAST smoke is green at `1f381cbd`. Real conversion/parity still requires separate license/operator authorization, then artifact regeneration and Apple evidence. |
| `vokra/reazonspeech-nemo-v2` | The source converter/runtime already bind the exact 965-tensor checkpoint, embedded 3,000-piece vocabulary and runtime axes. Regenerate the stale public artifact and repeat exact-head VAST plus Apple evidence. |
| `vokra/rmvpe` | Resolve the absence of an upstream license for the exact source repository; the live MIT stamp cannot be accepted by inference. |
| `vokra/sbv2-v2-jp-extra-base` | Exact JP-Extra model/repository/AGPL identity and retired-label rejection are landed at `0ee8359c`; strict tensor renames already exist and active parity/helper filenames are canonical at `ef6bf10a`. Close production Japanese G2P, then regenerate and verify the artifact on VAST before the Apple run. |
| `vokra/speechbrain-spkrec-ecapa-voxceleb` | Replace or repair the artifact whose tensor data extends outside the declared file bounds, then rerun strict parity. |
| `vokra/voice-gender-classifier` | Exact-head corrected conversion and official CPU parity are green at `df7f5574`; regenerate its authenticated Apple packet for the final CPU/Metal worker. Publish the replacement only after that passes and separate upload authorization is given. |
| `vokra/wespeaker` | Resolve Apache-vs-CC-BY-4.0 provenance and attribution, then produce the strict artifact. |
| `vokra/xy-tokenizer` | Production conversion remains inspection-only and the arbitrary synthetic payload helper is test-confined at `aea7dc10`. Authenticate a real tensor manifest/topology, implement the native route and regenerate the metadata-only public file. |
| `vokra/yue-xcodec-mini` | Add the missing PCM encode path: acoustic/HuBERT, RepCodec, fusion and RVQ contracts. Decode-only is incomplete. |

## Bound but incomplete native runtimes (19)

For every row below, finish the native first-party forward/composite route,
complete tokenizer/codec/config and license dependencies, build an independent
upstream reference, and run real-weight CPU parity on VAST. Each then needs a
portable no-fallback Apple worker for the final Scaleway batch.

| Repository | Principal remaining boundary |
|---|---|
| `vokra/audioldm2` | Projection model, sidecars/scheduler and the full text/audio diffusion composite. |
| `vokra/audioldm2-large` | Same complete composite contract at the Large checkpoint identity. |
| `vokra/canary-qwen-2.5b` | Native SALM route, tokenizer and exact dependency/dataset closure. |
| `vokra/chatterbox-multilingual-v3` | Full generation, conditioning, watermark and PCM-output path. |
| `vokra/chatterbox-nano-v1` | Full generation, conditioning, watermark and PCM-output path. |
| `vokra/chatterbox-turbo-v1` | Full generation, conditioning, watermark and PCM-output path. |
| `vokra/chattts` | Fixed DVAE/GFSQ/decoder/Vocos axes are exact at `d2cca9f5`, but no tensor binder is implied. Complete the native GPT+Embed+DVAE+decoder+Vocos composite plus AGPL/source, CC-BY-NC weight, dependency and personality/voice policy closure. |
| `vokra/clap-htsat-fused` | The fixed Transformers 5.10.4 wheel/tree, source-only processor preprocessing, source-derived expected manifest and model-free audit are green at `1f381cbd`. Complete owner review of the dedicated dependency/license closure, authenticate the real state-dict role/shape manifest, implement the fused binder/forward and run real parity. |
| `vokra/cosyvoice2-0.5b` | HiFT checkpoint/source/config, strict converter/binder and guarded VAST CPU-parity runner are complete at source level. Run the model-free closure phase, obtain explicit owner approval before any HiFT model/reference execution, then obtain exact-head VAST CPU parity. LLM/Flow are component-bound but the full flow/codec/vocoder composition remains incomplete; the official broad closure still imports forbidden `soxr`. |
| `vokra/dia-1.6b` | The strict 343-tensor binder, exact 44.1-kHz/nine-codebook DAC connection and authenticated model-free official-source contract are complete; the source-only VAST contract is green at `1f381cbd`. Finish the real main-model/DAC bind and same-execution delayed-AR/DAC parity plus dependency review. |
| `vokra/firered-asr-aed-l` | The PCM/fbank/CMVN/encoder/token seam, exact external CMVN/dictionary authentication, token rendering and artifact/worker sidecar binding are complete through `d9bcd5a6`. Finish released-weight AED decoding/beam execution, dependency/config closure and independent real parity. |
| `vokra/fun-cosyvoice3-0.5b-2512` | Find an exact allowed route around the current `soxr` closure, then finish the full composite. |
| `vokra/irodori-tts-500m-v3` | Find an authenticated Python-3.12 reference route that avoids the current `librosa -> soxr` dependency; more RAM or Scaleway cannot solve this. |
| `vokra/kyutai-stt-2.6b-en` | The dedicated `dep_q=0` text decoder and strict 323-BF16-tensor binder already exist. Complete Mimi/tokenizer/streaming ASR, review and register a fixed parity bound from the first VAST measurement, then run Apple CPU/Metal. |
| `vokra/owsm-v4-medium-1b` | The strict mel/GlobalMVN tensors and native source-authenticated PCM frontend are complete through `3432d3b9`, with a dedicated frozen ESPnet reference project. The exact-head VAST source-only worker correctly stops before source/checkpoint acquisition at `BLOCKED_UNREVIEWED_TRANSITIVE`. Complete that transitive dependency/license review, generate independent VAST frontend evidence, then finish the writer, subsampling/E-Branchformer/decoder, joint CTC-attention search and token semantics. |
| `vokra/sortformer-diar-4spk-v1` | The inspection gate correctly records CC-BY-NC-4.0 research-only at `1f31deda`. Resolve mutable NeMo build provenance, bind the real archive/config and complete native FastConformer/Transformer/arrival-order diarization plus independent parity. |
| `vokra/vibevoice-1.5b` | Close dependency approvals, complete native runtime and execute the real workers. |
| `vokra/voxcpm-0.5b` | Add the missing AudioVAE/tokenizer companions and full native composite. |
| `vokra/zonos-v0.1-transformer` | The typed transformer/conditioner/DAC path now strictly rejects dtype, non-finite, delay, packet, codebook and sample-rate drift at `3ae4d351`. Authenticate the real 246-tensor artifact, conditioning packet and exact DAC, then execute independent CPU parity and the Apple worker. |

## Generic no-runtime-binder rows (14)

Each row needs a pinned primary source, exact converter and metadata contract,
strict binder, native forward/composite route, CLI surface, dependency/license
decision, independent reference, VAST real-weight CPU parity, Metal preflight
and a final Apple worker.

| Repository | Important known constraint |
|---|---|
| `vokra/ace-step-1.5` | License/dependency/dataset decisions and native DiT/LM/VAE composition remain open. |
| `vokra/baichuan-audio` | Matcha/source roles are not fully pinned; MPL/PSF dependency policy and the native composite remain open. |
| `vokra/granite-speech-4.1-2b` | Verify Sigstore/crypto evidence, dependencies/datasets and the native speech-language composite. |
| `vokra/hibiki-2b` | Pinned artifacts/sources, exact config/delay/depformer schedule and a pure delay aligner are landed at `c8d1a170`. Authenticate tensor roles and implement native translation/demux/Moshi/Mimi plus dependency, license and dataset contracts. |
| `vokra/kimi-audio` | Authenticate and implement the roughly 42.6-GB multi-component release; all model work is VAST-only. |
| `vokra/kyutai-tts-1.6b-en-fr` | Pinned artifact/source identities, the exact scalar config/depformer schedule and a pure delay aligner are landed at `589f4337`. Authenticate the 418 tensor roles/shapes before adding a GGUF binder; then separate voice/model/source licenses and implement the actual second-stream demux, conditioners, native forward and Mimi composition. |
| `vokra/qwen2-5-omni-7b` | Complete native multimodal streaming, dependency/license and independent parity contracts. |
| `vokra/qwen2-audio-7b-instruct` | Resolve the missing exact source-repository license, then implement the native audio-language path. |
| `vokra/sgmse-voicebank` | Linux/VAST validation is complete at exact clean commit `855833c6`: the authenticated 647-tensor conversion, official score and 4,096-sample enhancement references, native CPU parity and repository gates all passed with no upload. Preserve the evidence archive identified above and run the existing final Apple worker on Scaleway for CPU/reference, Metal/reference and Metal/CPU verdicts with no CPU fallback. |
| `vokra/step-audio2-mini` | Complete multi-component native S2S and all source/weight/dependency licenses. |
| `vokra/vibevoice-asr` | Authenticate the release and implement the complete native ASR path, dependencies and parity. |
| `vokra/vibevoice-realtime-0.5b` | Replace arbitrary BF16 pass-through staging with a strict complete realtime runtime and parity. |
| `vokra/vieneu-tts-v3-turbo` | Complete dependency/license/voice-cloning policy and the composite runtime. |
| `vokra/xtts-v2` | Resolve CPML/source/dependency and voice-cloning consent gates, then implement native GPT/DVAE/HiFiGAN composition. |

## Routed but intentionally partial composites (2)

- `vokra/csm-1b`: replace the synthesized bridge with the complete released
  companion and audio-generation contract; resolve certifi/tqdm/
  typing-extensions/NumPy policy and run independent real parity.
- `vokra/ultravox-v0-5-llama-3-2-1b`: output conversion is no-clobber at
  `240737b8`, and its mixed BF16/F32 precision boundary is explicit. Authenticate
  the gated Meta companion digest/tokenizer/chat boundary and its
  license/dependency closure; a native Whisper tower plus projector alone is
  not a complete model.

## Non-artifact repository (1)

- `vokra/seamless-m4t-v2-large`: either produce a real, gated, independently
  verified GGUF or withdraw the empty public repository. Both options change
  public state and require exact repository-scoped owner authorization.

## Per-model done condition before Scaleway

Every unresolved row must complete the applicable parts of this chain before
it enters the final Apple packet:

1. Authenticate the exact upstream source revision, weight/config/tokenizer or
   codec identities, source license, weight license and dataset restrictions.
2. Complete strict conversion metadata, binder, native forward/composite, CLI
   route and complete-backend preflight. No ONNX/ORT runtime and no silent CPU
   fallback.
3. Dump the reference from the independent pinned upstream implementation; do
   not validate a Vokra implementation against a handwritten mirror of itself.
4. Run no-upload conversion and real-weight CPU numerical/output parity on a
   disposable VAST instance. Run package/workspace tests, all-target Clippy,
   `cargo deny` and `cargo audit` at the exact reviewed commit.
5. Preserve hashes and small logs only, update the live audit honestly, commit
   the model family separately, and destroy the VAST instance unless its packet
   is being transferred immediately to Scaleway.
6. Generate the Apple CPU/Metal worker and authenticated input manifest. A
   device-less Metal build is not an Apple-hardware PASS.

## Final pre-Scaleway exit gate

Do not provision Scaleway until all of the following are true:

- PR #79 and any follow-up implementation PRs are clean, reviewed and green.
- All 63 public rows have completed their non-Apple source, artifact, license,
  dependency, reference and VAST CPU work, or have an explicit owner-approved
  withdrawal/withholding disposition.
- Native BF16, HiFTNet and BigVGAN cross-cutting work required by those rows is
  complete and green on VAST.
- The exact final branch HEAD passes every static/local no-model gate and the
  remote workspace, Clippy, license and advisory gates.
- The live audit has no CPU partial, no missing binder and no non-artifact row.
- Apple workers cover the existing prepared set (GigaAM v3, GigaAM
  Multilingual, OmniASR CTC 1B, ReazonSpeech NeMo v2, BiCodec and Voice Gender
  Classifier) plus every newly completed checkpoint that lacks an authenticated
  Apple verdict.
- Fresh VAST Apple-transfer packets are generated at the exact final commit,
  hash-verified, transferred directly to Scaleway and then removed with their
  disposable VAST instances.

Scaleway then becomes the last compute environment: record its hardware
fingerprint and run all named CPU/Metal workers with explicit no-fallback
verdicts. Hardware failures return to an implementation/VAST wave and mean the
campaign is not complete.

## Work that necessarily follows the Scaleway compute run

Scaleway can be the final compute service, but it cannot safely be the literal
last external action. Corrected public artifacts should be published only
after their Apple evidence is green. Each exact repository still needs separate
upload authorization and must go through `publish-one.sh`; then the read-only
live audit must be repeated until CPU `partial=0`, `no-runtime-binder=0`,
`not-artifact=0` and Metal `blocked-by-cpu=0`, `cpu-only=0`,
`not-artifact=0`. No upload authorization is inferred from this ledger.
