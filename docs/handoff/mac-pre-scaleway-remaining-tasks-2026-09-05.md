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

The implementation head advanced through `316f6ab5` in this wave. The PR
remote at the start of the wave was `d241305f`; all checks at that remote
commit were green and GitHub reported the PR mergeable. The four
implementation/test commits below were deliberately kept unpushed until their
VAST verification completed. Do not merge the PR as the final Mac-coverage
change while the remaining inventory below is still open.

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
- HT-Demucs Multi commit `d03266ec` binds both variants to the exact ordered
  five-member ensemble, full member SHA-256 values, pinned source revision,
  clean Vokra HEAD and external approval record. Its VAST inspection packet
  includes the complete restricted-load tensor/config manifest and an external
  manifest sidecar hash; paths are absent, non-symlinked and disjoint, and all
  publication remains `NO_UPLOAD`. The former report-only worker no longer
  labels an upstream reference dump as CPU parity: it records
  `REFERENCE_ONLY_CPU_PARITY_NOT_RUN`. The unresolved Python-3.12 torchaudio
  lock and package/license review, MUSDB18 provenance, weight redistribution
  decision, native binder and real CPU parity keep the row explicitly blocked.
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

These commits improve readiness only. They do not decrement the 63-row public
audit until corrected artifacts and final Apple evidence exist under their
separate approval gates.

## Cross-cutting implementation before the final Apple run

These tasks affect multiple model rows and must not be mistaken for Scaleway
work:

- **Native BF16 compute:** replace the remaining upcast-to-F32 shim; validate a
  real BF16 checkpoint plus independent AVX512-BF16 and Arm-BF16 parity. The
  raw-BF16 Metal foundation exists, but that does not close the full task.
- **HiFTNet full GPU generator:** the complete resident CPU/Metal graph, strict
  328-tensor converter/binder, exact pinned config and a nonzero synthetic
  one-final-readback parity harness now exist. The no-upload VAST runner binds
  the authenticated inspection evidence to conversion and independent
  real-weight CPU parity. Run its model-free closure phase first; the reference
  and real-weight phases require explicit owner license approval. After a green
  exact-head VAST result, preserve the GGUF/reference packet for the final
  Apple CPU/Metal worker.
- **BigVGAN full GPU path:** the Linux and Darwin dependency/native archive
  graphs are evidence-complete, but owner/legal sign-off remains fail-closed.
  After that approval, VAST-compile the model adapter and run a fixed real
  artifact against an independent reference; leave Metal hardware parity for
  the final Scaleway run.
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
| `vokra/audiogen-medium` | Replace the LM-only artifact with the authenticated T5 conditioner and official 16-kHz EnCodec companion contract. |
| `vokra/bicodec` | Replace the permissive provenance with the audited CC-BY-NC-SA-4.0 research-only identity and bind the strict runtime. |
| `vokra/canary-1b-flash` | Convert the full released encoder, four-layer AED decoder and tokenizer instead of the encoder-only live GGUF. |
| `vokra/canary-1b-v2` | Replace the duplicated timestamp auxiliary checkpoint with the correct main checkpoint, decoder and tokenizer. |
| `vokra/conv-tasnet-libri1mix` | Keep the corrected 345-tensor topology, but resolve the conflicting CC-BY-SA/WHAM declarations before any replacement. |
| `vokra/htdemucs-multi` | Authenticate the five-member ensemble configuration, ordering, weights, dependency/license closure and native runtime; the digest inspection alone is not parity. |
| `vokra/lang-id-voxlingua107` | Add the XVector classifier, label vocabulary and exact topology; the live embedding-only artifact is incomplete. |
| `vokra/mms-1b-all-base` | Define a dedicated CC-BY-NC backbone-plus-language-adapter contract and vocabulary; the 8.9-MB adapter is not the 1B model. |
| `vokra/moss-audio-4b-instruct` | Authenticate its distinct topology and add a strict binder; the broad `moss_tts` tag is insufficient. |
| `vokra/moss-audio-8b-instruct` | Authenticate its distinct topology and add a strict binder; the broad `moss_tts` tag is insufficient. |
| `vokra/moss-audio-tokenizer-nano` | Replace the artifact mis-stamped as the Full variant with correct Nano name, variant and provenance. |
| `vokra/moss-tts-local-transformer-v1.5` | Complete the distinct 48-kHz stereo tokenizer-v2 companion boundary. |
| `vokra/nsnet2` | Resolve live MIT provenance against the audited upstream CC-BY-4.0 identity before replacement. |
| `vokra/qwen3-asr-0.6b` | Regenerate with the three execution metadata keys and all five authenticated tokenizer/chat/generation sidecars. |
| `vokra/qwen3-asr-1.7b` | Regenerate with the three execution metadata keys and all five authenticated tokenizer/chat/generation sidecars. |
| `vokra/qwen3-tts-12hz-0.6b-base` | Regenerate with variant topology, speaker-encoder contract, embedded BPE sidecars and the explicit 12-Hz speech-tokenizer companion. |
| `vokra/qwen3-tts-12hz-0.6b-customvoice` | Correct the Base mis-stamp, add variant metadata/BPE sidecars and authenticate the 12-Hz companion. |
| `vokra/qwen3-tts-12hz-1.7b-base` | Correct the 0.6B mis-stamp/topology, add BPE sidecars and authenticate the 12-Hz companion. |
| `vokra/qwen3-tts-12hz-1.7b-customvoice` | Correct the 0.6B mis-stamp/topology, add BPE sidecars and authenticate the 12-Hz companion. |
| `vokra/reazonspeech-nemo-v2` | Regenerate with the embedded 3,000-piece vocabulary and runtime-axis metadata; repeat exact-head VAST evidence. |
| `vokra/rmvpe` | Resolve the absence of an upstream license for the exact source repository; the live MIT stamp cannot be accepted by inference. |
| `vokra/sbv2-v2-jp-extra-base` | Replace raw legacy tensor names with the strict converter/runtime metadata and close the production Japanese G2P boundary. |
| `vokra/speechbrain-spkrec-ecapa-voxceleb` | Replace or repair the artifact whose tensor data extends outside the declared file bounds, then rerun strict parity. |
| `vokra/voice-gender-classifier` | Exact-head corrected conversion and official CPU parity are green at `df7f5574`; regenerate its authenticated Apple packet for the final CPU/Metal worker. Publish the replacement only after that passes and separate upload authorization is given. |
| `vokra/wespeaker` | Resolve Apache-vs-CC-BY-4.0 provenance and attribution, then produce the strict artifact. |
| `vokra/xy-tokenizer` | Provide a real authenticated tensor payload and verify topology/dependency closure; the live file is metadata-only. |
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
| `vokra/chattts` | Clean native composite plus AGPL/source, CC-BY-NC weight, dependency and personality/voice policy closure. |
| `vokra/clap-htsat-fused` | Complete the released audio/text preprocessing and fused inference contract with real parity. |
| `vokra/cosyvoice2-0.5b` | HiFT checkpoint/source/config, strict converter/binder and guarded VAST CPU-parity runner are complete at source level. Run the model-free closure phase, obtain explicit owner approval before any HiFT model/reference execution, then obtain exact-head VAST CPU parity. LLM/Flow are component-bound but the full flow/codec/vocoder composition remains incomplete; the official broad closure still imports forbidden `soxr`. |
| `vokra/dia-1.6b` | Exact DAC proof, dependency review and complete native composition. |
| `vokra/firered-asr-aed-l` | Resolve the remaining dependency rows and binary CMVN/tokenizer/config contracts, then implement complete AED inference. |
| `vokra/fun-cosyvoice3-0.5b-2512` | Find an exact allowed route around the current `soxr` closure, then finish the full composite. |
| `vokra/irodori-tts-500m-v3` | Find an authenticated Python-3.12 reference route that avoids the current `librosa -> soxr` dependency; more RAM or Scaleway cannot solve this. |
| `vokra/kyutai-stt-2.6b-en` | Implement the dedicated decoder seam required by its `dep_q=0` release instead of forcing the shared Moshi `dep_q>=1` contract. |
| `vokra/owsm-v4-medium-1b` | Finish writer contract, ESPnet S2T frontend/subsampling/E-Branchformer/decoder, joint CTC-attention search and token semantics. |
| `vokra/sortformer-diar-4spk-v1` | Bind the real archive/config and complete native diarization plus independent real parity. |
| `vokra/vibevoice-1.5b` | Close dependency approvals, complete native runtime and execute the real workers. |
| `vokra/voxcpm-0.5b` | Add the missing AudioVAE/tokenizer companions and full native composite. |
| `vokra/zonos-v0.1-transformer` | Complete the DAC/code/PCM contract and execute the real worker. |

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
| `vokra/hibiki-2b` | Complete native streaming translation, dependency and dataset contracts. |
| `vokra/kimi-audio` | Authenticate and implement the roughly 42.6-GB multi-component release; all model work is VAST-only. |
| `vokra/kyutai-tts-1.6b-en-fr` | Separate voice/model/source licenses and implement native demux plus Mimi composition. |
| `vokra/qwen2-5-omni-7b` | Complete native multimodal streaming, dependency/license and independent parity contracts. |
| `vokra/qwen2-audio-7b-instruct` | Resolve the missing exact source-repository license, then implement the native audio-language path. |
| `vokra/sgmse-voicebank` | Exact NCSN++ role mapping, strict binder, score graph, sampler and Apple score/full-enhancement worker are source-complete. Rerun the exact runtime-bound score comparator, then generate/verify the full 4,096-sample official enhancement packet and run Linux CPU parity on VAST. Only the resulting packet proceeds to final Scaleway CPU/Metal execution. |
| `vokra/step-audio2-mini` | Complete multi-component native S2S and all source/weight/dependency licenses. |
| `vokra/vibevoice-asr` | Authenticate the release and implement the complete native ASR path, dependencies and parity. |
| `vokra/vibevoice-realtime-0.5b` | Replace arbitrary BF16 pass-through staging with a strict complete realtime runtime and parity. |
| `vokra/vieneu-tts-v3-turbo` | Complete dependency/license/voice-cloning policy and the composite runtime. |
| `vokra/xtts-v2` | Resolve CPML/source/dependency and voice-cloning consent gates, then implement native GPT/DVAE/HiFiGAN composition. |

## Routed but intentionally partial composites (2)

- `vokra/csm-1b`: replace the synthesized bridge with the complete released
  companion and audio-generation contract; resolve certifi/tqdm/
  typing-extensions/NumPy policy and run independent real parity.
- `vokra/ultravox-v0-5-llama-3-2-1b`: authenticate the gated Meta companion
  digest/tokenizer/chat boundary and its license/dependency closure; a native
  Whisper tower plus projector alone is not a complete model.

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
