# Mac CPU / Metal completion plan (2026-08-30)

## Objective and authority boundary

The objective is full Mac CPU and Apple Metal coverage for every public Vokra
model repository, with real-artifact and independent-reference evidence.  This
plan does not narrow completion to the currently staged three-model Apple
packet.  The authoritative implementation ledger remains
`docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md`; this document fixes
the execution order and cloud boundary for the remaining work.

Model conversion, checkpoint preparation, upstream reference generation,
real-weight execution, every `-p vokra-models` Cargo command and workspace-wide
Cargo run must stay on cloud infrastructure.  The maintainer Mac is limited to
source review, management documentation, shell/static gates, `cargo fmt`, cheap
Cargo metadata inspection and small no-model self-tests.  No model artifact is
pulled back to the maintainer Mac.

VAST is the Linux/x86_64 conversion, reference, CPU-parity and heavy-Cargo
worker.  Scaleway Apple Silicon is the macOS arm64 CPU/Metal worker.  Uploading
or withdrawing a Hugging Face artifact remains a separate irreversible action:
no `--push`, repo deletion or public replacement is authorized by this plan.

## Audited baseline

The live read-only Hugging Face audit was repeated at branch HEAD
`e312b997b3c92f434a75cdc63fa00d06e79adf2d` on 2026-08-30.  It returned 194
public repositories, 193 GGUF-bearing repositories and 198 GGUF files:

| Dimension | Complete | Remaining |
|---|---:|---:|
| Mac CPU | 131 | 42 partial + 20 no-runtime-binder + 1 non-artifact |
| Apple Metal | 128 | 62 blocked by CPU + 3 CPU-only + 1 non-artifact |

The three CPU-complete/Metal-open repositories are
`vokra/omniasr-ctc-1b`, `vokra/sber-gigaam-multilingual` and
`vokra/sber-gigaam-v3`.

The 63 CPU-open rows split into execution classes rather than one misleading
flat list:

| Class | Count | Required route |
|---|---:|---|
| Public-artifact-specific blocker | 27 | 22 replacement/contract repairs and 5 artifact-specific missing binders; VAST no-upload conversion/parity, then separately authorized publication |
| Bound but incomplete runtime | 18 | Complete the native forward/composite contract, then VAST CPU parity and Scaleway Metal parity |
| Generic no-runtime-binder | 15 | Implement converter/binder/native runtime from pinned primary sources, then the full parity chain |
| Routed but intentionally partial | 2 | Complete the CSM and Ultravox companion/runtime boundaries |
| Non-artifact repository | 1 | Produce a gated GGUF or withdraw the repository under separate authorization |

The detailed repository names and per-model blockers remain in Waves 1-4 of
the authoritative ledger.  No row is promoted merely because an inspection,
converter, synthesized forward or device-less build succeeds.

## Execution order

### Wave A: remove the three CPU-only Metal gaps

1. Route the complete learned graphs of GigaAM v3 and GigaAM Multilingual
   through the `Compute` seam.  Selecting Metal must either execute every
   learned operation on Metal or return a structured unsupported error before
   execution; silent host execution is forbidden.
2. Audit OmniASR CTC's complete learned-op route and conservative Metal
   inventory entry.  Keep real Apple evidence separate from source-level
   reachability.
3. Replace the two GigaAM `OPEN_UNSUPPORTED`-only Apple contracts with
   authenticated CPU/Metal parity workers.  Preserve exact-token gates and the
   existing independent official VAST reference packets; do not change numeric
   bounds from observations.
4. Commit the source wave, transfer an unpushed git bundle, and run formatting,
   workspace tests, all-target Clippy, license/advisory checks and focused
   real-weight CPU parity on VAST.
5. Only after the VAST result is green, transfer the authenticated 4.9-GB
   packet directly to Scaleway, verify the packet manifest, and run CPU/Metal
   parity.  Hardware-only failures return to the implementation wave.

### Wave B: close the 22 partial public artifacts

Process the existing no-upload workers in small families: ASR first, then TTS,
then codec/music/separation/classification.  Each family must bind fixed source
revisions, exact artifact identities, dependency/license evidence, an
independent upstream oracle, a strict runtime manifest and a portable Apple
command before conversion begins.  All conversion/reference/model execution is
VAST-only even below the historical 2-GB threshold.

For source-complete routes such as Qwen3-ASR, Canary, ReazonSpeech and Qwen3-TTS,
the target is an authenticated replacement dry-run rather than a second model
implementation.  A dry-run can close code and evidence gates, but the live HF
audit stays partial until the exact repository replacement is separately
authorized and published through the gated publish workflow.

### Wave C: complete 18 bound-partial runtimes and two routed composites

Finish one model family per commit group.  Before coding, authenticate the
upstream configuration, tensor roles, tokenizer/codec companions, source and
weight licenses.  Implement only native first-party Rust runtime paths; no
ONNX/ORT/protobuf runtime and no external runtime crates.  Every learned-op set
must preflight one complete backend.  Generate independent references and run
real-weight CPU parity on VAST before requesting an Apple run.

CSM and Ultravox remain in this wave even though they have routed code: their
released-artifact companion boundaries are part of completion and may not be
represented as complete by a tower-only or synthesized bridge.

### Wave D: implement the 20 no-runtime-binder repositories

Handle the five artifact-specific binder failures first, because their public
bytes already provide concrete negative contracts.  Then implement the fifteen
generic no-binder families in bounded, non-overlapping model waves.  Each wave
includes converter, binder, native forward, CLI route, provenance/license
policy, independent reference, CPU parity, Metal preflight, Apple worker and
documentation in the same logical change.

Unsupported or nonredistributable weights stay fail-closed.  Engine support may
land without official weight distribution when the audited license requires a
research-only or no-distribution boundary; that does not close a public model
row until its exact done-condition is satisfied.

### Wave E: non-artifact and publication closure

`vokra/seamless-m4t-v2-large` must receive a real, gated GGUF or be withdrawn.
Either path changes public state and requires explicit repository-scoped
authorization.  After all replacement candidates have green VAST and Scaleway
evidence, request one exact authorization list and publish only through the
repository's gated workflow.  Repeat the live audit after every public batch.

## Per-wave evidence and commit contract

Each implementation wave is committed separately after Sol review.  The
minimum evidence attached to a wave is:

1. Clean source diff and `git diff --check`.
2. Local no-model gates: formatting, shell syntax/self-tests, zero-dependency,
   forbidden-symbol, converter/binder and applicable license/document checks.
3. VAST commit identity matching the local git bundle.
4. VAST workspace tests, all-target Clippy, `cargo deny`, `cargo audit` and the
   named real-weight CPU parity/convert jobs.
5. Scaleway hardware fingerprint, input SHA-256 manifest, exact CPU/Metal test
   names, numerical/token verdicts and retained small evidence.
6. Updated live-audit row and an explicit `NO_UPLOAD` record unless that exact
   repository was authorized.

VAST instances are destroyed after their evidence and required Apple transfer
are verified.  The currently stopped retained instance `49168183` may be
resumed only for the recorded packet transfer or the next named remote job; its
storage charge continues until final destruction.

## Final completion proof

Completion requires all of the following at the same reviewed HEAD and public
artifact state:

- CPU `partial=0`, `no-runtime-binder=0`, `not-artifact=0`.
- Metal `blocked-by-cpu=0`, `cpu-only=0`, `not-artifact=0`.
- Every public GGUF has an authenticated real-file CPU verdict and independent
  numerical/output evidence.
- Every supported checkpoint has an Apple-hardware Metal verdict with no silent
  CPU fallback.
- All local static gates and remote workspace/license/advisory gates pass.
- No unintended VAST instance, retained cloud artifact or unverified public
  upload remains.

Until every item above is evidenced, this campaign remains open.
