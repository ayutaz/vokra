# Mac CPU / Metal completion plan (2026-08-30)

## Objective and authority boundary

The objective is full Mac CPU and Apple Metal coverage for every public Vokra
model repository, with real-artifact and independent-reference evidence.  This
plan does not narrow completion to the currently staged five-model Apple set
across three packets. The authoritative implementation ledger remains
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

## Audited baseline (reconciled 2026-08-31)

The live read-only Hugging Face audit was repeated at clean branch commit
`9f69277d8a0d5df574c1ee95563bd1f005de91d0` on 2026-08-31. It returned 194
public repositories, 193 GGUF-bearing repositories and 198 GGUF files:

| Dimension | Complete | Remaining |
|---|---:|---:|
| Mac CPU | 131 | 42 partial + 20 no-runtime-binder + 1 non-artifact |
| Apple Metal | 129 | 62 blocked by CPU + 2 CPU-only + 1 non-artifact |

The two CPU-complete repositories still classified as source-level CPU-only are
`vokra/sber-gigaam-multilingual` and `vokra/sber-gigaam-v3`.  OmniASR-CTC-1B
is now classified as a complete Metal code route, raising source-level Metal
coverage from 128 to 129, but it still needs the same authenticated Apple CPU
and Metal hardware evidence as the two GigaAM rows.  The Wave A Scaleway packet
therefore deliberately still contains all three models.

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

## Wave A pre-Scaleway checkpoint (2026-08-30)

Wave A source preparation is complete at clean runtime/Apple target commit
`bc9d1db2bbf230f09ce4f3f68003a1c11f80e0e1`.  Later management-documentation
commits are not part of the authenticated packet identity; Scaleway must check
out this exact code commit.  No model artifact was copied to or executed on the
maintainer Mac.  The only attempted local Rust test was stopped when its
dev-dependency expansion reached `vokra-models`; all broad Cargo and
real-weight work stayed on VAST after that point.

VAST instance `49168183` produced the following no-upload evidence:

- `cargo test --locked --workspace` passed at behavior commit `bc8a8e36` after
  its only discovered fixture-extent defect was corrected.  Final HEAD
  `bc9d1db2` adds only documented Clippy suppressions to the same compute-seam
  signatures.
- `cargo clippy --locked --workspace --all-targets --all-features -- -D
  warnings` passed at exact Apple target `bc9d1db2`.
- `cargo deny check licenses advisories bans` and `cargo audit` passed.  The
  existing unmatched `libfuzzer-sys` license-exception warning remains
  non-fatal and unrelated to Wave A.
- GigaAM v3, GigaAM Multilingual and OmniASR-CTC-1B real-weight CPU parity all
  passed at exact Apple target.  The GigaAM approvals record that code commit,
  GGUF and reference-manifest digests; OmniASR recorded exact tokens and
  `frontend_max_abs=2.918243408e-4`, `encoder_max_abs=1.520276070e-3` and
  `logits_max_abs=1.450002193e-3`.

The immutable Scaleway input packet is
`/root/scratchpad/apple-transfer-bc9d1db2` on retained VAST storage: 4.9 GB,
30 regular files, no symlinks.  Every file passes
`apple-transfer-bc9d1db2.SHA256SUMS`; the manifest digest is
`c96eee3c61ec85b589a488deff21668097ed4e94f96b4654b990706098f6f606`.
The instance was returned to the stopped state after packet verification and
must not be destroyed until the packet and small evidence are copied to and
verified on Scaleway.

This Wave A packet can decide only its three Apple CPU/Metal rows. The later
ReazonSpeech and BiCodec packets expand the current Apple-ready set to five
models, but Scaleway still cannot close the 63 CPU-open public rows,
publication/replacement gates, RMVPE license decision or SeamlessM4T
non-artifact decision described below. Those remain separate
VAST/source/license/publication waves.

The first Wave B source/CPU item has since advanced: ReazonSpeech-NeMo-v2 is
green at exact code commit `a59c48c8da103ac14fe837cd2e0252b5266ac093` for
conversion, independent official-NeMo reference, native CPU encoder/ALSD
tokens and text, CLI output, workspace tests, Clippy, license checks and
advisory audit. Its verified Scaleway packet is
`/root/scratchpad/apple-transfer-reazon-a59c48c8`; the packet-manifest SHA-256
is `48874cf71497e347019c156f49409d74428734e840cc0302d8626ae5780679ed`.
This closes the Reazon source/VAST leg only. Apple CPU/Metal and an explicitly
authorized public replacement remain open, so the live 63-row CPU-open count
does not change. The other Wave B-D source/VAST/license tasks are also still
open; Scaleway alone cannot complete them.

The next Wave B dependency-evidence checkpoint completed without model
acquisition on temporary VAST instance `49232927`.  The reviewed source HEAD is
`afe0b77551290ba8525edb7baf581e0c221fbdda`; Bark and Parler-TTS were audited at
its immediate Qwen-only predecessor `152a4cccec785f4f419433f2f95eb234772fe163`.
The exact Linux closures matched: Qwen3-ASR 91 installed rows plus four explicit
inactive rows out of 95 lock rows, Bark 34 installed rows plus one virtual row,
and Parler-TTS 26 installed rows plus one virtual row.  Exact locked PyPI sdist
inspection recovered publisher bytes for Qwen Cython/tokenizers, Bark
safetensors/tokenizers and Parler tokenizers.  It also proved that the exact
`tqdm==4.70.0` sdist has no bounded LICENSE/COPYING/NOTICE/COPYRIGHT candidate;
Qwen additionally retains one no-sdist blocker (`dynet38==2.2`) and four other
exact-sdist no-candidate blockers.  Fixed HF model/DAC LICENSE paths remain 404
for both Qwen checkpoints, both Bark checkpoints and all three Parler model/DAC
objects; Bark also lacks a pinned source-license revision.  No license class or
owner sign-off was inferred from those facts.

The recovered JSON SHA-256 values are
`052a11f747b6840b6179f3f85044a9585e151a3349d349bbffa96b63cc8ce07f`
(Qwen),
`3e589a4d74cce49a12674840a745a7f8b911ccfbcaf54638ebb50593feace517`
(Bark) and
`ef2c7631d18d644750d1d485ef81f58368cd38f8c1ecb6a011d27ee144224f03`
(Parler-TTS).  No checkpoint, model weight, Cargo command or upload was part of
the job.  Instance `49232927` was destroyed after evidence recovery; retained
Scaleway-transfer instance `49168183` remains stopped and unchanged.

The following MOSS/Ultravox/NeuTTS dependency-audit correction checkpoint is
clean at `f22bfdfd98c47218057bf435f2aef5dc49fb0057`.  MOSS now retains the raw
lock artifact rows needed to authenticate exact sdists (`0d5e3c9a`), while
NeuTTS (`2f01d6c4`) and Ultravox (`f22bfdfd`) derive their Linux x86_64
installed closures by marker-aware traversal from the virtual project root.
All six model-free audit/wrapper/worker self-tests, shell syntax, ShellCheck,
zero-dependency, forbidden-symbol, fixture-EOL and pipefail gates passed.  No
model, Torch import, conversion or Cargo model build ran on the maintainer Mac.
The current verified incremental transfer bundle is
`/private/tmp/vokra-dependency-audit-corrections-a606b3c2.bundle`, requires
`d2ef5cfb8373a640f438a199fa779b3daaadc103`, resolves to management HEAD
`a606b3c2a70a511d15d8bd61ecc5d7d5fbc34a15`, and has SHA-256
`fbdc286fd0d0f384c28f0ed3a0634fe76b92239a267ce0578569d72eb8ee7b89`.
Historical disposable audit instance `49242592` is no longer an active retained
instance or restart target. The corrected three-family MOSS/Ultravox/NeuTTS
Linux rerun is not claimed by this plan and remains a separate evidence task.

BiCodec subsequently completed its exact remote evidence path. The strict
binder authenticates all 840 F32 tensors and the native decode-only runtime
reaches the semantic latent, d-vector, prenet and waveform outputs. Official
VAST parity passed with maximum absolute errors `1.907348633e-6`,
`1.847743988e-6`, `7.539987564e-6` and `6.183981895e-7` respectively. PCM
encode remains explicitly unsupported, the model stays research-only and
`NO_UPLOAD`, and real Apple CPU/Metal execution remains open. Its verified
Apple packet is `/root/scratchpad/apple-transfer-bicodec-5cd97d12`, 600 MB
across 12 regular files with no symlinks; the manifest SHA-256 is
`0a80edb51e88d17ce8f243ee58523551baf7d9fc5a848a17dc9c3fdecaf8d18f`.

The model-free XY-Tokenizer audit accounts for all 57 active dependency rows:
51 have exact bounded evidence, while SciPy, setuptools, soxr, SymPy,
tokenizers and tqdm remain fail-closed. The report SHA-256 is
`604e9cc74a5814f97bcd2be106e1f620f5f4d2d45052ce3c78fb485583f17210`
and the partial evidence SHA-256 is
`3e2471835be2b5cb767f3181050c98ff82dc12e039c9b4257af684d713306ffc`;
no model or Torch route was imported. HT-Demucs Multi is separately
`BLOCKED_UNSATISFIABLE_PY312_TORCHAUDIO`: its exact upstream
`torchaudio>=0.8,<2.1` constraint has no Python 3.12-compatible release, so
neither VAST nor Scaleway can clear it.

The complete Wave D model-free worker preflight is also green at management
HEAD `50635dca`. Nineteen fixed-revision VAST workers cover all twenty Wave D
rows (the single MOSS-Audio worker covers both historical 4B and 8B files).
With `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, every
worker's explicit `--self-test` passed, followed by Bash syntax and ShellCheck
for the same nineteen files. These tests exercised only fake/materialized
trees, argument and fail-closed contracts; no checkpoint, model execution,
dependency synchronization, conversion, Cargo command or network access was
used. This proves the remote launch contracts are staged, not that any tensor
inventory, native runtime, CPU parity or Metal parity is complete.

The corresponding model-free preflight for the currently open Wave B/C
families exercised forty-four inspection/validation workers at management HEAD
`52c52d29`: forty-three `--self-test` paths passed and the CosyVoice3 inspector
alone returned its intentional exit 2 before acquisition because its dedicated
reference lock is forbidden and absent. All forty-four files passed Bash
syntax and ShellCheck. The fixed upstream
`examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` binds
`matcha.utils.audio.mel_spectrogram` in both the GAN and feature-extractor
graphs, matching the recorded `librosa`/`soxr` closure blocker. Although the
same fixed source exposes lower-level flow and HiFT classes, no reviewed,
dependency-clean independent-reference import route has yet been
authenticated, so a VAST run alone must not clear that gate.

All forty-three staged Apple worker `--self-test` paths also passed at the same
management HEAD, followed by Bash syntax and ShellCheck for the same files.
Those are offline launch/evidence-contract tests only: no Apple hardware,
checkpoint, model execution, Cargo command or numerical verdict was involved.

## Completed VAST batch and next external stage (2026-08-31)

The current unblocked Linux/VAST batch is complete at exact code commit
`9f69277d8a0d5df574c1ee95563bd1f005de91d0`. Workspace tests recorded 310
passing result groups and zero failures (log SHA-256
`c6a9c5b1604ed53c02902bd311062f7a4646f7f9a455993489f625c96769b139`);
strict Clippy, `cargo deny` and `cargo audit` also passed with log SHA-256 values
`373ce57e806cb33ec0a7b16e49174ffcf0b274b38cfbb8d02bb7813b976aa33c`,
`43ba882d8949aa5a6145e86a1bdf66d602057591b4d462390aa8c4519c0e9666`
and `82e60f15564fdf549048e5f14a0d6a8e97a09b05fc875a389efe7da180d60c36`.
No VAST restart or additional Linux run is required before the prepared Apple
stage.

Read-only status confirmation shows retained instances `49168183` and
`49261078` with `cur_state=stopped`, `intended_status=stopped` and
`actual_status=exited`. They consume no compute; their 500 GB and 200 GB
retained storage continues to incur charges. Historical instance `49242592` is
not an active restart target. The two retained instances hold three verified
packets:

1. `/root/scratchpad/apple-transfer-bc9d1db2` for GigaAM v3, GigaAM
   Multilingual and OmniASR CTC 1B: 4.9 GB, 30 regular files, no symlinks,
   manifest SHA-256
   `c96eee3c61ec85b589a488deff21668097ed4e94f96b4654b990706098f6f606`.
2. `/root/scratchpad/apple-transfer-reazon-a59c48c8` for ReazonSpeech NeMo v2:
   11 regular files, manifest SHA-256
   `48874cf71497e347019c156f49409d74428734e840cc0302d8626ae5780679ed`.
3. `/root/scratchpad/apple-transfer-bicodec-5cd97d12` for BiCodec: 600 MB,
   12 regular files, no symlinks, manifest SHA-256
   `0a80edb51e88d17ce8f243ee58523551baf7d9fc5a848a17dc9c3fdecaf8d18f`.

The next stage is Scaleway Apple Silicon. No Scaleway instance or SSH access
has been supplied and no Apple run has started. Provision an official M4-M
with 32 GiB RAM, 1.02 TB storage, macOS/Scaleway Dev OS and Xcode; an M4 Pro XL
with 64 GiB is optional. Do not use Asahi Linux or FileVault. Share only the
resulting SSH command, never a private key or API token. Resume the retained
VAST instances only for direct packet transfer, verify each manifest on
Scaleway, execute the five model-specific CPU/Metal workers, recover the small
signed evidence and then destroy both VAST instances.

This Apple stage can decide only those five prepared rows. It cannot close the
remaining 62 CPU-blocked repositories, the six XY dependency blockers, the
HT-Demucs Python 3.12 contradiction, MOSS/Ultravox/NeuTTS remote reruns,
publication/replacement gates or owner/license decisions. No Hugging Face
upload is part of the stage.

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
are verified. The currently stopped/exited retained instances `49168183` and
`49261078` may be resumed only for their recorded packet transfer; their
storage charges continue until final destruction.

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
