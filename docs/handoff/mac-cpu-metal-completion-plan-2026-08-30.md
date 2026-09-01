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

## Audited baseline (reconciled 2026-09-01)

The active branch is workspace `0.3.0`; immediately before this documentation
refresh its remote head was
`d8a93bc3acdb8f9648ecb8dd37ef41657fbf425b` in open PR #79, with 109 passing
checks, 13 expected skips, and no failures or pending checks. The authenticated
runtime/VAST checkpoint `9f69277d8a0d5df574c1ee95563bd1f005de91d0` and
evidence/package checkpoint `5cd97d124bc9eb9d2bb7b0367541dcd1492e4d1e`
remain historical workspace `0.2.0` evidence.

The latest live read-only Hugging Face audit was repeated at clean local branch
commit `8b63dea72350a45a4c831d661ad707a9c664b565` on 2026-09-01. It returned
194 public repositories, 193 GGUF-bearing repositories and 198 GGUF files:

| Dimension | Complete | Remaining |
|---|---:|---:|
| Mac CPU | 131 | 43 partial + 19 no-runtime-binder + 1 non-artifact |
| Apple Metal | 131 | 62 blocked by CPU + 1 non-artifact |

The total number of CPU-open GGUF repositories remains 62. OWSM v4 medium 1B
moved from `no-runtime-binder` to `partial` because its strict structural binder
is now present; this is a classification change, not a CPU or Metal PASS. The
remote PR head remains `5bb06a42` until the seven code-bearing commits through
`8b63dea7` receive their exact-head full-workspace VAST closure and are pushed.

`vokra/sber-gigaam-multilingual`, `vokra/sber-gigaam-v3` and OmniASR-CTC-1B
are now classified as complete Metal code routes.  The two GigaAM graphs route
their complete learned-op sets through the selected `Compute` backend, and the
pull-request audit now fails closed if a CPU-complete public architecture is
missing from the conservative Metal registry.  The live audit therefore has
zero source-level CPU-only rows.  This does not establish an Apple-hardware
verdict: all three models still need authenticated Apple CPU and Metal evidence,
so the Wave A Scaleway packet deliberately continues to contain all three.

The 63 CPU-open rows split into execution classes rather than one misleading
flat list:

| Class | Count | Required route |
|---|---:|---|
| Public-artifact-specific blocker | 27 | 22 replacement/contract repairs and 5 artifact-specific missing binders; VAST no-upload conversion/parity, then separately authorized publication |
| Bound but incomplete runtime | 19 | Complete the native forward/composite contract, then VAST CPU parity and Scaleway Metal parity |
| Generic no-runtime-binder | 14 | Implement converter/binder/native runtime from pinned primary sources, then the full parity chain |
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

At the 2026-08-31 checkpoint this Apple stage could decide only those five
prepared rows. It could not close the remaining 62 CPU-blocked repositories,
the six XY dependency blockers, the HT-Demucs Python 3.12 contradiction,
MOSS/Ultravox/NeuTTS remote reruns, publication/replacement gates or
owner/license decisions. No Hugging Face upload is part of the stage.

## 2026-09-01 cloud evidence update

Disposable VAST instance `49422639` checked out clean code commit
`8f968961a786fc843ba059175d9f8c2e98c04f58` and completed the next
Scaleway-independent batch. No model ran on the maintainer Mac and no public
artifact was uploaded.

Voice Gender Classifier now has authenticated CPU evidence for the corrected
dedicated architecture. The fixed upstream checkpoint was
`JaesungHuh/voice-gender-classifier` at
`db1222153bd60337e900be22add7af180452adc0`; its 61,907,512-byte
`model.safetensors` matched SHA-256
`2d8e0be1fdf159d60d5087416e6f6277c5e30ce9e33a61c767a9a409e6c503c5`.
The independent source oracle was pinned at
`49bcbecfd929ba5a043bde645fdff1a375eb79c7`. The converter wrote 202 tensors
to the corrected GGUF with SHA-256
`afb03696d8a640d5d701ea0c136bb065cac648cbfe905a5dcc4eae04e0769b1a`.
The preregistered FP32 bound remained `0.01`; observed maximum absolute errors
were `0.000072479` for features, `0.000058383` for embeddings,
`0.000005677` for logits and `0.000002176` for probabilities. The official
label also matched exactly. The evidence archive SHA-256 is
`fc9e46d1715cca5ee094ba082817a63b42376809998f007c1de2d183135e4b7d`.
The existing 202-tensor GGUF/reference packet
`/private/tmp/voice-gender-apple-packet-f74374ab.tar.gz` has SHA-256
`f755fbdc3a2146ca41942501d80415eff573f1dc8e56201ee474d93d4d3d7261`;
Scaleway must use the current branch worker and verify this packet before
running Apple CPU/Metal parity. No Metal verdict is claimed yet.

The same exact code commit passed `cargo test --locked --workspace`, strict
all-target/all-feature Clippy, `cargo deny check licenses advisories bans` and
`cargo audit` on VAST. Tests and doctests completed with zero failures. Clippy
retained only the existing non-fatal `clippy.toml`/`Cargo.toml` MSRV warning,
and `cargo deny` retained only the existing unmatched `libfuzzer-sys`
exception warning.

FireRedASR-AED-L advanced from source-only inspection to authenticated
checkpoint preparation without being misclassified as runnable. The exact
4,678,597,714-byte `model.pth.tar` at revision
`e57f5960d03cff1071ff7acbb409314d1e70ed3d` matched SHA-256
`12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3`.
Safe tensor-only loading authenticated 940 F32 tensors with no duplicate names
or shared storage and no stripped counters. The measured checkpoint contract
is 80 input bins, 7,832 output symbols, 16 encoder and 16 decoder layers,
`d_model=1280`, 20 heads, `d_inner=5120` and kernel size 33. The prepared
4,678,403,512-byte safetensors file had SHA-256
`5e8608d5a23af0761cb6bb52d08ee19a6476b8c324799eff3c63c9785cef583e`.
Only the two manifests and validation log were recovered; their archive
SHA-256 is
`475fa7e54c4636db4dd17bfb340d7c9f2709ab5af24c71433222f93d6ce79c78`.
Status remains `NOT_IMPLEMENTED_FAIL_CLOSED`: an independently pinned upstream
importer, native converter/runtime, CPU parity and complete Metal graph are
still required. The `0.01` FP32 bound is preregistered but was not run.

The corrected MOSS Audio Tokenizer v2, Ultravox and NeuTTS Air dependency
audits were then rerun in their exact frozen Linux x86_64 environments. All
three completed factual collection and remained `BLOCKED`, not because a job
was skipped but because exact publication evidence is unavailable:

- MOSS: `tqdm==4.70.0` has a hash-verified sdist with no bounded
  LICENSE/COPYING/NOTICE/COPYRIGHT candidate; `triton==3.3.1` has neither an
  installed publisher file nor a locked sdist. Report SHA-256:
  `e3fb6d4f7c66d9a2df362bd42643d07266d859f01ee993d6443f6ee4293d9c46`.
- Ultravox: the same `tqdm` fact remains, the exact upstream LICENSE path
  returned 404 and the gated Llama companion LICENSE path returned 401.
  Report SHA-256:
  `21da50de30ee63b18343376ff9b292d7d10def38cfb188b8eb689f00da4f22aa`.
- NeuTTS Air: the same `tqdm` fact remains and the fixed gated upstream LICENSE
  path returned 401. Report SHA-256:
  `bbac6840aee9431ef74354301954db0d1bc4e6b756ced9d510105c8cb5b8061a`.

No license class, owner sign-off or publication approval was inferred. These
three remote reruns are now complete as evidence tasks; their legal/owner
blockers remain open.

After all small evidence archives were recovered and their remote/local
SHA-256 values matched, disposable instance `49422639` was destroyed and the
VAST API returned `instances: null` for that id. Retained transfer instances
`49168183` and `49261078` remain intentionally stopped. A live 2026-09-01
status read reported storage-only costs of `$0.074074/h` and `$0.022222/h`, or
about `$0.096296/h` (`$2.31/day`) combined. They must stay stopped until the
three retained packets are transferred directly to Scaleway and verified, then
both must be destroyed. The unrelated running instance labeled
`cutetts-s1-preprocess` was not touched.

The prepared Scaleway set is therefore six models: GigaAM v3, GigaAM
Multilingual, OmniASR-CTC-1B, ReazonSpeech-NeMo-v2, BiCodec and Voice Gender
Classifier. This does not change the live Hugging Face audit count before a
separately authorized public replacement, and it does not make FireRed or the
three blocked dependency families Apple-ready.

## 2026-09-01 exact-head continuation checkpoint

The next source wave and its Linux regression closure are clean at exact commit
`3001362ff1d0b21a0055f925bb95b0e8e407b52f`, which was pushed to open PR #79
as the VAST-tested base for this continuation.  The local branch and the VAST
checkout both matched that commit with clean worktrees.  Later focused fixes
are recorded below.  They received a second exact-head remote closure at code
commit `910e49581e2fd5f5b1b42de102ec5c73a31c5745`; this documentation-only
follow-up does not alter that tested source tree.  No checkpoint was converted
or executed on the maintainer Mac.

VAST instance `49469101` ran the exact-head workspace gates.  The workspace
test log contains 310 result groups and totals 7,796 passed, zero failed and 75
ignored tests.  Strict all-target/all-feature Clippy, `cargo deny check
licenses advisories bans` and `cargo audit` also exited zero.  The four logs
passed a bounded credential-pattern scan before recovery and have the following
SHA-256 values:

- workspace tests:
  `318aa433cc171e845ff7c24e67980f1ab201e44ee31e69dac9948f360e25fb73`
- strict Clippy:
  `7ef33bcb9ac37055e56bcbfade71a64236d0919492f0cb1e170a972747971304`
- cargo-deny:
  `3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`
- cargo-audit:
  `4dcc8a000d5f63fd15912dc615eaa80db14ecd5951106e0883f029ea45fccfc9`

The recovered log copies live only under
`/private/tmp/vokra-gates-3001362f/`; their local hashes match the remote
values.  GitHub dependency-review subsequently identified a separate Python
lock vulnerability in FireRed's exact `setuptools==80.9.0` pin
(`GHSA-h35f-9h28-mq5c`, fixed in 83.0.0).  Commit `e1745690` replaces the
incompatible closure with exact `kaldiio==2.18.1` and `setuptools==83.0.0`.
The official KaldiIO wheel SHA-256 is pinned, it has no `pkg_resources`
reference, and the lock, dependency-audit, upstream-reference and worker
self-tests are green without loading a model.  No advisory allow-list bypass
was used.  GitHub dependency-review must still confirm the pushed commit.

The second exact-head run used disposable VAST instance `49485395` with 64
effective x86_64 CPU cores and 128,791 MiB RAM.  The checkout was clean and
matched `910e4958`; Rust 1.98.0, cargo-deny 0.20.2 and cargo-audit 0.22.2 were
recorded before execution.  The workspace log contains 310 result groups and
totals 7,797 passed, zero failed and 75 ignored tests.  Formatting,
zero-dependency, forbidden-symbol, strict workspace/all-target/all-feature
Clippy, `cargo deny check licenses advisories bans` and `cargo audit` all
exited zero.  A bounded credential-pattern scan passed before recovery.  The
principal log SHA-256 values are:

- workspace tests:
  `f6758271b940220650fa19387c1cfb765b3c0043b843939456e4fe0a3ce5c046`
- strict Clippy:
  `a01f1bcc3bb146e6e3f2ee1e02791e3a956223fba5e9dadb93afd14d68222ddc`
- cargo-deny:
  `3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`
- cargo-audit:
  `4dcc8a000d5f63fd15912dc615eaa80db14ecd5951106e0883f029ea45fccfc9`

The recovered 708-KB evidence directory is
`/private/tmp/vokra-gates-910e4958/`; every copied log matches its remote
checksum.  Instance `49485395` was then destroyed with its saved disk, and a
read-back returned `instances: null`.

Three additional evidence paths advanced without changing a public-artifact
verdict:

- microWakeWord Path C completed 512 streaming invocations, eleven named
  intermediates, the final output and four reset replays at code checkpoint
  `38b51165`.  The verified report SHA-256 is
  `f805d84e7a34973d8db55ed1aaf0b138bccb48fb0b3e356907591bf5f25f6a26`.
- FireRedASR-AED-L gained native encoder/decoder primitives and strict prepared
  artifact contracts, but its independently pinned Python closure remains
  `BLOCKED_UNREVIEWED_TRANSITIVE` for 27 rows.  No reference PASS is claimed
  and the public row remains partial.
- SGMSE-VoiceBank generated a real upstream NCSN++ score reference at
  checkpoint `35679572`.  The six 64-frame F32 planes and manifest were
  verified; the manifest SHA-256 is
  `3dd740f473e547c3e44ad5156619ad37d6b3f34522a75bff20d14f68e815dc83`.
  Commit `03d67143` adds a consumer that hard-pins that digest, verifies the
  manifest's run-log and six artifact hashes, rejects non-finite/extra/tampered
  inputs, and applies the unchanged FP32 `atol=0.01` gate to future native
  score planes.  Commit `d8886ea3` wires that consumer into the VAST reference
  worker behind an optional native-score directory, after the independent
  reference verifier succeeds.  The consumer, worker self-test and recovered
  real fixture pass locally without model execution.  This authenticates the
  independent oracle and a future comparison path only.  Strict checkpoint
  conversion, complete native graph binding, production of a native score
  dump, native CPU parity and Apple CPU/Metal parity remain open.
- Kyutai STT remains fail-closed.  The pinned primary contract uses
  `dep_q=0`, while the shared Moshi API requires `dep_q>=1`; substituting one
  would invent an unsupported codebook and `1/1/1` depth topology.  A dedicated
  decoder-only runtime seam plus authenticated GGUF/Mimi/tokenizer provenance
  and independent compound parity are still required.

PR #79 also exposed four independent contract failures at the tested base:
the FireRed GGUF prefix was absent from `docs/abi-changelog.md`, the
`sgmse_voicebank` binder was absent from the CLI `BOUND_ARCHES` registry, the
Metal group-normalization slice had two strict-Clippy defects, and a FireRed
no-clobber test asserted the Unix-only text `File exists` on Windows.  These
are fixed in commit `b05db741`: the ABI row matches all 16 converter metadata
keys plus both string arrays, the SGMSE binder remains visibly fail-closed, the
Metal validator uses a typed argument bundle and an explicit safety contract,
and the no-clobber test checks `ErrorKind::AlreadyExists` portably.  The same
commit closes two additional strict-Clippy findings without changing parity
math: the existing test module moves after production items and the mixed-BF16
seed keeps the identical numeric value with consistent hex casing.  Package-
scoped Metal strict Clippy, the focused FireRed test, both registry/ABI shell
gates, formatting and diff checks are green locally.  The exact code commit is
covered by the second VAST workspace run above.

The read-only live Hugging Face audit was repeated at `3001362f` after its ten
offline unit tests passed.  It remains 194 public repositories, 193 GGUF
repositories and 198 GGUF files: CPU `full=131`, `partial=42`,
`no-runtime-binder=20`, `not-artifact=1`; Metal `full=131`,
`blocked-by-cpu=62`, `not-artifact=1`.  There are still zero source-level
CPU-complete/Metal-unsupported rows.  The campaign is therefore not at a
Scaleway-only finish: all 63 CPU-open public rows still require the applicable
source, artifact, dependency, publication and VAST legs before Apple hardware
can provide their final Metal verdicts.

The subsequent source wave records the unresolved Qwen3-ASR license/dependency
evidence without promoting it, hardens SGMSE inspection evidence, makes YuE
XCodec Mini reject duplicate JSON identities during parsing, proves that the
WeSpeaker approval gate precedes work-directory creation, and adds an OWSM v4
medium 1B strict structural binder.  The OWSM binder matches all 1,172
authenticated checkpoint tensor names, shapes and F32 dtypes, but deliberately
returns `NotImplemented` for PCM transcription.  Per-tensor payload mapping,
the native frontend/decoder/tokenizer/CTC-attention route and independent
real-weight parity therefore remain open; this is not a CPU or Metal PASS.

The first exact-bundle regression at code `b61f1d38` passed all 310 workspace
result groups with 7,801 passed, zero failed and 75 ignored tests, then strict
Clippy found 30 missing-doc findings on the new OWSM public surface.  No lint
allowance was added.  Commit `63f137b9` documents the actual units, authenticated
values and structural-only boundary.  Disposable VAST instance `49497103`
then checked out that exact clean commit from bundle SHA-256
`530fb6ac62be48ed4a64d4b5ba9005e25f065895285daa297d3a0531f9aab53c`
and passed formatting, zero-dependency, forbidden-symbol, architecture
handshake, Rust public-API, workspace tests, strict all-target/all-feature
Clippy, `cargo deny check licenses advisories bans` and `cargo audit`.  The
workspace totals remained 7,801 passed, zero failed and 75 ignored across 310
result groups.  The four principal log SHA-256 values are:

- workspace tests:
  `ebf45c458bf671f931fafa7876c2926155d7cb443fbde0baaefec7749b8324ca`
- strict Clippy:
  `e29ac9019f2f3564c91e8828751b9e344caa8c80a1c2eb8027d8bf897b727353`
- cargo-deny:
  `3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`
- cargo-audit:
  `4dcc8a000d5f63fd15912dc615eaa80db14ecd5951106e0883f029ea45fccfc9`

The recovered evidence is under
`/private/tmp/vokra-gates-63f137b9.eYlt4B/`; its bounded credential-pattern
scan passed.  Instance `49497103` was destroyed after recovery.  With the
owner's exact destructive confirmation, redundant instances `49447911`,
`49469101`, `49494353` and `49495037` and their saved disks were also destroyed;
all four read back absent.  Only stopped packet instances `49168183` and
`49261078` remain for their recorded direct Scaleway transfer.  Unrelated
running instance `49466383` is outside this campaign and was not touched.

### 2026-09-01 exact-head model-contract evidence

Disposable VAST instance `49511760` checked out the clean exact local commit
`8b63dea72350a45a4c831d661ad707a9c664b565` from incremental bundle SHA-256
`e2246d3680373c5bbcdefb81260f361b0f31a3704d17530ca26e165131dcfb20`.
The bundle requires the current remote PR base `5bb06a42`. The worker ran eight
fixed stages: OWSM, HTDemucs and SGMSE inspection self-tests and real
inspections, plus the SGMSE reference self-test and real reference. Every
stage matched its expected exit code; the run summary has SHA-256
`a3f1f6bf92bc8ad5a631699d64eda365d9c7b6195e32e58aa404f94ef1b62553`
and records `status=COMPLETE` and `publication=NO_UPLOAD`.

- OWSM authenticated all 1,172 F32 tensors. Its structural manifest SHA-256 is
  `82de20eea3cf3a247624c76cd8e108e562addda0c8582577515cf88abb3053d9`
  and its payload-manifest file SHA-256 is
  `04515dbd3dc7b0c65b6d59ae7e038564b45cf07a36573a12c60a7147b3941cdf`.
  The payload contract has canonical manifest digest
  `f695e97891b9351a2a9e91ac33a631119b1973cdd7857c1bacc9c2b27dfb5f6b`.
  Exactly one source tensor, `frontend.logmel.melmat`, is non-contiguous; its
  logical F32 values were canonicalized to copied little-endian C-order bytes.
  The result deliberately remains `BLOCKED_WRITER_CONTRACT`. No GGUF was
  written and the native frontend/encoder/decoder/search/parity work is open.
- HTDemucs' five fixed ensemble members all report `SAFE_LOADED`; tensor counts
  are 525 for `5c90dfd2` and 533 for each of `f7e0c4bc`, `d12395a8`,
  `92cfc3b6` and `04573f0d`. The exact safe-global allowlist is `BOUND` to
  `demucs/htdemucs.py` git blob `5d2eaaa1eb2620a5d2147eb86361e9964fb94528`.
  The recovered manifest SHA-256 is
  `365179b0127f2ae579b767b4b30e2ef225eb88037068bfef6a20fa1305b3533c`.
  Full-weight digest review, license/provenance review and the native runtime
  remain blockers; safe loading alone does not promote the public row.
- SGMSE safely loaded all 647 finite checkpoint tensors with 65,590,822
  parameters. Its inspection manifest SHA-256 is
  `335bb8c3213af0566bd4fc9ea076dc139b51434fd00f352623925b3e0cf550ba`.
  The independent pinned source run completed six F32 planes of shape
  `[1, 1, 256, 64]`; its reference-manifest file SHA-256 is
  `e0b0e061c144161a935cdbb5864e1c2db9504b7845c2c4bf620767b90e3bf5b9`
  and status is `REFERENCE_COMPLETE_NO_UPLOAD`. Construction evidence binds
  438 named modules, all 77 `dnn.all_modules` entries and all 647 state-dict
  rows with canonical SHA-256
  `695b9ef5b24685fbe41c73fa1b0041e3622cf5315cae7ec03aa799d5f468246e`.
  Exact native NCSN++ tensor-role mapping, native score production, CPU parity
  and Apple CPU/Metal parity remain open.

Only logs and JSON evidence (7.3 MB) were recovered under
`/private/tmp/vokra-model-contract-evidence-8b63dea7/`; no checkpoint, GGUF or
F32 fixture payload was copied to the maintainer Mac. Project dependencies
remain managed through `uv add` and locked execution uses `uv run --frozen` or
`uv sync --frozen`. The `uv pip --system` use in VAST `provision.sh` is limited
to bootstrapping a disposable stock image and does not mutate project
dependency metadata. Instance `49511760` was destroyed after evidence recovery
and a fresh inventory confirmed it absent. Stopped packet instances `49168183`
and `49261078` remain intentionally retained for direct Scaleway transfer;
unrelated running instance `49466383` was not touched. This management-only
record does not alter the code tested at `8b63dea7`.

## Execution order

### Wave A: close the three prepared Apple-hardware evidence gaps

Steps 1-4 below are complete at the recorded source and VAST checkpoints.  The
remaining action is step 5 on authenticated Scaleway Apple Silicon.

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
