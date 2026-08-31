# Mac CPU / Metal full-coverage execution ledger (2026-08-28)

> **Active execution plan (2026-08-31):** the cloud boundary, blocker-class
> ordering and final proof for continuing this ledger are fixed in
> `docs/handoff/mac-cpu-metal-completion-plan-2026-08-30.md`.  Scaleway is a
> hardware-verification stage, not a substitute for the remaining CPU/binder,
> artifact, license or Metal source work.

## Scope and branch

- Working branch: `feat/mac-cpu-metal-full-coverage-2026-08-28`
- Base: `41ce9ffdd4b0959497f55afa5016822f77a8a7b6`
- Reference merge: PR #74, `e3b12c450318a884961a9fa430b5ec69fc67b545`
- Live inventory command:

  ```text
  uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py
  ```

## Authoritative current snapshot (2026-08-31)

The authoritative runtime implementation/code snapshot is
`9f69277d8a0d5df574c1ee95563bd1f005de91d0` on
`feat/mac-cpu-metal-full-coverage-2026-08-28`, historically workspace version
`0.2.0`; the
pre-refresh evidence/package checkpoint was
`5cd97d124bc9eb9d2bb7b0367541dcd1492e4d1e`. The active branch is now
workspace `0.3.0`; immediately before this documentation refresh its remote
head was `d8a93bc3acdb8f9648ecb8dd37ef41657fbf425b` in open PR #79, which was
mergeable and non-draft with 109 passing checks, 13 expected skips, and no
failures or pending checks. GitHub `main` remains
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`. The source-level Metal inventory
correction is `8f0d8572d46fe9972bfdd88241efa937e17e63ac`. The repeated live
public-artifact audit at that commit reports 194 repositories, 193
GGUF-bearing repositories and 198 GGUF files: CPU `full=131`, `partial=42`,
`no-runtime-binder=20`, `not-artifact=1`; Metal `full=131`,
`blocked-by-cpu=62`, `not-artifact=1`. There are zero source-level
CPU-complete/Metal-unsupported rows. GigaAM v3 and GigaAM Multilingual now have
complete conservative Metal code routes, but both still require authenticated
Apple CPU/Metal evidence and therefore remain in the prepared Scaleway set.

Five models are ready for authenticated Apple execution: GigaAM v3, GigaAM
Multilingual, OmniASR CTC 1B, ReazonSpeech NeMo v2 and BiCodec. Their immutable
VAST-to-Scaleway inputs are:

- Wave A at exact code `bc9d1db2bbf230f09ce4f3f68003a1c11f80e0e1`:
  `/root/scratchpad/apple-transfer-bc9d1db2`, 4.9 GB, 30 regular files, no
  symlinks; manifest SHA-256
  `c96eee3c61ec85b589a488deff21668097ed4e94f96b4654b990706098f6f606`.
- ReazonSpeech at exact code
  `a59c48c8da103ac14fe837cd2e0252b5266ac093`:
  `/root/scratchpad/apple-transfer-reazon-a59c48c8`, 11 regular files;
  manifest SHA-256
  `48874cf71497e347019c156f49409d74428734e840cc0302d8626ae5780679ed`.
- BiCodec from evidence checkpoint `5cd97d124bc9eb9d2bb7b0367541dcd1492e4d1e`:
  `/root/scratchpad/apple-transfer-bicodec-5cd97d12`, 600 MB, 12 regular
  files, no symlinks; manifest SHA-256
  `0a80edb51e88d17ce8f243ee58523551baf7d9fc5a848a17dc9c3fdecaf8d18f`.
  Its GGUF SHA-256 is
  `a77004b9a85aa1619abb9413de2d7158d6603d8097f1eeebb83a4bb8bd26637c`
  and reference-manifest SHA-256 is
  `8d159e0e8b19cc7ad88a925f072ae56cf870a180e3e7b2acecb82023b103c696`.

VAST instances `49168183` (500 GB retained storage) and `49261078` (200 GB)
both report `cur_state=stopped`, `intended_status=stopped` and
`actual_status=exited`. They consume no compute, but storage billing continues.
Resume them only for direct transfer of the named packets; after Scaleway
checks the manifests and the required small evidence is preserved, destroy
both instances. The older `/root/scratchpad/apple-transfer-568dc192` packet is
historical and superseded by the exact Wave A packet above.

The Scaleway stage has been reached, but no Scaleway instance, SSH access or
Apple run exists yet. Use an official Apple Silicon M4-M host with 32 GiB RAM,
1.02 TB storage and macOS/Scaleway Dev OS with Xcode; M4 Pro XL with 64 GiB is
an optional higher-memory alternative. The 32 GiB minimum is required by the
OmniASR packet and also covers the recorded ReazonSpeech/BiCodec needs. Do not
use Asahi Linux or FileVault. Provisioning references are the official
[Apple silicon datasheet](https://www.scaleway.com/en/docs/apple-silicon/reference-content/apple-silicon-datasheet/),
[Scaleway Dev OS guide](https://www.scaleway.com/en/docs/apple-silicon/reference-content/scaleway-dev-os/)
and [SSH guide](https://www.scaleway.com/en/docs/apple-silicon/how-to/connect-to-mac-mini-ssh/).
Only the resulting SSH command is needed; no private key or API token should be
shared. No Hugging Face upload has occurred or is authorized.

The 2026-08-28 through 2026-08-30 inventory and preparation notes below are
historical evidence. This 2026-08-31 snapshot supersedes their current-state
wording; Apple/Metal and the remaining 62 CPU-blocked rows remain open.

The 2026-08-28 live audit reports 194 public repositories, 193 GGUF-bearing
repositories and 198 GGUF files. The live-artifact reachability split is CPU
`full=128`, `partial=45`, `no-runtime-binder=20`, `not-artifact=1`; Metal is
`full=128`, `blocked-by-cpu=65`, `not-artifact=1`. There is no public model
whose released artifact is CPU-complete but Metal-unsupported.

This branch retains all 65 blocked repositories as the implementation target.
It does not treat a converter, header probe, synthesized-weight forward,
device-less cross-build, or missing fixture as model completion.

## Completion evidence required per repository

A GGUF repository moves to complete only when all applicable columns have
evidence:

1. The exact public artifact, or an explicitly staged replacement, passes a
   strict manifest, metadata, provenance and license bind.
2. A real native Mac CPU route reaches the model's final declared output.
3. An independent upstream implementation produces the reference tensors or
   output. Self-authored mirror implementations are not numerical references.
4. The CPU result passes pre-registered numerical/output gates on VAST. A
   missing environment variable or artifact is a skip, not a pass.
5. The complete learned-op set is preflighted through one Metal backend with
   no silent CPU fallback.
6. Apple Silicon compares Metal against the independently validated CPU route;
   generated token/code sequences use exact equality where applicable.
7. The Hub revision users download is the revision that passed, or the ledger
   explicitly records that replacement remains unauthorized.

All checkpoint download, conversion and real-weight validation runs on VAST.
Workspace-wide Cargo and every `-p vokra-models` Cargo command also run on
VAST. Apple evidence uses a disposable Apple Silicon worker; scripts that
require 32/64 GB must not be weakened to run on the 16-GB maintainer Mac.

Hugging Face upload is a separate irreversible permission. This branch may
convert and perform a no-upload dry-run, but must not add `--push` without
explicit authorization for the exact target repository and artifact.

## Wave 0: execute the real-weight workers already staged by PR #74

These routes already have model-specific non-publishing workers. Run their
self-tests locally, then execute the real conversion/reference/CPU gates on a
clean VAST checkout. Preserve only small evidence; never pull the GGUFs back
to the maintainer Mac.

- [ ] Qwen3-ASR 0.6B and 1.7B — `run-qwen3-asr-validation.sh`
- [ ] Canary-1B-Flash — `run-canary-1b-flash-validation.sh`
- [ ] Canary-1B-v2 — `run-canary-1b-v2-validation.sh`
- [x] ReazonSpeech-NeMo-v2 — `run-reazonspeech-nemo-v2-validation.sh`
- [ ] SpeechT5-TTS — `run-speecht5-tts-validation.sh`
- [ ] Bark Small and Full — `run-bark-validation.sh`
- [ ] Parler-TTS Mini English and Multilingual — `run-parler-tts-validation.sh`
- [ ] MOSS Audio Tokenizer v2 — `run-moss-audio-tokenizer-v2-validation.sh`
- [ ] Ultravox audio + Llama companion — `run-ultravox-validation.sh`
- [ ] MOSS-Audio 4B and 8B — `run-moss-audio-validation.sh`
- [ ] MusicGen Medium/Large companion composition —
      `run-musicgen-companion-validation.sh`
- [ ] NeuTTS Air + NeuCodec composition — `run-neutts-air-validation.sh`

The matching remote Apple workers now exist for all 12 Wave-0 families,
including both Canary releases, ReazonSpeech, Bark, Parler-TTS and MOSS Audio
Tokenizer v2. Their offline self-tests pass; none has received a real-hardware
verdict on this branch yet.

### 2026-08-29 pre-Scaleway execution evidence

The no-upload VAST run on an x86_64 host with 503 GiB RAM completed both
Canary release workers. Only the small evidence trees and logs were retained;
the `.nemo`, prepared safetensors and generated GGUF files were not copied to
the maintainer Mac.

- Canary-1B-Flash passed at commit
  `9235b960f896799e41e711b4f86a11ba87134bd5`. The authenticated NeMo input
  SHA-256 is
  `3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324`,
  the generated GGUF SHA-256 is
  `fbcda72d79ae9889dd42ebfa05c92a20f930b92beaca86e446e12ff000ca80c7`,
  and both the English ASR and English-to-German AST exact-token gates passed.
- Canary-1B-v2 passed at commit
  `e3cd475b37f4eb1a80b1e514bec2b11d99e3163f`. The authenticated NeMo input
  SHA-256 is
  `ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094`,
  the generated GGUF SHA-256 is
  `3fe5701def703485d4eaf8eae38f81f8d80d6ac47f94d92fd435b6e733bdb2d9`,
  and both the English ASR and English-to-German AST exact-token gates passed.
  The latter confirms that the official Canary2 empty-context prompt begins
  with aggregate-vocabulary token `16053`; the public CLI then emitted the
  complete expected English and German sentences.
- The Canary-1B-v2 worker also completed `cargo test --locked --workspace`,
  strict workspace Clippy, `cargo deny check licenses advisories bans` and
  `cargo audit` at the same commit.
- The final VAST worker contract matrix at `e3cd475b...` reports 95 total,
  94 passing, one expected CosyVoice3 inspection block for the deliberately
  absent dedicated lock, and zero failures. The portable Apple-worker matrix
  reports 41 of 41 passing at the same commit. These are contract/self-test
  results, not Apple-hardware verdicts.
- A fresh live-artifact audit after those runs remains 194 public repositories,
  193 GGUF-bearing repositories and 198 GGUF files. CPU remains
  `full=128`, `partial=45`, `no-runtime-binder=20`, `not-artifact=1`; Metal
  remains `full=128`, `blocked-by-cpu=65`, `not-artifact=1`. The eight focused
  audit unit tests pass. No Hugging Face upload was performed, so the live
  public-artifact counts are expected to remain unchanged.

### 2026-08-30 pre-Scaleway closure

The remaining feasible Linux-side collection was completed on VAST at code
commit `0ba68abb331275dbf2b42919ac644b6734632bdd`. This section supersedes the
pre-run wording in the preparation notes below. No model row is promoted by an
inspection-only result: every unsupported runtime remains fail-closed, and no
CPU or Metal verdict is inferred from an authenticated file inventory.

- The final authenticated, no-upload inspection wave includes Canary-Qwen
  2.5B, FireRedASR-AED-L, Kyutai STT 2.6B EN, GigaAM v3 and GigaAM
  Multilingual. The broader completed inspection set also includes the
  no-binder or composite Qwen2-Audio, ACE-Step 1.5 and VibeVoice-ASR routes,
  together with the earlier inspection families recorded in this ledger.
  Successful collection is recorded as `AUTHENTICATED_EVIDENCE_COMPLETE` plus
  `INSPECTION_ONLY`; native binders, parity and publication remain blocked.
- Canary-Qwen's exact bounded safetensors inventory contains 1,718 tensors:
  1,686 BF16 and 32 I64 tensors, with 2,559,447,600 parameters. The corrected
  contract binds the actual Qwen config identity, input length, leaderboard,
  tensor dtypes and generation EOS instead of accepting nearby metadata.
- FireRed's 4.67-GB checkpoint is admitted only through
  `torch.load(..., weights_only=True)`. The sole additional safe global is the
  checkpoint's observed `argparse.Namespace`; no broad or unsafe pickle
  fallback was added.
- GigaAM v3 binds the live 1,024-piece tokenizer and therefore the exact
  1,025-class RNNT head. GigaAM Multilingual separately retains its 71-class
  CTC contract. Kyutai STT now binds the authenticated live tokenizer identity
  `tokenizer_en_audio_4000.model`; no silent filename alias is used.
- The final VAST gates at `0ba68abb...` pass `cargo test --locked --workspace`,
  strict workspace Clippy, `cargo deny check licenses advisories bans` and
  `cargo audit`. The VAST worker matrix reports 95 total, 94 passing, one
  expected CosyVoice3 block for the absent dedicated lock, and zero failures.
  The portable Apple-worker contract matrix reports 41 of 41 passing. These
  remain self-tests and portable contracts, not Apple-hardware verdicts.
- All 37 no-argument check scripts applicable to this checkout pass in one
  final run. Four scripts are explicitly non-applicable without their required
  external input or generated distribution tree:
  `check-no-market-claims.sh`, `check-model-size.sh`,
  `check-cpu-vulkan-only-no-nvidia.sh` and
  `check-godot-package-no-nvidia.sh`.
- The repeated live audit remains unchanged at 194 public repositories, 193
  GGUF-bearing repositories and 198 GGUF files. CPU is `full=128`,
  `partial=45`, `no-runtime-binder=20`, `not-artifact=1`; Metal is `full=128`,
  `blocked-by-cpu=65`, `not-artifact=1`. Thus all presently unblocked,
  in-scope work feasible before an Apple worker is collected, but the 65
  blocked repositories are not closed. Their remaining gates require real
  Apple hardware, native binder/parity implementation, exact external inputs
  or reference packets, owner license and dependency approvals, or some
  combination of those requirements.

The small evidence trees and logs were copied to
`/private/tmp/vokra-evidence-e3cd475b`; no checkpoint or generated model
artifact was retained locally. No Hugging Face upload or Scaleway execution
occurred in this closure.

### 2026-08-30 authenticated CPU lock before Scaleway

This later lock supersedes the pending GigaAM/OmniASR wording below without
promoting any unsupported Metal route. The code-bearing VAST checkout was
`d06ab75f560d194f44dc0449c1cc50e9d917af80`; the following documentation-only
ABI record advanced the checkout to `568dc192d5fc20b43441f861235c88b2b7af84cf`.
No Hugging Face upload occurred.

- GigaAM v3 completed conversion and independent official CPU parity. The
  prepared safetensors SHA-256 is
  `cee04765f031d6ee5088849ecb0e5c1db4e58ca28a345ce4d049015cd683a64e`,
  the GGUF SHA-256 is
  `287c3657d0ebc41637b6ce7535af2eae9fbb9a8d4a2faa0fb848540b536da1b8`,
  and the reference-manifest SHA-256 is
  `9e20afc5d9155a9b4ed38169f4a54c5cfd0ee2e7beefce628ddfb5e6f0ea9d43`.
  The CPU gate recorded maximum absolute errors of `1.335144043e-4` for
  log-mel, `4.753470421e-6` for the encoded trace and `1.296997070e-4`
  for RNNT logits, with exact decision/token output. Metal remains
  `OPEN_UNSUPPORTED`; no CPU fallback is treated as a Metal verdict.
- GigaAM Multilingual completed conversion and independent official CPU
  parity. The prepared safetensors SHA-256 is
  `1c4aa78524c87edce9ad4fab7e8fdfeebdb2dc7c546c826b37cd59f8d2541995`,
  the GGUF SHA-256 is
  `e80019e784d345e16e28a2b4441ad88c6a14209f315426a2dd7fc5a2900a10cf`,
  and the reference-manifest SHA-256 is
  `f71b3d53a662ebee9604dcd457c0d4c02c9251092f0cbedf860c80f2e255e38d`.
  The CPU gate recorded maximum absolute errors of `1.007579267e-4` for
  the encoded trace and `4.072189331e-4` for logits, with exact token IDs.
  Its Apple worker remains an explicit `OPEN_UNSUPPORTED` contract until the
  learned-op Metal route is wired; it cannot emit a false PASS.
- OmniASR CTC 1B completed the authenticated 807-tensor conversion and real
  CPU parity against the pinned official implementation, using the first
  second of the committed public-domain JFK fixture. The prepared checkpoint
  SHA-256 is
  `cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5`,
  the GGUF SHA-256 is
  `abf6f0ee8c028e7c79955f68d841d4445fa9664a2e87ff26c80a37a3b4a3561e`,
  and the reference-manifest SHA-256 is
  `7a37e36e56c90370390c741bac421e211834116f14c4d0305a84f8b87552dd1b`.
  The 49-frame CPU gate recorded maximum absolute errors of
  `2.918243408e-4` for the frontend, `1.520276070e-3` for the encoder and
  `1.450002193e-3` for logits; the five emitted token IDs matched exactly.
  Apple CPU repetition and Metal execution remain pending the authenticated
  Apple Silicon worker.
- At the code-bearing commit, `cargo test --locked --workspace`, strict
  workspace Clippy over all targets/features, `cargo deny check licenses
  advisories bans` and `cargo audit` all pass on VAST. At the ABI-record
  commit, all 37 applicable no-argument check scripts pass; the same four
  external-input/distribution checks listed above remain non-applicable. The
  expanded VAST worker matrix is 100 total: 99 pass and only the expected
  CosyVoice3 dedicated-lock refusal exits 2. The expanded portable Apple
  contract matrix is 43 of 43 passing. These portable tests are not
  Apple-hardware verdicts.
- The remote-only Apple transfer packet is
  `/root/scratchpad/apple-transfer-568dc192` on VAST: 4.9 GB across 33 files.
  Every file verifies against the relative-path manifest
  `/root/scratchpad/apple-transfer-568dc192.SHA256SUMS`, whose SHA-256 is
  `fa36488ad7cf93e62b3d4f3899b122b989ade98c4415bbbd7ac5bc344659304d`.
  The packet has not been pulled to the maintainer Mac. The current VAST
  instance is retained intentionally only for direct transfer to the approved
  Scaleway Apple worker and must be destroyed after that transfer/evidence
  handoff.

### 2026-08-30 ReazonSpeech ALSD CPU lock before Scaleway

ReazonSpeech-NeMo-v2 completed its no-upload VAST conversion and independent
official-NeMo CPU parity at exact code commit
`a59c48c8da103ac14fe837cd2e0252b5266ac093`. The released checkpoint's
alignment-length synchronous decoding contract is now bound fail-closed rather
than approximated by greedy decoding: beam size 4, maximum target length 1.0,
score normalization enabled, default nested search, temperature 1.0 and
return-best enabled. Legacy public GGUFs that lack this exact metadata are
rejected before tensor loading.

- The authenticated NeMo archive SHA-256 is
  `d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae`.
  The generated 2,477,292,896-byte GGUF SHA-256 is
  `31b93b620b9fdcaee13cda89f16bc35d4e191f0e161d735bc77f87f8839bd12f`.
- The official reference manifest SHA-256 is
  `0263d2ad5e9ff218be8c8ef3560edf22998233ad76d7b7919dc92091410f18d2`;
  its encoder payload is
  `e06a1dfa3fee5b9426c5a5e4625ab7ad252f7c3b844afb6e86d9d81c8c945b40`
  and its token payload is
  `54cae203650bacae778fe0a9bfab0e014f0d597471c549cf7224a457db0c1be0`.
  The 138-frame encoder comparison passed with maximum absolute error
  `7.414817810e-4` and mean absolute error `2.220646638e-5`. Native ALSD emitted
  exact token IDs `[2, 96, 25, 49, 214, 1]`, exact text `カントリー。`, and the
  public CLI emitted the same transcription.
- The focused release parity passed in 63.13 seconds. The same exact checkout
  then passed `cargo test --locked --workspace`, `cargo clippy --locked
  --workspace --all-targets -- -D warnings`, `cargo deny check licenses
  advisories bans` and `cargo audit`. The validation-summary SHA-256 is
  `fa6ed1ddc940fefd13a8ae86265ad9010f8c64ed421491ed52c3f9e67a85d03b`.
- The Scaleway input packet is retained only on VAST at
  `/root/scratchpad/apple-transfer-reazon-a59c48c8`: 11 regular files,
  including the GGUF, reference, approval evidence and git bundle. Every file
  verifies against `apple-transfer-reazon-a59c48c8.SHA256SUMS`, whose SHA-256
  is `48874cf71497e347019c156f49409d74428734e840cc0302d8626ae5780679ed`.
  The included bundle SHA-256 is
  `fe881c310c3f1d6970f5cfaf0ae3d9064cb45e25ac4b31b2be9e4ef9a8668261`.

VAST instance `49168183` is stopped after packet verification; retained
storage continues to incur a charge. ReazonSpeech still needs authenticated
Apple CPU/Metal execution on Scaleway. Its live public row also remains
partial until a separately authorized replacement is published through the
gated workflow. No model artifact was copied to or executed on the maintainer
Mac, and no Hugging Face upload occurred.

### 2026-08-30 model-free dependency evidence lock

The Qwen3-ASR, Bark and Parler-TTS dependency audits then ran on temporary VAST
instance `49232927` without downloading or executing a model and without Cargo.
The implementation commits are `4e34f67c` (Parler exact locked-sdist evidence),
`b09c92a6` (Bark), `152a4ccc` (Qwen), and `afe0b775` (the Qwen virtual-project
row correction discovered by the first real audit).  Every local manager
self-test and pre-commit static gate passed.  The temporary VAST checkout was
clean and matched the named commit supplied by verified incremental git
bundles; no branch was pushed.

- Qwen accounts for all 95 lock rows: 91 exact Linux installed rows and four
  inactive rows (the virtual root, Darwin Torch alternative, `colorama` and
  `tzdata`).  The closure has no missing or unexpected distribution.  Exact
  locked sdists yielded bounded publisher bytes for `cython==3.3.0` and
  `tokenizers==0.22.2`.  `dynet38==2.2` has no locked sdist; the exact
  `gradio-client==2.5.0`, `qwen-omni-utils==0.0.9`, `soynlp==0.0.493` and
  `tqdm==4.70.0` sdists contain no accepted LICENSE/COPYING/NOTICE/COPYRIGHT
  candidate.  Both fixed Qwen model `LICENSE` paths return 404.
- Bark accounts for 34 exact Linux installed rows plus its virtual project row.
  Exact sdists yielded publisher bytes for `safetensors==0.4.5` and
  `tokenizers==0.22.2`; the exact `tqdm==4.70.0` sdist has no accepted license
  candidate.  Both fixed Bark model `LICENSE` paths return 404, and the existing
  identity contract still has no pinned Bark source-license revision.
- Parler-TTS accounts for 26 exact Linux installed rows plus its virtual row.
  The `tokenizers==0.20.3` exact sdist yielded publisher bytes while the same
  `tqdm==4.70.0` sdist remained blocked.  The fixed Parler source LICENSE was
  acquired as factual, unclassified primary-source bytes; both model LICENSE
  paths and the DAC LICENSE path return 404.

The small recovered reports are
`qwen3-asr-v4.json` SHA-256
`052a11f747b6840b6179f3f85044a9585e151a3349d349bbffa96b63cc8ce07f`,
`bark-v4.json` SHA-256
`3e589a4d74cce49a12674840a745a7f8b911ccfbcaf54638ebb50593feace517`
and `parler-tts-v3.json` SHA-256
`ef2c7631d18d644750d1d485ef81f58368cd38f8c1ecb6a011d27ee144224f03`.
These reports preserve blockers rather than inferring a license class or owner
approval.  Instance `49232927` was destroyed after recovery; stopped retained
instance `49168183` and its Scaleway packets were not changed.  Consequently
all Qwen, Bark and Parler CPU/Metal rows remain open pending the remaining
license/operator decisions, real model validation and Apple execution.

### 2026-08-31 final VAST/source batch before Scaleway

The final code-bearing commit for this batch is
`9f69277d8a0d5df574c1ee95563bd1f005de91d0`. The clean VAST checkout on
instance `49261078` reached that exact commit only through verified incremental
git bundles. No branch was pushed and no Hugging Face upload occurred.

- BiCodec's official SparkAudio worker now has an exact Python 3.12 reference
  lock for `einx==0.4.3` and transitive `frozendict==2.4.7`; both are
  reference-only MIT dependencies and never enter the Rust runtime or an
  upload. The complete no-upload worker passed at code commit `1e4e27e8`.
  It reproduced GGUF SHA-256
  `a77004b9a85aa1619abb9413de2d7158d6603d8097f1eeebb83a4bb8bd26637c`
  and reference-manifest SHA-256
  `8d159e0e8b19cc7ad88a925f072ae56cf870a180e3e7b2acecb82023b103c696`.
  The CPU comparison passed semantic latent (`max_abs=1.907348633e-6`,
  `rmse=3.059676605e-7`), d-vector (`1.847743988e-6`, `2.558271893e-7`),
  prenet (`7.539987564e-6`, `1.297758877e-6`) and waveform
  (`6.183981895e-7`, `1.134617555e-7`) with the pre-registered bounds and one
  exact `backend=cpu` PASS sentinel. The later `9f69277d` change is only the
  Clippy-equivalent removal of a needless `return`; it changes no selector or
  numerical behavior. Real Apple CPU/Metal execution remains pending.
- The model-free XY-Tokenizer dependency collector accounts for all 57 active
  lock rows. It collected exact artifact, primary-license and bounded native
  evidence for 51 rows. The six retained fail-closed rows are SciPy and SymPy
  (compound license documents that cannot be reduced to one SPDX class),
  setuptools (vendored LGPLv3 metadata), soxr (LGPL-2.1-or-later), and
  tokenizers/tqdm (no accepted distribution-owned primary license bytes).
  The collection-report SHA-256 is
  `604e9cc74a5814f97bcd2be106e1f620f5f4d2d45052ce3c78fb485583f17210`;
  the partial evidence SHA-256 is
  `3e2471835be2b5cb767f3181050c98ff82dc12e039c9b4257af684d713306ffc`.
  No model, checkpoint or Torch source route was imported by this audit.
- HT-Demucs Multi remains blocked before model execution because its exact
  upstream `torchaudio>=0.8,<2.1` constraint has no Python 3.12-compatible
  release. More RAM, VAST or Scaleway cannot resolve that upstream dependency
  contradiction; the worker remains `BLOCKED_UNSATISFIABLE_PY312_TORCHAUDIO`
  and `NO_UPLOAD`.
- At exact final commit `9f69277d`, VAST `cargo test --locked --workspace`
  has 310 passing result groups and zero failed result groups (log SHA-256
  `c6a9c5b1604ed53c02902bd311062f7a4646f7f9a455993489f625c96769b139`).
  Strict workspace Clippy has zero errors (SHA-256
  `373ce57e806cb33ec0a7b16e49174ffcf0b274b38cfbb8d02bb7813b976aa33c`),
  `cargo deny check licenses advisories bans` passes (SHA-256
  `43ba882d8949aa5a6145e86a1bdf66d602057591b4d462390aa8c4519c0e9666`),
  and `cargo audit` passes (SHA-256
  `82e60f15564fdf549048e5f14a0d6a8e97a09b05fc875a389efe7da180d60c36`).
- The repeated live public-artifact inventory remains 194 repositories, 193
  GGUF-bearing repositories and 198 GGUF files. CPU is `full=131`,
  `partial=42`, `no-runtime-binder=20`, `not-artifact=1`; Metal is `full=129`,
  `blocked-by-cpu=62`, `cpu-only=2`, `not-artifact=1`. The zero-CPU-only
  invariant therefore correctly refuses the two GigaAM rows. The fail-closed
  inventory log SHA-256 is
  `276ea4e27b5feb97199e42327c277a61c8d7db708baf6c6a78ab15d65f32c619`;
  the explicit summary SHA-256 is
  `99272fead6ada58ef0e11628bc411ef71b0c011f64fbf341945d50e187f40370`.

  This bullet preserves the exact fail-closed audit result at the named
  `9f69277d` VAST checkpoint. The later source-level inventory correction at
  `8f0d8572` classifies both GigaAM routes as Metal-complete and produces the
  current `full=131`, `blocked-by-cpu=62`, `not-artifact=1` split with zero
  CPU-only rows. It does not supply an Apple-hardware verdict.

This closes the current unblocked Linux/VAST evidence batch, not all 62
repositories blocked by incomplete CPU artifacts or binders. Scaleway can
close only Apple-ready CPU/Metal packets. Missing native binders, unresolved
dependency or weight licenses, gated companion access, owner sign-off and
publication authorization remain separate blockers and are not solved by
Apple hardware. The BiCodec Apple packet and final branch bundle are verified:
`/root/scratchpad/apple-transfer-bicodec-5cd97d12` contains 12 regular files
with no symlinks and manifest SHA-256
`0a80edb51e88d17ce8f243ee58523551baf7d9fc5a848a17dc9c3fdecaf8d18f`.
Its branch bundle was created as
`/private/tmp/vokra-mac-final-5cd97d12.bundle`, has SHA-256
`a0b3a892cd29439a65d57e28c8c976d5914e53a0340272d682b7b28606341e71`
and requires GitHub `main` base
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`. VAST `49261078` and `49168183`
are both stopped/exited and retained only for direct Scaleway transfer; their
storage billing continues until destruction.

### Branch preparation status

The following source work is present and locally reviewed on this branch. None
of these bullets closes a model row until the named real run succeeds:

- All 12 Wave-0 workers have a passing offline self-test. Canary-1B-Flash,
  Canary-1B-v2 and ReazonSpeech-NeMo-v2 gained the previously missing hermetic
  contract tests. Sol repeated all 12 self-tests on 2026-08-29; all passed.
  `bash -n` also passed for the complete set. The three quoted EXIT traps in
  Bark, Parler-TTS and MOSS Audio Tokenizer v2 now use explicit helper
  functions; manager-repeated self-tests, `bash -n` and ShellCheck all pass
  without the former `SC2154` warnings while retaining the original failure
  summary and exit-code contract.
- `vastai-safe.sh` redacts credential-valued URL query parameters and
  credential fields in JSON or `key=value` output, including upper- and
  lower-case instance/Jupyter/container keys, plus single-quoted Python-dict
  and unquoted dict-field diagnostics, while preserving the wrapped CLI exit
  status. Its hermetic regression and live lifecycle probes report sensitive
  values as `[REDACTED]`. The lifecycle runbook and `run-one.sh` route local
  Vast CLI commands through it.
- `provision.sh` no longer resolves HF dependencies by running `uv add` in the
  checkout. `run-one.sh` owns its pinned transient HF environment instead. A
  hermetic regression proves that provision self-test leaves the checkout file
  set and dependency-file hashes unchanged.
- The 2026-08-29 manager-wide cheap gates currently report
  `git diff --check` green and the eight focused live-audit unit tests green.
  Luna applied repository rustfmt mechanically across the staged Rust work;
  Sol repeated `cargo fmt --all -- --check` on 2026-08-29 and it is now green.
- All 38 staged Apple Silicon workers pass `bash -n` and ShellCheck. An
  offline manager sweep originally passed 36. The CosyVoice3 worker still
  deliberately returns blocked because its dedicated lock is absent. The Dia
  worker initially let `uv run --frozen` attempt a local dedicated-environment
  sync before license approval; DNS then failed while fetching
  `typing-extensions`. The newly created partial Dia `.venv` was moved intact
  to `/private/tmp/vokra-dia-selftest-partial-venv`. Luna changed Dia's
  self-test to a no-project, no-sync route and kept real Apple manifest
  validation in the general tools environment. Sol repeated the corrected Dia
  reference, validator, VAST-worker and Apple self-tests without creating a
  dedicated `.venv`; all passed. The current Apple contract count is therefore
  37 passing and one intentionally blocked. No Apple real-weight execution
  occurred.
- OmniASR has an authenticated VAST manifest for the pinned 1B prepared/source
  identity and exact 807-F32 tensor contract; the native waveform frontend,
  grouped positional convolution, projected 48-layer pre-norm Transformer,
  and CTC-token binder are staged. Real VAST numerical parity and the Apple
  CPU/Metal verdict remain pending, and the external tokenizer is outside the
  GGUF (the runtime boundary produces token IDs). No public artifact
  replacement or publish is claimed here.
- No-upload VAST validation workers are staged for NSNet2, ECAPA-TDNN,
  WeSpeaker and SpeechBrain VoxLingua107 Lang-ID. They pin source revisions,
  authenticate available checkpoint/public-artifact digests, generate an
  independent upstream oracle, convert on VAST and run named nonzero CPU tests.
- WeSpeaker additionally has a safe exact-219-tensor preparer and a dedicated
  official-combined CPU parity test. The test has not yet run against the real
  converted artifact.
- Device-gated Apple workers are staged for NSNet2, the corrected ECAPA and
  official-combined WeSpeaker pair, and VoxLingua107. The Lang-ID worker
  deliberately reports `MEASURED_NOT_GATED` until CPU/Metal bounds are reviewed.
- MOSS Audio Tokenizer Nano now has a fixed-revision no-upload replacement
  validation worker and a real-file Apple CPU/Metal worker. The replacement
  gate requires the corrected Nano name, variant and provenance and refuses
  the historical metadata-repair path. Numerical bounds remain deliberately
  unregistered until the first independent real run is reviewed.
- RMVPE now has a fixed-upstream, fixed-public-revision no-upload validation
  worker and real-file Apple CPU/Metal worker. The official Python oracle is
  constructed under a temporary `torch.load(..., weights_only=True)` guard;
  the guard is restored in `finally` and has no unsafe fallback. The historical
  public bytes remain rejected because the upstream weight repository declares
  no license, so the corrected dry-run is stamped `unknown` and cannot be
  uploaded.
- YuE xcodec-mini now has an immutable-public-artifact VAST measurement worker
  and a 32-GB-class Apple CPU/Metal worker for its released 12-codebook
  token-to-44.1-kHz path. The oracle imports the fixed upstream residual-VQ
  implementation plus the pinned Vocos wheel, requires
  `torch.load(..., weights_only=True)`, and rejects non-finite or zero-norm
  measurements. It remains `MEASURED_NOT_GATED`; PCM encode is still an
  explicit unsupported boundary requiring a separate clean-room/license audit
  of the acoustic/HuBERT/RepCodec path.
- CLAP HTSAT-fused now rejects every non-empty metadata envelope instead of
  constructing a misleading runtime handle. A fixed-revision VAST inspection
  worker records the official Transformers state-dict manifest, config,
  runtime source hash and independent 512-d audio embedding. Until that
  manifest is produced and reviewed, all CLAP evidence is labeled
  `INSPECTION_ONLY`; the Apple script records only host readiness and emits no
  CPU/Metal parity verdict. The converter deliberately does not stamp an
  unauthenticated fixed revision onto arbitrary safetensors.
- MMS-1B-All conversion and runtime binding are now both explicitly
  `INSPECTION_ONLY`. The converter rejects permissive relabeling and never
  emits an adapter-only GGUF; the pinned VAST worker accepts the official
  language identifier grammar, fetches the matching nested vocabulary, and
  records both the source tensor manifests and the Transformers-composed
  state dict. Native work remains blocked on review of that real VAST
  composition evidence, so the Apple worker emits no CPU/Metal verdict.
- Voice Gender Classifier now has a dedicated MIT-only 202-tensor converter
  and CPU/Metal runtime instead of being mislabeled as the 200-tensor
  SpeechBrain ECAPA model. The frontend, zero-padded convolutions, official
  Res2Net branch order, attentive pooling, embedding and two-class head are
  covered by an independent fixed-source dumper plus no-upload VAST and
  Apple workers. The CLI now routes the dedicated arch to a distinct
  classification task, requires 16-kHz mono WAV input, executes the strict
  selected-backend binder once, and preserves the official `[male, female]`
  probability order on stdout and in optional little-endian f32 output.
  Numerical bounds remain unregistered until those real workers run.
- Qwen3-TTS now has accepted fail-closed staging for all four released main
  variants plus the separately authenticated 12-Hz decoder. Its dedicated
  Python 3.12 lock was reduced from 90 to 49 package entries by loading the
  exact official `QwenLM/Qwen3-TTS` commit through an authenticated shallow
  checkout instead of installing its demo/ONNX/SoX extras. The official
  wrapper imports `librosa`, `soundfile` and `soxr` unconditionally, so those
  packages and their required native transitive closure remain explicit,
  unresolved review rows; `gradio`, `onnxruntime`, `protobuf` and `sox` are
  absent and prohibited by code. The gate hard-codes exact lock/project bytes,
  all source/model/decoder/config/common-asset identities and six component
  rows, rejects placeholder and owner-evidence drift, and exits 2 first at
  `accelerate==1.12.0` without creating work or scratch. VAST emits one
  reusable Apple invocation containing five GGUFs, four reference directories
  and all nine expected hashes. Apple authenticates those inputs before Cargo
  and accepts only one exact named result and singleton full-line sentinels.
  Prompt ids and all 16 generated codebook rows remain exact-equality gates;
  PCM stays `MEASURED_NOT_GATED`. The native greedy helper preserves the
  official `min_new_tokens = 2` default, and the packet verifier has an
  independent SHA-256 known-vector test. Sol repeated the offline lock,
  gate/dumper/VAST/Apple self-tests, Python compilation, Bash syntax,
  ShellCheck, focused whitespace checks and the blocked production proof. No
  source/model fetch, sync, Cargo, VAST, Apple run or upload occurred;
  historical pre-contract public GGUFs therefore remain rejected and the four
  coverage rows remain open.
- Qwen3-ASR now has a dependency-free pre-sync gate for its dedicated Python
  3.12 reference project. The gate binds the exact lock and project bytes,
  canonical version/source/marker/dependency rows, both fixed Qwen model
  revisions, the fixed reference-audio digest, separate model-license rows,
  every version-keyed dependency license/native/bundled review row and an
  approval scope plus external evidence file. A fully approved temporary
  manifest passes its self-test and independent lock/model/license/audio/
  dependency/scope/signer/evidence tampering is rejected. The tracked
  production manifest intentionally remains `PENDING_REVIEW` with null
  conclusions. Normalized empty, pending, owner-review and unresolved
  placeholders, plus missing, extra or duplicate package/model identities,
  all fail closed. The gate is the first substantive VAST operation and exits
  2 on the 0.6B model-license row before host/tooling checks, scratch,
  dependency sync, checkpoint download or Cargo. VAST emits one reusable
  two-variant Apple command containing both GGUF paths/hashes and both
  reference directories/manifest hashes. Apple authenticates all four
  caller-supplied expected hashes before inner reference files or Cargo, then
  requires one exact named result and singleton full-line markers for both
  0.6B and 1.7B. Sol repeated the 95-package offline lock, gate, VAST and Apple
  self-tests, all three reference unit tests, Python compilation, shell syntax,
  ShellCheck, scoped whitespace checks and the production no-scratch proof. No
  model was downloaded or converted, no `vokra-models` Cargo command ran
  locally, and no VAST, Apple or upload operation was performed; both coverage
  rows remain open.
- Ultravox now has a dependency-free pre-sync license/identity gate covering
  the exact lock/project bytes, canonical package graph, all 40 version-keyed
  dependency license/native/bundled rows, the fixed public GGUF, the fixed
  Fixie model/source snapshot and the separately gated Meta companion. The
  Fixie `model.safetensors` payload remains correctly bound to
  `f3a3bf7e9137f3219a0d27ba71668deeee8c60aaf0ea587b48d8f71178763f31`;
  it is not attributed to Meta. The Meta companion payload digest remains
  null and unauthenticated until the gated fixed revision is inspected, so
  the production gate exits 2 before scratch creation, token checks, sync,
  download or build. VAST and remote Apple workers authenticate every
  reference payload and require one exact named result plus one backend
  sentinel; missing and duplicate results/sentinels are rejected. Sol repeated
  the gate, dumper, VAST-worker and Apple-worker self-tests, production exit-2
  check, Python compilation, shell syntax, ShellCheck and scoped whitespace
  checks. The real integration test still skips honestly in an ordinary
  workspace run with no opted-in inputs; that skip cannot become validation
  evidence because the workers require the model-level sentinel. No real
  model, VAST, Apple parity or upload action occurred.
- Conv-TasNet now binds through the default strict compliance policy and
  requires an explicit research opt-in for the unresolved/unknown upstream
  weight license. Its VAST worker reproduces the authenticated 345-tensor
  Asteroid checkpoint and the existing 2026-08-24 CPU bounds without upload.
  The Apple worker executes the real Metal branch and records CPU-relative
  metrics under an explicit `MEASURED_NOT_GATED` marker; no unreviewed Metal
  tolerance is treated as a pass.
- SBV2 now owns one explicit backend selector across the text encoder,
  JA/EN/ZH BERT sidecars, BERT bridge, speaker/style projections, stochastic
  duration predictor, flow inverse and conditioned HiFi-GAN decoder. CPU
  retains the previously measured JA/EN/ZH path; the Apple worker replays the
  same fixture request and seed on Metal and records only
  `MEASURED_NOT_GATED` CPU-relative metrics. The historical public main GGUF
  and production Japanese G2P boundary remain artifact/runtime blockers and
  are not repaired by this source-route wiring.
- BiCodec pins the exact 625 MB Spark-TTS checkpoint, 1,164-byte config and
  840-tensor F32 manifest. The authenticated converter rejects permissive
  license relabels and stamps the CC-BY-NC-SA-4.0 research/share-alike class.
  A decode-only native candidate strictly binds the complete manifest and
  loads the 319 semantic/global-token, Vocos prenet and waveform-generator
  tensors reachable from detokenize; PCM encode is an explicit unsupported
  operation. Its VAST-only official-source worker compares semantic latent,
  d-vector, prenet output and waveform and never uploads. That numerical run
  is still report-only with no selected tolerance or pass claim, and real
  Apple Metal parity remains pending.
- HT-Demucs Multi now rejects the historical unauthenticated 2,132-tensor
  ensemble bag instead of flattening it into a runtime-looking GGUF. Its
  VAST-only inspector pins the archived official source revision, the four
  ordered fine-tuned members, the single six-source member and their identity
  ensemble matrices, records complete safe-loaded tensor manifests, and uses
  `torch.load(..., weights_only=True)` without a pickle fallback. A config,
  filename-hash or safe-load blocker is preserved in evidence and propagated
  as worker exit 2. The Apple script is host-readiness-only; no CPU/Metal
  verdict is claimed before the ensemble evidence and native binder exist.
- SGMSE-VoiceBank now rejects the historical 647-tensor pass-through and every
  arbitrary safetensors candidate. Its VAST-only inspector pins the exact
  SpeechBrain model revision and checkpoint digest, safe-loads only with
  `torch.load(..., weights_only=True)`, and records the candidate EMA tensor
  container without emitting weights. The executable SpeechBrain 1.0.3
  `ScoreModel`/`SGMSEEnhancement` route and the separately licensed upstream
  NCSN++/OUVE/sampler implementation are fixed by commit and file hashes;
  weight, SpeechBrain-code and algorithm-code licenses remain distinct. A
  missing implementation marker, ambiguous tensor container or unsafe-load
  failure is preserved as a blocker and propagated as exit 2. Native
  STFT/NCSN++/OUVE sampling and Apple parity remain `INSPECTION_ONLY` pending
  review of the real VAST evidence.
- VibeVoice-ASR now rejects the former arbitrary one-file pass-through for the
  canonical Microsoft 9B eight-shard release. Its 128-GiB VAST inspector
  verifies complete index/shard coverage with a BF16-safe one-tensor resident
  scope, inventories every processor/tokenizer/config companion and all
  tracked Microsoft source files, and fixes both the VibeVoice source commit
  and the exact Transformers 4.51.3 implementation commit used by its Qwen2
  decoder/generation path. Weight and source-license declarations are derived
  from the fixed snapshots rather than stamped from table constants. Any
  shard, behavioral-marker, license or dependency mismatch writes a BLOCKED
  manifest and exits 2. BitNet/VibeASR.cpp derivatives are not substituted;
  native ASR, diarization/timestamp output and Apple parity remain
  `INSPECTION_ONLY`.
- VieNeu-TTS-v3-Turbo now rejects the stale July topology constants, arbitrary
  safetensors and license relabels instead of presenting the current composite
  release as a native GGUF. Its VAST inspector fixes the live model, official
  source and MOSS Audio Tokenizer Nano revisions; compares Hugging Face server
  trees with the complete local snapshots; records canonical config/token ids,
  every tensor, and each ONNX graph/input/output/opset/initializer without
  executing ONNX. External-data paths, symlinks and byte ranges are validated
  fail-closed. The official v3 Turbo engine and all tracked source files are
  hashed, while weight, source, codec, dependency and preset/voice-cloning
  rights remain separate evidence. Dependency-license review is still an
  explicit blocker, so this wave cannot produce an inspection-complete,
  runtime or Apple parity verdict even after the first VAST collection.
- XY-Tokenizer now pins the canonical OpenMOSS HF revision, exact
  2,137,328,977-byte checkpoint, official config hash and fixed source
  revision. The fixed official source `readme.md` at
  `5df5609c5883e555bd39a2d0b1005ca8f1a8f12e` is authenticated by Git blob
  SHA-1 `cfe231b384040a2162a516c400fbd9282b3317b7` and SHA-256
  `c5e9b83f8382a819063e270489a0f85994628360432fae1054fa2e65ec24d8f7`;
  its `## License 📜` section explicitly declares `XY-Tokenizer is released under
  the Apache 2.0 license.` No full `LICENSE`/`COPYING`/`NOTICE`/`COPYRIGHT`
  file is tracked, and that absence remains an evidence fact rather than a
  source-license-absent claim. The separate fixed HF weight README declares
  `license: apache-2.0`. Native runtime is still unimplemented; any future
  implementation must follow a clean-room boundary and copy no source code.
  Its 128-GiB VAST inspector uses only `torch.load(..., weights_only=True)`,
  safely handles scalar integer/float/BF16 buffers, records per-tensor raw-byte
  hashes and all tracked source files, and prepares evidence without claiming
  a runtime. The authenticated run at
  `/private/tmp/vokra-vast-evidence-509108cb-xy/` pins 1,079 tensors
  (1,071 F32 / 8 bool), prepared safetensors SHA-256
  `743f8f105159dcab57c363b330875d371d1618033cfc8caf372de4696370ebdf`,
  and the fixed source/config/checkpoint identities. The manifest still marks
  `TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER`: tensor names and config axes do not
  authenticate the forward order, frontend/RVQ/Vocos semantics, or an
  independent parity fixture. A VAST-only official-reference adapter now
  calls the pinned upstream `XY_Tokenizer.inference_tokenize` and
  `inference_detokenize` methods and captures frontend, encoder, RVQ, decoder,
  Vocos, and waveform f32 taps; it does not mirror the equations or run locally.
  Its execution is fail-closed behind a separate exact lock and
  version-keyed primary-source dependency/license audit. The current source
  `requirements.txt` is unpinned and the broad parity lock does not provide
  that XY-specific closure review, so the worker stops with
  `DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER` before source/model import.
  The public converter therefore remains an
  unconditional `INSPECTION_ONLY` refusal; a self-asserted sidecar cannot
  authorize a GGUF or native binder.
- ACE-Step 1.5 now rejects its historical pass-through conversion path and
  treats the canonical 10.1-GB composite release as inspection-only. Its
  128-GiB VAST worker fixes the model and source revisions, parses every
  root/component JSON and safetensors index, validates tensor-to-shard
  ownership, and safe-loads PyTorch containers only with
  `weights_only=True`. The corrected server packet now binds the requested
  and resolved HF revision, every regular Git blob, every canonical LFS
  pointer blob and every materialized payload SHA-256 with bounded streaming;
  only the root Hugging Face transport cache is excluded, and it cannot
  supply license evidence. The fixed source must be clean, all tracked files
  regular, and 15 reviewed role files must match their index, HEAD and
  streamed working-tree Git objects. Role-local markers bind the component
  registry, 48-kHz output, silence loading, generation parameters, diffusion
  guidance and 5-Hz LM rather than searching concatenated source text. The
  source `uv.lock` records the complete package/version inventory and hashed
  artifacts, including Transformers 4.57.6 and Diffusers 0.37.1. The
  canonical bundle is the ordered four-component
  `acestep-v15-turbo` / `vae` / `Qwen3-Embedding-0.6B` /
  `acestep-5Hz-lm-1.7B` composition plus exactly one
  `silence_latent.pt`. Manager review accepts this as a fail-closed evidence
  collector, not a model completion: component-local license declarations,
  dependency licenses and dataset/training provenance remain blockers, and
  the corrected worker has not run on VAST. No native DiT/LM/VAE route or
  CPU/Metal parity verdict exists.
- Sortformer Diar 4spk v1 now rejects both conversion and runtime binding
  unconditionally instead of allowing the prior incomplete architecture to
  look executable. The fixed-revision inspector authenticates the complete
  NeMo archive and its real frontend/transformer axes, checks archive paths,
  types, duplicates and embedded NULs without extracting it, and records the
  exact processor timing contract. Source-history review adds a narrower,
  non-provenance reference boundary: the first public Sortformer model/module
  merge is NeMo commit `505acacf6444a67ff9a4020fb03a5e6d59953e05`
  (2024-11-27), and the relevant Sortformer roles did not change before the
  end-of-2024-12-09 NeMo tree at
  `8583201ca8c5bf5d7114ee156d6b8ab684c9b5bf`. The original `.nemo` payload
  was uploaded in HF commit `5bd87d8c7e6fa303c6d9338f85a5e158537627e1`
  on 2024-12-09, whereas `model.safetensors` and the direct HF config were
  added later in `1dd84ea...` on 2025-10-29. NVIDIA's card still points to
  mutable NeMo `main` rather than naming the implementation used to build the
  weights. The 2024 public merge may therefore be pinned as an independently
  reviewed execution source, but it must not be relabeled as authenticated
  weight-build provenance. Manager review now accepts the corrected local
  inspection contract: it binds the fixed eight-file HF tree, canonical LFS
  pointer objects and payload digests, validates the structured model-card
  frontmatter without mistaking nested YAML for the license, and authenticates
  every fixed NeMo role through index, HEAD and streamed working-tree objects.
  The worker has not run on VAST, so real archive/tensor evidence, native
  diarization and Apple parity remain explicit blockers.
- XTTS v2 now rejects the historical multi-file pickle conversion path. Its
  VAST-only inspector fixes the 2.09-GB Coqui release and exact TTS source
  commit/origin, compares the server and local trees, inventories archive
  members and unsafe globals, and permits only `torch.load(...,
  weights_only=True)`. Both normal and unexpected-error manifests remain
  `BLOCKED` / `INSPECTION_ONLY`; CPML, source/dependency review, voice-cloning
  consent and native GPT/DVAE/HiFi-GAN execution remain unresolved.
- Baichuan-Audio-Instruct now rejects the stale 404-source converter and
  records the canonical 21.1-GB five-shard release instead. Its 128-GiB VAST
  inspector safely checks direct shard paths, bounded safetensors headers and
  index-to-physical-shard tensor ownership and keeps conversion disabled.
  The corrected server contract separates non-LFS Git blobs, canonical LFS
  pointer objects and streamed payload SHA-256/size and requires the exact
  repository, requested revision, resolved revision and complete materialized
  file set. Source HEAD/origin/cleanliness is recorded, while the unavailable
  exact role-object table remains explicitly unauthenticated.
  Manager review accepts the worker only as honest blocked staging: Matcha is
  not pinned and remaining custom model/tokenizer/config semantics and
  component/dependency/dataset licenses are explicit blockers. A dedicated
  14-row Python 3.12 inspection lock is now staged for the narrow
  Hugging-Face-Hub acquisition client, with exact lock, package-row and
  package/dependency-marker digests. Its pre-sync gate runs in a no-project
  standard-library environment and remains fail-closed for MPL-2.0,
  PSF-2.0 and native-extension policy review; source-role and Matcha identity
  are still deliberately unauthenticated. Sol independently repeated the
  lock check, inspector and worker self-tests, intentional gate exit 2,
  Python compilation, shell checks and lock digest check. No dependency sync,
  VAST collection, native composite execution, parity or upload has occurred.
- Qwen2-Audio-7B-Instruct now rejects the former arbitrary single-shard
  pass-through and records the canonical 16.8-GB five-shard release without
  converting it. Its VAST inspector fixes the official Qwen project and the
  first stable Transformers tag that actually contains Qwen2-Audio
  (`v4.45.0`), with separate Qwen2-Audio, Whisper frontend and Qwen2 decoder
  role hashes. It authenticates HF Git/LFS objects, strict JSON/config/tokenizer
  fields, bounded safetensors headers and each index-to-physical-shard tensor
  assignment. The official Qwen project snapshot has no license file, so its
  source license remains an explicit unknown blocker even though the model and
  Transformers snapshots declare Apache-2.0. Native audio/text-to-text
  execution, dataset provenance and Apple parity remain inspection-only.
- Step-Audio-2-mini now rejects the former arbitrary safetensors/license
  pass-through and records the canonical 16.6-GB four-shard language model
  together with its CampPlus, speech-tokenizer, flow and HiFT token2wav
  companions. Its fixed-revision VAST inspector authenticates the complete
  HF Git/LFS tree, bounded safetensors headers and exact index ownership,
  inspects ONNX graphs without external-data execution, and permits PyTorch
  companions only through `weights_only=True`. The official Step-Audio2 and
  Transformers v4.49.0 roles are recorded separately. Manager review accepts
  the corrected local contract: it uses the canonical underscored custom-code
  names and `audio_encoder_config` axes, resolves bounded ONNX external data
  relative to each graph, and distinguishes an authenticated collection from
  an inspection error while keeping both fail-closed. The worker has not run
  on VAST. Native S2S execution, component/dataset provenance and CPU/Metal
  parity remain blocked; the worker contains no conversion or publication
  route.
- Kimi-Audio-7B-Instruct now rejects its former single-safetensors
  pass-through and treats the canonical 42.6-GB composite snapshot as one
  inseparable inspection boundary. Its 128-GiB VAST worker authenticates the
  unusual 35-of-35 plus five-tensor 36-of-36 shard layout, embedded Whisper,
  audio detokenizer and vocoder, and verifies local Git/LFS content against a
  complete recursively walked server tree. Pickle-bearing components are
  inventoried and admitted only through `weights_only=True`; non-empty unsafe
  globals remain blockers. The Kimi source, its pinned GLM-4-Voice gitlink and
  submodule origin are recorded separately because the source root has no
  single license file. Native composite execution, dependency/license review,
  parity and publication remain fail-closed.
- Granite Speech 4.1-2B now rejects its historical arbitrary-shard conversion
  path and fixes the canonical three-shard IBM release, its auxiliary output
  projection, IBM wrapper source and the exact Transformers implementation
  revision as one inspection boundary. The VAST inspector validates the full
  recursively walked Git/LFS tree and binds every `RepoFile` to authenticated
  HEAD commit, size and ETag metadata. Regular Git blobs, canonical LFS pointer
  objects and LFS payload digests remain distinct; the four reviewed Xet-backed
  weight files are additionally fixed by payload size and SHA-256. The inspector
  also checks bounded duplicate-key safetensors headers, exact index ownership
  and the nested Sigstore v0.3 DSSE/in-toto resource manifest. It checks every
  declared resource hash and the exact serialization policy while retaining an
  explicit blocker because certificate, transparency log and signature
  cryptography are not yet verified. Collection errors now force
  `inspection_status=INSPECTION_ERROR`; only a fully matched collection may emit
  `AUTHENTICATED_EVIDENCE_COMPLETE`. Manager correction
  replaced nonexistent IBM model-role paths with five real SDK wrapper roles
  and fixes their Git objects, six exact Transformers implementation roles and
  both LICENSE objects; each role must match HEAD and streamed working bytes in
  a clean regular-file checkout. Native encoder,
  projector and Granite decoder composition, dependency/dataset provenance,
  CPU parity and Apple Metal execution remain inspection-only.
- Qwen2.5-Omni-7B now rejects arbitrary safetensors and treats its canonical
  22.4-GB five-shard Thinker/Talker/Token2Wav release as a single inspection
  boundary. The 128-GiB VAST worker materializes a fixed HF revision, compares
  the complete Git/LFS tree with local content, validates every bounded header
  and index-to-shard assignment, and safely inventories the fixed speaker
  archive with `weights_only=True`. Its config contract follows the real nested
  Thinker audio/text/vision, Talker, DiT and BigVGAN paths, and records the
  fixed Qwen source plus the exact Transformers Qwen/vision/audio/Whisper role
  files. Native multimodal streaming input, text output, speech generation,
  component/dataset provenance, numerical parity and Apple execution remain
  fail-closed.
- Hibiki-2B now rejects arbitrary or partial safetensors instead of emitting a
  runtime-looking translation GGUF. Its 128-GiB VAST inspector fixes the exact
  six-file Kyutai release, separates Hugging Face LFS content SHA-256 from Git
  pointer-blob identity, validates bounded LM/Mimi headers, the 48,000-piece
  SentencePiece structure and the real root/nested streaming configuration
  paths. It also records clean, tracked Hibiki and Moshi source snapshots at
  fixed commits, including the actual conditioner and TTS entry-point paths.
  Native simultaneous FR/EN streaming translation, dependency/dataset
  provenance, upstream parity and Apple execution remain inspection-only.
- Kyutai TTS 1.6B EN/FR now rejects arbitrary safetensors and license
  overrides instead of flattening the delayed-streams model into a misleading
  single-file GGUF. Its VAST inspector authenticates the exact six-file model
  release plus a separately selected CC0 voice, keeps model/source/voice
  licenses distinct, validates LM and Mimi headers, the 8,000-piece tokenizer,
  fixed config paths and scheduled depformer axes, and compares complete local
  materialization with Git/LFS server identities while excluding Hugging Face
  cache metadata. Fixed Moshi and delayed-streams source checkouts must be
  clean and all implementation/license roles tracked. Native demux,
  conditioning, Mimi decode, parity and Apple execution remain fail-closed.
- VibeVoice Realtime 0.5B now rejects the former arbitrary BF16 pass-through
  and cannot emit a runtime-looking GGUF. Its VAST-only inspector fixes the
  exact Microsoft model, VibeVoice source, Transformers 4.51.3 implementation
  and separately selected Qwen2.5 tokenizer revisions; authenticates the full
  model Git/LFS tree, bounded safetensors header and BF16 parameter total; and
  validates the real streaming decoder, diffusion-head, acoustic-tokenizer and
  preprocessor config paths. Source checkouts must be clean and every runtime
  role tracked. A completed evidence collection is explicitly distinguished
  from an inspection error before the worker accepts the expected exit 2.
  Native streaming state, diffusion/CFG, acoustic decode, tokenizer behavior,
  policy/dependency review, CPU parity and Apple execution remain fail-closed.
- FireRedASR-AED-L now rejects its historical arbitrary safetensors bridge and
  every license override without creating an output. Its 128-GiB VAST-only
  inspector fixes the exact eight-file Apache-2.0 release and official source
  commit, authenticates Git/LFS identities, safely inventories the nested
  PyTorch checkpoint with `weights_only=True`, and rejects unsafe archives,
  unsupported objects, cycles, unbounded metadata and non-finite tensors. The
  1,000-piece SentencePiece model, 7,832-entry token dictionary and textual
  2-by-81 CMVN statistics are structurally checked; the empty published config
  and binary CMVN interpretation remain explicit blockers. Completed evidence
  and inspection failure have distinct manifest states. Native Conformer/AED
  execution, beam decoding, dependency/dataset provenance, CPU parity and
  Apple Metal execution remain fail-closed.
- OWSM v4 medium 1B now rejects arbitrary safetensors, PyTorch checkpoints and
  license relabels without creating an output. Its 128-GiB VAST-only inspector
  fixes the exact ESPnet release and tagged source revision; authenticates the
  selected checkpoint, 50,000-piece SentencePiece model, embedded 50,002-token
  list, global-MVN statistics, model card and complete remote tree; and safely
  inventories the nested checkpoint with `weights_only=True`. The exact
  E-Branchformer/Transformer/frontend/preprocessor configuration is checked,
  including the list-shaped `espnet/yodas_owsmv4` model-card declaration.
  Completed evidence and inspection failure remain distinct manifest states.
  Native ESPnet S2T execution, joint CTC/attention decoding, dependency and
  dataset provenance, CPU parity and Apple Metal execution remain fail-closed.
- Canary-Qwen 2.5B now rejects arbitrary single-file conversion instead of
  presenting the NeMo/Qwen composite as a runtime-ready GGUF. Its 128-GiB
  VAST-only inspector fixes the exact six-file NVIDIA release, NeMo v2.5.0
  source commit and separately selected Qwen3-1.7B tokenizer revision. It
  authenticates complete Git/LFS identities, strict README and leaderboard
  YAML, the Canary/FastConformer/LoRA configuration, bounded header-only BF16
  tensor structure and the Qwen tokenizer's BPE/vocabulary/merge contract
  without downloading the companion language-model weights. Canary CC-BY-4.0,
  Qwen Apache-2.0 and NeMo Apache-2.0 evidence remain component-separated.
  The historical public GGUF is recorded as a stale pre-contract placeholder;
  native SALM composition, tokenizer behavior, dependency/dataset provenance,
  CPU parity and Apple Metal execution remain fail-closed.
- GigaAM v3 and GigaAM Multilingual now reject arbitrary checkpoints,
  preparation and runtime binding instead of conflating two different output
  heads. Their 128-GiB VAST-only inspector fixes both complete HF trees and the
  official source revision, separates Git LFS pointer identity from downloaded
  payload identity, and safely inventories each PyTorch checkpoint with
  `weights_only=True`. The v3 contract is the exact 1,025-class RNNT decoder
  and joint network; the multilingual contract is the exact 71-class CTC head
  with 70 ordered symbols plus an implicit blank, not an invented blank token.
  Exact frontend/Conformer axes, model cards, the v3 SentencePiece structure
  and both stale public GGUF identities are recorded. Native RNNT/CTC forward,
  dependency and dataset provenance, CPU parity and Apple Metal execution
  remain fail-closed.
- Kyutai STT 2.6B EN now rejects arbitrary safetensors and public GGUF loading
  instead of binding deterministic synthetic weights behind a runtime-looking
  artifact. Its 128-GiB VAST-only inspector fixes the exact six-file release,
  authenticates the separate LM, Mimi and 4,000-piece SentencePiece identities,
  and records clean fixed revisions of both delayed-streams-modeling and Moshi.
  The published config names `tokenizer_en_audio_4000.model` while the fixed
  tree contains `tokenizer_spm_4k_en.model`; that mismatch is retained as a
  blocker rather than silently aliased. The stale historical GGUF is recorded,
  and native composite binding, streaming STT, dependency/dataset provenance,
  CPU parity and Apple Metal execution remain fail-closed.
- ChatTTS now rejects arbitrary single-file conversion, license relabeling and
  every public GGUF load route instead of treating the historical GPT-only
  artifact as a complete synthesizer. Its VAST-only inspector pins the exact
  23-file release, authenticates the nine safe GPT, Embed, DVAE, Decoder,
  Vocos and tokenizer assets against both Hugging Face Git/LFS identities and
  the fixed v0.2.5 source SHA map, and parses the AGPL source config without
  importing or executing it. The GPT, VQ, DVAE, standalone decoder and Vocos
  axes and tokenizer structure are recorded separately from the
  CC-BY-NC-4.0 weight license and AGPLv3+ implementation license. A bounded
  official-run adapter now stages bounded binary values for the tokenizer,
  Embed output, generated IDs and hidden states, the exact probability tensor
  passed to each `torch.multinomial` call, DVAE/Decoder mel and Vocos PCM for
  both `use_decoder` settings. It has not run on VAST. The unreachable
  standalone-Decoder/DVAE tap was removed; stored float32 bytes are decoded
  again, integer IDs and masks are range-checked, and probability/artifact
  cardinality is bound to one execution identity. The dedicated project now
  lists the fixed-source `numba` and `tqdm` import dependencies. A dedicated
  Python 3.12 lock is now staged with 41 concrete package rows plus its virtual
  root; Torch and Torchaudio resolve only through the official CPU index. The
  lock SHA is
  `36986402c3badb45b50c9d18ffbc811409be618cf45e2438f97e99c6751235db`,
  the concrete inventory digest is
  `f8b00a8226662347ccf2e0ef7420922614ec570524ca6216852ee699f32db98a`,
  and the all-row digest binds package resolution markers and every dependency
  version/source/marker edge as
  `9714e1a005af4800608f608c9617e0ce90dec0c563427e7c693d9c603ea2cf52`.
  A stdlib/no-project preflight rejects dedicated environment sync and source
  or model acquisition until the unresolved PSF/MPL/Unlicense and bundled
  numerical-library notices receive explicit owner approval; its audit record
  digest is
  `38d0b49ad2b3fafd34bf19eaf1c955e53f0d7b5eb362612d0292a23d3e59148a`.
  The typed Rust component descriptions and unexecuted adapter are not a native
  binder. Native clean-room composition, dependency/dataset and
  personality-rights approval, CPU parity, Apple Metal execution and any
  non-commercial publication remain fail-closed.
- Chatterbox Multilingual v3, Nano and Turbo now reject every arbitrary
  single-checkpoint conversion instead of presenting a T3-only GGUF as a
  complete synthesizer. Their shared 128-GiB VAST inspector pins all three
  complete Hugging Face server trees and the official source revision, checks
  Git blob and canonical LFS-pointer identities, and materializes only the
  selected T3, tokenizer, conditioning, voice-encoder and S3Gen/meanflow
  components. Safetensors are bounded header-only inventories; `conds.pt` is
  accepted only through `torch.load(..., weights_only=True)` plus a bounded
  deterministic tensor manifest. The effective Llama/GPT-2 topology,
  vocabulary, merge table and 19 paralinguistic tags are checked against the
  pinned implementation. A T3-only source-shaped contract and one-step
  official-reference adapter are staged, including probability-tensor capture.
  The corrected inspector now requires each complete exact server identity and
  Base selects both `Cangjie5_TC.json` and `mtl_tokenizer.json`. The reference
  follows the official Turbo one-step initial-sample route, reproduces the Nano
  and Turbo `tfmr.wte` removal, and the Rust contract follows the pinned
  ascending-tail Top-P semantics with tie-split rejection and a single Top-K
  selection. Focused inspector/reference/worker self-tests, Python compilation,
  shell checks and scoped Rust formatting pass locally. A dedicated Python
  3.12 lock is now staged and routes Torch/Torchaudio exclusively through the
  official CPU index; its 39-package closure excludes CUDA/NVIDIA, Triton and
  soxr. The exact lock SHA and a canonical digest over all 41 package rows
  (39 unique packages, including version/source/marker identity) are now bound
  into both workers and the evidence schema. The full version-keyed license
  inventory records certifi/tqdm/NumPy as unmodified VAST-only,
  non-redistributed reference dependencies. `typing-extensions==4.16.0`
  remains the sole license blocker because its PSF-2.0 declaration is outside
  the repository's Apache/MIT/BSD allowlist and has no owner clearance. Sol
  repeated lock check, tamper-aware reference self-test, intentional license
  exit 2, both worker self-tests, Python compilation, shell checks and diff
  hygiene successfully. No real source import, multilingual tokenization or
  model trace has run on VAST. The
  historical public GGUFs
  remain partial strict checkpoints; complete tokenization, autoregressive
  generation, conditioning, VE, S3Gen/meanflow, watermarking, PCM parity and
  Apple Metal execution remain fail-closed.
- Fun-CosyVoice3-0.5B-2512 now rejects arbitrary safetensors and sidecars
  without producing a GGUF. Its 128-GiB VAST-only inspector authenticates the
  complete fixed 20-file Hugging Face tree, local LFS payloads and canonical
  pointer Git blobs, records the Qwen tokenizer/config, safe tensor headers,
  weights-only LLM/flow/HiFT checkpoint manifests, ONNX evidence-only assets,
  the official CosyVoice source revision and exact Matcha gitlink. The
  historical 293-tensor public binder remains explicitly LLM-only and cannot
  produce PCM. The corrected source-executed non-stream reference now includes
  `cfm_solver_trace` in both required role sets, binds the exact
  `1-cos(i*pi/20)` ten-step grid and all estimator x/mu/t/output/mask/speaker/
  conditioning values, and verifies official CFG row order: row 0 is
  conditional, row 1 is zero for mu/speaker/conditioning, while both mask rows
  equal the input mask. RAS evidence fixes sampling/top-p/top-k/window/tau and
  vocabulary, maps every ignored-control retry plus optional repetition
  fallback to its outer sampling call, and uses a true float32 four-ULP gate
  with five-ULP rejection. Manager review accepts these corrections as staged
  evidence only. The dedicated `uv.lock` is absent and no official packet has
  run, so native tokenizer, LLM decode, speech tokenizer,
  CampPlus, flow, HiFTNet, dependency/dataset review, CPU parity and Apple
  Metal execution remain fail-closed.
- CosyVoice2-0.5B now rejects all arbitrary single-file conversion surfaces
  and every public GGUF load instead of constructing a partial LLM handle.
  Its fixed-revision VAST inspector authenticates the complete release tree,
  canonical Git-LFS pointer and payload identities, safe safetensors headers,
  and each PyTorch component only through `torch.load(...,
  weights_only=True)`. Every checkpoint inventory carries a deterministic
  tensor manifest, and the fixed CosyVoice/Matcha source revisions and exact
  tokenizer, LLM, flow and HiFT roles are recorded. The corrected internal
  route and official adapter now model every sampling attempt within its outer
  generation step: only ignored EOS retries in place, an ID above EOS is an
  accepted control result that consumes the step without yielding, `min_len`
  is checked against outer steps, and the next `forward_one_step` input keeps
  or collapses the previous row count according to that accepted result. The
  trace also separates the official nucleus draw from the optional
  repetition-triggered full-softmax fallback, captures the exact
  `Tensor.multinomial` receiver, and binds non-streaming `finalize=true` to the
  source's generated-only mel return without a second prompt slice. Manager
  review accepts these source-state corrections as staged evidence, not as a
  native success: the dedicated reference `uv.lock` is still absent, no
  official packet has run, and the Rust path remains behind an unauthenticated
  composite binder. The historical public artifact remains LLM-only; full
  tokenizer, speech-tokenizer, CampPlus, locked source-exact flow/CFM/HiFT
  evidence, CPU parity and Apple Metal execution remain fail-closed.
  A later source-closure audit confirmed that absence is intentional rather
  than an unfinished lock-generation command: the fixed official frontend and
  the HyperPyYAML model graph resolve
  `matcha.utils.audio.mel_spectrogram`, whose pinned implementation imports
  `librosa`; the Python 3.12 closure then includes forbidden `soxr`. The
  dedicated project is now explicitly inventory-only and forbids sync and
  acquisition until an exact official route without that closure is
  authenticated and owner-approved. The same conclusion applies to the
  CosyVoice3 reference project. Sol repeated both reference self-tests, the
  CosyVoice3 validator and corrected general-environment Apple self-test,
  Python compilation, shell syntax, ShellCheck and whitespace checks. Both
  dedicated locks and `.venv` directories remain absent; no dependency or
  model was acquired.
- Dia-1.6B now rejects arbitrary safetensors conversion and classifies the
  historical 343-tensor public artifact as a partial diagnostic checkpoint.
  Its 128-GiB/40-GiB-tmpfs VAST inspector authenticates the exact six-file
  upstream tree, separately validates Git-LFS pointer blobs and materialized
  payloads, inventories the PyTorch checkpoint only with
  `weights_only=True`, compares the deterministic PTH and safetensors
  mappings, records the fixed official source roles, and validates the exact
  public GGUF identity/header. The branch now also stages an internal
  source-shaped batch-one encoder/decoder route with persistent paired CFG
  caches, the byte-first `[S1]`/`[S2]` text boundary, the exact
  `[0,8,9,10,11,12,13,14,15]` delay schedule, constrained first and later
  sampling steps, 15-step EOS drain, prompt/BOS-excluding output slicing and
  a fail-closed exact-top-k tie boundary. The official adapter now captures
  one complete `_encode_text` result, paired conditional/unconditional
  encoder rows, every sampler probability/selection, delayed and reverted
  nine-codebook frames, DAC latent and PCM, and its independent validator
  requires `pcm_samples == reverted_frames * 512`. It deliberately cannot
  emit `REFERENCE_COMPLETE`: the exact Descript 44.1-kHz checkpoint body,
  installed source/package tree and a content-bound mapping to Vokra's
  `DacVariant::Khz44` 328-tensor manifest are not yet pinned, and the adapter
  raises before model loading instead of accepting a prose equivalence claim.
  A dedicated 34-package Python 3.12 reference lock is now staged. It routes
  Torch/Torchaudio through the official CPU index and excludes Gradio,
  librosa, soxr, Triton, NVIDIA packages and the broad
  `descript-audio-codec` wheel closure. The adapted reference instead requires
  a separately authenticated minimal DAC source shell and explicitly disables
  `torch.compile`. Its license status remains
  `BLOCKED_UNREVIEWED_TRANSITIVE`: all 34 package rows now carry canonical
  version/source/marker hashes and license conclusions, with certifi,
  typing-extensions and bundled/native runtime policy cases explicitly left
  unresolved. The exact DAC proof path still raises before model loading.
  Manager repetition of lock check, adapter/validator self-tests, both worker
  self-tests, corrected no-sync Apple self-test, shell syntax, ShellCheck and
  diff hygiene passes. No dependency sync, official reference packet or real VAST CPU
  execution exists yet. Complete `crate::dac::Dac` composition, independent
  tokenizer/generation/PCM parity and Apple Metal execution remain
  fail-closed.
- CSM-1B now runs the strict weight-license gate before its unconditional
  production refusal, so a license-less GGUF cannot hide behind a generic
  runtime blocker. Even a permissively stamped legacy file remains
  `INSPECTION_ONLY`. The corrected inspector binds the exact 20-file composite
  tree, authenticated HEAD/Xet-LFS identities, restricted checkpoint load,
  embedded tokenizer/Mimi roles, the fixed Sesame source and the exact
  Transformers 4.52.1 CSM implementation including `generation_csm.py`.
  Source roles and both standard Apache licenses are fixed by Git blob and
  substantive license clauses. A 28-row CPU-only Python 3.12 lock is now
  staged and excludes CUDA/NVIDIA, Triton, soundfile/libsndfile, librosa and
  soxr, but remains behind an affirmative lock-bound license gate. Every lock
  row carries exact version/source/resolution-marker identity and a canonical
  digest; every row also has a versioned license conclusion. Certifi/tqdm MPL,
  `typing-extensions` PSF-2.0 and NumPy bundled/native runtime notices remain
  explicit owner-policy blockers. Manager source review found that the first
  proposed ndarray route was not executable: pinned Transformers calls
  `requires_backends(load_audio, ["librosa"])` before checking whether the
  audio value is already an ndarray. The corrected adapter therefore renders
  the official chat template with `tokenize=False`, then calls the fixed
  official CSM processor directly with the rendered string and authenticated
  caller-owned PCM16 WAV data decoded by the Python standard library. A
  text-only packet supplies `audio=None`; non-empty audio placeholders supply
  the exact ordered array list, matching the pinned processor's empty-audio
  contract. The greedy evidence records processor IDs/mask, models the depth
  decoder's two-token prefill then
  30 single-token calls per frame, and independently checks the previous-
  codebook input chain. It separates the official 31-codebook stop condition
  from the all-32-codebook codec cutoff and binds PCM to the 1,920-sample Mimi
  frame hop. Sol independently repeated the lock check, tamper-aware reference
  and both worker self-tests, Python compilation, shell syntax, ShellCheck and
  diff hygiene; no dedicated `.venv` was created. No dependency sync,
  model/source acquisition or official VAST reference run has occurred. Exact
  composite weight-license sign-off, native
  tensor mapping/tokenization/Mimi decode, CPU parity and Apple Metal execution
  remain fail-closed before a production binder can exist.
- Irodori-TTS-500M-v3 now rejects arbitrary single-file conversion without
  emitting a GGUF, and its legacy `with_codec` surface explicitly refuses a
  vanilla `DacCodecGguf`. The required
  `Semantic-DACVAE-Japanese-32dim` decoder is a distinct continuous-latent
  component. Source-shaped RF schedule, independent/joint/alternating CFG,
  duration and codec-evidence contracts plus no-upload VAST/Apple workers are
  staged. The corrected adapter now calls the learned official runtime when
  its fixed source `uv.lock`, exact model/codec packets and selected immutable
  tokenizer assets are present; it captures official token IDs, duration,
  source-owned RF noise/schedule, final patched latent and 48-kHz Semantic-
  DACVAE PCM. No such run has occurred. The fixed source revision
  `8224dafb46d0aba89209a8f905f1cb7e3299d9c1` authenticates its existing
  source `uv.lock` as
  `8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c`.
  That lock proves the mandatory route
  `dacvae@1.0.0` (commit `414c20785fc3a28373073ea8ef7a1316eeeaca6e`)
  through `descript-audiotools@0.7.2`, `librosa@0.11.0`, `soxr@1.0.0`,
  `soundfile@0.13.1`, `cffi@2.0.0` and `pycparser@3.0`; the source codec also
  imports SoundFile directly and therefore brings a native libsndfile route.
  A stdlib/no-project gate now records that exact closure and exits 2 before
  source/model download, dependency sync, source import, bundle consumption or
  Cargo on both VAST and Apple. A separate dedicated lock would not remove this
  authenticated blocker. The selected tokenizer revision is adapted evidence
  rather than an upstream pin, while exact safetensors role mapping and a
  native Semantic-DACVAE binder remain unaccepted. Tokenizer/reference
  conditioning, duration, RF-DiT sampling, native PCM decode and independent
  parity remain inspection-only.
- AudioGen Medium's released 588-tensor GGUF contains only the language-model
  surface. Vokra already has native AudioCraft LM/generation primitives for
  CPU and Metal, but the exact release additionally requires the authenticated
  text tokenizer/conditioner and AudioGen's 16-kHz, four-codebook EnCodec
  RVQ/SEANet path. FunCodec and SpeechTokenizer are different codec contracts
  and cannot be substituted. A corrected VAST inspector now binds the exact
  four-file release tree, canonical LFS pointer and payload identities, safely
  walks the nested checkpoint/config structures with bounded
  `weights_only=True` loading, and binds all selected AudioCraft v1.0.0 source
  roles. It records that execution source separately from the earlier weight
  upload instead of claiming it as weight-build provenance, and distinguishes
  the current CC-BY-NC-4.0 declarations from the historical v0.0.2
  CC-BY-NC-ND-4.0 license. Focused local self-tests and shell checks pass. The
  dedicated reference lock is absent and no real inspection has run; the exact
  external text-conditioner name/size must still be recovered from authenticated
  checkpoint evidence. A fixed official reference, complete companion
  composition, native codec execution, real VAST CPU parity and Apple Metal
  execution are all still required; the public artifact remains partial.
- AudioLDM2 base and large are not complete native artifacts despite their
  4.47-GB and 5.95-GB public GGUF sizes. The current offline preparers merge
  the VAE, 2-D conditional UNet, SpeechT5 HiFi-GAN, GPT-2 language model,
  CLAP and FLAN-T5-large tensors but omit the mandatory
  `AudioLDM2ProjectionModel`; they also allow missing component directories
  and do not preserve the scheduler, feature extractor, tokenizers and config
  sidecars needed to execute the pipeline. The permissive converter accepts
  arbitrary safetensors, while the runtime only inventories a nonempty tensor
  list and unconditionally refuses generation. The exact official snapshots
  are `cvssp/audioldm2@c8e7e189d324425c05c4c2f81214041ef4107983`
  and `cvssp/audioldm2-large@4b0b875a9e0c5305dfc917da808584e50e1c7ed4`;
  their selected safetensors total 4,474,362,800 and 5,958,866,688 bytes,
  respectively, so every conversion and validation run belongs on VAST.
  Native completion requires exact FLAN-T5-large and full unfused-CLAP text
  and audio towers, GPT-2, the learned projection/SOS/EOS bridge, the 2-D
  cross-attention UNet, AutoencoderKL, exact 200-step DDIM schedule and
  SpeechT5 HiFi-GAN. The existing flow sampler, fused-CLAP audio-only route and
  1-D VAE descriptions are not substitutes. Both public repositories remain
  fail-closed. A corrected strict fixed-tree inspector is now staged for both
  revisions: it authenticates the exact 11-component model index, bounded
  component-local safetensors headers, complete Git/LFS tree, Diffusers
  v0.21.0 tag/role objects, the CC-BY-NC-SA-4.0 weight declaration and the
  distinct Apache-2.0 source license. Authenticated collection and inspection
  errors are separate manifest states. The fixed Diffusers setup and pipeline
  imports are now recorded as an explicit dependency contract, but upstream
  supplies only broad lower bounds and no authenticated exact Python 3.12
  Torch/Transformers graph. The dedicated reference `uv.lock` therefore remains
  intentionally absent instead of guessing versions; the worker stops before
  acquisition and no real inspection has run. A complete replacement contract,
  independent reference, native CPU implementation and Apple Metal execution
  remain open.
- MOSS-TTS Local Transformer v1.5 now has an explicit `[rows,13]` prompt and
  generation contract with the distinct MOSS Audio Tokenizer v2 stereo
  decoder. Its fixed-revision official custom-code runner authenticates the
  complete HF Git/LFS tree, canonical LFS pointer identities, bounded
  438-tensor header and the exact loaded source roles. The real VAST/Apple
  test hard-fails unless deterministic CPU rows and 12-codebook assistant
  codes exactly match the official reference and Metal exactly matches CPU;
  PCM finiteness, rate, channel and length consistency are executed but
  end-to-end official PCM parity remains explicitly `COMPOSITE_PCM_NOT_RUN`.
  The independent v2 codec evidence stays separate until the real workers run.
  Its composite VAST and Apple workers now require two distinct external
  approval files: one for the Local 50-package closure and one for the already
  reviewed Audio Tokenizer v2 closure. Both exact gates run before host,
  work/cache, synchronization, network or Cargo activity. VAST accepts only an
  absent canonical work target disjoint from the checkout, both projects,
  prompt input and approval files; Apple likewise accepts only an absent
  evidence target disjoint from the checkout, all transferred inputs and both
  approvals. The Local gate rejects duplicate JSON keys and binds the exact
  lock/project bytes, strict package/source/dependency/artifact schemas, the
  unique virtual project, all version-keyed reviews and the fixed model/source
  identity. Sol repeated the 50-package offline lock check, Local and v2 gate
  self-tests, both worker self-tests, Python compilation, Bash syntax,
  ShellCheck and whole-tree whitespace checks. A production-shaped invocation
  exited 2 at the unresolved Local and v2 reviews without creating work,
  scratch or cache data. No dependency sync, model acquisition, conversion,
  Cargo, VAST, Apple or upload run occurred, so the row remains open.
- Zonos-v0.1-transformer's fixed 3,248,843,808-byte safetensors payload has a
  26,584-byte header containing exactly 246 tensors. Header-only HTTP range
  inspection authenticates the complete schema without downloading the model:
  26 transformer blocks each contribute fused QKV, output, packed SwiGLU and
  two affine LayerNorm pairs; the remaining tensors are final norm, nine input
  embeddings, nine heads and the complete seven-conditioner stack. The latter
  includes the 189-row fixed phoneme-symbol embedding, speaker projection,
  Fourier weights, language embedding, every learned unconditional vector and
  the final 2048-to-2048 prefix projection plus LayerNorm. A caller-supplied
  already-projected prefix therefore cannot count as the native conditioner
  route. Official generation also drains all nine delayed codebooks after CB0
  EOS before reverting the delay pattern and sending codes to the 44.1-kHz DAC.
  The branch now stages that raw typed conditioning route, the 26-layer
  transformer with persistent KV state, delayed-codebook generation and the
  complete `crate::dac::Dac` boundary. Its offline packet carries the actual
  asymmetric speaker/emotion controls and an authenticated content digest;
  the official fixed-source reference must emit both exact codes and 44.1-kHz
  PCM in the same evidence directory. The no-upload VAST worker requires the
  public/upstream 246-tensor manifests to match and executes native CPU codes
  before it may record authenticated evidence. The Apple worker in turn runs
  Metal only after that CPU exact-code prerequisite and preserves truthful
  failure statuses outside the checkout. None of those real runs has occurred,
  so PCM remains `MEASURED_NOT_GATED` and the repository row stays open.
- The fixed VoxCPM-0.5B upstream snapshot contains a 1,304,698,606-byte
  `pytorch_model.bin`, a separate 301,494,192-byte `audiovae.pth`, and four
  text-tokenizer/config files. The historical 1,304,607,744-byte public GGUF
  has only the 377-tensor main-model payload and no companion files, so its
  current "complete official checkpoint" runtime wording is false for PCM.
  Primary-source review also invalidates the stale AudioVAE-v2 description in
  the historical Vokra scaffold: fixed source commit
  `38a76704ee67935ccbafbe5b6725e83dbb1e9305` has `audio_vae.py`, not
  `audio_vae_v2.py`, and `AudioVAE()` uses encoder rates `[2,5,8,8]`, decoder
  rates `[8,8,5,2]`, latent width 64, decoder width 1536 and 16-kHz input/output
  (both rate products are 640). The `[8,6,5,2,2,2]` decoder / 48-kHz topology
  belongs to a later V2/2B source and must not be applied to the 0.5B artifact.
  The same fixed source confirms split-half `rotate_half` RoPE, not adjacent
  even/odd pairing. `MiniCPMLongRoPE` chooses one short/long factor table for
  the configured cache length relative to
  `original_max_position_embeddings`, duplicates the resulting frequencies
  across both head halves, and multiplies both cosine and sine caches by
  `sqrt(1 + log(max/original) / log(original))`. The released 0.5B axes have
  `max == original == 32768`, so the multiplier is one and the short table is
  selected, but a reusable runtime must preserve the generic rule rather than
  selecting a table per token position.
  Completion requires an authenticated AudioVAE plus text-tokenizer contract,
  native LM/residual-LM/CFM/DiT execution and final waveform parity; strict
  binding of the main tensor tree alone is only a partial diagnostic. The
  branch now stages the source-shaped batch-one route end to end: MiniCPM
  base/residual LM execution with persistent KV state, FSQ hidden-stream
  quantization, prompt AudioVAE encode with the exact 1280-sample padding and
  final prompt-patch drop, local encoder/DiT, dynamic-prefix split CFG with
  zero negative mu, unit-sway UnifiedCFM, first/last generated-latent trimming,
  and 16-kHz AudioVAE decode. Caller-owned draw packets are exact-length and
  no hidden RNG or backend fallback is permitted. Fragmentary composite
  loaders were removed; public loading remains deliberately closed until VAST
  fixes the exact combined main+AudioVAE tensor manifest, tokenizer/source
  provenance and independent CPU/Metal parity. No real conversion or device
  run has occurred, so the row remains open.
- The fixed VibeVoice-1.5B upstream weight snapshot has three safetensors
  shards whose 2,704,021,985 BF16 parameters are represented by the
  historical public GGUF's 1,204-tensor main manifest, but that snapshot does
  not ship the text tokenizer. The prior inspection pin to current
  `microsoft/VibeVoice@94da20d9...` was not release-compatible: Microsoft
  reset the repository on 2025-09-05 and removed the TTS implementation,
  while the current non-streaming files were added later with the ASR wave.
  The original Microsoft commit object remains directly addressable at
  `2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c` (2025-08-25, the model-release
  day) and contains the missing `modeling_vibevoice_inference.py` generation
  route. It fixes the effective sampler as deterministic order-2 midpoint
  `dpmsolver++` with a cosine alpha-bar schedule—not `sde-dpmsolver++`—and
  the full positive/negative cache, acoustic decode and semantic re-encode
  loop. Header-only range inspection also authenticates the complete 1,204
  tensor, 5,408,043,974-byte three-shard schema: the 28-layer Qwen2 decoder,
  four-layer diffusion head, both connectors, complete acoustic
  encoder/decoder and semantic encoder are all present. The official
  `final_sigmas_type="zero"` rule forces the last DPM-Solver++ step back to
  first order even with 20 inference steps; applying the second-order history
  correction on that final step is numerically wrong. The separately named
  `Qwen/Qwen2.5-1.5B` tokenizer companion is
  immutable at `8faed761d45a263340a0528343f099c05c9a4323`. The current runtime
  now stages the strict 28-layer Qwen2 full/KV route, mixed-embedding prefill,
  independent positive/negative cache forks, the four-layer diffusion head,
  deterministic order-2 DPM-Solver++ seam and both complete connector
  projections. Prompt audio replaces designated token embeddings with the
  acoustic connector output, and each generated speech frame feeds the sum of
  acoustic and semantic connector outputs back as the next `inputs_embeds`
  row. The fixed tokenizer config is causal 24-kHz mono with six ratios
  `[8,5,5,4,2,2]` (hop 3200, 7.5 Hz), depths `3-3-3-3-3-3-8`, 32 base
  filters, depthwise 7-tap mixers, affine RMSNorm at `1e-5`, layer scale
  `1e-6`, and no last norm; acoustic latents are 64-wide with `fix_std=0.5`,
  while semantic latents are deterministic and 128-wide. The branch now also
  stages exact causal Conv1D/ConvTranspose tokenizer caches, the acoustic
  encoder/decoder, semantic encoder and a batch-one composite loop. Prompt
  rows use only the sampled acoustic connector, the CFG negative branch starts
  from the single speech-start context and advances only with generated speech
  embeddings, diffusion conditions are read before the selected diffusion
  token is replaced, and speech-end zeroes established codec caches without
  terminating the LM. Random draws are caller-owned and max-step truncation is
  explicit. A no-upload VAST worker now authenticates the fixed Microsoft
  model/source, Qwen tokenizer companion and the historical 1,204-tensor
  public composite before running the actual upstream processor and
  `VibeVoiceForConditionalGenerationInference.generate`. The reference
  adapter captures the tokenizer prompt draw and the `[2*active_batch,64]`
  diffusion draw only at their pinned upstream call sites, maps one positive
  row per generated speech-diffusion token, and records official PCM plus
  scheduler latents. The native test exact-checks generated control tokens;
  PCM and captured diffusion latents deliberately remain
  `MEASURED_NOT_GATED` because native latent comparison has not been
  registered. The matching Apple worker replays the same evidence on CPU and
  Metal without fallback. Its dedicated Python 3.12 reference lock contains
  32 package rows, routes Torch only through the official CPU index, and
  excludes the unused soundfile/libsndfile/librosa/soxr path. The lock SHA is
  `a1aa0b371e5036a7f5bc72f2a5e1ba82ef21a6fa9ba8993e5612fb7612107806`;
  its `package-resolution-and-dependency-markers-v2` digest
  `ae07242d3b0e4d8fdda8b7435956b835a996e003a6615660358a01dbfd9bddf6`
  covers every package resolution marker and dependency-level
  version/source/marker qualifier. The version-keyed license-row digest is
  `6cca02093a2b76c728f0957193657f614e6f443e13805705423b384c5aa6c0ca`.
  Certifi, Filelock, NumPy bundled notices, PyYAML, Tqdm and
  Typing-Extensions remain unapproved, so a stdlib/no-project gate exits 2
  before dedicated reference use, acquisition, bundle validation or Cargo.
  None of those real workers has run, so this remains source/validation
  staging and the repository row cannot be checked.

Local manager verification has passed for every reviewed worker and preparer
listed above: their hermetic self-tests, `bash -n`, `shellcheck`,
`cargo fmt --all -- --check`, the applicable repository shell gates and
`git diff --check`. Counts are intentionally omitted while this execution
branch is still adding workers. No model download, conversion,
`-p vokra-models` Cargo run, workspace Cargo run or Apple hardware run was
performed locally.

An earlier 2026-08-29 manager matrix ran every then-staged worker
`--self-test` except the three files being edited concurrently. Of the 33
inspection workers exercised, 31 passed; CosyVoice3 and Dia stopped at their
intentional pre-download gate because the dedicated reference `uv.lock` is
absent at that time. All 35 Apple workers exercised in that earlier subset
passed. The later complete 38-worker Apple sweep and its current Dia/CosyVoice3
findings are recorded near the top of this section. Of the 33 exercised VAST
validation workers, 31 passed; AudioGen Medium and AudioLDM2 exposed a broken
self-test assertion that greps the reference `pyproject.toml` for the literal
text `uv.lock`. Those two assertions have since been corrected and their
focused manager self-tests pass. Their real paths remain independently blocked
before download because the dedicated locks are absent: AudioGen still needs
authenticated checkpoint-derived conditioner selection, while AudioLDM2 lacks
an upstream-selected exact Python 3.12 dependency graph. These are
worker-contract results only, not model, CPU, Metal or parity verdicts.

The initial dedicated-project inventory had no `uv.lock` for 11 directories:
AudioGen Medium, AudioLDM2, CosyVoice2, CosyVoice3, FireRed ASR LLM-L, Higgs
Audio v3 TTS 4B, MAGNeT Medium, MAGNeT Small, MelodyFlow, MicroWakeWord and
RMVPE. The three AudioCraft projects and MicroWakeWord now have reviewed child
locks, leaving six intentionally lockless directories: AudioGen Medium,
AudioLDM2, CosyVoice2, CosyVoice3, FireRed ASR LLM-L and Higgs Audio v3 TTS
4B. Each remains blocked on its separately recorded source, dependency, license
or artifact-identity boundary; none may inherit the parent workspace lock as
an implicit approval.

The lock audit also found a separate pre-execution safety blocker in all three
AudioCraft conversion-only projects: MAGNeT Small, MAGNeT Medium and
MelodyFlow fell back from `torch.load(..., weights_only=True)` to an unrestricted
legacy `torch.load` when the keyword was unavailable. The accepted correction
removes that retry completely, adds a hermetic one-call regression test and
gives each project a Python 3.12, official CPU-Torch-index lock with 14 concrete
package rows. Their lock SHA-256 values are respectively
`2b167917010f8b58ac3b1bb6ded945045cebf79d376143e18812f77b4ef3e123`,
`b8391e4eede2d8aa51951a47b2b463b533d59be88b7cc97c9814f4b7d6e4575c`
and `76527ac18290907525f0b931a9ef7c8ce992100ed2449e83f6f5e9eb0cb7033a`.
Their canonical package-row digests are
`42a4d9a37b51432aa6281b70a6efe9e390019b0cb1dc589ba19725cc7ea90f46`,
`6309c2e85f8e17dc98fe08fa5521fd0b6c9f78180996977222f3eddafdd4c174`
and `6d2d61526f6c17786d3802f574bd42a5cebe2708ba82471ecfff068136970f47`;
the common license-row digest is
`31c7b824efab5aaac83d8cba8e884de80ec0648c85f5a7769fe8493ebaa4290c`.
The Sol manager re-ran all three offline lock checks, preparer and gate
self-tests, all six worker self-tests, Python compilation, `bash -n`,
ShellCheck and focused whitespace checks. Real gates correctly exit 2 because
the exact HF/source revisions and primary license evidence remain unresolved.
All three artifacts meet the project's 2-GB VAST threshold, so their checkpoint
preparation, validation and any resulting lock import smoke remain VAST-only;
no acquisition, conversion or parity run has occurred.

MicroWakeWord is now isolated in a Python-3.12-only 17-package child lock with
SHA-256
`43e17e20616bc06072424abadaaed520244673db2f964a29ea2472e22e72afbe`,
complete package/dependency digest
`3250cac13ab9f8cf0a67ffc1f590988afa8cac3b346edf52d0e03924ec08ef06`
and version-keyed license digest
`2bcae92a909b92617e1ddc96a7cf4704a6c9305dcd94651584da4b68c49a7906`.
The source is fixed at
`kahrendt/microWakeWord@4665173cd35f1cff9a61e06fc427f124766c488e`; the
model mirror is fixed at
`esphome/micro-wake-word-models@05b65922cc433c9df13e98e32a7fe520758c837e`,
with the target TFLite, JSON and both license Git objects recorded separately
from the still-unknown artifact-byte SHA-256. Sol repeated the child offline
lock check, inspector/worker self-tests, Python compile, Bash syntax,
ShellCheck and whitespace checks. The real gate exits 2 before sync or
acquisition because the compiled ai-edge-litert/protobuf notices, certifi and
tqdm MPL terms, typing-extensions PSF terms, NumPy/PyYAML/ml-dtypes bundled
notices and target byte identity remain unapproved. No model/source artifact,
package environment, conversion, Cargo run or upload was performed.
RMVPE is now isolated in a Python-3.12-only 40-package child lock with SHA-256
`747057f4e8596d801d5d0450e6e10a33fc467ab9e9a6cf2063460d1ea019919d`,
complete package/dependency digest
`ecc622c63e8a487c4440cdc838f22af7b31fae783cca41f693b0f870dd9a1819`,
resolution-marker digest
`70a0c0d228b605430c8219bfc8e4ed66652a5f06d64cab841fee543266f3bffa`
and version/source-keyed license digest
`2afebac3c079863d28415885412c11fd2acf7e3f3b9a686e2c855455da8eedec`.
Torch and Torchaudio resolve only through the official CPU wheel index; every
other external package resolves through PyPI, and both RMVPE and
MicroWakeWord are explicitly excluded from the parent parity project. The
checkpoint-to-safetensors weights-only bridge also runs inside this dedicated
project, so it cannot silently use the parent's different Torch resolution.
Sol repeated both offline lock checks, the inspector, VAST-worker and Apple
self-tests, Python compilation, Bash syntax, ShellCheck and whitespace checks.
The real dependency gate exits 2 before acquisition because the fixed upstream
has no license file, checkpoint terms and byte SHA remain unknown,
`librosa.filters.mel` closes over soxr/libsndfile, and the native/bundled wheel
notices remain unreviewed. No dependency sync, source/model acquisition,
conversion, Cargo run, parity or upload was performed.

The 2026-08-29 Sol review then independently re-ran the focused Python
compile/self-test, worker self-test, `bash -n`, ShellCheck and whitespace
checks for the corrected XY-Tokenizer and VibeVoice-ASR inspection routes.
Both passed those local staging checks. XY-Tokenizer now authenticates nine
fixed source-role Git objects plus the fixed source README's explicit
Apache-2.0 declaration (with the absence of a full license file recorded as
an evidence fact), while retaining the unreviewed topology as an explicit
fail-closed blocker. VibeVoice-ASR now authenticates the fixed Microsoft and
Transformers role objects, materializes the complete eight-shard HF tree into
an explicit `local_dir`, and records the missing `Qwen/Qwen2.5-7B` revision as
an unselected, not-downloaded dependency. Neither review executed a model,
established numerical parity, or changed the corresponding CPU/Metal rows.

The corrected SpeechT5-TTS chain is now accepted as fail-closed staging, not
as a model verdict. Its dedicated Python 3.12 project resolves exactly 36
package rows, fixes Microsoft SpeechT5-TTS and HiFi-GAN revisions and payloads,
fixes the historical public GGUF, and binds the official reference route to
Transformers 5.5.0. Exact lock/project bytes, all 36 version/source-keyed
dependency reviews and five ordered model/source rows are part of the approval
scope. The production preflight exits 2 at `annotated-doc@0.0.5`, the first
unresolved dependency review, before host/tool inspection, work or scratch
creation, dependency synchronization, model acquisition or Cargo. VAST records
the SHA-256 of the independently generated
`reference.json`; remote Apple replay requires that expected digest, verifies
it before the reference's embedded artifact hashes and scalar files, and then
requires exactly one named Cargo pass, one total Cargo result, and one complete
CPU and Metal sentinel. Duplicate, prefixed, suffixed and failure-shaped
evidence is rejected.
Sol repeated the offline lock check, gate and reference-dumper self-tests,
both worker self-tests, Python compilation, Bash syntax, ShellCheck and focused
whitespace checks, then directly confirmed the production gate's exit 2. No
dependency sync, model download, conversion, Cargo, VAST or Apple run occurred,
so both the CPU and Metal completion rows remain open.

Bark Small/Full is likewise accepted only as fail-closed staging. Its
dedicated Python 3.12 project resolves 35 package rows and fixes the two Suno
checkpoints, both historical public GGUFs, both model configurations, the
shared generation configuration and Transformers 5.5.0 source identity. The
canonical approval scope covers exact lock/project bytes, package and license
review rows, every artifact identity and an explicit `NO_UPLOAD` decision;
production rows and owner evidence remain unresolved. This preflight is the
first substantive worker operation and exits 2 before VAST probing, scratch,
sync, download or Cargo. VAST records reusable expected Small/Full reference
manifest hashes, while Apple authenticates those manifests and their exact
inner files before requiring one named Cargo pass and complete, singleton
CPU/Metal metric sentinels for each variant. Sol repeated the offline lock,
gate/dumper/worker/Apple tamper suites, Python compilation, Bash syntax,
ShellCheck and whitespace checks and directly proved the production worker
left its scratch path absent. No dependency sync, model download, Cargo, VAST
or Apple run occurred; the model verdicts therefore remain open.

A subsequent 2026-08-29 Sol review accepted three more corrected inspection
routes as staging only. VibeVoice-1.5B now fixes every Microsoft and
Transformers role object, the complete model and Qwen tokenizer Git/LFS
identities, the official `generate(inputs, ..., **kwargs)` boundary and its
greedy token path; its inspector/reference/worker self-tests, Python compile,
ShellCheck, Rust formatting and whitespace checks pass. Hibiki-2B now uses the
real doubled `moshi/moshi/...` Python paths, exact Hibiki/Moshi role objects and
separate exact license-object tables while accepting legitimate non-role
`100755` source files; the same focused local checks pass. VieNeu v3 Turbo now
fixes ten source roles including the pinned source lock and Apache-2.0 license,
records absent optional ONNX wrapper paths at the selected commit, and can
distinguish a fully authenticated collection from an inspection error while
retaining exact-topology, dependency-license, native runtime and parity
  blockers. Its focused local checks also pass. That inspection review itself
  generated no dedicated reference lock; VibeVoice's separately reviewed lock
  and still-blocked preflight are recorded above. No model was downloaded or
  executed, and none of these reviews supplies a CPU, Metal or
  numerical-parity verdict.

The corrected Parler-TTS worker is now accepted as fail-closed staging only.
Its dedicated Python 3.12 project resolves 27 package rows with the exact
`torch==2.11.0+cpu` / `torchaudio==2.11.0+cpu` pair and Transformers 4.46.1.
The current official TorchAudio 2.11 installation document states that its
stable ABI also supports later PyTorch versions, so the earlier 2.13/2.11 ABI
mismatch claim remains withdrawn; the matching 2.11 pair is retained here for
reproducibility, not because 2.13 is unsupported. The worker authenticates the
official Parler source commit directly and uses its integrated Transformers DAC
route without pulling the broad `descript-audiotools` training/evaluation
closure. Exact source, English, Multilingual, public-GGUF and DAC identities are
bound to lock/project bytes, versioned dependency reviews, four exact component
review rows and operator evidence. Placeholder spellings and missing, extra or
duplicate rows fail closed. Production exits 2 at the first unresolved
`certifi@2026.7.22` review before host probing, scratch, source/model acquisition,
sync or Cargo. VAST records the two expected reference-manifest hashes and a
reusable Apple verifier command; Apple verifies both manifests and payloads,
then requires exact singleton Cargo results and complete CPU/Metal sentinels.
Sol repeated the offline lock, gate/VAST/Apple self-tests, Python compilation,
Bash syntax, ShellCheck and focused whitespace checks and directly proved that
the blocked production worker leaves both requested work and scratch absent.
No source/model/DAC download, sync, Cargo, VAST, Apple run or upload occurred,
so the Parler CPU and Metal coverage rows remain open.

MOSS Audio Tokenizer v2 is now accepted as fail-closed staging only. Its
dedicated Linux/x86_64/Python 3.12 project replaces the unrelated 215-package
parity environment with 52 code-bound rows. The gate fixes the official commit,
all eight snapshot files by byte count and SHA-256, and the 2,094-tensor,
2,123,701,248-parameter, 8,494,804,992-byte f32 topology plus its 48-kHz,
stereo, 12-quantizer decode contract. Exact package reviews, three canonical
license rows and owner evidence remain unresolved, so production exits 2 at
`accelerate==1.10.1` before VAST probing, scratch, sync, download, merge,
conversion, CUDA reference or Cargo. The lock is explicitly limited to the
VAST target; the final offline lock check resolves all 52 rows successfully.
VAST now emits a repo-relative, shell-safe Apple command with expected
GGUF/reference hashes and a separate external-approval placeholder. Apple
runs the same lock/project/manifest gate before host, input, evidence or Cargo
work, authenticates the transferred regular non-symlink inputs, requires an
absent evidence directory canonically disjoint from the checkout, inputs and
approval, accepts the dumper's real four-column source-file rows, and requires
exactly one named Cargo pass, one result and one complete CPU/Metal measurement
sentinel. Sol repeated the 52-package offline lock, gate and both worker
self-tests, Python compilation, Bash syntax, ShellCheck and whitespace checks,
and directly proved the blocked VAST and Apple production shapes leave work,
scratch and evidence absent. No dependency sync, model download, merge,
conversion, Cargo, VAST, Apple run or
upload occurred; the numerical state deliberately remains
`MEASURED_NOT_GATED` and both coverage rows stay open.

NeuTTS Air plus Distill NeuCodec is also accepted only as fail-closed staging.
Its dedicated Python 3.12 project binds the exact 39-package lock/project and
canonical dependency graph. The gate keeps the 1,495,883,328-byte public
NeuTTS GGUF and 1,025,417,504-byte Distill NeuCodec companion as distinct fixed
artifacts, fixes the gated upstream revision and exact seven-file name set, and
fixes `neuttsair/neutts.py` at 9,035 bytes and its source commit/SHA-256. The
seven upstream byte/SHA identities, upstream license and source license are not
known from authenticated primary evidence; they remain code-level `null`
values and independently make the gate impossible to approve. Filling review
or sign-off rows alone cannot bypass that blocker. Production exits 2 before
token, host or tooling checks and leaves requested work and scratch absent.
VAST emits a reusable Apple command containing expected hashes for both GGUFs
and the reference manifest. Apple authenticates all three inputs, requires the
exact six-file regular non-symlink reference tree and its inner hashes, and
accepts only singleton exact Cargo results plus full-line CPU/Metal parity and
composition markers. Sol repeated the 39-package offline lock, gate/dumper/VAST
and Apple self-tests, Python compilation, Bash syntax, ShellCheck, whitespace
checks and the direct production no-work/no-scratch proof. No sync, source or
model acquisition, Cargo, VAST, Apple run or upload occurred; CPU and Metal
completion therefore remain open.

MOSS Audio Tokenizer Nano is now accepted only as fail-closed staging. Its
dedicated Linux/x86_64/Python 3.12 project binds the exact 52-package
lock/project and canonical dependency graph, the fixed
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano@6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`
revision and its exact seven-file payload name set. Authenticated byte/SHA
identities for those seven files, the official Transformers compatibility
route, remote-code source identities, quantizer/decoder shape contract,
package/license reviews and owner approval remain unresolved code-level
blockers. Review/sign-off data alone cannot bypass the null identity fields.
The VAST worker gates before host, scratch, cache, synchronization, download,
conversion or Cargo, authenticates an exact regular non-symlink snapshot and
requires the official reference to contain one pinned source row, stable
`transformers_modules/...` source paths, exact runtime versions, the fixed
32-code packet, one quantizer tap, a nonzero contiguous decoder sequence and
the exact ordered decoder shape list. It emits a shell-safe portable Apple
command carrying the generated GGUF/reference hashes and a separate external
approval placeholder. Apple runs the same gate before host, input, evidence or
Cargo work, repeats that reference validation and accepts only singleton named
Cargo results plus complete CPU/Metal measurement sentinels. Sol repeated the
52-package offline lock, Nano gate/dumper, VAST and Apple self-tests, the shared
v2 VAST/Apple regression self-tests, Python compilation, Bash syntax,
ShellCheck and whitespace checks. Direct VAST and Apple production shapes exit
2 at the unresolved Transformers route with work, scratch, cache and evidence
absent. No dependency sync, source/model acquisition, conversion, Cargo, VAST,
Apple run or upload occurred; its CPU and Metal rows remain open.

MossFormer2-SS-16K is now accepted only as fail-closed staging. Its dedicated
Python 3.12 project binds the exact 48-package lock/project and canonical
dependency graph, the 223,058,240-byte public GGUF, the 670,353,271-byte
upstream checkpoint and the fixed ClearerVoice-Studio source revision. The
upstream checkpoint license/source-role identity, source license and the
byte/SHA identities of the exact six required source files remain code-level
unresolved values; operator review or sign-off alone cannot bypass them. The
VAST worker gates before host, scratch, cache, synchronization or network
activity, verifies the checked-out source before import/preparation and
strictly validates the generated CUDA reference: exact manifest schema,
runtime versions, six regular non-symlink float32-le payloads, shapes, byte
counts and inner hashes. It emits a shell-safe portable Apple command carrying
the fixed GGUF hash and generated reference-manifest hash. Apple authenticates
those transferred inputs before Cargo, enables the Metal feature explicitly
and accepts only singleton exact Cargo results plus complete CPU/Metal
measurement sentinels. Sol repeated the 48-package offline lock check,
gate/dumper/VAST/Apple self-tests, Python compilation, Bash syntax, ShellCheck,
whitespace checks and a direct production proof: exit 2 at the unresolved
upstream license blocker with work, scratch and cache all absent. No dependency
sync, source/model acquisition, checkpoint preparation, Cargo, VAST, Apple run
or upload occurred; its CPU and Metal rows therefore remain open.

MOSS-Audio 4B/8B is now accepted only as fail-closed staging. Its dedicated
Python 3.12 project binds the exact 50-package closure, lock digest
`f26e7504e980c5a62fdcb1bd2ed1d9726da09c839cb9f251412b4d4145fbd59f`,
project digest
`d321bfae5af886eb9ef0fc2fd3696c425c77c5c247353957e05317ab1efb43d0`,
official source revision `5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883`, and the
fixed 4B/8B model revisions. Exact source/LICENSE bytes and Git blobs, model
LICENSE identities, and complete index/shard identities remain unresolved
code-level blockers that operator approval cannot bypass. The VAST worker now
runs this gate before host, scratch, cache, synchronization, network or Cargo
work, verifies exact regular source/snapshot trees, and emits a portable quoted
Apple command with caller-bound model and reference hashes. The Apple worker
requires exact reference keys/files/inner hashes and singleton Cargo result and
parity sentinels. Sol repeated the gate, VAST and Apple self-tests, offline
50-package lock check, Python/shell syntax, ShellCheck and whitespace checks.
The production path exited 2 on the unresolved `accelerate@1.12.0` review
without creating scratch, cache or work data. No model acquisition, conversion,
Cargo, VAST, Apple or upload work ran, and CPU/Metal completion remains open.

An execution-order scan initially found eleven validation workers that
synchronized dependencies without a visible pre-sync closure/license gate.
Bark, Parler-TTS, MOSS Audio Tokenizer v2, NeuTTS Air, MOSS Audio Tokenizer
Nano and MossFormer2 are now protected by the reviewed fail-closed chains
above. SpeechBrain Lang-ID and YuE XCodec Mini now also have exact lock,
project, manifest and external-evidence gates before any work; both produce
portable Apple commands carrying explicit approval placeholders. MOSS-TTS
Local composite is now protected by the dual-gate chain recorded above. The
remaining execution-order item is WeSpeaker. ChatTTS,
CSM, FireRed-ASR-LLM and Higgs Audio were initial text-scan false positives:
each invokes an existing dependency gate under a different function name
before synchronization. This queue is safety hardening, not a count of new
runtime implementations.

The focused repository gates are currently green for zero runtime dependencies,
forbidden symbols and bound-arch registry coverage. The converter/binder
handshake gate found eleven bookkeeping-visible gaps after the new native
binders landed. Four `NO_READER` rows (`granite_speech`, `htdemucs_multi`,
`sgmse`, `xtts`) are stale because readers now exist. Sortformer is a deliberate
inspection-only binder with no converter that can emit `sortformer`, and the
inspection-only Canary-Qwen, CosyVoice2, CSM, Irodori, Kyutai STT and
Sortformer converters intentionally cannot stamp their binders' required
metadata until the fixed VAST identities/composite contracts are authenticated.
Those seven gaps must be recorded in the double-sided `NO_CONVERTER`/
`NO_STAMP` ledgers; stamping guessed values or reviving arbitrary-input
conversion would violate the fail-closed boundary. The Luna-owned ledger-only
correction removed the four stale rows, records Sortformer in `NO_CONVERTER`
and records all six inspection-only metadata groups in `NO_STAMP`. Sol repeated
the 49-case self-test and production gate: leg (a) now accounts for 107
converter arches, leg (b) for 134 binder arches with the one declared
Sortformer gap, and leg (d) for 1,044 required metadata reads with eight
declared no-stamp groups. Bash syntax and scoped whitespace checks also pass.

The live Hugging Face audit was repeated on 2026-08-29 after those staging
reviews. It reported the same 194 public repositories, 193 GGUF-bearing
repositories and 198 GGUF files, with CPU `full=128`, `partial=45`,
`no-runtime-binder=20`, `not-artifact=1` and Metal `full=128`,
`blocked-by-cpu=65`, `not-artifact=1`. The 65-row execution target therefore
has not drifted; no row was closed by inspection-only work.

## Wave 1: routed partial ASR and audio understanding

- [ ] `vokra/canary-qwen-2.5b`
- [ ] `vokra/firered-asr-aed-l`
- [ ] `vokra/kyutai-stt-2.6b-en`
- [ ] `vokra/omniasr-ctc-1b`
- [ ] `vokra/sber-gigaam-multilingual`
- [ ] `vokra/sber-gigaam-v3`
- [ ] `vokra/qwen3-asr-0.6b`
- [ ] `vokra/qwen3-asr-1.7b`
- [ ] `vokra/reazonspeech-nemo-v2`
- [ ] `vokra/canary-1b-flash`
- [ ] `vokra/canary-1b-v2`
- [ ] `vokra/ultravox-v0-5-llama-3-2-1b`

Qwen3-ASR, ReazonSpeech, Canary and Ultravox have complete source routes but
their live public artifacts remain incomplete or pre-contract. Their first
work item is real VAST evidence and a replacement dry-run, not a second model
implementation. The other rows retain genuine forward/configuration gaps.

## Wave 2: routed partial TTS and speech generation

- [ ] `vokra/chatterbox-multilingual-v3`
- [ ] `vokra/chatterbox-nano-v1`
- [ ] `vokra/chatterbox-turbo-v1`
- [ ] `vokra/chattts`
- [ ] `vokra/cosyvoice2-0.5b`
- [ ] `vokra/fun-cosyvoice3-0.5b-2512`
- [ ] `vokra/csm-1b`
- [ ] `vokra/dia-1.6b`
- [ ] `vokra/irodori-tts-500m-v3`
- [ ] `vokra/moss-tts-local-transformer-v1.5`
- [ ] `vokra/qwen3-tts-12hz-0.6b-base`
- [ ] `vokra/qwen3-tts-12hz-0.6b-customvoice`
- [ ] `vokra/qwen3-tts-12hz-1.7b-base`
- [ ] `vokra/qwen3-tts-12hz-1.7b-customvoice`
- [ ] `vokra/vibevoice-1.5b`
- [ ] `vokra/voxcpm-0.5b`
- [ ] `vokra/zonos-v0.1-transformer`

The Qwen3-TTS code route is source-complete, but all four historical main
artifacts predate the strict contract and the required 12-Hz companion is not
public. MOSS Local also has a distinct 48-kHz stereo companion boundary.

## Wave 3: routed partial music, codec, separation and classification

- [ ] `vokra/audiogen-medium`
- [ ] `vokra/audioldm2`
- [ ] `vokra/audioldm2-large`
- [ ] `vokra/clap-htsat-fused`
- [ ] `vokra/conv-tasnet-libri1mix`
- [ ] `vokra/lang-id-voxlingua107`
- [ ] `vokra/mms-1b-all-base`
- [ ] `vokra/moss-audio-tokenizer-nano`
- [ ] `vokra/nsnet2`
- [ ] `vokra/rmvpe`
- [ ] `vokra/sbv2-v2-jp-extra-base`
- [ ] `vokra/sortformer-diar-4spk-v1`
- [ ] `vokra/speechbrain-spkrec-ecapa-voxceleb`
- [ ] `vokra/voice-gender-classifier`
- [ ] `vokra/wespeaker`
- [ ] `vokra/yue-xcodec-mini`

Several rows already have a complete native code route but fail on exact live
bytes or provenance. Preserve that fail-closed distinction: a code path is not
permission to accept a corrupt, incomplete or incorrectly licensed artifact.

`mms-1b-all-base` is not an artifact-only repair despite reusing the Wav2Vec2
family. The current shared converter's default provenance path still stamps
Apache-2.0 for the MMS variant, while the audited MMS weights are
CC-BY-NC-4.0, and the runtime deliberately rejects every `mms-1b-all` model id.
Completion requires a dedicated fail-closed backbone-plus-language-adapter
contract, vocabulary/label identity and noncommercial policy; the 8.9-MB
public adapter must never be presented as the 1B backbone.

Conv-TasNet is not a native-CPU implementation gap. The 2026-08-24 VAST run
already authenticated the 345-tensor Asteroid topology and measured the
encoder, bottleneck, mask and terminal waveform against the independent
Asteroid 0.7.0 oracle. The remaining branch work is a no-upload worker that
reproduces that evidence, a real Apple CPU/Metal measurement, and a strict
runtime license-policy entry point. The corrected topology cannot replace the
public 16/8 artifact while the upstream CC-BY-SA/WHAM license declarations
remain contradictory; default provenance stays `unknown` and publication
stays blocked.

The immutable upstream payload identity is now explicit for that staging
work. At revision `bb8a876bc157b5cf3c405994accb798c49146016`, the sole model
payload is `pytorch_model.bin`, 20,130,704 bytes, with Git-LFS SHA-256
`dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0`.
The same fixed card records kernel 32, stride 16, 512 filters, eight blocks,
three repeats and one 16-kHz output stream. Its YAML front matter declares
CC-BY-SA-4.0, while the body declares the resulting work CC-BY-SA-3.0 and the
WHAM-derived material CC-BY-NC-4.0 Research-only. These are evidence inputs to
the external approval gate, not a basis for selecting one convenient license.

The independent oracle remains the official `asteroid==0.7.0` release. Its
PyPI wheel is SHA-256
`ea97a24901d9d9851b4a594171bd7c6dd900fee2c132b9ce045aa09926d489c7`
and its source distribution is SHA-256
`0326f28c5342495cb08ba0520efd0e21e39435dfd78854837fdd5a6c9c9ca410`.
The release metadata requires the real Asteroid dependency family, including
Asteroid Filterbanks, SciPy, SoundFile, Hugging Face Hub, PyYAML, pandas,
PyTorch Lightning, TorchMetrics, Torchaudio, pb-bss-eval, torch-stoi,
torch-optimizer and Julius. Do not replace that independent oracle with a
handwritten mirror. The dedicated Python 3.12 Linux x86_64 lock must instead
route Torch/Torchaudio to the official CPU index and contain no CUDA, NVIDIA
or Triton packages; every resulting package stays pending license/native
payload review until authenticated evidence resolves it.

The current legacy Conv-TasNet workers are not that final chain. Root review
found that the VAST script points at the broad `tools/parity` environment and
adds Asteroid dynamically with `--with asteroid==0.7.0` even though a dedicated
`tools/parity/conv_tasnet` lock exists. It also accepts a pre-existing empty
work directory without an authenticated approval argument, while the Apple
script accepts a pre-existing empty evidence directory and likewise has no
approval binding. Replace those paths with the dedicated frozen closure,
strict external license/policy approval, gate-first execution and absent,
canonical-disjoint work/evidence contracts before any replay is accepted.

The cross-gate duplicate-JSON audit also reopened the SpeechBrain Lang-ID and
YuE XCodec Mini staging chains: SpeechBrain currently uses ordinary
`json.loads` for both manifest and approval evidence, and YuE's strict loader
raises `ValueError` for duplicate approval keys outside its caught exception
set. Both must reject duplicate-key manifest/evidence documents through a
controlled exit-2 path with focused regression tests; their production gates
remain blocked in the meantime.

The same audit reopened both MOSS Audio Tokenizer staging chains. The v2 and
Nano license gates currently parse manifest and approval JSON with ordinary
last-key-wins `json.loads`; Nano's snapshot-manifest verifier does the same.
Their VAST workers also still obtain approval indirectly, accept pre-existing
empty work directories and do not apply the canonical checkout/approval
overlap contract already used by their Apple workers. Before either chain is
accepted again, require explicit `--approval-evidence`, strict duplicate-key
JSON rejection at every gate input, a gate-first production order, and an
absent/non-symlink/canonically-disjoint VAST work directory with no pre-gate
scratch or cache creation.

The exact-lock audit also reopens the MOSS Local composite gate narrowly. Its
tracked lock digest still prevents an unreviewed production lock from passing,
but the structural parser admits an empty `upload-time`, accepts any HTTPS
registry host and treats several actual package-row fields as optional instead
of enforcing the committed row variants. That is weaker than the exact
artifact contract now applied to the other staging gates and makes its
self-test an incomplete regression oracle. Before the Local chain is accepted
again, require exact top-level/package/dependency/source/virtual metadata row
schemas, fixed registry hosts, nonempty `upload-time`, positive non-boolean
artifact sizes, and tamper cases for every missing/extra field. This is a gate
hardening task only; it does not change the already separate Local main/codec
runtime boundary or claim model parity.

### 2026-08-30 cross-gate hardening closure

The reopened worker-contract findings above were re-audited at clean commit
`c3a653e4`. This closes only the gate-hardening findings; it does not promote
any public model row or infer a VAST/Apple parity result.

- Conv-TasNet now uses the dedicated frozen project, binds one explicit
  external approval document to the exact manifest/lock/project scope, rejects
  duplicate or tampered approval/reference JSON, and validates absent,
  non-symlink, canonically disjoint input/work/evidence/fixture paths before
  any cache, download or output creation. The approval digest is rechecked
  after validation. Its unresolved CC-BY-SA-3.0/4.0 versus WHAM
  CC-BY-NC-4.0 policy and Asteroid dependency review remain fail-closed, so
  the production license gate still exits 2.
- MOSS Audio Tokenizer v2 and Nano now reject duplicate keys in every staged
  manifest/approval/snapshot index path and assert controlled exit 2 with no
  work/evidence side effects for missing, duplicate, relative, symlinked or
  overlapping inputs. Explicit approval, gate-first ordering and `NO_UPLOAD`
  remain mandatory. Nano's custom-code identity and numerical bound remain
  unresolved; v2 and Nano still require real VAST and Apple runs.
- SpeechBrain Lang-ID, YuE XCodec Mini and MOSS TTS Local were found to already
  contain the strict duplicate-JSON, controlled-exit, exact-lock and disjoint
  path contracts described by the reopened findings. Manager-repeated Python
  and shell self-tests passed; no source edit was needed. MOSS Local still has
  its separate 48-kHz stereo companion and real-weight boundaries.

The manager repeated Bash syntax, ShellCheck, the focused VAST/Apple worker
self-tests, the Conv-TasNet and MOSS license-gate self-tests, and
`git diff --check` without running a model, downloading weights, synchronizing
a parity environment or invoking Cargo. The implementation commits are
`358fe1b2` (MOSS tokenizer staging gates) and `c3a653e4` (Conv-TasNet approval
binding).

SBV2 likewise has prior real CPU evidence rather than an unproven forward.
The 2026-08-18 VAST records cover the JP-Extra main model with JA/EN BERT and
the optional ZH BERT leg through the final waveform without widening bounds.
The pre-documentation implementation baseline stages one explicit backend selector and preflights the
complete learned-op set across the text encoder, JA/EN/ZH BERT sidecars,
bridge, speaker/style projections, duration predictor, flow inverse and
conditioned HiFi-GAN decoder. This closes the former source-level Mac GPU
routing gap, but it is not a Metal verdict: the real staged fixture must still
replay on disposable Apple Silicon and its CPU-relative measurements remain
`MEASURED_NOT_GATED` until reviewed. The historical public main GGUF and the
production Japanese G2P boundary remain separate artifact/runtime blockers.

## Wave 4: no complete live-artifact runtime binder

- [ ] `vokra/ace-step-1.5`
- [ ] `vokra/baichuan-audio`
- [ ] `vokra/bicodec`
- [ ] `vokra/granite-speech-4.1-2b`
- [ ] `vokra/hibiki-2b`
- [ ] `vokra/htdemucs-multi`
- [ ] `vokra/kimi-audio`
- [ ] `vokra/kyutai-tts-1.6b-en-fr`
- [ ] `vokra/moss-audio-4b-instruct`
- [ ] `vokra/moss-audio-8b-instruct`
- [ ] `vokra/owsm-v4-medium-1b`
- [ ] `vokra/qwen2-5-omni-7b`
- [ ] `vokra/qwen2-audio-7b-instruct`
- [ ] `vokra/sgmse-voicebank`
- [ ] `vokra/step-audio2-mini`
- [ ] `vokra/vibevoice-asr`
- [ ] `vokra/vibevoice-realtime-0.5b`
- [ ] `vokra/vieneu-tts-v3-turbo`
- [ ] `vokra/xtts-v2`
- [ ] `vokra/xy-tokenizer`

`bicodec`, `htdemucs-multi`, both historical MOSS-Audio files and
`xy-tokenizer` have artifact-specific failures in addition to missing runtime
completion. The remaining fifteen are generic no-binder rows in the live
audit and need model-family implementation work.

## Non-artifact repository

- [ ] `vokra/seamless-m4t-v2-large` receives a real gated GGUF, or is withdrawn
      through a separately authorized publication action.

## Cross-cutting backend work

- [ ] Native BF16 compute replaces the current upcast-to-F32 shim with parity
      and ISA/backend-specific evidence.
  The first bounded foundation is now staged: GGUF and safetensors expose
  dtype/shape/length-authenticated little-endian BF16 `u16` payloads without
  materializing F32 weights, including a windowed safetensors reader, and the
  CPU backend accepts those raw bits through scalar, AVX512-BF16 and
  Neon-BF16 GEMM paths with F32 accumulation, checked tail padding and explicit
  unsupported-ISA errors. A second CPU-only seam now multiplies unrounded F32
  activations by caller-owned raw BF16 weights. It widens at most one `k x 8`
  weight panel, reuses that panel across `8 x 8` output tiles, dispatches the
  ordinary F32 SIMD kernels, and explicitly rejects both RVV generations until
  they gain a dedicated implementation. Sol repeated `vokra-core`'s 591
  library tests, `vokra-backend-cpu`'s 145 library tests and the core
  no-default-features check successfully; the focused mixed test selected
  `neon-dotprod` on this Darwin arm64 host and exactly matched the widened-F32
  oracle. Ultravox's separately acquired Llama companion is the first staged
  consumer: on CPU its dense layer and tied-vocabulary-head weights remain
  mmap-backed BF16 and are transposed only in bounded eight-column panels;
  GPU routes currently retain the prior full-F32 materialization path. The
  Metal backend foundation now also owns a distinct raw-BF16 device tensor and
  an F32-activation/raw-BF16-weight GEMM. The shader reconstructs each BF16
  value by widening its exact `u16` bits, accumulates in F32, rejects shape,
  ownership and dimension failures before dispatch, and has no CPU fallback.
  The shared host/resident validator also rejects any `m*k`, `k*n` or `m*n`
  product above `u32::MAX`, matching the MSL kernel's 32-bit row-major index
  arithmetic even when each individual dimension and the host `usize` product
  are otherwise representable. Pure boundary tests accept an exact
  `u32::MAX` product, reject the first overflow on each axis and preserve the
  zero-`k` reduction contract; Sol repeated both focused unit tests.
  Sol exercised the corrected special-value oracle, zero-`k` semantics,
  cross-context rejection and the complete real Apple-GPU backend suite: all
  48 tests passed with no skip, including the new autorelease-pooled
  host-in/host-out wrapper. The accepted Ultravox Metal consumer now keeps its
  separately acquired companion's dense layer and vocabulary-head weights as
  mmap-backed BF16, transposes one complete logical matrix as raw `u16`, and
  sends that matrix through one contiguous Metal submission without an F32
  weight mirror or silent CPU fallback. The CPU-only strided panel seam remains
  bounded and Metal explicitly rejects that seam instead of hiding per-row
  allocation/submission. This backend and source wiring does not establish
  model support: the consumer has not been compiled or exercised locally
  because `vokra-models` verification is VAST-only under the project memory
  guard. The box stays open: no real
  BF16 checkpoint, AVX512-BF16 host or compatible Arm BF16 silicon has supplied
  independent parity evidence, and the Ultravox consumer still needs its VAST
  compile/parity and remote Apple real-weight runs.
- [ ] Complete HiFTNet GPU generator path; existing Metal primitives alone do
      not establish whole-generator residency.
  The backend foundation and a complete resident graph are now staged, but do
  not close this box. Metal now
  exposes context-owned resident tensors plus dilation-aware Conv1d,
  PyTorch-layout ConvTranspose1d, reflect/replicate padding, LeakyReLU, tanh,
  Snake/SnakeBeta, anti-aliased upsample and downsample, scale and clamp device
  operations. HiFT-specific ELU, absolute and tanh linear projections,
  nearest-neighbour expansion, channel-major deterministic SineGen, centered
  periodic-Hann STFT, logits-to-complex reconstruction and unclamped iSTFT are
  also device-resident; the configured terminal audio limit is applied only
  after iSTFT. Every public
  resident operation rejects tensors from another context before dispatch,
  dimension/index conversions fail closed, all new pipeline objects are
  released, and an explicit readback counter can prove that only the final
  caller-selected tensor crosses D2H. `HiFTGenerator::forward_with_resident_ops`
  now keeps all five F0 convolutions, NSF source construction, both learned
  source-fusion stages, every ResBlock/MRF branch, terminal complex spectrum,
  iSTFT and configured clamp on that resident seam. A nonzero instrumented
  scalar adapter matches the existing scalar graph within `1e-5`, proves one
  final readback and proves zero readbacks after an injected graph error.
  CosyVoice2's injected `HiFTChain` now selects the scalar route for CPU and a
  dedicated Apple Metal adapter for the same whole graph; unsupported targets
  and backends fail explicitly without a CPU fallback.

  Sol repeated the Apple-target backend test build, the `vokra-ops` library
  check, formatting and focused whitespace checks. An explicitly approved
  direct run on the maintainer Apple-Silicon Mac compiled the new MSL and began
  all 44 backend integration tests. The F0 primitives, channel-major source
  mixer, odd-FFT STFT oracle and one-final-readback chain test passed, but the
  process then received `SIGSEGV`. A serial exact-test replay isolated the
  crash to the empty-input/even-FFT STFT semantics test; the odd-FFT STFT test
  passed alone. The accepted correction preserves a zero logical tensor length
  while allocating a four-byte placeholder Metal buffer instead of passing an
  empty host pointer to `newBufferWithBytes`; output byte-size multiplication
  is also checked. Sol then repeated the formerly crashing exact test, the
  odd-FFT iSTFT exact test and the complete parallel suite on the real Apple
  GPU: all 44 tests passed with no skip. The operations remain one synchronous
  commit/wait each. The model-crate adapter has not compiled on VAST, and no
  real-weight CPU/Metal oracle or performance verdict exists yet.
- [ ] Complete BigVGAN GPU generator path; distinguish correctness coverage
      from fully device-resident/fused execution.
  The complete source topology is now staged behind a backend-independent
  resident trait: conv-pre, every ConvTranspose stage, Snake/SnakeBeta with
  stored alias-free up/down filters, both convolutions and residual additions
  in every AMP block, MRF branch averaging, post activation/conv and terminal
  tanh or clamp remain resident. The Apple Metal adapter maps that trait to the
  context-owned primitives and the model dispatcher requires exactly one final
  D2H readback; unsupported feature/target combinations return an explicit
  error without CPU fallback. Sol required and reviewed a nonzero structural
  oracle covering two stages, two MRF kernels, two dilations, SnakeBeta and the
  even-tap filters; it matches the scalar graph, reads back once, and an
  injected mid-graph failure reads back zero times. `vokra-ops` library check,
  scoped formatting and whitespace checks passed before concurrent HiFT edits
  began. A dedicated no-upload validation chain now fixes NVIDIA/BigVGAN at
  source commit `7d2b454564a6c7d014227f635b7423881f14bdac`, the
  `nvidia/bigvgan_base_24khz_100band` snapshot at
  `0f6305d0e010eaafdbf649978f46c3b5af099343`, and the exact checkpoint,
  config and both MIT-license payload digests. It uses a dedicated CPU-only
  Python 3.12 lock, safe tensor-only checkpoint loading, a VAST-generated
  independent upstream reference with no committed-fixture fallback, named
  one-test CPU/Metal sentinels and mandatory artifact/reference hashes for the
  remote Apple replay. The pre-sync license gate binds the complete lock rows,
  review rows and fixed artifact identities and deliberately exits 2 because
  the native/bundled dependency closure and fixed human sign-off are not yet
  approved. Sol repeated the gate tamper suite, VAST-worker self-test, Apple
  worker self-test, shell syntax check and focused whitespace check
  successfully. The box remains open because the blocking review has not been
  resolved, the model-crate Metal adapter has not compiled on VAST, and neither
  the fixed real artifact nor its independent upstream fixture has run through
  this resident route on Apple hardware.
- [ ] Keep the live audit invariant at zero CPU-complete/Metal-unsupported
      repositories after every wave.

## VAST credential incident

The first local `vastai` status probe on this branch failed DNS resolution and
the third-party CLI included its API credential in an exception request URL.
The value is not recorded here. The owner confirmed that credential was
rotated before VAST reuse.

A later instance-list response exposed a `jupyter_token` JSON field because
the first wrapper version covered URL query values only. The unrelated running
instance was not touched; the status response exposed only its SSH port mapping
at that time. Treat the emitted Jupyter credential as compromised even though
no Jupyter port was advertised. The wrapper now redacts credential-bearing URL
queries, JSON fields and `key=value` records, including upper- and lower-case
instance/Jupyter/container variants. Its network-free regression passes, and
a live status probe showed `[REDACTED]` instead of the credential while
preserving the real command exit status.

The first fresh-instance creation attempt then returned `instance_api_key` in
a single-quoted Python-dict diagnostic, a format not covered by the second
wrapper version. The value is not recorded here. Instance `49106108` was never
used: no checkout, bundle, model, conversion or validation reached it, and it
was immediately destroyed; a subsequent list probe confirmed its removal.
The wrapper now recognizes `instance_api_key` and all credential keys in both
quoted Python dictionaries and unquoted dict-like fields. Hermetic stdout and
stderr regressions cover those formats before any replacement instance is
created.

## Final branch exit gates

- [ ] Live audit: CPU `partial=0`, `no-runtime-binder=0`, Metal
      `blocked-by-cpu=0`, and `cpu-only=0`.
- [ ] Every GGUF repository has a named real-file CPU verdict and independent
      numerical/output evidence.
- [ ] Every supported Metal checkpoint has an Apple-hardware verdict.
- [ ] `cargo fmt --all -- --check`, `git diff --check`, all applicable
      `scripts/check-*.sh` gates, workspace tests, all-target Clippy, license
      audit and advisory checks pass at the exact branch HEAD; heavy Cargo runs
      are VAST evidence.
- [ ] No VAST instance or retained model artifact remains unintentionally.
- [ ] No upload was performed without exact-repository authorization.
