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

The branch has advanced through `1ce957df` locally. The PR remote remains at
`559b8b36`; all checks at that remote commit are green and GitHub reports the
PR mergeable. Do not merge it as the final Mac-coverage change yet: the local
audit and BF16 commits have not been pushed or checked by GitHub.

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

The remaining factual dependency-license cases were checked against primary
release sources and must stay fail-closed:

- `gradio-client==2.5.0` has official annotated tag
  `gradio_client@2.5.0` at commit
  `43f5de68579919b0632ceb6107a99c629483ea2f`, whose project metadata and root
  license both say Apache-2.0. The tag was created after the PyPI artifacts,
  so an exact artifact-to-tag binding still needs to be recorded before this
  can replace the absent sdist license file.
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

## Cross-cutting implementation before the final Apple run

These tasks affect multiple model rows and must not be mistaken for Scaleway
work:

- **Native BF16 compute:** replace the remaining upcast-to-F32 shim; validate a
  real BF16 checkpoint plus independent AVX512-BF16 and Arm-BF16 parity. The
  raw-BF16 Metal foundation exists, but that does not close the full task.
- **HiFTNet full GPU generator:** finish and bind the complete resident graph,
  VAST-compile the `vokra-models` adapter, and obtain a real-weight CPU oracle
  and performance evidence.
- **BigVGAN full GPU path:** finish the dependency/native graph closure and
  owner sign-off, VAST-compile the model adapter, then run a fixed real artifact
  against an independent reference.
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
| `vokra/voice-gender-classifier` | Replace the artifact incorrectly stamped as ECAPA with the authenticated classifier identity and contract. |
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
| `vokra/cosyvoice2-0.5b` | Find an exact allowed source/reference route: the current upstream closure imports forbidden `soxr`; then finish LLM, flow, codec and vocoder composition. |
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
| `vokra/sgmse-voicebank` | Finish exact NCSN++ tensor-role mapping, score model and CPU parity; inspection/reference construction is not runtime parity. |
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
