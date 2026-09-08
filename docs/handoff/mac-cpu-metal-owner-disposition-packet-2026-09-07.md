# Mac CPU / Metal owner disposition packet (2026-09-07)

Updated with owner-independent evidence on 2026-09-08.

The closing model-free batch was verified on a clean VAST checkout at exact
head `80c17e290cc163d639d88550ffaca2187f2870fb`. Workspace tests, all-target and
all-feature Clippy with warnings denied, deny and audit were green; the
preflight and full-run log SHA-256 values are
`af90a3890c757533636ed594aaed18f5300dac8575e3756ebb8c4eb448c2144e` and
`a746dd9aca6fc3615735356c464e4de6e6f034b64d4ddebf7136d1d73d7130b1`.

The subsequent MOSS Nano, FireRed and CLAP binding batch was verified at clean
exact head `504858bcfe4f7809090ef6b25f5105e40b42c509`. The workspace result was
8,008 passed, zero failed and 100 explicitly ignored tests across 322 suites;
Clippy, deny, audit and the focused fail-closed gates were green. This advances
review evidence only and does not record an owner approval.

## Purpose and decision boundary

This packet records the owner/legal decisions that are still required before
the affected model families may advance from `APPROVAL_BLOCKED` to
`VAST_READY`. It is evidence for review, not approval. It does not authorize a
model download, model execution, Hugging Face upload, public-repository
withdrawal or Scaleway allocation.

An approval applies only to the exact scope hash shown below. A changed source
revision, model identity, lockfile, package row, native payload or license
evidence invalidates that decision and requires a new scope. A blank or
ambiguous decision remains fail-closed.

The owner's uncommitted
`tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` is outside
this packet. Its contents were not inspected or modified while preparing this
record.

## Hash-bound scopes ready for an owner decision

| Family | Manifest | Approval scope SHA-256 | Remaining decision |
|---|---|---|---|
| Qwen3-ASR 0.6B / 1.7B | `tools/parity/qwen3_asr/license_gate_manifest.json` | `da581832351b223b890814c0bf45ba036174da24dd7fd47a58236c1dde33ced1` | Review two model-license rows and 40 package rows; approve or withhold source/model/operator execution. |
| Qwen3-TTS four variants | `tools/parity/qwen3_tts/license_gate_manifest.json` | `46662c9a1a1135c37a4dc00583c1a637f72aa745ca10f2172a38f77df9d1b1da` | The exact-head no-checkpoint API smoke is green for all four variants. Resolve the remaining component/dependency owner reviews; no model was loaded and publication remains `NO_UPLOAD`. |
| SpeechT5-TTS + HiFi-GAN | `tools/parity/speecht5_tts/license_gate_manifest.json` | `99116b392c560ec40c574589305492f35d9d30e8e2f44a9c03392885c77e85ba` | Dependency/model rows are reviewed; operator approval and the still-unverified API smoke must close before execution. |
| MOSS Audio 4B / 8B | `tools/parity/moss_audio/license_gate_manifest.json` | `08eeab246dac53187c683cfed54e07f4a64a15abd119f3c4cc186bd23b88e69e` | The exact-head no-checkpoint API smoke is green and the fixed index/shard identities are bound for both variants. The source tree and model repositories have no license file at their fixed revisions; cardData `apache-2.0` remains provenance only. Resolve source/model SPDX, dependency rows and owner/operator decisions; no payload was acquired or loaded. |
| Ultravox + Meta Llama companion | `tools/parity/ultravox/license_gate_manifest.json` | `35e73acdfdffa729464400a11cdc2f890b216dc59476a79acebd24bfe8ae555b` | Decide whether to accept the gated Meta conditional-license companion after its payload hash and Python closure are complete. |
| WeSpeaker corrected replacement | `tools/parity/wespeaker/license_gate_manifest.json` | `0133cb13d4869903f89d6bbcaee9e784a69cf158804ae76bf4a52c7a0ca3efd0` | Review all five pending source/checkpoint/replacement rows and approve replacement or withholding. |
| YuE XCodec Mini | `tools/parity/yue_xcodec_mini/license_gate_manifest.json` | `6f8378213db1ef19924c42cb76a194ed013093c048aba910808eb14e2dcef262` | Resolve the missing source license, mixed MIT/CC-BY-NC RepCodec scope, public-artifact/weight rows and package review. |

The presence of a scope hash means only that the proposed decision is
immutable. None of these rows is owner-approved by this document.

## Scopes that are not yet complete enough to sign

| Family or issue | Missing evidence before a scope can be signed |
|---|---|
| Parler-TTS English / Multilingual | Approval-scope hash and remaining package review. Fixed source/model/DAC/GGUF identities alone are insufficient. |
| BigVGAN | One combined approval-scope hash over the fixed source/model/checkpoint/config, Linux and Darwin package/native-payload closures and license payload evidence. |
| MOSS Audio Tokenizer Nano | The final CPU-only 37-row locked closure and exact AutoConfig/meta-device source/API/tap route are now hash-bound. All 37 package rows still require owner review, the approval digest is absent, and no real-weight runtime or parity was executed. |
| MOSS-TTS Local | Dependency-license review, complete composite PCM execution boundary and a scope hash over the 438-tensor identity and companion. |
| SpeechBrain Lang-ID | Source, weight, Python closure and fixture-license review plus a scope hash. |
| Conv-TasNet Libri1Mix | A legal disposition for the CC-BY-SA-3.0/4.0 and WHAM CC-BY-NC-4.0 conflict; publication remains `NO_UPLOAD`. |
| HT-Demucs Multi | Weight redistribution terms, the MUSDB18/extra-training-data disposition and owner disposition for the exact Linux dependency evidence. The repaired Python 3.12 reference closure is reproducible, but its NumPy wheel bundles GPL-with-GCC-exception `libgfortran` and LGPL `libquadmath`, which the current fail-closed policy does not approve. |
| CLAP HTSAT fused | The exact model-free audit, dependency inventory and summary are bound to candidate payload `91a8a82f8c420ac5f456f12f021bd385c43b50f947e9e05e547272ae3cec85aa`. The existing model-license row is commercial, but dependency review and explicit runtime owner approval remain pending; no checkpoint was acquired or executed. |
| FireRedASR-AED-L | CMVN, output-dictionary and native source seams are authenticated/source-implemented, but the empty config, dependency/training provenance review, complete transcription route and real CPU parity remain blocked. No approval scope is inferred from the source-ready labels. |
| AudioGen Medium | Exact external T5 revision/weight identity, compression checkpoint build provenance, dependency closure, real execution/parity and an approval-scope hash. The checked-in model-free evidence is deliberately `signable=false`. |
| XY-Tokenizer | Complete dependency/license evidence, exact reviewed tensor manifest and scope hash. Current SciPy/SymPy, setuptools, soxr, tokenizers and tqdm evidence is incomplete. |
| CosyVoice2 HiFT | Packet completeness is deliberately unassessed here because the owner manifest is dirty and outside this campaign's staging scope. |
| BiCodec | A research-only/non-commercial execution and publication disposition. Decode evidence does not approve the missing PCM encode route or upload. |
| NSNet2 / RMVPE / corrected SpeechBrain and WeSpeaker artifacts | Exact replacement-versus-withdrawal decision and missing provenance/license sign-off. RMVPE's absent exact-source license may not be inferred as permissive. |
| SeamlessM4T-v2-Large | Decide between a real gated research-only artifact and withdrawal of the empty public repository. |
| `dynet38`, `qwen-omni-utils`, `soynlp`, Triton/NVIDIA payload issues | Resolve the exact release/source mismatch, GPL/LGPL conflict or bundled native-payload review before the affected family can receive a scope hash. |

### HT-Demucs Multi primary-source boundary (2026-09-08)

The fixed source checkout remains `facebookresearch/demucs` at
`e976d93ecc3865e5757426930257e200846a520a`. Its repository `LICENSE` is MIT
and the repository README says that Demucs is released under that license.
That proves the source-code boundary; it does not separately state
redistribution terms for the five externally hosted `.th` checkpoint files.

The same official README states that HT-Demucs was trained on MUSDB HQ plus an
additional 800-song dataset. The official `sigsep-mus-db` README distinguishes
its MIT-licensed parser from the full music dataset and says access to the
tracks is restricted to academic-purpose use. Therefore this packet does not
infer a commercial checkpoint-redistribution right from the source-code MIT
license or from the parser license. Until the owner selects an exact
disposition, the safe proposal is engine support plus direct-upstream,
no-upload reference validation only; Vokra must not mirror the checkpoint
bytes in its official model zoo.

The Python 3.12 conflict is narrower than the upstream requirements snapshot:
the pinned `demucs.audio` module imports `torchaudio` and `lameenc` at module
load, but its reference-path `convert_audio` function calls the official
`julius.resample_frac` implementation and does not access either package. The
reference closure committed at `4c91f173` keeps the upstream snapshot
byte-identical, loads the fixed PCM16 WAV fixture without `torchaudio`, and
provides process-local fail-closed stubs for `lameenc`, `torchaudio` and the
otherwise-unused `openunmix.filtering.wiener` import seam. Every fixed member
must report `cac=true`, `wiener_iters=0` and `end_iters=0` before execution;
any Wiener call fails. Unused `dora-search` is also absent from the active
closure. These constraints remove all four packages from the exact lock
without changing the official resampler or active model numerics.

A model-free VAST audit at clean commit
`baef9b9c8bb7559f8d9cb1dc714718030a138450` verified the Linux x86_64 Python
3.12 lock as an exact 16-row closure, with zero factual collection failures
and all model, weight, audio, source-repository, Cargo and upload activity
false. The lock SHA-256 is
`8d1b65d5c4a84e18c646d11539977797ec072620f75311077c65e41fa61d3bec`;
candidate package/license row SHA-256 values are respectively
`4b6fc6cc81a0da62b06c4c275a4b1cdc496e40796e28228f984c962ed6cb25d3`
and `4b3cabcae55752a24cd23cd26a935a21f0020111d19594346055ead552ee8a9b`.
The 829,655-byte evidence JSON has SHA-256
`2ccc87081b52d2e8fc430421d17085fce4105d2515b95ff1118478a9a438e056`.

That exact NumPy 2.5.3 wheel records bundled
`numpy.libs/libgfortran-*.so` as `GPL-3.0-or-later WITH GCC-exception-3.1`
and `numpy.libs/libquadmath-*.so` as `LGPL-2.1-or-later` in its own primary
license bytes. The current dependency policy rejects GPL/LGPL rows, so the
collector correctly remains `BLOCKED_OWNER_REVIEW` / `NO_UPLOAD`. This is not
runtime parity evidence and does not authorize checkpoint acquisition,
execution or publication.

Primary sources:

- <https://github.com/facebookresearch/demucs/blob/e976d93ecc3865e5757426930257e200846a520a/LICENSE>
- <https://github.com/facebookresearch/demucs/blob/e976d93ecc3865e5757426930257e200846a520a/README.md>
- <https://raw.githubusercontent.com/facebookresearch/demucs/e976d93ecc3865e5757426930257e200846a520a/demucs/audio.py>
- <https://github.com/sigsep/sigsep-mus-db/blob/master/README.md>

## Owner decision vocabulary

Every eventual signed record must choose exactly one disposition for its exact
scope:

1. `APPROVE_COMMERCIAL_EXECUTION` — source, model, dependencies, native
   payloads and operator execution are accepted for the hash-bound scope.
2. `APPROVE_RESEARCH_ONLY_EXECUTION` — execution is accepted with the
   non-commercial/research-only gate retained; this does not approve upload.
3. `WITHHOLD_EXECUTION` — do not acquire or execute the affected model or
   companion; retain the explicit unsupported result.
4. `REPLACE_PUBLIC_ARTIFACT` — approve preparation and validation of corrected
   bytes; publication still requires separate repository-scoped permission.
5. `WITHDRAW_PUBLIC_REPOSITORY` — approve withdrawal instead of replacement;
   this is a separate destructive action and must name the exact repository.

For an approval, record the signer, UTC timestamp, decision, exact scope
SHA-256 and any retained restrictions. For a withholding or withdrawal, record
the reason and exact repositories. Until that record exists, all affected rows
remain `APPROVAL_BLOCKED` or `SOURCE_BLOCKED` and remain in the 63-row
denominator.

## Current execution consequence

The owner does not need to decide these rows while the remaining source and
model-free work is still being completed. No row may be advanced merely to
keep VAST busy. Once every incomplete scope above has either been completed or
proved unresolvable, present a single consolidated sign-off request before any
blocked real-weight job is started.
