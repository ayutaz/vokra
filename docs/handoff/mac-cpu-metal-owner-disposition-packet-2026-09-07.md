# Mac CPU / Metal owner disposition packet (2026-09-07)

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
| Qwen3-ASR 0.6B / 1.7B | `tools/parity/qwen3_asr/license_gate_manifest.json` | `e368984045fe3d73015c8e4d5e8696e050d5eb983796738a794ab112c5519a4a` | Review two model-license rows and 31 package rows; approve or withhold source/model/operator execution. |
| Qwen3-TTS four variants | `tools/parity/qwen3_tts/license_gate_manifest.json` | `46662c9a1a1135c37a4dc00583c1a637f72aa745ca10f2172a38f77df9d1b1da` | Resolve the 61 of 67 component/dependency rows still pending owner approval. Publication remains `NO_UPLOAD`. |
| SpeechT5-TTS + HiFi-GAN | `tools/parity/speecht5_tts/license_gate_manifest.json` | `99116b392c560ec40c574589305492f35d9d30e8e2f44a9c03392885c77e85ba` | Dependency/model rows are reviewed; operator approval and the still-unverified API smoke must close before execution. |
| MOSS Audio 4B / 8B | `tools/parity/moss_audio/license_gate_manifest.json` | `50948bb28d02a01ad2e1f754052a6e1b8ff8b1bf9f90dd9c525a1c464de58001` | Resolve source/model licenses, checkpoint index/shard identities and five review rows. The model-free API contract is ready, but real weights remain blocked. |
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
| MOSS Audio Tokenizer Nano | Final Python dependency closure and a scope hash. The fixed source/API/tap contract and Apache source/weight review do not approve real execution. |
| MOSS-TTS Local | Dependency-license review, complete composite PCM execution boundary and a scope hash over the 438-tensor identity and companion. |
| SpeechBrain Lang-ID | Source, weight, Python closure and fixture-license review plus a scope hash. |
| Conv-TasNet Libri1Mix | A legal disposition for the CC-BY-SA-3.0/4.0 and WHAM CC-BY-NC-4.0 conflict; publication remains `NO_UPLOAD`. |
| HT-Demucs Multi | MUSDB18 provenance, weight redistribution terms and a Python-3.12-compatible dependency closure. The current torchaudio constraint is not resolvable as written. |
| XY-Tokenizer | Complete dependency/license evidence, exact reviewed tensor manifest and scope hash. Current SciPy/SymPy, setuptools, soxr, tokenizers and tqdm evidence is incomplete. |
| CosyVoice2 HiFT | Packet completeness is deliberately unassessed here because the owner manifest is dirty and outside this campaign's staging scope. |
| BiCodec | A research-only/non-commercial execution and publication disposition. Decode evidence does not approve the missing PCM encode route or upload. |
| NSNet2 / RMVPE / corrected SpeechBrain and WeSpeaker artifacts | Exact replacement-versus-withdrawal decision and missing provenance/license sign-off. RMVPE's absent exact-source license may not be inferred as permissive. |
| SeamlessM4T-v2-Large | Decide between a real gated research-only artifact and withdrawal of the empty public repository. |
| `dynet38`, `qwen-omni-utils`, `soynlp`, Triton/NVIDIA payload issues | Resolve the exact release/source mismatch, GPL/LGPL conflict or bundled native-payload review before the affected family can receive a scope hash. |

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
