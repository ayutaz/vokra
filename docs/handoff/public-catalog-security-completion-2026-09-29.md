# Public catalog and security completion plan (2026-09-29)

**Status: active.** This is the execution queue for the owner's request to
finish the public Mac CPU/Apple Metal catalog, repair public artifacts, and
resolve the dependency and OpenSSF findings. It is not a completion claim or
an upload authorization.

## Audited starting point

- Clean GitHub `main`: `79745a4d7c8330f8b760e405cf7b9276c4a7fb49`;
  no open PR at this snapshot.
- Read-only `uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py --format summary`:
  194 public repositories,
  193 with GGUF, 198 GGUF files. CPU code/artifact reachability is 136 `full`,
  43 `partial`, 14 `no-runtime-binder`, and 1 `not-artifact`; Metal is 136
  `full`, 57 `blocked-by-cpu`, and 1 `not-artifact`. Thus 58 rows are unresolved
  in the live public inventory. `full` is not a blanket real-weight Apple
  hardware parity verdict.
- GitHub Dependabot open-alert API: 258 open, 226 with a named patched version
  and 32 without one. Open code-scanning findings: `CIIBestPracticesID`,
  `CodeReviewID`, `MaintainedID`, `SASTID`, and `VulnerabilitiesID` (one each).
- The 2026-09-21 M5 owner checklist has 53 checked and 29 unchecked literal
  boxes. Its prose-only GA and platform conditions remain additional work.
- VAST account readback: one running instance labelled
  `ralomi-m4r-bounded-context-v2-v21`, unrelated to Vokra; no retained
  volumes. Do not operate on that instance. Old Vokra instance IDs in dated
  records are not available transfer sources.

The live audit logic and HF revisions, GitHub alert API, exact source/weight
manifests, and measured parity packets supersede this dated count whenever
they change. Do not replace historical measurements with a current total.

### 2026-09-29 continuation readback (01:56 UTC)

The read-only HF summary still reports 194 public repositories, 193 with
GGUFs, 198 GGUF files, CPU `136 full / 43 partial / 14 no-runtime-binder /
1 not-artifact`, and Metal `136 full / 57 blocked-by-cpu / 1 not-artifact`.
This does not prove real-weight or Apple parity. PR #146 merged at
`d8759c6dc870a9ba5c4ef48ed342c13eb9cb1800`; PR #135 was rebased onto
that main head and awaits its fresh CI. PRs #137 and #142 remain stacked on
#135 and #137 respectively, and PR #147 remains draft. No model row is
promoted by these model-free changes.

The unmerged XY-Tokenizer closure candidate at clean head
`e7806fe3f37d497362956b89841f4db26172c043` has lock SHA-256
`c445c7c2f6dfc036620ff7cf123b974f0e519f2081c62ca9bde2a73c61c1ae57`.
Its exact-head VAST model-free collector reports 38 active dependency rows,
35 successful and three blocked: SciPy 1.18.1, setuptools 84.0.0 and
tokenizers 0.23.1. The recovered collection report SHA-256 is
`d78e22695d668a4c2e57317f8dfb975544041136c0973f54f7bbf9328aa82b7c`;
the license-evidence SHA-256 is
`15166ba9dc0ad739defbdc6fa5787bf5ea6c4b65c331749b84933e0e88de0ec8`.
The subsequent exact-artifact probe found GPL-with-GCC-exception and LGPL
native terms in SciPy's wheel license payload, LGPL vendored terms in
setuptools, and no unambiguous distribution-owned primary license bytes in
tokenizers. A permissive classifier was therefore not added. This candidate
remains `BLOCKED` / `NO_UPLOAD`, with no checkpoint acquisition, model
execution, sign-off or Apple verdict. Both disposable XY VAST workers were
destroyed after the model-free checks; unrelated instances were untouched.

### 2026-09-29 VibeVoice 1.5B model-free compatibility readback (06:54 UTC)

The unmerged security candidate at clean head
`1515ab2d29f4f821c0d711097e25e19b39728b77` pins Transformers 5.10.4,
Torch 2.13.0 and Diffusers 0.38.0. A disposable VAST worker (`53318907`)
verified that head, the clean checkout, its frozen Python 3.12 environment and
official VibeVoice source revision
`2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c`. The model-free import
probe then stopped with `BLOCKED_MODEL_FREE_API` / `ValueError` in
`module_import`; its traceback tail reaches the [official tokenizer's
`AutoModel.register` call](https://github.com/microsoft/VibeVoice/blob/2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c/vibevoice/modular/modular_vibevoice_tokenizer.py#L1188)
and Transformers `auto_factory.py:429,674`. The fixed upstream
[`pyproject.toml`](https://github.com/microsoft/VibeVoice/blob/2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c/pyproject.toml#L22)
specifies `transformers==4.51.3` and warns that later versions may be
incompatible. The diagnostic's message classifier did not affirm a duplicate
registration, so the exact exception wording and a safe compatibility remedy
remain unverified; do not bypass the upstream registration check by assumption.

The sanitized [stage log](vibevoice-1-5b-import-stages-2026-09-29.ndjson)
and [probe result](vibevoice-1-5b-import-probe-2026-09-29.json) have SHA-256
`ac2e6783b34eef626d1475afb17593ebbee5748d100e42df7c24067d4bdd89fb`
and `9b5cb2d647f1a7494baca5c398bd340e51fa45c225b890b137dad1502514e329`.
This is not an API, license, CPU-parity or Apple pass. No checkpoint, weights,
model forward or upload was requested. The worker and owned volumes were
destroyed; the Vokra-labelled instance and volume readback was empty. The
`vokra/vibevoice-1.5b` row remains partial and in the 58-row queue. Next,
resolve the fixed-upstream API/security conflict and dependency-license
decisions before any approved real-weight reference run.

### 2026-09-29 VibeVoice import diagnosis supersession (10:48 UTC)

The earlier `auto_model_registration_duplicate=false` field is a diagnostic
false negative, not evidence that registration succeeded. In the fixed
[Transformers 5.10.4 source distribution](https://files.pythonhosted.org/packages/f7/5d/1df789ca27a436ce09de67c8fee6acd2a528d34c28685991f8203e5418ae/transformers-5.10.4.tar.gz),
`auto_mappings.py` maps `vibevoice_acoustic_tokenizer` to
`VibeVoiceAcousticTokenizerConfig`, `modeling_auto.py` maps the same type to
`VibeVoiceAcousticTokenizerModel`, and `auto_factory.py:674` raises
`ValueError` when another class with that config name is registered without
`exist_ok`. The authenticated Microsoft source calls `AutoModel.register`
without `exist_ok` at the line reached by the VAST traceback. Together these
fixed-source facts identify a registration collision; the previous classifier
missed the library's actual "already used by a Transformers model" wording.

A local, unmerged diagnostic candidate at `2554a00f` now categorizes only
that exact VibeVoice `module_import` `ValueError` and redacts arbitrary
exception text. Its model-free self-test passed, but the candidate has **not**
been replayed on VAST and does not make the upstream import compatible.
Dependency-license review, a safe compatibility remedy, independent real-weight
reference, CPU parity, Apple CPU/Metal and publication all remain open. No
model row is promoted by this source-level diagnosis.

### 2026-09-29 VibeVoice bounded import replay (11:20 UTC)

An unmerged compatibility candidate at clean head
`9d181d7205d19f10284327df52a05aab374d2f70` limits the
Transformers 5.10.4 `AutoModel.register` override to the two fixed upstream
tokenizer class pairs and the exact duplicate-registration error, restores the
registry method after import, and removes the reference dumper's broad
file-by-path import fallback. Both model-free local self-tests and the
forbidden-symbol/zero-dependency gates passed. This is a candidate, not an
upstream API or parity verdict.

A disposable VAST worker (`53348901`) replayed that exact head against clean
Microsoft source `2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c` and locked
Python 3.12 environment (uv.lock SHA-256
`1b481e4774ec3c5cb53a061cbf7f68b0a98f52d4005b86fda1c19b01302d20fd`).
The registration collision no longer appears as the first failure. The
model-free import still stops at `module_import` with `ModuleNotFoundError`:
the fixed [Microsoft text tokenizer](https://github.com/microsoft/VibeVoice/blob/2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c/vibevoice/modular/modular_vibevoice_text_tokenizer.py#L7)
imports `transformers.models.qwen2.tokenization_qwen2_fast` and subclasses
`Qwen2TokenizerFast`, while the locked Transformers 5.10.4 installation has
only `tokenization_qwen2.py` with `Qwen2Tokenizer(TokenizersBackend)` in that
package. A name alias is not proof of tokenizer-semantic equivalence and was
not added. The sanitized [API result](vibevoice-1-5b-api-smoke-9d181d72-2026-09-29.json)
has SHA-256 `2f2f54755ff319b205d6708edb419bc27443cabb0d3189f4f9a1e97e77f1fe0e`.
No checkpoint, weight, forward, numerical reference, CPU parity or upload ran.
The worker and an earlier unusable stopped contract (`53348753`) were both
destroyed; neither left an owned volume. An unrelated VAST instance was not
modified. The public row stays partial and in the 58-row queue.

### 2026-09-29 GitHub security-alert readback (07:05 UTC)

With GitHub `main` at `09b39079a5b205e859ea24392ffe56b3bcd2371b`, a
fresh paginated `GET /repos/ayutaz/vokra/dependabot/alerts?state=open` returned
342 open alert records: 299 have `security_vulnerability.first_patched_version`
and 43 do not. They span 63 manifest paths and 61 distinct GHSA IDs; a
manifest alert is not a unique advisory. The separate open code-scanning alert
API still returned five records. This supersedes the 258 / 226 / 32 starting
snapshot above for current queue sizing, without rewriting that dated result
or claiming that the count change has a known cause. Of the 342 records, 88
have a 2026-09-29 `updated_at` date; 63 of those name Torch, 21 Transformers,
two Accelerate and two Diffusers. An update timestamp alone does not prove a
newly introduced vulnerability or a completed fix. Re-read the API after each
merge and close alerts through the dependency graph rather than dismissal.

### 2026-09-29 OWSM `torch-complex` license readback (12:56 UTC)

The fixed [PyPI `torch-complex` 0.4.4 release](https://pypi.org/project/torch-complex/0.4.4/)
lists an Apache license *classifier*, but its sdist (SHA-256
`4153fd6b24a0bad689e6f193bfbd00f38283b1890d808bef684ddc6d1f63fd3f`)
and wheel (SHA-256
`6ab4ecd4f3a16e3adb70a7f7cd2e769a9dfd07d7a8e27d04ff9c621ebbe34b13`)
contain no LICENSE/COPYING/NOTICE text. Both package metadata records say
`License: UNKNOWN`. The package's PyPI homepage points to an absent
`kamo-naoyuki/torch_complex` repository; the author's existing
[`kamo-naoyuki/pytorch_complex` source](https://github.com/kamo-naoyuki/pytorch_complex)
has a `v0.4.4` tag at `8a2ad1e47f3df25a30eb426f6ad781b89103fab3`, but
its GitHub repository license field is `null` and that exact tag's recursive
source tree has no license file. These primary-source checks do not
authenticate the terms for the fixed release. Keep the OWSM dependency gate
blocked; obtain explicit upstream license bytes/permission for that release or
an independently reviewed compatible replacement before any real-weight
reference or CPU parity. This readback is not an owner sign-off or model pass.

### 2026-09-29 VibeVoice Realtime tokenizer-license readback (13:07 UTC)

The fixed [`Qwen/Qwen2.5-0.5B` tokenizer companion at
`060db6499f32faf8b98477b0a26969ef7d8b9987`](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/LICENSE)
contains an Apache License 2.0 `LICENSE` file, not merely a model-card tag.
This resolves the narrow question of whether primary license text exists for
that fixed companion; it does not authenticate the entire dependency closure,
dataset rights, exact redistribution scope, or a new owner decision. The
`vokra/vibevoice-realtime-0.5b` row remains partial, with converter emission,
complete native synthesis, independent real-weight CPU parity, Apple
CPU/Metal, and publication still blocked. No weights or model were run locally.

### 2026-09-29 VibeVoice Realtime fixed-header readback (13:33 UTC)

A disposable VAST worker read only bytes `0-79439` of the pinned
`microsoft/VibeVoice-Realtime-0.5B` safetensors at revision
`6bce5f06044837fe6d2c5d7a71a84f0416bd57e4`. The redirect advertised
the previously pinned LFS SHA-256
`7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514`,
and the final HTTP response was `206` with
`Content-Range: bytes 0-79439/2035332888`. The eight-byte prefix declared a
79,432-byte JSON header; the 79,440 retrieved bytes have SHA-256
`73c4658be17469d62e22a0f4b7f042cc10aa83d409055a5e3a350d5b8d8f26cb`.
This range readback is bound to the server's immutable revision/LFS identity;
it does **not** hash or authenticate the entire tensor payload.

The header lists 605 tensors, all BF16, with contiguous payload offsets from
zero through 2,035,253,448. Its 26 `model.prediction_head.*` names and shapes
match the exact loader contract in VibeVoice Realtime diffusion-head
implementation commit `21dded9b7b0630266ee6c223b87d5ff7e569e2bc`:
four outer projections, four layers of five tensors, and two final-layer
projections. No tensor has rank above three. This removes the narrow
*head-name/header-shape* uncertainty; full-checkpoint hash, GGUF conversion,
independent real-weight reference, numerical CPU parity, integrated
CFG/scheduler, acoustic decode, Apple CPU/Metal and publication remain open. No model ran on
the maintainer Mac. The header-only VAST instance `53365673` was destroyed;
the exact-ID API readback returned `instances: null`.

### 2026-09-29 VibeVoice Realtime CFG sampler contract (13:47 UTC)

Implementation commit `e0390fdf76dc27a6be67a874bc139739b5e705db` adds a
bounded CPU-only, 20-step classifier-free-guidance sampler around the native
Realtime prediction head. It gives the conditional and unconditional branches
the same current latent, applies guidance before the DPM-Solver++ step, and
rejects a non-CPU backend explicitly. This is a model-free seam, not a full
streaming synthesis path or a GPU speed decision.

At that exact commit, disposable VAST instance `53367191` ran
`cargo test -p vokra-models --lib vibevoice_streaming::sampler` (5 passed)
and `cargo test -p vokra-models --lib vibevoice_streaming` (25 passed, 2
ignored because exact Qwen sidecars were absent), with Rust 1.98.1. The
instance and its storage were destroyed; the exact-ID readback returned
`instances: null`. No weights were downloaded or executed in this test.
Official-source numerical reference, full-checkpoint authentication, real-
weight CPU parity, Apple CPU/Metal no-fallback testing, and same-input CPU/GPU
benchmarking are still required. The public row remains partial, so the
136-full / 58-unresolved audit counts do not change.

### 2026-09-29 VibeVoice Realtime acoustic-decoder layout readback (14:02 UTC)

Disposable VAST instance `53368332` reread only the fixed
`model.safetensors` bytes `0-79439` at the same immutable HF revision. The
response was HTTP `206`, `Content-Range: bytes 0-79439/2035332888`, with
79,440 bytes and the same range SHA-256
`73c4658be17469d62e22a0f4b7f042cc10aa83d409055a5e3a350d5b8d8f26cb`.
The 605 tensor entries include exactly 276 BF16
`model.acoustic_tokenizer.decoder.*` entries. Canonical sorted JSON of their
`name`, `shape`, and `dtype` fields (`jq -cS`, one line with a trailing newline)
has SHA-256
`4576ebac2def6293d72668f3fa068461b7c256c6d75e9f12eb6c16207ccfbfbb`.
The descriptor list has the existing decoder's 64-to-2048 stem, seven stage
widths `2048/1024/512/256/128/64/32` with depths `8/3/3/3/3/3/3`, six
transposed-convolution upsamplers and `[1,32,7]` PCM head. This supports a
strict Realtime-specific binding to the shared native decoder implementation;
it does not prove payload identity or numerical parity. The [fixed Microsoft
streaming source](https://github.com/microsoft/VibeVoice/blob/94da20d98b2fa7688e9cbfaf7692ddb4954f7600/vibevoice/modular/modeling_vibevoice_streaming_inference.py#L716-L724)
unscales each generated latent using the checkpoint's scalar scale and bias
before cached acoustic decoding, which remains an implementation and parity
gate. Both the earlier unused header worker `53367911` and the completed
worker `53368332` were destroyed with exact-ID `instances: null` readbacks;
no probe instance was retained. No payload or model was run locally.

## Completion rule for each model

Move a row only when the evidence for that stage exists: exact upstream
source/checkpoint/tokenizer/codec and license facts; reviewed owner decision
where required; strict converter, binder, native forward and CLI route;
independent fixed-upstream reference; no-upload real-weight VAST CPU parity;
fresh hash-bound transfer packet; Apple CPU/reference, Metal/reference and
Metal/CPU measurements with explicit no-fallback; separately authorized
publication or withholding; and a final live public audit. A build, synthetic
fixture, inspection route, or model-free self-test does not skip a stage.

For execution after validation, benchmark CPU and GPU on the same model,
input and quality contract, and use the faster supported backend. Record the
measurement and selected device; do not assume GPU is faster for a small
workload or omit mandatory CPU/reference and Metal/no-fallback evidence. An
unsupported GPU operation remains an explicit error, never a silent CPU
fallback. This is an execution policy, not a claim that every model has been
benchmarked or that automatic backend selection is implemented.

The protected
`tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` must not be
read, edited, staged, reverted, or cleaned. The maintainer Mac performs no
model download or execution and no workspace or `vokra-models` Cargo work.
Disposable VAST workers handle large artifacts and workspace tests, with
small evidence recovered before destroy. Scaleway is the final Apple-hardware
stage after the non-Apple gates are closed or explicitly withheld.

## Complete 58-row work queue

The following groups are disjoint and are derived from the live TSV audit at
the starting head. The next action is a gate, not a claim that later stages
are already satisfied. Re-run the TSV audit and inspect per-row evidence before
moving any entry.

### Fourteen rows without a complete runtime binder

For each entry, authenticate source, artifact and license; implement the
strict binder, native forward and CLI route; then produce an independent
reference and real-weight CPU parity before Apple execution:

| Public repository | Immediate boundary |
|---|---|
| `vokra/ace-step-1.5` | Complete native binder/composite route. |
| `vokra/baichuan-audio` | Complete native binder/composite route. |
| `vokra/granite-speech-4.1-2b` | Complete native binder/composite route. |
| `vokra/hibiki-2b` | Complete native binder/composite route. |
| `vokra/htdemucs-multi` | Authenticate member ordering, per-member configuration and ensemble weights; resolve checkpoint/data/dependency terms. |
| `vokra/kimi-audio` | Complete native binder/composite route. |
| `vokra/kyutai-tts-1.6b-en-fr` | Complete native binder/composite route. |
| `vokra/qwen2-5-omni-7b` | Complete native binder/composite route. |
| `vokra/qwen2-audio-7b-instruct` | Resolve exact source-license boundary, then complete native route. |
| `vokra/step-audio2-mini` | Complete multi-component native route and source/weight/dependency review. |
| `vokra/vibevoice-asr` | Authenticate dataset, tokenizer and timestamp/diarization semantics before native route. |
| `vokra/vieneu-tts-v3-turbo` | Resolve voice-cloning/dependency policy and complete native route. |
| `vokra/xtts-v2` | Authenticate safe checkpoint and CPML/consent boundary before native composite. |
| `vokra/xy-tokenizer` | Obtain a real checkpoint payload; current GGUF has no tensors. |

### Twenty-one partial rows with incomplete released-artifact CPU forward

The live audit's reason for each is an incomplete CPU forward despite an
existing route or binder. Review each implementation against its own exact
source and license contract, finish the native forward/composite, then run
independent reference and real-weight CPU parity:

`vokra/audioldm2`, `vokra/audioldm2-large`, `vokra/canary-qwen-2.5b`,
`vokra/chatterbox-multilingual-v3`, `vokra/chatterbox-nano-v1`,
`vokra/chatterbox-turbo-v1`, `vokra/chattts`, `vokra/clap-htsat-fused`,
`vokra/cosyvoice2-0.5b`, `vokra/csm-1b`, `vokra/dia-1.6b`,
`vokra/firered-asr-aed-l`, `vokra/fun-cosyvoice3-0.5b-2512`,
`vokra/irodori-tts-500m-v3`, `vokra/kyutai-stt-2.6b-en`,
`vokra/owsm-v4-medium-1b`, `vokra/sortformer-diar-4spk-v1`,
`vokra/ultravox-v0-5-llama-3-2-1b`, `vokra/vibevoice-1.5b`,
`vokra/vibevoice-realtime-0.5b`, and `vokra/voxcpm-0.5b`.

### Twenty-two partial rows with a named artifact or companion boundary

| Public repository | Immediate boundary |
|---|---|
| `vokra/audiogen-medium` | Authenticate and bind the T5 conditioner and EnCodec decoder companions. |
| `vokra/canary-1b-flash` | Convert the complete AED decoder, aggregate tokenizer and authenticated manifest. |
| `vokra/canary-1b-v2` | Replace the misidentified auxiliary CTC GGUF with the complete AED conversion. |
| `vokra/conv-tasnet-libri1mix` | Resolve source-license/topology conflict and replace the stale artifact. |
| `vokra/lang-id-voxlingua107` | Publish a verified classifier plus ordered-label artifact after the gated replacement. |
| `vokra/mms-1b-all-base` | Supply the verified backbone rather than treating an adapter as the whole model. |
| `vokra/moss-audio-4b-instruct` | Correct provenance and tokenizer/chat/processor sidecars; run real-weight parity. |
| `vokra/moss-audio-8b-instruct` | Same independent correction and parity for the 8B identity. |
| `vokra/moss-audio-tokenizer-nano` | Replace the GGUF mis-stamped as Full with correct Nano provenance. |
| `vokra/moss-tts-local-transformer-v1.5` | Bind and verify its 48 kHz stereo MOSS Audio Tokenizer v2 companion. |
| `vokra/nsnet2` | Correct the public weight license/provenance and attribution. |
| `vokra/qwen3-asr-0.6b` | Replace old GGUF with exact audio metadata and five authenticated sidecars. |
| `vokra/qwen3-asr-1.7b` | Same gated replacement for the 1.7B identity. |
| `vokra/qwen3-tts-12hz-0.6b-base` | Apple parity, then variant/BPE-correct main and public tokenizer companion. |
| `vokra/qwen3-tts-12hz-0.6b-customvoice` | Apple parity, then correct identity/BPE sidecars and companion. |
| `vokra/qwen3-tts-12hz-1.7b-base` | Apple parity, then correct 1.7B topology/BPE sidecars and companion. |
| `vokra/qwen3-tts-12hz-1.7b-customvoice` | Apple parity, then correct 1.7B identity/BPE sidecars and companion. |
| `vokra/rmvpe` | Resolve exact-source license and replace the falsely permissive GGUF. |
| `vokra/sbv2-v2-jp-extra-base` | Correct public tensor names/metadata and production Japanese G2P. |
| `vokra/speechbrain-spkrec-ecapa-voxceleb` | Replace GGUF with out-of-bounds tensor data. |
| `vokra/wespeaker` | Correct CC-BY-4.0 weight provenance and attribution. |
| `vokra/yue-xcodec-mini` | Independently bind acoustic/semantic/fusion/RVQ PCM encode; decode alone is partial. |

### One non-artifact public repository

`vokra/seamless-m4t-v2-large` has no GGUF. Prepare a real gated artifact with
independent evidence or obtain an exact owner-approved withdrawal decision.
Neither outcome may be inferred from an empty repository.

## Other Apple and artifact debt outside the 58-row count

- GigaAM v3 passed a new no-upload VAST real-weight CPU replay at the Torch
  2.13.0 update in PR #131. Its older Apple verdict belongs to an older
  dependency/head; regenerate the packet and measure the current Apple CPU,
  Metal and no-fallback legs.
- Qwen3-TTS four variants passed independent VAST CPU parity in PR #109;
  their Apple measurements and gated public replacements remain pending.
- Distil-Whisper and Kotoba-Whisper passed real-weight VAST CPU parity; the
  final Apple measurements and repository disposition remain pending.
- Charsiu and other prepared rows still require their own Apple/evidence
  reconciliation; do not equate a full code route with device parity.
- `vokra/voxtral-small-24b-2507` has a stale false Mini-3B provenance stamp
  in its live GGUF. The historical retained VAST instance cited in the M5
  checklist is gone. Recreate any required 48-GB artifact on a new disposable
  VAST worker, then use the gated publisher only after exact artifact approval.

## Security queue

1. For each Python reference environment, select the alerts with published
   fixes, update its `pyproject.toml` and `uv.lock`, run model-free import/API
   and license gates, and repeat its independent real-weight reference/parity
   on VAST. Close alerts through a reviewed PR and GitHub dependency-graph
   rescan, not dismissal. Start with the existing isolated Ultravox and
   X-Codec2 Transformers branches; X-Codec2's Torch hard pin remains a
   separate blocker. Process the other environments independently rather
   than bulk-upgrading oracles.
2. For the 32 alerts without a published fix, recheck upstream releases and
   direct usage; remove or replace the dependency only with an independently
   verified reference contract. Keep unresolved alerts visible with an exact
   advisory and environment disposition.
3. Resolve Scorecard's five findings on their actual terms: external Best
   Practices registration; repository-age recheck; independent code review;
   CodeQL coverage of the rolling changeset window; and dependency
   remediation. Re-run Scorecard and inspect SARIF after the relevant changes.

The current detailed security execution record is
[`security-remediation-2026-09-21.md`](security-remediation-2026-09-21.md).
Its 261-alert snapshot predates PR #131 and the 258-alert starting API result;
the later 342-alert readback above supersedes both for current queue sizing.
Retain the dated observations and add a supersession note when refreshing them.

## Execution order and ownership

1. Keep the live 58-row TSV and GitHub alert snapshot as the starting audit;
   refresh them after each model/security merge. Preserve exact revision and
   evidence hashes. Use one model family or one Python environment per PR.
2. Close owner-independent source, dependency, runtime and reference work.
   The Sol manager reviews; Luna implementers own bounded implementation
   artifacts and focused checks. No oracle is accepted from a Vokra mirror.
3. Present only the exact unresolved source/license/operator decisions for
   owner review. A general instruction to finish the campaign does not
   supply a missing upstream license or change a non-commercial policy.
4. Run approved real-weight conversion/reference/CPU parity and workspace
   gates on disposable VAST workers. Recover hashes and small evidence;
   destroy instances and storage.
5. Freeze a clean head and regenerate direct transfer packets. Run the final
   Scaleway Apple CPU/reference, Metal/reference and Metal/CPU no-fallback
   batch. A failed device result returns to the implementation/VAST stage.
6. Apply separately authorized, repository-scoped upload or withdrawal
   decisions through `publish-one.sh` and its license/provenance gates;
   reconcile the live HF inventory, security alerts, Scorecard, docs and M5
   owner checklist. Never claim full catalog completion from code reachability
   alone.

## Final audit

Completion requires a current per-row disposition for all 194 public
repositories, with every executable row's source, license, native route,
independent CPU reference, VAST CPU result, Apple CPU/Metal/no-fallback result
and public artifact evidence linked to exact hashes. Every remaining security
alert must either be fixed or have a current evidence-backed upstream/policy
disposition, and all five Scorecard findings must be re-evaluated. Required
CI, workspace checks, documentation gates and branch protection must be
reviewed; no unrelated VAST/Scaleway resource may be modified. The worktree
and cloud resources must be reconciled before any completion claim.
