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

### 2026-09-29 VibeVoice Realtime acoustic-decoder contract replay (14:24 UTC)

After PR #160 was rebased onto `main` at `289f4416`, the unpushed CFG sampler
and acoustic-decoder commits were rebased onto PR head `3e070e4a`. Disposable
VAST instance `53370944` checked out exact integrated head
`70d4dc314c8c109478f754ddf8abd0b08072085a` from a git bundle and ran
Rust 1.98.1 model-free checks: `cargo test -p vokra-models --lib
vibevoice_streaming` (28 passed, two ignored for absent exact Qwen sidecars),
`cargo test -p vokra-models --lib realtime_decoder_tensor_count_gate` (one
passed), `cargo test -p vokra-models --lib vibevoice::tokenizer` (13 passed),
and `cargo clippy -p vokra-models --lib -- -D warnings` (passed). The
Realtime wrapper rejects a non-Realtime checkpoint before decoder loading;
the shared decoder's existing model-free topology and causal tests also pass.
The worker and its storage were destroyed; exact-ID API readback returned
`instances: null`.

These checks did not download or run weights. They do not establish complete
checkpoint authentication, independent waveform reference, real-weight CPU
parity, Metal parity, no-fallback execution, or a same-input CPU/GPU speed
comparison. The Realtime acoustic boundary remains CPU-only and explicitly
rejects Metal until that evidence exists; the public row remains partial.

### 2026-09-30 VibeVoice Realtime real-weight and device-selection readback

Follow-up [PR #161](https://github.com/ayutaz/vokra/pull/161) builds on merged
PR #160 without promoting the public row. The pinned Microsoft
`model.safetensors` at revision
`6bce5f06044837fe6d2c5d7a71a84f0416bd57e4` was downloaded and hashed
**on VAST only**: 2,035,332,888 bytes, SHA-256
`7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514`.
The exact 605 BF16 tensors were converted to a private 2,035,313,984-byte
GGUF, SHA-256
`908a10b917bee4b6389de09182b2603b833dcb1a642ed08bc8ff40c0212e0624`.
An ignored VAST integrity test compared all names, shapes and payload bytes
against the source, and the real GGUF passed the strict native binder. This
supersedes the earlier *header-only* identity limitation, not its historical
receipt. Neither weight file was moved to the maintainer Mac or uploaded.

The independent narrow reference imports the clean Microsoft source at
`94da20d98b2fa7688e9cbfaf7692ddb4954f7600`, binds the exact checkpoint,
then calls the official text/TTS/EOS and acoustic-connector path. The
decoder-only release legitimately lacks 276
`model.acoustic_tokenizer.encoder.*` parameters; any other missing or
unexpected parameter is rejected. BF16 same-input timing on a VAST RTX 4090
had CPU median 16.1034 s and CUDA median 0.01907 s, but the maximum output
difference was 0.25, above the unchanged provisional 0.05 selection guard;
CPU remained selected. Casting that same authenticated checkpoint to FP32
gave CPU median 23.495962 s and CUDA median 0.017232 s across three measured
iterations, with maximum output difference 0.0000457764; CUDA was selected
for **this fixed reference operation only**. The BF16 and FP32 packet SHA-256
values are respectively
`4c3590f5ef5b2bcc5f67b228dc8ce263ac95c5c579709fddfb2483c94a05b5a9`
and `af87fbc9959da3249971a637690bd00b890997647ff566f8a8a1ea8bdf2866e4`.
These comparisons are not native Rust parity, a release tolerance, or an
Apple Metal speed verdict.

The VAST clean head `31ab7d84` passed `cargo test --workspace` (8,100
passed, 0 failed, 105 ignored across 323 test binaries; log SHA-256
`ca3476bce2cb38bd1ed90e7d85fc0c5c8d77adcb785e32cc3d9c7b603023c59b`)
and `cargo clippy --workspace --all-targets -- -D warnings` (status 0;
log SHA-256
`be923e2675304dfbe7cb0379fef1f59e2699dfab11a215e0f38378cfbdc1fc29`).
The rebased PR head `9879093f` has the identical Git tree
`f24576cd3e385339c1450e6609b288039ed33114`; only commit ancestry
changed after #160 auto-merged. Disposable VAST instance `53372081` was
destroyed after small evidence recovery; exact-ID readback returned
`instances: null`. The reference Python dependency-license audit, complete
native synthesis, Rust CPU parity, independent waveform reference, Apple
CPU/Metal/no-fallback verification and public artifact gates remain open.
The row therefore remains partial and the 136-full / 58-unresolved count
does not change.

### 2026-09-30 VibeVoice Realtime staged native follow-ups

Draft [PR #163](https://github.com/ayutaz/vokra/pull/163), stacked on
PR #161, adds the four-layer text LM, twenty-layer TTS LM, shared embedding
and hidden-state splice, and the EOS classifier. Its fixed-upstream reference
runner can export official LM hidden states, while an ignored VAST-only Rust
diagnostic reports native-versus-official errors without declaring a parity
pass. The Rust tree matched VAST-tested commit `6bdb1f72`: workspace tests
passed 8,105/0/106 ignored across 323 suites and all-target Clippy passed.
The installed reference closure remains `OWNER_REVIEW_REQUIRED/NO_UPLOAD`;
the diagnostic has not been accepted as real-weight parity.

Draft [PR #164](https://github.com/ayutaz/vokra/pull/164), stacked on
PR #163, adds authenticated five-token text windows and incremental LM/TTS
KV-cache operations. Six speech iterations per window is a maximum, not an
EOS signal; positive and negative CFG prompts retain independent caches.
At exact VAST head `0ea381f9cd0a4398ee067c792746c32b2ccda7e7`, workspace
tests passed 8,113/0/106 ignored across 323 suites (log SHA-256
`41a7d9d38b587923e12f410da60cee9255a1779ce961ca6d6a5ed8c811d2e4a3`)
and all-target Clippy passed (log SHA-256
`5025862a5cdf2366e84dfefb99ea7355af609f4764e1f84a0fe8d30be86d47c0`).
The verification instance `53393052` and its storage were destroyed with an
exact-ID `instances: null` readback. These model-free checks do not establish
an integrated synthesis loop, native real-weight CPU parity, Apple CPU/Metal
parity, or public artifact readiness. The row remains partial and the live
audit remains 136 full / 58 unresolved.

### 2026-09-30 VibeVoice Realtime connector and preset boundary

Draft [PR #168](https://github.com/ayutaz/vokra/pull/168) stages a
model-free generation control plane on #164; it does not join the complete
streaming synthesis path. Draft [PR #171](https://github.com/ayutaz/vokra/pull/171)
then binds the distinct Realtime `64 -> 896 -> 896` acoustic connector and
implements the pinned `fc1 -> RMSNorm -> fc2` path with strict tensor names,
shapes and dense dtypes. Its exact VAST head `8a548f17` passed workspace tests
(323 suites, 8,121 passed, 0 failed, 106 ignored; log SHA-256
`019a943de9f1a78349509d10be5c0d42e7c7a1a3c78ba94da355e0344196a343`)
and all-target Clippy with warnings denied (log SHA-256
`7eb8ad0908269675e7236f795f1e291d812cbab15b4ff0c72f1d956e467862a2`).
Disposable VAST instance `53417369` was destroyed after validation; exact-ID
readback returned `instances: null`. The official-reference connector check is
still ignored and diagnostic-only, not real-weight native parity. Metal has
the required GEMM/RMSNorm Compute seam, while unsupported CUDA RMSNorm fails
explicitly; neither observation is an Apple speed or no-fallback verdict.

The [pinned Microsoft file demo](https://github.com/microsoft/VibeVoice/blob/94da20d98b2fa7688e9cbfaf7692ddb4954f7600/demo/realtime_model_inference_from_file.py)
loads a voice preset `.pt` with `weights_only=True` and narrowly allowed
`BaseModelOutputWithPast`/`DynamicCache` types before generating audio. The
[pinned generation path](https://github.com/microsoft/VibeVoice/blob/94da20d98b2fa7688e9cbfaf7692ddb4954f7600/vibevoice/modular/modeling_vibevoice_streaming_inference.py)
consumes four separate prefilled outputs: `lm`, `tts_lm`, `neg_lm` and
`neg_tts_lm`. A fabricated prompt or cache cannot validate this path.
[Microsoft's Realtime guidance](https://github.com/microsoft/VibeVoice/blob/94da20d98b2fa7688e9cbfaf7692ddb4954f7600/docs/vibevoice-realtime-0.5b.md)
describes embedded prompts as a deepfake mitigation and directs voice
customization requests to the team. The repository's MIT source license does
not by itself settle per-preset voice consent, provenance, redistribution or
Vokra publication. Record a fixed preset identity and owner/legal decision,
then implement a safe offline cache bridge and independent complete-waveform
reference before real-weight native CPU parity. The public row remains partial
and the 136-full / 58-unresolved audit count is unchanged.

### 2026-09-30 VibeVoice Realtime fixed voice-preset inventory

A read-only [GitHub tree API check](https://api.github.com/repos/microsoft/VibeVoice/git/trees/94da20d98b2fa7688e9cbfaf7692ddb4954f7600?recursive=1) of Microsoft's fixed source commit
`94da20d98b2fa7688e9cbfaf7692ddb4954f7600` returned `truncated=false`
and 25 blobs under `demo/voices/streaming_model/`. The English Carter
candidate used by the pinned official file-demo instructions is
`demo/voices/streaming_model/en-Carter_man.pt`, 4,256,002 bytes, Git blob
SHA-1 `1d795ef667e6641eecb8b22452bb853b089bfdbe`. This is a Git object
identity and size, **not** a verified payload SHA-256, cache-schema audit,
voice-consent finding or permission to redistribute. The [fixed HF model
revision](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/tree/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4) `6bce5f06044837fe6d2c5d7a71a84f0416bd57e4` lists the model,
config and preprocessor files but no voice preset; a complete reference must
bind the separate Git source artifact. No preset bytes, model weights or
token were downloaded to the maintainer Mac.

Microsoft's fixed file demo loads the `.pt` through `weights_only=True` with
only `BaseModelOutputWithPast` and `DynamicCache` safe globals, and its
Realtime guidance describes embedded prompts as a deepfake mitigation. The
[fixed model card](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4/README.md) says generated audio carries an audible disclaimer and an
imperceptible watermark. These source facts clarify the required safe cache
bridge and full-waveform guard but do not establish that a native Vokra path
preserves those safeguards. The preset's voice rights, execution scope and
publication remain for owner/legal disposition; native real-weight parity and
the public row remain blocked/partial.

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

## 2026-09-30 continuation readback (2026-09-29 23:39 UTC)

At clean `main` `4715e07f12a1e32247e848cdda4c99773d8800df`, the read-only HF
audit still reports 194 public repositories, 193 with GGUFs and 198 GGUF
files. CPU code/artifact reachability is 136 full, 43 partial, 14 without a
runtime binder and one non-artifact; Metal is 136 full, 57 blocked by CPU and
one non-artifact. The 58 unresolved rows remain in the queue. These code and
artifact classifications do not prove independent real-weight or Apple parity.

Draft [PR #147](https://github.com/ayutaz/vokra/pull/147) stages a patched
SpeechBrain 1.1.1 source-ready candidate for VoxLingua107 without the removed
TorchAudio compatibility shim. At exact clean VAST head `4c38e6f0`, frozen
Python 3.12 sync installed 37 packages and the offline model-free probe
validated the official `EncoderClassifier` import and required method names.
The production preflight returned RC 2 on unresolved package review; the
model-free worker returned `BLOCKED_OWNER_REVIEW`,
`PENDING_REVIEW_NOT_OWNER_SIGNABLE` and `NO_UPLOAD`. No checkpoint was acquired,
instantiated or executed. The exact-head workspace run passed 8,041 tests,
failed zero and ignored 101 across 306 suites; fmt and all-target Clippy
passed. The workspace log, Clippy log and blocked audit JSON SHA-256 values
are `b23570b22161f0cd6098d94cef532dfc27cb9e5e7de4cce3645b1e398c759ed7`,
`1ebbe83595d30433f5d65a14bf9d66f567c6ec96c3b287055ac5ebf27897987a`
and `9b95641328c5637980fd48bc752610a07e5e29830abc10783d5da74a36869b4a`.
Disposable VAST instance `53429267` and storage were destroyed; exact-ID
readback returned `instances: null` and the full instance list was empty.
Package/native and source/model/fixture owner reviews, independent real-weight
reference, VAST CPU parity, Apple CPU/Metal and public replacement remain open;
the Lang-ID row stays partial.

The paginated GitHub Dependabot open-alert API at the same `main` head returned
236 manifest alert records across 45 paths: 205 name a first patched version
and 31 do not. This supersedes the dated 258 and 342 snapshots for queue
sizing, without inferring why GitHub's asynchronous inventory changed or
equating manifest alerts with unique advisories. The open code-scanning API
still returned five Scorecard rules. The successful [Scorecard run
`36645717529`](https://github.com/ayutaz/vokra/actions/runs/36645717529)
on that head produced SARIF SHA-256
`fe7950d1a40ac51928290dff97c7090028b22ae453e572cf65a24a3a4241dcea`:
SAST reported 29/30 recent commits, Vulnerabilities reported 56 existing
findings, Code-Review reported 0/27 approved changesets, Maintained reported
repository age below 90 days, and CII-Best-Practices reported no registration.
The same-head CodeQL run `36645717641` was still in progress when this
Scorecard result was inspected. Re-run Scorecard after CodeQL completes before
deciding whether the SAST finding persists; no finding is dismissed by this
readback. Independent review, elapsed repository age and external Best
Practices registration cannot be manufactured by a dependency update.

At 2026-09-29 23:48 UTC, the same-head [CodeQL run
`36645717641`](https://github.com/ayutaz/vokra/actions/runs/36645717641)
completed successfully. The subsequently dispatched [Scorecard run
`36646909160`](https://github.com/ayutaz/vokra/actions/runs/36646909160)
also completed successfully on `4715e07f`; its downloaded SARIF SHA-256 is
`5f701095b34e339d249018a29c2a27fda2ccf4f634f8b65a7ac082518918f6fa`.
SAST still reports 29/30 recent commits, so the finding did **not** clear
merely by completing the latest CodeQL job. The other four Scorecard findings
remain unchanged. Do not attribute the missing historical commit to a cause
without commit-level evidence.

Draft [PR #152](https://github.com/ayutaz/vokra/pull/152) now stages a
decoder-only XCodec2 dependency candidate that overrides the official
`xcodec2==0.1.5` Torch/TorchAudio 2.5.0 hard pins to patched Torch 2.13.0
and stable-ABI TorchAudio 2.11.0 from the explicit PyTorch CPU index. Its
model-free VAST replay on instance `53434693` at clean code commit `e3e90572`
and `uv.lock` SHA-256
`d59f4541f665d3517bec3498e8b5b48fdc24aa0299885cac7ae0574f9fc1d9d4`
passed frozen Python 3.12 sync, lock check, and the official decoder-only
import guard. The instance and storage were destroyed; exact-ID lookup returned
`instances: null` and the full list `[]`. Follow-up head `305be9b4` records
this evidence. No real weight, independent CPU reference, Apple CPU/Metal
parity, GPU speed/quality comparison, or owner/legal sign-off was produced;
the PR remains draft and no public artifact changed.

## 2026-09-30 Catalog, security and verification readback (08:01 UTC)

The read-only HF metadata/card audit at clean `main`
`06240fe9b95bbd9ff3837c2d012c63e1b4abd148` still reports 194 public
repositories, 193 with GGUFs and 198 GGUF files. CPU code/artifact status is
136 full, 43 partial, 14 no-runtime-binder and one non-artifact; Metal is
136 full, 57 blocked-by-cpu and one non-artifact. The recovered summary
SHA-256 is
`42efbeee107584ba481691e00b80d1a2853fc5a0ed4743cb2fa144f79ac6ae97`.
No model payload was acquired or executed locally. All 58 unresolved rows
remain in scope, and these source/artifact classifications do not certify
real-weight CPU or Apple hardware parity.

The paginated GitHub open-alert API returned 211 manifest alerts: 183 name a
first patched version and 28 do not. Three are critical, all for
`GHSA-53q9-r3pm-6pq6`: XCodec2 alert #159 and CosyVoice2/3 alerts #318/#340.
This supersedes the earlier queue counts without dismissing an alert or
attributing closure to an unmerged candidate. XCodec2's isolated integration
candidate at `1c4eae72f118cc037bb1fe5e852906ae26e87e3b` incorporates this
main head, preserves the decoder-only patched CPU dependency boundary and
passes lock/documents-only checks. The recorded VAST import evidence remains
the earlier `e3e90572` run, not a fresh integrated-HEAD replay; primary
dependency/native-license review and real-weight parity remain open. The
candidate is not pushed. CosyVoice2/3's full-reference closures remain blocked
by their authenticated forbidden dependency path; version updates alone do
not close the composite or legal boundary. The owner's protected CosyVoice2
LLM manifest was not inspected or changed.

The separate Realtime implementation candidate remains clean at
`8dd161e1e9eb80d60fd2daeb42b4b336c0578b24`. Formatting, diff hygiene,
first-party-only dependencies, forbidden-symbol, native-boundary and
no-dynamic-loading gates pass. Its new real-packet consumer and Metal dispatch
still require remote compilation and actual authorized parity execution.
Default ignored tests, generic cache/API probes or compilation alone cannot
supply those numerical or Apple verdicts.

The reviewed single-rent controller (SHA-256
`288e2a8449a0469cab588bea34326ad1881e859f06b3636ba2e48f8841ca2228`)
created disposable worker `53492602`. Its API reported running, eight effective
CPU cores and an explicit SSH port mapping. The SSH probe returned zero but
did not emit its expected readiness marker. Bootstrap then stopped with
connection closed by the remote host, before any valid Rust verification
result. The controller's missing marker assertion is a diagnosed readiness
check defect; the cause of the remote host closing the connection is not
established. No test-pass count, Clippy, CUDA-feature, deny or audit result is
claimed. Bootstrap-log SHA-256 is
`4a0fdaafec585509a27ef7ba85194edd3adb2d7c60380728a702b84bcd129149`.
The controller attempted log collection, which failed with connection refused,
then destroyed this exact worker and its 200-GB storage. Destroy returned zero,
and strict individual readback returned `instances: null`; readback SHA-256 is
`817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057`.
A subsequent all-pages account readback returned zero instances and no next
page. No unrelated resource was operated on, and no automatic re-rent followed.
Fix and prove the readiness marker contract before the next remote attempt.

PR #138's refreshed head `0e74236ee04979471c129056ef1e47a1fb8b2fb3`
was still open at the latest observation, with 65 successful checks, one skip
and three pending checks. This is a dated queue observation, not a merge
verdict. PR #139 is merged at the main head above. CPU/GPU selection remains
based on unchanged same-input correctness plus whole-path measured speed;
mandatory native CPU and Apple CPU/Metal/no-fallback evidence is not waived.
No model artifact was published by this continuation.

## 2026-09-30 Merged security change and verification restart (08:13 UTC)

This supersedes the PR/check and alert observations above without replacing
their dated evidence. PR #138 merged at `2026-09-30T08:00:10Z` as
`740cd3b38d2bdb659c4db74f9be0b4bb1bece2e2`; the remote `main` reference
and local main now identify that commit. A later complete check-run readback
for its head `0e74236ee04979471c129056ef1e47a1fb8b2fb3` found 69 successful
checks, one skipped check and no unfinished check. The paginated open-alert
readback now reports 209 manifest alerts: 182 with a first patched version,
27 without, and three critical alerts. These are current queue observations,
not proof of numerical compatibility or closure of unmerged candidates.

The readiness controller was corrected and reviewed at SHA-256
`8a6c583b9848e79b058abbd284dcd357b2e4c667514230577c59ad8b124e9bf2`.
It now requires both a successful SSH exit and an exact remote nonce, then
verifies the transferred bundle SHA-256 before bootstrap. Syntax, shellcheck
and model-free mock checks passed; the mocks are not remote Rust evidence.
The account inventory was empty before the separately reviewed, single-rent
restart on disposable worker `53494404`. This different host reports 16
effective CPU cores, and the controller sets `CARGO_BUILD_JOBS=16`. Its actual
readiness nonce and bundle digest passed, and provision is running. The clean
Rust target remains `8dd161e1e9eb80d60fd2daeb42b4b336c0578b24`; no Rust test,
Clippy, CUDA-feature, deny, audit, real-weight or Apple verdict is yet claimed
for this restart. The controller must recover bounded evidence and destroy
the exact worker and its storage at termination. No model acquisition or
publication is part of this job.

The XCodec2 follow-up is a model-free license/native-evidence collector,
not an execution approval. It must bind the actual Linux dependency closure,
distribution hashes, primary license bytes and ELF payloads. Metadata-only
MIT classifiers, another model's scoped NumPy exception, historical VAST
imports, or a successful factual collection cannot clear its owner-review,
real-weight parity or `NO_UPLOAD` boundary. The public catalog's 58 unresolved
rows and the separate artifact/security/Apple work remain in scope. GPU
selection still requires same-input correctness and a faster measured whole
path; it does not waive the mandatory CPU or Apple/no-fallback checks.

### Termination of the restarted verification (08:15 UTC)

The restart above reached actual Rust compilation but stopped at
`workspace-test` with exit code 101. The unmerged candidate's integration test
`parity_vibevoice_realtime_streaming.rs` imports
`VibeVoiceRealtimeTokenizer` from the parent module, whereas the public type
is under `vibevoice_streaming::tokenizer`. This is a compile failure, not a
numerical-parity failure or a hardware result. Workspace tests did not
complete; all-feature Clippy, the focused CUDA-feature test, deny and audit
were not reached. The recovered full workspace log SHA-256 is
`f66c5659d56a0c7b0be4953cbe64f600c771141f56eb07e9902ae6ad6e0e2463`.
Eight recovered evidence files passed their remote/local SHA-256 comparison;
the final HEAD is the exact `8dd161e1` target and its observed worktree status
is empty. The bounded import correction is assigned for implementation review
before another fixed-head verification.

Worker `53494404` and its 200-GB storage were destroyed. The destroy command
returned zero, strict individual readback returned `instances: null`
(SHA-256 `817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057`),
and a subsequent all-pages account readback reported zero instances with no
next page. There is no retained VAST worker or storage from this restart.
No model payload, owner approval, numerical pass, artifact upload, or public
catalog promotion is inferred from these setup and compilation checks.

### Corrected-head Rust verification (08:32 UTC)

The bounded tokenizer import correction is committed at clean candidate head
`46948802f5fc939d0ba038517cd57504f6e50b87`. A separately reviewed
single-rent controller (SHA-256
`2406290da597c7ff305124ce7e29dafb0a53d65409acac16e9fec7076ae972a8`)
verified that exact head on disposable VAST worker `53495557`. Workspace tests
completed 324 suites with 8,142 passed, zero failed and 108 ignored. Ignored
tests supply no verification verdict. Workspace/all-target/all-feature Clippy
with `-D warnings`, the focused CUDA-feature Realtime test (68 passed, zero
failed, five ignored), locked cargo-deny and cargo-audit each exited zero.
Compilation and feature tests are not measured CUDA execution or numerical
parity with an independent real-weight reference.

All 16 recovered evidence files matched their remote SHA-256 values. The
observed worktree status was empty and final HEAD matched the target. Raw
workspace-log SHA-256 is
`3122f4be1d2dd7faa9429d74dec55686406146fc190263294683431def414ac2`;
Clippy-log SHA-256 is
`2f6434160e02910b60902cbb0995cc44558e3b7197f3035e11cb7f5e40cd400b`;
the recovered packet checksum-list SHA-256 is
`85b34615e38db28fd7a8ca1202bf97bc5882320c86dba250a928ceca5f36a1a8`.
Worker `53495557` and its 200-GB storage were destroyed after recovery.
Strict individual readback returned `instances: null`; a subsequent paginated
account readback found zero instances and no next page. No model payload was
downloaded or executed in this job, and no cloud worker was retained.

The metadata/card-only HF audit against code identical to main
`740cd3b38d2bdb659c4db74f9be0b4bb1bece2e2` still reports 194 public
repositories, 193 GGUF repositories and 198 files: CPU 136 full, 43 partial,
14 no-runtime-binder and one non-artifact; Metal 136 full, 57 blocked-by-CPU
and one non-artifact. Its summary SHA-256 is
`42efbeee107584ba481691e00b80d1a2853fc5a0ed4743cb2fa144f79ac6ae97`.
The 58 unresolved rows are unchanged. Same-input correctness and faster
whole-path measured performance remain prerequisites for GPU selection;
native CPU, Apple CPU/Metal/no-fallback and publication gates remain open.

### PR integration readback (08:46 UTC)

Documentation PR #183 merged after 56 successful checks, one skip and no
unfinished or failed check. Its squash commit is
`a76f8c5317a4284772a4045d382eb86b385094ba`; local main and origin/main
were fast-forwarded to that head. Draft PR #182 now points to the verified
`46948802` candidate, and its title/body distinguish Metal dispatch and
packet-consumer implementation from pending real-weight and Apple evidence.
Fresh CI on that candidate is running; the draft has not been promoted or
merged. No publication or model-row completion follows from these PR changes.

### Kyutai incremental-LM verification failure (09:09 UTC)

Disposable VAST worker `53500171` verified the clean, unpushed Kyutai
candidate `1ec30f6c2733d33664ec6df05eb23b7dc60d206c` with 16 effective
build jobs. Its API hardware snapshot identifies an AMD EPYC 7713 and RTX
A4000; selected CPU ISA/kernel flags were not captured and remain a diagnostic
gap. No model payload or independent upstream reference was executed.

Workspace testing stopped with exit 101 in the models crate: 3,206 tests
passed, one failed and 23 were ignored in that test binary. The failing
`kyutai_stt::streaming_lm::tests::step_matches_full_component_before_and_after_context_boundary`
requires exact equality between one-frame incremental and full-component
synthetic logits. The logged values differ at approximately 10^-7 scale.
This is an unresolved self-consistency failure, not evidence of flakiness,
independent real-weight parity or a justified tolerance change. Clippy, the
focused CUDA-feature test, deny and audit were not reached. Diagnosis must
record the failing frame/bin and compare actual scalar/default kernel paths
before any comparison or numerical bound is changed.

All eight recovered evidence files matched their remote SHA-256 values;
final HEAD matched the target and observed worktree status was empty. The
raw workspace-log SHA-256 is
`e8cfca881c9ca2d7bdabb86b1972244282d1e91a8dda642b9d25b5d5c7d69525`,
and the local packet checksum-list SHA-256 is
`a77b8489e30f5f225f8345b10faaab936f101005e57ab81f5fe0af5dac9a4da6`.
The worker and its 200-GB storage were destroyed after recovery. Strict
individual readback returned `instances: null`; the subsequent all-pages
account readback reported zero instances with no next page. No model row,
GPU performance selection, Apple/Metal gate or publication is promoted.

### Security and diagnostic preparation readback (09:26 UTC)

The paginated GitHub open-alert API readback reports 209 Dependabot alerts,
all in the pip ecosystem: 182 have a published first patched version and
27 do not. GitHub main at this observation is
`a76f8c5317a4284772a4045d382eb86b385094ba`. This dated observation
supersedes earlier queue-sizing snapshots, not their historical evidence;
it is not a numerical-validation verdict or a reason to dismiss alerts.

The clean, unpushed Kyutai diagnostic candidate is
`a9f462465f32ce43535f1e84e507c6b54b7552d4`. It preserves the strict
assertion and production arithmetic, adding test-only frame/bin, bit/ULP,
CPU/ISA and cache-position reporting. Listed operation shapes are candidates,
not measured first-divergence taps. Focused scalar/default VAST diagnosis
remains pending; no tolerance or model-completion verdict changed.

A separate reviewed hook candidate,
`f6cea6557484564ac4645abbacf07eb9ffe17e0e`, distinguishes literal Git
index staging from model execution. Its complete dispatcher self-test passes,
including the local-model, memory and protected-path guards. The active
maintainer hook was not changed by this preparation. Account-wide VAST
readback reports zero instances and no next page; no worker was retained.

### Same-worker Kyutai ISA diagnosis (09:50 UTC)

The frozen, clean candidate
`107de0b30ff06389da1681f11c4ae634bab913a4` was diagnosed on one
disposable AMD EPYC 7713 VAST worker, `53505648`, with 16 build jobs.
Both focused processes used the same compiled test binary and synthetic
fixture; the ordinary process explicitly unset `VOKRA_CPU_ISA`, and the
second process set it to `scalar`. Actual runtime observations were `Avx2`
and `Scalar`, respectively. The recorded feature tree includes the CPU
default, parallel and SIMD-transcendental features.

The ordinary path exited 101: two tests passed and the strict full-versus-step
test failed. Its first differing logit was frame 1, index 0:
`-1.3057351` (`0xbfa72254`) versus `-1.305735` (`0xbfa72253`), a one-ULP
difference with absolute delta `1.1920929e-7`. The scalar path exited zero:
three tests passed, none failed or were ignored. This narrows the synthetic
self-consistency defect to an ISA-sensitive route; it does not identify the
first divergent operation, justify relaxing equality, or establish independent
real-weight parity. QKV, QK, softmax, weighted-V and output-stage taps remain
the next diagnostic work. Production arithmetic and the assertion are unchanged.

All ten recovered packet files matched their remote SHA-256 values. Final
HEAD matched the target and observed worktree status was empty. Default and
scalar log SHA-256 values are, respectively,
`b0d2d1c8b32e548732c8ff306fb380cc55d678fd75763a7924140402f9197b43`
and `5b6f6b8d4dd230e0d95662eaed78ee5eb73d0086a02bc8fdc8394005adcef181`;
the remote checksum-list SHA-256 is
`b6698ebb5294f5a55f486b0b07f42aeb5b399fef45a5267a5a4b78225613db5d`.
The worker and 200-GB storage were destroyed. Individual readback returned
`instances: null`; account-wide readback reported zero instances and no next
page. Controller success means diagnosis, recovery and cleanup completed,
not that both test paths passed. No model payload, GPU execution, upstream
reference, Apple run or upload occurred.

The separate reviewed Mimi ELU slice is committed, clean and unpushed at
`336de430e4c9d40bb81906cd09381b43797c22de`, based on merged main
`ef64003b453840cf23516567d13beef7ac1f53e4`. It preserves the existing CPU
`exp_m1()` bit behavior and dispatches non-CPU activation through the existing
Compute ELU seam, with preallocated scratch and explicit unsupported errors.
Static checks passed; Rust compilation/tests and Metal device execution remain
pending. QK, weighted-V, RVQ and other host arithmetic are not closed by this
slice. The catalog denominator remains 58 unresolved rows. The GPU execution
policy still requires same-input correctness and faster measured whole-path
time, including transfers; mandatory CPU and Apple/no-fallback gates remain.

### XCodec2 model-free collection and stage-trace preparation (10:13 UTC)

The frozen clean XCodec2 collector at
`097c5884053092d07d758e7a60854a4c9ed95318` ran on disposable VAST
worker `53508157`, using the reviewed offer `51324724` (eight effective
CPU cores, 128,983 MB advertised RAM, 200-GB requested disk). The collector's
Linux Python-3.12 environment exactly matched all 62 external lock rows;
the reachable closure includes one additional first-party project row.
It collected 56 package/license evidence rows and reported six factual
failures, so its verdict remains `BLOCKED_FACTUAL_COLLECTION` / `NO_UPLOAD`.
These are not owner approvals or model-parity results.

The failures were repeated-download `FileExistsError` for the locked Antlr
and XCodec2 source archives, unsupported wheel `.data` relocation for Dill
and SymPy, the bounded publisher-RECORD selection for setuptools, and the
archive member-count bound for Torch. Fixing collection mechanics must not
approve unverified source-built payloads, infer a license grant or silently
relax archive safety limits. All license rows remain unresolved; NumPy's
bundled GPL-with-GCC-exception/LGPL components still require disposition.

The isolated collector wrapper reported no Torch/XCodec2 model-code imports
before or after collection. The setup phase's sdist build hooks are explicitly
`UNTRUSTED_NOT_ASSESSED`; the collector guard is not proof that setup executed
no package code. No model checkpoint, audio, forward, Cargo or upload task was
requested or recorded in this run. All 15 recovered manifest entries passed
local SHA-256 verification, final HEAD matched the target and observed Git
status was empty. Report SHA-256 is
`ea811881c0c20faa47a140c17e4095d1ab9e006def0cae25a14b598571cc3b32`;
manifest SHA-256 is
`f2f4febbbb1aff8ddc9e991e60a520c237714040ece44453b69eae1dfea7660a`.

The collector and remote shell both exited 2. The historical controller
`c1af86a7938eaf78501c353915d439185d53587103940293b813c5e731af8957`
incorrectly returned zero because cleanup overwrote its global exit-code
variable. Its zero is not a green verdict. The subsequently reviewed
controller preserves the original exit with an offline trap regression;
the recovered report and raw exit files remain authoritative. Recovery,
destroy and strict individual readback each succeeded; the worker and its
200-GB disk were destroyed, and the all-pages account readback reported zero
instances with no next page.

The separate clean Kyutai candidate
`a987c7a16ae5f9c5400db3a0efad73637f80956f` adds test-only copies immediately
after the actual full/streaming operations, rather than a diagnostic mirror.
Replay-integrity checks reject stage interpretation if replayed logits drift
from the original outputs. Static checks passed; the new focused VAST
stage-trace run remains pending. Production arithmetic, strict equality and
numerical bounds are unchanged. Neither slice closes an unresolved public
row or an Apple/Metal/publication gate; the unresolved denominator remains 58.

### Actual Kyutai first-stage readback (10:22 UTC)

The clean candidate `a987c7a16ae5f9c5400db3a0efad73637f80956f` was
diagnosed on disposable EPYC 7713 worker `53509305`, with 16 build jobs.
For the same synthetic fixture and compiled binary, default `Avx2` again
reported two passes and one strict failure; `Scalar` reported three passes
and no failures or ignores. The actual full/streaming taps now locate the
first differing operation at frame 1, layer 0, head 1 softmax. Its visible
probabilities were `[0.74857235, 0.25142762]` and
`[0.74857235, 0.25142765]`; the second element differs by one ULP,
`2.9802322e-8` (`0x3e80bb1f` versus `0x3e80bb20`). The replay-integrity
checks did not report drift. Earlier compared QKV/cache/QK/masked-score
stages matched for this fixture. This replaces the earlier operation-shape
hypothesis with a measured first-stage observation, not independent
real-weight parity or proof that all other inputs have identical GEMMs.

The existing AVX2 softmax uses vector exponentials/reduction for a full
eight-column row and scalar tails for a two-column streaming row. That is
a candidate explanation, not a measurement separating the exponential
approximation from reduction order. Strict equality, numerical bounds and
production arithmetic are unchanged in this diagnostic head. The separate
all-feature Clippy leg exited 101 with hygiene lints, including test-only
trace borrowing and production-only unused enumeration; it was not green.

All 12 recovered manifest entries passed local SHA-256 checks. Actual final
HEAD matched the target and Git status was empty. Default/scalar log hashes
are respectively
`63c48e5ab2d345f2c33b65c1187641003c6de5581e6621138c346f4865ef29cf`
and `2079359927b9124f90483a3c55b188de6405e8214d5488a06ff6467f8278a2dc`;
the checksum-list hash is
`71c2d8674314e259699b2338bd82e4672d5c3cf2c2c6ada60db3619672653e`.
The worker and 200-GB disk were destroyed; strict individual readback returned
`instances: null`, and the all-pages account readback reported zero instances
with no next page. Diagnostic-controller success is not a passing-test or
Clippy verdict. No model payload, upstream reference, actual GPU/Metal run or
publication occurred.

The XCodec2 collection-mechanics follow-up is separately committed, clean
and unpushed at `6185ea8e63382964d9b93be35b06be38d5c83383`. Model-free
tests verify hash/size-bound archive reuse, partial-download cleanup and root
publisher-RECORD selection without discarding vendored RECORD payloads. Its
next VAST collection remains pending. Source-built payload identity, wheel
`.data` relocation, the Torch archive bound and owner/license decisions
remain blocked; no collection or model row is promoted by the local tests.

### Mimi control-plane preflight and Kyutai CPU candidate (2026-09-30)

The reviewed Mimi candidate remains clean at
`336de430e4c9d40bb81906cd09381b43797c22de`. A full-verification controller
with SHA-256
`c11e321909acfa951a269aadff0d0185cb4fa1cc9a22b7e14dc83731418cc420`
created disposable worker `53511195` at 10:31 UTC. The exact owned worker
reported intended/cur state `stopped`; actual state first reported running
and later exited. No SSH verification or Rust test started. Its readback
also distinguishes eight effective CPU cores from 32 physical cores, and
uses `disk_space=200` while `disk` is null. These facts exposed controller
preflight defects, not a model or test failure. The manager terminated the
controller (exit 143); its EXIT cleanup destroyed the worker and disk.
Individual strict readback returned `instances: null`, the exact-label
readback was empty, and the subsequent all-pages account readback reported
zero instances. The individual readback SHA-256 is
`817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057`.
Full Mimi workspace/Clippy/deny/audit verification remains pending. Linux
verification cannot establish Apple/Metal execution or complete no-fallback.

Kyutai's separate clean candidate is now
`9950e896a8554f8161a0148a53ac5c77c6a64475`. It includes Clippy hygiene and
CPU-only visible-context softmax dispatch through the existing native-ISA
`Compute` path. Non-CPU backends retain batched softmax; no scalar forcing,
CPU fallback, seed change or tolerance relaxation was added. Synthetic
self-consistency coverage includes context 1/3 and widths 8/9/12 before and
after window saturation, with explicit masked zeroes. Root review,
formatting, diff hygiene and five static repository gates passed. The
original strict eight-frame streaming test is unchanged; its default-AVX2
rerun and independent real-weight reference remain pending. Neither this
candidate nor the failed preflight closes any of the 58 unresolved rows.

### Mimi attention candidate and active full-verification readback (11:01 UTC)

The replacement single-worker controller is frozen at SHA-256
`40601ecf9099b3c14eaf66380e3d3846be64b5f3b11f77944f01cdbef239408d`.
Its disposable worker `53512626` verifies the unchanged ELU candidate
`336de430e4c9d40bb81906cd09381b43797c22de`, with 14 allocated CPU cores
and 200 GB disk. The Mimi-focused result is 40 passed, zero failed and zero
ignored; all-features/all-targets Clippy with warnings denied also exits zero.
At 11:01 UTC the serial workspace regression is still active, with recent
log updates and an active test process. Workspace, deny, audit, packet
recovery and strict destruction are not yet final results. This Linux
run is neither an Apple/Metal verdict nor independent real-weight parity.

A distinct, reviewed attention candidate is clean at
`6328ad62aeafd3483221cd9ec04d56fbbc879168`. Non-CPU causal QK and AV
contractions dispatch through `Compute::gemm_f32`, using preallocated
chronological ring scratch; the CPU accumulation order is retained. New
synthetic tests cover scratch reset and errors, unsupported backend
selection, and device-gated batch/step ring-wrap and reset comparisons.
Formatting, diff hygiene and five static gates passed, but this new HEAD
has not been compiled or tested remotely. The active `336de430` run must
not be attributed to `6328ad62`. LayerScale residual and RVQ encoder host
paths, independent reference and Apple validation remain open. No public
model row is promoted and no artifact is uploaded by these changes.

### Recovered Mimi ELU full-verification packet (11:05 UTC)

The `336de430` run above is now final: focused Mimi tests, all-features /
all-targets Clippy, workspace tests, deny and audit all exit zero. The
workspace log contains 323 result summaries, 8,106 passed tests, zero
failures and 105 ignored tests; ignored tests are not passed tests or
real-weight/hardware proof. Deny reports an unmatched pre-existing
`libfuzzer-sys` license-exception warning, with advisories, bans and licenses
otherwise successful. Audit scans 22 lockfile dependencies without a
reported vulnerability.

All 18 packet entries were recovered and independently SHA-256 checked.
The actual final HEAD is `336de430e4c9d40bb81906cd09381b43797c22de`;
observed worktree status and diff-check output are empty. Packet manifest
SHA-256 is `f3fa28666744869483f7309e40c8124633f262433acbb504b028bbab322483e6`;
workspace log SHA-256 is
`7f3bdbbd3fb502390c34bba6770e081d976e9accf1cf59633cdf51d55537ad16`.
This is Linux code regression evidence only, not independent model parity,
Apple CPU/Metal coverage or verification of the newer attention HEAD.

Controller exit is zero; collection, destroy and readback all exit zero.
Worker `53512626` and its disk were destroyed. Individual strict readback
returns `instances: null`, and exact-label readback is empty. A fresh
all-pages account readback also reports zero instances (SHA-256
`0bae59a551bcb5491b24cb982cdb029b20509d86fa38b75209f12ca13415f63c`).
No model row, legal gate or public artifact is advanced by this run.

### Kyutai visible-softmax rerun and Mimi PR readback (11:22 UTC)

The reviewed softmax controller SHA-256
`e99b91e62186114850dba43c49d93ae7430d17853431ef15e0021c43fb4dd7a1`
ran exact clean Kyutai HEAD `9950e896a8554f8161a0148a53ac5c77c6a64475`
on disposable EPYC 7713 worker `53516739`, with 16 allocated CPU cores and
200 GB disk. Default AVX2 remains red: two passed and one failed (exit 101).
Scalar passes three tests and the visible-context softmax self-consistency
test passes one. Neither is independent upstream/real-weight parity.

The first recorded production-stage divergence is now frame 2, layer 0,
head 0, raw QK element 1: full `0.16708244` / bits `0x3e2b17a9`, step
`0.1670824` / bits `0x3e2b17a6`, absolute delta `4.4703484e-8` (three ULPs).
Earlier recorded taps match for this fixture. Full QK geometry is `[8,8,4]`
and step geometry `[1,3,4]`; these measurements do not establish independent
model correctness or justify a relaxed bound. The strict eight-frame test,
seed and equality assertion remain unchanged. All-feature Clippy also exits
101 for `option_as_ref_deref` at the diagnostic trace reborrow. Both failures
remain open; the next source candidate aligns CPU visible-window contraction
geometry without forcing scalar or changing non-CPU dispatch.

All 16 packet entries passed SHA-256 checks, with actual matching HEAD and
empty observed status. Manifest SHA-256 is
`33437c12429e8a665937f0b90a1fd3d914afe7460bdd5bcee21df1de96bfffdb`;
default log SHA-256 is
`07161a3d964a403ae2d77740ee78138b96f9520883b9162c33c14126abf3ff41`.
The controller retains exit 1; recovery, destroy and strict individual
readback all exit zero. Worker and disk are destroyed, and a fresh all-pages
account readback reports zero instances. No gate is promoted by the red run.

The independently verified Mimi ELU HEAD `336de430` was pushed unchanged as
[draft PR #185](https://github.com/ayutaz/vokra/pull/185). Its fresh CI is
pending at this readback; newer attention/LayerScale candidates are not part
of that PR or its VAST verdict. No PR was merged and no artifact was uploaded.

### Reviewed CPU-attention and Metal LayerScale candidates (11:32 UTC)

The CPU visible-window attention candidate is committed and clean at
`1dacaa14c77b07eba75df5d3ad4539d2ccf5afc6`, based on the red `9950e896`
run above. CPU full-forward QK and probability-times-V now use the same
one-row visible-window GEMM geometry as the incremental path, with checked
reusable scratch allocated outside query loops. Non-CPU batched dispatch is
unchanged. The original eight-frame seed and strict equality are unchanged;
additional synthetic context/window self-consistency coverage and the Clippy
trace-reborrow fix are included. Formatting, diff hygiene and five static
gates pass. Compilation and default-AVX2/scalar reruns are still pending;
this commit does not yet close the measured failure or establish independent
upstream or real-weight parity.

The distinct Mimi LayerScale candidate is committed and clean at
`e49ca0c51244d9bc47e7af0797ddc376f2747525`, based on `6328ad62`. Its learned
channel-scale/residual update dispatches through a checked Compute seam and
a dedicated Metal kernel, with complete shape and dimension checks before
caller-output mutation and explicit errors for uncovered backends. CPU
operand order is preserved, including immediate empty-shape no-ops. The
Mimi coverage registry requires the new operation; synthetic CPU shape and
operand-order tests and Apple-device-gated tests are added. Formatting, diff
hygiene and five static gates pass. Linux regression, Apple execution,
independent reference and speed measurements remain pending. Neither this
HEAD nor the attention parent is included in PR #185 or its `336de430` VAST
verdict; RVQ encoder and other host paths remain open.

At the latest PR #185 readback, Linux/macOS/Windows test jobs have passed,
while Metal and other CI jobs are still running. The PR remains draft and
unmerged. VAST has no retained worker from the preceding runs. New controllers
are under review, not yet rented. No model row or artifact is promoted.

The user's GPU preference applies to future work when the selected backend
is correct and faster end-to-end, including transfer/setup costs. It does not
authorize replacing required CPU ISA parity or Apple CPU/Metal/no-fallback
checks with a GPU result, inventing a speedup, silently falling back to CPU,
or executing models on the maintainer PC.

### Mimi ELU merge, LayerScale regression and Kyutai setup readback (11:53 UTC)

PR #185's exact head `336de430e4c9d40bb81906cd09381b43797c22de`
finished with 78 successful checks, five intentional skips, no failures and
no pending checks. All 16 required checks passed. After the reviewed body was
updated and the draft flag removed, the PR was squash-merged without an
administrator bypass at 11:44:39 UTC as
`81d72f2317a3bd2462cb7abbca15ac55b7518d56`. This accepts the ELU dispatch
component only; it does not establish complete Mimi Apple or real-weight
parity.

A separate clean VAST replay of
`e49ca0c51244d9bc47e7af0797ddc376f2747525` completed successfully on the
AMD EPYC 7C13 worker `53519879`. The focused Mimi selection passed 75 tests
with no failures or ignored tests. All-feature/all-target Clippy, serial
workspace tests, deny and audit each exited zero. Workspace output contains
323 result summaries, 8,111 passed tests, no failed tests and 105 explicit
ignores. The five named attention and LayerScale regressions were present
and passed; this result is not attributed to PR #185 or a later rebased HEAD.
All 18 recovered packet entries passed independent local SHA-256 checks;
actual final HEAD matches and observed worktree status is empty. The manifest
SHA-256 is
`ae73e5160116df9311cd70c2b13452c0f6ebc770748e7492c43364c9b36d0cfd`.
Controller exit, evidence recovery, destroy and readback all exited zero.
Individual readback returned `instances: null`, and all-pages account
readback returned zero instances with `next_token: null`. Worker and its
200-GB disk were destroyed; no retained handoff was created. Linux-only
regression does not close Apple execution, independent reference, real-weight
parity, RVQ encoding or speed measurement gates.

The distinct `1dacaa14` Kyutai attempt on worker `53520465` did not start
Rust verification. The instance API reported the owned instance running,
but exposed the direct SSH port under `ports["22/tcp"]` while
`direct_port_start` was -1. The reviewed controller accepted only the legacy
field and failed closed before SSH/bootstrap. Controller exit remains one;
no test pass or numerical conclusion is inferred. Destroy and strict
individual/label readbacks passed, including `instances: null` and a
terminal empty label list. The worker and its disk were destroyed. A bounded
endpoint-parser correction and offline regression fixtures are under review
before any new rent; proxy guessing and automatic rerent are not used.

The XCodec2 static sdist inspector was reviewed and committed cleanly as
`018c01742f29441bf335960b1926739eb033943b`. Its isolated, offline self-test
passed; it neither imports nor builds upstream code. Static archive findings
do not approve a build backend, resolve uv markers or establish source-built
wheel provenance. A separate model-free byte audit authenticated the exact
locked setuptools 84.0.0 wheel (818,216 bytes, SHA-256
`51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670`).
Its embedded `setuptools/_vendor/autocommand-2.2.2.dist-info/METADATA`
declares LGPLv3, and its primary LICENSE contains LGPL version 3, with
SHA-256
`ade78d04982d69972d444a8e14a94f87a2334dd3855cc80348ea8e240aa0df2d`.
Top-level MIT metadata therefore does not approve the complete package.
No package was installed, imported or built; no XCodec2 real-weight job was
started. The legacy isolated-build backend remains unbound/unapproved.

The live coverage classification remains 136 code/artifact-full and 58
unresolved public rows. No source regression result above promotes an
independent-reference or Apple gate, and no public artifact was uploaded.

### Kyutai same-CPU regression and queued Mimi integration (12:22 UTC)

The endpoint-parser correction passed independent offline regression checks.
The next Kyutai setup attempt, worker `53522056`, reached SSH/bootstrap but
failed the explicit AMD EPYC 7713 CPU-model guard: the manager had selected a
Xeon offer without checking that requirement. This was a host-selection error,
not a Rust test result. The guard was not weakened. The worker and its 200-GB
disk were destroyed, with successful strict individual and label readbacks.

A subsequent replay of the same frozen Kyutai head
`1dacaa14c77b07eba75df5d3ad4539d2ccf5afc6` on worker `53522575` observed
the required AMD EPYC 7713 CPU. Default execution explicitly reported Avx2;
forced Scalar execution explicitly reported Scalar. Each selection passed
four tests with zero failures or ignores. The additional context-boundary
selection passed one test, and all-feature library/tests Clippy exited zero.
The original strict synthetic step/full assertion was not relaxed. These
results close that focused regression, not independent upstream, real-weight,
workspace-wide or Apple parity. All 16 packet entries passed independent
SHA-256 verification. Final HEAD matched and observed worktree status was
empty. Manifest SHA-256:
`6e96a73409891c6bb7a7e164fd8d6ee27e9c0e1f9251e4986a3881e54815a35b`.
Controller exit, recovery, destroy and strict individual/label readbacks
passed; the worker and its 200-GB disk were destroyed. A later all-pages
account read returned zero instances and a terminal null next token.

The reviewed Mimi RVQ encode dispatch was committed as
`e5dd3b703e906d538aab03bbd4d88c6bdee2cdc7`. It adds a genuine Metal
distance/argmin/residual route, deterministic tie ordering and fail-closed
shape/backend checks. The attention, LayerScale and RVQ slices were then
integrated on the actual main base
`81d72f2317a3bd2462cb7abbca15ac55b7518d56` as clean head
`a84817d43a01bac0fb3d93c5f58d805d214aa72d`. Its complete tree matches
the frozen RVQ source; independent formatting and static gates passed.
The reviewed full-regression controller is bound to this exact head and to
eight required named focused tests. Its SHA-256 is
`28c8d1805244c5328dad472275e8cd56dae1eff1574767b6a1b43bf87168090e`.
The first rent request returned `no_such_ask` because that offer was no longer
available. Both exact-label and all-pages account readbacks were empty; no
test or cleanup success is inferred from an unknown instance ID. A separately
reviewed fresh offer is being used for the next single-rent attempt. Full
Linux and actual Apple results at `a84817d4` are still pending; the earlier
`e49ca0c5` packet is not relabelled as evidence for this integration.

Source-only investigation also confirmed that the existing upstream Mimi
fixture exercises eight-codebook RVQ decode, not the complete 32-codebook
STT encode/decode contract. Independent real-weight codes/PCM, Apple CPU,
Metal/no-fallback execution and transfer-inclusive speed measurements remain
open. GPU use is preferred only where correctness and measured end-to-end
speed support it; mandatory CPU verification is retained. No model was
executed locally and no public artifact was uploaded. The live classification
remains 136 code/artifact-full and 58 unresolved public rows.

Historical Mimi evidence must not be confused with the missing current
rerunnable gate. The
[July 16 campaign report](../bench-baselines/m1-real-weight-eval-2026-07-16/report-campaign2.md)
does record an independent full CPU comparison: 4,384/4,384 codes matched,
including semantic and acoustic books, and same-codes PCM maximum delta was
approximately `3.67e-6`. Its reported baseline is `92dbc92`; these historical
observations are preserved, not attributed to the new Metal integration or
to the authenticated STT companion without checking exact input identity.
The current gap is first-class reproducible full encode/decode replay at the
frozen implementation/input hashes, followed by actual Apple/Metal and speed
evidence. Existing Python package names or a historical workflow do not by
themselves approve a new source revision or its wheel/native license closure.

### Mimi integration regression and derived-source audit (12:52 UTC)

The queued Mimi replay above subsequently completed at exact clean head
`a84817d43a01bac0fb3d93c5f58d805d214aa72d`. Worker `53524722` passed
78 focused Mimi tests, with no failures or ignores, and the serial workspace
run reported 8,114 passed, zero failed and 105 ignored across 323 result
summaries. All-feature/all-target `vokra-models` Clippy, deny and advisory
audit exited zero. Final HEAD matched, observed status was empty and all
18 recovered packet SHA-256 checks passed independently. The packet manifest
SHA-256 is
`bd2009c862476de199e06ee1212311c1ca0edfce56eac6946d2df4c8a7c3afc5`.
The worker and its 200-GB storage were destroyed after recovery; individual
readback returned `instances: null` and the exact-label list was terminal and
empty. The reviewed source was pushed as
[draft PR #186](https://github.com/ayutaz/vokra/pull/186); its new CI is pending.
This is Linux regression evidence, not Apple Metal execution, independent
real-weight parity or a public-row completion verdict.

The XCodec2 derived-sdist builder was reviewed and committed cleanly at
`f260f8842baec412580565f9205fbac2461e9342`. A separate source checkout on
the same owned worker received the reviewed Git bundle and the two small,
hash-pinned ANTLR 4.9.3 and XCodec2 0.1.5 source archives. Initial static
verification stopped because `uv` was absent from PATH; resuming with the
installed absolute `uv` path passed all six named standard-library tests,
the builder self-test and repackaging of both actual archives. No upstream
setup/backend, package import, installation, model or upload was executed.
All four recovered output hashes matched independently:

- ANTLR derived wheel:
  `06e39bb3d804f5e14ec6dfcb2be2cf125419d700bbea43e5a52eb4269c1aa7e2`.
- ANTLR provenance:
  `8f8de04c9713bd9752a6fdbd1f22c5da1d548b0bd872ca061fc04d6c2026fe41`.
- XCodec2 derived wheel:
  `aa20f3975710b221a9f7f33e7a1be80a2f28b1f0093acba36a4df714ef5d5b78`.
- XCodec2 provenance:
  `5af99ab73f76c1976db517fb2e360769dc1bb370fd61bd79ebcbfb1e4516c1fa`.

These are locally derived package distributions, not official wheels or
model artifacts. Their provenance remains `DERIVED_NOT_OFFICIAL` /
`UNAPPROVED_NO_EXECUTION` / `NO_UPLOAD`. This closes only the bounded,
source-byte-preserving repackaging check. The runtime dependency closure and
the setuptools vendored LGPL finding above remain unapproved. The side
checkout was removed with the disposable worker; no XCodec2 real-weight run
or public artifact replacement is inferred.

The previously reviewed Kyutai streaming stack was replayed onto actual main
`81d72f2317a3bd2462cb7abbca15ac55b7518d56` as clean head
`27060dd2b8eceb744fb338fb1783caab9c5717b8`. Static checks passed. Its
independently reviewed full-verification controller has SHA-256
`b385bea61ba85d84370764bb5f22c544a98892d02b8f625b94cb42e4d7fff67d`;
it requires the actual EPYC 7713 CPU, exactly four named tests under observed
Avx2 and forced Scalar, all six verification exits zero and authenticated
recovered evidence. Red evidence remains recoverable, and both success and
failure paths destroy the owned worker. A fresh single-rent attempt is now
pending; no result from the earlier focused head is relabelled as full proof
for this integrated head.

The current full Mimi upstream-dump proposal remains an uncommitted,
preparatory contract, permanently blocked before checkpoint access or
third-party import. Canonical owner, dependency and native-payload gates and
first-class consumer verification are not yet closed. No model ran locally,
no HF token was transferred and no public artifact was uploaded. Coverage
classification remains 136 code/artifact-full and 58 unresolved public rows.

### Current security API readback (12:52 UTC)

A fresh GitHub API readback at `2026-09-30T12:52:14Z` contains 209 open
Dependabot alerts across 37 distinct manifests: 182 have a patched version
and 27 have none. Severity counts are three critical, 31 high, 82 medium
and 93 low. The complete paginated readback SHA-256 is
`e866594e13746f816b71130c43b4e4f9b0e08b2415083359a28aaf022e495a40`.
This is a new dated snapshot; the original 258/226/32 counts remain history,
not current completion evidence.

The three critical alerts concern Torch versions below 2.6.0: #340 in
`tools/parity/cosyvoice3_reference/pyproject.toml`, #318 in
`tools/parity/cosyvoice2_reference/pyproject.toml`, and #159 in
`tools/parity/xcodec2/uv.lock`. Main `81d72f23` still contains those
vulnerable declarations/resolution. Draft PR #152 contains an XCodec2 update
candidate, but is behind main; the locally accepted derived-source builder
is not yet in its remote head. Neither that candidate nor a declaration-only
CosyVoice version bump is a completed, numerically validated dependency fix.
CosyVoice's reference closure and package/native license gates remain open.

The Scorecard code-scanning API contains three open findings:
`CIIBestPracticesID`, `CodeReviewID` and `VulnerabilitiesID`. Readback SHA-256:
`f2b133092885fc6e0198afb7508d776731c458438fce19a38c3cd0abb77481d3`.
API state alone does not prove that every underlying policy requirement is
closed. The historical five-finding scope remains in the final audit, with
current per-finding disposition still required. No alert was dismissed to
obtain these counts, and no protected owner manifest was inspected.

### Integrated PRs and active Kyutai replay (13:02 UTC)

PR #186's new CI exposed two genuine source-gate failures at `a84817d4`:
`cargo-semver-checks` rejected additions to the public exhaustive `HotOp`
enum and changes to existing implicit discriminants; the Apple Metal build
passed, but Metal Clippy rejected two undocumented unsafe blocks. These
failures are not waived or attributed to flakiness. A bounded API-preserving
capability correction and accurate safety comments are under implementation
review. The green Linux packet above remains evidence for its named head,
not proof that this PR is currently mergeable or that Apple parity passed.

XCodec2's accepted source candidate was integrated conflict-free with actual
main as `35e4c70e830a675e3f9a124347e8fa3d23b77373` and pushed to
[draft PR #152](https://github.com/ayutaz/vokra/pull/152). Against main,
changes are restricted to its dedicated reference directory; all runtime
source remains byte-identical to main. Root independently passed the static
inspector and builder self-tests. The first builder invocation with Python
isolated mode failed sibling-module resolution before any audit; the ordinary
stdlib-only `-S` invocation then passed. The implementer reports 21 named
stdlib tests and dependency self-tests green. New CI is pending. No package
closure approval, real-weight run, public-row completion or upload is inferred.

The new Kyutai replay owns worker `53528022` (200-GB disk) at integrated
head `27060dd2`. Actual EPYC 7713 default Avx2 and forced Scalar each passed
exactly four tests with zero failures or ignores. The serial workspace leg
is actively running; no final result is recorded yet. A fresh terminal
all-pages account read contains only this exact owned, running worker. It is
needed for current verification, and its controller will recover evidence
then destroy it on both success and failure. No idle retained worker or HF
token transfer was introduced. Scaleway has not been newly allocated.

### Kyutai recovered results and Mimi API correction (13:12 UTC)

The integrated Kyutai worker above is now terminal. At exact clean head
`27060dd2b8eceb744fb338fb1783caab9c5717b8`, all six recovered verification
exit files are zero. Default Avx2 and forced Scalar each report four passed,
zero failed and zero ignored tests on the actual EPYC 7713. The serial
workspace log contains 323 successful result summaries: 8,111 passed, zero
failed and 105 ignored. Clippy, deny and advisory audit exited zero. Root
independently matched all 20 recovered SHA-256 entries, final HEAD and empty
observed status. The manifest SHA-256 is
`cb0cc153ff04747f46913041aee1109f823c8d16d2b64143d6448372c80509a9`.

The controller nevertheless exited one: both focused logs contain
`focused_result_count=FAIL`. A test emits its ISA diagnostic between the
test-name prefix and the following `ok` line, which the controller's
same-line result parser does not accept. This is a verification-controller
defect, not a waived Rust failure. Preserve the original controller and
packet unchanged; a separately reviewed parser correction and offline
readback are still required before claiming the integrated pipeline green.
No rerun or new allocation is inferred from that review. Worker `53528022`
and its 200-GB storage were destroyed after recovery; individual readback
returned `instances: null`, and the terminal all-pages account list contained
zero instances.

Mimi's CI correction was reviewed and committed locally as clean head
`d33ba0b9e0a8be06d390fe32fbfb1f8571888071`. It restores all 35 existing
public `HotOp` variants and their original discriminants, moves the two new
Mimi capabilities into a private fail-closed gate, preserves the unavailable
Metal error, and documents the two unsafe Metal calls. Formatting,
diff hygiene, zero-dependency and forbidden-symbol gates passed. An initial
attempt to invoke a nonexistent `check-forbidden-deps.sh` exited 127; the
actual repository gate `check-forbidden-symbols.sh` subsequently passed.
No local Cargo compilation or model execution ran. This correction is not
yet pushed or remotely tested; the older `a84817d4` Linux packet does not
prove the new head, and PR #186 remains draft with its earlier CI failures.

PR #152 at `35e4c70e830a675e3f9a124347e8fa3d23b77373` is also still
draft. A fresh check readback contains 60 successes, one skip and nine
in-progress jobs, not a final green verdict. Its package/native-license and
real-weight gates remain open. No owner approval, Apple result or artifact
upload is inferred; the public classification remains 136 code/artifact-full
and 58 unresolved rows.

### Authenticated Kyutai readback and source-audit tooling (13:28 UTC)

The Kyutai parser correction was reviewed separately, with SHA-256
`9ae38da37a061c988516d7e5447f826dc88e3355ee96de032d3f9161e2194c60`.
Root independently passed syntax, ShellCheck, offline self-tests and the
unchanged packet readback above. The readback authenticates the exact source
and base, all 20 file hashes, all six zero exits, all four named focused
results and the Avx2/Scalar observations. Missing, duplicate and conflicting
terminal statuses are rejected. No bound or original packet was modified,
and the original controller exit one remains recorded. The reviewed source
was pushed as [draft PR #187](https://github.com/ayutaz/vokra/pull/187).
Its fresh CI is not yet green: `typos (advisory)` failed before checking source
because GitHub returned HTTP 500 for the pinned tool download. An attempted
single-job rerun was refused while the parent workflow still had queued/live
jobs. This is distinguished from the genuine source failures in PR #186;
neither failure is waived or recorded as a successful job.

The bounded, standard-library-only Mimi upstream source collector and its
tests were reviewed and committed separately at
`592bf0ca1674ac8906edbd2185756ddc0f097fe6`. The blocked full-reference
dumper and its test remain untracked, frozen and unstaged. The implementer
reports 16 tests green; root's attempted independent local invocation was
refused by the local-model guard before execution and was not retried through
another invocation. Independent remote static replay is pending. Root did
independently authenticate the actual generated inventory JSON SHA-256
`2157000f7e0257125bd11db8929bdefe89d5cf1a18fe00720b27142aa1797cb0`
and its internal inventory digest
`d122055281ebc997611f1069fcb8a09268cbb86ae96f7cc93da609dc8a73f0b0`.
The clean official Moshi checkout at
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` contributed 42 selected
tracked source/license/metadata files, totalling 368,627 bytes. Both actual
`moshi/LICENSE` and `moshi/LICENSE.audiocraft` primary texts were read.
No upstream import, build, dependency resolution or model access occurred.
The inventory remains `PENDING_REVIEW_NOT_OWNER_SIGNABLE` / `NO_UPLOAD`,
not a complete runtime or native-license closure.

In particular, the fixed source declares `torch >= 2.2.0, < 2.10`; no local
override to 2.13 is inferred from another reference project's candidate.
Official-source and advisory compatibility still require review before any
Mimi real-weight reference run. Root reviewed a new six-leg Mimi controller
for clean implementation head `d33ba0b9`, with a separate exact singleton ABI
test, ten required Mimi test names, all six exits required zero, and a 20-file
authenticated packet. Its SHA-256 is
`f2b9abe95c2f0ecefa348cd04cf6fc2f0cecaeaf8182ea523559c39c4590d8cd`.
Syntax, ShellCheck and offline controller tests passed. A single reviewed
offer attempt is now active; there is no result for this head yet. It runs
source/Rust checks only, transfers no HF token, and destroys its owned worker
and storage after recovery on either result. Public coverage remains 136/58;
no new Apple or publication verdict is claimed.

### Mimi compiler failure and reviewed correction (13:41 UTC)

The six-leg verification of `d33ba0b9` above has finished with a genuine
source failure. Focused Mimi tests, the singleton ABI test, model Clippy and
workspace tests each exited 101: adjacent string literals inside `format!`
are not valid Rust. Deny and advisory audit exited zero, but this does not
make the implementation green. Root's prior formatting/static review missed
the compile error. All 20 recovered file hashes match the unchanged packet;
its manifest SHA-256 is
`90c9b1c948080384dc99edf03e32f8467ff0898e92f4cdd29e56132598636bcb`.
The parent controller exited one. Worker `53532437` and its 200-GB storage
were destroyed; individual readback returned `instances: null` and the
terminal all-pages account list contained zero instances.

The implementer joined the error message into one valid format string.
Root reviewed the single-file diff and committed it at
`3d9fffbf7e05b623c6fb4919c73a39db3016b662`. Formatting, diff hygiene,
zero-dependency and forbidden-symbol gates passed. No semantics, public ABI,
capability coverage or backend behavior changed. This new head remains
unverified remotely and unpushed; a separately reviewed fresh-head controller
is being prepared, not a modification or restart of the failed worker.

At this readback PR #152 has 70 successful checks and one skip, all terminal.
Its draft status, package/native-license and real-weight gates remain open;
CI alone is not an approval to execute or merge that candidate. PR #187's
HTTP-500 tool-download failure was rerun only after its parent workflow
finished. Attempt two succeeded at the same `27060dd2` head, and all 16
required checks are successful. The remaining live checks still require
terminal review before a merge decision. Neither PR establishes full-model
or Apple parity.

The owner's device-selection instruction is retained: choose GPU for a
correctness-approved operation when its measured end-to-end time, including
setup and transfer, is faster. Mandatory CPU/reference and Apple Metal
verdicts are not replaced. Rust compilation and these source regression
checks use CPU/RAM; GPU acceleration is not inferred for Cargo. Independent
official Rust Mimi reference compatibility and its dependency/native-license
closure remain under source-only investigation. No new model access, HF
upload, Apple allocation or catalog completion is claimed; scope remains
194 public rows, 136 code/artifact-full and 58 unresolved.

### Kyutai merge and Linux source-audit portability failure (13:56 UTC)

PR #187 finished with 77 successful checks, five intentional skips and zero
failures at `27060dd2`. All 16 strict required checks passed, and root
reviewed the clean merge state and protection rules before marking the
bounded component change ready. It was squash-merged at 13:50:37 UTC as
`909c527b65c2a1308848450652962df42244753d`, now the reviewed GitHub
`main`. Full PCM-to-text ASR, independent real-weight reference and Apple
verdicts remain pending; the merge does not change public-row classification.

The source-audit collector's independent Linux replay did not pass. At exact
clean `592bf0ca`, the standard-library suite ran 16 tests: one passed and 15
errored because its temporary-directory helper hard-coded macOS
`/private/tmp`, which does not exist on the worker. Root verified all 14
recovered evidence hashes, the final head and clean status, and both tool
hashes. The original test exit and side-controller exit are one. No fake
macOS directory was created to mask this portability defect. The failed
packet remains unchanged. The reviewed portable helper correction was
committed separately at `b34739839715bc482212793fae99e3215edab79a`;
its test-file SHA-256 is
`69191653382e78e9622afb736de0373ce073d3aea69710383f9ac1456801bcd0`.
No independent test pass is recorded for this new head yet.

The Mimi `3d9fffbf` six-leg controller is actively using its single owned
worker `53534094`, with effective 16 CPUs, approximately 128 GB host RAM
and 200 GB rented storage. Its recovered-to-date focused log reports
79 passed, zero failed and zero ignored; the separate ABI and Clippy exits
are also zero. Workspace verification is still live, so no overall green
result or final packet is claimed. The side audit used a distinct checkout
and does not own this worker's destruction. The parent will recover small
evidence and destroy the worker and storage on either final result. No
idle retained instance is introduced.

To preserve that frozen verification, root integrated new `main` into a
different clean checkout, producing
`0dca06f2d6e585c698f9dc61c2961e5fcbf193ea`. The four Mimi implementation
files remain byte-identical to `3d9fffbf`; formatting, diff hygiene,
zero-dependency and forbidden-symbol gates passed. Its new-base remote
verification is still pending and will not start another worker while the
current lifecycle is live. Official Rust core-only reference licensing,
checkpoint compatibility and converted-Mimi provenance binding remain
separate source/real-weight gates. No HF token, artifact upload or new Apple
allocation occurred; scope remains 136 code/artifact-full and 58 unresolved
out of 194 public rows.

### Mimi six-leg pass and collection-race correction (14:10 UTC)

The exact clean `3d9fffbf` verification above is now terminal and successful:
all six exit files are zero. Root independently matched all 20 packet hashes,
final HEAD, source/base and empty status. Focused Mimi tests report 79 passed,
zero failed and zero ignored; the ABI selection is one passed with no failure
or ignore. The serial workspace has 323 successful summaries, 8,116 passed,
zero failed and 105 ignored. Model Clippy, deny and advisory audit passed.
The manifest SHA-256 is
`2ef2c09046ee9583be2094fba3322f6fd852708a3a55076d25e67d36e2e1b1fe`.
The reviewed parent exited zero after recovery and strict destruction of
worker `53534094` and its 200-GB storage; individual readback is null.
This is Rust regression evidence, not independent full-model or Apple parity.

The portable `b347398` source-audit attempt is not a recovered pass. Root
reviewed its separate controller, SHA-256
`99af1a51640c6778ebeffb56311b321e1b3b4b73d1899dac4e8ad1830b5afaf5`,
including independently passing, re-signed positive/negative packet tests.
The remote work command completed, but the parent's normal destruction raced
the side packet collection. Only setup evidence was recovered; remaining SSH
connections were refused. The side controller correctly exited one. Neither
the 16-test verdict nor its final HEAD can be certified from that incomplete
packet. Do not restart or retain the destroyed worker. The next batch will
put this standard-library suite inside the parent lifecycle so all required
evidence is recovered before destruction, rather than use a competing side
collector. The original failed and incomplete packets remain unchanged.

Root also reviewed and committed measured Mimi input identity at
`cd7fcde72ae00fd588dea616dd6744b5766b1d4e`: the converter hashes its
actual input buffer and emits the existing checkpoint SHA-256/byte-count keys.
It guesses no origin, revision, filename or sign-off and cannot self-embed an
output whole-file digest. Its synthetic tests inspect both converted GGUF
stamps, including a valid input-byte mutation. Formatting and static gates
passed; execution remains pending. The new-base batch now has clean head
`00632846da5b11b3225c2864b7f75b41d1b9a4f3`, combining current `main`,
reviewed Mimi changes and the source-audit/portable-test commits. This head is
not yet remotely verified or pushed. No publication or Apple verdict follows.

A fresh unfiltered terminal `instances-v1 --all` read at 14:10 UTC contains
one unrelated running instance, `53535326`, labelled
`jtalm-gen_action_v051`. It was not created by this Vokra lifecycle and was
not modified. The prior controller's empty list was filtered by its owned
label, not an unfiltered account-wide zero. Vokra owns no retained worker or
storage; do not infer that the entire account is empty or destroy the
unrelated instance. An initial read-only attempt using unsupported
`--all-pages` exited two; the correct `--all` query succeeded with no next
page. Catalog scope remains 194 rows, 136 code/artifact-full and 58 unresolved.

### Integrated Mimi nine-leg verification started (14:18 UTC)

Root reviewed the frozen controller for clean integrated HEAD
`00632846da5b11b3225c2864b7f75b41d1b9a4f3`, based on merged `main`
`909c527b65c2a1308848450652962df42244753d`. The controller SHA-256 is
`a6ea375471e32048d70832a6dbf6e3d801c2683788e9804f9c2cfaba2e884ea9`.
Root independently passed shell syntax, ShellCheck and offline controller
self-tests, including source-hash and re-signed failure/absent/duplicate-test
rejections. The source checkout passed formatting, diff hygiene,
zero-dependency and forbidden-symbol checks. These checks are not remote
test results or model parity.

The single owned disposable worker is `53537966`, labelled
`vokra-mimi-metal-integration-006-full-verify-00632846-20260930T141851Z-26604`.
Its readback reports 16 effective CPUs and 200 GB rented storage; the reviewed
offer advertised approximately 128 GB host RAM. Bootstrap is in progress.
The parent lifecycle will run the 16-test standard-library source-audit suite,
Mimi focused tests, the single ABI pin, converter focused tests, converter
Clippy, model Clippy, workspace tests, deny and advisory audit. Its 26-member
packet and nine zero exit files must be recovered and authenticated before
a green result is recorded. No competing side collector is used; destruction
of this exact worker and its data follows evidence recovery on either result.

The user requested GPU use when it is faster. Model execution may choose a
GPU only after correctness within unchanged numerical gates and end-to-end
time, including setup and transfers, justify that choice. Mandatory CPU and
Apple Metal/no-fallback legs remain separate. This batch is Rust regression
and model-free audit work, so a GPU does not accelerate its selected commands.
No model runs on the maintainer Mac. Official Rust Mimi reference closure,
checkpoint compatibility and full PCM-to-text composition remain open; no
new checkpoint, publication or Apple allocation is authorized by this note.
The unfiltered pre-rent account read still showed unrelated `53535326`;
it was not modified. Scope remains 194 public rows, 136 code/artifact-full
and 58 unresolved, with the nine-leg verdict pending.

### Official Rust reference source-archive readback (14:30 UTC)

The e6 Moshi core-only source plan is now frozen at SHA-256
`76d11bc0cc0166e890a4115c310a0aed198d476430ab2abf1d750e8d4216b703`.
It distinguishes the 166-record static lock candidate graph from an activated
Linux CPU graph. An isolated audit workspace must preserve inherited package
and dependency declarations; full-workspace feature unification, an unbound
derived lock, or an unbound metadata JSON cannot establish that CPU closure.
No Cargo resolution, build, model/header acquisition or reference execution
was performed while preparing this plan.

Root acquired three official registry source archives only, totalling 675,933
compressed bytes, and independently matched their hashes to the e6 lock:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `candle-core-0.9.1.crate` | 239065 | `a9f51e2ecf6efe9737af8f993433c839f956d2b6ed4fd2dd4a7c6d8b0fa667ff` |
| `candle-nn-0.9.1.crate` | 67671 | `c1980d53280c8f9e2c6cbe1785855d7ff8010208b46e21252b978badf13ad69d` |
| `candle-transformers-0.9.1.crate` | 369197 | `186cb80045dbe47e0b387ea6d3e906f02fb3056297080d9922984c90e90a72b0` |

The normalized manifests declare empty default features and `build = false`.
All three archive VCS records identify Candle commit
`cd96fa80da255e34f7b16b4ff98b6a31d557201b`, with their respective package
paths. The core archive bundles an Apache-2.0 `LICENSE`; the other two do not
bundle a `LICENSE`/`COPYING`/`NOTICE` under those exact names. Root also read
the [primary Apache license](https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/LICENSE-APACHE)
and [primary MIT license](https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/LICENSE-MIT)
at that immutable commit. Their hashes are respectively
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
and `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`;
the bundled core license matches the former. These are direct source facts,
not a transitive/native license closure, owner signature or execution approval.
No package was extracted, installed, built or imported on the maintainer Mac.

Root reviewed the initial native Kyutai PCM composite and Rust-core audit
tool drafts and requested corrections before accepting either. The PCM draft
has an actual step/logits type mismatch and needs production-path control,
poison/reset and provenance-negative tests; the audit draft needs corrected
workspace construction, repository placement, pinned snapshot authentication,
root-reachable metadata binding and bounded archive handling. Neither draft
has a verified commit or numerical verdict. The integrated `00632846` VAST
regression remains separate and live, with six observed zero exits and
workspace verification pending. Final packet authentication and destruction
are not yet claimed. Scope remains 194 public rows and 58 unresolved.

### Integrated `00632846` terminal readback (14:43 UTC)

The primary nine-leg controller terminated with exit **1**, not a green
verdict. Root independently authenticated all 26 recovered packet members;
the manifest SHA-256 is
`12828a079824ea648724d8564e6d262a3c8691536b712741d37cc806c56ba657`.
The packet binds clean head
`00632846da5b11b3225c2864b7f75b41d1b9a4f3` to base
`909c527b65c2a1308848450652962df42244753d`; observed status and diff-check
output are empty. Every command leg recorded exit 0:

- source-audit unit tests: 16 ran, `OK`;
- focused Mimi tests: 79 passed, zero failed or ignored;
- public HotOp discriminant test: exactly one passed;
- Mimi converter tests: seven passed, including all three required names;
- converter Clippy and all-feature model Clippy: exit 0;
- workspace tests: 8,122 passed, zero failed, 105 ignored across 323 summaries;
- deny and audit: exit 0 (deny retains its unmatched fuzz-license-exception
  warning).

These command results do **not** override the controller failure. The source
audit's output-limit and timeout tests emitted `ResourceWarning` for unclosed
subprocess stdout/stderr streams. Those warnings split two named unittest
result lines, so the unchanged strict sixteen-name result gate rejected the
log. The next action is to close streams on every subprocess exit path and
replay a new exact clean head; suppressing warnings, loosening the result
matcher or pushing this head as fully verified is not accepted. This is a
resource-cleanup defect, not evidence of numerical flakiness.

Evidence collection and deletion succeeded independently of that verdict.
Owned worker `53537966` and its 200-GB storage were destroyed; individual
readback returned `instances: null`, and its exact label was absent. A fresh
unfiltered account read showed only unrelated running `53535326`, labelled
`jtalm-gen_action_v051`; it was not modified. No Vokra instance remains from
this replay. PR #186 remains draft at remote head `a84817d4`, with 89 successful,
nine skipped and two failed old-head checks; no corrected push is claimed.

Root also found that the official e6 core directory `moshi-core` contains
package **`moshi`**, inherited version `0.6.4`. The new Rust-reference audit
draft must bind that actual manifest/lock identity, not the directory name.
Further draft corrections remain under review and are not included in this
authenticated packet. No model, independent real-weight parity, Apple run or
upload was performed. Scope remains 194 public rows and 58 unresolved.

Root subsequently reviewed and froze the next logical changes in three local
commits, each based on `00632846`: `84098e8` closes the source-audit subprocess
streams and adds warning-aware regression coverage; `9d2eded` adds the
crate-private authenticated Kyutai PCM composite with shared production
buffer/lifecycle tests; `5040643` adds the bounded official Rust Mimi source
inventory and its synthetic contract tests. Static diff, format where
applicable, zero-dependency and forbidden-symbol gates are green. Local
compilation/model execution was not attempted; the Python unit executions
remain deferred after the local guard blocked them. These commits are
source-reviewed, **not** remotely verified or pushed. The next controller
must bind their combined clean head, retain the unchanged gates, and run both
new focused suites along with full regression. It must not treat the source
inventory's explicitly `OPEN` dependency/native/license status as parity or
execution approval.

### Integrated replay preparation and fresh readbacks (15:03 UTC)

Root independently verified that the clean integrated head
`a527c5c5a86e7eab16b36b3350039bfcbe2ffa5a` contains `00632846` and the
three reviewed follow-up slices. Its scoped diff against `909c527b` has 13
files, 5,127 additions and 93 deletions. An initial integration omitted the
Mimi integration ancestor; that candidate was rejected and corrected before
any rental or verification. The corrected head is still **not remotely
verified or pushed**. The next eleven-leg replay must require all 17 source
audit, 22 Rust-source audit and 18 PCM-control test names, plus the unchanged
regression/audit gates and a complete 30-member authenticated packet.

Review also caught nonportable self-test mutations, leaked shell tracing and
a proposed hardcoded PCM source-hash log. These must be corrected before
execution; remote evidence must measure the file hash, not print an expected
constant. This preparation does not establish real-weight parity or Apple
completion. The remaining reference dependency-activation binding and FireRed
PCM composition are separate bounded work, not part of this frozen head.

A fresh paginated GitHub read still reports 209 open Dependabot alerts:
182 with a patched version and 27 without; severity is three critical,
31 high, 82 medium and 93 low across 37 manifests. These counts do not mean
the reference environments have been updated or numerically revalidated.
A fresh unfiltered VAST read contains only unrelated running `53541309`,
labelled `jtalm-gen_action_v051b`; it was not modified. No Vokra worker had
been rented for this integrated replay at that readback.

The user's device-selection instruction is preserved: use cloud GPU where
it is faster end-to-end and satisfies the unchanged numerical guards.
Rust compilation/source-contract tests remain CPU/RAM work; mandatory CPU
and Apple CPU/Metal/no-fallback evidence cannot be replaced by CUDA results.
No local model execution, new publication or row promotion occurred. Scope
remains 194 public rows and 58 unresolved.

### Integrated replay terminal failure and cleanup (15:30 UTC)

The eleven-leg replay at exact head
`a527c5c5a86e7eab16b36b3350039bfcbe2ffa5a` finished with controller and
remote-verification exit 1. The frozen controller SHA-256 was
`984a64e49024e63c283116f1edb4d81ba00f2514cee6c4708f025c8eb54cdd59`.
Root recovered all 30 packet members and independently verified their
checksums; the packet manifest SHA-256 is
`cbb53546df11f1c3eb3f7bf9a08819a5bc745217b29ad03eed732c0a3eb976a5`.
The recovered final head matches the target; worktree status and diff-check
output are empty. This is authenticated **failure evidence**, not a green
regression result.

The source-audit suite ran 17 tests and the Rust-source audit suite ran 22,
both with `OK` and exit 0. Converter tests, converter Clippy, deny and audit
also recorded exit 0. Mimi tests, ABI, PCM tests, model Clippy and workspace
tests recorded exit 101: `E0599` identifies a call to the nonexistent
`KyutaiSttAsr::from_component_gguf` in the new PCM constructor. No successful
PCM execution or numerical result is claimed. The reviewed correction uses
the existing strict `KyutaiSttWeights::from_component_gguf`, authenticated
configuration and `KyutaiSttAsr::new`; blocked public loaders remain blocked.
That correction requires a new clean-head replay before any code push.

Owned worker `53544589`, including its 200-GB storage, was destroyed after
small-evidence recovery. Saved and fresh individual readbacks return
`instances: null`; the saved label readback contains no matching worker.
The fresh unfiltered account read contains only unrelated running `53545562`,
labelled `jtalm-train_action_v051`, which was not modified. No Vokra worker
or retained storage remains from this replay.

Separately reviewed local FireRed commit
`43fc8e6ce7328b547f1ce262580261fd62c37e93` adds explicit PCM-to-native-beam
composition and five structural/control tests. It is not included in this
failed packet and has not been remotely compiled or pushed. The pinned
official decoder allows immediate EOS with empty content, so that valid case
is retained rather than rejected by an invented nonempty-output requirement.
Its real-weight independent reference and Apple verification remain open.

The renewed device instruction is to use GPU when it is faster end-to-end
under unchanged numerical gates, with actual device/precision/timing recorded.
Rust compilation and model-free control tests remain CPU/RAM work. GPU
reference generation does not replace required native CPU or Apple
CPU/Metal/no-fallback evidence. No local model execution, model upload,
license approval or catalog-row promotion occurred; scope remains 194 public
rows with 58 unresolved.

### Reviewed repair freeze and live replay (15:46 UTC)

The next reviewed, clean integrated head is
`76cd41757be011ddf35bc66d49690bfeed90a4b6`, a descendant of both the Mimi
integration ancestor and `main` `909c527b`. It adds the FireRed PCM composition,
the component-binder API correction and the Moshi core metadata binder. The
last binder slice is local commit `3635475`, integrated as `76cd417`; its
15 model-free unit tests have not yet been run at this snapshot. Its registry,
target activation, native/license, owner and real-weight parity gates remain
explicitly OPEN, even if those unit tests subsequently pass.

Root independently confirmed all eight implementation/test hashes, clean
status, ancestor relationships, formatting, zero-dependency and forbidden
symbol gates. The new controller SHA-256 is
`5c522e2e594019c99f2ddf01776f317521ae66ebae8b78a50e70db2814eb15e7`.
Root also ran its syntax, ShellCheck and complete offline mock successfully.
The thirteen-leg replay requires the unchanged 17/22/18 named suites plus
five FireRed PCM tests and 15 metadata-binding tests, all regression/audit
legs, and exactly 34 authenticated packet members. It must not weaken result
matchers or promote skipped, missing, duplicate or hash-mismatched evidence.

The first creation request, for offer `39529883`, terminated with
`no_such_ask`; no verification process started. Label recovery and a fresh
unfiltered account read found no worker from that attempt, only unrelated
`53545562`. After confirming terminal failure and absence, a different
reviewed offer `31947962` created owned worker `53547744`, labelled
`vokra-mimi-kyutai-integrated-repair-verify-76cd4175-20260930T154240Z-95510`.
Its resource readback records 16 effective CPU cores, 200-GB storage and
16 build jobs. Provisioning completed; live SSH evidence records the
17- and 22-test source suites passing while Rust compilation is still running.
There is **no terminal replay verdict or deletion claim yet**. The existing
controller owns evidence recovery and exact-instance destruction.

Fresh GitHub reads still show `main` at `909c527b`, 209 open Dependabot
alerts, and draft PR #186 at unchanged remote head `a84817d4` with 89
successful, nine skipped and two failed old-head checks. No corrected code
push, model acquisition, upload, Apple run or row promotion is claimed.

### Authenticated source-only side probe and device policy (16:02 UTC)

A separate source-only Moshi probe reused owned worker `53547744` without
changing the frozen primary checkout, running models, building native code or
renting another worker. The accepted probe controller SHA-256 is
`f5a3df08f89aaa76254e4c91124f9a9a317540d4cd00b6ad49c1b0adc77e83f9`.
It ran the metadata binder at exact Vokra head `76cd4175` against pinned
Moshi `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`.

The real `cargo metadata --locked --offline` command exited 101 because the
isolated empty dependency cache did not contain `candle-core`. The binder
reported `BLOCKED_METADATA_COMMAND`, and the outer probe exited 2. This is
authenticated negative evidence, not metadata, license or parity approval.
The recovered 502,485-byte packet contains 53 hashed members plus its
manifest. All 53 checksums pass; the manifest SHA-256 is
`ed09a0a5fdf6a0344a394c9090dc61cf5a781277d41699fe24b6fbd2f4694a25`.
Both source checkouts retained their exact heads and empty before/after
status; the original and copied lock remained byte-identical. The source
inventory digest is
`5556c249f038140993fe1482cbe507271912ddcdda124a9b93df935247be2ec5`.
The next source-cache acquisition is a separate planned step, not an excuse
to remove offline/locked binding or execute unreviewed native dependencies.

Meanwhile, read-only primary evidence confirms Mimi 79 tests, ABI one test,
PCM 18 tests, FireRed five tests and metadata-binding 15 tests passed; model
Clippy also exited 0. Workspace and later regression legs are still pending
at this snapshot, so the replay has no terminal green verdict or cleanup
claim. The controller remains responsible for exact-instance destruction.

Reviewed local commit `9d7fa408ec9b02b1411e72768e01d9ff7f482d77` adds a
bounded, explicit-provenance, signed-int16 little-endian PCM packet to the
independent FireRed reference. It preserves the upstream formula, model calls
and numerical bound. Its six model-free wire tests are not yet run and this
commit is not in the live primary replay; no push or real-weight verdict is
claimed.

The user's renewed GPU preference requires measured end-to-end improvement
under unchanged accuracy/precision gates, including transfers. Rust builds
and model-free tests remain CPU-parallel work. GPU reference/backend results
must never replace mandatory native CPU or Apple CPU/Metal/no-fallback
evidence. No local model execution, upload, legal approval or catalog-row
promotion occurred; the 194-row scope and 58 unresolved rows are unchanged.

### Terminal reviewed-repair replay and code publication (16:13 UTC)

The primary controller completed with exit 0 at exact clean head
`76cd41757be011ddf35bc66d49690bfeed90a4b6`. Root independently checked all
34 recovered packet hashes against the remote manifest; its SHA-256 is
`2a8641463367bd48608e14aca322a0b96847767dd577c2b4a23a9f4005d8af44`.
The final head matches, and observed status and diff-check output are empty.
All thirteen regression/audit exit files contain 0. Focused counts are
17 source-audit, 22 Rust-source, 15 metadata-binding, 79 Mimi, one public ABI,
18 PCM and five FireRed tests. Workspace output records 323 suites, 8,145
passed, zero failed and 105 ignored. Ignored tests are not numerical or
hardware evidence; the actual offline metadata cache failure above remains
valid negative evidence.

Owned worker `53547744`, including 200-GB storage, was destroyed after
evidence recovery. Both saved and fresh individual API readbacks return
`instances: null`; saved label recovery and fresh account listing find no
owned worker. No unrelated instance was modified. This terminal record
supersedes the live/pending replay snapshots above, not their historical
facts. The reviewed `76cd4175` was fast-forward pushed to the existing PR
#186 branch, replacing `a84817d4`; no unverified follow-up or unrelated hook
change was transferred. New required PR CI and review remain pending; the
later FireRed wire/consumer slices are outside this packet.
No real-weight parity, Apple run, owner decision, model upload or public-row
completion is claimed; the full 194-row scope and 58 unresolved rows remain.

### Post-push platform-specific CI failures (16:24 UTC)

New PR #186 CI is bound to `76cd4175`, not the previous remote head. Fresh
GitHub readback reports two failed jobs and keeps merging blocked. The
[Windows test job](https://github.com/ayutaz/vokra/actions/runs/36742668157/job/109980803877)
fails in `authenticated_mmap_rejects_length_and_hash_mismatch`: its temporary
filename includes the Rust test thread's namespace separators, which Windows
rejects with OS error 123. The
[Metal job](https://github.com/ayutaz/vokra/actions/runs/36742667914/job/109980803790)
builds successfully, then fails Clippy's `match_like_matches_macro` check in
`MimiHotOp::covered_by_backend` when the Apple/Metal configuration makes its
capability predicate constant. These are authenticated platform-specific
failures; the green Linux VAST replay does not supersede them.

Bounded source repairs are delegated, preserving backend capability and
length/hash-rejection semantics. They are not yet reviewed, committed or
verified. Do not rerun unchanged failures, weaken checks, or merge until the
corrected exact head passes the required CI. FireRed wire/real-consumer and
Moshi source-cache follow-ups remain separate, unverified work. There is no
new worker, model execution, artifact upload or catalog-row promotion.

### Platform-fix replay started (16:45 UTC)

Reviewed commit `ecf0990db79047d142090220d2dbf1ac49623ce6` changes only the
Mimi capability predicate and portable PCM authentication-test filename. It
preserves the existing backend availability and byte-count/hash rejection
semantics. Formatting, diff hygiene, zero-dependency and forbidden-symbol
checks pass; it is not yet pushed to PR #186.

The separate replay controller has SHA-256
`8a2e1f0c6778d1338f37bce7be4a05645eb2bccc85b853c2fd52cf2a918ee5f1`.
Root reviewed its complete delta and independently ran its full offline mock
with `bundle-observation-self-test=PASS`. It preserves thirteen actual
regression/audit legs and the exact 34-member evidence contract, adds a strict
compute-source hash guard inside the Mimi leg, and does not include the
unverified FireRed wire/consumer or Moshi cache-acquisition work.

Reviewed offer `17191369` created owned worker `53555925`, labelled
`vokra-mimi-pr186-platform-fix-verify-ecf0990d-20260930T164510Z-99087`.
Its readback records 32 effective logical CPUs, 128,609 MB RAM, 200-GB storage,
16 build jobs, and total rental rate `$0.2051851851851852/hour`. Before rent,
the account read found no owned verification worker; unrelated training
`53554562` was not modified. Bundle authentication/transfer completed and
remote tool setup is progressing. There is no terminal test verdict or
deletion claim yet. The existing live controller owns bounded evidence
recovery and destruction of this exact instance, including storage.

Fresh CI at the still-published `76cd4175` retains the same Windows and Metal
failures; it does not test `ecf0990`. Real-weight CPU/reference, Apple
CPU/Metal/no-fallback, license/owner and publication gates remain open where
previously recorded. No model acquisition, upload or public-row promotion is
claimed; the scope remains 194 rows with 58 unresolved.

### Platform-fix replay closed and published (17:00 UTC)

The existing controller session completed with exit 0 at exact clean HEAD
`ecf0990db79047d142090220d2dbf1ac49623ce6`; it was not restarted. All thirteen
regression/audit exit files are zero. Root independently verified all 34
recovered packet-member hashes, the exact final HEAD, empty observed status,
and the workspace totals: 323 suites, 8,145 passed, zero failed and 105 ignored.
Packet-manifest SHA-256:
`2902b5b55ac88810bd9014bbecc2c16c3897b3fc85e11ed1eb6d78a69b367e53`.
The local evidence directory is
`/private/tmp/vokra-mimi-pr186-platform-fix-verify-logs.fAQRfG/evidence`.

Worker `53555925` and its 200-GB storage were destroyed after evidence
recovery. Controller collection, destroy, individual readback and own-label
readback all succeeded; strict individual readback is `instances: null` and
the own-label result is empty. A separate fresh individual API read also
returns `instances: null`. Unrelated training was not modified.

The reviewed two-file repair was fast-forward pushed to PR #186, and its
body was updated to the corrected-head packet and remaining limits. A fresh
PR read confirms HEAD `ecf0990`, new Windows/Metal and other CI jobs pending,
and merge state `BLOCKED`. Linux verification does not prove the repaired
Windows or Apple/Metal jobs; all sixteen required contexts and the Metal job
must pass at this exact head before merge.

Separately, root ran the source-cache probe's actual helper/scanner fixtures:
six tests with five passes and one Linux-only skip on the maintainer Mac,
then six passes with no skips on the same VAST worker. The probe and test
SHA-256 values were checked before remote execution:
`c05de666fe12289cb0fbe5ed36e43c196d5ac7a57106aea52522fd0d9f204643`
and `fff381f890e8be6f7be70ba40b8dfe1cd83791fee8c9d3811449f9f75f575e68`.
These are model-free source-cache safeguards, outside the immutable primary
packet; no actual dependency cache acquisition or native/model result is
claimed.

The separately reviewed FireRed wire and final-result consumer are committed
and integrated at clean follow-up HEAD
`ba61396cf51bf2bee889a002c36a201047a434bf`, not pushed to PR #186 and not yet
Cargo-tested. The consumer fixes SHA-256 authentication and source-marker
validation, adds external standard hash vectors and full synthetic-schema
rejection tests, and keeps the actual model test ignored/VAST-only. Its
source SHA-256 is
`1ca1e55d517e3053be7ad62039a9f7a07eae5c542f3dce2d32ed0b495491d2ec`.
Full per-layer/logit/beam-parent/timing parity remains open. No synthetic
fixture, build, source inventory or CI result promotes a public model row.
The catalog scope remains 194 rows with 58 unresolved; required independent
CPU and Apple CPU/Metal/no-fallback checks are not replaced by faster GPU
routing.

### Corrected-head platform CI and next-wave review (17:21 UTC)

At exact PR #186 HEAD `ecf0990db79047d142090220d2dbf1ac49623ce6`,
all sixteen required contexts are completed successfully. Root checked the
commit's check-run records against the live main-branch protection contexts,
including their GitHub App IDs (`15368`, or `57789` for CodeQL); no required
context is missing. The repaired
[Metal job](https://github.com/ayutaz/vokra/actions/runs/36748200588/job/109999715994)
also completed successfully at 17:16:42 UTC. Its build, Clippy, feature tests,
Metal backend tests and pinned-GGUF C ABI steps pass. The log reports an
Apple Paravirtual Metal device and passing native Mimi RVQ dispatch,
attention/ELU synthetic tests and device-in/out kernel checks. These are
bounded CI/kernel results, not independent real-weight completion evidence
for the unresolved catalog rows. The earlier failed Windows and Metal jobs
remain historical failure evidence, superseded only for this repaired head.

The optional Voxtral mini job has completed; the Unity packaging job
`110006855931` is still live, progressing through its platform cross-builds.
The latest rollup has no failure and still reports `UNSTABLE` while packaging
runs. No unchanged job was restarted and PR #186 has not yet been merged.

Root fully reviewed the separate FireRed follow-up controller at SHA-256
`495cf49371cd81d7cdc69a0ab35b229b8a10937087cb2bf584b9aa55abdd881a`
and independently passed syntax, ShellCheck and its complete offline mock.
It targets clean follow-up HEAD `ba61396cf51bf2bee889a002c36a201047a434bf`,
requires sixteen actual regression/audit legs and a strict 40-member packet,
and authenticates six PCM-wire tests plus seven namespaced Rust consumer
tests and exactly one intentionally ignored real-model test. The namespaced
matcher correction is covered by a negative namespace-drift fixture. No
remote Cargo or real-model verdict is claimed for this follow-up yet.

The separate source-cache transport driver is not accepted: root's review
found transferred-basename and real unittest-summary mismatches, an escaped
marker, and missing whole-operation/active-log bounds. Corrections and actual
offline orchestration mocks are delegated before any new rental. The frozen
source-cache probe itself is unchanged; actual dependency acquisition,
license disposition and native/reference execution remain open. An encoder
trace follow-up is also delegated in a separate clone, attaching observations
to actual native calls rather than inventing a reference mirror.

A fresh paginated VAST account read returned `success: true`,
`next_token: null` and no Vokra worker or retained storage. The sole returned
instance, `53558591` (`jtalm-train_action_v051_seeds`), belongs to unrelated
training and was not modified. No new rental, model acquisition, upload,
Scaleway run or catalog-row promotion occurred. The full scope remains
194 public rows with 58 unresolved.

### PR #186 merged after all CI completed (17:25 UTC)

The same Unity packaging job completed successfully at 17:24:22 UTC.
Final pre-merge readback for exact HEAD `ecf0990` reports 92 successful
checks, ten intentional skips, zero pending checks and zero failures, with
`CLEAN` / `MERGEABLE`. All sixteen required contexts and the Metal job were
checked separately above. The earlier `UNSTABLE` readback is superseded for
this head; it was not bypassed or reinterpreted as a completed run.

PR #186 was squash-merged at 17:25:59 UTC as
`91a7ebcebf6a643a2235040e133c767a78b8f8b3`. The initial normal-merge attempt
was rejected without changing the PR because repository policy permits only
squash merging. Root read back that policy, used the allowed method with
the exact-head match guard, and confirmed `MERGED` plus the merge-commit ID.
The live main ref now points to that merge commit. The tested PR head and
merged main are distinct identities; no exact-main VAST replay is invented.

Unverified FireRed wire/consumer/encoder-trace and Moshi source-cache transport
work remain separate from this merged implementation. The catalog scope,
legal/source blockers and outstanding independent real-weight CPU and Apple
CPU/Metal/no-fallback verdicts are unchanged. No new instance was rented,
artifact uploaded or unresolved public row promoted by the merge.

### FireRed primary-source trace correction identified (17:28 UTC)

Root read the official FireRed source tree at revision
`834635e4cf277ed8ca92049fc375b17c3dc20748` and authenticated the
12,651-byte `conformer_encoder.py` Git blob
`b41e68e6eb2bed90611e65c6bc5e2025dc6753df` by recomputing its Git-blob SHA-1.
Its SHA-256 is
`ea6412fdffc33b558a610d67143fd3211b76fc5a7a9ccac88ae2110f1fd3d320`.
The [fixed official implementation](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/fireredasr/models/module/conformer_encoder.py)
returns the centered relative-position tensor from `RelPositionalEncoding`;
it does not return a tuple containing the hidden stream. Consequently the
uncommitted trace draft's hidden-stream positional tap is incorrect and is
not accepted, committed or numerically tested. Its correction must observe
the actual native positional slice, not a recomputed reference mirror.

The same source pads six zero-valued time frames before encoder subsampling
by default, retains the original input lengths, and constructs a downsampled
validity mask from those original lengths. The existing native route's
padding/mask equivalence is under investigation; no numerical verdict or
tolerance change is inferred. Source inspection does not execute a model or
establish new license/operator approval. Decoder/beam and real-weight CPU and
Apple gates remain open.

### FireRed trace accepted and padded-mask execution review (17:44 UTC)

The corrected encoder diagnostic trace was fully reviewed and committed as
`72b94b8`: it observes the actual centered positional tensor and all sixteen
native layer results through the same arithmetic path as ordinary execution.
Root independently passed formatting, diff hygiene, the first-party lock gate
and forbidden-symbol gate. No local model or Cargo compilation/test was run;
the model-free trace tests and real nineteen-stage parity remain unexecuted
on this head.

A separate clean follow-up checkout now starts from the live squash-merged
main `91a7ebcebf6a643a2235040e133c767a78b8f8b3`. Its three reviewed PCM-wire,
consumer and trace commits end at
`a20f8e275b02455cc39f47d93952a19f9a7ac9d4`; root confirmed its tree matches
the pre-restack trace head. This is an integration identity, not a new remote
verification verdict. The maintainer's management branch was left intact.

Root re-fetched and re-authenticated the same official Conformer source blob
named above. Review of the pending six-frame padding/mask repair found that
the actual padded tensor length must be derived from the full input length,
independently of the valid-prefix length. The official convolution masks its
computed branch but preserves the residual at invalid rows, and the official
block does not zero rows after final LayerNorm. The existing native masking
differs at both boundaries; source-grounded corrections and focused
model-free regressions are delegated, with numerical bounds unchanged.

The source-cache transport driver's third revision independently passed its
eight offline tests, syntax and ShellCheck. It is still not accepted for a
rental: its normal watchdog cancellation, portable local deadlines and
truthful incomplete-run evidence require further correction. No real SSH,
source-cache acquisition, model acquisition, upload or new cloud allocation
occurred in this continuation. Faster GPU routing must be established by
same-input/same-precision, transfer-inclusive measurement; required CPU and
Apple CPU/Metal/no-fallback gates remain mandatory. The 194-row scope and
58 unresolved classifications are not promoted by these source changes.

### FireRed encoder repair integrated; GPU selection remains measured (18:04 UTC)

The source-grounded six-frame encoder padding and validity-mask correction
was reviewed and committed as `35214a8`. It derives the padded output shape
from total input length separately from the original valid-prefix length,
retains the official masked-convolution residual and final LayerNorm behavior,
and passes the resulting memory mask into both native decoding routes. Root
passed formatting, diff hygiene, zero-dependency and forbidden-symbol gates;
these are static results, not a numerical or compilation verdict.

The clean post-merge follow-up checkout now ends at
`0f016bc5c7f167c5315ee4b97fbfc05cf1abe00b`, based on squash-merged main
`91a7ebcebf6a643a2235040e133c767a78b8f8b3`. Root verified its tree matches
the reviewed pre-restack encoder-repair tree. Its pending remote controller
adds a focused native FireRed test leg to the previous sixteen-leg contract;
no rental or execution is inferred from controller preparation.

Public HTTPS metadata at the fixed FireRed model revision
`e57f5960d03cff1071ff7acbb409314d1e70ed3d` declares `apache-2.0` and lists
eight repository files, with no separate license file. The 989-byte metadata
response has SHA-256
`c0724d432e2604003a253869a137d8d32da882095157d72a2d9c8ad650665872`;
the 1,177-byte tree response has SHA-256
`9aa71a99642f927a9cb8620262c3b758c173aeaf8d2524555e72714c593e19d0`.
The tree confirms `config.yaml` is zero bytes. Root also authenticated the
6,458-byte [fixed model card](https://huggingface.co/FireRedTeam/FireRedASR-AED-L/blob/e57f5960d03cff1071ff7acbb409314d1e70ed3d/README.md)
against Git blob `5baa221616743b808a12ba7bfb25e8ba28e1689f`; its SHA-256 is
`a5a905edac140af027719a5ba6bab2f34f21ae5bfd279725d536519ddc00cf8a`.
Only metadata and text were acquired. These declarations do not close the
dependency/training-provenance review or manufacture an operator approval.

The [fixed official decoder](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/fireredasr/models/module/transformer_decoder.py)
was authenticated as the 11,033-byte Git blob
`2088b0832b84da4421883e2dc7b518f734c3e0b2`, SHA-256
`f0dd5d0ba224ec0be9d2778d3d4ae514ef5ab24c879436aad756353b81f4eedb`.
Its fixed-width beam rows, finite inactive/finished sentinels, cache-parent
selection and PAD key masking differ from the current native route. A separate
bounded repair is delegated without changing the frozen verification head or
the reference oracle. Decoder-stage/beam parity and real-weight CPU parity
remain open.

Following the owner's GPU preference, routing must use same-input,
same-precision correctness checks and transfer-inclusive elapsed time. The
FireRed CUDA backend currently lacks required Conv2d, Relu and Silu operations
and must fail explicitly rather than fall back to CPU. Apple Metal is a
candidate for the final measured comparison, not a demonstrated speedup.
The fixed FireRed reference environment remains CPU-only; required CPU and
Apple CPU/Metal/no-fallback verdicts are not waived. No model execution,
upload, new cloud allocation or public-row promotion occurred in this step.

### Source-cache transport accepted offline; current security and docs PR (18:14 UTC)

Root completed review of the bounded source-cache transport driver, including
portable local SSH/SCP deadlines, process-group cleanup after leader exit,
remote watchdog cancellation, authenticated packet membership and mandatory
before/after identity evidence. The terminal verifier now requires each result
key exactly once with its expected value, including the target HEAD, Moshi
revision and Cargo-lock hash. Contradictory `probe_exit`/`transport_complete`
keys, missing or incorrect identities, missing status files and mutated
evidence are rejected.

The frozen driver SHA-256 is
`d5b962572911da9a8264da46980e950ea12fc40bc500ca5371a3308541f12b92`;
its offline self-test SHA-256 is
`e3f3a699a479826e745bd0b73818479a50e61d67e1d134953311e1a78f51212b`.
Root independently passed Bash syntax, ShellCheck and all nine offline tests
at these identities. The earlier deficient drafts are superseded, not treated
as accepted execution evidence. No actual SSH, source-cache acquisition,
Linux fixture run, native build, model or upload occurred. The next worker
must execute the actual Linux fixtures, recover authenticated evidence and
retain the source/license result as blocked until its own gates close.

A fresh all-pages GitHub readback still contains 209 open Dependabot alerts:
182 with a named patch, 27 without one, across 37 manifests. Severities remain
three critical, 31 high, 82 medium and 93 low. The separately queried critical
alerts are #340/#318 in the CosyVoice3/2 reference inventories and #159 in
XCodec2's lock, all `GHSA-53q9-r3pm-6pq6` for Torch below 2.6.0. CosyVoice's
complete official dependency path remains blocked; source/dependency
investigation is not replacement numerical evidence or approval. No alert
was dismissed or legal gate bypassed.

Root reviewed all four documentation-only files in
[PR #188](https://github.com/ayutaz/vokra/pull/188), authenticated the existing
`v0.3.0` release and its four Python wheels, and passed the existing 52-case
mocked pre-push classifier/integration test. Because the PR was behind main,
root requested an update with exact old-head guard
`0ff54a16a138e0d4880a81d6965904b06a55899d`. GitHub produced new head
`4b434bbeb6a2076b5ef0f5deabb6d26c0ed7bd97`; the new CI is actually queued
and running, with no observed failure at this readback. Old CI is not proof
for this head, and the PR is not merged. Main and the frozen FireRed
verification checkout remain unchanged. No VAST allocation or public-model
row promotion occurred in this continuation.

### FireRed beam/PAD repair committed for remote verification (18:21 UTC)

Root reviewed the complete bounded native beam/PAD correction and committed
the three-file slice as
`961c4733efc9d3624d30aeb5c213749fcec92213` in the separate clean source
checkout based on `0f016bc5`. Production and mechanism tests share one search
driver: all B rows start with the official finite scores, finished rows still
execute the decoder/cache callback, B×B slots are retained, selected parents'
caches are gathered, and only EOS marks a row finished. PAD self-attention
key validity is separate from query/residual validity. Strengthened tests use
branch-specific cache markers and mutate actual current K/V projections
through the shared layer call, so cache-parent swaps and ignored PAD masks
are observable failures rather than untested assertions.

Root independently passed formatting, diff hygiene, zero-dependency and
forbidden-symbol gates. The commit is not pushed and has no compilation,
model-free execution, real-weight or Apple parity verdict yet; synthetic
mechanism tests are not independent reference evidence. The frozen `0f016bc5`
verification checkout remains unchanged. Next diagnostic work must observe
actual official decoder calls, preserve step/beam/parent/slot/cache lineage,
and compare the official full-prefix last query with native current-query
outputs. Official hidden-prefix caches and native projected K/V caches are
different representations; no direct raw-cache parity or Torch tie-order
equivalence is inferred. The 58 unresolved public rows remain in scope.

### Docs PR merged; strict FireRed controller accepted offline (18:34 UTC)

Root re-read the attachment and canonical campaign boundaries: all 194 public
rows remain in scope, including the 58 unresolved code/artifact rows, public
artifact/provenance repairs and the reference-environment security campaign.
The owner's GPU preference applies only after same-input/same-precision
correctness and transfer-inclusive timing; mandatory CPU and Apple
CPU/Metal/no-fallback gates remain unchanged.

[PR #188](https://github.com/ayutaz/vokra/pull/188) completed at exact head
`4b434bbeb6a2076b5ef0f5deabb6d26c0ed7bd97` with 63 successful checks,
four intentional skips and no pending or failed check. Root independently
matched all 16 strict branch-protection contexts to their required application
IDs on that head, then squash-merged with an exact-head guard. GitHub records
the merge at 18:33:21 UTC as `b6589befecf189f942ca35abf3b4d967c079b605`.
This documentation-only merge does not verify subsequent native candidates.

The standalone post-merge FireRed controller is frozen at SHA-256
`0cd58dc9cde57bad913fbdf5f01ade6c04f467b54e63f15d7ddb5658826fd286`,
targeting `0f016bc5c7f167c5315ee4b97fbfc05cf1abe00b`. Root reviewed the
local and generated-remote gates and independently passed Bash syntax,
ShellCheck and the complete offline mock/self-test. The focused FireRed gate
requires exactly one result summary, success for each required test, unique
names for all selected tests, the correct namespace and exact agreement
between actual success lines and the claimed count. Local and remote negative
fixtures reject missing/hash/zero/skip/duplicate/namespace/summary/failure
and count tampering. This is controller evidence, not a Cargo, real-model or
hardware verdict. A separate controller is being prepared to recover the
frozen source-cache probe on the same disposable worker; its blocked
source/license result must remain separate from the seventeen Rust/static
verification legs.

Root reviewed the next FireRed decoder/search observation diff and returned
it for correction: the draft has a mutable-vector compile issue, a now
test-only helper needing proper compilation gating, incomplete actual-decoder
trace/no-trace tests and unaccounted metadata in its memory budget. That draft
is not accepted, committed or remotely verified. No diagnostic observation is
treated as independent upstream parity.

A fresh all-pages VAST inventory returned an empty instance list and
`next_token: null`; no resource was modified. Affordable 128-GB/16-core offers
were observed, but none was rented. The first search mistakenly supplied the
raw RAM number to a CLI field expressed in GB; root inspected the installed
CLI conversion and repeated the read-only search with the correct units.
No model payload, source-cache run, upload or Apple allocation occurred.

### Shared FireRed decoder trace reviewed and committed (18:51 UTC)

Root accepted the corrected three-file native decoder/search observation
slice and committed it as `a374eb004dcbfc028c12127171174e17a483b388`
on top of `961c4733`. The trace observes the shared production decoder and
search driver, without recomputing a separate numerical mirror. Event,
tensor-value, checked-overflow and metadata-inclusive byte budgets fail
closed; the implementation includes mechanism tests for duplicate/order,
shape/nonfinite/resource failures and a real small-shape sixteen-layer shared
decoder call with two cache positions and a PAD key mask. These tests are
not independent upstream reference or real-weight evidence.

Root independently passed `cargo fmt --all -- --check`, `git diff --check`,
zero-dependency and forbidden-symbol gates. The clean committed checkout has
native-source SHA-256
`e89eb262f70693285042e7813e21c6c6ed85e5d38da2ab432b43865fe5b4bd53`,
module-source SHA-256
`41aed8c4e22166895b4e7e93e5dff4a7109e3a43d73d3256f95ce25055877075`
and design-record SHA-256
`f16755f9062f571c70eac7e7668e59599a343bb210b368bffacd9c5e2ba9c0d3`.
No Rust compilation, actual test execution, real-model run, code push or
hardware verdict is claimed for this candidate. The frozen `0f016bc5`
checkout and accepted standalone controller remain unchanged; a separate
combined controller must be reviewed and retargeted to this candidate before
its one-worker verification, so the old candidate is not rented and tested
redundantly.

The GPU preference is conditional on authenticated input, unchanged precision
and numerical gates, complete backend coverage, and transfer/setup-inclusive
timing. The current FireRed CUDA Compute seam lacks Conv2d, ReLU and SiLU;
its source explicitly rejects these operations rather than silently running
them on CPU. Metal has separate source dispatches but still requires actual
Apple validation. Rust compilation and source-cache collection are CPU work;
renting a GPU-equipped worker alone is not GPU model execution. The mandatory
CPU reference and Apple CPU/Metal/no-fallback gates remain in scope.

The separate CosyVoice source-only evidence collector and combined VAST
controller were returned for concrete safety/test corrections. Neither is
accepted for live acquisition or execution yet. Source authentication is not
full composite dependency/legal closure, and the forbidden soxr dependency
remains unresolved. No source or model payload was acquired in this slice.

Root published only the independently checked management journal in
[PR #189](https://github.com/ayutaz/vokra/pull/189), head
`c7af1357f69e6a5e1de495b31cfec043e59d15ce`, based on merged main
`b6589bef`. That one-file documentation change excludes the unreviewed local
hook commit and all FireRed implementation commits. CI is actually queued and
running on this exact head, with no failure at the readback; it is not merged
and pending checks are not green evidence. All 194 public rows, 58 unresolved
code/artifact rows and attachment security/provenance tasks remain in scope.

### GPU preference and source-only collector acceptance (19:14 UTC)

The owner reaffirmed GPU-first execution when it is faster. Select GPU only
for a supported complete path at unchanged precision and numerical gates,
using setup/transfer-inclusive end-to-end timing; keep the independent CPU
baseline and final Apple CPU/Metal/no-fallback legs. This is an execution
preference, not permission to change a frozen reference device, loosen a
bound, hide a CPU fallback, download/run models locally, or publish artifacts.
The current FireRed CUDA coverage gaps remain unchanged.

Root accepted and committed the bounded CosyVoice primary-source collector as
`d5c6ccd5fb8b35823bd7dd2d5df8d203a1dd34f8` in its separate clean checkout.
The three owned files are the collector, its offline tests, and its design
record; their SHA-256 values are respectively
`5eee79b03455ede6311c82e1665614446edc62582c23cf46f3834578d2fd86c0`,
`fc5ce41429901c6b52e1426ff4791ec76d555a567e64312053b295953484e766`
and `6336025aab814bcfcbefba168970c50807e6340535d5a2808b537dd7fd49eb86`.
Root independently ran all 23 offline tests through UV/Python 3.12, with
zero failures, and passed diff, dependency, forbidden-symbol and documentation
gates. The collector authenticates a fixed 28-file, 405,766-byte source set
with exact URL/Git-blob/SHA-256 identities, bounded aggregate/metadata size,
hard read deadlines and exclusive no-follow output creation. Static AST/YAML
facts do not approve dependency/license closure, execute upstream code or
close the full PCM composite.

The local acquisition command was stopped by the existing no-local-model
guard before execution. No source packet or model payload was acquired by
that attempt; the guard was not bypassed. Source acquisition is moved to the
same disposable VAST verification worker after its reviewed controller is
ready, rather than creating a separate retained instance.

Root caught an obsolete design-record digest in the combined FireRed/source
controller before renting a worker. The corrected draft targets frozen
`a374eb004dcbfc028c12127171174e17a483b388` and checks fifteen actual
checkout file digests, including design digest `f16755f...`. Root independently
passed Bash syntax, ShellCheck and its offline self-test. Additional nested
process/session cleanup and actual destroy-order proof were requested before
live acceptance; those mock results are not remote Rust or model evidence.

The separate v2 decoder-trace reader remains unaccepted. Review found missing
constant imports, incomplete cache/type/resource checks, and an ignored
consumer that discards intermediate trace values and compares only event
counts plus final output. Corrections were returned to its implementer;
no v2 capture, numerical trace parity, code push or hardware pass is claimed.
The reviewed native `a374eb` checkout is unchanged.

PR #189 remains on fixed head `c7af1357f69e6a5e1de495b31cfec043e59d15ce`.
At the latest readback it had 54 successful checks, two in progress, one
intentional full-history-gitleaks skip and no failures. Its sixteen strict
required check/application pairs were read from branch protection, but pending
checks are not a complete green verdict; no merge was performed in this slice.
The 194-row scope, 58 unresolved code/artifact rows and attachment security,
provenance and public-artifact work remain open.

### Exact PR merge and source preparation readback (19:30 UTC)

PR #189 was merged at `2026-09-30T19:19:06Z`, after root reviewed all 57
checks on exact head `c7af1357f69e6a5e1de495b31cfec043e59d15ce`: 56
successes, one intentional full-history-gitleaks skip, zero pending checks
and zero failures. Each of the sixteen strict required context/application
pairs had exactly one successful match. The squash commit is
`97447185361a37af64c1b30fe87e8e2618d96e20`; a fresh read-only GitHub API
readback confirmed the merged state. This documentation-only merge does not
include the subsequent FireRed or source-collector implementation candidates.

Root accepted the separate FireRed v2 schema-reader candidate at
`e9b9b058b4802107aab56623062328ba7d98185c`, based on frozen native
`a374eb004dcbfc028c12127171174e17a483b388`. Consumer and design SHA-256
values are respectively
`56e74871a21a7f0e935aeab23d2ebe3341b0f43cf7a9d6e7145c21fcd538d051`
and `c46038d665aa64198b29887ec4118bef4fd46c373e45b2c923478c75c289d42e`.
The reader adds eleven model-free schema tests while preserving seven v1
tests; eighteen passing tests and two ignored real legs are the expected
remote selection, not an executed result. Root passed formatting, diff,
dependency, forbidden-symbol and documentation gates only. The v2 real leg
explicitly stops OPEN before GGUF/model loading: source-bound capture,
v1-envelope digest binding, full stage-tensor comparison and Torch tie
behavior are still missing. No Rust compilation or real parity is claimed.

The authenticated primary decoder source was reviewed for the next capture
boundary. Its existing v1 module hooks and top-k wrapper do not observe
embedding, actual projected self K/V, post-rewrite candidates, cache lineage
or termination/GNMT locals. A separate source-instrumentation builder is being
prepared with exact source authentication and an original-AST preservation
check; this is not permission to execute unapproved upstream/model code.

Root also accepted and committed the separate Realtime source-only collector
as `abf7b0086c7e4c4ca6c91fbddcc79d59164df721`. Collector, test and design
SHA-256 values are respectively
`f109e1089d221c0a42b10b05a5e4b92fa4a973f6eecdcc2801cf791c64d9506d`,
`ff77947ae6578bc2c101baae6a23bf5ddf0a74b5537730d31a5e1692077e4be0`
and `83c0f62f531c61184270b064e13377a1b43212154c284ab2fb601fb202e94c73`.
Root independently ran all 25 offline UV/Python-3.12 tests successfully and
passed diff, dependency, forbidden-symbol and documentation gates. The fixed
allowlist is seventeen Microsoft files plus four Transformers/Qwen2 files;
source bodies have not yet been acquired. Per-file, aggregate and metadata
bounds, hard HTTP deadlines, partial-write accounting and explicit unknown
dynamic-call facts remain fail-closed. Static source facts do not establish
runtime import closure, legal approval, native synthesis or device parity.

The combined VAST controller still requires actual nested-session cleanup and
production destroy-order proof before acceptance. The next reviewed fork
must target `e9b9b058` and collect both frozen source-only inventories on the
same disposable verification worker. No live controller was started or
instance rented in this slice. The latest all-pages VAST instance readback
returned an empty list and `next_token=null`; no separate volume audit is
claimed. The GPU preference remains setup/transfer-inclusive and conditional
on complete supported execution at unchanged precision. CPU reference and
final Apple CPU/Metal/no-fallback gates remain mandatory; all 194 public
rows, the 58 unresolved code/artifact rows and attachment security/provenance
tasks remain in scope.

### Source-builder acceptance and actual cleanup failure (19:50 UTC)

Root accepted the separate FireRed source-instrumentation builder as
`0dc33c2e422d4cbbf2e3d25da37c0d273d1f1ff0`, based on frozen v2 reader
`e9b9b058b4802107aab56623062328ba7d98185c`. Root independently passed
the authenticated official-source AST roundtrip and all eleven owned offline
tests, plus diff, forbidden-symbol, zero-dependency, documentation-reference
and runbook-citation gates. Builder, test and design SHA-256 values are
respectively `106504cd323abbbdbe197ce4fc77342e3a420e5334215b61292630414b2aa882`,
`0ec62a045a1bf839988a96579c5a20fda004a1697b2c5e78e793c8f877071db8`
and `4ea28abdefa05e9aca05ec13da28969f332e4bf13c5ed765cbe0ab50a99082c8`.
The twenty source-observed events include actual decoder-layer input,
projected Q/K/V, search/cache lineage and termination/length-penalty locals.
Removing only the inserted reserved-sink calls restores the original AST.
Exclusive no-follow directory-descriptor writes and bounded process-owned
descriptor diagnostics preserve failure evidence without reading a replaced
path. A zero-iteration loop reports an absent final step instead of introducing
an unbound-local error. These are source-builder tests, not upstream/model
execution, real numerical parity, Rust compilation or hardware evidence.
Actual capture, v1 digest binding, module-role registration and stage comparison
remain open; the candidate has not been pushed or published.

The combined controller's stronger test actually invoked the frozen driver
through mock SSH/SCP and recorded the driver, executor, SSH and TERM-ignoring
leaf PIDs. Parent-TERM/timeout cleanup did not prove all descendants gone;
one recorded PID remained observable for twenty seconds. Process-state
inspection was denied by the local sandbox, so the record does not invent a
running-versus-zombie diagnosis. The old driver lacks direct-child wait/reap
in its EXIT cleanup and uses a one-second grace period; this is a source-level
cause candidate, not a verified fix. A separate versioned cleanup fork is
under review. The failed evidence is retained and no worker is rented until
the actual lifecycle and production destroy-order tests pass.

The three previously identified Critical Torch alerts remain unresolved:
CosyVoice2 #318, CosyVoice3 #340 and XCodec2 #159. The latter is an alert
number, not the unrelated PR number. A separate inventory-only CosyVoice
candidate is being prepared from merged main `97447185361a37af64c1b30fe87e8e2618d96e20`.
The official [TorchAudio installation contract](https://docs.pytorch.org/audio/master/installation.html)
states that its 2.11-and-later stable ABI supports PyTorch 2.11 and later;
the [official PyTorch 2.13 instructions](https://pytorch.org/get-started/previous-versions/#v2130)
provide a CPU wheel index. This supports investigating Torch 2.13/TorchAudio
2.11, not declaring API compatibility or parity. The authenticated official
frontend's forbidden soxr closure, primary package/native-license review,
dependency lock, API smoke, independent CPU parity and Apple gates remain open.
No local dependency installation, model acquisition or model execution occurs.

The safe VAST wrapper's offline redaction/exit-status/destruction-confirmation
self-test passed. A fresh read-only `show instances-v1 --all --raw` returned
zero instances and `next_token=null` at 19:50 UTC; no separate volume audit is
claimed. GPU selection remains conditional on setup/transfer-inclusive speed,
complete supported backend coverage and unchanged precision/numerical bounds.
Independent CPU reference and final Apple CPU/Metal/no-fallback validation
remain mandatory. All 194 public rows, 58 unresolved code/artifact rows and
the attachment's security/provenance/public-artifact work remain in scope.

### Security inventory and source-logits correction (20:01 UTC)

Root reviewed and committed the four-file CosyVoice security inventory as
`e7350b24c919759a6dc4971975a9db35f6b15d78`, based on merged main
`97447185361a37af64c1b30fe87e8e2618d96e20`. Both blocked reference trees
now record isolated Torch 2.13.0/TorchAudio 2.11.0 CPU-index candidates and
retain the pinned original requirements as separate provenance. CosyVoice2's
Hub candidate is aligned to the existing Transformers candidate; compatibility
is still unverified. The requested `uv add --frozen --no-sync` combination was
rejected by the installed CLI before changing any dependency environment.
Its [official CLI contract](https://docs.astral.sh/uv/reference/cli/#uv-add)
states that `uv add --frozen` skips resolution and environment synchronization.
The implementer therefore used the CLI-valid frozen/offline/Python-3.12 form
successfully for both trees, without creating a lockfile or virtual environment.
Root independently passed both TOML-only tests, dependency-automation fixture
and actual lock-coverage checks, diff, forbidden-symbol, zero-dependency,
documentation-reference and runbook-citation gates. No upstream import, model
run, numerical result, owner approval or alert closure is inferred; this
unpublished candidate does not resolve the forbidden official soxr closure.

The official decoder source revealed that logits are rank two
`[batch * beam, vocab]`, not a full-prefix tensor. Root accepted the separate
reader correction as `f8570bfde3be66fc2c309fd17723b5491d908b99`, based on
frozen builder `0dc33c2e`. It requires an exact source-batch-row selection,
retains finite/type/individual and cumulative resource checks, and rejects
invented prefix axes, padding and replicated rows. Consumer and design
SHA-256 values are respectively
`d3f7ec76827792dd0614bee88915c9c3a79e976617dabe18da7b00773c0c9cd8`
and `594c320e23ae0745cb3f1c10a5d60a5fb537b0b50e64ea793ff8791d2e6267eb`.
Root passed formatting and relevant static/documentation gates; the expected
remote selection is now 24 nonignored tests and two ignored real legs, not
executed Rust evidence. The v2 capture, source-patch and v1-envelope binding,
module-role registry, candidate lineage and real stage comparison remain open.

A subsequent stronger lifecycle failure retained its evidence and permitted
read-only process-state diagnostics. The remaining mock leaf had `PPID=1`
and sleeping state `S`, not zombie state, after the driver/executor/SSH PIDs
had gone. Natural disappearance later is not cleanup success. Whether the
cause is signal forwarding/group ownership, sandbox signal permissions, or
both is still being tested; no cause is declared from group IDs alone.
The draft controller and cleanup fork remain unaccepted and no worker is
rented. Their eventual integrated verification must use frozen `f8570bf`,
both accepted source-only collectors and actual lifecycle proof; it must not
rent an obsolete `a374eb`/`e9b9b058` batch. GPU eligibility, independent CPU
reference, final Apple/no-fallback gates and the complete public/attachment
scope remain unchanged.

### Decoder identity acceptance and bounded lifecycle proof (20:23 UTC)

Root accepted the separate FireRed module-identity registry as
`3cae7c7e1125f108f7e55d5f0dc0629ef0853fb0`, based on frozen reader/builder
`f8570bfde3be66fc2c309fd17723b5491d908b99`. It registers the explicitly
observed decoder, layer, self-attention and cross-attention object identities,
retains their lifetimes, and resolves the builder's actual integer-ID events
without inferring roles from tensor shape or call order. Source bytes/URLs,
caller-supplied layer count and actual `n_layers`/layer-stack geometry are
checked; aliases, unknown IDs, event-arity drift, mutation, symlinks and
nonregular/oversized records fail closed. Nonblocking file opens prevent a
FIFO from stalling authentication. Root independently passed all nine tests
with the authenticated source record; the portable default passes seven
synthetic tests and explicitly skips the two optional source-record tests.
Diff, zero-dependency, forbidden-symbol, documentation and runbook gates
passed. This is not a capture writer, upstream model run or numerical result;
v1/v2-envelope and patch binding, full beam/stage comparison, owner/legal,
real-weight CPU and final Apple gates remain open.

The bounded cleanup forks now have independent lifecycle evidence. Root
passed the nine outer-executor tests and the controller's complete offline
suite, then the real nested mock driver/SSH/TERM-ignoring-leaf cases for
parent termination, outer timeout and production collect-before-destroy.
The retained proof is
`/private/tmp/vokra-firered-postmerge-source-cache-success.0nCsdc`; all fifteen
recorded PIDs were absent in a subsequent read-only process-state check.
The shared absolute seven-second group-cleanup deadline does not restart in
the signal/timeout/finally paths; controller termination allows ten seconds
for that grace and the bounded leader reap. The strict twenty-second PID
absence gate was not relaxed. Outer, driver and controller SHA-256 values are
respectively `47e2a1f7e22c30250adfae48c0a59dfee578523f10c844a6e9d1e4dde1f50690`,
`9ee6f8ca766b2270b1a2a4e227bb273ecb98a485590db0cb5bccd12b156605eb`
and `a20a0e49d6df4b890d68bf7dbc80ccb96b0eb4cb6b32e29eb1167075f23b3437`.
Earlier failure evidence remains historical failure, not retroactive success.
These accepted cleanup mechanisms do not authorize launching the obsolete
`a374eb` batch: a separate integrated fork must bind the latest accepted code
and source-only collectors to merged main before a disposable VAST run.

Root also accepted the five-file VibeVoice Diffusers candidate as
`0f204ea9575a210727c94a0480bf96c31baa8b28`, based on main
`97447185361a37af64c1b30fe87e8e2618d96e20`. The 1.5B reference lock now pins
Diffusers 0.38.0 for GHSA-7wx4-6vff-v64p and GHSA-98h9-4798-4q5v. The
original Vokra reference pin is retained separately; it is not relabelled an
authenticated Microsoft upstream requirement. `uv add --no-sync` updated
lock metadata without environment installation. Realtime resolution instead
proved a conflict with its existing `safetensors==0.5.3` pin and preserved
the original lock, with an explicit blocked candidate record. Root passed
three TOML-only tests, both offline lock checks (44/57 package records),
dependency-automation fixture/coverage and relevant static/documentation
gates. No API, dependency-license, model, CPU/Metal or alert-closure verdict
is inferred; the unpublished candidate requires those later gates.

A fresh all-pages GitHub read still reports 209 open Dependabot alerts:
182 have a patched version and 27 do not; severity remains three critical,
31 high, 82 medium and 93 low. Scorecard has three open findings
(`VulnerabilitiesID`, `CodeReviewID`, `CIIBestPracticesID`). Main remains
`97447185361a37af64c1b30fe87e8e2618d96e20`. The all-pages VAST read at
20:10 UTC returned zero instances and `next_token=null`; no separate volume
audit is claimed. No worker was rented, no model ran locally, and no code was
pushed or artifact uploaded in this slice. GPU use remains conditional on
transfer/setup-inclusive speed, supported operations and unchanged precision;
independent CPU reference and final Apple CPU/Metal/no-fallback are mandatory.
All 194 public rows, 58 unresolved code/artifact rows and attachment repair,
provenance, legal and security tasks remain in scope.

### Realtime dependency conflict resolution and resource readback (20:36 UTC)

Root reviewed and committed the separate Realtime dependency candidate as
`7901e95543bcd45fdf2b535f55db516e037dad8d`, on top of the accepted
`0f204ea9575a210727c94a0480bf96c31baa8b28` inventory candidate. It resolves
the earlier direct constraint conflict using `uv add --no-sync` with
Diffusers 0.38.0 and safetensors 0.8.0. The original Vokra reference pins
remain historical provenance, not independently authenticated Microsoft
requirements. Torch, tokenizer, backend, source and precision constraints
remain unchanged. No environment was installed or model executed.

Root independently passed the six TOML/lock-only tests, the offline Realtime
lock check (57 package records), zero-dependency, forbidden-symbol,
documentation-reference and diff gates. The primary PyPI 0.8.0 JSON readback
confirmed Python >=3.10, extras-gated Requires-Dist, the selected Linux wheel
SHA-256 `fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774`
(516,040 bytes), and sdist SHA-256
`fabaf3e0f18a6618d9b36560682562157f77c2b71fcffc7b432be2baed9d753d`
(325,846 bytes). Both PyPI license fields were null. This is package metadata,
not a license grant or native-payload audit. The previous installed-closure
audit is explicitly stale for the new lock. Release-specific primary license,
API compatibility, independent real-weight CPU parity, Apple/no-fallback and
alert closure remain open; no push or upload was performed.

The source-only FireRed observer-envelope candidate remains unaccepted while
Linux test portability, the actual preparation/output-path tests, malformed
registry roles and accepted-builder identity binding are corrected. Its
preparation-only envelope is not consumed by the Rust v2 reader and cannot
substitute for real upstream capture or numerical parity. The integrated
VAST controller likewise remains a preparation candidate until its actual
CosyVoice/VibeVoice source-collection phases and small evidence packet are
fully bound and reviewed. No worker was rented in this slice.

An initial resource query without the task environment returned an invalid-key
error and is not treated as inventory evidence. After silently loading the
existing task `.env`, the authoritative all-pages VAST instance read returned
`instances=[]`, `total_instances=0`, `success=true`, `next_token=null`; the
separate `show volumes --type all --raw` read returned `[]`. Thus this
readback found neither retained instances nor separate storage volumes. No
credential was printed, passed as a CLI argument or transferred to a worker.
The full 194-row scope and all 58 unresolved rows remain unchanged.

### Observer preparation acceptance and integrated test-name review (20:43 UTC)

Root accepted the source-only FireRed observer-envelope preparation slice as
`333fc8c5d0f3f300f14f11473d5e6f271eb1841a`. The reviewed builder is loaded
only after its exact 27,190-byte size and SHA-256
`106504cd323abbbdbe197ce4fc77342e3a420e5334215b61292630414b2aa882`
are authenticated. The original upstream source is parsed, not executed.
The tests now use portable temporary paths, reject malformed registry roles
and tampered/aliased builder bytes, and cover the actual preparation output
and no-clobber path. Root independently passed ten tests with the explicit
authenticated source record; the portable default passed eight and skipped
two optional source-record tests. Zero-dependency, forbidden-symbol,
documentation-reference and diff gates passed. The envelope still says
`PREPARATION_ONLY_NO_UPLOAD` / `OPEN_NOT_CAPTURED`; external artifact
digests are declared prerequisites, not independent file authentication, and
the Rust v2 consumer does not yet consume this envelope. No real capture,
model execution, CPU/Metal parity or owner/legal approval is inferred.

Root separately repeated the integrated source-only tools' documented
`--self-test` entry points: CosyVoice 23, Realtime 25, FireRed builder 11
plus authenticated-source AST roundtrip, and FireRed module registry nine
with no source-record skips. No upstream source was imported or executed.
These are local standard-library checks, not remote Cargo or model results.

Inspection of the actual Rust test source found that the v2 tests are in
`decoder_trace_v2`, not `tests`, and the ignored function is
`decoder_trace_v2::firered_asr_aed_l_real_decoder_trace_v2_consumer`, not
the initially proposed controller name. The controller's local/remote
matchers and mocks are being corrected against those actual names before
rent. Its registry gate also requires an explicitly fetched authenticated
source record; accepting the portable two skips would weaken that gate and
is not permitted. The integrated source-collection/packet work remains
unaccepted pending complete controller review. No VAST worker was rented
and no implementation was pushed in this slice; all public/attachment scope
and real-weight/Apple/publication gates remain open.

### Source-license preparation and production-controller review (21:01 UTC)

Root independently reviewed and committed the four-file safetensors/Diffusers
source-license preparation slice as
`2a6cc35a476c817ad624fc626d5b27b78b1f94e7`, based on the accepted Realtime
dependency candidate `7901e95543bcd45fdf2b535f55db516e037dad8d`. Its standard-library
self-tests passed 11 and nine tests respectively, including the optional
saved-primary-record mock collection on this host. These are offline tests,
not actual Linux collection; the optional tests skip when those saved records
are absent. The collectors authenticate fixed tag chains, source LICENSE
bytes and PyPI artifact/Requires-Dist metadata, enforce a whole-run deadline,
and preserve exclusive bounded output through directory file descriptors.
Synthetic reports explicitly remain `SYNTHETIC_UNTRUSTED`, including the
safetensors PyPI license-status field. Root's fresh primary Diffusers tag,
LICENSE and PyPI readbacks matched the fixed identities; no distribution
payload was downloaded. Documentation-reference, forbidden-symbol,
zero-dependency and diff gates passed. Distribution/native and transitive
license review, owner decisions, API compatibility, real-weight CPU parity,
Apple/no-fallback and security-alert closure remain open. No upload is implied.

Root also reviewed and committed the two-file FireRed decoder comparator
preparation slice as `4678f40f4667971a5649e5a71abcf56967fe3ce1`, based on
the accepted observer preparation `333fc8c5d0f3f300f14f11473d5e6f271eb1841a`.
The parser retains typed row/layer/cache/candidate/prune observations and
charges checked container storage before allocation against the unchanged
128-MiB ledger. The shared comparison helper consumes actual native trace
getter slices without recomputing decoder arithmetic; synthetic positive and
drift/resource-negative tests are preparation only. An unconditional OPEN
abort remains before real model opening. The observer envelope does not
authorize capture; authenticated capture/provenance/owner binding and the
source rank-four projected-K/V to native-axis mapping remain unresolved.
Root independently passed formatting, zero-dependency, forbidden-symbol and
diff checks. Rust compilation, these new tests and Clippy have not run yet;
they require a later exact-head VAST batch. No numerical or hardware verdict
is claimed.

The integrated clone remains frozen at
`58177c040625f8cb714d2f8689c697e4a441cbe5`, based on main
`97447185361a37af64c1b30fe87e8e2618d96e20`. Its controller candidate
`5f833f4ea52fabce29bdbaa5c9e514770b67cdd1dea4693a2dfc1fcb1de24067`
passed agent-reported offline mocks but is not accepted for rent. Root found
production-path defects: the source subdirectory violates the flat main
packet's exact-member check, registry CLI verbosity does not match the named
test gate, and unescaped remote heredoc variables can expand locally before
SSH. Source-packet pre-read bounds/pin binding and unnecessary full reference
provisioning also require correction. Those corrections are delegated before
any paid execution; the two newer commits above are intentionally not added
to this frozen batch. Earlier controller mocks are not evidence that these
production paths work.

The completed read-only HF metadata audit at the integrated head still reports
194 public repositories, 193 GGUF-bearing repositories and 198 files; CPU
code/artifact classifications remain 136 full, 43 partial, 14 without a runtime
binder and one non-artifact, with 58 unresolved rows. Metal remains 136 full,
57 blocked by CPU and one non-artifact. This is not a real-weight or Apple
completion verdict. All public rows and the attachment's artifact, provenance,
legal and security tasks remain in scope. The user's GPU preference is applied
when supported operations and unchanged precision yield a setup/transfer-inclusive
speed advantage; independent CPU reference and final Apple CPU/Metal/no-fallback
remain mandatory. This source/Cargo-only batch is CPU-oriented. No model ran
locally and no implementation was pushed in this slice.

The fresh all-pages VAST readback returned `success=true`, `instances=[]`,
`total_instances=0` and `next_token=null`; the independent
`show volumes --type all --raw` read returned `[]`. No VAST worker was rented
in this slice and no retained compute or separate storage volume was found.

### Projected-K/V review and stale Canary closure (21:20 UTC)

Root accepted the two-file FireRed projected-K/V preparation change as
`26e583347ae7fd51e45cdb9d6ff304d76d879483`, based on
`4678f40f4667971a5649e5a71abcf56967fe3ce1`. The reader validates the
authenticated source rank-four `[N*B, heads, prefix, head_dim]` axes and
explicit source row/last-prefix selection before flattening the selected
head-major row to the native observation. Real packets bind the existing
authenticated 20-head decoder configuration; smaller schema-only fixtures
cannot authorize the real leg. The parser retains and checks projected head
geometry across rows and layers. An unconditional OPEN abort before model
opening is unchanged.

Review corrected an ambiguous helper lifetime, a prospective eight-argument
Clippy warning, a non-finite negative that previously broke shape before
reaching the finite-value guard, and an overflow negative whose first product
still fit in `usize`. Its first revised value also overflowed the fixture's
signed JSON encoding rather than the intended multiplication. The accepted
fixture uses representable dimensions and checks the exact multiplication
overflow diagnostic without allocating a large payload; NaN/infinity tests
now retain valid nested geometry and check the actual finite-value diagnostic.
Distinct multi-head values and unequal prefix/head axes exercise ordering.
Root independently passed formatting, forbidden-symbol, zero-dependency,
documentation-reference and diff checks, and counted 32 test declarations:
30 non-ignored and two ignored. Those are source counts, not executed Rust
test results. Rust compilation, Clippy and real-weight numerical/Apple checks
remain pending. This commit is intentionally not added to frozen integrated
head `58177c040625f8cb714d2f8689c697e4a441cbe5`; nothing was pushed.

Read-only review of the Canary Flash/v2 and SpeechBrain Lang-ID replacement
paths found no new VAST-ready real-weight row. Existing Canary dependency
facts contain primary license/native evidence for NumPy, SciPy and soxr, but
remain blocked by their recorded forbidden-license signals and pending exact
owner/legal decisions. These are historical facts for audited head
`87da78dc7709075d9dc23b797fc978b9c678c777`, not the current closure: the
manifest's source snapshot binds project SHA-256
`31c002238c213d64f78f38f68f613f168d12991df6b3aac48dd95662da85245e`
and lock SHA-256
`004f0b4d60ba51caf655789eff6e02afb3fa896f837f8d4def909b5cce33b730`,
whereas the current checkout has project
`473d101eb8ac955cba98c605dd41715899c21b6e4e57292685edd2edcbd57838`
and lock
`8daa624c8f8f37dedb553aed7af4a0fc52e8161e95f1525727359c921796fbe3`.
The current project pins Torch 2.13.0. Fresh exact-closure audit/API evidence
is therefore needed before those older package facts can support an execution
decision. The recorded Lightning/OneLogger import incompatibility remains a
separate historical blocked result, not a fresh probe of the changed lock.
No license exception, model execution or approval is inferred.

The integrated controller's fixed collector-source AST pin binding and stale
base correction were reviewed. Its production recovery path still allowed
recursive source SCP after a failed remote size/count guard, and the remote
enumeration skipped report-size and directory-symlink hazards. Further bounded
pre-transfer rejection and regression coverage were delegated before rent;
agent-reported offline success is not acceptance of that production path.
The full 194-row catalog and attachment scope remain open, with no new
real-weight, Apple or publication verdict in this slice.

The subsequent all-pages `show instances-v1 --all --raw` read returned
`success=true`, `instances=[]`, `total_instances=0` and `next_token=null`;
the independent volume read returned `[]`. No worker was rented in this
slice. Documentation-reference, runbook-path, forbidden-symbol,
zero-dependency and diff gates passed for this management update.

### 2026-10-01 XCodec2 exact-head branch update (01:15 UTC)

Fresh GitHub reads confirmed main at
`97447185361a37af64c1b30fe87e8e2618d96e20` and draft PR #152 at
`35e4c70e830a675e3f9a124347e8fa3d23b77373`, behind main with base
`81d72f2317a3bd2462cb7abbca15ac55b7518d56`. The three source/audit helper
commits `6185ea8e`, `018c0174` and `f260f884` are already ancestors of that
PR head; root also found matching collector, inspector, builder, builder-test,
project and lock hashes in the PR and helper checkouts. They must not be added
again as supposedly unintegrated work. Alert #159 remains open for Torch
`<2.6.0`, with first patched version 2.6.0. A patched candidate is not an
accepted real-weight result or a closed main-branch alert.

Root requested GitHub's branch update guarded by that exact old head. The
resulting PR head is `7dc3c492fcd44d6cfa7bbdc915b190fa7330ebd1`, based on
the confirmed main head above. It remains open and draft. New-head CodeQL
run `36800209413` and secret-scan run `36800209421` were in progress; main CI
run `36800209675` and the quality/platform/security/coverage workflows were
queued, and pins-sync-check `36800209448` had succeeded at readback. Old-head
CI is not proof for this merge head. No merge, model execution, owner/legal
approval, artifact update or security-alert dismissal occurred. The frozen
FireRed VAST batch remains at its separate exact head.

### 2026-10-02 resumed goal and bootstrap-failure readback (02:37 UTC)

The active goal remains all 194 public repositories, including the attachment's
artifact/provenance/legal and security scope. Completion means authenticated
native routes, independent real-weight VAST CPU parity, Apple CPU/Metal and
explicit no-fallback measurements, separately authorized correct public bytes,
and reconciled security/CI/cloud disposition. The dated 136 code/artifact-full
and 58 unresolved classification is not an Apple-completion count. A withheld
row remains explicitly unsupported, not a supported-model success.

The reviewed controller SHA-256
`75d88822fb09d8cb178f6e4b0006b5880ab6f24c427370acfbb032c6048f5939`
passed root's full offline self-test before renting disposable worker
`53663629` for clean target `58177c040625f8cb714d2f8689c697e4a441cbe5`.
Its actual remote run stopped during toolchain bootstrap: the authenticated
Rustup binary was executed as `vokra-rustup-init`, which Rustup rejected as
an unknown proxy name. No Rust tests or model execution followed. This attempt
is failed setup, not verification evidence. The recovered bootstrap-log
SHA-256 is `cb22551ce0bfc69b3e7c805b466b1776dd3b38f670f707462c7158bb148e115d`.
Bounded main diagnostics were recovered; the source-packet guard failed
because bootstrap had not produced the required evidence. Cleanup recorded
`destroy_rc=0`, strict individual readback with `instances=null`, and an empty
owned-label list. The controller-log SHA-256 is
`5a5b56de7477551a2e3a72fd650f99d65479015f3427d0b9a6f4d739ffe9bc24`.
The minimal bootstrap repair and its offline regression were delegated;
they are not yet accepted or rerun.

Fresh GitHub reads still show main at
`97447185361a37af64c1b30fe87e8e2618d96e20` and draft PR #152 at
`7dc3c492fcd44d6cfa7bbdc915b190fa7330ebd1`. All checks are terminal;
`documentation-links` is the sole failure and merge remains blocked. Bounded
official e-Gov API GETs returned HTTP 200 with law ID `415AC0000000057` in
their actual bodies. The proposed current USCode Chapter 5 alternative also
returned HTTP 200, but its body was an `Under Maintenance` page; it was rejected
as a substitute for the statute. Only the two verified e-Gov citations were
delegated for a minimal link correction. No legal prose, checker exclusions,
owner approval, model result, alert dismissal or publication is changed by
these observations.

The current all-pages VAST list returned `success=true`, `next_token=null`
and one stopped, unrelated `ralomi-development-*` instance; no Vokra worker
remains. The independent volume list is empty. The unrelated instance and its
storage were not modified. The resumed goal remains incomplete.

### 2026-10-02 PR lint repair and second bootstrap readback (02:57 UTC)

The verified e-Gov citation-only correction was committed as
`152a3695508ed7372ae7cfd06a18e53e11af1fee` and pushed to draft PR #152.
Its new-head `documentation-links` check passes. Main remains
`97447185361a37af64c1b30fe87e8e2618d96e20`; the PR remains open, draft and
blocked. Multiple new-head Clippy/backend/parity-matrix jobs fail on the same
Rust 1.99 `single_element_loop` diagnostic in the SBV2 converter, not on a
measured model-parity mismatch. The minimal direct-lookup repair preserves
primary speaker-key precedence and the existing fallback. It was reviewed and
committed as `1e4bd71f33956f56e3360787dffc341b999dce8b`; integrated verification
HEAD is `3570e4e86ef936a92f62e857e9302d57aba6e974`. The code has not been pushed
or described as remotely verified.

The retargeted controller SHA-256
`11be1f9349349bf8dd7b9cb273d3e92fae7d0186d06c8e893f6c22f8e386fc8f`
passed shell syntax, ShellCheck and the full offline controller self-test.
Root also confirmed that only the target constant differed from the accepted
bootstrap-repair controller and that the target checkout was clean. It rented
exactly one disposable worker, `53794282`, with 16 effective CPU cores,
128,483 MB RAM and 200 GB storage. The real Rustup repair succeeded: Cargo and
Rust 1.99, UV 0.12.5 and managed Python 3.12 were available. Bootstrap then
failed with exit 128 because a second bundle fetch attempted to update the
already checked-out `verification` branch. No Rust test, real-weight parity,
Apple run or model upload occurred. The bootstrap-log SHA-256 is
`30b7b891af985d3d72acafa4afb2598a94dfeb4cab11425d40aafa2f0111baf9`.

The controller destroyed `53794282` including its storage. Cleanup records
`destroy_rc=0`, strict individual readback with `instances=null`, and an empty
owned-label list. Recovery bounds rejected the incomplete result packets; no
green verification packet is claimed. Controller-log SHA-256 is
`6174ebb5f6ed650151821a4ba210dddd2d910fcf50a91543c43c7deac45a0b5a`.
The redundant-fetch repair and an actual-Git offline regression were delegated
before another rent; target, source pins and cleanup contracts remain frozen.

A fresh all-pages GitHub Dependabot read found 209 open alerts: 182 have a
published patch and 27 do not. These are current residuals, not dismissals or
successful reference-environment updates. All public-model, independent
reference, owner/legal, Apple/no-fallback and publication requirements remain
in scope; this setup attempt closes none of them.

### 2026-10-02 Scorecard supersession (03:03 UTC)

The current all-pages Code Scanning read and actual SARIF from successful
main run `36847827407` agree that only three Scorecard findings remain open.
`MaintainedID` and `SASTID` are fixed without dismissal; Best Practices,
independent Code Review and Vulnerabilities remain open. Exact-head evidence,
SARIF hash and current Dependabot counts are recorded in the
[security readback](security-remediation-2026-09-21.md#2026-10-02-current-security-readback).
This supersedes the older five-open observations, not the full completion
scope or the outstanding security remediation.

### 2026-10-02 first real compile and bounded correction (03:16 UTC)

Root reviewed controller
`c10e747ffeaa383e6e29ee90be1d5ec46f9a88c4567c5407c13ec4781e9dac55`
and independently passed its full offline test, including the production-bound
single-fetch guard and the actual-Git failure/success regression. The replay
of clean head `3570e4e86ef936a92f62e857e9302d57aba6e974` used exactly one
disposable worker, `53795614`. Bootstrap succeeded and an authoritative SSH
read confirmed that exact head, `jobs=16` and live Cargo/Rust compiler
processes. Source-audit and Rust-source-audit tests exited zero.

Real compilation exposed seven `E0425` errors in FireRed's nested unit tests:
`super::AUTHENTICATED_DECODER_N_LAYER` resolved to the `native` module rather
than its parent. Focused model tests could not compile; this is not a numeric
parity failure. Root reviewed the exactly seven reference corrections and
committed them in an isolated clean checkout as
`824fd33f38cb245e996cdd3c64a2bb51ed6665a8`. Production code, dimensions,
assertions and bounds are unchanged. Formatting, forbidden-symbol,
zero-dependency and diff checks passed; the new head is not remotely verified
or pushed. A separately retargeted controller is being prepared without
modifying the completed run's source or controller.

The original run ended with exit 1 and its disposable worker was destroyed
including storage: `destroy_rc=0`, strict individual `instances=null` and
owned-label absence all passed. Incomplete packets failed the recovery bounds;
no recovered green result is claimed. The controller-log SHA-256 is
`dceecee3c290d9c19ee9eaa936b4c9ece7db3d94115b62569a7720f49e71fc9d`.
Diagnostic-recovery limitations are being investigated separately; they do
not weaken completion gates.

The next-family review retains current execution blocks for Qwen3-TTS,
Canary and Lang-ID: changed dependency locks or incomplete unsigned scope
cannot inherit a historical approval. SGMSE is not reopened. Its unchanged
converter/runtime, accepted Apple consumer `bed6cc3f` and separately published
revision `c37e93159b4129b2c582c44f8170b44cf6e3e531` preserve the completed
exact-artifact verdict in the [Apple reconciliation record](mac-cpu-metal-scaleway-results-2026-09-11.md).
Later verifier plumbing changed neither runtime, references nor bounds;
repeating that completed model would not close another unresolved row.

### 2026-10-02 corrected-head replay and remaining compile/guard failures

The separately frozen controller
`b37470f9631e437ba09ed3ccede410c74abf3856c6452c63bc5b294c3510b5aa`
targets clean head `824fd33f38cb245e996cdd3c64a2bb51ed6665a8`. It explicitly
declares the repository path in the separate verification SSH shell; the
preceding run's actual `REMOTE_REPO: unbound variable` diagnostic explains
why bootstrap's variable did not survive into that shell. Root reviewed the
bounded fix and passed syntax, ShellCheck and the full offline regression,
including extracted-production declaration ordering and missing-declaration
negative cases. No model identity, numerical bound or cleanup gate changed.

Exactly one disposable worker, `53797800`, replayed that head. Bootstrap
succeeded; an authoritative SSH read confirmed the exact head and 16 build
jobs. Actual focused Mimi, ABI, PCM and FireRed composition tests exited zero.
The converter focused tests and warning-denied converter Clippy also exited
zero. CosyVoice and VibeVoice source-only collectors exited zero. These are
partial, directly observed results, not a recovered complete green packet or
real-weight/Apple parity evidence.

The actual FireRed consumer compile exposed two errors: a synthetic JSON
object iterator yielded borrowed keys where owned strings were required, and
an intentionally failing OPEN-path diagnostic used adjacent string literals
without Rust concatenation. Warning-denied model Clippy also exposed unused
test helpers, redundant final Option reborrows, equivalent collection checks
and missing field documentation. Bounded corrections were delegated; the
intentional OPEN verdict, assertions, tensor contracts and numerical guards
must remain unchanged.

Two controller defects were diagnosed separately. The full FireRed guard
retained the previous native-source SHA in its remote comparisons, rejecting
the correctly retargeted source with exit 125. The source-record step invoked
a child `sh` without exporting its evidence-directory variable, so subsequent
source instrumentation and registry tests could not find their required
authenticated record. The running controller and source remained frozen;
corrections and production-bound negative regressions are being prepared in
a separate controller. No source or hash check is bypassed.

The replay ended with exit 1. Incomplete evidence failed the existing recovery
bounds; no local green packet is claimed. The controller destroyed the worker
and all storage: destroy and readback exited zero, the individual response
was `instances=null`, and the exact owned-label list was empty. Controller-log
SHA-256 is
`e021637fc6cb577a91996249c5997d8cc09beb0ce359d61f678f13a0056c676e`;
bootstrap-log SHA-256 is
`6e74ef4eea7ee68aa10c6172da4cf120e954042d18c7401543c88612235d6c6e`.
No upload, Apple execution or additional model completion occurred.

The subsequent bounded consumer/model-lint correction was reviewed and
committed in a separate clean checkout as
`6ebd68964be69fb57c771fce3cf95c73ab7229c1` (three files, 22 insertions and
16 deletions). The consumer retains its intentionally failing, ignored
real-weight v2 OPEN test. Test-only helpers are excluded from production;
collection checks and final trace reborrows are equivalent. Root independently
passed format, diff, zero-dependency and forbidden-symbol checks. No remote
success is claimed for this new head; the corrected controller must bind its
source, native and consumer hash comparisons to that exact head before replay.

### 2026-10-02 exact-head tests and remaining mask/lint failures (03:55 UTC)

The separately frozen controller SHA-256
`81bd80c0960b00d64d3f492be8d7bdb2ace44b3e8ac26d1eaa7772046da61f2c`
replayed clean head `6ebd68964be69fb57c771fce3cf95c73ab7229c1` on exactly
one disposable VAST worker, `53800018`. An authoritative SSH read confirmed
that head and a live serial-test workspace invocation. Bootstrap installed
Rust/Cargo 1.99, UV 0.12.5 and managed Python 3.12. No source was changed
during the run.

Direct SSH observations before cleanup showed exit zero for source and Rust
source audits; focused Mimi, Mimi ABI, PCM and FireRed tests; converter tests
and converter Clippy; the FireRed consumer tests; CosyVoice and VibeVoice
source collectors; metadata binding; FireRed wire, source builder, module
registry and authenticated source record; and Cargo deny/audit. The corrected
remote source hashes and evidence-directory inheritance therefore progressed
beyond the preceding failures. These observations are not a complete recovered
packet, real-weight parity, Apple evidence or publication authorization.

The full FireRed unit selection ran 68 tests: 67 passed and one failed.
`decoder_self_attention_causal_and_incremental_cache_match` supplied a key
mask inconsistent with the runtime's cache-plus-current-query contract and
failed with `InvalidArgument` at the masking comparison. The runtime guard
must not be relaxed to make the test green; a separately reviewed correction
must preserve the causal, masked/unmasked and incremental comparisons.
Consumer Clippy reported six diagnostics: four constant assertions and two
redundant closures. Consumer compilation and ordinary tests now pass, but
warning-denied Clippy, full FireRed tests and workspace tests do not. Mimi
Clippy also exited 101; its detailed diagnostic was not recovered and its
cause is not inferred here.

The controller ended with exit 1. Main-packet recovery still rejected the
result with `SKIPPED_BOUNDS_FAILURE`; the local evidence directory is empty.
The observations above came from direct SSH output before destruction, not a
locally checksum-verified archive. Recovery is being investigated separately
without weakening mandatory records, hash checks or size limits. No Moshi
source-cache success is inferred from this failed main verification.

The worker and its storage were destroyed. Individual readback returned
`instances=null`; the owned-label list returned `success=true`, an empty
instance list and `next=null`, and strict cleanup passed. Controller-log
SHA-256 is
`266a4675eb211b7fa0e0a65b888c8db6d0ebdd4f8e936ab0881213e6c80b60fc`;
bootstrap-log SHA-256 is
`7f3ac185f305bfdf35d44b87e653d1b0624b4962253a9cf5991684726703e34e`.
No unrelated resource was modified. No model weight, HF token, Apple worker
or upload was used by this source-only run.

The next VibeVoice structural candidate was prepared separately from current
main `97447185361a37af64c1b30fe87e8e2618d96e20`, at clean head
`c08f3b2659716bfd07b224c91b921a4d932e1753`. It assembles 22 existing
dependency-ordered language, continuation, control-plane, connector, cache
and preset commits plus the reviewed SBV2 lint correction. Root independently
passed formatting, diff, zero-dependency and forbidden-symbol gates. Review
is still in progress; this candidate has no new remote compile/test verdict.
The native full-runtime aggregate, full streaming reference and real-parity
consumer are not included. Preset consent/rights, exact execution approval,
real cache replay, complete waveform parity and Apple CPU/Metal/no-fallback
remain open. Existing staged code must not be duplicated or counted complete
merely because it has been assembled.

The campaign goal remains all 194 public repositories, not only this FireRed
batch or the 136 code/artifact-full rows. Withholding must have an explicit
exact-scope owner disposition and is not supported-model completion. The next
steps are bounded source/lint corrections, reliable failed-run evidence
recovery, exact-head remote verification and the remaining real-reference
work; Scaleway remains the final hardware leg rather than a substitute for
these missing gates.

The subsequently reviewed correction is committed in a separate clean
checkout as `53192b06eb6111ce15982cba875f319923bc533d` (two files, 54
insertions and 34 deletions). The mask fixture retains the extreme stored K/V
row and uses the correct three-entry mask for two past frames plus one current
query. Its original masked-versus-past-only equality, unmasked difference,
causal and incremental comparisons remain, with an additional all-masked
rejection. No production mask validation or numerical bound changed. Consumer
target checks remain runtime fail-closed on Linux x86_64 little-endian only;
the ignored real-weight v2 OPEN verdict remains intentionally failing.
Root independently passed format, diff, zero-dependency and forbidden-symbol
checks. This new commit is neither remotely verified nor pushed.

A fresh all-pages VAST inventory after destruction returned no Vokra
instance. The sole account instance was the unrelated stopped
`ralomi-development-41259f7-20261001-urgent-03`, which was not modified; this
is not a claim that the entire account has no resources or storage charges.

### 2026-10-02 coherent FireRed / Realtime source-only candidate (04:14 UTC)

The earlier structural-only Realtime checkpoint remains historical. A separate
full-stack checkout at `0340a146cf5c14dec6f21d25c020c350409a15d5` combines
the reviewed structural commits with the existing native streaming composition,
selected-backend Metal dispatch, official streaming-reference caller,
model-free cache-compatibility probe and ignored real-packet consumer. It does
not duplicate the previously staged implementation or supply an owner approval.

The next coherent source-only verification candidate is clean head
`d24366cbcbb9018e33262ad37a24d734d4d75167`, based on the reviewed FireRed
correction `53192b06eb6111ce15982cba875f319923bc533d`. Exactly 30 existing
Realtime commits were cherry-picked without conflict, excluding the SBV2 lint
fix already present in the parent. The resulting 19 changed paths (8,651
insertions and 129 deletions) are byte-identical to their full-stack counterparts.
FireRed module, native implementation and consumer remain byte-identical to the
reviewed parent. Historical candidates and controllers remain frozen.

Root independently passed formatting, diff, first-party-only lock and
forbidden-symbol checks, plus the streaming-reference and cache-probe stdlib
self-tests on this exact combined checkout. The implementer also passed the
preset-exporter and installed-closure audit self-tests. These checks did not
compile or execute models, import model dependencies, load a preset, download
weights or produce numerical parity. Workspace/model Cargo remains VAST-only.

The native composition selects CPU or Metal explicitly and rejects unsupported
backends; dispatch support alone is not actual Apple execution. The independent
caller invokes the fixed official Microsoft generation path, but its real run
requires an external hash-bound execution scope, exact dependency disposition
and proved preset consent. Its PCM consumer reports measured error with an
explicitly OPEN numerical gate; neither structure agreement nor device-selection
diagnostics may be promoted to waveform parity. The ordinary synthesis/CLI
completion and real-weight CPU/Apple/no-fallback gates remain open.

The failed-run recovery controller is being corrected separately. Review found
that local diagnostic archive acceptance still needed an exact filename allowlist,
complete member/manifest equality and expanded-size validation before extraction.
Those corrections are delegated; no reviewed/frozen controller has been modified
mid-run and no new worker has been rented for this candidate. Diagnostic recovery
must stay failure-only and preserve the original failure exit, mandatory green
records, source hashes, transfer caps and frozen Moshi source-cache budgets.

The full 194-row goal remains unchanged. The dated 136 code/artifact-full and
58 unresolved classifications are not an Apple-completion count, and withheld
rows must remain separately accounted rather than described as supported.

### 2026-10-02 accepted recovery controller and active verification (04:36 UTC)

The 04:14 preparation record above remains historical. Root review accepted
the separately frozen diagnostic controller at SHA-256
`7f8595f15b669675f1b7bd0a2782a8513b6fefc040dbebf1c67d0dc199971dd5`.
Its diagnostic-only recovery rejects compressed, sparse, duplicate, unknown,
non-regular and oversized archive members before extraction, requires exact
file/manifest membership and verifies recovered checksums. Positive-control
extraction and oversized/compressed no-extraction fixtures passed the root's
independent self-test, along with shell syntax and ShellCheck. A recovered
failed diagnostic never promotes the original run to a green verdict; the
mandatory green records, source hashes and Moshi budgets remain unchanged.

Disposable VAST instance `53804203` started the exact combined candidate
`d24366cbcbb9018e33262ad37a24d734d4d75167`, with 128,483 MB reported RAM,
16 effective CPU cores and 200 GB disk. Compilation uses the allocated 16
jobs; tests remain serial to avoid known allocator-counter interference.
Direct SSH confirmed the clean exact HEAD and live workspace-test process.
The 19 completed source/focused/metadata/converter checks, including FireRed
full tests and consumer Clippy and Mimi all-target model Clippy, each recorded
exit zero. Workspace verification was still running at this snapshot, so no
full-workspace, Moshi source-cache or recovered-packet verdict is claimed.

This worker is source-only: no model weight, preset execution, HF token,
Apple worker or upload is involved. Evidence recovery and destruction,
including instance storage and authoritative absence readback, are required
before its lifecycle is complete. No unrelated account resource was modified.

A future reference-only correction remains outside the frozen candidate.
It preserves an authenticated CPU packet before optional CUDA diagnostics,
but root review found a production callback-binding defect and requested a
correction and a production-bound regression test. It is not accepted, merged,
remotely verified or used as real-reference evidence by this snapshot.

### 2026-10-02 terminal recovery failure and reviewed follow-ups (04:41 UTC)

The active-worker snapshot above is superseded by terminal controller exit 1
for VAST `53804203`. Remote verification recorded exit 1; source collection
reported `REMOTE_COLLECT_GUARD_FAIL:SOURCE_TOTAL_LIMIT`. Main fallback recovery
reported `MAIN_RECOVERY_GUARD_FAIL:UNEXPECTED_NAME`, and diagnostic-only recovery
reported `FAILED_DIAGNOSTIC_ONLY:MAIN_UNEXPECTED_NAME`. The local evidence
directory is empty. The earlier direct-SSH observations prove only the 19
specific completed checks, not the final workspace, deny, audit, source packet
or Moshi verdict. No green packet, full-workspace success or numerical/model
parity is inferred. The rejected filename and exact source-size cause are not
recoverable from these bounded reason codes; a separate controller correction
and realistic production-output tests are being investigated before another run.

The instance and its storage were destroyed. Cleanup records `destroy_rc=0`,
individual readback `instances=null`, an empty owned-label instance list with
`success=true` and `next_token=null`, and strict cleanup PASS. A subsequent
fresh all-pages account read returned only unrelated instance `53677077`
(`ralomi-development-41259f7-20261001-urgent-03`, exited, 500 GB); it was not
modified. There is no remaining Vokra instance in that readback, which is not
a claim that the entire account has no storage charges. Controller-log SHA-256
is `0acf5419a56ddb1cca126b727fd3f0800190d8b2202e35b71569aa5ee7e41e36`;
remote-collect-log SHA-256 is
`bd275b1cbeb5dd89841b5c8fee197dd47eddf85986de68958b1b052f5c99fb2b`.

Root accepted the separate reference-only correction as commit
`ed820f4e31bbdce29bc82c457eca48bb064f59c9`. It authenticates both tensor and
official cache-metadata CPU records, binds the CPU trace manifest to the packet,
atomically finalizes CPU evidence before optional CUDA, preserves dedicated CPU
integrity failures and user interruption, and reports staging/rollback failures
explicitly. The reviewed production callback binding is now defined. Root's
offline stdlib self-test, diff, zero-dependency and forbidden-symbol checks pass.
This is not an owner approval or actual CPU/CUDA/model parity; the live target
was never changed. The reference script SHA-256 is
`e6f7cc0595b7b5e747b998099e344fdfad16bcfb4c3c0fe7d0dd2810b0731d61`.

The minimal SBV2 lint repair was separately applied to exact PR #152 head
`152a3695508ed7372ae7cfd06a18e53e11af1fee`, without unrelated FireRed or
Realtime changes. Its sole converter file is byte-identical to the version
observed passing converter Clippy on the worker. Root's full format and static
checks pass; the reviewed five-addition/twelve-deletion repair is committed as
`dd6f015` and remains unpushed pending appropriate remote verification. PR #152
is still draft and blocked at its earlier head; no fresh-head CI or merge
success is claimed.

### 2026-10-02 strict failed-comparison JSON follow-up (04:53 UTC)

The future Realtime reference candidate is now clean commit
`04571ee1c020d822be9056568e39b1fede1d9a65`, following `ed820f4`.
Root review found that shape mismatch or non-finite comparisons could otherwise
serialize `Infinity`/`NaN`, making a preserved CPU packet unusable by the strict
Rust JSON consumer. The correction checks finite inputs and computed differences,
records explicit failure status with null magnitudes, and uses strict atomic JSON
writes. It does not widen the existing device-selection guard or normalize an
invalid CPU tensor into success; CPU tensor recording still rejects non-finite
values before finalization.

Root independently passed the offline stdlib self-test and static gates. The
actual comparison function was exercised with stdlib fake arrays for normal,
out-of-guard, stage/PCM shape mismatch, NaN and finite-input subtraction overflow;
PCM failure cases also traversed finalization, preserved CPU hashes and produced
strictly parseable FAILED/selected-CPU packets. These are model-free control-flow
regressions, not numerical parity or a real NumPy/Torch/CUDA run. Script SHA-256
is `9c7cef7b3cf3827c39cca46d3dc1c540b17968697d50d3a64cd6e317c074d647`.
The original VAST target `d24366c` and frozen recovery controller remain unchanged.
This follow-up has not been pushed, merged or remotely verified.

### 2026-10-02 recovery-contract investigation (05:02 UTC)

Read-only inspection found two concrete producer/collector mismatches in the
frozen controller, without recovering the destroyed worker's missing bytes.
The pinned CosyVoice collector declares separate 512-KiB payload and 512-KiB
manifest limits; its 28 pinned payloads total 405,766 bytes. The old transfer
guard instead counted payload and manifest together against 512 KiB. The
VibeVoice collector likewise separates its 4-MiB payload and 1-MiB metadata
budgets. A new controller must preserve these independently declared limits,
source counts, hashes and validation rather than enlarge an arbitrary combined
budget. The old run's exact rejected family and manifest size remain unknown.

The bootstrap also writes `cargo-deny.tar.list` and `cargo-audit.tar.list`
inside the evidence directory, although neither basename belongs to its exact
green-packet allowlist. This source-level mismatch is proved; it is not a
measurement of the old run's rejected basename. Bootstrap-only listings must
move to the existing tool-staging directory while retaining archive-member
validation, not expand the green-evidence allowlist.

Root also found that the final focused and ABI log predicates in
`verify_local_green` were not explicitly failure-chained. In a conditional
shell invocation, a later successful predicate can hide an earlier failure.
Explicit rejection and production-verifier negative fixtures are required.
These corrections are delegated in a new controller; the historical controller
and candidate remain frozen. Full controller self-test acceptance and remote
replay are still pending. No model, parity, Apple or publication result is
promoted by these source findings.

A fresh all-pages VAST read returned `success=true`, no pagination token and
only the unrelated, exited instance `53677077`. No Vokra instance was present;
the unrelated resource was not modified. The credential-redaction/exit-status/
destroy-confirmation wrapper self-test also passed locally without models.

### 2026-10-02 fresh public inventory audit

The read-only Hugging Face API/README audit completed again: 194 public
repositories, 193 GGUF-bearing repositories and 198 GGUF files. CPU
classification remains 136 full, 43 partial, 14 without a runtime binder and
one non-artifact; Metal remains 136 full, 57 blocked by CPU and one
non-artifact. The 58 unresolved rows remain in scope. The audit script and
engine source used for this readback are byte-identical to GitHub `main`
`97447185361a37af64c1b30fe87e8e2618d96e20`, whose remote HEAD was freshly
confirmed unchanged. The audit's 14 local stdlib classification tests passed.
No model weights or tokenizer/preset payload were acquired or executed.

These are code/public-metadata classifications, not independent real-weight
or Apple verification verdicts. PR #152 was freshly read as OPEN, draft and
BLOCKED at `152a3695508ed7372ae7cfd06a18e53e11af1fee`; its isolated SBV2
repair is still unpushed, so no new CI or merge result is claimed.

### 2026-10-02 accepted recovery correction and new active replay (05:20 UTC)

The separately frozen recovery correction has SHA-256
`effa82e425e9c8a2666b85d49d5907256920800b0324898a9203cd43ac50762f`.
Root reviewed its diff and independently passed shell syntax, ShellCheck and
the full offline controller self-test, including the source-budget positive
control, unknown-name diagnostic rejection, bootstrap staging assertions and
explicit focused/ABI failure chaining. The implementer's independent full
self-test also exited zero. Historical controller `7f8595...` is unchanged.
Diagnostic SSH step fields say `process_exit`, not a completed gate verdict.

A separate controller at SHA-256
`c363807fff2654c733f853d06ea9df8dcfd4df970969218aa440a5b5988f9fbe`
changes only the default checkout and target HEAD to clean candidate
`04571ee1c020d822be9056568e39b1fede1d9a65`. Root reviewed that two-line
retargeting and passed syntax/ShellCheck; its full offline self-test also
passed under the implementer. Mandatory green records, source hash pins,
cleanup rules and frozen Moshi budgets are unchanged.

The fresh prelaunch account read contained no Vokra worker and only the
unrelated `ralomi-*` resource, which was not modified. VAST instance
`53808967` was then created for exactly this source-only replay, with 16
effective CPU cores, 128,483 MB RAM, 200 GB disk and an observed total rate
of USD 0.1237037037/hour. Direct SSH confirmed the exact clean target HEAD,
18 completed step exit files equal to zero and the live all-feature model
Clippy process. The new run's CosyVoice and VibeVoice manifests measured
334,054 and 433,176 bytes respectively; these new-run measurements do not
recover the old failed run's unobserved bytes or rejected filename.

Workspace, final audit, recovered packet, Moshi and cleanup results remain
pending at this snapshot. No model/preset execution, weights, HF token,
Apple run or upload is involved. The lifecycle must recover small evidence
and destroy this instance and its storage before completion is claimed.

### 2026-10-02 terminal replay and PR head supersession (05:38 UTC)

This supersedes the active-worker and unpushed-PR snapshots above, without
changing their historical observations. The replay at exact clean HEAD
`04571ee1c020d822be9056568e39b1fede1d9a65` is terminal with overall exit one.
All 22 recovered step exit files are zero; the workspace log contains 325
successful summaries totaling 8,219 passed, zero failed and 110 ignored.
Those process results are not an overall green verification verdict.

The FireRed consumer's actual Rust result is 24 passed, zero failed and two
ignored, but its strict identity predicate reports
`firered_consumer_result_count=FAIL`: `--nocapture` interleaves expected caught
panic diagnostics with successful test identities. Independently, the broad
Mimi filter produces 79 passed tests rather than the local verifier's exact
ten-test contract. The source builder reports 11 tests with two skips because
its inner tests use a fixed Mac fixture path instead of the explicit remote
source record. These output-contract defects remain open; no test failure,
skip, real-weight parity or Apple result is promoted. Fixes must use actual
observed test summaries, never manufacture a Rust success summary.

The main and source-only packets were recovered and both local checksum
verifiers passed. The source collector separately authenticated 28 CosyVoice
payloads totaling 405,766 bytes with a 334,054-byte report, and 21 VibeVoice
payloads totaling 368,221 bytes with a 433,176-byte report. Frozen Moshi
source-cache execution was not reached. Evidence SHA-256 values are:

- workspace log: `5a6008c13ae401f99853a56c523ffe0281dda59fc03d9f6db3dee244d528a8ed`;
- remote packet manifest: `6f8568739c2345a1ba477b2abb286ea7f4166e4a48c068986ce76274adc573e9`;
- lifecycle controller log: `eded3d6b06ca27bf26662bf72a1b6b28b242be444718907463aeed2a68d61d65`.

VAST instance `53808967` and its storage were destroyed. Exact destroy
readback returns `instances: null`; the all-pages owned-label readback has
`success=true`, no pagination token and no matching instance. Strict cleanup
verification passes. The unrelated `ralomi-*` resource was not modified.

The isolated PR #152 SBV2 converter repair at
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a` passed exact-head, warnings-denied
converter Clippy on a separate checkout on that same VAST worker before
destruction. It was then fast-forward pushed to the existing PR branch; the
campaign root branch was not pushed or merged. A fresh GitHub read at this
snapshot confirms the new PR head, 68 successful checks, three skips, six
in-progress checks and zero failures. PR #152 remains OPEN, draft and BLOCKED;
no completed-CI or merge verdict is claimed.

### 2026-10-02 source-test repair, terminal PR CI and live security readback

The portable source-test repair is fixed at clean candidate
`87adbdafe507888df05a861f1f2c15035f73195f`, parent
`04571ee1c020d822be9056568e39b1fede1d9a65`. Only the source instrumentation
builder and its owned unittest module change. An explicit authenticated source
record is relayed to the inner suite; an explicit missing/invalid record fails
instead of becoming an optional-fixture skip. Verbose output identifies each
test. Stdlib regressions cover environment restoration after success, failure
and exception, missing/invalid/symlink records and authentication failures.
The normal `prepare` JSON is captured and asserted by its unittest without
changing production stdout or converting a failed result into success.

Root reviewed the final two-file diff and independently passed all 12 source
tests, zero skips, using the recovered authenticated source JSON; a missing
explicit record exited one. The source builder and test SHA-256 values are
respectively `cb77fc88d814297330464f146768b16db60dd2e010ae1a344d0b799bcd86a968`
and `5941901ffd1484b59e9327719febe0e89f03789a4789e65127424e43b1a86a48`.
No upstream model/package, Torch, weights, tokenizer, preset or Cargo was run
locally. This is source-protocol verification, not real-weight parity.
The new lifecycle controller remains under review: its positive controls must
match real output, including the actual unittest module/class namespace and
each of the ten independent exact-filter Rust summaries. No fixed success
summary may be emitted to replace observed results.

Fresh PR #152 CI at exact pushed head
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a` is now terminal: 76 successful
checks, three skips, zero failures and no pending checks, including the later
Unity packaging job. All 16 required checks pass. Its title and body were
updated to the actual head, converter/citation scope and terminal CI evidence.
It remains draft; XCodec2 package-license, hard-pin override and owner/legal
execution/publication gates are not closed by CI.

A fresh all-pages VAST inventory returns `success=true`, no pagination token
and only the unrelated stopped instance `53677077`. No Vokra worker is present
and no unrelated resource was changed. Fresh all-pages GitHub Dependabot
readback still has 209 open alerts: 182 with a patched version and 27 without
one. Package counts are Torch 156, ONNX 20, Transformers 20, Diffusers eight,
Accelerate two, ModelScope two and NLTK one. No alert was dismissed or claimed
fixed from the unmerged draft PR.

### 2026-10-02 strict replay launch and decoder-capture review (06:12 UTC)

The full objective remains reconciliation of all 194 public repositories,
with independent real-weight CPU and Apple CPU/Metal/no-fallback evidence for
every supported row, exact authorized dispositions for withheld rows, public
artifact reconciliation, security findings and disposable-resource cleanup.
This is not a claim that every Vokra feature is complete. The 58 unresolved
metadata rows are not reduced by model-free tests or by this replay launch.

Root accepted the revised source-only lifecycle controller at SHA-256
`b0a6831a9057f1d1bd1221af1da7a0acdd82500a90537c910ebabf1e359a8604`.
Its FireRed consumer predicate uses captured test output without `--nocapture`;
the Mimi predicate requires ten actual exact-filter summaries and identities;
the source-builder predicate requires the actual twelve-test namespace with
zero skips. No synthetic aggregate Rust success summary replaces observed
results. The main-packet member count is derived from its validated manifest,
not the obsolete fixed value 42. Root independently passed the complete
offline suite before the final three explanatory ShellCheck suppressions;
root syntax and plain ShellCheck then passed at the final hash. The
implementer also passed the complete final offline suite. The checksum
tamper-negative control emits its expected warning, not a production failure.

The clean, hash-checked target remains
`87adbdafe507888df05a861f1f2c15035f73195f`. An all-pages VAST readback confirmed
no Vokra worker before exactly one new instance, `53815435`, was created from
offer `49438897`. Its owned label binds that target and this launch. Instance
readback reports running, 16 effective CPU cores, 200 GB disk and total rate
USD 0.1995555556/hour. The controller retains its 128,000-MB RAM gate; cheaper
64-GB offers were rejected before creation. The execution handle is live;
remote checks, packet recovery, the evidence-only Moshi side probe and strict
destroy/readback are pending. No weights, HF token, model/preset execution,
Apple run or upload is part of this replay. Unrelated resources are untouched.

A separate, uncommitted decoder-capture candidate passed ten stdlib tests,
but root review found missing opt-in imports, incorrect frontend-derived
maximum-length metadata, module restoration and multi-output publication
issues, incomplete identity/shape/budget validation, and absent exact packet
binding. Corrections and integration regressions were delegated; that
candidate is not accepted and is not included in the frozen VAST target.
Neither its helper tests nor this source-only replay prove decoder parity.

### 2026-10-02 terminal replay, recovery-path defect and comparison map

This supersedes the live-worker snapshot above. Instance `53815435` is
terminal and destroyed, not a retained transfer source. The execution handle
returned exit one. The bounded SSH readback contains twenty named step results
with process exit zero, including workspace, deny and audit, and the final
clean target `87adbdafe507888df05a861f1f2c15035f73195f` with zero status bytes.
Its remote summary nevertheless has `overall=1`. The two direct source-audit
results and the failing strict predicate are not independently recoverable;
no workspace aggregate, exact test identities/counts or overall green verdict
is inferred from the process exit markers.

Main, source-only and diagnostic recovery all stopped at `EVIDENCE_MISSING`.
Root traced a separate concrete defect: three remote reader scripts retain
`/root/vokra-firered-integrated-verify-evidence`, while the new producer writes
`/root/vokra-firered-output-contract-verify-evidence`. A self-test substitution
of the old literal hid that production path mismatch. It explains the lost
packet, not the earlier remote `overall=1`; that verdict remains unlocalized.
Moshi source-cache execution was not reached. The frozen controller is not
edited in place. A new recovery candidate must bind every reader to the actual
producer path, test the rendered production scripts, emit bounded sanitized
predicate verdicts, and preserve compact failure evidence before destruction.
The strict packet allowlists and resource ceilings must not be relaxed.

The instance's destroy operation succeeded; exact readback is
`instances: null`, and the all-pages owned-label readback has `success=true`,
no pagination token and no matching instance. Strict cleanup passes. No model,
weights, HF token, preset, Apple execution or upload occurred. Evidence hashes:

- controller log: `83b6c0057e920d3223aff930cc2ef6242fe54f72a17fbdfbf3a7239ac7869c2d`;
- bounded remote summary: `27666d12b4d2447460c5f8750c0f4edfa80dbafe6d5c129434be7977801ec08c`;
- recovery guard: `d906c51985237188ecb6f2451262495a84187772234fa2d9821ad2cdbec2d74b`;
- exact destroy readback: `817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057`.

Read-only comparison mapping also confirms that the current v2 Rust reader
discards numeric event payloads and its real gate deliberately stops before
model execution. The next consumer must retain typed events, bind the exact
V1/PCM/checkpoint/GGUF and observer identities, and compare every required
native vector and beam/cache lineage. Source K/V are head-major rank four;
native newly projected rows are full `d_model` vectors. Complete authenticated
row selection is required, not two scalar samples. Source finished state is
observed after pruning, not separately for every unselected candidate; Torch
tie correspondence remains open. A revised capture candidate is still under
implementation and has not been accepted, committed or declared parity-green.

### 2026-10-02 decoder capture review rejection (06:38 UTC)

The separate, uncommitted source-capture candidate with runner SHA-256
`2bdf1f67837609bc62d22268b2cc0812b614483dbcf892cbc4ad6b40d822ffe3`
is not accepted. Its implementer-reported twelve stdlib tests exercise a
one-layer, one-beam, one-step fixture; that scope does not prove the fixed
sixteen-layer, three-beam upstream capture path. Root independently reproduced
the step-wide ordering rejection when a second layer input follows the first
layer output. The actual source performs this nested sequence for every layer.

Root re-read the authenticated decoder source, SHA-256
`f0dd5d0ba224ec0be9d2778d3d4ae514ef5ab24c879436aad756353b81f4eedb`.
At line 92 the observed pruned scores are already `[N,B]`, not `[N,B*B]`;
the full candidate scores precede pruning. At line 121 the length-penalty
tensor has the `[N,B]` length shape, not a scalar shape. The loop-exit anchor
is outside the loop and cannot be required once per step. These are source
contract defects, not tolerance failures. The candidate also needs explicit
role-specific cross-attention observation-only handling, complete index/final
state payload retention, and bounded allocation checks before tensor copies
and list materialization. Its broad full-array hashing fallback is not an
acceptable substitute for a lossless bounded production path.

Corrections and multi-layer, multi-step, three-beam model-free regressions were
delegated to the implementation owner. No source-capture commit, weight run,
CPU parity, Apple verdict or upload is authorized by this review. The recovery
controller remains a separate candidate targeting frozen
`87adbdafe507888df05a861f1f2c15035f73195f`; the rejected capture is not included.
The full public-catalog and security completion scope remains unchanged.

### 2026-10-02 packed capture candidate review (06:51 UTC)

The next uncommitted capture runner, SHA-256
`b366289ab5dc87cb454b93182208bea2cd780894c5d83f620ec24e57186c7533`,
fixes the earlier pruning and penalty shapes and adds packed/base64 selected
values. Root independently ran its stdlib unittest-discovery suite: thirteen
passed, zero failed. No Torch, weights, model arithmetic or Cargo was run.
This result is mechanism evidence, not numerical parity or acceptance.

The capacity assertion covers only 5,387,400 required values at the proposed
sixteen-layer, three-beam, twenty-five-step bound. The actual selection policy
still retains both layer inputs and duplicate pre-cache-store outputs,
adding 3,072,000 values. That makes 8,459,400 before masks, indices and final
state, already above the unchanged 8,388,608-value cap. The candidate therefore
does not prove its full-step capacity. Those duplicate vectors may become
explicit observation-only records; no required vector may be sampled or
dropped to pass the cap. A complete saved-payload ledger is required.

The tests also route through fake-array ASCII bytes rather than the production
byte-view/chunk path, and their synthetic shapes, dtypes and cache/registry
semantics do not yet authenticate the real source contract. Whole-step nested
layer/QKV completeness, source-cleanliness and stale-import rejection remain
review corrections. They were delegated together with a lossless packed-byte
round trip and full-bound mechanism test. The candidate remains unaccepted,
uncommitted and outside the frozen VAST target; no row is promoted.

### 2026-10-02 source-fidelity review and recovery-v2 replay (07:05 UTC)

Root independently ran eighteen stdlib unittest-discovery tests for the
uncommitted capture runner with SHA-256
`d698ace5f1702a829a815f234a9b70bcf71b0da52d968a1872438d4c97984a86`:
all passed without Torch, weights or Cargo. The candidate is nevertheless
not accepted. Authenticated upstream lines 51, 71–72, 104–109 and 208–209
show that decoder caches are optional rank-three layer outputs, not pairs of
rank-four projected K/V tensors. The proposed validator rejects the real
initial `None` cache and subsequent `[N*B,t,H]` output cache. Source lines
112–114 also permit full-beam EOS termination before the maximum length;
capacity must cover the full horizon, but validity must not require the
source to ignore its own early-exit condition.

The fake byte-view object lacks `reshape`, so the purported byte-view test
still enters its fallback. Its dtype packing and several beam/mask/index
shapes are not source-faithful. The full-horizon test calculates the minimum
vector formula rather than exercising the complete production selection and
auxiliary-payload ledger. Corrections were delegated with source-valid cache,
dtype, byte-view, early-EOS, class-origin and whole-call-order regressions.
The candidate remains outside the frozen remote target, with no parity or
catalog promotion.

The separate recovery-v2 controller was accepted after root review,
`bash -n`, ShellCheck and the complete offline mocked regression succeeded.
Its frozen SHA-256 is
`0f564546b96e3ccd01fdfd3ba5d91bea5e80aef0e5dcdb30256d1558ba083bfb`.
Fresh all-page VAST readback showed zero Vokra workers and one unrelated
worker, which was not modified. One replay was then started on disposable
instance `53822210`, offer `32178462`, for exact clean target
`87adbdafe507888df05a861f1f2c15035f73195f`. The reviewed offer reports sixteen
effective CPU cores, 258,023 MB RAM and USD 0.17407407407407408/hour with
200 GiB storage. Bootstrap completed; verification and cleanup are still
pending. This is source/code verification without upstream checkpoint
acquisition or real-weight execution, and without HF credentials; synthetic
workspace tests execute only on VAST. It is not an Apple or real-weight
result. The controller must recover
small evidence and destroy its owned instance before a cleanup claim.

PR #174 remains draft at `f4879d4ffd948144c4abc385255bdf5964c94f33`.
Documentation run `36658360528`, attempt one, reported zero link errors and
two timeouts for the same official U.S. Code URL. A bounded current request
returned HTTP 200 in 1.398321 seconds. Only the failed CI jobs were requested
for rerun after confirming the unchanged PR head; no link check was weakened
and no license disposition or merge was inferred.

### 2026-10-02 consumer-output diagnosis and reviewed capture (07:29 UTC)

The same recovery-v2 session on `53822210` reached the exact clean target
`87adbdafe507888df05a861f1f2c15035f73195f`. Workspace, deny and audit
reported process exit zero and their remote predicates passed. The original
overall result remains failure: the FireRed consumer process passed 24 tests
with zero failures and two expected real-weight tests ignored, but Rust's
authenticated ignore-reason suffix did not match the frozen controller's
plain `ignored` predicate. This is an output-contract defect, not numerical
parity evidence. The original logs and failed verdict are not rewritten.
Evidence recovery is in progress; destruction has not yet been proved.

A separate recovery-v3 controller, SHA-256
`ffa8e5d4846afa713d44c1932ac2c898efa835467a77641ab6033680dc29bbbb`,
passed root review, syntax, ShellCheck and the full offline mock. Its local
and rendered remote predicates require the exact named 24 successful tests,
the exact two ignored names with only their authenticated reason or legacy
plain form, exactly 26 test lines and the unchanged zero-failure summary.
Extra names, duplicate/missing tests and wrong reasons are rejected. It has
not been used to allocate another worker or retroactively promote v2.

The corrected raw source-capture foundation is committed in its isolated
clean clone at `5b919cca1bb3277456803798d165626e201239f3`. Root's independent
stdlib run passed 23/23 tests without Torch, checkpoints or local Cargo.
The mechanism tests include the full 16-layer, 25-step observational sink,
source-shaped optional caches, complete event ordering, early EOS, strict
dtype/geometry checks and manifest-last no-clobber output recovery. The
runner SHA-256 is
`f137de54c50c53cae89836696755f6bcc77efd4dc91df1a3bd892e05f30d690e`.
These are fake-tensor mechanism tests, not execution of the independent
upstream checkpoint. Raw-to-Rust-v2 normalization, PCM/checkpoint/GGUF
binding, native CPU parity and Apple CPU/Metal/no-fallback remain open.

The reviewed typed-retention correction was committed at
`3299c649d91ae39e23f5d124502d6ec09768220c`; combining the disjoint capture
commit produces clean candidate `c03e3010472521f0064a1d30e1a05e0d5af6389a`.
Its schema fixture uses two steps, three beams, two layers and four-wide
distinct complete vectors, and allocations are charged before retained
copies. The consumer source now contains 25 non-ignored tests and the same
two explicitly ignored real-weight legs. Formatting and diff hygiene pass;
the new Rust tests have not yet been compiled or executed on VAST. This
candidate is unpushed and is not covered by the running `87adbdaf` replay.

**Terminal recovery update:** the original recovery-v2 session subsequently
exited with status one, retaining the original failed output-contract
verdict. Both recovered packet checksum gates passed and collection returned
zero. Destroy returned zero; strict individual readback returned
`instances: null` for `53822210`, and the controller's post-destroy label
readback passed. The worker and its attached 200-GiB data were destroyed.
The integrated candidate above still requires a new exact-head remote run;
the earlier failure is not retroactively replaced by an offline mock.

PR #174's unchanged-head documentation rerun, attempt two of `36658360528`,
also failed. The previous U.S. Code timeouts disappeared, but 18 GitHub links
returned HTTP 503/504. A bounded subsequent probe still returned 503 for one
fixed Microsoft source link. No repeated rerun, URL exclusion, gate
relaxation, legal disposition or merge is inferred from this availability
failure; dependency-review and workflow-security passed in that attempt.

### 2026-10-02 source-mask and fixture supersession (07:48 UTC)

Further primary-source review found two defects before the next allocation.
The authenticated decoder constructs the target/self-attention masks as
`torch.uint8` at source lines 142–148, while candidate `c03e301` incorrectly
required bool. The encoder's source/cross-attention masks are genuinely bool:
the fixed conformer source returns bool at lines 45–51 and recreates its
subsampled mask by an integer comparison at lines 110–117. Its 12,651-byte
source has Git-blob SHA-1 `b41e68e6eb2bed90611e65c6bc5e2025dc6753df` and
SHA-256 `ea6412fdffc33b558a610d67143fd3211b76fc5a7a9ccac88ae2110f1fd3d320`.
No source or model was executed to obtain these facts. The capture fix at
`1706223ffa6cdb464d66044ee3827144d9a2eb7e` preserves the exact uint8/bool
distinction without a cast or broad acceptance. Root's independent stdlib
mechanism suite passed 24/24 tests in 48.179 seconds. The revised runner
SHA-256 is
`6b1dfeed33b326ccd0e38a30da582196206e890d51462cb2ad8615d5802a7ffd`.

The four-wide Rust fixture also had an eight-value encoder memory but asserted
only four values. The reviewed correction at
`905551c58d245a810bca92bcd416b5d4e4a5b133` now checks all eight distinct
components, exact shape and dtype. Its consumer SHA-256 is
`503a38165cf712e789bd796525a45bf78e092b8c90fa7aac484275e4dcf5158a`.
This remains statically reviewed, not remotely executed Rust evidence. The
previous `c03e301` candidate is superseded; neither defect is hidden by a
rerun or a relaxed completion gate.

The two reviewed corrections are integrated at clean candidate
`1c555807f11ad4e15f6e738e19c7a4c4d0bc5d9a`. Its Rust consumer retains
25 non-ignored tests and two deliberately ignored real-weight legs.
Formatting and diff hygiene pass. This head remains unpushed and requires
an exact-head VAST run; the preceding worker's evidence does not verify it.
The next independent work item is lossless raw-capture normalization and
explicit source/reference/input binding for a future native comparator.
That mechanism must distinguish source observations from derived integer
projections and must not claim parity while GGUF identity is unbound.

A separate read-only query to the [fixed FireRed HF API revision](https://huggingface.co/api/models/FireRedTeam/FireRedASR-AED-L/revision/e57f5960d03cff1071ff7acbb409314d1e70ed3d)
returned that exact revision and card license `apache-2.0`, with no license
file in the sibling list and no dataset field in card metadata. The observed
API-response SHA-256 is
`67cd36d96228b107407ecd5ff7320ac1d73a75fee32911cdcf8ed71e0d29203a`;
this binds the dated observation, not mutable API counters. The
[fixed README](https://huggingface.co/FireRedTeam/FireRedASR-AED-L/raw/e57f5960d03cff1071ff7acbb409314d1e70ed3d/README.md)
is 6,458 bytes with SHA-256
`a5a905edac140af027719a5ba6bab2f34f21ae5bfd279725d536519ddc00cf8a`.
Only metadata and documentation were read, not weights or tokenizer assets.
This records the upstream card declaration; it does not settle training
rights, the reviewed mixed Python closure, approval or redistribution. The
existing blocked execution gate and `NO_UPLOAD` remain unchanged.

Fresh VAST readback after the previous terminal run found zero Vokra-labelled
instances, one unrelated instance and zero persistent volumes. The unrelated
instance was untouched. GitHub main remains `97447185361a37af64c1b30fe87e8e2618d96e20`;
PR #174 is still draft at its unchanged head and its documentation check is
still failed. The previously failing fixed Microsoft source URL still returns
HTTP 503, so no availability-based rerun or merge was requested.

### 2026-10-02 binding-adapter and remote-controller review

Root rejected the first uncommitted capture-binding adapter proposal. Its
eight synthetic tests accepted a new identity/tensor layout rather than the
actual raw capture and `vokra.firered.decoder.binding-manifest.v1` producer
output. It did not verify the manifest's exact file bytes, and a monotonic
global event rank incorrectly rejected the next decoder layer's input after
the previous layer's output. The proposal is not accepted evidence or a
candidate for publication. Revision requires fixtures generated through the
existing fake-tensor capture producer, exact serialized reference/capture
hash binding, nested event order and pre-decode aggregate resource limits.
No real model or native numerical comparison was run by those eight tests.

Root also found a remote-only controller defect: the rendered wire guard
used `FIRERED_WIRE_REFERENCE_SHA256` without assigning it in the SSH script.
The initial offline mock passed but did not exercise that variable boundary.
The corrected v4 controller adds the remote assignment and a rendered
`set -u` wire/run-step regression. The exact earlier
`source-cache-process-self-test=FAIL` was diagnosed from its retained
`destroy-order.stderr`, which contains `target-source-hash-self-test=FAIL`:
the nested child used the old default checkout, not the new integrated target.
Its destroy-order file remained empty, while its orchestration sequence had
completed and its recorded timeout PIDs were gone. This is a target-binding
failure, not an unexplained timing failure or a relaxed cleanup gate. The v4
default checkout now names the clean integrated candidate.

The corrected controller SHA-256 is
`379bc3e0ba2e9eeb8f2e0579fde8138b21e146d194a9f69ea104f501fd6e5da6`.
Agent syntax, ShellCheck and full offline mocks pass; root syntax and
ShellCheck pass, and its independent complete mock also exited zero with
`bundle-observation-self-test=PASS`. V2/V3
controllers and the original failed remote verdict remain unchanged. No new
worker was rented during those mocks. The pre-rent all-pages VAST instance
read reported zero Vokra instances and one unrelated instance, untouched.

After root's independent mock passed, the single-rent v4 invocation started
at 08:02 UTC with reviewed offer `49438897`, quoted at USD
0.19955555555555554/hour including its 200-GiB disk. Creation returned
`success: true`, instance `53829210`. The owned-label readback confirms that
exact ID running with 16 effective CPU cores and 200-GiB allocated disk;
reported host CPU RAM is 515,830 MB. The code-only verification target remains
clean `1c555807f11ad4e15f6e738e19c7a4c4d0bc5d9a`, and the unaccepted adapter
is not included. The root controller is live while remote bootstrap proceeds;
no remote green result, model parity, publication or destruction is claimed.
Its cleanup trap must recover bounded evidence, destroy this exact owned
instance including its data, and verify absence.

**08:08 UTC remote failure diagnosis:** the owned worker's exact source-capture
suite exited one: 24 tests ran in 70.169 seconds, with seven errors and the
other 17 tests passing. All seven errors are `FileNotFoundError` in
`tempfile.TemporaryDirectory(dir="/private/tmp")`; that Mac-specific parent
does not exist on this Linux worker. They affect the bundle/rollback/alias,
import-scope restoration and source-record-cleanliness mechanism tests.
This is a test-portability defect, not numerical model evidence. Root read
only the bounded failed log over SSH and delegated a portable canonical-temp
parent correction in a separate clone. The frozen verification checkout and
failed verdict are unchanged; no remote compatibility directory, skip or
weaker production guard was introduced.

The same live run's FireRed Rust consumer and consumer Clippy steps returned
zero and their exact-output gates passed. The complete run, evidence recovery
and instance destruction are still pending. The capture-binding adapter is
still unaccepted: its newly added producer-generated, full-geometry fake
fixture exposes the remaining real producer-format mismatch. Such a fixture
is mechanism coverage only, not execution of the upstream checkpoint.

### 2026-10-02 portability correction and terminal VAST readback

The frozen `1c555807f11ad4e15f6e738e19c7a4c4d0bc5d9a` v4 execution
terminated with exit one, preserving the source-capture test failure above.
Workspace verification exited zero: 325 successful result summaries,
8,220 passed, zero failed and 110 ignored. Its recovered log SHA-256 is
`e5151d9da070aec5dd14119e4972815285c1ba3d21f0b51d9445ede795a6c44f`.
FireRed consumer verification reports 25 passed, zero failed and two ignored;
the ignored real-weight legs remain open. Consumer log SHA-256 is
`0d4d583645e75b0bc34f7cade2c256bd3c7df560641bf2c66b208480cbb57258`.
Deny and audit exited zero, with respective log SHA-256 values
`f35734dc54f6283b1df3d005396eeec011459fa5fa1fc9259dbf3e49b66065a9`
and `2de900cde21af07fc0cff26eb7a1fb7a16600c40da5faf87fefa9dd4d47f10dc`.
The failed capture log remains bound to
`ba7c6408243cc77c0f18eb5a209a5e6d8a6d4725fe8fe0a6c5c5149bf35dbd0a`.

The controller recovered bounded evidence, verified both local packet and
source-only packet checksums, and reported collection exit zero. It destroyed
owned instance `53829210` and its 200-GiB disk: destroy and individual
readback exited zero, the individual response is `instances: null`, and the
all-pages post-destroy response has no Vokra instance. It lists one unrelated
instance label, which was untouched. The sequential source-cache probe did
not run after the failed main gate. No real model, upload or Apple verdict is
inferred from this code-only execution.

Root reviewed and accepted the separate test-only portability correction at
clean commit `218ce74d5e8476a420efae6576c25a9190029299`. Seven fixtures now
use resolved `tempfile.gettempdir()` rather than a Mac-specific directory;
the existing alias test also checks that canonical parent. Production
no-follow guards and the 24 test names are unchanged. Root's independent
Python 3.12 UV stdlib replay passed 24/24 in 45.920 seconds, and agent
documentation gates passed. The test-file and design-file SHA-256 values are
respectively
`471d9d4663837bc1a89e8f88809e061bb6a111b805cc9a33fe5c1ff18ebf9fec`
and `32ff0b478390977a58684db0451cb2c6598d47961738ed5ce464951c924976d9`.
There is no import patch-target change; that agent-report claim was corrected
against the actual diff. Exact-head Linux replay is still required.

The uncommitted three-file capture-binding adapter now accepts a fixture made
through the real capture/bundle producer, but root has not accepted it. Its
semantic tamper tests currently stop at stale serialized-byte mismatches,
without exercising the claimed event/shape rejection paths; the fixture also
depends on an external Mac-specific source-record path. Root requested
portable producer fixtures, re-sealed semantic tamper tests, authenticated
contract/V1 links, distinct selected/full tensor hashes, producer-equivalent
pre-decode aggregate budgets and explicit per-beam projections. A self-test
that only prints success is not verification. These are mechanism and
binding requirements, not independent real-weight numerical evidence.

Fresh PR #174 readback retains unchanged head
`f4879d4ffd948144c4abc385255bdf5964c94f33`, draft/behind state, no pending
checks and the failed `documentation-links` check. No rerun, merge or gate
relaxation was performed.

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
