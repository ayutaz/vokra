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

**08:28 UTC corrected-head replay:** the reviewed v5 controller changes only
the target/default clone, the two portability-file hashes and isolated v5
namespaces. V4 remains unchanged. The v5 SHA-256 is
`b42a8ef2b88abcaa2d8d5e941b98405c92969cdc79184e2be896f1a94bc087e7`.
Agent and root complete offline mocks exited zero; root also checked syntax
and ShellCheck. Before the new single rent, a fresh all-pages read found zero
Vokra instances, one unrelated instance and zero persistent volumes. The
200-GiB-adjusted offer `32178462` quoted USD 0.17407407407407408/hour.
The controller created owned instance `53832348` for clean target
`218ce74d5e8476a420efae6576c25a9190029299`; its readback confirms the exact
owned label, running state, 16 effective CPU cores and 200-GiB disk. This
code-only replay is live; its result, source-cache leg, evidence recovery and
destruction remain unproven. The unaccepted adapter is not in this target.

**08:35 UTC live progress:** corrected-head `firered-source-capture` returned
process exit zero and its unchanged 24-test output gate passed on Linux.
FireRed consumer/Clippy, source-record, source-builder and module-registry
steps also passed. Whole-run success, the later source-cache leg and cleanup
are still pending; the old failed execution is not rewritten.

The source-only integration audit also closed mask polarity, not numerical
parity: authenticated decoder lines 142-148 build `(ys != pad_id)` intersected
with the causal triangle, and lines 258-264 mask positions equal to zero.
Thus source target `uint8` one and source encoder `bool` true mean allowed.
Cached decoder lines 185-196 select the last query row; source lines 48-50
repeat the encoder mask across beams. Native self-attention uses prior valid
keys plus the current `previous != pad_id` key, with a current-query causal
boundary. Exact source/native row correspondence still needs authenticated
beam-parent and prefix mapping. The top-level encoder/cross-attention source
mask must not be compared with native self-attention `TraceStep.key_masks`.
Source hidden-output prefix caches are not projected K/V caches. Source exit
step is zero-based, while native `AllFinished.step` records the completed
step count. No real decoder, cache-value, tie-order or Apple PASS follows from
these source facts.

Root's independent producer-backed adapter suite passed six tests, but the
adapter remains unaccepted. Further review found aggregate selected-value and
Base64-retained-byte checks occurring after decoding, missing nested tensor
digest coverage, noncanonical Base64 acceptance and filesystem publication
rollback gaps. These must close with targeted regressions before that code is
committed or included in a remote target. A separate bounded Rust row/head
projection implementation was delegated in a new clone; the running portable
verification checkout remains frozen.

Fresh PR #174 readback retains unchanged head
`f4879d4ffd948144c4abc385255bdf5964c94f33`, draft/behind state, no pending
checks and the failed `documentation-links` check. No rerun, merge or gate
relaxation was performed.

### 2026-10-02 live catalog/security audit and projection review (08:45 UTC)

Root repeated the read-only metadata/README inventory at management checkout
`36d8d8377fc2b2494285e4c7860070562e46e0dc`. It still reports 194 public
repositories, 193 with GGUF and 198 GGUF files: CPU code/artifact status is
136 full, 43 partial, 14 no-runtime-binder and one non-artifact; Metal is
136 full, 57 blocked-by-cpu and one non-artifact. No model payload was
downloaded or executed. Audit-script and engine SHA-256 values are respectively
`690d603f7182fc5c0645e784f8857acee9604b20050892b2bf0eadeb41f8a905`
and `cc22a9074531fa738c68101f4697b1b4393252b913b1bb2bc61092da93cf0b1b`.
The 58 unresolved rows and all independent real-weight/Apple gates remain.

Fresh paginated GitHub APIs report 209 open Dependabot records: 182 name a
patched version and 27 do not. Three open code-scanning records remain:
`CIIBestPracticesID`, `CodeReviewID` and `VulnerabilitiesID`. GitHub's main
commit API still reports `97447185361a37af64c1b30fe87e8e2618d96e20`.
PR #174 remains draft/behind at unchanged head `f4879d4f`, with no pending
checks. Inspection of its failed documentation job finds 18 link errors with
HTTP 503/504 responses, spanning Vokra, Microsoft, Demucs and other GitHub
pages, rather than evidence of eighteen removed source files. No link-gate
exclusion, blind rerun or merge was performed.

Root reviewed the separate Rust projection draft. Legacy V2 parsing and the
ignored real decoder gate remain unchanged; the new helpers are not yet an
authenticated producer-format consumer. Review requires explicit batch/beam
row correspondence, checked full-source geometry distinct from retained-value
caps, pre-allocation budget regressions, and actual full-rank logits selection.
Digest syntax preservation alone is not byte authentication. These corrections
and the Python adapter safety corrections are delegated in isolated clones;
neither draft is accepted, committed or in the live VAST target. Session
readback confirms the same v5 controller remains live; workspace verification,
source-cache evidence recovery and owned-worker destruction are still pending.

### 2026-10-02 portable VAST replay terminal readback (08:59 UTC)

The v5 controller session is terminal with exit 1. Its remote verification at
clean implementation HEAD `218ce74d5e8476a420efae6576c25a9190029299`
completed successfully: all recorded remote gates passed, workspace tests
reported 325 result summaries / 8,220 passed / zero failed / 110 ignored,
and the FireRed consumer reported 25 passed / zero failed / two deliberately
ignored real-weight tests. The portable capture suite passed 24/24 on Linux.
Models all-feature/all-target Clippy, deny and audit exited zero. Deny retained
the existing unused `libfuzzer-sys` exception warning; audit loaded 1,279
advisories and scanned the 22 first-party lock entries.

Root independently verified all 55 recovered manifest members and the exact
final HEAD, empty observed worktree and empty diff-check output. Log SHA-256
values are: workspace
`c344f2ecfbd4852e1df3718ad30a4fce25eade814dc28a4d0d08350fc5d411e6`,
consumer
`e36a53d5fcc0a17f99ad20507cf67d1fbd6f3dde6a6500b4fbf030b80a8ea4f6`,
capture
`f79dcd13658ea7602f875b1670273021673b52e68f7316524517ecfb779dc681`,
deny
`907da4b7650fa729da1d4032c1b772616791810ee1e791eb7817b7493f7a172f`
and audit
`2de900cde21af07fc0cff26eb7a1fb7a16600c40da5faf87fefa9dd4d47f10dc`.
The recovered manifest SHA-256 is
`a07f08339cb63ad3c4f6543fc77a3e4878521446acbfb7ba7f939ea10fcb855a`.

The outer controller nevertheless rejected its main-packet/local-green or
source-cache boundary. This is not an end-to-end successful lifecycle verdict:
the first rejecting condition is under review, the Moshi source-cache leg is
not proven executed, and no retry or new rent is inferred. Preserve the
terminal failure rather than relabelling the whole execution green.

Owned instance `53832348` and its 200-GiB instance storage were destroyed.
Controller readback and a fresh root individual API read both return
`instances: null`; a fresh all-pages read reports zero Vokra instances,
one unrelated instance and no next page, and the persistent-volume API
reports zero volumes. The unrelated instance was not modified.

The safe PR-preparation clone contains management-only commit
`d09cf7ddfd7d61e2ba0a4775ac4208bf72dab6a8` above the remotely tested
implementation HEAD. Root verified its implementation paths unchanged.
That clone has no configured pre-commit hook; root separately ran equivalent
format, forbidden-symbol, zero-dependency, fixture-EOL (190 cases) and
pipefail (306 cases) gate legs, plus documentation and diff checks. Do not
claim an automatically executed hook or a remote run at the documentation
HEAD. No push or PR submission has occurred at this checkpoint.

Both new Python capture-binding and Rust row-projection drafts remain
unaccepted in separate clones. Further review requires inode-owned,
fd-relative publication/rollback, producer-equivalent working-memory caps,
correct parent-finished/EOS lineage, truthful full-matrix row provenance and
complete allocation accounting. Model-free mechanism tests are not
independent real-weight parity. The 194-row denominator, 58 unresolved
metadata/code rows, security findings, owner/legal gates, Apple CPU/Metal and
separate publication gates remain open. No model ran on the maintainer Mac.

### 2026-10-02 reviewed binding and projection integration (09:09 UTC)

Subsequent source review accepted the fail-closed Python capture-binding
preparation at `10f88b8253c6d6a94b0dcf099c7cd3b7620b1192`. Root independently
reran 17 stdlib regressions with ResourceWarning errors enabled and the
authenticated decoder source record; all passed, as did its self-test.
The reviewed Rust projection slice is
`9e0eb0b163d657b3f6ca74967ce8359d3e13fdde`; its focused Rust tests and Clippy
remain VAST-pending. Management integration plan
`ea82c04aa1e7b6682d236b748b30bd9a00017580` records the actual missing
producer-to-Rust/native comparison seam rather than claiming schema parity.

The isolated clean integration candidate is
`8af2cc7bd4d1083602047e721f61bb39614da90a`, above safe PR-preparation
checkpoint `197b8c5b23612e8c3e64ea8a0930e52c4ee93d37`. Root confirmed only the
six intended files changed and reviewed-file hashes remained identical.
This candidate has not been remotely compiled, pushed or submitted as a PR.
The earlier upstream capture producer's filesystem-publication race boundary
is now assigned as a separate bounded correction before the next remote
target is frozen. No new VAST instance was allocated. The Python binding
still reports `INCOMPLETE_PARITY_BINDING`, `parity_ready=false` and unbound
GGUF identity; real-weight, owner/legal, native comparison and Apple gates
remain open, and no catalog row is promoted.

### 2026-10-02 v5 local-green failure diagnosis (09:18 UTC)

The earlier terminal failure is now localized: the first rejecting check is
`firered_module_registry_green_result_ok`, called by `verify_local_green`.
The recovered module-registry log reports all nine named tests passing and
`OK`, but contains none of the three required SHA-256 file-identity lines.
Its SHA-256 is
`5964384a0a32dd0df357204c70fc0dbfe494ec9d1e63f22f8a1ee302c5ff361b`.
The frozen v5 controller remains unchanged at SHA-256
`b42a8ef2b88abcaa2d8d5e941b98405c92969cdc79184e2be896f1a94bc087e7`.

Root confirmed that the remote verification heredoc does not forward the
three module-registry identity variables, and its `run_step` does not measure
or prepend those hashes or apply the strict module-registry predicate.
The generic remote process-success marker therefore does not prove that
the stricter identity-bearing gate passed. This supplements, rather than
replaces, the 08:59 remote test results and terminal exit-1 record. No hashes
were fabricated or inserted into the original evidence, and the local gate
was not weakened.

The sequencing function returns immediately when this local-green check
fails, before invoking `run_source_cache_side`. The Moshi source-cache leg
is consequently `NOT_RUN`, not merely unproven and not a numerical result.
A separate v6 controller correction is delegated for offline review before
any new disposable-worker allocation. Its target must be frozen only after
the producer-safety correction and actual Rust capture-binding reader are
reviewed; the clean `8af2cc7` preparation is not remotely verified.

Root also reviewed the first producer-publication safety draft and returned
specific post-link stat-error and rollback/descriptor-cleanup paths for
correction. Its nine focused filesystem test passes are not acceptance or
real-weight evidence. The actual Python `capture-binding.v1` to Rust reader
is being implemented separately; unbound GGUF identity and
`parity_ready=false` must still reject a real-weight run before model access.
No catalog row, Apple verdict, publication approval or completed-model count
is changed by this diagnosis or by these model-free drafts.

### 2026-10-02 capture publication review and commit (09:25 UTC)

After two correction rounds, root accepted the bounded producer publication
slice at `f72e0968cdcdfbb45609e5f19b49a7e3cbdab5f0`, based on frozen
integration preparation `8af2cc7bd4d1083602047e721f61bb39614da90a`.
Only the source-capture producer, its tests and its design record changed.
The isolated publication-safety clone is clean. It has no configured
pre-commit hook; root separately ran diff, forbidden-symbol and zero-dependency
checks and reviewed all three file changes.

Root independently reran the 13 focused stdlib filesystem tests with
ResourceWarning errors enabled: 13 passed in 0.037 seconds. No model or
upstream inference executed. Direct test-script invocation was refused by
the local model guard; the normal stdlib unittest runner successfully ran
only the inspected filesystem cases, without disabling any guard.
The reviewed producer SHA-256 is
`82fa23a9a5c0c42a5600a00443eb6dc8f37b6150f533197aaad0fca594dd3ce5`,
test SHA-256
`a25779e75671aa5bf25ed370cef9a99a45ff6b9b154677fb192cde5a613b1016`
and design SHA-256
`da1b4d3e09baa04931ab6ffd7ce6bc79778bca7fdf488070d25b8e6067ed221c`.
The complete 33-method suite and integrated Rust compilation remain
VAST-pending. This is filesystem mechanism evidence, not real-weight parity.

The actual Rust capture-binding reader draft remains unaccepted: root
requested producer-backed positive and resealed-negative tests, typed tensor
retention, byte/digest and derived-projection checks, and correction of
first-party JSON API assumptions. A separate read-only native observation-gap
review is underway to prepare the real comparison seam. No synthetic parser
or filesystem test advances an Apple or model-completion verdict.

Fresh root API reads confirm GitHub main is still `97447185361a37af64c1b30fe87e8e2618d96e20`
and PR #174 remains open, draft and behind at
`f4879d4ffd948144c4abc385255bdf5964c94f33`. VAST reports zero Vokra instances,
one unrelated instance, no next page and zero persistent volumes. No new
instance was rented, no unrelated resource was modified, and no model was
uploaded or catalog row promoted.

### 2026-10-02 explicit completion goal and draft review (09:38 UTC)

The owner asked to make the goal explicit. The full completion objective is
unchanged: account for every one of the 194 public repositories, finish each
executable row's authenticated native/CLI route and independent real-weight
CPU verification, then obtain Apple CPU/reference, Metal/reference and
Metal/CPU no-fallback evidence. A row may instead have an exact owner-approved
withholding or withdrawal disposition, but it must not silently disappear
from the denominator. Metadata/code-full, synthetic mechanism tests and
successful compilation are not hardware-completion verdicts. Current
license, provenance, security/CI, separately authorized publication and
disposable-worker cleanup remain part of the final audit below.

The immediate milestone is reviewed FireRed capture-binding consumption and
native final-ranking observation, followed by compilation and mechanism
tests at a frozen clean VAST head. That milestone is preparation, not a
replacement for authenticated real-weight comparison or the full catalog
goal. Scaleway remains the final hardware-validation stage, after the
non-Apple requirements in the execution plan have been met or explicitly
dispositioned. No model download or execution is allowed on the maintainer
Mac.

Root reviewed the new uncommitted Rust reader draft above
`8af2cc7bd4d1083602047e721f61bb39614da90a`. Real and schema-only reads now
share the capture/manifest validator, while the real path retains the
complete V1 reference parser. However, root found that step filtering calls
the required integer accessor on final events without a `step`, which would
reject the positive producer fixture. Authenticated selected bytes are still
discarded after digest checking, derived mask/lineage projections need
semantic validation, and the actual four generated JSON fixture files and
resealed-negative reader coverage are not yet available. The stdlib fixture
generator was refused by the local model guard; that guard was not disabled.
These items were returned to the implementer and the reader is not accepted.

Root also reviewed the uncommitted final-ranking observation draft above
`f72e0968cdcdfbb45609e5f19b49a7e3cbdab5f0`. Score normalization and sorting
are now shared, but the draft double-counts final rows and their event
reservations, introduces a second result-vector allocation in the no-trace
path, and labels an immediate all-finished exit as a repeated-EOS test.
Specific corrections and actual continuation/budget regressions were
requested. Formatting, diff hygiene, forbidden-symbol and zero-dependency
checks passed for the inspected draft; this is not source acceptance or a
Rust compilation/test result. All Rust compilation and tests remain
VAST-pending.

The separate v6 controller passes root's shell syntax and ShellCheck checks
but is not executed against a paid worker or treated as remote verification.
Its provisional old target is not a final integration head. The frozen v5
failure and its original recovered evidence remain unchanged.

Fresh API readback at 09:38 UTC reports main
`97447185361a37af64c1b30fe87e8e2618d96e20` and PR #174 open, draft and behind
at `f4879d4ffd948144c4abc385255bdf5964c94f33`. VAST's instance list reports
zero Vokra instances and one unrelated instance; persistent-volume count is
zero. No new allocation, upload, PR change, merge or catalog promotion was
performed. The full goal remains active and unproven.

### 2026-10-02 reviewed final-ranking preparation (09:42 UTC)

The earlier unaccepted final-ranking draft is superseded by the reviewed
source-mechanism preparation at
`d328a5a401dc07f6cd4890bb0e8fa6fc51aa6b58`, parent
`f72e0968cdcdfbb45609e5f19b49a7e3cbdab5f0`. The isolated native-trace clone
is clean and frozen. Only `native.rs`, `mod.rs` and its design record changed.
The source hashes are respectively
`0005408e008abd8179fef10fec47aca04721878e491565b67389de8f33c6a5b9`,
`390fb3326a66e057be6e89117ac877a4c12b7e4b0d07a7d20132d5ccfbc2a262`
and `26d240db7c1f2a37b5b3179d0cba3f6f9bb6ab707f7f2cbc4e0a80ad066fcde7`.

After further correction and root review, final-row reservations are consumed
once; the ordinary path keeps its original single hypothesis-vector
allocation; and traced/untraced ranking uses one shared generic arithmetic,
sort and truncation implementation. Synthetic regressions now describe
actual mixed finished/unfinished continuation, repeated EOS, max-length
termination, selected-hypothesis correspondence, exact two-/three-row event
reservation consumption and charged metadata budgets. These regressions are
not claimed to have run: Rust compile, focused tests and Clippy remain
VAST-pending. Root independently confirmed formatting, diff hygiene,
forbidden-symbol, zero-dependency, 190-file EOL and 306-file pipefail checks.
The source preparation was committed but not pushed or promoted to parity.

The producer-backed reader still requires correction. Root compared its
projection logic with the actual existing Python adapter and found three
positive-fixture mismatches: target masks require the final query row of
`[NB,width,width]`, source masks use the last dimension of `[NB,1,Ti]`, and
original candidate source rows are the three beam groups rather than nine
flattened candidate slots. These were returned for correction, together with
decoded-byte pre-allocation charging and no-clobber generator checks. The
generated JSON fixtures are still pending; no missing fixture is skipped or
relabelled as a passing real reference. The clean native preparation is not
yet the final integrated verification target. Real CPU, Apple, owner/legal,
publication and full-catalog completion remain open.

### 2026-10-02 stdlib fixture and offline controller checks (09:52 UTC)

Root independently ran the fixture builder's four normal stdlib unittest
methods with Python 3.12 through UV, the authenticated source record and
ResourceWarning errors enabled. The first run exposed the macOS `/var`
temporary-root alias; the test setup was corrected without relaxing the
production symlink checks. The subsequent run passed all four tests in
1.041 seconds, including deterministic generation into owned temporary
directories. No Torch, checkpoint or model executed and no guard was
disabled. The inspected generator SHA-256 is
`2c5018efa36f4848e6f11f682f958396b7eedcdc93456c789b408ebf8a846d27`.
The checked-in four-file synthetic fixture and complete Rust reader
regressions are still pending, and real V1 acceptance remains unproved.

Root also reviewed the scoped v5-to-v6 controller diff and independently ran
shell syntax, ShellCheck and the complete offline self-test. The self-test
terminated with exit 0 and `bundle-observation-self-test=PASS`; its deliberate
checksum-tamper warning is a negative-test result, not recovered production
evidence. The reviewed v6 SHA-256 is
`a2dc4e161f3b18bcdf2d733316fc4d0a884576f111c28b4d83926d0d471a931f`.
The repaired embedded registry gate measures real frozen source files and
rejects wrong, missing and duplicate identities; the actual worker-side
registry tests still require the final remote run. This template retains
provisional target `218ce74d5e8476a420efae6576c25a9190029299`, not the
new final-ranking or reader integration. It must be retargeted and reviewed
before paid execution. No VAST rental, remote source-cache probe, Rust
compilation, Apple verification or publication is proved by this offline run.

### 2026-10-02 fresh catalog inventory and rejected bootstrap review (10:08 UTC)

A public-metadata-only audit completed at `2026-10-02T10:06:56Z` against
clean preparation head `197b8c5b23612e8c3e64ea8a0930e52c4ee93d37`, using
Python 3.12.7 through UV and four workers. The first sandboxed requests failed
DNS resolution; the subsequently authorized public-network requests both
terminated successfully. No token, checkpoint, tokenizer or model was used.
The inventory remains 194 public repositories, 193 GGUF-bearing repositories
and 198 GGUF files. CPU code/artifact classifications are 136 full, 43 partial,
14 no-runtime-binder and one non-artifact; Metal code classifications are
136 full, 57 blocked-by-CPU and one non-artifact. All 58 unresolved names,
current public revisions and reasons were recovered. These classifications
are not independent real-weight or Apple hardware verdicts.

Root inspected the unresolved-row report and independently verified the
recovered report hashes. The audit-tool and CLI-engine SHA-256 values are
`690d603f7182fc5c0645e784f8857acee9604b20050892b2bf0eadeb41f8a905`
and `cc22a9074531fa738c68101f4697b1b4393252b913b1bb2bc61092da93cf0b1b`.
The summary, complete TSV and 58-row TSV SHA-256 values are respectively
`42efbeee107584ba481691e00b80d1a2853fc5a0ed4743cb2fa144f79ac6ae97`,
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`
and `51f989a34453b0c0ad7c72bbe6f5cfb45aebb27a1c55e50c83826dfb415fca14`.
The reports remain local bounded evidence, not committed model artifacts.

Root reviewed the complete model-free fixture bootstrap draft at SHA-256
`53487554d7e99579a7c679fe4a46dfb2beccf142f72e7b2c6be6029cf25bc0cc`.
Shell syntax, ShellCheck and its resource/packet self-test pass, but the
actual remote path is rejected: it constructs an alternate empty-event
packet instead of invoking the frozen producer-backed generator, uses the
wrong output filenames, assumes unprovisioned offline UV/Python, and lacks
the work-timeout definition. Pre-create disk-inclusive pricing, uncertain
create recovery, individual deletion verification and unique packet-member
checks also need correction. No worker was rented. The existing generator
remains frozen at `2c5018efa36f4848e6f11f682f958396b7eedcdc93456c789b408ebf8a846d27`;
`82fa23a9a5c0c42a5600a00443eb6dc8f37b6150f533197aaad0fca594dd3ce5`
identifies the separate producer, not that generator. The corrected bootstrap
must invoke the actual generator and recover its four JSON files plus README.

The Rust reader draft now routes resealed four-file mutation tests through
the same reader used by its fixture-positive test, but it is not accepted.
Root found moved-value and mutable-reference assignment errors, negative
cases that can fail an earlier unrelated header check, and missing exact
role-specific geometry and complete derived-lineage checks. Corrections
were returned to the implementer, including matching the intended rejection
reason instead of accepting any panic. Formatting is not evidence that the
Rust code compiles. Fixture JSON generation, Rust compilation/tests, final
integration and the exact-head VAST replay remain pending.

Fresh paginated VAST API readback reports zero Vokra instances, one unrelated
instance and zero persistent volumes. The unrelated instance was not changed.
GitHub still reports main `97447185361a37af64c1b30fe87e8e2618d96e20`
and PR #174 open, draft and behind at
`f4879d4ffd948144c4abc385255bdf5964c94f33`. No allocation, upload, push,
PR update or merge occurred. A separate owner-independent planning audit
of the complete 58-row queue is in progress; FireRed preparation does not
replace the full 194-row completion objective.

### 2026-10-02 source-role and immediate-EOS review (10:24 UTC)

Root reviewed the reader draft at SHA-256
`e635ec422400eb5ad435387b2b756c24abee8dbf9150130e4aeb14e25df892b1`
against the actual frozen producer and instrumentation contract, rather than
assuming that a tensor index is a payload slot. Explicit payload targeting
replaces that ambiguous mutation helper in the draft. Further corrections
are still required: Q observations and cross-attention K/V are metadata-only,
while only self-attention K/V retain last-prefix numeric bytes. The draft's
role checks incorrectly require retained values for all Q/K/V. Negative tests
also need to match the first actual rejecting validator and independently
prove an aligned over-budget input. This draft is not accepted or compiled;
the missing checked fixture and remote Rust verification remain open.

A separate UV/Python 3.12 stdlib-only probe reproduced an immediate-EOS
contract mismatch without Torch, weights or model execution. The existing
fake capture producer emits a valid `torch.int64` metadata tensor of shape
`[0]`, zero elements and bytes, and the SHA-256 of empty bytes for final
content. The Python binder rejects that tensor as an invalid shape. The
authenticated source's final-content slice permits this empty output. A
separate implementer owns a narrow semantic-role-bound repair; this is not
permission to allow zero dimensions globally or claim real parity.

The narrowed Nano audit at clean preparation head
`197b8c5b23612e8c3e64ea8a0930e52c4ee93d37` found no justified additional
owner-independent source implementation. Root independently checked the
inspector, dependency audit, license gate, native binder and real-test hashes
reported by that audit, and inspected the official meta API route, strict
Nano binder and measurement-only real test. The audit record SHA-256 is
`81f04d67b6958157c28fa0cb8beff0b7a64097db6f35cf303196c03b6ade561f`.
Existing source preparation does not prove a successful actual API replay,
an owner/legal decision, real CPU parity, reviewed numerical bounds, Apple
CPU/Metal/no-fallback or publication. Those gates remain open. The old
generic planning matrix is not accepted as an execution-ready per-row audit.

The model-free bootstrap remains under review and no new paid allocation,
Rust compilation, model run, upload, push or PR mutation was performed during
this readback. All 194 public rows and the 58 unresolved-row denominator
remain in scope; no row was promoted by these source and fake-only findings.

### 2026-10-02 immediate-EOS correction and bootstrap review (10:49 UTC)

The previous goal turn clarified the objective but did not change implementation
or complete a verification gate. This continuation made a concrete source
correction authoritative: the isolated clean commit
`087ed12485987f528062c9c0caea0b2acbddbc83`, based on native trace head
`d328a5a401dc07f6cd4890bb0e8fa6fc51aa6b58`, contains the two reviewed Python
binding/test files and a dated design record. Root independently passed all
21 focused binding tests in 1.515 seconds through offline UV Python 3.12,
with ResourceWarning treated as an error. These are standard-library fake
producer tests, not real immediate-EOS numerical parity.

Only the direct final-hypothesis `new_hyp.yseq` role permits a metadata-only
`torch.int64` tensor of shape `[0]`, zero elements and bytes, and the canonical
SHA-256 of empty bytes. Generic zero extents, nested empty tensors, scalar
and mapping replacements, wrong dtypes and non-metadata payloads remain
rejected. Selected-digest paths and numerical bounds are unchanged. The
reviewed binder and test SHA-256 values are respectively
`9ffde9756cb96d1e93913d3a964bb8a6172a96e1bd8ce66ee5232a4ee0bee457`
and `929b16c3ecdbb31006bf044b892880b3f67d18aab4c937b85905c8632e7290fd`.
The commit is clean; the original frozen native-trace checkout is preserved.
Formatting, forbidden-symbol, zero-dependency, fixture-EOL, pipefail lint,
documentation-reference and runbook-path checks passed for this slice.

Root also independently passed the four actual canonical generator unittest
cases in 1.112 seconds with the authenticated v5 source record. No persistent
fixture build or model execution was performed by that verification. The
bootstrap input remains clean
`e8e585b4fb2a8dea0c593488d381ef6c94052cc7`; its generator SHA-256 is
`2c5018efa36f4848e6f11f682f958396b7eedcdc93456c789b408ebf8a846d27`.
The authenticated source record SHA-256 is
`11c96d58909bb2863e2e92b2f1420b4545d7e7b5a08021e8b90954210293475b`.

The Rust reader draft now mirrors the semantic-role restriction. Root found
and returned a further whole-reader test setup error: the canonical fake
producer's original `yseq` is an encoded sequence, not a tensor carrying
`raw_bytes_hashed`. The correction replaces that actual sequence slot without
inventing source-byte accounting. A raw-byte rejection diagnostic was also
made contextual so a negative test proves the intended validator, not any
panic. Latest reviewed reader SHA-256 is
`3f79aacc702c57b16bbb013ef973c5cb5e1dcf1db6bf6e017441928584806602`.
Formatting and diff hygiene pass; Rust compilation/tests, the canonical
checked fixture and real-weight comparison remain unproved.

The bootstrap controller is not yet approved for paid execution. Review
requires a portable bounded runner on this Mac, whole create/work/destroy
offline lifecycle cases, signal-safe cleanup, and bounded evidence recovery.
Mock shell outputs are control-flow tests, not actual producer output. The
official UV installer was retrieved for hash review only and was not executed
locally: version 0.12.22, SHA-256
`58488ae8dbd0773134c92c85e901430e33f99d975bd7f929d26aa9ab0c2f9390`.
Only that exact installer is approved for the remote bootstrap.

Fresh paginated VAST readback using the approved existing environment reports
zero Vokra instances, one unrelated instance and a null next token. The
initial sandbox DNS failure and stale saved-key 401 were diagnostic failures,
not proof of absence; the subsequent authenticated API succeeded. No new
allocation or unrelated-resource mutation occurred. PR #152 remains draft at
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, with 76 successful and three
skipped checks. PR #174 remains draft and behind at
`f4879d4ffd948144c4abc385255bdf5964c94f33`; its link-check rerun failed on
18 GitHub HTTP 503/504 responses. That failure is not a green CI result and
was not hidden by exclusions or a merge.

No new real-weight run, Apple run, HF publication, push, PR mutation or merge
occurred in this continuation. All 194 public rows remain in scope, and the
58 unresolved code/artifact classifications are not promoted by these
mechanism-only tests.

### 2026-10-02 controller safety regressions and transfer preparation (11:12 UTC)

The reviewed integration checkout is clean at
`5b78f8e5e988bcb22f3e05669f839b97262592da`. Root revalidated the pinned
binder, binding-test and canonical-generator hashes, then passed their combined
25 standard-library unittest cases in 1.754 seconds with ResourceWarning
treated as an error. There were no skips. This is model-free mechanism
evidence, not independent real-weight CPU parity or Rust consumer evidence.
The canonical persistent fixture remains absent.

Root reproduced a portable-deadline failure in bootstrap draft SHA-256
`e4d9bbf221fdf569f48bdfa6299c41bbe4d97f258a0d1c8397f5b41f89fba6e9`.
With the external timeout runner disabled, a TERM-ignoring shell remained
live seven seconds after starting a two-second deadline. The isolated
diagnostic process group was killed and reaped; the reviewed script was not
changed by the diagnostic. Paid execution remains rejected pending a portable
TERM-to-KILL runner and whole-lifecycle ownership, signal and recovery tests.

A separate read-only audit of the preserved full-replay v6 controller,
SHA-256 `a2dc4e161f3b18bcdf2d733316fc4d0a884576f111c28b4d83926d0d471a931f`,
passed syntax and ShellCheck but found further safety gaps: the positive
create-response ID was treated as owned before exact-label verification;
general CLI/SSH/SCP operations lacked a shared deadline; and the final
label-absence validator accepted malformed non-dictionary rows. Its old
`218ce74d` target is also not the integrated source head. The old controller
is preserved, while ownership and strict absence regressions are delegated
to a separate v7 draft. Neither draft is approved for paid execution, and
retargeting the full replay still requires a verified persistent fixture and
the subsequent final clean commit.

Root created and verified a complete-history transfer bundle for clean
`5b78f8e5` at
`/private/tmp/vokra-firered-reviewed-bootstrap-input-20261002.2gpaUF/repo.bundle`.
Its size is 57,873,950 bytes, below the 128-MiB input limit, and its SHA-256 is
`d2a8ba44c4c10e6b8f1b43dae368cf7936958db978e70017b57917de9fe2ebd3`.
The bundle contains the intended HEAD and passes `git bundle verify`; it has
not been transferred or executed remotely. The authenticated source record
and remote-only UV installer still match their previously recorded hashes.
The offline Vast wrapper redaction, exit-status and explicit destroy-confirmation
regressions also pass.

Fresh authenticated paginated VAST readback reports zero Vokra instances,
one unrelated instance and a null next token. The unrelated resource was not
modified. No paid allocation, persistent fixture generation, Rust compilation,
model execution, Apple run, publication, push, PR mutation or merge occurred.
All 194 public rows remain in scope; no code/artifact classification or
hardware-completion state is promoted by this preparation.

### 2026-10-02 owned-instance recovery and cleanup-signal review

The full-replay v7 controller is frozen at SHA-256
`b553f79eae126fc8e71fddc92a0427c248ec3771456c0f333d8bdf4bd9f2f254`.
Root reviewed its complete diff against preserved v6, passed Bash syntax and
ShellCheck, and independently completed the full offline self-test with exit
zero. Ownership and bundle-observation markers passed; the checksum-tamper
warning belongs to the expected negative case. Common registration now retains
an individually proven owned ID solely for cleanup after a nonzero create
exit, while aborting work. Ambiguous exact-label recovery checks every ID
individually, destroys owned candidates, and checks individual null responses
plus an empty final list with an explicit null pagination token. This accepts
only that recovery/ownership scope: global deadlines, cleanup interruption,
transfer limits and retargeting to the eventual final integration HEAD remain
pending. The old `218ce74d` target is not approved for production replay.

Root separately reproduced a cleanup-runner defect: inherited SIGTERM-ignore
was overwritten by the ordinary interruption handler, yielding exit 143 before
the tiny diagnostic command completed. A subsequent embedded-runner snapshot
from bootstrap SHA-256
`8e9b1a00f2923a994b14c29af9403cdccf9cae4dd32b9108f1957261b112df9e`
passed four independent cases. Cleanup mode completed after either SIGINT or
SIGTERM, while ordinary mode stopped with exit 130 or 143 respectively.
The embedded code SHA-256 is
`156c8fef00043a939466d5b506b2ae278561666833c9cfe39ada8997f276298d`.
This is runner-only evidence, not approval of the UV launcher, whole cleanup
lifecycle, schema fixture, real-weight parity, or paid execution.

Root subsequently completed the same frozen bootstrap's full offline self-test
with exit zero: four actual generator unittests, five mocked remote-body output
files, and eleven lifecycle cases passed. Those cases include nonzero create,
ambiguous ownership, unrelated/unknown IDs, residual readback and synchronized
INT/TERM with a second process-group signal during cleanup. This additionally
accepts the tested model-free lifecycle integration, but not real remote
generation or operational readiness. Remote prerequisites and pre-transfer
log recovery remain a separate stage; the canonical persistent fixture is
still absent and paid execution remains unapproved.

The latest authenticated VAST readbacks at approximately 11:44 UTC report
zero Vokra instances, two unrelated instances, an explicit null pagination
token, and zero persistent volumes. The unrelated instances were not modified.
No new paid worker was allocated during these checks.

Read-only Realtime reconciliation at clean integration `5b78f8e5` confirms that
the isolated candidate already contains the native source-ordered streaming
runtime composition; that integration is absent from the maintainer checkout.
It must not be implemented a second time. The unchanged CLI remains
inspection-only. The Carter preset manifest,
SHA-256 `d56c5fbd82d42c80f81ee0f550ad6a4ff4c160cb8004d394552a3a04e6b9c957`,
retains `INSPECTION_ONLY`, `NO_MODEL_FORWARD_NO_AUDIO`, `NO_UPLOAD`, and
unproven rights/voice consent. The recorded dependency audit, SHA-256
`3e41d14ef2260f4eafe7775ae4345e626593f0e524da973149bb75109d2faa5b`,
still requires owner review and reports undiscovered installed license files
for safetensors, tokenizers, tqdm and triton. Source license discovery is a
separate pending investigation, not owner approval. The existing narrow
reference packets do not authorize full streaming generation; an exact
external execution scope and independent CPU waveform evidence remain open.

These checkpoint-free diagnostics and reconciliations do not advance any
public-model completion state. All 194 public rows remain in scope, and no
model/preset/tokenizer download or execution, Rust compilation, Apple run,
artifact publication or PR mutation occurred in this continuation.

### 2026-10-02 archive-license helper review and descendant-leak rejection

A separately reviewed implementation candidate is committed at
`501215c3b6a54d62a07c745b056e00327c3429b5`, based on clean integration
`5b78f8e5e988bcb22f3e05669f839b97262592da`, in the isolated checkout
`/private/tmp/vokra-vibevoice-dependency-archive-audit-20261002`.
Its three-file change is absent from the maintainer checkout and has not been
pushed. The archive collector SHA-256 is
`7f3b8d524b2fd6df1841d79fbd06663e635a2baab506611275e6fc9342550fa1`;
root independently passed its 24 named tiny-archive cases and three focused
unittests. Zero-dependency, the 190-file EOL-pin check and staged diff hygiene
also passed. No real archive was acquired or executed.

The collector authenticates the historical audit, exact lock and archive
identity; separates distribution-owned primary licenses from vendored and
NOTICE evidence; rejects non-regular, empty, ambiguous and unsafe inputs;
and caps aggregate license evidence at 8 MiB. Metadata body text cannot
fabricate a License-File header. Sdist evidence remains distinct from installed
wheel evidence and cannot fill owner sign-off or authorize model execution.

The historical audit records a 47,564-byte lock with SHA-256
`5cfaad7532ca8e144173cc3552b7f3f9585a15be3b81b29e34a0b0c6355a5603`.
Bounded isolated-candidate history lookup found only the different
58,249-byte tracked lock at
`4482e80bd4a61e5cb6ff62516835a8e7cfd19492`, SHA-256
`e8787e3c9e7bfdb3383fca1bba22026edbec4b9e61191692797e43e801f74352`.
At that lookup, the required historical original had not been found. No
replacement lock, reconstructed identity, dependency approval or real license
closure was inferred. The subsequent maintainer-history lookup below supersedes
the availability limitation, not the license or execution gates.

Root rejected the full-replay v8 draft frozen at SHA-256
`cc1fedda900949eb3a844a2ca4be7d1f5fe3ce849dd22be64dd86adb20acfd6d`.
An independent tiny-process diagnostic returned runner exit zero while its
child process group remained live after the parent exited. Root killed the
diagnostic's own remaining group. The draft therefore has no paid-execution
approval despite its reported deadline-only test result. Whole-group teardown
and legacy-suite deadline isolation are delegated corrections; the accepted
v7 ownership snapshot remains unchanged. Bootstrap transfer-prerequisite and
partial-log recovery corrections are also still under review.

Authenticated paginated VAST readback at approximately 12:09 UTC reports zero
Vokra instances, two unrelated instances and an explicit null pagination token;
the volume readback reports zero persistent volumes. No unrelated resource was
modified and no paid worker was allocated. PR #152 remains draft/open at
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, with 76 successful and three
skipped checks; no PR mutation or merge was performed. All 194 public rows,
real-weight CPU gates, Apple CPU/Metal/no-fallback gates and separately
authorized publication remain in scope and unpromoted by this slice.

### 2026-10-02 exact historical Realtime lock recovered from Git history

Root expanded the lookup to the maintainer checkout's existing local refs,
without merging or pushing its divergent branches. Two historical commits,
`9879093f91b0900964673f91482c0907a12fceeb` and
`31ab7d84828599d6edf83fe5049917fee88dc9c3`, contain the identical recorded
47,564-byte lock, Git blob `4fee981efbd3d1bfe1826b569920c281800f2455`.
Root independently hashed each blob to the required
`5cfaad7532ca8e144173cc3552b7f3f9585a15be3b81b29e34a0b0c6355a5603`.
The previous missing-original result was therefore limited to the isolated
candidate's history, not all local history.

Exact-byte recovery and archive-candidate metadata extraction are delegated
as separate evidence files; the current reference project's lock is not being
replaced. The authenticated original permits the historical archive evidence
collector's input gate to be satisfied once the recovered file is verified.
It does not approve installing this old dependency closure, importing its
packages, executing a model, or publishing an artifact. Real archive/license
collection remains VAST-only and has not run.

### 2026-10-02 completion scope and recovered historical archive identities

The user-facing goal clarification preserves all 194 public model rows, not
only the presently prepared FireRed and Realtime slices. Completion requires
source/license decisions, a working converter/binder/native/CLI route,
independent real-weight VAST CPU parity, final Apple CPU/Metal/no-fallback
evidence, authorized public-artifact reconciliation, reviewed PR/CI/docs,
and removal of unnecessary campaign compute/storage. Withholding must remain
an explicit disposition, not an implementation or hardware PASS. The latest
recorded 136 code/artifact-full and 58 unresolved classification is not a
136-model Apple-completion verdict. Scaleway remains the final hardware leg
after non-Apple gates are closed or explicitly disposed of.

The recovered evidence directory
`/private/tmp/vokra-realtime-historical-lock-evidence-20261002` is now frozen.
Root verified the regular, non-symlink 47,564-byte lock against the historical
audit's SHA-256 `5cfaad7532ca8e144173cc3552b7f3f9585a15be3b81b29e34a0b0c6355a5603`.
The manifest explicitly separates Git blob object ID
`4fee981efbd3d1bfe1826b569920c281800f2455` from that content digest.
Its current SHA-256 is
`91b65a06e2be6a0d15477db9c9d6c12163af82e3ca65f6f47260b5516d1ad5c6`;
the evidence README SHA-256 is
`92bcaf97ec817adbc7b5969cfedf128573516403620c69c6a45aa1558418b8d9`.
Earlier reported manifest hashes are superseded, not additional inputs.

Root's independent Python 3.12 stdlib-only comparison verified all eleven
manifest archive rows' exact URL, SHA-256, byte size and upload timestamp
against the original lock, and the recorded Linux/x86_64/Python/UV platform
fields against the authenticated audit. Four Linux wheel candidates are
identified for safetensors 0.5.3, tokenizers 0.21.4, tqdm 4.67.1 and triton
3.3.1. These are candidate archive identities, not proof of the originally
installed wheel bytes. No real archive was downloaded, installed, imported
or executed; primary license-byte collection and policy review remain open.

The replacement full-replay controller v8r1 is frozen at SHA-256
`9acfac1320dd21c85434a148abb2d5da5a4a6642d376363cba6192b39a5bbe25`.
Root independently passed syntax and ShellCheck and the previously failing
tiny parent-exit diagnostic: exit zero now leaves no descendant process
group. Root then completed the full offline regression with terminal exit zero
(session `65978`), including ownership, deadline/signal, wire and bundle
observation checks. The frozen file's SHA-256 remained unchanged. This accepts
that offline lifecycle snapshot, not a remote code/parity run.
The controller still targets the older `218ce74d` input;
current integration/fixture retargeting and production approval remain pending.
The actual canonical FireRed schema fixture is still absent, and operational
bootstrap corrections remain separately under review. No model-completion
state, real-weight parity, Apple verdict or publication is promoted by this
evidence-recovery checkpoint.

Authenticated VAST readbacks at approximately 12:39 UTC report zero Vokra
instances, one unrelated instance and an explicit null pagination token;
the persistent-volume query reports zero volumes. No unrelated resource was
modified and no new paid worker was allocated during this checkpoint.

### 2026-10-02 actual schema bootstrap and disposable-worker cleanup

This later checkpoint supersedes only the preceding fixture-absence and
bootstrap-pending statements. It does not promote a model's CPU, Apple,
license or publication state. The root-reviewed schema controller stage3
passed its full offline regression (terminal session `14957`, exit zero):
four generator tests, five mocked output files and twenty lifecycle cases.

The first production attempt ended before creation (session `96973`, exit
one): an exact-ID offer search returned an empty list. A subsequent fresh
eligible-offer list still included that offer, so the evidence does not prove
that it was sold out. Stage4 changed preflight to a fresh eligible list with
an exact local ID/cap match. Its production attempt (session `84409`, exit
one) created instance `53861014` but stopped before SSH or fixture generation
because the service reported no direct endpoint. The valid SSH proxy fields
were present. The controller destroyed that instance including its storage;
root independently verified individual `instances: null` and a strict empty
label-scoped list with an explicit null pagination token.

Stage5 added validated SSH proxy transport when a direct endpoint is absent,
plus bounded endpoint readiness polling. This is a transport alternative,
not inference CPU fallback. Its frozen SHA-256 is
`624944f2e632e4d48ed4a3a95aaa3da66ec924780f09e64a4d248993c67ac161`.
Syntax and ShellCheck passed; the delegated full offline regression completed
with four generator tests, five mocked outputs and twenty-one lifecycle cases.
Root also independently checked the endpoint decoder against the failed
worker's actual metadata before approving the next attempt.

The actual stage5 run (terminal session `12813`, exit zero) used exact clean
generator HEAD `5b78f8e5e988bcb22f3e05669f839b97262592da`. Four generator
tests passed remotely and five files were recovered under
`/private/tmp/vokra-firered-schema-fixture-bootstrap-run-20261002-1301/packet`.
Root independently verified the exact five-name allowlist, regular/non-symlink
files, all five SHA-256 values and their 5,210,545-byte aggregate. The packet
is explicitly `SYNTHETIC_SCHEMA_ONLY=true`, `REAL_V1_ACCEPTANCE=false` and
`PARITY_READY=false`. It contains no model checkpoint, tokenizer or Torch
execution evidence. Its authenticated source-record digest remains
`11c96d58909bb2863e2e92b2f1420b4545d7e7b5a08021e8b90954210293475b`;
the generator digest is
`2c5018efa36f4848e6f11f682f958396b7eedcdc93456c789b408ebf8a846d27`.

The successful worker `53861834` was also destroyed including saved data.
Root verified individual `instances: null` and a strict empty label-scoped
list with an explicit null pagination token. Neither attempt retained a
worker or storage. No unrelated instance was modified, and no HF credential
or `.env` was transferred. Canonical fixture installation, combined clean
HEAD fixation and latest-head remote Rust gates remain the next steps;
independent real-weight parity, Apple CPU/Metal and publication remain open.

### 2026-10-02 canonical fixture integration and fresh metadata audit

Root reviewed the five recovered fixture files against their source packet
with byte-for-byte comparisons, reviewed the LF-pin/handoff diff and ran the
fixture EOL, zero-dependency, forbidden-symbol and diff checks. The isolated
fixture commit is `32fe24a6884419812ba7a528bc6c17c6b549fd3f` (seven files:
five fixtures, `.gitattributes` and the matching dated handoff). Its explicit
schema-only/non-accepting/non-parity flags remain unchanged. A root attempt
to run the binder test script locally was refused before execution by the
maintainer model-execution guard; the guard was not bypassed. The delegated
stdlib-only binder report records 21 passed tests, but latest-head VAST must
independently rerun those tests and all required Rust gates.

The previously reviewed archive-helper commit `501215c3` was mechanically
integrated as a separate three-file commit,
`597164c7d1a81abf016ebc26400e597a91ab8811`. Root verified the helper,
focused test and design hashes remained exactly
`7f3b8d524b2fd6df1841d79fbd06663e635a2baab506611275e6fc9342550fa1`,
`b45f4cbecb4c7d125b9d4f8ff85dc32037a09c456bc58e492c27ae5c61b24f71`
and `18600ee813ae734d1660035d9435b45172fbddc813e0102b3b2c698cc23b7798`.
The combined tree is clean and is the next
remote verification target; root's separate documentation branch is not a
wholesale merge or code-push input. Combined-tree EOL, zero-dependency,
documentation references (110 IDs), runbook citations (1,132 anchors) and
diff checks passed. Latest-head Cargo/Clippy/deny/audit, real-weight CPU,
Apple and publication results are not claimed.

At approximately 13:15 UTC, the existing read-only HF audit completed
successfully at that exact combined HEAD (terminal session `8868`, exit
zero). It requested only HF API metadata and README cards, not GGUF tensor,
checkpoint, tokenizer or preset bytes. It reports 194 public repositories,
193 GGUF-bearing repositories and 198 GGUF files; CPU classification is
136 full, 43 partial, 14 no-runtime-binder and one non-artifact; Metal code
classification is 136 full, 57 blocked-by-CPU and one non-artifact. Thus the
58 unresolved code/artifact rows remain, and 136 full is still not an Apple
hardware verdict. Audit source SHA-256 is
`690d603f7182fc5c0645e784f8857acee9604b20050892b2bf0eadeb41f8a905`;
engine source SHA-256 is
`cc22a9074531fa738c68101f4697b1b4393252b913b1bb2bc61092da93cf0b1b`.
The canonical inventory and execution plan receive the same dated refresh.

The separate VAST-only archive collection leaf is frozen at SHA-256
`eed0d883ae2569ee67ea78896cde7a89beeddde33356376cb9aadb3e1c36a03c`.
Root reviewed its bounded HTTPS/input/output and whole-process-group cleanup
paths, then independently passed all fourteen tiny, networkless cases
(terminal session `16710`, exit zero) and authenticated the actual historical
audit/lock/manifest/helper inputs without a network request. This accepts
only the offline collector snapshot; actual wheel/license-byte retrieval,
installed-wheel identity proof and policy decisions remain open.

Fresh read-only PR results still show PR #152 open/draft at `dd6f0154`,
clean with 76 successful and three skipped checks; PR #174 is open/draft at
`f4879d4f`, behind with 68 successful, one skipped and one failed
`documentation-links` check. Neither was modified, re-run or merged during
this checkpoint. The two bootstrap workers remain individually confirmed
destroyed; no new paid worker was allocated for fixture integration or this
metadata refresh.

### 2026-10-02 preflight rejection and current-source inventory correction

The next paid runs remain gated on controller review. Root rejected an
archive-controller draft at SHA-256
`90029d3cbec2bd589df0decb3bf23aa7d6c2d5841612f3902094cb2f191f00fe`:
its later function definitions overrode the inherited lifecycle regression
suite with differently named cases, including no-op checks. It also hashed
the transfer payload only after copying it locally, and its remote archive
manifest path did not match the staged worker path. These are review defects,
not a successful lifecycle or archive-evidence result. The implementation
owner was instructed to preserve the actual inherited create/ownership,
signal/deadline, failure-recovery and destruction cases, bind a remote
transfer checksum manifest before copying, and recover bounded negative
evidence without promoting it to PASS. Neither this rejected draft nor an
in-progress correction is authorized for a production run.

A separate read-only audit of clean candidate `597164c7` found the FireRed
reader contains 43 test attributes: 40 ordinary cases and three explicitly
ignored real-weight consumers. Root independently checked the attributes,
ignore declarations and unchanged reader SHA-256
`3f79aacc702c57b16bbb013ef973c5cb5e1dcf1db6bf6e017441928584806602`.
The ordinary cases split into seven `tests::`, 23 `decoder_trace_v2::` and
ten `capture_binding_v1::` cases. The earlier planning inventory of 37
ordinary plus three ignored was inaccurate; it was not an executed Rust
result. The old controller's 25-pass/two-ignore reader expectation and
24-case source-capture expectation are also stale. Its successor must bind
the actual complete test identities, including the 33 source-capture,
21 capture-binding, four generator and three archive-helper unittest cases.
The helper's 24 tiny cases are a separate self-test inventory. No latest-head
Rust or real-weight success is inferred from this source enumeration.

Fresh read-only cloud and GitHub checks completed in terminal sessions
`67196`, `58013` and `58840`, all exit zero. VAST reported zero Vokra
instances, one unrelated instance and an explicit null pagination token;
the unrelated instance was not changed. Two eligible offer candidates were
observed with 200-GB-inclusive estimated rates below USD 0.20/hour, but no
instance was rented; availability and pricing must be rechecked immediately
before allocation. GitHub main remains `97447185361a37af64c1b30fe87e8e2618d96e20`.
PR #152 remains open/draft/clean at `dd6f0154` with 76 successful and three
skipped checks. PR #174 remains open/draft/behind at `f4879d4f` with 68
successful, one skipped and one failed check. No PR or remote branch was
modified. All 194 public rows, independent real-weight CPU validation, final
Apple CPU/Metal/no-fallback validation and publication boundaries remain in
scope and incomplete.

### 2026-10-02 row-level audit, controller regression and ECAPA identity correction

The row-level metadata-only audit at clean integration HEAD
`597164c7d1a81abf016ebc26400e597a91ab8811` completed in terminal session
`23770`, exit zero. Its retained TSV is
`/private/tmp/vokra-public-coverage-live-20261002-1358.tsv`, 37,881 bytes,
SHA-256 `9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
It enumerates all 194 public rows and confirms the earlier 136
code/artifact-full and 58 unresolved classification. Only HF API metadata
and README cards were requested; no model, tokenizer or preset bytes were
downloaded. This inventory is not a new real-weight or Apple verdict.

Review of current source and the row-level artifact identities found that
Qwen3-ASR 0.6B/1.7B, NSNet2 and SpeechBrain ECAPA already have source
converter/binder/CLI/reference routes. They are not thereby VAST-ready:
Qwen's live artifacts lack required execution metadata/sidecars and its
dependency/license decisions remain pending; NSNet2's live MIT/permissive
stamp conflicts with its recorded CC-BY-4.0 model identity; ECAPA's public
artifact remains out of bounds. Existing owner records do not supply the
required exact replacement/execution approval for NSNet2 or ECAPA. These
facts do not authorize model acquisition, sign-off, replacement or upload.

A primary HF API metadata check completed in tool cell `1503`, exit zero,
without retrieving the GGUF. At
`vokra/speechbrain-spkrec-ecapa-voxceleb`, exact revision
`3dc7704b5861e348edafc0d800291887a284a476` reports the file
`spkrec-ecapa-voxceleb.restamped.gguf`, 83,239,904 bytes, with API-reported
LFS content-object SHA-256
`f03292e93c215037b7d855281bb5786d600a1db3e51e9450c8caeb1360f1d571`.
The old worker revision `3dc7704b2dcb80b8ea8eb2d3db7280f682ac3657`
returned HTTP 404; its old `75e74d4e...` content pin is not the current
replacement target. This is metadata identity evidence, not a locally
measured model hash or repaired artifact.

The implementation owner corrected that worker in an isolated clone as
commit `5f6ce6fb1ec0b0b302ef71ed446e704dd015d052`, followed by a
documentation-precision correction at
`3315339ebbb1f664969ad0172ab205c68813af13`. Root reviewed both diffs and
independently passed the worker's five-case offline self-test, Bash syntax,
ShellCheck, zero-dependency, forbidden-symbol, documentation-reference
(110 IDs), runbook-citation (1,132 anchors across 123 runbooks) and diff
checks. The two-file logical change preserves the upstream/checkpoint,
license/approval, restricted loader, parity bounds and no-upload gates. The
clean isolated clone at `/private/tmp/vokra-ecapa-target-identity-20261002`
is the next code-verification candidate; it is not pushed or merged, and
its workspace/Clippy/deny/audit and real-weight/Apple gates remain pending.
Root's separate management branch is not a wholesale code-push input.

Root's full offline FireRed-controller regression at frozen SHA-256
`cffc200a00c157f1348099648b9ae34e7298fbc459081f4c68e00801914eb90e`
ended with exit one in terminal session `73182`. Ownership and
deadline/signal regressions passed, but the actual extracted remote reader
gate failed on an unbound `FIRERED_CONSUMER_DESIGN_SHA256` variable. Root
also found old 25/24 reader and 24/23 source-capture mutation strings that
no longer mutate the current 40-pass/three-ignore and 33-case fixtures.
The implementation owner must correct the actual inherited harness rather
than supply a separate permissive test. The failed snapshot is not approved
for production; focused reports do not override this full-regression result.

The separate archive-controller's restored inherited lifecycle suite was
independently run by root at frozen SHA-256
`8923c25bc58090c92991fce54348f18902a5553c83b0b7f8c772ebc4acaefbb6`.
Terminal session `90803` exited zero: the actual collection leaf's fourteen
tiny cases, remote-body check, 21 packet checks and original 21
create/ownership/recovery/signal/destroy lifecycle scenarios passed. Syntax
and ShellCheck also passed. This accepts only that first correction stage,
not production: semantic JSON/result verification and bounded negative-JSON
recovery still need implementation and review. Root additionally identified
that the draft's old 128-MiB bundle bound would reject the authenticated
159,334,746-byte archive total; the archive budget must follow the frozen
collector/manifest contract, without relaxing packet/log caps.

A fresh read-only VAST readback in terminal session `98706`, exit zero,
reported zero Vokra instances, one unrelated instance and an explicit null
pagination token. The unrelated instance was not modified, and no paid
worker was created in this checkpoint. All 194 public rows, security/CI
requirements, independent real-weight CPU validation, final Apple
CPU/Metal/no-fallback verification and authorized publication remain in
scope and incomplete.

### 2026-10-02 subsequent harness failure and read-only security review

The FireRed implementation owner corrected the extracted remote harness
bindings and current-count negative mutations in controller SHA-256
`af69b8e6793ce3f76c6894c53fb76a13ad2d61e342214f766e2fa6b976af88f7`.
Its focused extracted reader, source-capture, capture-binding and generator
gates passed in terminal session `1174`, exit zero, with syntax and ShellCheck
also green. However, the owner's actual full offline regression in terminal
session `79457` ended with exit one at `green-exit-self-test=FAIL`. This is
not a production-approved controller. The previously frozen v8r1 full suite
had passed, so the new failure must be diagnosed against the actual child
packet/exit evidence rather than declared an unrelated pre-existing defect.
The strict valid-packet, tampered-packet and cleanup-exit checks remain
required. Archive-helper test integration and the `3315339e` retarget are
still subsequent work, not completed gates.

A delegated authenticated, read-only GitHub API audit used complete
pagination and completed synchronously without a persistent process. Current
open Dependabot metadata remains 209 alerts: 182 have a published
`first_patched_version` and 27 do not. This is patch availability, not 182
repairs in the repository. The current API represents 37 manifest paths and
30 project roots; those metrics are not interchangeable with the earlier
local 49-tree inventory. The no-fixed-version rows cover Accelerate in
NanoCodec/Ultravox, NLTK in the Misaki integration, and three Torch advisories
across eight manifests. Missing upstream patches do not establish that
removing an unused dependency or obtaining another allowed official route is
impossible; any alternative still needs compatibility and license evidence.

The same read-only review confirms PR #152 remains open/draft/clean at exact
head `dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, base
`97447185361a37af64c1b30fe87e8e2618d96e20`, with 76 successful and three
skipped checks and no failures. Its XCodec2 reference project overrides the
official Torch/TorchAudio 2.5.0 requirement with 2.13.0/2.11.0 and pins
Transformers 5.10.4. Historical model-free import evidence exists for an
earlier head, but current-head API compatibility is not thereby proved;
version-string checks alone are not an independent compatibility test.
The branch's manifest remains `BLOCKED_PENDING_PRIMARY_BYTES` / `NO_UPLOAD`,
with 62 external distribution license/notice rows, native-payload review and
owner decisions still unresolved. The source-only lock change must not be
treated as real-weight reference or parity approval. No PR, alert, branch,
package environment or cloud resource was mutated by this review, and no
merge or model-execution authorization is inferred.

### 2026-10-02 archive-consumer schema defect and XCodec2 execution-gate audit

Root's read-only review of the archive controller's in-progress SHA-256
`c53719358eb08c34cc76b41e95acc06772324a6055e682dbc2803fe66aad1b61`
found a production-input mismatch, not a primary-license verdict. Its new
package JSON verifier requires an exact ten-key object and a four-key
artifact object, but the frozen helper `7f3b8d5` actually returns additional
`recorded_installed_status`, `archive`, `supporting_license_evidence`,
`distribution_identity`, `evidence_scope` and `fallback_policy` fields,
and the artifact includes `kind`. Consequently, a positive mock using only
the invented smaller schema cannot prove acceptance of the real collector
output. The implementation owner must bind positive and negative fixtures
to the frozen helper's actual schema, validate the exact wheel filename in
the summary, and retain the original lifecycle and bounded negative-packet
checks. This WIP is not approved for a paid run.

A separate delegated read-only review of PR #152 at exact HEAD
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a` traced the direct reference
path. `tools/parity/xcodec2/dump_reference.py` imports NumPy, Torch and GGUF
at lines 28-30; its lines 64-78 validate version strings, lines 111-123
import the official decoder, and lines 189-218 authenticate a supplied
GGUF and execute the decoder. Neither this direct path nor the documented
dependency guard invokes the license manifest or a hash-bound owner
execution approval. The guard's documents-only mode exits before third-party
imports, but its ordinary mode is an API/dependency check, not legal approval.
No dedicated XCodec2 VAST worker was found in that exact tree. Thus the
blocked manifest and documented VAST boundary do not themselves enforce
fail-closed direct execution. A bounded isolated implementation has been
assigned to put the executable gate before third-party import and model
access; no approval or primary-license bytes are invented. `NO_UPLOAD`
remains a publication restriction, not by itself a prohibition on separately
approved no-upload reference validation. Current-head API compatibility and
real-weight CPU/Apple parity remain unproved, and PR #152 is not accepted
for merge by this audit.

Root's fresh read-only VAST query completed in terminal session `66824`,
exit zero, with zero Vokra instances, one unrelated instance and an explicit
null pagination token. The unrelated instance was not modified; this query
does not establish Scaleway or persistent-volume state. No paid worker or
model execution was started in this checkpoint. The 194-row completion
scope, 136 code/artifact-full / 58 unresolved classification, and all
independent CPU, Apple and publication requirements remain unchanged.

### 2026-10-02 independent controller failure and producer-contract review (15:02 UTC)

Root independently ran the FireRed controller's mock-only trap-child path
at SHA-256 `084360e3e08e952c7192e3c13b6df4232339a37d46b00a7933840dd3f6596a61`.
Terminal session `44384` returned one: ownership passed but the deadline/signal
self-test failed before the expected trap-child exit. The implementation
owner's full-suite success is retained as a separate observation, not a
supersession of this failure. A diagnostic-only revision at SHA-256
`997b751f34a7f6654966cde376d98b7f0778b17d9b2407eab7e94da598e88cca`
preserves assertions and deadlines. Root's independent deadline-only run,
terminal session `78214`, also returned one without a case diagnostic; the
remaining early and cleanup failure branches need instrumentation before a
cause can be claimed. Neither snapshot is production-approved. The archive
helper integration and final `3315339e` retarget remain subsequent work.

Root also reviewed the isolated XCodec2 pre-import execution-gate candidate
against the actual PR #152 evidence producer, not just its synthetic tests.
The initial gate used `parents[2]` (the `tools` directory) as the repository
root, required a different activity schema, required embedded license bytes
in the producer's three-field summary, rejected the producer's ELF
`readelf_returncode`, and selected the first lock artifact rather than the
installed wheel. The implementation owner corrected those paths and reported
focused tests, but root's next review at gate SHA-256
`1528ae6a4eb21391cc9a12d12f01386374fc0533d206bab4f5c491f24a9485e2`
found another incompatibility: the producer names artifact kinds
`locked_wheel` / `locked_sdist`, while the positive fixture and lock matcher
use `wheel` / `sdist`. That fixture bypasses the installed-wheel binding
branch and cannot establish acceptance of factual producer output. This
uncommitted four-file candidate remains under correction and is not a merge,
owner-approval, model-execution or parity verdict. Root's local direct-test
invocation was refused by the model guard; it was not rerouted to bypass the
guard, and independent execution remains a remote verification requirement.

The clean code-verification candidate is still
`3315339ebbb1f664969ad0172ab205c68813af13`. Root's terminal session `87242`
returned zero for the zero-external-dependency and forbidden-symbol shell
gates and diff hygiene; no Cargo or model execution was performed locally.
Latest-head workspace/Clippy/deny/audit and independent real-weight CPU,
Apple CPU/Metal/no-fallback and publication gates remain pending.

A fresh read-only VAST query, terminal session `66379`, returned zero with
zero `vokra-*` instances, two other instances and an explicit null pagination
token. The other instances were not modified; this query does not establish
persistent-volume or Scaleway state. No paid worker was created in this
checkpoint. The full 194-row scope and 136 code/artifact-full / 58 unresolved
classification remain unchanged; excluded or withheld rows require an exact
owner disposition and must not be represented as supported models.

### 2026-10-02 XCodec2 safety commit and independent controller readback (15:22 UTC)

The isolated XCodec2 safety patch is now a local, reviewed logical commit,
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, on parent
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`. Root verified the clean clone,
the four-file commit scope and unchanged reviewed hashes. The reference
entry point now validates the exact execution approval and factual dependency
evidence before importing third-party packages or touching the GGUF. The
gate reuses the producer's installed-wheel selection, cross-binds retained
license/native payload records, accepts unknown publisher RECORD fields
without accepting known hash/size mismatches, and reads external JSON through
bounded no-follow snapshots. The gate SHA-256 is
`36fef37d2d30239975f1e80247a3c3ebc277cc5ad99725da6f3fcb55962d0f67`.
The implementation owner reports 11 focused and 32 full stdlib tests passing,
AST parsing, zero-dependency, forbidden-symbol and documentation checks.
Root independently confirmed commit diff hygiene and artifact hashes, but
its direct local test remains guard-refused; independent remote execution
is still required. No push, manifest/sign-off change, model execution or
publication occurred. The blocked production manifest stays blocked, and
this source-review acceptance is not a merge or parity verdict.

Root's whole mock-only FireRed controller run at frozen SHA-256
`50f852e6847976ac39dfd87b45b66a7787fd0f76988267e8fa56cb8fe4c8c130`,
session `29257`, is terminal with exit one. Ownership and deadline checks
passed in the outer run, but `green-exit-self-test=FAIL` followed. The
implementation owner inspected the retained green-child packet and found
that its repeated deadline prerequisite failed before green-exit validation;
its stderr had no named-case diagnostic. A bounded diagnostic-only revision
now enables the existing failure diagnostics in those children. The actual
cause is not yet established; neither an environment explanation nor the
owner's earlier successful run supersedes this independent failure. No
paid FireRed run is approved by these mock results.

The clean `3315339e` integration candidate still needs current-head remote
workspace/Clippy/deny/audit verification. A separate model-free disposable
verification controller has been delegated so that those Rust checks and
the isolated XCodec2 stdlib tests can progress without waiting for model
approval or claiming real-weight parity. It is not yet reviewed or executed.
All 194 public rows, the 136 code/artifact-full / 58 unresolved distinction,
owner/legal dispositions, independent real-weight CPU results, final Apple
CPU/Metal/no-fallback checks and separately authorized publication remain
in the completion scope.

### 2026-10-02 independent archive-controller acceptance and launch refusal (15:50 UTC)

Root independently verified the frozen Realtime archive-collection controller,
SHA-256 `38be2988e3fea19c46dd399c28f23292df50795563dec9a4a399f2cbcc78cd03`.
Bash syntax and ShellCheck passed. Whole offline terminal session `12550`
returned zero: the actual leaf's 14 synthetic cases, remote-body test, 33
packet/contract cases and 22 mocked lifecycle cases passed. Root compared
the packet verifier's canonical member paths and distribution-owned primary
license rules with the reviewed archive helper at clean `3315339e`. This
accepts only the bounded collection controller, not a package license grant,
installed-wheel identity, model execution approval or numerical parity.

The production launch was refused before execution by the local-model guard:
the transfer-only `--helper` argument names a `tools/parity/*.py` file and
was classified as local execution. No worker was created, and the command
was not hidden, rerouted through a self-test or retried with a guard bypass.
A narrowly hash-bound recognition of the reviewed remote-only controller
has been delegated; that active hook change still needs root review and
independent regression checks before the launch can be retried.

Fresh metadata-only VAST terminal session `62736` returned zero with no
`vokra-*` instances, one unrelated instance and an explicit null pagination
token. The unrelated instance was not modified. Offer session `50548`
returned eligible offers, but no allocation is inferred from that search.
Root separately confirmed the small official UV installer SHA-256
`58488ae8dbd0773134c92c85e901430e33f99d975bd7f929d26aa9ab0c2f9390`;
no binary, wheel or model was downloaded or executed on the Mac.

The separate model-free Rust controller remains under correction. Root's
review of its frozen candidate found that the actual manifest producer
omits a required status file while the mocked producer supplies it, and
the local success verifier does not enforce every raw and derived exit
status. Failed remote transfer preflight, overall deadlines, authenticated
resource/ownership checks and allocated-core scheduling also need bounded
corrections. Its mocked green result is not accepted as production readiness.
Official primary metadata confirms its fixed UV 0.12.5 release digest and
Rustup 1.29.0 checksum, but actual remote toolchain/tests remain unrun.

The latest clean code candidate remains `3315339e`, with the isolated
XCodec2 safety commit `e6552853` separately awaiting remote test execution.
The full 194-row completion scope, 136 code/artifact-full / 58 unresolved
classification, independent real-weight CPU evidence, final Apple CPU/Metal
no-fallback results and separately authorized publication remain unchanged.

### 2026-10-02 hook-fixture isolation failure and recovery requirement (15:55 UTC)

Root rejected the next hook candidate after inspecting its actual fixture
lifecycle. Following a symlink-rejection case, the self-test wrote a new
fixture through that still-present symlink into the temporary production
controller. The reviewed `38be2988` bytes are no longer at that path: root
observed SHA-256 `b3ba5208edfcfcf7dc6b159b6e2988b44992ae6e9e318ee0071830c70037ecab`
and a one-line inert fixture. Root's guard session `65451` returned zero,
but that result is not accepted: it proves neither fixture isolation nor
production-file immutability. No hook change has been committed or accepted.

The implementation owner must unlink only the fixture symlink before any
fixture write, verify that production bytes are unchanged across self-tests,
and recover the original controller from retained authoritative bytes or
patch records. The frozen hash may not be changed to bless the overwritten
file. A new nonmatching reconstruction would require a new source review
and independent whole-suite validation. The earlier `12550` result remains
historical evidence for the exact `38be2988` bytes, not approval to run the
current file. Production allocation remains unstarted; no model, wheel or
public artifact was acquired or executed on the Mac.

### 2026-10-02 actual VAST code verification and FireRed compile failure (16:22 UTC)

Root independently accepted the frozen FireRed model-free controller at
SHA-256 `7c323719fb414bbaa1b001463effb029fcbe4d32d5ef8e630f9bf3302f898faa`:
whole offline session `37194` returned zero, including ownership, absolute
deadline, wire-process, packet and bundle-observation checks. Bash syntax
and independent ShellCheck session `67515` passed. The actual main packet
contains 61 payload entries (28 logs, 26 exits and seven metadata files),
not an invented larger member count. The archive-helper gate checks the
actual three unit tests and 24 named synthetic helper cases; none of this
is independent real-weight parity.

Root then started production controller session `21345` with reviewed
offer `32178462`. Creation registered owned instance `53883938` under the
unique `vokra-firered-output-contract-verify-v9-3315339e-20261002T161319Z-61629`
label. The reviewed allocation has 16 effective CPU cores, approximately
258,023 MB RAM and 200 GB rented storage; the controller uses all 16
allocated cores. Its work deadline is 10,800 seconds with a separate
180-second cleanup budget. The clean remote code target is `3315339e`.
No model bytes, HF token or local `.env` were transferred, and this is
Rust/source verification, not a GPU model run or Apple result.

The actual remote gates passed source audits, focused Mimi/ABI/PCM tests,
FireRed focused/full tests and the wire contract. Both the FireRed consumer
test and its Clippy leg returned raw exit 101. Root's bounded, read-only
SSH observations (`16241` and `43875`, both terminal zero) confirm 13
consumer compile errors: inaccessible integer-parser helpers, an undefined
status constant, stale selected-axis/index struct-field accesses, an
unresolved optional-shape type and a `usize`/`u64` arity mismatch. This is
not a numerical-parity failure. A library unused-method warning also needs
review. The failure remains authoritative; no skipped test, relaxed bound,
mocked packet or successful preceding leg supersedes it.

Root's bounded log observation `41154` also returned zero and distinguished
two controller-wiring failures: capture-binding ran 21 tests with 22 errors
because `VOKRA_FIRERED_SOURCE_RECORD` was absent, and the fixture generator
returned raw zero but skipped its authenticated-record generation test.
The semantic fixture gate correctly failed rather than accepting that skip.
The official source-record fetch and builder passed, but the actual remote
commands omitted that record's environment binding for these two legs.
The next controller revision must pass the exact authenticated record and
regress the extracted commands; synthetic all-green lifecycle logs alone
did not establish this production precondition. The current frozen file
must remain unchanged. Other completed source and archive-helper legs do
not supersede these failures.

At this timestamp the controller is still live and completing the remaining
bounded code/source legs. Evidence collection and owned-instance destruction
are not yet verified, and the earlier zero-instance readback predates this
allocation. The consumer correction is delegated to a separate clean
candidate; frozen production inputs must not change during this run.
The hook/archive recovery and separate all-feature/XCodec2 controller also
remain under review. No code push, merge, public upload or Scaleway allocation
is inferred. The complete scope remains 194 public rows, 136
code/artifact-full and 58 unresolved, with independent real-weight CPU,
final Apple CPU/Metal/no-fallback and separate publication gates intact.

### 2026-10-02 failed-run diagnostics, destruction and bounded correction (16:30 UTC)

Production session `21345` is now terminal with exit one. At exact clean
`3315339e`, the remaining all-feature model Clippy leg failed because the
unit-test-only `ranked_score` wrapper was compiled into the library, and the
workspace leg failed on the same 13 FireRed consumer compile errors. Deny,
audit, source-packet construction, final HEAD, clean worktree and diff-check
gates passed. This is a failed code-verification run, not a real-weight,
source-cache or Apple verdict; the separately frozen Moshi source-cache side
was not executed because the main leg failed.

The failure-path collector recovered the 61 small main payload files, without
symlinks, including logs, raw exits, environment and source-record metadata.
However, source-evidence SCP returned 124 within the finite cleanup budget:
`collect_remote_logs_rc=1`, the local source directory is empty and both
`remote-packet.sha256` and `source-only.packet.sha256` are absent. Complete
packet/provenance acceptance is therefore unproved. Root's local diagnostic
hashes are `4a30e10eecc83137ad45a2d7623a8d949acb067c4a8cffd8e95781740d001dd2`
for the consumer log and
`35da99a9498ffa71085e6532aa5df02971db1fdd6909ee581f2961ad7d4b346c`
for capture-binding; they identify recovered diagnostic bytes, not a remote
manifest verification. The next controller must prioritize remote manifests
and bounded failure diagnostics before optional source payloads, while
preserving the independent destroy reserve and absolute deadlines.

Cleanup returned `destroy_rc=0`, `destroy_readback_rc=0`,
`post_destroy_list_rc=0`, `post_destroy_label_absent=0` and strict readback
PASS. Root independently read the individual `instances: null` response
and the exact-label empty list with explicit `next_token: null`. Fresh
all-instance observation `83068` returned zero with no `vokra-*` instances,
two unrelated instances and null pagination. Only owned `53883938` and its
storage were destroyed; unrelated resources were untouched.

The isolated FireRed correction is source-reviewed for remote validation.
It fixes integer/status/type/arity compilation, tests established sample and
beam-row bounds instead of inventing full-matrix selection fields, retains
the existing parser's axis/index tamper tests and makes the test-only score
wrapper `cfg(test)`. No bound, license, source identity or publication gate
was relaxed. Focused static checks pass; Cargo execution remains VAST-only.
A new controller, rather than a mutation of frozen V9, is required for the
authenticated-record wiring and evidence-order fixes.

The hook owner fixed the symlink write site but could not recover the exact
`38be2988` controller from authoritative retained bytes. Root's independent
guard test `79367` passed after the fix and confirmed the corrupted production
file's hash remained unchanged. That proves fixture isolation, not recovery
or production readiness. A distinct V3 controller has been delegated for
new review and whole-suite validation; the old frozen hash will not be
repinned to the inert fixture. No row, merge or artifact is promoted by these
bounded corrections. The full 194-row completion scope remains active.

### 2026-10-02 clean correction candidate and PR link-check readback

Read-only Git inspection confirmed the isolated FireRed correction is
committed at `db26d377f90c308e690e37a0871195c712d88a80`, with a clean
worktree. The separate XCodec2 pre-import execution-gate tree remains clean
at `e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`. Neither candidate has a
new independent remote result yet. The main-branch API readback is
`97447185361a37af64c1b30fe87e8e2618d96e20`; these isolated candidates are
not represented as merged main or published artifacts.

PR #174 remains draft at `f4879d4ffd948144c4abc385255bdf5964c94f33`.
CI Security run `36658360528`, attempt three, is terminal failure;
documentation-links job `110932221071` returned exit two with 22 errors,
each a GitHub 503 response. This followed a single failed-job rerun after
three representative primary URLs returned HTTP 200 on bounded HEAD probes.
Those probes do not establish CI success. No link exclusion, retry-policy
change, code push or merge was made, and the failed check remains open.

The replacement remote controllers are still under root review. The generic
workspace/XCodec2 controller's owner reported its offline actual-body tests
passing, but review requires authenticated effective-core normalization and
reserved destroy time before production acceptance. FireRed's replacement
must create and validate real manifests before priority transfer, and the
Realtime archive replacement must validate the actual helper's primary
license schema instead of accepting empty package objects. No new worker was
rented for these review-only steps; no owner decision, model execution,
numerical parity or publication gate is promoted. The full 194-row objective
and final Apple CPU/Metal/no-fallback requirements remain active.

### 2026-10-02 controller acceptance and new code-verification start

Root independently ran the complete offline FireRed V10 suite in terminal
session `32551`, exit zero, after Bash syntax and warning-level ShellCheck
passed in session `78505`. Controller SHA-256 remained
`1469c3827bfcacb498fee7ea18cc17ee5909a5643bd9aeca954457116ce7595a`
before and after execution. The normal and fallback priority regressions
execute the embedded manifest producers on fresh tiny fixtures and validate
the retained manifest hashes; failed preflight performs no SCP, and optional
transfer failure preserves the main manifest and consumer failure records.
Authenticated-record wiring, ownership, deadlines/signals and inherited
whole-suite gates passed. These are controller tests, not model execution.

The generic workspace/XCodec2 controller's frozen `da8fec65` independent
suite passed in session `69520`; its subsequent `819cf31f` suite passed in
session `59913`. Root nevertheless found an incomplete tool-destination
bootstrap condition and requested an actual absent-directory/symlink
regression before production acceptance. Neither snapshot authorizes a new
generic worker; its all-target/all-feature workspace and separate clean-head
XCodec2 32-case remote result remain pending. The replacement Realtime
archive controller and guard repin also remain under review.

Before the new FireRed replay, independent all-instance session `31673`
returned zero Vokra workers, one unrelated worker and explicit null
pagination. Fresh offer session `64299` confirmed offer `32178462` with
16 effective CPU cores and a 200-GB-inclusive estimated rate of
USD `0.17407407407407408` per hour, below the USD 0.20 review cap.
Production session `69864` then created exactly one owned instance,
`53891836`, labeled
`vokra-firered-output-contract-verify-v10-db26d377-20261002T171715Z-64772`.
Independent individual readback `80889` confirmed that exact ID/label,
running state, 16 effective cores, 200-GB storage and the same estimated
hourly rate. SSH readiness and input-bundle transfer subsequently passed.

The replay remains live at clean target
`db26d377f90c308e690e37a0871195c712d88a80`; small logs are retained under
`/private/tmp/vokra-firered-output-contract-recovery-v10-logs.V2embv`.
Remote code/source results, the conditional Moshi source-cache side,
complete packet acceptance and storage-inclusive destruction are not yet
claimed. The controller preserves its 10,800-second work deadline,
180-second cleanup deadline and separate destroy reserve. No model weights,
tokenizer or preset are requested, no HF token is transferred, and no
publication, PR push or merge occurs in this replay. Unrelated resources
are untouched. All 194 rows, 58 unresolved metadata classifications,
independent real-weight CPU and final Apple CPU/Metal/no-fallback gates
remain in scope and incomplete.

### 2026-10-02 terminal FireRed schema failure, cleanup and corrected-head replay

The preceding FireRed V10 production session `69864` terminated with exit
one at `db26d377f90c308e690e37a0871195c712d88a80`. The consumer compiled,
but its actual test result was 35 passed, five failed and three ignored;
the workspace stopped on the same five failures. Both raw exit files are
101. Four failures reject the upstream registry's legitimate
`decoder` / `layer_index: null` identity; the fifth test expects a source-head
diagnostic although the authenticated-head invariant correctly rejects the
mutation earlier. These are schema/control failures, not measured numeric
drift, and no tolerance was changed. Source capture binding passed 21 tests,
the fixture producer passed four and the module registry passed nine, without
skips. The retained Clippy, deny and audit exit records are zero. The
conditional Moshi source-cache leg did not execute after the failed main leg.

Priority recovery retained the raw failures and manifests. Root verified
57 recovered members of the 61-entry remote checksum manifest; four named
members are absent (`cargo-metadata.json`, `jobs`, `source.txt`, and the
authenticated decoder source-record JSON). Manifest SHA-256 is
`8d0b5086253175bd525eb06c28a27bc0135359e25226ead8d2369e72bead932b`.
Consumer and workspace log SHA-256 values are respectively
`1a5a1750d8e114e36485f3f52550bc12f81d31b51cd7b9044b9d93c862583a8a`
and `c239a78f8e721a71f6e475b180cafd5c1d9f8ddb3b412737a149351dd4ed962b`.
Bulk collection returned one following an SCP timeout; this is retained
failure evidence, not acceptance of the complete packet. The controller's
destroy/readback/label checks passed. Independent session `28150` then
confirmed `53891836` returns explicit `instances: null`, the exact label is
absent, the complete account listing has zero Vokra workers and explicit
null pagination, and one unrelated worker remains untouched.

Root reviewed Luna's isolated correction commit
`7b6b49cd192b31f392350922c8765e8e0a1234eb`, directly descended from
`db26d377`. Only the FireRed consumer changes: decoder identities require
explicit null, indexed roles require an in-range integer, and wrong types,
unknown roles and missing fields remain rejected. The producer-shaped fixture
now uses the real decoder identity, and the negative head test checks the
actual authenticated-head diagnostic. Focused format, diff, forbidden-symbol,
zero-dependency, EOL-pin and pipefail gates passed; local Rust compilation and
model execution did not run. The commit is clean but not pushed or merged.

The distinct generic V2 controller has SHA-256
`4166a8cf43a85d757b6f7575224740314990a79b0dc44cd7d200dfed1631fdeb`.
Root reviewed its bounded retargeting diff and independently passed Bash
syntax, ShellCheck and the full offline suite in session `77916`; its digest
remained unchanged. Those mock transport/lifecycle tests do not establish
actual Rust or Python results. Production session `43017` created a single
disposable worker `53895587`, labeled
`vokra-clean-heads-model-free-v2-20261003-20261002T174809Z-90591`, for clean
`7b6b49cd` workspace all-target/all-feature tests and Clippy plus the separate
clean `e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf` XCodec2 32-case gate.
Independent readback session `94342` confirms the exact ID/label, loading
status, 16 effective cores, 200-GB disk and
USD `0.1277037037037037` per hour, below the USD 0.20 cap. Its terminal result,
complete evidence recovery and storage-inclusive destruction remain pending.
No checkpoint, tokenizer, preset, HF credential or upload is requested.

Separately, root recomputed all 13 retained ESPnet source receipts at
`cccc29023d43a3f504e28df7d1324bb4eb6daedd`: 168,873 raw bytes agree with
each SHA-256, Git blob SHA-1, primary API blob and declared size. Receipt and
source-facts JSON SHA-256 values are respectively
`754fc50c11bfd501a786c7c9ef3707a66775bea939d0d9cf87a6322e4eb33c42`
and `b5f4e2a88448a74f89e71dd6b073d8a68c269cb0d4bb1f980e7244dc08149707`.
OWSM's existing inspector already binds `selfattn`, `abs_pos` and latest
position mode; the next bounded work authenticates remaining source semantics
and designs the missing native Conv2d8/encoder integration. These source-only
facts do not approve dependency installation, model execution or publication.
All 194 public rows, 58 unresolved metadata classifications, genuine
independent real-weight CPU and final Apple CPU/Metal/no-fallback gates remain
active and incomplete. The Realtime archive controller and dirty hook remain
under their separate review; no unrelated change is staged here.

### 2026-10-02 release-asset bootstrap failure and independent destruction

Production session `43017` subsequently terminated with exit one before
workspace or XCodec2 execution. The pinned rustup installer checksum passed
and Rust installation completed, but the uv archive failed its checksum.
No complete output packet exists, and the secondary collection preflight
reported a missing output member. These are setup/collection failures, not
Rust test failures or real-weight evidence. Preserve the immutable V2
controller digest and its prior offline-test result; those tests skipped real
bootstrap acquisition and did not prove this branch.

Root queried the [official uv release API](https://api.github.com/repos/astral-sh/uv/releases/tags/0.12.5)
without downloading an archive: the exact Linux x86-64 asset has 23,015,306
bytes and digest `sha256:68a509da24b06b4223a1c0175fb5eb5bc79342b76cbeff0cfe51ac3f5b17b6b2`,
which agrees with the existing pin. Bounded primary HEAD probes return HTTP
302 with zero redirects when not following, and HTTP 200 after one HTTPS
redirect when following. The controller's GitHub release curl commands omit
redirect following. This supports correcting bounded HTTPS-only acquisition,
not replacing the authenticated digest with an observed unchecked download.
A separate V3 correction and actual tiny-fixture bootstrap-branch tests are
delegated; no new worker is authorized by an unreviewed candidate.

Saved cleanup reports `cleanup_rc=0`. Independent session `98208` confirms
`53895587` returns explicit `instances: null`, its exact label is absent,
and the account listing has explicit null pagination and zero Vokra workers.
One unrelated worker remains untouched. No instance or storage is retained
from this attempt. The corrected FireRed and XCodec2 code results are still
pending; all 194 public rows and independent real-weight/Apple/publication
completion requirements remain unchanged.

### 2026-10-02 corrected bootstrap and live code replay (18:27 UTC)

Root reviewed the distinct generic V4 controller at SHA-256
`a00da7ce62a92ed832565c0f6d8832271abae4be4907f2da06998de966b1eb2f`.
It preserves the verified release digests while following at most three
HTTPS-only redirects. Its tiny fixtures exercise the actual extracted
bootstrap branches, require the redirect contract separately for uv, deny
and audit, and reject each asset's missing-follow, wrong-protocol and
wrong-limit mutations. Fixture installation uses a task-specific root,
without repurposing `HOME`. Root's Bash syntax, ShellCheck and complete
offline suite passed in session `4062`; the frozen digest remained unchanged.
These are mechanism tests, not real model or Rust results.

Root independently recomputed the retained official release API asset tag,
URL, size and digest for uv, cargo-deny and cargo-audit, the uv/deny checksum
sidecars, and the rustup checksum receipt. The receipt JSON SHA-256 is
`150789a505c47860f390caf3f93ab5032eea2b6827276d3e1e404ddb472546a9`.
No archive or executable was downloaded or executed on the maintainer Mac.
The earlier V2 failure and V3 candidate remain immutable historical evidence.

Production session `98581` created exactly one disposable worker `53899353`,
label `vokra-clean-heads-model-free-v4-20261003-20261002T182131Z-92897`.
Independent API readback session `78463` confirms running status, the exact
ID/label, 16 effective allocated cores, 200-GB disk and USD
`0.17407407407407408` per hour, below the USD 0.20 cap. The real bootstrap log
now reports matching rustup, uv, cargo-deny and cargo-audit checksums and a
completed Python 3.12 installation. Read-only SSH session `28475` confirms
model HEAD `7b6b49cd192b31f392350922c8765e8e0a1234eb`, XCodec2 HEAD
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, `allocated_cores=16`, `jobs=16`,
toolchain exit zero and a progressing workspace test log. An earlier
read-only probe failed its endpoint type preflight; it did not run SSH,
restart the controller or establish a worker failure.

The actual all-target/all-feature workspace tests and Clippy, deny/audit,
XCodec2's 32-case gate, complete packet recovery and storage-inclusive
destruction are still pending. The controller retains bounded cleanup and
exact-ID/label ownership checks. No checkpoint, tokenizer, preset or HF
credential was transferred; no upload or Scaleway allocation occurred.
Separately, the published PR #152 check readback in session `73014` reports
76 successes, three skips and no other states. It does not validate the
unpublished correction candidates.

Root also independently ran the real pinned archive leaf's 14 offline cases
and the archive-license helper's 24 offline cases, both with terminal exit
zero and unchanged SHA-256 values `eed0d883ae2569ee67ea78896cde7a89beeddde33356376cb9aadb3e1c36a03c`
and `7f3b8d524b2fd6df1841d79fbd06663e635a2baab506611275e6fc9342550fa1`.
These synthetic tests do not establish actual wheel collection or installed
dependency approval. The archive lifecycle controller remains under separate
readiness/schema review; its dirty hook is not included in this change.

The retained ESPnet primary `LICENSE` was independently recomputed: 11,372
bytes, SHA-256 `4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5`,
Git blob `6afea95e3feee6ed39ecd4840662576c1f513173`, matching API-decoded bytes
and fixed revision `cccc29023d43a3f504e28df7d1324bb4eb6daedd`. API receipt
SHA-256 is `c72ca5e36acda11ae74a7211a1db50e63fe504eff27aa795d0f32ca944c86466`.
This authenticates Apache-2.0 source facts, not weights, dependencies or an
owner publication decision. A bounded native OWSM Conv2d8/absolute-position
stem is delegated with existing CPU/Metal Compute seams and no silent CPU
fallback. Its inspector correction and native diff are not accepted yet;
full encoder/decoder/tokenizer, genuine independent real-weight CPU and
final Apple verification remain unfinished.

All 194 public rows and the metadata-only 136 full / 58 unresolved split
remain intact. No row is promoted by bootstrap, synthetic tests or a live
code replay, and the full completion audit below remains required.

### 2026-10-02 terminal code replay, reviewed stem and archive transfer boundary (18:47 UTC)

The existing V4 production session `98581` is terminal with exit one, not a
successful workspace replay. Its exact model HEAD was
`7b6b49cd192b31f392350922c8765e8e0a1234eb` and separate XCodec2 HEAD was
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`. Workspace tests exited `101`,
stopping at the FireRed consumer binary: 37 passed, four failed and three
ignored in that binary. The log contains 160 reported result lines totaling
6,624 passed, four failed and 68 ignored; because execution stopped early,
these are partial replay counts, not a full workspace success or real-weight
parity verdict. Toolchain, workspace Clippy with warnings denied, cargo-deny,
cargo-audit and XCodec2's 32 tests all exited zero. The derived workspace
contract alone is nonzero.

Root independently verified all 22 required recovered files, their exact
allowlist, regular-file and size bounds, both HEADs and exit records. Total
packet bytes are `564699`; `collection-manifest.sha256` SHA-256 is
`557a39b813c509bed861378db192a1c49ff15ac56b8a51ab3ca5f0a4b01e9372`.
The controller recorded `cleanup_rc=0`. A fresh exact-ID readback confirms
worker `53899353` has explicit `instances: null`; a separate paginated
account readback confirms the exact label is absent, `next_token: null` and
zero Vokra instances. One unrelated worker was not modified. No retained
compute or storage from this replay remains.

Three FireRed failures originate from comparing source-observed raw BOOL
bytes, preserved as integer `0/1` by the Python producer, against a consumer
test's re-cast JSON boolean array. Root checked the committed producer's
`_tensor_values` and strict consumer binary-value gate. Reviewed local commit
`2ae2e4597d0e378f16abdefae389c1205946f57e` preserves raw integer observations
while retaining typed boolean state for derived EOS logic. It also aligns
the noncanonical-Base64 negative test's expected diagnostic with the actual
strict padding-bit rejection and adds explicit BOOL/UINT8/FP32 and byte-two
contract coverage. It changes one consumer test file only; SHA-256 is
`b2d4cdb2676e6a7579842e96d55cc91de0464862728c2aa6c77ea481dfc1168f`.
No producer, authenticated fixture, schema rejection or tolerance was relaxed.
Remote Rust results for this correction are pending.

Reviewed local OWSM commit `00daef0655c2c8ad5245d6e1fbed5e0dc84e4ea6`
adds the authenticated Conv2d8/absolute-position stem, its source inspector,
Apache source attribution and a bounded design record. Root reviewed native
GGUF-derived geometry, finite/overflow checks, actual three-stage synthetic
pipeline tests, channel/time layout, prefix/mask/position ordering and
Conv2d/ReLU/GEMM Compute seams. Metal selection explicitly rejects the
CPU-only PCM frontend instead of silently running it. The public weights
binder signature is preserved. Root independently passed the stdlib
inspector self-test and parsed the five roles from the retained authenticated
ESPnet source files without importing or executing upstream model code.
Conv2d8 and positional expressions are checked separately from structural
encoder/attention/CGMLP facts. Constructor-only defaults are not substituted
for the unresolved S2T task-kwargs or runtime attention-backend contracts.
Rust file SHA-256 is
`3f57c10aae6181a24355bd6324501cb9a9ade7d7d58a561edd9aa54b6a7f5085`;
inspector SHA-256 is
`c5e48068863904ea6ee914a672c77dfd4abffb3d9e59fd3910394c501b91d389`.
Both implementation commits have parent `7b6b49cd`; their distinct clean
integration and remote compilation remain pending. The complete encoder,
decoder, tokenizer and independent real-weight/Apple parity are unfinished.

The archive lifecycle V6 controller has frozen SHA-256
`1d81150ebe1b464233191a535e85f83d6ecaa00e421f8724f3ca683329d8c031`.
Root reviewed the minimal V5-to-V6 delta and independently passed Bash
syntax, ShellCheck and the complete offline suite (session `19587`, exit
zero, 25 lifecycle cases). Oversize, aggregate-size, symlink and missing
manifest cases now execute the actual transfer-preflight shell body;
refreshed endpoints recheck the exact instance ID, label and resource
contract before SSH. The exact hook pin and isolated fixture tests passed
twice independently in session `41859`; neither self-test altered the
production controller. All other local model/download paths remain blocked.

Actual archive collection was attempted only after that review. Auto-review
rejected the launch before process creation because explicit authorization
for the five-file external transfer was not recognized. A bounded read-only
input inspection established `358621` total bytes, unchanged fixed hashes,
public PyPI package/URL/hash metadata and audit code, and no matches for
the inspected secret-literal patterns; `.env`, keys and weights are excluded.
The evidence-backed re-review of the same operation was also rejected before
process creation. No instance was rented and no file was sent. The operation
is held for explicit five-file approval; do not bypass the rejection through
a wrapper, other runner or alternate transfer. Actual wheel collection,
installed dependency approval and full Realtime synthesis/parity are still
pending. This restriction does not justify loosening any license gate.

Separately, root removed five exact, clean, terminal synthetic-test clone
directories totaling `1791260` KiB (about 1.7 GiB). Their input bundles and
all small evidence were retained for reconstruction; no working candidate or
production packet was deleted. No local model execution, HF credential
transfer, artifact upload, PR mutation, merge or Scaleway allocation occurred.
All 194 public rows and the metadata-only 136 full / 58 unresolved split
remain unchanged. The full completion audit below still applies.

### 2026-10-02 clean integrated candidate and distinct code-only replay (18:54 UTC)

The separate clean integration is
`39b7570f3cccd89ccbb049c91077f635f7026c0f`, combining reviewed FireRed
`2ae2e459` and a local cherry-pick of OWSM `00daef06`. Root independently
confirmed the five-file scope and all five unchanged accepted file digests;
the input clones remain clean. This is not a push, merge or Rust/parity pass.

The new generic V5 controller has SHA-256
`a2d4c25c56d9f95000a4eff8ff40869e76b2cb8977dab3772a7f4fa55a78651b`.
Root reviewed its entire V4-to-V5 delta: only the name/label namespace and
fixed model clone/HEAD changed. The validated bootstrap, limits, security
and exact ownership/recovery/destruction behavior did not change. Root
passed Bash syntax and ShellCheck, checked both clean HEADs and retained
offline success/failure/cleanup evidence, and reviewed the implementer's
complete terminal-zero offline run in session `43592` (15 cases). Root did
not duplicate that unchanged full controller suite. V4's frozen digest
remains `a00da7ce62a92ed832565c0f6d8832271abae4be4907f2da06998de966b1eb2f`.

The separately authorized source-git-bundle code verification was accepted
by auto-review and started in session `65914`, with logs retained under
`/private/tmp/vokra-clean-heads-model-free-logs.EDCR8t`. It created only
worker `53902589`, label
`vokra-clean-heads-model-free-v5-20261003-20261002T185119Z-20122`.
Independent fresh API session `22615` confirms exact ID/label,
`actual_status=running`, `cur_state=running`, 16 effective cores, 200-GB disk and USD
`0.17407407407407408` per hour, below the USD 0.20 cap. The actual bootstrap
reached a completed Python 3.12 installation. The model bundle is fixed to
`39b7570f`; separate XCodec2 remains `e6552853`. Both bundle verification
messages were observed. Workspace tests/Clippy, deny/audit, XCodec2's 32
cases, complete packet recovery and storage-inclusive destruction are still
pending; the same bounded teardown is active. Poll the existing session;
do not restart it merely because observations yield without output.

This is source-bundle code verification only. It does not transfer the
denied archive collection's generated audit inputs/leaf, execute that
operation, install its target packages, or acquire a model. The five-file
archive approval boundary remains unchanged. No HF credential, checkpoint,
tokenizer, preset or public artifact is sent; no Scaleway allocation or
model-row promotion is inferred. The preceding zero-Vokra readback applies
to the completed V4 teardown, not this deliberately active V5 worker.
All 194 rows and every independent real-weight CPU, Apple CPU/Metal/
no-fallback, security and publication requirement remain unfinished unless
their exact earlier row-scoped evidence already proves them.

### 2026-10-02 terminal integrated replay and authenticated task source (19:10 UTC)

The source-bundle-only V5 job in session `65914` is terminal with exit one
at exact clean model HEAD `39b7570f3cccd89ccbb049c91077f635f7026c0f`
and separate XCodec2 HEAD `e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`.
Its retained packet is
`/private/tmp/vokra-clean-heads-model-free-logs.EDCR8t`.
Root independently verified the complete 22-file allowlist, regular-file
and size bounds, every SHA-256, both exact HEADs and all eleven exit-code
records. The collection-manifest SHA-256 is
`75c8b9b9c1bef524baa5e3fceae4dd3eca536501944ad3c2e67ff79932e84531`;
the recovered proof totals `563259` bytes. Collection completeness proves
packet recovery, not successful verification.

Workspace tests exit `101`, with derived workspace contract exit `1`.
The other nine exit records are zero, including Clippy, deny, audit and
XCodec2's actual 32-test run (`1.678` seconds, no skips). The workspace log
contains 160 result lines, totaling 6,636 passed, one failed and 68 ignored
before Cargo stops. Those are partial-run totals, not full workspace counts.
FireRed's binary has 41 passed, one failed and three ignored. The new BOOL
observation test and the three previously failing producer-consumer cases
now pass. The remaining
`producer_fixture_rejects_resealed_event_and_projection_mutations` case
correctly rejects the resealed `mask-projection-row` value `9`, but the exact
projection equality assertion lacks the expected `mask step` diagnostic.
Keep the rejection and exact comparison; a bounded contextual-diagnostic
correction is delegated rather than accepting the mutation or weakening
the negative-test assertion. The new OWSM stem tests, including the complete
tiny-weight stem pipeline, pass in the actual remote Rust run. This is
synthetic code evidence, not independent upstream/model-weight parity.

The lifecycle reports `cleanup_rc=0 instance=53902589`. Root's separate
fresh API readback in session `17405` confirms explicit `instances: null`
for that ID, followed by a complete `instances-v1 --all` response with
`next_token: null` and the exact ID/label absent. The account has zero Vokra
workers and one unrelated worker, which was not changed. The owned worker's
storage was destroyed; no idle verification worker is retained. This
supersedes the preceding V5-live snapshot, not its historical observations.

OWSM's constructor/task investigation also gained two bounded primary-source
receipt packets at the same ESPnet revision
`cccc29023d43a3f504e28df7d1324bb4eb6daedd`:

- `/private/tmp/vokra-owsm-task-source-receipts-20261003` contains five source
  files totaling `150081` raw bytes and `212204` API-response bytes. Its
  manifest SHA-256 is
  `5cb879ed339e209aa3e9a1c9f3d3dc01b72bd45da18da87a3f33bbd7388b3e00`.
- `/private/tmp/vokra-owsm-s2t-model-source-receipts-20261003` contains six
  source files totaling `35250` raw bytes and `54797` API-response bytes.
  Its manifest SHA-256 is
  `dd905cd30e35a787b7d94ba7be78010a1bcdb0727fb2c967bc30b1d16aae7ee3`.

For all eleven files, root independently checked exact fixed-revision URLs,
paths, sizes, SHA-256, Git blob SHA-1, retained API JSON and strict Base64
decoding equality with raw source, then parsed AST without executing the
upstream code. The authenticated S2T task forwards actual `args.*_conf`
kwargs; absent model config bytes still prevent runtime-effective resolution.
The S2T model source authenticates frontend/normalization/encoder ordering,
optional inter-CTC conditioning and token interfaces. Its previous-text
`<sop>/<sos>` preparation is the training attention-loss path, not a proved
inference beam-search route. Config, tokenizer payload, full native
encoder/decoder/transcription and independent real-weight CPU/Apple parity
remain unresolved. A separate bounded inference-source investigation is
ongoing, not a completion claim.

The archive five-input external transfer remains held; no approval was
inferred from this goal continuation and no archive worker was created.
No local model execution, HF credential transfer, artifact upload, PR
mutation, merge or Scaleway allocation occurred. Preserve all 194 public
rows, the metadata-only 136 full / 58 unresolved split and the final audit.

### 2026-10-02 reviewed diagnostic replay and complete security readback (19:25 UTC)

Root reviewed the complete FireRed diagnostic-only diff in isolated clean
candidate `8f98e110dc61864a8b0f002ce10486825ecb6da8`, following commits
`070a8758` and `8f98e110`. Every mask/lineage/KV comparison remains exact;
no mutated input is accepted and no negative test or parity bound is relaxed.
Root caught the later `lineage step` diagnostic mismatch before renting,
returned it for correction, and independently passed fmt, forbidden-symbols,
zero-deps and diff checks (session `29562`, exit zero). Rust tests remain
remote-only. The consumer file SHA-256 is
`9053ed2413cd0ce1af85e19adf08595e06c86237770083fcab8935f6e7624212`.

V7 controller SHA-256 is
`9e43114241ae0149ca06b37cd4a7c5e9c15598ea7ee49d79a097ba78283cc1d6`.
Root reviewed the full mechanical retarget, Bash syntax, ShellCheck and the
implementer's terminal-zero 15-case offline run (session `71178`); root did
not repeat the unchanged full lifecycle suite. V6's frozen controller digest
is unchanged. The separately authorized source-git-bundle replay started in
session `59021`, logs
`/private/tmp/vokra-clean-heads-model-free-logs.qyG9Ea`, owned worker
`53905822`, exact label
`vokra-clean-heads-model-free-v7-20261003-20261002T191742Z-58723`.
Fresh API and read-only SSH in session `12948` independently confirm running
status, exact model/XCodec2 HEADs, 28 allocated/build cores, 200-GB disk and
USD `0.1488888888888889` per hour, below the unchanged USD 0.20 cap.
Toolchain setup exits zero and workspace compilation has begun; terminal
gates, full evidence recovery and storage-inclusive destruction are pending.
Poll the same job; do not restart after an observation timeout. This is not
the denied archive transfer or a model/reference/Apple run.

Two further bounded ESPnet packets authenticate inference/beam/scorer/loading
source, without executing upstream code:
`/private/tmp/vokra-owsm-s2t-inference-source-receipts-20261003`, manifest
`59ea644715e0f8fadb4a831929332da4c1334bf0bc484d6d2eaafae7a062c4a0`,
and `/private/tmp/vokra-owsm-inference-scorer-source-receipts-20261003`,
manifest `9f061daaa45740476718300581043188d981c38c591bc0a733d2a1bffffb0d81`.
Root independently verified all ten API/raw/AST identities and size bounds.
Inference uses `[sos, language, task]`, appends `notime` only when timestamp
prediction is disabled, and conditionally prepends valid previous text.
These source facts preserve the earlier authenticated 1,172-tensor structural
binder; they do not resolve effective model configuration, payload execution,
tokenizer identity, complete native inference or independent CPU/Apple parity.

The fresh complete GitHub alert readback retains three pages of 100/100/9
records, exact HTTP/Link receipts and bounded raw JSON under
`/private/tmp/vokra-security-critical-readback-20261003`. Root independently
recomputed 209 unique open alerts: three critical, 31 high, 82 medium and
93 low; 182 have patched-version metadata. The decoded JSON SHA-256 is
`fc3bfbeb2ac0ed26c137a9383f6a41b55b9c922b92f903a2548e7f210fedff2a`.
This is a current observation, not alert closure. CosyVoice3 remains an
inventory-only, no-lock, forbidden-soxr composite route; available patched
versions do not establish a compatible, approved reference environment.
No dependency is installed and no approval is inferred from metadata.

The implementer removed only ten exact, clean, terminal V6/V7 synthetic
work directories totaling `3458160` KiB (about 3.30 GiB), retaining input
bundles and every small proof for reconstruction. Production packets, frozen
input candidates and the active job were not touched. No local model work,
HF credential transfer/upload, PR mutation, merge or Scaleway allocation
occurred. The archive five-file approval is still outstanding. All 194 rows
and the metadata-only 136 full / 58 unresolved split remain unchanged.

### 2026-10-02 terminal V7 and reviewed V8 replay (19:46 UTC)

V7 session `59021` is terminal with exit one. At exact model head
`8f98e110dc61864a8b0f002ce10486825ecb6da8` and XCodec2 head
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, workspace tests exit 101
and the workspace contract exits one; the other nine exit records are zero.
The incomplete workspace log has 160 result lines totaling 6,636 passed,
one failed and 68 ignored, not a full-workspace success count. FireRed's
binary reports 41 passed / one failed / three ignored. The mask and pruned
parent mutations now reach the intended rejection checks, but the later
test setup incorrectly treats the `lineage_projection` object as an array.
It panics before the rejection helper can inspect that mutation.

Root independently verified all 22 recovered files, exact HEADs and size
bounds under `/private/tmp/vokra-clean-heads-model-free-logs.qyG9Ea`:
563,413 proof bytes, manifest SHA-256
`984b779fdbf4824f191b4cefd558d2d9af968a451b19d6f00c419163926f1f2a`.
Clippy, deny/audit and XCodec2 remain green only in their recorded scopes.
Controller destruction is corroborated by a fresh individual query returning
explicit `instances: null` and a complete paginated inventory with the exact
ID/label absent. No Vokra worker remained at that deletion snapshot; the
single unrelated worker was untouched.

The bounded correction at clean commit
`4edc870fbce14ead1f6b90f4183a6bea71bfbcda` changes only two negative-test
setups to `lineage_projection.per_step_integer_lineage[]`, matching the
authenticated schema and the preceding pruned-parent case. Root reviewed
the full diff and independently passed fmt, forbidden-symbols, zero-deps and
diff checks (session `83498`, exit zero). No comparison, rejection condition,
expected reason or parity bound is relaxed. Consumer SHA-256:
`6b95555bf49fe5e4b6c5918cc3c4c022db071cd4ea5d42dacf20ee35a8cd4ced`.

V8 controller SHA-256 is
`24e2e0e4fcba5f6cc4a9bb2bf8615f03217ef9d96a26d52f00cbe8614700b194`.
Root reviewed its complete mechanical retarget and passed Bash syntax and
ShellCheck; lifecycle and remote gate logic are unchanged from V7's accepted
15-case offline suite, which was not duplicated. The distinct code-only run
is live in session `4952`, logs
`/private/tmp/vokra-clean-heads-model-free-logs.vYHbgD`, owned worker
`53908796`, exact label
`vokra-clean-heads-model-free-v8-20261003-20261002T194226Z-37619`.
Fresh API/read-only SSH (session `44095`, exit zero) confirms running status,
both exact HEADs, 28 allocated/build cores, 200-GB disk and USD
`0.1488888888888889` per hour. Toolchain setup exits zero and Rust tests
are progressing. Terminal gates, proof recovery and destruction are pending;
poll the same process rather than restarting after an observation timeout.

The fresh actual HF metadata/card-only audit (session `39881`, exit zero)
reconfirms 194 public repositories, 193 GGUF-bearing repositories and 198
GGUF files: CPU 136 full / 43 partial / 14 no-runtime-binder / one
non-artifact; Metal 136 full / 57 blocked-by-CPU / one non-artifact. All
repository IDs are unique and their revisions are fixed 40-character hashes.
TSV SHA-256 is
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
The 58 unresolved rows remain; metadata `full` is not Apple parity.

CosyVoice3's isolated declaration candidate is still being reviewed. Its
three candidate pins were reproduced with offline `uv add --frozen`, without
resolution, installation, a lock or virtual environment; the blocked
composite and unresolved Torch/Torchaudio pair remain explicit. Root's raw
PyPI readback identified provenance-field wording to correct before accepting
the candidate. No security-alert closure, license approval, API compatibility
or complete reference environment is claimed. The separate five-input
archive transfer is still held. No local model execution, HF credential
transfer/upload, PR mutation, merge or Scaleway allocation occurred.

### 2026-10-02 live V8 progress and family security review (20:02 UTC)

The same V8 session `4952` remains live; do not restart it merely because
polls have no new controller output. A fresh independent API/read-only SSH
observation (session `20897`, exit zero) confirms owned worker `53908796`,
the exact V8 label, running state and 28 allocated cores. The previously
failing `producer_fixture_rejects_resealed_event_and_projection_mutations`
test is now `ok`. The later workspace log is executing FFT/RFFT benchmarks;
only the toolchain exit is recovered in this live observation. Terminal
workspace, Clippy, deny/audit and XCodec2 verdicts, the complete checksummed
packet and independent destruction readback remain pending. This is code
verification, not independent real-weight or Apple parity.

CosyVoice3's reviewed isolated candidate is committed at clean
`4f389505db43b130776e335d64973671dd4a96d4`, based on public main
`97447185361a37af64c1b30fe87e8e2618d96e20`. Its sole changed file is
`tools/parity/cosyvoice3_reference/pyproject.toml` (SHA-256
`ab1fa2186d7d0c29f0cc9b294035a3a2651c361f6d2f5f7bd1d2a64711151d62`).
The ONNX 1.22.0, Diffusers 0.38.0 and ModelScope 1.27.0 declarations were
reproduced with offline `uv add --frozen --no-python-downloads --no-build`.
No dependency resolution, package installation, lock or virtual environment
was created. Root independently matched the retained official PyPI JSON
metadata, including raw license fields, expressions and capture hashes.
Primary license bytes and API compatibility are unreviewed; the forbidden
soxr composite closure and unresolved Torch/TorchAudio pair remain blocked.
No security alert is declared closed by this declaration-only commit.

The read-only XCodec2 review at exact `e6552853` found remaining execution
boundary defects: declared distribution/source identities are not fully
checked against actual installed payloads; decoder source is checked only
after its import; file hashing and document reads lack complete bounded
snapshot/ancestor-path guarantees; codes/output handling lacks a hardened
no-clobber boundary; payload paths and aggregate counts require stricter
validation; and an unverified locked-sdist build can pass the evidence
validator despite the documented block. A separate isolated hardening
candidate is being implemented. The current 32-test result does not prove
these uncovered properties, authorize an unverified sdist, or establish
independent CPU/Metal parity.

Fresh GitHub readback still has 19 open draft PRs. PR #152 is clean at
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, with 76 successful and three
skipped checks; those checks do not validate the unpushed `e6552853` delta.
PR #149's older PCM seam already exists on current main, while its old
branch is dirty relative to main. The FireRed integration is therefore being
reconstructed from reviewed FireRed-only paths on current main, not by
pushing the mixed 74-commit integration stack. Its fresh exact-head Rust
verification remains required before a PR update or merge.

OWSM's next native component is the source-authenticated CGMLP branch, using
existing Compute operations; it is not a completed E-Branchformer or ASR
route. Effective model flags, complete payload binding, independent reference
and real CPU/Apple results remain unproved. The separate five-file archive
transfer remains approval-held, without a worker or transferred inputs.
No local model execution, HF credential transfer/upload, PR mutation, merge
or Scaleway allocation occurred. Preserve all 194 rows and the metadata-only
136 full / 58 unresolved split; the current live V8 worker must be destroyed
after bounded evidence recovery.

### 2026-10-02 terminal V8 proof and family-only FireRed candidate (20:17 UTC)

V8 session `4952` is terminal with exit zero, not a process to resume or
restart. At exact clean model head
`4edc870fbce14ead1f6b90f4183a6bea71bfbcda` and XCodec2 head
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, all 11 raw/contract exit
records are zero. The full all-target/all-feature serial workspace test log
has 308 result lines totaling 8,289 passed, zero failed and 111 explicitly
ignored tests. Clippy with warnings denied, deny/audit and the separate
32-test XCodec2 stdlib suite pass. Explicit ignores and device-less feature
builds are not real-weight or Apple verification.

Root independently verified all 22 recovered regular-file checksums, per-file
and aggregate bounds, exact HEADs and both input bundle digests under
`/private/tmp/vokra-clean-heads-model-free-logs.vYHbgD`. Proof payload:
713,522 bytes; collection-manifest SHA-256:
`f2030d5091e63a5fcfe4cf86b568204edf1574698ed3fbe6f260031d46dc3ccf`.
The recorded allocation/build count is 28. Controller cleanup reports zero;
fresh independent individual and complete paginated API queries (session
`37687`, exit zero) prove worker `53908796` absent via explicit
`instances: null` and exact ID/label absence with `next_token: null`.
No Vokra worker remains in that account snapshot; the one unrelated worker
was not operated on. Storage was destroyed, not retained by stop.

The reviewed FireRed-only assembly is committed at clean
`a3fb0fc227b745e30f477f37f2cd264b5948b30a`, directly on main
`97447185361a37af64c1b30fe87e8e2618d96e20`. All 31 selected files are
byte-identical to the corresponding reviewed `4edc870f` files, independently
checked against both index and worktree. The only shared-file change is the
FireRed schema-fixture EOL pin; Cargo.lock and other model families are
unchanged. Root's fmt, forbidden-symbol, zero-dependency, EOL and diff checks
pass, and the isolated commit's normal hooks pass. The new family HEAD still
needs its own remote verification before push/PR update; V8's mixed-head
green result is not relabelled as an exact `a3fb0fc2` verdict.

Root's first additional Python binding invocation correctly failed closed
because `VOKRA_FIRERED_SOURCE_RECORD` was absent. After independently
verifying the retained fixed-revision record and supplying that explicit
environment, all 21 tests passed (session `78803`, exit zero). The 16,493-byte
GitHub record SHA-256 is
`11c96d58909bb2863e2e92b2f1420b4545d7e7b5a08021e8b90954210293475b`;
its 11,033 decoded source bytes match Git blob
`2088b0832b84da4421883e2dc7b518f734c3e0b2` at source revision
`834635e4cf277ed8ca92049fc375b17c3dc20748`. This authenticates source and
schema tests, not a real checkpoint reference or numerical parity.

OWSM's added LayerNorm receipt was independently matched across raw bytes,
API base64, fixed revision, Git blob and constructor AST: 958 raw bytes,
SHA-256 `5f2b40be03a3f345a6d86231ba38e3acf1f22eb28ec81d17474d908cb06ad8fd`.
The source explicitly uses `eps=1e-12`. Root review caught and requested
correction of per-row CGMLP splitting, eval dropout handling, public binding
gate preservation and source-inspector coverage. That candidate remains
uncommitted and uncompiled locally. XCodec2's additional hardening likewise
remains unaccepted after review found residual snapshot, output, sdist-proof,
payload-closure and test-coverage gaps. Neither addition is covered by V8.

No local model execution, HF credential transfer/upload, PR mutation, merge
or Scaleway allocation occurred. The separate five-input archive transfer
remains held without a worker. Preserve all 194 public rows, the last retained
metadata-only 136 full / 58 unresolved classification, and every independent
real-weight CPU, final Apple CPU/Metal/no-fallback and publication gate.

### 2026-10-02 independent corrective review and PR readback (20:26 UTC)

The root worktree started clean at `88ae2c36734f5e682863786221054d8dc71b890f`.
The prior V8 job is terminal; this review did not start another VAST worker.
Root independently ran the new uncommitted XCodec2 hardening candidate's full
stdlib discovery through offline Python 3.12 with `-S`: 39 tests, 38 passed
and one Linux-only sealed-memfd test explicitly skipped (command receipt
`c84125`, exit zero). This is not a Linux or real-model execution verdict.
Review still found that a nonempty executable-file map does not authenticate
the complete installed execution footprint, that the source-distribution
receipt field is not emitted by the current producer, and that bounded
growth, bad-code-before-import and output/source replacement behavior need
stronger implementation and tests. Corrections remain delegated; no candidate
commit or safe-execution approval is inferred from the passing suite.

Root's OWSM inspector self-test failed at the new LayerNorm positive fixture
(`951f63`, exit one). A separate source-only check (`7c7c7f`, exit zero)
independently matched the retained primary CGMLP and LayerNorm raw SHA-256
values. The current CGMLP AST checks accept the fixed primary source, while
the LayerNorm AST check rejects its actual nested transpose expression.
That is an inspector false rejection, not an upstream LayerNorm defect; fix
the source consumer and retain mutation rejection rather than altering the
authenticated source or weakening numerical bounds. Constructor/default-axis
and ordered-operation coverage are also requested. The partial native CGMLP
candidate remains unaccepted, uncommitted and not compiled locally.

Fresh read-only GitHub queries confirm main remains
`97447185361a37af64c1b30fe87e8e2618d96e20`. FireRed PR #149 is still
open/draft/dirty at `9a929c141a40cad524be4945135d3dee1cba3460`, based on
`09b39079a5b205e859ea24392ffe56b3bcd2371b`, with 76 successful and four
skipped checks. XCodec2 PR #152 is open/draft/clean at
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, based on the current main,
with 76 successful and three skipped checks. These verdicts do not cover
the fresh FireRed family HEAD or either new uncommitted correction.

The next three-family code-only controller is preparation work only: missing
reviewed OWSM/XCodec2 pins must prevent execution. No local model acquisition
or execution, external archive-input transfer, HF upload, PR mutation, merge
or Scaleway allocation occurred. The separate five-input archive transfer
remains held. Preserve all 194 public rows, the last retained metadata-only
136 full / 58 unresolved split, independent real-weight CPU and final Apple
CPU/Metal/no-fallback evidence, security closure and publication gates.

### 2026-10-02 reviewed OWSM family candidate and next source boundary (20:33 UTC)

The preceding OWSM inspector failure is corrected, not overwritten. Root's
full UV Python 3.12 `-S` self-test passed (`53ac93`, exit zero), and its separate
primary-source AST invocation (`744184`, exit zero) matched the fixed CGMLP
and LayerNorm raw SHA-256 values and passed all current checks. Ordered
projection/gating operations, explicit default-axis checks and negative
mutations are included; no source or numerical bound was relaxed. Root's fmt,
forbidden-symbol, first-party lock and diff checks also passed.

The reviewed partial slice is assembled and committed at clean
`12c32984b51e2d7bddc85c9becd4c5ca007bdc18` in
`/private/tmp/vokra-owsm-family-pr-20261003`, directly on main
`97447185361a37af64c1b30fe87e8e2618d96e20`. Root independently checked
the parent, exactly four paths, clean worktree and matching reviewed hashes
(`3fdc23`); its additional family-clone self-test passed (`1064bc`). The four
paths are the OWSM Rust module, inspector, design record and NOTICE; the main
delta is 2,132 insertions / 68 deletions. The required focused Rust module
has 18 source-defined tests and no ignored test. Those have not been compiled
or executed locally; the exact family HEAD still needs remote verification.
This is the bounded frontend/stem/CGMLP component, not full ASR or parity.

The next controller's full V8-to-family diff review caught invalid OWSM test
argv, a success-marker mismatch, missing family verification legs, stale
XCodec2 test counts, missing allocated/cgroup RAM caps and a pagination fixture
that stopped at pending pins rather than reaching the API. Bash syntax and
ShellCheck alone pass but cannot prove the requested behavior. The frozen V1
SHA-256 is
`1d5e2defe58f0e55dc33674df03248e84bdaab296fe5cd0ad552d52af5cc889d`;
it is unaccepted and must not run. A corrected inert V2 remains delegated.

Root independently ran the further uncommitted XCodec2 candidate's stdlib
suite: 44 tests, 43 passed and one explicit Linux-only skip (`05c2da`, exit
zero). Review still found that deriving an execution footprint from the same
possibly incomplete payload map does not prove completeness, that output
files are written directly rather than atomically, and that scope validation
and behavioral import-order coverage remain incomplete. Producer/archive/
installed-inventory reconciliation and atomic anchored output corrections
remain delegated. No commit, execution approval or real-weight result is
inferred from that suite.

Root also inspected the authenticated ESPnet attention source, raw SHA-256
`722f4499d555472df6355fa1c811f71f96f2d63adbbf5602acfc5344f20bcf23`.
Its Flash Attention branch is guarded by `self.training and
self.use_flash_attn`; eval reaches the ordinary Q/K/V, scaled-score and
masked-softmax path. This source fact guides a separate native attention
candidate and stricter AST coverage; it does not authenticate missing actual
task kwargs or enable full encoder/decoder/reference execution. Preserve the
frozen CGMLP family HEAD while implementing that next required component.

No replacement VAST worker, local model acquisition/execution, held external
archive-input transfer, HF upload, PR mutation, merge or Scaleway allocation
occurred. Keep all 194 public rows and the retained metadata-only 136 full /
58 unresolved split, with every independent real-weight CPU, final Apple
CPU/Metal/no-fallback, security and publication gate still accounted for.

### 2026-10-02 accepted XCodec2 family and corrective attention/controller review (20:51 UTC)

Root accepted the corrected XCodec2 model-free hardening after independent
source review and repeated offline Python 3.12 `-S` discovery. The final
hardening suite has 48 tests: 47 pass and the Linux-only sealed-memfd test
explicitly skips on the maintainer Mac (`db237a`, exit zero). The correction
is committed with normal hooks at clean
`72ffcf0f31d56975d1fc88bf56f4cc52e702e5d7`; it changes only five reviewed
XCodec2 files. The producer independently inventories executable ZIP members,
publisher RECORD and installed metadata; the consumer reconciles the current
bounded, unique, root-contained metadata inventory before imports. Python,
versioned shared libraries and extensionless ELF are included; metadata-only
empty executable sets are permitted only when reconciled. Generated bytecode
and selected sdists without reviewed installed-build proof remain blocked.
Anchored temporary writes, fsync and no-clobber publication preserve existing
outputs; invalid code values are tested through the actual entry point before
memfd allocation or third-party imports. This accepts preparation code, not
an execution approval, immutable whole-environment claim or numerical parity.

The reviewed family-only assembly is committed separately at clean
`99e6a157b47face677f66b61a6dad2ba40ff84ec`, directly on main
`97447185361a37af64c1b30fe87e8e2618d96e20`. Root checked exactly 16
XCodec2-only paths and byte equality with the reviewed hardening tree
(`f61d94`), then independently repeated the 48-test suite (`5b549a`, one
explicit Linux skip) and forbidden-symbol/first-party-lock/diff gates
(`ffa8ce`). Parent, committed scope and cleanliness were rechecked
(`d91586`). This does not carry unrelated SBV2 or e-Gov changes into the
family PR. Exact-head remote Linux execution must require 48 tests, no
failure, no skip and no expected failure; previous 32-test results do not
cover this candidate.

A bounded public HF metadata query independently matched XCodec2
`model.gguf`: 3,291,064,672 bytes, SHA-256
`7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e`, at
revision `2b6adcf787a8f9ec957b985c8c1664ba2007f7c2`. The 1,267-byte API
response SHA-256 is
`3bcd298c9fcc81d46bdb005ee2138b0bf3c307a0d546b14b773e2735efd658d9`.
This authenticates the public size/hash preflight without downloading GGUF
bytes; it is not an artifact validation or parity run. Primary endpoint:
<https://huggingface.co/api/models/vokra/xcodec2?blobs=true>.

The next OWSM attention candidate remains under review. Root caught invented
query-output zeroing absent from the authenticated eval source, weak projection
and mask assertions, and an incorrect source-byte count. The fixed primary
`attention.py` is 16,772 bytes, not 9,153; its previously authenticated SHA-256
remains unchanged. Corrections are delegated without altering source bytes or
numerical bounds. The accepted CGMLP family remains frozen at `12c32984` with
18 pending remote Rust tests; no later attention change is silently included.

For the OWSM reference dependency, public release metadata resolves
`torch-complex` v0.4.4 to the official `kamo-naoyuki/pytorch_complex` tag
commit `8a2ad1e47f3df25a30eb426f6ad781b89103fab3`. Its complete,
non-truncated 17-entry Git tree has no LICENSE/COPYING/NOTICE-named file;
the 4,158-byte tree response SHA-256 is
`8e6ecea37419763bbc85a55b7403aac79082675b04b600a84c0bf8f3124ee3a4`.
PyPI's Apache classifier is descriptive metadata, not retained primary
license text. No archive was acquired, installed or imported; archive/license
and owner review remain open. Primary metadata endpoints:
<https://pypi.org/pypi/torch-complex/0.4.4/json> and
<https://api.github.com/repos/kamo-naoyuki/pytorch_complex/git/trees/8a2ad1e47f3df25a30eb426f6ad781b89103fab3?recursive=1>.

The inert next VAST controller is not accepted. Root review additionally
found cgroup byte limits compared directly with KiB, a unittest success check
that only accepts fake fixed-duration output, and incomplete independent
source/recovered-proof authentication. Corrections and realistic offline
negative fixtures remain required before any rent. No worker was launched to
work around these failures or the separately held five-input archive transfer.

A fresh complete VAST readback at 20:51:32 UTC (`761b41`, exit zero) has
`next_token: null`, no Vokra instance and one unrelated instance left untouched.
The redacted response SHA-256 is
`7e370b8b2581a9a3f075a9b3c39d822b217617cfc9ed5828724098b1800c5b95`.
No local model acquisition/execution, HF upload, PR mutation, merge or
Scaleway allocation occurred. Preserve all 194 rows and the last retained
metadata-only 136 full / 58 unresolved classification, with independent
real-weight CPU, final Apple CPU/Metal/no-fallback, security and publication
gates still required.

### 2026-10-02 accepted OWSM attention and remaining proof gates (21:12 UTC)

Root accepted the bounded standard self-attention implementation after
independent source and diff review. The clean commit is
`387d1f79b3da45eba63ec5c93214a39817e6fad3`, parent
`12c32984b51e2d7bddc85c9becd4c5ca007bdc18`, in
`/private/tmp/vokra-owsm-attention-slice-20261003`. Its three-path delta is
927 insertions / 16 deletions: the OWSM Rust module, source inspector and
design record. Root rechecked the committed identity and clean worktree
(`e7be46`). The final commit passed all five normal pre-commit gates
(`eea297`); the temporary clone's initially missing hook configuration was
corrected by an immediate amendment using explicit `.githooks`, offline UV
and no dependency sync. This receipt applies to the final commit, not the
superseded pre-amendment object.

The authenticated primary attention source remains 16,772 bytes, SHA-256
`722f4499d555472df6355fa1c811f71f96f2d63adbbf5602acfc5344f20bcf23`.
Root's final primary-source probe passed 20 AST checks and rejected five
actual-source mutations affecting dtype, softmax input, head merge, output
projection and key transpose (`28cbeb`). The inspector self-test also passed
(`314eb0`). The attention component is authenticated, but the retained overall
packet remains `BLOCKED_SOURCE_SEMANTICS` with five other source roles missing.
Fmt, forbidden-symbol, first-party-lock and diff gates pass. The module now
defines 23 Rust tests and no ignored test; none has been compiled or executed
on the maintainer Mac. Require these 23 tests at this exact HEAD remotely.
Synthetic structural checks are not independent real-weight reference parity.
Full encoder/decoder integration, effective task configuration and the
reference dependency/license closure remain unfinished.

The next code-only VAST controller remains unaccepted. Review now requires
independent parsing and API/raw/record hash reconciliation of recovered proof,
failure-path log recovery, realistic negative cases for changed gates and the
new 23-test OWSM pin. Separately, the new XCodec2 derived-sdist proof candidate
still needs corrections for lock-snapshot identity, installed RECORD changes,
anchored no-follow paths and strict input parsing. Passing its initial small
test suite did not prove these properties. It does not change the frozen
`99e6a157` 48-test target, approve actual package builds or unlock execution.
The separately held five-input archive transfer remains held.

Fresh GitHub reads retain main
`97447185361a37af64c1b30fe87e8e2618d96e20` (`0580ff`). Draft PR #152
is mergeable at head `dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`;
draft PR #149 has a dirty merge state at head
`9a929c141a40cad524be4945135d3dee1cba3460` (`343244`, `2cc3c9`).
Those PR heads do not cover the new fixed family candidates. No push, PR
update, merge or CI restart occurred in this review.

The complete VAST readback at 21:05:56 UTC (`5342e9`) has `next_token: null`,
zero Vokra instances and one unrelated instance left untouched. Its redacted
response SHA-256 is
`9dbb112c795a2933946397200586fae0eb343afb7161002f5d70793103ceec24`.
Read-only offer research created no instance or storage. No local model
acquisition/execution, actual archive build/install, HF upload or Scaleway
allocation occurred. Preserve all 194 public rows and the last retained
metadata-only 136 full / 58 unresolved split. Independent real-weight CPU,
final Apple CPU/Metal/no-fallback, security and publication evidence remain
required; no row is promoted by this component acceptance.

### 2026-10-02 reviewed derived-sdist preparation helper (21:18 UTC)

The preceding XCodec2 preparation review has now closed its identified code
corrections. Root fully reviewed the changes and independently ran focused
9/9 tests (`b17463`) and the complete stdlib discovery (`ce9f8c`): 57 run,
56 passed, one explicit Linux-only sealed-memfd skip. Forbidden-symbol,
first-party-lock and diff gates pass. The exact three-file candidate is
committed at clean `38e5c0edbedcee5e3564b3e65df9e88f2eaec902`, parent
`99e6a157b47face677f66b61a6dad2ba40ff84ec`, with 767 insertions and no
other path changes (`0185be`). All five normal commit hooks passed (`93c268`)
offline and without syncing third-party dependencies.

The production API and CLI enforce the fixed source-artifact identities
using a single authenticated lock snapshot. The private synthetic seam is
separate. Installed source, metadata and RECORD tampering are rejected;
directory/entry/depth/byte bounds and anchored no-follow reads/writes prevent
unbounded traversal or symlink-based publication. The verifier SHA-256 is
`1c52cbb964b1500307a7417f32fd95ced09dd5bb77e5c243f3fd170cae522142`.
These source/synthetic checks prove the bounded preparation contract only.
No actual production archive, installer-generated mapping, package build,
upstream compatibility, independent reference or model parity was executed.
The output retains `UNAPPROVED_NO_EXECUTION`, `NO_UPLOAD` and unresolved owner
review; ANTLR script relocation and installer RECORD rewriting remain blocked.
The existing execution consumer is unchanged and still blocks selected sdists.

Keep this candidate separate from the current code-only controller's frozen
XCodec2 target `99e6a157` / 48 tests. OWSM remains `387d1f79` / 23 and FireRed
remains `a3fb0fc2`; controller acceptance and exact-head remote results are
pending. No new cloud resource, held archive-input transfer, model activity,
PR mutation or upload occurred. Preserve all 194 rows, metadata-only counts
and every real-weight CPU, final Apple/no-fallback and publication gate.

### 2026-10-02 production-lock counterexample and fresh catalog readback (21:30 UTC)

Root's source-packet review found a counterexample to the earlier synthetic
preparation coverage. A metadata-only invocation against the actual tracked
XCodec2 `uv.lock` (`66a94a`) produces `KeyError: 'hash'` in the new collector
and rejects the frozen preparation verifier before reaching a nonexistent
archive, with `lock sdist identity is malformed`. No archive or target package
was acquired, imported, built, installed or executed by that probe.

The actual UV sdist row uses `url`, `size`, prefixed `hash` and `upload-time`;
the builder's fixed identity uses `url`, `bytes` and unprefixed `sha256`.
Raw-dictionary comparisons or rejecting the known UV metadata cannot prove a
valid production path. A corrective child must normalize the asset identity
strictly, preserve the complete authenticated lock snapshot and SHA, and test
the actual tracked lock before any fetch/archive boundary. Wrong URL, size,
hash, unknown fields and malformed metadata must still fail closed. The
synthetic tests and commit `38e5c0ed` remain historical evidence, not a
production installed-build verdict. No execution consumer was unlocked.

The new source collector is preparation only and remains unaccepted. Its
public path must enforce VAST/platform and production pins before HTTP,
separate its private fixture seam, include every first-party producer
dependency identity, and distinguish source-archive network from model
activity. The current fixed `99e6a157` / 48-test code target remains unchanged.
The three-family controller also remains under corrective review: negatives
must test the intended gate rather than fail in BSD-incompatible fixture
mutation, and failure recovery must prove local recovered files, not merely
the fake remote files. No replacement worker was rented.

The fresh read-only HF API/model-card audit at 21:25 UTC (`ac5c9b`, exit zero)
still has 194 public repositories, 193 GGUF-bearing repositories and 198
GGUF files. CPU reachability is 136 full / 43 partial / 14 no-runtime-binder /
one non-artifact; Metal is 136 full / 57 blocked-by-CPU / one non-artifact.
The script's CPU-full/Metal-unsupported invariant passes. This audit downloads
no model payload and supplies no real-weight or Apple parity verdict. Preserve
every public row, source/license decision, full native and independent CPU
reference requirement, final Apple/no-fallback gate and publication boundary.

### 2026-10-02 corrected source packet and linked-worktree preflight (21:45 UTC)

Root accepted and committed the separate five-file XCodec2 correction at
clean `e9c7b8d054bb76f768ca4d13e129088e288544d3`, parent `38e5c0ed`.
The production verifier and new source-only collector normalize UV asset
identity against fixed builder pins, accept the known typed `upload-time`
metadata, and reject malformed or unknown fields. Actual tracked-lock
positives reach the archive boundary; wrong URL, size and SHA still reject
before acquisition. Public collection requires the VAST/Linux/CPython gate
and production pins before HTTP. Producer identities include the verifier;
source-archive transport is separate from always-false model activity.

Root's independent full stdlib run (`3bd33e`) has 72 tests: 71 passed and one
explicit Linux-only skip. The reviewed negative fixtures now parse as valid
TOML and assert their intended altered fields before gate checks, rather than
merely fail during fixture parsing. The normal five pre-commit hooks all
passed (`473f2f`); the exact five-file commit and clean readback are `b062f9`.
No actual source archive, build backend, installer, target-package import,
model weight, independent reference, parity, cloud or upload ran in this
source-preparation review. Existing execution consumers remain locked.

The separate three-family controller at `87f3131c` passed syntax, ShellCheck
and root's offline lifecycle suite (`1bf5a4`). Root independently verified
local recovered hashes and the intended source/API, named-test, skip and
cgroup mutations, with successful fake remote legs before local rejection;
the source-fetch-failure case recovered its local status, receipt and
workspace log (`65fae4`). This is offline controller evidence only.

The attempted actual-layout preflight (`090015`) then stopped with
`review-pending` before any safe-CLI/API or create call. All three pinned HEADs
are correct, but the clean XCodec2 tree is a valid linked Git worktree whose
`.git` is a regular file; the controller incorrectly requires a directory
(`8cb9bd`). Its corrective child must support authenticated clones and linked
worktrees without weakening exact-head/clean checks. Local controller evidence
`/private/tmp/vokra-clean-heads-model-free-logs.vnmGAo` records
`cleanup_rc=0 instance=none`, with no create request record (`f135d6`).
The fresh complete account readback at 21:42 UTC has no Vokra instance and one
unrelated instance (`ad3ba2`), which was not changed.

OWSM's next selected-layer candidate is separately unaccepted. Root's
authenticated-primary-source mutation probe (`67d5a8`) shows that wrong
pre-norm branch inputs and a changed macaron coefficient still pass its
new semantic checks; stronger dependency/order/scale checks and meaningful
binding negatives are required before freezing that child. Preserve every
public row, metadata-only counts and all native/independent real-weight CPU,
final Apple/no-fallback, owner/legal, security and publication gates.

### 2026-10-02 accepted selected layer and latest code-only targets (22:01 UTC)

Root accepted the bounded OWSM selected E-Branchformer composition and
committed its three reviewed files with all five normal pre-commit hooks.
The independently read-back clean candidate is
`15ec2f0d73e7d40202ca54b79933e76a09478a1c`, a child of the historical
`387d1f79` attention candidate. The Rust implementation binds 38 selected
layer tensors after global license/config/inventory checks and follows
source-ordered macaron FFN, attention/CGMLP branches, concat-plus-convolution
merge, main FFN and final normalization through existing Compute seams.
Unsupported paths remain explicit errors, not silent CPU fallback.

Root independently authenticated the primary layer source and Positionwise
FFN source, accepted their 29 and six AST checks, and rejected altered
attention/CGMLP norms, FFN scaling, branch copying and normalization order.
The final corrective review additionally found a missing-macaron-source
exception and a copy-before-macaron acceptance gap. Both were corrected;
independent primary-source mutations now return `BLOCKED_SOURCE_AST` rather
than an exception or a false acceptance. Inspector self-test, formatting,
diff hygiene, forbidden-symbols and first-party-only lock gates pass. The
28 Rust test definitions have not been compiled or executed locally.

The final reviewed file SHA-256 identities belong to the separate candidate
commit, not necessarily the versions in this management checkout:

| File | SHA-256 |
|---|---|
| `crates/vokra-models/src/owsm_v4_medium_1b.rs` | `c7fee8537d90f8ba2b1d805ff541d7a2293d01b252d610a75299a01804da6209` |
| `tools/parity/owsm_v4_medium_1b_inspect.py` | `b088c592cca58f4079f7338fbdc9142344d89d2d70d09d2a7b50f9b85594434b` |
| `docs/design/owsm-v4-medium-1b-source-contract.md` (candidate-only; does not exist in this checkout) | `4a7c87bf7c72ba1b99cd95fe9bd656e1b7bdd1b485f0589f77c9715cc0799e1a` |

Root also independently passed the corrected V4 controller's full offline
suite, including its actual linked-worktree fixture and instrumented invalid
repository preflight. This is controller evidence only, not a remote Cargo
verdict. A distinct V5 is now being prepared for the latest clean candidates:
FireRed `a3fb0fc2` unchanged, OWSM `15ec2f0d` / 28 focused tests, and XCodec2
`e9c7b8d0` / 72 stdlib tests with no permitted Linux skip. The new pins and
full controller suite still require root review before any rent. XCodec2's
actual source collector is not invoked by that code-only wave.

The separate proposed two-public-sdist source-only controller did not pass
root review: its clone/worktree gate, remote-to-local producer identity
mapping, cleanup ownership/resource separation, nested packet validation,
toolchain authentication and real-path offline lifecycle tests need
corrections. It remains inert and has made no live API, archive or cloud
request. No permission is inferred for the separately held five-input
transfer, installed builds, independent model execution or publication.

No replacement VAST or Scaleway worker was created by this review. All 194
public rows remain in scope; the last dated HF metadata snapshot remains
136 full / 58 unresolved and is not an Apple-completion count. Effective
S2T kwargs, full native encoder/decoder/tokenizer, independent real-weight
CPU parity, final Apple CPU/Metal/no-fallback and publication gates remain
unproved for this slice.

### 2026-10-02 live latest family HEAD replay and effective-config gap (22:14 UTC)

Root reviewed the immutable V5 controller diff: only its name/label and the
OWSM/XCodec2 candidate paths, exact HEADs and required test counts change
from the corrected V4 lifecycle. Syntax, ShellCheck and the independent full
offline suite pass. Controller SHA-256 is
`e1b7a555afe2c9f0428b8ccf7328c56e3569246c7d50946f1b810c04f010bbe9`.

The 22:03 UTC complete paginated account query had no Vokra worker and one
unrelated instance; response SHA-256 was
`0e281018d866f4064d5e3f5720da54ee7e5849932f5803bb0cd49f23c996a8c9`.
After the fresh capacity/price gate, the distinct code-only run created
`53926242`, exact label
`vokra-family-model-free-v5-20261003-20261002T220812Z-63522`.
Independent API readback at 22:10 UTC confirms that exact ID/label is running,
with 48 allocated CPU cores, 200-GB disk and reported storage-inclusive
`dph_total=0.19999999999999998`. The unrelated instance is outside this run.

Independent read-only SSH at 22:14 UTC confirms:

| Tree | Actual remote HEAD |
|---|---|
| FireRed | `a3fb0fc227b745e30f477f37f2cd264b5948b30a` |
| OWSM | `15ec2f0d73e7d40202ca54b79933e76a09478a1c` |
| XCodec2 | `e9c7b8d054bb76f768ca4d13e129088e288544d3` |

The remote resource summary has `allocated_cores=48`,
`allocated_ram_mb=515823`, `host_ram_kb=528202592`,
`cgroup_v2_limit_bytes=259621126144`, `effective_ram_kb=253536256`
and `jobs=48`. Cargo is live and workspace tests are progressing. These are
observations of this same process, not terminal test verdicts. Final focused
OWSM/XCodec2 gates, workspace/security results, recovered checksums and
destruction are still pending. The controller has bounded work and owned-ID
cleanup; independently confirm individual absence and complete paginated
ID/label absence after its terminal result. Do not restart a live job because
an observation command failed or timed out.

This wave transfers clean source bundles only, without an HF key or actual
model/config/tokenizer payloads. It does not run the separate source-archive
collector or the held five-input transfer. The source-only controller still
requires independent root acceptance before any separate rent.

The additional read-only OWSM source audit narrows the earliest unresolved
effective-kwargs fact: pinned `AbsTask.build_model_from_file` loads persisted
YAML into a Namespace, and `S2TTask.build_model` passes `args.encoder_conf`
directly to the selected encoder. Constructor/default-config-generation facts
therefore cannot authenticate the actual six effective flags without the
fixed target config's authenticated bytes. The retained metadata guards name
HF revision `e10985c8f1d592e905c24d2ac2b2c53e3feb24dc` and an advertised
494,398-byte config Git blob `fbf425c85d183f9103cb5e2c84ebffb0f425a930`;
those constants are not a live source/config readback. No config body was
downloaded locally or promoted into an effective runtime contract.

All 194 public rows remain accounted for. No real-weight CPU/reference,
Apple CPU/Metal/no-fallback, owner/legal or publication gate advances from
this live code verification. The metadata-only 136 full / 58 unresolved
snapshot remains distinct from actual Apple-complete rows.

### 2026-10-02 source controller review and full encoder scope (22:25 UTC)

Root polled the same controller session again; it is still live. A separate
read-only SSH observation finds Cargo and the workspace test log progressing,
not a terminal workspace verdict. Do not restart `53926242` or mutate its
three frozen input HEADs. Packet recovery and owned-resource destruction
remain pending.

The separate XCodec2 source-only V6 controller has SHA-256
`eb5a66dd31873f97d18897cb6d350ba2fc3f5ce840e3c0c0b2dc89a290eefe56`.
Root independently ran syntax, ShellCheck and its exposed offline self-test:
the 24 source-producer/consumer tests and fake controller lifecycle suite
terminate successfully. The previously absent missing-packet, wrong-remote-
exit, malformed-pagination, bounded-readiness-hang and controller INT/TERM
branches are now actually invoked. This does **not** accept V6 for live use:
the 30-second preliminary cleanup lane exceeds its eight-second test cleanup
budget; existing UV is only version-checked; fault cases do not yet prove
their intended diagnostic/phase; and complete-list destruction must check
both exact ID and label. A separate immutable corrective controller is
required. No live API, archive collector, installation or model execution was
performed by this offline test.

Root reauthenticated the fixed ESPnet E-Branchformer encoder source against
its retained API content and source receipt: revision
`cccc29023d43a3f504e28df7d1324bb4eb6daedd`, 19,432 bytes, Git blob
`1928fb98e4999064ebef4f0a7ee15e261c6c1815`, SHA-256
`adef32dd5ce8da01c5004d62c43b0b9036d2c455f0b6aac919eac7186c4d5b07`.
Its forward applies the encoder sequence before `after_norm` and computes
output lengths from the resulting mask. The next separately owned child of
accepted `15ec2f0d` targets all 18 native encoder layers and final
normalization, reusing the reviewed layer and stem with explicit caller
behavior. This is assigned work, not an accepted implementation or parity
verdict. Target-config authentication, decoder/tokenizer, independent
real-weight CPU parity and final Apple/no-fallback remain open.

In a non-overlapping child, the OWSM parser source-fact collector is being
prepared for exact locked PyYAML source/wheel identities and native-license
provenance. Synthetic stdlib tests do not approve acquiring or executing a
dependency, and primary project license text alone does not authenticate the
actual wheel/native closure. The separately held five-input transfer is not
retried or replaced by this preparation. All 194 public rows remain in
scope; the metadata-only 136 full / 58 unresolved split is unchanged.

### 2026-10-02 terminal family replay and production gate corrections (22:48 UTC)

The V5 code-only controller is terminal with exit 1, not an overall pass.
Root independently verified all 59 recovered manifest entries (793,671 bytes
total), the three frozen HEADs and the fixed FireRed API/raw/Git-blob source
identity. Workspace tests terminate successfully across 307 result lines:
8,239 passed, zero failed and 108 explicitly ignored. The OWSM selected-layer
leg passes all 28 focused tests with no ignore. Cargo deny and audit also
pass. These results remain code-only; no real model, target config, tokenizer,
dependency-archive collector or Apple run was started.

The failed gates identify four correction categories, not a reason to waive
verification:

- Rust 1.99.0 Clippy rejects the one-element candidate loop in the SBV2
  converter. Both workspace Clippy legs terminate with exit 101; the next
  isolated change must preserve the tensor lookup and fallback semantics.
- FireRed's generator leg uses a nonexistent test filename and runs zero
  tests (exit 5). Its actual generator/self-test entry must be identified and
  exercised before assigning a test-count contract.
- FireRed registry tests reject the controller's normalized six-field source
  record: the registry requires the complete fixed GitHub API record. The
  test also assumes a spaced size field. Correct the producer/fixture wiring
  without relaxing authenticated source checks.
- XCodec2 runs 72 stdlib tests but one errors because this Linux CPython build
  omits the sealing constants exported by `fcntl`. A separate child must
  preserve kernel-enforced immutable snapshots, verify installed seals and
  exercise the missing-export case; skipping production Linux verification
  or using a mutable snapshot is not acceptable.

The controller recovered failure logs and destroyed owned instance
`53926242`, including its storage. Root's fresh individual query returns
`instances: null`; the complete account query has `next_token: null`, neither
the exact ID nor label, zero Vokra instances and one unrelated instance that
was not modified. The SSH failure observed during readback was not used as
terminal evidence; the controller handle and recovered exit records establish
termination.

Root separately reviewed the new native OWSM 18-layer composition and final
normalization. Its inspector self-test, formatting and diff checks pass.
The authenticated 19,432-byte primary source passes the extended AST checks;
six mutations of that actual source (repeat destination/count, fast-path
arguments, final norm, lengths and ordinary return) are independently rejected.
The new Rust tests remain unexecuted and require a later exact-head VAST leg.
The configured git-only staging command was rejected before execution by the
local model guard. Root then inspected the authoritative current guard and
used its existing literal `git add` exemption, followed by a separate commit
with all five normal hooks enabled. The clean candidate is
`94a8341cad776159d65e3c5326685ec7770a368e`; no guard was changed, bypassed or
weakened. Its 30 source-defined Rust tests still require exact-head remote
execution, not the older selected-layer 28-test result.

The OWSM YAML helper's retained primary-license positive and synthetic source
tests now pass, but actual PyYAML archives, the wheel's LibYAML closure and
execution approval remain unproved. The source-only V7 controller's root
offline suite is terminal and green; V8 requires independent review and cannot
substitute for the failed production Linux XCodec2 gate. The separately held
five-input transfer was not retried. All 194 public rows remain in scope, with
the metadata-only 136 full / 58 unresolved snapshot unchanged. Independent
real-weight CPU, final Apple/no-fallback, security and separately authorized
publication/disposition gates remain open.

### 2026-10-02 source-only Linux replay and corrective source review (23:10 UTC)

Root independently reviewed the immutable XCodec2 V9 controller, SHA-256
`4115b7d1259904b72a8cdf42a771c6d89f585c3bae88902e15025cc2f6fbbdb8`,
and reproduced its full offline suite: 24 source-producer/consumer tests plus
the fake lifecycle, recovery, signal and authenticated-bootstrap cases pass.
The exact clean target remains
`a5e4c810f853c1d0c25d1df78635f115d7c7d153`. This did not approve dependency
execution or real model work. Two admission attempts stopped at the fresh
offer gate before any create call; their selected offer was absent from that
query's result. They were not failed or retained instances.

The subsequent admitted source-only replay used owned instance `53932071`,
label `vokra-xcodec2-source-audit-20261002T230553Z-94731`, with 200-GB disk
and observed total price `$0.07481481481481482/h`. The controller pinned and
verified the UV 0.12.5 archive before extraction, installed CPython 3.12.14,
and authenticated the fixed candidate HEAD. Its Linux stdlib leg ran all
76 tests in 4.473 seconds, without skips, but ended with three errors. All
three are `FileNotFoundError: uv` in subprocess CLI-negative tests: the
authenticated task-owned binary was invoked by absolute path but its
directory was not exported into PATH. The controller is terminal with exit
24, not a 76-test pass. The source-only collector was never reached; neither
of the two pinned PyPI sdists, installed builds nor any actual model payload
was acquired by that collector. A separate immutable V10 correction is
assigned to expose only the authenticated binary to subprocesses and to
recover verbose named-test evidence before the same scope is repeated.

The bounded failure log was recovered at
`/private/tmp/vokra-xcodec2-source-audit-v9c-evidence-20261003/logs/failure-stdlib.log`.
The controller destroyed `53932071`, including its storage. Root's independent
fresh individual readback returns `instances: null`; the complete account
readback has `next_token: null`, neither the exact ID nor label, zero Vokra
instances and one unrelated instance left untouched. No failed worker is a
restart target, and no storage is retained.

The isolated SBV2 one-element-loop correction is reviewed and committed at
clean FireRed child `893544169f6e87922c6760462a43b01978d4ac77`, directly on
`a3fb0fc2`. It preserves checked last-dimension conversion and the existing
`emb_g.weight` fallback. Root reproduced formatting, diff, zero-dependency
and forbidden-symbol checks and all five normal commit hooks. The identical
reviewed commit was cherry-picked into a separate OWSM child, clean
`a9007c643eaf4ce9ccda007ea82ddc3f9e959f23`, directly on the accepted
`94a8341c` full encoder. Its five normal commit gates also pass. No local
Rust compile/test or push occurred; the 30 source-defined OWSM tests and both
new Clippy HEADs still require remote verification.

Root rejected the prepared family V6 controller because its complete FireRed
source record still used compact JSON, while the actual registry mutation
test requires a spaced size field. A full-schema record alone therefore does
not close both earlier registry failures. A new V7 is assigned with the
production serialization, actual nine-test registry and four-test generator
executed against retained primary source, plus the corrected frozen HEADs;
no live family replay is authorized by the old V6 self-test alone.

The separate 18-layer ordinary OWSM decoder candidate is also unaccepted.
Root's full diff review found invented GELU behavior: the fixed upstream
`TransformerDecoder` constructs `PositionwiseFeedForward` without an
activation override, and that authenticated constructor uses ReLU. Its new
claimed AST proof actually used whole-file string positions, not role-local
method bodies. Corrections must use the actual retained primary sources,
semantic mutations and full 18-layer composition tests before review; tiny
attention tests and inventory checks are insufficient. A 5,000-position
implementation safety bound must not be described as authenticated target
configuration. No decoder compile, real-weight, independent reference or
Apple result is inferred.

All 194 public rows remain in scope. The last dated metadata-only 136 full /
58 unresolved snapshot is not an Apple-pass count and is not refreshed by
this code/source replay. The held five-input transfer was not retried, no HF
key was sent to this worker, no model or target config was run locally, and
no upload or withdrawal occurred. Source/license closure, independent
real-weight VAST CPU evidence, final Scaleway/no-fallback, security/CI and
separately authorized publication/disposition remain required.

### 2026-10-02 named Linux gate and new family replay (23:24 UTC)

The distinct XCodec2 V10 replay is terminal with controller exit 25.
The authenticated bootstrap and corrected task-owned UV PATH work: the
fixed `a5e4c810f853c1d0c25d1df78635f115d7c7d153` Linux suite passes all
76 tests in 5.197 seconds, with zero failures, errors or skips. Root
independently counted 76 named cases and 76 success markers in the verbose
recovered log, SHA-256
`79d9e1f9284fac7028fcdd1c4fbde70e0a40a1c31508f8c75fe93f6c02e8737d`.
The actual Linux UAPI/missing-Python-export, seal failure/FD closure,
incomplete installed-seal/FD closure and immutable memfd snapshot tests are
named and successful. This closes the failed model-free Linux sealing leg,
not real-weight reference or Apple parity.

The next collector phase stops before archive transport: `--lock uv.lock`
produces a noncanonical parent `.` and the sealed-snapshot gate refuses it.
Keep that gate; use the exact absolute lock path rather than weakening the
producer. No source archive packet or installed-build verdict was produced.
Owned worker `53933207` and its storage were destroyed. Root's fresh
individual API returns `instances: null`; its complete account API has
`next_token: null`, neither ID nor exact label, zero Vokra instances and one
unrelated instance untouched.

The corrective source-only V11 has SHA-256
`6bfdd4aa4fb93db7d2a620525b19631da3c13377d36f15bb745ee91c8f7eccd3`.
Root reviewed the exact diff and reproduced its full offline suite, including
the actual collector CLI with only the runtime gate mocked and an archive
boundary sentinel replacing transport. Relative input is rejected; the
actual pinned absolute lock reaches that sentinel. Syntax, ShellCheck, the
24 producer tests and fake lifecycle suite pass. V11 has not been rented,
and this mocked offline proof does not approve actual dependency execution.
The held five-input transfer is unchanged and was not retried.

Root also independently passes the retargeted family controller's full
offline suite, including the actual retained-source four-test generator and
nine-test registry. The immutable file is
`/private/tmp/vokra-family-model-free-vast-controller-v8-20261003.sh`, SHA-256
`eb9586f597e742a57996484e96c0504927daee0764f8f20c4568ae41d7fa7f99`;
its internal controller/label prefix remains V7, so identify it by this exact
file digest and allocation label, not the filename alone. The frozen inputs
are FireRed `893544169f6e87922c6760462a43b01978d4ac77`, OWSM
`a9007c643eaf4ce9ccda007ea82ddc3f9e959f23` (30 expected focused tests) and
XCodec2 `a5e4c810f853c1d0c25d1df78635f115d7c7d153` (76 no-skip tests).

The new code-only replay is live on owned `53934208`, label
`vokra-family-model-free-v7-20261003-20261002T232218Z-80331`. Root's direct
API readback confirms actual and intended status `running`, 48 effective
allocated cores, 200-GB disk and total price `$0.20/h` (API floating-point
representation is slightly lower). The local controller session is still
live; do not restart it or mutate its frozen inputs. Work is capped at
6,000 seconds with a separate 120-second cleanup lane. Terminal workspace,
Clippy, security and focused results, recovered checksums, and owned-ID/
label destruction are pending. This worker receives source bundles only,
not an HF key, actual model/config/tokenizer or source dependency archives.
The separate decoder candidate is not substituted into this running wave.

All 194 rows remain accounted for. No independent real-weight CPU or final
Scaleway/no-fallback result, owner/legal decision or publication status is
promoted by the Linux unit gate. Preserve the last dated metadata-only
136 full / 58 unresolved snapshot as metadata, not Apple completion.

### 2026-10-02 actual two-sdist inspection and decoder counterexamples (23:45 UTC)

The separately reviewed source-only V11 is terminal with controller exit 0
at fixed XCodec2 `a5e4c810f853c1d0c25d1df78635f115d7c7d153`. Its actual
Linux suite passes all 76 named tests in 5.601 seconds with zero failures,
errors or skips. Root independently checks the names, success markers and
summary; the 12,720-byte log has SHA-256
`0860a79cfc6eab7aab972deffbc4805c820e9210b4c1dd9921ec5822710dce60`.

Unlike the earlier relative-lock failure, this run actually acquires the two
fixed public source archives on VAST. ANTLR 4.9.3 is 117,034 bytes at SHA-256
`f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b`;
XCodec2 0.1.5 is 22,329 bytes at SHA-256
`dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b`.
Their retained member inventories have 68 and 28 entries respectively.
Neither contains primary LICENSE/NOTICE/COPYING members. The collector
therefore intentionally returns packet status 2 with `BLOCKED_FACTUAL`,
`UNAPPROVED_NO_EXECUTION`, owner review `REQUIRED` and `NO_UPLOAD`. A green
lifecycle controller means successful bounded collection and cleanup; it is
not a clean-license or model-execution verdict. Obtain exact release-linked
primary license/provenance evidence before any approved derived build or
installation; do not infer a grant from classifiers or another release.

The recovered 30,952-byte packet has SHA-256
`a3719c196cdf8a9a203c30d8df8d1b7121112d2652191b9cd4cae9059c3e8e12`.
Root independently verifies it against the exact local lock and all four
producer/dependency script byte counts and hashes, both artifact identities,
unique member counts, failure/status consistency and false model/import/
installation/build activity. Source-only network transport is recorded
separately as true. No actual package code or model was imported or executed,
and no HF key was transferred. The held five-input transfer was not retried.

To use the family job's waiting time, this short, independent source-only
scope ran concurrently on owned `53935735`, label
`vokra-xcodec2-source-audit-20261002T233701Z-39113`, with eight effective
cores and total price `$0.07481481481481482/h`. The controller work bound
remains 1,200 seconds plus an independent 120-second cleanup budget. The
worker and its storage were destroyed. Root's fresh individual readback is
`instances: null`; the complete account query has `next_token: null` and
neither its ID nor exact label. It retains only the in-use Vokra family
worker `53934208` and one unrelated instance, which was not modified.

The same family controller session remains live, with its three HEADs
unchanged. Read-only SSH confirms 48 build jobs and effective cgroup-bounded
RAM of 253,536,256 KiB; workspace tests are progressing. Its terminal
workspace/Clippy/security/focused gates, authenticated recovery and owned
destruction remain pending. Do not restart it because observation yields no
new output and do not substitute the next decoder candidate into it.

The separate, uncommitted OWSM decoder now has the correct 26 tensor roles
per layer and 40 source-defined focused Rust tests, not executed locally.
Root authenticates all three actual decoder source files, totaling 44,034
bytes, and reproduces the expanded primary-source mutation checks. Earlier
passing inspector self-tests nevertheless missed wrong constructor wiring,
residual operands and a cache-only attention call; root reproduced these
counterexamples and sent corrections back to the implementer. Ordinary-path
residual ordering and the new internal many-argument helpers still require
final review and remote Clippy/testing. This is source/self-consistency work,
not an independent numerical reference or accepted full-ASR runtime.

Fresh read-only PR results remain #149 open/draft/dirty at `9a929c14` with
76 successful and four skipped checks, and #152 open/draft/clean at
`dd6f0154` with 76 successful and three skipped checks. These results do not
cover the later isolated candidates and do not authorize a merge of them.

All 194 rows remain in scope. The last dated metadata-only 136 full / 58
unresolved snapshot is not an Apple-pass count. No real-weight parity,
owner/legal disposition, final Scaleway/no-fallback result, upload or
withdrawal is promoted by this source-only collection. An inert PyYAML
source-only controller is being prepared separately, with no archive
acquisition, installation, config parsing or new allocation authorized by
that preparation alone.

## 2026-10-03 terminal code-only family replay and decoder candidate (00:04 UTC)

The same V8 controller session terminates with exit zero; its internal V7
label remains unchanged. Exact code-only targets are FireRed
`893544169f6e87922c6760462a43b01978d4ac77`, full-encoder OWSM
`a9007c643eaf4ce9ccda007ea82ddc3f9e959f23` and XCodec2
`a5e4c810f853c1d0c25d1df78635f115d7c7d153`. Root independently verifies all
59 recovered file checksums, three HEADs, transferred bundle hashes and the
actual FireRed API/raw/record byte and Git-blob identity. Recovered evidence
totals 777,239 bytes; collection-manifest SHA-256 is
`a4bd60e6df0041f958e4d09aaf04d2dadd9fe3698ce3f66edab4f8c6fb8dbb7b`.
All 31 exit files are zero.

Workspace all-target/all-feature tests finish across 307 suites with 8,239
passed, zero failed and 108 explicit ignores. The 702,158-byte workspace log
has SHA-256
`4baf06a563dd77a717a584f1b88db3da01af732316bc1b4dd55efe34975af04e`.
Workspace and OWSM all-target/all-feature Clippy with warnings denied, deny
and audit finish with exit zero. OWSM's focused 30 named tests pass without
ignores. FireRed instrumentation/capture/PCM/binding/generator/registry pass
their required 12/33/6/21/4/9 counts; XCodec2 passes its required 76 tests
without skips. These are model-free source/structure and regression gates,
not real-weight reference or Apple parity. Explicit workspace ignores do not
become passed tests. The recorded configuration uses 48 build jobs within
253,536,256 KiB of effective cgroup-bounded RAM.

Owned `53934208`, exact label
`vokra-family-model-free-v7-20261003-20261002T232218Z-80331`, and its storage
are destroyed. The controller records `cleanup_rc=0`; root subsequently
obtains `instances: null` from the individual API and an independent
`show instances-v1 --all --raw` response with explicit `next_token: null`.
Neither exact ID nor label remains. There are zero Vokra instances and one
unrelated instance, which was not modified. Do not restart the old terminal
session or leave storage for an unspecified future task.

The separate ordinary-decoder correction is committed at
`3f963df624f0842709da7013d870af5fa022703f`; the clean integration candidate
adds only the reviewed SBV2 Clippy correction and is fixed at
`0c616382ebd000491b85bcf612b5b5ce1eb4075a`. Root reviews selected ordinary-path
residual ordering and typed helper inputs, reproduces the actual authenticated
three-role/44,034-byte source self-test with 19 rejected mutations, and runs
all five normal compile-free hooks on the integration candidate. It defines
40 focused Rust tests without ignores; compilation, Clippy and execution at
this new HEAD remain pending and cannot borrow the earlier 30-test verdict.
The tiny tests remain self-consistency, not an independent numerical oracle.

The next family controller remains inert during phase-specific fault review.
The separate PyYAML controller also remains unaccepted: actual focused
collector tests alone do not validate its production remote paths, packet
consumer or lifecycle. Correct and exercise those boundaries before any
separately reviewed source-only allocation. No new model/archive acquisition,
dependency installation, HF upload, withdrawal or Scaleway run occurs in this
closing review. The earlier two-sdist license-file absence remains factual
blocked evidence, not a conclusion about every possible upstream grant.
All 194 rows and the last dated metadata-only 136 full / 58 unresolved
snapshot remain in scope; no Apple-pass or full-model completion is promoted.

## 2026-10-03 combined candidate, primary receipt validation and live replay (00:32 UTC)

Root compares the two reviewed candidate histories rather than interpreting
their different baselines as authorization to delete FireRed work. The new
isolated merge is clean at `dc7afa5e4c6bf5057f2efb1a63902e339034a881`, tree
`8fea6db12bbebc7845c3ff907e1741affd5cab2c`, with parents
`893544169f6e87922c6760462a43b01978d4ac77` and
`0c616382ebd000491b85bcf612b5b5ce1eb4075a`. Only the four expected OWSM/NOTICE
paths differ from the FireRed parent; they match the accepted OWSM candidate
byte-for-byte. All other FireRed tracked content is preserved. Root passes
the normal five compile-free gates and the genuine three-role decoder-source
self-test. The 40 focused Rust tests remain self-consistency, not an
independent real-weight numerical oracle.

A new source-only supplemental helper is reviewed and committed separately
at `f28259c0abb5823498522b3f61dbb689f0c17441`, directly on XCodec2
`a5e4c810f853c1d0c25d1df78635f115d7c7d153`. It changes only the helper, its
tests and the XCodec2 parity README; runtime and approval state are unchanged.
Root first reproduces four invalid-input acceptances in the uncommitted
candidate (changed status, setup member hash, collector hash and duplicate
artifact), then verifies their fail-closed corrections. The final helper
pins the original 30,952-byte source packet and 143,165-byte lock, all four
producer identities, both unique artifacts, five primary receipt files and
the actual ANTLR setup-member linkage. It rejects duplicate/non-finite JSON,
symlink paths, oversized reads, FIFO inputs, missing tag proof and output
overwrite. Root passes all eight tests with the explicit retained inputs,
the normal five compile-free hooks and the same validator/writer through its
explicit genuine self-test. Its root-produced supplemental receipt is 3,259
bytes, SHA-256
`04d8b90cb4279aa6bba9ae1af477c9aec6392918b64d9d705da835b7b0c1ad16`.
Path-dependent receipt bytes need not equal another run's output.

The supplemental primary evidence binds the upstream
[ANTLR 4.9.3 license](https://github.com/antlr/antlr4/blob/4.9.3/LICENSE.txt)
and [Python setup source](https://github.com/antlr/antlr4/blob/4.9.3/runtime/Python3/setup.py)
to tag commit `e4c1a74c66bd5290364ea2b36c97cd724b247357`. The 527-byte setup
SHA-256 matches the inspected locked sdist. Retained API-text copies add one
terminal newline; raw license/setup bytes are unchanged and decoded content,
SHA-256 and Git blob identities are checked. This supplements, rather than
rewrites, the original package-license-file absence. Output remains
`REVIEW_REQUIRED` / `OWNER_REVIEW_REQUIRED` / `UNAPPROVED_NO_EXECUTION` /
`NO_UPLOAD`; the exact XCodec2 package-linked primary license, installed-build
proof and owner/legal decisions remain unresolved. No archive is reacquired
and no upstream package is imported, built or installed in this review.

Root's complete pre-rent VAST readback has explicit `next_token: null`, zero
Vokra instances and one unrelated instance left untouched. Root passes V12's
full offline lifecycle/fault suite with the genuine reviewed inspector hook,
then independently reviews V13's retarget-only diff, shell syntax and
ShellCheck. V13 SHA-256 is
`049e82fe46a5aacc27dbafb5053228e24c4f76dd8a778ffb0031875fc637cefd`;
protocol/body is unchanged. Root also separately runs the genuine inspector
on the final combined tree. The new owned worker is `53940991`, label
`vokra-family-model-free-v13-20261003-20261003T002634Z-9847`, at a verified
200-GB-disk-inclusive price of approximately USD 0.20/hour. Independent API
readback says running; independent SSH through the controller's already
trusted gateway host key confirms both combined HEADs and unchanged XCodec2
HEAD. Effective cgroup-bounded RAM is 253,536,256 KiB with 48 build jobs.

The same run is still live, not green. Compilation emits an unused-variable
warning for `hidden` in the new tiny decoder test helper, which would fail
warnings-denied Clippy. Root confirms the unused binding in source and
assigns a separate correction; do not retarget or restart the live worker.
Continue the remaining gates, recover authenticated small evidence and
destroy this owned worker and storage with independent readbacks. The new
PyYAML controller remains explicitly run-disabled and unaccepted pending
actual producer/validator and lifecycle proof. No actual model, HF credential
transfer, upload, withdrawal or Scaleway run occurs. All 194 rows, the last
dated metadata-only 136 full / 58 unresolved snapshot, real-weight CPU gates
and final Apple CPU/Metal/no-fallback scope remain unchanged.

## 2026-10-03 source-fault review and Kyutai composition boundary (00:49 UTC)

The review starts from clean management HEAD
`5153b7519d76793e15531f59090e6b22adf95321`. A fresh metadata-only Hub audit
still reports 194 public repositories, 193 GGUF repositories and 198 files:
CPU code/artifact classification is 136 full, 43 partial, 14 without a binder
and one non-artifact. Metal classification is 136 full, 57 CPU-blocked and
one non-artifact, with no CPU-only row. The fresh all-pages Dependabot read
still reports 209 open alerts, 182 with a named patched version: three
critical, 31 high, 82 medium and 93 low. These observations are neither
real-weight/hardware verdicts nor dependency repairs.

Root independently runs the PyYAML source-only controller V8 and V9 offline
suites to terminal exit zero (sessions `82022` and `81291`). V8 SHA-256 is
`dfadd9c49f32b812d6ed261b3b3754d915124ca86f7192fdb96a597b5ed4837b`;
V9 SHA-256 is
`2a237c8c125560c99fc500d3064e330a8cda2ffbe1a32f900f0c44468cfa0e46`.
Both authenticate the retained primary receipts, run the six collector tests
and consume synthetic output from the actual accepted collector through the
production packet validator. The shared create helper is exercised against
a fake provider, not a real API. V9 additionally proves timed-out creation
can recover a uniquely labelled owned instance using a separate finite
reserve after the work deadline, then destroy it and check both readbacks.
Foreign-only, duplicate-label and missing-pagination inputs remain rejected.

The V9 finish-status cases return 130/143 and invoke `finish`; they do not
deliver actual INT/TERM or execute the production signal traps. Root does
not accept them as signal-lifecycle proof and assigns that specific missing
test and possible partial-creation recovery correction to a new version.
Bounded regular-file/JSON input review also remains open. Both frozen versions
keep `--run` disabled at `PENDING_REVIEW`: no actual PyYAML archive is obtained,
installed, imported or approved by these tests.

Read-only Kyutai review confirms the existing native Mimi encoder and
`KyutaiSttAsr::forward_text_logits` are separate execution components.
`transcribe` still explicitly returns `NotImplemented`, and the wire-state
tracks frames/tokens rather than decoder KV cache or generation state. A
bounded PCM-to-Mimi-to-logits bridge is assigned in a separate candidate;
explicit backend selection and caller-supplied text inputs must not invent
initial-token, EOS, sampling or delay policy. Full transcription, independent
real-weight CPU parity and final Apple CPU/Metal/no-fallback remain required.
No implementation or public row is promoted by this investigation.

Root repeatedly polls the same live controller session `91310` and reads
the frozen worker's logs through its already trusted SSH host key. The
workspace all-target run has progressed into benchmarks; no terminal pass
is claimed. Worker `53940991` remains in use at the exact previously recorded
HEADs, not retargeted or replaced. Its terminal evidence recovery and
storage-inclusive destruction remain pending. No new paid allocation,
model execution, HF credential transfer/publication, PR mutation or Scaleway
run occurs in this checkpoint. The full 194-row goal remains unchanged.

## 2026-10-03 terminal combined replay, correction and generation source (01:12 UTC)

The management baseline is clean `20427439adeb7dea5eda9006b5f263f871494f3e`.
This checkpoint supersedes the earlier live-job observations, not their
historical evidence. Controller session `91310` is now terminal with exit
one. It preserved combined FireRed/OWSM
`dc7afa5e4c6bf5057f2efb1a63902e339034a881` and XCodec2
`a5e4c810f853c1d0c25d1df78635f115d7c7d153` throughout the run.

Root authenticates all 63 recovered files, totaling 805,875 bytes, against
collection manifest SHA-256
`75851cc38d1359c8b8e2509f697360f79713aa5c628acf9c880b2df49eaf2b96`.
The actual workspace totals are 307 result-bearing suites, 8,274 passed,
zero failed and 108 explicit ignores. OWSM's 40 named tests and XCodec2's
76 Linux stdlib tests pass without ignores/skips. All named FireRed test,
source-authentication and capture contracts, the three-role decoder-source
leg, cargo-deny and cargo-audit pass. This is code-only evidence, not an
independent real-weight reference or Apple verdict.

Workspace and OWSM Clippy both exit 101. The actual log identifies 24
missing public documentation items, one indexed mask loop and one unused
test-helper binding. Root reviews a separate correction at clean
`741ba4a213a07b927dd052f64dce6de52945bd31`: one file, 32 insertions and three
deletions, accurate component-only API documentation, equivalent mask
iteration after the existing exact-length check, and removal of the unused
binding. Normal commit hooks pass all five compile-free gates. This commit
was not substituted into V13 and has no remote compile/Clippy verdict yet.

After bounded evidence recovery, the controller destroys its exact owned
worker `53940991` and records `cleanup_rc=0`. Root makes additional fresh
read-only API calls: the individual response has `instances: null`; the
complete paginated response explicitly has `next_token: null`, neither
owned ID nor label, and zero Vokra-labelled instances. The one unrelated
account instance is not modified. No worker or storage is retained for this
wave, and no replacement is rented at this checkpoint.

Root also independently authenticates seven small primary-source roles at
Moshi revision `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`: `lm.py`,
`lm_utils.py`, `loaders.py`, `sampling.py`, `transformer.py`,
`LICENSE-APACHE` and `LICENSE-MIT`. All API-decoded bytes equal the raw
receipts, with matching SHA-256 and Git blob identities; Python roles pass
AST parsing only. Total raw size is 116,210 bytes; the receipt manifest
SHA-256 is
`9438a1c26fb9d89d03b4d8f4075e82df9212c94614f1afde1f13b643bcdf9da6`.
These source facts define initial embedding rows, delayed token-cache writes
and a distinct per-layer circular KV cache. They do not establish the
caller-level EOS/flush policy, an effective model config or actual reference
execution. A real stateful single-frame decoder component is assigned in
a separate candidate; the PCM bridge remains under corrective review.
Synthesized wiring tests are self-consistency only, never an independent
oracle. Public loading/transcription remains fail-closed.

The source-only PyYAML lifecycle review finds two real false-proof risks:
V10's default self-test exits before reaching its signal child, and V11
accepts a caller-controlled provider path plus that provider's own hash.
Those versions remain frozen and unaccepted. V12 pins the fake-provider body
independently and root reproduces its default offline suite to exit zero;
V13 additionally uses a dedicated execution marker and the default
provider's own read-only hash in negative tests; root reproduces its default
offline suite to terminal exit zero as well. Actual acquisition remains
disabled, and broader bounded regular-file/JSON input review is still open.
No actual PyYAML archive, dependency import/install, model payload, HF
credential transfer/publication, PR change or Scaleway execution occurs here.

The full 194-row completion objective is unchanged. The last dated live
metadata audit remains 136 code/artifact-full and 58 unresolved, not 136
hardware-complete models. Source/license decisions, strict native routes,
independent real-weight CPU parity, final Apple CPU/Metal/no-fallback and
separately authorized public-artifact reconciliation remain required.

## 2026-10-03 PCM integration, baseline correction and caller source (01:29 UTC)

The management baseline is clean `51bb0606642ca24562ce58ab36c555712db4475e`.
Root's fresh read-only GitHub main response remains
`97447185361a37af64c1b30fe87e8e2618d96e20`. The earlier proposed new decoder
was based on an older management checkout. Inspection of the combined
main-derived candidate instead finds the already merged `streaming_lm.rs`
(Git blob `e7f0c8dd5911bac7c48bf6c0083ccc098ba680eb`, from PR #187) and
crate-private `pcm_session.rs` (blob
`78f11b8e064d0d97154465a63a8a4508b8943da9`). The former already performs
per-layer KV, absolute-position RoPE, attention, FFN and logits. The isolated
duplicate draft is uncommitted and not adopted. Focused corrective work now
targets actual gaps in that existing implementation: argument failures
poisoning before mutation, a position-overflow check after layer mutation,
and missing allocation-capacity preflight. This corrects the earlier queue,
not the remaining real-weight/Apple requirements.

Root reviews the PCM bridge integrated onto OWSM correction `741ba4a2`,
preserving those existing components and Mimi backend/ELU changes. Normal
commit hooks pass all five compile-free gates, fixing the two-file candidate
at clean `3c15ea1d8adc20e8b1643698ae61f27b45bc0836`. File SHA-256 values are
`8deacd7f706b22981409747fed8009aa36d5471f412cccc016534bd6e4bebd90`
for `crates/vokra-models/src/kyutai_stt/mod.rs` and
`70a5fe7fa5653b8c54195ab2b68c36aa4336c7dfb84f742bd3b3df53843bd017`
for `crates/vokra-models/src/mimi/encoder.rs`. The bridge requires explicit
text tokens and complete finite PCM frames; it retains Mimi state, not
decoder history. Both learned backend capability sets are checked before
Mimi state allocation, and invalid token/shape/capacity inputs are checked
before encoding. The encoder's capacity guard derives from its actual bound
topology, including seven batch hidden-width scratch vectors and two FFN
scratch vectors. Existing public load/transcribe gates stay fail-closed.
The eight newly defined bridge/encoder tests have not run locally or remotely;
synthetic wiring is self-consistency, not an independent reference. The next
immutable controller and exact-head VAST replay remain pending.

Root separately authenticates the official
[DSM PyTorch STT caller](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_from_file_pytorch.py)
at revision `4c4f65e147df056adf3346290d64c7b9649b18c9`: 8,452 bytes,
SHA-256 `2ac2d9bff71d3d6a874bed070eb9d4e60736209e3cbaa9697fbe735dc79d2955`,
Git blob `cf3fb05b0e0c1f265a667276d2886ce2664d79ff`; decoded GitHub contents
response SHA-256 is
`b8f00e48832c4f41747c56cef46040f884afa81e632ba21d147ddd4bb9f7d2b5`.
The API-decoded bytes match the blob/size and pass AST parsing only. The caller
uses greedy generation, rounds an incomplete audio frame up with zeros,
prepends/appends the configured silence/delay frame counts and processes
the entire resulting chunk iterator. It filters text IDs zero and three for
piece display. EOS is considered separately in timestamp post-processing;
the generation loop does not stop early on EOS. This source evidence is not
reference execution, an effective model-config grant or license approval.
No upstream imports, model, config or tokenizer payloads are acquired.

Source-only PyYAML controller V15/V16 still fails root acceptance. The worker
reports a successful V16 offline suite after adding input snapshots, but root
finds active offer parsing, provider-output capture and hashing still not
bounded, and duplicate-key validation separated from a later parse. The
earlier initial INT-cleanup failure also needs retained diagnostic evidence,
not dismissal after a successful rerun. A new immutable correction is
assigned; `--run` remains disabled and no actual archive or dependency
execution is approved.

A fresh complete paginated VAST API readback explicitly has `next_token:
null`, zero Vokra instances and one unrelated account row, which is untouched.
No paid worker, storage, model execution, HF transfer/publication, push, PR
mutation or Scaleway allocation is started in this checkpoint. Preserve all
194 public rows: source/license, strict native/CLI paths, independent
real-weight CPU parity, final Apple CPU/Metal/no-fallback, security/CI and
separately authorized public-artifact reconciliation remain unproved where
their exact evidence is missing.

## 2026-10-03 streaming guard, PCM policy and offline cleanup review (01:41 UTC)

The management baseline is clean
`918f77c690b3afcb4f5ae30a6866c54af5297c0e`. Root rechecks the clean separate
streaming-LM guard candidate at
`deba8c6a537ce134be5343100185c51462d78820`, accepted through the normal five
compile-free commit gates. Its sole changed file is Kyutai's
`streaming_lm.rs` in that main-derived candidate, not a file present in this
older management checkout. Its SHA-256 is
`793fee60f92f522e5f5140863532b8da7a49b5f4db47d55b2ac5138e92861aec`.
The correction checks caller arguments and absolute-position overflow before
neural/cache mutation, preflights aggregate KV/layer/scratch allocation bytes,
and reserves bounded cache capacity. Partial neural failures still poison
state; reset remains explicit. Existing source initialization, cache ordering,
tracing and numerical bounds are preserved. Five new source-defined tests
have not run locally or remotely. Combining this correction with `3c15ea1d`
and obtaining a new immutable exact-head model-free replay remain pending.

Root inspects the current combined `3c15ea1d` source and confirms an important
reference-boundary mismatch. The legacy MLX-derived contract prepends 24,000
raw samples, appends 84,000 samples and floors complete 1,920-sample frames;
its empty-input schedule is 56 frames. Its private PCM control makes two LM
calls for the first Mimi row, hiding the first sampled result. In contrast,
the authenticated fixed DSM PyTorch caller recorded above rounds incomplete
input up to a whole frame and applies ceiling to silence-prefix and delay
frame counts, without an extra second of suffix. For the strict 1.0/2.5-second
configuration at 12.5 frames/second, this is 13 prefix and 32 suffix frames
(45 for empty input), with one LM call/output per encoded frame. Neither
schedule stops generation early on EOS. They must not be treated as the same
PCM policy or silently exchanged in historical evidence.

A separate source-bound PyTorch native engine/session route is assigned to
the implementation agent in an isolated candidate. The existing public
loader remains fail-closed. The current decoder reference tool executes
official `LMModel.forward_text` on explicit text/Mimi-code rows, not PCM
encoding or the stateful `LMGen` loop; its scope is
`KYUTAI_STT_DECODER_PARITY` and its streaming runtime status is
`BLOCKED_NOT_EXECUTED`. An independent full PCM oracle and real-weight VAST
measurement therefore remain separate requirements. Source authentication,
native synthetic wiring and decoder parity do not satisfy those requirements.

Root reviews source-only controller V19, SHA-256
`79308aee6a05b6e7266aa596aca5e9dd6b4f2a7d1ce39b12b33fc00d369fec44`,
and independently runs its default offline self-test to terminal exit zero
(session `14894`). Actual collector tests pass six cases; production packet
validation, bounded provider capture, partial owned-ID recovery, exact owned
destruction and signal cleanup tests pass. New negative cases reach the
actual cleanup path and reject duplicate `instances` keys in individual
readback and duplicate `next_token` keys in complete paginated readback,
while still destroying the fake owned instance. This is preparation-only
evidence: no provider, network, actual archive or model was used. It is not
Linux-specific zombie-reaping proof, nor diagnosis of the missing earlier
V15 failure trace. V19's `--run` remains disabled. A distinct two-artifact
PyYAML source-metadata proposal remains under review; previously rejected
five-archive transfers are not retried or rerouted.

No new paid VAST/Scaleway allocation, model execution, HF transfer/upload,
push or PR mutation occurs at this checkpoint. Preserve the full 194-row
objective and the dated 136 code/artifact-full / 58 unresolved classification;
no row gains real-weight CPU, Apple/no-fallback or publication completion
from these source and offline tests.

## 2026-10-03 integrated PyTorch PCM and source-audit readback (02:26 UTC)

The management baseline is clean `433030531342487ae3dccfa01f64d6ae396db692`.
Root reviews and normally commits the private PyTorch PCM candidate at
`729c85a9d783ccb89ac4a6bf856d8e1cd9be17fc`, then accepts the identical four-file
change on the clean streaming-guard baseline `a0ef540a`. The resulting clean
combined HEAD is `1cbc4abce1752f4a267f6ae024de5fdcbf0feb32`. Both normal commits
pass all five compile-free hooks; the integrated change has 190 fixture pins
and 306 shell files checked. The guard source digest remains
`793fee60f92f522e5f5140863532b8da7a49b5f4db47d55b2ac5138e92861aec`.

The actual private session preserves the historical MLX route and adds the
authenticated PyTorch caller's frame-ceiling policy, 13 prefix/32 suffix
frames, one LM call per row, visible first sampled result, and continued
processing after EOS. It consumes prepared mono 24 kHz PCM; file decoding,
resampling, VAD, independent reference and the public loader are not promoted.
The six new Rust regressions bring this PCM module to 24 defined tests. The
three new stdlib source-audit tests and retained raw/API audit also remain
unexecuted at the integrated HEAD; no local heavy Cargo or model run occurred.

Root's immutable code-only controller V16 passes its complete default offline
suite. Its first actual attempt terminates on local ENOSPC before remote
verification. Owned worker `53950549` is manually destroyed after exact label
authentication, then independently absent in individual and complete API
readbacks. Cleanup removes only redundant fake work/input copies from an old
terminated self-test, retaining its logs, output and destruction markers.
The old model HEAD remains recoverable in a separately verified retained
bundle. Local free space rises from about 370 MiB to 5.4 GiB; no active clone,
owner manifest, model artifact or unrelated account resource is removed.

The distinct replacement session `20301` remains live at this dated readback,
frozen at `a0ef540a2cd77838f6521898e08f72fd8dc353aa` for model/OWSM and
`a5e4c810f853c1d0c25d1df78635f115d7c7d153` for XCodec2. Its owned worker is
`53951140`, label `vokra-family-model-free-v15-20261003-20261003T020743Z-77962`;
the legacy label is distinct from the V16 controller version. The authenticated
resource record binds 16 build jobs and effective RAM 126,303,232 KiB.
Several zero-exit test, Clippy, source and dependency logs are already
recovered, but full checksum reconciliation, terminal verdict and destruction
remain pending. This job does not cover the newer integrated PCM HEAD.

Root independently passes source-only PyYAML V22's default offline suite.
Its actual run terminates before package acquisition because UV rejects
`UV_PYTHON_DOWNLOADS=allow`; worker `53951238` is destroyed and independently
absent. The minimal V23 correction uses `auto` only for authenticated runtime
bootstrap, then returns to offline/no-download invocation. Root's full V23
offline suite passes actual collector, production packet, provider-capture,
cleanup and activation-gate cases. This does not approve package execution.

The new V23 actual session `95373` terminates with exit 26. Authentic UV/Python
bootstrap and all six stdlib collector tests pass; the producer then reports
`Linux wheel filename is not the exact lock artifact name`. The controller
used a generic local wheel basename, which the unchanged strict producer
correctly refuses. No accepted source packet, native-license decision or
dependency execution is inferred. Worker `53952644` and storage are destroyed;
root's fresh individual readback is null and complete account pagination is
explicitly null. The remaining Vokra row is the active code worker above;
two unrelated account rows are untouched. A correction must retain the same
one-sdist/one-wheel URLs, hashes and source-only scope. Previously rejected
five-archive proposals are not retried.

Root also reviews the independent PCM capture draft against retained official
source. Its assumed `LMGen.state` field and unconditional emitted token do not
match `_streaming_state` and delayed `None` output in the fixed implementation.
The draft's schema-only tests are not proof of real caller wiring; corrections
also require source/dependency/import identity, effective schedule and bounded
capture validation. It remains unaccepted and uncommitted. No upstream import,
real-weight capture, HF credential transfer/upload, push, PR mutation or
Scaleway allocation occurs in this checkpoint. Preserve all 194 public rows,
the dated metadata-only 136 full/58 unresolved classification, independent
real-weight CPU evidence and final Apple CPU/Metal/no-fallback requirements.

## 2026-10-03 terminal code-only readback (02:30 UTC)

This supersedes the live-job status in the 02:26 record, not its frozen scope.
Session `20301` is terminal with exit zero at model/OWSM
`a0ef540a2cd77838f6521898e08f72fd8dc353aa` and XCodec2
`a5e4c810f853c1d0c25d1df78635f115d7c7d153`. Root independently verifies all
66 recovered manifest entries; manifest SHA-256 is
`40fb64c9a5c025f0ee9c683e092e291d0e281a1f44d7af6fe46d732bc672e885`.
Every one of the 35 recovered exit files is zero. Workspace tests report
307 suites, 8,287 passed, zero failed and 108 ignored. All 13 exact named
Kyutai bridge/streaming-guard tests and OWSM's 40 focused tests pass with
zero ignores; XCodec2 reports 76 tests and no skip. Both warning-denied Clippy
legs, source-contract legs, cargo-deny and cargo-audit pass.

Controller cleanup returns zero. Owned worker `53951140` and its 200-GiB
storage are destroyed; root's fresh individual API returns null and complete
account readback explicitly has `next_token: null`, no Vokra row and one
unrelated account row, left unchanged. Do not poll, restart or SSH this
retired worker. Model/source payloads are not kept as a cloud handoff.

Clean integrated `1cbc4abce1752f4a267f6ae024de5fdcbf0feb32` is a later candidate
and cannot inherit this exact-head pass. Its 24 PCM tests and three retained
source-audit tests still need remote execution. The separate controller draft
requires additional source/PCM fault cases before rent. The corrected PCM
capture draft also remains unaccepted pending root review; synthetic schema
or instrumentation tests do not establish an independent numerical reference.
No real-weight result, owner/legal approval, Apple/no-fallback verdict or
publication advances. All 194 public rows and the dated metadata-only
136 full/58 unresolved classification remain in scope.

## 2026-10-03 source-only audit readback (02:46 UTC)

Root reviews the minimal V23-to-V24 wheel-basename correction. Controller
SHA-256 is `cb564442c994663be4896331b2a49e8e596c8fe72a97d0ddd6ec3a530f81b857`;
its legacy embedded V23 identifier does not change the immutable V24 bytes.
The full offline suite passes collector tests, actual synthetic packet
production/validation, exact/generic basename cases, bounded provider capture,
owned cleanup, signal handling and pre-provider activation gates. An initial
root run fails because `UV_NO_SYNC=1` emits a warning into the signal PID
log; removing that root-supplied variable produces the complete green run.
No real upstream package or model is executed locally.

The distinct actual session `45321` terminates with exit zero at fixed
producer `e8b93b4bf1588b67f371ce5195184af3a9c5771b`. All six stdlib tests pass.
The recovered 225,307-byte source packet has SHA-256
`fff2c6be49300f2278511d2d70f19bce13a3d6f5c819d31f862fab867a507a34`.
Root independently checks the packet hash, both exact locked artifact
sizes/digests and both 1,101-byte primary PyYAML license members against
`8d3928f9dc4490fd635707cb88eb26bd764102a7282954307d3e5167a577e8a4`.
The wheel uses its complete locked basename; the prior filename failure is
resolved. Its observed native ELF is 2,679,264 bytes with digest
`957a099a4521c1f7669306fb6f79ca63fa4b9a0b4c463f7cac833a65a5c5c0cb`.
This does not establish that binary's full LibYAML source/build/license
closure. The packet retains `NATIVE_REVIEW_REQUIRED`, `UNAPPROVED`,
`NO_UPLOAD` and a factual owner-review blocker; no install, build, package
import, ESPnet/reference execution, model/config/tokenizer access or upload
is authorized or performed.

Owned worker `53954812`, label
`vokra-owsm-pyaml-source-audit-20261003T024128Z-72454`, and 200-GiB storage
are destroyed. Root's separate individual API returns `instances: null`;
complete account pagination has `next_token: null`, zero Vokra rows and two
unrelated rows, left unchanged. This retired worker is not a restart target.

Before the run, root removes only five clean, terminal synthetic work copies
and seven byte-identical input-bundle pairs from an old offline fixture.
All logs, outputs, destruction markers and both verified complete-history
bundles remain. That fixture shrinks from about 2.6 GiB to 111 MiB and local
free space rises from about 1.5 GiB to 3.8 GiB. These are reproducible fake
copies, not user work, model payloads or live cloud storage.

The next PCM oracle review confirms corrected sampling aliases, official
Mimi `cardinality` and fixed suppression IDs, but rejects the draft's
metadata-only dependency verification and missing positive full-capture
orchestration test. Corrections remain assigned; no reference or numerical
pass is inferred. A separate source-only collector for three fixed official
Moshi API files passes four stdlib tests, but its local acquisition command
is refused by the protection hook. It is not rerouted, renamed or executed
through a bypass; raw/API receipts remain uncollected pending the next
reviewed remote code-only job. New integrated `1cbc4abc` still requires its
own verification. All 194 public rows and the dated metadata-only 136 full /
58 unresolved classification remain; real-weight CPU and final Apple
CPU/Metal/no-fallback, legal/owner and publication gates stay separate.

## 2026-10-03 PCM oracle review (02:54 UTC)

Root reads the complete 1,296-line capture, 426-line test file and README.
The capture SHA-256 is
`b2f452da1bf262de5e7b1a247b63763ea6ef796a31d9622f65d7adb86394cbe1`;
the test SHA-256 is
`222cbbb82d7ea682d3695ef52faa43ccb9efbae40d65d1a58db396c9915f3a25`.
An independent offline UV/Python 3.12 stdlib run passes all 13 tests without
upstream imports, network or model execution. Actual dependency files are now
opened and hashed, an improvement over the previous metadata-only candidate.
This review does not accept the draft for execution.

The remaining concrete defects are pre-import admission of ignored bytecode
or extra source; lock rows with no selected distribution bypassing artifact
matching; incomplete RECORD entries and no binding between installed files
and the locked archive; a native disposition explicitly requiring owner
review being accepted for capture; and checking an existing output only after
the caller has run. Record-count and metadata/output budgets also need checks
before allocation. The documented `python -S` entrypoint does not add its
bound dependency environment to the import path, while the installed Moshi
origin requirement conflicts with the separate authenticated source checkout.
Lazy imports during the caller require origin validation as well.

The positive orchestration test mocks the source, dependency, artifact and
loaded-origin validators, so its success does not prove a complete accepted
filesystem path. Root returns these findings to the same bounded implementer,
requiring an actual small synthetic filesystem test and focused negatives;
real primary identities, license approvals and parity values must not be
invented. Scheduling-cache capture is still not transformer per-layer KV.
New `1cbc4abc` remote code verification and remote-only Mimi primary receipt
collection remain pending. No provider is allocated by this review, and no
model, Apple or publication verdict advances. All 194 rows remain in scope.

## 2026-10-03 PCM/KV preparation readback (03:15 UTC)

Root independently passes all 18 offline stdlib tests and normally commits
the reviewed source-only capture preparation at clean
`fc977f3b7637d7f3c8249cccf594d33971fee019`, based on `a0ef540a`.
Capture, tests and README SHA-256 values are respectively
`6ecfaad7e78ff3bbed184353cba40a9ff98490b781ddc5df212cde9ce15d3e2d`,
`462b87d755555dda5fc9ae37b47f42e3dd2673b0b014399f8951d6643e0ff92d`
and `1a65cbf3d576681ee6a256631f07a3b5ea2f21b01c8e402da06c94f81540bd95`.
The five compile-free commit hooks pass, including 190 fixture pins and
306 shell files. No push or real upstream execution occurs.

This supersedes the rejected preparation, not its execution verdict. The
candidate binds installed RECORD files to locked archive members, rejects
pending native-owner dispositions, preflights no-clobber output, bounds
archive iteration and capture budgets, and admits executable import origins
before their loaders run. Tests use tiny synthetic ZIP/ELF/filesystem bytes,
not acquired dependencies. The positive orchestration still substitutes the
fixed-source checkout validator and upstream numerical caller; its focused
source test uses a simulated Git readback. Neither is evidence of an actual
complete upstream checkout, dependency closure or numerical reference.

Separately, reviewed native observation is fixed at clean
`6c015bb9e2e2824e1411838a950bac83d6aba948`, based on `1cbc4abc`.
`KyutaiSttStreamingLmLayerKvView` borrows existing per-layer positions and
row-major key/value slices without changing arithmetic or eviction. Root
reviews the lazy first-frame fix and exact context-window/geometry guards;
all three source-defined Rust tests remain `NOT_RUN`. The existing Rust
public-API snapshot scans core/ops/capi, not models, so that snapshot is not
changed by this models-only view. The earlier fixed replay must not be
retargeted to either later candidate.

Retained official Moshi `transformer.py` at
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` has raw SHA-256
`f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622`
and API-body SHA-256
`be42048eda224e70264c1d91456d42fb81aece8bb18244e372d7948c689b911d`.
Its `RingKVCache` stores `(2, batch, KV heads, capacity, head width)`;
`complete()` returns the official keys, values and absolute positions.
Reset changes offsets without zeroing old tensor bytes. The independent tap
must identify main-transformer layer caches, preserve native dtype, and use
the returned positions to select valid chronological rows. Do not derive
reference values from Vokra or confuse this state with LMGen's scheduling
cache. Actual `kv_repeat`, dtype/context, streaming lifecycle, full primary
dependency/license/owner closure and real-weight parity remain unresolved.
No row advances, no new provider is allocated by this preparation, and all
194 public rows remain in scope. Final Apple/no-fallback and publication
remain separate gates.

## 2026-10-03 integrated candidate and live code replay (03:34 UTC)

Root independently completes V21's offline self-test in session `76959`
with exit zero, including its positive Mimi raw/API validation and the
missing-role, hash and identity negative cases. Controller SHA-256 is
`f754c533e95871e126684481dba6db121a016de4bc5efe54453da33130bcb83a`.
Root compares the generated tiny-fixture controller with that frozen file:
only the eight intended repository, HEAD and count substitutions differ.
Those synthetic provider/source outputs are not primary receipt acquisition
or real code/numerical evidence.

After fresh complete account readback has zero Vokra instances, the distinct
authorized code-only run starts at 03:25 UTC on owned worker `53959731`,
label `vokra-family-model-free-v21-20261003-20261003T032514Z-49548`.
Session `42297` remains live. The provider records 32 effective CPU cores,
257,589 MiB RAM, 200 GB disk and USD 0.17555555555555558/hour. Its fixed
model/OWSM HEAD is `1cbc4abce1752f4a267f6ae024de5fdcbf0feb32` and
XCodec2 HEAD is `a5e4c810f853c1d0c25d1df78635f115d7c7d153`.
The scope is model-free Rust/Python/static verification plus the exact three
fixed-revision Moshi Python raw/API receipts. No HF credential is transferred,
and no model, tokenizer, configuration, preset or package-reference workload
is authorized by this run. Terminal verdicts, recovered checksums and owned
worker/storage destruction are still pending. The earlier `a0ef540a` pass
cannot be transferred to this new HEAD.

Separately, implementation snapshot
`b068ab9abf5dd527b2240dbe8035deed8a525ac2`, parent `1cbc4abc`, combines
the already reviewed native KV observation and source-only PCM capture
preparation. Documentation correction
`858bfa464fb0b101b1d4f1a4f8bc56cd96bdc088` is the current clean integrated
HEAD. Root verifies all four implementation/README hashes equal the reviewed
standalone bytes, checks the complete six-file integration scope, reviews
the dated handoff corrections, and independently passes all 18 offline
stdlib PCM tests. Diff hygiene passes. The three native KV Rust tests remain
`NOT_RUN`; this candidate needs a separate later exact-head remote run and
must not retarget live V21. Preparation review is recorded in normal root
management commit `367206c948a3224cf9eaf2027655fcbde325ab5e`.

The new, uncommitted official KV tap is not accepted for capture. Root reads
its complete source, tests and handoffs and finds that retained borrowed ring
views can be overwritten by later frames, resets or eviction. Pointer-only
storage identity cannot distinguish K/V regions within the shared real
backing tensor. Poison and reset-budget guards occur after upstream mutation,
and a partial reset can incorrectly clear a poison. Assigned corrections
must preserve source dtype in bounded snapshots or establish an immediate
ephemeral consumer, validate exact storage regions, and reject inadmissible
operations before mutation. Its five reported synthetic tests are not real
upstream or numerical evidence. Full tracked-Python source receipt preparation
is also assigned separately; it has not acquired a checkout or resolved the
dependency/license/owner closure.

To preserve local recovery space, root removes only three redundant bundles
from the terminal synthetic debug directory
`/private/tmp/vokra-clean-heads-model-free-logs.4UkT3M`. All logs and source
receipts remain. The live-run input bundles in
`/private/tmp/vokra-clean-heads-model-free-logs.97TzDj` are byte-identical
and independently verified as complete-history backups: model/OWSM SHA-256
`172870252098b86c1b20d848bd66ef7a73b721c38c0051e438fd87b05c40ba71`,
XCodec2 SHA-256
`87c1a296db86b52e11ff4392db4111b562d96efb4aeb35e3238d75c6f9dd0b35`.
The old directory drops from about 165 MiB to 544 KiB. No unique commit or
evidence is removed, and the live-run inputs are not modified.

All 194 public rows remain in scope. The metadata-only 136 full / 58
unresolved split is not real-weight or Apple completion. No source/license,
CPU parity, Apple/no-fallback or publication verdict advances here.

## 2026-10-03 official KV preparation review (03:52 UTC)

The corrected official main-transformer KV tap supersedes the earlier source
review rejection, not the outstanding execution verdict. Root reads all 745
implementation lines, 408 test lines and both documents, compares the retained
official `RingKVCache.complete()` and `_complete_kv()` source, and independently
passes nine offline stdlib tests with no skips. The implementation authenticates
both base storage and exact K/V shape, stride, storage offset, dtype and device.
It creates detached, bounded snapshots before subsequent ring writes can
overwrite them. Event/byte and poison guards precede upstream complete/reset
calls; partial reset clears only selected poisoned batches. Patch restoration
attempts each owned patch independently. Type-name checks remain scope checks,
not upstream source authentication.

Normal logical commit `2c0f855e2c4aaabf50a52a227571c31a71186743`, parent
`fc977f3b7637d7f3c8249cccf594d33971fee019`, contains only the adapter, its
tests, README and dated handoff. Its implementation SHA-256 is
`42032a71fa95e925f27f4e995cae45cbd83fcd0329f17fa24fa11370d7c26920`;
test SHA-256 is
`8fec95c1dc5b823b89e314ec75af0f5bdb87f3a6f5b67864b71f0e03758ec493`.
No upstream/Torch/model execution, network acquisition, Cargo or push occurs.

Separately, normal commit `9829a0f0521d8a335faa4afd6fb0e9ae188b564d`, parent
`858bfa464fb0b101b1d4f1a4f8bc56cd96bdc088`, permits zero-byte tracked Python
files only in the supplemental source closure. Required DSM/Moshi roles
remain strictly positive and exact-hash bound. Root reviews the full four-file
diff and independently passes 19 PCM tests, including a tiny real Git checkout
with an empty tracked `__init__.py`, missing-file and wrong-zero-hash cases.
The tiny checkout substitutes source identity constants for a synthetic caller;
it proves validator orchestration, not an actual upstream checkout. Capture
implementation SHA-256 is
`b624d5fb75a0814fa4740625a811d946f344ef9c4cdb7a8609ed36d8abd1a285`;
test SHA-256 is
`312867a6b0a41e0f76f6c9cc260c60f5c3c3f83f458958e23d0835a3c1b1f742`.

The reviewed tap is cherry-picked into that existing isolated integration tree
as clean HEAD `11d97f14964d21482741cd0a7a5dee9d408f2b77`, parent
`9829a0f0521d8a335faa4afd6fb0e9ae188b564d`. Root verifies exact implementation
hashes and the eight-file combined scope. The implementation owner reruns all
19 PCM and nine tap tests, diff hygiene, AST, forbidden-symbol and zero-dependency
gates. The three native KV Rust tests are still `NOT_RUN`. Full primary source,
dependency/native-owner, composite license/approval, actual precision/context,
independent real-weight PCM/CPU and Apple gates remain open.

The independent read-only VAST probe in terminal session `46518` reports zero
exits for workspace tests, workspace Clippy, cargo-deny, cargo-audit, decoder
source, OWSM source and OWSM focused tests. Its remaining Cargo process is live.
Controller session `42297` remains live and fixed at `1cbc4abc` / `a5e4c810`;
these partial observations do not establish its terminal result. Required
checksums, source receipts, remaining contracts and destruction readbacks must
still be recovered and verified. New `11d97f14` must not inherit that older
HEAD's results or enter its running scope.

Root's targeted disk check reports about 0.87 GiB free, so no new clone or large
local environment is created. Source-only collector preparation uses existing
retained primary bytes and synthetic API envelopes; its local test success is
not a newly acquired full source closure. All 194 public rows remain in scope,
and the metadata-only 136 full / 58 unresolved classification is not a hardware
completion count. No owner sign-off, real-weight/Apple verdict or publication
authorization is invented.

## 2026-10-03 terminal V21 code/source readback (04:02 UTC)

Frozen V21 controller session `42297` terminates with exit zero at model/OWSM
HEAD `1cbc4abce1752f4a267f6ae024de5fdcbf0feb32` and XCodec2 HEAD
`a5e4c810f853c1d0c25d1df78635f115d7c7d153`. Root independently verifies all
80 recovered file checksums (895,482 bytes), all 39 exit files equal zero,
exact HEAD markers and `collection-status=complete`. Collection-manifest
SHA-256 is `df20206083c87fed0cdf1957c8e5674a0ab0118ea471f3c5157bc994f2a308da`.
Workspace results are 307 suites, 8,293 passed, zero failed and 108 ignored;
ignored real-artifact/device tests are not completion evidence. OWSM focused
tests pass 40 without ignores, the 13 bridge/guard and 24 PCM cases pass by
name, and XCodec2 passes 76 tests. Clippy and security contracts pass; deny
retains its unmatched `libfuzzer-sys` exception warning. The resource summary
records 32 allocated cores and 32 build jobs. No model/reference package or
HF credential workload runs.

At fixed Moshi revision `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`, root
independently binds the three acquired raw files to their API base64 contents,
Git blob IDs, fixed paths/URLs, sizes and manifest hashes:

- `compression.py`: 17,429 bytes, SHA-256
  `edc6fbf34e4d84b35f2c2f4d6f1f263c6ba86329a7c686b091f304a6b3cbfc8f`.
- `streaming.py`: 8,433 bytes, SHA-256
  `bafaaafd12291727a6a613e9ac68623dc924fa6cdf57cdc9874f66fb30464d38`.
- `utils/compile.py`: 11,047 bytes, SHA-256
  `08cfd2422318bf9425c57fcbdbd98326de64fe2b8f093c6c9b75eb3997814ae8`.

These 36,909 bytes are primary source evidence, not upstream execution or
numerical parity. `streaming.py` lines 139–156 show optional-mask normalization
and recursive child-state reset. A read-only diagnosis is assigned to check
whether the direct main-module reset wrapper observes parent-initiated resets;
synthetic direct-reset tests alone do not prove that lifecycle.

Controller cleanup reports `cleanup_rc=0 instance=53959731`. Root separately
performs fresh individual and complete account API readbacks in terminal
session `17294`, exit zero: individual `instances=null`, explicit terminal
`next_token=null`, zero Vokra instances and one unrelated instance. Worker and
storage are destroyed; the unrelated resource is not modified. Fresh individual
and all-page response SHA-256 values are respectively
`817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057`
and `3d60fabd8a1f9b967d8b5cd1a9094eb817fd90c4ce8f35a393020f828d8a28d7`.

New preparation commit `fd7159f5597dd82f7a563a4adac9ea263c1b09b5`, parent
`11d97f14964d21482741cd0a7a5dee9d408f2b77`, normally commits only the bounded
full-Python source collector, its tests and README. Root reviews the collector
and portable retained-source tests and independently passes 31 tests without
skips. Its production-branch positive uses the five already authenticated
source-role bytes inside synthetic API envelopes; it is not full upstream
tree acquisition. Collector SHA-256 is
`258a1c1ba35108fbf0452a775a2944fe9be967bcb9b5cef46fa96eb072155379`.
No new provider is allocated. Its native KV Rust tests, full source checkout,
dependency/native-owner and composite approval, independent real-weight CPU,
final Apple/no-fallback and publication gates remain open. Keep all 194 public
rows; neither this preparation nor V21's older exact-head code pass promotes
the metadata-only 136 full / 58 unresolved split to hardware completion.

## 2026-10-03 recovery-space and controller review (04:09 UTC)

Root's narrowly scoped read-only inventory finds 301 old controller log roots
with large top-level bundles, including 234 terminal synthetic roots marked
`cleanup_rc=0 instance=7`. This is local debug data, not a provider-instance
count. Before removal, a filtered process check finds no live matching
controller. Root hashes and byte-compares exactly 64 `xcodec.bundle` copies
against retained backup
`/private/tmp/vokra-clean-heads-model-free-logs.zUpL0K/xcodec.bundle`.
Every copy is 57,600,230 bytes with SHA-256
`7d944ce4f6ef408a815860ad9dde380decbf0b79c4a0b48d50c42bb362a94ac9`.
`git bundle verify` confirms complete history, including HEAD
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, its remote tip and `v0.3.0` tag.

Only those exact 64 redundant files are removed in terminal session `41155`,
exit zero (3,686,414,720 apparent bytes). The backup is rehashed and reverified
afterwards. All source receipts, logs, model bundles, unique histories and
unrelated directories remain. Deleted copies can be restored from that
byte-identical complete-history backup. Local free-space readback increases
from approximately 0.77 GiB to 4.85 GiB; this does not relax the next provider
job's minimum recovery-space guard.

Targets are `xcodec.bundle` in log roots with the prefix
`/private/tmp/vokra-clean-heads-model-free-logs.` and the following exact
suffixes:

```text
0ST4jy 1lGE0H 26v8EF 2EYLcw 2IIdHk 2VZwxL 2q26Cf 3CD916
3CsALk 3QiaM9 3ZUdaR 4BWDFe 4ilf4j 4xrMCw 5G4JM8 5P8GHy
5mEQvc 6THtpB 6fN5rh 6xlmEy 7nHeny 7sBLsg 873Ot5 8Ihghb
8iyIfE 9A788g 9AB98u 9BCom6 9NGEhY AHh8QA AJynaH ASUvsC
Aao5Xg B4V473 BF2Q25 BG4eAB BJjVcO BRYNFs BVvCD5 C8VHYx
CURwnq CUwwMU CgBs4X CqG6FU D2r1GR D3s8Kj D5pcgi DVUInt
EgJmfU F2Sudb F3CMHT F9MmYA FDKGLJ FRPCWU FtCTsh GAu5Pu
GKXtc5 Ghv4jy GnLSCL H6eEcs HaGI7K Hce8Nk HewFHK HvUCaq
```

Root also reads the complete 117-line V22 draft and its handoff. It safely
refuses provider execution, but does not implement the requested remote
source/KV wave. Its self-test calls separate handwritten `audit`, `envelope`,
`deadline` and `clean` functions rather than actual production paths; its
empty-list cleanup example differs from the real individual-null contract.
These 31 toy checks cannot establish actual collector, native Rust, recovery
or lifecycle coverage. The implementation owner is assigned to reuse the
reviewed real VAST lifecycle and tracked collector, with fault tests against
the actual leaf/harness and tiny input bundles. No new provider is authorized
from this disabled draft. Full source/dependency/composite owner closure,
independent real-weight CPU, final Apple/no-fallback and publication remain
open; all 194 public rows are preserved.

## 2026-10-03 actual-path and current-scope review (04:25 UTC)

Root commits the reviewed recovery-space/controller record normally as
`1c609174d0b3c1d7dc2e93f832c09ec5d8910f3d` (three management documents,
68 insertions). All five normal compile-free hooks pass and the root worktree
is clean. Doc references, runbook citations and diff hygiene also pass.

Authenticated fixed-source inspection confirms that parent Moshi reset
traverses child streaming states directly, then reaches `_MHAState.reset`
and `RingKVCache.reset`; it does not call the child's `reset_streaming`
method. The fixed DSM caller itself uses one fresh streaming context without
an intervening reset. That narrower caller does not prove general lifecycle
support. A four-file correction replaces the direct-main wrapper with cache
reset wrappers. Root reads its complete diff and independently passes all
14 stdlib synthetic tests in terminal tool result `b81353`, without Torch,
upstream imports, model access, network or Cargo.

Root nevertheless rejects that uncommitted correction: an incomplete
multi-layer reset still permits complete/capture/reset-record consumption;
stale-cache reset identity is unchecked; and an all-false reset failure can
lose its poison after an all-false retry. The tests also need the genuine
parent-to-state-to-cache bypass and upstream device normalization topology.
The implementation owner is assigned those bounded corrections and their
regressions. No real KV values, numerical parity or hardware verdict is
inferred from the 14 synthetic passes.

Root also reads the entire 421-line V22 draft and its handoff at SHA-256
`0c023ab2389ed051c8f244d18d62af131cecec7eb2b49738b838165de2024e59`.
Its 31 validator cases pass independently in terminal session `19708`,
exit zero; `--run` exits 78 before any provider/SSH/network action.
Frozen V21 still hashes to `f754c533e95871e126684481dba6db121a016de4bc5efe54453da33130bcb83a`.
The draft is not accepted: extra fixture files can make negative cases pass
for the wrong reason, several mutations are no-ops, Python summaries are
parsed as Rust, remote cwd/source environment are incomplete, the original
source-receipt revision chain is discarded, recovery caps are unenforced,
and bootstrap/create/pagination/ownership/cleanup paths are unproved or
incorrect. Require reuse of the actual audited lifecycle in a new file and
tiny fault fixtures against the real generated leaf; do not restart V21.

Fresh read-only GitHub queries retain main
`97447185361a37af64c1b30fe87e8e2618d96e20` and 19 open PRs, all draft.
PR #152 remains clean at `dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`,
with 76 successful and three skipped checks. Dependabot alert #159 remains
open and critical: `GHSA-53q9-r3pm-6pq6`, Torch in
`tools/parity/xcodec2/uv.lock`, patched floor 2.6.0. The current local project
and lock still pin Torch 2.5.0. Neither code-only passes nor clean CI settle
that advisory, package-license closure or real-weight execution.

SpeechT5's current 28-package compact audit already binds its active
Transformers 5.10.4 / Torch 2.13.0+cpu project and lock. Do not mechanically
repeat that audit as though it were missing. Its current approval scope is
`6ee461bab29ef90b6d425f5ca179eb79ac7cd5e0278db1c4c2a28efa4834f8c1`,
while the checked-in historical approval names `99116b392c560ec40c574589305492f35d9d30e8e2f44a9c03392885c77e85ba`.
Current operator/native/build-only review and authenticated API smoke remain
open. Historical approvals or Apple results cannot be transferred to changed
scopes, nor described as missing solely from an older decision table.

The retained row-level TSV is rehashed to
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
All 194 metadata rows remain in scope (136 route/artifact-full, 58 explicitly
unresolved). Detailed review of priority families is not a hardware audit of
every row. No provider allocation, model acquisition, upstream/reference
execution, upload, PR mutation or Scaleway action occurs in this review.

## 2026-10-03 accepted KV preparation and execution boundaries (04:36 UTC)

This record supersedes the KV draft rejection at 04:25 UTC; that earlier
measurement remains historical evidence. Root reviews the corrected actual
parent/state/cache reset topology and accepts the bounded four-file change
normally committed as `42ba5cf2ed2dacd0800167f512309e4f215a39b1`.
The correction blocks capture/complete during incomplete multi-layer resets,
commits one generation only after every bound cache resets, retains poison
after failure or incomplete-context exit, and restores wrapped methods.
No new tensor arithmetic or reference numbers are introduced.

The separately reviewed upstream caller citation correction is normally
committed as `07e04668195bc4910f01c5ed73bb19f0552dfe79`. All five normal hooks
pass and that integrated worktree is clean. Root independently reruns the
20 stdlib synthetic tap tests on that exact head: 20 passed, zero skipped,
in 0.007 seconds, terminal tool result `389fd5`. The integrated runbook gate
also passes (1,140 anchored citations, 129 runbooks, 482 paths).
These prove preparation only. The three native KV Rust tests on this head,
full primary source acquisition, composite dependency/legal closure,
independent real-weight CPU parity and Apple CPU/Metal/no-fallback remain
unverified; earlier V21 code-only results are not reused for this new head.

Root reads the complete 681-line V22 controller at SHA-256
`a191855ac4ea68008af674108f3c095af42609a27d548e06639bb75475bd6fc2`.
Syntax and all 31 production-validator cases pass independently; session
`4746` terminates with exit zero. The installed Vast CLI's own help confirms
that `show instances-v1 --all --raw` is supported, correcting the earlier
unverified suspicion about that command. Nevertheless the draft remains
disabled and unaccepted: its mock lifecycle does not invoke the actual
production controller or generated leaf; destruction lacks a fresh exact
ID/label owner recheck; cleanup/recovery lack a shared total deadline;
remote bootstrap PATH and allocated/cgroup CPU/RAM are not carried through;
and recovered source rows need actual Git-blob binding and named-test exit
receipts. The implementation owner is assigned these bounded corrections
with tiny fixtures, without local Cargo, model execution or provision.

SpeechT5's current manifest SHA-256 is
`63a508e172a2c0a753ce434eef20d7fe2b731a04a476cf979cc16868d0a639e6`;
its compact current audit is
`2ce6275c59c9dcf9731438f355484360bc1ee7ab36268a4caac17d872f71a434`.
The audit binds the Torch 2.13.0+cpu native libgomp bytes and reports the
patchelf build-only tool absent from the final environment, but does not
itself retain the primary GCC-exception/patchelf license text needed for a
new operator decision. The original full-audit artifact is sought before
any repeat audit. Current scope `6ee461...` remains pending; historical scope
`99116...` is not substituted, and no speculative sign-off is recorded.

A fresh complete Vast readback in terminal session `8418` reports
`next_token: null`, zero Vokra-labelled workers and one unrelated stopped
project instance. It is not modified. No new allocation, model acquisition,
upstream execution, upload, push, PR mutation or Scaleway action occurs in
this checkpoint. All 194 public rows remain in the completion denominator.

## 2026-10-03 primary evidence recovery and official reference candidate (04:58 UTC)

The original SpeechT5 full audit is recovered, not regenerated: 277,065 bytes,
SHA-256 `bcd5c811713a23f0373db17039d3c3844968c75936388ef55d445c7643443082`.
Root independently verifies the hash, parses its schema and policy, and
compares its project/lock hashes against the current files. They match
`a6f51a2e3300ba0b8750dca6050c6fc2d645e657c8a507dcfc334c291d9e0bff`
and `36d0f01df62e4d9f90f80d4c7d15bfa9df612b5a4f99ad4716c1c62458a6864b`.
The audit records clean historical head
`a7eb478e1d6cd42e920767aade102f2e8738746c`, raw dependency/native facts,
build-only tools absent, and no Torch/model import, model acquisition,
license classification, owner sign-off or upload. Its embedded historical
approval is not an approval for the current manifest. This supersedes the
04:36 statement that the original artifact was still sought, without
promoting the current operator/legal gate or API/real-weight tests.

The official [HKUSTAudio Transformers-native checkpoint card](https://huggingface.co/HKUSTAudio/xcodec2-hf)
documents `decode(audio_codes).audio_values` and identifies noncommercial
weights. The [official Transformers X-Codec2 documentation](https://huggingface.co/docs/transformers/main/en/model_doc/xcodec2)
dates the contribution to 2026-06-25, and its
[upstream implementation directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/xcodec2)
contains a checkpoint converter and native model implementation. These
primary sources change the next action: investigate a released, immutable
Transformers reference and its conversion mapping rather than assuming the
old `xcodec2==0.1.5` dependency pin is the only official route. No exact
version/revision, old/new weight identity, decoder equivalence, dependency
license closure or numerical result is established by these pages. The
critical old-Torch alert remains open; no lock change or approval occurs.

V22 implementation is still under review. The offline fixture must not
enable the public provider entry point through environment variables, touch
physical remote bootstrap paths on the maintainer Mac, or substitute a
handwritten receipt builder for the actual first-party collector. Passing
synthetic validator counts alone is not production lifecycle evidence.
No allocation, local model execution, model acquisition, upstream numerical
execution, publication, push, PR mutation or Scaleway action occurs in this
checkpoint. All 194 public rows and their actual final hardware gates remain
in scope.

## 2026-10-03 bounded reference preparation and transfer-safety review (05:22 UTC)

Root accepts the source-only XCodec2 collector after complete implementation
and test review, then independently runs its 22 stdlib tests through offline
UV/Python 3.12 with `-S`: zero failures in 0.031 seconds. Deep output paths,
portable temporary paths, fixed-source identity checks, deterministic Linux/
Mac execution gates and fake-response transport faults are covered. This is
synthetic orchestration evidence only; no real source, model or numerical
execution occurs. The three files are normally committed as clean integrated
`dd7db2cf61be539a2e046c884741632a263dc5ae`, descended from accepted `07e04668`.
Diff checks, forbidden-symbol and zero-dependency gates pass; the runbook gate
reports 1,142 citations across 130 runbooks and 484 distinct paths. All five
normal pre-commit checks pass; no hook bypass, push or PR mutation occurs.

Metadata-only official GitHub API readbacks bind the `v5.13.0` tag object
`d9055545a3a17df7ba8682e36d88c9bf4122c7c6` to code commit
`6af945f436d85f2b0c5dff9b14feccd27b1d470b`. The comparison shows that the
non-causal fix `7f5d5d1aaca3cc3d236c80ec8cb34d06f08a5fb8` is ancestral.
At that fixed release, the XCodec2 directory has five files and no checkpoint
converter. The collector therefore holds only those five source files and
the repository license, not a dependency/import closure or a conversion
mapping. The annotated tag reports `verification.verified=false`; this is
an authenticated metadata readback, not a claim of a verified tag signature.

A separate official API readback identifies original-conversion commit
`9b6af5d77de78f0a57a098b2809009ad6ec0cfe3` and its
`src/transformers/models/xcodec2/convert_xcodec2_checkpoint.py`: 15,239 bytes,
Git blob `9b925564edab74774d225da25285091bff678391`. Its repository license is
11,418 bytes, Git blob `68b7d66c97d66c58de883ed0c451af2b3183e6f3`.
No converter body is recovered or executed in this checkpoint. Prepare a
separate immutable source receipt and review the mapping rather than infer
old/new checkpoint or decoder equivalence. The old Torch critical alert,
CC-BY-NC weight boundary and independent real-weight gate remain open.

Root's full V22 review finds remaining production issues after the implementer
reports 42 offline cases passing: an unchecked proxy host takes precedence
over the validated direct endpoint, a nonzero remote output-cap preflight is
recorded but not enforced before recovery transfers, and signal traps are
removed before final cleanup. Require corrections and actual shared-path
fault coverage. The public provider entry point remains `PENDING_REVIEW`;
the reported fixture count is not root acceptance or a real cloud result.

The latest complete account readback at 05:06 UTC exits zero with
`next_token=null`, zero Vokra instances and one unrelated stopped instance;
the unrelated resource is untouched. Current local recovery headroom is
about 2.87 GiB; no storage cleanup is performed in this checkpoint. Historical
clone candidates contain dangling histories and are retained. There is no
allocation, local model acquisition/execution, upstream numerical execution,
publication or Scaleway action. All 194 public rows and their final hardware
gates remain in scope; the 136 code/artifact-full versus 58 unresolved
classification is not a count of hardware-complete models.

## 2026-10-03 reviewed source-batch and lifecycle checkpoint (05:40 UTC)

Root accepts a second immutable XCodec2 source profile after full diff review
and 26 independent offline UV/Python 3.12 tests with `-S` (0.039 seconds).
Clean commit `6fb2720f4491db42bf75c00eee008bfadaf00b08` keeps modern-code
`6af945f436d85f2b0c5dff9b14feccd27b1d470b` (six files) distinct from
original-conversion `9b6af5d77de78f0a57a098b2809009ad6ec0cfe3` (license and
converter). Neither source profile has been acquired or executed in this
checkpoint; the old GGUF/new checkpoint mapping, safe dependency route,
critical Torch advisory and independent real-weight parity remain open.

Root also completes review of the new three-file SpeechT5 primary legal
collector and independently passes its explicit offline `--self-test` path:
16 tests, zero failures in 0.008 seconds. The earlier twelve-case draft
result is not substituted for this corrected revision. The collector permits
only five fixed raw license/context files, no archives, packages or models,
with exact Git blobs, bounded HTTP reads and Linux/VAST opt-in. General GCC
license/exception context is explicitly not a proof that the retained Torch
`libgomp.so.1` was built from that GCC revision; the PyPI packaging license
and upstream patchelf license are separate identities. No owner/operator
sign-off or publication permission is inferred. Normal commit
`0f26b6ea256d198d7d479da44aedf09353b1c908` contains only these three files;
all five ordinary pre-commit checks pass, including 190 byte-hashed fixture
pins and 306 shell files. Forbidden-symbol, zero-dependency and diff checks
pass. The correctly named runbook path gate passes with 1,144 citations in
131 runbooks and 486 distinct paths; the initial attempted
`scripts/check-runbook-commands.sh` does not exist and is not counted as a
successful check. The integrated worktree is clean at this exact HEAD.

Root independently runs frozen V22 controller SHA-256
`d853bc054d05f83556b89fdcca770a4a2d23de61c6290eba7da45315472745c5`:
session `65117` ends with exit zero and 46/46 offline shared-path cases.
The harness covers direct-endpoint preference and malicious proxy rejection,
pre-transfer size enforcement, diagnostics-only failed-preflight recovery,
partial-create ownership recovery, positive recovery, faulty leaf results,
complete cleanup readbacks, paginated residual rejection and actual TERM
cleanup (exit 143). Fixed offline mocks prevent provider/model execution.
This accepts the bounded harness, not a live cloud run; public V22 `--run`
remains disabled. A new controller is delegated to bind the reviewed clean
HEAD and batch these source receipts with exact-head remote code tests.

The complete VAST account readback at 05:30 UTC has `next_token=null`, zero
Vokra instances and one unrelated running instance. The unrelated resource
is untouched; account-wide resource/cost zero is not claimed. An offers-only
query at 05:40 UTC returns eligible CPU/RAM candidates under the USD 0.20/hour
ceiling; no instance is created. Current recovery headroom is about 2.56 GiB,
above the controller's 2-GiB floor; no cleanup is performed. No local models,
real source acquisition, upstream numerical execution, upload, code push,
PR mutation or Scaleway action occurs. All 194 rows and their final hardware
gates remain in scope; 136 code/artifact-full and 58 unresolved are still
route classifications, not hardware-complete counts.

## 2026-10-03 capacity and independent lifecycle review (06:26 UTC)

Root restores local recovery headroom without removing worktrees or primary
evidence: 140 terminal offline mock input Git-bundle duplicates, totaling
8,086,885,744 bytes, are compared with retained valid parent bundles before
removal. Twenty parent bundles, all checkouts and test logs remain available.
One intermediate cleanup command uses the wrong legacy directory name for
84 nonexistent paths and exits one; it is not recorded as a successful
112-file deletion. Root inspects the actual unversioned paths, verifies those
84 existing copies with independent fail-closed checks and removes only that
corrected explicit file list. Starting headroom is about 1.6 GiB, the immediate
post-cleanup measurement is about 9.0 GiB, and the 06:26 measurement is about
8.1 GiB. These are point-in-time readings, not guaranteed future capacity.

Frozen V23 SHA-256
`ccd6a1230ae82635952cb55abd986ace16d7d8d3afc9a6d7355be39e6dba2985`
passes root's complete offline self-test: session `81583`, terminal exit zero,
52/52 cases. The corrected validator accepts realistic Cargo summary suffixes,
requires zero failures in every summary and a nonzero aggregate pass count,
and rejects the exercised zero-workspace and failed-workspace fixtures.
Partial-diagnostic caps now cover all added source/workspace logs before
transfer. The actual TERM fixture records exit 143 and mock destruction.
Earlier root sessions `23348` and `33541` are failures, not passes: inherited
`UV_NO_SYNC` warnings contaminated mock JSON in the first, and the second
TERM launcher attempted an unwritable default UV cache. Explicitly unsetting
`UV_NO_SYNC` and selecting a writable task cache resolves those environment
failures without changing the frozen controller. Its public production entry
is still disabled; a bounded activation delta is assigned for separate review.
The clean integrated execution HEAD remains `0f26b6ea`, not a new model result.

The fresh complete VAST readback has `next_token=null`, zero Vokra instances
and one unrelated exited instance, which is untouched. GitHub main remains
`97447185361a37af64c1b30fe87e8e2618d96e20` and is an ancestor of the integrated
candidate. No provider mutation, local model acquisition/execution or code
push occurs at this checkpoint. A local BigVGAN metadata-only HTTP proposal
is rejected by the memory hook before execution and is not retried; the three
non-base primary receipts remain unacquired. Their remote-only collector is
being prepared separately. Qwen3-TTS's current full dependency evidence still
needs recovery/regeneration; model-free fact collection does not release its
four unresolved Torch/TorchAudio rows or authorize real-weight execution.

All 194 public rows remain in scope. The 136 code/artifact-full and 58
unresolved split is not an Apple-completion count. Source/legal closure,
independent real-weight CPU parity, final Apple CPU/Metal/no-fallback evidence,
current CI/security review and separately authorized publication remain ahead.

## 2026-10-03 actual source-batch failure and correction (06:45 UTC)

Root independently accepts activated V23 SHA-256
`1ec35f619430cd5524a0a58a48f9bf03593f4f0e223850948d7489e855248633`:
session `71112` terminates zero with 53/53 actual shared-path offline cases.
This supersedes the earlier disabled-entry checkpoint, not a model verdict.
Production session `22092` then creates only owned worker `53977035`, label
`vokra-kyutai-source-kv-vast-v23-20261003-26087`, for exact clean implementation
HEAD `0f26b6ea256d198d7d479da44aedf09353b1c908`. Its selected offer has 14
effective CPU cores, 200 GB requested storage and USD 0.1985185185185185/hour.
No HF credential, model/checkpoint/config/tokenizer/preset acquisition or model
execution is part of this source/code run.

The modern and original XCodec2 source collectors and the five-file SpeechT5
primary legal-context collector each exit zero. Their recovered tests pass
26, 26 and 16 respectively, without skips. Workspace compilation, however,
exits 101 on Rust 1.99.0 with E0689 at line 1497 of
`crates/vokra-models/src/kyutai_stt/streaming_lm.rs` (candidate-only; does not exist in this checkout): the new test's
untyped frame counter cannot resolve `saturating_sub`. No workspace pass
count, Clippy pass, native-KV test pass or full Kyutai-source acquisition is
claimed for this HEAD. The remote leaf exits 101, the complete-output
preflight exits one, and the diagnostic preflight exits zero. Controller
session `22092` is terminal one, not a green lifecycle run.

Recovered diagnostics are retained in
`/private/tmp/vokra-kyutai-source-kv-v23-logs.x2SvPA`. The 3,312-byte workspace
log SHA-256 is
`6005d4d3fc4ef1eaea19577b2d14195ade402a0f8ee10e740e657edccc94f1ac`.
The controller assembles its primary source packet only after the workspace
legs; consequently, the acquired profile bytes are not recovered by this
failed run. Exit-zero acquisition logs do not substitute for authenticated
local primary receipts or legal approval. Reacquire/recover the bounded source
packet during the corrected replay rather than promoting these logs.

Automatic cleanup passes both individual-null and complete-pagination checks.
Root's separate session `73243` also terminates zero, confirms
`53977035` has `instances=null`, and reads a complete `next_token=null`
account inventory with zero Vokra rows. The one unrelated exited instance is
untouched. Worker storage is not retained.

Root reviews Luna's isolated one-line correction, changing only the test loop
to `0usize..5`, and commits it as
`52e4ddbebfeb32cb5f7aa241580b2493b20ecdb0`. The fixed checkout is clean.
That isolated clone initially has no installed Git-hook setting; root runs
the normal five compile-free gates explicitly in session `74736`, terminal
zero, and installs the standard hook setting for subsequent commits. This
does not prove corrected-head compilation or numerical parity. Frozen V23 and
its old execution HEAD are not retargeted; a separate replay is being bound to
the correction.

Separately, root reviews the three-file BigVGAN metadata-only collector,
including fixed-endpoint admission, per-read deadlines, final-read socket
closure, response caps and synthetic evidence markers. Independent offline
self-test and diff hygiene pass; normal commit gates pass with 190 fixture
pins and 306 shell files. Clean preparation commit
`af462cffda14e97b044d3ebd718b23c08572c841` contains no manifest approval or
primary acquisition. The six-test generic suite remains `NOT_RUN` locally
after the earlier hook refusal and is assigned to VAST, not retried locally.
Its three real metadata responses and per-variant license/card bytes are still
absent. Qwen3-TTS's separate controller remains disabled after root finds
additional recovery-deadline, endpoint/ownership and fault-coverage defects.

All 194 rows remain in scope. The retained 136 code/artifact-full and 58
unresolved classification is not Apple completion. Independent real-weight
CPU results, legal decisions, final Apple CPU/Metal/no-fallback, exact-head
security/CI and separately authorized public reconciliation remain open.

## 2026-10-03 corrected replay and native PCM boundary (07:01 UTC)

Root's independent V24 session `1117` is terminal zero with 53/53 offline
shared-path cases and unchanged controller SHA-256
`7026300c313606812c035537ce4a856d55224a96ceeb3a8f7b6f467a7b1605bc`.
The capped partial-source fixture reports `VALIDATED_FOR_REVIEW_ONLY`, not
model readiness. V24 fixes the execution HEAD at clean
`52e4ddbebfeb32cb5f7aa241580b2493b20ecdb0` and assembles the source-profile
packet before workspace compilation. Frozen V23 remains historical.

Actual sessions `64079` and `40652` terminate one before instance creation:
their selected 36- and 14-core offers disappear from the fresh pre-create
search. No cloud worker or storage is created by either attempt. A distinct
attempt, session `94842`, creates owned worker `53978818`, label
`vokra-kyutai-source-kv-vast-v24-20261003-85980`, on an available eight-core
offer at USD 0.07481481481481482/hour with 200 GB requested storage. Its logs
are in `/private/tmp/vokra-kyutai-source-kv-v24-logs.Yx9e85`. Provider readback
and both SSH probes succeed; Rust 1.99.0 and UV-managed Python 3.12.14 bootstrap
complete. Session `94842` is confirmed live at this checkpoint: no workspace,
Clippy, primary-recovery or cleanup outcome is inferred yet. Continue observing
that same handle, recover bounded evidence and destroy this owned worker and
storage; do not restart merely because an observation times out.

A source-only review of the corrected candidate also distinguishes its
private native PCM session from the public caller-text bridge. The existing
private DSM route already composes authenticated local artifact construction,
Mimi encoding, streaming LM steps, greedy token selection and tokenizer text
rendering. This is implementation preparation, not authenticated real-weight
execution or independent parity. The separate historical MLX and DSM padding
policies must not be combined, and public `transcribe` remains explicitly
deferred. Exact composite source/artifact/tokenizer identities, dependency and
legal closure, official-caller evidence, independent real-weight CPU parity and
final Apple CPU/Metal/no-fallback results are still required.

Qwen3-TTS's separate model-free controller and BigVGAN's remote metadata leaf
remain under review; neither has performed a production run. Preserve all 194
rows and the metadata-only 136 full / 58 unresolved snapshot. No model weights,
configs, tokenizers or presets are acquired or executed in this code/source
wave, no HF credential is forwarded, and no public artifact changes.

## 2026-10-03 terminal code pass and incomplete source recovery (07:40 UTC)

This supersedes the live V24 observation, not its frozen execution scope.
Controller session `94842` terminates one with `recovery-file`; exact clean
implementation HEAD remains `52e4ddbebfeb32cb5f7aa241580b2493b20ecdb0`.
The remote leaf and complete-output preflight both exit zero. Recovered
workspace tests report 307 result-bearing suites, 8,223 passed, zero failed
and 108 explicit ignores; default and all-feature Clippy both exit zero.
The 702,851-byte workspace log SHA-256 is
`1a8099c159302fe84f9873d1b58e72a8851253d01abc47b749655c656a7cb58a`.
These are exact-head code results, not independent real-weight or Apple parity.

Root independently runs the unchanged V24 leaf-log validator over recovered
evidence and obtains a terminal pass: PCM Rust has 24 passed with zero ignores;
Python PCM, tap and full-source tests pass 19, 20 and 31 respectively; the
three named native-KV tests pass without ignores. Modern/original XCodec2
source tests each pass 26, and SpeechT5 legal-context tests pass 16. All
corresponding collection/test exit codes are zero. Acquisition logs do not,
alone, prove primary-byte recovery or legal approval.

The source manifest lists 128 files totaling 1,233,698 bytes, with SHA-256
`1f6c595e3c0d142f284883121eedb2c642a8a899ba74f2aa1019be93a09cea91`;
the full-source receipt lists 111 held roles. The 300-second shared recovery
budget expires during repeated SSH-plus-SCP transfers. Only ten source files
remain in the original `remote-flat`; no full flat/source/profile-validator
pass is claimed. A separate bounded bulk-copy attempt transfers no source
bytes: its direct endpoint lacks an authenticated matching host-key entry,
and host verification is not disabled. Preserve these failures. A source-only
bulk leaf is assigned to reacquire the missing fixed-revision packet without
rerunning green Cargo. No replacement worker is provisioned at this checkpoint.

The controller verifies ownership before destruction and passes individual-null
and full-pagination checks. Root's independent API session `80431` is also
terminal zero: `53978818` has `instances=null`, the account has explicit
`next_token=null`, the owned ID/label is absent and zero Vokra-labelled
instances remain. One unrelated exited instance is untouched. No Vokra worker
or storage is retained from this wave.

A narrow local-inspection guard correction is fully reviewed, independently
tested, committed and integrated as
`282227af2de733d55eb81a02c27eae603b9737c4`; all five normal compile-free
commit gates pass. It recognizes literal, canonical `git -C` inspection;
new diff/show admission requires `--no-ext-diff --no-textconv`. Actual model
execution/download, heavy Cargo, protected paths and immutable-controller
admission remain guarded. Explicit external diff/textconv is refused;
ordinary Git management is not blanket-blocked. No approval/sandbox setting
or hook trust is disabled. Source review then uses the stronger read-only
form, not an execution override.

The initial PR #152 correction passes root's 24 small stdlib tests, collector/
audit self-tests and documents-only guard. Full diff review then finds that
RECORD presence does not authenticate entry hashes or import resolution, a
callable loader still bypasses pre-import checks, and the aggregate unpacked
cap can prevent legitimate full Torch inspection. Further bounded corrections
are assigned; neither this candidate nor the green upstream PR is accepted
for merge. The paired Qwen/BigVGAN controller is also returned for trusted-leaf
reuse and fail-closed combined-transfer checks; no production/model result is
inferred from its fixture tests.

All 194 rows and the metadata-only 136 full / 58 unresolved snapshot remain.
Source/legal closure, independent real-weight CPU parity, final Apple CPU/
Metal/no-fallback, exact-head CI/security and separately authorized public
reconciliation are required. No model weights/configs/tokenizers/presets,
HF credential transfer, upload or Scaleway execution occurs in this checkpoint.

## 2026-10-03 import-gate and bulk-transfer review (07:55 UTC)

The estimate-only reply changed no implementation or external state. This
continuation independently revalidates the next actions against clean root
`72ac4a4c` and clean implementation candidate `52e4ddbe`, preserving the
already recorded actual code pass and failed source recovery separately.

Fresh GitHub readback of PR #152 is terminal zero at unchanged
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`: it remains draft, `CLEAN` /
`MERGEABLE`, with 76 successful and three skipped completed checks. Those
checks do not cover the new, uncommitted eight-file correction. Root reviews
its full diff through literal Git inspection with external diff and textconv
disabled and independently runs 29 tiny stdlib tests, terminal zero with no
skips. No real package, native payload or model is imported by these tests.

The correction improves installed RECORD hashes/sizes, import origins and
symlink handling, but installed RECORD is not publisher/archive authentication
or owner approval. The implementer confirms that direct `dump_reference.py`
does not consult the blocked manifest and that the installed-environment
branch of `dependency_guard.py` can import the official decoder despite
`BLOCKED_PENDING_PRIMARY_BYTES`. Root assigns a shared, stdlib-only execution
gate before both direct and callable official imports. Do not invent an
approved record or let a Boolean/environment variable or the in-process proof
object authorize this blocked scope. Additional deadline corrections cover
the real urllib socket path and cached-file reads; repeated streamed reads
must be counted and documented honestly. The later reported 30-test revision
still needs root's final review and independent verification before commit.

The paired Qwen/BigVGAN controller remains disabled. Root identifies a fresh
remote-shell dependency on undeclared BigVGAN cap constants and incomplete
receipt-digest checks, then assigns explicit cap bindings and real receipt
comparisons. Root's attempted independent self-test at initial SHA
`301ed26ecaeca8b666abb680ea5cb6c131af8dfe41d871026e3da10af0249cba`
prints a pass but exits two after the file changes during execution. It is
invalid evidence, not an accepted root pass. Freeze the revised file and bind
terminal exit zero to identical before/after hashes before activation.

The source-only recovery draft also remains disabled. Its first harness counts
one SCP command with 128 remote operands, not one connection. The reviewed
[OpenSSH portable SCP source](https://raw.githubusercontent.com/openssh/openssh-portable/master/scp.c)
(OpenBSD revision 1.278, dated 2026-10-01; read on 2026-10-03) reconnects inside
the remote-to-local operand loop. Require one validated packet/file transfer,
strict member/type/size/hash checks before materialization, bounded subprocess
and log handling, and reuse of the existing four collectors and original
full/profile receipt validators. Do not replace actual collection with an
arbitrary synthetic 128-row manifest or replay green Cargo for missing bytes.

Fresh independent VAST API session `52976` is terminal zero: `53978818` has
`instances=null`; complete account readback has explicit `next_token=null`,
zero Vokra-labelled instances and one unrelated instance left untouched.
No replacement VAST or Scaleway worker is provisioned. No HF credential is
forwarded and no artifact is downloaded, executed or published in this review.
All 194 rows, real-weight CPU/reference, final Apple CPU/Metal/no-fallback,
exact-head CI/security and separately authorized publication remain in scope.

## 2026-10-03 committed import gate and source-only production boundary (08:04 UTC)

Root independently reviews the complete eight-file PR #152 correction and
passes 32 tiny stdlib tests without skips, both audit/collector self-tests and
the documents-only dependency guard. Before/after helper hashes agree. Normal
five-stage commit hooks then pass, including format, forbidden-symbol,
first-party-lock, 190 fixture-pin and 306-file shell-lint checks. The isolated
candidate is clean at `b0add994b4d0350c338e60cb29b8165d8145fe32`, whose parent
is PR #152's `dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`. It is not pushed;
the earlier PR CI does not cover this correction.

The shared execution gate uses the existing complete audit contract before
direct/callable third-party imports. Current primary/native/owner blockers
remain effective. RECORD/origin integrity is not publisher authentication,
legal approval or numerical parity, and `NO_UPLOAD` alone is not an execution
ban for a future separately reviewed private scope.

The source-only draft is still disabled: its reported synthetic 128-file
packet does not execute the four production collectors or the original
full/profile receipt validators. Root assigns that missing production path,
corrects the candidate-checkout identity, and preserves the real distinction
between 111 DSM/Moshi held files and the 17 additional profile/bundle files.
Do not rewrite the full receipt to pretend it lists all 128 files. Require
production rejection of synthetic receipts and file caps before reads.
The paired controller also needs log-only caps: process file-size limits
would incorrectly constrain legitimate package/data writes. No new provider
allocation, model acquisition, HF upload or Apple execution occurs here.
All 194 rows and the remaining real-weight, legal and final Apple gates stay
in scope; an estimate is not a completion result.

## 2026-10-03 recovery headroom and exact-head CI readback (08:29 UTC)

Root verifies 112 obsolete synthetic cap-test files in 31 terminal paired-V2
fixture directories by exact path, regular-file ownership, size and complete
content SHA-256 before deleting only those files. The removed logical size is
553,547,736 bytes. Their zero/NUL-padding contents are reproducible from the
retained frozen mock constructors; they are not primary source or model data.
Recorded fixture process IDs are absent, no directory symlinks are found,
and controllers, logs, Git bundles, checkouts and all other files are retained.
Do not cite these deleted negative fixtures as retained evidence. Subsequent
filesystem readback reports 5,793,348 KiB available; the APFS availability
change is not attributed solely to the removed logical bytes.

The CLI's saved credential first returns a JSON 401 error despite exit zero.
Root's authorized retry uses the existing `.env` VAST key in the subprocess
environment only, without logging it, putting it in arguments or changing
credential settings. Individual readback then returns `error=false` and
`instances=null` for destroyed owned worker `53978818`. This individual
readback is not a new complete-account inventory. No replacement worker is
allocated in this checkpoint.

Fresh GitHub readback confirms PR #152 is open and `CLEAN` at unchanged
`dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, with 76 successful and three
skipped checks, all completed. The clean `b0add994` correction is still local;
those checks do not cover it. Root also re-runs the three model-free audit,
collector and documents-only guard self-tests successfully. Bounded source
recovery and model-free verification revisions remain under review, not
accepted production or model results. All 194 rows, independent real-weight
CPU evidence, final Apple CPU/Metal/no-fallback and publication gates remain
required.

## 2026-10-03 independent production-path tests and pre-allocation failure (09:06 UTC)

At root documentation head `4ba7d302`, both actual input checkouts are clean:
Qwen `df4c2dbf51ea4f9c5977d45ae5c2abecb6040fff` and BigVGAN
`af462cffda14e97b044d3ebd718b23c08572c841`. Root independently verifies
their pinned dependency files, the BigVGAN leaf and the safe provider wrapper.
The Qwen API work directory must not pre-exist; the corrected shared remote
path and that rejection case pass the frozen V7 offline test, session `81578`.

The activated paired V8 then passes offline session `51379`, but root finds
that conditional-call shell semantics can swallow an intermediate override
rejection. Frozen V9 fixes this with explicit returns. Root independently
passes V9's full offline session `30071` and verifies that even default-valued
provider/SSH/SCP/identity overrides exit two before allocation. V9 SHA-256 is
`48211e39f551c28ae143ebc4f519456037eeac2861fc1846adc5718822daae73`.

Actual V9 execution, function cell `3916`, is terminal two: both bundle
helpers use a raw commit SHA without an advertised ref and Git refuses the
empty bundles. The run stops before offer search or instance creation.
Retained log directory
`/private/tmp/vokra-qwen-bigvgan-model-free-v9.eCE7tM` contains
`cleanup_rc=0 instance=none`. Require bundle creation from `HEAD`, independently
verified against the fixed candidate SHA, and explicit helper failure returns
before retrying. The offline fixture did not cover this actual Git preparation
path. No new Qwen API, BigVGAN metadata, real-weight or Apple result is claimed.

Root also independently passes source-only V6 session `41726` at unchanged
SHA `1c14c4dececcff2e79396a6b250c9cda3760764f65ffe37ba76d101ad9bf5550`.
Its rendered remote leaf executes the real flatten and packet-building paths
over 111 invented full-source rows plus the 17 profile/bundle rows; original
strict validators reject those invented bytes. V7 session `8577` also exits
zero, but its operation-marker list is not proof that two builder bodies are
identical. Final process-group cleanup and bounded transport review remain
pending; there is no authenticated replacement source packet. Original V5
`318c1358...` was overwritten by its implementer and has no recovered backup;
do not cite it as a retained immutable script. V1–V4 and the later frozen
versions remain available.

The PR #152 transfer bundle independently verifies as complete history at
`b0add994b4d0350c338e60cb29b8165d8145fe32`: 57,385,353 bytes, SHA-256
`a88a00db99e49bff50b05f52464c919e65de257153e4797afb047b0df95362d7`.
Its V5 leaf requires all three pinned base scripts at their exact remote
paths and a new cold checkout. This is transfer preparation, not the actual
Linux 32-test/helper replay, PR CI, import authorization or publication.

No maintainer-local model acquisition/execution, workspace Cargo, HF credential
forwarding, artifact upload or Scaleway allocation occurs in this checkpoint.
The whole 194-row scope, owner/legal, independent real-weight CPU/reference,
final Apple CPU/Metal/no-fallback, exact-head security/CI and public-artifact
reconciliation remain required. Scaffold tests and failed execution preparation
do not promote any model row.

## 2026-10-03 SSH failure and local-space boundary (09:28 UTC)

At clean root head `830b5abc32bdab631bdacf952a695926e88f786e`, root
independently passes paired A10 offline session `35441` and verifies its
unchanged SHA-256
`e95aca152a8c399489efa85789825ed176f4d5cc5a3d3e1813e17e992d43c7fc`.
The corrected shared helper creates bundles from advertised `HEAD` and checks
the exact fixed commit; actual execution now gets past the old empty-bundle
failure. Function cell `3929` / session `33635` is terminal 255. Owned VAST
`53991834`, label `vokra-qwen3-tts-mf-v1-20261003T091040Z-19228`, has
128,480 MiB RAM and 200 GB disk, at the selected offer's $0.1237037037/hour.
Its API reports running, but the first SSH operation fails before remote
directory creation or model-free work. API running is not SSH readiness.
An unavailable direct port correctly selects the validated proxy endpoint;
that fact alone does not prove the cause of the SSH failure.

Retain `/private/tmp/vokra-qwen-bigvgan-model-free-a10.KGitcy`, including
the 57,705,238-byte candidate bundle, 57,734,788-byte BigVGAN bundle, provider
and destruction receipts. Cleanup records `cleanup_rc=0 instance=53991834`.
The individual API is null and the all-pages exact-label readback is empty.
Root independently confirms absence in session `1042`; the later fresh
session `16442` also exits zero and reports complete account pagination,
zero instances and no Vokra instances. No unrelated resource is modified.

Frozen A11 adds bounded SSH probes on the same instance, endpoint, identity
and known-hosts path. Its root offline invocation exits one before entering
the fixture because `mktemp` fails with `ENOSPC`: observed available space is
119,176 KiB. The old readonly assignment masks that failure and subsequent
directory/log writes fail at unintended root paths. No provider is called.
A12 explicitly validates temporary-directory creation and is prepared, not
independently accepted. Root additionally finds that conditional fixture
calls disable shell errexit: a readiness failure must explicitly return,
and the timeout regression must prove no directory/transfer/bootstrap work
occurs, not merely a later nonzero exit. A further narrow correction and
adequate local headroom remain required before any retry. Preserve logs,
authenticated sources and Git backups during any synthetic-data cleanup.

Root independently passes frozen source-only leaf V9 session `78284` at
unchanged SHA-256
`edef78ad919c83069e1aba8530e158032bbeb86ea877391dea6a92154acc6104`.
The actual rendered path rejects invented full/profile bytes and exercises
bounded transfer and child-process cleanup. This accepts leaf preparation
only; the source-only provider lifecycle remains a draft, and no replacement
authenticated 128-file packet is recovered. Do not rerun the already-green
workspace merely to recover source evidence.

A read-only Nano audit confirms that the existing source inspector and
dependency auditor already cover the required preparation; do not add a
duplicate inspector. The current exact Python closure's 35 review rows are
unresolved, current owner approval remains unsigned, real VAST API/native
closure facts and independent real-weight parity are missing, and the
historical Full-stamped Nano artifact still requires a corrected replacement.
Old weight sign-off does not approve this new dependency/operator scope.

No local model acquisition/execution, new real-weight verdict, HF upload or
Scaleway allocation occurs. The metadata-only 136 full / 58 unresolved split
is not an Apple-completion count; preserve the whole 194-row scope and all
remaining source/legal, real-weight CPU, final Apple/no-fallback, exact-head
security/CI and public-artifact reconciliation gates.

## 2026-10-03 bounded synthetic-copy cleanup (09:32 UTC)

After root documentation commit `ef01c02641bca97d7f10523236f7a012caeb0566`,
root independently verifies the two retained complete-history input bundles
and the retained successful synthetic counterpart. The model bundle is
57,926,048 bytes, SHA-256
`ff04d8b18a30db3d8d90dbafeaba40cc7ec40321d3bef7db41427ab75712cba9`,
at exact `db26d377f90c308e690e37a0871195c712d88a80`. The XCodec bundle is
57,600,230 bytes, SHA-256
`7d944ce4f6ef408a815860ad9dde380decbf0b79c4a0b48d50c42bb362a94ac9`,
at exact `e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`. Both are retained
under `/private/tmp/vokra-clean-heads-model-free-logs.OthKNz/self/`.

Root rechecks all four candidate repositories as clean, confirms UID 501
and non-symlink directories, then deletes only these two reproducible work
roots, deletion session `4557` terminal zero:

- `/private/tmp/vokra-clean-heads-model-free-logs.OthKNz/self/state-failed-leg/root/vokra-clean-heads-model-free-20261003/work`
- `/private/tmp/vokra-clean-heads-model-free-logs.OthKNz/self/state-preflight-failure/root/vokra-clean-heads-model-free-20261003/work`

Each work root was 357,904 KiB. Their input directories are empty; outputs,
case logs, origin shell, lifecycle markers, retained bundles and the complete
successful counterpart are not removed. The cases' completion outputs and
failure logs are synthetic results, not numerical or cloud verdicts; no
persisted live session/PID handle exists for those old cases. Source/Git
content can be reconstructed from the verified bundles. Root checks absence
of only the selected work roots and continued presence of both bundles and
case logs. Available space rises to 2,533,124 KiB immediately after cleanup
and is 2,529,856 KiB at this checkpoint, above the unchanged 2 GiB floor.

Frozen paired A13 SHA-256
`4b774ab6e561c923e5f2604df374719c73a1ede97d9ae506b2a1cfd1f569a807`
passes root syntax/hash checks. Its root offline test nevertheless exits one
before lifecycle cases: the temporary-directory-count regression enumerates
unrelated `/private/tmp` entries and encounters an OS permission denial.
Require task-prefix-only enumeration; do not relax filesystem permissions
or bypass hooks. A13 also still uses the A12 log prefix, so the regression
prefix must be aligned in the next frozen correction. No provider, model,
workspace Cargo, publication or Apple execution occurs in this checkpoint.
All 194 rows and the outstanding end-to-end gates remain in scope.

## 2026-10-03 repeated-failure investigation (09:55 UTC)

At root management head `37082099b1c9025fff938dfdf10cbcc4cfb85c72`,
the owner requests diagnosis and countermeasures rather than another blind
retry. Root holds further VAST allocations until the orchestration regressions
and failure-evidence recovery pass independent review. This does not pause or
reduce the full 194-row catalog objective.

### Confirmed causes and test blind spots

| Failure | Confirmed cause / evidence | Countermeasure and acceptance boundary |
|---|---|---|
| Raw-SHA bundle creation | A9 advertises no ref and stops before allocation. | A10 uses literal `HEAD`, verifies the expected commit and advertised ref, and gets past actual bundle preparation. Keep this pre-allocation gate. |
| Temporary directory failure | A11's `readonly name="$(mktemp ...)"` masks `ENOSPC`; subsequent paths are invalid. | Assign and check first, then mark readonly. Keep the 2 GiB free-space floor; never lower it to make a test pass. |
| SSH readiness exits on first probe | A14's `bounded` helper globally re-enables `set -e` before returning 255. Its conditional-function fixture suppresses that shell behavior. | A15 captures `wait` in an explicit conditional without changing caller flags; test the outer lifecycle in a normal `set -e` subshell, not only `if run_live`. |
| Actual collector failure is not diagnosable | A15 runs unconditional BigVGAN `sha256sum` under `set -e` with five required files absent. The remote job exits before later reason classification; successful-artifact recovery requires those missing files. | Record each leg's exit/reason before manifest operations; recover a separate capped diagnostic packet even when success manifests do not exist. Preserve the original nonzero result. |
| Source-only outer lifecycle remains unproved | Frozen source-only V5's exposed self-test never calls its provider lifecycle. Independent audit also finds incomplete partial-create parsing, reset transfer deadlines and absent failure-log recovery. | Keep activation disabled. Require actual shared-path create/readiness/leaf/transfer/signal/cleanup fault tests, bounded JSON/logs and complete exact-label ownership readback. |

Root independently reproduces the Bash semantics without files, models,
network or Cargo in function cell `4008`: readonly assignment around a failing
substitution returns zero; a function called as an `if` condition continues
after `false` and can report success; a helper that restores `set -e` aborts
its caller at return 255; the explicit `if wait` version lets the caller
capture 255 normally. These are orchestration failures, not evidence of a
numerical-parity or GPU-performance problem.

### Actual A14 and A15 outcomes

Frozen A14 SHA-256 is
`a7bfe1d304b2ef7f01c63a5c2eb3263659d9539b16c11849c5c65b3013df8df6`.
Root offline session `46533` passes, but actual session `55342` exits 255
with an empty readiness log. Owned `53994450` is destroyed; fresh independent
session `20551` confirms absence and complete account pagination, with one
unrelated instance left untouched. This is direct evidence that the old
conditional fixture missed a production-shell failure.

Frozen A15 SHA-256 is
`8461e717ff92b9ce6893cdb95a1b669bffe60ad9a1ad423089eded94459519c8`.
Root session `99682` passes its normal-shell regression. Actual session
`54931` reaches readiness on probe 17 after sixteen 255 results and clones
both exact candidate heads. It then exits one. Retain actual evidence in
`/private/tmp/vokra-qwen-bigvgan-model-free-a15.MSS6BW`.
Its remote log reaches BigVGAN manifest creation and names five absent output
files, but does not contain the original collector log or its recorded exit.
The exact collector failure cannot be reconstructed from those warnings;
do not invent a network, API, package or license cause. Earlier Qwen
audit/API failures could coexist, but the bootstrap-failure path returns
before BigVGAN and is not an explanation for reaching that manifest command.
Cleanup records zero for owned `53995269`; the retained individual receipt is
null and the complete exact-label receipt is empty. A fresh independent
post-destroy confirmation is still pending at this snapshot.

### Recovery correction under review; no new allocation

Luna prepares A16 at SHA-256
`4848dcd87b3275dce8c9b437570b3eb38192aa480949dd0d558b54e65255fb13`;
root does **not** accept it for activation. Review finds that BigVGAN logs
are selected from the wrong directory, logs above 1 MiB prevent the whole
diagnostic packet, only a generic remote-job exit is retained, manifest
failure still bypasses reason recording, and diagnostic failure overwrites
the original status. A bounded correction is delegated with normal-shell
bootstrap/audit/API/BigVGAN failure tests and exact log/reason assertions.
Syntax or agent self-test claims alone are not acceptance.

Local available space is 790,468 KiB at the last readback, below the unchanged
2 GiB floor. Large synthetic fixtures and new allocations are withheld.
Only independently authenticated, reproducible duplicate test work roots may
be removed; actual run evidence, outputs, source and complete-history backups
remain preserved. Root's process-inventory diagnostic is denied by the OS;
no alternate permission-bypassing inspection is attempted.

PR #152's dated independent readback at 09:41:48 UTC is open and **draft**,
remote head `dd6f0154acb0e6d7c2c47c25287ea6deaac1410a`, with 76 successful,
three skipped and zero pending/failing checks. These checks still do not cover
local `b0add994`. No model-free pass, source/legal approval, real-weight parity,
HF upload or Apple verdict is added by this investigation. Reuse the accepted
`52e4ddbe` code result rather than rerun green workspace Cargo for missing
source bytes. Final Scaleway/no-fallback and public-artifact reconciliation
remain required across the unchanged catalog scope.

## 2026-10-03 BigVGAN call contract and cleanup confirmation (10:03 UTC)

After management root-cause commit
`59c77d791a68753def81aa9424b86e995a11f6bf`, root examines the pinned real
BigVGAN leaf, not only the controller's replacement fixture. Frozen A15 line
572 creates `$work/big/evidence`; line 594 passes that existing directory as
`--evidence-dir` and its child as `--output-dir`. Real leaf SHA-256
`87184a9392d5e8d25cd126ab74d3abb206b99fe1c1bdc874d19733d936fb6b31`
lines 182–190 require both paths to be absent and non-overlapping, then create
the evidence directory inside the leaf. Root and an independent Luna source
review agree that the call violates both guards. If the preceding host/UV/repo
checks pass, the leaf rejects existing evidence before its own tests or
collector; even removing that directory alone leaves the overlap violation.
The controller's fake leaf previously used permissive `mkdir -p` and checked
neither condition. This is a concrete integration defect, not an inferred HF
network or collector defect. The original leaf exit/log still was not retained,
so do not claim its exact observed exception or that the actual collector ran.

A17 corrects the real invocation to fresh, separate `metadata-output` and
`metadata-evidence` siblings and gives the fixture both preconditions. Its
SHA-256 is
`279b6b890ec5a877183e25fb4f4eb6794d4dbc532f52e30fc874dcdb34069740`;
root passes syntax/hash and reviews the full delta. Activation is still
withheld: aggregate diagnostic excerpts can exceed their cap, first-failure
recording happens too late for early returns, and log-content/adversarial
packet assertions are incomplete. Those bounded corrections are delegated;
no production retry is performed on an unaccepted revision.

The two exit-code layers also differ: the real leaf expects its internal
metadata collector to return blocked code 2, validates that output, and then
returns 0 itself. Its `die` paths return 2. Treating outer leaf code 2 as
normal therefore misclassifies a guard or validation error. Require outer
leaf code 0 only; record any nonzero leaf result as a failed leg, with its
original code and log. Test missing output after a zero-return fixture
separately from an explicit nonzero leaf failure. These two contracts must
not be conflated in another permissive replacement fixture.

Fresh independent individual readback for `53995269` is null. Complete
account pagination returns success, one unrelated instance and no Vokra
labels. The retained 6,133-byte inventory receipt
`/private/tmp/vokra-qwen3-tts-cleanup-readback-20261003.json` has SHA-256
`466e8c8e222b739a17dd748b7af2bac2e6f0da76fac42873b8b748134bf458b5`;
root rechecks its hash and regular-file type. The unrelated resource is not
modified. Old empty-account snapshots must not replace this newer readback.

Root additionally authenticates and removes only the completed synthetic
`state-slow-cleanup` and `state-unknown` work roots under
`/private/tmp/vokra-clean-heads-model-free-logs.OthKNz/self/`, deletion
session `90521` terminal zero. Each is 357,904 KiB, with the same clean exact
model/XCodec heads and verified complete-history bundles recorded at 09:32.
Root confirms selected-path absence and preserved output directories; all
logs, case markers, input bundles and the successful counterpart remain.
Source/Git content remains reconstructible. Available space is 1,507,172 KiB
immediately afterward, still below the 2 GiB floor. A later independent
environment readback shows 4,940,764 KiB; that larger external change is not
attributed to these two deletions. No additional deletion is performed:
the audited remaining synthetic tree has no safe 650-MiB duplicate candidate,
and the actual V24 log tree has no `self` subtree. Offline regression can now
resume, but new VAST allocation remains held pending review and test evidence.
No public model row, legal approval, numerical result or Apple gate advances.

## 2026-10-03 independent failure-countermeasure verification (10:17 UTC)

Root reviews the complete A17-to-A18 delta and independently runs frozen A18,
SHA-256 `ded39f8c6115dbba5a34e92929471b6f642fae0dea99b0844863ff14898d075f`.
Session `5668` is terminal zero with the shared-path fixture-only PASS;
post-run hash is unchanged. A further narrow review identifies delayed Big
test failure recording: record its result before starting the next leaf so a
second failure cannot hide the first. Luna prepares a separate frozen A19;
A18 is not overwritten.

Final controller
`/private/tmp/vokra-qwen-bigvgan-model-free-vast-controller-a19-20261003.sh`
has SHA-256
`6920f84221eb61f09f9386f0e98c0a9a0002b11100fc2613190d1651a37ace63`.
Root reviews the full narrow delta, passes syntax/hash checks, then runs its
full offline self-test serially after A18. Session `56631` is terminal zero,
with `self-test PASS (shared run_live/remote_job/manifest/cleanup;
fixture-only)`, and its post-run hash is unchanged. Preserve root evidence in
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.KNFmPs`.

The normal `set -e` outer path now exercises bootstrap, audit, API,
missing-output-after-zero, collector error, leaf exit 2, Big-test error and
combined Big-test/leaf error. Assertions cover original exit/reason, log
contents, diagnostic recovery and cleanup/readback. Root directly reads the
combined case: tests exit 23, `first_failure=big_tests:23` appears once, leaf
exit 2 is also recorded, `big_reason=big_tests_exit` survives, and both actual
fixture log messages remain. The outer leaf accepts only zero. Other existing
regressions cover SSH 255-to-zero readiness, total readiness failure without
later transfer, malformed/oversized/linked packets, aggregate bounded log
excerpts, partial-create ownership and process-group cleanup. Each diagnostic
member gets a fixed 256-KiB budget, with original size and truncation markers;
the total is capped at 4 MiB and its archive at 8 MiB.

Acceptance is bounded to these reviewed orchestration corrections and offline
fixtures. No new provider allocation, network source acquisition, model,
workspace Cargo, HF upload or Apple execution occurs in this root-cause turn.
The separately audited source-only V5 still lacks its actual outer lifecycle
fault matrix and remains disabled; do not promote its validator-only tests.
Keep the accepted exact-head code evidence and the full 194-row goal rather
than rerun green Cargo or claim catalog completion from controller tests.
Available local space is 3,771,636 KiB at the last readback, above the unchanged
2 GiB floor; no further data cleanup is performed. Fresh owned-worker absence
and zero Vokra instances are recorded above, with the unrelated instance
untouched. The next paired VAST wave must bind these exact tested bytes and
retain failure diagnostics; it has not run at this snapshot.

## 2026-10-03 post-fix environmental checks (10:28 UTC)

At clean management head `ef9b9dd0701a075688925c04f8e1ceb85bb8e69e`,
the frozen, independently tested A19 bytes remain unchanged at SHA-256
`6920f84221eb61f09f9386f0e98c0a9a0002b11100fc2613190d1651a37ace63`.
Actual session `72498` terminates two during offer lookup: the restricted
network reports DNS resolution failure for the provider host. No create call
occurs. Retain controller evidence in
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.BoEFB9`; its cleanup record
is `cleanup_rc=0 instance=none`. This is not evidence of a provider outage
or a regression in the corrected remote job.

The network-authorized retry also terminates two, at `local_free_floor`,
before bundles, offer lookup or resource creation. Its retained directory
is `/private/tmp/vokra-qwen-bigvgan-model-free-a19.dMeWo9`, again with
`cleanup_rc=0 instance=none`. Available local space drops to 186,052 KiB
and later recovers to 2,023,580 KiB, still below the unchanged
2,097,152-KiB recovery floor. The reason for these external capacity changes
has not been established; do not attribute them to this job or delete
unrelated data to force allocation. No additional cleanup is performed.

An independent network-authorized read-only complete account query,
session `92970` terminal zero, succeeds with `all_pages_complete=true`,
one account instance and zero Vokra instances. The unrelated instance is
untouched. Neither actual attempt creates a worker or begins model-free
remote validation. Do not label them successful production verification.

The source-only implementer reports no V6 output, fixture, bundle, test or
provider mutation because of the capacity boundary. Frozen V5 remains
unchanged and disabled. Resume this independent correction only with safe
headroom and require its actual outer-lifecycle failure matrix; absence of
a new implementation is not acceptance. For the paired path, reuse frozen
A19 without another speculative rewrite, verify capacity before creation,
and use the already-approved network path rather than repeating the denied
lookup. Retain original exit/reason/logs and destroy only owned resources.
Do not repeat the accepted `52e4ddbe` workspace run to recover source bytes.
No legal, numerical, public-artifact or Apple verdict changes; all 194 rows
and the Scaleway-last gates remain in scope.

## 2026-10-03 recovery headroom and live paired replay (10:36 UTC)

Root resumes from clean `c025fabac59069e7e6ebcc9d94aa693be0838a7e`.
Independent cleanup inspection finds only two clean, completed synthetic
checkouts under `/private/tmp/vokra-clean-heads-model-free-logs.OthKNz/self/`
`state-success/root/vokra-clean-heads-model-free-20261003/work/`:
`model` at `db26d377f90c308e690e37a0871195c712d88a80` and `xcodec` at
`e6552853d5dcba0ca1bbe7e07f74914ea9a0f2cf`, approximately 350 MiB total.
Both have no untracked/ignored files or unique external test results; they
are owned regular directories with no symlink boundary. Root verifies the
retained complete-history bundles and their unchanged digests recorded at
10:03, then removes only those two work roots. Deletion is terminal zero;
selected-path absence and both retained bundles are checked. All case
outputs, logs, markers and backup history remain outside the removed roots,
so the checkouts can be reconstructed. The earlier retained-counterpart
statement describes the older snapshot, not a requirement to retain this
reproducible checkout indefinitely.

Available space immediately rises from 2,019,868 to 2,381,640 KiB.
A later readback shows 2,979,708 KiB; the additional external change is not
attributed to this deletion. An authorized read-only offer query, session
`69703` terminal zero, finds eight offers within the existing $0.20/hour
ceiling; no instance is created by that query. Root reconfirms frozen A19,
its leaf/wrapper hashes, both clean input heads and their complete-history
bundles. The operational launcher additionally reserves the unchanged
2 GiB recovery floor plus 115,422,927 bytes for bundles and 64 MiB for
bounded evidence before activating the tested controller.

Actual session `44590` is live on owned `54000184`, exact label
`vokra-qwen3-tts-mf-v1-20261003T103414Z-93402`, at approximately
$0.136296/hour. Retain evidence in
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.uvyJi3`. Its SSH readiness
log records eighteen 255 results followed by zero on probe 19; remote work
starts. Do not restart this live run or claim a paired pass from readiness.
The frozen scope is Qwen3-TTS dependency/API and BigVGAN primary metadata,
without models, HF credentials, workspace Cargo or upload. Recover small
authenticated output or failure diagnostics, destroy this owned worker and
storage, and independently verify absence after termination.

The separate source-only V6 correction is delegated as a small file-only
change, with no provider, bundle or large fixture generation. It remains
unaccepted until actual-path fault tests and root review. All 194 catalog
rows, owner/legal boundaries, real-weight CPU gates, final Scaleway
CPU/Metal/no-fallback gates and separate publication authorization remain.

## 2026-10-03 paired terminal diagnosis and independent cleanup (10:40 UTC)

The frozen A19 replay at the preceding checkpoint is now authoritative
terminal one, session `44590`. Its diagnostic receipt records `remote_rc=1`
and `diagnostic_rc=0`. Unlike the earlier incomplete-artifact failure, the
original per-leg reasons and capped logs are actually retained in
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.uvyJi3/diagnostics`:

- Bootstrap exits zero. Qwen dependency audit exits two with
  `license gate remains factually BLOCKED`; `first_failure=audit:2` is
  preserved. Do not treat that legal stop as authorization or invent a
  package/security conclusion from the single-line log.
- Qwen model-free API exits zero. Its log reports the all-variant validator
  digest `e4a10c6dbed47a8b2399bd119e6cf4be62fd777defebfbab454d14d40ccb9eb8`,
  and the summary records `checkpoint_load=NOT_PERFORMED`, `NO_UPLOAD` and
  pending owner approval. The aggregate success packet is rejected, so its
  underlying API JSON is not retained locally. This is a successful logged
  API leg, not a complete dependency/reference/parity packet.
- BigVGAN runs six unit tests; three error when tests or `self_test()` use
  `Path.cwd()` and attempt the variant manifest under `/root/tools/parity/`
  instead of the checked-out repository. The actual trace demonstrates a
  working-directory integration defect before those fixture assertions.
- Its collector/leaf exit two and log
  `sibling descriptor not bound to raw API`, followed by metadata-output
  validation failure. Three fixed metadata requests are recorded. The
  mismatching descriptor/raw fields are not in the recovered diagnostic
  packet, so the precise field-level cause is not established. Do not
  loosen binding or claim primary metadata acceptance.

Cleanup records zero for owned `54000184`; retained individual receipt is
`instances:null` and the exact-label receipt is complete and empty. Fresh
independent session `2993` terminates zero with individual absence, complete
account pagination, one unrelated instance and zero Vokra instances. No
unrelated resource is changed; owned worker/storage are not retained.

Root's attempted read-only source inspection of the BigVGAN tests/collector
and Qwen dependency-audit script is refused by the trusted local model-safety
hook. It is not rerouted through another path or delegated to evade that
denial. The logged causes above remain the verified boundary; detailed
source correction/review is still pending. No new allocation is made to
repeat unchanged inputs.

Source-only V6 is a separate unaccepted 97,832-byte patch. Root's full diff
review identifies still-missing actual outer-lifecycle fault tests, missing
pagination-key validation in recovery, diagnostic selection that rejects a
directory containing other artifacts, and signal/snapshot/reaping gaps.
Corrections and actual shared-path tiny fixtures (at most 32 MiB, no full
clones/bundles/provider/models/Cargo) are delegated at restored headroom;
do not promote syntax or fixture-injection wiring to acceptance. Frozen
V5/A19 stay unchanged. All 194 rows, source/legal boundaries, real-weight
CPU gates, final Scaleway CPU/Metal/no-fallback and separate publication
permission remain required.

## 2026-10-03 recurrence-prevention review (10:54 UTC)

Root starts this review at clean management HEAD
`06f0304a374f9121d51395aa4187b6787744c36f`. No new provider allocation,
model download/execution or Cargo run is made. The latest independent owned
resource absence evidence remains session `2993` at the preceding checkpoint;
it is not represented as a new provider query in this review.

Repeated attempts must be classified before another retry:

| Scope and evidence | Cause or remaining uncertainty | Action / acceptance requirement |
|---|---|---|
| Original V24: exact-head workspace green, only ten of 128 source files recovered | Serial recovery exhausts its 300-second budget | Recover the missing packet in one bounded bulk transfer; do not rerun already-green Cargo to retrieve bytes. |
| A19 BigVGAN: three of six tests error on `/root/tools/parity/` | Tests/self-test derive their repository root from the launch working directory | Correct the launch/root contract and add a regression from a different directory. Detailed source inspection remains refused by the trusted local hook; do not route around that refusal. |
| A19 BigVGAN: `sibling descriptor not bound to raw API` | The original raw fields were not recovered; the precise mismatch is unknown | Recover the small raw descriptor/validation diagnostics before selecting a field-level fix. Preserve identity checks; do not relax them to obtain green. |
| A19 Qwen: audit exits two, factual license gate blocked | This is a legal/factual gate, not evidence of a transient package-install error | Resolve the exact source/dependency facts and hash-bound decision separately. Generic autonomy approval does not invent sign-off. |
| Source-only V6/harness, not accepted | Previous fixtures could fail before the intended branch, or substitute a permissive mock for actual execution | Require positive controls, exact phase/reason/status assertions and the normal shared lifecycle. Synthetic transport acceptance must remain distinct from production primary-source acceptance. |

Root reviews harness candidate `424a5888` and subsequent `3a72da0c`,
and the V6 candidate read as
`870d88f03ccf406f4a1ee331ee407c9b7da73aeec73c23d45c885eb4a0349b0f`.
These hashes are review checkpoints only, not frozen accepted inputs.
The review identifies missing pre-inventory label argument, asynchronous
stdin preservation, mutation of caller `errexit`, recovery/cleanup budgets
not fully governed by one absolute phase deadline, incomplete regular-file
snapshot checks, broad diagnostic filename suffixes and skipped oversized
logs instead of capped excerpts. The positive fixture contract also remains
unfinished. None of these candidate findings is retroactively asserted to
have caused A19's original logged failures.

Two tiny, model-free shell probes independently confirm the language-level
risks: a 14-byte fixture piped to a background `wc -c` without explicit stdin
redirection is read as zero bytes, and a helper that sets `+e` then `-e`
changes a caller initially running without `errexit` to enabled. These probes
do not execute any controller, model or provider command. Their exact
corrections and integration assertions are delegated to the controller owner;
the independent harness owner is responsible for testing the same lifecycle,
not a second implementation. Root returns both candidates for correction.

Before any new source-only lease, require all of the following together:

1. Finish the known controller/harness correction checklist, agree the explicit
   tiny-fixture dispatcher contract, and freeze both files with matching hashes.
   Syntax/ShellCheck or a partially prepared test matrix is not acceptance.
2. Execute and independently review the complete offline matrix: positive
   transport and one-time SSH recovery; ambiguous creation and strict paginated
   ownership recovery; foreign/duplicate/malformed identities; permanent SSH
   failure; inbound and recovery deadline exhaustion; original leaf failure;
   bounded diagnostic excerpts; unsafe/oversized packet rejection; INT/TERM
   descendant termination and owned-only destroy/readback. Each negative case
   must demonstrate its intended branch, not merely return nonzero.
3. Assert actual bootstrap/leaf stdin bytes and hashes, explicit `root@host`
   transport, unchanged caller shell options, and production validators'
   rejection of synthetic source receipts. No invented full 111-file receipt
   or 128-file primary packet may be promoted to source readiness.
4. Retain the original failure, small diagnostics and final cleanup receipts.
   Keep tiny fixtures within 32 MiB total and the unchanged 2 GiB local recovery
   floor. No full test clone/bundle is needed for this regression matrix.
5. Retry only the corrected, frozen failing scope. Leave A19 and V5 unchanged;
   do not repeat the unchanged paired model-free wave while its original
   BigVGAN and Qwen blockers remain unresolved.

This review changes the acceptance plan and records concrete defects; it does
not claim that the returned implementation is fixed or that the fault matrix
has passed. No new cloud cost, model-row promotion, Apple result, owner/legal
sign-off or publication is claimed. Full-catalog scope remains all 194 rows;
the retained 136-full/58-unresolved code inventory is not blanket Apple parity.

## 2026-10-03 actual Bash 3.2 primitive verification (11:01 UTC)

This review begins at clean management HEAD
`abddd031a4a69c0836aa50c65e4fdc510a8510b2`. Root confirms the actual local
`bash` resolves to `/bin/bash`, version `3.2.57`. A tiny shell probe rejects
the candidate's dynamic file-descriptor syntax with
`exec: {diagnostic_fd}: not found`, while a reserved-FD probe preserves the
14-byte stdin fixture. A separate tiny probe reproduces `root: unbound
variable` from the harness's combined local declaration. Both are actual
shell evidence, without executing any model, upstream code or provider.

The controller owner corrects the Bash compatibility issue and supplies a
proper focused dispatcher. Root reviews its bounded-helper and primitive-test
bodies, then runs the frozen exact file
`/private/tmp/vokra-primary-source-bulk-v24-derived-outer-v6-20261003.sh`,
SHA-256
`a552efb20d445591985b66531cbada94dd04c46a41757b4a40b1991e2a8c5536`,
with `--self-test-bounded`. Session `75127` terminates zero. Hash checks before
and after match; Bash syntax also passes. The actual helper, not a substitute,
reports all of these passed:

- Stdin has 14 bytes and the expected SHA-256.
- Child exit 23 and timeout exit 124 are retained.
- Actual INT and TERM produce 130 and 143 respectively, with the marked
  descendants no longer alive.
- The caller's enabled/disabled `errexit` states are restored.

This accepts the focused bounded helper only. It does not prove that an
interrupt delivered to the entire outer controller reaps all active transfer
children or completes provider cleanup. It does not accept a synthetic
receipt as genuine source bytes, the full 128-file packet, API/dependency
closure, CPU/Metal parity or publication. The outer harness is still under
correction and has not passed its full matrix.

Root's remaining frozen-file review returns three bounded scopes to the
controller owner: a common nonblocking regular/stable-FD provider snapshot
loader with strict identity/row types; diagnostic archive overhead budgeting,
status/exit priority and exact truncation-marker validation; and explicit
production rejection of fixture-context environment overrides. The accepted
primitive bytes must remain reproducible while later corrections are made.
The independent harness owner continues normal-child integration using the
producer's explicit tiny-fixture builder, without authoring a second provider
lifecycle. No new VAST allocation, remote process or model execution is made.
The preceding independently verified owned-resource destruction remains the
cloud evidence; no fresh API query is claimed here. All 194 rows and final
Apple/no-fallback gates remain in scope.

## 2026-10-03 redundant transfer-bundle cleanup (11:07 UTC)

At management HEAD `95eb57a7`, the independent bounded-helper pass is
recorded and normal commit gates have passed. Subsequent space readback falls
to 1,983,424 KiB, below the unchanged 2,097,152-KiB floor. Root has no live
test/commit process or cloud worker. This is a local recovery-capacity hold,
not a model failure or proof that all external disk consumption is Vokra's.

An independent agent audits only the two transfer bundles from pre-allocation
failure directory `/private/tmp/vokra-qwen-bigvgan-model-free-a19.BoEFB9`.
Root then freshly checks the four files' regular/non-symlink type, UID 501,
size, digest and advertised sole HEAD, and verifies the retained actual A19
bundles' complete history. Session `36961` terminates zero after removing
only the old `candidate.bundle` and `bigvgan.bundle` in that failure directory.
No directory, checkout, model, unique raw receipt or log is deleted.

| Scope | Removed pre-allocation bundle | Retained actual-run bundle | Verified sole HEAD |
|---|---|---|---|
| Qwen candidate | 57,673,503 bytes; SHA-256 `f73279708fed493577ed591604e1f7df3ddb8852eccc8c96b809aa5ce781603b` | 57,674,781 bytes; SHA-256 `4b5a8dcec185553dcfcfac7cfcae07ea6e234ea3cacd1337d4699c79906b3218` | `df4c2dbf51ea4f9c5977d45ae5c2abecb6040fff` |
| BigVGAN candidate | 57,749,424 bytes; SHA-256 `3ce21770e686abd6b1fc4cdfd9d6cf3a06eb36ba0cd8e4d8f2879461801724fc` | 57,687,387 bytes; SHA-256 `dbf86e454b0522054e9e129f4879c501a4abd91955ab1856f8077a21aed27961` | `af462cffda14e97b044d3ebd718b23c08572c841` |

Retained bundles are in
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.uvyJi3`. The old failure
directory still contains `offers.json` and `controller.log`. Removed total
is 115,422,927 bytes (about 110 MiB). Packing bytes differ: the old archive
streams are not recoverable byte-for-byte, but their exact Git HEADs and
complete reachable history are reconstructible from the retained bundles.
This semantic redundancy, not a claim of matching file hashes, is the cleanup
decision. Immediately afterwards available space is 2,088,296 KiB; the later
readback is 2,087,740 KiB, still below the floor. No full fixture test or new
transfer is started, and no unrelated files are removed to force capacity.
The separately retained synthetic-model/XCodec bundles are not removed:
attempted ancestor checks against the A19 checkouts return invalid commit
names, so those are not proved redundant recovery copies.

The accepted primitive snapshot is retained at
`/private/tmp/vokra-primary-source-bulk-v6-a552efb20d445591985b66531cbada94dd04c46a41757b4a40b1991e2a8c5536.sh`
with its unchanged full SHA-256. A subsequent V6 candidate is frozen at
`ef18aaeb6bd863f3471c47eb6abeaecd079fa9313a89a183d65c94f754e4728a`;
its owner reports stable-provider-JSON and diagnostic-marker tests, but root
has not independently replayed those additions. Harness candidate `30178844`
is also unexecuted and still pins the earlier helper while its integration
contract is being corrected. Root's review returns mock-path, actual rendered
inline-command, diagnostic archive and signal/deadline integration gaps;
there is no full outer acceptance or permission to activate `--run`.
No provider/model/Cargo run occurs. All 194 rows, source/legal boundaries,
real-weight CPU gates and final Scaleway CPU/Metal/no-fallback remain required.

## 2026-10-03 independent regression review and bundle cleanup (11:23 UTC)

Root starts from clean management HEAD
`8b7f6b7956b4b1c675622600c47be36103acd431`. No live root test, transfer or
cloud process remains. The accepted `a552efb2` bounded-helper snapshot is
preserved; it does not accept later edits. Frozen candidates receive an
independent body review rather than being promoted from their authors'
reports:

| Reviewed candidate | Concrete finding | Required countermeasure |
|---|---|---|
| V6 `b125f414` | `transfer_bounded()` launches its supervisor in the background but has no subsequent PID tracking, wait/status propagation or shell-option restoration tail. | Restore the full lifetime contract and exercise the actual transfer helper, including nonzero status, deadline and parent interruption cases. |
| Harness `72ff24f2` | Its actual producer pin remains `a552efb2`, despite the reported `f472ee3a` update. Its inline-command matcher recognizes `python -S -c` but the allowed branch accepts only stdin `-`, rejecting the real bounded supervisor. | Bind the final reviewed producer hash and forward both exact stdlib invocation contracts without duplicating arguments or permitting unknown commands. |
| Same harness | Diagnostic execution uses nonexistent `remote-root/bin` rather than the prepared mock bin directory; parent-signal leaf cases fall through without marked live descendants. Stdin hashes are self-hashes, not a comparison with the expected rendered body. | Fix mock isolation, phase synchronization, finite watcher/reaping, marked descendant assertions and exact expected-body comparison. |
| V6 parent INT test | A background subshell is assumed to install a working INT trap. | Use a launch contract that resets inherited signal disposition and finite wait/cleanup; do not infer INT coverage from TERM coverage. |

The final finding has actual language-level evidence, not just inspection.
Root runs a tiny `/bin/bash` 3.2 probe: a background subshell installs an INT
trap, receives INT and nevertheless prints `parent_normal_exit`; its wait
status is zero and the trap does not run. This probe executes no controller,
provider, model or upstream code. Candidate `7f37a138` restores the transfer
tail on inspection, but runtime transfer/parent-signal acceptance is still
absent. Initialization signal races are also returned to the controller
owner. Both owners retain narrow, non-overlapping script ownership; root
does not author implementation fixes or activate `--run`.

Separately, independent inspection and fresh root type/UID/hash/sole-HEAD
checks establish that the following six old transfer bundles have complete
same-HEAD recovery copies in the retained actual A19 directory
`/private/tmp/vokra-qwen-bigvgan-model-free-a19.uvyJi3`. Root deletion session
`47796` terminates zero; only these six files are removed:

| Old trial directory under `/private/tmp` | Removed `candidate.bundle` | Removed `bigvgan.bundle` |
|---|---|---|
| `vokra-qwen-bigvgan-model-free-a10.KGitcy` | 57,705,238 bytes; SHA-256 `b505c6b80b169b879dc5837f65bad839c168c98f78961cae3f0806b815ecf8f8` | 57,734,788 bytes; SHA-256 `0d76ff88d74b7a5ff7f339728294c87694296dfa7b82988e7fc7fdc94fb70fde` |
| `vokra-qwen-bigvgan-model-free-a14.Zw2FIQ` | 57,676,361 bytes; SHA-256 `b0498b4adc1f198514d04bd239c1d48c15eb646ef91459c76db1556355e4376c` | 57,645,578 bytes; SHA-256 `60e55e2b35bd4b8caec3a9e33810f617ed90569262e32ba43884138182041c8a` |
| `vokra-qwen-bigvgan-model-free-a15.MSS6BW` | 57,634,313 bytes; SHA-256 `7d96ef9d4be4a2bfd78c741de2c1f3597cbee79f7998644fd180658777b55360` | 57,684,109 bytes; SHA-256 `9af87a9e43e4b92885d01e9231c4ed46dc4eff7806ccad021ae6720e5e6210fa` |

Removed total is 346,080,387 bytes (about 330 MiB). All original trial
directories, logs, readbacks and unique outputs remain. The retained A19
bundle digests and sole HEADs are rechecked and match the preceding cleanup
record. Their complete Git histories reconstruct the old contents; the
differently packed old archive streams are not recoverable byte-for-byte.
Immediately after deletion, free space is 2,061,892 KiB, still below the
unchanged 2,097,152-KiB floor. Subsequent readback at 11:19:51 UTC is
1,759,100 KiB and at 11:23:16 UTC is 1,684,024 KiB; these changes do not prove
that all other disk consumption belongs to Vokra. No full fixture generation,
new transfer, local model or provider allocation is started below the floor.
An additional bounded read-only audit identifies large task directories but
does not prove them disposable; unique clones, logs and caches remain intact
pending exact backup/use checks.

The unchanged A19 BigVGAN working-directory and raw-metadata binding failures,
and Qwen license stop, remain separate unresolved source/fact gates. The
denied local source inspection is not rerouted or bypassed. No fresh cloud API
query, legal approval, primary-source packet, model-row advancement or Apple
result is claimed. The retry gate remains: corrected frozen inputs, genuine
primary validators, independently passed focused/full lifecycle tests and
sufficient recovery headroom. Preserve already-green exact-head Cargo rather
than repeating it. All 194 public rows and final Scaleway CPU/Metal/no-fallback
verification remain in scope.

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
