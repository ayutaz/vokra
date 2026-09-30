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
