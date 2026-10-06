# VibeVoice Realtime 0.5B reference contract

The inspector portion of this directory records the original model-free
contract used by `tools/parity/vibevoice_realtime_0_5b_inspect.py`; the
dedicated runner below now records a separate narrow official real-weight
reference.

## 2026-09-30 supersession note

Earlier header-only/model-free statements in this historical document are
superseded for checkpoint identity by the VAST evidence: the complete pinned
`model.safetensors` payload was SHA-256 authenticated, the offline conversion
and native binder accepted the authenticated tensor topology, and the official
CPU/CUDA narrow reference completed against those real weights. The dated
model-free receipts below are intentionally retained as history. This does not
claim complete acoustic encoding, waveform synthesis, Rust numerical parity,
Apple CPU/Metal parity, or publication eligibility.

## 2026-10-06 PR #169 security-closure integration

This PR #182 preparation integrates the reviewed PR #169 installed-closure
collector, its focused regression tests, and its CPU-only project lock at
`b3e9d5c1cd2c75bcc0b2cbe81ce8b115f00e3690`. The lock pins
`torch==2.13.0+cpu` through the explicit PyTorch CPU index, excludes CUDA and
`nvidia-*` payload rows, and remains
`IDENTITY_ENRICHED_VAST_NOT_RUN_OWNER_REVIEW_REQUIRED`. The collector is an
evidence collector only: it does not install, import, download, execute, or
publish a model, and unresolved license/native-payload facts remain
`OWNER_REVIEW_REQUIRED_NO_UPLOAD`.

## 2026-10-07 strict JSON hardening supersession

The current collector now has 33 local regression tests covering fail-closed
duplicate JSON object keys at the top level, in the platform object, and in
selected artifact identity objects, for both equal and differing values. The
regressions also cover malformed-manifest stderr, no report or sidecar creation,
and preservation of pre-existing report/sidecar bytes. This is stdlib-only,
synthetic-input evidence; it does not establish that the current 41 locked
distributions were audited or installed.

The earlier 31-test collector originated in PR #169; the V16 actual model-free
run at exact source HEAD `3f82c214` is a separate historical scope. Neither
must be combined with this new 33-test local proof. Owner, license, runtime,
numerical-parity, and Apple CPU/Metal review remain unresolved, and the
collector remains `OWNER_REVIEW_REQUIRED_NO_UPLOAD`.

The PR #182 streaming/native reference responsibility is retained. This
integration does not alter `NATIVE_PARITY.md`, `STREAMING_REFERENCE.md`,
`export_preset_cache.py`, `probe_dynamic_cache_compat.py`, `run_reference.py`,
or `run_streaming_reference.py`; the broader streaming and runtime sections
below remain historical/current contract documentation as previously scoped.

Model-free focused checks (stdlib only; no package sync or model execution):

```text
uv run --no-project --no-sync --python 3.12 python -B -S \
  tools/parity/vibevoice_realtime_0_5b_reference/test_audit_installed_closure.py
uv run --no-project --no-sync --python 3.12 python -B -S \
  tools/parity/vibevoice_realtime_0_5b_reference/test_lock_contract.py
```

The first command exercises the 31 collector tests and the second exercises
the five CPU-lock contract tests. These checks do not establish installed
closure, license sign-off, real-weight parity, Apple CPU/Metal parity,
publication eligibility, or authority to run upstream.

## Fixed Carter preset cache bridge (inspection-only)

`export_preset_cache.py` is a VAST/Linux x86_64-only exporter for the fixed
official demo preset `demo/voices/streaming_model/en-Carter_man.pt`. Before
importing `torch` or `safetensors`, it requires the clean Microsoft source
checkout at `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`, the authenticated
4,256,002-byte Git blob
`1d795ef667e6641eecb8b22452bb853b089bfdbe`, and payload SHA-256
`a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897`.
The sparse VAST source checkout may use a separately staged fixed file through
`--preset-path`; its filename, size, Git blob, and payload hash remain fixed.

The load is exactly the upstream demo's `weights_only=True` operation with
only `BaseModelOutputWithPast` and `DynamicCache` safe globals. The exporter
does not construct a model, run a forward pass, decode audio, download, or
upload. It writes a new `cache.safetensors` and deterministic `manifest.json`
with classification `INSPECTION_ONLY` / `NO_UPLOAD`, preserving all four
outputs (`lm`, `tts_lm`, `neg_lm`, `neg_tts_lm`). Source KV tensors are checked
as `[batch,kv_head,position,head_dim]` (the observed source tensors are
BFLOAT16) and explicitly transposed to native
`[position,kv_head,head_dim]`; the observed hidden/cache lengths are
`lm=108/108`, `tts_lm=316/316`, and both negative branches `1/1`. These are
schema facts, not synthesis or voice-consent evidence. A
historical `DynamicCache` shape/API that cannot be inspected through these
explicit lists is a loud refusal, not a compatibility shim.

The native `vokra_models::vibevoice_streaming::preset` parser requires an
external expected manifest SHA-256, checks exact tensor names/shapes/F32
payload hashes/finiteness, and stages positive and negative paired imports on
empty language branches. This is a structural cache bridge only: voice
consent/redistribution rights, independent waveform CPU/Metal gates, and
production synthesis/parity remain unproved.

Model-free self-test (stdlib-only):

```text
uv run --no-project --python 3.12 python \
  tools/parity/vibevoice_realtime_0_5b_reference/export_preset_cache.py --self-test
```

Controlled VAST export (source-only inspection; no model construction or
forward):

```text
VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python \
  tools/parity/vibevoice_realtime_0_5b_reference/export_preset_cache.py \
  --source-root /root/VibeVoice --preset-path /root/en-Carter_man.pt \
  --output-dir /root/realtime-carter-preset
```

## 2026-09-30 VAST narrow-reference receipts

Both runs used the exact checkpoint/source contract above and three timed
repeats. BF16 measured CPU median `16.1034 s` and RTX 4090 CUDA median
`0.01907 s`, but the global max absolute difference was `0.25`; the unchanged
provisional `0.05` guard therefore selected CPU. The packet SHA-256 is
`4c3590f5ef5b2bcc5f67b228dc8ce263ac95c5c579709fddfb2483c94a05b5a9`.

FP32 measured CPU median `23.4959622710 s` and RTX 4090 CUDA median
`0.0172316080 s`; global max absolute difference was
`4.57763671875e-05`, so the same `0.05` guard passed and selected CUDA. The
packet SHA-256 is
`af87fbc9959da3249971a637690bd00b890997647ff566f8a8a1ea8bdf2866e4`.
These are narrow same-workload device-selection receipts, not Rust parity or a
release tolerance; the BF16 guard remains unchanged.

## Official real-weight reference runner

`run_reference.py` is a VAST-only runner for a deliberately narrow real-weight
operation. It imports the pinned Microsoft implementation from a clean checkout
at `94da20d98b2fa7688e9cbfaf7692ddb4954f7600` and calls its
`forward_lm`, `forward_tts_lm`, and `model.acoustic_connector` methods. It does
not contain a copied model implementation and has no CPU fallback for a failed
official import. The checkpoint is required to be the complete pinned
`model.safetensors` payload (`605` tensors, SHA-256
`7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514`) from HF
revision `6bce5f06044837fe6d2c5d7a71a84f0416bd57e4`.
The supplied `config.json` is also authenticated before model construction:
2117 bytes, SHA-256
`caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57`.
The source checkout must be clean as well as at the pinned Git revision.
The runner disables Python bytecode writes before importing the upstream source,
so repeated runs do not create untracked `__pycache__` files. A pre-existing
dirty checkout is still rejected and must be cleaned on VAST before execution.

The pinned Realtime checkpoint is decoder-only for the acoustic tokenizer. Its
official state dict therefore omits `model.acoustic_tokenizer.encoder.*`. The
runner permits exactly that missing prefix, records the sorted-name SHA and
count in `reference.json`, and rejects every other missing or unexpected tensor.
The selected official `forward_lm`, `forward_tts_lm`, EOS-classifier, and
acoustic-connector calls do not invoke the acoustic-tokenizer encoder. This is
not evidence for acoustic encoding, synthesis, or complete checkpoint binding.

The fixed probe uses token IDs `[1, 2, 3, 4]`, a four-token attention mask, and
a deterministic `[1, 1, 64]` acoustic latent. It writes the independent
official `forward_lm` `lm_last_hidden_state`, EOS logits, TTS hidden state,
acoustic-connector output, and a JSON packet containing source, checkpoint,
input, runtime, timing, shape, finiteness, and SHA-256 metadata. The LM hidden
state is captured directly from the official `forward_lm` return before it is
spliced into `forward_tts_lm`; it is not reconstructed from the TTS output.
When CUDA is available, the packet also contains separate CPU and CUDA copies
of each small output plus per-output diagnostics: absolute difference, relative
difference with a `1e-6` denominator floor, differing-element count, shape, and
finite status. The selected-device aliases are retained separately.
This is an official narrow reference run, not Rust parity or a complete
synthesis claim. The packet must remain `NO_UPLOAD` until the separate license,
provenance, and model-zoo gates are complete.

The runner measures the same workload on CPU and CUDA when CUDA is available.
CUDA is selected only when its median is lower than CPU and every output stays
within the fixed provisional `5e-2` comparison guard; otherwise CPU remains
selected. This guard is only a same-workload device-selection safety check. It
is not an independent numerical reference, a Rust parity tolerance, or a
release gate, and no performance claim is valid without the emitted timing and
diagnostic fields. A large CPU/CUDA difference must be investigated from the
paired outputs; it must not be resolved by widening this guard without
evidence.

The checkpoint tensors are BF16. The runner retains BF16 as the default compute
dtype and also accepts `--dtype float32`; FP32 is a cast of the same
authenticated checkpoint after exact state-dict binding, not a second fixture.
The selected dtype is recorded in `reference.json`. CUDA is still selected only
when the unchanged guard passes; changing dtype does not widen that guard.

The dedicated `uv.lock` is generated and pinned. On the exact VAST tree, commit
`7e11e027`, `uv sync --frozen` succeeded with 56 installed packages. A fresh
model-free official-source API smoke on that same pinned source and environment
also passed as `AUTHENTICATED_API_SMOKE`; its JSON evidence SHA-256 is
`1d5f9d037ef15e8cded3db06d323d86de4ef5e08bd9fc52e00f96655e241a189`. The
smoke imported and inspected the official API only: it did not download a
model, construct a model, execute a checkpoint, generate parity numbers, or
publish an artifact.

The installed-closure audit for the same project and lock is recorded at
`a2d2939ae7dddff33eade14b4eb70ccc6a7af2b732617c6de2531c43ec5bcfc6` and remains
`OWNER_REVIEW_REQUIRED/NO_UPLOAD`. It covers 56 installed packages and reports
missing bundled license files for `safetensors`, `tokenizers`, `tqdm`, and
`triton`. Owner/primary-source review is therefore still required before any
real-weight replay or publication. The command below is the controlled replay
command, not an authorization or assertion that replay is currently cleared:

```text
uv run --frozen --python 3.12 --project tools/parity/vibevoice_realtime_0_5b_reference python \
  tools/parity/vibevoice_realtime_0_5b_reference/run_reference.py \
  --source-root /root/VibeVoice \
  --config /root/realtime-checkpoint/config.json \
  --weights /root/model.safetensors \
  --dtype float32 \
  --output /root/realtime-reference
```

#### 2026-10-07 current-lock supersession

The preceding 56-package receipt is historical evidence for the earlier VAST
tree and lock (`7e11e027` and the superseded lock identity); it is not proof for
the current lock. The current source records lock SHA-256
`34f58e53b5c79ed96853c2a4b6f9b6b1eaf12066f010cd23ddffae2b816b3797`, with 42
package rows: one repository-root virtual row and 41 registry rows. The lock
contains 63 candidate wheel artifacts across the approved PyPI and explicit
PyTorch CPU indexes. A fresh selected-wheel manifest must choose exactly one
Linux x86_64 / CPython 3.12 wheel for each of the 41 registry rows and bind its
absolute archive path, filename, byte count, and SHA-256 to the current lock.
Manifest preparation is a separate mechanical archive step; this collector
does not download, install, import, or select wheels.

On a disposable Linux x86_64 VAST worker, first provide an already existing
CPython 3.12 venv and the separately prepared selected-wheel manifest. Do not
let this command create a venv, synchronize a project, resolve dependencies,
or acquire packages. Bind every path explicitly before running the stdlib-only
collector:

```text
REFERENCE_VENV=/root/realtime-reference-venv
REFERENCE_PY="$REFERENCE_VENV/bin/python"
REFERENCE_PROJECT=/root/vokra-reference-source/tools/parity/vibevoice_realtime_0_5b_reference/pyproject.toml
REFERENCE_LOCK=/root/vokra-reference-source/tools/parity/vibevoice_realtime_0_5b_reference/uv.lock
REFERENCE_COLLECTOR=/root/vokra-reference-source/tools/parity/vibevoice_realtime_0_5b_reference/audit_installed_closure.py
REFERENCE_SITE="$REFERENCE_VENV/lib/python3.12/site-packages"
REFERENCE_SCRIPTS="$REFERENCE_VENV/bin"
REFERENCE_MANIFEST=/root/realtime-selected-wheel-manifest.json
REFERENCE_SOURCE=/root/VibeVoice
REFERENCE_OUTPUT=/root/realtime-reference-dependency-audit.json
REFERENCE_BOOTSTRAP_DIR=/root/approved-clean-uv-bootstrap
UV_BOOTSTRAP_PY=/absolute/path/to/approved-clean-uv-managed-cpython312

cd "$REFERENCE_BOOTSTRAP_DIR"
env -u VIRTUAL_ENV -u CONDA_PREFIX -u PYTHONHOME -u PYTHONPATH \
  -u PYTHONSTARTUP VOKRA_PUBLISH_ON_VAST=1 uv run --offline --no-project --no-sync \
  --python "$UV_BOOTSTRAP_PY" "$REFERENCE_PY" -B -S "$REFERENCE_COLLECTOR" \
  --project "$REFERENCE_PROJECT" \
  --lock "$REFERENCE_LOCK" \
  --site-packages "$REFERENCE_SITE" \
  --venv-root "$REFERENCE_VENV" \
  --scripts-root "$REFERENCE_SCRIPTS" \
  --selected-wheel-manifest "$REFERENCE_MANIFEST" \
  --source-root "$REFERENCE_SOURCE" \
  --output "$REFERENCE_OUTPUT"
```

The `UV_BOOTSTRAP_PY` value must be an approved clean UV-managed CPython 3.12
interpreter, not the target venv interpreter, and `REFERENCE_BOOTSTRAP_DIR` must
be outside the target venv/project (with no implicit `.venv` selected). The
environment is scrubbed before invoking UV so inherited virtualenv/conda and
Python path/startup hooks cannot redirect interpreter discovery. UV initializes
the interpreter used for `--python` discovery before launching the requested
command, so the target venv must be passed only as the explicit command with
`-B -S`; otherwise an unreviewed `.pth` can execute before the collector. This
behavior is visible in the pinned UV interpreter path (see
<https://raw.githubusercontent.com/astral-sh/uv/0.12.5/crates/uv-python/src/interpreter.rs>).

The collector requires Linux x86_64 CPython 3.12, checks that the explicitly
invoked interpreter is the venv interpreter, and consumes only the pre-existing
selected wheel archives and installed files. It records exact locked
distribution Name/Version/source, project and lock identities, official
VibeVoice Git revision and clean-tree state, metadata/license files, and native
payload inventory/hashes where the bounded hash policy permits. It performs no
third-party import, dependency acquisition, model download, or model
execution. Its result intentionally remains `OWNER_REVIEW_REQUIRED_NO_UPLOAD`,
not an approval; independent primary-source license/native-payload review and
owner decision remain required before replay evidence can be treated as
execution-authorizing.

The command is intentionally not a local verification command. The full
checkpoint is larger than the 2 GB VAST threshold and must never be downloaded
or executed on the maintainer Mac.

The authenticated import closure for this runner is `torch`, `numpy`, `tqdm`,
`transformers`, `safetensors`, and the official `diffusers` scheduler. The
official package's UI/server and optional audio dependencies (`accelerate`,
`gradio`, `av`, `aiortc`, `uvicorn`, `fastapi`, `pydub`, `ml-collections`,
`absl-py`) are not imported by this fixed class path and are intentionally not
added. Neither are `librosa`, `soundfile`, `soxr`, `scipy`, `numba`, or
`llvmlite`; adding them would expand the reference closure without evidence
that the selected official forward path needs them.

## Patched Transformers compatibility gate

The official VibeVoice `pyproject.toml` at the pinned source currently declares
`transformers>=4.51.3,<5.0.0`. The isolated reference lock uses
`transformers==5.10.4` because dependency review rejects 4.51.3 for
`GHSA-xrqw-3rrv-vx5w`; 5.10.4 is above the patched minimum `5.10.0`. This is
an intentional source-declared-range exception and is not treated as compatible
by assumption.

Before any real-weight replay, run the model-free import/API smoke on the same
VAST environment and pinned clean source checkout:

```text
uv run --frozen --python 3.12 --project tools/parity/vibevoice_realtime_0_5b_reference python \
  tools/parity/vibevoice_realtime_0_5b_reference/run_reference.py \
  --compatibility-check --source-root /root/VibeVoice
```

The check imports only the official config/model classes, inspects the
`forward_lm`/`forward_tts_lm` signatures and `DynamicCache(config=...)`, and
constructs neither a model nor a checkpoint. The exact VAST run now reports
`AUTHENTICATED_API_SMOKE`; this authenticates only the import/API surface, not
model execution, numerical parity, or publication. A model-free package
inventory of the 5.10.4 wheel shows that
`transformers.models.qwen2.tokenization_qwen2_fast` is absent, while the pinned
official source imports it directly. The runner therefore installs an explicit,
temporary namespace shim that maps only that removed module and translates the
old `vocab_file`/`merges_file` constructor names to Transformers 5's native
`Qwen2Tokenizer`/`TokenizersBackend`. It does not emulate tokenization or model
execution. During the same import-only window, Transformers 5.10.4 also has a
native `VibeVoiceAcousticTokenizerConfig` with the same class name as the
older official source. The runner permits only that exact source
config/model pair to register with `exist_ok=True`, restores the registration
method immediately, and records the scoped override in the smoke packet. No
other auto registration is relaxed. The shim is recorded in the smoke packet as
`qwen2_fast_import=COMPATIBILITY_SHIM`, with the registration override recorded
as `SCOPED_VIBEVOICE_ACOUSTIC_TOKENIZER_OVERRIDE`. The exact VAST import/API
smoke passed, so the compatibility status is now
`AUTHENTICATED_API_SMOKE`; this does not claim model execution or numerical
parity.

The first VAST smoke after the 5.10.4 lock update reached the official source
but failed at that exact auto-registration collision (`AutoModel.register`),
before any model construction or checkpoint access. That historical failure is
superseded by the exact-source, exact-lock smoke receipt above; the registration
scope remains explicit and narrow.

The earlier VAST `uv sync` and dependency audit were for Transformers 4.51.3
and are invalidated by this lock change. The refreshed VAST sync and model-free
smoke are now recorded, but the refreshed installed-closure audit remains
`OWNER_REVIEW_REQUIRED/NO_UPLOAD` because the four bundled license files named
above are missing. No model was downloaded or run for this compatibility
update, and no publication is authorized.

## Fixed upstream identities

The gate binds all evidence to these immutable revisions:

- HF model: `microsoft/VibeVoice-Realtime-0.5B`
  `6bce5f06044837fe6d2c5d7a71a84f0416bd57e4`
- VibeVoice source: `microsoft/VibeVoice`
  `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`
- Transformers package: `5.10.4` (PyPI artifacts are hash-pinned in
  `uv.lock`; the pinned source/API smoke is `AUTHENTICATED_API_SMOKE`, with
  evidence SHA-256 `1d5f9d037ef15e8cded3db06d323d86de4ef5e08bd9fc52e00f96655e241a189`)
- Base tokenizer: `Qwen/Qwen2.5-0.5B`
  `060db6499f32faf8b98477b0a26969ef7d8b9987`

The VibeVoice source checkout is authenticated by exact Git blob identities
for the streaming processor, acoustic tokenizer processor, text tokenizer,
and acoustic tokenizer implementation. The gate additionally checks source
markers to document and enforce the expected streaming API surface; the exact
Git blob identities remain the source of file identity and provenance.

## Tokenizer roles

Only the following selected Qwen tokenizer files are structurally inspected;
the recursive server walk is still authenticated in full:

| File | Model-free role contract |
| --- | --- |
| `tokenizer_config.json` | Qwen2 tokenizer configuration with a positive `model_max_length` |
| `tokenizer.json` | Fast-tokenizer JSON whose model is a non-empty BPE vocabulary and merge table |
| `vocab.json` | Non-empty token-string to non-negative integer map |
| `merges.txt` | UTF-8 BPE pairs with no duplicates; the fixed snapshot is headerless, while an optional `#version: 0.2` line is tolerated |
| `LICENSE` | Non-empty text retained for a separate license review |

Tokenizer model weights are not selected or downloaded. The tokenizer license
and redistribution decision remain `SEPARATE_REVIEW_REQUIRED`.

The upstream text tokenizer defines slow and fast Qwen2 variants and adds
`<|vision_start|>`, `<|vision_end|>`, and `<|vision_pad|>` as additional
special tokens. The pinned source records their Realtime speech boundaries as
token strings; the exact pinned tokenizer sidecars authenticate their IDs as
`151652`, `151653`, and `151654`, respectively. The Rust
`vokra_models::vibevoice_streaming::tokenizer::VibeVoiceRealtimeTokenizer`
primitive accepts only the four byte-authenticated sidecars below and reuses
the first-party byte-level BPE implementation; any size, hash, JSON role, or
GGUF metadata drift fails closed:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer_config.json` | 7,228 | `c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9` |
| `tokenizer.json` | 7,031,645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |

The primitive's `streaming_text_ids` applies the upstream streaming boundary
(`text.strip()` followed by one newline and no added special tokens).
Historical tokenizer evidence at exact implementation HEAD
`89a211738705f603a88b6996d4151cea689ae6db` is bound only to that commit and
must not be reused as proof for a later rebased or integration HEAD. That
disposable VAST run compiled the Rust package, bound the exact sidecars, and
matched independently obtained pinned Transformers Qwen2TokenizerFast IDs for
representative ASCII, Unicode, whitespace, and punctuation inputs. Reserved
speech-boundary literals are rejected by policy; their official IDs are
checked separately against the authenticated sidecar records. A fresh
integration-HEAD receipt is required for the branch that carries this
contract. These are model-free tokenizer results, not model execution,
synthesis, or full parity.

## Streaming input/output contract

The fixed upstream processor intentionally rejects the ordinary `__call__`
path for realtime input. Streaming callers must use
`process_input_with_cached_prompt(...)` with one text input and a non-null
cached prompt. The model-free source contract records the output field names
`input_ids`, `tts_lm_input_ids`, `tts_text_ids`, `speech_tensors`,
`speech_input_mask`, `attention_mask`, `tts_lm_attention_mask`, and
`speech_masks` where applicable.

The acoustic processor requires audio, uses the upstream 24 kHz configuration,
normalizes to the configured target, and returns its processor field as
`audio`. The realtime preprocessor declares a speech compression ratio of
3200; speech mask lengths are therefore derived from the upstream ceiling
operation. The pinned `process_input_with_cached_prompt` path itself sets
`speech_inputs` to `None`; the native
`vokra_models::vibevoice_streaming::state` module therefore binds only its
model-free cached-prompt text boundary. It takes the two authenticated
hidden-state lengths, authenticates the VibeVoice Fast tokenizer's
`<|image_pad|>`-based `pad_id` from the exact sidecars (this intentionally
differs from the ordinary Qwen `pad_token` field), emits pad-filled pseudo
LM/TTS-LM IDs, and tracks newline-terminated text steps without owning a KV
cache. Audio preparation, sampling-rate conversion,
staged model execution, EOS/diffusion/acoustic decoding, and output numerical
parity are not implemented by this gate. The state module must not be read as
evidence that synthesis or real-weight parity is complete.

## 2026-10-07 source capability supersession

At the current source, the production Realtime composite, sampler, acoustic
connector, and causal decoder select CPU or Metal through the first-party
`Compute` registry and reject uncovered backends before binding. The legacy
model-free generation control plane remains CPU-only. The learned head and
decoder stages use the selected backend; the DPM scheduler remains explicit
host control and is not described as a GPU kernel. This source capability is
not Apple hardware execution or CPU/Metal numerical parity; those remain
separate authorized gates.

## Native prediction-head boundary (historical 6a baseline wording)

The CPU-only wording preserved in this section is historical documentation
from exact source HEAD `6a937b7782a5bd0b0d4aa3f28298c7942578a048`; it is stale
relative to that same HEAD's production CPU/Metal dispatch and must not be read
as a claim that the source lacked Metal support. The current source capability
and host-controlled scheduler distinction are recorded above.

The native `vibevoice_streaming::diffusion` module implements one
source-derived prediction-head forward step on an explicitly selected
`Compute` backend. Its small scalar-oracle test checks the calculation
contract only. A fixed range read of the HF safetensors header is now
authenticated: the 79,432-byte JSON header declares all 605 tensors as BF16,
and the 26 `prediction_head` tensor names/shapes match the native loader in
PR #160. The authenticated range SHA-256 is
`73c4658be17469d62e22a0f4b7f042cc10aa83d409055a5e3a350d5b8d8f26cb`.
This paragraph is a historical header-only receipt and is superseded by the
2026-09-30 full-payload SHA-256 authentication and VAST conversion/binder
evidence recorded at the top of this document. It remains useful for the
descriptor-only provenance, but it is not the current checkpoint identity.

The native `vibevoice_streaming::sampler::sample_vibevoice_realtime_cfg`
boundary now contains the fixed 20-step CPU-only CFG loop and an explicit
non-CPU rejection. Its tests use a deterministic synthetic predictor to check
branch order, shared latents, scheduler reset, and 20 steps. This is a
model-free contract test only: it is not an official numerical reference,
real-weight CPU parity, Apple CPU/Metal parity, or a full acoustic synthesis
route. The tokenizer, staged language-model execution, and publication gates
remain separate; the acoustic decoder boundary is described below.

At exact implementation commit `21dded9b7b0630266ee6c223b87d5ff7e569e2bc`,
a disposable VAST instance ran `cargo test -p vokra-models --lib
vibevoice_streaming::diffusion` (5 passed) and `cargo test -p vokra-models
--lib vibevoice_streaming` (20 passed, 2 ignored because the authenticated
Qwen sidecars were absent). Both used Rust 1.98.1 and the instance was
destroyed with an exact-ID readback of `instances: null`. This is model-free
Rust test evidence, not a speed comparison or real-weight parity receipt.

## Native Realtime acoustic decoder boundary

The fixed HF header range also records 276 BF16 tensors under
`model.acoustic_tokenizer.decoder.*`; its canonical descriptor digest is
`4576ebac2def6293d72668f3fa068461b7c256c6d75e9f12eb6c16207ccfbfbb`.
The descriptor shapes match the already-native causal VibeVoice acoustic
decoder (stem 64→2048, stage widths 2048/1024/512/256/128/64/32 with depths
8/3/3/3/3/3/3, six transposed convolutions, and a 32→1 head). The dedicated
Realtime constructor first applies
`VibeVoiceStreamingCheckpoint::from_gguf`, enforces exactly 276 decoder-prefix
tensors, then reuses the 1.5B decoder loader and scalar conversion logic. The
rank-0 `speech_bias_factor`/`speech_scaling_factor` tensors are loaded and
validated by the Realtime-specific `VibeVoiceLatentScale` loader after the
composite checkpoint gate; they are not part of the descriptor-only composite
tensor gate itself.

`VibeVoiceRealtimeAcousticDecoderStream::decode_scaled_latent` accepts one
scaled `[64]` frame, applies the official
`latent / scaling_factor - bias_factor` conversion, and forwards it through
the existing causal decoder stream to one 3,200-sample 24 kHz mono chunk. It
rejects non-CPU backends explicitly; no CPU fallback is implied. The tests are
model-free contract tests only. The header-only wording is superseded for
checkpoint identity by the 2026-09-30 full-payload and conversion/binder
evidence. No real-weight waveform parity, independent acoustic waveform
reference, Metal parity, or complete synthesis claim follows from this
boundary.

## Native composite execution boundary (unpromoted)

`vokra_models::vibevoice_streaming::runtime` now composes the authenticated
language stages, diffusion CFG sampler, acoustic connector, and causal
acoustic decoder into a borrow-scoped
`VibeVoiceRealtimeSynthesisSession`. The runtime imports the four official
`lm`, `tts_lm`, `neg_lm`, and `neg_tts_lm` preset branches first. After import,
text uses only incremental LM/TTS-LM steps; the negative text LM is not
advanced. Each source five-token text window is followed by up to six speech
steps, including the source zero-text continuation.

For every speech step the caller supplies a fresh finite 64-wide noise vector.
The native order is CFG sampling, acoustic decoding, and then the acoustic
connector receives the original sampled scaled latent (not the decoder's
unscaled latent). The connector result is sent to both positive and negative
TTS branches with token `1` and `is_text = false`; both hidden rows are kept as
the next independent CFG conditions, and the positive EOS classifier is
checked afterwards.

The session keeps the source's terminal details explicit. A decoded final
chunk is retained before the strict `max_length` check; when appending the
next TTS position would exceed the bounded `max_new_tokens` budget, neither
TTS cache is advanced. After positive EOS, the remaining source six-step
inner-loop cache updates are drained while their audio chunks are suppressed;
the caller receives `Draining` steps until the session reports `Finished`.
`max_new_tokens`, a separate finite speech control budget, and an external stop
are distinct terminal conditions. An explicit external stop or exhausted
caller speech budget takes precedence over an incomplete EOS drain and is
reported consistently by both the step result and `stop_reason()`. Any
operational error resets the positive, negative, and acoustic mutable state and
poisons the session so a partial cache cannot be reused.

The composite is currently CPU-only because its sampler and causal acoustic
decoder do not have a GPU implementation. Unsupported Metal/CUDA requests are
rejected before weight binding; there is no silent CPU fallback. This historical
6a baseline section records source-ordered native composition and model-free
sequencing tests only; the current source capability is recorded above.
Independent real-weight CPU parity, an independent waveform reference, a
caller noise packet, Apple CPU/Metal parity, voice rights/consent and other
owner/legal decisions remain open. The CLI and public model status must not be
promoted from this boundary, and no completion or publication claim follows.

## Verification boundary

Run only the model-free checks:

```text
uv run --no-project --python 3.12 python tools/parity/vibevoice_realtime_0_5b_inspect.py --self-test
uv run --no-project --python 3.12 python tools/parity/vibevoice_realtime_0_5b_inspect.py --gate-self-test
```

No model, tokenizer snapshot, or tensor body is loaded by these inspector
checks. The roughly 2 GB HF safetensors payload remains VAST-only; the narrow
real-weight packet is generated separately by `run_reference.py`. Independent
native parity, full acoustic waveform parity, owner/legal approval, dataset
provenance review, and public publication remain blocked and must not be
inferred from this structural contract.
