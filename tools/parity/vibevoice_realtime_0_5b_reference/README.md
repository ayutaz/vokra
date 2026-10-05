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

## 2026-10-06 locked-sdist primary-license supplement

This dated supplement starts from the clean PR #169 source candidate
`312369da7886f680cd708fea91b14818b4a5eb9d`, which includes main
`5ccde239ec261cbd8c3b5ac8a23fe2ee32fdddeb`. Its CPU-only lock remains
`cbf0ce675cdc8bc3c8cd32a4528f3a67f283b2e7b46841cb6dc32e4949af666e`.
The earlier installed-environment audit below remains bound to its original
HEAD and evidence hash. A license found in an sdist does **not** disprove a
missing bundled-license finding for a different, installed wheel.

Two independent read-only reviews authenticated the following exact lock-listed
publisher sdists by full size and SHA-256. Regular archive members were read
in memory with bounded reads; no archive was extracted to disk, dependency
installed/imported, build script run, model acquired/executed, or token used.

| Publisher sdist | Bytes | SHA-256 |
| --- | ---: | --- |
| [safetensors 0.5.3](https://files.pythonhosted.org/packages/71/7e/2d5d6ee7b40c0682315367ec7475693d110f512922d582fef1bd4a63adc3/safetensors-0.5.3.tar.gz) | 67,210 | `b6b0d6ecacec39a4fdd99cc19f4576f5219ce858e6fd8dbe7609df0b8dc56965` |
| [tokenizers 0.22.2](https://files.pythonhosted.org/packages/73/6f/f80cfef4a312e1fb34baf7d85c72d4411afde10978d4657f8cdd811d3ccc/tokenizers-0.22.2.tar.gz) | 372,115 | `473b83b915e547aa366d1eee11806deaf419e17be16310ac0a14077f1e28f917` |
| [tqdm 4.67.1](https://files.pythonhosted.org/packages/a8/4b/29b4ef32e036bb34e4ab51796dd745cdba7ed47ad142a9f4a1eb8e0c744d/tqdm-4.67.1.tar.gz) | 169,737 | `f8aef9c52c08c13a65f30ea34f4e5aac3fd1a34959879d7e59e63027286627f2` |

Located primary license members:

- `safetensors-0.5.3/safetensors/LICENSE` and
  `tokenizers-0.22.2/tokenizers/LICENSE` are each 11,357 bytes, SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- `tqdm-4.67.1/LICENCE` is 1,985 bytes, SHA-256
  `dc33252e829015e3b150086fb9b3a40f6ad6fb32c2f4610ce812fa677d35986a`.
  The spelling is `LICENCE`, not an absent `LICENSE` member.

Each sdist's `PKG-INFO` matches its locked Name and Version. Safetensors and
tokenizers have Apache license classifiers but no explicit `License` header;
their respective `PKG-INFO` digests are
`fe7d581363ed20b204701623da8f90dd8a3b9995e9a9290b616788d80356adbf`
(3,823 bytes) and
`15a5ddaf489f592b77e0a934c0eeb46b51130a94064ca129232afdb0a3868efc`
(7,254 bytes). Both tqdm `PKG-INFO` members are byte-identical:
57,675 bytes, SHA-256
`688a1632df525a198fec52dcfefb1246d31eacee9c035aacac2d7eaf8d8ff669`.
They declare `License: MPL-2.0 AND MIT` and `License-File: LICENCE`;
this must not be reduced to MIT-only or treated as a blanket MPL approval.

Disposition is **SDIST_PRIMARY_LICENSE_LOCATED / INSTALLED_BINDING_UNPROVEN**.
Exact selected-wheel/build/installed RECORD and native-library closure,
compatibility, package/license and operator owner review still need their
own evidence. Existing `OWNER_REVIEW_REQUIRED` and `NO_UPLOAD` gates are
unchanged. No real-weight replay, numerical/Apple parity, publication, or
whole-catalog completion is authorized or established by this supplement.

## 2026-10-06 selected-wheel primary-license supplement

Two independent bounded, in-memory reviews also authenticated these exact
artifacts from the same unchanged CPU-only lock. The native-package scope is
**glibc/manylinux Linux x86_64**, not every Linux wheel: the lock also lists
musllinux candidates, which were not fetched for this review.

| Selected publisher wheel | Bytes | SHA-256 |
| --- | ---: | --- |
| [safetensors 0.5.3 manylinux](https://files.pythonhosted.org/packages/a6/f8/dae3421624fcc87a89d42e1898a798bc7ff72c61f38973a65d60df8f124c/safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) | 471,642 | `cead1fa41fc54b1e61089fa57452e8834f798cb1dc7a09ba3524f1eb08e0317a` |
| [tokenizers 0.22.2 manylinux](https://files.pythonhosted.org/packages/2e/76/932be4b50ef6ccedf9d3c6639b056a967a86258c6d9200643f01269211ca/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl) | 3,274,982 | `369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67` |
| [tqdm 4.67.1 platform-independent](https://files.pythonhosted.org/packages/d0/30/dc54f88dd4a2b5dc8a0279bdd7270e735851848b762aeb1c1184ed1f6b14/tqdm-4.67.1-py3-none-any.whl) | 78,540 | `26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2` |

Each wheel has one `METADATA` member with the expected Name/Version; its
content matches the corresponding sdist `PKG-INFO` digest above. Safetensors
and tokenizers have neither a regular LICENSE/LICENCE/COPYING/NOTICE member
nor License/License-Expression/License-File headers. Their native members are
`safetensors/_safetensors_rust.abi3.so` and `tokenizers/tokenizers.abi3.so`;
these were enumerated, not loaded or executed. Located sdist licenses do not
automatically prove the source/build binding of these native members.

The tqdm wheel does contain `tqdm-4.67.1.dist-info/LICENCE`: 1,985 bytes,
the same `dc33252e...` digest as its sdist above. Its metadata declares
`MPL-2.0 AND MIT` and `License-File: LICENCE`, and no native member was found.
Thus "tqdm has no bundled license" must not be generalized to this selected
wheel. The older installed-environment receipt remains historical evidence,
not a newly revalidated installation or RECORD binding.

No archive was extracted or saved, package installed/imported, native code
executed, or model acquired. Disposition remains **PRIMARY_ARTIFACT_FACTS_BOUND /
INSTALLED_NATIVE_CLOSURE_UNPROVEN**. Owner review, execution restrictions and
NO_UPLOAD are unchanged; this is not numerical or Apple verification.

## 2026-09-30 PR #167 dependency-security boundary

The security review of the Torch 2.7.1 to 2.13.0 Dependabot update uses the
explicit CPU-only PyTorch index and locks `torch==2.13.0+cpu` for Linux x86_64.
The lock contains no `nvidia-*`, CUDA, cuDNN, cuBLAS, cuFFT, NCCL, or NVJITLINK
package/native payload. The lock SHA-256 is recorded in the project metadata
and was verified by the VAST sync receipt. This is intentional: the CUDA 13
closure has not received the required owner/legal review and must not enter the
model-free API/license gate implicitly.

This CPU-only security lock does not make a GPU performance claim. If an
approved future run shows that GPU is faster while meeting the fixed numerical
guard, it must use a separately reviewed GPU lock and record that evidence;
neither the current API smoke nor this lock authorizes a real-weight replay or
publication.

The fresh VAST receipt for this lock used disposable instance `53407724` and
implementation HEAD `c7dc975c`. `uv sync --frozen` installed 41 packages,
including `torch==2.13.0+cpu`; the lock SHA-256 is
`cbf0ce675cdc8bc3c8cd32a4528f3a67f283b2e7b46841cb6dc32e4949af666e` and no
`nvidia-*` package was present. The imported Torch runtime reported
`2.13.0+cpu`, `torch.version.cuda=None`, and `cuda_available=false`; its
`torch/lib` contained no CUDA/cuDNN/cuBLAS/cuFFT/NCCL/NVJITLINK-named native
library. The model-free lock contract test passed with evidence SHA-256
`70f593a0f94db5c51c05806dbd85fac295f1e223d135fc789848173dd98b3f8a`.

Against the clean Microsoft source checkout at revision
`94da20d98b2fa7688e9cbfaf7692ddb4954f7600`, the official import/API smoke
returned `AUTHENTICATED_API_SMOKE` with evidence SHA-256
`1d5f9d037ef15e8cded3db06d323d86de4ef5e08bd9fc52e00f96655e241a189`.
It reported `NO_MODEL_DOWNLOAD`, `NO_MODEL_EXECUTION`, and `NO_UPLOAD`.
The installed closure audit covered 41 packages and returned
`OWNER_REVIEW_REQUIRED_NO_UPLOAD`; bundled license files were missing for
`safetensors`, `tokenizers`, and `tqdm`. The audit evidence SHA-256 is
`23bc546d7fcf47f1dc3b66f587c53d37b8025e5b418c185148f3201f1417e06e`.

The disposable instance was destroyed after evidence collection and its
individual API readback returned `instances: null`.

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
a deterministic `[1, 1, 64]` acoustic latent. It writes the EOS logits, TTS
hidden state, acoustic-connector output, and a JSON packet containing source,
checkpoint, input, runtime, timing, shape, finiteness, and SHA-256 metadata.
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

The dedicated `uv.lock` is generated and pinned. The PR #167 security lock is
CPU-only (`torch==2.13.0+cpu`) and its VAST sync, package inventory, and
model-free official-source API smoke are recorded separately from the
historical CUDA-enabled receipts above. The smoke imports and inspects the
official API only: it does not download a checkpoint, construct a model,
execute a checkpoint, generate parity numbers, or publish an artifact.

The installed-closure audit for the CPU-only lock completed on VAST and remains
`OWNER_REVIEW_REQUIRED/NO_UPLOAD` because the bundled license files for
`safetensors`, `tokenizers`, and `tqdm` still require owner/primary-source
review. That review is therefore still required before any real-weight replay
or publication. The command below is the controlled replay command, not an
authorization or assertion that replay is currently cleared:

```text
uv run --frozen --python 3.12 --project tools/parity/vibevoice_realtime_0_5b_reference python \
  tools/parity/vibevoice_realtime_0_5b_reference/run_reference.py \
  --source-root /root/VibeVoice \
  --config /root/realtime-checkpoint/config.json \
  --weights /root/model.safetensors \
  --dtype float32 \
  --output /root/realtime-reference
```

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

The earlier VAST `uv sync` and dependency audit were for a CUDA-enabled Torch
lock and are historical only. The CPU-only lock audit is complete on VAST, but
the installed closure remains `OWNER_REVIEW_REQUIRED/NO_UPLOAD` because no
owner/legal approval has been granted for a real-weight replay or publication.
No checkpoint may be downloaded or run for this compatibility update.

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

## Native prediction-head boundary

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

## 2026-10-06 bounded installed-closure collector preparation

`audit_installed_closure.py` and its stdlib-only regression tests are a
source-only preparation for a future Linux x86_64 / CPython 3.12 audit. The
collector accepts an explicit selected-wheel manifest and a trusted
site-packages directory; it does not download, install, import, or execute
third-party packages or models. Every selected archive is rebound to the
current project and lock hashes, the lock-listed wheel URL/hash/filename/tag,
archive bytes, wheel `RECORD`, and installed metadata. `LICENCE` and
`License-File` spellings are resolved explicitly, and native members are
reported only when their archive and installed bytes are equal. The audit also
accepts wheel-declared `.data/purelib`/`.data/platlib` relocation and
`console_scripts`/`gui_scripts` wrappers only through an explicit canonical
venv root and scripts root. Relocated wrappers are hash-bound to the installed
RECORD but are separately classified because installers may rewrite shebangs.
Generated wrappers are classified as `UNPROVEN_INSTALLER_SOURCE`, and the
report's RECORD status is logical binding rather than RECORD-file byte equality.
Installer-only additions are limited to `INSTALLER`, `REQUESTED`,
`direct_url.json`, and bounded bytecode paths. Unknown, out-of-root, or
undeclared external RECORD paths fail closed.

The selected-wheel manifest is untrusted input, not owner approval or a source
provenance signature. Reports remain
`OWNER_REVIEW_REQUIRED_NO_UPLOAD`, with package license, native payload,
runtime compatibility, and owner/legal decisions unresolved. Missing bundled
license files are reported as unresolved rather than approved. The historical
installed-closure receipt is not rewritten or re-signed; a real collection
still requires the reviewed VAST workflow and an independently authenticated
source checkout.

The bounded synthetic regression entry point is:

```text
UV_NO_SYNC=1 UV_OFFLINE=1 uv run --no-project --no-sync --offline --python 3.12 python -S tools/parity/vibevoice_realtime_0_5b_reference/audit_installed_closure.py --self-test
```

Real collection additionally requires explicit `--venv-root` and
`--scripts-root` paths inside that same trusted environment; no implicit host
path discovery is permitted. The current family lock statically contains 41
registry rows and the reviewed CPU Torch row, but this collector intentionally
blocks before collection when any locked artifact hash is absent (the current
snapshot includes such a row, e.g. `jinja2`); no lock row is repaired or
re-signed by this preparation.

The collector's Metadata-Version 2.4+ `License-File` resolution follows
PEP 639 exactly: a header value is relative to the wheel's
`<dist-info>/licenses/` directory (so `LICENSE` and `licenses/LICENSE.MIT`
resolve to distinct nested members). Metadata-Version 2.1 legacy resolution
is retained separately. Project validation also binds the exact Linux CPU
Torch declaration, explicit PyTorch CPU index, and `2.13.0+cpu` reference
fact; a prefix match or alternate index is rejected. Venv `pyvenv.cfg`, the
CPython 3.12 interpreter, complete site/scripts inventory, wheel WHEEL tags,
installer-generated wrapper bytes, and bounded read/hash budgets are checked
before an evidence report is produced. These checks strengthen collection
integrity only; they do not establish license approval, native compatibility,
owner sign-off, or upload eligibility.

The venv boundary is strict: `pyvenv.cfg` must identify CPython 3.12 with
`include-system-site-packages = false`, the running interpreter and
`python`/`python3`/`python3.12` aliases must resolve to the same executable,
and the canonical `lib/python3.12/site-packages` plus `bin` layout is required.
Known activation and virtualenv bootstrap files are enumerated individually,
captured with bytes/SHA and marked `UNPROVEN_INSTALLER_BOOTSTRAP`; arbitrary
unregistered files are never ignored, and site `_virtualenv.py`/
`_virtualenv.pth` bootstrap entries must be regular non-symlink files. Installed package and bootstrap file
identities (inode, metadata, size, and SHA) are captured and revalidated after
source inspection. Generated wrappers, relocated wrappers, generated pyc, and
installer metadata carry their installed bytes/SHA and remain explicitly
`UNPROVEN_INSTALLER_SOURCE` where installer provenance is not archive-bound.
The initial installed inventory and clean source checkout are captured before
package inspection, then both are revalidated after package and archive
binding. WHEEL metadata may contain multiple distinct expanded platform tags,
but duplicate tags or filename/tag disagreement remain blocked.

## 2026-10-06 lock identity enrichment (VAST not run)

The current lock adds bounded artifact identity only. The exact selected
[`jinja2` wheel](https://download.pytorch.org/whl/jinja2-3.1.6-py3-none-any.whl)
is 134,899 bytes with SHA-256
`85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67`; the
exact selected
[`markupsafe` wheel](https://download.pytorch.org/whl/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl)
is 22,947 bytes with SHA-256
`d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d`.
The remaining eleven previously size-less wheel rows have bounded HTTP HEAD
size metadata recorded in `uv.lock` (mpmath, four NumPy, three Pillow,
SymPy, Torch, and typing-extensions); their bodies were not re-fetched by this
change. URLs, versions, markers, dependency edges, and pre-existing hashes
remain unchanged.

`uv.lock` now hashes to
`34f58e53b5c79ed96853c2a4b6f9b6b1eaf12066f010cd23ddffae2b816b3797`, and the
reference status is
`IDENTITY_ENRICHED_VAST_NOT_RUN_OWNER_REVIEW_REQUIRED`. The compatibility-smoke
and installed-closure receipts remain historical and are explicitly bound to
the prior lock SHA
`cbf0ce675cdc8bc3c8cd32a4528f3a67f283b2e7b46841cb6dc32e4949af666e`; they are
not evidence for this enriched lock. Package license, native closure, runtime
compatibility, owner/legal review, real execution, and `NO_UPLOAD` gates remain
unresolved/blocked. No dependency was installed or imported and no VAST
collection was run.
