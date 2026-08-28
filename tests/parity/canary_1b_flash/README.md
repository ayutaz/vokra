# Canary-1B-Flash structural reference

This directory pins the released NVIDIA state-dict contract used by the
native Canary-1B-Flash binder. It is a **structural reference**, not numerical
parity: no tensor payload and no forward result is committed here.

## Immutable source

- Repository: `nvidia/canary-1b-flash`
- Revision: `2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e`
- Archive: `canary-1b-flash.nemo`, 3,540,715,520 bytes
- Archive SHA-256 (HF LFS):
  `3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324`
- Embedded checkpoint: `model_weights.ckpt`, 3,539,329,811 bytes
- Embedded `data.pkl` SHA-256:
  `a60784f60aa5cea26d3c11d62c3ed7270e5c7bf52844d99b553656d9498a3617`
- Embedded `model_config.yaml` SHA-256:
  `42d71aebc1f4b9f387a20902db71e00128b324ff5156bdac63897e1afad55ff9`

The `.nemo` is an uncompressed POSIX tar. Only its small config/tokenizer
members, the PyTorch ZIP `data.pkl`, and ZIP directory bytes were obtained via
pinned HTTP Range requests. Tensor storage members were not downloaded on the
maintainer Mac.

`state_dict_manifest.json` was produced with:

```sh
uv run --no-project --python 3.12 python \
  tools/audit/torch_pickle_manifest.py data.pkl state_dict_manifest.json \
  --source nvidia/canary-1b-flash@2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e:canary-1b-flash.nemo/model_weights.ckpt
```

The fail-closed unpickler accepts only plain PyTorch tensor reconstruction
globals and never imports `torch`. The result contains 1,406 tensors with
the runtime binder's canonical `(name, dimensions)` manifest SHA-256
`861fbc862c01f6e4517e39661f5fa8a982988e6605b71d6178a4eb27a8dc8a11`.
The richer dtype/stride/storage manifest SHA-256 is
`f9b5a22c917c094131486740c3c42b857f5d1b46d07a09dd65cdf305138758a8`.
The inference-prep contract drops 32 integer
`batch_norm.num_batches_tracked` counters, leaving 1,374 float tensors with
strict GGUF manifest SHA-256
`f76f4c3d28147b418705c8272a81dab53425e3bd264b8a2040ffb0de03385cb6`.

## Public-artifact discrepancy

At the pinned audit date, upstream `model.safetensors` contains 1,292 tensors,
all under `encoder.*`; its `config.json` also declares
`nemo_decoder_type = "none"`. The public
`vokra/canary-1b-flash/canary-1b-flash.gguf` mirrors those 811,049,984 encoder
parameters and therefore cannot perform Canary ASR/AST. A complete artifact
must instead be prepared from the `.nemo` checkpoint on VAST and must include:

- 1,260 float `encoder.*` tensors (the 32 training-only integer counters are
  removed by the established NeMo prep script);
- 110 `transf_decoder.*` tensors;
- 2 `log_softmax.*` tensors;
- 2 preprocessor buffers (or an authenticated runtime-equivalent frontend);
- the five aggregate SentencePiece tokenizers and their exact ordering.

The binder rejects the encoder-only artifact explicitly; it never substitutes
a fabricated decoder or silently falls back to CPU.

## Tokenizer contract

The embedded config orders aggregate tokenizers as
`spl_tokens, en, de, es, fr`. Their piece counts are
`1152 + 1024 + 1024 + 1024 + 1024 = 5248`, exactly matching
`head.num_classes = 5248`. In the special-token tokenizer:

- pad = 2 (`<pad>`)
- EOS = 3 (`<|endoftext|>`)
- BOS = 4 (`<|startoftranscript|>`)
- English = 62, French = 69, German = 76, Spanish = 169
- PnC = 5, no-PnC = 6, ITN = 8, no-ITN = 9
- timestamp = 10, no-timestamp = 11
- diarize = 12, no-diarize = 13
- undefined emotion = 16

These IDs come from the released 1,152-entry tokenizer itself, not from a
hand-written mirror.

## Independent numerical gate

No forward value is fabricated or committed by the structural audit above.
The real-checkpoint gate runs only on a provisioned Linux/VAST host and imports
the official `nemo.collections.asr.models.EncDecMultiTaskModel` from the pinned
`nemo-toolkit[asr]==3.0.0` parity environment.
Decoder, Canary2 prompt and hypothesis-stripping semantics were separately
audited against official `NVIDIA-NeMo/Speech` commit
`837a31fa7a810a3de9e4826837e97dea837a5c42`; that source-audit revision is
recorded distinctly from the executed package version.

The host must also have the `rustfmt`/`clippy` Rust components and the
`cargo-deny`/`cargo-audit` executables installed. The worker checks these before
unpacking the 3.54 GB archive, so a missing verifier cannot waste a conversion
run.

Run it with:

```sh
VOKRA_PUBLISH_ON_VAST=1 \
  scripts/publish/vast-ai/run-canary-1b-flash-validation.sh \
  --nemo /workspace/canary-1b-flash.nemo
```

The worker authenticates and prepares the archive, builds the complete GGUF,
records CPU/ISA/Torch/CUDA environment data before generation, then requires
exact greedy token equality for both English ASR and English-to-German AST.
It also runs the workspace tests, workspace clippy, `cargo deny`, and
`cargo audit`. It never uploads, publishes or pushes. Pull only the small
`evidence/` directory and log before destroying the instance; do not copy the
multi-GB checkpoint, safetensors or GGUF back to the maintainer Mac.

As of 2026-08-26 this worker is staged but has not run, so this directory
proves the immutable structure and comparison procedure—not numerical parity.
VAST is Linux and cannot establish real-weight Metal parity; that final gate
requires an Apple-silicon runner and remains separate.
