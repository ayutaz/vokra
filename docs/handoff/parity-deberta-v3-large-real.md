# parity-deberta-v3-large-real — owner runbook

Tracked / public. Operational counterpart to
`.github/workflows/parity-deberta-v3-large-real.yml` for the standalone
`microsoft/deberta-v3-large` EN encoder gate. This isolates DeBERTa v3 from
the AGPL/ShareAlike checkpoints used by the combined SBV2 pipeline.

## What the workflow proves

The workflow has two explicit phases:

* **Phase A — pinned conversion and loadability.** It downloads
  `microsoft/deberta-v3-large` at revision
  `64a8c8eab3e352a784c658aef62be1662607476f`. The release ships
  `pytorch_model.bin`, so `tools/parity/bin_to_safetensors.py` performs a
  `torch.load(weights_only=True)` bridge and writes safetensors. The revision
  is passed to both download paths; no default-branch checkpoint can enter
  the run. `vokra-cli convert --model deberta-v3` writes the GGUF. The
  ignored converter smoke checks the freshly bridged safetensors; the Rust
  tests in Phase B then reparse the result through
  `DebertaV3Encoder::from_gguf` and `SbertTokenizer::from_gguf`.

* **Phase B — independent real numerical parity.** The Python 3.12 oracle is
  fully pinned by `tools/parity/deberta_v3/{pyproject.toml,uv.lock}` and runs
  only through uv. `deberta_v3_dump_reference.py` imports the real upstream
  Hugging Face `transformers` implementation, forces eager attention, and
  captures real `input_ids`, hidden states, attentions, and `final_hidden`.
  `crates/vokra-bert/tests/deberta_v3_real.rs` binds the converted GGUF, runs
  the Rust encoder over those input IDs, and compares `final_hidden` against
  the independent upstream bytes.

The Rust consumer landed on 2026-07-29. The old
`VOKRA_DEBERTA_V3_HARNESS_READY` gate and descriptions saying that no consumer
exists are obsolete and have been removed. Matching PRs and enabled schedules
run both phases. A manual dispatch also runs both by default; explicitly pass
`run_dumper=false` only for a conversion-only diagnostic run. That mode is
reported as a Phase B skip, never as a numerical pass (FR-EX-08).

## Locked oracle

The dedicated Linux x86_64 lock intentionally avoids the much larger shared
parity environment. Its direct pins are:

* Python `3.12.x`;
* CPU-only torch `2.4.1`;
* transformers `4.49.0`;
* huggingface-hub `0.29.3`;
* numpy `1.26.4`, protobuf `5.29.5`, safetensors `0.5.3`,
  sentencepiece `0.2.2`.

These packages are an offline/CI reference oracle only. They do not enter any
Vokra runtime or distribution, and the root Rust lock remains first-party
only.

## Current numerical coverage

The dumper writes embedding, every `layer_NN_output`, every
`layer_NN_attention`, and final output tensors. The active Rust gate compares:

* `input_ids` as the exact shared input;
* `final_hidden` (`[1, T, 1024]`) at the unchanged `6.0e-3` bound;
* tokenizer metadata/loadability in the sibling env-gated test.

Per-layer hidden and attention comparison remains a separate extension:
`DebertaV3Encoder::forward` currently returns only final hidden. Do not claim
those taps as covered until a real `forward_with_layer_taps()` API and measured
bounds land. This limitation does not make the existing final-output consumer
missing.

## 2026-08-18 VAST measurement

The full path was measured on VAST instance `47955178` (Linux x86_64,
AVX2) with the locked Python 3.12 oracle above. No bound was changed:

* real checkpoint revision: `64a8c8eab3e352a784c658aef62be1662607476f`;
* GGUF: 972,475,712 bytes, SHA-256
  `858c07bf3909b1b8d25d7ca7c1e0fed0e7c2bd9889a0068c6ebcfd689d0eb579`;
* tokenizer JSON: 128,000 pieces, SHA-256
  `71e5ff992a75528ded334d165bd4945cac67d615933b4ae71edf868290e1061e`;
* shared `input_ids.bin` SHA-256
  `8e3d37203282fe00a83480a265c24947f952a643e1c9100317c63a0009d928ce`;
* upstream `final_hidden.bin` SHA-256
  `8e8d9b5e73cf9703bfec1e4a9a4200a96a4b15d8d9a5f5de6c0a7987dfbf04d9`;
* measured final-hidden max absolute delta: `1.049042e-5` versus the unchanged
  `6.0e-3` bound;
* Rust real harness: 2 passed (numerical + tokenizer), converter real smoke:
  1 passed.

The measurement also exposed and fixed two genuine implementation drifts.
The v3 converter had emitted raw relative embeddings despite
`norm_rel_ebd=layer_norm` (`max|Δ|=9.343735`); applying the upstream
encoder-level normalization reduced it to `6.086731e-2`. The shared DeBERTa
FFN then used tanh-approximate GELU while transformers `ACT2FN["gelu"]` uses
the exact erf form. Matching that activation produced the final passing
measurement above. Raising the parity bound was neither needed nor attempted.

## Owner operation

Enable the weekly full gate:

```sh
gh api -X POST repos/ayutaz/vokra/actions/variables \
  -f name=VOKRA_DEBERTA_V3_ENABLE -f value=1
```

Run the initial full dispatch (no extra harness variable is needed):

```sh
gh workflow run parity-deberta-v3-large-real.yml
```

Use the following only when diagnosing conversion separately:

```sh
gh workflow run parity-deberta-v3-large-real.yml -f run_dumper=false
```

A completed full run must show:

* `run_conversion=true` and `run_dumper=true`;
* the pinned upstream revision in every checkpoint fetch;
* uv lock sync and the CPU/ISA/torch environment record before the result;
* GGUF conversion/loadability, embedded tokenizer metadata, and the real
  converter smoke passing;
* a real reference manifest plus `input_ids.bin` and `final_hidden.bin`;
* `deberta_v3_real_weight_final_hidden_parity` running without a skip;
* measured max absolute delta no greater than the existing bound;
* `git diff --exit-code Cargo.lock` succeeding.

The workflow is advisory, not a required check. Promotion remains an owner
decision after repeated stable runs.

## VAST policy

The bin, safetensors, GGUF, upstream model, and `vokra-bert`/converter build
belong on VAST or a hosted CI runner, not the M1 Mac. Python must use the
dedicated uv project:

```sh
uv sync --project tools/parity/deberta_v3 --frozen --python 3.12

uv run --project tools/parity/deberta_v3 --frozen python \
  tools/parity/deberta_v3_dump_reference.py \
  --hf-repo microsoft/deberta-v3-large \
  --revision 64a8c8eab3e352a784c658aef62be1662607476f \
  --output-dir /tmp/deberta-v3-refdata --do-dump
```

Do not copy the multi-GB model artifacts to the Mac. Preserve only hashes,
logs, and bounded reference fixtures when their redistribution terms permit.

## Troubleshooting

* **HTTP 401/403:** keep the pin unchanged. If the public model-card gate
  requires acceptance, provide a read-only `HF_TOKEN` through the environment,
  never argv or a tracked file.
* **Unsafe/corrupt pickle:** `bin_to_safetensors.py` must keep
  `weights_only=True` and fail loudly. Never fall back to unrestricted
  `torch.load`.
* **Missing tokenizer/config:** verify the pinned snapshot contains
  `config.json`, `tokenizer_config.json`, and `spm.model`; do not infer
  architecture metadata.
* **Numerical miss:** record CPU/ISA/torch first and inspect the actual worst
  element. Do not loosen `6.0e-3` to chase green.
* **Harness skip in a requested full run:** both
  `VOKRA_DEBERTA_V3_GGUF` and `VOKRA_DEBERTA_V3_REFDIR` must resolve. Treat a
  skip as workflow plumbing failure, not a pass.

## Related files

* Workflow: `.github/workflows/parity-deberta-v3-large-real.yml`
* Locked oracle: `tools/parity/deberta_v3/{pyproject.toml,uv.lock}`
* Reference dumper: `tools/parity/deberta_v3_dump_reference.py`
* Pickle bridge: `tools/parity/bin_to_safetensors.py`
* Rust consumer: `crates/vokra-bert/tests/deberta_v3_real.rs`
* Rust encoder: `crates/vokra-bert/src/deberta_v3.rs`
* Converter: `crates/vokra-convert/src/models/deberta_v3.rs`
