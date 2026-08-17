# parity-deepfilternet3-real — owner runbook

Tracked / public. Operational counterpart to
`.github/workflows/parity-deepfilternet3-real.yml` for the M4-20 T17
DeepFilterNet3 `denoise` parity harness
(`crates/vokra-ops/tests/parity_denoise_dfn3.rs`).

## What the workflow proves

The workflow has two explicit phases:

* **Phase A — conversion and GGUF loadability.** It fetches
  `Rikorose/DeepFilterNet`'s `models/DeepFilterNet3.zip` at commit
  `82b0c7ad33fc756284104053817d1e855d8d8386`, verifies SHA-256
  `49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2`,
  flattens `model_120.ckpt.best`, and runs
  `vokra-cli convert --model denoise`. The converter reparses the emitted
  GGUF through `DenoiseModel::from_gguf`, so successful conversion also proves
  runtime loadability.

* **Phase B — independent real numerical parity.** It creates
  `clean_48k.f32`, `noisy_48k.f32`, `enhanced_upstream.f32`, and every
  `taps/*.f32` input required by
  `dfn3_real_weight_stage_and_output_parity`. The upstream Python 3.11 oracle
  is pinned by `tools/parity/dfn3/{pyproject.toml,uv.lock}` and executed only
  through uv. `dfn3_dump_reference.py` calls the real DeepFilterNet 0.5.6
  checkpoint and requires its stage walk to match the upstream whole-model
  forward bit-for-bit before Rust sees the fixtures.

No mutable `VOKRA_DFN3_DATA_URL` bundle is needed. Matching PRs and enabled
schedules run both phases. A manual dispatch runs Phase A only by default and
runs both phases with `force_parity=true`; conversion-only output is visibly
marked as a Phase B skip, never as a numerical pass (FR-EX-08).

## Exact fixture recipe

The quality bounds were established on 2026-07-17 with:

1. `tests/fixtures/audio/jfk-30s.wav` (11 seconds of 16-kHz mono PCM16);
2. `torchaudio.functional.resample` from torch/torchaudio 2.1.2 for 16→48 kHz;
3. `np.random.default_rng(20260717)` float64 white noise;
4. raw full-signal power scaling to 5.000 dB construction SNR;
5. DeepFilterNet/DeepFilterLib 0.5.6 running the pinned real checkpoint.

The Rust harness separately computes zero-mean SI-SNR, hence its noisy anchor
is `5.002 dB`. Do not replace the resampler or use zero-mean power during
fixture construction: both change the input bytes and the upstream quality
anchor.

## Owner operation

Enable the weekly full gate:

```sh
gh api -X POST repos/ayutaz/vokra/actions/variables \
  -f name=VOKRA_DFN3_ENABLE -f value=1
```

Run the initial full dispatch:

```sh
gh workflow run parity-deepfilternet3-real.yml -f force_parity=true
```

The completed run must show:

* `run_conversion=true` and `run_parity=true`;
* checkpoint SHA verification and GGUF conversion/loadability success;
* upstream `self_check_max_delta` values all zero;
* all 21 stage/output comparisons within their existing bounds;
* noisy/upstream/Vokra SI-SNR around `5.002 / 14.768 / 14.768 dB`;
* no `skipping: set VOKRA_DFN3_GGUF` line;
* `git diff --exit-code Cargo.lock` success.

`gh` authentication must be valid before dispatching or setting the repository
variable. The workflow is not a required check; promoting it remains an owner
decision after consecutive stable runs.

## VAST reproduction

Run model-backed Python and Cargo work on VAST, not the Mac:

```sh
uv sync --project tools/parity/dfn3 --frozen --python 3.11

uv run --project tools/parity/dfn3 --frozen python \
  tools/parity/dfn3_prep_noisy.py \
  --clean-source tests/fixtures/audio/jfk-30s.wav \
  --out-dir /tmp/dfn3-refdata \
  --enhance --model-dir "${DFN3_MODEL_DIR}"

uv run --project tools/parity/dfn3 --frozen python \
  tools/parity/dfn3_dump_reference.py \
  --model-dir "${DFN3_MODEL_DIR}" \
  --noisy /tmp/dfn3-refdata/noisy_48k.f32 \
  --out /tmp/dfn3-refdata/taps

VOKRA_DFN3_GGUF="${DFN3_GGUF}" VOKRA_DFN3_DATA=/tmp/dfn3-refdata \
  cargo test --locked --release -p vokra-ops --test parity_denoise_dfn3 \
  dfn3_real_weight_stage_and_output_parity -- --nocapture
```

2026-08-18 VAST instance `47955178` evidence: the first run exposed that the
tracked prep recipe had drifted from the preserved 2026-07-17 script
(scipy/zero-mean instead of torchaudio/raw-power), yielding an honest
`14.790` vs `14.768 ± 0.01` quality-anchor failure. Restoring the exact recipe
made all existing bounds pass without changing any tolerance: enhanced
waveform max |Δ| `4.172e-7`; upstream and Vokra SI-SNR both `14.768 dB`.

## Troubleshooting

* **Checkpoint SHA mismatch:** retry once, then investigate the upstream
  artifact. Do not bump the pin without regenerating and reviewing parity.
* **Raw `.f32` “format not recognised”:** the dumper must use its explicit
  little-endian raw-f32 path; libsndfile cannot infer a headerless format.
* **Quality anchor moves but stage deltas pass:** compare the fixture recipe
  and uv lock first. Do not relax the SI-SNR bound.
* **Harness reports a skip:** both `VOKRA_DFN3_GGUF` and `VOKRA_DFN3_DATA`
  must resolve to existing paths; the workflow converts this condition into a
  hard failure during a requested Phase B run.

## Related files

* Workflow: `.github/workflows/parity-deepfilternet3-real.yml`
* Locked oracle: `tools/parity/dfn3/{pyproject.toml,uv.lock}`
* Checkpoint prep: `tools/parity/dfn3_prepare_checkpoint.py`
* Audio prep: `tools/parity/dfn3_prep_noisy.py`
* Upstream dumper: `tools/parity/dfn3_dump_reference.py`
* Rust gate: `crates/vokra-ops/tests/parity_denoise_dfn3.rs`
* Committed primitive fixtures: `tests/parity/dfn3/`
