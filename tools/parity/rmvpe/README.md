# RMVPE parity sidecar

This directory generates independent numerical fixtures for the exact native
RMVPE CPU/Metal implementation. It imports the upstream implementation rather
than reimplementing the model in Python.

Fixed inputs:

- upstream repository: `yxlllc/RMVPE`
- upstream commit: `0aabafba18289ca938a73af0b0297686abf4922d`
- model class: `src.inference.RMVPE`, which instantiates `src.model.E2E0`
- frontend: 16 kHz, hop 160, `n_fft = win_length = 1024`, 128 HTK mels
- decoder: nine-bin local average, threshold `0.03`, no Viterbi

The `yxlllc/RMVPE` repository has no LICENSE file. `Dream-High/RMVPE` is
Apache-2.0 but is not a GitHub fork relationship and does not establish terms
for the exact `yxlllc` checkpoint. Keep the weight and generated GGUF
fail-closed as `unknown`; this parity workflow does not authorize publication.

## Safety and execution location

Run the checkpoint conversion, upstream inference, and `vokra-models` Cargo
tests on VAST. Do not run them on the maintainer Mac. Python is 3.12 and every
invocation goes through `uv`; dependencies are pinned in the parent
`tools/parity/uv.lock` workspace lock.

The `.pt` checkpoint is a PyTorch pickle and may execute code while loading.
Fetch it only from the audited release and record its SHA-256. No Python,
PyTorch, upstream source, or pickle enters the Rust runtime.

## VAST recipe

After provisioning the repository with the standard VAST workflow:

```bash
cd ~/vokra/tools/parity/rmvpe
uv sync --python 3.12 --frozen

bash ./fetch_rmvpe_pt.sh \
  --output ~/rmvpe-fixtures/rmvpe.pt \
  --sha256 <audited-release-sha256>

git clone https://github.com/yxlllc/RMVPE.git ~/rmvpe-upstream
git -C ~/rmvpe-upstream checkout 0aabafba18289ca938a73af0b0297686abf4922d
test -z "$(git -C ~/rmvpe-upstream status --porcelain --untracked-files=all)"

uv run --python 3.12 python dump_reference.py \
  --pt-path ~/rmvpe-fixtures/rmvpe.pt \
  --upstream-src ~/rmvpe-upstream \
  --canned \
  --out-dir ~/rmvpe-fixtures/reference
```

For real audio, replace `--canned` with `--pcm /path/to/clip.wav`. The WAV must
already be 16-kHz PCM; the dumper refuses resampling so upstream and Rust
receive byte-identical samples.

The dumper writes raw little-endian data without `.npy` headers:

| File | Shape | Meaning |
|---|---:|---|
| `pcm.f32` | `[samples]` | exact 16-kHz mono input |
| `hidden.f32` | `[frames, 384]` | input captured at `fc.0.gru` |
| `probabilities.f32` | `[frames, 360]` | already-sigmoid E2E0 output |
| `argmax.u32` | `[frames]` | class index; `0xffffffff` means unvoiced |
| `f0.f32` | `[frames]` | upstream nine-bin local-average F0 |
| `meta.json` | — | revisions, hashes, shapes, and settings |

Class zero is a valid 31.7-Hz bin and therefore is not used as the unvoiced
sentinel.

## GGUF and test

Convert the same trusted checkpoint through the established `.pt` →
safetensors bridge and RMVPE converter. The strict runtime accepts only the
623 inference tensors, with the 118 BatchNorm counters optional. Unused
`unet.tf.*` weights constructed by `DeepUnet0` must not enter the runnable
GGUF because `E2E0.forward` never reads them.

The current public `vokra/rmvpe` tensor payload is useful for audit, but its
header incorrectly says `license=mit` / `weight_license=permissive`. The
production loader deliberately rejects it. For parity only, make an ephemeral
provenance-corrected copy stamped `unknown`; do not upload or replace the
public artifact without separate permission and a valid license grant.

Set the paths printed by the dumper, plus the corrected GGUF:

```bash
export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe-corrected.gguf
export VOKRA_RMVPE_REAL_PCM=~/rmvpe-fixtures/reference/pcm.f32
export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/reference/hidden.f32
export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=384
export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/reference/argmax.u32
export VOKRA_RMVPE_REAL_F0=~/rmvpe-fixtures/reference/f0.f32

CARGO_BUILD_JOBS=1 cargo test -p vokra-models --test parity_rmvpe -- --nocapture
```

The harness has three distinct gates:

1. exact 623/741 manifest, frontend, topology, and finite end-to-end smoke;
2. full PCM-to-F0 agreement against the independent upstream output;
3. post-CNN hidden-state agreement for the BiGRU, head, and decoder.

On Apple hardware it additionally compares the Metal path with CPU. Selecting
Metal validates all required learned operations before execution; unsupported
operations return an explicit error and never fall back to CPU.

Copy the small logs, `meta.json`, and hashes back to the repository evidence
directory, then destroy the VAST instance. Do not copy checkpoint payloads or
large generated model artifacts to the maintainer Mac.
