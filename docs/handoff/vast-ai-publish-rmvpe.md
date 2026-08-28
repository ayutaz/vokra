# RMVPE VAST validation handoff

Status: native exact E2E0 CPU/Metal implementation is in progress on
`feat/npu-delegate-execution-2026-08-24`. Real-weight validation has not yet
been recorded. This document authorizes neither a Hugging Face upload nor a
license override.

## Audited identity

| Item | Fixed value |
|---|---|
| Runtime arch | `rmvpe` |
| Upstream source | `yxlllc/RMVPE` |
| Upstream commit | `0aabafba18289ca938a73af0b0297686abf4922d` |
| Public HF repository | `vokra/rmvpe` |
| Public HF revision | `3eb5fa8946f1074ba3959074c5cde95ec22b8c91` |
| Public file | `rmvpe.gguf` |
| Public size | `181010688` bytes |
| Public LFS SHA-256 | `208fc73819586b4546f2cba7a829033c5900c44af1ad48fe9d3e727cc1a932fb` |
| Runtime inference tensors | 623 |
| Optional BatchNorm counters | 118 |

The public header audit used HTTP Range reads only; no public tensor payload
was processed on the maintainer Mac.

## Exact runtime contract

The fixed upstream `E2E0` forward is:

1. 16-kHz PCM → magnitude log-mel (`n_fft=win_length=1024`, hop 160,
   128 HTK mels, Slaney normalization);
2. initial BatchNorm;
3. five encoder layers, each with four residual Conv-BN-ReLU blocks and
   `AvgPool2d(2)`;
4. four intermediate residual layers;
5. five decoder layers with 3×3 stride-2 ConvTranspose, BN/ReLU, paired skip
   concat, and four residual blocks;
6. 16→3 Conv2d and `[3, 128]` collapse to 384 features per frame;
7. one-layer bidirectional GRU, 256 features per direction;
8. 512→360 Linear, sigmoid, and nine-bin local-average decode.

Class zero is exactly 31.7 Hz and classes are 20 cents apart. The upstream
decoder does not clamp voiced output to `fmin`/`fmax`.

The historical public GGUF header says `n_fft=2048` and
`base_hz=32.703197`. The tensor payload is the fixed E2E0 model, so the loader
recognizes only that exact historical pair and normalizes it to 1024/31.7.
Other metadata combinations fail explicitly.

Conv2d and ConvTranspose2d lower to the existing GEMM seam; GRU gates use
GEMV. A selected Metal backend validates both before executing. There is no
per-op CPU fallback.

## License blocker

`yxlllc/RMVPE` has no LICENSE file or GitHub license classification.
`Dream-High/RMVPE` has Apache-2.0 terms but is not related through GitHub's
fork graph and cannot establish terms for the exact released checkpoint.

The public GGUF is therefore incorrectly stamped:

```text
vokra.provenance.license = mit
vokra.provenance.weight_license = permissive
vokra.provenance.source = yxlllc/RMVPE
```

The strict runtime rejects that provenance. For validation, create an
ephemeral copy with `license=unknown` / `weight_license=unknown`; do not upload
it. Publishing, replacing the public file, or applying a permissive override
requires a separately verified grant and explicit user permission.

## VAST-only validation

Use `.agents/skills/vast-ai-workflow/SKILL.md`. Transfer the current commit or
a git bundle, provision the repository, run the work, pull only small evidence,
and destroy the instance. Never stop an unused instance.

The saved VAST credential must be valid before starting. Do not print the key
or place it in logs.

On the instance:

```bash
cd ~/vokra/tools/parity/rmvpe
uv sync --python 3.12 --frozen

bash fetch_rmvpe_pt.sh \
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

Prepare the strict 623-tensor GGUF through the established converter chain or
make a provenance-only ephemeral restamp of the audited public payload. If the
raw E2E0 state dict contains `unet.tf.*`, omit it: `DeepUnet0` constructs that
TimbreFilter but its forward never reads it.

Then run:

```bash
export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe-corrected.gguf
export VOKRA_RMVPE_REAL_PCM=~/rmvpe-fixtures/reference/pcm.f32
export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/reference/hidden.f32
export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=384
export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/reference/argmax.u32
export VOKRA_RMVPE_REAL_F0=~/rmvpe-fixtures/reference/f0.f32

CARGO_BUILD_JOBS=1 cargo test -p vokra-models --test parity_rmvpe -- --nocapture
```

Linux VAST proves the CPU route and independent upstream parity. Metal cannot
execute on a Linux VAST host; obtain the Metal measurement from an authorized
remote Apple runner. Do not run the real model on the maintainer Mac merely to
close that row. A cross-build is useful evidence but is not an execution
measurement.

## Evidence to retain

Copy back only small files:

- exact git commit and dirty-state check;
- VAST offer/instance/GPU metadata without credentials;
- checkpoint, GGUF, PCM, and fixture SHA-256 values;
- `meta.json` from the independent dumper;
- CPU test output and measured parity values;
- remote Apple runner identity and Metal/CPU values when available;
- `cargo fmt --all -- --check`, focused converter tests, `git diff --check`,
  zero-dependency and license gate output.

Record commands that were not run and why. After evidence is safely pulled,
destroy the VAST instance and confirm it no longer appears in the instance
list.
