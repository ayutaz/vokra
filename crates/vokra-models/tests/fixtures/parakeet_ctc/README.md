# Parakeet CTC real-weight parity fixture

This fixture was generated on 2026-08-22 by the pinned upstream
`transformers==5.15.0` implementation, not by a Vokra mirror. The dumper
imports `AutoProcessor` and `ParakeetForCTC`, hooks the official feature
extractor, subsampler, encoder and CTC head, and decodes through the official
`ParakeetProcessor` CTC grouping path.

## Pinned inputs

- model: `nvidia/parakeet-ctc-1.1b`
- authenticated model revision:
  `20e63a0fed6aedba145b74b826dbd41df0941730`
- pinned Transformers source revision:
  `d56c55bf564ddb176759eb6ec199442682564916`
- tokenizer SHA-256:
  `f3f1dd45c3889ed2b5bf67180caf05f51d7d7e4948c20e5f24d8c24df9cc47aa`
- audio: `tests/fixtures/audio/jfk-30s.wav`, the existing Public Domain
  LibriVox JFK fixture documented in that directory's README
- audio SHA-256:
  `58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f`

The 176,000 PCM samples produce 1,100 valid feature frames, 138 encoder
frames, 26 collapsed CTC tokens and the transcript in `text.txt`.

## Regeneration

Run on a VAST instance because the official checkpoint is larger than 2 GB:

```sh
uv run --project tools/parity --python 3.12 python \
  tools/parity/parakeet_ctc_dump_reference.py \
  --checkpoint /path/to/pinned/parakeet-ctc-1.1b \
  --audio tests/fixtures/audio/jfk-30s.wav \
  --output-dir /tmp/parakeet-ctc-reference
```

`pcm.f32`, `features.f32`, `subsampled_unscaled.f32`, `encoder.f32` and
`logits.f32` are little-endian F32. `raw_argmax.u32` and `tokens.u32` are
little-endian U32. `metadata.json` records the complete shapes, versions,
ids and transcript. Verify every committed byte with `SHA256SUMS`.

## Fixed gates and first measurement

The bounds were declared before the first measurement and were not widened:

- encoder: max `2e-4`, mean `2e-5`
- logits: max `1e-3`, mean `1e-4`
- raw frame argmax ids, collapsed tokens and decoded text: exact equality

The green VAST measurement on the real JFK fixture was:

- encoder: max `2.574920654e-5`, mean `1.575474698e-6`
- CTC head from the upstream encoder fixture: max `1.525878906e-4`, mean
  `2.181897435e-5`
- full native PCM logits: max `2.670288086e-4`, mean `3.449702854e-5`
- 138 raw argmax ids, 26 collapsed tokens and transcript: exact

Environment: Intel Xeon E5-2650 v2, 32 logical CPUs, AVX, Torch
`2.13.0+cu130`, `torch.backends.cpu.get_cpu_capability() == "DEFAULT"`.

During diagnosis, a one-second low-energy three-tone input exceeded the
predeclared end-to-end bound because small STFT/mel differences were amplified
by feature normalization and the checkpoint's `scale_input = sqrt(1024)`.
Feeding the upstream feature fixture into the native subsampler and encoder
reduced final encoder error to max `2.342462540e-5`, proving that tensor
mapping, attention/linear/convolution biases and BatchNorm were not the
source. The production-realistic JFK fixture passes the original bounds, so
no tolerance was changed to fit an observation.
