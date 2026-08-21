# FSMN-VAD implementation contract

This document pins the native runtime to the released FunASR FSMN-VAD model.

- Hugging Face: `funasr/fsmn-vad`
- Hugging Face revision: `df20e6b30c653645fa4ff125cacfcabd1020a669`
- ModelScope identity: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- reference code: FunASR commit `3c58cb56a56598232c3efffa15d313d7e82a4307`
- weight license: Apache-2.0; FunASR reference code: MIT
- `model.pt` SHA-256: `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5`
- `am.mvn` SHA-256: `df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739`
- `config.yaml` SHA-256: `486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93`

## Frontend

The streaming path matches `funasr/frontends/wav_frontend.py`:

1. normalized mono PCM is multiplied by 32768;
2. Kaldi fbank uses 16 kHz, 80 bins, 25 ms frames, 10 ms shift, Hamming
   window, DC removal, pre-emphasis 0.97, no dither, snip edges, power
   spectrum, HTK mel, and no per-utterance CMN;
3. LFR `m=5, n=1` prepends two copies of the first fbank row and emits only
   complete five-row windows on a non-final stream, retaining the four-row
   tail;
4. `am.mvn` is an affine transform, exactly `(x + AddShift) * Rescale`.

The VAD trait has no finalization method, so the runtime implements the
official non-final online contract and does not invent right-edge padding.

## Encoder

The exact `funasr/models/fsmn_vad_streaming/encoder.py` topology is:

```text
400 -> affine 140 -> affine 250 -> ReLU
 -> 4 × {
      bias-free linear 250 -> 128
      causal depthwise memory: 20 taps, stride 1, projected residual
      affine 128 -> 250 -> ReLU
    }
 -> affine 250 -> 140 -> affine 140 -> 248 -> softmax
```

There is no two-class head. The 248 outputs are acoustic pdfs, pdf 0 is the
sole silence pdf (`silence_pdf_num=1`, `sil_pdf_ids=[0]`), and the runtime VAD
score is `1 - probability[0]`.

## Canonical tensor manifest

The converter requires exactly 24 F32 weights:

- `encoder.in_linear{1,2}.linear.{weight,bias}`
- for blocks `0..4`:
  - `encoder.fsmn.<i>.linear.linear.weight` `[128,250]`
  - `encoder.fsmn.<i>.fsmn_block.conv_left.weight` `[128,1,20,1]`
  - `encoder.fsmn.<i>.affine.linear.weight` `[250,128]`
  - `encoder.fsmn.<i>.affine.linear.bias` `[250]`
- `encoder.out_linear1.linear.{weight,bias}` (`[140,250]`, `[140]`)
- `encoder.out_linear2.linear.{weight,bias}` (`[248,140]`, `[248]`)

`tools/parity/fsmn_vad_prepare_checkpoint.py` verifies the three source hashes
and embeds two additional reserved tensors for the 400-value AddShift and
Rescale vectors. The converter requires all 26 prepared tensors, writes only
the 24 network weights, and persists the two vectors as GGUF F32 arrays. A
plain weight-only safetensors file is rejected.

## GGUF identity and geometry

Canonical GGUFs stamp the HF and ModelScope identities, revision, all three
source hashes, Apache-2.0 weight provenance, and these required axes:

| key suffix | value |
|---|---:|
| `n_blocks` | 4 |
| `input_dim` | 400 |
| `input_affine_dim` | 140 |
| `linear_dim` | 250 |
| `proj_dim` | 128 |
| `lorder`, `rorder` | 20, 0 |
| `lstride`, `rstride` | 1, 0 |
| `output_affine_dim` | 140 |
| `output_dim` | 248 |
| `n_mels` | 80 |
| `lfr_m`, `lfr_n` | 5, 1 |
| `sample_rate` | 16000 |

The loader requires exact tensor shapes and all identity metadata. Historical
scaffold GGUFs used invented tensor names, identity CMVN, and a two-class head;
they fail closed rather than being guessed onto this topology.

## Real-weight parity

`tools/parity/fsmn_vad_reference.py` directly executes the official pinned
FunASR `WavFrontendOnline` and `FSMN` classes. Its committed fixture contains
the exact integer PCM stimulus, 96 non-final online LFR rows (`96×400`), all
posterior values (`96×248`), and VAD scores.

The VAST calibration measured:

- network posterior max absolute error: `8.344650269e-7`;
- one-shot PCM score max absolute error: `1.370906830e-6`;
- 173-sample chunked PCM score max absolute error: `1.370906830e-6`.

`parity_fsmn_vad_real` fixes the corresponding gates at `2e-6` for the network
and `5e-6` for PCM/streaming. These are measured parity bounds, not synthetic
mirrors or guessed tolerances.
