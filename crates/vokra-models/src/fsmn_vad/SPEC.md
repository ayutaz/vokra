# FSMN-VAD — implementation spec (SoTA plan Phase 5 VAD-2)

Single source for the FSMN-VAD subgraph: architecture, the GGUF weight
map, the exact numeric details pinned against upstream FunASR, and the
parity methodology. Source of truth for the code in this directory.

Upstream: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` (FunASR family)
— feed-forward sequential memory network for voice activity detection.
License MIT / MIT — commercial use OK
(`docs/license-audit.md` §3.1 row landed 2026-07-30 yousan).

## Design red lines (permanent)

- **First-class audio-dialect op (distinct posture from Silero VAD v5).**
  FSMN-VAD's architecture is a stack of stateless feed-forward + memory
  blocks over Kaldi fbank + LFR (Low Frame Rate) + CMVN — a natural fit
  for graph-level ops. This is why it does NOT get the 1:1-subgraph
  treatment Silero VAD v5 required (FR-LD-06 applies only when a model's
  topology cannot be lowered cleanly).
- **No silent CPU fallback (FR-EX-08).** Every shape / streaming /
  state-mismatch precondition is a hard error.
- **Streaming ⇔ batch invariance.** Per the FSMN-streaming contract,
  feeding `n_frames` in one call must match feeding them one-at-a-time
  with state carried; the numeric core (`vokra-ops::fsmn_vad`) pins this
  under an internal-oracle test.

## Architecture (upstream reference)

Source: FunASR `funasr/models/fsmn_vad_streaming/model.py` +
`funasr/models/fsmn_vad_streaming/encoder.py`.

```text
PCM chunk (16 kHz mono f32, typically 200 ms window)
 -> Kaldi fbank (80-d, 25 ms window, 10 ms hop, Povey window, snip-edges,
                 per-frame DC removal + pre-emphasis 0.97) via shared
                 vokra_ops::kaldi_fbank — matches upstream FunASR's WavFrontend
                 (kaldi.fbank + LFR + CMVN).
 -> LFR (Low Frame Rate) frame stacking: stack lfr_m (=5) consecutive
    frames and stride by lfr_n (=1), producing a lfr_m * n_mels-wide
    feature per output step.
 -> CMVN (global mean-variance normalization) — reads the checkpoint's
    mean_stats / var_stats.
 -> FSMN encoder stack: 4 FSMN blocks (upstream default), each block:
        input -> Linear (proj_dim -> hidden_dim) + Affine bias
             -> ReLU
             -> Linear (hidden_dim -> proj_dim)      // ffn2 + bias
             -> memory block (depthwise Conv1d with left_padding=lorder,
                              right_padding=rorder, groups=proj_dim)
             -> residual add (input + ffn_output + memory_output)
 -> Linear projection (proj_dim -> n_class) + softmax
 -> emit per-frame class probabilities [t, n_class] (n_class = 2:
    [silence, speech] by upstream convention).
```

## Config axes (upstream defaults for the released checkpoint)

Every value is transcribed from the upstream release; the runtime
rejects `0`-sentinels at load per FR-EX-08.

| axis | default | note |
|---|---|---|
| `n_blocks` | 4 | number of stacked FSMN encoder blocks |
| `input_dim` | 400 | LFR-stacked fbank dim (`lfr_m * n_mels` = 5 * 80) |
| `proj_dim` | 128 | FSMN block input/output width |
| `hidden_dim` | 128 | FSMN block inner (ReLU) width |
| `lorder` | 20 | past frames the memory block sees |
| `rorder` | 0 | future frames (0 for streaming) |
| `n_class` | 2 | output classes: [silence, speech] |
| `n_mels` | 80 | Kaldi fbank bins per frame |
| `lfr_m` | 5 | LFR stacking window |
| `lfr_n` | 1 | LFR stride |
| `sample_rate` | 16000 | input PCM Hz |

## GGUF metadata (`vokra.fsmn_vad.*`)

The converter (`vokra-convert::models::fsmn_vad`) writes these keys so
`FsmnVadConfig::from_gguf` can bind them back:

- `vokra.model.arch` = `"fsmn-vad"`
- `vokra.model.name` = `"fsmn-vad-zh-cn-16k-common"` (default; caller can
  override at CLI time)
- `vokra.model.category` = `"vad"` — silence-vs-speech classifier posture
- `vokra.provenance.upstream_hf` = `"iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"`
- `vokra.provenance.weight_license` = `"permissive"` (MIT)
- `vokra.provenance.license` = `"mit"`
- `vokra.fsmn_vad.n_blocks` = `4`
- `vokra.fsmn_vad.input_dim` = `400`
- `vokra.fsmn_vad.proj_dim` = `128`
- `vokra.fsmn_vad.hidden_dim` = `128`
- `vokra.fsmn_vad.lorder` = `20`
- `vokra.fsmn_vad.rorder` = `0`
- `vokra.fsmn_vad.n_class` = `2`
- `vokra.fsmn_vad.n_mels` = `80`
- `vokra.fsmn_vad.lfr_m` = `5`
- `vokra.fsmn_vad.lfr_n` = `1`
- `vokra.fsmn_vad.sample_rate` = `16000`

## Real-weight parity (deferred — owner sign-off)

The initial land ships the numeric primitives (op + model wrapper +
converter) plus synthetic-weight invariant tests. Real-weight parity
against the upstream FunASR Python pipeline is deferred to owner per
the standing rule for skeleton models
(`docs/license-audit.md` §3.1 sign-off already recorded 2026-07-30;
the real-weight harness gets added when the upstream checkpoint is
first pulled and the fbank / LFR / CMVN chain is oracled against
`funasr.utils.postprocess_utils.sentence_postprocess`).

The current tests pin:

- structural: `FsmnVadV1::from_gguf` binds every documented tensor
  under its published name;
- streaming ⇔ batch invariance (numeric core in `vokra-ops`);
- residual identity when the FFN + memory weights are zeroed
  (numeric core in `vokra-ops`);
- softmax numerical stability (numeric core in `vokra-ops`);
- all shape / state / config-mismatch violations are loud errors
  (FR-EX-08).
