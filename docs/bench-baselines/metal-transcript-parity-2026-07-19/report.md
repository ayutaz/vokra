# Whisper Metal-vs-CPU transcript parity — all four sizes (2026-07-19)

**cc-26** from the M4-residual audit. The audit found that the campaign-2
`driver.log` terminated mid-run, so the claim "small/turbo Metal byte-identical"
was not backed by a complete artifact. This run re-measures **all four sizes**
end to end and commits the evidence.

## Method

Release `vokra-cli` built with `--features metal` (Apple M1 iMac, macOS arm64).
For each size the same GGUF and the same audio are transcribed twice — once with
`--backend metal`, once with `--backend cpu` — and the emitted transcript lines
are compared with `diff`. Byte-identical is the pass condition (not an atol):
greedy decoding is discrete, so any numerical divergence large enough to change
a token would show up here.

```
vokra-cli run --model <gguf> --input jfk-30s.wav --backend metal
vokra-cli run --model <gguf> --input jfk-30s.wav --backend cpu
```

- GGUFs: `~/.cache/vokra-eval/gguf/whisper-{base,small,medium,turbo}.gguf`
  (real upstream checkpoints converted during the 2026-07-16 campaign).
- Audio: `~/.cache/vokra-eval/corpus/jfk-30s.wav` (11.0 s of speech, the
  committed `tests/fixtures/audio/jfk-30s.wav`).

## Result — 4/4 byte-identical

| size | Metal vs CPU | transcript |
|---|---|---|
| base | **byte-identical** | " And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country." |
| small | **byte-identical** | (same) |
| medium | **byte-identical** | (same) |
| turbo | **byte-identical** | (same) |

All four also match the canonical JFK reference transcript.

## Scope / honesty notes

- This pins **transcript equality**, i.e. that the Metal path produces the same
  greedy token sequence as the CPU path. It is not a per-tensor parity claim and
  not a performance measurement (Metal is memory-bound and slower than CPU at
  these sizes on this machine — see `docs/bench-baselines/m5-14-final-2026-07-18/`).
- CUDA has no analogue here: this machine has no NVIDIA GPU. The CUDA transcript
  check remains an owner leg (vast.ai), as recorded in the owner checklist.
