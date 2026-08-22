# Desktop CLI tutorial

**English** | [日本語](cli.ja.md)

`vokra-cli` is the umbrella command-line tool (`FR-TL-01`, `FR-TL-02`): four
subcommands — `run`, `convert`, `bench`, `f0` — over the same native runtime,
with hand-written argument parsing and no external dependency (`NFR-DS-02`).
This is the deep dive; for the 5-minute path see
[getting-started.md](../getting-started.md).

## 1. Build

```sh
cargo build --release
```

This produces `target/release/vokra-cli`. Run `vokra-cli <subcommand> --help`
for that subcommand's full option list.

## 2. `run` — inference with an auto-selected task

`run` loads a GGUF and picks the task from the model's `vokra.model.arch`
metadata (Whisper → ASR, Silero VAD → VAD, piper-plus → TTS), so you do not
name the task yourself:

```sh
# ASR — audio in, text out
./target/release/vokra-cli run --model whisper-base.gguf --input speech.wav

# TTS — text in, WAV out
./target/release/vokra-cli run --model voice.gguf \
  --text "Hello from Vokra." --output hello.wav
```

ASR has decoding controls: `--beam-size`, `--word-timestamps`,
`--length-penalty`, `--no-repeat-ngram`, `--language`. TTS has `--voice`,
`--style` and `--length-scale`:

```sh
./target/release/vokra-cli run --model whisper-base.gguf --input speech.wav \
  --beam-size 5 --word-timestamps
```

### CT-Punc paired token input

CT-Punc deliberately does not infer tokenization. Supply the token strings and
the exact vocabulary ids passed to the model in one versioned UTF-8 TSV file:

```
vokra-ct-punc-tsv-v1
101	we
202	build
303	世界
```

Each record is `<u32 id><TAB><escaped token>`. A token may contain literal
Unicode or the escapes `\\`, `\t`, `\n`, `\r`, and `\u{HEX}`. Empty records,
extra TSV columns, malformed escapes, and out-of-range ids are errors. The
single record stream makes a token/id length mismatch unrepresentable.

```sh
./target/release/vokra-cli run --model ct-punc.gguf \
  --tokens tokens.tsv --output restored.txt
```

Without `--output`, the restored text is printed as `ct_punc: <text>`. With
`--output`, the file is the exact restored UTF-8 text without a diagnostic
prefix or implicit newline.

### Mimi encode/decode and the portable code container

Mimi has explicit directions; nested Rust arrays are never used as an
interchange format:

```sh
./target/release/vokra-cli run --model mimi.gguf --codec-mode encode \
  --input speech-24k.wav --output speech.vmc
./target/release/vokra-cli run --model mimi.gguf --codec-mode decode \
  --input speech.vmc --output reconstructed.wav
```

`speech.vmc` is `VKRMCODE` version 1. Its fixed little-endian header pins mono
channel count, sample rate, frame rate in milli-Hz, frame count, original PCM
sample count, codebook count/size, feature width, and the SHA-256 of the
effective GGUF codebook tables. The payload is unsigned 32-bit codes in
time-major `[frame, codebook]` order. Decode rejects a different model hash or
topology, trailing/truncated bytes, out-of-range codes, and a PCM length that
does not equal `frames * model_hop`. Encode likewise requires an exact positive
frame-hop multiple; there is no implicit resample, padding, or trimming.

## 3. `convert` — checkpoint → GGUF (offline)

The runtime loads **GGUF only**; ONNX / safetensors are handled here, offline.
`--model` names the source kind, and `--quantize` K-quantizes on the way out:

```sh
./target/release/vokra-cli convert --model whisper \
  --input whisper-base/model.safetensors --output whisper-base.gguf

# smaller footprint via K-quant
./target/release/vokra-cli convert --model whisper \
  --input whisper-base/model.safetensors --output whisper-base.q4_k.gguf \
  --quantize q4_k
```

A piper-plus voice needs its `config.json` too; some models take a `--tokenizer`
or `--adapter-config` side-car:

```sh
./target/release/vokra-cli convert --model piper-plus \
  --input voice.onnx --config voice.config.json --output voice.gguf
```

## 4. `bench` — RTF / TTFA / jitter, with a regression gate

`bench` reports real-time factor, time-to-first-audio, jitter and p50/p95/p99
latencies. `--baseline` turns it into a **regression gate**: a >5% relative
slowdown versus the recorded baseline exits non-zero (`NFR-PF-13`).

```sh
# measure
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav \
  --iters 20 --warmup 3 --format json

# gate against a recorded baseline
./target/release/vokra-cli bench --model whisper-base.gguf --input speech.wav \
  --baseline baseline.json
```

## 5. `f0` — pitch extraction with no checkpoint

`f0` runs YIN or PyIN over a WAV. It is a separate subcommand rather than a
`run --task` because `run` requires a `--model` GGUF, and these two extractors
carry no weights at all — no checkpoint, no license class, no
`docs/license-audit.md` §3.1 row. There is no `--model` a caller could supply.

```sh
# YIN (default)
./target/release/vokra-cli f0 --input speech.wav

# PyIN, restricted to a speaking range
./target/release/vokra-cli f0 --input speech.wav --algo pyin \
  --fmin 65 --fmax 400
```

Rows are tab-separated and identical in shape to what `run` prints for a
neural F0 model, so whatever parses them does not change when you switch
extractor:

```
time_sec<TAB>hz<TAB>voiced<TAB>confidence
```

An unvoiced frame is `hz=0.000`, `voiced=false`. PyIN reports its real
per-frame voiced probability in `confidence`; YIN has no probability output,
so its confidence remains the explicit binary `1.0` / `0.0` alongside `voiced`.
rather than a fabricated score.

The sample rate is **not** fixed: both ops derive their lag search from the
rate the WAV carries, so nothing is silently resampled. The neural members of
the same family — RMVPE, FCPE, CREPE — do need a checkpoint and stay on
`run`.

## 6. Backend selection is explicit (`FR-EX-08`)

`--backend` chooses the compute backend. Vokra never silently falls back: an op
a GPU backend does not cover, or a device that is absent, is an explicit error,
not a quiet drop to CPU.

```sh
cargo build --release -p vokra-cli --features metal   # macOS
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend metal
```

Use `--backend cpu` to choose the CPU *deliberately* — that is a decision you
make, not one Vokra makes behind your back.

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `error: model file has no vokra.model.arch metadata` | The GGUF came from a non-Vokra tool (e.g. `llama.cpp`). Regenerate with `vokra-cli convert`. |
| `error: backend does not implement op X` | A GPU backend does not cover that op (`FR-EX-08`). Retry with `--backend cpu` or file the model/op. |
| `bench` exits non-zero with a regression message | The `--baseline` gate fired (>5% slower). Investigate the change or refresh the baseline intentionally. |
| `error: research flag required for CC-BY-NC weight` | A non-commercial weight was refused by the compliance gate; explicit opt-in is required for research use. |

## Next steps

- [Server (four compatibility APIs)](server.md) — the separate `vokra-server`
  binary, if you want HTTP endpoints rather than a CLI
- [Adding a backend](../backend-guide.md)
- [Migration Guide](../migration-guide.md) (from ONNX Runtime / whisper.cpp /
  sherpa-onnx)

## Keeping this page current

**Last verified: 2026-08-21 — against the `run` / `convert` / `bench` / `f0`
argument parsers in `crates/vokra-cli/src/`.**

- **Update responsibility**: a PR that adds or renames a CLI flag updates this
  page and its Japanese twin in the same PR. Every `vokra-cli` invocation here
  is checked against the real parsers by the `doc-examples` CI job, so a stale
  flag fails CI.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`).
- **Re-fetch the flag surface**:

```sh
grep -oE '"--[a-z0-9-]+"' crates/vokra-cli/src/run.rs crates/vokra-cli/src/convert.rs crates/vokra-cli/src/bench.rs crates/vokra-cli/src/f0.rs
```
