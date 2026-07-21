# Desktop CLI tutorial

**English** | [日本語](cli.ja.md)

`vokra-cli` is the umbrella command-line tool (`FR-TL-01`, `FR-TL-02`): three
subcommands — `run`, `convert`, `bench` — over the same native runtime, with
hand-written argument parsing and no external dependency (`NFR-DS-02`). This is
the deep dive; for the 5-minute path see
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

## 5. Backend selection is explicit (`FR-EX-08`)

`--backend` chooses the compute backend. Vokra never silently falls back: an op
a GPU backend does not cover, or a device that is absent, is an explicit error,
not a quiet drop to CPU.

```sh
cargo build --release -p vokra-models --features metal   # macOS
./target/release/vokra-cli bench --model whisper-large-v3.gguf \
  --input speech30s.wav --backend metal
```

Use `--backend cpu` to choose the CPU *deliberately* — that is a decision you
make, not one Vokra makes behind your back.

## 6. Troubleshooting

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

**Last verified: 2026-07-21 — against the `run` / `convert` / `bench` argument
parsers in `crates/vokra-cli/src/`.**

- **Update responsibility**: a PR that adds or renames a CLI flag updates this
  page and its Japanese twin in the same PR. Every `vokra-cli` invocation here
  is checked against the real parsers by the `doc-examples` CI job, so a stale
  flag fails CI.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`).
- **Re-fetch the flag surface**:

```sh
grep -oE '"--[a-z0-9-]+"' crates/vokra-cli/src/run.rs crates/vokra-cli/src/convert.rs crates/vokra-cli/src/bench.rs
```
