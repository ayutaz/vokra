# vokra-piper-g2p

Opt-in bridge that runs **real 8-language text→speech** on Vokra's native
piper-plus TTS, using the client's own [`piper-plus-g2p`] for grapheme→phoneme
conversion.

```text
text ──piper-plus-g2p──▶ phoneme ids + prosody (A1,A2,A3) + language id ──▶ Vokra native piper-plus ──▶ WAV
```

## Why it lives outside the workspace

Vokra's runtime is **zero-external-dependency** (NFR-DS-02): the root
`Cargo.lock` may contain only `vokra-*` crates, enforced in CI. `piper-plus-g2p`
pulls `jpreprocess`, `regex`, `serde`, … — all third-party — so it **cannot**
live in the runtime workspace.

This crate is therefore its **own isolated workspace** (note the empty
`[workspace]` table in its `Cargo.toml`), with its own `Cargo.lock`. The root
workspace `exclude`s `integrations/`, so building or testing Vokra never sees
these dependencies. The third-party G2P is injected into the runtime across the
`vokra_piper_plus::Phonemizer` trait boundary — the runtime crates it links
(`vokra-models`, `vokra-piper-plus`, `vokra-core`) stay zero-dependency.

`piper-plus-g2p` is pinned by git rev in `Cargo.toml`; the first build fetches
it and the Japanese dictionary (`naist-jdic`, bundled into the binary — no
runtime download).

## Usage

```sh
cargo run --release -- --voice voice.gguf --text "こんにちは" --lang ja --out hello.wav

# Zero-shot voice cloning from a reference utterance (native CAM++ encoder):
cargo run --release -- --voice voice.gguf --text "Hello" --lang en \
    --ref reference.wav --speaker-gguf campplus.gguf --out cloned.wav

# Inspect the G2P output (phoneme ids / prosody / language id) without synthesizing:
cargo run --release -- --voice voice.gguf --text "こんにちは" --lang ja --dump
```

`--voice` is a GGUF produced by `vokra-convert` from a piper-plus checkpoint
(e.g. `ayousanz/piper-plus-zero-shot-multi-6lang-v7`). `--lang` is optional; the
multilingual phonemizer auto-detects the dominant language otherwise.

## Faithfulness

The pipeline mirrors piper-plus's own inference
(`piper_plus.api.PiperTTS._phonemize`): `MultilingualPhonemizer` →
`PiperEncoder.encode_with_prosody`, with the encoder's phoneme id map
reconstructed from the **voice's own** `vokra.piper.phoneme_symbols`, so ids are
byte-correct for that checkpoint. The native model's `[3 → 16]` prosody
projection that consumes the `(A1, A2, A3)` triples is parity-checked against
onnxruntime with a non-zero prosody buffer
(`tests/parity/piper_plus_v7_prosody/`).

Japanese and English are exact; `zh` currently falls back to a passthrough
phonemizer (no bundled loanword dict here).

[`piper-plus-g2p`]: https://github.com/ayutaz/piper-plus/tree/main/src/rust/piper-plus-g2p
