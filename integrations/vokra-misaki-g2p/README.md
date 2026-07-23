# vokra-misaki-g2p

Opt-in bridge that runs **real text → Kokoro-82M speech** on Vokra's native
Kokoro TTS, using the **upstream [`misaki`](https://github.com/hexgrad/misaki)**
Python package for grapheme → phoneme conversion.

```text
text ──(subprocess: python misaki_bridge.py)──▶ IPA phoneme string
     ──(this crate: KokoroConfig.phoneme_symbols lookup)──▶ phoneme ids
     ──▶ vokra-models::kokoro::KokoroTts::synthesize_phonemes ──▶ WAV
```

## Why it lives outside the workspace

Vokra's runtime is **zero-external-dependency** (NFR-DS-02): the root
`Cargo.lock` may contain only `vokra-*` crates, enforced in CI. `misaki` is a
Python package — it cannot live in the runtime workspace, and re-writing it in
Rust would drift from Kokoro's training distribution by construction (the
reference IS the Python code).

This crate is therefore its **own isolated workspace** (empty `[workspace]`
table in `Cargo.toml`), with its own `Cargo.lock`. The root workspace
`exclude`s `integrations/`, so building or testing Vokra never sees this
crate. The Python bridge is confined to `python/misaki_bridge.py`; the runtime
crates it links (`vokra-core`, `vokra-models`) stay zero-dependency.

Contrast with `integrations/vokra-piper-g2p`, which links a **Rust** G2P
(`piper-plus-g2p`) — same isolation posture, different transport.

## Prerequisites

- **Python 3.9+** with the `misaki` package. Install into a venv:

  ```sh
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

  `requirements.txt` pulls `misaki[en,ja,zh,ko]` — the four languages Kokoro-82M
  ships trained voices for. Drop the extras you do not need (e.g. keep only
  `[en]`) if you want a leaner env.

- **Kokoro GGUF converted with `--config`** so the `vokra.kokoro.phoneme_symbols`
  table is present:

  ```sh
  vokra-cli convert --model kokoro --config kokoro-config.json \
      --input kokoro.pth --output kokoro-82m.gguf
  ```

  A voice converted without `--config` has no symbol table, and this bridge
  will fail loudly on the id lookup step.

## Usage

```sh
# American English with an inline voice name:
cargo run --release -- \
    --kokoro kokoro-82m.gguf --text "Hello world" \
    --lang en --voice af_bella --out hello.wav

# Japanese with a venv-scoped Python (recommended):
cargo run --release -- \
    --kokoro kokoro-82m.gguf --text "こんにちは" \
    --lang ja --voice jf_alpha \
    --python .venv/bin/python --out hi.wav

# Inspect the phoneme id sequence without synthesizing:
cargo run --release -- \
    --kokoro kokoro-82m.gguf --text "Hello" --lang en --dump
```

Full options: `cargo run --release -- --help`.

Supported `--lang` values: `en` (US default), `en-gb`, `ja`, `zh`, `ko`. These
are the sub-modules misaki exports today; anything else fails at parse time
with a loud error.

## What is fail-closed

The bridge treats any of the following as a hard error, never a silent skip:

- misaki not installed for the requested language (`ImportError` is surfaced).
- misaki raising during `G2P(text)` (upstream error text is quoted verbatim).
- a phoneme character misaki emits that the Kokoro voice's `phoneme_symbols`
  table does not contain (names the offending character and its Unicode code
  point).
- an empty or duplicated `phoneme_symbols` table (a converter drift signal).

## Zero-dependency invariant (NFR-DS-02)

- Root `Cargo.lock`: unchanged. This crate is not a workspace member.
- This crate's `Cargo.lock`: contains `vokra-*` + std only (no third-party Rust
  deps today; the JSON parsing is hand-written to avoid pulling `serde_json`).
- Third-party `misaki`: reached only via Python subprocess, never linked.

## License

Apache-2.0 (matches the Vokra runtime). misaki is Apache-2.0 upstream; the
Rust wrapper does not vendor its source.
