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

- **Python 3.12** with the `misaki` package. Create the pinned environment
  through uv from the repository root:

  ```sh
  uv sync --project integrations/vokra-misaki-g2p --frozen
  ```

  `pyproject.toml` + `uv.lock` pin `misaki[en,ja,zh,ko]` — the four languages
  Kokoro-82M ships trained voices for. The `parity` dependency group adds the
  independently pinned Kokoro/torch stack used by `parity-kokoro-real.yml`.
  Misaki 0.9.4 omits its POSIX Korean `mecab` provider from the `ko` extra,
  so this project also pins `python-mecab-ko`; uv locks its matching Korean
  dictionary wheel transitively.
  The English spaCy model is also locked explicitly; without that pin,
  `misaki.en` downloads `en_core_web_sm` on first use and the next frozen uv
  sync removes it as an undeclared package.

  Korean G2P additionally asks NLTK for the `cmudict` corpus on first import.
  That corpus is not a Python wheel and is therefore outside `uv.lock`; keep
  `NLTK_DATA` on persistent storage when using Korean, and do not describe a
  successful package sync alone as an offline-complete Korean data setup.
  Japanese explicitly uses upstream `JAG2P(version="pyopenjtalk")`; the
  default cutlet/fugashi path needs an untracked UniDic post-install download
  and is therefore not used by this bridge. `pyopenjtalk` still downloads its
  own Open JTalk dictionary into its installed package directory on first use,
  so keep the uv environment itself on persistent storage if that data should
  survive between runs. A locked Python environment is reproducible; these two
  upstream language-data downloads are separate, explicit setup inputs and
  prevent an unfetched installation from being called offline-complete.

  Upstream `kokoro`/`misaki[en]` resolves `phonemizer-fork` and
  `espeakng-loader`. Those GPL components are confined to this opt-in,
  workspace-excluded reference environment; they are never linked, loaded,
  invoked, bundled, or distributed by Vokra's runtime, model, C ABI, Unity,
  Godot, or official package artifacts. Do not promote this integration's
  Python environment into a shipped Vokra component.

- **Kokoro GGUF converted with `--config`** so the `vokra.kokoro.phoneme_symbols`
  table is present:

  ```sh
  vokra-cli convert --model kokoro --config kokoro-config.json \
      --input kokoro.pth --output kokoro-82m.gguf
  ```

  A voice converted without `--config` has no symbol table, and this bridge
  will fail loudly on the id lookup step.

## Usage

**Execution policy:** this integration has a path dependency on
`vokra-models`; its `cargo run` recipe compiles the model runtime and then
executes a real Kokoro model. Run conversion, build, and model execution on
VAST or another appropriately sized host, not on the 16 GB maintainer Mac.
The uv environment setup above is the only local preparation expected here;
this documentation update did not run the model.

```sh
# American English with an inline voice name:
cargo run --release -- \
    --kokoro kokoro-82m.gguf --text "Hello world" \
    --lang en --voice af_bella --out hello.wav

# Japanese with a venv-scoped Python (recommended):
cargo run --release -- \
    --kokoro kokoro-82m.gguf --text "こんにちは" \
    --lang ja --voice jf_alpha \
    --python integrations/vokra-misaki-g2p/.venv/bin/python --out hi.wav

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

## License boundary

Apache-2.0 (matches the Vokra runtime). misaki is Apache-2.0 upstream; the
Rust wrapper does not vendor its source. The Korean binding is BSD-3-Clause
and its dictionary wheel is Apache-2.0. The uv lock records the isolated Python
reference environment, including the upstream GPL eSpeak loader noted above;
none of those Python packages are part of a Vokra binary or official runtime
distribution.
