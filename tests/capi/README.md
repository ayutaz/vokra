# C ABI smoke tests (M0-09)

C programs that exercise the Vokra C ABI through **`include/vokra.h` only**
(no Rust internals) — the WP M0-09 completion condition "call VAD / ASR / TTS
from C". They double as the single-header check (IF-01).

| File | Task | Model source |
|------|------|--------------|
| `smoke_vad.c` | Silero VAD stream: create → open → push → poll | committed `tests/parity/silero_vad/silero-vad-v5.gguf` (2 MB) |
| `smoke_vad_bytes.c` | Bytes-based create (M4-02): read GGUF → `create_from_bytes` → VAD stream | committed Silero fixture (also the WebGL emcc verify body) |
| `smoke_aec.c` | AEC (M4-03): create → ref_push → process → reset → destroy | **none** — model-free, synthetic PCM |
| `smoke_s2s.c` | Full-duplex S2S + attribution (M4-06): duplex open/push/pull/text/interrupt + `vokra_model_attribution` | committed Silero (error paths + permissive attribution) + **env** `VOKRA_MOSHI_GGUF` for the duplex leg |
| `smoke_asr.c` | Whisper: create → transcribe → free string | **env** `VOKRA_WHISPER_GGUF` (uncommitted ~290 MB) |
| `smoke_tts.c` | piper-plus native TTS: create → synthesize → free PCM | **env** `VOKRA_PIPER_GGUF` (uncommitted ~77 MB) |

ASR/TTS **SKIP cleanly (exit 0)** when their env var is unset, matching the
M0-05/06/07 parity gating (the large GGUFs are not committed). `smoke_s2s`'s
full duplex leg SKIPs the same way on `VOKRA_MOSHI_GGUF` — its error-path and
permissive-attribution legs always run. VAD / VAD(bytes) / AEC always run from
the committed fixture (AEC needs no model at all).

Audio input is raw little-endian **float32 PCM** read with `fread` — the C side
has no WAV parser and no `strtod` / locale-dependent parsing (NFR-RL-01,
enforced by `scripts/check-forbidden-symbols.sh`). The fixtures under
`fixtures/` are derived from the parity assets by `fixtures/gen_fixtures.py`.

## Run

```sh
# All three, with the symbol check and header drift check:
scripts/run-capi-smoke.sh

# ASR + TTS + the S2S duplex leg live as well:
VOKRA_WHISPER_GGUF=whisper-base.gguf VOKRA_PIPER_GGUF=voice.gguf \
    VOKRA_MOSHI_GGUF=moshi.gguf scripts/run-capi-smoke.sh
```

Or build one by hand (from the repo root, after
`cargo build -p vokra-capi --release`):

```sh
cc tests/capi/smoke_vad.c -I include -L target/release -lvokra \
    -Wl,-rpath,target/release -o /tmp/smoke_vad
/tmp/smoke_vad
```

## Followups (M0-09-T12/T13/T15)

- Wire a CI `capi` job (Linux/macOS/Windows matrix) that installs cbindgen,
  runs `scripts/gen-c-abi.sh --check`, and builds + runs these tests. The
  Windows leg builds `vokra.dll` and links the import lib with MSVC (`cl.exe`).
- ASR real detokenization needs the tokenizer embedded in the GGUF (the M0
  converter does not yet — hence the bracketed token-id transcript here).
