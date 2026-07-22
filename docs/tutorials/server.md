# Server tutorial

**English** | [日本語](server.ja.md)

`vokra-server` (`integrations/vokra-server` <!-- anchor: integrations/vokra-server -->)
is the HTTP server: **four compatibility APIs from a single binary**, targeting
the server platform Vokra treats as first-class (`NFR-PT-01` — x86-64 / ARM64
servers are not optional). It is an isolated workspace, so it does not perturb
the root `Cargo.lock` zero-dependency invariant (`NFR-DS-02`). This is a
different binary from `vokra-cli` — see the [CLI tutorial](cli.md) for that.

## 1. Build and launch (single binary, Docker-free)

Vokra does not require Docker (`FR-SV-01`): the server is one static-friendly
binary you launch with model GGUFs.

```sh
cargo build --release -p vokra-server
./target/release/vokra-server \
  --http-bind 127.0.0.1:8080 \
  --whisper-base whisper-base.gguf --whisper-base-tokenizer tok.gguf \
  --piper-plus voice.gguf --piper-g2p
```

`--piper-g2p` enables plain-text TTS via the real 8-language G2P; without it,
plain-text `/api/tts` returns an explicit 400 and only raw phoneme-id payloads
work (`FR-EX-08` — no silent behaviour change).

## 2. The four compatibility APIs

One process serves all four (`IF-05` covers the piper-plus / Home Assistant
interfaces):

| API | Endpoint | Requirement |
|---|---|---|
| OpenAI Whisper | `POST /v1/audio/transcriptions` | `FR-SV-02` |
| vLLM-compatible | `POST /v1/completions`, `POST /v1/chat/completions` | vLLM HTTP compatibility |
| piper-plus HTTP | `POST /api/tts` | `FR-SV-04` |
| Wyoming Protocol | `--wyoming-bind` (Home Assistant) | `FR-SV-05` |

```sh
# OpenAI-compatible transcription (faster-whisper drop-in)
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F model=whisper-base

# piper-plus TTS
curl -s http://127.0.0.1:8080/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Vokra."}' --output hello.wav
```

## 3. Model and backend flags

Load the models you need by flag; the whisper family covers base / small /
medium / turbo / large-v3, each with its `-tokenizer` companion, and there are
`--piper-plus`, `--kokoro`, `--voxtral` and `--silero-vad` slots:

```sh
./target/release/vokra-server --http-bind 0.0.0.0:8080 \
  --whisper-large-v3 whisper-large-v3.gguf \
  --whisper-large-v3-tokenizer tok.gguf \
  --backend cuda
```

`--backend` selects the compute backend explicitly; an unavailable device or an
uncovered op is an explicit error, never a silent CPU fallback (`FR-EX-08`).

## 4. Multi-session and concurrency

The server handles concurrent sessions (`FR-SV-06`); cap them with
`--max-concurrent-sessions` so a burst cannot exhaust memory:

```sh
./target/release/vokra-server --http-bind 0.0.0.0:8080 \
  --whisper-base whisper-base.gguf --whisper-base-tokenizer tok.gguf \
  --max-concurrent-sessions 8
```

## 5. Bind address and security posture

`--http-bind` and `--wyoming-bind` take an explicit `host:port`. Bind to
`127.0.0.1` for local-only use; bind to `0.0.0.0` only behind your own
authenticating reverse proxy — the server does not add authentication itself.
Every configuration surface is validated at startup, and a missing model file
is a hard error rather than a lazy failure on first request.

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| plain-text `/api/tts` returns 400 | Launch with `--piper-g2p`; without it only phoneme-id payloads are accepted (§1). |
| startup error naming a model path | A `--whisper-*` / `--piper-plus` / … path does not exist; the server fails fast rather than 500-ing later. |
| `BackendUnavailable` at launch with `--backend cuda` | No usable CUDA driver/GPU; drop to `--backend cpu` *explicitly* (`FR-EX-08`). |
| Home Assistant does not see the server | Pass `--wyoming-bind host:port` and register that address in HA (`FR-SV-05`). |

## Next steps

- [Desktop CLI](cli.md) — the `convert` step that produces the GGUFs the
  server loads
- [Migration Guide](../migration-guide.md) — faster-whisper / OpenAI API
  drop-in details
- [Adding a backend](../backend-guide.md)

## Keeping this page current

**Last verified: 2026-07-21 — against the flag surface in
`integrations/vokra-server/src/config.rs`
<!-- anchor: integrations/vokra-server/src/config.rs -->.**

- **Update responsibility**: a PR that adds or renames a server flag or an
  endpoint updates this page and its Japanese twin in the same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`).
- **Re-fetch the flag surface**:

```sh
grep -oE '"--[a-z][a-z0-9-]+"' integrations/vokra-server/src/config.rs | sort -u
```
