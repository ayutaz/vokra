# vokra-server

Single-binary Vokra API server: OpenAI-compatible + vLLM-compatible HTTP
endpoints, piper-plus `/api/tts`, and a Wyoming Protocol JSONL-over-TCP
listener. Deliberately **out of the Vokra root workspace** so the HTTP /
async / serde stack it links (axum, hyper, tokio, tower, serde,
serde_json) never touches the root `Cargo.lock` (NFR-DS-02, enforced by
`scripts/check-zero-deps.sh`).

This README documents the M2-09 (v0.5) delivery: single-binary launch,
CLI/config surface, bind/security posture, musl static distribution,
the 4-API compatibility matrix, and connection examples for
faster-whisper and Home Assistant Wyoming.

## Single-binary launch (Docker-free)

`vokra-server` is a single statically-linked executable. **No Docker,
no container runtime, no companion services are required** (FR-SV-01).

```
# CPU-only build (default, zero-dep root Cargo.lock preserved)
cd integrations/vokra-server
cargo build --release

# Launch with a Whisper base GGUF and a piper-plus voice GGUF
./target/release/vokra-server \
    --asr-base /path/to/whisper-base.gguf \
    --tts-piper /path/to/piper-voice.gguf
```

By default the HTTP listener binds `127.0.0.1:8080` and the Wyoming
JSONL-over-TCP listener binds `127.0.0.1:10300`. Both are loopback-only
on purpose (§Security below). Publishing either to the LAN is opt-in
via `--http-bind` / `--wyoming-bind`.

Graceful shutdown drains in-flight HTTP requests and Wyoming sessions
on `SIGINT` / `SIGTERM`. Startup calls `setlocale(LC_NUMERIC, "C")` so
JSON number parsing is deterministic under European locales
(NFR-RL-01, plan §D7 / R4).

### CLI / config surface

Configuration is layered: CLI flag > env var > TOML config file >
built-in default.

| Flag | Env var | Config key | Default | Notes |
|---|---|---|---|---|
| `--http-bind` | `VOKRA_HTTP_BIND` | `http.bind` | `127.0.0.1:8080` | Public exposure requires reverse proxy |
| `--wyoming-bind` | `VOKRA_WYOMING_BIND` | `wyoming.bind` | `127.0.0.1:10300` | HA Wyoming reference port |
| `--asr-base` | `VOKRA_ASR_BASE` | `models.asr_base` | (required) | Whisper base GGUF |
| `--asr-large-v3` | `VOKRA_ASR_LARGE_V3` | `models.asr_large_v3` | (unset → unavailable) | Whisper large-v3 GGUF (M2-06) |
| `--tts-piper` | `VOKRA_TTS_PIPER` | `models.tts_piper` | (required for TTS) | piper-plus native voice GGUF |
| `--tts-kokoro` | `VOKRA_TTS_KOKORO` | `models.tts_kokoro` | (unset → unavailable) | Kokoro-82M GGUF (M2-07 skeleton) |
| `--backend` | `VOKRA_BACKEND` | `runtime.backend` | `cpu` | `cpu` \| `metal` \| `cuda` |
| `--config` | `VOKRA_CONFIG` | — | (none) | Path to TOML config file |
| `--max-body-bytes` | `VOKRA_MAX_BODY` | `http.max_body_bytes` | `26214400` (25 MiB) | OpenAI parity |
| `--request-timeout-secs` | `VOKRA_REQ_TIMEOUT` | `http.request_timeout_secs` | `60` | Per-request deadline |
| `--max-connections` | `VOKRA_MAX_CONN` | `http.max_connections` | `100` | DoS ceiling |

Unknown model names, missing GGUFs, and unimplemented backend/op
combinations return an explicit error — **no silent CPU fallback**
(FR-EX-08).

### Bind & security posture

- **Default bind is loopback** on both HTTP (8080) and Wyoming
  (10300). Publishing to the LAN requires explicit `0.0.0.0` flags.
- **TLS and authentication are delegated to a reverse proxy**
  (nginx / Caddy / traefik). vokra-server itself does not terminate
  TLS in v0.5.
- **Forward-compatible `Authorization: Bearer <key>` parsing** is
  wired but disabled by default; enable via the reverse proxy or a
  future `--api-key` flag.
- **CORS is restrictive** by default (no `Access-Control-Allow-Origin`
  header emitted). Opt-in per deployment.
- **Panic isolation**: every HTTP handler and Wyoming session runs
  under a panic guard; a panic maps to a 500 response and closes the
  offending Wyoming connection cleanly, never the whole runtime
  (NFR-RL-07).

Example nginx reverse proxy config (TLS termination + hostname
enforcement, keeps vokra-server on loopback):

```
server {
    listen 443 ssl http2;
    server_name vokra.example.com;
    ssl_certificate     /etc/letsencrypt/live/vokra.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/vokra.example.com/privkey.pem;

    client_max_body_size 25m;
    proxy_read_timeout   60s;

    location / {
        proxy_pass         http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Authorization     $http_authorization;
    }
}
```

For Wyoming (JSONL over raw TCP, not HTTP), publish the listener on
the trusted VLAN only (`--wyoming-bind 10.0.0.5:10300`) or SSH-tunnel
from the Home Assistant host. A reverse proxy is not required.

### musl static distribution

The v0.5 reference distribution target is `x86_64-unknown-linux-musl`,
producing a single fully-statically-linked executable with **zero
runtime dynamic dependencies**. Verified in CI (T19):

```
rustup target add x86_64-unknown-linux-musl
cd integrations/vokra-server
cargo build --release --target x86_64-unknown-linux-musl

file target/x86_64-unknown-linux-musl/release/vokra-server
# → ELF 64-bit LSB executable, ... statically linked, ...

ldd  target/x86_64-unknown-linux-musl/release/vokra-server
# →       not a dynamic executable
```

macOS ARM64 and Windows x86_64 ship as regular dynamically-linked
binaries via `cargo build --release`. Docker is never required for any
target (FR-SV-01).

The server binary is intentionally larger than the core distribution
(NFR-DS-01 < 5 MB target applies to `vokra-core`, not to the server
executable which links the HTTP + async + serde stack). No ONNX,
protobuf, or gRPC dependency is ever linked in — enforced by the root
`Cargo.lock` scan and reasserted in T19.

## Compatibility matrix (v0.5)

Legend: **OK** = spec-compliant response; **stub** = schema-conformant
placeholder response with explicit non-implementation notice
(FR-EX-08 / plan §D9); **gated** = requires the corresponding GGUF at
launch, otherwise the model is `unavailable` and requests receive an
explicit error, never a silent substitution.

| API | Endpoint | whisper-base | whisper-large-v3 | whisper-turbo | piper-plus | Kokoro-82M |
|---|---|---|---|---|---|---|
| OpenAI | `POST /v1/audio/transcriptions` | OK | gated (M2-06) | out of scope (v1.0+) | n/a | n/a |
| OpenAI | `POST /v1/audio/speech` | n/a | n/a | n/a | out of scope (v1.0+) | out of scope (v1.0+) |
| vLLM | `POST /v1/completions` | stub (501) | stub (501) | stub (501) | n/a | n/a |
| vLLM | `POST /v1/chat/completions` | stub (501) | stub (501) | stub (501) | n/a | n/a |
| piper-plus HTTP | `POST /api/tts` | n/a | n/a | n/a | OK | gated + M2-07 skeleton → 501 |
| Wyoming | `transcribe` / audio events | OK | gated (M2-06) | out of scope | n/a | n/a |
| Wyoming | `synthesize` / audio events | n/a | n/a | n/a | OK | gated + M2-07 skeleton → 501 |

Notes:

- `whisper-turbo` (FR-MD-04 follow-on) is out of scope for v0.5 in
  every API row; requests naming it receive 501 + `type:
  "not_implemented"`, never a base-model substitution.
- vLLM `/v1/completions` and `/v1/chat/completions` are **contract-only
  stubs** in v0.5 — schema-conformant JSON is returned but no LLM is
  loaded (plan §D9). Real completion generation lands with the
  CosyVoice2 / Voxtral (v0.9, formerly v1.0) and Moshi / Helium
  (v1.0-rc+, formerly v1.5+) work.
- Kokoro entries reflect the M2-07 skeleton whose `TtsEngine::synthesize`
  currently returns `NotImplemented`. The registry accepts the GGUF at
  launch; requests fail with 501 until M2-07 lands the vocoder.
- CC-BY-NC weights (F5-TTS / Fish-Speech / EnCodec) are refused at
  registry load without a research flag (plan §D11 / M2-13); they
  never appear in this matrix.
- Backend selection (CPU / Metal / CUDA) is fixed at server startup
  and applies uniformly across all rows above.

## Connection examples

### faster-whisper drop-in client (Python)

`faster-whisper`'s `OpenAI`-style client and any tool that speaks the
`POST /v1/audio/transcriptions` multipart API work unchanged — point
`base_url` at the vokra-server HTTP listener:

```python
# pip install openai  (or use any OpenAI-compatible client library)
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-required-loopback",  # any non-empty string; auth is
                                       # handled by the reverse proxy
)

with open("hello.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",  # → mapped to Whisper base (OpenAI naming)
        file=f,
        # model="whisper-large-v3",  # → Vokra large-v3 (M2-06), if
        #                              the GGUF was registered at boot
        response_format="json",  # verbose_json is v1.0+ scope
    )

print(result.text)
```

Model naming: `whisper-1` maps to Whisper base for OpenAI parity.
`whisper-large-v3` selects the M2-06 large-v3 weights when their GGUF
was passed at boot; otherwise the request returns 404 with
`type: "model_not_found"` — never a base-model substitution
(FR-EX-08).

### Home Assistant — Wyoming Server registration

Vokra exposes a Wyoming Protocol JSONL-over-TCP listener on
`127.0.0.1:10300` by default (the same port used by the Rhasspy /
Home Assistant Wyoming reference services). The listener is available
as soon as `vokra-server` starts; no separate Docker container is
required (FR-SV-01).

Steps to register the server with Home Assistant OS / Container:

1. **Boot `vokra-server` on a host reachable from HA.** Default bind
   is loopback; publish to the LAN by passing `--wyoming-bind
   0.0.0.0:10300` explicitly. The bind change is opt-in on purpose —
   the default is loopback (plan §D10).

   ```
   vokra-server --wyoming-bind 0.0.0.0:10300 \
                --asr-base   /path/to/whisper-base.gguf \
                --tts-piper  /path/to/piper-voice.gguf
   ```

2. **In Home Assistant, open Settings → Devices & Services → Add
   Integration.**

3. **Select "Wyoming Protocol".**

4. **Enter the host + port** for the machine running `vokra-server`
   (e.g. `192.168.1.42` + `10300`). HA discovers the available
   services (STT, TTS, wake-word) by sending a `describe` event; the
   Vokra listener responds with an `info` event describing the ASR
   and TTS models registered at boot (Whisper base / large-v3 for
   STT, piper-plus native / Kokoro for TTS, subject to GGUF
   availability — plan §D6).

5. **Assign the discovered services to a HA Voice Assistant pipeline.**

Notes on the current M2-09 delivery:

- **`stream=true` and word-level timestamps** are not yet exposed
  over Wyoming (v0.5 scope, plan §4 out-of-scope).
- **Watermark / C2PA are forward-compat only** (依頼者 drop 2026-07-04,
  plan §D11 / M2-13). TTS payloads over Wyoming carry no
  AudioSeal / C2PA marker in v0.5.
- **Backend selection** (CPU / Metal / CUDA) is fixed at server
  startup; per-request backend switching is out of scope for M2-09.
- **Silent CPU fallback is disabled** (FR-EX-08). If a requested
  model or backend combination has an unimplemented op, the Wyoming
  session emits an error event and closes cleanly.

Framing invariant: Wyoming events are JSONL over TCP. Each event's
JSON header terminates at the first `\n`; the binary payload region
(announced by `payload_length` / `data_length`) is read with
`read_exact(N)`, NEVER with a line-buffered reader. This is asserted
by the unit test
`framing_invariant_read_exact_over_payload_region` in
`tests/wyoming_compat.rs` and runs on every push, whether or not the
T14+ event loop is wired.

Real Home Assistant hardware verification (VoicePE satellite, HA
Assist pipeline) is deferred to M2-15 (依頼者 quarterly Go/No-go,
Kill switch J).
