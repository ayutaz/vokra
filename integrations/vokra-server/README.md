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
    --whisper-base /path/to/whisper-base.gguf \
    --piper-plus /path/to/piper-voice.gguf

# Add --piper-g2p to synthesize from plain text (real 8-language G2P,
# derived from the voice GGUF metadata). Without it, plain-text TTS
# requests return an explicit 400 and only raw phoneme-id payloads work.
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
| `--http-bind` | `VOKRA_HTTP_BIND` | (CLI/env only) | `127.0.0.1:8080` | Public exposure requires reverse proxy |
| `--wyoming-bind` | `VOKRA_WYOMING_BIND` | (CLI/env only) | `127.0.0.1:10300` | HA Wyoming reference port |
| `--whisper-base` | `VOKRA_WHISPER_BASE` | `whisper_base` | (unset → ASR unavailable) | Whisper base GGUF |
| `--whisper-base-tokenizer` | `VOKRA_WHISPER_BASE_TOKENIZER` | `whisper_base_tokenizer` | (unset) | Optional external tokenizer side-car |
| `--whisper-large-v3` | `VOKRA_WHISPER_LARGE_V3` | `whisper_large_v3` | (unset → unavailable) | Whisper large-v3 GGUF (M2-06) |
| `--whisper-large-v3-tokenizer` | `VOKRA_WHISPER_LARGE_V3_TOKENIZER` | `whisper_large_v3_tokenizer` | (unset) | Optional external tokenizer side-car |
| `--piper-plus` | `VOKRA_PIPER_PLUS` | `piper_plus` | (unset → TTS unavailable) | piper-plus native voice GGUF |
| `--piper-g2p` | `VOKRA_PIPER_G2P` | `piper_g2p = true` | off | Inject the real 8-language G2P (plain-text TTS); off = explicit-error passthrough (raw phoneme ids only) |
| `--kokoro` | `VOKRA_KOKORO` | `kokoro` | (unset → unavailable) | Kokoro-82M GGUF |
| `--voxtral` | `VOKRA_VOXTRAL` | `voxtral` | (unset → unavailable) | Voxtral GGUF |
| `--silero-vad` | `VOKRA_SILERO_VAD` | `silero_vad` | (unset → unavailable) | Silero VAD GGUF |
| `--config` | `VOKRA_CONFIG` | — | (none) | Path to TOML config file (flat keys mirror the flag names with underscores) |

Request-body size is capped at 25 MiB (OpenAI parity) as a compiled-in
limit (`api/openai.rs::MAX_BODY_BYTES`); there is no CLI flag for it.

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

| API | Endpoint | whisper-base | whisper-small/medium/turbo | whisper-large-v3 | piper-plus | Kokoro-82M |
|---|---|---|---|---|---|---|
| OpenAI | `POST /v1/audio/transcriptions` | OK | gated (cc-39) | gated (M2-06) | n/a | n/a |
| OpenAI | `POST /v1/audio/speech` | n/a | n/a | n/a | OK (cc-38) | gated + M2-07 skeleton → 501 |
| OpenAI | `GET /v1/models` | OK | OK (advertised when gated-in) | OK | OK | OK |
| vLLM | `POST /v1/completions` | stub (501) | stub (501) | stub (501) | n/a | n/a |
| vLLM | `POST /v1/chat/completions` | stub (501) | stub (501) | stub (501) | n/a | n/a |
| piper-plus HTTP | `POST /api/tts` | n/a | n/a | n/a | OK | gated + M2-07 skeleton → 501 |
| Wyoming | `transcribe` / audio events | OK | gated (cc-39) | gated (M2-06) | n/a | n/a |
| Wyoming | `synthesize` / audio events | n/a | n/a | n/a | OK | gated + M2-07 skeleton → 501 |

Notes:

- `whisper-small` / `whisper-medium` / `whisper-turbo` are served when
  their GGUF is supplied at launch (`--whisper-small` etc., cc-39;
  `whisper-turbo` also answers to its upstream id
  `whisper-large-v3-turbo`). **An unconfigured size is a 404, never a
  substitution by another size** — verified against real weights in
  `tests/real_gguf_slots.rs`. This supersedes the earlier "out of scope
  (v1.0+)" row: model-side support landed with M4-14 and all four sizes
  transcribe byte-identically to onnxruntime
  (`docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md`).
- `POST /v1/audio/speech` (cc-38) returns `audio/wav`. Three deliberate
  deviations from OpenAI, each an explicit status rather than a
  plausible-looking response: compressed `response_format`
  (mp3/opus/aac/flac) and OpenAI's headerless 24 kHz `pcm` are **501**
  (Vokra links no audio encoder or resampler, and adding one would mean a
  third-party codec dependency); `speed` other than `1.0` is **501** (the
  native runtime does not wire per-request `length_scale`); and OpenAI's
  stock voice names (`alloy`, `nova`, …) are **404** rather than being
  folded onto the one loaded voice. Omitting `response_format` yields
  `wav`, not OpenAI's `mp3` default. `model: "tts-1"` is accepted as the
  stock alias for the default TTS engine (the same convention as
  `whisper-1` → base); `tts-1-hd` is **not** aliased — it names a quality
  tier this server does not have.
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
- Backend selection is fixed at server startup: `--backend <cpu|metal|
  cuda|vulkan>` sets the default and `--model-backend <SLOT>=<BACKEND>`
  overrides individual engines (cc-30). A backend can only be *selected*
  if it was *compiled in* — the GPU backends are opt-in Cargo features
  that forward to `vokra-models` (`cargo build --release --features
  metal`). Requesting an uncompiled backend is a **hard startup error**
  naming the feature to rebuild with, never a silent CPU fall back
  (FR-EX-08). Two engines are exceptions worth knowing: Silero VAD has no
  backend selector at all (CPU-only by construction — an explicit
  `--model-backend silero-vad=…` is rejected, and a non-CPU global default
  is announced as not applying to it), and Voxtral only began honouring
  the setting with cc-30 (it previously stayed on CPU regardless).

## Stability & versioning tier (experimental)

**The `vokra-server` network APIs — the OpenAI-/vLLM-/piper-plus-compatible
HTTP endpoints AND the Wyoming Protocol JSONL/TCP listener — are a
protocol-tracking / experimental governance tier and are NOT covered by
Vokra's v1.0 semver stability guarantee.**

- These surfaces are **not part of the Vokra C ABI freeze** (the frozen,
  semver-stable interface is `include/vokra.h` at v1.0 GA). See
  `docs/handoff/m4-12.md` §(e)-1 ("HTTP/gRPC/WebSocket API is NOT part of the
  C ABI freeze"; "the future Wyoming Protocol Server lives at a separate
  governance tier").
- The **Wyoming endpoint tracks upstream `rhasspy/wyoming` semantics** and
  follows *that project's* versioning — a breaking change in the
  HA-community-driven Wyoming Protocol is **not** a Vokra semver violation.
  Treat the Wyoming surface as experimental and pin your integration to a
  tested Vokra build rather than assuming cross-version wire stability.
- The HTTP endpoints aim for drop-in compatibility with the upstream shapes
  they mirror (OpenAI audio, vLLM, piper-plus, faster-whisper) and likewise
  track those upstreams, not Vokra semver.

(The formal `## Non-C-ABI surface areas` section of `include/vokra.h`'s
STABILITY block is added at **M5-13** — the C ABI freeze — per
`docs/handoff/m4-12.md` §(e)-1; this README documents the exclusion ahead of
that.)

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
                --whisper-base /path/to/whisper-base.gguf \
                --piper-plus   /path/to/piper-voice.gguf --piper-g2p
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
