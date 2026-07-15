# Wyoming Protocol — Server Design (M2-09 T14)

Scope: design record for Vokra's Wyoming Protocol server implementation
(ASR + TTS surface). Consumed by T15 (Wyoming ASR) and T16 (Wyoming TTS).
Implementation details (event handlers, streaming bridge) live in `src/api/wyoming.rs`.

## 1. Spec provenance

- Upstream: `rhasspy/wyoming` (Apache-2.0), the reference Python implementation
  and prose specification of the protocol.
- Confirmed against: `rhasspy/wyoming` README / `docs/` describing the wire
  format, event catalogue (info / describe / audio-start / audio-chunk /
  audio-stop / transcribe / synthesize), and PCM audio conventions.
- No invention: any field or event not documented upstream is deferred to a
  follow-up ticket rather than added speculatively (CLAUDE.md hallucination-ban).

## 2. Wire format (JSONL over TCP + binary payload)

Wyoming is **newline-delimited JSON events over a raw TCP socket** (not HTTP,
not WebSocket). Each event is a single line of UTF-8 JSON followed by `\n`.
An event MAY carry two out-of-band binary attachments described in the header:

```
<json-header>\n              # exactly one line, terminated by 0x0A
<data       bytes...>        # exactly `data_length` bytes if present, else 0
<payload    bytes...>        # exactly `payload_length` bytes if present, else 0
```

Header JSON shape (subset used by Vokra):

| Field            | Type           | Notes                                                              |
|------------------|----------------|--------------------------------------------------------------------|
| `type`           | string         | Event kind, e.g. `audio-chunk`, `transcribe`, `synthesize`, `info` |
| `data`           | object \| null | Small structured payload (rate, width, channels, text, ...)         |
| `data_length`    | integer \| 0   | Byte count of the optional structured `data` blob (rarely used)     |
| `payload_length` | integer \| 0   | Byte count of the raw binary payload (PCM samples for audio-chunk)  |
| `version`        | string \| null | Optional protocol version tag                                       |

Absent / zero `data_length` and `payload_length` mean the next event begins
immediately at the next byte (no attachment). This document records
`payload_length` as the load-bearing field for audio framing; a client that
ignores it and line-buffers the socket will corrupt any payload containing
`0x0A`.

### 2.1 Event catalogue used by Vokra (v0.5)

- **ASR path (T15):** `describe` / `info` (capability advertisement),
  `audio-start` (rate/width/channels), `audio-chunk` (PCM frames in payload),
  `audio-stop`, `transcribe` (language selection + finalize), `transcript`
  (server -> client final text).
- **TTS path (T16):** `synthesize` (text + voice), `audio-start`,
  `audio-chunk`, `audio-stop`. Vokra emits mono int16 PCM at the voice's
  native rate; conversion is the client's responsibility unless negotiated
  in `info`.

## 3. Parser pattern (mandatory — see risk R5)

The parser MUST NOT use a line-buffered reader for binary payloads. The
canonical read loop, in Rust / tokio terms:

```rust
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

let mut reader = BufReader::new(socket);
let mut header_line = String::new();

loop {
    header_line.clear();
    // 1. Read the JSONL header line (terminated by \n).
    let n = reader.read_until(b'\n', unsafe { header_line.as_mut_vec() }).await?;
    if n == 0 { break; } // clean EOF

    let header: WyomingHeader = serde_json::from_str(header_line.trim_end())?;

    // 2. If data_length > 0, read EXACTLY that many bytes with read_exact.
    let mut data_buf = vec![0u8; header.data_length];
    if header.data_length > 0 {
        reader.read_exact(&mut data_buf).await?;
    }

    // 3. If payload_length > 0, read EXACTLY that many bytes with read_exact.
    //    NEVER call read_until / read_line here — payload bytes may include 0x0A.
    let mut payload_buf = vec![0u8; header.payload_length];
    if header.payload_length > 0 {
        reader.read_exact(&mut payload_buf).await?;
    }

    dispatch(header, data_buf, payload_buf).await?;
}
```

Rules:

1. Header line: `read_until(b'\n', ...)`. Enforce a max header length
   (e.g. 64 KiB) to bound memory before `serde_json::from_str`.
2. `data` bytes: `read_exact(&mut buf[..data_length])`.
3. `payload` bytes: `read_exact(&mut buf[..payload_length])`. The payload is
   opaque binary — treating it as text WILL corrupt PCM whenever a sample
   byte equals `0x0A`.
4. Writer side is the mirror: serialize header JSON, write `\n`, then write
   the exact `data_length` and `payload_length` bytes with `write_all`. Never
   interleave writes from multiple tasks on the same socket without a mutex.

## 4. Streaming bridge to Vokra runtime

- Wyoming TCP handler runs as one `tokio::spawn` task per connection.
- ASR: incoming `audio-chunk` payloads (PCM int16 LE) are converted to
  `Vec<f32>` and pushed into either `WhisperAsr::transcribe` (batched, on
  `audio-stop`) or `Session::open_step_stream` for streaming.
  `Stream::push` is sync + wait-free, so the tokio task moves the producer
  into a `spawn_blocking` region or a dedicated `std::thread` to avoid
  blocking the async runtime.
- TTS: `synthesize` -> `SynthesizeService::synthesize` -> emit `audio-start`
  (with rate/width/channels), then chunk the output PCM into `audio-chunk`
  events (target ~40 ms per chunk to feed HA satellites smoothly), then
  `audio-stop`.
- Unknown / unsupported events -> respond with an `error` event or close the
  connection with a structured log line. Never silently drop.

## 5. Safety and error handling

- Panics inside a connection task MUST NOT tear down the runtime. Each task
  is wrapped so a panic closes only that connection (see T05, NFR-RL-07).
- Unsupported backend ops surface as explicit errors (FR-EX-08); no silent
  CPU fallback is inserted at the Wyoming layer.
- Default bind is loopback (`127.0.0.1:10300`); non-loopback bind is opt-in
  via CLI (T20).
- Max frame sizes: header <= 64 KiB, `payload_length` <= 16 MiB per event
  (bounds DoS via oversized `payload_length` fields before allocating).

## 6. Test hooks (T17)

- `tests/wyoming_compat.rs` will implement a mock client that speaks JSONL
  + `payload_length` framing exactly per section 3 and asserts byte-level
  round-trip of PCM chunks (input == output within tolerance) — this is the
  regression guard for risk R5.
- HA on-device verification is deferred to M2-15 (owner: requester).

## 7. Barge-in (M4-19, FR-ST-03)

The accept loop's full ASR+TTS handler (`run_wyoming_connection`) supports
barge-in on the TTS emit path. Semantics and provenance:

- **Trigger (契機)**: a new `audio-start` event received on the same
  connection *while a TTS `synthesize` is emitting* is treated as the barge-in
  trigger (the satellite's wake word / next utterance beginning). `audio-start`
  is a documented upstream `rhasspy/wyoming` event (§2.1); we do **not** invent
  a bespoke control event (CLAUDE.md 発明禁止). **Owner-confirmable follow-up**:
  if upstream defines a *dedicated* barge-in / interrupt control event, adopt
  it here and keep the `audio-start` heuristic as a fallback — recorded so the
  choice is a deliberate future edit, not silent drift.
- **Effect**: the `audio-chunk` emit loop polls a connection-scoped barge-in
  flag (`wyoming::BargeIn`, an `Arc<AtomicBool>` mirroring the M3-14
  `vokra_core::stream::InterruptHandle` Release/Acquire semantics) at each
  chunk boundary. When raised it stops emitting the remaining chunks and sends
  `audio-stop` immediately — from an HA satellite's view the audio output cuts
  (the barge-in体感). The still-buffered event (the trigger `audio-start`) is
  then processed as the start of a fresh ASR utterance.
- **Batch vs streaming**: v0.5 TTS is *batch synth* (`SynthesizeService`
  returns the whole `SynthesizedAudio` up front), so "barge-in" here means
  "stop sending the remaining `audio-chunk`s"; the un-emitted PCM tail is
  discarded. There is no SPSC ring to `EventPoller::drain_all` on this path —
  that becomes load-bearing only for a future *streaming* synth form, at which
  point true mid-synthesis stop (halting the synth kernel) lands (follow-up).
  For a streaming ASR path (`Session::open_step_stream`, §4) the real
  `Stream::interrupt()` / `InterruptHandle` apply directly.
- **Concurrency / framing**: a per-connection reader-pump task frames every
  inbound event (header + `read_exact` payload, §3 R5) onto a bounded channel;
  the emit loop watches that channel for the trigger. Because the pump owns the
  reader exclusively, watching for a mid-emit trigger never cancels a partial
  read (no cancel-safety hazard).
- **FR-EX-08**: an *unknown* control event mid-emit is surfaced as an `error`
  event, never silently dropped.
