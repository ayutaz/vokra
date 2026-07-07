# Wyoming ↔ Home Assistant — local Docker smoke (M2-15 pre-check)

Wire-level reachability smoke: confirms that a Home Assistant container on
this host can reach `vokra-server`'s Wyoming TCP port. Full protocol
handshake (`describe` → `info`) is **NOT** covered — it requires the T14/T15/T16
event loop to be wired into the accept loop, which is still pending
(see "Findings" below).

Kill switch J judgment (HA Voice adopting Vokra) is owner-side; this smoke
does not attempt to make that call.

## What was tested

1. Build `vokra-server` from the isolated workspace under `integrations/vokra-server`.
2. Launch the release binary bound to `--wyoming-bind 0.0.0.0:10300` on the host.
3. Pull `homeassistant/home-assistant:stable` and start a container with
   `-p 8123:8123 -v /tmp/vokra-ha-config:/config` (Docker Desktop for macOS
   does not honour `--network host`, so the published-port form is used).
4. Verify HA's own HTTP surface (`GET /manifest.json`) responds — proves the
   HA process is up inside the container.
5. From **inside** the HA container, TCP-connect to
   `host.docker.internal:10300` and to the M1 iMac's LAN IP
   (`192.168.11.5:10300`). Try to send a Wyoming `describe` JSONL header
   and read a JSONL response.
6. Cleanup: stop and remove the HA container; kill the `vokra-server` process.

## Environment (recorded)

| Field                    | Value                                                                                     |
|--------------------------|-------------------------------------------------------------------------------------------|
| Date (UTC)               | 2026-07-07T03:06:24Z                                                                       |
| Host                     | Apple M1 iMac (Darwin 25.3.0, `aarch64`)                                                   |
| Docker                   | `Docker version 24.0.6, build ed223bc` (Docker Desktop, Server Version 24.0.6)             |
| Home Assistant image     | `homeassistant/home-assistant:stable` (digest `sha256:f73512ba4fe06bb4d57636fe3578d0820cdec46f81e8f837ab59e451662ff3cb`, local ID `291e4b204813`, 2.33 GB) |
| Vokra commit at test     | `c04d3442ea2d782a96c1a610f6299a5825422ce3` (working tree with a local, uncommitted `run_with_config` fix — see below) |
| `vokra-server` binary    | `integrations/vokra-server/target/release/vokra-server` (1.83 MB)                          |
| Ports                    | HA `127.0.0.1:8123 → container:8123`; vokra-server `0.0.0.0:10300` (host)                  |
| Docker→host bridge       | `host.docker.internal` and `gateway.docker.internal` both resolve to `192.168.65.254`      |

## Raw observed output (not synthesized)

### vokra-server startup log (`/tmp/vokra-server.log`)

```
vokra-server: HTTP listening on 127.0.0.1:8080, Wyoming listening on 0.0.0.0:10300
```

The process stayed alive for the duration of the smoke (was previously
exiting immediately — see "Findings", item 1).

### Host probe (M1 iMac loopback → `127.0.0.1:10300`)

```
TCP_CONNECT_OK dt=0.2ms
TX/RX error (expected for T03 placeholder): ConnectionResetError(54, 'Connection reset by peer')
```

### HA container probe (inside container → `host.docker.internal:10300`)

```
RESOLVED host.docker.internal -> 192.168.65.254
TCP OK host.docker.internal:10300 dt=2.6ms
  EOF after send (server closed) — bytes=0
RESOLVED gateway.docker.internal -> 192.168.65.254
TCP OK gateway.docker.internal:10300 dt=2.3ms
  EOF after send (server closed) — bytes=0
```

### HA container probe (inside container → LAN IP `192.168.11.5:10300`)

```
192.168.11.5 (192.168.11.5:10300) open
```

### HA container probe (inside container → send `describe` via `nc`)

```
(no bytes returned; nc exited after -w 2 timeout)
```

## Results — verdicts

| Check                                                                       | Result                    |
|-----------------------------------------------------------------------------|---------------------------|
| Docker available                                                            | ✅ yes (24.0.6)            |
| `vokra-server` release build                                                | ✅ succeeded               |
| `vokra-server` process stays alive after "listening" log                    | ✅ after local fix (see 1) |
| `vokra-server` listens on `0.0.0.0:10300`                                   | ✅ confirmed (`lsof`)      |
| HA container starts and serves `/manifest.json`                             | ✅ ~1 s to ready           |
| HA container TCP-connects to host port 10300 via `host.docker.internal`     | ✅ 2.6 ms                  |
| HA container TCP-connects to host port 10300 via LAN IP `192.168.11.5`      | ✅ "open"                  |
| HA container receives a Wyoming `info` reply to `describe`                  | ❌ EOF, 0 bytes            |
| Cleanup (stop + rm container, kill server)                                  | ✅ done                    |

## Findings (bugs / gaps observed during this smoke)

1. **`vokra-server` main entrypoint exits immediately after logging.**
   `run_with_config` (`src/server.rs`) calls `spawn_server(...).await?`,
   prints the "listening" line, and then returns `Ok(())` from the
   `block_on` future. The listener tasks live on the runtime, not on
   that future, so `block_on` unwinds and the runtime is dropped —
   killing both the HTTP and the Wyoming listeners before the first
   real request can be served.

   Reproduced twice on this machine: PID observed running for < 1 s,
   `nc -zv 127.0.0.1 10300` reports "Connection refused" a second later.

   Workaround applied locally so this smoke could proceed:
   `signal.wait().await` before returning from the `block_on` future.
   The change is a single line + clone of the existing `ShutdownSignal`
   handle; it matches the docstring intent in `main.rs`
   ("waits for `ctrl_c` / `SIGTERM` before draining") and the pre-existing
   (but incorrect) comment inside `run_with_config`. Uncommitted at
   the time of this smoke — the observation is recorded here so the
   owner can decide whether to land the fix separately.

2. **Wyoming JSONL event loop is not wired into the accept loop.**
   `wyoming_accept_loop` in `src/server.rs` is still the T03 placeholder
   (`tokio::spawn(async move { drop(stream); })`). The real event
   dispatcher exists in `src/api/wyoming.rs` (T14/T15/T16 —
   `run_asr_connection`, `handle_synthesize`, `AsrSession`, `InfoBody`,
   the JSONL framing helpers) but is not called from the accept loop.
   Consequence: HA can complete a TCP handshake, but sending a
   `describe` header produces an immediate EOF (server closes the
   socket without responding). The existing test
   `wyoming_compat::wyoming_describe_round_trip` already documents this
   with an explicit skip: "server closed with no data — T14/T15/T16
   event loop not yet wired; contract test pending."

   This is the load-bearing gap between "reachable" and "usable" for
   Home Assistant. Wiring `run_asr_connection` + `handle_synthesize`
   into the accept loop (with panic-isolated per-connection tasks per
   NFR-RL-07) is the follow-up that turns this smoke green end-to-end.

## Known limitations of this smoke

- **Wire-level only.** This test asserts that a container reaches the host
  port; it does not exercise the Wyoming event vocabulary (`describe` /
  `info` / `audio-start` / `audio-chunk` / `audio-stop` / `transcribe` /
  `synthesize` / `transcript`). The engine handlers are unit-tested inside
  `src/api/wyoming.rs` but are not reachable over the socket yet (see
  finding 2).
- **HA integration wizard is not driven.** Adding "Wyoming Protocol" as
  an integration inside the HA UI and selecting Vokra as the ASR/TTS
  backend is a manual, cookie-authenticated flow. Owner runs that step
  in M2-15.
- **Docker Desktop `--network host` is a no-op on macOS.** This test uses
  the published-port form (`-p 8123:8123`) and relies on `host.docker.internal`
  for reverse container→host reachability. On Linux hosts, `--network host`
  would be the simpler form and this smoke would need to be re-run.
- **HA container is `aarch64`.** The image is multi-arch; verified this
  smoke on Apple Silicon only. `linux/amd64` HA image is expected to
  behave the same for TCP reachability but is not covered here.
- **No timing / RTF measurements.** vokra-server has no models loaded on
  this smoke path (no `--config` was passed), so no ASR / TTS work was
  attempted. RTF gates for the Wyoming path land in M3-01 regression.

## Owner follow-ups

- Decide whether the `run_with_config` fix (finding 1) lands as a small
  standalone `fix(server)` commit or is folded into the T14 wiring PR.
- Land the T14 accept-loop wiring (finding 2). Suggested shape:
  in `wyoming_accept_loop`, spawn each connection into a task that
  routes `describe` / ASR events into `run_asr_connection` and
  `synthesize` events into `handle_synthesize` behind a `select!` that
  also observes `signal.wait()` for graceful shutdown.
- Once wired, re-run this smoke and record the actual `info` payload
  (voice catalogue / ASR model names) that HA will consume when it
  probes the Wyoming service.
- M2-15 owner call: after the above, drive the HA "Wyoming Protocol"
  integration wizard against `host.docker.internal:10300` on this
  machine and record the observed voice / ASR entities.
