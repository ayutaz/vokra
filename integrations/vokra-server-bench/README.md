# vokra-server-bench

**Excluded-workspace HTTP-boundary TTS latency bench client** — measures real HTTP `POST` round-trip TTFA / total-latency against a running `vokra-server` process.

- **Status**: Wave 14 land, PR #4 (commit `c2abfcb`, 2026-07-11)
- **Scope**: `docs/m3-15-server-latency-handover.md` § 4 "Option C" gap-fill
- **Third-party deps** (isolated Cargo.lock): `ureq` + `serde_json`. NOT in the root `vokra-*` graph — this crate is deliberately excluded to preserve zero-dep NFR-DS-02 on the root workspace.
- **Sibling crate**: [`integrations/vokra-cli-bench-server/`](../vokra-cli-bench-server/README.md) provides a **pure-`std`** variant (no `ureq` / `serde_json`) for operators who want a bench binary whose `Cargo.lock` contains only `vokra-*` names.

## Purpose

The in-process bench at `integrations/vokra-server/benches/tts_latency.rs` measures the schema-layer floor (deterministic FakeSynth, no network) — it is not `NFR-PF-05` (75 ms TTFA) evidence because it does not cross the HTTP wire. This crate measures the wire boundary against a live `vokra-server` on localhost or a LAN peer, which is what NFR-PF-05 actually specifies.

## Run

Start `vokra-server` on port 8080, then:

```
cd integrations/vokra-server-bench
cargo run --release -- --host 127.0.0.1 --port 8080 --iterations 100 --format json
```

Full argv + exit code contract + JSON schema: `docs/m3-15-server-latency-handover.md` § 4.

## Exit codes

- `0` — success (metrics emitted)
- `2` — invalid argv
- `3` — server unreachable / connection refused
- `4` — HTTP error response

## Tests

```
cargo test    # in this directory
```

Covers argv parsing, exit code mapping, JSON emit format, KV emit format.
