# vokra-cli-bench-server

**Zero-dependency HTTP-boundary TTS latency bench client** — the sibling of [`integrations/vokra-server-bench/`](../vokra-server-bench/README.md) that removes even the excluded-workspace's transitive third-party surface.

- **Status**: Wave 14 land, PR #4 (commit `c2abfcb`, 2026-07-11)
- **Scope**: `docs/m3-15-server-latency-handover.md` § 4 "Option C" — pure-`std` variant
- **Third-party deps**: **none**. HTTP/1.1 hand-written on `std::net::TcpStream`, JSON request/response escaped and parsed by hand. `Cargo.lock` contains only `vokra-*` names — belt-and-suspenders for NFR-DS-02 posture.

## Why a second bench binary?

The sibling `integrations/vokra-server-bench/` already covers Option C using `ureq` + `serde_json`. It works and is documented. This crate exists specifically for operators who want a bench binary whose Cargo.lock contains **only** `vokra-*` names — an even stricter posture than the sibling crate's excluded-workspace isolation.

**Both binaries produce byte-compatible JSON output for the same schema** (`docs/m3-15-server-latency-handover.md` § 4). Choose whichever fits your dependency policy.

## Run

Start `vokra-server` on port 8080, then:

```
cd integrations/vokra-cli-bench-server
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

Covers argv parsing, exit code mapping, JSON emit format, KV emit format, hand-written HTTP/1.1 framing, hand-written JSON escape/parse.
