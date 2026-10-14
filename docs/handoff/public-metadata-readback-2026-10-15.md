# Public metadata inventory readback — 2026-10-15 JST

The manager runs the existing read-only coverage audit against accepted main
`6f135822075f9b98de08dfb7329a88d786d36531`. The actual process exits zero;
the complete result is independently parsed at 19:59:12 UTC on 2026-10-14.
This is a fresh metadata retrieval, not a real-weight or Apple measurement.

The only remote inputs are the public Hugging Face author API and each
GGUF-bearing repository's README at its returned immutable revision. No
weight, config or tokenizer is downloaded; no credential is used. The only
local runtime-source input is `crates/vokra-cli/src/engine.rs`. The protected
CosyVoice2 LLM manifest is not read, hashed or modified.

## Actual result

| Metadata/code classification | Count |
|---|---:|
| Public repositories | 194 |
| GGUF-bearing repositories | 193 |
| GGUF filenames | 198 |
| CPU/Metal code-and-public-metadata full | 136 |
| CPU partial | 43 |
| No complete runtime binder | 14 |
| No artifact | 1 |
| Unresolved public rows | 58 |

All 194 IDs are unique, revision values are 40-character SHA-1 values and
the complete TSV has 195 lines / 37,881 bytes. Its SHA-256 is
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`,
exactly matching the previous recorded inventory. The returned revisions,
filename counts, architecture labels and source classifications have not
changed. The retained local receipt is not a model artifact.

The audit source SHA-256 is
`690d603f7182fc5c0645e784f8857acee9604b20050892b2bf0eadeb41f8a905`;
the engine source SHA-256 is
`cc22a9074531fa738c68101f4697b1b4393252b913b1bb2bc61092da93cf0b1b`.
The existing audit fetches no GGUF payload or tensor header. Its `full`
classification is a declared code/public-reachability classification, not
proof of fresh conversion, independent reference, CPU parity or Apple
CPU/Metal/no-fallback execution for all 136 rows.

The [58-row prerequisite readback](current-58-static-prerequisite-readback-2026-10-15.md)
and [full execution plan](next-steps-2026-10-14.md) therefore remain open.
No owner/legal decision, public artifact, numerical bound, 63-row frozen
owner scope or final transfer packet changes. PR #248's current source CI
and private XCodec2 Rust verification are separate acceptance gates.
