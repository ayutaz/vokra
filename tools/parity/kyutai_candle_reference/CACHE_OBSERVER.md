# Moshi cache/state observer seam candidate (source-only)

This candidate adds an observational prerequisite for a future independent
full-PCM reference. It is not a numerical mirror, a parity result, a runtime
compatibility result, or a publication approval. The patcher never compiles or
imports Moshi/Candle source and never acquires model, tokenizer, or weight
artifacts.

All four inputs are authenticated as complete bytes from Kyutai Moshi commit
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` before any source edit:

| input | official raw URL | Git blob SHA-1 | bytes |
| --- | --- | --- | ---: |
| `rust/moshi-core/src/asr.rs` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/asr.rs` | `0f3ad70082f152e3250aa81ab4e0b48ffef550db` | 9125 |
| `rust/moshi-core/src/transformer.rs` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/transformer.rs` | `4382d71044bd723ebbedade5043ff883683a70b0` | 40480 |
| `rust/moshi-core/src/lm.rs` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/lm.rs` | `6775ec8c5453c40a4f777303bd583db76e089dd8` | 37671 |
| `rust/LICENSE` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11357 |

`pinned-asr.rs`, `pinned-transformer.rs`, `pinned-lm.rs`, and
`pinned-LICENSE` are exact authenticated fixtures. The root license is Apache
License 2.0 source evidence and preserved attribution; it is not an owner
signoff, a new license row, or permission to publish model weights.

The source-only accessor contract is bound to the published Candle 0.9.1 API
at commit `cd96fa80da255e34f7b16b4ff98b6a31d557201b`, path `candle-nn`, via
[`kv_cache.rs`](https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/candle-nn/src/kv_cache.rs).
The published archive checksum is
`c1980d53280c8f9e2c6cbe1785855d7ff8010208b46e21252b978badf13ad69d`; this
does not claim a crate-member byte comparison or compilation. Used accessors
are `RotatingKvCache::k/v/k_cache/v_cache`, `RotatingCache::offset/dim/max_seq_len/current_seq_len`,
and the Moshi `KvCache::current_seq_len/positions` methods. These are distinct
accessor families. License approval is explicitly `null`; no license signoff
is inferred.

## Candidate contract

The existing v1 observer seam is composed byte-for-byte and remains intact.
The new optional `observer_v2` defaults to `None`; no cache observation or
other new fallible work occurs in that default path. At the existing per-LM
step callback point—immediately after selected `text_tokens.to_vec1` and
before any item state update—the v2 callback receives:

- an owned metadata snapshot containing the actual global
  `model_step_idx`, each private item’s actual `step_idx` and `text_token`,
  and the normal LM cache observation;
- the original read-only callback tensors and values (`StreamMask`,
  `text_logits`, `transformer_out`, and selected tokens);
- for every normal-transformer layer, the upstream
  `current_seq_len`, the exact upstream `positions(0)` result under the
  explicit name `upstream_positions_for_zero_query`, and the actual
  `last_reset_pos.clone()`;
- the actual rotating-cache K/V handles when present, plus their upstream
  offset, dimension, capacity, and current sequence counters. Empty caches
  remain `None`; no zero tensor or inferred retained count is synthesized.

These handles alias upstream storage and are not deep immutable snapshots. A
collector must serialize synchronously and must not mutate tensor storage;
later forwards may change aliased storage. In fixed Candle 0.9.1,
`current_data()` is `None` when empty, narrows along `cache.dim()` from offset
0 to `current_seq_len` before capacity, and returns a clone of the full
physical rotating-buffer order at/after capacity; `k()`/`v()` delegate to it.
`all_data()` is the full allocated physical buffer and may include unused
capacity. Neither view is chronologically reordered here, and
`positions(0)` is only the upstream mapping result—no reordering or tensor copy
is implied.

The `LmModel::batched` route is explicit unsupported for this observation and
never falls back to the normal transformer. Callback errors remain loud and
propagate as `candle::Result` errors; callers must not resume a failed state.
The patch does not promise that an arbitrary callback is nonmutating: the
reviewed collector must serialize observations without mutation.

The seam does not close raw K/V event capture, a separate correctly
source-bound Candle collector/consumer, a full-PCM driver, or an independent
reference oracle. In particular, it does not satisfy the separate schema that
requires KV-event positions plus independent keys/values tensor artifacts, and
it does not label Rust outputs as a Python oracle. Those remain open work.

`cache_observer_patch.py` writes three fresh patched source files and one fresh
receipt with exclusive-create semantics. It refuses source drift, whole-file
license mutation, wrong or duplicate anchors, symlinks/unsafe ancestors,
existing destinations, and output/receipt path aliasing. `revert_*_bytes()`
proves byte-exact reversal in memory before writing. Acceptance requires a
successful process exit (`0`), a complete receipt, and all three patched-output
hash/size bindings in that receipt. A write failure leaves any partial artifact
for inspection; failed or partial outputs/receipts remain unaccepted and must
not be used as build inputs.

## Offline verification

Use the repository’s fixed Python 3.12 environment and no network:

```text
UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 uv run --no-project --offline --python 3.12 python -m unittest discover -s tools/parity/kyutai_candle_reference -p 'test_cache_observer_patch.py' -v
UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 uv run --no-project --offline --python 3.12 python tools/parity/kyutai_candle_reference/cache_observer_patch.py --self-test
```

These tests authenticate the full pinned fixtures, reject same-size mutations,
wrong/missing/duplicate/partial anchors, verify callback ordering and default
behavior textually, prove exact reversal, and exercise no-overwrite,
distinct-path, symlink, unsafe-ancestor, and partial-write handling. They do
not compile, import, or run upstream/model code.

## Explicit holds

- Rust compilation, Moshi/Candle runtime, model execution, CPU, and Metal:
  `NOT_RUN`.
- Raw K/V collector/consumer integration and full independent PCM reference:
  `NOT_RUN`.
- Model/tokenizer/weight acquisition and provider operations: `NOT_ACQUIRED`.
- License owner review and publication: `BLOCKED_PENDING_REVIEW` / `NO_UPLOAD`.
- This source candidate is not `VAST_READY` and does not activate any
  compatibility or parity job.
