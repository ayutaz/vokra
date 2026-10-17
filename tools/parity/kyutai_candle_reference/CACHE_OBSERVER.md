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

### Hosted source-processing evidence (2026-10-17)

At reviewed head `777d7881a0dc4ea207e32153cec9541f0d71cfc9`,
[source-test run 38998090705](https://github.com/ayutaz/vokra/actions/runs/38998090705)
succeeds on Ubuntu, macOS and Windows. The manager independently checks each
original job log: fifteen cache tests, seventeen ASR tests and the explicit
cache self-test pass on each platform, without failures or skips. These are
source-processing tests, not upstream Rust compilation or model execution.

The earlier head's typos v1.50.3 failure reports the verbatim upstream API
identifiers `Tensor::arange` and `from_mmaped_safetensors`. The two exact
authenticated fixture paths are excluded under the existing vendored-source
policy; their bytes are unchanged and no global word override is added.
Superseded still-running CI jobs are cancelled after the fresh-head jobs are
confirmed; the original failure remains historical evidence, not a pass.

## Patched-library compatibility preparation (2026-10-17)

`check_cache_observer_compatibility.py` is a separate, unactivated remote-only
runner. It normally imports the unchanged CPU checker helpers; it does not
monkey-patch the historical checker or change any frozen producer. Its new
opt-in marker is distinct from the consumed PR #255 technical scopes.

The prepared sequence authenticates the original eight-file application
delivery and frozen helpers, replays the original applier into fresh roots,
authenticates the actual projection's four source/license inputs, and applies
the existing observer patch. All three output bindings and byte-exact
reversals are checked. Detailed projection/vendor/selected-registry snapshots
must show exactly the three intended source changes, with no manifest, lock
or dependency-source change. The before-build snapshot is explicitly patched.

The intended remote path then builds the full locked offline CPU library,
checks the unchanged 105-package graph and compiler-artifact identities, and
compiles an uncalled API probe for both observers and actual cache/item field
types. Network-namespace and no-new-privileges checks, Rust/Cargo 1.99.0,
fresh target/output paths, a shared 900-second deadline and bounded original
streams are retained. Snapshots are checked immediately after build and again
after the API probe. Failures remain non-success and preserve bounded evidence.

Manager independently passes all 27 new offline tests, in addition to the
15 cache and 17 ASR source tests. The orchestration tests mock every external
stage; they do not compile or execute upstream code. Negative tests cover
changed provenance/approval/gates, wrong output bindings, partial patches,
failed reversals, stale or overlapping outputs, wrong toolchain, source drift,
compiler failure and timeout retention. This is preparation, not an actual
compatibility result.

Original application artifact `11985400172` is no longer retrievable (API
404 on 2026-10-17). The preserved private evidence commit
`e9f236e9c6a811160e9cc69518aef0b49c6f9b45` is clean and its eight files,
4,333,048 bytes total, independently match every SHA-256 in the
[original application record](../../../docs/handoff/kyutai-candle-paste-application-result-2026-10-16.md).
No replacement artifact has been uploaded and the private repository root
must not be passed as an eight-file delivery because it also contains `.git`.
An actual job needs a reviewed delivery of only those exact eight files and
a fresh bounded execution scope; neither consumed compiler label is reused.
No actual compiler activation workflow or label is supplied by this preparation.

## Explicit holds

- Rust compilation, Moshi/Candle runtime, model execution, CPU, and Metal:
  `NOT_RUN`.
- Raw K/V collector/consumer integration and full independent PCM reference:
  `NOT_RUN`.
- Model/tokenizer/weight acquisition and provider operations: `NOT_ACQUIRED`.
- License owner review and publication: `BLOCKED_PENDING_REVIEW` / `NO_UPLOAD`.
- This source candidate is not `VAST_READY` and does not activate any
  compatibility or parity job.
