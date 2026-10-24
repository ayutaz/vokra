# Moshi ASR observer seam candidate (source-only)

This is a source-only candidate patch utility, not implementation acceptance
or a status promotion. It targets only the official
`kyutai-labs/moshi` source at commit
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`:

| input | fixed official raw URL | Git blob SHA-1 | bytes |
| --- | --- | --- | ---: |
| `rust/moshi-core/src/asr.rs` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/asr.rs` | `0f3ad70082f152e3250aa81ab4e0b48ffef550db` | 9125 |
| `rust/LICENSE` | `https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11357 |

`pinned-asr.rs` and `pinned-LICENSE` are the exact authenticated upstream
fixtures. The latter is the upstream Apache License 2.0 text and attribution
evidence; it is not an owner/license signoff or a new license row.

`asr_observer_patch.py` authenticates both complete local byte inputs against
those fixed Git blob identities before it can patch. It rejects source drift,
malformed/non-regular files, symlink components, double patching, existing or
symlink output/receipt paths, and never modifies the source input. Output and
receipt are exclusive-create artifacts, so the original remains recoverable.
There is no automatic deletion on any write failure: partial output or receipt
artifacts remain for inspection, are unaccepted, and must not be used as build
inputs. A concurrent creator or replacement is preserved by construction.
Authentication checks the descriptor size before reading and reads at most
`expected_bytes + 1`; it rejects oversize, short-read, growth, and replacement
drift before hash acceptance.

The additive observer contract is deliberately narrow:

- `State` defaults to `observer = None`; `set_observer` installs an optional
  `Send` callback returning `candle::Result<()>`.
- Exactly one call is inserted immediately after the existing
  `text_tokens.to_vec1::<u32>()?` conversion and before any selected-token
  state updates.
- The callback receives the actual global `model_step_idx`, current
  `ItemState` slice, `StreamMask`, `text_logits`, `transformer_out`, and the
  actual selected token slice. It performs no sampling, dtype, shape, counter,
  or numeric transformation.
- Callback errors propagate and abort the step loudly; callers must treat the
  `State` as failed and must not resume it after an observer error.

`revert_bytes()` is a byte-level reverse proof for the five exact additive
insertions. A successful invocation generates a receipt recording the original
source/license identities and the patched output identity. The expected
patched output identity is:

| artifact | Git blob SHA-1 | raw SHA-1 | bytes |
| --- | --- | --- | ---: |
| patched `asr.rs` candidate | `bf670f6cf2d25c0680b4d3681005e0d2b7bd2aa0` | `e3b48e0ee0e69127a0811198fb21f597d6e00f51` | 9887 |

The expected patched output SHA-256 is
`053bc81d853a86eeda061d2b4ff17193b94187554b1db485b000a3fb1f39a52a`; the
authenticated source SHA-256 is
`4bbe8dc8b12a2520ff0f228ca8c18b004927446d699b2b7712ea4fdec0e80e03` and the
license SHA-256 is
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

Receipts are generated only after a successful invocation; no receipt or
patched output is checked in here. A generated receipt's `reversible: true`
field is backed by `revert_bytes(patched) == authenticated_source`.

## Offline tests

The 17 tests exercise the authenticated full pinned source/license pair and a
tiny synthetic Rust-shaped fixture. They cover bounded oversize/short-read/
growth rejection, drift/size/license failures, same-size mutations, symlink
and no-overwrite guards, single-call insertion, reversibility, receipt
behavior, and concurrent output replacement. They do not claim that the
upstream source patch compiles or that Moshi runtime behavior is validated.
They perform no model execution and grant no numerical-parity approval.

```text
uv run --no-project --offline --python 3.12 python -m unittest discover \
  -s tools/parity/kyutai_candle_reference -p 'test_asr_observer_patch.py' -v
```

## Unrun gates and holds

- Remote Linux Rust build/check and runtime observer execution: `NOT_RUN`.
- Candle/Moshi dependency, model, tokenizer, checkpoint, and weight work:
  `NOT_RUN` / `NOT_ACQUIRED`.
- Owner/license review and any model publication approval: blocked/pending;
  the Apache-2.0 source license receipt is evidence only, not an owner signoff.
- CPU/Metal/GPU compatibility and numerical parity: `NOT_RUN`; this seam does
  not close the PCM oracle or parity gate.

The remaining Rust/API risks for root review are borrow-checker validation of
the callback field access, downstream `State` worker `Send` requirements, and
the exact public API decision for a boxed `FnMut`. Local Cargo and model
execution are intentionally prohibited.

## Current Candle advisory evidence boundary (2026-10-20)

The applied-advisory and one-shot CPU workflows now create a fresh source-only
producer in the same workflow run. The producer uploads the historical exact
eight-member application packet plus a separate bounded envelope containing the
member byte/hash map, application-receipt self-hash, source checkout head,
workflow-definition SHA, caller identity, and upload/API artifact digest and
size. Consumers authenticate the current run through the GitHub REST API: the
overall run may still be `in_progress`, but the unique producer job must be a
completed success with the matching run, attempt, head, caller workflow, and
reusable-workflow reference. Both artifact IDs, names, digests, sizes, expiry,
and the complete bounded API listings are rechecked before any advisory or CPU
work. An expired historical packet is never accepted as a fallback, and the
frozen baseline/candidate/applied profiles remain unchanged.
