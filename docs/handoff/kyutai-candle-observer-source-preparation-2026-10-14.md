# Kyutai Candle observer source preparation — 2026-10-14

Prepared against accepted main
`d4a54d6212b3e60c702089c626d6d0f4ece23894`. PR #235 was normally merged at
09:39:44 UTC after exact head `64422404` passed 69 checks, with one skip and
all 16 required context/application pairs successful. Local main was updated
by fast-forward; this separate candidate does not alter that main checkout.

This is source-text preparation for an independent upstream reference, not a
Rust build, API compatibility result, full PCM oracle or real-weight result.
The [58-row inventory](current-58-prerequisite-inventory-2026-10-14.md), owner
scope and numerical bounds remain unchanged. No model/tokenizer payload,
local model execution, paid worker, sign-off or upload is involved.

## Actual observation seam, not a second forward implementation

The fixed official ASR callback runs before `LmModel::forward_cond`, so it
cannot directly observe the last step's selected token or logits. The
[source-only patch helper](../../tools/parity/kyutai_candle_reference/README.md)
adds an optional observer to the authenticated official implementation. It
captures actual logits, transformer output and selected tokens after the
existing selection/conversion and before selected-token state updates.
It does not add or replace a forward, sampling algorithm, dtype conversion,
padding rule, counter reset or numeric bound.

The source remains recoverable byte-for-byte by removing the five exact
additions. Global `model_step_idx` has already been incremented at the
observation point; it is not silently reset or relabeled. The observer defaults
to `None`, requires `Send`, and propagates errors. An observer error is terminal
for the run: the partially advanced State must not be resumed.

| Input/output | Fixed identity | Bytes |
|---|---|---:|
| Official `rust/moshi-core/src/asr.rs` at `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` | Git blob `0f3ad70082f152e3250aa81ab4e0b48ffef550db` | 9,125 |
| Official `rust/LICENSE` at the same commit | Git blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11,357 |
| Expected instrumented ASR source | SHA-256 `053bc81d853a86eeda061d2b4ff17193b94187554b1db485b000a3fb1f39a52a` | 9,887 |

The exact upstream Apache-2.0 license accompanies the source fixture. That
source receipt is not an owner/license approval for the reference closure or
weights. The generated patched source and per-invocation receipt are not
checked in as if they were build/runtime evidence.

## Safety corrections and actual static verification

Manager review caught an unowned-file deletion path in failed output handling.
The corrected helper performs no automatic deletion. Exclusive creation
refuses existing files, and failed partial output/receipt files remain
unaccepted for inspection; they must not become build inputs. Original inputs
are never modified. Two redundant generated copies in the isolated development
checkout were removed; original temporary evidence and reproducible fixtures
remain available.

Input size is checked on the opened regular-file descriptor before reading;
reads are capped at the expected byte count plus one. Oversize, growth,
short-read, replacement, same-size hash drift, symlinks and unsafe destinations
fail closed. Linux tests use a canonical system temporary directory rather
than assuming the Mac-only `/private/tmp` path. Production path rules are not
relaxed.

Luna and manager independently pass 17 stdlib tests using offline/no-project
UV/Python 3.12. These include the real pinned source/license pair, additive
reversal, bounded-read failures, no-overwrite/concurrent-replacement cases,
CLI refusal and temporary-directory portability. Existing workflow-security
gets one explicit invocation with the existing pinned UV setup; no action,
permission, dependency or job is added. Actionlint passes. These results do
not claim a Linux Rust compile, runtime observation or model parity.

## Remaining gates and execution order

1. Preserve this reviewed source preparation as a separate private logical
   commit while the existing dependency PR runs.
2. Finish PR #234's latest-base CI and normal acceptance. Its updated candidate
   is `54340ba5`; old `2bb9c662` checks do not cover the new head. Normally
   incorporate that accepted main before this preparation's own fresh PR CI.
3. Obtain the actual Linux core-only graph and reviewed precise memmap2 update,
   license/native/advisory evidence and exact source/model/operator decisions.
   The [Candle admission record](kyutai-candle-reference-readback-2026-10-14.md#source-metadata-only-offer-admission-readback)
   retains the fixed RAM/price gates and no-worker verdict; do not repeat the
   known-blocked unchanged lock on a paid worker.
4. Only after admission, validate the observer's Rust/API/worker properties
   remotely and bind the complete PCM/Mimi/tokenizer/reset/scheduling writer.
   Candle and the private PyTorch schedule are distinct contracts, not an
   unapproved reference swap or token-position correction.
5. Approved no-upload independent reference and real-weight CPU parity must
   pass before the final clean HEAD and regenerated Apple packet. No part of
   this source-only preparation closes that original step-4 goal.
