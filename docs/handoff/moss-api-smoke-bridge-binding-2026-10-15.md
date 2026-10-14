# MOSS API-smoke bridge binding — 2026-10-15 JST

Private source-only follow-up `78f72c9fea4d8114a361d4f7d111c3c12d3c53aa`
extends the [urllib3 candidate reconciliation](moss-api-smoke-urllib3-reconciliation-2026-10-15.md).
It is not PR acceptance, an owner approval, an API smoke run or CPU parity.

## Actual consumer mismatch and authoritative input

The VAST validation wrapper uses the separate API-smoke project and checks
the main preflight bridge before environment setup. That bridge still used
older project/lock identities and Torch/TorchAudio 2.9.1 expectations. The
prepared update binds its actual current project and corrected lock:

| Input | Identity |
|---|---|
| API-smoke project SHA-256 | `17431a2f3ddfb6a9853c47c51f2e99209fcf8c0497e0675c5eb4016cbed58aec` |
| Corrected lock SHA-256 | `517ed7226440a5b6b6d2e462619ad705ba24bc7bc8e87cdecc59317a77b1d4be` |
| Declared direct pins | Torch 2.13.0 / TorchAudio 2.11.0 / Transformers 5.10.4 |
| Scope cardinality | 45 full lock records / 44 registry identity rows, excluding the virtual project |

These are whole-lock declarations, not 44 active Linux packages or installed
API compatibility. Main MOSS's separate 48-row manifest and pending reviews
remain unchanged. Historical approval/evidence bytes are not restamped.

## Why updating constants alone was insufficient

The earlier evidence validator checked internal package/scope digests but did
not compare package identities to authenticated lock-derived rows. In a
bounded fixture, the original validator with only current bridge constants
accepted a mutated package row plus its matching review and recomputed
self-hashes. The corrected real validator rejects the same data:

```text
old-naive-bridge=(True, 'PASS')
final-authoritative-bridge=(False, 'API smoke evidence package rows do not match the authenticated lock')
```

This demonstrates the latent weakness of a constants-only correction, not
that the original production worker passed: its old project contract and
independent license/owner gates already blocked current execution.

The shared loader uses only the checked-in API-smoke parser. It requires the
supplied verifier to be byte-equal before loading trusted code, then derives
rows from actual authenticated project/lock bytes. Evidence validation now
requires the authoritative project and compares the complete row list and
digest. An input-supplied forged verifier is rejected without executing it.

## Verification and retained holds

The new [focused regression](../../tools/parity/moss_audio/test_api_smoke_preflight_binding.py)
calls real production validators. It covers current structural inputs,
stale project/lock, forged verifier sentinel, stale row/scope digests,
self-consistent changed rows, omissions and duplicates. It copies only three
known project files into owned scratch, never an entire project or environment.
Synthetic PASS data uses explicit self-test identities and is not retained
as a real approval/evidence packet. CI Security explicitly runs this test.

Luna and manager pass the focused regression, preflight and API-smoke
`--self-test` checks and diff hygiene. Workflow hygiene/actionlint pass;
all five normal pre-commit gates pass. Fresh integrated-head CI remains
separate. No target-package/model import, dependency sync/install, model/config/
tokenizer acquisition, compatibility run, heavy Cargo, cloud worker, owner
signature, license decision or publication occurred. Source/model grants,
native/API/owner review, actual CPU parity and the final Apple packet remain
unfinished; no public row or frozen owner scope is promoted.
