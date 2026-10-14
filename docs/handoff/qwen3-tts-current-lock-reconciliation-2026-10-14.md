# Qwen3-TTS current-lock reconciliation — 2026-10-14

## Scope and cause

The dependency refresh merged through PR #223 changed `tools/parity/qwen3_tts/uv.lock`,
but the gate and manifest still required the previous lock digest. This change repairs
that input binding; it does not approve a new dependency closure or model execution.

| Input | SHA-256 |
|---|---|
| Previous lock | `98cef03a391c9a31116b0b63c1dca4bb456c1a58ec407a4567e78e648eaf8857` |
| Current lock | `e03f60fb15066ed5efbcd4d682c8a2f61eefc5ca261f982d1943988e84f85fd6` |
| Unchanged project | `277ac9a2f414a6a41fa473c956c71bb6d7680c83c011afc53cff08276d102711` |
| Retained historical dependency audit | `cc22b0e23b225a26e47ad503033c280c47b8c456f05682252265406655ad1d42` |

The current lock changes urllib3 from 2.7.0 to 2.8.0 and changes the platform-marker
coverage of six Torch dependencies. Locked archive metadata is not a fresh installed
payload, native inventory, or license observation. The new urllib3 row is therefore
unresolved; the four Torch/TorchAudio variants remain unresolved as before.

## Approval boundary

The new proposed scope digest is
`37b431de525248bcc3950394c9ef93b0a36f67b0528e2983dec769ba48e09d7e`.
All 59 package rows and six component rows have null current-scope approval fields.
Unchanged factual license/native/payload fields are retained, not presented as a
new audit. The existing approval-subject wire format and fail-closed algorithm are
unchanged. Historical dependency and operator evidence files are not rewritten.

The actual gate now passes the current lock input check and refuses the historical
dependency audit. A controlled stdlib-only fixture invokes `license_gate.run()` and
checks this rejection ordering. It does not import a model or resolve dependencies.
The API probe scripts retain their previous lock binding and are not made executable
against the new closure by this repair.

## Verification and remaining work

Manager review and local verification:

- `license_gate.py --self-test`: PASS.
- `test_lock_pending_review.py --self-test`: PASS; actual gate exits 2 on the
  historical dependency audit, rather than rejecting the current lock digest.
- Normalized manifest comparison: only the lock/package bindings, derived hashes,
  current-scope approval fields, and urllib3 unresolved row changed.
- Historical audit digest and local model-safety guard digest remain unchanged.
- `git diff --check`: PASS.

Python checks used offline `uv run --no-project --python 3.12`. No model acquisition,
local model execution, Cargo build, cloud lease, or artifact publication occurred.
This is not `VAST_READY`: current archive/native/license evidence, exact owner
decision, and approved source/API/reference prerequisites remain to be resolved.
Real-weight reference and CPU parity belong to the later approved VAST stage;
they are not prerequisites that must already exist before that stage can start.
