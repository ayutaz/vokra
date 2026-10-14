# Reference gate binding readback — 2026-10-14

This is a source-only investigation at code-equivalent HEAD
`6330a1a2f089aec89bf6a8cdda7797df9bf7ed76`. It follows the
[58-row inventory](current-58-prerequisite-inventory-2026-10-14.md), not a new
model execution, dependency approval or numerical result. The protected
CosyVoice2 manifest was excluded without reading or hashing it.

## Why hash differences alone were insufficient

A first static scan found Qwen3-TTS probe hashes and old Dia/Canary license
snapshots differing from sibling current locks. The initial report SHA-256 was
`f5f2cc1b35b651476b458138944d6b323323529d0bb37c2f9982a24fff978793`.
The subsequent call-site/README investigation, report SHA-256
`1976ab849af61bb8eff04c8d0902c55c0424ecd461444e7b4fc1f7b6ac3d7c8e`,
corrected the initial assumption that the old Dia/Canary manifests were
current worker gates. They are historical records and were not rewritten.

The initial broad scan also mis-associated a nested MOSS API-smoke project
with its parent lock. The main and nested projects match their own bindings.
CLAP's explicitly named `PREVIOUS_LOCK_SHA256` is historical, not a broken
current approval. SpeechT5 and microWakeWord adjacent probe findings remain
outside the current 58-row denominator.

## Dia: active contract still needs separate work

- `tools/parity/dia_1_6b_reference/license_gate.py` and its old manifest are
  fixed to historical Torch 2.6, 26 packages and 159 native files. The reference
  README calls that closure superseded. The current worker does not call that
  historical gate.
- `scripts/publish/vast-ai/audit-dia-1-6b-dependencies.sh`, current
  `dependency_audit.py` and `owner_review_scope.py` bind lock
  `06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449`
  and project
  `4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb`.
- The current validation worker explicitly rejects the fixed official source's
  Torch/TorchAudio 2.6 pins against the security-refresh candidate with
  `BLOCKED_UPSTREAM_PINNED_COMPATIBILITY`, and requires external dependency
  and source/model approvals before execution.
- The active `tools/parity/dia_1_6b_dump_reference.py` and
  `tools/parity/dia_1_6b_validate_evidence.py` still encode the older reference
  lock, direct Torch identity and lock-audit contract. Reconciliation requires
  current dependency/native/API evidence; blindly changing hashes or carrying
  historical approvals forward would not resolve that compatibility gate.
- The reference README records a dated current-candidate audit with 27
  packages, 147 publisher-license entries, 164 native files and unsigned
  `PENDING_OWNER_REVIEW` / `NO_UPLOAD` scope. Its external report, preparation,
  scope and signed approval were not available in the inspected checkout and
  temporary evidence locations. Documentation hashes are pointers, not a
  recovered current receipt or approval.

## Canary: retain historical closure, use the current chain

`tools/parity/canary_1b_reference/license_closure_gate.py` and its manifest
record the historical Hydra 1.3.2 / 134-row closure. Current workers instead use
`tools/parity/canary_1b/preflight_gate.py`,
`scripts/publish/vast-ai/audit-canary-1b-dependencies.sh`, the current dedicated
auditor and `dependency_approval_gate.py` with external signed approval.
The current lock/project are
`a931dfb0884bfe2b74f43ca5e372abe6a523de347e7789802105077c5be53a4c` /
`b4d21fdcf9c329fde6ad70da090ad90d8f60ebf209817949e3863fbb87ff465b`.
Historical archive/license counts cannot satisfy that current approval chain.
Current dependency/native/API review, complete artifact replacement and
independent CPU parity remain separate unresolved gates.

## Qwen3-TTS: current probes are a distinct repair

The [current license-gate reconciliation](qwen3-tts-current-lock-reconciliation-2026-10-14.md)
already binds the current `e03f60fb...` lock without carrying approval forward.
Unlike the historical Dia/Canary snapshots, both current Qwen3-TTS API probes
still pin `98cef03a...`. The model-free probe's stdlib self-test itself calls
`verify_project()` against the current lock, so its old binding also prevents
the controlled fixture from reaching its intended validation.

The bounded implementation changed only the two production lock constants and
focused stdlib regressions in those probes and `test_lock_pending_review.py`.
Manager diff review confirmed that host/source/checkpoint/CPU-index/security
guards and approval algorithms were unchanged. Both actual current-lock readers
passed, while old bindings and altered lock fixtures were rejected. The AST
check reads only the unique module-level constants, not self-test reassignment.

Manager independently ran both probe `--self-test` entry points, the pending-
review regression and license-gate self-test through offline UV/Python 3.12;
all exited 0. The 59 package and six component rows still have null approval
schema/signer/digest. Their current license/native/owner fields remain pending;
synthetic self-test evidence is never an actual API import or model result.
Historical evidence was not rewritten. Remote workspace verification, PR
acceptance and authorized current API/weight evidence remain separate gates.

No model, archive, native package, Cargo or cloud operation was performed by
this call-site investigation. No owner decision, signer, approval digest,
license class, numerical tolerance or protected input was changed.
