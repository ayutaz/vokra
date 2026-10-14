# Dia dependency-scope validator reconciliation — 2026-10-14

Source-only candidate based on accepted main
`fb1a692a24ebe4bb928556699eff6fbce4750b46`. This is a validator correction,
not a new measured dependency receipt, owner approval or reference result.

## Mismatch and correction

The current dedicated candidate lock is SHA-256
`06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449`;
its project is SHA-256
`4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb`.
The current audit/owner-scope producer uses a 27-package Linux closure and
147 recorded publisher-evidence bytes. The approval validator still required
the historical 26-package / 50-byte contract, so its model-free self-test
could pass without accepting the current producer's scope shape.

The implementation changes only those two production expectations. Schema,
canonical scope digest, expected HEAD, native-row validity, three evidence
digests, approval-file SHA and external signer binding remain strict. The
unsigned scope must remain `PENDING_OWNER_REVIEW`, owner decision `None`,
`BLOCKED_UNREVIEWED_TRANSITIVE` and `NO_UPLOAD`. No real approval is generated.

Model-free synthetic tests include the current shape, the rejected legacy
26/50 shape, independent cardinality/publisher mutations, duplicate/unsorted
identities, malformed native/evidence inputs, populated owner decision,
nonempty audit failures, old approvals against changed valid evidence digests,
and wrong supplied approval-file SHA. Both implementer and manager run the
stdlib-only self-test offline through UV Python 3.12 and obtain PASS. These
fixtures use synthetic evidence and signer names; they are not owner records.

## Remaining legitimate holds

The authenticated official Dia source stays at
`2811af1c5f476b1f49f4744fabf56cf352be21e5`. Its original Torch/TorchAudio
2.6.0 requirements differ from the candidate Torch 2.13.0 / TorchAudio 2.11.0
route. The upstream compatibility probe correctly remains
`BLOCKED_UPSTREAM_PINNED_COMPATIBILITY`; this correction does not bypass it.

The active dumper and evidence validator still bind the older lock/project,
direct-dependency and optional TorchAudio-seam contract. They need a separate
coordinated source/API correction and independently measured evidence before
execution. Native/transitive license review and external owner decisions
remain pending. Do not restamp historical approval/audit/reference receipts
or infer API compatibility from source or synthetic tests. No model/package
import, weight/config/tokenizer acquisition, reference execution, numerical
bound change, local heavy Cargo, paid worker or artifact upload occurred.

See the [reference binding readback](reference-gate-binding-readback-2026-10-14.md)
and [current unresolved inventory](current-58-prerequisite-inventory-2026-10-14.md).
Dia remains unresolved; final Apple packet and Scaleway are separate later gates.
