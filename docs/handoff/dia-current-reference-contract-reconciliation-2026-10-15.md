# Dia current reference contract reconciliation — 2026-10-15 JST

Source-only candidate based on pending PR #245 head
`f89e092735ce5a00cb0a5f2c6e8dfae019774c87`, which normally includes accepted
main `f0130b5de5c37e4441612a85ecaa4e39ae0344f9`. This is coordinated
metadata-contract correction, not new dependency/API/runtime evidence or an
execution approval. The [earlier admission/scope correction](dia-dependency-scope-reconciliation-2026-10-14.md)
and its unconditional execution refusal remain intact.

## Actual current inputs, not restamped historical receipts

The unchanged dedicated lock is SHA-256
`06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449`;
the unchanged project is
`4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb`.
The full inventory has 31 lock records, whereas the Linux x86_64 active closure
has 27. The difference is two inactive Torch/TorchAudio records, Windows-only
colorama and the virtual project. Both inventories are retained and separately
hashed; one count is not substituted for the other.

The new stdlib `tools/parity/dia_1_6b_reference/lock_contract.py` uses the
existing dependency collector's project/marker/identity semantics. Dumper and
validator bind that exact current contract rather than old 29-record metadata.
Validation reopens local pinned project bytes, checks exact schemas, full and
active membership, canonical digests, allowed index/source values, duplicate
identities and integer counts. It does not import distributions or access
model/archive payloads. The lock input must be canonical, regular, non-symlink
and named `uv.lock`, not silently replaced by another file in its directory.

Seven current direct dependencies are bound. Project declarations use Torch
2.13.0 and TorchAudio 2.11.0; the separately checked Linux installed-version
map expects 2.13.0+cpu and 2.11.0+cpu. A declaration is not evidence that those
distributions are installed, compatible or approved.

## Explicit pending and legacy-seam semantics

Historical static `*_REVIEWED` package labels are not reused as current legal
decisions. The factual lock packet is `BLOCKED_UNREVIEWED_TRANSITIVE`, license
classification `NOT_PERFORMED` and review `PENDING_OWNER_REVIEW`.
No primary license fact, owner decision or receipt is generated or relabeled.

The historical no-audio TorchAudio module stub is still only a fail-closed
legacy seam: all audio load/use aborts and `audio_prompt=None` stays fixed.
Its metadata now declares the locked distribution separately, with installed
state `UNKNOWN_NOT_CHECKED` and compatibility evidence
`LEGACY_UNVALIDATED_NO_COMPATIBILITY_EVIDENCE`. The old claim that the current
project has no installed distribution is not accepted as current evidence.
This does not prove that the stub is a compatible full reference route.

Both CLI and programmatic `run()` continue to refuse before input/path/import
access with `BLOCKED_UPSTREAM_PINNED_COMPATIBILITY`. The worker's same
unconditional guard is unchanged. Pinned official source still requires
Torch/TorchAudio 2.6.0; exact current upstream/API/ABI review and independently
measured evidence remain prerequisites to any future enabling change.

## Verification and remaining gates

Luna and manager independently run offline/no-project UV Python 3.12 tests.
The new suite tests the actual dumper and validator consumers, rejecting stale
hashes, old 29-record/six-dependency contracts, malformed sources/digests/counts,
wrong full or active membership, missing/mismatched synthetic distributions,
legacy seam descriptors and altered local lock bytes. Installed-version mocks
are explicitly synthetic: they are not an installed-package/API result.
Existing dumper, validator and direct-admission self-tests also pass. The new
suite is wired into the Linux security job for fresh exact-head CI; local
success does not replace that pending CI or a real VAST run.

Source/model/DAC identities, worker and external approval gates, numerical
operations/bounds, project/lock bytes, manifests and historical receipts are
unchanged. Native Rust is unchanged. No target package/model import, weight,
config/tokenizer acquisition, local heavy Cargo, paid VAST/Scaleway worker or
publication occurred. Dia remains unresolved; dependency/native/license/owner,
full real-weight reference/CPU parity and the final Apple packet remain open.
