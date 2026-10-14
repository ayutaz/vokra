# MOSS-Audio current-lock reconciliation — 2026-10-14

Candidate for PR #242, based on accepted main
`fb1a692a24ebe4bb928556699eff6fbce4750b46`. This is model-free input binding
and metadata repair, not dependency/API execution or approval.

## Exact change boundary

The original Dependabot head `3e4f895e6b89e1a28c933884ef41e50bd909d3da`
updated urllib3 and also changed Torch/TorchAudio dependency markers. The
markers and edges are restored byte-for-byte to accepted-base values. Full
TOML comparison preserves all 48 package records, resolution markers,
project metadata and non-urllib3 dependency semantics; no dependency is added.
urllib3 2.8.0 wheel/sdist URLs, hashes and sizes match the accepted Nanocodec
row and the existing [primary release review](../license-audit.md#shared-parity-urllib3-release-review-2026-10-05-pr-194).

Nine PyTorch wheels in the base lock lacked `size`, which the unchanged strict
artifact schema requires. Credential-free HTTP HEAD requests to the exact
locked official URLs returned HTTP 200 and these Content-Length values:

| Package/version and Python 3.12 platform | Bytes |
|---|---:|
| Torch 2.9.1, macOS arm64 | 74,388,972 |
| Torch 2.9.1+cpu, Linux aarch64 | 104,126,462 |
| Torch 2.9.1+cpu, Linux x86_64 | 184,378,187 |
| Torch 2.9.1+cpu, Windows amd64 | 110,856,378 |
| Torch 2.9.1+cpu, Windows arm64 | 94,843,899 |
| TorchAudio 2.9.1, macOS arm64 | 808,135 |
| TorchAudio 2.9.1, Linux aarch64 | 476,017 |
| TorchAudio 2.9.1+cpu, Linux x86_64 | 495,619 |
| TorchAudio 2.9.1+cpu, Windows amd64 | 663,086 |

Only those size fields are added. All PyTorch artifact URLs, hashes and upload
times are unchanged. The [official CPU TorchAudio index](https://download.pytorch.org/whl/cpu/torchaudio/)
also confirms the existing Linux x86_64 hash
`43cf20a2965cf081945c91d2dc8844377e5e3f1b172c0d0c18399ca3ecf1f899`.
It has 64 hex characters in both accepted base and candidate; an intermediate
claim of a pre-existing short hash was incorrect and is not an accepted
finding. HTTP metadata is not authentication of wheel bytes. No wheel body
was downloaded, installed or imported.

## Rebound, still unsigned inputs

The dedicated project SHA-256 stays
`00dbde83ac503f0ce91d74b3da9fb2f6010d827efedd62831b276e3abeb28687`.
Current lock SHA-256 is
`eaa1cd47a253bfd72f962d30f05362ed76b231ebef6cd2bce5e5f3cc8ed47d8c`;
package-row SHA-256 is
`37f246d9206db8b7867d53098dd384eba2472bf0c2c18d76461e3f008c43b04b`.

The old manifest contained a pending `psutil@7.2.2` review row even though
psutil was absent from the base/current lock. Removing that stale review is
not removal of an installed or resolved dependency. The pending urllib3 row
is updated to 2.8.0, giving 48 review rows matching the lock. Dependency-review
SHA-256 is
`dc1084afd693097fb587999c48486233cb07f0f938eff3ee893d7b35578a9c3a`.
The new approval scope is
`6cfe95b3317173af13db255569631f37b6f467933e1d6fd1c16e5826ef0d6ae5`;
historical scope `cdda6be3a60e4b703c0d26d249a21703b69c2024f10bfabedf4896e2d7bb2b9a`
and its evidence must not be reused as current approval.

All component/dependency reviews remain pending, operator signer/digest remain
null, and publication remains `NO_UPLOAD`. Source/model identities and the
separate patched-reference/API-smoke project are unchanged. The production
gate reaches unresolved review, not `VAST_READY`. Missing primary model/source
licenses, native/dependency review, API compatibility, operator decision and
real-weight/reference/CPU evidence remain open. This does not restamp any
measured audit, change parity bounds or create a license grant.

Offline UV Python 3.12 preflight and identity self-tests, complete TOML/manifest
state comparisons and diff hygiene pass. Persistent binding/tamper tests are
synthetic checks, not real execution evidence. No model/config/tokenizer
payload, local heavy Cargo, cloud allocation, reference execution, artifact
upload or owner sign-off occurred. Keep both MOSS-Audio variants in the
unresolved inventory; final Apple packet and Scaleway are later gates.
