# VibeVoice Realtime static dependency-closure audit — 2026-10-13

> **2026-10-14 source-only follow-up:** [Exact source-license reconciliation](vibevoice-source-license-reconciliation-2026-10-14.md) verifies the two pinned upstream LICENSE identities and safetensors' official repository move. It does not approve the audited wheels, native payloads or execution. The old raw-evidence directory below was not present at the new inspection; its availability statement is retained as the original dated observation, not a current recovery guarantee.

## Verdict and scope

The single authorized VAST static audit finished with controller **PASS**,
remote/recovery exits **0/0**, and independently accepted worker/data cleanup.
The collector verdict remains **`OWNER_REVIEW_REQUIRED_NO_UPLOAD`**:
license `UNRESOLVED`, native payload `UNRESOLVED_FOR_RELEASE`, owner
`UNAPPROVED`, and runtime compatibility `UNPROVEN`. Operational PASS is not
an owner/legal sign-off, API test, model result, or publication decision.

This is a dated audit of seven frozen **private candidate inputs**, not a
validation of the reference environment currently checked into `main`
`7d38061c3dcb56f6a16bcf4b8085d966b3cdab09`. The private review HEAD was
`915b591f11f9684de98e4aa68643f61e46ae83bf`. In particular, main's project/lock
SHA256 values are `4498804bacda712defe082a6c6d8b79df4e29c1b35fac37db872cd7986cc9cd9`
and `e8787e3c9e7bfdb3383fca1bba22026edbec4b9e61191692797e43e801f74352`;
they differ from the audited inputs below, and `audit_installed_closure.py`
is absent from main. Draft PRs #169 and #182 are not accepted-main evidence.

No model was downloaded, loaded, converted or executed; no independent
model reference or API-compatibility probe ran; no HF token was transferred
and no artifact was uploaded. No Apple CPU/Metal/no-fallback check ran.

## Authorized inputs and environment

The authorization was one static audit, at most USD 0.20/hour including
storage and at most 30 minutes, followed by destruction of the worker and
stored data and withdrawal of the exact temporary guard exception.
The selected quote was USD 0.11844444444444445/hour (not an invoice total).
Worker `55648877` had 16 effective CPU cores, 256.498 GB RAM and 150 GB disk.
Execution and destruction completed within the authorized window on
2026-10-13 UTC; terminal cleanup readback completed at approximately 01:09 UTC.

Seven source/control inputs totaling 4,997,026 bytes were transferred over
encrypted SSH. Their exact SHA256 identities are:

| Input | SHA256 |
|---|---|
| Remote static runner | `157ab4e157bcd274af83bbde18fc6db9f1582e82cd7e2a6fa5edfb8ecf4179cd` |
| Static plan | `2cf398ad0dedc2c028dd611ce70d567022ff30a2236ccaca449e770a64030788` |
| Root-reviewed wheel intent | `7d3d051646182672454c241bd6308ebf4e6de475fe3a78fd9da8a9fff54bcd24` |
| Private pyproject.toml | `0d2642a580732293884f7d35a79e22fd37479632afcda6321efc2d74ed6f4c54` |
| Private uv.lock | `34f58e53b5c79ed96853c2a4b6f9b6b1eaf12066f010cd23ddffae2b816b3797` |
| Installed-closure collector | `df8e2f2a92631589ba87c0f223285201dcbde77d4ef4b7c4d677b50eb360620e` |
| Receipt validator | `ee56f10f62b7e29f16ee7e801ee50ec06c36ea7fc99d899e62d06d44ebfe83d7` |

Owner-record SHA256: `303e590123a1e85260a24d3c8ddfb2400944184bd987314cf182da0578d23e0b`.
The toolchain was UV 0.12.5 / managed CPython 3.12.14 x86_64. The 41 selected
wheel archives totaled 253,756,867 bytes. Upstream source identity was
Microsoft/VibeVoice revision `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`;
only its LICENSE and project metadata were fetched for this audit.

The audit used offline wheel installation and first-party standard-library
inspection, not imports of installed third-party/native code. It checked
the exact `.lock` zero-length regular-file/nlink identity transition and
the bounded SymPy share/man relocation. These checks do not demonstrate
runtime importability or execution compatibility.

## Findings and evidence integrity

All 41 selected distributions were inspected. Bundled license evidence was
missing for **safetensors 0.5.3** and **tokenizers 0.22.2**. This is an evidence
gap, not a finding that those projects grant no license. Exact-version primary
license/provenance and redistribution review remains required.

Ten packages contain native payloads requiring exact-scope review:
charset-normalizer, hf-xet, markupsafe, numpy, pillow, pyyaml, regex,
safetensors, tokenizers and torch. Neither their presence nor successful
static enumeration approves them for release or model execution.

| Recovered evidence | SHA256 |
|---|---|
| Controller receipt | `7d8eaa6265008c2c3f7e0b87a1bf2ee1adfe3b8efa0567f12d924dd3a4c69f67` |
| Remote receipt | `a56d0a05ff06f2f214c741107fa7c270a1cb65c135d8c96bc099b5310fb7ba12` |
| Collector report | `7f68f345a251a660228cc62d589b6a1469334f01c7afbc3ce711c6eeb9fca7a7` |
| Selected-wheel manifest | `528c049cd7a361dc6d8a26e0bb9e21b73e966657285f042976ff861ac14d215d` |
| Lock transition report | `ac225562e9772bd6334356ac796df204281e057e50303592412c9d17ada56935` |
| Source-file evidence | `3a4cc006a675f36512770cafc92b9d00567a5797e1c81d69fa1b115446ffba0e` |

Raw evidence remains local-only at
`/private/tmp/vokra-vibevoice-data-lock-audit41-own-dist-info-approved-evidence-20261013-0140`.
This directory is not a tracked artifact and is not available from a public
clone; this summary does not imply otherwise. The manager independently
validated the recovered receipts and report using UV/Python 3.12 without
executing models. Provider credentials and raw account diagnostics are not
included in this public record.

## Cleanup and original safety rules

Destroy/readback/inventory/cleanup statuses are all PASS. The owned worker's
readback was explicitly `{instances: null}`, and the owned-ID inventory
selection was empty. Its stored data was destroyed. This is **not** a claim
that the entire VAST account is empty: unrelated baseline resources were
outside this scope and were not changed. No new provider inventory query or
allocation was performed during the documentation refresh.

The authorized temporary **133 lines** in `guard-local-models.sh` were
withdrawn. Its original SHA256 was independently rechecked as
`0b8f3b1a343e1a8fd10e47f47d09dc65cbf9dc43aabb53bbeb3b68c9611edc62`.
The original safety rules are restored; this audit does not establish a
standing exception or permit local model execution.

The separate protected-path metadata repair remains an **uncommitted** local
change in `.codex/hooks/guard-protected-path.sh`, `scripts/check-codex-hooks.sh`
and new `scripts/check-protected-path-metadata.sh`. It is not part of accepted
main. The manager's full hook regression and focused 21-case metadata suite
passed after restoration. The protected CosyVoice2 manifest was not read,
hashed, modified, staged or restored for this documentation task.

## Remaining order

1. Review and accept dependency/reference changes on exact heads, without
   carrying old-lock approvals forward; resolve exact license/native/owner
   and upstream gates. Static closure evidence is now available for this
   candidate, so repeating the same paid audit is not the next step.
2. Only for an execution-approved exact scope, perform a bounded API/reference
   compatibility check; do not silently expand this static authorization.
3. For ready rows, perform real-weight conversion, independent reference and
   CPU parity on VAST, recover evidence, then destroy owned resources.
4. Freeze final clean HEAD and regenerate the exact Apple packet.
5. Run Scaleway Apple CPU/Metal/no-fallback checks, then record results in a
   follow-up PR. Public artifact changes require a separate exact approval.

The [current documentation snapshot](documentation-refresh-2026-10-13.md),
[execution plan](mac-cpu-metal-execution-plan-2026-09-07.md) and
[owner packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md) retain
the catalog and legal completion boundaries.
