# MMS current evidence admission readback — 2026-10-15 JST

Read-only investigation at accepted main
`f0130b5de5c37e4441612a85ecaa4e39ae0344f9`. This corrects the
[58-row static prerequisite report](current-58-static-prerequisite-readback-2026-10-15.md)
for MMS: an intentionally omitted wheel-size field is not the current first
gate failure. No model, configuration, tokenizer or vocabulary payload was
acquired, loaded or executed. No cloud worker was created, package installed,
approval granted, receipt restamped or lock modified.

## Current bindings and actual gate outcome

The dedicated project SHA-256 is
`3a03b5297505dfabf1ad72a6827fa76d765fab5e5c008bf20bb759ba307f9baf`;
the lock is
`2552c24935775517139f1235325ffb590a4b04982b79d4efdbc6c042f5d9397a`.
The manifest, dependency evidence and API evidence bind this same lock.
There are 35 full lock records: one virtual project and 34 registry packages.
The dependency evidence has 35 package rows with a matching recomputed digest,
and 34 non-virtual package facts. These are different denominators, not drift.

The lock/project, license gate and both evidence generators are unchanged from
the evidence source head
`de5684a759e84e39aa824ec3ed2cdab620ba4f28`. The source-head difference from
current main is therefore not itself a stale-input binding. Do not compare
Vokra source HEAD with a different repository's upstream/public revision.

Current `validate_dependency_audit(...)` passes, including the omitted-size
receipt path. Normal production admission still exits 2 with:

> dependency/native package review and owner approval are pending; no model
> evidence may be acquired

The pending manifest remains `PENDING_OWNER_APPROVAL`, with model evidence
`BLOCKED_PENDING_AUTHENTICATED_MANIFEST` and publication `NO_UPLOAD`. It has
not supplied approved package/native review rows, an execution scope or an
owner signer. The existing API evidence says `MODEL_FREE_API_VALIDATED` for
official Transformers signatures only; it does not establish checkpoint
loading, forward execution or real-weight parity.

## Why a missing size does not require a lock repair

The exact official artifact is CPU Torch `2.13.0+cpu`:

- URL: <https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl>
- Locked SHA-256: `4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2`.
- Locked upload time: `2026-07-08T19:29:03Z`.
- Existing archive receipt: 191,817,609 bytes and that same SHA-256.

The current license gate explicitly permits a PyTorch CPU index artifact
tuple containing URL, hash and upload time without size. The dependency
auditor then requires separately recovered archive byte-count/hash evidence.
This artifact meets that contract. A new credential-free official HTTP HEAD
returns 200 and `Content-Length: 191817609`; no response body was downloaded.
HEAD is transport metadata, not fresh authentication of the wheel's bytes.

Adding size solely to prettify the lock would change its digest and invalidate
the existing dependency/API/manifest binding. Such a change would require new
exact-input evidence and coordinated review; it is unnecessary to close the
actual current first blocker. Do not purchase another audit to repeat these
already validated facts.

Linux x86_64 scope is unchanged: lock supported/resolution markers and project
environment all select `platform_machine == 'x86_64' and sys_platform == 'linux'`.
No platform expansion or TorchAudio dependency is introduced.

## Verification and next prerequisites

Read-only offline UV Python 3.12 self-tests pass for the license gate,
dependency audit and API inspector. The actual pending production gate
rejects for the owner/native review reason above; whitespace and clean-main
checks pass. Model/package imports and installations were not part of these
local tests. The historical API import happened only in its named VAST record.

Next, review the exact package/native/publisher license facts, authenticate
complete backbone plus one explicitly selected adapter/language vocabulary and
the metadata contract, and obtain the exact external owner decision. Then
prepare authorized real-weight runtime/reference/CPU parity at a reviewed
clean source HEAD. The adapter-only public artifact is not a complete MMS
backbone. MMS remains unresolved; this readback does not advance CPU, Apple or
publication counts and does not generate the final Apple packet.
