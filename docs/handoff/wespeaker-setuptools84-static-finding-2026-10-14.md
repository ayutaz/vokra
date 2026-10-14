# WeSpeaker setuptools 84 static finding — 2026-10-14

> **Source-contract follow-up:** the [fixed-source reconciliation](wespeaker-source-pin-reconciliation-2026-10-14.md)
> identifies incorrect pooling digests and separates namespace/leaf import
> facts from Torch's locked setuptools requirement. The archive and LGPL
> finding below remain unchanged; no omission or execution approval follows.

This factual software-only audit precedes any paid current WeSpeaker closure
audit or reference execution. The source lock remains
`65715f60a9f86aa80792e4a11303bc4eb2067cb4a3cd670b5b6d724d478c9d51`.
No model, installation, target-package import, native payload execution, Cargo,
cloud worker or owner approval was involved. It was Mac-side ZIP inspection,
not Linux installation or API compatibility evidence.

## Exact archive, not a top-level license assumption

One small public software archive was fetched from the exact locked
[setuptools 84.0.0 wheel](https://files.pythonhosted.org/packages/95/9c/c510029fc6ef33a6275cd2c5d3cecd6613dfd6aa401d57c54f1c18852ccf/setuptools-84.0.0-py3-none-any.whl):
818,216 bytes, SHA-256
`51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670`.
Direct HTTPS used no proxy, redirect, credentials, query or fragment, within
the fixed eight-MiB download, 32-MiB uncompressed and 60-second bounds.
The sdist and all other target distributions were not acquired by this audit.

The ZIP has 343 members and 2,773,629 uncompressed bytes, without duplicate,
traversal, symlink or encrypted members. Its top-level RECORD validates every
non-RECORD member; its own entry has the standard blank hash/size. One nested
vendored wheel RECORD refers to missing `setuptools/_vendor/bin/wheel`; this
anomaly remains unresolved rather than being silently treated as clean.

Top-level metadata and LICENSE identify setuptools itself as MIT. However,
the same archive embeds `autocommand 2.2.2`; its actual METADATA declares
LGPLv3 and its actual full LICENSE is GNU LGPL v3. The 7,634-byte license
payload SHA-256 is
`ade78d04982d69972d444a8e14a94f87a2334dd3855cc80348ea8e240aa0df2d`.
Manager independently rehashed both the archive and retained payload, and
streamed that exact member from the verified ZIP to reproduce the license hash.
The generated validation NOTICE also identifies MPL-2.0 and BSD-3-Clause
components. Top-level MIT metadata is not sufficient to classify the entire
dependency as clean under the [project policy](../license-audit.md).

The archive also carries eight Windows `.exe` launchers despite its
`py3-none-any` tag. They were inventoried/hashed, never executed. A report of
shared objects alone must not be confused with complete non-shared executable
or bundled-payload coverage.

## Disposition of this run, not of the public repository

The current 14-row reference closure is not admitted for execution or described
as license-clean. A paid repeat of the same known-blocked configuration is not
the next action. First determine whether a source-supported inference-only
candidate can omit setuptools, or whether a different licensed reference
route is needed. The Qwen omission pattern is a precedent, not proof that
WeSpeaker/TorchAudio behaves correctly without that dependency. Any candidate
requires its own lock/package/hash reconciliation, native review and authorized
model-free API verification; no dependency was changed by this finding.

This does not withdraw or replace `vokra/wespeaker`, does not promote its
corrected sibling's status into the legacy public row, and is not an owner
license decision. Current provenance/attribution, artifact replacement,
independent CPU parity and final Apple gates remain separate.

## Retained small evidence

The wheel, all 17 LICENSE/NOTICE payloads, full byte inventory, METADATA/RECORD
tables and report are retained under
`/private/tmp/vokra-setuptools84-software-audit-AtAWxn`.
Report SHA-256:
`3a05b1f3704ccb8d55593765070c7c5dab85ac71559d9dfe0c5170879dc6aa11`.
These are small software/evidence files, not model caches. No protected
CosyVoice2 manifest was read, hashed, changed or cleaned, and no public
artifact or approval field was updated.
