# Candle paste replacement: primary source facts — 2026-10-15

This is a source-collection and limited license-text readback at accepted
main `7701cc67414d9cb84e4fe3cd6cf228a10bb8b792`. It does not replace the
[actual five-finding advisory result](kyutai-candle-source-advisory-result-2026-10-15.md),
approve a dependency alias, or establish macro/build/numerical compatibility.
PR #255 remains draft. The full non-Apple campaign, independent real-weight
CPU verification, final clean HEAD and fresh Apple packets remain unfinished.

## Actual collection and independent verification

A bounded, opt-in stdlib collector reads exactly the official
[pastey 0.2.3 version metadata](https://crates.io/api/v1/crates/pastey/0.2.3)
and [release archive](https://static.crates.io/crates/pastey/pastey-0.2.3.crate).
Its two source/test files remain private diagnostic tooling, not installed
dependencies. Manager reviews both files, independently passes all eight
offline tests, then runs the source-only collection. The first sandboxed
attempt exits 2 at DNS resolution; a separately authorized network-enabled
attempt succeeds. The failed receipt is retained separately, not rewritten.

The original complete recovered JSON is **176,563 bytes**, SHA-256
`66d2def8d41978ce93d07692c008cb593b730242550a51b91681d95d712b2c88`.
It has no terminal newline and remains unchanged in the private evidence
directory. The [tracked JSON copy](kyutai-candle-pastey-source-2026-10-15.json)
adds exactly one terminal LF through the required patch editor: **176,564
bytes**, SHA-256
`d4ca85e59f16d9ae6dc36f406473b6c534fe86895c353c3aca18930bce6edba2`.
Manager verifies this exact one-byte relationship; JSON values and every
embedded API/source/license body are unchanged. It is not a byte-identical
copy or a rewritten success receipt.
Manager independently recomputes all thirteen retained member hashes and byte
counts, totaling **86,484 UTF-8 bytes**, and the raw API response hash/length.
It reads both license texts in full. The separate initial DNS-failure receipt
has SHA-256 `692c3566eb1d5dc7ab88d9f7c66c1125422f0efd88f5b81eb16603e7f523f723`.

The collector is SHA-256
`7a478a12f1cfdf012bfb95cd5b1561bf33c1654d650382a4fa7b14d357cbd5d6`;
its eight-test module is
`cb902c712bd4a8a5ecf6eec709f3392d451acf95f57ceb3d94de55ec6ec6dc34`.
It authenticates and reuses the earlier source supplement's reviewed
deadline/bounded-read/path/fresh-output helpers at SHA-256
`72592fa24131869eba41a4cbc9780f784e942e75186d393d52de21a4cc6a351d`.
Limits are one shared 60-second absolute deadline, 64 KiB metadata,
2 MiB compressed archive, 16 MiB expanded archive, 128 members,
256 KiB per text member and 4 MiB aggregate retained text. Redirects,
unsafe/duplicate paths, links, devices and PAX metadata are refused.
Decompression is bounded before retaining expanded bytes. No archive is
written or extracted, and no target package is imported or executed.

## Authenticated package and license facts

| Item | Actual result |
|---|---|
| Version metadata | 1,560 bytes; SHA-256 `bbcb63b609e5094eeb99e36c68ae0ab036b60d6f1ef9f5e294da624c17f18255` |
| Compressed release | 20,361 bytes; SHA-256/checksum `2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4` |
| Expanded release | 97,280 bytes; SHA-256 `e22aa3b6e728ba9b4c093fc69a63168ee2baec5debc8adf6db237f222e118999` |
| Package declaration | `pastey` / `0.2.3`, `MIT OR Apache-2.0`, Rust 1.54, edition 2018; API reports not yanked |
| Normalized manifest | `proc-macro = true`, `build = false`; no ordinary dependency or feature table |
| VCS declaration | commit `c377d07c9d1a418e9922af59dc932736752a95cc`, empty `path_in_vcs`; publisher declaration, not an independent whole-tree Git audit |
| MIT license member | 1,023 bytes; SHA-256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| Apache license member | 9,723 bytes; SHA-256 `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |

The two license members contain the MIT permission/notice/warranty terms and
Apache-2.0 sections 1–9, respectively. The exact thirteen-member archive has
no separate NOTICE, COPYING or COPYRIGHT member. This closes the narrower
question of actual bundled license bytes for this candidate; it is not an
owner/operator decision or a finding about the complete Candle closure.

The archive contains five Rust source members, manifests, a lockfile,
README/changelog, VCS metadata and the two licenses. It has no binary payload
or build script in that exact member set. A Rust proc-macro still becomes
executable compiler code when built; source collection does not approve that
execution. The original `paste 1.0.15` finding remains unresolved.

The fixed upstream [manifest](https://github.com/as1100k/pastey/blob/c377d07c9d1a418e9922af59dc932736752a95cc/Cargo.toml)
and [migration instructions](https://github.com/as1100k/pastey/blob/c377d07c9d1a418e9922af59dc932736752a95cc/README.md)
corroborate the package declaration and describe a dependency alias. Those
instructions are not proof that every transitive Candle parent is compatible.

## Next gate, not an acceptance shortcut

The existing graph investigation identifies fourteen CPU normal-dependency
parents of `paste`, and sixteen whole-lock normal-dependency parents including
two Metal versions. A root direct-dependency alias is not a completed change
to those parents. Prepare and review any exact parent-manifest source patch,
authenticate all affected archive/source/license identities, prove ordinary
dependency resolution and macro/API compatibility, and rerun the strict
whole-lock advisory audit. Do not disguise the unmaintained package by a
name/version-only change or waive its warning.

The recovered JSON intentionally retains `collection_only: true`, null
approval, `license_review: NOT_PERFORMED`, `compatibility: NOT_TESTED`,
`model_execution: NOT_AUTHORIZED` and `upload: NO_UPLOAD`. This dated manager
readback adds only the limited text inspection above; it does not restamp
the original collector receipt. No model, compiler, Cargo command, package
installation, provider credential, paid worker or publication is used.
