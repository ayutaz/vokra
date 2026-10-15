# Candle build-source supplement — 2026-10-15 JST

## Actual bounded collection

After the [license/build root review](kyutai-candle-archive-review-readback-2026-10-15.md),
the manager independently passes all **17 offline tests**, revalidates the
exact 14-identity plan and runs the reviewed collector once. Collection exits
zero with **14/14 archives**, no missing identities and no failures. This is
static software-source inspection, not package installation or execution.
Accepted Vokra main remains `5ceb432545cc1d9bdf2ff5c5ff289d683276539a`.

The full private receipt is **1,392,663 bytes**, SHA-256
`6aa9df96517e8f3ef668ed7449d3c65961b744590ad5c7d6b0b8ea4a979ca5ef`.
Its [compact fact index](kyutai-candle-build-source-supplement-2026-10-15.json)
retains all archive identities/checksums, source paths, lengths/hashes and
body indexes, without publishing the source text bodies. The original
104-archive receipt remains byte-identical at SHA-256
`b081d421f62cf41c445cf4171ccab875758106a55d629cd44ae32c9e9d972d69`.

| Collection measure | Actual value |
|---|---:|
| Complete compressed archive bytes | 359,536 |
| Bounded expanded tar bytes, sequentially inspected | 1,902,080 |
| Regular `.rs` member entries | 218 |
| Unique UTF-8 source bodies | 215 |
| Deduplicated source-body bytes | 1,247,972 |
| Original build/proc-macro roots rebound byte-identically | 14 |

Manager independently checks every complete archive checksum and Cargo.toml
hash against the original receipt, all 218 member body hashes/lengths and
indices, all 215 unique body hashes with no orphan bodies, and the five exact
required helper paths. These are collection/authentication facts. Presence
of test sources does not mean the upstream tests ran.

## Fixed profile and observed safety corrections

The 14 fixed identities are crossbeam-utils 0.8.21, proc-macro2 1.0.94,
thiserror 1.0.69, autocfg 1.4.0, bytemuck_derive 1.9.3, displaydoc 0.2.5,
num_enum_derive 0.7.3, paste 1.0.15, seq-macro 0.3.6, serde_derive 1.0.219,
thiserror-impl 1.0.69, tracing-attributes 0.1.28, yoke-derive 0.7.5 and
zerofrom-derive 0.1.6. Exact registry source and checksum bindings come from
the authenticated patched lock and original full receipt, not a new resolve.

The collector is 19,532 bytes, SHA-256
`72592fa24131869eba41a4cbc9780f784e942e75186d393d52de21a4cc6a351d`;
its tests are 14,522 bytes, SHA-256
`06214f5c5407fa09881919e44a4e8bee659e99cd2dc1e148fbab6c384e9bfae2`.
It verifies the frozen original collector hash before importing first-party
inspection helpers. No upstream crate code is imported or evaluated.

Before actual collection, manager review rejects a no-op deadline guard and
missing required-helper checks, then rejects suffix-only helper matching and
an incorrect crossbeam helper path. The final 17-test suite independently
passes the actual blocking-read interruption/partial-receipt case, exact-root
positive/negative cases, checksum-before-gzip, redirects, identity mutations,
unsafe members, source caps, inherited timer refusal and fresh output rules.
The original nine-test prototype is not substituted for these properties.

Actual collection preserves the reviewed 240-second absolute deadline;
compressed caps are 2 MiB/package and 8 MiB/total, expanded caps 16 MiB/package
and 64 MiB/total, individual text 256 KiB, deduplicated text 2 MiB and JSON
output 4 MiB. Official `static.crates.io` archives are authenticated in memory
before inspection; redirects, unsafe/duplicate/link/device members are
refused. No archive is extracted to disk or retained.

## Exact helper receipt

| Member | Bytes | SHA-256 |
|---|---:|---|
| `crossbeam-utils-0.8.21/no_atomic.rs` | 218 | `fc1baa4489d9842988bacaaa545a7d7d0e2f8b93cfa0b7d1ae31f21256e4cb0a` |
| `crossbeam-utils-0.8.21/build-common.rs` | 482 | `502cb7494549bed6fa10ac7bea36e880eeb60290dc69b679ac5c92b376469562` |
| `proc-macro2-1.0.94/build/probe.rs` | 885 | `df0d73191f20c207bb1051a4944fb6962e1f632d1e0535aba4b995aa7feba8d1` |
| `thiserror-1.0.69/build/probe.rs` | 832 | `2e2198c1aa004aabf8b36cc34e057b4dfce56d23401fe0a7582d8480225817f7` |
| `autocfg-1.4.0/src/lib.rs` | 20,515 | `b40828305eb18bcaad11608fbbf82d97fe381803fbf1b89a1511f0675ba49e6c` |

## Actual helper review and module inventory

An independent reviewer reads the five required helper bodies and autocfg's
three internal modules. Manager independently reads both crossbeam helpers,
both compiler probes, all three autocfg internal modules and the relevant
autocfg constructor/probe/emit paths from the authenticated receipt.

- Crossbeam's 13-line `build-common.rs` rewrites a Linux target's vendor to
  `unknown`; its nine-line `no_atomic.rs` is a target-name table. Neither
  body invokes a process, filesystem or network API.
- The proc-macro2 probe exercises unstable Span/Literal compiler APIs; the
  thiserror probe exercises `Error::provide`. Both use
  `option_env!("RUSTC_BOOTSTRAP")` for their cache key. Their bodies do not
  themselves spawn processes; the already reviewed build scripts compile them.
- Autocfg `src/lib.rs:171–220` reads `OUT_DIR`/`TARGET`, discovers the compiler
  version and probes std/no-std. Lines 274–311 spawn the configured compiler,
  send generated source through stdin and request LLVM IR under `OUT_DIR`.
  Lines 475–480 implement the expression-to-Cargo-cfg path used by num-traits.
  The rustc module binds `RUSTC`, `RUSTC_WRAPPER` and
  `RUSTC_WORKSPACE_WRAPPER`; the version module runs `--version --verbose`.
  Those configured external commands can have side effects. The review does
  not execute or approve them.

Autocfg module SHA-256 values are
`a8a213ddb64a05c1a1af933bcb331a98879e942b167c33d8f94f9f60ebb14e29`
(`src/rustc.rs`, 2,732 bytes),
`4f7d23b36f01c7be1871be86c038d6cb4689e145d67c82d3793690e9aa05b133`
(`src/version.rs`, 2,172 bytes), and
`fd8ff67c64f7cd1b9f81325a81de4baa34c39d6ae298bdb33f9829cc91acac39`
(`src/error.rs`, 1,806 bytes). No explicit network API or native-link
directive appears in these reviewed helper bodies; that is not a whole
dependency-tree or runtime-side-effect guarantee.

The source inventory resolves all explicit module declarations in the ten
proc-macro roots, including serde_derive's `internals/mod.rs` children, to
exact archived Rust members. It does not execute expansion or prove every
macro's semantics/reachability. The reviewer finds no include/path directives
in the archived Rust inventory; non-Rust members are not retained in this
source-only supplement, and external dependency calls remain a separate
source/lock/review boundary. Do not promote inventory coverage to a blanket
no-native or execution verdict.

## Remaining execution boundary

The recovered helper paths and their limited source review close the
specific root-only capture gaps; inventory completeness is not a whole
reachability/native-code audit. The
known pulp 0.21.4 assembly generator is not removed or reclassified as
no-native. A separate current advisory audit of the 161-record patched lock
is still unperformed; the selected 105-package CPU graph is a different
scope, not the whole lock. Root/runtime CI audits do not cover this lock.

Collection status `COMPLETE` is collection-only, with
`SOURCE_ONLY_REVIEW_PENDING`, null approval, `NOT_AUTHORIZED` model execution
and `NO_UPLOAD` unchanged. No upstream compiler/build script/proc-macro/test,
package installation, model/config/tokenizer acquisition, paid VAST/Scaleway
worker, owner signature or publication runs here. Independent full-PCM
reference, authorized real-weight CPU work, all-row non-Apple closure and
fresh final clean-HEAD Apple packets remain required before Scaleway.
