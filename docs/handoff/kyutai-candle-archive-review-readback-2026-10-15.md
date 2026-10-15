# Candle archive license/build review readback — 2026-10-15 JST

> Subsequent collection: the [14-archive supplement](kyutai-candle-build-source-supplement-2026-10-15.md)
> passes the corrected 17-test suite and actual bounded source-only collection.
> All five exact required helper paths are recovered; the prototype defects
> described below retain their pre-execution review boundary, not a live
> collection failure. Helper semantics/advisory/approval and real-weight gates
> remain separate.

This read-only review follows the [actual 104-archive collection](kyutai-candle-primary-archive-facts-2026-10-15.md).
The full input receipt remains 651,624 bytes, SHA-256
`b081d421f62cf41c445cf4171ccab875758106a55d629cd44ae32c9e9d972d69`.
The compact [archive facts](kyutai-candle-primary-archive-facts-2026-10-15.json)
and their collection-time declarations/gaps are not rewritten by this review.
Reviewed source baseline is PR #253 candidate
`92f63d6245761b2954b6954b46342767c156013f`, based on accepted main
`1d86328008c505aeeed3740009faf3558ac7ba37`. PR #253 was normally squash-merged
at 07:07:47 UTC as `5ceb432545cc1d9bdf2ff5c5ff289d683276539a`. All 73 checks
completed (71 success, two intentional skips), all 16 required context/app
pairs passed and all ten started workflows completed successfully, including
Unity packaging's 24 steps. Protection was not bypassed. Local main was
fast-forwarded and clean. This accepts source and audit records, not model,
dependency/license/operator execution or final Apple packet readiness.

## Coverage and independent review boundary

The license reviewer reports reading all 65 unique license bodies and mapping
them to 178 archived member entries, preserving the exact Cargo declarations,
including `AND`, `OR` and historical slash forms. Manager independently proves
that every body hash matches its UTF-8 bytes and maps to at least one original
member: 65 exact hashes, 178 entries and zero orphan bodies. Manager separately
reads the material libm, embedded Unicode, contributor-notice, BSD and Boost
conditions below; this is not a claim that an automated keyword scan or every
package's entire source tree has received legal approval.

The three missing bundled-license rows remain archive facts. The prior
fixed-source supplement establishes separate MIT/Apache primary documents
for candle-nn 0.9.1, candle-transformers 0.9.1 and ug 0.4.0; it does not put
those files into their original archives or approve the complete closure.

## Material primary-text distinctions

Line numbers below are one-based lines of the exact raw archived text bodies,
not Markdown table line numbers or mutable upstream HEADs.

| Text SHA-256 | Package/member and exact finding |
|---|---|
| `3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7` | libm 0.2.11 `LICENSE.txt`: lines 1–20 give the whole-library MIT statement; lines 23–24 distinguish contributor MIT/Apache alternatives. Lines 228–258 retain Rust authorship, musl and CORE-MATH provenance and refer to individual source-file notices. Preserve the Cargo declaration `MIT AND (MIT OR Apache-2.0)`; the outer Apache body alone is not the entire fact. Per-file notice coverage is not proved merely by reading this document. |
| `74db5baf44a41b1000312c673544b3374e4198af5605c7f9080a402cec42cfa3` | regex-syntax 0.8.5 `src/unicode_tables/LICENSE-UNICODE`: lines 27–41 contain a separate 1991–2018 Unicode notice and permission/notice-retention conditions; lines 54–57 restrict promotional use of copyright-holder names. The package-level `MIT OR Apache-2.0` declaration does not by itself describe this additional embedded table document. Do not silently discard it or automatically invent a new SPDX expression. |
| `90eb64f0279b0d9432accfa6023ff803bc4965212383697eee27a0f426d5f8d5` | Rand/rand_chacha/rand_core/rand_distr `COPYRIGHT`: lines 1–4 retain contributor authorship; lines 6–12 point to MIT/Apache files and same-license Rust-derived code. The short notice is not a substitute for the actual license bodies or every per-file exception. |
| `01c266bced4a434da0051174d6bee16a4c82cf634e2679b6155d40d75012390f` | aho-corasick/byteorder/memchr `COPYING`: three lines identify the Unlicense/MIT alternatives, not a complete standalone MIT grant. Separate actual license members remain recorded. |
| `0be96d891d00e0ae0df75d7f3289b12871c000a1f5ac744f3b570768d4bb277c` | num_enum/num_enum_derive `LICENSE-BSD`: lines 4–16 distinguish source/binary notice retention and non-endorsement conditions. Presence of another alternative does not erase this captured document. |
| `c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566` | ryu 1.0.20 `LICENSE-BOOST`: lines 10–15 state the notice-retention requirement and machine-executable-object exception. Preserve the declaration `Apache-2.0 OR BSL-1.0`, not mandatory Boost-only terms or an invented global allowlist exception. |

Previously independently checked ISC and Unicode V3 bodies remain part of the
same isolated reference scope. In particular, unicode-ident's
`(MIT OR Apache-2.0) AND Unicode-3.0` cannot be reduced to MIT-only, and the
four ICU4X-derived crates retain their Unicode document. Root runtime policy,
exceptions for other projects and earlier Torch/reference approvals are not
automatically applied to this Candle scope.

## Build/proc-macro root review and exact remaining source scope

The receipt has 14 active root build entry points and ten proc-macro entry
points, with 24 distinct body hashes. Reading these roots does not prove all
included helpers, generated files, dependency calls or subprocess behavior.
The independent reviewer has now read all 24 bodies. Manager independently
rechecks their receipt ownership and reads the complete 281-line pulp 0.21.4
build script (6,538 bytes, SHA-256
`a4865ce25e92138d34f9b396e1fef2c59a5f733dc1a909dd0b843e3308d2b5f9`).
The separate pulp 0.18.22 row declares `build = false`; a name-only match
would wrongly attribute the newer build script to that older version.

| Exact source root | Static behavior and remaining boundary |
|---|---|
| pulp 0.21.4 `build.rs` | Lines 199–207 choose an x86 register prefix and an `OUT_DIR/x86_64_asm.rs` destination; lines 211–276 construct `global_asm!`, exported assembly routines and `extern "C"` declarations; line 278 writes the generated Rust text. This is native-code generation, not evidence for a pure-Rust/no-native closure. The source is read, not run; generated output and target-dependent compilation are not verified. |
| crossbeam-utils 0.8.21 `build.rs` | Includes root `no_atomic.rs` and `build-common.rs` at lines 17–18. Their bodies were not in the original root-only capture. |
| proc-macro2 1.0.94 / thiserror 1.0.69 `build.rs` | Compile `build/probe.rs` through Rust compiler/wrapper subprocesses and use a probe directory under `OUT_DIR`. These probe sources remain a separate capture/review target. |
| num-traits 0.2.19 `build.rs` | Invokes `autocfg::new().emit_expression_cfg`; the root does not contain autocfg's implementation. |
| getrandom / libc / paste / serde / zerocopy build roots | Environment, compiler-version or platform-dependent subprocess branches are explicit. Reading them does not prove the chosen host path or all external command behavior. |
| Ten proc-macro entry points | Token parsing/generation is visible at the entry layer, but declared module bodies and external helper dependencies must not be inferred from those roots alone. |

A bounded supplemental source-only collector is being prepared for exactly
14 registry identities: crossbeam-utils, proc-macro2, thiserror, autocfg,
bytemuck_derive, displaydoc, num_enum_derive, paste, seq-macro, serde_derive,
thiserror-impl, tracing-attributes, yoke-derive and zerofrom-derive. Versions
and complete-archive checksums must come from the frozen receipt and patched
lock, not a current registry resolution. Collection and its offline tests are
not yet accepted at this snapshot. Manager review found that the first
prototype's deadline context did not actually interrupt a blocking read and
its required-helper presence checks were absent. Actual source collection is
held while those two concrete defects and causal regressions are corrected;
the prototype's nine passing tests do not establish those missing properties.
No complete build/native review or package
execution is inferred from the completed 24-root review.

## Legacy VCS path supplements

A second read-only reviewer checked the official root `Cargo.toml` at each
of the following recorded commits. The package name/version/license tuple
matches the archived manifest semantically, preserving historical
`MIT/Apache-2.0` spelling. The source and published normalized manifest have
different bytes; these lookups do not establish byte identity, a reconstructed
whole source tree or whole-closure provenance approval. The original null
`path_in_vcs` fields remain unchanged.

Manager separately re-fetches and parses all four official fixed manifests,
checks the name/version/license tuples and independently reproduces each Git
blob ID and byte length below. Their SHA-256 values, in table order, are
`7b6c78f666996f4ba6213af4c79f864e9a87e049f93ba59ca668df57e52f2318`,
`b4f5eefea5453292bbe9a91f618666eb346fac484f940bbc53fe927b8c2e0071`,
`90923641c7cd59e9f3df14ab37d39523d4a6e59cf02d7a6aae79abcaca07e3b9`
and `b46a09c5bab6e4239b9b2d4dc17516301eda0ad8af4d0c3d39d8758e9d8a5a47`.

| Registry identity | Official repository / recorded commit | Root Cargo.toml Git blob / bytes |
|---|---|---|
| bit-vec 0.6.3 | contain-rs/bit-vec / `975189772f60f87ee84b77b8cd648beac6b86703` | `98c0bed5995119fd30f58d63ff99a2266e1c711a` / 708 |
| bitflags 1.3.2 | bitflags/bitflags / `ed185cfb1c447c1b4bd6ac021c9ec3bb02c9e2f2` | `be9e05ada4596d05ec13f8212f84a97c7b83fe52` / 1,035 |
| cfg-if 1.0.0 | recorded alexcrichton/cfg-if, canonical API rust-lang/cfg-if / `e60fa1efeab0ec6e90c50d93ec526e1410459c23` | `0a133160982d6f6280556d94008a1e280bb05569` / 792 |
| stable_deref_trait 1.2.0 | Storyyeller/stable_deref_trait / `66f9d8a15b7209c45f58edee6c1b6bb497b7bd31` | `04e30b48280864c0d495969c01f3fefbe3cbc2ea` / 596 |

Primary fixed manifests: [bit-vec](https://raw.githubusercontent.com/contain-rs/bit-vec/975189772f60f87ee84b77b8cd648beac6b86703/Cargo.toml),
[bitflags](https://raw.githubusercontent.com/bitflags/bitflags/ed185cfb1c447c1b4bd6ac021c9ec3bb02c9e2f2/Cargo.toml),
[cfg-if](https://raw.githubusercontent.com/rust-lang/cfg-if/e60fa1efeab0ec6e90c50d93ec526e1410459c23/Cargo.toml)
and [stable_deref_trait](https://raw.githubusercontent.com/Storyyeller/stable_deref_trait/66f9d8a15b7209c45f58edee6c1b6bb497b7bd31/Cargo.toml).
The original collector did not retain `Cargo.toml.orig`; no comparison with
that archive member or archive re-fetch is claimed for these supplements.

## Execution consequence

No package installation/import/compilation, model acquisition/execution,
provider operation or publication is performed by this review. Approval and
signatures are unchanged. The collector's `COMPLETE` remains collection-only;
its `NOT_PERFORMED` license review, `NOT_RUN` advisory audit,
`NOT_AUTHORIZED` model execution and `NO_UPLOAD` are not promoted in place.

Next close the precise remaining license/attribution, build-source/native,
advisory and operator prerequisites and the independent full-PCM reference.
Real-weight CPU and final clean-HEAD Apple packets are still outstanding for
the campaign. No public row, frozen owner scope or numerical tolerance changes.
