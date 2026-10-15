# Candle CPU primary-archive facts — 2026-10-15 JST

This actual source-only collection follows [PR #251's metadata capture](kyutai-candle-source-metadata-ci-2026-10-15.md).
That PR was normally accepted at 06:34:34 UTC as main
`1d86328008c505aeeed3740009faf3558ac7ba37`, after all 71 checks completed
(70 success, one intentional skip), all 16 required context/application pairs
passed and all ten started workflows succeeded. Its exact measured source
head remains `b111f377bfbfc2f8d079f10791d61befb2d3ff5d`.

## Actual collection and independent receipt review

Manager reviews Luna's disposable stdlib collector and independently passes
its nine synthetic tests before the actual collection. The collector is
SHA-256 `bcbfe36db605b1c41bc45d9c83d099e865e2050ab3628c6f73408d0370130b59`;
tests are `dbe872c506e185da278d35e3e61ec5ab33cfbf3ff5fcc4112b7f6a13319de0e7`.
Neither is a production license gate. The fixed hosted input hashes are:

| Input | SHA-256 |
|---|---|
| Initial CPU metadata | `904ba6da53c222ad8feda4c5a82d90dac12b9c84defa6ef081a613e7c9b4b736` |
| Final graph/lock receipt | `99a22ac5c6ab9b34e5b21c7ceee308011ae42bf880d3c711403e9aeec1847a02` |
| Patched lock | `303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01` |

The active CPU closure has 105 packages: one authenticated local Moshi and
104 registry packages. The collector reads the exact post-update closure,
binds every registry name/version/source/checksum to the patched lock and
fetches only `https://static.crates.io/crates/{name}/{name}-{version}.crate`.
The sole initial-to-final identity change is memmap2 0.9.5 to 0.9.11; the new
archive's actual manifest is read instead of inheriting the old declaration.

All **104 complete archive SHA-256 values match**. Collection failures and
missing archive identities are empty. Compressed inputs total **7,110,892
bytes**; sequential, in-memory gzip expansion totals **41,978,368 bytes**.
No archive is saved or extracted. A shared absolute 480-second alarm,
per-file/aggregate bounds, redirect refusal, path/link/device/duplicate
refusals and fresh-output protections remain in force. Root/nested manifest
collisions, declared license paths, missing VCS metadata and partial-failure
preservation are covered by the synthetic tests.

The full local receipt is 651,624 bytes, SHA-256
`b081d421f62cf41c445cf4171ccab875758106a55d629cd44ae32c9e9d972d69`.
The [tracked compact receipt](kyutai-candle-primary-archive-facts-2026-10-15.json)
preserves all 104 archive identities and checksums, original manifest/VCS
receipts, license/source member hashes and unresolved facts. Raw license and
build/proc-macro text bodies remain in the full local receipt, not this compact
file. Manager independently checks the full receipt's 104 unique identities,
all lock/archive bindings, **178 license-member** hashes and lengths, all
source-text hashes and lengths, aggregate byte totals and blocked statuses.
The compact receipt is 239,172 bytes, SHA-256
`74f18fe424932205e769d875739539fcf66a8979eaffe09c52b80e4ec3f6a383`.

There are **65 distinct license text bodies**, totaling 238,078 bytes, and
**24 distinct build/proc-macro entry-point bodies**, totaling 167,023 bytes.
Presence or a matching hash does not mean the entire body or its reachable
code has received legal/native approval.

## Explicit gaps, not inferred approvals

Three archives contain no license-like or declared license-file member:

| Package | Authenticated archive SHA-256 | Published VCS commit/path |
|---|---|---|
| candle-nn 0.9.1 | `c1980d53280c8f9e2c6cbe1785855d7ff8010208b46e21252b978badf13ad69d` | `cd96fa80da255e34f7b16b4ff98b6a31d557201b` / `candle-nn` |
| candle-transformers 0.9.1 | `186cb80045dbe47e0b387ea6d3e906f02fb3056297080d9922984c90e90a72b0` | `cd96fa80da255e34f7b16b4ff98b6a31d557201b` / `candle-transformers` |
| ug 0.4.0 | `90b70b37e9074642bc5f60bb23247fd072a84314ca9e71cdf8527593406a0dd3` | `854edf4f9d97c86723a48ab89362fc1339675180` / `ug-core` |

Their manifests declare `MIT OR Apache-2.0`, but metadata is not a bundled
primary license. At the archive-collection boundary, fixed official-source
primary documents still require a separate check. An absent archive member
does not itself prove absence of a grant; preserve that archive fact.

### Subsequent fixed-source primary supplement

The separate check locates both root license bodies at the exact published
VCS commits. Manager independently reads/hashes Candle's
[Apache body](https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/LICENSE-APACHE)
and [MIT body](https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/LICENSE-MIT).
At that same commit, both package manifests inherit `license.workspace=true`
and the workspace declares version 0.9.1 and `MIT OR Apache-2.0`.

For ug, fixed commit `854edf4f9d97c86723a48ab89362fc1339675180` binds tree
`6605d017f694e2acb48bbc1f0cfe4b1a73a09fd5`. Its non-truncated official Git
tree binds root `LICENSE-APACHE` to blob
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` and `LICENSE-MIT` to blob
`31aa79387f27e730e33d871925e152e35e428031`. The connector's raw-file route
returns 404; manager first reads the exact official Git blobs, then directly
confirms HTTP 200 and complete body equality through the official
[Apache raw source](https://raw.githubusercontent.com/LaurentMazare/ug/854edf4f9d97c86723a48ab89362fc1339675180/LICENSE-APACHE)
and [MIT raw source](https://raw.githubusercontent.com/LaurentMazare/ug/854edf4f9d97c86723a48ab89362fc1339675180/LICENSE-MIT).
Both bodies are byte-identical to the corresponding Candle body. The earlier
GitHub Apache web-page check returned HTTP 500; no link exclusion is added.
The same tree's root manifest and `ug-core/Cargo.toml` establish version
0.4.0 and inherited `MIT OR Apache-2.0`, without guessing a repository root.

| Recovered primary body | Bytes | SHA-256 |
|---|---:|---|
| Apache-2.0, both fixed repositories | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| MIT, both fixed repositories | 1023 | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |

This resolves those three rows' missing **primary-source document** lookup,
not their unchanged missing bundled archive members or the entire dependency
license/native/operator review. The compact archive receipt is not rewritten
to imply that the files were bundled. No owner approval is generated.

Four older VCS JSON records have an unknown `path_in_vcs`: bit-vec 0.6.3,
bitflags 1.3.2, cfg-if 1.0.0 and stable_deref_trait 1.2.0. The receipt records
the gaps instead of inventing paths or aborting unrelated archive collection.

Fourteen packages have active root build entry points: crossbeam-utils,
getrandom, libc, libm, num-traits, paste, proc-macro2, pulp, rayon-core, serde,
serde_json, thiserror, zerocopy and zip. Ten have proc-macro entry points:
bytemuck_derive, displaydoc, num_enum_derive, paste, seq-macro, serde_derive,
thiserror-impl, tracing-attributes, yoke-derive and zerofrom-derive. This is an
entry-point inventory, not a complete transitive build-source or behavior
audit. Included helpers, generated code, subprocesses and actual build/native
outputs require separate review before package execution.

The binary-filename heuristic finds two Windows test DLLs in libloading:
`tests/nagisa32.dll` (3,072 bytes) and `tests/nagisa64.dll` (2,560 bytes).
These are archive facts, not proof about all native contents or a Linux
execution/linkage result. No binary is run or installed.

## License sample independently checked against fixed sources

For the six selected ISC/Unicode packages, full archive hashes agree with the
same patched lock. Manager also reads and hashes these official primary bodies:

| Fixed primary source | Bytes | SHA-256 |
|---|---:|---|
| [libloading ISC LICENSE](https://github.com/nagisa/rust_libloading/blob/fa0694e4ba662768d2b81843bf058f115e67ce10/LICENSE) | 736 | `b29f8b01452350c20dd1af16ef83b598fea3053578ccc1c7a0ef40e57be2620f` |
| [unicode-ident Unicode license](https://github.com/dtolnay/unicode-ident/blob/93ab72c02e41056e63a3414d80878d18f2f7a962/LICENSE-UNICODE) | 1995 | `f7db81051789b729fea528a63ec4c938fdcb93d9d61d97dc8cc2e9df6d47f2a1` |
| [yoke/derive ICU4X license](https://github.com/unicode-org/icu4x/blob/6bd4893cc44c2ca2718de47a119a31cc40045fe5/LICENSE) | 2195 | `f367c1b8e1aa262435251e442901da4607b4650e0e63a026f5044473ecfb90f2` |
| [zerofrom/derive ICU4X license](https://github.com/unicode-org/icu4x/blob/f4290a877dfcb0f87cad6de4abdd65f0cbb33c9c/LICENSE) | 2195 | `f367c1b8e1aa262435251e442901da4607b4650e0e63a026f5044473ecfb90f2` |

These bodies match the authenticated archive members. Unicode-ident retains
`(MIT OR Apache-2.0) AND Unicode-3.0`, not a simplified MIT-only expression.
Copyright/permission retention and other license conditions remain facts for
the exact isolated reference scope. Existing root runtime allowlists or
exceptions for other projects are not automatically extended to Candle.

## Remaining admission and campaign gates

`COMPLETE` in this receipt means **archive collection only**. Approval remains
null, license review `NOT_PERFORMED`, advisory audit `NOT_RUN`, model execution
`NOT_AUTHORIZED`, publication `NO_UPLOAD`. No owner/signature, runtime
dependency policy, model status, parity tolerance or publication gate changes.
The separate exact-source/native/advisory/operator review and independent
full-PCM reference remain necessary before approved real-weight CPU work.

No model/config/tokenizer acquisition or execution, package installation or
compilation, heavy local Cargo, paid worker, Apple run or publication occurs.
The protected CosyVoice2 owner manifest is neither read nor changed. The
metadata-only 136 full / 58 unresolved inventory and frozen 63-row owner
scope remain distinct and unchanged. All non-Apple prerequisites, approved
real-weight CPU measurements and final clean-HEAD Apple packets remain the
campaign objective; this collection does not close it.
