# Candle paste parents and updated dependency sources — 2026-10-15

This record closes the specific missing **source-collection** inputs for the
sixteen resolved normal parents of `paste` and the three precisely updated
registry packages. It does not close license/native/operator decisions,
macro compatibility, a clean advisory audit, real-weight CPU parity or the
original step-4 clean-HEAD/fresh Apple-packet objective. The
[actual candidate audit](kyutai-candle-candidate-advisory-result-2026-10-15.md)
still fails on `paste 1.0.15`; PR #255 stays draft.

## Sixteen exact parent archives, with partial evidence retained

Fixed candidate lock SHA-256 is
`d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8`.
Its authenticated CPU/whole metadata identifies fourteen CPU parents:
`gemm`, `gemm-c32`, `gemm-c64`, `gemm-common`, `gemm-f16`, `gemm-f32` and
`gemm-f64`, each at 0.17.1 and 0.18.2. The unfiltered whole graph additionally
contains `metal` 0.27.0 and 0.29.0. Each has a resolved normal `paste` edge.
The declared `num_enum 0.7.3` dev dependency is not a resolved edge here;
its actual lock/node depends only on `num_enum_derive`. Do not add a
seventeenth patch or infer active reachability from declarations alone.

The source collector authenticates the fixed lock, CPU/whole metadata and
old 104-archive facts. It checks each official release archive against its
lock checksum before gzip/tar inspection. Manager reviews the implementation
and independently passes eighteen offline tests, including a real-parser
sixteen-archive synthetic run, before actual collection.

Actual collection recovers all fourteen CPU parent archives, then stops at
the existing 2 MiB compressed limit for `metal 0.27.0`. The unmodified partial
receipt is **139,310 bytes**, SHA-256
`05a2ba200afac19d2e4ae8404ee43c24a7c3c9df10cb009c92ccba8c3c53404d`.
It remains `INCOMPLETE`, fourteen of sixteen, not a successful full capture.

Two crates.io version-API reads return HTTP 403, and the web reader cannot
open those API resources. Separate successful HEAD requests to the exact
official static archive URLs report 3,028,990 and 3,033,188 bytes respectively.
Those are size declarations until the subsequent full archive authentication.
The reviewed **distinct two-package resume mode** binds exactly these sizes
and the unchanged lock checksums; it does not raise the original default
2 MiB contract, compressed aggregate 8 MiB, expanded 16 MiB per archive /
64 MiB aggregate, retained text 4 MiB or JSON 4 MiB bounds. It authenticates
the old fourteen-file-set receipt and opens only the two missing archive URLs.
Manager independently passes all twenty-two corrected offline tests before
this actual two-archive operation.

The resume succeeds. Its original combined receipt is **218,909 bytes**,
SHA-256 `0ee688bfd89e7e4e792857038b4b3def31a42e88f77b87e6674c619b6fa6fe21`,
with no terminal LF. The required patch editor adds one terminal LF to the
[tracked copy](kyutai-candle-paste-parents-source-2026-10-15.json):
**218,910 bytes**, SHA-256
`234a530ba183472daf0a34444703b934df741a50520f4c48cf421be25ec3ff36`.
The original remains unchanged; the tracked copy is not claimed byte-identical.
All embedded values and retained source/license bodies are unchanged.

Manager independently verifies the whole sixteen-parent identity set,
archive checksum bindings, every retained text body and manifest/original/
license/VCS reference against the complete regular-member inventory, every
Rust source hash reference, all sixteen parsed package/dependency identities,
and exact equality of the first fourteen archive objects to the original
partial receipt. All fourteen normalized gemm manifests match their old
facts' lengths and hashes. Compressed total is **6,211,252 bytes**, expanded
total **8,638,976 bytes**; the text store has **51 bodies / 65,443 UTF-8 bytes**.
No archive is persisted or extracted; only source evidence is retained.

The fourteen gemm manifests declare MIT and include their license member.
Both Metal manifests declare `MIT OR Apache-2.0` and include both bodies.
Metal archives also include example PNGs and precompiled `.metallib` files;
their existence/hashes are recorded, not executed or approved as a pure-Rust
closure. All sixteen VCS metadata bodies are retained, but a publisher's VCS
declaration is not an independent whole-Git-tree identity proof.

### Tool-retention limitation

The first collector's `086f9b8a1ec13cc9519c5c382b666a1cce007069c021426e76c7efdc3a30feac`
hash was reported before execution, but its byte-identical source snapshot
was not recovered after resume preparation. The later private file named
`collect_parents_first.py` has SHA-256
`a64b382dcdead3b2688a927733d273ebdbaa825351439341c988d3e6f9a403df`;
it is **not** accepted as that original snapshot. Do not relabel it or claim
full first-tool reproducibility from its filename. The original partial data
receipt remains exact and independently revalidated as described above.

The accepted resume collector is independently inspected/hashed at
`82cd5f9af498116402683172d1bf7c57133ee25cfece0ed63b6ced7aed29ef1e`;
its twenty-two-test module is
`3c55df8a3ccfdb6b530c6a1fe1e7ddf0fbb201e636d8f48bb01ad683e9fb9987`.
Future actual overlay application must independently authenticate its source
archives and every unchanged source member; this collection and missing
old-tool snapshot are not substituted for that application proof.

After accepting the current code, manager freezes its exact source/test and
both data receipts in a private local evidence repository at commit
`d22ed6c284eba19fdf29fbd086c7a4bb1d5ffb56`. This prevents subsequent edits
from destroying the accepted resume source. It does not recover the missing
old snapshot or imply that this private commit was published to GitHub.

## Updated registry sources: two CPU replacements, one whole-only package

Exact comparison finds only two changes between the old 104 registry CPU
identities and the candidate's 104 registry CPU identities (plus local Moshi):
crossbeam-epoch 0.9.18 → 0.9.20 and rand 0.9.0 → 0.9.3.
Anyhow 1.0.103 is whole-lock-only, not part of that CPU graph. Old archive
facts do not cover these new versions merely because package names match.

The separately reviewed collector authenticates the exact candidate inputs
and the three release archives, with one 120-second deadline, compressed
2 MiB per archive / 6 MiB aggregate, expanded 16 MiB / 48 MiB, source text
256 KiB per body / 4 MiB aggregate and bounded JSON/member counts. Redirects,
unsafe/duplicate paths, links/devices, PAX, overwritten output and inherited
active timers are rejected. Manager independently passes twelve offline
tests, including full three-archive serialization, aggregate-cap failure and
actual short-timer interruption of a blocking synthetic read.

Actual collection succeeds for all three archives. The original raw receipt
is **860,623 bytes**, SHA-256
`6196fd0c4af1dc57434725e76d0fa749a8f5c119a1e8dcfdbe73c0616636a4b5`,
with no terminal LF. The [tracked compact facts](kyutai-candle-updated-source-facts-2026-10-15.json)
are a deliberately distinct derivative: **123,042 bytes**, SHA-256
`a8ddbf72f531f0f3e35b0da072d24e5daa318b2f11c4991c64a7d91107efa302`.
They retain complete member/hash inventories, manifests, license/VCS/build
bodies; other Rust bodies, including the Anyhow probe source described below,
remain in the original private receipt, with their hashes in the compact
record. This is not a restamped raw capture.

| Package / scope | Archive bytes | Rust bodies | Actual build fact |
|---|---:|---:|---|
| crossbeam-epoch 0.9.20 / CPU and whole | 47,545 | 18 | Root build.rs, 443 bytes; captured, not executed. |
| rand 0.9.3 / CPU and whole | 99,786 | 31 | Explicit build=false; no active build root. |
| anyhow 1.0.103 / whole only | 48,746 | 36 | Root build.rs, 6,936 bytes; captured, not executed. |

The three archives total **196,077 compressed / 948,224 expanded bytes**.
The retained source-text counter is **752,823 bytes**. Manager independently
checks all retained manifest/original/Rust/license/VCS/build body lengths and
hashes against their regular-member inventories and the three lock checksums.
All three declare `MIT OR Apache-2.0`; seven license/COPYRIGHT members are
captured. These are material primary facts, not a completed legal/operator
or build-side-effect approval.

The collector SHA-256 is
`f465406a06dfe8e8e4f808d5289ef42332861d3562554777c0238e72d5a08508`;
its test module is
`7ee67add5d8bdc7b5269c54ccc2806519cf2c5b547ab62e8540dc0b8b378ccb8`.
Both captures keep null approval, `NOT_TESTED` compatibility,
`NOT_AUTHORIZED` model execution and `NO_UPLOAD` boundaries.

### Limited source/license and build-side-effect review

An independent reviewer reads the seven license/COPYRIGHT members, manifests,
VCS bodies and the two complete build roots. Crossbeam's MIT notice names
the Crossbeam Project Developers; Rand retains Rand-contributor and Rust
Project attribution. Preserve these notices and both declared alternatives;
this is a limited primary-text review, not a complete closure or owner sign-off.

Manager independently reads both complete build scripts. The 443-byte
Crossbeam root reads `CARGO_CFG_SANITIZE` and emits Cargo cfg/rerun directives;
that root has no filesystem writes, subprocess or native-link directive.
Rand explicitly declares `build = false`. Anyhow's 6,936-byte root is a
different execution boundary: it reads compiler/wrapper/target/encoded-flags
environment variables, creates `OUT_DIR/probe`, may run two compile probes,
runs `rustc --version`, and removes its probe directory. Its probe input is
**`src/nightly.rs`**, not an invented `build/probe.rs` or `called.rs`.
Manager independently reads the exact **1,563-byte** body at SHA-256
`1c0aeadfcdbe22f8f807b185dec1a7448c378bddb37689714944b60473b58049`.
It exercises the unstable generic Error member-access API through
`anyhow_build_probe`. The source is captured and hashed, not compiled.

No explicit network or native-link directive appears in these two reviewed
build roots. This does not make an environment-selected compiler/wrapper
subprocess inert or prove all external dependency reachability. The raw
receipt's `license_review: NOT_PERFORMED` is preserved rather than restamped
as approval. Future executable compatibility work needs the exact applicable
operator/environment gate separately.

Manager freezes the accepted updated-source tool/test and original raw
receipt in a separate private local evidence repository at commit
`a2ece844f8f61a2ac90199ab00513889c9f92aba`. No provider or public upload
is involved in this evidence retention.

## Next exact action, not an execution shortcut

Prepare a reviewable offline manifest-overlay plan for the sixteen parents:
preserve the dependency key `paste`, alias it to actual package `pastey`
at exact 0.2.3, and patch each authenticated parent path independently so
both gemm/Metal versions coexist. Prove the full parsed-manifest delta is
limited to that version/package alias. Keep every numerical Rust source and
every other member as an explicit unchanged-application requirement.

The [pastey primary receipt](kyutai-candle-pastey-source-readback-2026-10-15.md)
does not establish macro equivalence. The next hosted source-only producer
must authenticate the complete applied source tree, exact dependency delta
and CPU/whole metadata before a new unfiltered strict audit can be registered.
Compilation/macro tests, license/native/operator decisions, independent
full-PCM reference and real-weight CPU remain separate gates. No new paid
worker, model/config/tokenizer acquisition, heavy local Cargo, target-package
execution, Scaleway allocation or publication occurs in this work.
