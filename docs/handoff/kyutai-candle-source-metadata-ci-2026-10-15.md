# Kyutai Candle source-metadata CI preparation — 2026-10-15 JST

## Normal acceptance and subsequent static collection — 06:34 UTC

PR #251 is normally squash-merged at 06:34:34 UTC as main
`1d86328008c505aeeed3740009faf3558ac7ba37`, after 70 successful checks,
one intentional skip, all 16 required context/application pairs and all ten
started workflows succeeded at exact `b111f377`. Local main is fast-forwarded
and clean. This accepts the reviewed source correction and actual metadata
diagnostic, not the dependency/native/legal/operator or real-weight gates.
The subsequent [104-archive primary-fact record](kyutai-candle-primary-archive-facts-2026-10-15.md)
advances the separate source review without changing those approval fields.

## Actual semantic capture and precise update — 06:06 UTC

At exact `b111f377bfbfc2f8d079f10791d61befb2d3ff5d`,
[run 38690241603](https://github.com/ayutaz/vokra/actions/runs/38690241603),
job `116274220275` succeeds. All **40 stdlib tests** pass on the hosted
Linux worker in 1.751 seconds. Fixed Rust 1.99.0/Linux checks, actual source
capture, the complete seven-file evidence check and bounded upload all pass.
Actual capture completes at 06:06:47 UTC; ordinary PR CI is a separate gate.
This supersedes the preparation-only pending statements below, not the
earlier failed runs themselves.

Artifact `11951567227` reports 210,734 ZIP bytes and expires at
2026-10-16 06:06:47 UTC. Manager recovers exactly the seven regular members
below, without symlinks, and independently hashes the extracted bytes. The
reported ZIP digest is not claimed as independently verified.

| Member | Bytes | Independently computed SHA-256 |
|---|---:|---|
| `source_receipt.json` | 8320 | `157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec` |
| `source/rust/Cargo.lock` | 150274 | `bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab` |
| `base/Cargo.lock` | 39842 | `eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de` |
| `patched/Cargo.lock` | 39843 | `303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01` |
| `initial_metadata.json` | 898699 | `904ba6da53c222ad8feda4c5a82d90dac12b9c84defa6ef081a613e7c9b4b736` |
| `whole_lock_metadata.json` | 1264493 | `61982123e29c2d51f3147d1b2a4755252da4b040ccae741a42c42d594a220e00` |
| `metadata.json` | 324957 | `99a22ac5c6ab9b34e5b21c7ceee308011ae42bf880d3c711403e9aeec1847a02` |

The first three hashes reproduce all four hosted attempts. Raw metadata
includes run-specific paths, so its whole-file hash is not represented as
cross-run byte-reproducible.

Manager independently revalidates the 592-record original to 161-record base
projection and the precise update. A separate full-record comparison proves
that **only memmap2 version/checksum change**: 0.9.5 to 0.9.11, with the
expected official-registry checksum already recorded below. All other 160
complete package records and root lock fields are exact, not merely
normalized. Initial and post-update CPU graphs are identical after only that
one package-ID relabel: 105 active packages, no GPU feature activation. The
whole metadata command preserves the base lock byte-identically. All four
final lock receipts bind to the recovered files, including the final
projection matching the patched lock.

Final status remains `METADATA_CAPTURED_REVIEW_PENDING`, with null approval,
`NO_UPLOAD`, model execution `NOT_AUTHORIZED`, and license/advisory audits
`NOT_RUN`. This is not upstream compilation/build-script execution, a native
payload/license review, full-PCM reference, model conversion or real-weight
CPU/Metal parity. The selected CPU declaration set includes Unicode/ISC
terms requiring separate primary-source review; metadata declarations alone
do not grant an exception to the runtime dependency policy. No local model,
paid worker, owner decision or publication is introduced.

## Evidence-backed semantic projection correction

The subsequent private candidate changes the production call to supply both
the initial CPU metadata and the separate whole-lock metadata. Without both
proofs, the original two-argument strict record-equality gate is unchanged.
The semantic path is technical validation only and explicitly returns
`SOURCE_ONLY_METADATA_REVIEW_PENDING`, not a license or execution approval.

The new path requires original package-identity containment and exact retained
non-dependency fields, including source, checksum and unknown fields. Cargo
dependency labels must resolve uniquely against their respective complete
locks. Canonical retained edges preserve their original order, without
additions or replacements, and every projected record must match the complete
whole-feature graph's non-dev edges. Package/node identity coverage, manifest
dependency names/sources/kinds/targets and plain-versus-typed edges are checked.
Active CPU packages must be contained and their declaration lists must match
the same whole-graph identities; the 105-package CPU universe is not required
to equal the 161-record whole lock.

Every removed edge needs an unambiguous optional non-dev declaration and an
inactive dependency alias under the whole parent feature closure. Mandatory
declarations, including mandatory/optional mixtures, are refused. Explicit
`dep:` activation, strong dependency-feature activation and implicit optional
aliases are distinguished from weak dependency-feature references. This
checks removals even when the target package remains in the base lock.

Manager independently runs the candidate against the actual recovered
592-record original and 161-record base/whole graph. Six actual-data mutations
are separately rejected for their intended reasons: checksum changes, unknown
fields, missing/duplicate whole nodes, a mandatory cc dependency removal and
cc parallel-feature activation. An AST comparison confirms that existing
helpers other than `capture` and `validate_base_lock` are unchanged, including
the precise memmap2 patch gate, CPU GPU-feature refusal, command allowlist,
source authentication, clean environment, bounds and deadline/process rules.

The regression fixture covers ten source-derived removals across six parents,
including five retained targets. Its small graphs/checksums are synthetic,
not complete registry graphs or package execution. Independent raw comparison
caught a missing hashbrown feature and a shared serde declaration from a
different parent; these are corrected before candidate publication. The
post-update lock-mutation regression now faults only after the fake precise
update and asserts that exact failure, rather than failing at an earlier gate.

Manager independently passes all **40 stdlib tests** on the final reviewed
candidate and confirms every fixture's source-derived features/declarations,
identities and retained-target facts against the recovered original data.
Actionlint and documentation references pass. Collector SHA-256:
`75bdaa3da0ce7a3add28a1018e7d319b48b9f18448d7c3093c55fa94fdb2773f`;
operational tests:
`138e4f885aae677c8929059bfa5cc769f60e79e158c80fd688137b20257a57da`;
workflow contract tests:
`5d65ac89e4e5c2b987051127c1ac74fe1c6c961882d2733c3accd84dba449580`.

The [official registry index at fixed commit `7ddd842c`](https://raw.githubusercontent.com/rust-lang/crates.io-index/7ddd842c05edd884b506c23165a282cf9de6bf65/me/mm/memmap2)
independently lists identical dependency declarations and feature maps for
memmap2 0.9.5 and 0.9.11, with neither version yanked. The old checksum is
`fd3f7eed9d3848f8b98834af67102b720745c4ec028fcd0aa0239277e7de374f`;
the expected new checksum is
`d1219ed1b7f229ee7104d281dd01d6802fe28bb6e95d292942c4daacdeb798c0`.
This supports the planned precise-update shape, not an actual Cargo result,
archive-license review or current advisory audit. No package was executed.

Fresh corrected-head hosted capture and actual precise memmap2 update remain
required. This candidate does not turn any earlier failed diagnostic green,
approve the dependency/native/license/operator closure, or supply full-PCM,
real-weight CPU, Apple or publication evidence. The following sections retain
their original published-head measurement boundaries.

## Actual whole-lock recovery — 05:24 UTC

At exact `072df353bc6f6506144674db50d0160c9df36aee`,
[run 38685726123](https://github.com/ayutaz/vokra/actions/runs/38685726123),
job `116259990428` passed all **31 stdlib tests** on Linux (1.700 seconds),
then actually recovered the complete all-features/no-platform-filter metadata.
Capture still stopped at the unchanged base-lock retained-record gate at
05:24:21 UTC; neither this run nor its missing-evidence check is green.
Bounded upload succeeded. No patched lock or final receipt was produced.

GitHub artifact `11950976136` contains exactly five recovered members and
reports 170,823 ZIP bytes, expiring 2026-10-16 05:24:21 UTC. Manager independently
hashes the extracted members, without claiming independent raw-ZIP hashing:

| Member | SHA-256 |
|---|---|
| Whole-lock metadata, 1,264,493 bytes | `02488f07884e4f01bb8b7c32eacfc0f1324013a95c1e9551a95530511495a9ad` |
| Initial CPU metadata | `48fce36881a0062d06f4e812e407912e65ac6e06bc1947aaba1a56a0c5f87880` |
| Source receipt | `157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec` |
| Original lock | `bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab` |
| Projected base lock | `eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de` |

The last three hashes reproduce all three hosted attempts. Raw initial
metadata includes run-specific filesystem/package paths; its new hash is not
represented as byte-reproducible across runs.

Manager independently parses all **161 whole-feature packages, 161 nodes and
161 base-lock records**. Package identity sets and node IDs match exactly,
without extras or missing records. Every base-lock canonical dependency set
matches the complete graph's non-dev edges: **zero edge differences**.
Before/after lock hashes are identical to the recovered base lock, and the
embedded post-command lock text is byte-identical. The output remains below
the unchanged 2 MiB JSON cap. These checks cover the whole lock, not only the
smaller selected CPU graph.

The six changed parents' whole-feature sets confirm no enabling feature for
the ten removed optional edges. In particular cc has no selected features
and only shlex as a resolved dependency; its parallel-controlled jobserver
and libc edges are absent. The other five parents have the feature sets
listed in the 04:59 record. This is consistent with changing the full source
workspace to its core-only projection and resolving that feature union;
the original workspace's feature graph itself was not executed here.

This now supports designing a canonical-identity/whole-graph projection
validator instead of treating raw dependency-label equality as semantic
identity. Require unchanged non-dependency fields, exact graph coverage,
no new/replaced/reordered edges and proved inactive optional removals. Do
not add a hash waiver or infer package/native/license/advisory/owner approval.
The implementation correction, actual precise memmap2 update, full-PCM
reference and real-weight CPU parity remain separate uncompleted gates.

At the same published head, hosted `documentation-links` job `116259990487`
and `workflow-security` job `116259990321` subsequently pass. The earlier
RMVPE citation failure is not ignored or carried forward as a successful run;
the corrected fixed-tree citation has new actual CI evidence. Other current
CI remains in progress, and the diagnostic's strict-gate failure is retained.

## Separate whole-lock diagnostic preparation

The next reviewed collector has SHA-256
`a33f8f690eea8bcbbfc49e8fe1a78472363c184836edab70b50b048f80f5db6c`.
After recovering the existing CPU metadata and base lock, it adds exactly
`cargo metadata --format-version 1 --all-features`, without a platform filter,
under the same clean Cargo environment and shared deadline. It never builds
or executes a GPU/model. The raw result is a separate seventh evidence file,
`whole_lock_metadata.json`, with `WHOLE_LOCK_METADATA_UNREVIEWED`, null
approval, `NO_UPLOAD`, model/GPU execution `NOT_AUTHORIZED`, null target/filter
and a separate CPU target provenance field. It is not a CPU execution profile.

Only that exact Cargo argv receives a 2 MiB per-stream capture cap; CPU/tree/
probe commands retain 512 KiB. The existing 2 MiB JSON-file cap is unchanged.
The receipt stores before/after lock hashes and bounded post-command lock
text before rejecting any lock mutation. The strict retained-record gate is
still unchanged. The workflow checks/uploads exactly seven literal paths,
without source trees or caches, and retains artifacts for one day.

Manager independently passes **31 stdlib tests**, including a same-argv
non-Cargo capture-bound negative, actual fake-CPU GPU refusal, separate
whole-feature GPU metadata, mutation evidence and the seven-path contract.
Actionlint and workflow hygiene also pass. These synthetic results do not
establish the whole-feature package universe, lock-edge equivalence, actual
output size or any license/advisory/owner/real-weight approval. Inspect the
new actual hosted artifact before choosing a semantic projection correction.

## Actual initial metadata recovery — 04:59 UTC

At exact `265d0e93a8987590f103732e6b9b22b3f63de340`,
[run 38683860254](https://github.com/ayutaz/vokra/actions/runs/38683860254),
job `116254161757` passed all **25 stdlib tests** and the pinned Linux/Rust
checks. Actual capture again stopped at the unchanged retained-record gate
at 04:59:54 UTC. This attempt has new diagnostic evidence, not a successful
projection or memmap2 update. The missing-evidence check remained red and
the bounded upload succeeded.

Artifact `11949397424` is 94,570 ZIP bytes according to GitHub and expires
2026-10-16 04:59:54 UTC. Manager recovered exactly four members: source
receipt, original lock, base lock and the new initial metadata. The first
three independently computed hashes are identical to the 04:38 attempt
below. `initial_metadata.json` is 898,699 bytes, SHA-256
`b16050f02e698918b7724f1362dedaab11eb91edd503f3d4ae9427cd25ad805d`.
Its status remains `INITIAL_METADATA_UNREVIEWED`, approval is null and upload
is `NO_UPLOAD`. GitHub's ZIP digest is not presented as an independently
verified ZIP hash.

The actual raw metadata has **105 packages and 105 resolve nodes**, all in
the inspected root normal/build closure (102 normal IDs and two build IDs;
these sets do not include the root). Its GPU feature list is empty. Those
105 package identities are a subset of the **161 whole-lock records**,
not an equal universe: 56 other lock records include `cc`, `bindgen_cuda`,
`candle-kernels`, `cudarc` and two `metal` versions. These are metadata facts,
not execution, native-payload or license approval. Fourteen selected packages
declare custom-build targets; that is not evidence that build scripts ran.

The [Cargo resolver documentation](https://doc.rust-lang.org/cargo/reference/resolver.html#features)
separates all-workspace feature resolution for lock generation from actual
selected features. The [metadata format](https://doc.rust-lang.org/cargo/commands/cargo-metadata.html#output-format)
separates declared dependencies from resolved edges. Neither permits treating
an inactive CPU edge as automatic proof that every whole-lock prune is valid.

| Changed retained parent | Actual selected features | Removed optional edges |
|---|---|---|
| hashbrown 0.15.2 | none | allocator-api2, equivalent, foldhash |
| toml_datetime 0.6.8 | none | serde |
| toml_edit 0.22.24 | parse | serde, serde_spanned |
| tracing 0.1.41 | attributes, default, std, tracing-attributes | log |
| tracing-core 0.1.33 | once_cell, std | valuable |
| cc 1.2.19 | absent from selected metadata | jobserver, libc; separately checked source declarations |

The first five parents' metadata declares the removed edges optional and
does not resolve them. Manager separately fetched the official
[cc 1.2.19 source archive](https://static.crates.io/crates/cc/cc-1.2.19.crate)
into memory without installation or execution. Its 105,910 bytes match the
original/base lock checksum
`8e3a13707ac958681c13b39b458c073d0d9bc8a22cb1b2f4c8e55eb72c13f362`.
The normalized Cargo.toml is 1,608 bytes, SHA-256
`cb2f267dc64e6a61ce0fc54411607d3014dc6de5dfc6edfb7cd5ec7cc6354e18`;
Cargo.toml.orig is 1,537 bytes, SHA-256
`90621aee01724b41779b87a2edd3355622e388a9d95d82e1428e15e5b37cdfa1`.
Both declare optional jobserver and Unix libc under the parallel feature.
Their license declarations do not approve this complete reference closure.

Five of the ten removed canonical edges target packages **still retained**
in the base lock: cc to libc, hashbrown to equivalent, toml_datetime and
toml_edit to serde, and tracing to log. A proposed filter that removes only
edges targeting absent packages is therefore rejected, as is equating the
metadata package universe with the entire lock. The strict gate remains
unchanged while the technical projection validation is designed and reviewed.
Patched-lock evidence, native/license/advisory/operator closure, independent
full-PCM reference and real-weight CPU parity remain missing.

## Actual hosted diagnostic — 04:38 UTC

At exact `8fd71d676f5a3f9263e83628c78a04cfe2b90621`,
[run 38681926040](https://github.com/ayutaz/vokra/actions/runs/38681926040),
job `116248167505` ran from 04:38:11 to 04:38:33 UTC. Checkout, interpreter,
all **23 stdlib tests**, pinned Rust and host verification passed. Actual
metadata capture failed at the unchanged retained-record gate with
`projected base lock mutated retained package: backtrace`. This is one
actual source attempt, separate from synthetic rejection messages in tests.
The missing-evidence check also failed as intended; bounded upload succeeded.
No patched-lock or final metadata receipt was produced, and no source/model
gate is accepted by this failed run.

Artifact `11948213954` is 47,201 ZIP bytes according to GitHub, with one-day
retention. Manager recovers exactly the source receipt, original Cargo.lock
and projected base Cargo.lock; there is no model payload. The original
150,274-byte lock independently matches primary Git blob
`70182edd0eb78002fac85d824c6c35b558328b0d`. The projected lock is 39,842 bytes.

| Recovered member | Independently computed SHA-256 |
|---|---|
| Source receipt | `157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec` |
| Original lock | `bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab` |
| Projected base lock | `eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de` |

Manager parses all records: 592 original and 161 projected, with no added
package identities. All version/source/checksum and other non-dependency
fields of retained records match. Exactly 45 records differ in dependencies:
39 resolve to identical canonical package identities, while six actually
remove edges. Those six are cc 1.2.19 (jobserver/libc), hashbrown 0.15.2
(allocator-api2/equivalent/foldhash), toml_datetime 0.6.8 (serde), toml_edit
0.22.24 (serde/serde_spanned), tracing 0.1.41 (log) and tracing-core 0.1.33
(valuable). None is silently dismissed as display-only or approved pruning.

Official fixed-version registry JSON for all six matches the recovered
package checksums and supplies feature declarations. Declarations alone do
not establish the actual activated feature set. The initial Cargo metadata
was inspected in memory but not recovered before this failure. The next
bounded diagnostic preserves it before the same strict rejection, without
extra Cargo commands, an equality waiver or a model run. That candidate is
not yet actual hosted evidence. The preparation snapshot below remains
dated history; its not-yet-run statements are superseded only by this record.

## Evidence-preservation correction — reviewed candidate

The corrected collector has SHA-256
`dd1503f496a0e4c9cade24c4177577bed2013a4910974a043da95f799d760fe5`.
Before the unchanged base-lock equality gate, it saves the already obtained
Cargo metadata object and inspected active graph as `initial_metadata.json`.
The receipt is explicitly `INITIAL_METADATA_UNREVIEWED`, with null approval,
`NO_UPLOAD` and model execution `NOT_AUTHORIZED`. The existing JSON size cap,
Cargo command allowlist, fixed source/toolchain and shared deadline remain
unchanged. This adds no Cargo command or permission to accept pruned edges.

The workflow checks and always recovers exactly six bounded paths: the
original five plus this initial metadata receipt. One-day retention and
nonzero capture/missing-evidence failures are unchanged. Manager independently
passes all **25 stdlib tests**, including failure-time evidence recovery and
the exact six-path contract. Those fake-process tests do not establish a
real active feature graph. Actual hosted evidence for this correction is
still pending; do not relabel the original failed run as successful.
Actionlint, workflow hygiene (49 workflows), documentation references and
`git diff --check` also pass for the corrected candidate.

## Original preparation snapshot

Prepared against accepted main
`8bd30d9ab3857f77f3d0fa1edce34c450ea8c102`. This moves the previously
reviewed model-free diagnostic collector onto an ordinary GitHub Ubuntu
runner, not onto the maintainer Mac and not onto another paid VAST worker.
It is not a new reference implementation or an execution approval.

## Why a bounded source diagnostic is next

The [actual source-only VAST attempt](kyutai-candle-reference-readback-2026-10-14.md#actual-source-only-capture-and-retained-lock-rejection)
stopped at `projected base lock mutated retained package: backtrace`.
Its worker/data were destroyed. The actual changed fields were not
recovered; dependency-display relabeling remains a hypothesis, not a cause
established by that log. Repeating a paid lease or weakening record equality
is not the chosen remedy.

The promoted collector is byte-identical to the reviewed diagnostic candidate:
SHA-256 `8bc8fbf2be559264edc67bb49da44288d54c3c43d54cd44c4ccdbbc4cc94bc3d`.
It authenticates 23 fixed official Moshi Rust source files, retains original
lock bytes and rejects changed retained records. Cargo commands are limited
to metadata, dependency trees and the precise memmap2 candidate update.
No upstream compilation, build-script/model execution, model/config/tokenizer
acquisition, license/advisory approval or independent PCM parity is included.

## Workflow and independent verification

The path-restricted workflow uses Ubuntu, read-only contents permission,
pinned actions, Rust 1.99.0 and Python 3.12. Checkout is non-cone sparse with
exactly the workflow and four owned Python files, with credentials not
persisted. Manager read the pinned checkout implementation and confirmed
that sparse checkout selects its internal `blob:none` fetch when the filter
input is absent. Model paths and the protected owner manifest are excluded.

Capture has a shared 480-second deadline and the job a 15-minute timeout.
The collector, not the workflow, creates its fresh owned output directory.
Nonzero capture status remains a failure. Even on failure, upload selects
only the five bounded receipt/lock/metadata paths, never the Cargo cache or
whole source tree, with one-day artifact retention. Missing evidence is not
silently accepted. No provider, HF or project secret is passed to capture.

Manager independently passes all 23 stdlib tests (14 original collector,
five lock diagnostics, four workflow/entrypoint contracts), actionlint and
diff hygiene. The only original operational-test change is a canonical
portable temporary parent instead of Mac-only `/private/tmp`; diagnostic
tests and the collector are otherwise byte-identical. Synthetic main-entry
success is not an actual source graph or successful patched-lock receipt.

## Remaining acceptance boundary

Actual hosted capture is not yet observed. Inspect its bounded field
diagnostics and recovered locks before choosing a correction. Preserve
checksum, source, package/dependency identity and fail-closed review. Then
complete active CPU graph, native/license/advisory and full-PCM/API contracts,
exact owner decisions and authorized real-weight CPU parity. Draft #191's
scope does not approve this distinct Candle reference scope. The 58 public
unresolved rows, frozen 63-row owner scope and final clean-HEAD/Apple packets
remain incomplete; no model acquisition, publication or Scaleway follows.
