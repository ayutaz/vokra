# XCodec2 dependency-license disposition — 2026-10-04

## Scope and decision

This record belongs to existing draft PR #152. It separates archive facts,
dependency remediation, and approval; it creates no new model-support or
publication claim. Main baseline:
`4d753d81b084f6d2369e330656035042fd00fb9f`. Archive/test candidate:
`a7bb24cbd7896d770d4da375febc3669b768ae9f` (A7).
The A7 lock SHA-256 is
`2ffd17d936fef2b660e3865e3d8cdecbde24f48ae7537254b1d78c9f2c77b220`;
older import receipts with a different lock retain their historical identity.

Disposition: **Draft / ARCHIVE_ONLY / BLOCKED_OWNER_REVIEW / NO_UPLOAD**.
GPL/LGPL policy conflicts are not resolved by a generic owner signature,
metadata-only license classification, model-free CI, or a no-upload label.
The dependency execution contract remains fail-closed. No owner/legal approval
has been added and no runtime dependency, numerical bound, source pin, model
artifact, or public catalog status is changed by this record.

## Evidence chain and limits

V5 at `9ff0cb4b423d98c8ac4e1181cabaf00bad225a08` collected 61 of 62 package
rows. Its Torch archive hit the 10,000-member guard; this was not evidence
that the archive actually contained duplicate members. The V6 supplement
authenticated the exact locked Torch wheel and found 12,248 members with
zero duplicates. Only that wheel used a hash-bound cap of 12,504; global and
other-package bounds were unchanged.

V6 ran on clean A7 with all 13 reviewed source/contract hashes unchanged
before and after. It retained the original 61 package rows exactly and added
the Torch row. `complete=true` means archive factual collection reached
62/62, **not** that missing primary licenses, ambiguous artifact selections,
installed/build bindings, or owner review have been resolved.

| Recovered small receipt | Bytes | SHA-256 |
| --- | ---: | --- |
| V5 `report.json` | 1,153,450 | `fb039b48d91bf9535c3341fe5530d496e3c6967095cc285a961c01b485eba775` |
| V6 `merged-report.json` | 1,648,939 | `fd90d688d043b396f751bae0231569fdcc6a9bb3f6c20ddd4ab7d057a3f2b45e` |
| V6 `torch-supplement.json` | 497,669 | `9f9810487f3112c90d36c08466286c390c5d9a3ad40d0663d4239429d53d0f14` |
| V6 `verification.json` | 4,239 | `ea3cf9333e74f23cb72e631ae852268a39b18a8710da3371057e9003a8142183` |

All nine recovered JSON/log sidecars were independently checked. The raw
receipts are retained in the local evidence archive, not committed as model
fixtures or advertised as available from a clean public checkout. This
record identifies them by content hash rather than claiming a repository
path for untracked evidence.

Remote tooling verification: 35 stdlib tests, zero failures/skips, and five
helper self-tests passed. No third-party package was installed or imported;
no native payload/model/weight/audio was executed, no Cargo was invoked, and
no HF token or artifact upload was involved. This is not workspace, parity,
or Apple verification. Later documentation commits do not acquire A7's
exact-HEAD verdict merely by citing it.

Owned VAST workers `54134969` (V5) and `54137450` (V6), including their stored
data, were destroyed after evidence recovery. Exact-ID readbacks returned
null and independent inventory reads confirmed absence. Other projects'
instances were not modified; global account emptiness is not claimed.

## Confirmed primary-license conflicts

| Component | Primary member | SHA-256 | Result |
| --- | --- | --- | --- |
| frozendict 2.4.7 | `frozendict-2.4.7.dist-info/licenses/LICENSE.txt` | `e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118` | LGPLv3; blocked by repository policy |
| setuptools 84 / autocommand 2.2.2 | `setuptools/_vendor/autocommand-2.2.2.dist-info/LICENSE` | `ade78d04982d69972d444a8e14a94f87a2334dd3855cc80348ea8e240aa0df2d` | LGPLv3 vendor payload; top-level MIT is insufficient |
| NumPy 2.0.2 | `numpy-2.0.2.dist-info/LICENSE.txt` | `493c7996721d9206971883292936f1323d3a30b52537d15284fb2da4ad58e13b` | Bundled libgfortran GPL/GCC-exception and libquadmath LGPL notices; no policy waiver inferred |

These are repository-policy conflicts, not a finding that every private use
violates the law. The controlling project rules are
[CONTRIBUTING section 3](../../CONTRIBUTING.md#3-dependency-license-policy)
and the [license audit](../license-audit.md). Existing scoped precedents do
not constitute a blanket GPL/LGPL exception for this new candidate closure.

## Torch native finding versus generic notices

Exact locked wheel: Torch `2.13.0+cpu`, CPython 3.12, Linux x86_64,
191,817,609 bytes; SHA-256
`4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2`.
It contains one METADATA, 98 LICENSE/NOTICE members, and 12 native ELF entries.

The archive contains `torch/lib/libgomp.so.1`: 253,849 bytes, SHA-256
`78511033caddec6ccae8f4d62b94d56135f377c4cee33120e6d7df2ef499f69f`.
Readelf `NEEDED` entries bind `libtorch_cpu.so`, `libshm.so`, and
`libtorch_global_deps.so` to that library name. No separately identified
libgomp license member was retained. The [official GCC libgomp copying
documentation](https://gcc.gnu.org/onlinedocs/libgomp/Copying.html) identifies
GPL terms, but does not authenticate this wheel binary's source revision,
build, or applicable exception. That exact binding remains **UNRESOLVED**;
neither a clean approval nor an exception-free legal conclusion is asserted.

NVTX, kineto's cpr test subtree, and llvm-openmp have GPL text in retained
third-party notices. None has been mapped to a selected native ELF payload
in this receipt. Generic license text alone is not labeled proof of runtime
component inclusion. Conversely, the actual bundled libgomp cannot be
cleared by ignoring the generic notices or reading Torch's metadata alone.

## Remediation feasibility and rejected shortcuts

| Dependency | Actual boundary | Next acceptable evidence |
| --- | --- | --- |
| frozendict | `xcodec2 -> vector-quantize-pytorch -> einx -> frozendict`; official decoder imports and constructs `ResidualFSQ` | An authenticated allowed-license upstream path retaining the independent official oracle, or an explicit project-policy decision; none established |
| setuptools | Required by the locked Torch closure even without a direct project edge | Exact replacement distribution, full external dependency/license closure, source/build/RECORD identity and compatibility; current de-vendoring is `CANDIDATE_NOT_BUILT` |
| NumPy | Current Linux PyPI wheel bundles Fortran runtimes | Pinned source build, artifact hash, build flags, RECORD and ELF/system-library closure audit; currently an unbuilt derived candidate |
| Torch/libgomp | Actual native library and `NEEDED` bindings, but source/license/exception identity unresolved | Primary source/build/notice evidence tied to the exact bundled bytes and project-policy disposition |

The official [pinned XCodec2 decoder](https://huggingface.co/HKUSTAudio/xcodec2/blob/e9463f16b1a4af077e9d96c06ae99beb8639c1ee/vq/codec_decoder_vocos.py)
uses `ResidualFSQ`; [einx 0.4.3 publisher metadata](https://pypi.org/pypi/einx/0.4.3/json)
declares frozendict. Reviewed older einx releases also retain this dependency;
no clean official replacement was found in this investigation. Replacing
FSQ with our equations would change the oracle and is not accepted as
independent reference evidence. Removing `Requires-Dist`, using `--no-deps`,
or deleting the lock edge cannot authenticate the official dependency path.

The [setuptools downstream de-vendoring guidance](https://setuptools.pypa.io/en/latest/history.html#v71-0-0)
is a candidate packaging strategy, not proof that a reviewed wheel exists.
NumPy's [official BLAS/LAPACK build options](https://numpy.org/doc/stable/building/blas_lapack.html)
make a different build technically conceivable; they do not prove that
system-library terms are acceptable or that this lock has been repaired.
Neither candidate should incur a build worker merely to produce more
artifacts while the required frozendict path remains unresolved. Reverting
to a vulnerable old Torch pair is not a license/security remedy.

## Remaining work and acceptance boundary

### Existing-PR CI rate-limit remediation

At pushed head `5a26355ed2e621fc9bfe1c73ec2645baaabe24b4`, CI Security run
`37196989338`, documentation-links job `111420816168`, failed with nine
Hugging Face HTTP 429 responses and zero timeouts (847 total links, 385
unique). The other two security jobs passed. This receipt does not show
that those URLs are missing, nor does it prove why the host rate-limited
this runner.

The follow-up workflow bounds global concurrency to 32, per-host concurrency
to two, and per-host request spacing to one second; the retry wait is five
seconds. These are supported by the action's pinned lychee v0.24.2 CLI.
The three retries, 20-second timeout, input scope, existing exclusions,
action pin, and fail-closed result are unchanged. No 429 acceptance, new
exclusion, or blanket CI rerun is used. Local actionlint and workflow-hygiene
checks cover this configuration change; the new-head remote link verdict
remains to be observed. CI remediation does not resolve the license conflicts
or authorize model execution, publication, or merge.

At follow-up head `4fa38735603d6657253b423bebcdf2ed5b55eda0`, run
`37198249231`, job `111424459319`, reported 847 total / 385 unique links,
819 successful checks, zero timeouts and one HTTP 404. No HTTP 429 remained
in that run; this single observation does not prove permanent rate-limit
resolution. The remaining 404 was the decoder citation above: its revision
had an extra `c`, producing 41 characters rather than the publisher's
40-character Git revision. The citation now uses the authenticated publisher
page. This corrects only the documentation URL, not a model/source execution
pin, approval, license classification, or numerical contract. Fresh CI must
verify the corrected documentation head.

### Dependency and model gates

1. Resolve the required frozendict path and Torch/libgomp primary binding
   without changing the independent-oracle or security contract.
2. If a permissible closure is viable, authenticate/build its exact artifacts
   on bounded VAST capacity and re-audit all changed transitive/native edges.
3. Recover missing bounded primary licenses for ANTLR 4.9.3, sentencepiece
   0.2.2, and XCodec2 0.1.5; bind the locked sdists to reviewed builds.
   Resolve the six multi-compatible-archive selections and scoped MPL reviews.
4. Verify installed RECORD/runtime/API compatibility and obtain hash-bound
   scope-specific owner/legal decisions before any third-party model import.
5. Run independent real-weight reference and native CPU parity on VAST.
   Apple CPU/Metal/no-fallback validation remains a separate hardware gate.

All of these remain incomplete. Source hardening may be reviewed and updated
in the existing draft PR, but this record does not make its dependency
candidate merge-ready, mark Mac/Metal support complete, authorize public
artifact replacement, or close security alerts.

## Exact-pin remediation readback — 2026-10-04

The existing PR API readback at
`1e3f0385cb21a96ef011781e1a69da9dd1b0f322` reports OPEN / Draft / CLEAN,
with 69 successful checks and one skipped check. This supersedes the pending
CI observation above only for that head; it does not clear dependency or
execution gates. The existing PR body also records the findings below.

The [publisher's XCodec2 0.1.5 metadata](https://pypi.org/pypi/xcodec2/0.1.5/json)
requires `vector-quantize-pytorch==1.17.8`, not an open older-version range.
Its sole published archive is `xcodec2-0.1.5.tar.gz`, 22,329 bytes, SHA-256
`dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b`.
The [official 1.17.8 ResidualFSQ source](https://github.com/lucidrains/vector-quantize-pytorch/blob/1.17.8/vector_quantize_pytorch/residual_fsq.py)
imports `einx.get_at`. A pre-einx release such as 1.12.12 is therefore an
out-of-contract, unbuilt investigation candidate, not an approved replacement
or proof of decoder API compatibility. Do not repair the audit by substituting
our own quantizer or removing the official dependency declaration.

For the exact reviewed Torch release, the
[v2.13.0 CMake options](https://github.com/pytorch/pytorch/blob/v2.13.0/CMakeLists.txt)
default `USE_OPENMP` to ON, and the
[same-tag dependency configuration](https://github.com/pytorch/pytorch/blob/v2.13.0/cmake/Dependencies.cmake)
conditionally links the OpenMP target. An OpenMP-disabled source build is a
technical candidate only: no such reviewed artifact was built or shown to
have an acceptable full native/system-library closure. It does not identify
the source/build/exception applicable to the actual bundled `libgomp` bytes.
No manual wheel stripping, affected-version downgrade, owner policy waiver,
dependency installation, model execution, public upload, or merge is implied.

The next acceptable action remains resolving the official quantizer path and
exact native/license binding, followed by artifact/RECORD/API review if a
permissible closure becomes viable. Existing GPL/LGPL conflicts and missing
primary licenses remain open; green model-free CI is not their remedy.

## Primary-source license supplement — 2026-10-04

Read-only publisher-source review recovered two versioned license texts.
GitHub tag readbacks resolve ANTLR `4.9.3` to
`e4c1a74c66bd5290364ea2b36c97cd724b247357` and SentencePiece `v0.2.2` to
`e0cce7d37b065b5140349dbe12c6bcf6192fdd78`. Their exact source files are:

| Primary source | Bytes / Git blob | Text SHA-256 | Observed terms |
| --- | --- | --- | --- |
| [ANTLR LICENSE.txt](https://github.com/antlr/antlr4/blob/e4c1a74c66bd5290364ea2b36c97cd724b247357/LICENSE.txt) | 2,699 / `2042d1bda6c933e504d9dc2fe3197a6e42a71fe2` | `b1b379fcaf3219593a4c433feb1b35c780bed23fafaae440b1ae2771a9521e3a` | BSD-3-Clause plus named JavaScript MIT notices |
| [SentencePiece LICENSE](https://github.com/google/sentencepiece/blob/e0cce7d37b065b5140349dbe12c6bcf6192fdd78/LICENSE) | 11,358 / `d645695673349e3947e8e5ae42332d0ac3164cd7` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Apache-2.0 |

This is **SOURCE_LICENSE_LOCATED / ARCHIVE_BINDING_UNPROVEN**, not a
replacement for the immutable V6 report. That report's missing-member facts
remain unchanged. The locked ANTLR sdist (`f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b`)
and selected SentencePiece wheel (`c8a168b040bc61681293f79a949b5d911c8e25086f4260285b8d97ab5f1195da`)
still need exact source/build/member/RECORD bindings and any applicable
vendored/native notice review. A source tag is not proof of a wheel's full
contents or of installed runtime provenance.

XCodec2 0.1.5's primary package-license binding remains unresolved; its PyPI
metadata or model-weight license is not substituted for that binding. No
archive was downloaded, dependency installed, owner sign-off added, native
payload executed, or model imported for this supplement. The policy conflicts,
secure dependency closure, real-reference gates and Draft / NO_UPLOAD posture
remain unchanged.

## Original publisher repository license readback — 2026-10-04

The publisher's [model card](https://huggingface.co/HKUSTAudio/xcodec2)
links the original [X-Codec-2.0 repository](https://github.com/zhenye234/X-Codec-2.0).
A read-only GitHub API check resolves its current main to
`e5d3b2601146b20da39fc2f7c5a1db1418292138`, with tree
`0d499b388ac42b66b5d5fcf38886f194e32e1614`. The exact
[LICENSE at that revision](https://github.com/zhenye234/X-Codec-2.0/blob/e5d3b2601146b20da39fc2f7c5a1db1418292138/LICENSE)
is MIT, with the named 2025 copyright holder: 1,064 bytes, Git blob
`ec2d7a448d3294011ea7cab32ec5aa100d1041c0`, SHA-256
`bc4f68c65be9d49d804447f4567368a6b6d25ba92d9c711621271f7984074f89`.
The file's history reports creation at
`c1ed8bf795c4397189e2b64c97236204a941a41a` on 2025-02-10.

This establishes **SOURCE_LICENSE_LOCATED / ARCHIVE_BINDING_UNPROVEN**
for another primary source. It does not prove that every file in the locked
XCodec2 0.1.5 sdist comes from that revision or is covered by that license,
nor does it supply missing bundled notices or installed RECORD evidence.
The immutable archive report's missing-license-member finding remains valid.
Code terms are not the CC-BY-NC model-weight terms, and neither clears the
transitive LGPL or native-library conflicts.

The existing PR was also read back at
`6e61207a6810a78a84fbecc5c31affc82ebe85dc`: OPEN/Draft/CLEAN, 69 successful
checks and one skipped check. That is the prior pushed head's CI, not CI for
this new documentation supplement. No package archive, model or dependency
was acquired or executed; no approval, execution pin, lock or license gate
was changed.

## Official Transformers-native candidate — 2026-10-04

The publisher's current model card and original repository now explicitly
point to [HKUSTAudio/xcodec2-hf](https://huggingface.co/HKUSTAudio/xcodec2-hf),
an official Transformers-native release. This is a **new oracle candidate to
investigate**, not permission to replace the pinned decoder or its checkpoint.
The prior statement that no clean official replacement was established applies
to the reviewed legacy package path; it is not proof that this newer route is
unusable.

The manager authenticated the source file at Transformers commit
`469230357aab0f2b303b0d638c1f8d06edb14184`:
[modeling_xcodec2.py](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/xcodec2/modeling_xcodec2.py),
48,515 bytes, Git blob `edafffa6cb91e2b8b3432442fc1e777fc8638964`, SHA-256
`781afd94a2d0cfd8eaef3fd05312779d58e74f93c282d0190cdfc54e3a20281e`.
Its Apache-2.0-header implementation contains the official FSQ module and
quantizer projections directly. The inspected file's imports do not contain
the old `vector-quantize-pytorch -> einx -> frozendict` path. This is not a
complete transitive import/license audit: Torch, NumPy and Transformers
internals remain, including their native/build/RECORD review requirements.

Independent source/metadata review found the new checkpoint's namespace differs
from the existing native `generator.*` contract. No exact tensor-name, shape,
payload-hash or numerical correspondence to the original fixed checkpoint has
been proved. Similar FSQ geometry is not that proof. The Hub metadata still
identifies the new model weights as CC-BY-NC-4.0; an Apache-2.0 source header
does not change the weight terms or authorize upload.

Before changing an oracle, establish an exact source/checkpoint mapping,
authenticate a secure permitted dependency and native-library closure, and
review the resulting reference contract separately. If these prerequisites
become viable, exact real-weight correspondence and independent native CPU
parity belong on VAST. No model was obtained, no dependency was installed or
imported, and no source, lock, approval or execution gate was changed by this
investigation. Draft / blocked execution / NO_UPLOAD remains the disposition.

## Official checkpoint-mapping source — 2026-10-04

Read-only review found the publisher-supported Transformers
[conversion script at the same fixed revision](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/xcodec2/convert_xcodec2_checkpoint.py).
The manager independently authenticated Git blob
`da0790db43235c26b16a5619ab345eb9e300a7cc`, 15,257 bytes, SHA-256
`d91c3e278d9b1ba5c43bd8467e8579878bc35deb8759e284b253617ab70cb456`.
Its contents were inspected, not imported or executed.

The script defines an ordered mapping rather than a single prefix rename:
the old generator backbone becomes the acoustic decoder, the final backbone
norm becomes the decoder norm, the output head becomes its linear head,
quantizer layers move to the top-level quantizer, and `fc_post_a` becomes the
decoder input projection. Acoustic-encoder and semantic-adapter paths also
have explicit hierarchy changes. The fused attention projection is split
along dimension zero into Q/K/V; **Q and K**, but not V, undergo the source's
RoPE permutation before being assigned to separate projection names.

The script is written to remove non-persistent buffers and unused semantic
layers, reject extra/missing state-dict names, and load with strict matching.
These are inspected checks, not observed successful conversion. Its complete
entry point additionally applies/removes encoder weight normalization,
requires a CUDA device, and obtains external semantic configuration and
feature-extractor information. Those inputs and dependency paths would need
their own fixed identities and review before any approved execution; do not
run the script as-is or permit its optional Hub-upload branch.

This establishes **OFFICIAL_STRUCTURAL_MAPPING_LOCATED**, not exact payload
correspondence, native binder compatibility, independent numerical parity,
or a permitted complete dependency closure. The current native converter
preserves old tensor names, so a newer checkpoint is not a drop-in replacement.
No mapper or manifest was changed, no dependency or weight was obtained,
and no model, conversion, CUDA work or upload was executed. The existing
license/security/owner gates and Draft / NO_UPLOAD posture remain unchanged.

## Official wheel-packaging provenance boundary — 2026-10-05 JST

The existing PR152 was revalidated at clean
`1f75166bd0f35eff087394debf67ff729414c65e`: OPEN/Draft, 69 successful
checks and one skipped. This is its prior exact-head CI, not approval of
the external/native closure or verification of this documentation addition.

Read-only investigation resolved the official PyTorch `v2.13.0` source to
commit `cf30153c4c131c8164ee7798e5022d810682e2cb`, tree
`7cda5eae52ace99ca4daa7e623920cc93782cc6c`. Each source below was checked
against the API byte count and Git blob header, then SHA-256 hashed without
importing it or obtaining a wheel, native binary or model.

| Fixed primary source | Bytes / Git blob | SHA-256 |
| --- | --- | --- |
| [wheel repair](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/manywheel/repair_wheel.py) | 14,927 / `ae964c5c6d0a79f8367f7b65dda72aeddea2fced` | `d3f798285acaa11ce98f3c57cd84c39b73be111afaf2a5918805fed89aebf559` |
| [x86_64 image recipe](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/docker/manywheel/Dockerfile_2_28) | 7,889 / `67b8b60f832076240058dd0ba629fdf2e160da56` | `db310e80bf585c105436141c7fba5c90b0bdfbcd4679a837204f6b19d5b98dc9` |
| [aarch64 image recipe](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/docker/manywheel/Dockerfile_2_28_aarch64) | 2,625 / `6aeffe2e44dfa0cfe4a0478cdf3477ea5a9b5b48` | `8f9d99e555bf71b4c187378ec80528efc1d25913698461061814789f0057fbcd` |
| [libgomp source-build recipe](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/docker/common/install_libgomp.sh) | 1,671 / `308915ec4f61888c0ddd0a1d3f85bfb19e5704c9` | `b9ec9a21ca62e11e700521b2f77732a8cc2aa105c77f5bad4a693e9479a40d5b` |

The repair source selects `/usr/lib/<architecture>-linux-gnu/libgomp.so.1`
on Ubuntu and `/usr/lib64/libgomp.so.1` otherwise. It copies that actual
filesystem library into `torch/lib/libgomp.so.1`, including the CPU branch.
The inspected code does not bind the copied file to an RPM/source-package
digest or to the observed `78511033...` library digest. Recipe existence
therefore supports the packaging mechanism, not the exact wheel's provenance.

The x86_64 recipe uses a manylinux/AlmaLinux environment with a configurable
GCC toolset (default 13). It does not invoke the inspected libgomp source-
build helper. The aarch64 recipe does invoke that helper; the helper names
GCC 13.3.0 and explicit `armv8-a` flags. That ARM source-build identity must
not be assigned to the audited x86_64 library. Compiler version, filesystem
library path and applicable exception are separate facts, not interchangeable.

The [generated nightly workflow](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.github/workflows/generated-linux-binary-manywheel-nightly.yml)
has a Python 3.12 CPU row naming `manylinux2_28-builder` and image tag
`cpu-78e737ad29420ffc4800e677c51e2a852caf8359`. This is a nightly recipe,
not an authenticated release-build run or immutable image digest for the
locked 191,817,609-byte wheel. Its presence does not prove that image
produced the selected artifact or clear installed RECORD/native terms.

Disposition remains **BUILD_RECIPE_LOCATED / EXACT_BINARY_BINDING_UNPROVEN**.
Required next evidence is the selected release wheel's authenticated build
run/image identity and the copied library's exact package/source/build/notice
binding, followed by project-policy review. No blanket absence of such
evidence is claimed beyond the inspected sources. No package was installed,
source recipe executed, VAST worker allocated, signature supplied, exception
approved or execution/publication gate relaxed by this supplement.

## Transformers-native config identity — 2026-10-06 JST

The new oracle candidate's Hub revision is now fixed to the full commit
`64bd034d12d441299cdd535b15c33efd6ccdf252` in `HKUSTAudio/xcodec2-hf`.
Two independent read-only reviews checked only the following small, ordinary
text files. Each response named that exact `x-repo-commit`; full byte counts,
content SHA-256 and Git blob headers matched, with no LFS pointer or linked
weight payload. Nothing was saved, imported or executed.

| Fixed primary metadata | Bytes / Git blob | SHA-256 |
| --- | --- | --- |
| [config.json](https://huggingface.co/HKUSTAudio/xcodec2-hf/blob/64bd034d12d441299cdd535b15c33efd6ccdf252/config.json) | 2,923 / `3508ca1c0ab9d77f398e02b44fcc260be8daad55` | `f3082487a22d44095e42f0824825e4ed36f5b464ba8405d64eb20dc440a5756e` |
| [preprocessor_config.json](https://huggingface.co/HKUSTAudio/xcodec2-hf/blob/64bd034d12d441299cdd535b15c33efd6ccdf252/preprocessor_config.json) | 291 / `95ed01d22ecb63d548ca24731b3e6bf62b58d5d3` | `39d2ebcd4c4b44e9b780721ce3665e44f46c31638e72125653c89e8c61b437de` |
| [README.md](https://huggingface.co/HKUSTAudio/xcodec2-hf/blob/64bd034d12d441299cdd535b15c33efd6ccdf252/README.md) | 6,234 / `20ed76aad04473886a447368c634325edebb7d9f` | `811bdb9111db6e8e36a87cc3f358708ea609f9105d1925844403474d961c4288` |

The fixed config identifies `model_type=xcodec2`, hidden size 1,024, twelve
layers, sixteen attention heads and a 16-kHz sampling rate. The preprocessor
also records 16 kHz. The fixed card retains **CC-BY-NC-4.0** weight terms;
the Apache-2.0 Transformers source header does not change them.

This closes the short-revision ambiguity for these three metadata files only:
**CONFIG_IDENTITY_BOUND / WEIGHT_CORRESPONDENCE_UNPROVEN**. It does not
authenticate any weight, shard or tensor header, establish exact correspondence
to the legacy 1,153-tensor `generator.*` decoder contract, or approve a new
oracle. The already inspected mapping source remains the reference for a
future strict mapping review; full encoding components are not silently added
to the existing decode-only runtime. Exact payload/name/shape/dtype mapping,
permitted dependency/native closure, owner decisions and real CPU/Apple parity
remain outstanding. No source, lock, model manifest, signature, numerical bound
or execution/publication gate was changed; Draft / blocked / NO_UPLOAD remains.
