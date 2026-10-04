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

The official [pinned XCodec2 decoder](https://huggingface.co/HKUSTAudio/xcodec2/blob/e9463f16b1a4af077e9d96c06ae99bebc8639c1ee/vq/codec_decoder_vocos.py)
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
