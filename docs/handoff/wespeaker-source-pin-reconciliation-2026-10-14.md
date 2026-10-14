# WeSpeaker source pin reconciliation — 2026-10-14

Management readback at accepted source baseline
`6cee4e4fc9ad1cf2ca1c0fa5c5e48e69dabde806`. This is an official small-source
identity investigation, not dependency installation, API/native validation,
checkpoint acquisition, inference, numerical parity, owner approval or upload.
The [58-row inventory](current-58-prerequisite-inventory-2026-10-14.md) and
the separate frozen owner packet are not promoted by this correction.

## Actual primary source and the recorded-input defect

The official source revision remains
`45941e7cba2c3ea99e232d02bedf617fc71b0dad` in `wenet-e2e/wespeaker`.
GitHub's contents/tree metadata and direct HTTPS raw bytes were independently
checked. Each raw request was limited to 20,000 bytes and 30 seconds; source
bytes were hashed from standard input and never imported or executed. No
model, configuration, tokenizer or package payload was acquired.

| Fixed official input | Bytes | Git blob | SHA-256 |
|---|---:|---|---|
| [ResNet](https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/models/resnet.py) | 9,564 | `17607e6d2c72627e15db4214cacfa9d7b89ca945` | `6f3c8219be2c9a8b9eabed8169c1abaec3e48670be7aaf1e792138b2b20e68c4` |
| [Pooling](https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/models/pooling_layers.py) | 10,255 | `47120eb9029ba1b00d18fcd61dd80f8afa2037c6` | `768910c91e353f1d0c2f3edc8f9ae758d2ff7e477d37fb470ae946eba69adac8` |
| [Apache-2.0 LICENSE](https://github.com/wenet-e2e/wespeaker/blob/45941e7cba2c3ea99e232d02bedf617fc71b0dad/LICENSE) | 11,357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The recorded ResNet and LICENSE identities match. The old pooling blob
`47120eead47a511939267470496539804c17b7d3` and SHA-256
`768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9`
do not identify the actual pooling file at the fixed revision. This is a
recorded source-contract error, not evidence of a changed upstream revision.
Correcting only the dumper would leave the worker and preflight inconsistent;
all active source bindings must agree before a run is admitted.

## Remote checkpoint name, not checkpoint-byte validation

Public, unauthenticated HTTP HEAD requests at the unchanged HF revision
`f0c48c298fd835726c27956a5d617bad7115627e` distinguish the remote inputs:
`avg_model.pt` returns HTTP 404, while `avg_model` returns redirect metadata
with `x-repo-commit` equal to that fixed revision, `x-linked-size` 45,053,131
and `x-linked-etag`
`9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449`.
Only headers were read: neither response body nor checkpoint/configuration
payload was acquired, and no HF token was used. Redirect `content-length`
describes the redirect response, not the checkpoint size.

The worker therefore requests exactly the remote `avg_model` and stages it
as the existing local `avg_model.pt`; this is an explicit normalization, not
a retry through alternative filenames. The local tensor/provenance contract,
declared checkpoint size and SHA-256 are unchanged. Header metadata matching
the declaration is not a fresh checksum of the actual checkpoint bytes.

The formerly recorded checkpoint Git OID
`7f92ddd059d244c7d2653650d3be85de9f136c41` was not authenticated by those
headers. Fixed-tree metadata retrieval was unavailable; it was not bypassed
and no LFS pointer was downloaded. The active checkpoint identity now records
`git_oid: null` and `git_oid_status: NOT_AUTHENTICATED` rather than presenting
that unproven value as verified provenance. This does not authenticate the
separately recorded configuration or public-replacement identities.

## Exact source import boundary, not omission compatibility

The fixed [top-level directory](https://github.com/wenet-e2e/wespeaker/tree/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker)
and [models directory](https://github.com/wenet-e2e/wespeaker/tree/45941e7cba2c3ea99e232d02bedf617fc71b0dad/wespeaker/models)
have no `__init__.py`. The normal import of `wespeaker.models.resnet` therefore
traverses namespace packages, without parent initialization code. The selected
ResNet/pooling source imports Torch, its neural-network/functional surface and
the relative pooling module. This is a static source finding, not an actual
target-package import or a claim about dynamic Torch internals.

The project's four direct requirements remain NumPy, safetensors, Torch and
TorchAudio. In the current exact 14-row lock, setuptools is a Torch transitive
edge, not a direct project requirement. No direct setuptools, pkg_resources
or autocommand import was found in this selected official source route. The
[actual LGPL vendored-payload finding](wespeaker-setuptools84-static-finding-2026-10-14.md)
still applies to the locked setuptools wheel; lack of a direct import is not
a redistribution grant or proof that the dependency can be omitted.

Any omission candidate needs a separate fixed lock/project, exact-wheel
metadata/native review and authorized Linux model-free API verification.
The [TorchAudio linkage finding](wespeaker-torchaudio211-static-facts-2026-10-14.md)
also remains open. No dependency is removed by this source correction.

## Scope, evidence and remaining work

The old pending approval scope
`77ff8539c507652b7fda4ae4481b2c0fa60dfad4b81eaaa43d0466e127599f22`
cannot authorize corrected inputs. A replacement scope must be derived by the
existing gate from the exact corrected source, unchanged lock/project and
complete pending dependency/component rows. Signers, approval digests and
operator decisions remain unapproved; `NO_UPLOAD` stays in force. Historical
source/API/native/parity receipts are not restamped into the new scope.

The intermediate source-only correction produced pending scope
`f4c1870171d5e98b74dae0820c414db99be50a0b49c0a0b44a1e8c56198b3419`.
Adding the explicit remote/staged filenames and truthful unauthenticated
checkpoint Git-OID state produces the current pending scope
`009e346051df3bbacfa07e3ee7828d48c3d38f36efb8e8d8ed67744722c84ec1`.
The 14 dependency rows, five component reviews and operator decision remain
pending, with no new signer or approval digest. The lock/project identities
remain `65715f60a9f86aa80792e4a11303bc4eb2067cb4a3cd670b5b6d724d478c9d51`
and `88c7416e8a4a0f57a537e4ac832f268d24c6747fe06777aa392981fd9ba98bba`.

Implementation review and focused static regressions must cover the dumper,
preflight, both manifest source bindings and VAST worker consistently, with
real small source bytes as positive controls and drift/old identity as negative
controls. Such tests are not a full upstream checkout, Linux API/native result
or model/reference run. Relevant PR CI and normal acceptance remain separate.

Manager independently passes fourteen stdlib tests through offline/no-project
UV/Python 3.12, including exact bytes, sizes and Git-blob hashing of all three
official fixtures, real fixture Git status/blob checks with only the official
commit identity mocked, one-byte drift and early-runtime refusal. Approval
positive/negative controls use synthetic temporary bundles, never tracked
sign-offs. Fixture Git routing/configuration is isolated from the maintainer's
repository, and raw Git-blob hashing disables external filters. An unchanged
new synthetic manifest with old evidence is specifically rejected at the
external manifest/scope binding check. Preflight and dumper self-tests, the
preparer self-test and the worker's five-case self-test also pass. The final
suite includes the offline worker self-test: synthetic curl checks the exact
remote filename, fake-token stdin/argv/environment boundary, unchanged existing
staged input, and directory/symlink destination races without nested payload.
The Linux production path uses an exclusive no-target-directory hard link;
the maintainer's macOS self-test does not prove that Linux path. The same suite
is added to Linux CI, which must pass at the final PR head before acceptance.
Actionlint and final workflow hygiene pass (48 workflows, 36 crons).
Documentation reference/example gates pass; their 30 pre-existing
deferred examples remain explicitly unverified. No fresh VAST or Linux
target-package/native run is claimed by these local static results.

Next: finish source-binding review and regressions, then close dependency,
native, provenance and exact owner/operator prerequisites. Only an admitted
scope can proceed to independent no-upload VAST CPU parity and a regenerated
Apple packet. The legacy public artifact, replacement, CPU/Metal results and
publication state remain unchanged. The protected CosyVoice2 input was not
read, hashed, changed or cleaned.
