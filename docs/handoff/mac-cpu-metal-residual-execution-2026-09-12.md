# Mac CPU / Metal residual execution ledger (2026-09-12)

## Scope and baseline

This ledger continues the public Mac CPU / Apple Metal campaign after the
bounded 2026-09-11 Apple run and the four approved artifact publications.
The starting clean repository head is
`1de3887d795439b07abd9f2cba2fb6357674ac19` on `main`.
The reviewed residual branch advanced to
`72f0ad0b53b5fe234ecae3bacca7b2672784f498` after the first VAST evidence
wave and its portable dependency-evidence consumer repair. This later head
does not rewrite the starting baseline above.

The latest live audit, evaluated with audit logic through `1be45e76`, reports
194 repositories, of which 193 carry GGUFs, and 198 GGUF files. CPU status is
`full=135`, `partial=43`, `no-runtime-binder=15`, `not-artifact=1`; Metal
status is `full=135`, `blocked-by-cpu=58`, `not-artifact=1`. Therefore 59
public rows remain unresolved. This ledger does not reduce that denominator for a build,
synthetic fixture, inspection, or unexecuted packet.

No model is executed on the maintainer Mac. Model artifacts of at least 2 GB,
all `vokra-models` Cargo work, and workspace Cargo work go to a disposable
VAST worker. Scaleway remains the final Apple CPU / Metal / no-fallback stage.
Publication remains a separate repository- and artifact-specific permission.

## Completion transition per public row

Each row advances only when the corresponding evidence exists:

1. source, topology, dependency, license and distribution facts fixed;
2. strict converter, native binder/forward and CLI route complete;
3. independent upstream reference and real-weight CPU parity pass on VAST;
4. exact clean-head packet regenerated and transferred directly;
5. Apple CPU/reference, Metal/reference and Metal/CPU no-fallback pass;
6. separately approved artifact is published through `publish-one.sh`, or an
   exact owner-approved withholding/withdrawal disposition is recorded;
7. the live read-only audit reports the row as `full`.

## Execution waves

### Wave R1: prove already-landed strict paths

- Charsiu: the fixed `charsiu/en_w2v2_fc_10ms` conversion, independent
  Transformers reference regeneration and native CPU parity worker passed at
  `e478117c`. The 377,632,160-byte GGUF SHA-256 is
  `16aa648273268130fa1c5b3942b6b31ed70d6f4241b5418a12da9f5fc33aa7d9`;
  max absolute CPU/reference error was `0.000007629` against the preregistered
  `0.000200000` bound. The recovered evidence archive SHA-256 is
  `2400d9860ed706045e415be68c3752b3bed76cdb10c868d3649888ccdeb94f36`.
- microWakeWord: exact head `72f0ad0b` regenerated the fixed `hey_jarvis`
  contract and completed the reviewed stateful Path-C worker. The raw inventory
  SHA-256 is
  `ce57a719f60af3a494cbd8fb22ff30fdb405b0a3037b049333f25f5794749989`;
  the locked LiteRT dependency evidence is 1,005,169 bytes with SHA-256
  `b6938057226de9d4fc5ed2db75448a0cf2febf650302a445cb71b5342042f540`,
  and the independent inspector PASS report SHA-256 is
  `a1f28fea3600fed330a5b433e15083f9c23ec08fe5bea1b21b91eed4ece41670`.
  The temporary reviewed GGUF was 43,680 bytes with SHA-256
  `840fb20bf6a3d8ba85cdf204a702fc2b6ee6b3952c58cbe9e904e08b8f01ad0a`.
  The reference manifest SHA-256 was
  `ea9944cd53324fcd0d9829c5f4504c86c7575cce1b2eeaf7c8afea73e07cec4e`.
  Path C passed 512 persistent invocations, eleven preserved intermediate
  stages, final output and four-invocation reset replay; the package-scoped
  Rust result was `4 passed / 0 failed / 0 ignored`. The recovered validation
  JSON and Path-C log SHA-256 values are respectively
  `b0671ca449f92358eda7737b31933c53e59db3fd9ec0466f1e0303c263a7f60a`
  and
  `c97cd08879213da32e14eac08c6a3f206744600f941f5d1b757ffcdef3c2ed43`.
  No artifact was uploaded and the model payload remained VAST-only.
- Only the small logs/manifests above were recovered. The disposable VAST
  worker was destroyed with its storage after their local hashes and strict
  JSON contracts were verified.

These runs determine whether the corresponding open M5 checklist rows need
implementation or only evidence reconciliation. Neither path uploads a model.

### Wave R2: repair stale structural-only parity surfaces

- Whisper extras: commit `4cb50e5e` replaces the old
  `NotImplemented`/shape-only test posture with real Distil-Whisper and
  Kotoba-Whisper native forwards compared against an independent pinned
  Transformers reference. Exact VAST head
  `216588cb69e62e729e1c25f1f50325125572ab23` passed both variants with
  539 tensors each and exact greedy-token equality. Distil encoder/decoder
  maximum absolute errors were `2.956390381e-5` and `9.155273438e-5`;
  Kotoba errors were `2.193450928e-5` and `4.196166992e-5`, all inside the
  preregistered `0.01` bound. The recovered evidence archive SHA-256 is
  `d325bca39cbfabc732ee8f33396dac75d0d9fbbe883993b2fffde7d897154e63`.
  No artifact was uploaded.
- UTMOS: the tensor-only `.safetensors` preparation route is implemented and
  the legacy Lightning `.ckpt` remains permanently refused. Numeric parity is
  still blocked on an independently authenticated safe wav2vec state-dict and
  an owner-approved upstream model-construction path; no unsafe pickle fallback
  or UTMOS numerical success is claimed. A read-only primary-source refresh on
  2026-09-12 found no safe export to adopt: the fixed official
  [UTMOS demo tree](https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/tree/47212055c2ecfb02d40cec2395233b83295d3d30)
  still distributes only `epoch=3-step=7459.ckpt` and `wav2vec_small.pt` as
  the learned payloads, and Hugging Face identifies pickle imports in both.
  The official
  [UTMOS22 source tree](https://github.com/sarulab-speech/UTMOS22/tree/master)
  likewise contains download scripts but no `.safetensors` state-dict. The
  newer UTMOSv2 project is a different model and cannot be substituted as
  evidence for UTMOS22-strong.
- BigVGAN: extend the real-weight VAST and Apple gate from the already-proven
  base variant to all four released, already-supported runtime variants. Do
  not invent missing revisions or payload hashes.

Implementation in this wave is delegated to non-overlapping Luna owners under
`AGENTS.md`; the Sol manager reviews the integrated diff and remote evidence.

### Wave R3: execute existing family workers

Run only rows whose model-free gates and exact source/license inputs are
complete. The planned family order is:

1. NeMo-ASR remaining members;
2. Whisper extras after Wave R2 lands;
3. Dia/Zonos DAC-terminal TTS paths;
4. Qwen3-TTS main plus tokenizer companion;
5. VoxCPM2/VibeVoice continuous-VAE paths;
6. Irodori Japanese TTS where an independent allowed reference closure exists;
7. HiFTNet-family components whose exact hash-bound owner gate is signed;
8. BigVGAN variants whose exact hash-bound owner gate is signed.

An inspection-only result does not count as CPU parity. A worker that reports a
forbidden dependency closure, missing sidecar, unsafe checkpoint, incomplete
native synthesis, or unsigned hash-bound manifest returns to implementation or
owner disposition instead of advancing to Scaleway.

### Wave R4: shared implementation closure

- model-level BF16 activation/runtime integration with real AVX512-BF16 and
  Arm/Apple evidence;
- complete HiFTNet and BigVGAN production integrations, preserving explicit
  unsupported errors for every unavailable backend;
- remaining native binder/CLI composition gaps. BiCodec and SGMSE exact-file
  CLI routing is complete and the live audit promotes both rows to full;
- SeamlessM4T-v2-large: publish a real gated artifact or record an exact
  owner-approved withdrawal of the empty public repository.

### Wave R5: final Apple and publication reconciliation

After all non-Apple work in a batch is green at one clean head, regenerate the
packet, run Scaleway Apple CPU/reference, Metal/reference and Metal/CPU parity,
and reject any fallback. Publish only artifacts with explicit scoped approval,
then run the live inventory audit and update current documentation without
rewriting dated evidence.

## Current blockers that compute cannot waive

- BigVGAN and CosyVoice2 HiFT execution manifests still require exact
  hash-bound owner sign-off even though their general license-audit rows are
  signed.
- UTMOS has a restricted tensor-only state-dict route, but still requires an
  independently authenticated safe wav2vec source and owner-approved upstream
  construction. Its historical unsafe-pickle parity is not reusable evidence.
- Irodori's official source reference closure imports forbidden audio/native
  dependencies; a clean independent reference boundary is required.
- Voice-conversion destination decisions and SeamlessM4T publication versus
  withdrawal are owner dispositions, not Scaleway tasks.

## Resource record

VAST instance `50719822` (`vokra-residual-cpu-wave1`) was created solely for
the R1 CPU-evidence wave. After the Charsiu, Whisper-extras and microWakeWord
small evidence was recovered and verified, it was destroyed with its 200-GB
storage on 2026-09-12. The individual API readback returned `instances: null`;
the full inventory contained no Vokra-labelled instance or retained Vokra
storage. The remaining `ralomi-m4r4s-*` entries are unrelated and were not
touched.

## 2026-09-13 Dia, Zonos and Canary source/API integration replay

The next owner-independent source/API batch was integrated from clean base
`240187302b531dfb42d50ddf3535845e58c6145b` at exact clean implementation
head `d9aeb5d2dbe325c470032c975e03fbe147ed90dc`. It adds Dia's explicit
production generation APIs and no-BLAS reference dependency boundary, an
authenticated Zonos synthesis CLI, and a model-free Canary Flash/v2
dependency audit. The protected CosyVoice2 LLM owner manifest was not read,
modified or staged.

Disposable VAST instance `50763529` (`vokra-source-apis-20260913`) replayed
the exact head. `cargo test --workspace --all-targets --locked` completed 306
result groups with 8,028 passing, zero failed and 101 ignored tests; its log
SHA-256 is
`4b00a8a3816ff2d449ed62fb50ba9f9d5f02e6a5c01404d69f9793bb49813806`.
Workspace/all-target/all-feature Clippy with warnings denied passed; its log
SHA-256 is
`2b87f439fa896da2a32a0430e139744be7cdd43088d28800c75c255c7d92b5ac`.
Format, locked metadata, diff hygiene, zero-dependency, forbidden-symbol,
fixture-EOL, pipefail, architecture-handshake, bound-architecture and model-zoo
gates passed; the static-gate log SHA-256 is
`200f3aaf4023c72767e5b5192594d8be0c3a7bcecd11e9ffd1878230edcf3b55`.
`cargo-deny` 0.20.2 and `cargo-audit` 0.22.2 passed with only the existing
unmatched `libfuzzer-sys` exception warning; the dependency-gate log SHA-256
is `32fcf309f976ea56990443035ae611430458952d759a23d3f7c89e05242a0654`.
The preflight log SHA-256 is
`ca3d0cde9aa3a13b5f5c6bcff40e8ada38eea666e4c9f2911f941e76743fe471`.

The three exact-head model-free audits intentionally stopped before model
acquisition:

- Canary reported `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`. It accounted
  for the virtual root plus 133 installed package facts and retained 200
  publisher-license files. The closure still includes GPL/LGPL/native review
  boundaries such as NumPy/SciPy native libraries and soxr. Its report
  SHA-256 is
  `72da310c35841584eb6c8427766d2ec72638dbb72231a8b9e9e5d8fe793ad844`.
- Zonos reported `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`. Its eight
  factual blockers are missing installed publisher-license bytes for
  `safetensors` and `tokenizers`, plus the NumPy wheel's
  `libgfortran`/`libquadmath`/OpenBLAS native boundary and dependent
  extensions. Its report SHA-256 is
  `ba02dfbb6215045089809bcb6a699bacc7866879b5f6e2adfaad9caa34e26f52`.
- Dia built and audited the exact 26-package reference closure with
  `blas=none` and `lapack=none`, with no missing or unexpected package. It
  reported `FACTS_COLLECTED_GATE_BLOCKED` / `NO_UPLOAD` because the exact
  publisher/native rows still require owner review. Its report SHA-256 is
  `70235d132b3c17e55deba18f02634eed849fbd687d2f9f030b26ad3e7e33d66c`.

No checkpoint, source model or Hugging Face token was acquired; no model was
imported or executed; no real-weight conversion, independent-reference CPU
parity, Apple/Scaleway run or upload occurred. These implementation and
dependency facts therefore do not reduce the live 59-row unresolved public
denominator.

Only the 6-MiB bounded evidence set was recovered and its local hashes were
verified. Instance `50763529` and its 150-GB storage were then destroyed. The
individual API readback returned `instances: null`; the remaining VAST
inventory contains no Vokra-labelled instance or retained Vokra storage. The
unrelated `ralomi-m4r-4t-horizon-20260913` instance was not touched.

## 2026-09-13 corrected exact-head replay and live reconciliation

The earlier source/API replay above exposed additional fail-closed evidence
contract defects before any model acquisition. Dia's blocked audit wrapper did
not preserve its intentional exit contract. Zonos preparation evidence needed
to bind the locked NumPy sdist member count, preparation JSON arguments,
checksums inside the preparation root, installed wheel `RECORD`, the exact six
files generated by uv, the prepared virtual environment, and disabled Python
bytecode generation during validation. Those corrections advanced the clean
implementation head to
`87da78dc7709075d9dc23b797fc978b9c678c777`, based on GitHub `main`
`08b206c4218f2e4eff03403f2dae31c2b00efbcb`. The protected CosyVoice2 LLM
owner manifest was not read, modified or staged.

Disposable VAST instance `50795698` (`vokra-source-api-26384196`) replayed the
exact corrected head. `cargo test --workspace --all-targets --locked` completed
306 result groups with 8,028 passing, zero failed and 101 ignored tests; its
log SHA-256 is
`106bc004891f9167e9eeb8e108bca99e870027f7d90d7361b85e627fee246dd4`.
Workspace/all-target/all-feature Clippy with warnings denied passed; its log
SHA-256 is
`b23db99372b499fb76974e5cc32d2de39f6ec7fe5f92d7abff231ce0446887f5`.
Format, locked metadata, diff hygiene, zero-dependency, forbidden-symbol,
fixture-EOL, pipefail, architecture-handshake, bound-architecture, model-zoo,
parity-citation and all 55 Apple-worker syntax/self-test gates passed; the
static-gate log SHA-256 is
`e9532c4bf4ae6694a16d94c632357564730e6ffcefbddb7600f243b75d8a7aec`.
`cargo-deny` 0.20.2 and `cargo-audit` 0.22.2 passed with only the existing
unmatched `libfuzzer-sys` exception warning; the dependency-gate log SHA-256
is `3e839201d50ae2b2a55e3e7ae07e0a15ab355b3de2d729ca2c64315ea549ce4f`.

The exact-head model-free audits all preserved their intended blocked
publication posture:

- Dia reported `FACTS_COLLECTED_GATE_BLOCKED` /
  `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`, with the exact 26-package
  closure, 159 native files, no missing or unexpected package and no collector
  failure. Its report SHA-256 is
  `8ce645073916f8f1148ed572b476653fc602a2236fbeee90183b0b01b1c57cd8`;
  the owner-scope SHA-256 is
  `0d51a44f9494313d01b6cb0314d35215c48c311c0c071d8516a10617a2f224e3`.
- Canary reported `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`, with 133 exact
  installed distributions, 134 package rows and dependency paths, 200 retained
  license entries and no collector failure. Its report SHA-256 is
  `37dd6ce7d30b1e7a3d3c184a5379c4519bb2dd1fa06f3b03349b0530fec51059`.
- Zonos reported `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`, with 25 exact
  installed distributions, 51 native files and 49 retained publisher
  license/notice files. The NumPy wheel evidence accounted for all 829 wheel
  `RECORD` rows plus exactly six hash-and-size-bound uv-generated rows, and the
  native/runtime configuration passed the no-forbidden-BLAS checks. Its report
  SHA-256 is
  `507db1c6ecc623430134396466988a0302277c4ba00d19aca4c9744edc661bce`.

No checkpoint, source model, Hugging Face token or model payload was acquired;
no model was imported or executed; no real-weight conversion, independent
reference parity, Apple/Scaleway run or upload occurred. The live read-only
Hugging Face reconciliation at the same head reported 194 public repositories,
193 GGUF-bearing repositories and 198 GGUF files. CPU status is `full=136`,
`partial=42`, `no-runtime-binder=15`, `not-artifact=1`; Metal status is
`full=136`, `blocked-by-cpu=57`, `not-artifact=1`, leaving 58 unresolved public
rows. The summary and TSV SHA-256 values are respectively
`8a7a6f8676d5022586602c74634e5a100635826a7a6f2d7f5d34d95da29000a4` and
`eb284b33f738f52e4e7df3d63a09bc60b92b5dc9eba0c0d17c6dea0f668881dc`.

The bounded, model-free 371-entry evidence archive was recovered as
`/private/tmp/vokra-evidence-87da78dc-d58ba75c.tar.gz`; its remote and local
SHA-256 both equal
`d58ba75c3985f81f1a7e5887d88ad8c09187fbef55116c8630d0c16600bad283`.
It contains no GGUF, checkpoint, wheel, sdist, virtual environment, Cargo
target or source-model payload. Instance `50795698` and its 150-GB storage were
then destroyed. The post-destroy inventory contains no Vokra-labelled
instance, and the independent volume inventory is empty. The unrelated
`50798096` (`ralomi-m4r-4u-matched-v1`) instance was not touched.

## 2026-09-13 Zonos Transformers security-closure replay

GitHub dependency review identified the pinned Zonos reference dependency
`transformers==4.48.1` as vulnerable. The dedicated project now pins
`transformers==5.10.4` and its compatible `huggingface-hub==1.5.0` closure.
This dependency update is not source/API compatibility evidence: the new
pre-acquisition gate stops with
`BLOCKED_UNVERIFIED_TRANSFORMERS_API_SMOKE` before any Zonos source,
checkpoint or model access until the exact dependency scope is approved and a
separate VAST source/API smoke is authorized.

The first exact-head replay exposed two model-free worker defects before an
audit report could be created. The dependency-approval self-test used the
macOS-only `/private/tmp` path, and the no-BLAS NumPy preparer still bound the
pre-update project and lock hashes. Commits `9e8be2f8` and `5a30084d` replace
the fixed temporary path with a platform-native root, add Linux-relevant
self-test coverage, restamp the project/lock identities and make ordinary
preparer self-tests reject future identity drift.

Disposable VAST instance `50829786`
(`vokra-zonos-transformers-31ea5cec`) replayed final clean head
`5a30084dcea18b9e72d7befb776f98a61a2a4286`. The frozen lock resolved 38
packages and installed 34 active distributions. Model-free imports reported
Torch/Torchaudio `2.6.0+cpu`, Transformers `5.10.4`, Hugging Face Hub `1.5.0`,
NumPy `2.2.2` and safetensors `0.5.3`. The compatibility gate self-test,
dependency-audit self-test and complete Zonos inspection-worker self-test
passed; the ordinary compatibility gate preserved its intentional exit 2 and
blocked marker.

The exact-head dependency audit then built the locked NumPy wheel with BLAS
and LAPACK disabled and returned the intended
`BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD` result with zero collector
failures. It recorded 34 active lock packages, 34 installed distributions, 50
native files and 59 retained publisher licence/notice files. The candidate
scope SHA-256 is
`59d9b214d811efad3c4c6a85dc8f82bff9b220026d71fc7b723f9d0ce088c30e`;
the audit JSON SHA-256 is
`1432d2ddb45a2f7b2d0f8490b83e2151c4a61e226899224963bd9f3d17ec7a9e`.
The locally built 10,105,723-byte no-BLAS NumPy wheel had SHA-256
`4018ba8c1e41e876983706adfa96f97e2314d3a01d627c92b13a58c2b18ee2c1`
and remained on the disposable worker.

Only the 208-KiB model-free evidence archive was recovered. Its remote and
local SHA-256 both equal
`99e1f87758d4661da26a29f374e34a28d2ddeaa165ef0006a942699b420cc0c6`,
and all 71 internal checksum entries passed locally. It contains no GGUF,
checkpoint, safetensors, PyTorch payload, wheel, sdist or virtual environment.
The report records `source_access=false`, `model_access=false` and
`checkpoint_access=false`; no real-weight conversion, independent reference,
CPU parity, Apple/Scaleway execution or upload occurred.

Instance `50829786` and its 200-GB storage were destroyed after evidence
recovery; its individual API returned `instances: null`. The remaining VAST
inventory contains only unrelated instance `50798096`
(`ralomi-m4r-4u-matched-v1`), which was not modified. This security closure
does not decrement the live inventory: 58 public rows remain unresolved, and
Zonos stays blocked before source/API and real-weight work pending the exact
dependency approval.

## 2026-09-13 BigVGAN exact-head dependency-closure replay

The model-free BigVGAN Linux and Darwin dependency closures were replayed on
disposable VAST instance `50849061` from clean public `main` head
`64394791db5e8a2f2f437e0b4c405e78312bf2f0`. The workers downloaded and
inspected only the exact CPython 3.12 wheels selected by the committed lock;
they did not install or import packages, acquire a source/model checkpoint,
execute a model, create an owner signature, or upload an artifact.

The Linux candidate reproduced SHA-256
`fd414613311cf1ca7da4504e85acbb79d43c200a4cb1dc221e2421fc67b26086`
with ten active packages and 142 native/bundled payloads. The arm64-Darwin
candidate reproduced SHA-256
`148e44365efa92c2cd95feeef156e327975be465aad21c6b20c979433f6d25fa`
with ten active packages and 21 native/bundled payloads. These are the exact
candidate identities already bound into approval scope
`73f8b60a0f71be420dfbaf1fc7213743701816a301303ff98ba46bbf2d09bce4`.

Only the candidates plus their LICENSE, NOTICE, METADATA and native-payload
inventories were recovered. The 274,515-byte local review archive has
SHA-256
`56572d0ad60586dfb1f405c082b1495f82d462b3dddf6ee60108afc737cd008b`;
all 82 extracted payload hashes matched the candidate records. The evidence
again includes Setuptools' vendored LGPL-3.0 and MPL-2.0 material and the
PyTorch native/bundled closure. It therefore confirms rather than removes the
manual legal-review boundary: the recorded `WITHHOLD_EXECUTION`,
`BLOCKED_UNREVIEWED_TRANSITIVE` and `NO_UPLOAD` posture remains in force.

## 2026-09-13 FireRedASR-AED-L source-state reconciliation

A read-only reconciliation against clean public `main` head
`64394791db5e8a2f2f437e0b4c405e78312bf2f0` corrects the older planning
shorthand that described the released-weight AED decoder and beam search as
unimplemented. The current source already contains the strict 940-tensor
runtime binder (551 encoder plus 389 decoder descriptors), native
CPU/Metal-dispatched encoder and incremental decoder primitives, the pinned
upstream `batch_beam_search` policy and native beam execution, authenticated
`cmvn.txt` and `dict.txt` sidecars, token rendering, and an explicit
PCM/fbank/CMVN-to-greedy-token composition seam.

The public row remains honestly `partial`: no current exact-head run proves
the 4,678,597,714-byte released checkpoint against an independent FireRed
reference, and the ordinary `transcribe_tokens` / `AsrEngine::transcribe`
surface deliberately returns `UnsupportedOp` until that VAST parity gate is
green. Therefore the next operation is not another speculative decoder or
beam implementation. It is an approved, no-upload VAST conversion plus
independent CPU encoder/beam/output parity; only after that evidence may a
small reviewed implementation change open the ordinary transcription route,
followed by the final Scaleway Apple CPU/Metal/no-fallback worker. No model,
source checkpoint or external payload was acquired or executed during this
reconciliation.

The exact-head VAST model-free worker then completed on instance `50849061`
with its intended exit 2 contract. It reported `BLOCKED_OWNER_REVIEW` /
`NO_UPLOAD`, `payload_status=NOT_ACQUIRED` and
`execution_status=NOT_PERFORMED`. The frozen Linux closure contains 27 rows
with digest
`b79e93fabc422b5b9a1c4347402829ad46d2e82afc342e7cf5125f50286e768c`;
the pending owner-review scope SHA-256 is
`02269d72f53a573458f10c919fc23270f38b967814cd15854a0bf0d853c46b6c`.
The recovered 24-KiB evidence archive is
`/private/tmp/firered-model-free-64394791-evidence.tar.gz`, and its remote and
local SHA-256 both equal
`f43f4d852787d845b5e1055d39653dcbdd8221b1953fe98552801a8655edce3c`.
It contains only the audit JSON and log. The JSON and log SHA-256 values are
respectively
`b7895bead57ffc179ab3c194cdd61fa8deca67db895df78b2ed25c2be02a3a04`
and
`4a8db751d8e35b625ba8fdff3686dd27bbbfed61535bdcb162a5766b158eb5e4`.
The remaining pre-acquisition decisions are the exact per-row publisher and
native-payload review, including the pinned `kaldi-native-fbank` closure, plus
training provenance. This evidence does not authorize checkpoint acquisition
or execution.

Commits `5d9d209b` and `77984f04` then add and harden a distinct
`--dependency-audit-only` worker stage. It prepares the frozen Linux CPU
closure, clones only the pinned FireRed and `kaldi-native-fbank` source trees
with Git LFS smudging disabled, and collects installed publisher, licence and
native-payload evidence without contacting the model repository. The first
exact-head VAST attempt correctly failed closed because two tracked FireRed
source aliases are Git mode `120000` relative symlinks rather than regular
files. The follow-up records their Git blob identities and relative targets
without following them outside the source tree; it never initializes a
submodule or fetches a model payload.

Those two commits were isolated in
[PR #98](https://github.com/ayutaz/vokra/pull/98) and merged to `main` as
`21acdbf9925ca6dd6c2c77144951cf8948754832`. Its review boundary was the
fail-closed dependency-audit packet only. The merge does not widen the
owner/legal scope or authorize model access.

The final exact-head worker at `77984f048069bd9130f8f414a4dd915595a3aee8`
completed evidence collection and retained its intended exit 2 posture:
`BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD`. It records 27 active closure
rows, 27 installed distribution-evidence rows, 27 owner-review rows, no
collection failure, and 42 authenticated tracked FireRed source entries. The
source `LICENSE` is 11,357 bytes with SHA-256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
All ten recognized Hugging Face token environment variables were absent. The
report records the model API as not contacted, `snapshot_download` as not
called, and checkpoint/payload access, model import, execution, reference and
conversion as not performed. The pending dependency audit scope SHA-256 is
`7c0498f231ffef3fc16c212ce8cefa3b4a0ae686cba4db1aab48312cfb151324`.

Only the JSON and validation log were recovered in
`/private/tmp/firered-dependency-audit-77984f04-evidence.tar.gz`; its remote
and local SHA-256 both equal
`6d82c4ef9abd391124be0a819778d76a1dd69edf0a4c0a5a8ebe52c4d5f98002`.
The JSON and log SHA-256 values are respectively
`430acd2ec0ccd4c4122fc9638ce3251017b4c57c99a9f3586668207a22a87d8b`
and
`cbc6b25bd03ff377af402fa3c9e1515bb32d71ae1c78393d1f9b72109b3cc943`.
The owner/legal review remains mandatory before any FireRed model acquisition
or real-weight execution. This packet makes that review concrete; it does not
self-approve it.

## 2026-09-13 Zonos Transformers source/API compatibility closure

The preceding security-closure record remains the evidence for its exact head,
but its final source/API blocker was superseded by a later, separately bounded
replay. Branch `feat/zonos-transformers-compat-smoke-20260913` was rebased
without patch changes from base `446ea9e706aac8fc3ca779512df86dda933bce2a`
onto clean `main` base `e7944b2dc7a5cf4c7d7919b8c49bbf326df1dbc6`.
`git range-diff` reported all ten commits unchanged. A subsequent security
closure pinned `setuptools==84.0.0`, a safe release after the first patched
`83.0.0` for GHSA-h35f-9h28-mq5c, and made the compatibility generator record
that distribution version. The final exact implementation head was
`c82ed76e308a13f5ca5324ae3d00f9935f40123b`.

Disposable VAST instance `50849061` (`vokra-zonos-setuptools-20260913`)
checked out that exact head and cloned only the fixed official Zonos source
revision `Zyphra/Zonos@bc40d98e1e1ab54fc65c483be127a90e3c7c0645`.
The first final-head probe exposed that the generator omitted Setuptools from
its package-version evidence; no model access occurred, and that diagnostic
run was not accepted as final evidence. After the generator repair, the
strict, model-free Transformers `5.10.4` source/API probe passed its fresh
hash-bound validator. Its JSON and log SHA-256 values are respectively
`a6890330da22270d8831b30d89d6852b087cc5eed84b98a95e1a6cb8de7daf74`
and `e3f133833b5ed18af0ab776b7b5785776f39d1e3be483821801d6a09746c29f2`.
The accepted report records `model_access=false`, `checkpoint_access=false`,
`hf_token_present=false`, `constructor_calls=0` and a clean source checkout
after import.

The frozen dependency audit was regenerated at the same head after building
the locked NumPy `2.2.2` wheel with BLAS and LAPACK disabled. It retained the
intentional exit 2 posture `BLOCKED_UNREVIEWED_TRANSITIVE` / `NO_UPLOAD` with
38 lock packages, 34 active packages, 34 installed distributions, 41 native
files, 59 retained publisher licence/notice files and no collector failure.
The audit JSON and log SHA-256 values are respectively
`d56994ebb79d1680f964984ec0e03fbc96d3ec6d05278031d7d8eec39c36361c`
and `46b87c3af3625d246572b314996a414dd4ed7d3fcd269e8b74c7977517859a4d`.
The candidate-scope SHA-256 is
`a71392aea86a7470c25fc7fb59c799205da0194c1247cba0c3b19165f5c4feba`,
and the publisher manifest SHA-256 is
`7358b563f1cc6a96c91eef602c432a06fd02f25c84986c368c04f809f8cd4378`.
The 10,105,816-byte no-BLAS wheel, SHA-256
`0e7a1f3be4d0ac45109c3bcfb9afacc22987aeaae92fac4c030ada528d56f2ca`,
remained on the disposable worker. Passing source/API compatibility therefore
removes only the narrower `BLOCKED_UNVERIFIED_TRANSFORMERS_API_SMOKE`
condition. It does not approve the transitive dependency closure, real-weight
acquisition or publication.

The same VAST checkout passed the locked workspace/all-target test suite with
`groups=306 passed=8028 failed=0 ignored=101`, workspace/all-target/all-feature
Clippy with `-D warnings`, `cargo deny`, `cargo audit`, and the complete static
gate group. Their log SHA-256 values are respectively
`584c3b0935062fe353968f0faaf0ed0d238be5bcd71e6feb4973eddd211a9810`,
`04e86037dda801d80eb80030edcd91815c07a124d038fddf9ca83acffcd87bdd`,
`3cf80bdc410003f3945b935691d26b6bf07dcdf648ce80b242447c564ff1991d`,
`d090e5b875632f40e1b302dfc89b0a50776cee61f7ecb60e79284d779b80e4b7`
and `f673d40c8e8793aae6497d91914b9611c4fc51d565cf2f4c568c86acfb9e5830`.
The static group included all 55 Apple-worker syntax/self-tests and reported
`syntax_failures=0`, `self_test_failures=0` and
`unsupported_or_blocked=0`.

Only the 317,680-byte evidence archive was recovered. Its remote and local
SHA-256 both equal
`bd5c4183e5a8b7372869f830efc2843acccf640ae10cb76e8f2d9194ffae2c09`.
All 72 internal checksum entries passed locally, yielding 73 files including
the checksum file, and all 59 publisher file hashes and byte counts were
independently revalidated against the retained manifest. The archive excludes
GGUFs, checkpoints, model payloads, wheels, sdists, prepared environments and
Cargo target data. Instance `50849061` was stopped after this recovery.
Exact-ID destroy authorization remains pending, so its storage continues to
bill at `$0.037037037/hour`; this record intentionally reports the stopped
state rather than claiming destruction. The unrelated protected instance
`50798096` (`ralomi-m4r-4u-matched-v1`) was not modified.

No real-weight conversion, independent numerical CPU parity, Apple/Scaleway
execution or upload occurred in this closure. The live inventory therefore
remains 136 full and 58 unresolved public rows. Zonos may advance beyond its
source/API compatibility gate only after the exact transitive dependency scope
is reviewed; publication remains separately prohibited until all five publish
gates and artifact-specific authorization pass.

## 2026-09-13 residual dependency-license closure wave

The captured BigVGAN dependency rows were reviewed and merged through
[PR #100](https://github.com/ayutaz/vokra/pull/100) as
`7a9c888128d82adbcb327a9b2aa17bf8ff1d94d9`. The FireRedASR-AED-L closure
was then merged through [PR #101](https://github.com/ayutaz/vokra/pull/101)
as `5e4199551d4a2948121f49c47ae7977f5b14e346`. Both changes preserve empty
owner signatures and `NO_UPLOAD`; they review captured dependency facts but
do not authorize source/model acquisition or publication.

Canary, Zonos and Dia were combined only for an exact-head validation wave.
The clean base was `5e4199551d4a2948121f49c47ae7977f5b14e346` and the
exact candidate head was
`74c4d1627cdcc3626555eaa9fe1d1c1bbf64cf67`. Its git bundle SHA-256 was
`5b73c04cac29d7269c19d66ac520bef28b8e5c7f4f7feac8050cdbe38307bde4`.
The corresponding [PR #102](https://github.com/ayutaz/vokra/pull/102) was
squash-merged on 2026-09-13 at `2026-09-13T09:17:28Z` as
`50981d60fe0d6d0542514545db2ddf254e7343d9`. Its required checks were green;
the still-running optional Unity package job did not block the merge.

The committed closure gates bind the following exact factual evidence while
retaining owner/legal review:

- Canary: 134 lock rows, 133 package facts and 200 publisher files; candidate
  owner scope
  `f271515af4a6a5c97ef84bd93b21f18e36da8d63c8885693004fcf2bd7369d1e`;
- Zonos: 34 active packages, 59 publisher files and 41 native files at captured
  head `c82ed76e308a13f5ca5324ae3d00f9935f40123b`; candidate scope
  `a71392aea86a7470c25fc7fb59c799205da0194c1247cba0c3b19165f5c4feba`;
- Dia: 26 active packages, 159 native/bundled files and 50 publisher entries;
  closure approval scope
  `1f8c9465007f01acd17b256e54f410733ce1e0778256d1a588190dcef6ee9866`;
- FireRedASR-AED-L: 27 distribution rows, 186 required paths and 135 unique
  payloads; closure approval scope
  `ee84a6d15aaa9e8e594fc2b2fffe59f0e3a70360f0fcae1629641ec3f1375d9e`.

Disposable VAST instance `50849061` checked the exact candidate head without
model/checkpoint acquisition or an HF token. The locked workspace/all-target
tests, workspace/all-target/all-feature Clippy with `-D warnings`, `cargo
deny`, `cargo audit`, the four license-gate self-tests, the Zonos external
publisher/native evidence gate, Dia and FireRed ordinary fail-closed gates,
and the forbidden-symbol, zero-dependency and Canary-review guards all passed.
The expected ordinary gates validated their evidence and then returned exit 2
because owner/legal approval remains absent. The recovered 136,282-byte log
archive has SHA-256
`27ccd7d7e9311f1e33f34526e4f19d42a8f852a8dc69305c57fd599780b30485`;
all recorded per-file hashes matched after recovery.

The worker was stopped after the archive and exact HEAD were verified.
Instance `50849061` now incurs storage only at `$0.037037037/hour` until an
exact-ID destroy authorization is supplied. Protected instance `50798096` was
not modified. This closure wave does not change the live support denominator:
the public audit remains 136 full and 58 unresolved rows. The next transition
for BigVGAN, FireRed, Canary, Zonos and Dia is an exact scoped owner/legal
decision, followed by real-weight VAST conversion and independent CPU parity;
only green packets advance to Scaleway Apple CPU/Metal/no-fallback execution.
