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
