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
  or UTMOS numerical success is claimed.
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
