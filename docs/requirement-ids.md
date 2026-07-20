# Requirement ID Glossary

**English** | [日本語](requirement-ids.ja.md)

Vokra's public documents — `README.md`, `CONTRIBUTING.md` and the pages under
`docs/` — cite requirement IDs such as `FR-EX-08` or `NFR-DS-02` as shorthand
for fixed design commitments. **The documents that define those IDs are not
published.** Requirement / system-requirement / deliverable / milestone
planning is maintained privately by the maintainer, so a reader who only has
the public repository has no way to resolve a cited ID.

This page closes that gap. For every requirement ID that appears anywhere in
the public documents it gives **one line: what the item governs**. That is
enough to read a code comment, a PR description or a review remark without
guessing.

## What this page is not

- **Not the requirement specification.** Each entry states the *subject* of a
  requirement, not its wording, acceptance criteria, numeric thresholds, or
  the release it is scheduled for. Those live in the private planning
  documents.
- **Not a complete index of every ID that exists.** It covers exactly the IDs
  cited in the public documents. Internal planning uses more.
- **Not a stable numbering.** Numbers within a family are frequently
  non-contiguous here (`FR-MD` jumps 02 → 09 → 10 → 13). That is expected: the
  gaps are IDs that exist but are never cited publicly, **not** missing
  entries.

## Keeping this page current

**Last verified: 2026-07-20 — 91 IDs cited across the public documents.**

The set this page must cover is mechanically derived, not curated by hand.
Regenerate it with:

```bash
git ls-files 'README.md' 'README.ja.md' 'CONTRIBUTING.md' 'docs/*.md' 'docs/**/*.md' \
  | xargs grep -IohE '\b(BR|FR|NFR|IF)-([A-Z]{2}-)?[0-9]+' \
  | sort -u
```

Two details of that command are load-bearing:

- The middle `[A-Z]{2}-` segment is **optional**. Two-segment IDs (`BR-02`,
  `IF-01`, …) exist, and a regex that requires the middle segment silently
  drops five of them.
- The corpus is `git ls-files`, i.e. **tracked files only**, and covers the
  README pair, `CONTRIBUTING.md` and `docs/`. Untracked or ignored working
  files are deliberately outside it.

**Who updates it**: whoever first cites a requirement ID in a public document
adds its row in the same pull request. This is not left to periodic review —
`scripts/check-doc-references.sh` compares the cited set against the rows
below in **both directions**, so an uncatalogued ID and a stale leftover row
both fail the check. The script runs in CI (advisory) and can be run locally:

```bash
bash scripts/check-doc-references.sh          # verify
bash scripts/check-doc-references.sh --list   # show the resolved sets
```

The Japanese twin, [requirement-ids.ja.md](requirement-ids.ja.md), must list
the same IDs; the same script enforces that too.

## Families

| Prefix | Domain |
|---|---|
| `BR` | Business requirements — why the project exists |
| `FR-LD` | Model loading (GGUF, safetensors, metadata) |
| `FR-EX` | Execution engine and IR |
| `FR-OP` | Speech operators (vocoder, codec, decode, enhancement, speaker) |
| `FR-BE` | Backends (CPU, GPU, NPU, build SKUs) |
| `FR-MD` | Model support and the process for adding a model |
| `FR-QT` | Quantization policy and verification |
| `FR-SV` | Server / API compatibility |
| `FR-ST` | Streaming behaviour |
| `FR-CP` | Compliance (watermarking, provenance, research flag) |
| `FR-TL` | Tooling (converter, CLI, eval, build scripts) |
| `NFR-DS` | Distribution and size |
| `NFR-PF` | Performance |
| `NFR-QL` | Numerical and audio quality |
| `NFR-RL` | Platform reliability constraints |
| `NFR-MT` | Maintenance, CI/CD, community process |
| `NFR-LC` | Dependency licensing |
| `NFR-LG` | Legal / regulatory |
| `NFR-PT` | Platform coverage |
| `IF` | External interfaces |

---

## BR

Business requirements. Only the two cited publicly are listed.

| ID | What it governs |
|---|---|
| `BR-02` | Multi-platform stability as a **precondition**: Windows / macOS / Linux / Android / iOS / Web (WASM) plus servers, behind one API and one binary. No platform is dropped; roadmap staging orders GPU/NPU acceleration, not platform support itself. |
| `BR-04` | Usability from Unity and Godot: a C ABI, a single binary, an Apache-2.0 licence (so GPL is avoided), and IL2CPP / GDExtension compatibility. |

## FR-LD

Model loading.

| ID | What it governs |
|---|---|
| `FR-LD-01` | Loading GGUF files directly, with `mmap`-backed zero-copy weights so cold start does not pay a full read. |
| `FR-LD-02` | Reading and writing speech-specific metadata as Vokra's own `vokra.*`-prefixed GGUF chunks, chosen so they cannot collide with llama.cpp's own keys. |
| `FR-LD-03` | Treating the feature-frontend description (`vokra.frontend.*`) as a mandatory chunk that the runtime inspects, so a model's mel frontend is reproducible rather than re-derived. |
| `FR-LD-04` | Loading safetensors directly — upstream checkpoints only, no pickle. |
| `FR-LD-05` | **Permanent constraint.** ONNX is handled by the offline conversion tool *only*; the runtime ships no ONNX loader and no protobuf / abseil / onnx dependency. |
| `FR-LD-06` | Loading Silero VAD as a dedicated, 1:1-preserved subgraph instead of rewriting it into generic operators. |
| `FR-LD-07` | Loading quantized weights (the K-quant family) straight from GGUF. |

## FR-EX

Execution engine and IR.

| ID | What it governs |
|---|---|
| `FR-EX-01` | Vokra having its own IR — an audio graph descriptor — rather than adopting an external graph format. |
| `FR-EX-08` | **Permanent constraint.** Every backend guarantees the same operator coverage. An operator a backend cannot run is an **explicit error**; silent CPU fallback is not the default. This is the most-cited ID in the repository. |
| `FR-EX-10` | Keeping sampler, beam search and CFG as runtime functions rather than baking them into the model graph, so changing decode settings needs no reconversion. |

## FR-OP

Speech operators. Vokra implements these natively rather than composing them
from generic tensor ops.

| ID | What it governs |
|---|---|
| `FR-OP-10` | The `hifigan_generator` vocoder operator, including the conditions under which reduced precision is allowed. |
| `FR-OP-11` | The `bigvgan_generator` operator, reimplemented from the paper because the reference implementation is not commercially licensed. |
| `FR-OP-12` | The `vocos_head` operator, kept distinct from an iSTFTNet head, with a minimum precision the configuration cannot lower. |
| `FR-OP-13` | The `snake_activation` operator and its internal-precision attribute. |
| `FR-OP-31` | FSQ-family codec operators (`wavtokenizer_vq`, `xcodec2_fsq`), implemented as a subgraph separate from RVQ because their cost profile differs. |
| `FR-OP-32` | How EnCodec is treated: the engine supports the operator, but the weights are kept out of the official model zoo. |
| `FR-OP-40` | The `beam_search` operator — beam width, length normalisation, early stopping, n-best output and word-level timestamps — provided as a host-side function. |
| `FR-OP-41` | The `ctc_decode` operator, including language-model fusion and hotword boosting. |
| `FR-OP-42` | The `rnnt_decode` operator and its selectable decoding strategies. |
| `FR-OP-60` | The `aec` (acoustic echo cancellation) operator and the runtime-managed, time-tagged reference-signal queue it needs — a prerequisite for full-duplex speech-to-speech. |
| `FR-OP-61` | The `denoise` (speech enhancement) operator. |
| `FR-OP-62` | The `agc` and `hpf` operators of the capture-side audio pipeline. |
| `FR-OP-63` | The `loudness_norm` operator (LUFS / EBU R128). |
| `FR-OP-80` | The `speaker_encode` operator — one API over several speaker-embedding architectures. It stays in the core runtime because zero-shot TTS depends on it. |
| `FR-OP-81` | The `speaker_verify` operator (similarity-based verification). |
| `FR-OP-82` | The `diarize` operator, behind an optional feature flag. |
| `FR-OP-93` | Evaluation metrics (mel loss, UTMOS, DNSMOS, WER, CER) built into the runtime so quantization can be checked automatically. |

## FR-BE

Backends. Only the IDs cited publicly are listed; each backend has its own.

| ID | What it governs |
|---|---|
| `FR-BE-01` | The CPU backend as a first-class backend, with a runtime ISA-dispatch ladder spanning x86-64, ARM64, RISC-V and WASM. |
| `FR-BE-05` | The WebGPU backend, written as a hand-rolled extern-import shim rather than through a binding crate, so the zero-dependency invariant holds. |
| `FR-BE-06` | Delegate-style NPU backends (Apple ANE via CoreML, Qualcomm Hexagon via QNN), reached through raw framework/dlopen FFI with no binding crate. |
| `FR-BE-09` | A critical-safe build SKU that compiles out vendor GPU/NPU paths and states the result in the SBOM. |

## FR-MD

Model support.

| ID | What it governs |
|---|---|
| `FR-MD-02` | Whisper base (ASR), natively reimplemented — encoder, decoder and beam search. |
| `FR-MD-09` | Moshi (full-duplex speech-to-speech), whose weights require an attribution-display capability. |
| `FR-MD-10` | F5-TTS and Fish-Speech: engine support only, with the non-commercial weights separated behind a research flag. |
| `FR-MD-13` | **Permanent process.** A PR that adds model support must update the licence audit and clear the legal-compliance checklist in the same PR. See [CONTRIBUTING.md](../CONTRIBUTING.md) §4. |

## FR-QT

Quantization.

| ID | What it governs |
|---|---|
| `FR-QT-02` | Per-layer quantization policy being config-driven rather than hard-coded. |
| `FR-QT-03` | Enforcing a minimum precision for operators that do not survive aggressive quantization, which configuration may not lower. |
| `FR-QT-04` | Making post-quantization quality verification a standard pipeline stage that flags degradation automatically. |
| `FR-QT-05` | Quantization of the KV cache. |

## FR-SV

Server and API compatibility.

| ID | What it governs |
|---|---|
| `FR-SV-01` | Shipping `vokra-server` as a separate single binary, with Docker optional rather than required. |
| `FR-SV-02` | An OpenAI-compatible audio-transcription endpoint, so existing clients work unchanged. |
| `FR-SV-04` | A piper-plus-compatible HTTP TTS endpoint, for the Home Assistant / Rhasspy ecosystem. |
| `FR-SV-05` | A Wyoming Protocol server implementation. |
| `FR-SV-06` | Serving concurrent sessions from a paged KV cache. |

## FR-ST

Streaming.

| ID | What it governs |
|---|---|
| `FR-ST-03` | Barge-in: an in-flight generation can be interrupted and its buffered audio flushed immediately. |

## FR-CP

Compliance.

| ID | What it governs |
|---|---|
| `FR-CP-01` | Watermarking of TTS / VC output by default, with opting out being explicit. See [docs/legal-compliance.md](legal-compliance.md) for current status. |
| `FR-CP-02` | Attaching and verifying C2PA manifests. |
| `FR-CP-03` | Making non-commercially-licensed weights loadable only through an explicit research flag, keeping them off the default path. |
| `FR-CP-05` | Exposing model provenance, licence and frontend description as GGUF metadata so downstream users can inspect them. |
| `FR-CP-06` | A compliance configuration API. |

## FR-TL

Tooling.

| ID | What it governs |
|---|---|
| `FR-TL-01` | The offline checkpoint-to-GGUF conversion tool — the only place ONNX handling is allowed to live (see `FR-LD-05`). |
| `FR-TL-02` | `vokra-cli`: `run`, `convert` and `bench`. |
| `FR-TL-03` | `vokra-eval`: running the quality metrics of `FR-OP-93` from one command. |
| `FR-TL-04` | The build scripts that generate the C header and the engine / binding packages. |
| `FR-TL-05` | **Retired.** This was an automated competitor-changelog workflow. It was withdrawn by maintainer decision and fully superseded by the manual quarterly review under `NFR-MT-05`. It is listed here only because the retirement is still referenced. |

## NFR-DS

Distribution and size.

| ID | What it governs |
|---|---|
| `NFR-DS-01` | A size budget for the core runtime binary, with a tighter one for mobile. |
| `NFR-DS-02` | **The zero-dependency invariant.** The runtime carries no protobuf / abseil / onnx and, in practice, no third-party crate at all: the resolved root `Cargo.lock` contains only first-party `vokra-*` crates, and static linking yields a single distributable file. Enforced by `scripts/check-zero-deps.sh` locally and in CI. The two sanctioned ways to add functionality without breaking it are documented in [CONTRIBUTING.md](../CONTRIBUTING.md) §7 and [architecture.md](architecture.md). |
| `NFR-DS-03` | The package channels Vokra is distributed through. |
| `NFR-DS-04` | Distributing models separately from the binary — one GGUF file carrying its own metadata. |

## NFR-PF

Performance. Entries name the workload being bounded; the numeric targets
themselves are part of the private specification.

| ID | What it governs |
|---|---|
| `NFR-PF-03` | The real-time-factor target for Whisper base on iOS. |
| `NFR-PF-04` | The real-time-factor target for Whisper large-v3 on CUDA. |
| `NFR-PF-05` | The server-side TTS latency target. |
| `NFR-PF-06` | The real-time-factor target for Whisper base on Android. |
| `NFR-PF-08` | The capability target for the Web (WASM / WebGPU) target. |
| `NFR-PF-11` | Cold start: `mmap`-based loading keeps model load time close to zero, which is the failure mode this project was started to avoid. |
| `NFR-PF-13` | The performance regression gate — RTF / TTFA / latency are measured per PR and a regression must be justified rather than merged silently. |

## NFR-QL

Numerical and audio quality.

| ID | What it governs |
|---|---|
| `NFR-QL-01` | Numerical parity against the PyTorch reference, verified per PR in CI. Per-model tolerances are stated in [`tests/parity/`](../tests/parity/) rather than being global constants — a tolerance is an architectural bound, not a knob to widen when CI is red. |
| `NFR-QL-02` | The bound on audio-quality degradation relative to the PyTorch reference. |
| `NFR-QL-04` | Nightly audio-quality regression runs over public evaluation subsets, where a threshold breach is treated as a blocking defect. |

## NFR-RL

Platform reliability constraints. These encode failure modes that are cheap to
avoid up front and very expensive to retrofit.

| ID | What it governs |
|---|---|
| `NFR-RL-03` | iOS forbidding dynamic library loading, hence static linking. |
| `NFR-RL-04` | The Android `StreamingAssets` jar-URL problem, hence a built-in extraction helper. |
| `NFR-RL-05` | No JIT anywhere (iOS W^X); acceleration comes from runtime dispatch only. |
| `NFR-RL-06` | A GPU backend incompatibility surfacing as an explicit error rather than a silent fallback — the backend-level counterpart of `FR-EX-08`. |
| `NFR-RL-07` | Memory safety: the core is Rust, `unsafe` and SIMD intrinsics are permitted inside operators, and API boundaries stay safe. The crates that may opt out are enumerated in [architecture.md](architecture.md). |

## NFR-MT

Maintenance, CI/CD and community process.

| ID | What it governs |
|---|---|
| `NFR-MT-01` | How development time is budgeted across engine work, CI, documentation, packaging, release engineering and community engagement — deliberately reserving a large share for the non-code work that starves in single-maintainer projects. |
| `NFR-MT-02` | The CI matrix tiering: which platforms run on every PR, which nightly, which weekly. |
| `NFR-MT-03` | The release process — release train, semantic versioning, changelog automation, reproducible builds and SBOM generation. |
| `NFR-MT-05` | The quarterly Go/No-go review of the project's exit criteria, built into the release process. It is the sole surviving monitoring mechanism after `FR-TL-05` was retired. |
| `NFR-MT-06` | Developing in the open: the repository is public from the start of implementation, and issues, PRs, CI results and benchmarks stay public so quality is externally verifiable. |
| `NFR-MT-07` | The CI quality gates: `main` is branch-protected and PR-only, with build, test, formatting, lint, numerical parity, performance regression, licence/vulnerability and documentation-example checks as required checks. The currently required set is listed in [CONTRIBUTING.md](../CONTRIBUTING.md) §2. |
| `NFR-MT-08` | Automated release publishing — release artefacts are built and published by CI, never hand-built. |

## NFR-LC

Dependency licensing.

| ID | What it governs |
|---|---|
| `NFR-LC-02` | The allowed dependency licences (Apache-2.0 / MIT / BSD family), the prohibition on GPL and LGPL, and the case-by-case treatment of MPL-2.0. |
| `NFR-LC-04` | Making a GPL/LGPL dependency a PR blocker through an automated licence check in CI. |

## NFR-LG

Legal and regulatory.

| ID | What it governs |
|---|---|
| `NFR-LG-01` | EU AI Act Article 50 — machine-readable marking of AI-generated audio, retained as a design requirement. Current implementation status is recorded in [docs/legal-compliance.md](legal-compliance.md); do not infer it from this line. |
| `NFR-LG-02` | California SB 942, the Tennessee ELVIS Act and the proposed NO FAKES Act, addressed through voice-cloning separation and consent handling. Status detail is likewise in [docs/legal-compliance.md](legal-compliance.md). |

## NFR-PT

Platform coverage.

| ID | What it governs |
|---|---|
| `NFR-PT-01` | Treating all platforms as a precondition: no mandatory dependency that exists on only one platform, and cross-build viability verified continuously in CI. Backend ordering stages acceleration, it does not select which platforms are supported. |
| `NFR-PT-02` | The breadth of CPU support, expressed as the instruction-set baseline assumed on x86-64 and ARM64. |

## IF

External interfaces.

| ID | What it governs |
|---|---|
| `IF-01` | The C ABI consumer interface — a single `include/vokra.h`, opaque handles, thread-local error state, and the ABI stability commitment. This is the interface Unity, Godot, Swift, Kotlin, Python and JS bindings all sit on, which is why it is the second-most-cited ID here. |
| `IF-05` | The piper-plus HTTP / Home Assistant interface (see `FR-SV-04`, `FR-SV-05`). |
| `IF-07` | The GGUF ecosystem interface: conformant GGUF plus `vokra.*`-prefixed chunks that cannot collide with llama.cpp's own keys. |

---

## Related reading

- [architecture.md](architecture.md) — crate map, execution model and the
  design red lines, for readers who want the structure rather than the
  vocabulary.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — pull requests, required checks,
  dependency policy and the red lines as review rules.
