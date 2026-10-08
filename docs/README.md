# Vokra documentation

**Current-state review:** 2026-10-08 JST

**Read-only external baseline:** GitHub `main` at
`d100d93778191ccab77bd1c37fe3552e3d889758` (API read on 2026-10-08). The
documentation branch starts at that exact checkout; unmerged PRs and their CI
are not part of this baseline. The detailed dated snapshot, including its
source boundaries and historical evidence, is retained in the
[documentation audit](handoff/documentation-refresh-2026-10-08.md#documentation-index-snapshot-retained-from-before-consolidation).

The public catalog is not fully Apple-complete. Conversion, binding, native
forward, independent parity, Apple CPU/Metal/no-fallback verification, and
publication remain separate gates; VAST or accepted hardware evidence applies
only to the exact rows and heads named in its dated record and does not
authorize an artifact upload by itself.

This directory contains public guides, generated-surface pointers, design
decisions, validation evidence, and dated engineering records. Start with the
guides below; use handoff and benchmark files only for the commit and
environment they name.

## Start here

| Need | Document |
|---|---|
| Install, build, and first inference | [Getting started](getting-started.md) / [日本語](getting-started.ja.md) |
| Desktop command line | [CLI tutorial](tutorials/cli.md) / [日本語](tutorials/cli.ja.md) |
| Rust, C, and binding surfaces | [API reference](api-reference.md) / [日本語](api-reference.ja.md) |
| Runtime and crate design | [Architecture](architecture.md) / [日本語](architecture.ja.md) |
| CPU and accelerator behavior | [Backend guide](backend-guide.md) / [日本語](backend-guide.ja.md) |
| Move from ONNX Runtime, whisper.cpp, or sherpa-onnx | [Migration guide](migration-guide.md) / [日本語](migration-guide.ja.md) |
| Platform examples | [`tutorials/`](tutorials/) |
| Contributor workflow | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Community conduct | [Code of Conduct](../CODE_OF_CONDUCT.md) / [日本語](../CODE_OF_CONDUCT.ja.md) |
| Vulnerability reporting | [Security Policy](../SECURITY.md) / [日本語](../SECURITY.ja.md) |
| Current dependency-security remediation | [2026-09-21 Dependabot and Scorecard plan](handoff/security-remediation-2026-09-21.md) |
| Model and dependency licensing | [Licence audit](license-audit.md) |
| Deployment policy and legal notes | [Legal compliance](legal-compliance.md) |
| C ABI changes | [ABI changelog](abi-changelog.md) |
| Release history | [`CHANGELOG.md`](../CHANGELOG.md) |
| Current Mac CPU/Metal campaign | [2026-09-11 Apple results](handoff/mac-cpu-metal-scaleway-results-2026-09-11.md), the [residual execution ledger](handoff/mac-cpu-metal-residual-execution-2026-09-12.md), and the [remaining-task ledger](handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md) (the live audit has 58 unresolved rows; the dated ledger retains its historical 63-row baseline) |
| Public catalog and security completion | [2026-09-29 execution plan](handoff/public-catalog-security-completion-2026-09-29.md) (dated starting counts; re-run the live audits for current values) |
| Whole-document refresh coverage and limitations | [2026-10-08 documentation audit](handoff/documentation-refresh-2026-10-08.md); the [2026-10-04 audit](handoff/documentation-refresh-2026-10-04.md) preserves the previous review |

Platform tutorials are available for Android, iOS, Unity, Godot, Python, web,
and the server in English and Japanese under [`tutorials/`](tutorials/).

## Additional references by reader

These documents are useful entry points but are not all current status pages.
Design records and dated handoffs retain the scope, source, and evidence of the
commit or campaign they name.

### Current protocol and runbooks

| Need | Document |
|---|---|
| Shared NPU bakeoff rules and report boundaries (protocol, not an execution result) | [NPU bakeoff protocol](handoff/npu-bakeoff-protocol.md) |

### API, conversion, and design

| Need | Document |
|---|---|
| Streaming C API ownership and backpressure | [Streaming codec decoder C API](c-api-streaming-codec.md) |
| NanoCodec conversion contract | [NanoCodec conversion](nanocodec-conversion.md) |
| FireRed PCM beam composition | [FireRed PCM design](design/firered-pcm-beam-composite.md) |
| Kyutai STT PCM boundary | [Kyutai STT PCM design](design/kyutai-stt-pcm-composite.md) |
| Kyutai incremental language model | [Kyutai STT streaming-LM design](design/kyutai-stt-streaming-lm.md) |
| Mimi checkpoint provenance | [Mimi checkpoint provenance](design/mimi-checkpoint-provenance.md) |
| Mimi Rust metadata binding | [Mimi metadata binding](design/mimi-rust-core-metadata-binding.md) |

### Governance

| Need | Document |
|---|---|
| GA judgment record template | [GA Definition-of-Done judgment template](governance/dod-judgment-template.md) |

### Dated benchmark and historical evidence

| Evidence | Document |
|---|---|
| M5-14 Wave-0 per-model hot-spot measurements | [Hot-spot tables](bench-baselines/m5-14-wave0-2026-07-18/hotspot-tables.md) |
| Audit campaign follow-up | [2026-08-14 audit follow-up](handoff/audit-followup-2026-08-14.md) |
| Codex operations handoff | [2026-08-28 Codex operations](handoff/codex-operations-2026-08-28.md) |
| Fun-CosyVoice3 dependency-route decision | [2026-09-08 CosyVoice3 route](handoff/cosyvoice3-soxr-route-2026-09-08.md) |
| Coverage-audit Wave A | [2026-08-03 coverage audit](handoff/coverage-audit-2026-08-03-wave-a.md) |
| Readback branch closeout | [2026-10-04 branch closeout](handoff/readback-branch-closeout-2026-10-04.md) |
| SBV2 parity owner handoff | [2026-08-11 SBV2 handoff](handoff/sbv2-parity-owner-handoff-2026-08-11.md) |
| X-06 nightly-benchmark operations | [X-06 owner handoff](handoff/x-06.md) |

## Reading model status correctly

Model support has separate stages: offline conversion, GGUF binding, native
forward execution, independent numerical parity, and publication. A model at
one stage must not be described as complete at a later stage.

Use the live sources for current answers:

- `vokra-cli convert --help` lists accepted converter identifiers;
- `vokra-cli run --help` lists CLI-routed inputs, outputs, and backends;
- [`crates/vokra-cli/src/engine.rs`](../crates/vokra-cli/src/engine.rs) records
  routed architectures and explicit deferred operations;
- the [Vokra model hub](https://huggingface.co/vokra) contains only published
  artifacts and their model cards;
- parity tests and fixtures provide architecture-specific numerical evidence.

The generated source or checker wins when prose disagrees with it:
[`include/vokra.h`](../include/vokra.h) for the C ABI, the generated Python
prototype table for Python FFI coverage, model manifests for tensor contracts,
and the publication scripts for release eligibility.

## Current release posture

Version `0.3.0` is the first GitHub-only source release. External package
registries remain explicitly disabled. Rust APIs, the C ABI, GGUF metadata,
and the model roster remain pre-1.0 and may change. The C header and Python
prototype table are checked for exact function-set equality; documentation
therefore avoids copying a function count that would drift on the next ABI
addition.

The exact release scope, GitHub-only channel decision, security-alert triage
and final gate list are recorded in the
[2026-09-17 release-preparation record](handoff/release-preparation-0.3.0-2026-09-17.md).

The default runtime keeps the root `Cargo.lock` first-party-only. GPU and NPU
features are opt-in, and unsupported operations must fail explicitly instead
of silently running on CPU. Model licences remain separate from the
Apache-2.0 source licence; consult the licence audit and each model card before
redistribution.

## Current sources versus dated records

The following directories preserve useful evidence but are not live status
pages:

- `handoff/` — branch- or campaign-specific transfer notes;
- `bench-baselines/`, `benchmarks/`, and `perf/` — measurements for named
  hardware, commits, flags, and fixtures;
- `adr/` — decisions at the status and date written;
- `_research/` — initial research snapshots;
- `superpowers/` — implementation plans and specifications.

Do not rewrite old measurements or historical test counts to resemble a new
head. Add a supersession note or a newer report. Local maintainer planning
files may be intentionally gitignored; public instructions come from
`AGENTS.md`, `CONTRIBUTING.md`, the tracked guides, and repository checks.

## Documentation conventions

- Executable Python recipes use Python 3.12 through `uv run` or `uv sync`.
- Public entry guides and platform tutorials keep English/Japanese twins.
  Audits, ADRs, benchmarks, and dated handoffs may be single-language.
- Large-model conversion and workspace-scale verification follow the VAST
  workflow; user-facing focused build examples remain valid on normal hosts.
- Credentials never belong in documentation or committed command examples.

## Lightweight validation

These checks validate documentation without compiling the workspace or
`vokra-models`:

```sh
uv run --no-project --python 3.12 python \
  tools/docs/check_doc_examples.py --self-test
uv run --no-project --python 3.12 python \
  tools/docs/check_doc_examples.py
scripts/check-doc-references.sh --self-test
scripts/check-doc-references.sh
scripts/check-runbook-path-citations.sh
scripts/check-community-docs.sh
scripts/check-workflow-hygiene.sh
git diff --check
```

`check-community-docs.sh` requires the English/Japanese Code of Conduct and
Security Policy pairs and validates their relative links and heading parity.
