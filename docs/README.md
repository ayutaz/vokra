# Vokra documentation

**Current-state review:** 2026-08-30

**Reviewed implementation baseline:** GitHub `main` at
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`; the pre-documentation code
baseline was branch `feat/mac-cpu-metal-full-coverage-2026-08-28` at
`c64b7b7237b70c5dc70ffd60394af325016d9a8d`.

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
| Model and dependency licensing | [Licence audit](license-audit.md) |
| Deployment policy and legal notes | [Legal compliance](legal-compliance.md) |
| C ABI changes | [ABI changelog](abi-changelog.md) |
| Release history | [`CHANGELOG.md`](../CHANGELOG.md) |

Platform tutorials are available for Android, iOS, Unity, Godot, Python, web,
and the server in English and Japanese under [`tutorials/`](tutorials/).

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

The workspace version is `0.2.0` development; no Git tag or published release
exists yet. Rust APIs, the C ABI, GGUF metadata, and the model roster remain
pre-1.0 and may change. The C header and Python prototype table are checked for
exact function-set equality; documentation therefore avoids copying a function
count that would drift on the next ABI addition.

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
