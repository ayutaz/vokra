# Vokra documentation

Start with the current guides below. **Last reviewed: 2026-10-23 UTC**,
against accepted main `0d51bb01e56da6ce5f72375cc5fdddc599beac3c`.

[Current project status](current-status.md) / [現状と残作業（日本語）](current-status.ja.md)
is the canonical current main, release, PR, catalog and security readback.
The [complete documentation audit](handoff/documentation-refresh-2026-10-23.md)
records coverage and verification limits.

The previous index's dated checkpoints are preserved in
[the archived index](handoff/documentation-index-checkpoints-through-2026-10-22.md).
They are historical observations, not current instructions or clean-HEAD claims.

## Start here

| Need | Document |
|---|---|
| Current status and remaining work | [English](current-status.md) / [日本語](current-status.ja.md) |
| Published CPU preview installation | [v0.4.0-rc.1 guide](release-0.4.0-rc1.md) / [日本語](release-0.4.0-rc1.ja.md) |
| Install, build, first inference | [Getting started](getting-started.md) / [日本語](getting-started.ja.md) |
| Desktop command line | [CLI tutorial](tutorials/cli.md) / [日本語](tutorials/cli.ja.md) |
| Rust, C and binding surfaces | [API reference](api-reference.md) / [日本語](api-reference.ja.md) |
| Runtime and crate design | [Architecture](architecture.md) / [日本語](architecture.ja.md) |
| CPU and accelerators | [Backend guide](backend-guide.md) / [日本語](backend-guide.ja.md) |
| Move from other runtimes | [Migration guide](migration-guide.md) / [日本語](migration-guide.ja.md) |
| Requirements vocabulary | [Requirement IDs](requirement-ids.md) / [日本語](requirement-ids.ja.md) |
| Contributor workflow | [CONTRIBUTING](../CONTRIBUTING.md) and [good first tasks](good-first-tasks.md) / [日本語](good-first-tasks.ja.md) |
| Community conduct | [Code of Conduct](../CODE_OF_CONDUCT.md) / [日本語](../CODE_OF_CONDUCT.ja.md) |
| Vulnerability reporting | [Security Policy](../SECURITY.md) / [日本語](../SECURITY.ja.md) |
| Model/dependency licensing | [License audit](license-audit.md) |
| Deployment/legal boundaries | [Legal compliance](legal-compliance.md) |
| Release and ABI changes | [Changelog](../CHANGELOG.md) / [C ABI changes](abi-changelog.md) |

## Platforms and bindings

English/Japanese tutorials cover [Android](tutorials/android.md),
[iOS](tutorials/ios.md), [Unity](tutorials/unity.md), [Godot](tutorials/godot.md),
[Python](tutorials/python.md), [web](tutorials/web.md) and
[server deployment](tutorials/server.md); each tutorial links its twin.
The [platform matrix](platform-support/v1.0-rc-support-matrix.md) separates
source support, published payloads and owner/device verification.

Use the binding's source documentation for its exact publication state:
[Python](../bindings/python/README.md),
[Unity](../bindings/unity/com.vokra.unity/README.md),
[Godot](../integrations/vokra-godot/README.md),
[Swift/XCFramework](../README-swift-package.md),
[web](../web/pkg/README.md) and
[isolated server](../integrations/vokra-server/README.md).
Workspace version `0.4.0-rc.1` does not mean that every package has a new
published release. The stable `v0.3.0` binding assets and CPU-only desktop RC
are distinct distributions.

## Reading model status correctly

Conversion, GGUF binding, native forward, independent numerical parity,
Apple CPU/Metal/no-fallback and publication are separate stages.
The [fresh complete inventory](handoff/model-inventory-2026-10-23.tsv) is
metadata/card-only: 194 public repositories, 136 code/artifact-full and 58
non-full. These are not full-catalog hardware passes.

Use source and generated contracts when prose disagrees:
the [CLI routing source](../crates/vokra-cli/src/engine.rs),
[C header](../include/vokra.h), Python generated prototype table, per-model
manifests, independent reference evidence and gated publication scripts.
CLI `convert --help` and `run --help` describe interfaces on an authorized
host; documentation checks do not execute the models.

For the Mac campaign, use the
[execution plan](handoff/mac-cpu-metal-execution-plan-2026-09-07.md),
[remaining-task ledger](handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md),
[frozen owner packet](handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md)
and [dated Apple results](handoff/mac-cpu-metal-scaleway-results-2026-09-11.md)
together with the current status. The frozen 63-row decision scope is not
today's 58-row public classification. BF16, HiFTNet, all four BigVGAN variants,
non-Apple prerequisites, real-weight CPU and final packets remain explicit
requirements before the final Scaleway stage.

## Protocols, design and governance

| Need | Document |
|---|---|
| Streaming C ownership/backpressure | [Streaming codec C API](c-api-streaming-codec.md) |
| NanoCodec conversion | [Conversion contract](nanocodec-conversion.md) |
| FireRed PCM beam composition | [Design](design/firered-pcm-beam-composite.md) |
| Kyutai PCM / incremental LM | [PCM design](design/kyutai-stt-pcm-composite.md) / [streaming LM](design/kyutai-stt-streaming-lm.md) |
| Mimi provenance / metadata | [Checkpoint provenance](design/mimi-checkpoint-provenance.md) / [Rust metadata binding](design/mimi-rust-core-metadata-binding.md) |
| NPU comparison protocol | [NPU bakeoff protocol](handoff/npu-bakeoff-protocol.md), not a measured result |
| Quarterly owner reviews | [Runbook](governance/quarterly-review-runbook.md) / [records index](governance/quarterly-reviews/README.md) |
| Go/No-go and GA judgment | [Go/No-go template](governance/vokra-go-nogo-v0.5.md) / [DoD template](governance/dod-judgment-template.md) |
| Security remediation | [Dated security plan](handoff/security-remediation-2026-09-21.md), with current routing at the top |

## Current sources versus dated records

- `handoff/`: dated branch/campaign evidence. Current routing is this index
  and `current-status.md`, not whichever handoff has “current” in its old title.
- `bench-baselines/`, `benchmarks/`, `perf/`: measurements for their named
  hardware, commits, flags and fixtures. Start with the
  [baseline provenance index](bench-baselines/README.md); do not restamp measurements.
- `design/`: design contracts and explicitly dated implementation notes;
  planned behavior is not necessarily implemented behavior.
- Governance templates retain blank owner fields; static checks are not a verdict.
- Ignored `adr/`, `_research/`, `superpowers/` and maintainer planning files
  are not automatically tracked public documentation. Do not force-add them.

The [2026-10-23 audit](handoff/documentation-refresh-2026-10-23.md) links the
per-file coverage records. Earlier audits and checkpoint failures remain dated;
no sign-off, numerical bound, release, provider inventory or public upload is
created by updating documentation.

## Documentation conventions

Python recipes use Python 3.12 through `uv run` / `uv sync`.
Public entry guides and platform tutorials keep English/Japanese twins.
Owner-approved remote environments handle model and workspace-scale work;
the maintainer Mac runs only safe static/focused checks. Credentials never
belong in documentation.

## Lightweight validation

These commands inspect documentation; they do not build the workspace or
run `vokra-models`:

```sh
uv run --no-project --python 3.12 python tools/docs/check_doc_examples.py --self-test
uv run --no-project --python 3.12 python tools/docs/check_doc_examples.py
scripts/check-doc-references.sh --self-test
scripts/check-doc-references.sh
scripts/check-runbook-path-citations.sh
scripts/check-community-docs.sh
scripts/check-workflow-hygiene.sh
git diff --check -- . ':(exclude)tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
```

The example checker covers its declared entry/tutorial set, not every fenced
block in every file. Its Swift/C#/Godot/UI Tier-C cases remain explicitly
unverified without their toolchains. Static links and source inspection are
not native/device or legal verification.
