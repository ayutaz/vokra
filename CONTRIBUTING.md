# Contributing to Vokra

Thank you for considering a contribution. Vokra is developed fully in the
open, and every change — including changes by the maintainer — goes through
the same pull-request and CI pipeline described below.

The design documents under `docs/` define requirements and scope; public entry
guides have English/Japanese twins while audit and handoff records may be
single-language. Requirement IDs (BR / FR / NFR) referenced below are indexed
in `docs/requirement-ids.md` and its Japanese twin.

## 1. Pull requests and branch protection

- `main` is protected: **direct pushes are not allowed**. Every change is
  made through a pull request and must pass CI before merging.
- Link each PR to the issue / work package (WP) it implements, so that the
  change is traceable to requirement IDs.
- The project currently runs with a single maintainer, so the required
  approving-review count may be 0 at this stage — the CI gates below are
  the non-negotiable blocker, not review count.

## 2. CI required checks

As verified through the GitHub branch-protection API on 2026-08-20, every PR
must pass **15 required status contexts**:

| Check | What it runs |
|---|---|
| `build (ubuntu-latest)`, `build (macos-latest)`, `build (windows-latest)` | `cargo build --release` on each OS |
| `test (ubuntu-latest)`, `test (macos-latest)`, `test (windows-latest)` | `cargo test --workspace` on each OS |
| `fmt` | `cargo fmt --all -- --check` |
| `clippy` | `cargo clippy --all-targets -- -D warnings` |
| `parity` | `cargo test -p vokra-parity` — numerical parity harness against reference implementations (`tests/parity/`) |
| `license` | `cargo deny check licenses advisories bans` + `cargo audit`, then the repository invariant gates under `scripts/`: zero-dependency, forbidden symbols, `no_std` subset, EnCodec weight exclusion, workflow hygiene, the converter ⇄ binder architecture handshake, bound-arch registry completeness, and the citation gates (`vokra_ops::`, `vokra_<crate>::`, parity sidecars, runbook paths). Each runs its own `--self-test` first, so a gate that has stopped being able to see a defect fails before it reports on your PR. |
| `workflow-security` | actionlint + ShellCheck + zizmor over every workflow |
| `dependency-review` | dependency license/vulnerability review plus OpenSSF Scorecard visibility for changed dependencies |
| `documentation-links` | lychee link validation for the public documentation surface |
| `CodeQL` | GitHub CodeQL Rust `security-extended` analysis |
| `pins.yaml ↔ workflow sync` | Bidirectional consistency between `.github/pins.yaml` and workflow pin literals |

Run lightweight equivalents locally and use CI or an adequately sized remote
host for the complete matrix. On the maintainer's 16 GB Mac, workspace-wide
Cargo and every `-p vokra-models` Cargo invocation are VAST-only. Ten core
contexts live in `.github/workflows/ci.yml`; the four security contexts live
in `.github/workflows/ci-security.yml` and `.github/workflows/codeql.yml`, and
the pin-catalog context lives in `.github/workflows/pins-sync-check.yml`.
The advisory checks were split out on 2026-07-23 into
`.github/workflows/ci-quality.yml` (lint / audit / doc-drift / API-compat) and
`.github/workflows/ci-platform.yml` (platform build targets / GPU backends /
regression gate). `.github/workflows/README.md` is the index of which job lives
where.

Beyond the 15 required contexts, CI also runs a **`gpu-backends`** job
(in `.github/workflows/ci-platform.yml`) that
keeps the optional `metal` / `cuda` GPU backends compiling and lint-clean
(`cargo build`/`clippy`/`test -p vokra-models -p vokra-cli --features
metal|cuda`). The `metal` leg runs its GPU parity tests on the Apple-silicon
macOS runner; the `cuda` leg is build/lint-only (GitHub runners have no NVIDIA
GPU, so the dlopen-probe-gated device tests skip cleanly). Both are
first-party `vokra-*` crates, so this does not affect the zero-dependency
invariant.

Performance-regression, documentation-example, rustdoc, platform, real-weight
parity, and nightly audio-quality jobs already run as advisory checks. They are
not branch-protection contexts today; promotion requires an owner decision
after stable green runs. The exact required/advisory split is maintained in
`.github/workflows/README.md`.

## 3. Dependency license policy

- **Allowed**: Apache-2.0, MIT, BSD-family licenses only.
- **Forbidden**: GPL and LGPL in any form — Vokra targets Unity / Godot and
  other proprietary embedding scenarios where (L)GPL is not acceptable.
- **MPL-2.0** (e.g. symphonia): limited use only, after evaluating the
  file-level copyleft implications case by case (see
  [docs/license-audit.md](docs/license-audit.md)).
- `cargo-deny` runs in CI and is a **PR blocker**: a PR that introduces a
  GPL/LGPL dependency cannot merge (NFR-LC-04).
- Keep new dependencies minimal and justified; prefer std / existing
  workspace code over adding a crate.

## 4. Adding support for a new model

A PR that adds model support must:

1. **Update [docs/license-audit.md](docs/license-audit.md)** in the same PR
   (license of code *and* weights, commercial usability, training-data
   provenance).
2. Respect the model-zoo policy: weights under **CC-BY-NC / CC-BY-NC-SA or
   with unclear training-data rights are excluded from the official model
   zoo** and may only be exercised behind an explicit research flag
   (engine support without weight distribution).
3. For TTS / VC models, go through the
   [docs/legal-compliance.md](docs/legal-compliance.md) checklist
   (EU AI Act Article 50 / California SB 942: AudioSeal watermarking ON by
   default, C2PA manifest support, disclosure requirements).
4. Update [NOTICE](NOTICE) when the addition carries attribution or
   distribution-relevant terms.

## 5. Design red lines

The following are fixed design decisions. PRs that cross them will be
declined regardless of implementation quality:

- **No ONNX graph loading in the runtime.** ONNX models are handled
  exclusively by the offline conversion tool; the runtime must stay free of
  onnxruntime / onnx / protobuf dependencies (FR-LD-05).
- **No onnxruntime in the piper-plus inference path.** The MB-iSTFT-VITS2
  inference stack is natively reimplemented in Rust (maintainer decision,
  2026-07-02); only the G2P text preprocessing is reused from piper-plus
  for the time being.
- **No eSpeak-NG** (GPL-3.0) in the core. G2P comes from piper-plus's own
  MIT implementation or IPA-dictionary-based approaches.
- **No NNAPI backend** (deprecated by Google as of Android 15).
- **No soxr / rubberband** (GPL). Resampling is a native implementation
  based on the speexdsp (BSD) resampler design.

## 6. Finding something to work on

Issues labeled **`good first issue`** are curated to be self-contained
entry points with clear acceptance criteria. If you want to take a larger
work package, comment on the corresponding WP issue first so scope can be
agreed before you invest time.

## 7. Local development hooks (recommended)

Vokra ships version-controlled git hooks under `.githooks/` (no external
hook manager — consistent with the zero-dependency policy). Activate them
once per clone:

```
bash scripts/install-git-hooks.sh   # sets core.hooksPath -> .githooks
```

- **pre-commit** (fast, no compile): `cargo fmt --all -- --check`,
  `scripts/check-forbidden-symbols.sh`, `scripts/check-zero-deps.sh`,
  `scripts/check-fixture-eol-pins.sh`, `scripts/compliance/lint-pipefail-grep-q.py`.
- **pre-push** (compiling deep path):
  `scripts/compliance/test-nvidia-scanner-sigpipe.sh` (always),
  `cargo clippy --all-targets -- -D warnings`,
  `cargo test --workspace` (or `cargo nextest run --workspace` when
  `cargo-nextest` is installed — the hook falls back to plain `cargo test`
  when it is missing). Doc-tests are CI-owned, not silently substituted by
  nextest.

**Fast-paths for iteration speed.** The pre-push hook classifies the diff
since the tracking upstream (or `origin/main` for brand-new branches). When
every file changed is Rust-build-neutral documentation/config, approved
`tools/parity` Python/uv sidecars, fixture hash sidecars, publish helpers, or
Claude hook helpers, the clippy + test legs are **skipped**; the compliance
scanner still runs. Any Rust/build input, general script/tool/test,
integration, hook self-change, or unrecognised extension returns to the deep
path. A deletion-only remote ref update also runs compliance and then skips
Cargo; mixed or malformed updates do not. On the maintainer Mac, a deep path
refuses before Cargo and directs the run to VAST. After a recorded green VAST
verification, use `VOKRA_SKIP_HOOKS=1` for that code push. Force the full path
on a capable approved host with `VOKRA_HOOK_DEEP=1`; the explicit local escape
hatch is `VOKRA_ALLOW_LOCAL_HEAVY=1`. The classifiers live in
`.githooks/lib-fastpath.sh` and are pinned by
`scripts/test-pre-push-fastpath.sh` (50 classifier/integration cases).

Uninstall with `git config --unset core.hooksPath`.

`scripts/check-zero-deps.sh` enforces the **zero-external-dependency**
invariant (NFR-DS-02): `Cargo.lock` must contain only first-party `vokra-*`
crates. This is stricter than `cargo deny` and is a hard local + CI gate.

Two patterns add functionality without breaking this invariant — they are the
only sanctioned ways to reach outside the runtime graph:

- **First-party optional features.** The GPU backends `vokra-backend-metal` /
  `vokra-backend-cuda` are ordinary `vokra-*` crates (hand-written raw FFI —
  no `metal` / `objc2` / `cudarc` binding crate), gated OFF by default behind
  the `metal` / `cuda` Cargo features so default (and Linux / Windows / WASM)
  builds never even name them. Adding a GPU/NPU path this way keeps
  `Cargo.lock` vokra-only.
- **Isolated integration workspaces.** Code that genuinely needs an external
  crate (e.g. the real 8-language G2P in `integrations/vokra-piper-g2p`, which
  pulls non-`vokra-*` crates) lives in its own workspace under `integrations/`
  with its own `Cargo.lock`, excluded from the root workspace, and is wired in
  across a trait boundary (`vokra_piper_plus::Phonemizer`) — never linked into
  the runtime graph checked here.

### Codex and Claude Code

Codex is the primary agent for this repository. Codex reads the committed
`AGENTS.md`, discovers reusable workflows under `.agents/skills/`, and loads
the repository policy hooks from `.codex/hooks.json` after they are reviewed
and trusted with `/hooks`.

- **Codex hooks** keep Rust edits formatted, re-assert the zero-dependency
  invariant after Cargo metadata edits, block `cargo add` and bare pip/conda
  mutations/direct Python, and route workspace-scale Cargo, all
  `vokra-models` Cargo, or 2 GB+ model work to VAST.
- **Codex skills** encode the recurring policy-heavy workflows:
  `add-speech-model`, `add-audio-operator`, `numerical-parity`,
  `license-audit`, `publish-model-to-hf`, and `vast-ai-workflow`.
- **Claude Code compatibility** remains available through the committed
  `.claude/settings.json` and `.claude/skills/`; its legacy hook scripts live
  in `scripts/claude-hooks/`. New Codex behavior must not depend on Claude-only
  environment variables or lifecycle events.

Machine-local approval settings are managed by Codex configuration and are
not committed to the repository. Do not add credentials or personal overrides
to the project policy files.

The current operational baseline and branch-retirement history are summarized
in `docs/handoff/codex-operations-2026-08-18.md`.
