# Contributing to Vokra

Thank you for considering a contribution. Vokra is developed fully in the
open, and every change — including changes by the maintainer — goes through
the same pull-request and CI pipeline described below.

The design documents under `docs/` (currently Japanese, Draft status) are
the source of truth for requirements and scope; requirement IDs (BR / FR /
NFR) referenced below are defined there.

## 1. Pull requests and branch protection

- `main` is protected: **direct pushes are not allowed**. Every change is
  made through a pull request and must pass CI before merging.
- Link each PR to the issue / work package (WP) it implements, so that the
  change is traceable to requirement IDs.
- The project currently runs with a single maintainer, so the required
  approving-review count may be 0 at this stage — the CI gates below are
  the non-negotiable blocker, not review count.

## 2. CI required checks

As of M0 (v0.1 spike) every PR must pass **6 required checks**:

| Check | What it runs |
|---|---|
| `build` | `cargo build --release` on Linux / macOS / Windows |
| `test` | `cargo test --workspace` on Linux / macOS / Windows |
| `fmt` | `cargo fmt --all -- --check` |
| `clippy` | `cargo clippy --all-targets -- -D warnings` |
| `parity` | `cargo test -p vokra-parity` — numerical parity harness against reference implementations (`tests/parity/`) |
| `license` | `cargo deny check licenses advisories bans` + `cargo audit` + `scripts/check-forbidden-symbols.sh` |

Run the same commands locally before pushing; the CI configuration is
`.github/workflows/ci.yml`.

Beyond the six required checks, CI also runs a **`gpu-backends`** job that
keeps the optional `metal` / `cuda` GPU backends compiling and lint-clean
(`cargo build`/`clippy`/`test -p vokra-models -p vokra-cli --features
metal|cuda`). The `metal` leg runs its GPU parity tests on the Apple-silicon
macOS runner; the `cuda` leg is build/lint-only (GitHub runners have no NVIDIA
GPU, so the dlopen-probe-gated device tests skip cleanly). Both are
first-party `vokra-*` crates, so this does not affect the zero-dependency
invariant.

Planned extension (M1 and later, completing the NFR-MT-07 set): a
performance-regression gate (5% threshold) and execution checks for code
examples in documentation will be added as required checks, and nightly
audio-quality threshold violations will block or revert the offending PR.

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
- **pre-push** (compiling, mirrors CI):
  `scripts/compliance/test-nvidia-scanner-sigpipe.sh` (always),
  `cargo clippy --all-targets -- -D warnings`,
  `cargo test --workspace` (or `cargo nextest run --workspace` + `cargo
  test --workspace --doc` when `cargo-nextest` is installed locally —
  ~60% faster on the workspace test leg; the hook falls back to plain
  `cargo test` when it is missing, never a hard error).

**Fast-paths for iteration speed.** The pre-push hook classifies the diff
since the tracking upstream (or `origin/main` for brand-new branches). When
every file changed is documentation-shape (`docs/**`, `.github/**`,
`*.md`, `*.yml` / `*.yaml`, `include/*.h`, root dotfiles / `LICENSE` /
`NOTICE` / `README` / `CONTRIBUTING` / `CHANGELOG`), the clippy + test
legs are **skipped**; the compliance scanner still runs. Any `.rs`,
`Cargo.toml`, `Cargo.lock`, `scripts/`, `tools/`, `tests/`,
`integrations/`, `.githooks/` change or an unrecognised extension puts the
hook back on the full path. Force the full path regardless of diff shape
with `VOKRA_HOOK_DEEP=1`. Skip everything (including the compliance
scanner) with `git push --no-verify` or `VOKRA_SKIP_HOOKS=1`. The
classifier lives in `.githooks/lib-fastpath.sh` and its behaviour is
pinned by `scripts/test-pre-push-fastpath.sh` (17 cases).

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

### Claude Code

The repository is configured for [Claude Code](https://claude.ai/code) via
committed `.claude/settings.json` and `.claude/skills/`:

- **Hooks** keep Rust edits formatted (`rustfmt` on write), re-assert the
  zero-dependency invariant after `Cargo.toml` / `Cargo.lock` edits, and
  block `cargo add` (which would introduce an external dependency). The hook
  scripts live in `scripts/claude-hooks/`.
- **Skills** encode the recurring, policy-heavy workflows so they stay
  consistent: `add-speech-model`, `add-audio-operator`, `numerical-parity`,
  `license-audit`.

Personal, machine-local overrides go in `.claude/settings.local.json`
(git-ignored).
