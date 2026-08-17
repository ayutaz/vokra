# Vokra Codex project instructions

This repository is maintained with Codex. Treat this file as the Codex-facing
project guide. The older local `CLAUDE.md` is supplementary historical context;
do not depend on Claude Code settings or hook behavior when working here.

## Project invariants

- Vokra is a Rust, audio-focused inference runtime and offline model converter.
- Keep the runtime zero-dependency invariant: root `Cargo.lock` must contain
  only first-party `vokra-*` crates. Do not use `cargo add`; implement with
  existing first-party code or escalate the design decision.
- Do not put ONNX/ONNX Runtime/protobuf runtime dependencies into the runtime.
  Offline conversion tools may handle source formats when the established
  converter pattern requires it.
- GPU backends are optional. Unsupported operations must return an explicit
  error; never add a silent CPU fallback.
- Do not invent model shapes, licenses, provenance, parity numbers, or source
  URLs. Preserve fail-closed license and publication gates.
- Python tooling is managed per tree with `pyproject.toml`, `uv.lock`, and
  Python 3.12. Use `uv run`/`uv sync` for every local Python invocation; do
  not use bare `python`, `python3`, pip, or conda.

## Memory and large-model safety

- Do not run workspace-wide cargo tests/builds/clippy locally on the maintainer
  machine. Use the `vast-ai-workflow` skill and a remote machine for heavy
  verification; destroy the instance afterward.
- Treat model artifacts of 2 GB or larger (including the sum of shards) as
  vast.ai work. The only narrow exception is provenance-only restamping through
  the established mmap path, without touching tensor data.
- Keep local verification package-scoped and serial (`CARGO_BUILD_JOBS=1`) when
  it is known to be safe. The Codex PreToolUse hook enforces the broad guard;
  do not bypass it unless the user explicitly authorizes that exact run.

## Skill routing

Use the repository skills in `.agents/skills/` when the task matches:

- `$add-speech-model`: add a TTS/ASR/S2S/VAD/speaker/music/separation/audio-LLM
  model, including converter, binder, license, and parity work.
- `$add-audio-operator`: add an audio-dialect operator or backend kernel.
- `$numerical-parity`: create or review reference fixtures and numerical gates.
- `$license-audit`: inspect dependencies, weights, attribution, and publication
  eligibility before changing model or codec support.
- `$publish-model-to-hf`: publish only through the repository's gated script.
- `$vast-ai-workflow`: convert, validate, publish, or benchmark memory-heavy
  artifacts and workspace-scale Rust code remotely.

Skills are guidance, not permission to skip tests or gates. Read the selected
skill before acting and follow its referenced project files.

## Verification and handoff

- Inspect `git status` before editing and preserve unrelated user changes.
- Prefer focused tests and repository shell gates locally. Use the remote
  workflow for workspace-scale verification.
- Run `git diff --check` and the relevant `scripts/check-*.sh` gates before
  handoff. Report commands that were not run and why.
- Keep documentation, license rows, provenance, and public API snapshots in
  the same logical change as the implementation.
- Do not commit, push, publish, or open a PR unless the user asks for that
  external state change.

The repository also contains legacy Claude configuration for users who still
need it. Codex behavior is defined by this file, `.agents/skills/`, and the
trusted hooks under `.codex/`.
