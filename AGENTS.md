# Vokra Codex project instructions

This repository is maintained with Codex. Treat this file as the Codex-facing
project guide. The older local `CLAUDE.md` is supplementary historical context;
do not depend on Claude Code settings or hook behavior when working here.

## Agent model and delegation policy

- The primary/root agent is the manager and must use `gpt-5.6-sol`. It owns
  requirements, investigation, planning, task decomposition, risk decisions,
  review, verification strategy, and the final user handoff.
- Any task that creates or changes implementation artifacts must be delegated
  by the Sol manager to the project `luna_implementer` agent, or to a spawned
  sub-agent explicitly using `gpt-5.6-luna` when the named agent is not exposed
  by the current client. Implementation artifacts include Rust and Python
  source, tests, shell scripts, build files, workflows, and product-behavior
  configuration. The Sol manager must not directly author those changes.
- The Luna implementer owns the bounded code edits and focused verification
  delegated to it. It must follow this file and every applicable project skill,
  preserve unrelated worktree changes, and report changed files, test evidence,
  unresolved risks, and any blocked work to the Sol manager.
- The Sol manager must inspect Luna's diff and verification evidence. Corrections
  to implementation must be sent back to Luna; Sol accepts and hands off the
  work only after completing its own review.
- Sol may directly perform read-only investigation and edit management-only
  documentation, handoffs, plans, and Codex agent configuration. If Luna is
  unavailable, Sol must stop and report the blocker instead of silently taking
  over implementation.
- Multiple Luna agents may be used only for independent, non-overlapping scopes.
  Sol remains responsible for coordination and the integrated result.

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

- Do not run workspace-wide or `vokra-models` Cargo commands that compile,
  test, check, run, document, or audit locally on the maintainer machine. Cheap
  `cargo fmt` / `cargo metadata` / `cargo tree` inspection remains local. Use
  the `vast-ai-workflow` skill for the heavy scope and destroy the instance
  afterward.
- Treat model artifacts of 2 GB or larger (including the sum of shards) as
  vast.ai work. The only narrow exception is provenance-only restamping through
  the established mmap path, without touching tensor data.
- Keep local verification package-scoped and serial (`CARGO_BUILD_JOBS=1`) when
  it is known to be safe. The Codex PreToolUse hook enforces the broad guard;
  do not bypass it unless the user explicitly authorizes that exact run.
- A deep pre-push path is also refused on the maintainer Mac. After a recorded
  green VAST run, `VOKRA_SKIP_HOOKS=1` may be used for the corresponding code
  push. A deletion-only ref update still runs the compliance regression, then
  skips Cargo because it transfers no commit.

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
- Treat model upload as a separate irreversible permission: VAST work, a clean
  dry-run, or a license sign-off does not authorize `--push` to Hugging Face.
- For the 2026-08-18 baseline, merged-PR history, retired-branch disposition,
  VAST evidence, and remaining M5 ledger, read
  `docs/handoff/codex-operations-2026-08-18.md`.

The repository also contains legacy Claude configuration for users who still
need it. Codex behavior is defined by this file, `.agents/skills/`, and the
trusted hooks under `.codex/`.
