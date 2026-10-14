# Compile-free pre-commit without parity-environment sync — 2026-10-15 JST

Private source-only candidate based on accepted Vokra main
`627a641ffe96441a1c1e7040f151602a5ea7facf`. This changes the interpreter
invocation of one existing static checker, not its scanning behavior or any
model, license, owner, numerical, memory or publication gate.

## Observed cause and bounded correction

The ordinary pre-commit hook has five checks: Rust formatting, forbidden
symbols, the first-party dependency invariant, fixture EOL pins, and the
pipefail/grep fail-open scanner. The last checker imports only `re`, `sys`
and `pathlib`, but its former `uv run --project tools/parity` invocation
synchronized the full parity software environment when it was absent.
Recent ordinary documentation commits therefore re-created 146 cached
software distributions after cleanup; `du` reported about `1.3G` for each
task environment. This is not equivalent physical-free-space evidence because
cache hard links can retain shared bytes. No model import or inference was
performed by that stdlib scanner.

The new invocation matches the existing CI Quality static-linter command:

```sh
uv run --no-project --python 3.12 python scripts/compliance/lint-pipefail-grep-q.py
```

UV and Python 3.12 remain mandatory. Production does not force offline mode,
so interpreter bootstrap is not newly prohibited. No parity package/lock,
scanner/parser, preceding check, ordering, failure propagation or existing
explicit hook-skip semantics changes. No skip flag is newly enabled.

## Causal regression and manager review

The new `scripts/test-pre-commit-stdlib.sh` executes a copy of the actual
production hook in an owned temporary Git fixture. Cargo and the three
preceding shell checks are mocked; UV and the unchanged stdlib scanner are
real. The test forces offline mode only within that fixture, unsets inherited
`VIRTUAL_ENV`, directs any UV project environment into owned scratch, and
keeps sync enabled to make an accidental project invocation observable.
Both root and parity-project sentinels declare unresolvable dependencies.

The tests prove clean scanner success; exact preceding command order;
Cargo exit 17 and each shell-check exit 23 stopping later commands; real lint
findings remaining nonzero; UV exit 42 propagating; and the existing explicit
skip behavior. Sentinel bytes remain unchanged; no environment or lock is
created, including dangling symlinks. All fixture writes and cleanup are
inside the owned temporary tree. The caller is read/status-compared only.

Manager review rejected an initial test that could overwrite/delete caller
files. Luna replaced it with the isolated fixture before acceptance. The
unsafe test was neither committed nor published. The regression is explicitly
wired into the existing CI Quality hygiene job, without changing action pins,
permissions or the required-check set.

Luna passes the focused regression, shell syntax/ShellCheck, actionlint,
workflow hygiene, the real offline stdlib scanner and diff hygiene. The
manager independently passes the production-hook regression and confirms
the scanner, pre-push hook, Codex guards and native Cargo/core paths are
unchanged. Normal pre-commit and fresh integrated-head CI must be recorded
separately; local fixture success is not remote CI or real-model evidence.

No dependency sync, target-package/model import, model/config/tokenizer
acquisition, heavy Cargo, paid VAST/Scaleway worker, protected-manifest access,
owner signature or upload is needed for this correction. The 58 unresolved
model rows, frozen 63-row owner scope, real-weight CPU work and final Apple
packets remain incomplete. This candidate must not be mistaken for campaign
completion.
