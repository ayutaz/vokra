#!/usr/bin/env bash
# check-find-hygiene.sh — enforce `find | head` idioms include a type filter.
#
# The bug class this script prevents: a workflow does
#
#     find "$X" -name '*.wav' | head -n1
#
# expecting a file, but a DIRECTORY whose basename happens to end in `.wav`
# will match too and hand a directory path to the downstream tool
# (`ffmpeg -i "$WAV"` etc.) which then crashes hard on every run. Vokra hit
# this class in `nightly-asr-wer.yml` (RTF companion) and fixed it in commit
# 8832ff0. This tripwire generalises the fix so the same defect cannot land
# again in another workflow via copy-paste.
#
# Contract:
#   For every `find` command inside .github/workflows/*.yml that pipes into
#   `head`, at least one `-type <t>` predicate must appear on the same line.
#   `-type f`, `-type d`, `-type l` are all accepted — the point is that the
#   author explicitly chose the entry kind rather than letting the basename
#   filter alone gate the pick.
#
# Not enforced (out of scope):
#   * `find` without `head` — the pattern is specifically the "grab the first
#     one" idiom; a `find | xargs` streaming pipeline handles multi-hits
#     differently and has different failure modes.
#   * `find` alone (no pipe) — that surfaces the full listing to the reader,
#     which is a documentation intent, not a "grab one" pattern.
#   * multi-line `find` continuations (`find "$X" \\`) — Vokra style rewrites
#     these to a single line before landing; a future extension can walk
#     joined lines if we ever adopt multi-line find idioms.
#
# CI wiring: Commit M in fix/nightly-asr-wer-2026-07-23 lands this alongside
# the sibling `check-workflow-advisory-suffix.sh` tripwire.
#
# Exit codes:
#   0 — every `find ... | head` line carries an explicit `-type <t>`.
#   1 — at least one line is missing the type filter (path:line printed).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# WORKFLOW_DIR is env-overridable so --self-test can point the same code at
# a synthetic tree. The default is the real repo tree.
: "${WORKFLOW_DIR:=$ROOT/.github/workflows}"

if [ ! -d "$WORKFLOW_DIR" ]; then
    echo "error: workflow directory not found at $WORKFLOW_DIR" >&2
    exit 1
fi

# Self-test: build a synthetic workflow tree and assert the checker flags
# the RED case (missing -type) and stays silent on the GREEN case (explicit
# -type). Kept as a hard leg because a checker bug that silently passes is
# the exact defect this checker exists to prevent.
if [ "${1:-}" = "--self-test" ]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/.github/workflows"

    # Negative: bare `find | head` with no type — must be flagged.
    cat > "$tmp/.github/workflows/bad.yml" <<'YML'
name: bad
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          F="$(find "$DIR" -name '*.wav' | head -n1)"
YML

    # Positive: explicit `-type f` — must be silent.
    cat > "$tmp/.github/workflows/good.yml" <<'YML'
name: good
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          F="$(find "$DIR" -type f -name '*.wav' | head -n1)"
YML

    # Positive: `-type d` also accepted.
    cat > "$tmp/.github/workflows/good-dir.yml" <<'YML'
name: gooddir
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          F="$(find "$DIR" -type d -name 'build' | head -n1)"
YML

    # Negative-in-loop: even inside a shell loop, missing -type is a hit.
    cat > "$tmp/.github/workflows/bad-loop.yml" <<'YML'
name: badloop
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          for f in $(find "$DIR" -name '*.txt' | head -3); do
            echo "$f"
          done
YML

    # Out-of-scope: `find` without head, must NOT be flagged.
    cat > "$tmp/.github/workflows/no-head.yml" <<'YML'
name: nohead
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          find "$DIR" -name '*.txt'
YML

    # Out-of-scope: comment lines that mention `find | head` textually
    # (this file's own gate step comment in ci.yml, workflow-level docs).
    # A false positive here defeats the checker's own installation.
    cat > "$tmp/.github/workflows/comment-only.yml" <<'YML'
name: commentonly
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - run: |
          # Prevents the `find | head` landmine — advisory text only, no actual command.
          # See scripts/check-find-hygiene.sh: `find "$X" -name '*.wav' | head` is banned.
          echo "safe"
YML

    out="$tmp/out"
    if WORKFLOW_DIR="$tmp/.github/workflows" bash "$0" > "$out" 2>&1; then
        echo "self-test FAIL: bad tree passed" >&2
        cat "$out" >&2
        exit 1
    fi
    if ! grep -q 'bad.yml' "$out"; then
        echo "self-test FAIL: bad.yml not flagged" >&2
        cat "$out" >&2
        exit 1
    fi
    if ! grep -q 'bad-loop.yml' "$out"; then
        echo "self-test FAIL: bad-loop.yml not flagged" >&2
        cat "$out" >&2
        exit 1
    fi
    if grep -q 'good.yml' "$out"; then
        echo "self-test FAIL: good.yml mis-flagged" >&2
        cat "$out" >&2
        exit 1
    fi
    if grep -q 'good-dir.yml' "$out"; then
        echo "self-test FAIL: good-dir.yml (-type d) mis-flagged" >&2
        cat "$out" >&2
        exit 1
    fi
    if grep -q 'no-head.yml' "$out"; then
        echo "self-test FAIL: no-head.yml (out of scope) mis-flagged" >&2
        cat "$out" >&2
        exit 1
    fi
    if grep -q 'comment-only.yml' "$out"; then
        echo "self-test FAIL: comment-only.yml (comment lines) mis-flagged" >&2
        cat "$out" >&2
        exit 1
    fi

    # Positive-only tree must be silent.
    rm -f "$tmp/.github/workflows/bad.yml" "$tmp/.github/workflows/bad-loop.yml"
    if ! WORKFLOW_DIR="$tmp/.github/workflows" bash "$0" > "$out" 2>&1; then
        echo "self-test FAIL: positive-only tree flagged" >&2
        cat "$out" >&2
        exit 1
    fi

    echo "check-find-hygiene: self-test OK"
    exit 0
fi

any_fail=0

shopt -s nullglob
for wf in "$WORKFLOW_DIR"/*.yml "$WORKFLOW_DIR"/*.yaml; do
    # Match a single line containing `find` and `| head` (allowing arbitrary
    # whitespace / other pipeline members between them). If the same line
    # lacks `-type`, print the offender.
    #
    # awk instead of grep because we need line-numbered output AND the
    # negative-match ("no -type on this line") in one pass. Grep supports
    # `-vP` but composing "match A AND NOT B" against the same line is
    # cleaner in awk.
    awk -v WF="$wf" '
        # A `find ... | head` line lacking `-type`.
        # Skip YAML/shell comment-only lines (leading `#`) — the tripwire
        # itself and workflow-level documentation mention the pattern as
        # prose ("prevents find | head ..." text) and would otherwise
        # match its own explanation.
        /find[^|]*\| *head/ {
            trimmed = $0
            sub(/^[[:space:]]+/, "", trimmed)
            if (substr(trimmed, 1, 1) == "#") next
            if ($0 !~ /-type[[:space:]]+[fdlpcbs]/) {
                printf("%s:%d: `find | head` without `-type f|d|...` — %s\n", WF, NR, trimmed)
                any_fail = 1
            }
        }
        END { exit any_fail }
    ' "$wf" || any_fail=1
done

if [ "$any_fail" -ne 0 ]; then
    echo "" >&2
    echo "error: at least one \`find ... | head\` line lacks an explicit \`-type\`." >&2
    echo "       Basename globbing alone (e.g. -name '*.wav') will match DIRECTORIES" >&2
    echo "       too — see commit 8832ff0 for the concrete failure this prevents." >&2
    echo "       Add \`-type f\` (or \`-type d\` if a directory is intended)." >&2
    exit 1
fi

echo "check-find-hygiene: OK (all \`find ... | head\` lines carry -type)"
