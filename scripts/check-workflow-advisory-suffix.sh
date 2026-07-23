#!/usr/bin/env bash
# check-workflow-advisory-suffix.sh — enforce §8.2 canonical advisory-step shape.
#
# .github/workflows/README.md §8.2 defines the canonical form of an advisory
# CI step:
#
#     - name: <what it measures> (advisory, record-only)
#       id: <slug>
#       continue-on-error: true
#       shell: bash
#       run: |
#         set -euo pipefail
#         if ! <probe>; then
#           echo "::warning::<what drifted> — advisory downgrade ..."
#           echo "<gate>=false" >> "$GITHUB_OUTPUT"
#           exit 0
#         fi
#         echo "<gate>=true" >> "$GITHUB_OUTPUT"
#
# The bug class this script prevents: a step declares itself advisory by
# lexical intent (a nearby comment says "advisory") but its `- name:` does not
# carry the canonical suffix, so a downstream grep-based audit that expects
# `advisory` in the step name silently misses the step. The `fix/nightly-asr-wer-
# 2026-07-23` P1 findings surfaced exactly this shape: a step documented as
# advisory-downgraded but LEXICALLY hard-failed on the very drift mode it
# claimed to tolerate. Encoding "the suffix is required" as a machine-checked
# invariant is cheaper than another human catching the same defect class.
#
# Contract:
#   For every step in .github/workflows/*.yml that has `continue-on-error: true`
#   at STEP scope (not job scope), the immediately-preceding `- name:` value
#   must contain the substring `advisory` (case-insensitive).
#
#   Job-scoped `continue-on-error: true` (whole-job advisory posture, §8.1)
#   is out of scope for this tripwire — that is a job-level attribute, and
#   the job's `name:` does not follow the same suffix convention.
#
# Mechanism: pure bash + awk + grep. No new dependencies. Handles multi-line
# step blocks by tracking the most recent `- name:` value while scanning down.
#
# CI wiring: Commit L in fix/nightly-asr-wer-2026-07-23 lands this as a
# `ci.yml` step (co-located with the other tripwires under §8.2 / §8.5
# canonical form). Runs on every PR.
#
# Exit codes:
#   0 — every advisory step carries the suffix.
#   1 — at least one step is missing the suffix (path:line + name printed).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# WORKFLOW_DIR is env-overridable so --self-test can point the same code at
# a synthetic tree. The default is the real repo tree.
: "${WORKFLOW_DIR:=$ROOT/.github/workflows}"

if [ ! -d "$WORKFLOW_DIR" ]; then
    echo "error: workflow directory not found at $WORKFLOW_DIR" >&2
    exit 1
fi

# Self-test mode: build a tiny synthetic workflow tree in a temp dir and
# assert both the passing and the failing shape produce the correct verdict.
# Kept redundant with the tripwire on purpose (§8.2 principle: an audit is
# only trustworthy if the audit itself has a red→green test).
if [ "${1:-}" = "--self-test" ]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/.github/workflows"
    # Positive case: advisory step with suffix should be silent.
    cat > "$tmp/.github/workflows/positive.yml" <<'YML'
name: pos
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - name: Foo drift check (advisory, record-only)
        id: foo
        continue-on-error: true
        run: echo ok
YML
    # Negative case: advisory step WITHOUT suffix must be reported.
    cat > "$tmp/.github/workflows/negative.yml" <<'YML'
name: neg
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    steps:
      - name: Foo drift check
        id: foo
        continue-on-error: true
        run: echo ok
YML
    # Job-scoped continue-on-error: this MUST NOT be reported (out of scope).
    cat > "$tmp/.github/workflows/job-scope.yml" <<'YML'
name: job
on: [push]
jobs:
  a:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - name: whatever
        run: echo ok
YML

    # Re-run this script pointing at the synthetic dir via a helper.
    scan() {
        WORKFLOW_DIR="$1" bash "$0" --scan-internal
    }
    if scan "$tmp/.github/workflows" > "$tmp/pos.out" 2>&1; then
        # Should have failed because negative.yml is present.
        echo "self-test FAIL: negative case was not flagged" >&2
        cat "$tmp/pos.out" >&2
        exit 1
    fi
    if ! grep -q 'negative.yml.*Foo drift check' "$tmp/pos.out"; then
        echo "self-test FAIL: negative case flagged the wrong step" >&2
        cat "$tmp/pos.out" >&2
        exit 1
    fi
    if grep -q 'positive.yml' "$tmp/pos.out"; then
        echo "self-test FAIL: positive (with suffix) was mis-flagged" >&2
        cat "$tmp/pos.out" >&2
        exit 1
    fi
    if grep -q 'job-scope.yml' "$tmp/pos.out"; then
        echo "self-test FAIL: job-scope continue-on-error was mis-flagged" >&2
        cat "$tmp/pos.out" >&2
        exit 1
    fi

    # Positive-only tree should be silent.
    rm -f "$tmp/.github/workflows/negative.yml" "$tmp/.github/workflows/job-scope.yml"
    if ! scan "$tmp/.github/workflows" > "$tmp/pos2.out" 2>&1; then
        echo "self-test FAIL: positive-only tree was flagged" >&2
        cat "$tmp/pos2.out" >&2
        exit 1
    fi

    echo "check-workflow-advisory-suffix: self-test OK"
    exit 0
fi

# --scan-internal is used by --self-test to point at a temp dir. The env var
# WORKFLOW_DIR overrides the default; the flag disables self-test recursion.
if [ "${1:-}" = "--scan-internal" ]; then
    : # WORKFLOW_DIR must already be set by the caller
fi

any_fail=0

# Iterate every workflow file. shopt -s nullglob so an empty dir is a no-op.
shopt -s nullglob
for wf in "$WORKFLOW_DIR"/*.yml "$WORKFLOW_DIR"/*.yaml; do
    # Parse: for each `continue-on-error: true` line, look back for the most
    # recent step-scope `- name:` and forward to confirm the enclosing scope
    # is a step (indented under `steps:`), not a job.
    #
    # Simple heuristic that matches Vokra's uniform 6/8/10-space indent style
    # (per .github/workflows/README.md §8): a step name line matches
    # `^\s*-\s+name:` and a job-level `continue-on-error:` is at lower indent
    # than any step's `- name:`. We locate each `continue-on-error: true`
    # occurrence and check the closest preceding `- name:` INDENTED MORE
    # DEEPLY THAN the `continue-on-error:` line itself is at the same or
    # deeper indent (i.e. step scope).
    awk -v WF="$wf" '
        function indent(s) { match(s, /^[ \t]*/); return RLENGTH }

        # Track the last seen `- name:` at each indent level.
        /^[ \t]*-[ \t]+name:/ {
            level = indent($0)
            # capture the value after `name:`
            v = $0
            sub(/^[ \t]*-[ \t]+name:[ \t]*/, "", v)
            # strip trailing comment / quotes
            sub(/[ \t]+#.*$/, "", v)
            gsub(/^["'\'']|["'\'']$/, "", v)
            last_name[level] = v
            last_name_line[level] = NR
        }

        # Detect job-scope continue-on-error by structural depth.
        # A step-scope continue-on-error sits at a deeper indent than its
        # `- name:` sibling (siblings share indent, so we compare against the
        # nearest ANY `- name:` above at the same-or-lesser indent).
        /^[ \t]*continue-on-error:[ \t]*true/ {
            coe_level = indent($0)
            # Find the deepest `- name:` seen so far whose indent is
            # <= coe_level (that is the sibling `- name:`).
            candidate_level = -1
            for (lvl in last_name) {
                if (lvl + 0 <= coe_level + 0 && lvl + 0 > candidate_level) {
                    candidate_level = lvl + 0
                }
            }
            if (candidate_level < 0) {
                # No preceding `- name:` at or above this indent → job scope.
                next
            }
            sib = last_name[candidate_level]
            sib_line = last_name_line[candidate_level]

            # Confirm this looks like a step (`- name:` at candidate_level),
            # not a job. Steps in Vokra style have `- name:` at indent > 6;
            # `continue-on-error:` under a JOB is at indent 4 with no
            # preceding `- name:` at indent <= 4 in scope.
            if (candidate_level < 6) {
                # Structurally a job-level attribute — skip.
                next
            }

            lc = tolower(sib)
            if (index(lc, "advisory") == 0) {
                printf("%s:%d: advisory step missing (advisory, record-only) suffix in `- name:`: %s\n", WF, sib_line, sib)
                any_fail = 1
            }
        }

        END { exit any_fail }
    ' "$wf" || any_fail=1
done

if [ "$any_fail" -ne 0 ]; then
    echo "" >&2
    echo "error: at least one advisory step (continue-on-error: true) is missing the" >&2
    echo "       canonical '(advisory, record-only)' suffix in its \`- name:\`." >&2
    echo "       See .github/workflows/README.md §8.2 for the canonical shape." >&2
    exit 1
fi

echo "check-workflow-advisory-suffix: OK (all continue-on-error step names carry the suffix)"
