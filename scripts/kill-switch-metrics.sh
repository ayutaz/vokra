#!/usr/bin/env bash
# kill-switch-metrics.sh — collect GitHub-side metrics for the Kill switch C / K
# quarterly Go/No-go review (VOKRA-GOV-001, NFR-MT-05).
#
# Source: extraction of the script embedded in
#   docs/governance/kill-switch-metrics-runbook.md §4 (lines 178-267).
# Ticket / WP: M2-15-T03 ("指標データ収集 runbook 整備 … Discord proxy の手動収集手順").
#
# Purpose: emit a machine-readable JSON snapshot containing
#   - GitHub stargazer count (Kill switch C threshold: >= 500)
#   - Non-bot / non-Claude contributor count (Kill switch D input, also used
#     downstream for the M3 quarterly review)
#   - Unique participants (issue/PR/comment/discussion authors) in the last
#     3 months, bot- and Claude-filtered — the GitHub proxy for the retired
#     "Discord active user" metric (2026-07-04 依頼者決定, runbook §0/§3)
#   - Kill switch C verdict (PASS if stars >= 500 && active >= 20, else FAIL)
#   - Kill switch K input snapshot (competitor comparison is owner judgement)
#
# This script is INTENTIONALLY not wired into CI (M2-15-T02 / M2-15-T03 explicit
# constraint: "スケジュール実行のワークフロー (.yml) にはしない" — do not resurrect
# the 2026-07-04-retired kill-switch-check.yml). It is designed to be executed
# by the owner on their local machine at each quarterly review.
#
# Runtime dependencies: `gh` (GitHub CLI, authenticated as `ayutaz`) and `jq`.
# Both are owner-side installed CLIs; neither is a Vokra runtime dependency, so
# NFR-DS-02 (zero-external-crate invariant on the Rust workspace) is preserved.
#
# Usage:
#   bash scripts/kill-switch-metrics.sh \
#     > docs/governance/quarterly-reviews/2026-Q3.metrics.json
#   bash scripts/kill-switch-metrics.sh --help
#
# Exit codes: 0 = JSON written to stdout, 1 = missing dep / gh auth failure /
# CLI arg error. Networked `gh api` errors surface as non-zero via `set -e`.
#
# One deliberate deviation from the runbook §4 verbatim listing: the aggregation
# pipeline that produces $ACTIVE wraps each `grep -v` filter in
# `{ ... || true; }` so `set -o pipefail` does not turn "zero surviving
# authors" (Kill switch C = 0 activity, which is exactly the case we want to
# detect) into a crash. Semantic (unique non-bot / non-Claude login count) is
# preserved; the fix only fills in the runbook's unspecified corner behaviour
# and is verified in this file's test entry (see `--self-test`).

set -euo pipefail

usage() {
    cat <<'USAGE'
kill-switch-metrics.sh — quarterly review metrics collector (VOKRA-GOV-001)

Usage:
  bash scripts/kill-switch-metrics.sh
  bash scripts/kill-switch-metrics.sh --help
  bash scripts/kill-switch-metrics.sh --self-test

Produces JSON on stdout. Redirect into
  docs/governance/quarterly-reviews/YYYY-QN.metrics.json
and record the Go/No-go decision in the sibling .md file (runbook §6).

Requires: gh (authenticated), jq. Network access to api.github.com.
USAGE
}

# --self-test exercises the ACTIVE aggregation with fixture inputs and NEVER
# touches the network. It is safe to run in an unauthenticated environment.
self_test() {
    local status=0

    # Fixture 1: zero-activity case (the Kill switch C failure mode). The
    # runbook §4 verbatim listing crashes here under `set -o pipefail`.
    local ISSUE_AUTHORS="" ISSUE_CREATORS="" DISC_AUTHORS="" ACTIVE=""
    ACTIVE=$(printf '%s\n%s\n%s\n' "$ISSUE_AUTHORS" "$ISSUE_CREATORS" "$DISC_AUTHORS" \
        | { grep -v -E 'bot|Claude' || true; } \
        | { grep -v '^$' || true; } \
        | sort -u \
        | wc -l \
        | tr -d ' ')
    if [ "$ACTIVE" != "0" ]; then
        echo "self-test FAIL: zero-activity expected 0, got '$ACTIVE'" >&2
        status=1
    else
        echo "self-test PASS: zero-activity -> ACTIVE=0"
    fi

    # Fixture 2: mixed inputs — 6 raw logins, 2 filtered (dependabot[bot] and
    # Claude Code), one dedup, expected 4 (alice, bob, charlie, dave).
    ISSUE_AUTHORS=$'alice\nbob\ndependabot[bot]\nClaude Code'
    ISSUE_CREATORS=$'alice\ncharlie'
    DISC_AUTHORS=$'dave\ndependabot[bot]'
    ACTIVE=$(printf '%s\n%s\n%s\n' "$ISSUE_AUTHORS" "$ISSUE_CREATORS" "$DISC_AUTHORS" \
        | { grep -v -E 'bot|Claude' || true; } \
        | { grep -v '^$' || true; } \
        | sort -u \
        | wc -l \
        | tr -d ' ')
    if [ "$ACTIVE" != "4" ]; then
        echo "self-test FAIL: mixed-input expected 4, got '$ACTIVE'" >&2
        status=1
    else
        echo "self-test PASS: mixed-input -> ACTIVE=4 (alice, bob, charlie, dave)"
    fi

    # Fixture 3: BSD vs GNU date probe — exercise the same branch the main
    # path uses to compute $SINCE. Not asserting exact value (calendar-local
    # to the test host), only that at least one branch succeeds.
    local SINCE=""
    if date -u -v-3m +%Y-%m-%dT%H:%M:%SZ >/dev/null 2>&1; then
        SINCE=$(date -u -v-3m +%Y-%m-%dT%H:%M:%SZ)
    else
        SINCE=$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)
    fi
    if [[ ! "$SINCE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
        echo "self-test FAIL: SINCE not RFC3339, got '$SINCE'" >&2
        status=1
    else
        echo "self-test PASS: SINCE=$SINCE"
    fi

    # Fixture 4: Kill switch C verdict thresholds.
    local STARS=500 ACTIVE_MET=20 KSC_VERDICT=""
    if [ "$STARS" -ge 500 ] && [ "$ACTIVE_MET" -ge 20 ]; then
        KSC_VERDICT="PASS"
    else
        KSC_VERDICT="FAIL"
    fi
    if [ "$KSC_VERDICT" != "PASS" ]; then
        echo "self-test FAIL: threshold PASS boundary broken" >&2
        status=1
    else
        echo "self-test PASS: threshold PASS boundary honours >= 500 / >= 20"
    fi

    STARS=499
    if [ "$STARS" -ge 500 ] && [ "$ACTIVE_MET" -ge 20 ]; then
        KSC_VERDICT="PASS"
    else
        KSC_VERDICT="FAIL"
    fi
    if [ "$KSC_VERDICT" != "FAIL" ]; then
        echo "self-test FAIL: threshold FAIL below 500 broken" >&2
        status=1
    else
        echo "self-test PASS: threshold FAIL when stars=499"
    fi

    if [ "$status" -eq 0 ]; then
        echo "kill-switch-metrics --self-test: OK"
    fi
    return "$status"
}

# CLI dispatch: --help / --self-test short-circuit before any network work.
case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
    --self-test)
        self_test
        exit $?
        ;;
    "")
        : # fall through to the main path
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac

# Pre-flight: fail fast with a clear message if gh / jq are missing. `set -e`
# would otherwise surface a cryptic "command not found" via the first `gh api`
# call. This is not a Vokra dep — it is an owner-side CLI prerequisite.
for tool in gh jq date; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool '$tool' not found on PATH" >&2
        echo "hint: install with 'brew install $tool' (macOS) or 'apt install $tool' (Debian/Ubuntu)" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# The remainder is the runbook §4 script verbatim, minus the deliberate
# `|| true` corner-case fix noted in the preamble. Do not edit this block
# without also updating docs/governance/kill-switch-metrics-runbook.md §4.
# ---------------------------------------------------------------------------

OWNER=ayutaz
REPO=vokra
TODAY=$(date -u +%Y-%m-%d)

# BSD date (macOS) と GNU date (Linux) 両対応
if date -u -v-3m +%Y-%m-%dT%H:%M:%SZ >/dev/null 2>&1; then
    SINCE=$(date -u -v-3m +%Y-%m-%dT%H:%M:%SZ)
else
    SINCE=$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)
fi

# 1. Stars
STARS=$(gh repo view "$OWNER/$REPO" --json stargazerCount --jq .stargazerCount)

# 2. Contributors (excluding bots and Claude Code)
CONTRIB=$(gh api "repos/$OWNER/$REPO/contributors?per_page=100" --paginate \
    | jq '[.[] | select(.login | test("bot|Claude") | not)] | length')

# 3a. Issues + PR active participants (3 months)
ISSUE_AUTHORS=$(gh api "repos/$OWNER/$REPO/issues/comments?since=$SINCE&per_page=100" --paginate \
    | jq -r '.[].user.login')
ISSUE_CREATORS=$(gh api "repos/$OWNER/$REPO/issues?since=$SINCE&state=all&per_page=100" --paginate \
    | jq -r '.[].user.login')

# 3b. Discussions (if enabled)
DISC_ON=$(gh repo view "$OWNER/$REPO" --json hasDiscussionsEnabled --jq .hasDiscussionsEnabled)
DISC_AUTHORS=""
if [ "$DISC_ON" = "true" ]; then
    DISC_AUTHORS=$(gh api graphql -f query='
    query($owner:String!, $repo:String!) {
      repository(owner:$owner, name:$repo) {
        discussions(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
          nodes {
            author { login } updatedAt
            comments(first: 100) { nodes { author { login } updatedAt } }
          }
        }
      }
    }' -F owner="$OWNER" -F repo="$REPO" \
        | jq -r --arg since "$SINCE" '
            .data.repository.discussions.nodes
            | map(select(.updatedAt >= $since))
            | (map(.author.login) + (map(.comments.nodes[]?.author.login))) []')
fi

# `{ grep -v ... || true; }` around each filter: without this, an empty
# post-filter stream causes `grep -v` to exit 1, which `set -o pipefail`
# propagates and crashes the script — exactly on the Kill switch C =
# "0 active participants" path we are trying to detect. Verified in
# `--self-test`. This is the sole deviation from the runbook §4 listing.
ACTIVE=$(printf '%s\n%s\n%s\n' "$ISSUE_AUTHORS" "$ISSUE_CREATORS" "$DISC_AUTHORS" \
    | { grep -v -E 'bot|Claude' || true; } \
    | { grep -v '^$' || true; } \
    | sort -u \
    | wc -l \
    | tr -d ' ')

# Kill switch C verdict
if [ "$STARS" -ge 500 ] && [ "$ACTIVE" -ge 20 ]; then
    KSC_VERDICT="PASS"
else
    KSC_VERDICT="FAIL"
fi

# JSON output
cat <<EOF
{
  "measurement_date": "$TODAY",
  "repo": "$OWNER/$REPO",
  "window_since": "$SINCE",
  "stars": $STARS,
  "contributors_non_bot_non_cc": $CONTRIB,
  "issues_discussions_active_3mo": $ACTIVE,
  "kill_switch_c": {
    "threshold": {"stars_min": 500, "active_min": 20},
    "verdict": "$KSC_VERDICT",
    "note": "Discord は非採用（2026-07-04）ゆえ 'active user' は GitHub Issues + Discussions の直近 3 ヶ月の unique participants で代替判定"
  },
  "kill_switch_k": {
    "note": "competitor comparison is owner judgement; addressable market 10% threshold. 競合値の選定と比較は依頼者判断（本 runbook では自動収集しない）。",
    "verdict_input": {
      "vokra_stars": $STARS,
      "vokra_active_3mo": $ACTIVE,
      "unity_asset_store_dl": null,
      "competitor_reference": "sherpa-onnx / whisper.cpp / Candle 等の star 数は依頼者が手動記入"
    }
  }
}
EOF
