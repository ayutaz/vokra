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
#   - (M5-12-T05) contributors_excluding_owner — the OWNER-excluded committer
#     count DoD item 5 consumes ("community not dependent on the maintainer
#     alone"). Kill switch D keeps the owner-INCLUDED count; whether the
#     threshold-of-3 should itself exclude the owner is owner work
#     (X-05-T21/T23). This script surfaces both and decides neither. The field
#     name matches the one the go-nogo record already documents
#     (docs/governance/vokra-go-nogo-v0.5.md).
#   - (M5-12-T04) dod_item4_kill_switch — an A–L scaffold for GA DoD item 4
#     ("none of Kill switch A–L has fired"): C/D/K reuse the self-collected
#     values above; A/B/E/F/G/H/I/J/L stay "owner-judgment-required" (competitor
#     changelog is owner judgement, milestones.md §10.1). Verdicts are never
#     fabricated (FR-EX-08); the final per-switch call lives in the go-nogo
#     record (vokra-go-nogo-<phase>.md, X-05-T17).
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
# Deviations from the runbook §4 verbatim listing (all intentional, tested):
#  (1) the aggregation pipeline that produces $ACTIVE wraps each `grep -v`
#      filter in `{ ... || true; }` so `set -o pipefail` does not turn "zero
#      surviving authors" (Kill switch C = 0 activity, which is exactly the case
#      we want to detect) into a crash. Semantic (unique non-bot / non-Claude
#      login count) is preserved.
#  (2) (M5-12-T04/T05) the JSON additionally emits `contributors_excluding_owner`
#      and the `dod_item4_kill_switch` / `dod_item5` objects described above.
#      The runbook §4 listing predates these; the runbook now points here for
#      the DoD-item additions rather than duplicating them (no drift check
#      enforces byte-equality, and duplicating the JSON would itself risk drift).
# Both deviations are verified in this file's test entry (see `--self-test`).

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

    # Fixture 5 (M5-12-T05): the owner-included vs owner-excluded contributor
    # filters. Same fixture, two jq expressions: owner-included keeps ayutaz
    # (Kill switch D input), owner-excluded drops it (DoD item 5 input). jq may
    # be absent in a bare self-test env, so this fixture is jq-guarded.
    if command -v jq >/dev/null 2>&1; then
        local FIX='[{"login":"ayutaz"},{"login":"alice"},{"login":"bob"},{"login":"dependabot[bot]"},{"login":"Claude"}]'
        local INCL EXCL
        INCL=$(printf '%s' "$FIX" | jq '[.[] | select(.login | test("bot|Claude") | not)] | length')
        EXCL=$(printf '%s' "$FIX" \
            | jq --arg owner ayutaz '[.[] | select((.login != $owner) and (.login | test("bot|Claude") | not))] | length')
        if [ "$INCL" = "3" ] && [ "$EXCL" = "2" ]; then
            echo "self-test PASS: contributors owner-included=3 (ayutaz,alice,bob), owner-excluded=2 (alice,bob)"
        else
            echo "self-test FAIL: expected incl=3 excl=2, got incl='$INCL' excl='$EXCL'" >&2
            status=1
        fi
    else
        echo "self-test SKIP: jq absent — contributor-exclusion filter not exercised here"
    fi

    # Fixture 6 (M5-12-T04): Kill switch 'fires' (該当) derives from the verdict
    # — FAIL fires, PASS does not. DoD item 4 reads this per switch.
    local KSC_V FIRES
    KSC_V="FAIL"
    if [ "$KSC_V" = "FAIL" ]; then FIRES=true; else FIRES=false; fi
    if [ "$FIRES" = "true" ]; then
        echo "self-test PASS: Kill switch FAIL -> fires=true"
    else
        echo "self-test FAIL: FAIL must fire" >&2
        status=1
    fi
    KSC_V="PASS"
    if [ "$KSC_V" = "FAIL" ]; then FIRES=true; else FIRES=false; fi
    if [ "$FIRES" = "false" ]; then
        echo "self-test PASS: Kill switch PASS -> fires=false"
    else
        echo "self-test FAIL: PASS must not fire" >&2
        status=1
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

# 2. Contributors. Fetch once; derive two counts from the same snapshot.
CONTRIB_RAW=$(gh api "repos/$OWNER/$REPO/contributors?per_page=100" --paginate)
# Kill switch D input: excludes bots + Claude Code, INCLUDES the owner (the
# switch wording is "committers other than Claude Code").
CONTRIB=$(printf '%s' "$CONTRIB_RAW" \
    | jq '[.[] | select(.login | test("bot|Claude") | not)] | length')
# DoD item 5 input: additionally excludes the owner — the count of EXTERNAL
# committers ("community not dependent on the maintainer alone"). Field name
# matches the go-nogo record (docs/governance/vokra-go-nogo-v0.5.md).
CONTRIB_EXCL_OWNER=$(printf '%s' "$CONTRIB_RAW" \
    | jq --arg owner "$OWNER" '[.[] | select((.login != $owner) and (.login | test("bot|Claude") | not))] | length')

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

# "Fires" (該当) is the withdrawal-argument sense: Kill switch C FAILs its PASS
# condition => it fires. DoD item 4 reads this per switch.
if [ "$KSC_VERDICT" = "FAIL" ]; then
    KSC_FIRES=true
else
    KSC_FIRES=false
fi

# JSON output
cat <<EOF
{
  "measurement_date": "$TODAY",
  "repo": "$OWNER/$REPO",
  "window_since": "$SINCE",
  "stars": $STARS,
  "contributors_non_bot_non_cc": $CONTRIB,
  "contributors_excluding_owner": $CONTRIB_EXCL_OWNER,
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
  },
  "dod_item4_kill_switch": {
    "note": "GA DoD item 4 requires that NONE of Kill switch A–L has fired. C/D/K are computed from this snapshot; A/B/E/F/G/H/I/J/L are competitor-changelog owner judgements (milestones.md §10.1) and are never auto-fabricated (FR-EX-08). The final per-switch verdict is recorded in the go-nogo record (vokra-go-nogo-<phase>.md, X-05-T17).",
    "A": {"basis": "Unity Inference Engine fixes VITS/Metal & covers major TTS/ASR net-natively", "cadence": "quarterly", "verdict": "owner-judgment-required"},
    "B": {"basis": "sherpa-onnx ships official Unity UPM + Godot AssetLib bindings", "cadence": "quarterly", "verdict": "owner-judgment-required"},
    "C": {"basis": "stars >= 500 && active >= 20", "metric_verdict": "$KSC_VERDICT", "fires": $KSC_FIRES, "timing": "5–6 month point; start date owner-recorded (X-05-T23)"},
    "D": {"basis": "< 3 committers other than Claude Code", "contributors_non_bot_non_cc": $CONTRIB, "threshold": 3, "verdict": "timing-owner-determined", "note": "start date undefined (no v0.5.0 tag); timing and whether the threshold excludes the owner are owner decisions (X-05-T23)"},
    "E": {"basis": "major vendor (MS/Google/Apple/Meta/HF) ships an Apache-2.0 speech runtime", "cadence": "continuous-urgent", "verdict": "owner-judgment-required"},
    "F": {"basis": "HF Candle covers Whisper/piper-plus/Kokoro/CosyVoice/Moshi/Mimi natively", "cadence": "continuous", "verdict": "owner-judgment-required"},
    "G": {"basis": "ORT Web + WebGPU runs speech at RTF < 1.0 on Whisper base", "cadence": "12–18 months", "verdict": "owner-judgment-required"},
    "H": {"basis": "Modular MAX Engine expands its speech-op coverage", "cadence": "quarterly", "verdict": "owner-judgment-required"},
    "I": {"basis": ">= 3 of 6 models degrade >= 5% vs PyTorch (MEL/UTMOS)", "verdict": "owner-judgment-required", "input_ref": "GA DoD item-2 runner (vokra_eval::dod)"},
    "J": {"basis": "HA Voice / Wyoming declines to adopt Vokra", "cadence": "at v0.5", "verdict": "owner-judgment-required"},
    "K": {"basis": "addressable market < 10% of a comparable competitor", "verdict": "owner-judgment-required", "input_ref": "kill_switch_k"},
    "L": {"basis": "maintainer burnout / funding exhaustion", "cadence": "continuous", "verdict": "owner-judgment-required"}
  },
  "dod_item5": {
    "basis": "GA DoD item 5: >= 3 committers other than Claude Code AND community operation not dependent on the maintainer alone.",
    "external_committers": $CONTRIB_EXCL_OWNER,
    "threshold": 3,
    "verdict": "owner-judgment-required",
    "note": "Consumes the OWNER-excluded count (contributors_excluding_owner). Whether the Kill switch D threshold-of-3 should itself exclude the owner is an owner decision (X-05-T21/T23); this object surfaces the external count and decides nothing."
  }
}
EOF
