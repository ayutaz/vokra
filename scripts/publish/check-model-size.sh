#!/usr/bin/env bash
# check-model-size.sh — decide local vs vast.ai for a HF model before we
# start convert.
#
# 2026-07-28 policy (memory feedback-large-models-on-vast-ai):
# convert + upload for a HF weight now defaults to vast.ai; the M1 iMac (16GB
# RAM) is only used for models that provably fit. This script is the *local*
# preflight owner runs before deciding where to convert — it queries the HF
# API for the safetensors sibling sizes and prints a verdict plus the
# reasoning, so the choice is data-driven, not eyeballed from a repo page.
#
# The thresholds match the runbook (docs/handoff/vast-ai-large-model-publish.md
# §1 / §4) verbatim:
#   ≤4 GiB total                                     -> LOCAL_SAFE
#   4-8 GiB total  AND max shard ≤6 GiB              -> LOCAL_OK       (single-tenant)
#   8-16 GiB OR shards ≥5                            -> LOCAL_BORDERLINE (single-tenant only, exit 0 with warning)
#   >16 GiB                                          -> VAST_AI_REQUIRED (exit 1)
#
# Exit code — meant for scripting: `check-model-size.sh <repo> && convert-here`
#   0 for LOCAL_SAFE / LOCAL_OK / LOCAL_BORDERLINE
#   1 for VAST_AI_REQUIRED
#   2 for usage / network error (distinct so callers can tell "genuinely too big"
#     from "we couldn't reach HF").
#
# Usage:
#   check-model-size.sh <hf-repo>              # human summary + verdict
#   check-model-size.sh --json <hf-repo>       # machine-readable single line JSON
#   check-model-size.sh --self-test
#
# HF token: HF_TOKEN or HF (only needed for gated repos; public metadata is
# fine without one).

set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: check-model-size.sh <hf-repo>
       check-model-size.sh --json <hf-repo>
       check-model-size.sh --self-test

thresholds (match docs/handoff/vast-ai-large-model-publish.md):
  ≤4 GiB total                    -> LOCAL_SAFE
  4-8 GiB total, max shard ≤6 GiB -> LOCAL_OK        (single-tenant)
  8-16 GiB or shards ≥5           -> LOCAL_BORDERLINE (single-tenant only)
  >16 GiB                         -> VAST_AI_REQUIRED (exit 1)
EOF
}

# --- verdict thresholds --------------------------------------------------
# In bytes so python side never mixes units.
readonly THRESH_SAFE=$((4 * 1024 * 1024 * 1024))       # 4 GiB
readonly THRESH_OK=$((8 * 1024 * 1024 * 1024))         # 8 GiB
readonly THRESH_BORDERLINE=$((16 * 1024 * 1024 * 1024)) # 16 GiB
readonly THRESH_MAX_SHARD=$((6 * 1024 * 1024 * 1024))  # 6 GiB
readonly THRESH_MANY_SHARDS=5                          # ≥5 shards trips borderline

# classify_bytes -- pure fn: (total_bytes, max_shard_bytes, shard_count)
#                              -> "LOCAL_SAFE|LOCAL_OK|LOCAL_BORDERLINE|VAST_AI_REQUIRED"
# Exercised directly from --self-test (no HF fetch).
classify_bytes() {
  python3 - "$1" "$2" "$3" \
    "$THRESH_SAFE" "$THRESH_OK" "$THRESH_BORDERLINE" \
    "$THRESH_MAX_SHARD" "$THRESH_MANY_SHARDS" <<'PY'
import sys
total, max_shard, shards = (int(sys.argv[i]) for i in (1, 2, 3))
t_safe, t_ok, t_bord, t_max_shard, t_many = (int(sys.argv[i]) for i in (4, 5, 6, 7, 8))
if total > t_bord:
    print("VAST_AI_REQUIRED"); sys.exit(0)
if total > t_ok or shards >= t_many:
    print("LOCAL_BORDERLINE"); sys.exit(0)
if total > t_safe or max_shard > t_max_shard:
    print("LOCAL_OK"); sys.exit(0)
print("LOCAL_SAFE")
PY
}

# fetch_hf_sizes -- (repo) -> "<total>\t<max_shard>\t<shard_count>"
# Uses HF public API; token only needed for gated repos.
#
# Python fetches directly (urllib.request) rather than curl-into-python. An
# earlier iteration piped `curl | python3 - <<'PY'` and lost the JSON: the
# heredoc redirect and the pipe both target python3's stdin, and the heredoc
# wins — python3 reads the script itself and tries to json.loads it. Doing
# the fetch inside python removes that whole marshalling class of bug.
fetch_hf_sizes() {
  local repo="$1"
  local token="${HF_TOKEN:-${HF:-}}"
  local out
  if ! out="$(python3 - "$repo" "$token" <<'PY'
import json, sys, urllib.request, urllib.error
repo, token = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/api/models/{repo}?blobs=true"
req = urllib.request.Request(url)
if token:
    req.add_header("Authorization", f"Bearer {token}")
try:
    body = urllib.request.urlopen(req, timeout=30).read()
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} from HF API for '{repo}' (URL: {url})", file=sys.stderr)
    sys.exit(2)
except (urllib.error.URLError, TimeoutError) as e:
    print(f"could not reach HF API for '{repo}' (URL: {url}): {e}", file=sys.stderr)
    sys.exit(2)
d = json.loads(body)
total = 0
max_shard = 0
count = 0
for s in d.get("siblings", []):
    name = s.get("rfilename", "") or ""
    if not name.endswith(".safetensors"):
        continue
    sz = int(s.get("size") or 0)
    total += sz
    if sz > max_shard:
        max_shard = sz
    count += 1
print(f"{total}\t{max_shard}\t{count}")
PY
  )"; then
    echo "check-model-size: could not fetch HF sizes for '$repo'" >&2
    return 2
  fi
  printf '%s\n' "$out"
}

# render_human -- pretty summary for a human running this at the terminal.
render_human() {
  local repo="$1" verdict="$2" total="$3" max_shard="$4" shards="$5"
  local total_gib max_gib
  total_gib="$(python3 -c "print(f'{$total/1024**3:.2f}')")"
  max_gib="$(python3 -c "print(f'{$max_shard/1024**3:.2f}')")"
  echo "check-model-size: repo=$repo"
  echo "  safetensors total: ${total_gib} GiB"
  echo "  max shard         : ${max_gib} GiB"
  echo "  shard count       : ${shards}"
  echo "  verdict           : ${verdict}"
  case "$verdict" in
    LOCAL_SAFE)
      echo "  -> convert here (docs/handoff/vast-ai-large-model-publish.md §1 'Safe locally')"
      ;;
    LOCAL_OK)
      echo "  -> convert here single-tenant (stop other Rust builds / tests / DL during convert)"
      ;;
    LOCAL_BORDERLINE)
      echo "  -> LOCAL BORDERLINE — convert here only single-tenant; running other work risks swap thrash"
      echo "     Prefer vast.ai (docs/handoff/vast-ai-large-model-publish.md §2)"
      ;;
    VAST_AI_REQUIRED)
      echo "  -> VAST.AI REQUIRED — 16 GB M1 iMac cannot mmap this without OS-level swap thrash"
      echo "     Runbook: docs/handoff/vast-ai-large-model-publish.md §2"
      echo "     Automated path: scripts/publish/vast-ai/provision.sh (once) + scripts/publish/vast-ai/run-one.sh (per model)"
      ;;
  esac
}

# render_json -- single-line JSON, stable field order, no floats (bytes only).
# Kept parseable so callers (e.g. publish-one.sh's own gate) can jq it.
render_json() {
  local repo="$1" verdict="$2" total="$3" max_shard="$4" shards="$5"
  python3 - "$repo" "$verdict" "$total" "$max_shard" "$shards" <<'PY'
import json, sys
repo, verdict, total, max_shard, shards = sys.argv[1:6]
print(json.dumps({
    "repo": repo,
    "verdict": verdict,
    "total_bytes": int(total),
    "max_shard_bytes": int(max_shard),
    "shard_count": int(shards),
}, separators=(",", ":")))
PY
}

# ---- self-test ----------------------------------------------------------
# Exercises the pure classifier directly (no HF fetch, no network). Both
# directions matter: refusing what must be refused, AND passing what is
# compliant — a too-strict classifier is a silent failure that would push
# small models to vast.ai unnecessarily.
run_self_test() {
  local fail=0 cases=0 got
  local gib=$((1024 * 1024 * 1024))

  # 1 GiB total, single shard -> LOCAL_SAFE
  cases=$((cases + 1))
  got="$(classify_bytes $((1 * gib)) $((1 * gib)) 1)"
  [[ "$got" == "LOCAL_SAFE" ]] || { echo "self-test FAIL: 1GiB/1shard != LOCAL_SAFE (got $got)" >&2; fail=1; }

  # exactly 4 GiB -> LOCAL_SAFE (boundary: >4 GiB moves to LOCAL_OK)
  cases=$((cases + 1))
  got="$(classify_bytes $((4 * gib)) $((4 * gib)) 1)"
  [[ "$got" == "LOCAL_SAFE" ]] || { echo "self-test FAIL: exactly 4GiB != LOCAL_SAFE (got $got)" >&2; fail=1; }

  # 5 GiB total, small shards -> LOCAL_OK
  cases=$((cases + 1))
  got="$(classify_bytes $((5 * gib)) $((3 * gib)) 2)"
  [[ "$got" == "LOCAL_OK" ]] || { echo "self-test FAIL: 5GiB/2sh != LOCAL_OK (got $got)" >&2; fail=1; }

  # 3 GiB total but max shard 7 GiB (impossible in reality; guards the OR branch) -> LOCAL_OK
  cases=$((cases + 1))
  got="$(classify_bytes $((3 * gib)) $((7 * gib)) 1)"
  [[ "$got" == "LOCAL_OK" ]] || { echo "self-test FAIL: big-shard trip != LOCAL_OK (got $got)" >&2; fail=1; }

  # 9 GiB total -> LOCAL_BORDERLINE
  cases=$((cases + 1))
  got="$(classify_bytes $((9 * gib)) $((5 * gib)) 3)"
  [[ "$got" == "LOCAL_BORDERLINE" ]] || { echo "self-test FAIL: 9GiB != LOCAL_BORDERLINE (got $got)" >&2; fail=1; }

  # ≥5 shards, small total -> LOCAL_BORDERLINE (many-shards trip)
  cases=$((cases + 1))
  got="$(classify_bytes $((6 * gib)) $((2 * gib)) 5)"
  [[ "$got" == "LOCAL_BORDERLINE" ]] || { echo "self-test FAIL: 5sh trip != LOCAL_BORDERLINE (got $got)" >&2; fail=1; }

  # 17 GiB total -> VAST_AI_REQUIRED
  cases=$((cases + 1))
  got="$(classify_bytes $((17 * gib)) $((8 * gib)) 4)"
  [[ "$got" == "VAST_AI_REQUIRED" ]] || { echo "self-test FAIL: 17GiB != VAST_AI_REQUIRED (got $got)" >&2; fail=1; }

  # 48 GiB (Voxtral-Small-24B real case) -> VAST_AI_REQUIRED
  cases=$((cases + 1))
  got="$(classify_bytes $((48 * gib)) $((5 * gib)) 11)"
  [[ "$got" == "VAST_AI_REQUIRED" ]] || { echo "self-test FAIL: Voxtral-24B != VAST_AI_REQUIRED (got $got)" >&2; fail=1; }

  # render_human on the Voxtral case actually emits the vast.ai pointer.
  # Cheap regression against silently dropping the recommendation.
  cases=$((cases + 1))
  local human
  human="$(render_human "test/voxtral-small-24b" "VAST_AI_REQUIRED" $((48 * gib)) $((5 * gib)) 11)"
  [[ "$human" == *"VAST.AI REQUIRED"* && "$human" == *"provision.sh"* ]] \
    || { echo "self-test FAIL: render_human dropped the vast.ai pointer" >&2; fail=1; }

  # render_json emits parseable single-line JSON with a stable schema.
  cases=$((cases + 1))
  local jline verdict_j
  jline="$(render_json "test/repo" "LOCAL_SAFE" $((1 * gib)) $((1 * gib)) 1)"
  verdict_j="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['verdict'])" "$jline")"
  [[ "$verdict_j" == "LOCAL_SAFE" ]] || { echo "self-test FAIL: render_json verdict round-trip (got $verdict_j)" >&2; fail=1; }

  if [[ "$fail" -ne 0 ]]; then
    echo "check-model-size self-test: FAIL ($cases cases attempted)" >&2
    return 1
  fi
  echo "check-model-size self-test: OK ($cases cases)"
}

# ---- main --------------------------------------------------------------

main() {
  local json=0
  local repo=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test) run_self_test; exit $? ;;
      --json)      json=1; shift ;;
      -h|--help)   usage; exit 0 ;;
      -*)          echo "check-model-size: unknown flag '$1'" >&2; usage; exit 2 ;;
      *)
        if [[ -n "$repo" ]]; then
          echo "check-model-size: unexpected extra arg '$1' (already have repo '$repo')" >&2
          usage; exit 2
        fi
        repo="$1"; shift ;;
    esac
  done

  if [[ -z "$repo" ]]; then
    usage; exit 2
  fi

  local sizes total max_shard shards verdict
  if ! sizes="$(fetch_hf_sizes "$repo")"; then
    exit 2
  fi
  total="$(printf '%s' "$sizes" | cut -f1)"
  max_shard="$(printf '%s' "$sizes" | cut -f2)"
  shards="$(printf '%s' "$sizes" | cut -f3)"

  if [[ "$shards" -eq 0 ]]; then
    echo "check-model-size: repo '$repo' has no .safetensors siblings (private, gated without token, or a non-safetensors model)" >&2
    exit 2
  fi

  verdict="$(classify_bytes "$total" "$max_shard" "$shards")"

  if [[ "$json" -eq 1 ]]; then
    render_json "$repo" "$verdict" "$total" "$max_shard" "$shards"
  else
    render_human "$repo" "$verdict" "$total" "$max_shard" "$shards"
  fi

  case "$verdict" in
    VAST_AI_REQUIRED) exit 1 ;;
    *) exit 0 ;;
  esac
}

main "$@"
