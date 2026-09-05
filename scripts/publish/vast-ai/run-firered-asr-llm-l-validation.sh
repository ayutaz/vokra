#!/usr/bin/env bash
# Terminally blocked VAST worker for FireRedASR-LLM-L.  No source/model
# acquisition, dedicated uv sync, Cargo, conversion, or upload is permitted
# until the upstream tarball contract and source revision are authenticated.
# shellcheck disable=SC2034 # identity constants are self-test contract data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/firered_asr_llm_l_inspect.py"
MODEL_REPOSITORY="FireRedTeam/FireRedASR-LLM-L"
MODEL_REVISION="9837461f78d15ee66565d00aaec0bc5497d7fbc1"
SOURCE_REPOSITORY="https://github.com/FireRedTeam/FireRedASR"
SOURCE_REVISION=""
PUBLICATION_STATUS="NO_UPLOAD"

die() { echo "run-firered-asr-llm-l-validation: $*" >&2; exit 2; }
self_test() {
  local self="${BASH_SOURCE[0]}" fail=0 gate_line
  [[ -f "$INSPECTOR" ]] || { echo 'self-test FAIL: inspector missing' >&2; fail=1; }
  for needle in "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REPOSITORY" "SOURCE_REVISION" "dependency-gate" "BLOCKED_SOURCE_FORMAT_AND_AUTHENTICATION" "NO_UPLOAD" "VAST"; do
    grep -Fq -- "$needle" "$self" || { echo "self-test FAIL: missing $needle" >&2; fail=1; }
  done
  if grep -En '(^|[;&|])[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then
    echo 'self-test FAIL: raw Python/pip invocation found' >&2; fail=1
  fi
  gate_line="$(grep -n 'INSPECTOR.*--dependency-gate' "$self" | tail -1 | cut -d: -f1)"
  [[ -n "$gate_line" ]] || { echo 'self-test FAIL: gate invocation missing' >&2; fail=1; }
  local post_gate normalized token uv_word sync_word run_word project_word snapshot_word hf_word clone_word mkdir_word work_word cargo_word bundle_word upload_word push_word
  post_gate="$(tail -n +"$gate_line" "$self")"
  normalized="$(printf '%s' "$post_gate" | tr -d "\"'\\")"
  uv_word='u'; uv_word+='v'; run_word="$uv_word run"; sync_word="$uv_word sync"; project_word="$run_word --project"
  snapshot_word='snapshot'; snapshot_word+='_download'; hf_word='hf_'; hf_word+='hub_download'; clone_word='git'; clone_word+=' clone'; mkdir_word='mkdir'; mkdir_word+=' -p'; work_word='WORK_DIR'; cargo_word='cargo'; bundle_word='git'; bundle_word+=' bundle'; upload_word='upload'; push_word='git'; push_word+=' push'
  for token in "$sync_word" "$project_word" "$snapshot_word" "$hf_word" "$clone_word" "$mkdir_word" "$work_word" "$cargo_word" "$bundle_word" "$upload_word" "$push_word" '--push'; do
    [[ "$normalized" != *"$token"* ]] || { echo "self-test FAIL: post-gate effect found: $token" >&2; fail=1; }
  done
  (( fail == 0 )) && echo 'run-firered-asr-llm-l-validation.sh self-test: PASS' || return 1
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test; exit 0
fi
[[ $# == 0 ]] || die 'arguments are not accepted'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
export UV_CACHE_DIR="${FIRERED_UV_CACHE_DIR:-/tmp/vokra-firered-uv-cache}"

# stdlib-only and terminal: the gate must pass before any project invocation,
# sync, work path creation, acquisition, source checkout, or Cargo operation.
uv run --no-project --python 3.12 python "$INSPECTOR" --dependency-gate || die 'FireRed source/dependency gate is unresolved'
die 'unreachable: FireRed gate unexpectedly passed without authenticated source closure'
