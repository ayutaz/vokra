#!/usr/bin/env bash
# Generate the authenticated SBV2 JP-Extra Japanese G2P contract.
#
# This worker is intentionally separate from the four-checkpoint SBV2 parity
# worker.  It downloads and executes only the pinned upstream ``text`` source;
# no model/checkpoint bytes, Cargo command, or publication/upload is allowed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
GENERATOR="$VOKRA_ROOT/tools/parity/sbv2_jp_extra/generate_contract.py"
PARITY_PROJECT="${SBV2_G2P_PROJECT:-$VOKRA_ROOT/tools/parity/sbv2}"
SOURCE_URL="https://github.com/litagin02/Style-Bert-VITS2.git"
SOURCE_COMMIT="ef93f388fc1ddf0dc0f598126c1964923f1df94f"

log() { printf '[sbv2-jp-extra-g2p] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: run-sbv2-jp-extra-g2p-contract.sh --expected-head <40-lower-hex> \
  --output <absent-absolute-json> \
  [--work-dir <absent-absolute-dir>]
       run-sbv2-jp-extra-g2p-contract.sh --self-test

VAST-only, model-free official Style-Bert-VITS2 Japanese G2P contract worker.
Vocabulary/tone dimensions are derived from authenticated source symbols;
model compatibility is checked separately by the strict binder/parity worker.
Publication is always NO_UPLOAD.
EOF
}

require_vast() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux ]] || { die 'source acquisition/execution is Linux/VAST-only'; return 2; }
}

require_tooling() {
  local tool
  for tool in git uv sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { die "required tool missing: $tool"; return 2; }
  done
  [[ -f "$GENERATOR" && ! -L "$GENERATOR" ]] || { die 'contract generator is missing or symlinked'; return 2; }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || { die 'pinned SBV2 reference uv.lock is missing'; return 2; }
}

reject_dot_components() {
  local path="$1"
  [[ "$path" == /* ]] || { die "path must be absolute: $path"; return 2; }
  case "$path" in
    */./*|*/../*|*/.|*/..) die "path contains dot components: $path"; return 2 ;;
  esac
}

reject_symlink_ancestors() {
  local path="$1" current=/ rest component
  reject_dot_components "$path"
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || { die "path has symlink ancestor: $path"; return 2; }
    current="$current/"
  done
}

require_absent_path() {
  local path="$1" label="$2"
  reject_symlink_ancestors "$path"
  [[ ! -e "$path" && ! -L "$path" ]] || { die "$label must be absent: $path"; return 2; }
  [[ -d "${path%/*}" && ! -L "${path%/*}" ]] || { die "$label parent must be a real directory: ${path%/*}"; return 2; }
}

require_expected_head() {
  local expected="$1" actual status
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { die 'expected HEAD must be lowercase 40-hex'; return 2; }
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || { die 'cannot inspect Vokra HEAD'; return 2; }
  [[ "$actual" == "$expected" ]] || { die "Vokra HEAD mismatch: $actual != $expected"; return 2; }
  status="$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)"
  [[ -z "$status" ]] || { die 'Vokra checkout must be clean'; return 2; }
}

run_self_test() {
  local failed=0 token
  UV_CACHE_DIR="${UV_CACHE_DIR:-$VOKRA_ROOT/.cache/uv-sbv2-jp-extra}" \
    uv run --no-project --offline --python 3.12 python "$GENERATOR" --self-test \
    || failed=1
  for token in \
    "$SOURCE_URL" "$SOURCE_COMMIT" 'VOKRA_PUBLISH_ON_VAST=1' 'Linux/VAST-only' \
    '--expected-head' '--output' '--work-dir' \
    '--project' 'SBV2_G2P_PROJECT' \
    'NO_UPLOAD' 'git clone' 'git checkout' 'verify_source_tree' \
    'model/checkpoint bytes' 'model_weight_acquisition' 'cargo' \
    'g2p_en' 'distance' 'num2words' '__vokra_num2words_sentinel__' 'numeric-text G2P' \
    'forbidden GPL frontend dependency'; do
    grep -Fq -- "$token" "${BASH_SOURCE[0]}" || { log "self-test missing contract token: $token"; failed=1; }
  done
  for token in symbols_blob japanese_blob mora_blob common_log_blob stdout_wrapper_blob init_blob license_blob; do
    grep -Fq -- "$token" "$GENERATOR" || { log "self-test missing authenticated blob token: $token"; failed=1; }
  done
  if grep -En '(^|[[:space:]])(python3?|pip)([[:space:]]|$)' "${BASH_SOURCE[0]}" | grep -v 'uv run' >/dev/null; then
    log 'self-test found forbidden bare Python/pip invocation'
    failed=1
  fi
  if grep -En '(^|[[:space:]])(cargo|hf|huggingface-cli)([[:space:]]|$)|--push|upload\.sh|publish-one\.sh' "${BASH_SOURCE[0]}" | grep -v 'grep -En' >/dev/null; then
    log 'self-test found forbidden Cargo/model/upload command'
    failed=1
  fi
  if "${BASH_SOURCE[0]}" --self-test --self-test >/dev/null 2>&1; then
    log 'self-test accepted duplicate --self-test'
    failed=1
  fi
  [[ "$failed" == 0 ]] || return 1
  echo 'run-sbv2-jp-extra-g2p-contract.sh self-test: PASS'
}

main() {
  local self_test=0 expected_head='' output='' work='' arg source_dir
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test) [[ "$self_test" == 0 ]] || { die 'duplicate --self-test'; return 2; }; self_test=1; shift ;;
      --expected-head) [[ -z "$expected_head" && $# -ge 2 ]] || { die 'duplicate or missing --expected-head'; return 2; }; expected_head="$2"; shift 2 ;;
      --output) [[ -z "$output" && $# -ge 2 ]] || { die 'duplicate or missing --output'; return 2; }; output="$2"; shift 2 ;;
      --work-dir) [[ -z "$work" && $# -ge 2 ]] || { die 'duplicate or missing --work-dir'; return 2; }; work="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ "$self_test" == 1 ]]; then
    [[ -z "$expected_head$output$work" ]] || { die '--self-test accepts no execution arguments'; return 2; }
    run_self_test
    return $?
  fi
  [[ -n "$expected_head" && -n "$output" ]] || { usage; die 'expected HEAD and output are required'; return 2; }
  require_vast
  require_tooling
  require_expected_head "$expected_head"
  if [[ -z "$work" ]]; then
    work="$(mktemp -d -p /tmp sbv2-jp-extra-g2p.XXXXXX)"
    rmdir "$work"
  fi
  require_absent_path "$work" 'worker directory'
  require_absent_path "$output" 'contract output'
  require_absent_path "$output.sha256" 'contract SHA-256 sidecar'
  [[ "$output" != "$work"/* && "$work" != "$output"/* ]] || { die 'output and worker directory must be disjoint'; return 2; }
  mkdir -m 700 "$work"
  source_dir="$work/official-source"
  git clone --quiet --no-checkout "$SOURCE_URL" "$source_dir"
  git -C "$source_dir" fetch --quiet --depth=1 origin "$SOURCE_COMMIT"
  git -C "$source_dir" checkout --quiet --detach "$SOURCE_COMMIT"
  log "verified source checkout at $SOURCE_COMMIT before official import"
  VOKRA_PUBLISH_ON_VAST=1 UV_CACHE_DIR="${UV_CACHE_DIR:-$work/uv-cache}" \
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$GENERATOR" \
    --vokra-root "$VOKRA_ROOT" --expected-head "$expected_head" \
    --source-dir "$source_dir" \
    --work-dir "$work" --output "$output"
  log 'contract generation complete; publication=NO_UPLOAD; model weights=NOT_ACQUIRED'
  return 0
}

main "$@"
