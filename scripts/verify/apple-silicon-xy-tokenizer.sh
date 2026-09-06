#!/usr/bin/env bash
# No-download Apple gate for XY-Tokenizer's authenticated inspection packet.
#
# The current packet has no authenticated topology/native runtime, so this
# verifier records a bounded BLOCKED result and never runs Cargo, Python model
# code, Metal, conversion, upload, or model execution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
MANIFEST=""
EXPECTED_MANIFEST_SHA256=""
EXPECTED_HEAD=""
EVIDENCE_DIR=""

die() { printf '[xy-tokenizer-apple] ERROR: %s\n' "$*" >&2; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-xy-tokenizer.sh \
  --inspection-manifest <blocked-manifest.json> \
  --expected-manifest-sha256 <64-hex> --expected-head <40-hex> \
  --evidence-dir <absent-dir>
       apple-silicon-xy-tokenizer.sh --self-test

This is a no-download, no-model, no-Cargo Apple verification scaffold.  The
authenticated inspection manifest must explicitly remain BLOCKED because the
native tensor topology and runtime are not authenticated.  It requires a
clean Darwin arm64 checkout and records no success-shaped parity result.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

reject_unsafe_path() {
  local path="$1" label="$2" cursor
  [[ "$path" == /* ]] || die "$label must be absolute"
  [[ "$path" != *"/../"* && "$path" != */.. && "$path" != *"/./"* && "$path" != */. ]] \
    || die "$label contains a dot component"
  cursor="$path"
  while [[ "$cursor" != / ]]; do
    [[ ! -L "$cursor" ]] || die "$label has a symlinked component: $cursor"
    cursor="${cursor%/*}"
    [[ -n "$cursor" ]] || cursor=/
  done
}

require_clean_expected_head() {
  [[ "$EXPECTED_HEAD" =~ ^[0-9a-fA-F]{40}$ ]] \
    || die '--expected-head must be exactly 40 hexadecimal characters'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die 'Apple checkout must be clean before evidence creation'
  local actual
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual" == "$EXPECTED_HEAD" ]] \
    || die "checkout HEAD $actual does not match expected $EXPECTED_HEAD"
}

require_blocked_manifest() {
  local manifest="$1" expected_sha="$2"
  [[ -f "$manifest" && ! -L "$manifest" ]] \
    || die 'inspection manifest must be a regular non-symlink file'
  [[ "$(sha256_file "$manifest")" == "$expected_sha" ]] \
    || die 'inspection manifest SHA-256 does not match the externally supplied digest'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$manifest" <<'PY'
import json
import pathlib
import sys

def reject(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
    expected = {
        "format": "vokra-xy-tokenizer-prepared-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("manifest is not the exact blocked inspection contract")
    if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise ValueError("inspection evidence is not authenticated")
    if manifest.get("collection_status") != "AUTHENTICATED":
        raise ValueError("inspection collection is not authenticated")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or "TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER" not in blockers:
        raise ValueError("topology blocker is missing")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"BLOCKED manifest gate: {error}")
PY
}

require_apple_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] \
    || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing maintainer-Mac execution'
  [[ "$(uname -s)" == Darwin ]] || die 'XY-Tokenizer Apple gate requires Darwin'
  [[ "$(uname -m)" == arm64 ]] || die 'XY-Tokenizer Apple gate requires arm64'
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" fail=0
  for required in '--expected-head' '--expected-manifest-sha256' 'require_clean_expected_head' \
    'TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER' 'NOT_IMPLEMENTED_FAIL_CLOSED' \
    'VOKRA_REMOTE_APPLE_SILICON' 'Darwin' 'arm64' 'NO_UPLOAD' 'NOT_RUN' \
    'no-download' 'no-model'; do
    grep -Fq -- "$required" "$script_path" || { printf 'self-test missing: %s\n' "$required" >&2; fail=1; }
  done
  if grep -En '^[[:space:]]*(curl|wget|python3?|pip|git[[:space:]]+(clone|fetch|pull|push))([[:space:]]|$)' "$script_path" >/dev/null; then
    printf 'self-test found forbidden acquisition command\n' >&2; fail=1
  fi
  for bad in '--expected-head' '--expected-head -bad' '--expected-head a --expected-head b' \
    '--expected-manifest-sha256' '--expected-manifest-sha256 -bad' \
    '--expected-manifest-sha256 a --expected-manifest-sha256 b' '--inspection-manifest' \
    '--evidence-dir'; do
    if eval "\"$script_path\" $bad" >/dev/null 2>&1; then
      printf 'self-test accepted malformed/duplicate option: %s\n' "$bad" >&2; fail=1
    fi
  done
  (( fail == 0 )) || return 1
  echo 'apple-silicon-xy-tokenizer.sh self-test: OK'
}

main() {
  local self_test=0 seen_manifest=0 seen_manifest_sha=0 seen_head=0 seen_evidence=0
  while (( $# > 0 )); do
    case "$1" in
      --inspection-manifest)
        (( seen_manifest == 0 )) || die 'duplicate --inspection-manifest'
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--inspection-manifest requires a path'
        MANIFEST="$2"; seen_manifest=1; shift 2 ;;
      --expected-manifest-sha256)
        (( seen_manifest_sha == 0 )) || die 'duplicate --expected-manifest-sha256'
        [[ $# -ge 2 && "$2" =~ ^[0-9a-fA-F]{64}$ ]] || die '--expected-manifest-sha256 requires 64 hex characters'
        EXPECTED_MANIFEST_SHA256="${2,,}"; seen_manifest_sha=1; shift 2 ;;
      --expected-head)
        (( seen_head == 0 )) || die 'duplicate --expected-head'
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--expected-head requires a path-like value'
        EXPECTED_HEAD="$2"; seen_head=1; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a path'
        EVIDENCE_DIR="$2"; seen_evidence=1; shift 2 ;;
      --self-test) (( self_test == 0 )) || die 'duplicate --self-test'; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self_test == 1 )); then
    (( $# == 0 && seen_manifest == 0 && seen_manifest_sha == 0 && seen_head == 0 && seen_evidence == 0 )) \
      || die '--self-test accepts no other arguments'
    run_self_test
    return
  fi
  [[ "$seen_manifest$seen_manifest_sha$seen_head$seen_evidence" == 1111 ]] \
    || die 'all inspection, hash, exact-head, and evidence options are required'
  reject_unsafe_path "$VOKRA_ROOT" 'Vokra root'
  reject_unsafe_path "$MANIFEST" 'inspection manifest'
  reject_unsafe_path "$EVIDENCE_DIR" 'evidence directory'
  require_clean_expected_head
  require_blocked_manifest "$MANIFEST" "$EXPECTED_MANIFEST_SHA256"
  require_apple_host
  [[ ! -e "$EVIDENCE_DIR" && ! -L "$EVIDENCE_DIR" ]] \
    || die 'evidence directory must be absent and non-symlink'
  [[ -d "$(dirname "$EVIDENCE_DIR")" && ! -L "$(dirname "$EVIDENCE_DIR")" ]] \
    || die 'evidence parent must be an existing non-symlink directory'
  mkdir "$EVIDENCE_DIR"
  {
    echo "verdict=BLOCKED_PENDING_AUTHENTICATED_TENSOR_MANIFEST"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_head=$EXPECTED_HEAD"
    echo "inspection_manifest=$MANIFEST"
    echo "inspection_manifest_sha256=$EXPECTED_MANIFEST_SHA256"
    echo "cpu_vs_official=NOT_RUN"
    echo "metal_vs_official=NOT_RUN"
    echo "metal_vs_cpu=NOT_RUN"
    echo "model_execution=NOT_PERFORMED"
    echo "download=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "publication=NO_UPLOAD"
  } > "$EVIDENCE_DIR/summary.txt"
  printf '[xy-tokenizer-apple] BLOCKED: authenticated tensor topology/runtime is not available; evidence=%s\n' "$EVIDENCE_DIR" >&2
  return 2
}

main "$@"
