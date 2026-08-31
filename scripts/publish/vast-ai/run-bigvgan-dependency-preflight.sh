#!/usr/bin/env bash
# VAST/Linux-only, model-free BigVGAN dependency closure preflight.
# This worker stages only the exact wheels named by the dedicated uv.lock;
# it does not install, import, execute, or publish any package or model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bigvgan"
PREFLIGHT="$PARITY_PROJECT/preflight_linux_closure.py"
AUDITOR="$PARITY_PROJECT/audit_linux_closure.py"
DEFAULT_LOCK="$PARITY_PROJECT/uv.lock"
EXPECTED_LOCK_SHA256="80ef4819e06ad5b78675da245917bf852ee7952847a1be69fbb2baf97f91b36e"

log() { printf '[bigvgan-dependency-preflight] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-bigvgan-dependency-preflight.sh --artifacts-dir <absent-dir> --output <absent-json> [--lock <uv.lock>]
       run-bigvgan-dependency-preflight.sh --self-test

The real path is VAST/Linux-only. It downloads the active x86_64 CPython
3.12 glibc wheels selected from the supplied pinned uv.lock, verifies each
locked URL/hash/size, and emits a no-upload owner-review candidate. No
package-manager installation, package import, model acquisition, native build
execution, owner signature, or publication is performed.
EOF
}

require_regular_lock() {
  local lock="$1"
  [[ -f "$lock" && ! -L "$lock" ]] || { die "lock is missing, non-regular, or symlinked: $lock"; return 2; }
}

verify_committed_lock() {
  local lock="$1" actual
  require_regular_lock "$lock"
  actual="$(sha256sum "$lock" | awk '{print $1}')"
  [[ "$actual" == "$EXPECTED_LOCK_SHA256" ]] || {
    die "lock bytes do not match committed BigVGAN lock identity: $lock"
    return 2
  }
}

require_clean_checkout() {
  [[ -d "$VOKRA_ROOT/.git" ]] || { die 'Vokra checkout is missing'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || { die 'VAST checkout must be clean'; return 2; }
}

require_absent_destination() {
  local target="$1" parent
  [[ "$target" = /* ]] || { die "destination must be an absolute path: $target"; return 2; }
  [[ ! -e "$target" && ! -L "$target" ]] || { die "destination must be absent: $target"; return 2; }
  parent="$(dirname "$target")"
  [[ -d "$parent" && ! -L "$parent" ]] || {
    die "destination parent must be an existing non-symlink directory: $parent"
    return 2
  }
}

run_python() {
  local script="$1"
  shift
  PYTHONDONTWRITEBYTECODE=1 UV_NO_CACHE=1 \
    uv run --no-project --offline --python 3.12 python "$script" "$@"
}

verify_candidate() {
  local output="$1" marker
  grep -Fq -- "\"lock_sha256\": \"$EXPECTED_LOCK_SHA256\"" "$output" || {
    die 'audit output lock identity does not match the committed BigVGAN lock'
    return 2
  }
  for marker in '"decision": "OWNER_REVIEW_REQUIRED"' \
    '"dependency_review": "BLOCKED_UNREVIEWED_TRANSITIVE"' \
    '"publication": "NO_UPLOAD"' '"status": "OWNER_SIGNOFF_REQUIRED"'; do
    grep -Fq -- "$marker" "$output" || {
      die "audit output is missing mandatory fail-closed marker: $marker"
      return 2
    }
  done
}

run_self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 bad_uv bad_install bad_native bad_model bad_upload bad_publisher
  for token in 'run-bigvgan-dependency-preflight.sh --self-test' \
    'preflight_linux_closure.py' 'audit_linux_closure.py' '--no-project --offline --python 3.12' \
    'OWNER_REVIEW_REQUIRED' 'BLOCKED_UNREVIEWED_TRANSITIVE' 'NO_UPLOAD' 'VOKRA_PUBLISH_ON_VAST=1' \
    'EXPECTED_LOCK_SHA256' 'lock_sha256' 'git status --porcelain --untracked-files=all'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing contract: $token"; fail=1; }
  done
  bad_uv='u'; bad_uv+='v sync'
  bad_install='u'; bad_install+='v pip install'
  bad_native='car'; bad_native+='go'
  bad_model='model'; bad_model+=' download'
  bad_upload='--'; bad_upload+='push'
  for forbidden in "$bad_uv" "$bad_install" "$bad_native" "$bad_model" "$bad_upload"; do
    if grep -Fq -- "$forbidden" "$path"; then
      log "self-test FAIL: forbidden execution token found: $forbidden"
      fail=1
    fi
  done
  bad_publisher='publish'; bad_publisher+='-one.sh'
  if grep -Fq -- "$bad_publisher" "$path"; then
    log 'self-test FAIL: publication helper found'
    fail=1
  fi
  if ! run_python "$PREFLIGHT" --self-test; then
    log 'self-test FAIL: locked wheel staging self-test'
    fail=1
  fi
  if ! run_python "$AUDITOR" --self-test; then
    log 'self-test FAIL: closure audit self-test'
    fail=1
  fi
  local probe custom_lock
  probe="$(mktemp -d "${TMPDIR:-/tmp}/bigvgan-preflight-gate.XXXXXX")"
  if VOKRA_PUBLISH_ON_VAST=0 "$path" --artifacts-dir "$probe/artifacts" --output "$probe/candidate.json" >/dev/null 2>&1; then
    log 'self-test FAIL: non-VAST invocation was accepted'
    fail=1
  fi
  [[ ! -e "$probe/artifacts" && ! -e "$probe/candidate.json" ]] \
    || { log 'self-test FAIL: non-VAST gate created output'; fail=1; }
  custom_lock="$probe/tampered.lock"
  cp "$DEFAULT_LOCK" "$custom_lock"
  printf '\n# self-test tamper\n' >> "$custom_lock"
  if verify_committed_lock "$custom_lock" >/dev/null 2>&1; then
    log 'self-test FAIL: tampered/custom lock was accepted'
    fail=1
  fi
  [[ ! -e "$probe/artifacts" && ! -e "$probe/candidate.json" ]] \
    || { log 'self-test FAIL: custom lock check created output'; fail=1; }
  rm -f "$custom_lock"
  rmdir "$probe"
  if [[ "$fail" -ne 0 ]]; then
    return 2
  fi
  printf 'run-bigvgan-dependency-preflight.sh self-test: PASS\n'
}

main() {
  local self_test=0 lock="$DEFAULT_LOCK" artifacts="" output="" arg
  while (($#)); do
    arg="$1"
    case "$arg" in
      --self-test)
        [[ "$self_test" -eq 0 && -z "$artifacts" && -z "$output" && "$lock" == "$DEFAULT_LOCK" ]] \
          || { usage; return 2; }
        self_test=1
        shift
        ;;
      --lock)
        [[ "$self_test" -eq 0 && $# -ge 2 && -z "$artifacts" && -z "$output" ]] \
          || { usage; return 2; }
        lock="$2"
        shift 2
        ;;
      --artifacts-dir)
        [[ "$self_test" -eq 0 && $# -ge 2 && -z "$artifacts" ]] || { usage; return 2; }
        artifacts="$2"
        shift 2
        ;;
      --output)
        [[ "$self_test" -eq 0 && $# -ge 2 && -z "$output" ]] || { usage; return 2; }
        output="$2"
        shift 2
        ;;
      *) usage; return 2 ;;
    esac
  done
  if [[ "$self_test" -eq 1 ]]; then
    (($# == 0)) || { usage; return 2; }
    run_self_test
    return
  fi
  [[ -n "$artifacts" && -n "$output" ]] || { usage; return 2; }
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; return 2; }
  [[ "$(uname -s)" == Linux ]] || { die 'dependency preflight is VAST/Linux-only'; return 2; }
  [[ "$(uname -m)" == x86_64 ]] || { die 'VAST host must be x86_64'; return 2; }
  command -v sha256sum >/dev/null 2>&1 || { die 'sha256sum is required'; return 2; }
  [[ -f "$PREFLIGHT" && ! -L "$PREFLIGHT" && -f "$AUDITOR" && ! -L "$AUDITOR" ]] \
    || { die 'BigVGAN preflight/audit scripts are missing or symlinked'; return 2; }
  require_clean_checkout
  verify_committed_lock "$lock"
  require_absent_destination "$artifacts"
  require_absent_destination "$output"
  run_python "$PREFLIGHT" --lock "$lock" --artifacts-dir "$artifacts"
  run_python "$AUDITOR" --lock "$lock" --artifacts-dir "$artifacts" --output "$output"
  verify_candidate "$output"
  log "PASS: locked Linux closure staged and candidate emitted at $output; owner review and NO_UPLOAD remain required"
}

main "$@"
