#!/usr/bin/env bash
# Fail-closed guard for the temporary Canary dependency-review exception.
# This is metadata/static validation only: it never resolves dependencies or
# imports, downloads, executes, or inspects a model.

set -euo pipefail

die() {
  echo "check-canary-dependency-review-guard: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW="$REPO_ROOT/.github/workflows/ci-security.yml"
CANARY_PROJECT="$REPO_ROOT/tools/parity/canary_1b_reference"
CANARY_LOCK="$CANARY_PROJECT/uv.lock"
CANARY_PYPROJECT="$CANARY_PROJECT/pyproject.toml"
CANARY_GATE="$CANARY_PROJECT/dependency_approval_gate.py"
CANARY_FLASH_WORKER="$REPO_ROOT/scripts/publish/vast-ai/run-canary-1b-flash-validation.sh"
CANARY_V2_WORKER="$REPO_ROOT/scripts/publish/vast-ai/run-canary-1b-v2-validation.sh"
CANARY_LOCK_RELATIVE="tools/parity/canary_1b_reference/uv.lock"

check_lightning_lock() {
  local lock_path="$1" summary
  [[ -f "$lock_path" && ! -L "$lock_path" ]] || {
    echo "lock is missing or symlinked: $lock_path" >&2
    return 1
  }
  summary="$(awk '
    function finish() {
      if (name == "lightning") {
        entries++
        if (version == "2.6.6") exact++
      }
    }
    /^\[\[package\]\]$/ {
      finish()
      name = ""
      version = ""
      next
    }
    /^name = "/ {
      name = $0
      sub(/^name = "/, "", name)
      sub(/"$/, "", name)
    }
    /^version = "/ {
      version = $0
      sub(/^version = "/, "", version)
      sub(/"$/, "", version)
    }
    END {
      finish()
      printf "entries=%d exact=%d\n", entries + 0, exact + 0
    }
  ' "$lock_path")"
  [[ "$summary" == "entries=1 exact=1" ]] || {
    echo "Canary lock must contain exactly one lightning==2.6.6 package ($summary)" >&2
    return 1
  }
}

check_pyproject_override() {
  local count
  count="$(grep -Ec '^[[:space:]]*"lightning==2\.6\.6",[[:space:]]*$' "$CANARY_PYPROJECT" || true)"
  [[ "$count" == "1" ]] || {
    echo "Canary pyproject must contain exactly one exact lightning==2.6.6 override (count=$count)" >&2
    return 1
  }
}

check_tracked_lightning_inventory() {
  local base_sha="${BASE_SHA:-HEAD^}" path
  local -a lightning_locks=()
  while IFS= read -r path; do
    if grep -Fqx -- 'name = "lightning"' "$REPO_ROOT/$path"; then
      lightning_locks+=("$path")
    fi
  done < <(git -C "$REPO_ROOT" ls-files -- '*uv.lock')
  [[ " ${lightning_locks[*]} " == *" $CANARY_LOCK_RELATIVE "* ]] || {
    echo "tracked Lightning lock inventory does not include the Canary lock" >&2
    return 1
  }
  for path in "${lightning_locks[@]}"; do
    if [[ "$path" != "$CANARY_LOCK_RELATIVE" ]]; then
      git -C "$REPO_ROOT" cat-file -e "$base_sha^{commit}" >/dev/null 2>&1 || {
        echo "cannot establish the dependency-review base commit: $base_sha" >&2
        return 1
      }
      if ! git -C "$REPO_ROOT" diff --quiet "$base_sha" -- "$path"; then
        echo "non-Canary tracked Lightning lock changed in this review: $path" >&2
        return 1
      fi
    fi
  done
}

check_compatibility_contract() {
  local path
  for path in "$CANARY_GATE" "$CANARY_FLASH_WORKER" "$CANARY_V2_WORKER"; do
    [[ -f "$path" && ! -L "$path" ]] || {
      echo "compatibility contract file is missing or symlinked: $path" >&2
      return 1
    }
    grep -Fq -- 'BLOCKED_SECURITY_INCOMPATIBLE_CANARY_CLOSURE' "$path" || {
      echo "security-block marker is missing: $path" >&2
      return 1
    }
  done
  for path in "$CANARY_FLASH_WORKER" "$CANARY_V2_WORKER"; do
    grep -Fq -- '--compatibility-check' "$path" || {
      echo "worker compatibility-check invocation is missing: $path" >&2
      return 1
    }
  done
}

check_allowlist_exception() {
  local count
  count="$(awk '
    /^[[:space:]]*allow-ghsas:[[:space:]]*>-/ { inside = 1; next }
    inside && /^[[:space:]]*vulnerability-check:/ { inside = 0 }
    inside { count += gsub(/GHSA-qqmf-gpg7-g8gw/, "") }
    END { print count + 0 }
  ' "$WORKFLOW")"
  [[ "$count" == "1" ]] || {
    echo "Canary false-positive exception must occur exactly once in allow-ghsas (count=$count)" >&2
    return 1
  }
}

run_guard() {
  git -C "$REPO_ROOT" ls-files --error-unmatch "$CANARY_LOCK_RELATIVE" >/dev/null 2>&1 || die "Canary uv.lock is not tracked"
  check_tracked_lightning_inventory || die "tracked Lightning lock inventory contract failed"
  check_lightning_lock "$CANARY_LOCK" || die "Canary uv.lock security contract failed"
  check_pyproject_override || die "Canary pyproject security contract failed"
  check_compatibility_contract || die "Canary compatibility security contract failed"
  check_allowlist_exception || die "dependency-review exception contract failed"
  echo "check-canary-dependency-review-guard: PASS"
}

run_self_test() {
  local tmp old_lock duplicate_lock
  for path in "$WORKFLOW" "$CANARY_LOCK" "$CANARY_PYPROJECT" "$CANARY_GATE" "$CANARY_FLASH_WORKER" "$CANARY_V2_WORKER"; do
    [[ -f "$path" && ! -L "$path" ]] || die "self-test contract file is missing or symlinked: $path"
  done
  grep -Fq -- 'allow-ghsas:' "$WORKFLOW" || die "self-test missing allow-ghsas contract"
  grep -Fq -- 'BLOCKED_SECURITY_INCOMPATIBLE_CANARY_CLOSURE' "$CANARY_GATE" || die "self-test missing security-block contract"

  tmp="$(mktemp -d -t vokra-canary-dependency-review.XXXXXX)"
  trap "rm -rf '$tmp'" EXIT
  old_lock="$tmp/old.lock"
  duplicate_lock="$tmp/duplicate.lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.5"' > "$old_lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' > "$duplicate_lock"
  check_lightning_lock "$CANARY_LOCK" || die "self-test rejected the tracked exact lock"
  if check_lightning_lock "$old_lock" >/dev/null 2>&1; then
    die "self-test accepted a vulnerable Lightning lock"
  fi
  if check_lightning_lock "$duplicate_lock" >/dev/null 2>&1; then
    die "self-test accepted duplicate Lightning lock entries"
  fi
  echo "check-canary-dependency-review-guard self-test: PASS"
}

case "${1:-}" in
  "") run_guard ;;
  --self-test)
    [[ "$#" == "1" ]] || die "--self-test accepts no other arguments"
    run_self_test
    ;;
  *) die "unknown argument: $1" ;;
esac
