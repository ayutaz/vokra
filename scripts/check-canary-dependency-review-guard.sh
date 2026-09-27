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

check_package_lock_version() {
  local lock_path="$1" package_name="$2" expected="$3" summary
  [[ -f "$lock_path" && ! -L "$lock_path" ]] || {
    echo "lock is missing or symlinked: $lock_path" >&2
    return 1
  }
  summary="$(awk -v package_name="$package_name" -v expected="$expected" '
    function finish() {
      if (name == package_name) {
        entries++
        if (version == expected) exact++
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
    echo "lock must contain exactly one $package_name==$expected package ($summary)" >&2
    return 1
  }
}

check_lightning_lock() {
  check_package_lock_version "$1" lightning 2.6.6
}

check_package_lock_before_target() {
  local lock_path="$1" package_name="$2" summary
  summary="$(awk -v package_name="$package_name" '
    function finish() {
      if (name == package_name) {
        entries++
        split(version, parts, ".")
        if (version ~ /^[0-9]+\.[0-9]+\.[0-9]+$/ &&
            ((parts[1] + 0) < 2 ||
             ((parts[1] + 0) == 2 && (parts[2] + 0) < 6) ||
             ((parts[1] + 0) == 2 && (parts[2] + 0) == 6 && (parts[3] + 0) < 6))) older++
      }
    }
    /^\[\[package\]\]$/ { finish(); name=""; version=""; next }
    /^name = "/ { name=$0; sub(/^name = "/, "", name); sub(/"$/, "", name) }
    /^version = "/ { version=$0; sub(/^version = "/, "", version); sub(/"$/, "", version) }
    END { finish(); printf "entries=%d older=%d\n", entries + 0, older + 0 }
  ' "$lock_path")"
  [[ "$summary" == "entries=1 older=1" ]]
}

check_patched_lightning_update() {
  local base_lock="$1" new_lock="$2" path="$3" tmp lock_file
  for lock_file in "$base_lock" "$new_lock"; do
    [[ -f "$lock_file" && ! -L "$lock_file" ]] || {
      echo "Lightning lock is missing or symlinked: $lock_file" >&2
      return 1
    }
  done
  if check_package_lock_version "$base_lock" lightning 2.6.6 >/dev/null 2>&1 &&
    check_package_lock_version "$base_lock" pytorch-lightning 2.6.6 >/dev/null 2>&1; then
    # Once both packages are securely patched, later lock refreshes may move
    # unrelated packages.  Keep the patched pair exact and unique.
    check_package_lock_version "$new_lock" pytorch-lightning 2.6.6 || return 1
    check_lightning_lock "$new_lock" || return 1
    return 0
  fi
  if ! check_package_lock_before_target "$base_lock" lightning ||
    ! check_package_lock_before_target "$base_lock" pytorch-lightning; then
    echo "non-Canary Lightning base must contain one pre-2.6.6 entry for each package ($path)" >&2
    return 1
  fi
  check_package_lock_version "$new_lock" pytorch-lightning 2.6.6 || return 1
  check_lightning_lock "$new_lock" || return 1

  # A patched-only update may refresh resolver markers, but it must not add,
  # remove, or change any other locked package or dependency edge.
  tmp="$(mktemp -d -t vokra-canary-dependency-review.XXXXXX)"
  awk '
    /^\[\[package\]\]$/ { package_name=""; next }
    /^name = "/ {
      package_name=$0; sub(/^name = "/, "", package_name); sub(/"$/, "", package_name)
      next
    }
    /^version = "/ && package_name != "" && package_name != "lightning" && package_name != "pytorch-lightning" { print package_name "|" $0 }
    /^[[:space:]]*\{ name = "/ && package_name != "" && package_name != "lightning" && package_name != "pytorch-lightning" {
      child=$0; sub(/^.*name = "/, "", child); sub(/".*$/, "", child)
      if ((child == "lightning" || child == "pytorch-lightning") && package_name ~ /^vokra-/) next
      print package_name "|" $0
    }
  ' "$base_lock" | sed -E 's/, marker = "[^"]*"//' | LC_ALL=C sort > "$tmp/base.inventory"
  awk '
    /^\[\[package\]\]$/ { package_name=""; next }
    /^name = "/ {
      package_name=$0; sub(/^name = "/, "", package_name); sub(/"$/, "", package_name)
      next
    }
    /^version = "/ && package_name != "" && package_name != "lightning" && package_name != "pytorch-lightning" { print package_name "|" $0 }
    /^[[:space:]]*\{ name = "/ && package_name != "" && package_name != "lightning" && package_name != "pytorch-lightning" {
      child=$0; sub(/^.*name = "/, "", child); sub(/".*$/, "", child)
      if ((child == "lightning" || child == "pytorch-lightning") && package_name ~ /^vokra-/) next
      print package_name "|" $0
    }
  ' "$new_lock" | sed -E 's/, marker = "[^"]*"//' | LC_ALL=C sort > "$tmp/new.inventory"
  if ! cmp -s "$tmp/base.inventory" "$tmp/new.inventory"; then
    rm -rf "$tmp"
    echo "unrelated locked package or dependency changed in this review: $path" >&2
    return 1
  fi
  rm -rf "$tmp"
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
  local base_sha="${BASE_SHA:-HEAD^}" path snapshot
  local -a lightning_locks=()
  git -C "$REPO_ROOT" cat-file -e "$base_sha^{commit}" >/dev/null 2>&1 || {
    echo "cannot establish the dependency-review base commit: $base_sha" >&2
    return 1
  }
  while IFS= read -r path; do
    if grep -Fqx -- 'name = "lightning"' "$REPO_ROOT/$path"; then
      lightning_locks+=("$path")
    fi
  done < <(git -C "$REPO_ROOT" ls-files -- '*uv.lock')
  [[ " ${lightning_locks[*]} " == *" $CANARY_LOCK_RELATIVE "* ]] || {
    echo "tracked Lightning lock inventory does not include the Canary lock" >&2
    return 1
  }
  while IFS= read -r path; do
    if git -C "$REPO_ROOT" show "$base_sha:$path" | grep -Fqx -- 'name = "lightning"'; then
      [[ -f "$REPO_ROOT/$path" && ! -L "$REPO_ROOT/$path" ]] || {
        echo "tracked Lightning lock was removed or symlinked in this review: $path" >&2
        return 1
      }
    fi
  done < <(git -C "$REPO_ROOT" ls-tree -r --name-only "$base_sha" -- '*uv.lock')
  snapshot="$(mktemp -d -t vokra-canary-dependency-review.XXXXXX)"
  for path in "${lightning_locks[@]}"; do
    if [[ "$path" != "$CANARY_LOCK_RELATIVE" ]]; then
      git -C "$REPO_ROOT" cat-file -e "$base_sha:$path" >/dev/null 2>&1 || {
        rm -rf "$snapshot"
        echo "non-Canary Lightning lock was added without a reviewed base: $path" >&2
        return 1
      }
      if ! git -C "$REPO_ROOT" diff --quiet "$base_sha" -- "$path"; then
        git -C "$REPO_ROOT" show "$base_sha:$path" > "$snapshot/base.lock"
        check_patched_lightning_update "$snapshot/base.lock" "$REPO_ROOT/$path" "$path" || {
          rm -rf "$snapshot"
          return 1
        }
      fi
    fi
  done
  rm -rf "$snapshot"
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
  local tmp old_lock duplicate_lock newer_lock patched_lock paired_base paired_new partial_new duplicate_pt unrelated_base unrelated_new unrelated_edge_new secure_base secure_new secure_downgrade secure_duplicate
  for path in "$WORKFLOW" "$CANARY_LOCK" "$CANARY_PYPROJECT" "$CANARY_GATE" "$CANARY_FLASH_WORKER" "$CANARY_V2_WORKER"; do
    [[ -f "$path" && ! -L "$path" ]] || die "self-test contract file is missing or symlinked: $path"
  done
  grep -Fq -- 'allow-ghsas:' "$WORKFLOW" || die "self-test missing allow-ghsas contract"
  grep -Fq -- 'BLOCKED_SECURITY_INCOMPATIBLE_CANARY_CLOSURE' "$CANARY_GATE" || die "self-test missing security-block contract"

  tmp="$(mktemp -d -t vokra-canary-dependency-review.XXXXXX)"
  trap "rm -rf '$tmp'" EXIT
  old_lock="$tmp/old.lock"
  duplicate_lock="$tmp/duplicate.lock"
  newer_lock="$tmp/newer.lock"
  patched_lock="$tmp/patched.lock"
  paired_base="$tmp/paired-base.lock"
  paired_new="$tmp/paired-new.lock"
  partial_new="$tmp/partial-new.lock"
  duplicate_pt="$tmp/duplicate-pt.lock"
  unrelated_base="$tmp/unrelated-base.lock"
  unrelated_new="$tmp/unrelated-new.lock"
  unrelated_edge_new="$tmp/unrelated-edge-new.lock"
  secure_base="$tmp/secure-base.lock"
  secure_new="$tmp/secure-new.lock"
  secure_downgrade="$tmp/secure-downgrade.lock"
  secure_duplicate="$tmp/secure-duplicate.lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.5"' > "$old_lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' > "$duplicate_lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.7"' > "$newer_lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' > "$patched_lock"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.5"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.5"' '[[package]]' 'name = "safe-package"' 'version = "1.0.0"' > "$paired_base"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "safe-package"' 'version = "1.0.0"' > "$paired_new"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.5"' > "$partial_new"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' > "$duplicate_pt"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.5"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.5"' '[[package]]' 'name = "safe-package"' 'version = "1.0.0"' 'dependencies = [' '    { name = "base-dependency" },' ']' > "$unrelated_base"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "safe-package"' 'version = "2.0.0"' 'dependencies = [' '    { name = "base-dependency" },' ']' > "$unrelated_new"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "safe-package"' 'version = "1.0.0"' 'dependencies = [' '    { name = "unrelated-dependency" },' ']' > "$unrelated_edge_new"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "safe-package"' 'version = "1.0.0"' 'dependencies = [' '    { name = "base-dependency" },' ']' > "$secure_base"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "safe-package"' 'version = "9.9.9"' 'dependencies = [' '    { name = "later-dependency" },' ']' > "$secure_new"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.5"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' > "$secure_downgrade"
  printf '%s\n' '[[package]]' 'name = "lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' '[[package]]' 'name = "pytorch-lightning"' 'version = "2.6.6"' > "$secure_duplicate"
  check_lightning_lock "$CANARY_LOCK" || die "self-test rejected the tracked exact lock"
  if check_lightning_lock "$old_lock" >/dev/null 2>&1; then
    die "self-test accepted a vulnerable Lightning lock"
  fi
  if check_lightning_lock "$duplicate_lock" >/dev/null 2>&1; then
    die "self-test accepted duplicate Lightning lock entries"
  fi
  if check_patched_lightning_update "$newer_lock" "$patched_lock" self-test >/dev/null 2>&1; then
    die "self-test accepted a Lightning downgrade"
  fi
  check_patched_lightning_update "$paired_base" "$paired_new" self-test || die "self-test rejected the paired patched update"
  if check_patched_lightning_update "$paired_base" "$partial_new" self-test >/dev/null 2>&1; then
    die "self-test accepted a partially patched Lightning closure"
  fi
  if check_patched_lightning_update "$paired_base" "$duplicate_pt" self-test >/dev/null 2>&1; then
    die "self-test accepted duplicate pytorch-lightning entries"
  fi
  if check_patched_lightning_update "$unrelated_base" "$unrelated_new" self-test >/dev/null 2>&1; then
    die "self-test accepted an unrelated package update"
  fi
  if check_patched_lightning_update "$unrelated_base" "$unrelated_edge_new" self-test >/dev/null 2>&1; then
    die "self-test accepted an unrelated dependency edge"
  fi
  check_patched_lightning_update "$secure_base" "$secure_new" self-test || die "self-test rejected a safe post-patch lock refresh"
  if check_patched_lightning_update "$secure_base" "$secure_downgrade" self-test >/dev/null 2>&1; then
    die "self-test accepted a post-patch downgrade"
  fi
  if check_patched_lightning_update "$secure_base" "$secure_duplicate" self-test >/dev/null 2>&1; then
    die "self-test accepted a post-patch duplicate"
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
