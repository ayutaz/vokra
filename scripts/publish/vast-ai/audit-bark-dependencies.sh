#!/usr/bin/env bash
# Audit the exact Bark Python closure on VAST without acquiring model weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bark"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=23000000
MIN_FREE_DISK_KIB=4000000

usage() {
  cat <<'EOF' >&2
usage: audit-bark-dependencies.sh --output <audit.json>
       audit-bark-dependencies.sh --self-test

The target environment must already have been synchronized by an authorized,
named VAST workflow. This command never runs uv sync, never invokes Cargo,
never imports Bark/Transformers model code, and never downloads model weights.
The audit fetches only the two exact upstream LICENSE URLs for the pinned
suno/bark-small and suno/bark revisions, plus exact locked PyPI sdist archives
needed when an installed wheel has no publisher LICENSE/NOTICE file. Sdist
archives are bounded, hash/size checked, and inspected in memory only. The Bark
source-code LICENSE pin is reported as a blocker when the existing contract
does not contain one.
EOF
}

run_audit() {
  local output="$1"
  local mem_kib free_kib root_real project_real output_real tracked
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || {
    echo "bark dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required on VAST" >&2
    return 2
  }
  local tool
  for tool in uv readelf git awk df uname; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "bark dependency audit: required tool is unavailable: $tool" >&2
      return 2
    }
  done
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
    echo "bark dependency audit: exact Linux x86_64 VAST host required" >&2
    return 2
  }
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_VAST_MEM_KIB" ]] || {
    echo "bark dependency audit: VAST host has less than 23 GiB RAM" >&2
    return 2
  }
  free_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_FREE_DISK_KIB" ]] || {
    echo "bark dependency audit: VAST checkout filesystem has less than 4 GiB free" >&2
    return 2
  }
  [[ -d "$VOKRA_ROOT/.git" ]] || {
    echo "bark dependency audit: VOKRA_ROOT is not a git checkout" >&2
    return 2
  }
  root_real="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "bark dependency audit: cannot resolve git checkout root" >&2
    return 2
  }
  [[ "$root_real" == "$VOKRA_ROOT" ]] || {
    echo "bark dependency audit: VOKRA_ROOT is not the canonical checkout root" >&2
    return 2
  }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "bark dependency audit: checkout must be clean" >&2
    return 2
  }
  for tracked in \
    tools/parity/bark/pyproject.toml \
    tools/parity/bark/uv.lock \
    tools/parity/bark/license_gate_manifest.json \
    tools/parity/bark/dependency_audit.py; do
    git -C "$VOKRA_ROOT" ls-files --error-unmatch -- "$tracked" >/dev/null 2>&1 || {
      echo "bark dependency audit: required tracked file is missing: $tracked" >&2
      return 2
    }
    [[ -f "$VOKRA_ROOT/$tracked" && ! -L "$VOKRA_ROOT/$tracked" ]] || {
      echo "bark dependency audit: required file is missing or symlinked: $tracked" >&2
      return 2
    }
  done
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || {
    echo "bark dependency audit: pyproject.toml is missing or symlinked" >&2
    return 2
  }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || {
    echo "bark dependency audit: uv.lock is missing or symlinked" >&2
    return 2
  }
  [[ -f "$AUDIT" && ! -L "$AUDIT" ]] || {
    echo "bark dependency audit: audit script is missing or symlinked" >&2
    return 2
  }
  [[ "$output" == /* ]] || {
    echo "bark dependency audit: --output must be absolute" >&2
    return 2
  }
  [[ "$output" != *"/../"* && "$output" != */.. && "$output" != ../* && "$output" != ".."/* ]] || {
    echo "bark dependency audit: --output must not contain '..'" >&2
    return 2
  }
  output_real="$(canonical_absent_path "$output")" || return 2
  project_real="$(cd -P "$PARITY_PROJECT" 2>/dev/null && pwd)" || {
    echo "bark dependency audit: parity project is inaccessible" >&2
    return 2
  }
  case "$output_real/" in
    "$root_real/"*|"$project_real/"*)
      echo "bark dependency audit: output overlaps checkout or parity project" >&2
      return 2
      ;;
  esac
  mkdir -p "$(dirname "$output")"
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "bark dependency audit: --output already exists; refusing overwrite" >&2
    return 2
  }
  # --no-sync is intentional: a separately authorized named VAST job must
  # perform the frozen sync before this wrapper is called.
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" --fetch-model-licenses
}

canonical_absent_path() {
  local target="$1" lexical current component suffix real
  [[ "$target" = /* ]] || { echo "bark dependency audit: path must be absolute" >&2; return 2; }
  lexical="${target#/}"
  current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ -n "$component" && "$component" != "." ]] || continue
    [[ "$component" != ".." ]] || { echo "bark dependency audit: path contains '..'" >&2; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || {
      echo "bark dependency audit: path contains a symlinked ancestor" >&2
      return 2
    }
  done
  [[ ! -e "$target" && ! -L "$target" ]] || {
    echo "bark dependency audit: output already exists" >&2
    return 2
  }
  current="$target"
  suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"
    suffix="/$component$suffix"
    current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || {
    echo "bark dependency audit: output parent is inaccessible or symlinked" >&2
    return 2
  }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || return 2
  printf '%s%s\n' "$real" "$suffix"
}

self_test() {
  local failed=0 bad_args
  grep -Fq -- 'VOKRA_PUBLISH_ON_VAST=1' "$0" || failed=1
  grep -Fq -- 'MIN_VAST_MEM_KIB' "$0" || failed=1
  grep -Fq -- 'git status --porcelain --untracked-files=all' "$0" || failed=1
  grep -Fq -- 'canonical_absent_path' "$0" || failed=1
  grep -Fq -- 'already exists; refusing overwrite' "$0" || failed=1
  grep -Fq -- '--no-sync' "$0" || failed=1
  grep -Fq -- 'LICENSE URLs' "$0" || failed=1
  grep -Fq -- 'locked PyPI sdist archives' "$0" || failed=1
  grep -Fq -- 'never downloads model weights' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(uv[[:space:]]+sync|snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  # The only Python token is the interpreter argument of the uv run command;
  # a bare shell-level Python/pip command is intentionally absent.
  grep -Fq -- 'uv run --no-cache --no-project --offline --python 3.12' "$0" || failed=1
  for bad_args in \
    "--self-test --output /tmp/audit.json" \
    "--output" \
    "--output relative.json" \
    "--output /tmp/../audit.json" \
    "--unknown"; do
    # shellcheck disable=SC2086
    if "$0" $bad_args >/dev/null 2>&1; then
      echo "audit-bark-dependencies.sh self-test: invalid argument accepted: $bad_args" >&2
      failed=1
    fi
  done
  (
    cd "$PARITY_PROJECT"
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
      python "$AUDIT" --self-test
  ) || failed=1
  if (( failed != 0 )); then
    echo "audit-bark-dependencies.sh self-test FAIL" >&2
    return 1
  fi
  echo "audit-bark-dependencies.sh self-test PASS"
}

while (( $# > 0 )); do
  case "$1" in
    --output)
      [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$OUTPUT" ]] || { usage; exit 2; }
      OUTPUT="$2"
      shift 2
      ;;
    --self-test)
      (( SELF_TEST == 0 )) || { usage; exit 2; }
      SELF_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "audit-bark-dependencies.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if (( SELF_TEST != 0 )); then
  [[ -z "$OUTPUT" ]] || { usage; exit 2; }
  self_test
else
  [[ -n "$OUTPUT" ]] || { usage; exit 2; }
  run_audit "$OUTPUT"
fi
