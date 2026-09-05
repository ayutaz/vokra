#!/usr/bin/env bash
# Audit the exact Qwen3-ASR Python closure on VAST without acquiring weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_asr"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=60000000

usage() {
  cat <<'EOF' >&2
usage: audit-qwen3-asr-dependencies.sh --output <audit.json>
       audit-qwen3-asr-dependencies.sh --self-test

The target environment must already have been synchronized by the named,
authorized VAST Qwen3-ASR worker after its approval gate. This command is
VAST/Linux-only, never runs uv sync, and never downloads model weights. The
audit itself fetches only exact locked PyPI sdist URLs when present for a
missing wheel license file, plus the two fixed upstream LICENSE URLs. It
never fetches weights, imports model packages, or runs Cargo, and rejects
unsafe or redirected non-license paths.
EOF
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ "$component" != .. ]] || return 1
    [[ -n "$component" && "$component" != . ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"
    [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_vast_linux() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || {
    echo "qwen3-asr dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required" >&2
    return 2
  }
  [[ "$(uname -s)" == "Linux" ]] || {
    echo "qwen3-asr dependency audit: Linux/VAST host is required" >&2
    return 2
  }
  [[ -r /proc/meminfo ]] || {
    echo "qwen3-asr dependency audit: /proc/meminfo is unavailable" >&2
    return 2
  }
  local memory
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && memory -ge MIN_VAST_MEM_KIB ]] || {
    echo "qwen3-asr dependency audit: 64-GB-class VAST memory is required" >&2
    return 2
  }
}

require_clean_checkout() {
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || {
    echo "qwen3-asr dependency audit: git checkout is missing" >&2
    return 2
  }
  local root
  root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "qwen3-asr dependency audit: git root is unavailable" >&2
    return 2
  }
  [[ "$(canonicalize_uncreated "$VOKRA_ROOT")" == "$(canonicalize_uncreated "$root")" ]] || {
    echo "qwen3-asr dependency audit: VOKRA_ROOT is not the checkout root" >&2
    return 2
  }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "qwen3-asr dependency audit: checkout must be clean" >&2
    return 2
  }
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || {
    echo "qwen3-asr dependency audit: output must be an absent absolute path" >&2
    return 2
  }
  canonical="$(canonicalize_uncreated "$output")" || return 2
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 2
  project="$(canonicalize_uncreated "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && {
    echo "qwen3-asr dependency audit: output overlaps checkout" >&2
    return 2
  }
  paths_overlap "$canonical" "$project" && {
    echo "qwen3-asr dependency audit: output overlaps parity project" >&2
    return 2
  }
  return 0
}

run_audit() {
  local output="$1"
  require_vast_linux || return 2
  require_clean_checkout || return 2
  command -v uv >/dev/null 2>&1 || {
    echo "qwen3-asr dependency audit: uv is unavailable" >&2
    return 2
  }
  command -v readelf >/dev/null 2>&1 || {
    echo "qwen3-asr dependency audit: readelf is unavailable" >&2
    return 2
  }
  [[ -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" ]] || {
    echo "qwen3-asr dependency audit: project pyproject.toml is missing or symlinked" >&2
    return 2
  }
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" ]] || {
    echo "qwen3-asr dependency audit: project uv.lock is missing or symlinked" >&2
    return 2
  }
  [[ -f "$AUDIT" && ! -L "$AUDIT" ]] || {
    echo "qwen3-asr dependency audit: audit script is missing or symlinked" >&2
    return 2
  }
  [[ "$output" == /* ]] || {
    echo "qwen3-asr dependency audit: --output must be absolute" >&2
    return 2
  }
  require_absent_output "$output" || return 2
  command -v git >/dev/null 2>&1 || {
    echo "qwen3-asr dependency audit: git is unavailable" >&2
    return 2
  }
  mkdir -p "$(dirname "$output")" || return 2
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" --fetch-model-licenses
}

self_test() {
  local failed=0 probe_root blocked_parent blocked_output probe_parent=/tmp
  [[ -d /private/tmp && ! -L /private/tmp ]] && probe_parent=/private/tmp
  grep -Fq -- '--no-sync' "$0" || failed=1
  grep -Fq -- '--fetch-model-licenses' "$0" || failed=1
  grep -Fq -- 'exact locked PyPI sdist' "$0" || failed=1
  grep -Fq -- 'never downloads model' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(uv sync|snapshot_download|cargo (build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  probe_root="$(mktemp -d "$probe_parent/qwen3-asr-audit-wrapper.XXXXXX")"
  blocked_parent="$probe_root/blocked-parent"
  blocked_output="$blocked_parent/audit.json"
  mkdir "$probe_root/bin"
  printf '%s\n' '#!/usr/bin/env bash' "touch '$probe_root/auditor-invoked'" > "$probe_root/bin/uv"
  chmod +x "$probe_root/bin/uv"
  if PATH="$probe_root/bin:$PATH" VOKRA_PUBLISH_ON_VAST=0 run_audit "$blocked_output" >/dev/null 2>&1; then failed=1; fi
  [[ ! -e "$probe_root/auditor-invoked" ]] || failed=1
  [[ ! -e "$blocked_parent" && ! -e "$blocked_output" ]] || failed=1
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  if ! require_absent_output "$probe_root/nested/audit.json" >/dev/null 2>&1; then failed=1; fi
  [[ ! -e "$probe_root/nested" ]] || failed=1
  if require_absent_output "$probe_root/../escaped.json" >/dev/null 2>&1; then failed=1; fi
  touch "$probe_root/existing.json"
  if require_absent_output "$probe_root/existing.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$probe_root" "$probe_root/link-parent"
  if require_absent_output "$probe_root/link-parent/new.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$probe_root"
  (
    cd "$PARITY_PROJECT"
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
      python "$AUDIT" --self-test
  ) || failed=1
  if (( failed != 0 )); then
    echo "audit-qwen3-asr-dependencies.sh self-test FAIL" >&2
    return 1
  fi
  echo "audit-qwen3-asr-dependencies.sh self-test PASS"
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
      echo "audit-qwen3-asr-dependencies.sh: unknown argument: $1" >&2
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
