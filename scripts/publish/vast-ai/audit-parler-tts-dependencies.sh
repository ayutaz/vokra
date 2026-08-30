#!/usr/bin/env bash
# Audit the exact Parler-TTS Python closure on VAST without acquiring weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/parler_tts"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=60000000

usage() {
  cat <<'EOF' >&2
usage: audit-parler-tts-dependencies.sh --output <audit.json>
       audit-parler-tts-dependencies.sh --self-test

The target environment must already have been synchronized by a separately
authorized, named VAST Parler-TTS job. This wrapper only inspects that frozen
environment: it never runs uv sync, imports model/Torch code, invokes Cargo, or
downloads model weights. The audit fetches only the four exact primary-source
LICENSE files already named by license_gate_manifest.json.
EOF
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . ]] || continue
    [[ "$component" != .. ]] || return 1
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
    echo "parler dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required" >&2
    return 2
  }
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
    echo "parler dependency audit: exact Linux x86_64 VAST host required" >&2
    return 2
  }
  [[ -r /proc/meminfo ]] || {
    echo "parler dependency audit: /proc/meminfo is unavailable" >&2
    return 2
  }
  local memory
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && memory -ge MIN_VAST_MEM_KIB ]] || {
    echo "parler dependency audit: 64-GB-class VAST memory is required" >&2
    return 2
  }
}

require_clean_checkout() {
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || {
    echo "parler dependency audit: git checkout is missing" >&2
    return 2
  }
  local root
  root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "parler dependency audit: git root is unavailable" >&2
    return 2
  }
  [[ "$(canonicalize_uncreated "$VOKRA_ROOT")" == "$(canonicalize_uncreated "$root")" ]] || {
    echo "parler dependency audit: VOKRA_ROOT is not the checkout root" >&2
    return 2
  }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "parler dependency audit: checkout must be clean" >&2
    return 2
  }
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || {
    echo "parler dependency audit: output must be an absent absolute path" >&2
    return 2
  }
  canonical="$(canonicalize_uncreated "$output")" || return 2
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 2
  project="$(canonicalize_uncreated "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && {
    echo "parler dependency audit: output overlaps checkout" >&2
    return 2
  }
  paths_overlap "$canonical" "$project" && {
    echo "parler dependency audit: output overlaps parity project" >&2
    return 2
  }
}

run_audit() {
  local output="$1" input
  require_vast_linux
  require_clean_checkout
  command -v uv >/dev/null 2>&1 || {
    echo "parler dependency audit: uv is unavailable" >&2
    return 2
  }
  command -v readelf >/dev/null 2>&1 || {
    echo "parler dependency audit: readelf is unavailable" >&2
    return 2
  }
  command -v git >/dev/null 2>&1 || {
    echo "parler dependency audit: git is unavailable" >&2
    return 2
  }
  for input in pyproject.toml uv.lock license_gate_manifest.json dependency_audit.py; do
    [[ -f "$PARITY_PROJECT/$input" && ! -L "$PARITY_PROJECT/$input" ]] || {
      echo "parler dependency audit: missing or symlinked input: $input" >&2
      return 2
    }
  done
  [[ "$output" == /* ]] || {
    echo "parler dependency audit: --output must be absolute" >&2
    return 2
  }
  require_absent_output "$output"
  mkdir -p "$(dirname "$output")"
  # --no-sync is intentional: a separately authorized named VAST sync must
  # happen before this wrapper is invoked.
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" --fetch-model-licenses
}

self_test() {
  local failed=0 probe_root
  grep -Fq -- '--no-sync' "$0" || failed=1
  grep -Fq -- 'four exact primary-source' "$0" || failed=1
  grep -Fq -- 'never downloads model weights' "$0" || failed=1
  ! grep -Eq '^[[:space:]]*(uv[[:space:]]+sync|snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  grep -Fq -- 'uv run --no-cache --no-project --offline --python 3.12' "$0" || failed=1
  if VOKRA_PUBLISH_ON_VAST=0 run_audit /private/tmp/parler-dependency-audit-self-test.json >/dev/null 2>&1; then
    failed=1
  fi
  probe_root="$(mktemp -d "${TMPDIR:-/tmp}/parler-dependency-audit-wrapper.XXXXXX")"
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  touch "$probe_root/existing.json"
  if require_absent_output "$probe_root/existing.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$probe_root/../escaped.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$probe_root" "$probe_root/link-parent"
  if require_absent_output "$probe_root/link-parent/new.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$probe_root"
  (
    cd "$PARITY_PROJECT"
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
      python "$AUDIT" --self-test
  ) || failed=1
  if (( failed != 0 )); then
    echo "audit-parler-tts-dependencies.sh self-test FAIL" >&2
    return 1
  fi
  echo "audit-parler-tts-dependencies.sh self-test PASS"
}

while (( $# > 0 )); do
  case "$1" in
    --output)
      [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$OUTPUT" ]] || { usage; exit 2; }
      OUTPUT="$2"; shift 2
      ;;
    --self-test)
      (( SELF_TEST == 0 )) || { usage; exit 2; }
      SELF_TEST=1; shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "audit-parler-tts-dependencies.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if (( SELF_TEST != 0 )); then
  [[ -z "$OUTPUT" ]] || { usage; exit 2; }
  self_test
else
  [[ -n "$OUTPUT" ]] || { usage; exit 2; }
  run_audit "$OUTPUT"
fi
