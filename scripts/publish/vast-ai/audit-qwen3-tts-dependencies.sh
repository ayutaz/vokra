#!/usr/bin/env bash
# Audit the exact Qwen3-TTS Python closure on VAST without acquiring weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/qwen3_tts"
AUDIT="$PARITY_PROJECT/dependency_audit.py"
OUTPUT=""
COMPACT_OUTPUT=""
SELF_TEST=0
MIN_VAST_MEM_KIB=60000000

usage() {
  cat <<'EOF' >&2
usage: audit-qwen3-tts-dependencies.sh --output <audit.json>
       audit-qwen3-tts-dependencies.sh --output <audit.json> --compact-output <compact.json>
       audit-qwen3-tts-dependencies.sh --self-test

The synchronized environment must have been prepared by a separately
authorized VAST job. This worker performs no uv sync, model download, model
import, Cargo operation, upload, or publication. It inspects exact installed
rows and may fetch only locked PyPI sdists, fixed source LICENSE paths, and
the exact HF model-info API cardData.license projection and sibling tree only
as a fallback for a pinned model revision whose exact LICENSE path returns
HTTP 404. Other LICENSE errors remain blocking.
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
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'
    path="$parent"; [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_vast_linux() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || {
    echo "qwen3-tts dependency audit: VOKRA_PUBLISH_ON_VAST=1 is required" >&2; return 2;
  }
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
    echo "qwen3-tts dependency audit: exact Linux x86_64 VAST host required" >&2; return 2;
  }
  [[ -r /proc/meminfo ]] || { echo "qwen3-tts dependency audit: /proc/meminfo unavailable" >&2; return 2; }
  local memory; memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && memory -ge MIN_VAST_MEM_KIB ]] || {
    echo "qwen3-tts dependency audit: 64-GB-class VAST memory is required" >&2; return 2;
  }
}

require_clean_checkout() {
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || return 2
  local root; root="$(git -C "$VOKRA_ROOT" rev-parse --show-toplevel 2>/dev/null)" || return 2
  [[ "$(canonicalize_uncreated "$VOKRA_ROOT")" == "$(canonicalize_uncreated "$root")" ]] || return 2
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "qwen3-tts dependency audit: checkout must be clean" >&2; return 2;
  }
}

require_absent_output() {
  local output="$1" canonical root project
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || return 2
  canonical="$(canonicalize_uncreated "$output")" || return 2
  root="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 2
  project="$(canonicalize_uncreated "$PARITY_PROJECT")" || return 2
  paths_overlap "$canonical" "$root" && return 2
  paths_overlap "$canonical" "$project" && return 2
  return 0
}

run_audit() {
  local output="$1" compact_output="$2" input
  require_vast_linux || return 2
  require_clean_checkout || return 2
  command -v uv >/dev/null 2>&1 || { echo "qwen3-tts dependency audit: uv unavailable" >&2; return 2; }
  command -v readelf >/dev/null 2>&1 || { echo "qwen3-tts dependency audit: readelf unavailable" >&2; return 2; }
  command -v git >/dev/null 2>&1 || { echo "qwen3-tts dependency audit: git unavailable" >&2; return 2; }
  for input in pyproject.toml uv.lock license_gate_manifest.json license_gate.py dependency_audit.py; do
    [[ -f "$PARITY_PROJECT/$input" && ! -L "$PARITY_PROJECT/$input" ]] || return 2
  done
  [[ "$output" == /* ]] || return 2
  require_absent_output "$output" || return 2
  if [[ -n "$compact_output" ]]; then
    [[ "$compact_output" == /* ]] || return 2
    require_absent_output "$compact_output" || return 2
    paths_overlap "$output" "$compact_output" && return 2
  fi
  mkdir -p "$(dirname "$output")" || return 2
  local -a compact_args=()
  if [[ -n "$compact_output" ]]; then
    mkdir -p "$(dirname "$compact_output")" || return 2
    compact_args=(--compact-output "$compact_output")
  fi
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 \
    python "$AUDIT" --project "$PARITY_PROJECT" --output "$output" "${compact_args[@]}" --fetch-model-licenses
}

self_test() {
  local failed=0 probe_root blocked_output probe_parent=/tmp
  [[ -d /private/tmp && ! -L /private/tmp ]] && probe_parent=/private/tmp
  grep -Fq -- '--no-sync' "$0" || failed=1
  grep -Fq -- 'fixed source LICENSE paths' "$0" || failed=1
  grep -Fq -- 'no model download' "$0" || failed=1
  grep -Fq -- "setuptools ; python_version < '0'" "$PARITY_PROJECT/pyproject.toml" || failed=1
  ! grep -Eq '^[[:space:]]*name[[:space:]]*=[[:space:]]*"setuptools"[[:space:]]*$' "$PARITY_PROJECT/uv.lock" || failed=1
  ! grep -Eq '^[[:space:]]*\{[[:space:]]*name[[:space:]]*=[[:space:]]*"setuptools"([,[:space:]])' "$PARITY_PROJECT/uv.lock" || failed=1
  ! grep -Eq '^[[:space:]]*(uv[[:space:]]+sync|snapshot_download|huggingface-cli|cargo[[:space:]]+(build|test|check|clippy))([[:space:]]|$)' "$0" || failed=1
  grep -Fq -- 'uv run --no-cache --no-project --offline --python 3.12' "$0" || failed=1
  probe_root="$(mktemp -d "$probe_parent/qwen3-tts-audit-wrapper.XXXXXX")"
  blocked_output="$probe_root/blocked.json"
  if VOKRA_PUBLISH_ON_VAST=0 run_audit "$blocked_output" "" >/dev/null 2>&1; then failed=1; fi
  [[ ! -e "$blocked_output" ]] || failed=1
  if require_absent_output "$VOKRA_ROOT" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$PARITY_PROJECT" >/dev/null 2>&1; then failed=1; fi
  require_absent_output "$probe_root/nested/audit.json" || failed=1
  [[ ! -e "$probe_root/nested" ]] || failed=1
  touch "$probe_root/existing.json"
  if require_absent_output "$probe_root/existing.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_output "$probe_root/../escaped.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$probe_root" "$probe_root/link-parent"
  if require_absent_output "$probe_root/link-parent/new.json" >/dev/null 2>&1; then failed=1; fi
  rm -rf "$probe_root"
  (cd "$PARITY_PROJECT" && UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
    python "$AUDIT" --self-test) || failed=1
  if (( failed != 0 )); then echo "audit-qwen3-tts-dependencies.sh self-test FAIL" >&2; return 1; fi
  echo "audit-qwen3-tts-dependencies.sh self-test PASS"
}

while (( $# > 0 )); do
  case "$1" in
    --output) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$OUTPUT" ]] || { usage; exit 2; }; OUTPUT="$2"; shift 2 ;;
    --compact-output) [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$COMPACT_OUTPUT" ]] || { usage; exit 2; }; COMPACT_OUTPUT="$2"; shift 2 ;;
    --self-test) (( SELF_TEST == 0 )) || { usage; exit 2; }; SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if (( SELF_TEST != 0 )); then
  [[ -z "$OUTPUT" && -z "$COMPACT_OUTPUT" ]] || { usage; exit 2; }
  self_test
else
  [[ -n "$OUTPUT" ]] || { usage; exit 2; }
  run_audit "$OUTPUT" "$COMPACT_OUTPUT"
fi
