#!/usr/bin/env bash
# VAST-only, model-free Zonos/Transformers compatibility smoke.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/zonos_v0_1_reference"
PROBE="$PROJECT/transformers_compatibility.py"
SOURCE_REPOSITORY="https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION="bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
DEFAULT_WORK="/dev/shm/vokra-zonos-transformers-compatibility"

die() { echo "zonos-transformers-compatibility: ERROR: $*" >&2; exit 2; }
usage() { echo "usage: $0 --expected-head HEX40 --output ABSENT_JSON [--work-dir ABSENT_DIR] | --self-test" >&2; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/${name}${suffix}"
    parent="${path%/*}"; [[ "${parent}" == "$path" ]] && parent=/; path="${parent}"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_path() {
  local path="$1" canonical protected protected_real
  [[ "$path" == /* && "$path" != */ && ! -e "$path" && ! -L "$path" ]] || die "path must be an absent absolute path: $path"
  canonical="$(canonicalize_uncreated "$path")" || die "path cannot be canonicalized: $path"
  for protected in "$ROOT" "$PROJECT" "$PROBE" "$SCRIPT_DIR"; do
    protected_real="$(canonicalize_uncreated "$protected")" || die "protected path cannot be canonicalized: $protected"
    paths_overlap "$canonical" "$protected_real" && die "path overlaps protected checkout path: $protected"
  done
}

self_test() {
  local self="${BASH_SOURCE[0]}" status=0
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" && -f "$PROBE" ]] || die 'Zonos project/probe is missing'
  for needle in "$SOURCE_REPOSITORY" "$SOURCE_REVISION" 'transformers_compatibility.py' 'PASS_COMPATIBLE' 'NO_UPLOAD' '--validate-evidence' 'HF_TOKEN' 'checkpoint_access' 'constructor_calls'; do
    grep -Fq -- "$needle" "$self" "$PROBE" || die "self-test contract missing: $needle"
  done
  if grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then die 'raw Python/pip invocation found'; fi
  if bash "$self" --self-test --expected-head 0000000000000000000000000000000000000000 >/dev/null 2>&1; then die 'extra self-test argument accepted'; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PROBE" --self-test || status=$?
  [[ "$status" == 0 ]] || die "probe self-test failed: $status"
  echo 'run-zonos-transformers-compatibility.sh self-test: OK'
}

expected_head=''; output=''; work_dir="$DEFAULT_WORK"; self_flag=0; seen_head=0; seen_output=0; seen_work=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( self_flag == 0 )) || die 'duplicate --self-test'; self_flag=1; shift ;;
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --output) (( seen_output == 0 )) || die 'duplicate --output'; [[ $# -ge 2 && "$2" != -* ]] || die '--output requires a path'; output="$2"; seen_output=1; shift 2 ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && "$2" != -* ]] || die '--work-dir requires a path'; work_dir="$2"; seen_work=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
  esac
done
if (( self_flag == 1 )); then
  [[ $seen_head == 0 && $seen_output == 0 && $seen_work == 0 ]] || die '--self-test accepts no other options'
  self_test
  exit 0
fi
(( seen_head == 1 && seen_output == 1 )) || { usage; die '--expected-head and --output are required'; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_ZONOS_VAST_VALIDATION:-0}" == 1 ]] || die 'VOKRA_ZONOS_VAST_VALIDATION=1 is absent'
[[ -z "${HF_TOKEN:-}" && -z "${HF:-}" ]] || die 'HF token environment variables must be absent'
for command in git uv sha256sum awk findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
[[ -f "$ROOT/.git" || -d "$ROOT/.git" ]] || die 'Vokra checkout is missing .git'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'Vokra checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'Vokra HEAD differs from --expected-head'
[[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || die 'dedicated Zonos project is missing or symlinked'
require_absent_path "$output"
require_absent_path "$work_dir"
work_parent="$(dirname "$work_dir")"
[[ -d "$work_parent" && "$(findmnt -T "$work_parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
mkdir "$work_dir"
trap 'rm -rf -- "$work_dir"' EXIT
source_dir="$work_dir/official-source"
step() { echo "zonos-transformers-compatibility: $*" >&2; }
step 'install frozen dedicated environment'
uv sync --project "$PROJECT" --frozen --python 3.12
step 'checkout exact official source'
git clone --filter=blob:none --no-checkout "$SOURCE_REPOSITORY" "$source_dir" >/dev/null 2>&1
git -C "$source_dir" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1
step 'run model-free official import/API contract probe'
UV_NO_CACHE=1 uv run --no-cache --project "$PROJECT" --frozen --python 3.12 python "$PROBE" \
  --run --vokra-root "$ROOT" --source-dir "$source_dir" --caller-script "$SCRIPT_DIR/run-zonos-transformers-compatibility.sh" \
  --expected-head "$expected_head" --output "$output"
echo "zonos-transformers-compatibility: PASS_COMPATIBLE; no model/checkpoint/token; evidence_sha256=$(sha256sum "$output" | awk '{print $1}')" >&2
