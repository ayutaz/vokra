#!/usr/bin/env bash
# VAST/Linux-only MMS backbone plus one-language-adapter inspection.
# This worker never uploads or claims native CPU/Metal parity: the Rust MMS
# binder remains fail-closed until the complete manifest is independently
# reviewed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/mms_1b_all"
PREPARER="tools/parity/mms_1b_all_prepare_checkpoint.py"
REFERENCE_DUMPER="tools/parity/mms_1b_all_dump_reference.py"
PREFLIGHT_GATE="$PARITY_PROJECT/license_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
UPSTREAM_REPO="facebook/mms-1b-all"
UPSTREAM_REVISION="3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
MMS_UV_CACHE_DIR="${MMS_UV_CACHE_DIR:-/tmp/vokra-mms-uv-cache}"

log() { printf '[mms-1b-all-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-mms-1b-all-validation.sh --language <official-code> --approval-evidence <file> [--work-dir <absent-dir>]
       run-mms-1b-all-validation.sh --self-test

The normal path is Linux/VAST-only. It resolves only the full upstream
backbone and one explicitly selected official language adapter at the pinned
revision, records hashes and complete manifests, and remains INSPECTION_ONLY.
No upload, publication, native runtime parity, or tolerance is produced.
EOF
}

license_preflight() {
  local language="$1" approval="$2"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
    --manifest "$PREFLIGHT_MANIFEST" --approval-evidence "$approval" --language "$language" \
    || die 'dedicated MMS closure/license/approval gate is unresolved'
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
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
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" candidate protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die 'work-dir must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die 'work-dir has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die 'protected input is symlinked'; return 2; }
    other="$(canonical_absent_path "$protected")" || { die 'protected path cannot be canonicalized'; return 2; }
    paths_overlap "$candidate" "$other" && { die 'work-dir overlaps protected input'; return 2; }
  done
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  # shellcheck disable=SC2016 # literal source contract tokens
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'uname -s' 'uname -m' 'MIN_VAST_MEM_KIB' \
    '/proc/meminfo' 'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'snapshot_download' 'HfApi' \
    'model_info' 'local_dir' 'materialized snapshot path' 'snapshot file-set drift' \
    'member.is_symlink' 'not member.is_file() and not member.is_dir()' '.cache/' 'extra non-cache directory' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$PREPARER" "$REFERENCE_DUMPER" \
    'prepared_manifest.json' 'reference_manifest.json' 'tensor_manifest' \
    'INSPECTION_ONLY' 'no upload' 'MMS_LANGUAGE' 'azj-script_cyrillic' \
    'cac-dialect_sanmateoixtatan' 'vocabs/' 'git status --porcelain' \
    'tools/parity/mms_1b_all/license_gate.py' '--prepared-manifest' '--reference-manifest' \
    '--language "$language"' 'work_disk_root' 'nearest existing canonical ancestor' \
    'mms_1b_all_prepare_checkpoint.py --self-test' \
    'mms_1b_all_dump_reference.py --self-test'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  # shellcheck disable=SC2016 # literal source contract token
  if ! grep -Fq -- 'license_preflight "$language" "$approval_evidence"' "$path"; then
    log 'self-test FAIL: approval scope is not bound to selected language'
    fail=1
  fi
  if grep -Fq 'ignore_mismatched_sizes=True' "$VOKRA_ROOT/$REFERENCE_DUMPER"; then
    log 'self-test FAIL: reference dumper permits silent composition mismatch'
    fail=1
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if "$path" --self-test --language eng >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  for bad in '--language' '--language -bad' '--language eng --language spa' '--approval-evidence' '--approval-evidence -bad' '--approval-evidence a --approval-evidence b'; do
    if eval "\"$path\" $bad" >/dev/null 2>&1; then
      log "self-test FAIL: malformed or duplicate option accepted: $bad"
      fail=1
    fi
  done
  local gate_line host_line path_line
  # shellcheck disable=SC2016 # match literal source token
  gate_line="$(grep -n 'license_preflight "\$language" "\$approval_evidence"' "$path" | tail -n 1 | cut -d: -f1)"
  host_line="$(grep -n 'uname -s' "$path" | tail -n 1 | cut -d: -f1)"
  # shellcheck disable=SC2016 # match literal source token
  path_line="$(grep -n 'require_absent_work_dir "\$work_dir"' "$path" | tail -n 1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$path_line" =~ ^[0-9]+$ && "$host_line" =~ ^[0-9]+$ && "$gate_line" -lt "$path_line" && "$path_line" -lt "$host_line" ]] || {
    log 'self-test FAIL: closure/path gate is not before host probe'
    fail=1
  }
  local temporary nested canonical disk_root
  temporary="$(cd -P "$(mktemp -d)" && pwd)"
  nested="$temporary/nested/deeper/work"
  canonical="$(canonical_absent_path "$nested")" || fail=1
  disk_root="$canonical"
  while [[ ! -d "$disk_root" ]]; do disk_root="$(dirname "$disk_root")"; done
  [[ "$disk_root" == "$temporary" ]] || { log 'self-test FAIL: disk probe did not use nearest existing ancestor'; fail=1; }
  rm -rf "$temporary"
  if ! UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --no-project --offline --python 3.12 \
    python "$VOKRA_ROOT/$PREPARER" --self-test >/dev/null; then
    log 'self-test FAIL: preparer self-test failed'
    fail=1
  fi
  if ! UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --no-project --offline --python 3.12 \
    python "$VOKRA_ROOT/$REFERENCE_DUMPER" --self-test >/dev/null; then
    log 'self-test FAIL: independent dumper self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-mms-1b-all-validation"
language=""
approval_evidence=""
self=0
seen_self=0; seen_language=0; seen_work=0; seen_approval=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --language) (( seen_language == 0 )) || die 'duplicate --language'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--language requires a nonempty official adapter code'; seen_language=1; language="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--work-dir requires a nonempty path'; seen_work=1; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_self" == 1 && -z "$language$approval_evidence" && "$work_dir" == "/workspace/vokra-mms-1b-all-validation" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi
[[ "$seen_approval" == 1 ]] || die '--approval-evidence is required'
[[ -n "$language" ]] || die '--language is required; refusing to assume English'
[[ "$language" =~ ^[a-z0-9]+([_-][a-z0-9]+)*$ ]] || die '--language contains unsafe filename characters'
license_preflight "$language" "$approval_evidence"
require_absent_work_dir "$work_dir" "$approval_evidence"
[[ "$(uname -s)" == Linux ]] || die 'inspection is Linux/VAST-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PREFLIGHT_GATE" ]] || die 'dedicated MMS license gate is missing'
[[ -f "$VOKRA_ROOT/$PREPARER" && -f "$VOKRA_ROOT/$REFERENCE_DUMPER" ]] || die 'MMS parity tools are missing'

mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work-dir must be empty'
canonical_work_dir="$(canonical_absent_path "$work_dir")" || die 'work-dir cannot be canonicalized'
# Disk probe uses the nearest existing canonical ancestor of nested absent work-dir.
work_disk_root="$canonical_work_dir"
while [[ ! -d "$work_disk_root" ]]; do work_disk_root="$(dirname "$work_disk_root")"; done
free_kib="$(df -Pk "$work_disk_root" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
{
  printf 'repository=%s\nrevision=%s\nlanguage=%s\n' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$language"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
} > "$work_dir/validation.log"
cargo fmt --all -- --check >> "$work_dir/validation.log" 2>&1
cargo metadata --no-deps --format-version 1 >> "$work_dir/validation.log" 2>&1

snapshot_path_file="$work_dir/snapshot-path.txt"
snapshot_dir="$work_dir/$UPSTREAM_REVISION"
UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$language" "$snapshot_dir" "$snapshot_path_file" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo, revision, language, materialized, output = sys.argv[1:]
materialized = Path(materialized)
if materialized.exists() or materialized.is_symlink():
    raise SystemExit(f"materialized snapshot path must be absent: {materialized}")
info = HfApi().model_info(repo_id=repo, revision=revision, files_metadata=False)
if info.sha != revision:
    raise SystemExit(f"server revision drift: {info.sha!r} != {revision!r}")
path = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=str(materialized),
    allow_patterns=[
        "config.json", "preprocessor_config.json", "tokenizer_config.json",
        "vocab.json", "special_tokens_map.json", "model.safetensors",
        f"adapter.{language}.safetensors",
        f"vocabs/{language}.txt",
    ],
))
expected = {
    "config.json", "preprocessor_config.json", "tokenizer_config.json",
    "vocab.json", "special_tokens_map.json", "model.safetensors",
    f"adapter.{language}.safetensors", f"vocabs/{language}.txt",
}
if path.resolve() != materialized.resolve() or path.name != revision:
    raise SystemExit(f"materialized snapshot path drift: {path!r}")
actual = set()
for member in path.rglob("*"):
    relative = member.relative_to(path).as_posix()
    if member.is_symlink():
        raise SystemExit(f"snapshot member is symlinked: {relative}")
    if not member.is_file() and not member.is_dir():
        raise SystemExit(f"snapshot member is not regular: {relative}")
    if relative == ".cache" or relative.startswith(".cache/"):
        # huggingface_hub may leave transport metadata below this one known
        # cache directory. It is never part of the authenticated model set.
        continue
    if member.is_file():
        actual.add(relative)
    elif member.is_dir() and relative != "vocabs":
        raise SystemExit(f"extra non-cache directory: {relative}")
if actual != expected:
    raise SystemExit(f"snapshot file-set drift: got {sorted(actual)!r}, expected {sorted(expected)!r}")
Path(output).write_text(str(path) + "\n", encoding="utf-8")
PY

snapshot_dir="$(< "$snapshot_path_file")"
UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$VOKRA_ROOT/$PREPARER" --snapshot-dir "$snapshot_dir" --language "$language" \
  --evidence-dir "$work_dir/prepared" \
  >> "$work_dir/validation.log"
UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$VOKRA_ROOT/$REFERENCE_DUMPER" --snapshot-dir "$snapshot_dir" --language "$language" \
  --output-dir "$work_dir/reference" >> "$work_dir/validation.log"
UV_NO_CACHE=1 UV_CACHE_DIR="$MMS_UV_CACHE_DIR" uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
  --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
  --manifest "$PREFLIGHT_MANIFEST" --approval-evidence "$approval_evidence" --language "$language" \
  --prepared-manifest "$work_dir/prepared/prepared_manifest.json" \
  --reference-manifest "$work_dir/reference/reference_manifest.json" \
  >> "$work_dir/validation.log" || die 'generated MMS manifests failed strict validation'
for required in config.json preprocessor_config.json tokenizer_config.json vocab.json \
  special_tokens_map.json model.safetensors "adapter.$language.safetensors" \
  "vocabs/$language.txt"; do
  sha256sum "$snapshot_dir/$required"
done > "$work_dir/upstream-file-sha256.txt"
sha256sum "$work_dir/prepared/prepared_manifest.json" "$work_dir/reference/reference_manifest.json" \
  > "$work_dir/evidence-sha256.txt"
{
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_UPLOAD'
  echo 'reason=complete manifest and native CPU/Metal route are not yet reviewed'
} | tee -a "$work_dir/validation.log"
log "inspection complete: evidence remains at $work_dir; no upload or publication was performed"
