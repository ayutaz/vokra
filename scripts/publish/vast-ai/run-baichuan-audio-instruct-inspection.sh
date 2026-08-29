#!/usr/bin/env bash
# VAST/Linux-only Baichuan-Audio-Instruct inspection. No conversion, runtime,
# execution, upload, or publication is performed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/baichuan_audio_instruct"
HF_REPOSITORY="baichuan-inc/Baichuan-Audio-Instruct"
HF_REVISION="1c86512d863376f9ea0c32bb77451b9f428283c8"
SOURCE_URL="https://github.com/baichuan-inc/Baichuan-Audio.git"
SOURCE_REVISION="805d456433dbf3e0edb2bdd302f733a4bd38ea84"
INSPECTOR="$ROOT/tools/parity/baichuan_audio_instruct_inspect.py"
REFERENCE_LOCK_SHA256="0e8ca64e2f81060732c317fd6d10e01df7c3a5eb122426ef5d695e9813df7625"
REFERENCE_PACKAGE_ROWS_SHA256="a276c50b73fcbc7f0ac22667d6d56516bf7dea7e9e420456562f128b4fa36b2b"
REFERENCE_RESOLUTION_MARKERS_SHA256="4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
UV_CACHE_DIR_VALUE="${BAICHUAN_UV_CACHE_DIR:-/tmp/vokra-baichuan-uv-cache}"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((40 * 1024 * 1024))

log() { printf '[baichuan-audio-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

self_test() {
  local script="${BASH_SOURCE[0]}" fail=0 token
  for token in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" \
    'resolved_revision' 'git_blob_sha1' 'lfs_pointer_git_blob_sha1' 'lfs_sha256' 'payload_bytes' 'SOURCE_ROLE_PATHS' 'fixed Git blob table' \
    'snapshot_download' 'list_repo_tree' 'server-tree' 'weights_only' 'safetensors' \
    'header-only' 'overlapping tensor ranges' 'gap in tensor data region' \
    'INSPECTION_ONLY' 'materialized payload size' 'payload_sha256' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'UNSUPPORTED' 'BLOCKED_BY_CPU' 'NOT_RUN' 'NO_UPLOAD' \
    'UNAUTHENTICATED_BLOCKER' 'UNREVIEWED_BLOCKER' 'CARGO_BUILD_JOBS=1' 'cargo metadata --locked --no-deps --format-version 1' \
    'dependency-gate' "$REFERENCE_LOCK_SHA256" "$REFERENCE_PACKAGE_ROWS_SHA256" "$REFERENCE_RESOLUTION_MARKERS_SHA256" 'dependency_license_audit' \
    'BLOCKED_UNREVIEWED_TRANSITIVE' 'uv sync --project' '--no-sync' 'exit 2'; do
    if ! grep -Fq -- "$token" "$script" && ! grep -Fq -- "$token" "$INSPECTOR"; then
      log "self-test FAIL: missing contract $token"; fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$script" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  if grep -En 'torch\.load|weights_only=False|pickle\.load' "$INSPECTOR" >/dev/null; then
    log 'self-test FAIL: unsafe loader found'; fail=1
  fi
  if grep -Eq 'CARGO_BUILD_JOBS=.*\$\{' "$script"; then
    log 'self-test FAIL: Cargo jobs can be overridden'; fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-project --python 3.12 python "$INSPECTOR" --self-test >/dev/null; then
    log 'self-test FAIL: inspector self-test failed'; fail=1
  fi
  local gate_line sync_line work_line download_line
  gate_line="$(grep -n -- '--dependency-gate' "$script" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^uv sync --project' "$script" | tail -1 | cut -d: -f1)"
  work_line="$(grep -n '^mkdir -p "' "$script" | tail -1 | cut -d: -f1)"
  download_line="$(grep -n 'snapshot_download(repo_id' "$script" | tail -1 | cut -d: -f1)"
  if [[ -z "$gate_line" || -z "$sync_line" || -z "$work_line" || -z "$download_line" || "$gate_line" -ge "$sync_line" || "$sync_line" -ge "$work_line" || "$work_line" -ge "$download_line" ]]; then
    log 'self-test FAIL: affirmative gate must precede sync, work-directory creation, and acquisition'; fail=1
  fi
  if sed -n "1,$((sync_line - 1))p" "$script" | grep -Eq 'uv run[^\n]*--project'; then
    log 'self-test FAIL: dedicated project invocation appears before uv sync'; fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/dev/shm/vokra-baichuan-audio-instruct-inspection"
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no other arguments'
  self_test; exit $?
fi
[[ $# == 0 ]] || die 'arguments are not accepted; revisions are fixed in the worker'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST host must be Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || die 'dedicated Baichuan inspection uv.lock is absent; refuse before acquisition'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'
if [[ -e "$work_dir" ]]; then
  [[ -d "$work_dir" ]] || die 'work path must be a directory'
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
fi
free_kib="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die 'tmpfs disk guard failed'
for command in cargo git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$UV_CACHE_DIR_VALUE"
uv run --no-project --python 3.12 python "$INSPECTOR" --dependency-gate || die 'dependency/license gate is not affirmatively approved; refuse before sync or acquisition'
uv sync --project "$PROJECT" --frozen --python 3.12
mkdir -p "$work_dir"
[[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
mkdir -p "$work_dir/hf" "$work_dir/source" "$work_dir/evidence"
{
  echo "runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED"
  echo "parity_status=NOT_RUN"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

# shellcheck disable=SC2129 # heredoc output is one validation stream
UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python - \
  "$HF_REPOSITORY" "$HF_REVISION" "$work_dir/hf" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["*"])
PY

# The packet is written only after materialization so `size` is the observed
# payload size, not an unverified API hint.  The inspector still compares this
# packet against the complete local tree and the server tree metadata.
# shellcheck disable=SC2129 # heredoc output is one validation stream
UV_CACHE_DIR="$UV_CACHE_DIR_VALUE" uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python - \
  "$HF_REPOSITORY" "$HF_REVISION" "$work_dir/hf" "$work_dir/server-tree.json" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json, re, sys
from pathlib import Path
from huggingface_hub import HfApi
repo, rev, snapshot, output = sys.argv[1:]
api = HfApi()
info = api.model_info(repo, revision=rev)
if info.sha != rev or not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
    raise RuntimeError("HF revision identity mismatch")
rows = []
for item in api.list_repo_tree(repo, revision=rev, recursive=True):
    path, kind = getattr(item, "path", None), getattr(item, "type", None)
    if kind in {"directory", "folder", "dir"}:
        continue
    size, blob = getattr(item, "size", None), getattr(item, "oid", None) or getattr(item, "blob_id", None)
    lfs = getattr(item, "lfs", None)
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    materialized = Path(snapshot) / path if isinstance(path, str) else None
    if kind != "file" or materialized is None or not path or "\\" in path or "\x00" in path or path.startswith("/") or ".." in Path(path).parts or not isinstance(size, int) or size < 0 or not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob) or (lfs_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", lfs_sha)) or not materialized.is_file() or materialized.is_symlink():
        raise RuntimeError(f"incomplete canonical HF identity or materialization: {path}")
    payload_size = materialized.stat().st_size
    if payload_size != size:
        raise RuntimeError(f"materialized payload size differs from HF metadata: {path}")
    rows.append({"path": path, "type": "file", "size": payload_size, "git_blob_sha1": blob if lfs_sha is None else None, "lfs_pointer_git_blob_sha1": blob if lfs_sha is not None else None, "lfs_sha256": lfs_sha})
if len({row["path"] for row in rows}) != len(rows):
    raise RuntimeError("duplicate canonical HF tree path")
Path(output).write_text(json.dumps({"repository": repo, "revision": rev, "resolved_revision": info.sha, "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n")
PY

git clone --no-tags --filter=blob:none "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'
set +e
uv run --no-sync --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" \
  --snapshot "$work_dir/hf" --source "$work_dir/source/repo" \
  --server-tree "$work_dir/server-tree.json" --output "$work_dir/evidence" \
  >> "$work_dir/evidence/validation.log" 2>&1
status=$?
set -e
[[ "$status" == 2 ]] || die 'inspector must exit 2 fail-closed'
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'blocker manifest missing'
grep -Fq '"runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED"' "$work_dir/evidence/manifest.json" || die 'runtime status missing'
grep -Fq '"publication": "NO_UPLOAD"' "$work_dir/evidence/manifest.json" || die 'publication status missing'
log "inspection blocked by contract; evidence preserved at $work_dir/evidence"
exit 2
