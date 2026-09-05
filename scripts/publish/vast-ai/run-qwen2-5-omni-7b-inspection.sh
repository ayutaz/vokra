#!/usr/bin/env bash
# VAST-only Qwen2.5-Omni-7B inspection. Never converts, executes, uploads, or publishes.
set -euo pipefail
HF_REPOSITORY="Qwen/Qwen2.5-Omni-7B"
HF_REVISION="ae9e1690543ffd5c0221dc27f79834d0294cba00"
SOURCE_REPOSITORY="https://github.com/QwenLM/Qwen2.5-Omni.git"
SOURCE_REVISION="d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_TAG="v4.52.3"
TRANSFORMERS_REVISION="f4fc42216cd56ab6b68270bf80d811614d8d59e4"
INSPECTOR="tools/parity/qwen2_5_omni_7b_inspect.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((80 * 1024 * 1024))

die() { echo "run-qwen2-5-omni-7b-inspection: $*" >&2; exit 2; }
self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 required status
  root="$(cd "$(dirname "$self")/../../.." && pwd)"; [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  for required in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_TAG" "$TRANSFORMERS_REVISION" "$INSPECTOR" "SHARD_BYTES" "spk_dict.pt" "weights_only=True" "MAX_HEADER_BYTES" "INSPECTION_ONLY" "NO_UPLOAD" "BLOCKED" "git_blob_sha1" "lfs_sha256" "thinker_config" "token2wav_config" "qwen2_5_omni" "local_dir" ".cache"; do
    if ! grep -Fq -- "$required" "$self" && ! grep -Fq -- "$required" "$root/$INSPECTOR"; then echo "self-test FAIL: missing $required" >&2; fail=1; fi
  done
  if ! grep -Fq '"resolved_revision": info.sha, "files"' "$self"; then echo "self-test FAIL: resolved_revision must use verified HF revision" >&2; fail=1; fi
  for required in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'git status --porcelain --untracked-files=all' 'snapshot_download' 'model_info' 'list_repo_tree' 'CARGO_BUILD_JOBS'; do
    if ! grep -Fq -- "$required" "$self"; then echo "self-test FAIL: missing VAST gate $required" >&2; fail=1; fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: mutation/conversion/Cargo test found" >&2; fail=1; fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then echo "self-test FAIL: raw Python/pip found" >&2; fail=1; fi
  if bash "$self" --self-test --work-dir /tmp/qwen2-omni-self-test >/dev/null 2>&1; then echo "self-test FAIL: extra arg accepted" >&2; fail=1; else status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }; fi
  (( fail == 0 )) && echo "run-qwen2-5-omni-7b-inspection.sh self-test: OK" || return 1
}
work_dir="/dev/shm/vokra-qwen2-5-omni-7b-inspection"; self=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self=1; shift;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires path"; work_dir="$2"; shift 2;;
    -h|--help) echo "usage: $0 [--work-dir TMPFS] | --self-test"; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
if ((self == 1)); then [[ "$work_dir" == "/dev/shm/vokra-qwen2-5-omni-7b-inspection" ]] || die "--self-test accepts no other arguments"; self_test; exit $?; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"; [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
parent="$(dirname "$work_dir")"; [[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die "work path is not directory"; [[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work path is not empty"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 128 GiB"
free="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_DISK_KIB" ]] || die "disk below guard"
for command in git rustc cargo rustfmt uv sha256sum findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool $command"; done
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
cache="$work_dir/cache"; model_dir="$work_dir/model"; snapshot_file="$work_dir/snapshot"; tree="$work_dir/server-tree.json"; source="$work_dir/source"; transformers="$work_dir/transformers"; evidence="$work_dir/evidence"; mkdir -p "$cache" "$model_dir" "$evidence"
"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$cache" "$model_dir" "$snapshot_file" "$tree" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, snapshot_download
repo, rev, cache, model_dir, output, tree = sys.argv[1:]; api = HfApi(); info = api.model_info(repo_id=repo, revision=rev)
if info.sha != rev: raise SystemExit(f"HF revision drift: {info.sha} != {rev}")
snapshot = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir=cache, local_dir=model_dir, allow_patterns=["*"], token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if snapshot.resolve() != Path(model_dir).resolve(): raise SystemExit("materialized local_dir mismatch")
files = []
for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True):
    if not isinstance(item, RepoFile): continue
    lfs = getattr(item, "lfs", None); lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    git_oid = getattr(item, "blob_id", None)
    if not isinstance(item.path, str) or not isinstance(item.size, int) or not isinstance(git_oid, str) or len(git_oid) != 40 or any(c not in "0123456789abcdefABCDEF" for c in git_oid): raise SystemExit(f"incomplete server Git blob entry: {item}")
    if lfs_sha is not None and (not isinstance(lfs_sha, str) or len(lfs_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in lfs_sha)): raise SystemExit(f"incomplete server LFS entry: {item}")
    files.append({"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": git_oid, "lfs_sha256": lfs_sha})
expected = {row["path"] for row in files}; actual = set()
for path in snapshot.rglob("*"):
    if ".cache" in path.relative_to(snapshot).parts: continue
    if path.is_symlink():
        if not path.exists() or not path.is_file(): raise SystemExit(f"snapshot dangling/non-file symlink: {path}")
        if snapshot not in path.resolve().parents: raise SystemExit(f"snapshot symlink escapes root: {path}")
        actual.add(path.relative_to(snapshot).as_posix())
    elif path.is_file(): actual.add(path.relative_to(snapshot).as_posix())
    elif not path.is_dir(): raise SystemExit(f"snapshot non-regular member: {path}")
if actual != expected: raise SystemExit(f"server/local file-only walk mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
Path(output).write_text(str(snapshot) + "\n", encoding="utf-8")
Path(tree).write_text(json.dumps({"repository": repo, "revision": rev, "resolved_revision": info.sha, "files": sorted(files, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
snapshot="$(< "$snapshot_file")"
git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1; git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1; [[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "source revision mismatch"
git clone --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1; git -C "$transformers" checkout --detach "$TRANSFORMERS_REVISION" >/dev/null 2>&1; [[ "$(git -C "$transformers" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] || die "Transformers revision mismatch"; [[ "$(git -C "$transformers" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] || die "Transformers tag mismatch"
set +e; "${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot" --source "$source" --transformers "$transformers" --server-tree "$tree" --output "$evidence"; status=$?; set -e
[[ "$status" == 2 ]] || die "inspection did not return exit 2"; grep -Fq '"status": "BLOCKED"' "$evidence/manifest.json" || die "missing BLOCKED manifest"; grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$evidence/manifest.json" || die "missing evidence stage"; echo "Qwen2.5-Omni inspection BLOCKED; evidence=$evidence" >&2; exit 2
