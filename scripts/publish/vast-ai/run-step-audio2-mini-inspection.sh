#!/usr/bin/env bash
# VAST-only Step-Audio-2-mini composite inspection. No conversion, runtime,
# ONNX execution, parity, upload, or publication is performed.
set -euo pipefail

HF_REPOSITORY="stepfun-ai/Step-Audio-2-mini"
HF_REVISION="e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36"
SOURCE_REPOSITORY="https://github.com/stepfun-ai/Step-Audio2.git"
SOURCE_REVISION="76e272b56c3917a8d7188f18bbb5a65dfc8a0845"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers"
TRANSFORMERS_TAG="v4.49.0"
TRANSFORMERS_REVISION="a22a4378d97d06b7a1d9abad6e0086d30fdea199"
INSPECTOR="tools/parity/step_audio2_mini_inspect.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((80 * 1024 * 1024))

die() { echo "run-step-audio2-mini-inspection: $*" >&2; exit 2; }

self_test() {
  local self="${BASH_SOURCE[0]}" root fail=0 required status
  root="$(cd "$(dirname "$self")/../../.." && pwd)"
  [[ -f "$root/$INSPECTOR" ]] || die "inspector missing"
  for required in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_TAG" "$TRANSFORMERS_REVISION" "$INSPECTOR" "SHARD_BYTES" "COMPANIONS" "campplus.onnx" "speech_tokenizer_v2_25hz.onnx" "flow.pt" "hift.pt" "flow.yaml" "configuration_step_audio_2.py" "modeling_step_audio_2.py" "audio_encoder_config" "n_mels" "n_audio_ctx" "n_audio_state" "n_audio_head" "n_audio_layer" "n_codebook_size" "llm_dim" "kernel_size" "adapter_stride" "preprocessor_config.json" "generation_config.json" "inspection_status" "AUTHENTICATED_EVIDENCE_COMPLETE" "collection_status" "AUTHENTICATED" "INSPECTION_ERROR" "UNVERIFIED" "load_external_data=False" "weights_only=True" "INSPECTION_ONLY" "NO_UPLOAD" "BLOCKED" "MAX_HEADER_BYTES" "requested_revision" "git_blob_sha1" "lfs_pointer_git_blob_sha1" "lfs_payload_sha256" "lfs_payload_size"; do
    if ! grep -Fq -- "$required" "$self" && ! grep -Fq -- "$required" "$root/$INSPECTOR"; then
      echo "self-test FAIL: missing contract: $required" >&2; fail=1
    fi
  done
  for required in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'findmnt' 'git status --porcelain --untracked-files=all' 'snapshot_download' 'model_info' 'list_repo_tree' 'CARGO_BUILD_JOBS'; do
    if ! grep -Fq -- "$required" "$self"; then echo "self-test FAIL: missing VAST gate: $required" >&2; fail=1; fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check))([[:space:]]|$)' "$self" >/dev/null; then
    echo "self-test FAIL: mutation/conversion/Cargo test command found" >&2; fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$self" >/dev/null; then
    echo "self-test FAIL: raw Python/pip command found" >&2; fail=1
  fi
  if bash "$self" --self-test --work-dir /tmp/step-audio2-self-test >/dev/null 2>&1; then
    echo "self-test FAIL: extra argument accepted" >&2; fail=1
  else
    status=$?; [[ "$status" == 2 ]] || { echo "self-test FAIL: expected exit 2, got $status" >&2; fail=1; }
  fi
  if ((fail == 0)); then echo "run-step-audio2-mini-inspection.sh self-test: OK"; else return 1; fi
}

work_dir="/dev/shm/vokra-step-audio2-mini-inspection"
self=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires a path"; work_dir="$2"; shift 2 ;;
    -h|--help) echo "usage: $0 [--work-dir TMPFS] | --self-test"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if ((self == 1)); then
  [[ "$work_dir" == "/dev/shm/vokra-step-audio2-mini-inspection" ]] || die "--self-test accepts no other arguments"
  self_test; exit $?
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "inspection requires Linux x86_64 VAST"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
parent="$(dirname "$work_dir")"
[[ -d "$parent" && "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die "work parent must be tmpfs"
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die "work path is not a directory"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work path must be empty"
mem="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"; [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge "$MIN_MEM_KIB" ]] || die "RAM below 128 GiB"
free="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"; [[ "$free" =~ ^[0-9]+$ && "$free" -ge "$MIN_DISK_KIB" ]] || die "tmpfs free space below 80 GiB"
for command in git cargo rustc rustfmt uv sha256sum findmnt; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; export CARGO_BUILD_JOBS=1
snapshot="$work_dir/model"; tree="$work_dir/server-tree.json"; source="$work_dir/source"; transformers="$work_dir/transformers"; evidence="$work_dir/evidence"; mkdir -p "$evidence"

"${UV_CMD[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$snapshot" "$tree" <<'PY'
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, snapshot_download
repo, rev, destination, tree = sys.argv[1:]
api = HfApi(); info = api.model_info(repo_id=repo, revision=rev)
if info.sha != rev: raise SystemExit(f"HF revision drift: {info.sha} != {rev}")
snapshot = Path(snapshot_download(repo_id=repo, revision=rev, local_dir=destination, allow_patterns=["*"], token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if snapshot.resolve() != Path(destination).resolve(): raise SystemExit("local materialization path drift")
files = []
for item in api.list_repo_tree(repo_id=repo, revision=rev, recursive=True):
    kind = getattr(item, "type", None)
    if kind in {"directory", "folder", "dir"} or item.__class__.__name__ == "RepoFolder": continue
    if not isinstance(item, RepoFile) and kind != "file": raise SystemExit(f"recursive tree returned unsupported entry: {item}")
    lfs = getattr(item, "lfs", None); lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None); lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", item.size); pointer_blob = getattr(item, "blob_id", None)
    local = snapshot / item.path if isinstance(item.path, str) else None
    if not isinstance(item.path, str) or not item.path or "\\" in item.path or "\x00" in item.path or Path(item.path).is_absolute() or ".." in Path(item.path).parts or not isinstance(item.size, int) or isinstance(item.size, bool) or item.size <= 0 or local is None or not local.is_file() or local.is_symlink() or local.stat().st_size != item.size: raise SystemExit(f"incomplete server tree entry: {item}")
    if not isinstance(pointer_blob, str) or len(pointer_blob) != 40 or any(c not in "0123456789abcdef" for c in pointer_blob.lower()): raise SystemExit(f"missing Git identity: {item.path}")
    if lfs is None:
        files.append({"path": item.path, "size": item.size, "git_blob_sha1": pointer_blob, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None})
    else:
        if not isinstance(lfs_sha, str) or len(lfs_sha) != 64 or any(c not in "0123456789abcdef" for c in lfs_sha.lower()): raise SystemExit(f"missing LFS payload identity: {item.path}")
        if lfs_size != item.size: raise SystemExit(f"LFS size mismatch: {item.path}")
        files.append({"path": item.path, "size": item.size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": pointer_blob, "lfs_payload_sha256": lfs_sha, "lfs_payload_size": item.size})
if len({row["path"] for row in files}) != len(files): raise SystemExit("duplicate server path")
Path(tree).write_text(json.dumps({"repository": repo, "requested_revision": rev, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(files, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
for forbidden in preprocessor_config.json generation_config.json; do
  [[ ! -e "$snapshot/$forbidden" ]] || die "forbidden canonical root file present: $forbidden"
done
git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source" >/dev/null 2>&1; git -C "$source" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1
[[ "$(git -C "$source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "source revision mismatch"
git clone --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1; git -C "$transformers" checkout --detach "$TRANSFORMERS_REVISION" >/dev/null 2>&1
[[ "$(git -C "$transformers" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] || die "Transformers revision mismatch"
[[ "$(git -C "$transformers" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] || die "Transformers tag mismatch"
set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot" --source "$source" --transformers "$transformers" --server-tree "$tree" --output "$evidence"
status=$?
set -e
[[ "$status" == 2 ]] || die "inspection did not return required blocker exit 2"
grep -Fq '"status": "BLOCKED"' "$evidence/manifest.json" || die "missing BLOCKED manifest"
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$evidence/manifest.json" || die "missing INSPECTION_ONLY stage"
"${UV_CMD[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
from pathlib import Path

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate manifest key: {key}")
        result[key] = value
    return result

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=unique)
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "collection_status": "AUTHENTICATED",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "cpu_status": "UNSUPPORTED",
    "metal_status": "BLOCKED_BY_CPU",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}
if any(manifest.get(key) != value for key, value in required.items()):
    raise SystemExit("inspection evidence is incomplete or an inspection error")
PY
echo "Step-Audio-2 inspection BLOCKED; evidence=$evidence" >&2; exit 2
