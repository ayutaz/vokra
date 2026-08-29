#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audioldm2_reference"
INSPECTOR="$ROOT/tools/parity/audioldm2_inspect.py"
REFERENCE="$ROOT/tools/parity/audioldm2_dump_reference.py"
BASE_REPOSITORY="cvssp/audioldm2"
BASE_REVISION="c8e7e189d324425c05c4c2f81214041ef4107983"
LARGE_REPOSITORY="cvssp/audioldm2-large"
LARGE_REVISION="4b0b875a9e0c5305dfc917da808584e50e1c7ed4"
SOURCE_REPOSITORY="https://github.com/huggingface/diffusers.git"
SOURCE_REVISION="29f15673ed5c14e4843d7c837890910207f72129"
SOURCE_TAG="v0.21.0"
die(){ echo "audioldm2-vast: BLOCKED: $*" >&2; exit 2; }
if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'uv.lock' "$INSPECTOR" "$REFERENCE" "$0"
  grep -Fq 'dedicated AudioLDM2 uv.lock is absent' "$0"
  grep -Fq 'snapshot_download' "$0"
  grep -Fq 'list_repo_tree' "$0"
  grep -Fq "$SOURCE_TAG" "$0"
  grep -Fq 'RepoFolder' "$0"
  grep -Fq 'cc-by-nc-sa-4.0' "$INSPECTOR"
  grep -Fq '0.20.0.dev0' "$INSPECTOR"
  grep -Fq 'NO_UPLOAD' "$INSPECTOR"
  UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test
  UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test
  echo 'run-audioldm2-inspection.sh self-test: OK'
  exit 0
fi
[[ $# == 0 ]] || die 'usage: run-audioldm2-inspection.sh [--self-test]'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioLDM2 uv.lock is absent; fail before downloads'
evidence_dir="${AUDIO_LDM2_EVIDENCE_DIR:-}"
[[ -n "$evidence_dir" ]] || die 'evidence directory is required'
work_dir="${AUDIO_LDM2_WORK_DIR:-/dev/shm/vokra-audioldm2-inspection}"
parent="$(dirname "$work_dir")"
[[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die 'work path is not a directory'
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work path must be empty'
mkdir -p "$work_dir"
if [[ -n "${AUDIO_LDM2_LARGE:-}" ]]; then HF_REPOSITORY="$LARGE_REPOSITORY"; HF_REVISION="$LARGE_REVISION"; LARGE_FLAG=1; else HF_REPOSITORY="$BASE_REPOSITORY"; HF_REVISION="$BASE_REVISION"; LARGE_FLAG=0; fi
cache="$work_dir/cache"; model_tree="$work_dir/model"; source_tree="$work_dir/source"; mkdir -p "$cache" "$model_tree"
UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
  uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$cache" "$model_tree" <<'PY'
import hashlib, json, os, re, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download
repo, revision, cache, destination = sys.argv[1:]
api = HfApi(); info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision: raise SystemExit("resolved revision drift")
from audioldm2_large_prepare_checkpoint import REQUIRED_TREE as LARGE_TREE
from audioldm2_prepare_checkpoint import REQUIRED_TREE as BASE_TREE
expected = LARGE_TREE if repo.endswith("-large") else BASE_TREE
snapshot = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, local_dir=destination, allow_patterns=sorted(expected), token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if snapshot.resolve() != Path(destination).resolve(): raise SystemExit("local_dir materialization mismatch")
rows=[]; seen=set()
for item in api.list_repo_tree(repo_id=repo, revision=revision, recursive=True, expand=True):
    if isinstance(item, RepoFolder): continue
    if not isinstance(item, RepoFile): raise SystemExit(f"unknown server item: {item}")
    path=item.path; blob=getattr(item,"blob_id",None); size=item.size
    if not isinstance(path,str) or not path or "\\" in path or path.startswith("/") or ".." in path.split("/") or path in seen or path not in expected: raise SystemExit(f"unsafe/unexpected file: {path!r}")
    if not isinstance(size,int) or isinstance(size,bool) or size<0 or not isinstance(blob,str) or re.fullmatch(r"[0-9a-f]{40}",blob) is None: raise SystemExit(f"invalid server identity: {path}")
    lfs=getattr(item,"lfs",None); lfs_sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None); lfs_size=lfs.get("size") if isinstance(lfs,dict) else getattr(lfs,"size",None)
    if lfs_sha is None:
        row={"path":path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_pointer_git_blob_sha1":None,"lfs_payload_sha256":None,"lfs_payload_size":None}
    else:
        if not isinstance(lfs_sha,str) or re.fullmatch(r"[0-9a-f]{64}",lfs_sha) is None or not isinstance(lfs_size,int) or isinstance(lfs_size,bool) or lfs_size != size: raise SystemExit(f"invalid LFS identity: {path}")
        pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {size}\n".encode(); pointer_blob=hashlib.sha1(f"blob {len(pointer)}\0".encode()+pointer).hexdigest()
        if pointer_blob != blob: raise SystemExit(f"LFS pointer mismatch: {path}")
        row={"path":path,"type":"file","size":size,"git_blob_sha1":None,"lfs_pointer_git_blob_sha1":blob,"lfs_payload_sha256":lfs_sha,"lfs_payload_size":size}
    rows.append(row); seen.add(path)
if seen != expected: raise SystemExit(f"server tree mismatch: {sorted(expected-seen)}")
Path(destination,".vokra-server-tree.json").write_text(json.dumps({"repository":repo,"requested_revision":revision,"resolved_revision":info.sha,"walk":"recursive_file_only","files":sorted(rows,key=lambda row:row["path"])},sort_keys=True,indent=2)+"\n")
PY
git init "$source_tree" >/dev/null 2>&1 || die 'source init failed'
git -C "$source_tree" remote add origin "$SOURCE_REPOSITORY" || die 'source origin failed'
git -C "$source_tree" fetch --filter=blob:none --depth=1 origin "refs/tags/$SOURCE_TAG:refs/tags/$SOURCE_TAG" "$SOURCE_REVISION" >/dev/null 2>&1 || die 'source tag/commit fetch failed'
[[ "$(git -C "$source_tree" rev-parse "refs/tags/$SOURCE_TAG^{commit}")" == "$SOURCE_REVISION" ]] || die 'source tag object drift'
git -C "$source_tree" checkout --detach "$SOURCE_REVISION" >/dev/null 2>&1 || die 'source checkout failed'
if (( LARGE_FLAG )); then large_arg=(--large); else large_arg=(); fi
UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
  uv run --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" \
    --snapshot "$model_tree" --source "$source_tree" --output "$evidence_dir" "${large_arg[@]}"
