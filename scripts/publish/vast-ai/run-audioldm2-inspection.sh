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
usage(){ cat >&2 <<'EOF'
usage: run-audioldm2-inspection.sh --expected-head <40-hex> \
       --approval-evidence <file> --approval-sha256 <64-hex> [--self-test]
EOF
}
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
  grep -Fq -- '--expected-head' "$0"
  grep -Fq -- '--approval-sha256' "$0"
  if "$0" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1; then die 'duplicate --expected-head was accepted'; fi
  if "$0" --expected-head 0000000000000000000000000000000000000000 --approval-evidence a --approval-evidence b >/dev/null 2>&1; then die 'duplicate --approval-evidence was accepted'; fi
  if "$0" --expected-head 0000000000000000000000000000000000000000 --approval-sha256 "$(printf '0%.0s' {1..64})" --approval-sha256 "$(printf '1%.0s' {1..64})" >/dev/null 2>&1; then die 'duplicate --approval-sha256 was accepted'; fi
  UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$INSPECTOR" --self-test
  UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
    uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test
  echo 'run-audioldm2-inspection.sh self-test: OK'
  exit 0
fi
expected_head=''; approval_evidence=''; approval_sha256=''; seen_head=0; seen_approval=0; seen_approval_sha=0
while (($#)); do
  case "$1" in
    --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen_head=1; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a path'; approval_evidence="$2"; seen_approval=1; shift 2 ;;
    --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha256="$2"; seen_approval_sha=1; shift 2 ;;
    *) usage; die "unexpected argument: $1" ;;
  esac
done
(( seen_head == 1 )) || { usage; die '--expected-head is required'; }
(( seen_approval == 1 )) || { usage; die '--approval-evidence is required'; }
(( seen_approval_sha == 1 )) || { usage; die '--approval-sha256 is required'; }
if [[ -n "${AUDIO_LDM2_LARGE:-}" ]]; then LARGE_FLAG=1; variant_args=(--large); else LARGE_FLAG=0; variant_args=(); fi
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioLDM2 uv.lock is absent; fail before downloads'
[[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die 'approval evidence must be a non-empty regular file'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ "$(git -C "$ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
UV_CACHE_DIR="${AUDIO_LDM2_UV_CACHE_DIR:-/private/tmp/vokra-audioldm2-uv-cache}" \
  uv run --no-project --offline --python 3.12 python "$INSPECTOR" --validate-approval \
    --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" \
    --expected-head "$expected_head" "${variant_args[@]}" >/dev/null || die 'external approval evidence is invalid'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
evidence_dir="${AUDIO_LDM2_EVIDENCE_DIR:-}"
[[ -n "$evidence_dir" ]] || die 'evidence directory is required'
[[ ! -e "$evidence_dir" && ! -L "$evidence_dir" ]] || die 'evidence directory must be absent/non-symlink'
work_dir="${AUDIO_LDM2_WORK_DIR:-/dev/shm/vokra-audioldm2-inspection}"
parent="$(dirname "$work_dir")"
[[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
[[ ! -e "$work_dir" && ! -L "$work_dir" ]] || die 'work path must be absent/non-symlink (no-clobber)'
if (( LARGE_FLAG )); then HF_REPOSITORY="$LARGE_REPOSITORY"; HF_REVISION="$LARGE_REVISION"; else HF_REPOSITORY="$BASE_REPOSITORY"; HF_REVISION="$BASE_REVISION"; fi
root_real="$(cd -P "$ROOT" && pwd)"
project_real="$(cd -P "$PROJECT" && pwd)"
evidence_real="$(cd -P "$(dirname "$evidence_dir")" && pwd)/$(basename "$evidence_dir")"
work_real="$(cd -P "$parent" && pwd)/$(basename "$work_dir")"
approval_real="$(cd -P "$(dirname "$approval_evidence")" && pwd)/$(basename "$approval_evidence")"
paths_overlap(){ local left="$1" right="$2"; [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]; }
paths_overlap "$evidence_real" "$root_real" && die 'evidence directory overlaps checkout'
paths_overlap "$evidence_real" "$project_real" && die 'evidence directory overlaps reference project'
paths_overlap "$evidence_real" "$work_real" && die 'evidence directory overlaps work directory'
paths_overlap "$evidence_real" "$approval_real" && die 'evidence directory overlaps approval evidence'
paths_overlap "$work_real" "$root_real" && die 'work directory overlaps checkout'
paths_overlap "$work_real" "$project_real" && die 'work directory overlaps reference project'
paths_overlap "$work_real" "$approval_real" && die 'work directory overlaps approval evidence'
paths_overlap "$work_real" "$evidence_real" && die 'work directory overlaps evidence directory'
mkdir "$work_dir"
cache="$work_dir/cache"; model_tree="$work_dir/model"; source_tree="$work_dir/source"; mkdir "$cache" "$model_tree"
mkdir "$evidence_dir"
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
    --snapshot "$model_tree" --source "$source_tree" --output "$evidence_dir" \
    --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" \
    --expected-head "$expected_head" "${large_arg[@]}"
