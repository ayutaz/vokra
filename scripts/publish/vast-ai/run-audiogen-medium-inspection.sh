#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/audiogen_medium_reference"
INSPECTOR="$ROOT/tools/parity/audiogen_medium_inspect.py"
REFERENCE="$ROOT/tools/parity/audiogen_medium_dump_reference.py"
HF_REPOSITORY="facebook/audiogen-medium"; HF_REVISION="1277dd7dfd8fa57a205a70acc5de0ee90804502f"
SOURCE_URL="https://github.com/facebookresearch/audiocraft.git"; SOURCE_REVISION="a2b96756956846e194c9255d0cdadc2b47c93f1b"
MIN_MEM_KIB=$((64*1024*1024)); MIN_DISK_KIB=$((16*1024*1024))
die(){ echo "audiogen-medium-vast: BLOCKED: $*" >&2; exit 2; }
validate_manifest(){
  local path="$1"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import json,sys
from pathlib import Path
def unique(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise SystemExit(f"duplicate manifest key: {key}")
  out[key]=value
 return out
m=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),object_pairs_hook=unique)
required={"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","runtime_status":"LM_ONLY_PCM_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD"}
for key,want in required.items():
 if m.get(key)!=want: raise SystemExit(f"manifest marker mismatch: {key}={m.get(key)!r}")
up=m.get("upstream")
if not isinstance(up,dict) or up.get("repository")!="facebook/audiogen-medium" or up.get("requested_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f" or up.get("resolved_revision")!="1277dd7dfd8fa57a205a70acc5de0ee90804502f": raise SystemExit("upstream identity mismatch")
if m.get("inspection_status") in {"INSPECTION_ERROR","FAILED"} or m.get("collection_status")!="AUTHENTICATED": raise SystemExit("incomplete/error evidence rejected")
PY
}
self_test(){
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  grep -Fq 'LM_ONLY_PCM_FAIL_CLOSED' "$INSPECTOR"
  grep -Fq 'AUTHENTICATED_EVIDENCE_COMPLETE' "$INSPECTOR"
  grep -Fq 'snapshot_download' "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh"
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test
  UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --self-test
  echo 'run-audiogen-medium-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then self_test "$@"; exit 0; fi
[[ $# == 0 ]] || die 'usage: run-audiogen-medium-inspection.sh [--self-test]'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$PROJECT/uv.lock" ]] || die 'dedicated AudioGen uv.lock absent; fail before downloads'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '64 GiB memory guard failed'
for command in git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
work=/dev/shm/vokra-audiogen-medium-inspection
if [[ -e "$work" ]]; then [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection tmpfs must be empty'; else mkdir -p "$work"; fi
mkdir -p "$work/model" "$work/source" "$work/evidence"
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die '16 GiB tmpfs guard failed'
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/model" "$work/tree.json" <<'PY' >"$work/evidence/acquisition.log" 2>&1
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder,snapshot_download
repo,rev,destination,out=sys.argv[1:]; api=HfApi(); info=api.model_info(repo_id=repo,revision=rev)
if info.sha != rev: raise SystemExit("resolved HF revision mismatch")
snapshot=Path(snapshot_download(repo_id=repo,revision=rev,local_dir=destination,allow_patterns=["*"]))
if snapshot.resolve()!=Path(destination).resolve(): raise SystemExit("snapshot_download escaped local_dir")
expected={".gitattributes","README.md","compression_state_dict.bin","state_dict.bin"}; rows=[]
fixed={".gitattributes":(1519,"a6344aac8c09253b3b630fb776ae94478aa0275b"),"README.md":(2240,"31a77819df582937de900237706f104a325e223f"),"compression_state_dict.bin":(235740815,"0cc8de6c4cf0c16326ee3c693385370b98bbf0f2","5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095"),"state_dict.bin":(3678455287,"ae572ad32705a0a9ba679b0d2813cbae716d869e","f3b20997834de1ca47d6a31d00a5dc37019b279c7c8f250fd482d56def04faaa")}
def pointer_sha1(sha,size):
 pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode(); h=hashlib.sha1(f"blob {len(pointer)}\0".encode()); h.update(pointer); return h.hexdigest()
for item in api.list_repo_tree(repo_id=repo,revision=rev,recursive=True,expand=True):
 if isinstance(item,RepoFolder): continue
 if not isinstance(item,RepoFile): raise SystemExit(f"unknown HF tree entry: {item!r}")
 if item.path not in expected: raise SystemExit(f"unexpected HF tree file: {item.path}")
 if not isinstance(item.size,int) or item.size<=0 or not isinstance(item.blob_id,str) or len(item.blob_id)!=40 or item.size!=fixed[item.path][0]: raise SystemExit(f"incomplete/fixed HF identity: {item.path}")
 lfs=getattr(item,"lfs",None); sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None); lfs_size=lfs.get("size") if isinstance(lfs,dict) else getattr(lfs,"size",None)
 if len(fixed[item.path])==2:
  if sha is not None or item.blob_id!=fixed[item.path][1]: raise SystemExit(f"regular Git identity mismatch: {item.path}")
  row={"path":item.path,"type":"file","size":item.size,"git_blob_sha1":item.blob_id,"lfs_pointer_git_blob_sha1":None,"lfs_payload_sha256":None,"lfs_payload_size":None}
 else:
  if sha is not None and (not isinstance(sha,str) or sha!=fixed[item.path][2]) or lfs_size is not None and lfs_size!=item.size: raise SystemExit(f"invalid LFS metadata: {item.path}")
  if item.blob_id!=fixed[item.path][1] or pointer_sha1(fixed[item.path][2],item.size)!=item.blob_id: raise SystemExit(f"invalid fixed LFS identity: {item.path}")
  row={"path":item.path,"type":"file","size":item.size,"git_blob_sha1":None,"lfs_pointer_git_blob_sha1":item.blob_id,"lfs_payload_sha256":fixed[item.path][2],"lfs_payload_size":item.size}
 rows.append(row)
if {row["path"] for row in rows}!=expected or len(rows)!=len(expected): raise SystemExit("HF tree is not exact four-file set")
Path(out).write_text(json.dumps({"repository":repo,"requested_revision":rev,"resolved_revision":info.sha,"walk":"recursive_file_only","files":sorted(rows,key=lambda row:row["path"])},sort_keys=True,indent=2)+"\n")
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/acquisition.log" 2>&1
git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/acquisition.log" 2>&1
UV_CACHE_DIR="${AUDIOGEN_UV_CACHE_DIR:-/private/tmp/vokra-audiogen-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/repo" --server-tree "$work/tree.json" --output "$work/evidence" >>"$work/evidence/acquisition.log" 2>&1 || [[ $? == 2 ]]
validate_manifest "$work/evidence/manifest.json" || die 'inspection did not produce complete authenticated evidence'
exit 2
