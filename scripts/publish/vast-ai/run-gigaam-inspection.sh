#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INSPECTOR="$ROOT/tools/parity/gigaam_inspect.py"
SOURCE_URL="https://github.com/salute-developers/GigaAM.git"
SOURCE_REVISION="7447938d791c4f3e643386ee22c33777004293a5"
die(){ echo "gigaam-vast: ERROR: $*" >&2; exit 2; }
self_test(){
 local path="${BASH_SOURCE[0]}" token fail=0 tree_source
 for token in "$SOURCE_URL" "$SOURCE_REVISION" 'weights_only=True' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'NO_UPLOAD' 'CARGO_BUILD_JOBS=1' 'model_class' 'RNNT' 'CTC' 'RepoFile' 'RepoFolder' 'path_in_repo' 'recursive=False'; do
  grep -Fq -- "$token" "$path" || grep -Fq -- "$token" "$INSPECTOR" || { echo "missing contract $token" >&2; fail=1; }
 done
 if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi
 tree_source="$(mktemp "${TMPDIR:-/tmp}/gigaam-hf-tree-self-test.XXXXXX.py")"
 awk '/^import json,sys,re,os$/{capture=1} capture && /^PY$/{exit} capture{print}' "$path" >"$tree_source"
 if ! PYTHONPATH="$ROOT" GIGAAM_HF_TREE_SELF_TEST=1 UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$tree_source" v3 /tmp/gigaam-hf-tree-self-test.json; then
  echo 'gigaam-vast: tree collector self-test failed' >&2
  fail=1
 fi
 rm -f -- "$tree_source" /tmp/gigaam-hf-tree-self-test.json
 UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" v3 --self-test || fail=1
 UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" multilingual --self-test || fail=1
 ((fail==0)) || return 1
 echo 'run-gigaam-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit 0; fi
[[ $# == 1 && ( "$1" == v3 || "$1" == multilingual ) ]] || die 'usage: run-gigaam-inspection.sh {v3|multilingual}'
variant="$1"; [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'; [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'; [[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'; free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((32*1024*1024)) ]] || die 'tmpfs disk guard failed'
work="/dev/shm/vokra-gigaam-$variant"; [[ ! -e "$work" ]] || [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory must be empty'; mkdir -p "$work/model" "$work/source" "$work/evidence"; export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${GIGAAM_UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}"
{ cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1; } >"$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129 # validation output is one evidence stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$variant" "$work/tree.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys,re,os
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder
from tools.parity.gigaam_inspect import VARIANTS
variant,out=sys.argv[1:]; spec=VARIANTS[variant]
def field(item,name,default=None):
 return item.get(name,default) if isinstance(item,dict) else getattr(item,name,default)
def classify_entry(item):
 if isinstance(item,RepoFolder):
  if field(item,"type") not in (None,"directory"): raise RuntimeError(f"invalid RepoFolder type: {item!r}")
  return "directory"
 if isinstance(item,RepoFile):
  if field(item,"type") not in (None,"file"): raise RuntimeError(f"invalid RepoFile type: {item!r}")
  return "file"
 raise RuntimeError(f"unknown Hugging Face tree entry: {type(item).__name__}")
def walk_tree(api,repository,revision,expected_count):
 pending=[("",0)]; seen=set(); visited=0; max_items=max(1024,expected_count*32)
 while pending:
  path_in_repo,depth=pending.pop()
  if path_in_repo in seen: continue
  if depth>64: raise RuntimeError(f"Hugging Face tree depth exceeds bound: {path_in_repo!r}")
  seen.add(path_in_repo)
  entries=api.list_repo_tree(repo_id=repository,revision=revision,path_in_repo=path_in_repo,recursive=False)
  for item in entries:
   visited+=1
   if visited>max_items: raise RuntimeError("Hugging Face tree item bound exceeded")
   kind=classify_entry(item); path=field(item,"path")
   if not isinstance(path,str) or not path or path.startswith(("/","./")) or "\\" in path or ".." in Path(path).parts or not Path(path).name:
    raise RuntimeError(f"unsafe Hugging Face tree path: {path!r}")
   if kind=="directory": pending.append((path,depth+1))
   else: yield item
def file_row(item):
 lfs=field(item,"lfs"); lfs_sha=field(lfs,"sha256") if lfs is not None else None; lfs_size=field(lfs,"size") if lfs is not None else None
 path=field(item,"path"); blob=field(item,"blob_id") or field(item,"oid"); size=field(item,"size")
 if not (isinstance(path,str) and path and "\\" not in path and ".." not in Path(path).parts and not path.startswith(("/","./")) and Path(path).name and isinstance(size,int) and not isinstance(size,bool) and size>=0 and re.fullmatch(r"[0-9a-f]{40}",str(blob))):
  raise RuntimeError(f"invalid Hugging Face file metadata: {path!r}")
 if lfs is not None and not (isinstance(lfs_sha,str) and re.fullmatch(r"[0-9a-f]{64}",lfs_sha) and isinstance(lfs_size,int) and not isinstance(lfs_size,bool) and lfs_size==size):
  raise RuntimeError(f"invalid Hugging Face LFS metadata: {path!r}")
 row={"path":path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha}
 return row
if os.environ.get("GIGAAM_HF_TREE_SELF_TEST")=="1":
 file_entry=RepoFile(path="model.bin",size=1,oid="a"*40); file_entry.type=None
 nested_file=RepoFile(path="nested/file.bin",size=2,oid="b"*40); nested_file.type=None
 folder_entry=RepoFolder(path="nested",oid="c"*40); folder_entry.type=None
 class Paged:
  def __init__(self,pages): self.pages=pages; self.consumed=False
  def __iter__(self):
   for page in self.pages:
    yield from page
   self.consumed=True
 class FakeApi:
  def __init__(self): self.calls=[]; self.pages={"":Paged([[folder_entry],[file_entry]]),"nested":Paged([[nested_file]])}
  def list_repo_tree(self,**kwargs):
   self.calls.append(kwargs)
   if set(kwargs)-{"repo_id","revision","path_in_repo","recursive"}: raise AssertionError(f"unexpected API kwargs: {kwargs}")
   return self.pages[kwargs["path_in_repo"]]
 fake=FakeApi(); rows=list(walk_tree(fake,spec["repository"],spec["revision"],len(spec["files"])))
 assert [field(item,"path") for item in rows]==["model.bin","nested/file.bin"]
 assert fake.pages[""].consumed and fake.pages["nested"].consumed
 assert fake.calls[0]["path_in_repo"]=="" and fake.calls[0]["recursive"] is False
 assert fake.calls[1]["path_in_repo"]=="nested" and "path" not in fake.calls[0]
 lfs_file=RepoFile(path="lfs.bin",size=3,oid="e"*40); lfs_file.type=None; lfs_file.lfs={"sha256":"f"*64,"size":3}
 assert file_row(lfs_file)["lfs_sha256"]=="f"*64
 for bad_lfs in ({"size":3},{"sha256":"f"*64,"size":2}):
  lfs_file.lfs=bad_lfs
  try: file_row(lfs_file)
  except RuntimeError: pass
  else: raise AssertionError(f"invalid LFS metadata accepted: {bad_lfs!r}")
 for invalid in (object(),{"path":"dict.bin","type":"file"}):
  try: classify_entry(invalid)
  except RuntimeError: pass
  else: raise AssertionError(f"unknown tree entry accepted: {invalid!r}")
 print("GigaAM Hub tree class/path/pagination self-test: PASS"); raise SystemExit(0)
api=HfApi(); info=api.model_info(spec["repository"],revision=spec["revision"]); assert info.sha==spec["revision"]; rows=[]
for item in walk_tree(api,spec["repository"],spec["revision"],len(spec["files"])):
 rows.append(file_row(item))
assert {row["path"] for row in rows}==set(spec["files"])
Path(out).write_text(json.dumps({"repository":spec["repository"],"revision":spec["revision"],"resolved_revision":info.sha,"walk":"recursive_file_only","files":rows},sort_keys=True,indent=2)+"\n")
PY
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$variant" "$work/model" <<'PY' >>"$work/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
from tools.parity.gigaam_inspect import VARIANTS
spec=VARIANTS[sys.argv[1]]; snapshot_download(repo_id=spec["repository"],revision=spec["revision"],local_dir=sys.argv[2],allow_patterns=list(spec["files"]))
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/validation.log" 2>&1; [[ "$(git -C "$work/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'; [[ "$(git -C "$work/source/repo" remote get-url origin)" == "$SOURCE_URL" ]] || die 'source origin mismatch'
set +e; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" "$variant" --snapshot "$work/model" --source "$work/source/repo" --server-tree "$work/tree.json" --output "$work/evidence" >>"$work/evidence/validation.log" 2>&1; rc=$?; set -e; [[ "$rc" == 2 ]] || die 'inspector must exit 2'; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json,sys
p=json.loads(open(sys.argv[1]).read()); assert p["status"]=="BLOCKED" and p["inspection_status"]=="AUTHENTICATED_EVIDENCE_COMPLETE" and p["evidence_stage"]=="INSPECTION_ONLY" and p["runtime_status"]=="NOT_IMPLEMENTED_FAIL_CLOSED" and p["publication"]=="NO_UPLOAD"
PY
exit 2
