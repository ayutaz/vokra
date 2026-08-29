#!/usr/bin/env bash
# VAST-only Chatterbox composite inspection.  It never converts, publishes, or uploads.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/chatterbox_family_inspect.py"
REFERENCE="$ROOT/tools/parity/chatterbox_t3_reference.py"
REFERENCE_PROJECT="$ROOT/tools/parity/chatterbox_t3"
REFERENCE_LOCK_SHA256="83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37"
SOURCE_URL="https://github.com/resemble-ai/chatterbox.git"
SOURCE_REV="5de7a54aa4e5e2baadb0182dde554908b48b85c2"
WORK="${CHATTERBOX_WORK_DIR:-/dev/shm/vokra-chatterbox-family-inspection}"
UV_CACHE_DIR="${CHATTERBOX_UV_CACHE_DIR:-/tmp/vokra-chatterbox-uv-cache}"
MIN_MEM_KIB=$((128*1024*1024)); MIN_TMPFS_KIB=$((32*1024*1024))
log(){ printf '[chatterbox-vast] %s\n' "$*" >&2; }
die(){ log "ERROR: $*"; exit 2; }
usage(){ echo 'usage: run-chatterbox-family-inspection.sh [--work-dir DIR] | --self-test'; }
license_audit_preflight(){
 set +e
 local audit_output audit_rc
 audit_output="$(UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$REFERENCE" --license-audit 2>&1)"
 audit_rc=$?
 set -e
 if [[ "$audit_rc" == 2 ]]; then
  [[ "$audit_output" == *"$REFERENCE_LOCK_SHA256"* ]] || die 'license audit did not report the reviewed lock identity'
  log "$audit_output"
  return 1
 fi
 [[ "$audit_rc" == 0 ]] || die 'dependency license audit command failed unexpectedly'
 return 0
}
self_test(){
 local fail=0 token
 for token in 'ResembleAI/chatterbox' '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18' 'ResembleAI/chatterbox-nano' '71ccd1d0081b430592cea481f4307e764e07bc64' 'ResembleAI/chatterbox-turbo' '749d1c1a46eb10492095d68fbcf55691ccf137cd' '5de7a54aa4e5e2baadb0182dde554908b48b85c2' 'SOURCE_ROLE_BLOBS' 'git_blob_sha1' 'lfs_sha256' 'path_in_repo' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'NO_UPLOAD' 'CARGO_BUILD_JOBS=1' 'chatterbox_t3/pyproject.toml' 'uv.lock' '--license-audit' 'BLOCKED_UNRESOLVED' 'https://download.pytorch.org/whl/cpu' '2.6.0+cpu' '83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37' 'f5cfab32caf3cc2340b434c1e9e0d3f8dbbab73a519925fbb6f08457c03e7e98' 'package_rows' 'license_conclusions'; do
  grep -Fq -- "$token" "$INSPECTOR" "$0" || { log "self-test FAIL missing $token"; fail=1; }
 done
 if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$0" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("Chatterbox tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"Chatterbox tree walk has incompatible path keyword: {call}")
PY
 then
  log 'self-test FAIL: frozen HfApi.list_repo_tree path_in_repo contract regression'
  fail=1
 fi
 if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        if getattr(entry, "type", None) not in {None, "directory"}:
            raise RuntimeError("unknown RepoFolder type")
        return "directory"
    if isinstance(entry, RepoFile):
        if getattr(entry, "type", None) not in {None, "file"}:
            raise RuntimeError("unknown RepoFile type")
        return "file"
    raise RuntimeError(f"unknown HF tree entry: {entry!r}")

file_entry = RepoFile(path="README.md", size=1, oid="a" * 40)
file_entry.type = None
assert classify_entry(file_entry) == "file"
folder_entry = RepoFolder(path="nested", oid="b" * 40)
folder_entry.type = None
assert classify_entry(folder_entry) == "directory"
try:
    classify_entry(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown HF tree entry was accepted")
print("Chatterbox RepoFile/RepoFolder self-test: PASS")
PY
 then
  log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
  fail=1
 fi
 grep -Eq '^[[:space:]]*(git[[:space:]]+push|hf_hub_upload|upload_file)' "$0" && { log 'self-test FAIL publication command'; fail=1; } || true
 UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
 (( fail == 0 )) || return 1
 log 'self-test PASS'
}
work="$WORK"; self=0
while (($#)); do case "$1" in
 --self-test) self=1; shift;;
 --work-dir) (($#>=2)) || die '--work-dir requires DIR'; work="$2"; shift 2;;
 -h|--help) usage; exit 0;; *) die "unknown argument: $1";; esac; done
if ((self)); then [[ "$work" == "$WORK" ]] || die '--self-test accepts no work-dir'; self_test; exit $?; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated Chatterbox reference pyproject.toml + uv.lock are required before any model download'
if ! license_audit_preflight; then die 'dependency license audit is unresolved; no Chatterbox model acquisition or reference execution is permitted'; fi
for tool in awk cargo df find findmnt git uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool $tool"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'; ((mem_kib>=MIN_MEM_KIB)) || die '128 GiB memory guard failed'
mkdir -p "$(dirname "$work")"; [[ ! -e "$work" || -z "$(find "$work" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
[[ "$(findmnt -T "$(dirname "$work")" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$(dirname "$work")" | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid tmpfs free-space value'; ((free_kib>=MIN_TMPFS_KIB)) || die '32 GiB tmpfs guard failed'
mkdir -p "$work"/evidence "$work"/source
work="$(cd "$work" && pwd)"; export CARGO_BUILD_JOBS=1 UV_CACHE_DIR
printf '%s\n' 'status=BLOCKED' 'evidence_stage=INSPECTION_ONLY' 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED' 'cpu_status=UNSUPPORTED' 'metal_status=BLOCKED_BY_CPU' 'parity_status=NOT_RUN' 'publication=NO_UPLOAD' > "$work/evidence/validation.log"
{
 cargo fmt --all -- --check
 cargo metadata --locked --no-deps --format-version 1 >/dev/null
} >> "$work/evidence/validation.log" 2>&1
for v in base nano turbo; do
 case "$v" in
  base) repo='ResembleAI/chatterbox'; rev='5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'; patterns='README.md Cangjie5_TC.json conds.pt grapheme_mtl_merged_expanded_v1.json mtl_tokenizer.json s3gen_v3.safetensors t3_mtl23ls_v3.safetensors tokenizer.json ve.safetensors';;
  nano) repo='ResembleAI/chatterbox-nano'; rev='71ccd1d0081b430592cea481f4307e764e07bc64'; patterns='README.md added_tokens.json conds.pt merges.txt s3gen.safetensors s3gen_meanflow.safetensors special_tokens_map.json t3_nano_v1.safetensors t3_nano_v1.yaml tokenizer_config.json ve.safetensors vocab.json';;
  turbo) repo='ResembleAI/chatterbox-turbo'; rev='749d1c1a46eb10492095d68fbcf55691ccf137cd'; patterns='README.md added_tokens.json conds.pt merges.txt s3gen.safetensors s3gen_meanflow.safetensors special_tokens_map.json t3_turbo_v1.safetensors t3_turbo_v1.yaml tokenizer_config.json ve.safetensors vocab.json';;
 esac
 mkdir -p "$work/$v/model" "$work/$v/evidence"
 UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$repo" "$rev" "$work/$v/server_tree.json" "$patterns" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys,re
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder
repo,rev,out,patterns=sys.argv[1:]; api=HfApi(); info=api.model_info(repo,revision=rev)
if info.sha!=rev: raise RuntimeError('resolved HF revision mismatch')
def get(item,name,default=None):
    return item.get(name,default) if isinstance(item,dict) else getattr(item,name,default)
rows=[]; pending=['']; seen=set()
while pending:
 p=pending.pop()
 if p in seen: continue
 seen.add(p)
 for item in api.list_repo_tree(repo,revision=rev,path_in_repo=p,recursive=False):
  if isinstance(item,RepoFolder):
   if get(item,'type') not in {None,'directory'}: raise RuntimeError('invalid RepoFolder type')
   typ='directory'
  elif isinstance(item,RepoFile):
   if get(item,'type') not in {None,'file'}: raise RuntimeError('invalid RepoFile type')
   typ='file'
  else: raise RuntimeError(f'unknown HF tree entry type: {type(item).__name__}')
  path=get(item,'path')
  if not isinstance(path,str): raise RuntimeError('invalid tree item path')
  if typ=='directory': pending.append(path); continue
  lfs=get(item,'lfs'); lfs_sha=(lfs.get('sha256') or lfs.get('oid')) if isinstance(lfs,dict) else (get(lfs,'sha256') or get(lfs,'oid')) if lfs is not None else None
  git_id=get(item,'blob_id') or get(item,'oid'); size=get(item,'size')
  if not isinstance(git_id,str) or not re.fullmatch(r'[0-9a-f]{40}',git_id) or not isinstance(size,int) or isinstance(size,bool): raise RuntimeError('invalid identity')
  if lfs_sha is not None and (not isinstance(lfs_sha,str) or not re.fullmatch(r'[0-9a-f]{64}',lfs_sha)): raise RuntimeError('invalid LFS identity')
  rows.append({'path':path,'type':'file','size':size,'git_blob_sha1':git_id,'lfs_sha256':lfs_sha})
if len({x['path'] for x in rows})!=len(rows): raise RuntimeError('duplicate path')
Path(out).write_text(json.dumps({'repository':repo,'revision':rev,'resolved_revision':info.sha,'files':sorted(rows,key=lambda x:x['path'])},indent=2,sort_keys=True)+'\n')
PY
 # shellcheck disable=SC2086
 UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$repo" "$rev" "$work/$v/model" $patterns <<'PY' >>"$work/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
repo,rev,dest,*patterns=sys.argv[1:]
snapshot_download(repo_id=repo,revision=rev,local_dir=dest,allow_patterns=patterns)
PY
 done
 if [[ ! -d "$work/source/chatterbox" ]]; then git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work/source/chatterbox" >>"$work/evidence/validation.log" 2>&1; git -C "$work/source/chatterbox" checkout --detach "$SOURCE_REV" >>"$work/evidence/validation.log" 2>&1; fi
 set +e
 for v in base nano turbo; do
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$INSPECTOR" --variant "$v" --snapshot "$work/$v/model" --server-tree "$work/$v/server_tree.json" --source "$work/source/chatterbox" --evidence "$work/$v/evidence" >>"$work/evidence/validation.log" 2>&1
  rc=$?; [[ "$rc" == 2 ]] || { set -e; die "inspector $v returned $rc"; }
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$work/$v/evidence/manifest.json" "$v" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys
def pairs(items):
    out={}
    for key,value in items:
        if key in out: raise ValueError(f'duplicate manifest key: {key}')
        out[key]=value
    return out
m=json.loads(open(sys.argv[1],encoding='utf-8').read(),object_pairs_hook=pairs)
required={'status':'BLOCKED','evidence_stage':'INSPECTION_ONLY','runtime_status':'NOT_IMPLEMENTED_FAIL_CLOSED','cpu_status':'UNSUPPORTED','metal_status':'BLOCKED_BY_CPU','parity_status':'NOT_RUN','publication':'NO_UPLOAD','inspection_status':'AUTHENTICATED_EVIDENCE_COMPLETE'}
if any(m.get(k)!=v for k,v in required.items()): raise SystemExit('manifest status mismatch')
if m.get('inspection_status')=='INSPECTION_ERROR': raise SystemExit('inspection error was accepted')
if m.get('variant')!=sys.argv[2]: raise SystemExit('manifest variant mismatch')
PY
 done
 set -e
die 'inspection evidence preserved; composite runtime/parity remain blocked'
