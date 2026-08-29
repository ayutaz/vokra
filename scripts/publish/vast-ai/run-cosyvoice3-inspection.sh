#!/usr/bin/env bash
# VAST-only full CosyVoice3 evidence run. No conversion, publication, or upload.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/cosyvoice3_inspect.py"
PROJECT="$ROOT/tools/parity/cosyvoice3_reference"
WORK="${COSYVOICE3_WORK_DIR:-/dev/shm/vokra-cosyvoice3-inspection}"
UV_CACHE_DIR="${COSYVOICE3_UV_CACHE_DIR:-/tmp/vokra-cosyvoice3-uv-cache}"
HF_REPO='FunAudioLLM/Fun-CosyVoice3-0.5B-2512'; HF_REV='29e01c4e8d000f4bcd70751be16fa94bf3d85a18'
SOURCE_URL='https://github.com/FunAudioLLM/CosyVoice.git'; SOURCE_REV='0d990d60740bf174904a5185cce910b847bd3684'
MATCHA_URL='https://github.com/shivammehta25/Matcha-TTS.git'; MATCHA_REV='dd9105b34bf2be2230f4aa1e4769fb586a3c824e'
MIN_MEM_KIB=$((128*1024*1024)); MIN_TMPFS_KIB=$((32*1024*1024))
log(){ printf '[cosyvoice3-vast] %s\n' "$*" >&2; }
die(){ log "ERROR: $*"; exit 2; }
usage(){ echo 'usage: run-cosyvoice3-inspection.sh [--work-dir DIR] | --self-test'; }
self_test(){
  local fail=0 token
  for token in "$HF_REPO" "$HF_REV" "$SOURCE_REV" "$MATCHA_REV" 'git_blob_sha1' 'lfs_sha256' 'path_in_repo' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'NO_UPLOAD' 'weights_only=True' 'CARGO_BUILD_JOBS=1'; do
    grep -Fq -- "$token" "$INSPECTOR" "$0" || { log "self-test missing $token"; fail=1; }
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$0" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("CosyVoice3 tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"CosyVoice3 tree walk has incompatible path keyword: {call}")
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
print("CosyVoice3 RepoFile/RepoFolder self-test: PASS")
PY
  then
    log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
    fail=1
  fi
  [[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || { log 'dedicated CosyVoice3 uv.lock absent; self-test deliberately blocked'; return 2; }
  grep -Eq '^[[:space:]]*(git[[:space:]]+push|hf_hub_upload|upload_file|convert)' "$0" && { log 'self-test publication/conversion command'; fail=1; } || true
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
  ((fail == 0)) || return 1
  log 'self-test PASS'
}
work="$WORK"; self=0
while (($#)); do case "$1" in
  --self-test) self=1; shift;;
  --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work="$2"; shift 2;;
  -h|--help) usage; exit 0;; *) die "unknown argument: $1";;
esac; done
if ((self)); then [[ "$work" == "$WORK" ]] || die '--self-test accepts no work-dir'; self_test; exit $?; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ -f "$PROJECT/pyproject.toml" && -f "$PROJECT/uv.lock" ]] || die 'dedicated CosyVoice3 uv.lock missing; refuse before acquisition'
for tool in awk cargo df find findmnt git uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool $tool"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
((mem_kib >= MIN_MEM_KIB)) || die '128 GiB memory guard failed'
mkdir -p "$(dirname "$work")"; [[ ! -e "$work" || -z "$(find "$work" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
[[ "$(findmnt -T "$(dirname "$work")" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$(dirname "$work")" | awk 'NR==2{print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid tmpfs free-space value'
((free_kib >= MIN_TMPFS_KIB)) || die '32 GiB tmpfs guard failed'
mkdir -p "$work"/{model,evidence,source,matcha}; work="$(cd "$work" && pwd)"; export CARGO_BUILD_JOBS=1 UV_CACHE_DIR
{
  printf '%s\n' 'status=BLOCKED' 'evidence_stage=INSPECTION_ONLY' 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED' 'cpu_status=UNSUPPORTED_FULL_TTS_PENDING' 'metal_status=BLOCKED_BY_CPU' 'parity_status=NOT_RUN' 'publication=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$PROJECT" --python 3.12 python - "$HF_REPO" "$HF_REV" "$work/server_tree.json" "$work/model" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,re,sys
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder,snapshot_download
repo,rev,packet,dest=sys.argv[1:]; api=HfApi(); info=api.model_info(repo,revision=rev)
if info.sha != rev: raise RuntimeError('HF resolved revision mismatch')
def get(x,k,default=None):
    return x.get(k,default) if isinstance(x,dict) else getattr(x,k,default)
def classify_entry(item):
    if isinstance(item,RepoFolder):
        if get(item,'type') not in (None,'directory'): raise RuntimeError('invalid RepoFolder type')
        return 'directory'
    if isinstance(item,RepoFile):
        if get(item,'type') not in (None,'file'): raise RuntimeError('invalid RepoFile type')
        return 'file'
    raise RuntimeError(f'unknown HF tree entry type: {type(item).__name__}')
rows=[]; pending=['']; seen=set()
while pending:
    p=pending.pop()
    if p in seen: continue
    seen.add(p)
    for item in api.list_repo_tree(repo,revision=rev,path_in_repo=p,recursive=False):
        typ=classify_entry(item); path=get(item,'path')
        if not isinstance(path,str): raise RuntimeError('invalid HF tree entry path')
        if typ=='directory': pending.append(path); continue
        lfs=get(item,'lfs')
        lsha=(get(lfs,'sha256') or get(lfs,'oid')) if lfs is not None else None
        blob,size=get(item,'blob_id') or get(item,'oid'),get(item,'size')
        if not isinstance(blob,str) or not re.fullmatch(r'[0-9a-f]{40}',blob) or not isinstance(size,int) or isinstance(size,bool): raise RuntimeError('invalid server identity')
        if lsha is not None and (not isinstance(lsha,str) or not re.fullmatch(r'[0-9a-f]{64}',lsha)): raise RuntimeError('invalid LFS identity')
        rows.append({'path':path,'type':'file','size':size,'git_blob_sha1':blob,'lfs_sha256':lsha})
if len({x['path'] for x in rows}) != len(rows): raise RuntimeError('duplicate server path')
Path(packet).write_text(json.dumps({'repository':repo,'revision':rev,'resolved_revision':info.sha,'walk':'recursive_file_only','files':sorted(rows,key=lambda x:x['path'])},indent=2,sort_keys=True)+'\n')
snapshot_download(repo_id=repo,revision=rev,local_dir=dest)
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/CosyVoice" >>"$work/evidence/validation.log" 2>&1
git -C "$work/source/CosyVoice" checkout --detach "$SOURCE_REV" >>"$work/evidence/validation.log" 2>&1
git -C "$work/source/CosyVoice" submodule update --init --recursive >>"$work/evidence/validation.log" 2>&1
git clone --filter=blob:none "$MATCHA_URL" "$work/matcha" >>"$work/evidence/validation.log" 2>&1
git -C "$work/matcha" checkout --detach "$MATCHA_REV" >>"$work/evidence/validation.log" 2>&1
set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$PROJECT" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/CosyVoice" --matcha-source "$work/matcha" --server-tree "$work/server_tree.json" --output "$work/evidence" >>"$work/evidence/validation.log" 2>&1
rc=$?
set -e
[[ "$rc" == 2 ]] || die "inspector returned $rc"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,sys
m=json.loads(open(sys.argv[1],encoding='utf-8').read())
expected={'status':'BLOCKED','evidence_stage':'INSPECTION_ONLY','runtime_status':'NOT_IMPLEMENTED_FAIL_CLOSED','cpu_status':'UNSUPPORTED_FULL_TTS_PENDING','metal_status':'BLOCKED_BY_CPU','parity_status':'NOT_RUN','publication':'NO_UPLOAD','inspection_status':'AUTHENTICATED_EVIDENCE_COMPLETE'}
if any(m.get(k)!=v for k,v in expected.items()): raise SystemExit(f'manifest status mismatch: {m}')
if m.get('inspection_status')=='INSPECTION_ERROR': raise SystemExit('inspection error accepted')
PY
die 'inspection evidence preserved; full TTS binder/parity remain blocked'
