#!/usr/bin/env bash
# VAST-only Chatterbox composite inspection.  It never converts, publishes, or uploads.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/chatterbox_family_inspect.py"
REFERENCE="$ROOT/tools/parity/chatterbox_t3_reference.py"
REFERENCE_PROJECT="$ROOT/tools/parity/chatterbox_t3"
REFERENCE_LOCK_SHA256="2fa167c5d2587d7fef6ac2c589a193f9cbd9a8d4495e22487a53a7ba5da6798f"
SOURCE_URL="https://github.com/resemble-ai/chatterbox.git"
SOURCE_REV="5de7a54aa4e5e2baadb0182dde554908b48b85c2"
WORK="${CHATTERBOX_WORK_DIR:-/dev/shm/vokra-chatterbox-family-inspection}"
UV_CACHE_DIR="${CHATTERBOX_UV_CACHE_DIR:-/tmp/vokra-chatterbox-uv-cache}"
MIN_MEM_KIB=$((128*1024*1024)); MIN_TMPFS_KIB=$((32*1024*1024))
log(){ printf '[chatterbox-vast] %s\n' "$*" >&2; }
die(){ log "ERROR: $*"; exit 2; }
sha256_file(){ sha256sum "$1" | awk '{print $1}'; }
require_clean_expected_head(){
 local expected="$1" actual
 [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be exactly 40 lowercase hexadecimal characters'
 [[ -d "$ROOT/.git" ]] || die 'checkout is missing .git'
 [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
 actual="$(git -C "$ROOT" rev-parse HEAD)" || die 'cannot resolve checkout HEAD'
 [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from expected $expected"
}
require_regular_approval_path(){
 local input="$1" path="$1" rest component current base
 [[ -n "$path" && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || return 2
 if [[ "$path" != /* ]]; then base="$(pwd -P)" || return 2; path="$base/$path"; fi
 [[ "$path" != */../* && "$path" != */.. && "$path" != *'/./'* && "$path" != *'/.' ]] || return 2
 rest="${path#/}"; current="/"
 while [[ -n "$rest" ]]; do
  if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
  [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 2
  current="$current$component"; [[ ! -L "$current" ]] || return 2; current="$current/"
 done
 [[ -f "$input" && ! -L "$input" ]]
}
require_approval_binding(){
 local approval="$1" expected_sha="$2"
 [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 must be exactly 64 lowercase hexadecimal characters'
 require_regular_approval_path "$approval" || die 'approval evidence must be a regular file with safe non-symlink ancestry'
 [[ "$(sha256_file "$approval")" == "$expected_sha" ]] || die 'approval evidence SHA-256 differs from caller binding'
}
require_blocked_approval(){
 local approval="$1" expected_head="$2" scope='{"source_url":"https://github.com/resemble-ai/chatterbox.git","source_revision":"5de7a54aa4e5e2baadb0182dde554908b48b85c2","variants":["base","nano","turbo"]}'
 local scope_sha
 scope_sha="$(printf '%s' "$scope" | sha256sum | awk '{print $1}')"
 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$expected_head" "$scope_sha" <<'PY'
import hashlib, json, pathlib, sys
def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate approval key: " + key)
        out[key] = value
    return out
try:
    value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    expected = {
        "schema", "decision", "status", "evidence_stage", "no_upload", "expected_head",
        "source_url", "source_revision", "variants", "scope_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("approval schema is not exact")
    if value["schema"] != "chatterbox-vast-approval-v1" or value["decision"] != "BLOCKED" or value["status"] != "BLOCKED" or value["evidence_stage"] != "INSPECTION_ONLY" or value["no_upload"] is not True:
        raise ValueError("approval is not the blocked inspection disposition")
    if value["expected_head"] != sys.argv[2] or value["source_url"] != "https://github.com/resemble-ai/chatterbox.git" or value["source_revision"] != "5de7a54aa4e5e2baadb0182dde554908b48b85c2" or value["variants"] != ["base", "nano", "turbo"] or value["scope_sha256"] != sys.argv[3]:
        raise ValueError("approval identity or scope mismatch")
    raise RuntimeError("BLOCKED_APPROVAL/INSPECTION_ONLY")
except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as error:
    raise SystemExit(str(error))
PY
}
validate_absent_work(){
 local work="$1" component rest current parent candidate item root_real project_real
 local -a suffix=()
 [[ "$work" == /* && "$work" != *$'\n'* && "$work" != *$'\r'* ]] || return 2
 [[ "$work" != */../* && "$work" != */.. && "$work" != *'/./'* && "$work" != *'/.' ]] || return 2
 rest="${work#/}"; current="/"
 while [[ -n "$rest" ]]; do
  if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
  [[ -n "$component" ]] || return 2
  current="$current$component"; [[ ! -L "$current" ]] || return 2; current="$current/"
 done
 [[ ! -e "$work" && ! -L "$work" ]] || return 2
 parent="$work"
 while [[ ! -e "$parent" ]]; do
  [[ ! -L "$parent" ]] || return 2; item="${parent##*/}"; [[ -n "$item" ]] || return 2
  suffix+=("$item"); [[ "$parent" != / ]] || return 2; parent="${parent%/*}"; [[ -n "$parent" ]] || parent=/
 done
 [[ -d "$parent" && ! -L "$parent" ]] || return 2
 candidate="$(cd -P "$parent" && pwd)" || return 2
 for ((item=${#suffix[@]}-1; item>=0; item--)); do candidate="$candidate/${suffix[item]}"; done
 root_real="$(cd -P "$ROOT" && pwd)" || return 2
 [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || return 2
 if [[ -d "$REFERENCE_PROJECT" ]]; then
  project_real="$(cd -P "$REFERENCE_PROJECT" && pwd)" || return 2
  [[ "$candidate" != "$project_real" && "$candidate/" != "$project_real/"* && "$project_real/" != "$candidate/"* ]] || return 2
 fi
}
claim_absent_directory(){ local path="$1"; [[ ! -e "$path" && ! -L "$path" ]] || return 2; mkdir "$path" || return 2; [[ -d "$path" && ! -L "$path" ]]; }
usage(){ echo 'usage: run-chatterbox-family-inspection.sh --approval-evidence FILE --approval-sha256 SHA --expected-head HEAD [--work-dir DIR] | --self-test'; }
license_audit_preflight(){
 set +e
 local audit_output audit_rc
 audit_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --license-audit 2>&1)"
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
 local fail=0 token tmp approval approval_sha
 for token in 'ResembleAI/chatterbox' '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18' 'ResembleAI/chatterbox-nano' '71ccd1d0081b430592cea481f4307e764e07bc64' 'ResembleAI/chatterbox-turbo' '749d1c1a46eb10492095d68fbcf55691ccf137cd' '5de7a54aa4e5e2baadb0182dde554908b48b85c2' 'SOURCE_ROLE_BLOBS' 'git_blob_sha1' 'lfs_sha256' 'path_in_repo' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'NOT_IMPLEMENTED_FAIL_CLOSED' 'NO_UPLOAD' 'BLOCKED_APPROVAL/INSPECTION_ONLY' '--approval-sha256' '--expected-head' 'CARGO_BUILD_JOBS=1' 'chatterbox_t3/pyproject.toml' 'uv.lock' '--license-audit' 'BLOCKED_UNRESOLVED' 'https://download.pytorch.org/whl/cpu' '2.6.0+cpu' 'transformers==5.10.4' 'source_declared_transformers' 'isolated_transformers_security_floor' 'isolated_transformers_pin' 'GHSA-xrqw-3rrv-vx5w' '2fa167c5d2587d7fef6ac2c589a193f9cbd9a8d4495e22487a53a7ba5da6798f' '1feb25cd45b465dc7fb37dce07599c16218584211640357d541ba969917342d8' 'package_rows' 'license_conclusions'; do
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
 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
 tmp="$(cd -P "$(mktemp -d)" && pwd)"; approval="$tmp/approval.json"; printf '{}\n' >"$approval"
 if require_blocked_approval "$approval" "$(printf '0%.0s' {1..40})" >/dev/null 2>&1; then fail=1; fi
 scope='{"source_url":"https://github.com/resemble-ai/chatterbox.git","source_revision":"5de7a54aa4e5e2baadb0182dde554908b48b85c2","variants":["base","nano","turbo"]}'
 scope_sha="$(printf '%s' "$scope" | sha256sum | awk '{print $1}')"
 printf '{"schema":"chatterbox-vast-approval-v1","decision":"BLOCKED","status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","no_upload":true,"expected_head":"%s","source_url":"https://github.com/resemble-ai/chatterbox.git","source_revision":"5de7a54aa4e5e2baadb0182dde554908b48b85c2","variants":["base","nano","turbo"],"scope_sha256":"%s"}\n' "$(printf '0%.0s' {1..40})" "$scope_sha" >"$approval"
 approval_sha="$(sha256_file "$approval")"
 require_approval_binding "$approval" "$approval_sha" || fail=1
 if require_blocked_approval "$approval" "$(printf '0%.0s' {1..40})" >/dev/null 2>&1; then fail=1; fi
 if validate_absent_work "$tmp/../escape" >/dev/null 2>&1; then fail=1; fi
 mkdir "$tmp/real"; ln -s "$tmp/real" "$tmp/link"
 if validate_absent_work "$tmp/link/work" >/dev/null 2>&1; then fail=1; fi
 rm -rf "$tmp"
 (( fail == 0 )) || return 1
 log 'self-test PASS'
}
work="$WORK"; approval_evidence=''; approval_sha256=''; expected_head=''; self=0
seen_self=0; seen_work=0; seen_approval=0; seen_sha=0; seen_head=0
while (($#)); do case "$1" in
 --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift;;
 --work-dir) ((seen_work==0 && $#>=2)) || die 'duplicate or missing --work-dir'; work="$2"; seen_work=1; shift 2;;
 --approval-evidence) ((seen_approval==0 && $#>=2)) || die 'duplicate or missing --approval-evidence'; approval_evidence="$2"; seen_approval=1; shift 2;;
 --approval-sha256) ((seen_sha==0 && $#>=2)) || die 'duplicate or missing --approval-sha256'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'invalid --approval-sha256'; approval_sha256="$2"; seen_sha=1; shift 2;;
 --expected-head) ((seen_head==0 && $#>=2)) || die 'duplicate or missing --expected-head'; [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die 'invalid --expected-head'; expected_head="$2"; seen_head=1; shift 2;;
 -h|--help) usage; exit 0;; *) die "unknown argument: $1";; esac; done
if ((self)); then [[ "$work" == "$WORK" && $seen_approval == 0 && $seen_sha == 0 && $seen_head == 0 ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
[[ $seen_approval == 1 && $seen_sha == 1 && $seen_head == 1 ]] || die '--approval-evidence, --approval-sha256, and --expected-head are required'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required for caller binding'
require_clean_expected_head "$expected_head"
require_approval_binding "$approval_evidence" "$approval_sha256"
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated Chatterbox reference pyproject.toml + uv.lock are required before any model download'
if ! license_audit_preflight; then die 'dependency license audit is unresolved; no Chatterbox model acquisition or reference execution is permitted'; fi
if require_blocked_approval "$approval_evidence" "$expected_head"; then
 die 'blocked approval unexpectedly authorized execution'
else
 die 'BLOCKED_APPROVAL/INSPECTION_ONLY: Chatterbox acquisition remains disabled'
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
for tool in awk cargo df find findmnt git uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool $tool"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'; ((mem_kib>=MIN_MEM_KIB)) || die '128 GiB memory guard failed'
validate_absent_work "$work" || die 'work directory must be absent, disjoint, and free of symlink ancestors'
[[ "$(findmnt -T "$(dirname "$work")" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$(dirname "$work")" | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid tmpfs free-space value'; ((free_kib>=MIN_TMPFS_KIB)) || die '32 GiB tmpfs guard failed'
claim_absent_directory "$work" || die 'work directory could not be atomically claimed'
mkdir "$work/evidence" "$work/source"
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
 mkdir "$work/$v" "$work/$v/model" "$work/$v/evidence"
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
require_clean_expected_head "$expected_head"
die 'inspection evidence preserved; composite runtime/parity remain blocked'
