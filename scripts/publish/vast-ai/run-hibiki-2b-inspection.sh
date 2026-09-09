#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2317
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF_REPOSITORY="kyutai/hibiki-2b-pytorch-bf16"; HF_REVISION="bd71144c96f26040612f6414716f5f48ee4fce69"
HIBIKI_URL="https://github.com/kyutai-labs/hibiki.git"; HIBIKI_REVISION="f1cf9293e35c1dceffbe60dd325bdd702bc8305e"
MOSHI_URL="https://github.com/kyutai-labs/moshi.git"; MOSHI_REVISION="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
INSPECTOR="$ROOT/tools/parity/hibiki_2b_inspect.py"; MIN_MEM_KIB=$((128*1024*1024)); MIN_DISK_KIB=$((16*1024*1024))
GATE="$ROOT/tools/parity/hibiki_2b_gate.py"
die(){ echo "hibiki-vast: ERROR: $*" >&2; exit 2; }
validate_manifest(){
 local manifest_path="$1"
 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$manifest_path" <<'PY'
import json,sys
from pathlib import Path
def no_dupes(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise SystemExit(f"duplicate manifest key: {key}")
  out[key]=value
 return out
manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"),object_pairs_hook=no_dupes)
required={"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD"}
for key,value in required.items():
 if manifest.get(key)!=value: raise SystemExit(f"inspection evidence incomplete: {key}={manifest.get(key)!r}")
if manifest.get("inspection_status") in {"INSPECTION_ERROR","FAILED"}: raise SystemExit("inspection failure was treated as complete")
model=manifest.get("model")
if not isinstance(model,dict) or model.get("repository")!="kyutai/hibiki-2b-pytorch-bf16" or model.get("requested_revision")!="bd71144c96f26040612f6414716f5f48ee4fce69" or model.get("resolved_revision")!="bd71144c96f26040612f6414716f5f48ee4fce69": raise SystemExit("model revision identity is incomplete")
PY
}
self_test(){
 local path="${BASH_SOURCE[0]}" token fail=0 fixture_dir
 for token in "$HF_REPOSITORY" "$HF_REVISION" "$HIBIKI_URL" "$HIBIKI_REVISION" "$MOSHI_URL" "$MOSHI_REVISION" 'requested_revision' 'recursive_file_only' 'lfs_pointer_git_blob_sha1' 'lfs_payload_sha256' 'SentencePiece' 'INSPECTION_ONLY' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'NO_UPLOAD' 'BLOCKED_UNRESOLVED_HIBIKI_2B_COMPOSITE' '--expected-head' '--approval-evidence' '--approval-sha256' 'CARGO_BUILD_JOBS=1'; do
  if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$INSPECTOR"; then echo "missing contract $token" >&2; fail=1; fi
 done
 if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi
 gate_line="$(grep -nF 'python "$GATE" --verify' "$path" | tail -n1 | cut -d: -f1)"; uname_line="$(grep -nF 'VAST requires Linux x86_64' "$path" | tail -n1 | cut -d: -f1)"; work_line="$(grep -n '^work=/dev/shm' "$path" | tail -n1 | cut -d: -f1)"; snapshot_line="$(grep -nF 'snapshot_download(repo_id' "$path" | tail -n1 | cut -d: -f1)"
 [[ "$gate_line" =~ ^[0-9]+$ && "$uname_line" =~ ^[0-9]+$ && "$work_line" =~ ^[0-9]+$ && "$snapshot_line" =~ ^[0-9]+$ && $gate_line -lt $uname_line && $gate_line -lt $work_line && $gate_line -lt $snapshot_line ]] || fail=1
 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --self-test || fail=1
 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$INSPECTOR" --self-test || fail=1
 if "$path" --self-test --self-test >/dev/null 2>&1; then fail=1; fi
 if "$path" --self-test --expected-head bad >/dev/null 2>&1; then fail=1; fi
 fixture_dir="$(mktemp -d)"
 printf '%s\n' '{"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","collection_status":"AUTHENTICATED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":"kyutai/hibiki-2b-pytorch-bf16","requested_revision":"bd71144c96f26040612f6414716f5f48ee4fce69","resolved_revision":"bd71144c96f26040612f6414716f5f48ee4fce69"}}' >"$fixture_dir/valid.json"
 printf '%s\n' '{"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"INSPECTION_ERROR","collection_status":"UNVERIFIED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":"kyutai/hibiki-2b-pytorch-bf16","requested_revision":"bd71144c96f26040612f6414716f5f48ee4fce69","resolved_revision":null}}' >"$fixture_dir/error.json"
 validate_manifest "$fixture_dir/valid.json" || fail=1
 if validate_manifest "$fixture_dir/error.json"; then fail=1; fi
 ((fail==0)) || return 1; echo 'run-hibiki-2b-inspection.sh self-test: OK'
}
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
expected_head=''; approval_evidence=''; approval_sha256=''; seen_expected=0; seen_approval=0; seen_sha=0
while (($#)); do
 case "$1" in
  --expected-head) ((seen_expected+=1)); shift; (($#)) || die '--expected-head requires a value'; expected_head="$1" ;;
  --approval-evidence) ((seen_approval+=1)); shift; (($#)) || die '--approval-evidence requires a value'; approval_evidence="$1" ;;
  --approval-sha256) ((seen_sha+=1)); shift; (($#)) || die '--approval-sha256 requires a value'; approval_sha256="$1" ;;
  --self-test) die '--self-test cannot be combined with normal arguments';;
  -h|--help) sed -n '1,18p' "$0"; exit 0 ;;
  *) die "unknown argument: $1" ;;
 esac
 shift
done
[[ $seen_expected == 1 && $seen_approval == 1 && $seen_sha == 1 ]] || die 'exactly one --expected-head, --approval-evidence, and --approval-sha256 is required'
gate_log=""; gate_rc=0
if gate_log="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --verify --expected-head "$expected_head" --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" --root "$ROOT" 2>&1)"; then gate_rc=0; else gate_rc=$?; fi
[[ $gate_rc == 2 && "$gate_log" == *BLOCKED_UNRESOLVED_HIBIKI_2B_COMPOSITE* ]] || die "Hibiki approval gate did not reach the expected terminal blocker: $gate_log"
die 'BLOCKED_UNRESOLVED_HIBIKI_2B_COMPOSITE: current blocked approval cannot authorize host checks, acquisition, or inspection'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project is missing before acquisition'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'
work=/dev/shm/vokra-hibiki-inspection; if [[ -e "$work" ]]; then [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory must be empty'; else mkdir -p "$work"; fi; mkdir -p "$work/model" "$work/hibiki" "$work/moshi" "$work/evidence"
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die 'tmpfs disk guard failed'
for command in cargo git uv awk find df; do command -v "$command" >/dev/null || die "missing tool: $command"; done
export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${HIBIKI_UV_CACHE_DIR:-/tmp/vokra-hibiki-uv-cache}"
# shellcheck disable=SC2129 # validation output is one stream
{ cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1; } >"$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129 # heredoc output is one validation stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/tree.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder,snapshot_download
repo,rev,out=sys.argv[1:]; api=HfApi(); info=api.model_info(repo_id=repo,revision=rev); assert info.sha==rev
destination=Path(sys.argv[3]).parent/"model"; snapshot=Path(snapshot_download(repo_id=repo,revision=rev,local_dir=destination,allow_patterns=["*"])); assert snapshot.resolve()==destination.resolve()
expected={".gitattributes","README.md","config.json","hibiki-pytorch-ccef4858@200.safetensors","mimi-pytorch-e351c8d8@125.safetensors","tokenizer_spm_48k_multi6_2.model"}
rows=[]
for x in api.list_repo_tree(repo_id=repo,revision=rev,recursive=True,expand=True):
 kind=getattr(x,"type",None)
 if isinstance(x,RepoFolder): continue
 if not isinstance(x,RepoFile) or kind not in {None,"file"}: raise SystemExit(f"unsupported server entry: {x}")
 if x.path not in expected: raise SystemExit(f"unexpected HF file: {x.path}")
 lfs=getattr(x,"lfs",None); lfs_sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None); lfs_size=lfs.get("size") if isinstance(lfs,dict) else getattr(lfs,"size",None); blob=getattr(x,"blob_id",None)
 if not isinstance(x.path,str) or not isinstance(x.size,int) or not isinstance(blob,str) or len(blob)!=40: raise SystemExit(f"incomplete server identity: {x}")
 if lfs_sha is None:
  row={"path":x.path,"type":"file","size":x.size,"git_blob_sha1":blob,"lfs_pointer_git_blob_sha1":None,"lfs_payload_sha256":None,"lfs_payload_size":None}
 else:
  if not isinstance(lfs_sha,str) or len(lfs_sha)!=64 or lfs_size!=x.size: raise SystemExit(f"invalid LFS identity: {x.path}")
  pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {x.size}\n".encode(); pointer_git=hashlib.sha1(f"blob {len(pointer)}\0".encode()+pointer).hexdigest()
  if pointer_git!=blob: raise SystemExit(f"LFS pointer Git blob mismatch: {x.path}")
  row={"path":x.path,"type":"file","size":x.size,"git_blob_sha1":None,"lfs_pointer_git_blob_sha1":blob,"lfs_payload_sha256":lfs_sha,"lfs_payload_size":x.size}
 rows.append(row)
if {row["path"] for row in rows}!=expected or len(rows)!=len(expected): raise SystemExit("HF tree is not the exact complete file set")
Path(out).write_text(json.dumps({"repository":repo,"requested_revision":rev,"resolved_revision":info.sha,"walk":"recursive_file_only","files":sorted(rows,key=lambda row:row["path"])},sort_keys=True,indent=2)+"\n")
PY
git clone --filter=blob:none "$HIBIKI_URL" "$work/hibiki/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/hibiki/repo" checkout --detach "$HIBIKI_REVISION" >>"$work/evidence/validation.log" 2>&1; [[ "$(git -C "$work/hibiki/repo" rev-parse HEAD)" == "$HIBIKI_REVISION" ]] || die 'Hibiki source revision mismatch'; [[ "$(git -C "$work/hibiki/repo" remote get-url origin)" == "$HIBIKI_URL" ]] || die 'Hibiki source origin mismatch'
git clone --filter=blob:none "$MOSHI_URL" "$work/moshi/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/moshi/repo" checkout --detach "$MOSHI_REVISION" >>"$work/evidence/validation.log" 2>&1; [[ "$(git -C "$work/moshi/repo" rev-parse HEAD)" == "$MOSHI_REVISION" ]] || die 'Moshi source revision mismatch'; [[ "$(git -C "$work/moshi/repo" remote get-url origin)" == "$MOSHI_URL" ]] || die 'Moshi source origin mismatch'
set +e
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --hibiki-source "$work/hibiki/repo" --moshi-source "$work/moshi/repo" --server-tree "$work/tree.json" --output "$work/evidence" --expected-head "$expected_head" --approval-evidence "$approval_evidence" --approval-sha256 "$approval_sha256" >>"$work/evidence/validation.log" 2>&1
rc=$?; set -e; [[ "$rc" == 2 ]] || die 'inspector must exit 2'
validate_manifest "$work/evidence/manifest.json" || die 'inspection manifest is not authenticated complete evidence'
exit 2
